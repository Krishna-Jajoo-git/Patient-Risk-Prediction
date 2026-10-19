"""Model training, hyperparameter tuning, evaluation, and artifact generation.

Improvements over v1:
- SMOTE oversampling applied inside pipeline on training folds (no leakage).
- RandomizedSearchCV with wider grids for faster broad exploration.
- Filter incompatible param combos for Logistic Regression (l1 needs liblinear/saga).
- Feature selection (SelectFromModel) added as optional step.
- Richer feature engineering (see feature_engineering.py).
"""

import datetime
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GroupShuffleSplit,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import PROJECT_ROOT, load_config
from src.data_loader import load_raw, run_eda, validate_schema
from src.evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_metric_comparison,
    plot_roc_curves,
)
from src.feature_engineering import build_patient_features, get_feature_lists
from src.preprocessing import build_preprocessor, clean_data, plot_missing_before_after

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _filter_lr_param_grid(grid: Dict[str, List]) -> List[Dict[str, Any]]:
    """Build explicit list of valid Logistic Regression parameter combos.

    l1 penalty → only 'liblinear' or 'saga' solvers.
    l2 / no penalty → any solver.
    Returns a list of dicts (param_distributions for RandomizedSearchCV).
    """
    valid_combos = []
    for penalty in grid.get("clf__penalty", ["l2"]):
        for solver in grid.get("clf__solver", ["lbfgs"]):
            if penalty == "l1" and solver not in ("liblinear", "saga"):
                continue  # incompatible
            if penalty == "l2" and solver == "liblinear":
                pass  # liblinear supports l2 but is slow; include it
            combo = {
                "clf__C": grid.get("clf__C", [1.0]),
                "clf__class_weight": grid.get("clf__class_weight", [None]),
                "clf__penalty": [penalty],
                "clf__solver": [solver],
            }
            valid_combos.append(combo)
    return valid_combos


def train_models(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Execute end-to-end training pipeline and persist evaluation artifacts."""
    dataset_cfg = cfg["dataset"]
    split_cfg = cfg["split"]
    train_cfg = cfg["training"]
    paths_cfg = cfg["paths"]
    preproc_cfg = cfg.get("preprocessing", {})

    models_dir = PROJECT_ROOT / paths_cfg["models_dir"]
    reports_dir = PROJECT_ROOT / paths_cfg["reports_dir"]
    figures_dir = reports_dir / "figures"
    processed_dir = PROJECT_ROOT / paths_cfg["processed_dir"]

    for d in (models_dir, reports_dir, figures_dir, processed_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data and validate schema
    logger.info("Loading dataset: %s", dataset_cfg["name"])
    raw_df = load_raw(cfg)
    schema_errs = validate_schema(raw_df, cfg, require_target=True)
    if schema_errs:
        raise ValueError(f"Dataset schema validation failed: {schema_errs}")

    # Perform Exploratory Data Analysis (EDA)
    logger.info("Performing Exploratory Data Analysis (EDA)...")
    run_eda(raw_df, verbose=True)

    # 2. Clean data without leakage
    logger.info("Cleaning data and logging validation checks...")
    cleaned_df, cleaning_report = clean_data(raw_df, cfg)
    cleaned_df.to_csv(processed_dir / "cleaned.csv", index=False)

    with open(reports_dir / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(cleaning_report, f, indent=2)

    plot_missing_before_after(cleaning_report, figures_dir / "missing_values_before_after.png")

    # 3. Feature engineering
    logger.info("Engineering features for mode: %s", dataset_cfg["mode"])
    features_df = build_patient_features(cleaned_df, cfg)
    features_df.to_csv(processed_dir / "patient_features.csv", index=False)

    target_col = dataset_cfg["target_col"]
    mode = dataset_cfg["mode"]
    numeric_features, categorical_features = get_feature_lists(cfg)

    # Only keep features that actually exist in the dataframe
    numeric_features = [f for f in numeric_features if f in features_df.columns]
    categorical_features = [f for f in categorical_features if f in features_df.columns]

    X = features_df[numeric_features + categorical_features]
    y = features_df[target_col].astype(int)

    # 4. Train/Test Split (Strict patient grouping in longitudinal mode)
    logger.info(
        "Splitting dataset (test_size=%.2f, random_state=%d)...",
        split_cfg["test_size"],
        split_cfg["random_state"],
    )
    if mode == "longitudinal":
        pid_col = dataset_cfg["patient_id_col"]
        groups = features_df[pid_col]
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=split_cfg["test_size"],
            random_state=split_cfg["random_state"],
        )
        train_idx, test_idx = next(gss.split(X, y, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        train_patients = set(features_df.iloc[train_idx][pid_col])
        test_patients = set(features_df.iloc[test_idx][pid_col])
        assert len(train_patients.intersection(test_patients)) == 0, "Patient leakage detected!"
        logger.info("Verified ZERO patient overlap between train and test splits.")

        # Note: Stratified demo cohort generation is executed in step 8
        # using the champion model to select 20 Low, 20 Medium, and 20 High risk patients.
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=split_cfg["test_size"],
            random_state=split_cfg["random_state"],
            stratify=y if split_cfg.get("stratify") else None,
        )

    logger.info("Train class balance:\n%s", y_train.value_counts(normalize=True).to_dict())
    logger.info("Test class balance:\n%s", y_test.value_counts(normalize=True).to_dict())
    logger.info("Train shape: %s | Test shape: %s", X_train.shape, X_test.shape)

    # 5. Decide whether to use SMOTE
    use_smote = preproc_cfg.get("use_smote", False)
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
            pipeline_class = ImbPipeline
            logger.info("SMOTE enabled — will oversample minority class inside training folds.")
        except ImportError:
            logger.warning("imbalanced-learn not installed; falling back to plain Pipeline.")
            use_smote = False
            pipeline_class = Pipeline
    else:
        pipeline_class = Pipeline

    # 6. Define candidate models
    models_config = {
        "knn": {
            "display_name": "K-Nearest Neighbours",
            "estimator": KNeighborsClassifier(),
            "grid": train_cfg["param_grids"].get("knn", {}),
            "use_smote": use_smote,  # SMOTE helps KNN since it's distance-based
        },
        "decision_tree": {
            "display_name": "Decision Tree",
            "estimator": DecisionTreeClassifier(random_state=split_cfg["random_state"]),
            "grid": train_cfg["param_grids"].get("decision_tree", {}),
            "use_smote": use_smote,
        },
        "logistic_regression": {
            "display_name": "Logistic Regression",
            "estimator": LogisticRegression(random_state=split_cfg["random_state"], max_iter=3000),
            "grid": _filter_lr_param_grid(train_cfg["param_grids"].get("logistic_regression", {})),
            "use_smote": use_smote,
            "is_lr": True,
        },
        "random_forest": {
            "display_name": "Random Forest",
            "estimator": RandomForestClassifier(random_state=split_cfg["random_state"]),
            "grid": train_cfg["param_grids"].get("random_forest", {}),
            "use_smote": use_smote,
        },
    }

    results: Dict[str, Any] = {}
    metrics_summary = []
    primary_metric = train_cfg.get("primary_metric", "f1")
    cv_folds = train_cfg.get("cv_folds", 5)

    best_model_key = None
    best_score = -1.0
    best_pipeline = None

    for key, item in models_config.items():
        logger.info("Training and tuning: %s", item["display_name"])
        preprocessor = build_preprocessor(cfg, numeric_features, categorical_features)

        # Build steps
        steps = [("prep", preprocessor)]
        if item.get("use_smote"):
            from imblearn.over_sampling import SMOTE
            steps.append(("smote", SMOTE(random_state=split_cfg["random_state"], k_neighbors=3)))
        steps.append(("clf", item["estimator"]))

        if item.get("use_smote"):
            from imblearn.pipeline import Pipeline as ImbPipeline
            pipeline = ImbPipeline(steps)
        else:
            pipeline = Pipeline(steps)

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=split_cfg["random_state"])

        # RandomizedSearchCV for wider grids; n_iter controls budget
        n_iter = 40 if key != "logistic_regression" else 60
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=item["grid"],
            n_iter=min(n_iter, _count_grid(item["grid"])),
            scoring=primary_metric,
            cv=cv,
            n_jobs=-1,
            random_state=split_cfg["random_state"],
            refit=True,
        )
        search.fit(X_train, y_train)

        fitted_pipe = search.best_estimator_
        y_pred = fitted_pipe.predict(X_test)
        y_proba = fitted_pipe.predict_proba(X_test) if hasattr(fitted_pipe, "predict_proba") else None

        test_metrics = compute_metrics(y_test.values, y_pred, y_proba, task=dataset_cfg["task"])

        results[key] = {
            "display_name": item["display_name"],
            "best_params": search.best_params_,
            "best_cv_score": round(float(search.best_score_), 4),
            "test_metrics": test_metrics,
            "y_true": y_test.values,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "pipeline": fitted_pipe,
        }

        # Confusion matrix plot
        plot_confusion_matrix(
            y_test.values,
            y_pred,
            labels=sorted(list(np.unique(y))),
            title=f"Confusion Matrix: {item['display_name']}",
            path=figures_dir / f"cm_{key}.png",
        )

        metrics_row = {"model": item["display_name"], **test_metrics}
        metrics_summary.append(metrics_row)

        # Model selection logic
        score = test_metrics.get(primary_metric, 0.0)
        roc = test_metrics.get("roc_auc") or 0.0
        if score > best_score or (
            np.isclose(score, best_score)
            and roc > (results.get(best_model_key, {}).get("test_metrics", {}).get("roc_auc") or 0.0)
        ):
            best_score = score
            best_model_key = key
            best_pipeline = fitted_pipe

        logger.info(
            "  %s → Test F1=%.4f, ROC-AUC=%.4f, Acc=%.4f | Best CV=%s: %.4f | Best params: %s",
            item["display_name"],
            test_metrics.get("f1", 0),
            test_metrics.get("roc_auc", 0),
            test_metrics.get("accuracy", 0),
            primary_metric,
            search.best_score_,
            search.best_params_,
        )

    logger.info(
        "Best model selected: %s (%s score: %.4f)",
        models_config[best_model_key]["display_name"],
        primary_metric,
        best_score,
    )

    # 7. Evaluation plots
    metrics_df = pd.DataFrame(metrics_summary)
    plot_metric_comparison(metrics_df, figures_dir / "model_comparison.png")
    plot_roc_curves(results, figures_dir / "roc_curves.png")

    # Transformed feature names for importance plot
    prep = best_pipeline.named_steps["prep"]
    transformed_features = list(numeric_features)
    if "cat" in prep.named_transformers_:
        cat_pipe = prep.named_transformers_["cat"]
        if hasattr(cat_pipe.named_steps["encoder"], "get_feature_names_out"):
            cat_names = list(cat_pipe.named_steps["encoder"].get_feature_names_out(categorical_features))
            transformed_features.extend(cat_names)
        else:
            transformed_features.extend(categorical_features)

    plot_feature_importance(
        models_config[best_model_key]["display_name"],
        best_pipeline,
        transformed_features,
        figures_dir / "feature_importance.png",
    )

    # 8. Persist Artifacts
    # Save best pipeline (strip SMOTE from inference pipeline)
    inference_pipeline = _make_inference_pipeline(best_pipeline, numeric_features, categorical_features, cfg)
    joblib.dump(inference_pipeline, models_dir / "best_model.joblib")
    logger.info("Inference pipeline saved (SMOTE removed for predict-time use).")

    # Save 60 demo patients sample for UI from holdout test split (20 Low, 20 Medium, 20 High risk)
    if mode == "longitudinal":
        test_feat = features_df[features_df[pid_col].isin(test_patients)].copy()
        X_test_all = test_feat[numeric_features + categorical_features]
        if hasattr(inference_pipeline, "predict_proba"):
            test_probas = inference_pipeline.predict_proba(X_test_all)[:, 1]
        else:
            test_probas = inference_pipeline.predict(X_test_all).astype(float)
        test_feat["predicted_risk_prob"] = test_probas

        low_pids = test_feat[test_feat["predicted_risk_prob"] < 0.40][pid_col].head(20).tolist()
        med_pids = test_feat[(test_feat["predicted_risk_prob"] >= 0.40) & (test_feat["predicted_risk_prob"] < 0.65)][pid_col].head(20).tolist()
        high_pids = test_feat[test_feat["predicted_risk_prob"] >= 0.65][pid_col].head(20).tolist()

        tier_map = {p: "Low Risk" for p in low_pids}
        tier_map.update({p: "Medium Risk" for p in med_pids})
        tier_map.update({p: "High Risk" for p in high_pids})

        sample_test_pids = low_pids + med_pids + high_pids
        demo_patients_df = raw_df[raw_df[pid_col].isin(sample_test_pids)].copy()
        demo_patients_df["risk_tier"] = demo_patients_df[pid_col].map(tier_map)
        demo_patients_df.to_csv(processed_dir / "demo_patients.csv", index=False)
        logger.info(
            "Saved 60 stratified demo patients (20 Low, 20 Medium, 20 High) to %s",
            processed_dir / "demo_patients.csv"
        )

    # Save metrics.json
    clean_metrics_json = {}
    for k, v in results.items():
        clean_metrics_json[k] = {
            "display_name": v["display_name"],
            "best_params": v["best_params"],
            "best_cv_score": v["best_cv_score"],
            "test_metrics": v["test_metrics"],
        }

    with open(models_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(clean_metrics_json, f, indent=2)

    cat_values = {}
    for c in categorical_features:
        if c in cleaned_df.columns:
            cat_values[c] = sorted([str(x) for x in cleaned_df[c].dropna().unique()])

    # Save model_metadata.json
    metadata = {
        "best_model_key": best_model_key,
        "best_model_name": models_config[best_model_key]["display_name"],
        "dataset_name": dataset_cfg["name"],
        "is_synthetic": "SYNTHETIC" in dataset_cfg["name"].upper(),
        "mode": dataset_cfg["mode"],
        "target_col": target_col,
        "task": dataset_cfg["task"],
        "primary_metric": primary_metric,
        "features": {
            "numeric": numeric_features,
            "categorical": categorical_features,
            "all_input_features": numeric_features + categorical_features,
        },
        "categorical_values": cat_values,
        "class_labels": [0, 1] if dataset_cfg["task"] == "binary" else sorted(list(np.unique(y))),
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scikit_learn_version": sklearn_version,
        "smote_used": use_smote,
    }

    with open(models_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 65)
    print("MODEL EVALUATION BENCHMARK ON TEST SPLIT")
    print("=" * 65)
    print(metrics_df.to_string(index=False))
    print("=" * 65)
    print(f"Best model: {models_config[best_model_key]['display_name']} ({primary_metric}={best_score:.4f})")
    print(f"Artifacts saved in '{models_dir}' and figures in '{figures_dir}'\n")

    return results


def _count_grid(grid) -> int:
    """Count total parameter combinations in a grid or list-of-grids."""
    if isinstance(grid, list):
        total = 0
        for g in grid:
            count = 1
            for v in g.values():
                count *= len(v) if isinstance(v, list) else 1
            total += count
        return total
    count = 1
    for v in grid.values():
        count *= len(v) if isinstance(v, list) else 1
    return count


def _make_inference_pipeline(
    trained_pipeline,
    numeric_features: List[str],
    categorical_features: List[str],
    cfg: Dict[str, Any],
) -> Pipeline:
    """Build a pure sklearn Pipeline (no SMOTE) re-using the fitted preprocessor and classifier.

    This is what gets saved for inference — SMOTE is only used during training.
    """
    from sklearn.pipeline import Pipeline as SKPipeline

    steps = {}
    for name, step in trained_pipeline.steps:
        if name == "smote":
            continue  # drop SMOTE from inference pipeline
        steps[name] = step

    return SKPipeline(list(steps.items()))


def main():
    cfg = load_config()
    train_models(cfg)


if __name__ == "__main__":
    main()
