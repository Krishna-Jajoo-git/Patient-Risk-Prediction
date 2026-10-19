"""Evaluation metrics computation, plotting, and clinical interpretation helpers."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    task: str = "binary",
) -> Dict[str, float]:
    """Compute standard classification evaluation metrics."""
    avg = "binary" if task == "binary" else "macro"
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
    }

    if y_proba is not None:
        try:
            if task == "binary":
                # Probability of positive class
                score_col = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                roc = roc_auc_score(y_true, score_col)
            else:
                roc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            metrics["roc_auc"] = round(float(roc), 4)
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def class_imbalance_note(y: pd.Series) -> str:
    """Generate explanatory note regarding class imbalance and metric selection."""
    counts = y.value_counts(normalize=True)
    min_share = float(counts.min()) if not counts.empty else 0.5
    if min_share < 0.4:
        return (
            f"Minority class represents {min_share * 100:.1f}% of samples. In clinical risk prediction, "
            "standard accuracy can be deceptive; F1 score, Recall (minimising missed high-risk patients), "
            "and ROC-AUC provide more reliable measures of screening efficacy."
        )
    return "Classes are relatively balanced across the evaluation split."


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[Any],
    title: str = "Confusion Matrix",
    path: Optional[str] = None,
) -> plt.Figure:
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Ground Truth", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()

    if path:
        out_p = Path(path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=150)
        plt.close(fig)

    return fig


def plot_roc_curves(results: Dict[str, Dict[str, Any]], path: Optional[str] = None) -> plt.Figure:
    """Plot ROC curves for all models on a single figure."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name, res in results.items():
        y_true = res.get("y_true")
        y_proba = res.get("y_proba")
        display_name = res.get("display_name", model_name)
        auc_score = res.get("test_metrics", res.get("metrics", {})).get("roc_auc")

        if y_true is not None and y_proba is not None:
            proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            fpr, tpr, _ = roc_curve(y_true, proba)
            lbl = f"{display_name} (AUC = {auc_score:.3f})" if auc_score else display_name
            ax.plot(fpr, tpr, lw=2, label=lbl)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance (AUC = 0.50)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if path:
        out_p = Path(path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=150)
        plt.close(fig)

    return fig


def plot_metric_comparison(metrics_df: pd.DataFrame, path: Optional[str] = None) -> plt.Figure:
    """Plot grouped bar chart comparing performance across all four candidate models."""
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics_melted = metrics_df.melt(
        id_vars=["model"],
        value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"],
        var_name="Metric",
        value_name="Score",
    )
    sns.barplot(data=metrics_melted, x="Metric", y="Score", hue="model", palette="Set2", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Model Benchmark Comparison on Test Set", fontweight="bold")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if path:
        out_p = Path(path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=150)
        plt.close(fig)

    return fig


def plot_feature_importance(
    model_name: str, pipeline: Any, feature_names: List[str], path: Optional[str] = None
) -> Optional[plt.Figure]:
    """Extract and plot top feature importances or coefficients."""
    if not hasattr(pipeline, "named_steps") or "clf" not in pipeline.named_steps:
        return None

    clf = pipeline.named_steps["clf"]
    importances = None

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        metric_title = "Feature Importance (Gini / MDI)"
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
        metric_title = "Absolute Coefficient Magnitude"
    else:
        # KNN does not have direct feature importances
        return None

    if len(importances) != len(feature_names):
        # Fallback if transformed names differ slightly
        feature_names = [f"F_{i}" for i in range(len(importances))]

    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    imp_df = imp_df.sort_values(by="Importance", ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=imp_df, x="Importance", y="Feature", color="#2563eb", ax=ax)
    ax.set_title(f"Top Predictors: {model_name} ({metric_title})", fontweight="bold")
    ax.set_xlabel(metric_title, fontweight="bold")
    plt.tight_layout()

    if path:
        out_p = Path(path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=150)
        plt.close(fig)

    return fig
