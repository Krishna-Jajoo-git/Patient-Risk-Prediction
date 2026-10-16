# BUILD SPECIFICATION: Patient Health Risk Prediction (Python + Streamlit + Colab)

> **How to use this document:** Give this entire file to an AI coding agent as its task brief.
> The agent must read all of it before writing code, then build both deliverables in Section 3
> and finish by meeting every item in the Acceptance Checklist (Section 13).

---

## 1. Project summary

Build an academic, supervised machine-learning system that predicts a patient health-risk
outcome from structured, de-identified medical records, optionally using longitudinal
(repeated, dated) records per patient.

- **Title:** Machine Learning-Based Patient Health Risk Prediction Using Longitudinal Medical Records
- **Task:** Binary (default) or multiclass classification. The target comes from the dataset.
- **Models to compare (exactly these four):** KNN, Decision Tree, Logistic Regression, Random Forest.
- **Audience:** Technical reviewers evaluating data cleaning, model comparison, UI/deployment,
  code quality, report quality and understanding of the ML pipeline.
- **Language/stack:** Python 3.10+, pandas, numpy, scikit-learn, matplotlib, seaborn, plotly,
  streamlit, joblib, pyyaml, pytest.

This is an **academic project, not a diagnostic system**. Predictions are model outputs, not medical advice.

---

## 2. Non-negotiable rules

1. **Target must come from the dataset.** Never define the target as "BP is increasing / Improving / Stable / Deteriorating" by rule. That would be a rule-based trend detector, not ML.
2. **Longitudinal changes are input features, never the label.** Compute them automatically in the backend. The UI must never ask the user to type "BP change", "Glucose change", etc.
3. **No data leakage.**
   - Split train/test **before** fitting any imputer, scaler or encoder.
   - All preprocessing lives inside a single scikit-learn `Pipeline` / `ColumnTransformer` that is fit on training data only.
   - For longitudinal data, split **by patient**, never by row. The same patient must never appear in both train and test.
   - If the outcome has a date, only use records on or before the prediction/index date to build features.
4. **Same preprocessing at training and inference.** The saved artifact is the full fitted pipeline (preprocessing + model), not just the model.
5. **Real results only.** Never hard-code or invent metrics. Every number shown in the UI, README or notebook must come from an actual run.
6. **Dataset-agnostic via config.** All dataset-specific names (columns, target, paths) live in `config.yaml`. No column names hard-coded inside `src/` logic.
7. **Separate training from inference.** `train.py` trains and saves. `predict.py` only loads and predicts. `app.py` never trains.
8. **No secrets, no absolute paths.** Use `pathlib` and paths relative to the project root.
9. **Do not add** LLMs, RAG, vector DBs, OCR, React/Express, Docker/Kubernetes, databases, or auth. Out of scope for Version 1.
10. Use a fixed `random_state` (default 42) everywhere for reproducibility.

---

## 3. Deliverables

### Deliverable A: Full-stack Python project (GitHub-ready)
A runnable project with data pipeline, training code, saved model, tests, and a Streamlit app. Structure in Section 5.

### Deliverable B: Google Colab notebook (`Patient_Risk_Prediction_Colab.ipynb`)
A **single self-contained notebook** containing the main ML logic, written to run top-to-bottom in Google Colab with clear explanations for each step. Specification in Section 9.

Both deliverables must use the **same logic and the same function names**, so what is shown in Colab matches the repository code.

---

## 4. Dataset handling

The final public dataset is **not yet chosen**. Build the system so the dataset can be swapped by editing `config.yaml` only.

### 4.1 Two supported modes
| Mode | Description | Example shape |
|---|---|---|
| `longitudinal` | Multiple dated records per patient. Features are aggregated per patient into one row. | `Patient_ID, Date, Age, Gender, BMI, BP, Glucose, Cholesterol, Medication, Outcome` |
| `single_record` | One row per patient. No aggregation. | `Age, Gender, BMI, BP, Glucose, ..., Outcome` |

### 4.2 Demo dataset (for testing the pipeline only)
Because the real dataset is not locked, create `src/generate_demo_data.py` that writes a **synthetic** longitudinal dataset to `data/raw/demo_longitudinal.csv`.

Requirements for the generator:
- ~600 patients, 3 to 8 visits each, monthly-ish dates.
- Columns: `patient_id, date, age, gender, bmi, systolic_bp, glucose, cholesterol, medication_count, outcome`.
- `outcome` (0/1) generated from a **noisy latent risk score** (age, mean BMI, mean BP, mean glucose, plus random noise) so it is not a deterministic rule. Roughly 25 to 35% positive class (imbalanced).
- Deliberately inject ~5% missing values, a few duplicate rows, and a few impossible values (e.g. BP = 0, BMI = 250) so the cleaning code has real work to do.
- Print and write a clear banner/comment: **"SYNTHETIC DATA: for pipeline testing only. Replace with a real public dataset before reporting any results."**
- The README and the Streamlit footer must also say when the demo dataset is in use.

### 4.3 Swapping in the real dataset
Document in the README: place the real CSV in `data/raw/`, edit `config.yaml`, run `python -m src.train`. No code changes should be needed. If the real data has a different shape than the two modes, state that clearly in the README.

### 4.4 `config.yaml` schema (create with these keys)
```yaml
dataset:
  name: "demo_longitudinal (SYNTHETIC)"
  path: "data/raw/demo_longitudinal.csv"
  mode: "longitudinal"            # longitudinal | single_record
  patient_id_col: "patient_id"    # longitudinal only
  date_col: "date"                # longitudinal only
  target_col: "outcome"
  task: "binary"                  # binary | multiclass
  positive_label: 1               # binary only
  static_numeric: ["age"]                        # taken as-is (latest value)
  static_categorical: ["gender"]
  time_varying_numeric: ["bmi", "systolic_bp", "glucose", "cholesterol", "medication_count"]
  valid_ranges:                   # values outside become NaN (impossible-value handling)
    age: [0, 120]
    bmi: [10, 70]
    systolic_bp: [60, 260]
    glucose: [30, 600]
    cholesterol: [50, 500]
    medication_count: [0, 30]

split:
  test_size: 0.2
  random_state: 42
  stratify: true
  group_by_patient: true          # forced true in longitudinal mode

preprocessing:
  numeric_imputer: "median"
  categorical_imputer: "most_frequent"
  scaler: "standard"
  outlier_strategy: "valid_ranges_then_iqr_report"   # report IQR outliers, do not delete rows silently

training:
  cv_folds: 5
  primary_metric: "f1"            # f1 | f1_macro | roc_auc | recall
  param_grids:
    knn:                 { clf__n_neighbors: [3, 5, 7, 11, 15], clf__weights: ["uniform", "distance"] }
    decision_tree:       { clf__max_depth: [3, 5, 8, null], clf__min_samples_leaf: [1, 5, 10] }
    logistic_regression: { clf__C: [0.01, 0.1, 1, 10], clf__class_weight: [null, "balanced"] }
    random_forest:       { clf__n_estimators: [100, 300], clf__max_depth: [null, 8, 12], clf__min_samples_leaf: [1, 3] }

paths:
  processed_dir: "data/processed"
  models_dir: "models"
  reports_dir: "reports"
```
Validate the config on load (missing keys, unknown mode, columns absent from the CSV) and raise clear, human-readable errors.

---

## 5. Repository structure (create exactly this)

```
patient-risk-prediction/
├── app.py                          # Streamlit UI (inference only)
├── config.yaml
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── raw/                        # demo_longitudinal.csv (generated) + README.md on how to add real data
│   └── processed/                  # cleaned.csv, patient_features.csv, demo_patients.csv
├── notebooks/
│   └── Patient_Risk_Prediction_Colab.ipynb     # Deliverable B
├── src/
│   ├── __init__.py
│   ├── config.py                   # load + validate config.yaml
│   ├── generate_demo_data.py
│   ├── data_loader.py              # load csv, schema validation, template csv
│   ├── preprocessing.py            # cleaning + sklearn ColumnTransformer builder
│   ├── feature_engineering.py      # longitudinal aggregation
│   ├── train.py                    # split, compare 4 models, save artifacts
│   ├── evaluation.py               # metrics, confusion matrix, ROC, plots
│   └── predict.py                  # load artifacts, validate input, predict
├── models/                         # best_model.joblib, metrics.json, model_metadata.json (created by train.py)
├── reports/
│   └── figures/                    # EDA + evaluation PNGs saved by code
├── tests/
│   ├── test_feature_engineering.py
│   ├── test_preprocessing.py
│   ├── test_split_no_leakage.py
│   └── test_predict_roundtrip.py
└── docs/
    └── architecture.md             # pipeline diagram (mermaid or ASCII)
```

---

## 6. Module specifications

Use type hints and docstrings. Keep functions small and pure where possible. Log with the `logging` module, not bare prints (except the CLI summary).

### 6.1 `src/config.py`
- `load_config(path="config.yaml") -> dict`: parse, validate, return.
- `PROJECT_ROOT` via `pathlib.Path(__file__).resolve().parents[1]`.

### 6.2 `src/data_loader.py`
- `load_raw(cfg) -> pd.DataFrame`
- `validate_schema(df, cfg, require_target: bool) -> list[str]`: returns a list of problems (missing columns, wrong dtypes, unparseable dates). Empty list means valid.
- `get_upload_template(cfg) -> pd.DataFrame`: a 2-row example frame with the correct columns, used by the "Download CSV template" button in the UI.
- `dataset_summary(df, cfg) -> dict`: rows, columns, patients, records per patient, class balance, missing count per column, date range. Used by the notebook and the report's dataset table.

### 6.3 `src/preprocessing.py`
- `clean_data(df, cfg) -> tuple[pd.DataFrame, dict]`
  Steps in order:
  1. Parse dates, strip whitespace/standardise categorical casing.
  2. Drop exact duplicate rows (record how many).
  3. Set values outside `valid_ranges` to NaN (record counts per column).
  4. Compute an IQR outlier **report** per numeric column (count only; do not remove).
  5. Return the cleaned frame and a `cleaning_report` dict with before/after missing counts per column, duplicates removed, impossible values nulled, and outlier counts.
  
  Do **not** impute here. Imputation happens inside the sklearn pipeline so it is fit on training data only.
- `build_preprocessor(cfg, numeric_cols, categorical_cols) -> ColumnTransformer`
  - Numeric: `SimpleImputer(median)` → `StandardScaler`.
  - Categorical: `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")`.
- `plot_missing_before_after(report, path)` saves a bar chart.

### 6.4 `src/feature_engineering.py`
- `build_patient_features(df, cfg) -> pd.DataFrame`
  - **Longitudinal mode:** sort by patient and date, then group by `patient_id` and produce **one row per patient** with:
    - For each `time_varying_numeric` column `X`: `X_latest`, `X_first`, `X_mean`, `X_min`, `X_max`, `X_std` (pandas default `ddof=1`; set to 0.0 when only one observation), `X_change` (latest minus first, ignoring NaN), `X_pct_change` (change / first, NaN if first is 0 or missing).
    - Global: `n_records`, `span_days` (last date minus first date), `mean_gap_days` (0 if single record).
    - Static columns: latest non-null value of each `static_numeric` and `static_categorical`.
    - The target: the patient-level outcome (last non-null value, or as defined by config). Document the assumption.
  - **Single-record mode:** return the frame unchanged (selecting configured features and target).
- `get_feature_lists(cfg) -> tuple[list, list]`: returns the numeric and categorical engineered feature names so the preprocessor and the app agree.
- `summarize_patient(patient_df, cfg) -> dict`: values shown on the UI summary card (average, latest, change for each key vital).

**Worked example the tests must reproduce** (this is also shown in the notebook): a patient with systolic BP over 4 visits = `125, 130, 138, 145` gives `latest = 145`, `first = 125`, `mean = 134.5`, `change = 20`, `pct_change = 0.16`, `std ≈ 8.81` (ddof=1).

### 6.5 `src/evaluation.py`
- `compute_metrics(y_true, y_pred, y_proba, task) -> dict`: accuracy, precision, recall, F1 (binary or macro), ROC-AUC (when probabilities are available).
- `plot_confusion_matrix(y_true, y_pred, labels, title, path)`
- `plot_roc_curves(results, path)`: all four models on one plot (binary).
- `plot_metric_comparison(metrics_df, path)`: grouped bar chart.
- `plot_feature_importance(model_name, pipeline, feature_names, path)`: tree importances for RF/DT, absolute coefficients for Logistic Regression, skip for KNN.
- `class_imbalance_note(y) -> str`: returns a sentence explaining why F1/recall matter when the minority share is below 40%.

### 6.6 `src/train.py` (`python -m src.train`)
1. Load config and raw data, validate schema.
2. `clean_data` → save `data/processed/cleaned.csv` and `reports/cleaning_report.json`.
3. `build_patient_features` → save `data/processed/patient_features.csv`.
4. Split: use `GroupShuffleSplit` by patient in longitudinal mode, otherwise `train_test_split(stratify=y)`. Assert zero patient overlap and print train/test class balance.
5. For each of the four models build `Pipeline([("prep", build_preprocessor(...)), ("clf", Model())])`, run `GridSearchCV` (stratified CV; group-aware if possible, else document the limitation) on the **training set only**, scoring with `primary_metric`.
6. Evaluate each tuned model on the held-out test set. Save per-model metrics, confusion matrices and the ROC plot.
7. Select the best model by `primary_metric` on the test set (tie-break: higher ROC-AUC, then simpler model). Document in the README that selection used the test set and that a nested or validation split would be stronger.
8. Save: `models/best_model.joblib` (full pipeline), `models/metrics.json` (all four models, best params, CV scores, test scores), `models/model_metadata.json` (best model name, feature names, class labels, config snapshot, training date, sklearn version, dataset name and whether it is synthetic), and `data/processed/demo_patients.csv` (a small sample of raw records for the "existing patient" mode; the sampled patients come from the **test** split).
9. Print a clean results table at the end.

### 6.7 `src/predict.py`
- `load_artifacts() -> tuple[Pipeline, dict, dict]`: model, metadata, metrics. Raise a friendly error if `models/best_model.joblib` is missing ("Run `python -m src.train` first").
- `validate_input(df, cfg) -> list[str]`: reuse `validate_schema` with `require_target=False`.
- `predict_from_history(history_df, cfg, model) -> dict`: clean → engineer features (single patient) → `predict` + `predict_proba` → return `{"prediction", "label_text", "probability", "class_probabilities", "features_used", "patient_summary"}`.
- `predict_from_features(features_df, model) -> pd.DataFrame` for batch use.
- Handle a patient with a single record, missing optional columns and all-NaN columns without crashing.

---

## 7. Streamlit application (`app.py`)

**Rules:** inference only. Load the model with `@st.cache_resource`. Wrap the whole app in friendly error handling (no raw stack traces for users). Display a persistent disclaimer: *"Academic project. Predictions are model outputs, not a medical diagnosis or advice."* If the model metadata says the dataset is synthetic, show a visible banner: *"Demo model trained on SYNTHETIC data."*

### Layout
Sidebar: title, input-mode radio (`Select Existing Patient` / `Upload Patient History`), dataset name, best model name.

Main area uses three tabs: **Prediction**, **Model Performance**, **About / Method**.

### Mode 1: Select Existing Patient
1. Dropdown of patient IDs from `data/processed/demo_patients.csv`.
2. Show the patient's raw history table.
3. Button **Analyze**.

### Mode 2: Upload Patient History
1. File uploader (CSV) and a **Download CSV template** button (from `get_upload_template`).
2. Validate with `validate_input`. Show specific, readable errors (e.g. "Missing column: glucose"). Stop cleanly on invalid input.
3. If the file contains several patients, let the user choose one patient ID.
4. Button **Analyze**.

### Single-record mode
If `config.dataset.mode == "single_record"`, replace both modes with one form generated **dynamically from config** (numeric inputs bounded by `valid_ranges`, select boxes for categorical columns). Do not hard-code field names.

### Prediction tab output (after Analyze)
1. **Patient summary cards** (`st.metric`): average, latest and change for each key time-varying vital, plus number of records and the time span. These are **calculated** values.
2. **ML prediction:** risk label, probability (progress bar plus %), and the model name. Add a caption: "Probability is a model output, not medical certainty."
3. **Trend charts** (Plotly): one line chart per time-varying feature across visits, with markers.
4. **Calculated features** in an expander (the exact vector sent to the model).

### Model Performance tab
Read `models/metrics.json` and show the four-model comparison table, the metric bar chart, ROC plot, confusion matrix of the best model (images from `reports/figures/`), best hyper-parameters, and feature importance.

### About / Method tab
Short description of the pipeline, target definition (from config), dataset source, preprocessing, limitations and safety notes. Pull from config and metadata, do not hard-code.

---

## 8. Tests (`pytest`)

| File | What it checks |
|---|---|
| `test_feature_engineering.py` | The BP worked example (`125,130,138,145`) gives the expected latest, mean, change, pct_change, std; a single-record patient yields std 0 and change 0; output has exactly one row per patient. |
| `test_preprocessing.py` | Impossible values (BP = 0) become NaN; duplicates are removed; the preprocessor handles unseen categories; a pipeline fit on train never sees test statistics (compare fitted scaler mean to train mean). |
| `test_split_no_leakage.py` | In longitudinal mode, the intersection of train and test patient IDs is empty; class ratio is approximately preserved. |
| `test_predict_roundtrip.py` | After training on the demo data: `predict_from_history` returns a valid label and a probability in [0, 1]; invalid schema returns errors, not exceptions. |

All tests must pass with `pytest -q` from the project root, after `python -m src.generate_demo_data && python -m src.train`.

---

## 9. Google Colab notebook specification (Deliverable B)

**File:** `notebooks/Patient_Risk_Prediction_Colab.ipynb`, created with `nbformat`.

### Hard requirements
- **Fully self-contained.** All functions are defined inside the notebook (no `import src...`), using the **same function names** as `src/` so it maps one-to-one to the repository.
- Runs top-to-bottom on a fresh Colab runtime with **no edits**, using only pip-installable libraries available on Colab. Include a first cell `!pip -q install ...` for anything missing.
- **Data source cell** with three options, switchable by a variable `DATA_SOURCE = "demo" | "upload" | "drive"`:
  - `"demo"`: generates the synthetic dataset inline (with the SYNTHETIC banner).
  - `"upload"`: uses `google.colab.files.upload()`.
  - `"drive"`: mounts Google Drive and reads a path variable.
- A **CONFIG cell** near the top (a Python dict mirroring `config.yaml`) where the student sets column names, target and valid ranges.
- Every section starts with a **Markdown cell**: what this step does, **why** it is needed, and the one-sentence takeaway. Use plain, simple English and short sentences.
- Every code cell has brief comments. Outputs (tables, plots) must be visible and readable.
- Use `random_state=42`. Run the notebook end to end before finishing and make sure it has **no errors** (use `nbconvert --execute` or equivalent, then keep the executed outputs).

### Section outline (one Markdown + one or more code cells each)
0. **Title and project overview.** Problem, why ML (not rules), the four models, safety note.
1. **Setup.** Installs and imports.
2. **Configuration.**
3. **Load data.** The three-source switch, then `dataset_summary` shown as a table (this is the "Dataset summary table" for the report).
4. **Why ML and not rules** (mini demo). Show the BP example `125,130,138,145`: a rule can say "increasing", but only a model trained on many patients with known outcomes can predict risk.
5. **Data validation.** `validate_schema`, dtypes, shape.
6. **Exploratory Data Analysis.** Class distribution, feature distributions (histograms), correlation heatmap, longitudinal trend plot for a few sample patients (coloured by outcome), missing-value bar chart. Explain each in one line.
7. **Data cleaning.** `clean_data` with before/after missing-value chart and the cleaning report table (duplicates, impossible values, IQR outlier counts).
8. **Longitudinal feature engineering.** Run the worked BP example first, then `build_patient_features` for all patients. Show the engineered table head and explain each feature type.
9. **Train/test split.** Patient-level split, plus a printed check that no patient is in both sets. A Markdown cell explaining **data leakage** in simple words.
10. **Preprocessing pipeline.** Build the `ColumnTransformer` + explain imputation, scaling, one-hot encoding, and **why KNN and Logistic Regression need scaling**.
11. **Model training.** One short Markdown explanation for each model (how it works, strength, weakness, in 2 to 3 sentences): KNN, Decision Tree, Logistic Regression, Random Forest. Then train with `GridSearchCV` on the training set; print best parameters and CV score per model.
12. **Evaluation.** Test-set metrics table for all four models (accuracy, precision, recall, F1, ROC-AUC), metric comparison bar chart, all-model ROC plot, confusion matrix for each model. Include `class_imbalance_note` and explain precision vs recall in healthcare terms (a missed high-risk patient vs a false alarm).
13. **Best model and interpretation.** Pick the best by the primary metric, show feature importance / coefficients, discuss why it likely won, and name the limitations of this comparison.
14. **Save and reload.** Save the full pipeline with joblib, reload it, and verify identical predictions.
15. **"What happens when a new patient is entered?"** End-to-end demo: take a small dated history for one new patient (typed inline as a DataFrame) → clean → engineer features → predict → show probability and a trend plot. This mirrors what the Streamlit app does.
16. **Limitations and ethics.** Synthetic vs real data, bias, missingness, generalisation, no clinical validation, need for external validation.
17. **Optional:** a cell that exports `best_model.joblib` and `metrics.json` via `files.download` so they can be dropped into the Streamlit project's `models/` folder.

---

## 10. Supporting files

### `requirements.txt` (pin minimum versions)
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
plotly>=5.18
streamlit>=1.30
joblib>=1.3
pyyaml>=6.0
pytest>=7.4
nbformat>=5.9
```

### `.gitignore`
Python cache, `.venv/`, `.ipynb_checkpoints/`, `.DS_Store`, `.env`, `*.log`. Keep `models/` and `data/raw/demo_longitudinal.csv` **tracked** (the app and demo need them); ignore any other large raw data files with a comment explaining why.

### `README.md` must contain
1. Project title, one-paragraph description, safety disclaimer.
2. Pipeline diagram (copy from `docs/architecture.md`).
3. Setup: `python -m venv`, `pip install -r requirements.txt`.
4. Quick start:
   ```
   python -m src.generate_demo_data
   python -m src.train
   pytest -q
   streamlit run app.py
   ```
5. Dataset section: source, size, columns, target definition, license, and a **prominent note when the demo synthetic data is in use**; instructions to swap in a real dataset via `config.yaml`.
6. Methodology summary (cleaning, features, split, four models, metrics).
7. Results table, **auto-generated from `models/metrics.json`** by a small helper script or a clearly marked section updated after training (no invented numbers).
8. Colab notebook link/instructions and how it maps to `src/`.
9. Deployment on Streamlit Community Cloud (steps: push to GitHub, select `app.py`, make sure `models/best_model.joblib` is committed).
10. Limitations and future scope (SHAP explainability, prescription/lab OCR integration, hospital data, fairness evaluation, time-series models after a reliable baseline, clinical validation).

### `docs/architecture.md`
A pipeline diagram:
```
Dataset → Validation & Cleaning → EDA → Feature Engineering → Train/Test Split (by patient)
→ Preprocessing (fit on train only) → KNN | Decision Tree | Logistic Regression | Random Forest
→ Metrics + Confusion Matrices → Best Model + Pipeline → joblib → Streamlit Inference UI
```

---

## 11. Code quality standards
- Meaningful names, small functions, type hints, docstrings, no dead code.
- No duplicated logic between `train.py`, `predict.py` and `app.py`. Shared logic lives in `feature_engineering.py` and `preprocessing.py`.
- Never use `print` for application logic (use `logging`), no wildcard imports, no hard-coded absolute paths.
- Clear error messages for end users in the UI.
- Make small, meaningful git commits if the agent has git access (e.g. "feat: add feature engineering", "feat: add training pipeline").

---

## 12. Recommended order of work for the agent
1. Create the folder structure, `requirements.txt`, `.gitignore`, `config.yaml`, `src/config.py`.
2. `generate_demo_data.py`, then generate the CSV.
3. `data_loader.py`, `preprocessing.py`, `feature_engineering.py`, plus their tests.
4. `evaluation.py`, `train.py`. Run training; confirm artifacts and figures exist.
5. `predict.py` plus its test.
6. `app.py`. Launch it, test all input modes, including a deliberately bad CSV.
7. Colab notebook. Build, execute end to end, fix errors.
8. README, `docs/architecture.md`. Fill the results table from real `metrics.json`.
9. Run the full Acceptance Checklist.

---

## 13. Acceptance checklist (the agent must verify every item and report it)

**Project**
- [ ] `python -m src.generate_demo_data && python -m src.train` runs without errors on a clean environment.
- [ ] `pytest -q` passes all tests.
- [ ] `streamlit run app.py` launches; both input modes work; invalid CSV shows a friendly error; prediction, summary cards, trend charts and the performance tab all render.
- [ ] No leakage: patient-level split verified; preprocessing fit on train only.
- [ ] All four models trained, tuned and compared with accuracy, precision, recall, F1, ROC-AUC, and confusion matrices.
- [ ] `models/best_model.joblib`, `metrics.json` and `model_metadata.json` exist and are consistent.
- [ ] No hard-coded column names in `src/` logic (everything flows from `config.yaml`).
- [ ] No absolute paths, no secrets.
- [ ] UI never asks the user to type engineered features such as "BP change".
- [ ] Synthetic-data banner appears wherever the demo dataset is used.

**Colab notebook**
- [ ] Runs top-to-bottom on a fresh Colab runtime with `DATA_SOURCE="demo"` and no errors.
- [ ] Contains all sections with clear Markdown explanations.
- [ ] Function names match `src/`.
- [ ] Includes the BP worked example, the leakage explanation, the four-model comparison and the new-patient end-to-end demo.
- [ ] Executed outputs are saved in the notebook.

**Documentation**
- [ ] README complete per Section 10, with real (not invented) results.
- [ ] Architecture diagram present.

### Final report from the agent
When finished, the agent must reply with: (1) a file tree, (2) the exact commands to run the project, (3) the real results table, (4) any assumptions made, (5) known limitations, and (6) what the student must do next (choose and plug in the real dataset, retrain, re-run the notebook, update the README).

---

## 14. Safety and academic limitation (include verbatim in README and app)
> This is an academic machine-learning project, not a diagnostic or treatment system. It must use only public, de-identified data. Predictions are model outputs, not medical advice. Results depend on the training data, may be biased or fail to generalise, and would require external and clinical validation before any real-world use.
