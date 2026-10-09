# Machine Learning-Based Patient Health Risk Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

> **Academic Project & Clinical Disclaimer:**  
> This is an academic machine-learning project, not a diagnostic or treatment system. It must use only public, de-identified data. Predictions are statistical model outputs, not medical advice. Results depend on the training data, may be biased or fail to generalize, and would require external and clinical validation before any real-world use.

---

## 1. Project Overview

This repository implements an academic, reproducible, supervised machine-learning system designed to predict patient health-risk outcomes from structured medical records. The pipeline specializes in **longitudinal patient trajectories**—automatically computing vital changes, percentage rates of change, visit variance, and cumulative exposure without requiring clinical users to manually calculate or type engineered features.

Four distinct classical machine-learning algorithms are systematically trained, cross-validated, tuned via grid search, and benchmarked on a strictly held-out test split:
1. **K-Nearest Neighbours (KNN)**
2. **Decision Tree Classifier**
3. **Logistic Regression**
4. **Random Forest Classifier**

---

## 2. Architecture & Pipeline

```mermaid
flowchart TD
    A[Raw Longitudinal Dataset] --> B[Schema Validation & Range Checks]
    B --> C[Data Cleaning & Deduplication]
    C --> D[Longitudinal Feature Aggregation]
    D --> E[Patient-Level Train/Test Split (No Leakage)]
    E --> F[ColumnTransformer: Imputation + Scaling/Encoding]
    F --> G[5-Fold Stratified CV Grid Search]
    G --> H[Model Benchmark on Test Set]
    H --> I[Artifact Serialization (models/best_model.joblib)]
    I --> J[Streamlit UI & Interactive Dashboard]
```

---

## 3. Quick Start Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.11)

### Installation
```bash
# Clone the repository and enter the directory
git clone <repo-url>
cd patient-risk-prediction

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
.\.venv\Scripts\activate        # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Reproduce End-to-End Pipeline
```bash
# 1. Generate synthetic demonstration dataset
python -m src.generate_demo_data

# 2. Run data cleaning, feature engineering, and model training
python -m src.train

# 3. Execute automated test suite
pytest -q

# 4. Launch interactive Streamlit web application
streamlit run app.py
```

---

## 4. Benchmark Evaluation Results

The four candidate models were trained and cross-validated on the training split (2,400 patients), then evaluated on the strictly held-out test cohort (600 patients, patient-grouped via GroupShuffleSplit, zero patient leakage). The empirical results obtained from `models/metrics.json`:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Best Hyperparameters |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **K-Nearest Neighbours** | 0.7083 | 0.7367 | 0.8870 | 0.8049 | 0.6924 | `n_neighbors: 15, weights: 'uniform', metric: 'euclidean'` |
| **Decision Tree** | 0.6750 | 0.7477 | 0.7862 | 0.7665 | 0.6878 | `max_depth: 5, min_samples_leaf: 20, criterion: 'entropy'` |
| **Logistic Regression** | 0.7133 | 0.7364 | **0.8993** | 0.8097 | **0.7447** | `C: 0.001, penalty: 'l2', solver: 'lbfgs'` |
| **Random Forest (Winner)** | **0.7250** | **0.7597** | 0.8698 | **0.8110** | 0.7436 | `n_estimators: 100, max_depth: 8, min_samples_leaf: 3, max_features: 'log2'` |

*Selected Best Model:* **Random Forest** achieved the highest test F1-Score (**0.8110**) alongside strong test Accuracy (**72.50%**), Precision (**75.97%**), Recall (**86.98%**), and ROC-AUC (**0.7436**), effectively stratifying risk on the longitudinal test cohort.

---

## 5. Dataset Configuration & Real Data Integration

> **Notice:** The default cohort `data/raw/demo_longitudinal.csv` is **SYNTHETIC** and provided for pipeline demonstration only.

### Swapping in a Real Dataset
To use a real de-identified public dataset (e.g., MIMIC-IV, NHANES, or PhysioNet):
1. Place the CSV file in `data/raw/<dataset_name>.csv`.
2. Open [`config.yaml`](file:///c:/Users/Krishna/OneDrive/Desktop/ml%20project%202/config.yaml) and update:
   - `dataset.name`: Descriptive name of your cohort
   - `dataset.path`: `"data/raw/<dataset_name>.csv"`
   - `dataset.mode`: `"longitudinal"` or `"single_record"`
   - `dataset.target_col`: Target label column
   - Feature lists (`static_numeric`, `static_categorical`, `time_varying_numeric`)
   - `valid_ranges`: Clinically plausible boundaries for outlier detection
3. Retrain and regenerate all models and figures:
   ```bash
   python -m src.train
   ```
No code modifications in `src/` are required.

---

## 6. Project Structure

```
patient-risk-prediction/
├── app.py                          # Streamlit UI (Inference & Dashboard)
├── config.yaml                     # Centralized dataset & training configuration
├── requirements.txt                # Pinned dependencies
├── README.md                       # Comprehensive project documentation
├── .gitignore                      # Git ignore rules
├── data/
│   ├── raw/                        # demo_longitudinal.csv + instructions
│   └── processed/                  # cleaned.csv, patient_features.csv, demo_patients.csv
├── notebooks/
│   └── Patient_Risk_Prediction_Colab.ipynb  # Self-contained Colab notebook
├── src/
│   ├── __init__.py
│   ├── config.py                   # Config parser and schema validator
│   ├── generate_demo_data.py       # Synthetic data generator
│   ├── data_loader.py              # CSV loader, schema verification, templates
│   ├── preprocessing.py            # Data cleaning & ColumnTransformer builder
│   ├── feature_engineering.py      # Longitudinal aggregations & patient KPI cards
│   ├── evaluation.py               # Metrics computation & publication figures
│   ├── train.py                    # Cross-validation, model comparison, artifact saver
│   └── predict.py                  # Leak-free inference engine
├── models/
│   ├── best_model.joblib           # Full serialized scikit-learn pipeline
│   ├── metrics.json                # Verified benchmark metrics across models
│   └── model_metadata.json         # Feature list and training run metadata
├── reports/
│   └── figures/                    # Generated ROC curves, confusion matrices, etc.
├── tests/
│   ├── test_feature_engineering.py # Verifies worked examples & single visits
│   ├── test_preprocessing.py       # Verifies range clipping & no leakage
│   ├── test_split_no_leakage.py    # Verifies patient separation across splits
│   └── test_predict_roundtrip.py   # Verifies end-to-end inference & error handling
└── docs/
    └── architecture.md             # Detailed pipeline documentation & diagrams
```

---

## 7. Deliverable B: Google Colab Notebook

The self-contained notebook [`notebooks/Patient_Risk_Prediction_Colab.ipynb`](file:///c:/Users/Krishna/OneDrive/Desktop/ml%20project%202/notebooks/Patient_Risk_Prediction_Colab.ipynb) is organized into 18 faculty-friendly sections:
- **Zero-Dependency Setup:** Defines all logic inline matching `src/` function signatures.
- **Three Data Source Modes:** Switch between `"demo"`, Google Colab file `"upload"`, or Google `"drive"`.
- **Viva Cheat-Sheet:** Includes direct Q&A for faculty defense on data leakage, scaling rationale, and metric choice.

---

## 8. Deployment on Streamlit Community Cloud

1. Commit and push the repository to GitHub:
   ```bash
   git add .
   git commit -m "feat: complete patient risk prediction system"
   git push origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io).
3. Connect your repository, select branch `main`, and specify `app.py` as the entry file.
4. The application will launch with zero additional configuration needed.

---

## 9. Academic Limitations & Future Scope

- **Clinical Validation:** The synthetic dataset simulates biological noise but lacks institutional multi-center validation.
- **Explainability:** Future iterations will incorporate SHAP (SHapley Additive exPlanations) for local patient feature attributions.
- **Multi-Modal Data:** Extending tabular EHR with clinical notes via natural language processing or ECG waveforms using 1D convolutional networks.
#   P a t i e n t - R i s k - P r e d i c t i o n  
 