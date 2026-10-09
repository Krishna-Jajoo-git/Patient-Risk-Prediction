# System Architecture & ML Pipeline

This document outlines the end-to-end data processing, feature engineering, model training, and inference architecture of the **Patient Health Risk Prediction** system.

```mermaid
flowchart TD
    A[Raw Dataset (CSV)] --> B[Schema Validation & Type Checking]
    B --> C[Data Cleaning (Impossible Ranges -> NaN, Duplicates Dropped)]
    C --> D[Longitudinal Feature Engineering (Per-Patient Trajectories)]
    D --> E[Patient-Level Train/Test Split (GroupShuffleSplit)]
    
    subgraph Training Pipeline (Train Split Only)
        E --> F[ColumnTransformer: Median Imputation + StandardScaler / OneHotEncoder]
        F --> G[5-Fold Stratified Cross-Validation & Grid Search]
        G --> H[Candidate Models: KNN | Decision Tree | Logistic Regression | Random Forest]
    end
    
    subgraph Evaluation (Held-out Test Split)
        H --> I[Benchmark Test Metrics: Accuracy, Precision, Recall, F1, ROC-AUC]
        I --> J[Confusion Matrices, ROC Curves, Feature Importance Plots]
        J --> K[Select Best Estimator by Primary Metric (F1-score)]
    end
    
    subgraph Artifact Storage
        K --> L[models/best_model.joblib (Full Fitted Pipeline)]
        K --> M[models/metrics.json & model_metadata.json]
    end
    
    subgraph Production Deployment & Exploration
        L --> N[Streamlit Web Application (app.py)]
        L --> O[Interactive Colab Notebook]
    end
```

## Core Pipeline Invariants

1. **Strict Zero-Leakage Guarantee:**
   - In longitudinal mode, grouping is enforced on `patient_id` using `GroupShuffleSplit`. The intersection of train and test patient sets is strictly empty:
     $$\text{Patients}_{\text{train}} \cap \text{Patients}_{\text{test}} = \emptyset$$
   - Imputers (`SimpleImputer`), encoders (`OneHotEncoder`), and scalers (`StandardScaler`) are fit exclusively on the training partition.

2. **Unified Preprocessing & Inference:**
   - The serialized `best_model.joblib` artifact is the full Scikit-Learn `Pipeline` containing both the `ColumnTransformer` and the final estimator.
   - Inference on raw multi-visit patient records executes the identical transformation without manual feature entry.

3. **Multi-Model Algorithmic Paradigms:**
   - **K-Nearest Neighbours:** Non-parametric instance-based classifier.
   - **Decision Tree:** Single interpretable decision tree with regularized depth and leaf constraints.
   - **Logistic Regression:** Linear probabilistic log-odds classifier with L2 regularization.
   - **Random Forest:** Bagged ensemble of decision trees with feature sub-sampling.
