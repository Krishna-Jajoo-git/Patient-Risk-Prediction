# Model Performance Benchmark — Cardiovascular Disease Dataset (CardioTrain)

> Evaluated on a **strictly held-out 14,000-patient test set** (20% stratified test split from 70,000 real patient examinations, zero leakage).

---

## 📊 Results Comparison across 4 Supervised Architectures

| Model Architecture | Accuracy | Precision | Sensitivity / Recall | F1-Score (Primary) | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** *(Winner 🏆)* | **73.28%** | **0.7491** | **0.6995** | **0.7235** | **0.7972** |
| Logistic Regression | 72.97% | 0.7593 | 0.6722 | 0.7131 | 0.7937 |
| Decision Tree | 72.24% | 0.7367 | 0.6915 | 0.7134 | 0.7837 |
| K-Nearest Neighbours | 71.26% | 0.7190 | 0.6975 | 0.7081 | 0.7702 |

> [!IMPORTANT]
> **Random Forest is the winning champion estimator** with **Accuracy = 73.28%**, **ROC-AUC = 0.7972**, and **F1-Score = 0.7235**.
> All 4 models achieve consistent, well-calibrated discriminatory performance on this real cardiovascular cohort.

---

## 🔧 Cardiovascular Domain Feature Engineering

| Feature | Type | Clinical Description & Formula |
|---|---|---|
| **`age_years`** | Numeric | Normalized patient age in calendar years ($Age_{days} / 365.25$). |
| **`bmi`** | Numeric | Body Mass Index ($Weight_{kg} / (Height_m)^2$). |
| **`pulse_pressure`** | Numeric | Arterial pulse pressure ($SBP - DBP$). Established marker of arterial stiffness. |
| **`map`** | Numeric | Mean Arterial Perfusion Pressure ($DBP + \frac{1}{3}(SBP - DBP)$). |
| **`bp_category`** | Categorical | AHA/ACC Clinical Stages: Normal, Elevated, Stage 1 HTN, Stage 2 HTN, Crisis. |
| **`bmi_category`** | Categorical | WHO Weight Classification: Underweight, Normal, Overweight, Obese. |
| **`age_x_bp`** | Numeric | Non-linear interaction between age and systolic blood pressure. |
| **`composite_risk_score`** | Numeric | Multivariable Framingham-inspired cardiovascular risk index. |

---

## 🛡️ Data Quality & Zero Leakage Validation

1. **Blood Pressure Transposition Correction**: Cases where systolic and diastolic blood pressure were transposed during clinical data entry ($ap\_hi < ap\_lo$) were verified and corrected.
2. **Physiologically Impossible Outlier Handling**: Values outside clinically possible biological ranges (e.g., negative blood pressures, typing errors > 300 mmHg) are converted to `NaN` and imputed via `SimpleImputer(strategy='median')` strictly learned from training folds.
3. **Strict Stratified Holdout**: Class balance (~50% CVD positive, 50% healthy) is precisely preserved across training (56,000) and holdout test (14,000) partitions.

---

## 🚀 How to Run and Retrain

```bash
# Execute training pipeline (5-fold CV on 56,000 records, evaluates on 14,000 test cases)
python -m src.train

# Run unit test suite
python -m pytest

# Launch Streamlit Clinical Web Interface
streamlit run app.py
```
