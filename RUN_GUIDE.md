# Program Execution Guide: Patient Health Risk Prediction System

This guide is organized into three dedicated sections:
* **Part A:** Complete setup from scratch (prerequisites, Python 3.11 venv creation, dependency installation, model compilation, tests, and execution).
* **Part B:** Instant fast-track terminal commands for daily development and server launches on `localhost`.
* **Part C:** Deep-dive: How the Benchmark Metrics Table works (Dynamic vs. Static lifecycle explained).

---

# Part A: Fresh Setup from Scratch (Environment & Execution)

Use this section if you are setting up the project on a new system, after cloning, or inside a clean directory.

### A.1 Prerequisites
Ensure your machine has the following installed:
1. **Python 3.11** (Recommended and required for scikit-learn 1.6.1 binary wheels).
   * Verify via: `py -3.11 --version` or `python --version`
2. **Git** (Optional, for version control).
3. **PowerShell** (Windows) or **Bash / Zsh** (Linux / macOS).

---

### A.2 Clone / Open Project Directory
Navigate into the root directory of the project:

```powershell
# Windows PowerShell
cd "C:\Users\Krishna\Desktop\ml project"
```

```bash
# Linux / macOS
cd /path/to/"ml project"
```

---

### A.3 Create an Isolated Virtual Environment (Python 3.11)
Create a clean virtual environment named `.venv` explicitly targeting Python 3.11:

```powershell
# Windows PowerShell (ensures Python 3.11 is selected even if 3.14+ is installed)
py -3.11 -m venv .venv
```

```bash
# Linux / macOS
python3.11 -m venv .venv
```

---

### A.4 Activate the Virtual Environment

* **Windows PowerShell:**
  ```powershell
  # If PowerShell blocks script execution, run this once:
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

  # Activate .venv
  .\.venv\Scripts\Activate.ps1
  ```

* **Windows Command Prompt (CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

* **Linux / macOS (Bash/Zsh):**
  ```bash
  source .venv/bin/activate
  ```

*(When activated, your terminal prompt will display `(.venv)` in front of the prompt).*

---

### A.5 Upgrade pip & Install Project Dependencies
Install all required libraries specified in [`requirements.txt`](requirements.txt):

```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
```

```bash
# Linux / macOS
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
```

---

### A.6 (Optional) Generate or Re-scale Dataset
If you want to generate a fresh synthetic longitudinal research cohort with 5,000 patients:

```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe src/generate_demo_data.py --patients 5000
```

```bash
# Linux / macOS
./.venv/bin/python src/generate_demo_data.py --patients 5000
```

---

### A.7 Train and Compile the Machine Learning Pipeline
Before launching the web app for the first time, compile the data preprocessing pipeline and train the benchmark models:

```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe src/train.py
```

```bash
# Linux / macOS
./.venv/bin/python src/train.py
```

**What this command does:**
1. Loads the longitudinal dataset from `data/raw/demo_longitudinal.csv`.
2. Validates schema and applies biological plausibility filters.
3. Computes longitudinal trajectory features (latest, baseline delta, variance, temporal spans).
4. Performs a zero-leakage `GroupShuffleSplit` on `patient_id`.
5. Trains and cross-validates 4 candidate models: **KNN**, **Decision Tree**, **Logistic Regression**, and **Random Forest**.
6. Selects the champion model based on **F1-Score** and serializes artifacts to:
   * `models/best_model.joblib`
   * `models/model_metadata.json`
   * `models/metrics.json`
   * `reports/figures/roc_curves.png`
   * `reports/figures/cm_*.png`

---

### A.8 Run Automated Verification Tests
Verify pipeline integrity and ensure all components pass unit tests:

```powershell
# Windows PowerShell
.\.venv\Scripts\pytest.exe
```

```bash
# Linux / macOS
./.venv/bin/pytest
```

*(All 14 test suites covering data ingestion, imputation, feature engineering, zero-leakage splitting, and model inference will execute and pass).*

---

### A.9 Launch the Streamlit Web Application on Localhost
Start the interactive clinical dashboard:

```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

```bash
# Linux / macOS
./.venv/bin/streamlit run app.py
```

* **Localhost URL:** Open [http://localhost:8501](http://localhost:8501) in your browser.
* **Network URL:** Accessible on your local Wi-Fi network (displayed in terminal output).

---

# Part B: Fast-Track Quick Start (Environment Already Set Up)

> **For returning developers:** If your virtual environment (`.venv`) is already created, use these exact commands.

---

### B.1 The Single One-Liner (Instant App Launch)
Open Windows PowerShell inside the project directory and paste:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

* The server starts immediately and opens your browser at **`http://localhost:8501`**.

---

### B.2 Full End-to-End Pipeline (Generate + Retrain + Test + Launch)
If you updated the raw dataset or modified pipeline code and want to re-run everything end-to-end:

```powershell
.\.venv\Scripts\python.exe src/train.py; .\.venv\Scripts\pytest.exe; .\.venv\Scripts\python.exe -m streamlit run app.py
```

---

### B.3 Alternative Port or Custom Host Binding
If port `8501` is already in use by another service:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

To run in headless mode without popping up the default browser automatically:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true
```

---

### B.4 How to Stop the Engine
To shut down the running server at any time:
* Press **`Ctrl + C`** in the active terminal window.

---

# Part C: Is the Benchmark Metrics Table Static or Dynamic?

### Answer: The table is **100% Dynamic** (Driven by Trained Model Artifacts).

The table labeled **"2. Standardized Holdout Metrics Table"** under the **"Model Benchmarking & Performance"** tab is **NOT hardcoded**. Here is how its data lifecycle works:

```
+------------------------------------+
|  1. Raw Dataset                    |
|  (data/raw/demo_longitudinal.csv)  |
+------------------------------------+
                 │
                 ▼  (Triggered by: `python src/train.py`)
+------------------------------------+
|  2. Training & Cross-Validation    |
|  - 5-Fold CV on 4,000 patients     |
|  - Test evaluation on 1,000 holdout|
|    (KNN, Decision Tree, LogReg, RF)|
+------------------------------------+
                 │
                 ▼  (Calculates fresh metrics)
+------------------------------------+
|  3. Disk Artifact                  |
|  (models/metrics.json)             |
|  Stores: accuracy, f1, precision,  |
|  recall, roc_auc, cv_score         |
+------------------------------------+
                 │
                 ▼  (Read dynamically by Streamlit)
+------------------------------------+
|  4. Web Application (app.py)       |
|  - Renders multi-metric bar chart  |
|  - Builds Holdout Metrics Table    |
|  - Highlights champion in teal     |
+------------------------------------+
```

### Does it update automatically when the dataset changes?

1. **If you replace or edit the CSV file alone:**
   - Machine learning evaluation scores cannot calculate themselves on raw data without training.
   - The table will continue displaying the metrics of the **currently trained model artifact**.

2. **When you re-run the training script:**
   ```powershell
   .\.venv\Scripts\python.exe src/train.py
   ```
   - `train.py` ingests the new dataset, fits all 4 algorithms on the new data, and overwrites `models/metrics.json` and `models/best_model.joblib`.
   
3. **Reflecting the changes in the Web App:**
   - `app.py` automatically checks the modification timestamp of `models/metrics.json`. When it sees a newer file timestamp, it **automatically invalidates the cache and reloads the new metrics**.
   - You can also click the **"🔄 Reload Model Artifacts"** button in the sidebar or press **'R'** in the browser to immediately refresh the table and graphs.

---

### Quick Reference Command Matrix

| Purpose | Windows PowerShell Command |
| :--- | :--- |
| **Launch Web App (Instant)** | `.\.venv\Scripts\python.exe -m streamlit run app.py` |
| **Retrain Models & Update Metrics** | `.\.venv\Scripts\python.exe src/train.py` |
| **Generate/Rescale Dataset (5k Patients)** | `.\.venv\Scripts\python.exe src/generate_demo_data.py --patients 5000` |
| **Run All 14 Automated Tests** | `.\.venv\Scripts\pytest.exe` |
| **End-to-End Retrain, Test & Launch** | `.\.venv\Scripts\python.exe src/train.py; .\.venv\Scripts\pytest.exe; .\.venv\Scripts\python.exe -m streamlit run app.py` |
| **Docker Compose Launch** | `docker compose up` |
| **Local Dashboard URL** | [http://localhost:8501](http://localhost:8501) |
