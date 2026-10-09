# Program Execution Guide: Patient Health Risk Prediction System

This guide is organized into two dedicated sections:
* **Part A:** Complete from-scratch setup for new machines or clean environments (prerequisites, virtual environment creation, dependency installation, model compilation, tests, and execution).
* **Part B:** Instant fast-track terminal commands for developers who already have the environment set up and simply want to start all engines on `localhost`.

---

# Part A: Fresh Setup from Scratch (Environment & Execution)

Use this section if you are setting up the project on a new system, after cloning, or inside a clean directory.

### A.1 Prerequisites
Ensure your machine has the following installed:
1. **Python 3.10 or 3.11** (Verify via `python --version` or `py --version`).
2. **Git** (Optional, for version control).
3. **PowerShell** (Windows) or **Bash / Zsh** (Linux / macOS).

---

### A.2 Clone / Open Project Directory
Navigate into the root directory of the project:

```powershell
# Windows PowerShell
cd "C:\Users\Krishna\OneDrive\Desktop\ml project 2"
```

```bash
# Linux / macOS
cd /path/to/"ml project 2"
```

---

### A.3 Create an Isolated Virtual Environment
Create a clean virtual environment named `.venv`:

```powershell
# Windows
python -m venv .venv
```

```bash
# Linux / macOS
python3 -m venv .venv
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
Install all required libraries specified in [`requirements.txt`](file:///c:/Users/Krishna/OneDrive/Desktop/ml%20project%202/requirements.txt):

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### A.6 Train and Compile the Machine Learning Pipeline
Before launching the web app for the first time, compile the data preprocessing pipeline and train the benchmark models:

```bash
python src/train.py
```

**What this command does:**
1. Loads the longitudinal dataset from `data/raw/demo_longitudinal.csv`.
2. Validates schema and applies biological plausibility filters.
3. Computes longitudinal trajectory features (latest, baseline delta, variance, temporal spans).
4. Performs a zero-leakage `GroupShuffleSplit` on `patient_id`.
5. Trains and cross-validates 4 candidate models: **KNN**, **Decision Tree**, **Logistic Regression**, and **Random Forest**.
6. Selects the champion model based on **F1-Score** and serializes artifacts to:
   * `models/best_model.joblib`
   * `reports/benchmark_results.json`
   * `reports/roc_curve.png`
   * `reports/confusion_matrix_*.png`

---

### A.7 Run Automated Verification Tests
Verify pipeline integrity and ensure all components pass unit tests:

```bash
pytest tests/ -v
```

*(All 13 test suites covering data ingestion, imputation, feature engineering, zero-leakage splitting, and model inference should pass).*

---

### A.8 Launch the Streamlit Web Application on Localhost
Start the interactive clinical dashboard:

```bash
streamlit run app.py
```

* **Localhost URL:** Open [http://localhost:8501](http://localhost:8501) in your browser.
* **Network URL:** Accessible on your local Wi-Fi network (displayed in terminal output).

---

### A.9 (Optional) Run via Docker Container
If you prefer running inside an isolated container with Docker installed:

```bash
# Build and run using Docker Compose
docker compose up --build

# OR build and run with standard Docker CLI
docker build -t patient-risk-prediction .
docker run -p 8501:8501 patient-risk-prediction
```

Open [http://localhost:8501](http://localhost:8501).

---

# Part B: Fast-Track Quick Start (Environment Already Set Up)

> **For returning developers:** If your virtual environment (`.venv`) is already created and dependencies are installed, use these exact commands.

---

### B.1 The Single One-Liner (Fastest Method)
Open Windows PowerShell inside the project directory and paste:

```powershell
.\.venv\Scripts\python -m streamlit run app.py
```

* The server starts immediately.
* Opens your browser automatically at: **`http://localhost:8501`**

---

### B.2 Standard Two-Step Launch (Windows PowerShell)

```powershell
# Step 1: Activate .venv
.\.venv\Scripts\Activate.ps1

# Step 2: Start Streamlit App
streamlit run app.py
```

---

### B.3 Standard Two-Step Launch (Windows Command Prompt)

```cmd
:: Step 1: Activate .venv
.venv\Scripts\activate.bat

:: Step 2: Start Streamlit App
streamlit run app.py
```

---

### B.4 Full End-to-End Engine Pipeline (Retrain + Test + Launch)
If you updated the raw dataset or modified pipeline code and want to retrain the models, run tests, and launch the UI in a single automated chain:

```powershell
# Windows PowerShell
.\.venv\Scripts\python src/train.py; .\.venv\Scripts\pytest tests/ -v; .\.venv\Scripts\python -m streamlit run app.py
```

---

### B.5 Alternative Port or Custom Host Binding
If port `8501` is already in use by another service:

```powershell
.\.venv\Scripts\python -m streamlit run app.py --server.port 8502
```
Access at: **`http://localhost:8502`**

To run in headless mode without popping up the default browser automatically:

```powershell
.\.venv\Scripts\python -m streamlit run app.py --server.headless true
```

---

### B.6 How to Stop the Engine
To shut down the running server at any time:
* Press **`Ctrl + C`** in the active terminal window.

---

### Quick Reference Command Matrix

| Purpose | Windows PowerShell Command |
| :--- | :--- |
| **Start Web App (Instant)** | `.\.venv\Scripts\python -m streamlit run app.py` |
| **Retrain Models Only** | `.\.venv\Scripts\python src/train.py` |
| **Run Unit Tests Only** | `.\.venv\Scripts\pytest tests/ -v` |
| **Run Integration Tests** | `.\.venv\Scripts\python -m pytest tests/test_pipeline.py` |
| **Docker Compose** | `docker compose up` |
| **Default URL** | [http://localhost:8501](http://localhost:8501) |
