# README for real data

Place real CSV files here following the naming convention described below.

## How to add a real dataset

1. Copy your CSV file to this directory:
   ```
   data/raw/your_dataset.csv
   ```

2. Edit `config.yaml` at the project root:
   ```yaml
   dataset:
     name: "Your Dataset Name"
     path: "data/raw/your_dataset.csv"
     mode: "longitudinal"    # or "single_record"
     patient_id_col: "patient_id"
     date_col: "date"
     target_col: "your_outcome_column"
     ...
   ```

3. Update column lists in config.yaml to match your dataset's columns.

4. Retrain:
   ```bash
   python -m src.train
   ```

## Supported dataset shapes

| Mode | Required columns |
|---|---|
| `longitudinal` | `patient_id`, `date`, feature columns, `target` |
| `single_record` | feature columns, `target` (one row per patient) |

## Notes on large raw datasets

Files matching `data/raw/*.csv.gz` and `data/raw/real_*.csv` are excluded from
git tracking (see `.gitignore`) to prevent accidentally committing large data files.
The demo file `demo_longitudinal.csv` **is** tracked so the app works out of the box.
