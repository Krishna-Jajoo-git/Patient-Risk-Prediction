"""Configuration loader and validator."""

from pathlib import Path
from typing import Any, Dict
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load and validate the YAML configuration file.

    Args:
        path: Relative or absolute path to the configuration file.

    Returns:
        dict: Validated configuration dictionary.

    Raises:
        FileNotFoundError: If the config file cannot be found.
        ValueError: If mandatory configuration keys are missing or invalid.
    """
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Invalid configuration file: Root content must be a dictionary.")

    _validate_config_structure(cfg)
    return cfg


def _validate_config_structure(cfg: Dict[str, Any]) -> None:
    """Validate required keys and schema logic."""
    required_sections = ["dataset", "split", "preprocessing", "training", "paths"]
    for sec in required_sections:
        if sec not in cfg:
            raise ValueError(f"Missing required configuration section: '{sec}'")

    dataset = cfg["dataset"]
    required_dataset_keys = [
        "name", "path", "mode", "target_col", "task",
        "static_numeric", "static_categorical", "time_varying_numeric"
    ]
    for key in required_dataset_keys:
        if key not in dataset:
            raise ValueError(f"Missing required key in dataset configuration: '{key}'")

    mode = dataset["mode"]
    if mode not in ("longitudinal", "single_record"):
        raise ValueError(
            f"Invalid dataset mode '{mode}'. Supported modes are: 'longitudinal', 'single_record'."
        )

    if mode == "longitudinal":
        if "patient_id_col" not in dataset or not dataset["patient_id_col"]:
            raise ValueError("Longitudinal mode requires 'patient_id_col' in config.yaml")
        if "date_col" not in dataset or not dataset["date_col"]:
            raise ValueError("Longitudinal mode requires 'date_col' in config.yaml")

    task = dataset.get("task", "binary")
    if task not in ("binary", "multiclass"):
        raise ValueError(f"Invalid task '{task}'. Supported tasks are: 'binary', 'multiclass'.")

    # Paths validation
    for path_key in ("processed_dir", "models_dir", "reports_dir"):
        if path_key not in cfg["paths"]:
            raise ValueError(f"Missing path config in 'paths': '{path_key}'")
