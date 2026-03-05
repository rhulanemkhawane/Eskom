from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from loguru import logger
import yaml

# Load environment variables from .env file if it exists
load_dotenv()


_ENV_BOOL_TRUE = {"1", "true", "yes"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return value.lower() in _ENV_BOOL_TRUE if value not in (None, "") else default


def _resolve_path(path_value: str | Path, proj_root: Path) -> Path:
    candidate = Path(path_value)
    return candidate if candidate.is_absolute() else proj_root / candidate


# Paths
PROJ_ROOT = Path(__file__).resolve().parents[2]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DEFAULT_CONFIG_PATH = PROJ_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class ProjectConfig:
    # Paths
    proj_root: Path = PROJ_ROOT
    data_dir: Path = PROJ_ROOT / "data"
    raw_data_dir: Path = PROJ_ROOT / "data" / "raw"
    interim_data_dir: Path = PROJ_ROOT / "data" / "interim"
    processed_data_dir: Path = PROJ_ROOT / "data" / "processed"
    external_data_dir: Path = PROJ_ROOT / "data" / "external"
    models_dir: Path = PROJ_ROOT / "models"
    predictions_dir: Path = PROJ_ROOT / "models" / "predictions"
    reports_dir: Path = PROJ_ROOT / "reports"
    figures_dir: Path = PROJ_ROOT / "reports" / "figures"

    # Eskom data
    eskom_raw_filename: str = "ESK17472.csv"
    eskom_processed_filename: str = "eskom_processed.parquet"
    timestamp_col: str = "Date Time Hour Beginning"
    timestamp_format: Optional[str] = "%Y-%m-%d %I:%M:%S %p"
    residual_demand_col: str = "Residual Demand"
    total_re_col: str = "Total RE"
    target_col: str = "Total Energy Demand"
    freq: str = "h"
    timezone: str = "Africa/Johannesburg"

    # Split boundaries (inclusive end timestamps). Set explicitly before training.
    train_end: Optional[str] = "2024-09-30 03:00:00+02:00"
    val_end: Optional[str] = "2025-07-01 00:00:00+02:00"
    test_end: Optional[str] = "2026-03-31 23:00:00+02:00"

    # Rolling-origin backtesting
    initial_train_days: int = 365
    horizon_hours: int = 24
    step_hours: int = 24
    n_folds: int = 6
    expanding_window: bool = True
    sliding_window_days: Optional[int] = None
    seasonal_period: int = 168  # weekly

    # Features
    target_lags: List[int] = field(default_factory=lambda: [1, 2, 3, 24, 48, 168])
    rolling_windows: List[int] = field(default_factory=lambda: [24, 168])

    # Metrics
    mape_method: str = "exclude_zeros"  # options: exclude_zeros, epsilon
    mape_epsilon: float = 1e-6

    # Runtime
    run_test_eval: bool = False
    random_seed: int = 42
    enable_tree_model: bool = True
    enable_elasticnet: bool = True
    enable_ets: bool = True
    enable_lightgbm: bool = False
    enable_xgboost: bool = True
    enable_lstm: bool = True

    # LSTM settings
    lstm_sequence_length: int = 336
    lstm_units: int = 64
    lstm_dense_units: int = 32
    lstm_dropout: float = 0.2
    lstm_epochs: int = 30
    lstm_batch_size: int = 128
    lstm_patience: int = 6
    lstm_validation_split: float = 0.1

    # Output files
    fold_metrics_filename: str = "fold_metrics.csv"
    metrics_summary_filename: str = "metrics_summary.json"
    final_model_filename: str = "final_model.pkl"
    split_log_filename: str = "split_sanity_checks.txt"

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> "ProjectConfig":
        config_path = path or Path(os.getenv("ESKOM_CONFIG_PATH", DEFAULT_CONFIG_PATH))
        if not config_path.exists():
            raise FileNotFoundError(f"Config YAML file not found: {config_path}")

        raw = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config YAML must contain a mapping/object: {config_path}")

        base = asdict(cls())
        base.update(raw)

        proj_root = _resolve_path(base.get("proj_root", PROJ_ROOT), PROJ_ROOT)
        base["proj_root"] = proj_root

        path_fields = {
            "data_dir",
            "raw_data_dir",
            "interim_data_dir",
            "processed_data_dir",
            "external_data_dir",
            "models_dir",
            "predictions_dir",
            "reports_dir",
            "figures_dir",
        }
        for field_name in path_fields:
            base[field_name] = _resolve_path(base[field_name], proj_root)

        # Optional environment variable overrides for runtime tuning.
        base["initial_train_days"] = _env_int("INITIAL_TRAIN_DAYS", int(base["initial_train_days"]))
        base["horizon_hours"] = _env_int("HORIZON_HOURS", int(base["horizon_hours"]))
        base["step_hours"] = _env_int("STEP_HOURS", int(base["step_hours"]))
        base["n_folds"] = _env_int("N_FOLDS", int(base["n_folds"]))

        base["enable_tree_model"] = _env_bool("ENABLE_TREE_MODEL", bool(base["enable_tree_model"]))
        base["enable_elasticnet"] = _env_bool("ENABLE_ELASTICNET", bool(base["enable_elasticnet"]))
        base["enable_ets"] = _env_bool("ENABLE_ETS", bool(base["enable_ets"]))
        base["enable_lightgbm"] = _env_bool("ENABLE_LIGHTGBM", bool(base["enable_lightgbm"]))
        base["enable_xgboost"] = _env_bool("ENABLE_XGBOOST", bool(base["enable_xgboost"]))
        base["enable_lstm"] = _env_bool("ENABLE_LSTM", bool(base["enable_lstm"]))

        base["lstm_sequence_length"] = _env_int(
            "LSTM_SEQUENCE_LENGTH", int(base["lstm_sequence_length"])
        )
        base["lstm_units"] = _env_int("LSTM_UNITS", int(base["lstm_units"]))
        base["lstm_dense_units"] = _env_int("LSTM_DENSE_UNITS", int(base["lstm_dense_units"]))
        base["lstm_dropout"] = _env_float("LSTM_DROPOUT", float(base["lstm_dropout"]))
        base["lstm_epochs"] = _env_int("LSTM_EPOCHS", int(base["lstm_epochs"]))
        base["lstm_batch_size"] = _env_int("LSTM_BATCH_SIZE", int(base["lstm_batch_size"]))
        base["lstm_patience"] = _env_int("LSTM_PATIENCE", int(base["lstm_patience"]))
        base["lstm_validation_split"] = _env_float(
            "LSTM_VALIDATION_SPLIT", float(base["lstm_validation_split"])
        )

        return cls(**base)

    def to_dict(self) -> dict:
        return asdict(self)


CONFIG = ProjectConfig.from_yaml()


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
