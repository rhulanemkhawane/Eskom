from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
PREDICTIONS_DIR = MODELS_DIR / "predictions"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


@dataclass(frozen=True)
class ProjectConfig:
    # Paths
    proj_root: Path = PROJ_ROOT
    data_dir: Path = DATA_DIR
    raw_data_dir: Path = RAW_DATA_DIR
    interim_data_dir: Path = INTERIM_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR
    external_data_dir: Path = EXTERNAL_DATA_DIR
    models_dir: Path = MODELS_DIR
    predictions_dir: Path = PREDICTIONS_DIR
    reports_dir: Path = REPORTS_DIR
    figures_dir: Path = FIGURES_DIR

    # Eskom data
    eskom_raw_filename: str = "ESK17472.csv"
    eskom_processed_filename: str = "eskom_processed.parquet"
    timestamp_col: str = "Date Time Hour Beginning"
    residual_demand_col: str = "Residual Demand"
    total_re_col: str = "Total RE"
    target_col: str = "Total Energy Demand"
    freq: str = "H"
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

    # Meteostat settings
    meteostat_lat: Optional[float] = -26.2023
    meteostat_lon: Optional[float] = 28.0436
    meteostat_timezone: str = "Africa/Johannesburg"
    weather_vars: List[str] = field(
        default_factory=lambda: ["temp", "dwpt", "rhum", "prcp", "wspd", "pres"]
    )
    weather_prefix: str = "weather_"
    weather_impute_method: str = "ffill"  # options: ffill, interpolate, none
    weather_impute_limit: int = 6

    # Features
    target_lags: List[int] = field(default_factory=lambda: [1, 2, 3, 24, 48, 168])
    rolling_windows: List[int] = field(default_factory=lambda: [24, 168])

    # Metrics
    mape_method: str = "exclude_zeros"  # options: exclude_zeros, epsilon
    mape_epsilon: float = 1e-6

    # Runtime
    run_test_eval: bool = False
    random_seed: int = 42

    # Output files
    fold_metrics_filename: str = "fold_metrics.csv"
    metrics_summary_filename: str = "metrics_summary.json"
    final_model_filename: str = "final_model.pkl"
    split_log_filename: str = "split_sanity_checks.txt"

    def to_dict(self) -> dict:
        return asdict(self)


CONFIG = ProjectConfig()


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
