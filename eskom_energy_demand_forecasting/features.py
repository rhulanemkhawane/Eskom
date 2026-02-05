from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger
import typer

from eskom_energy_demand_forecasting.config import CONFIG

app = typer.Typer()


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=index)
    df["hour"] = index.hour
    df["dayofweek"] = index.dayofweek
    df["month"] = index.month
    df["is_weekend"] = (index.dayofweek >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def _lag_features(series: pd.Series, lags: list[int]) -> pd.DataFrame:
    df = pd.DataFrame(index=series.index)
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)
    return df


def _rolling_features(series: pd.Series, windows: list[int]) -> pd.DataFrame:
    df = pd.DataFrame(index=series.index)
    s_shift = series.shift(1)
    for window in windows:
        df[f"roll_mean_{window}"] = s_shift.rolling(window).mean()
        df[f"roll_std_{window}"] = s_shift.rolling(window).std()
    return df


def build_ml_features(
    df: pd.DataFrame,
    config=CONFIG,
    include_weather: bool = True,
    drop_target_na: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    if config.target_col not in df.columns:
        raise ValueError(f"Target column '{config.target_col}' not found in dataset.")

    index = df.index
    target = df[config.target_col]

    X = _calendar_features(index)
    X = X.join(_lag_features(target, config.target_lags))
    X = X.join(_rolling_features(target, config.rolling_windows))

    if include_weather:
        weather_cols = [c for c in df.columns if c.startswith(config.weather_prefix)]
        if weather_cols:
            X = X.join(df[weather_cols])
        else:
            logger.warning("No weather columns found while include_weather=True.")

    combined = X.join(target.rename("y"))
    if drop_target_na:
        combined = combined.dropna()
    else:
        combined = combined.dropna(subset=[c for c in combined.columns if c != "y"])
    X = combined.drop(columns=["y"])
    y = combined["y"]
    return X, y


def build_target_series(df: pd.DataFrame, config=CONFIG) -> pd.Series:
    if config.target_col not in df.columns:
        raise ValueError(f"Target column '{config.target_col}' not found in dataset.")
    return df[config.target_col].copy()


@app.command()
def main(
    input_path: Path = CONFIG.processed_data_dir / CONFIG.eskom_processed_filename,
    output_path: Path = CONFIG.processed_data_dir / "features.parquet",
) -> None:
    logger.info("Loading processed dataset for feature generation...")
    df = pd.read_parquet(input_path)
    X, y = build_ml_features(df, CONFIG, include_weather=True)
    features = X.join(y.rename(CONFIG.target_col))
    features.to_parquet(output_path, index=True)
    logger.success(f"Features saved to {output_path}")


if __name__ == "__main__":
    app()
