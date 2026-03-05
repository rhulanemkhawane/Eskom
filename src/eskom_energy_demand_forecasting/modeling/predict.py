from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger
import typer

from eskom_energy_demand_forecasting.config import CONFIG
from eskom_energy_demand_forecasting.features import build_ml_features

app = typer.Typer()


def _load_config_from_artifact(models_dir: Path) -> dict:
    config_path = models_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError("config.json not found in models directory.")
    return json.loads(config_path.read_text())


def _calendar_features_for_ts(ts: pd.Timestamp) -> dict:
    hour = ts.hour
    dayofweek = ts.dayofweek
    month = ts.month
    return {
        "hour": hour,
        "dayofweek": dayofweek,
        "month": month,
        "is_weekend": 1 if dayofweek >= 5 else 0,
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "dow_sin": float(np.sin(2 * np.pi * dayofweek / 7)),
        "dow_cos": float(np.cos(2 * np.pi * dayofweek / 7)),
    }


def _lag_and_roll_features(history: pd.Series, config=CONFIG) -> dict:
    feats = {}
    for lag in config.target_lags:
        feats[f"lag_{lag}"] = float(history.iloc[-lag]) if len(history) >= lag else np.nan
    for window in config.rolling_windows:
        if len(history) >= window:
            window_vals = history.iloc[-window:]
            feats[f"roll_mean_{window}"] = float(window_vals.mean())
            feats[f"roll_std_{window}"] = float(window_vals.std())
        else:
            feats[f"roll_mean_{window}"] = np.nan
            feats[f"roll_std_{window}"] = np.nan
    return feats


def _build_feature_row(
    ts: pd.Timestamp,
    history: pd.Series,
    feature_columns: list[str],
    config=CONFIG,
) -> pd.DataFrame:
    row = {}
    row.update(_calendar_features_for_ts(ts))
    row.update(_lag_and_roll_features(history, config))

    return pd.DataFrame([row], columns=feature_columns)


def _recursive_predict(
    model,
    history: pd.Series,
    forecast_idx: pd.DatetimeIndex,
    feature_columns: list[str],
    config=CONFIG,
) -> pd.Series:
    preds = []
    hist = history.copy()
    for ts in forecast_idx:
        X_row = _build_feature_row(ts, hist, feature_columns, config)
        pred = float(model.predict(X_row)[0])
        preds.append(pred)
        hist = pd.concat([hist, pd.Series([pred], index=[ts])])
    return pd.Series(preds, index=forecast_idx)


@app.command()
def main(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> None:
    models_dir = CONFIG.models_dir
    model_path = models_dir / CONFIG.final_model_filename
    if not model_path.exists():
        raise FileNotFoundError("Final model not found. Run training first.")

    config_payload = _load_config_from_artifact(models_dir)
    logger.info(f"Loaded model config from {models_dir / 'config.json'}")

    df = pd.read_parquet(CONFIG.processed_data_dir / CONFIG.eskom_processed_filename)
    df = df.sort_index()
    last_ts = df.index.max()

    if start is None:
        start_ts = df.index.min()
    else:
        parsed = pd.Timestamp(start)
        start_ts = parsed.tz_localize(CONFIG.timezone) if parsed.tzinfo is None else parsed.tz_convert(CONFIG.timezone)

    if end is None:
        end_ts = last_ts
    else:
        parsed = pd.Timestamp(end)
        end_ts = parsed.tz_localize(CONFIG.timezone) if parsed.tzinfo is None else parsed.tz_convert(CONFIG.timezone)

    model_artifact = joblib.load(model_path)
    model_name = model_artifact["model"]
    if model_name == "ETS":
        raise ValueError("ETS model does not support direct feature-based prediction.")

    model = model_artifact["fit"]
    df_hist = df.loc[:last_ts]
    X_hist, _ = build_ml_features(df_hist, CONFIG, drop_target_na=False)
    feature_columns = list(X_hist.columns)

    if end_ts <= last_ts:
        X = X_hist.loc[start_ts:end_ts]
        preds = pd.Series(model.predict(X), index=X.index, name="prediction")
    else:
        preds_list = []
        idx_list = []

        if start_ts <= last_ts:
            X_in = X_hist.loc[start_ts:last_ts]
            preds_in = pd.Series(model.predict(X_in), index=X_in.index)
            preds_list.append(preds_in)
            idx_list.append(X_in.index)

        forecast_start = max(start_ts, last_ts + pd.Timedelta(hours=1))
        forecast_idx = pd.date_range(forecast_start, end_ts, freq=CONFIG.freq, tz=CONFIG.timezone)
        y_history = df_hist[CONFIG.target_col].dropna()
        preds_future = _recursive_predict(model, y_history, forecast_idx, feature_columns, CONFIG)
        preds_list.append(preds_future)
        idx_list.append(forecast_idx)

        preds = pd.concat(preds_list).sort_index()
        preds.name = "prediction"

    output_path = CONFIG.models_dir / "predictions_latest.csv"
    preds.to_csv(output_path, index=True)
    logger.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    app()
