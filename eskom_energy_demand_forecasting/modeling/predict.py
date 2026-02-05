from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
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


@app.command()
def main(
    start: Optional[str] = None,
    end: Optional[str] = None,
    future_weather_path: Optional[Path] = None,
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

    if end_ts > last_ts:
        if future_weather_path is None:
            raise ValueError(
                "Forecast end is beyond available data. Provide future_weather_path "
                "or limit the forecast range."
            )
        future_weather = pd.read_parquet(future_weather_path)
        future_weather = future_weather.sort_index()
        df = pd.concat([df, future_weather], axis=0)

    X, y = build_ml_features(df, CONFIG, include_weather=True, drop_target_na=False)
    X = X.loc[start_ts:end_ts]

    model_artifact = joblib.load(model_path)
    model_name = model_artifact["model"]
    if model_name == "ETS":
        raise ValueError("ETS model does not support direct feature-based prediction.")

    model = model_artifact["fit"]
    preds = pd.Series(model.predict(X), index=X.index, name="prediction")

    output_path = CONFIG.models_dir / "predictions_latest.csv"
    preds.to_csv(output_path, index=True)
    logger.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    app()
