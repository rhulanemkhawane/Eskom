from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger
import typer

from eskom_energy_demand_forecasting.config import CONFIG

app = typer.Typer()


def _read_eskom_csv(path: Path) -> pd.DataFrame:
    import csv

    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        first_row = next(reader)

    names = header
    if len(first_row) > len(header):
        extras = [f"extra_col_{i+1}" for i in range(len(first_row) - len(header))]
        names = header + extras
        logger.warning(
            "Eskom CSV has more columns than header. Added extra columns: {}",
            extras,
        )

    df = pd.read_csv(path, names=names, skiprows=1, engine="python")
    return df


def load_eskom_data(config=CONFIG) -> pd.DataFrame:
    candidates: List[Path] = [
        config.processed_data_dir / config.eskom_processed_filename,
        config.raw_data_dir / config.eskom_raw_filename,
        config.external_data_dir / config.eskom_raw_filename,
    ]
    input_path: Optional[Path] = None
    for p in candidates:
        if p.exists():
            input_path = p
            break

    if input_path is None:
        raise FileNotFoundError(
            "Eskom data file not found. Checked: "
            + ", ".join(str(p) for p in candidates)
        )

    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = _read_eskom_csv(input_path)

    if config.timestamp_col in df.columns:
        df[config.timestamp_col] = pd.to_datetime(
            df[config.timestamp_col], errors="coerce"
        )
        df = df.dropna(subset=[config.timestamp_col])
        df = df.set_index(config.timestamp_col).sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            f"Timestamp column '{config.timestamp_col}' not found and index is not datetime."
        )

    if df.index.duplicated().any():
        dup_count = int(df.index.duplicated().sum())
        raise ValueError(f"Duplicate timestamps found in Eskom data: {dup_count}")

    if config.timezone:
        if df.index.tz is None:
            df.index = df.index.tz_localize(
                config.timezone, nonexistent="shift_forward", ambiguous="NaT"
            )
        else:
            df.index = df.index.tz_convert(config.timezone)

    expected_index = pd.date_range(
        df.index.min(), df.index.max(), freq=config.freq, tz=df.index.tz
    )
    missing = expected_index.difference(df.index)
    if len(missing) > 0:
        raise ValueError(
            f"Eskom data has missing timestamps. Missing count: {len(missing)}"
        )

    df = df.reindex(expected_index)
    return df


def engineer_target(df: pd.DataFrame, config=CONFIG) -> pd.DataFrame:
    for col in (config.residual_demand_col, config.total_re_col):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in Eskom data.")
    df = df.copy()
    df[config.target_col] = df[config.residual_demand_col] + df[config.total_re_col]
    return df


def fetch_meteostat_hourly(
    start: pd.Timestamp,
    end: pd.Timestamp,
    config=CONFIG,
) -> pd.DataFrame:
    if config.meteostat_lat is None or config.meteostat_lon is None:
        raise ValueError(
            "Meteostat lat/lon are not set. Please set CONFIG.meteostat_lat "
            "and CONFIG.meteostat_lon before fetching weather."
        )

    try:
        from meteostat import Hourly, Point
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "meteostat is not installed. Add it to requirements.txt and install."
        ) from exc

    point = Point(config.meteostat_lat, config.meteostat_lon, tz=config.meteostat_timezone)
    weather = Hourly(point, start, end, tz=config.meteostat_timezone).fetch()
    if weather.empty:
        raise ValueError("Meteostat returned empty weather data.")

    if config.weather_vars:
        keep_cols = [c for c in config.weather_vars if c in weather.columns]
        weather = weather[keep_cols]

    weather = weather.sort_index()
    if config.timezone and weather.index.tz is not None:
        weather.index = weather.index.tz_convert(config.timezone)

    weather = weather.rename(columns={c: f"{config.weather_prefix}{c}" for c in weather.columns})
    return weather


def merge_eskom_weather(
    eskom_df: pd.DataFrame, weather_df: pd.DataFrame, config=CONFIG
) -> pd.DataFrame:
    df = eskom_df.join(weather_df, how="left")

    weather_cols = [c for c in df.columns if c.startswith(config.weather_prefix)]
    if weather_cols:
        if config.weather_impute_method == "ffill":
            df[weather_cols] = df[weather_cols].ffill(limit=config.weather_impute_limit)
        elif config.weather_impute_method == "interpolate":
            df[weather_cols] = df[weather_cols].interpolate(
                method="time", limit=config.weather_impute_limit
            )
        elif config.weather_impute_method == "none":
            pass
        else:
            raise ValueError(
                f"Unknown weather_impute_method: {config.weather_impute_method}"
            )

    return df


def save_processed_dataset(df: pd.DataFrame, config=CONFIG) -> Path:
    config.processed_data_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.processed_data_dir / config.eskom_processed_filename
    df.to_parquet(output_path, index=True)
    return output_path


@app.command()
def main() -> None:
    logger.info("Loading Eskom data...")
    df = load_eskom_data(CONFIG)
    logger.info("Engineering target...")
    df = engineer_target(df, CONFIG)
    logger.info("Fetching Meteostat weather...")
    weather = fetch_meteostat_hourly(df.index.min(), df.index.max(), CONFIG)
    logger.info("Merging Eskom + weather...")
    df = merge_eskom_weather(df, weather, CONFIG)
    output_path = save_processed_dataset(df, CONFIG)
    logger.success(f"Processed dataset saved to {output_path}")


if __name__ == "__main__":
    app()
