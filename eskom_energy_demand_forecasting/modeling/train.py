from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
import typer

from eskom_energy_demand_forecasting.config import CONFIG
from eskom_energy_demand_forecasting.dataset import (
    engineer_target,
    load_eskom_data,
    save_processed_dataset,
)
from eskom_energy_demand_forecasting.features import build_ml_features, build_target_series
from eskom_energy_demand_forecasting.metrics import all_metrics

app = typer.Typer()


def _parse_ts(ts: str, tz: str) -> pd.Timestamp:
    parsed = pd.Timestamp(ts)
    if parsed.tzinfo is None:
        return parsed.tz_localize(tz)
    return parsed.tz_convert(tz)


def _hash_array(values: np.ndarray) -> str:
    return hashlib.md5(values.tobytes()).hexdigest()


def split_by_dates(df: pd.DataFrame, config=CONFIG) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not config.train_end or not config.val_end or not config.test_end:
        raise ValueError(
            "Split boundaries are not set. Please set CONFIG.train_end, CONFIG.val_end, CONFIG.test_end."
        )

    train_end = _parse_ts(config.train_end, config.timezone)
    val_end = _parse_ts(config.val_end, config.timezone)
    test_end = _parse_ts(config.test_end, config.timezone)

    if not (train_end < val_end < test_end):
        raise ValueError("Split boundaries must satisfy train_end < val_end < test_end.")

    val_start = train_end + pd.Timedelta(hours=1)
    test_start = val_end + pd.Timedelta(hours=1)

    df_train = df.loc[:train_end]
    df_val = df.loc[val_start:val_end]
    df_test = df.loc[test_start:test_end]

    if df_train.empty or df_val.empty or df_test.empty:
        raise ValueError("One or more splits are empty. Check split boundaries.")

    if df_train.index.max() >= df_val.index.min():
        raise ValueError("Train and validation splits overlap.")
    if df_val.index.max() >= df_test.index.min():
        raise ValueError("Validation and test splits overlap.")

    return df_train, df_val, df_test


def write_split_sanity_checks(
    df_val: pd.DataFrame, df_test: pd.DataFrame, output_path: Path, config=CONFIG
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    val_idx_hash = _hash_array(df_val.index.view("int64"))
    test_idx_hash = _hash_array(df_test.index.view("int64"))
    val_y_hash = _hash_array(df_val[config.target_col].to_numpy())
    test_y_hash = _hash_array(df_test[config.target_col].to_numpy())

    if val_idx_hash == test_idx_hash or val_y_hash == test_y_hash:
        raise ValueError("Validation and test splits appear identical.")

    content = [
        f"train_end: {config.train_end}",
        f"val_end: {config.val_end}",
        f"test_end: {config.test_end}",
        f"val_index_hash: {val_idx_hash}",
        f"test_index_hash: {test_idx_hash}",
        f"val_y_hash: {val_y_hash}",
        f"test_y_hash: {test_y_hash}",
    ]
    output_path.write_text("\n".join(content))


def make_backtest_folds(
    index: pd.DatetimeIndex, config=CONFIG
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    folds: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    initial_train_hours = config.initial_train_days * 24
    horizon = config.horizon_hours
    step = config.step_hours
    val_end_limit = _parse_ts(config.val_end, config.timezone)

    if len(index) < initial_train_hours + horizon:
        raise ValueError("Not enough data to create initial train + horizon window.")

    train_end = index[initial_train_hours - 1]
    for _ in range(config.n_folds):
        val_start = train_end + pd.Timedelta(hours=1)
        val_end = val_start + pd.Timedelta(hours=horizon - 1)
        if val_end > val_end_limit:
            break

        if config.expanding_window:
            train_start = index[0]
        else:
            if config.sliding_window_days is None:
                raise ValueError("sliding_window_days must be set when expanding_window=False.")
            window_hours = config.sliding_window_days * 24
            train_start = train_end - pd.Timedelta(hours=window_hours - 1)

        train_idx = index[(index >= train_start) & (index <= train_end)]
        val_idx = index[(index >= val_start) & (index <= val_end)]
        if len(val_idx) == horizon:
            folds.append((train_idx, val_idx))

        train_end = train_end + pd.Timedelta(hours=step)

    if not folds:
        raise ValueError("No backtest folds were created. Check parameters.")
    return folds


def _evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    config=CONFIG,
) -> Dict[str, float]:
    aligned = pd.concat([y_true, y_pred], axis=1).dropna()
    if aligned.empty:
        return {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
    metrics = all_metrics(
        aligned.iloc[:, 0].to_numpy(),
        aligned.iloc[:, 1].to_numpy(),
        config.mape_method,
        config.mape_epsilon,
    )
    return metrics


def _baseline_predictions(
    y_series: pd.Series, val_idx: pd.DatetimeIndex, config=CONFIG
) -> Dict[str, pd.Series]:
    preds = {}
    preds["Naive"] = y_series.shift(1).loc[val_idx]
    preds["SeasonalNaive"] = y_series.shift(config.seasonal_period).loc[val_idx]
    preds["RollingMean"] = (
        y_series.shift(1).rolling(config.seasonal_period).mean().loc[val_idx]
    )
    return preds


def _calendar_features_for_ts(ts: pd.Timestamp) -> Dict[str, float]:
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


def _lag_and_roll_features(history: pd.Series, config=CONFIG) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    for lag in config.target_lags:
        if len(history) >= lag:
            feats[f"lag_{lag}"] = float(history.iloc[-lag])
        else:
            feats[f"lag_{lag}"] = np.nan

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
    feature_columns: List[str],
    config=CONFIG,
) -> pd.DataFrame:
    row: Dict[str, float] = {}
    row.update(_calendar_features_for_ts(ts))
    row.update(_lag_and_roll_features(history, config))

    return pd.DataFrame([row], columns=feature_columns)


def _recursive_baseline_predictions(
    y_history: pd.Series, val_idx: pd.DatetimeIndex, config=CONFIG
) -> Dict[str, pd.Series]:
    results: Dict[str, List[float]] = {"Naive": [], "SeasonalNaive": [], "RollingMean": []}

    for model_name in results.keys():
        history = y_history.copy()
        preds: List[float] = []
        for ts in val_idx:
            if model_name == "Naive":
                pred = float(history.iloc[-1]) if len(history) >= 1 else np.nan
            elif model_name == "SeasonalNaive":
                if len(history) >= config.seasonal_period:
                    pred = float(history.iloc[-config.seasonal_period])
                else:
                    pred = np.nan
            else:  # RollingMean
                if len(history) >= config.seasonal_period:
                    pred = float(history.iloc[-config.seasonal_period:].mean())
                else:
                    pred = np.nan

            preds.append(pred)
            history = pd.concat([history, pd.Series([pred], index=[ts])])

        results[model_name] = preds

    return {k: pd.Series(v, index=val_idx) for k, v in results.items()}


def _recursive_model_predictions(
    model,
    y_history: pd.Series,
    val_idx: pd.DatetimeIndex,
    feature_columns: List[str],
    config=CONFIG,
) -> pd.Series:
    preds: List[float] = []
    history = y_history.copy()
    for ts in val_idx:
        X_row = _build_feature_row(ts, history, feature_columns, config)
        pred = float(model.predict(X_row)[0])
        preds.append(pred)
        history = pd.concat([history, pd.Series([pred], index=[ts])])
    return pd.Series(preds, index=val_idx)


def _get_tree_model():
    try:
        from lightgbm import LGBMRegressor

        return "LightGBM", LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=CONFIG.random_seed,
        )
    except ModuleNotFoundError:
        pass

    try:
        from xgboost import XGBRegressor

        return "XGBoost", XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=CONFIG.random_seed,
        )
    except ModuleNotFoundError:
        pass

    from sklearn.ensemble import HistGradientBoostingRegressor

    return "HistGB", HistGradientBoostingRegressor(random_state=CONFIG.random_seed)


def _train_ets(y_train: pd.Series, horizon: int) -> Optional[np.ndarray]:
    try:
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    except ModuleNotFoundError:
        logger.warning("statsmodels not installed; skipping ETS model.")
        return None

    try:
        model = ETSModel(
            y_train,
            error="add",
            trend="add",
            seasonal="add",
            seasonal_periods=CONFIG.seasonal_period,
        )
        fit = model.fit(disp=False)
        return fit.forecast(horizon).to_numpy()
    except Exception as exc:
        logger.warning(f"ETS failed to fit: {exc}")
        return None


def rolling_origin_backtest(
    df_trainval: pd.DataFrame, config=CONFIG
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    folds = make_backtest_folds(df_trainval.index, config)
    y_series = build_target_series(df_trainval, config)

    fold_metrics: List[Dict[str, object]] = []
    model_scores: Dict[str, List[float]] = {}

    predictions_dir = config.predictions_dir
    predictions_dir.mkdir(parents=True, exist_ok=True)

    for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
        logger.info(f"Backtest fold {fold_id}/{len(folds)}")
        y_train = y_series.loc[train_idx]
        y_val = y_series.loc[val_idx]
        df_train = df_trainval.loc[train_idx]

        # Baselines
        baseline_preds = _recursive_baseline_predictions(y_train, val_idx, config)
        for model_name, pred in baseline_preds.items():
            metrics = _evaluate_predictions(y_val, pred, config)
            metrics_row = {"fold": fold_id, "model": model_name, **metrics}
            fold_metrics.append(metrics_row)
            model_scores.setdefault(model_name, []).append(metrics["MAE"])
            pred_df = pd.DataFrame({"y_true": y_val, "y_pred": pred})
            pred_df.to_csv(predictions_dir / f"fold_{fold_id}_{model_name}.csv")

        # ETS
        if config.enable_ets:
            ets_pred = _train_ets(y_train, len(val_idx))
            if ets_pred is not None:
                ets_series = pd.Series(ets_pred, index=val_idx)
                metrics = _evaluate_predictions(y_val, ets_series, config)
                metrics_row = {"fold": fold_id, "model": "ETS", **metrics}
                fold_metrics.append(metrics_row)
                model_scores.setdefault("ETS", []).append(metrics["MAE"])
                pred_df = pd.DataFrame({"y_true": y_val, "y_pred": ets_series})
                pred_df.to_csv(predictions_dir / f"fold_{fold_id}_ETS.csv")

        # ML models
        X_train, y_train_ml = build_ml_features(df_train, config)

        if X_train.empty:
            logger.warning("Skipping ML models for fold due to empty feature set.")
            continue

        # Tree-based
        if config.enable_tree_model:
            tree_name, tree_model = _get_tree_model()
            tree_model.fit(X_train, y_train_ml)
            tree_pred = _recursive_model_predictions(
                tree_model, y_train, val_idx, list(X_train.columns), config
            )
            metrics = _evaluate_predictions(y_val, tree_pred, config)
            metrics_row = {"fold": fold_id, "model": tree_name, **metrics}
            fold_metrics.append(metrics_row)
            model_scores.setdefault(tree_name, []).append(metrics["MAE"])
            pred_df = pd.DataFrame({"y_true": y_val, "y_pred": tree_pred})
            pred_df.to_csv(predictions_dir / f"fold_{fold_id}_{tree_name}.csv")

        # ElasticNet
        if config.enable_elasticnet:
            from sklearn.linear_model import ElasticNet

            enet = ElasticNet(random_state=config.random_seed)
            enet.fit(X_train, y_train_ml)
            enet_pred = _recursive_model_predictions(
                enet, y_train, val_idx, list(X_train.columns), config
            )
            metrics = _evaluate_predictions(y_val, enet_pred, config)
            metrics_row = {"fold": fold_id, "model": "ElasticNet", **metrics}
            fold_metrics.append(metrics_row)
            model_scores.setdefault("ElasticNet", []).append(metrics["MAE"])
            pred_df = pd.DataFrame({"y_true": y_val, "y_pred": enet_pred})
            pred_df.to_csv(predictions_dir / f"fold_{fold_id}_ElasticNet.csv")

    metrics_df = pd.DataFrame(fold_metrics)
    summary = {
        model: {
            "MAE": float(np.nanmean(scores)),
        }
        for model, scores in model_scores.items()
    }
    return metrics_df, summary


def _select_final_model(summary: Dict[str, Dict[str, float]]) -> str:
    candidates = {k: v for k, v in summary.items() if k not in {"Naive", "SeasonalNaive", "RollingMean"}}
    if not candidates:
        raise ValueError("No trainable models available for final selection.")
    return min(candidates, key=lambda k: candidates[k]["MAE"])


def _train_final_model(
    df_trainval: pd.DataFrame,
    model_name: str,
    config=CONFIG,
):
    y_series = build_target_series(df_trainval, config)

    if model_name == "ETS":
        ets_pred = _train_ets(y_series, horizon=1)
        if ets_pred is None:
            raise ValueError("ETS training failed for final model.")
        return {"model": "ETS", "fit": None}

    X_all, y_all = build_ml_features(df_trainval, config)
    if model_name == "ElasticNet":
        from sklearn.linear_model import ElasticNet

        model = ElasticNet(random_state=config.random_seed)
    else:
        _, model = _get_tree_model()

    model.fit(X_all, y_all)
    return {"model": model_name, "fit": model}


@app.command()
def main() -> None:
    logger.info("Loading Eskom data...")
    df = load_eskom_data(CONFIG)
    logger.info("Engineering target...")
    df = engineer_target(df, CONFIG)
    processed_path = save_processed_dataset(df, CONFIG)
    logger.info(f"Processed dataset saved to {processed_path}")

    df_train, df_val, df_test = split_by_dates(df, CONFIG)
    split_log_path = CONFIG.reports_dir / CONFIG.split_log_filename
    write_split_sanity_checks(df_val, df_test, split_log_path, CONFIG)
    logger.info(f"Split sanity checks saved to {split_log_path}")

    df_trainval = pd.concat([df_train, df_val]).sort_index()
    metrics_df, summary = rolling_origin_backtest(df_trainval, CONFIG)

    CONFIG.reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = CONFIG.reports_dir / CONFIG.fold_metrics_filename
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Fold metrics saved to {metrics_path}")

    CONFIG.models_dir.mkdir(parents=True, exist_ok=True)
    summary_path = CONFIG.models_dir / CONFIG.metrics_summary_filename
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Metrics summary saved to {summary_path}")

    final_model_name = _select_final_model(summary)
    final_model = _train_final_model(df_trainval, final_model_name, CONFIG)
    final_model_path = CONFIG.models_dir / CONFIG.final_model_filename
    joblib.dump(final_model, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    config_path = CONFIG.models_dir / "config.json"
    config_path.write_text(json.dumps(CONFIG.to_dict(), default=str, indent=2))

    if CONFIG.run_test_eval:
        logger.info("Running test evaluation...")
        X_test, y_test = build_ml_features(df, CONFIG)
        test_idx = df_test.index.intersection(X_test.index)
        if test_idx.empty:
            raise ValueError("No test indices available after feature generation.")

        if final_model["model"] == "ETS":
            ets_pred = _train_ets(build_target_series(df_trainval, CONFIG), len(test_idx))
            y_pred = pd.Series(ets_pred, index=test_idx)
        else:
            model = final_model["fit"]
            y_pred = pd.Series(model.predict(X_test.loc[test_idx]), index=test_idx)

        if np.array_equal(y_test.loc[test_idx].to_numpy(), y_pred.to_numpy()):
            raise ValueError("Test predictions identical to test targets (sanity check failed).")

        test_metrics = _evaluate_predictions(y_test.loc[test_idx], y_pred, CONFIG)
        test_metrics_path = CONFIG.models_dir / "test_metrics.json"
        test_metrics_path.write_text(json.dumps(test_metrics, indent=2))
        logger.info(f"Test metrics saved to {test_metrics_path}")
    else:
        logger.info("Test evaluation disabled (RUN_TEST_EVAL=False).")


if __name__ == "__main__":
    app()
