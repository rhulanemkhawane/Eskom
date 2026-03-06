from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
import typer

from eskom_energy_demand_forecasting.config import CONFIG

app = typer.Typer()


def _get_xgboost_fold_file(predictions_dir: Path) -> Path | None:
    """Return the first fold prediction file for XGBoost, falling back to any model."""
    xgb_files = sorted(predictions_dir.glob("fold_*_XGBoost.csv"))
    if xgb_files:
        return xgb_files[0]
    all_files = sorted(predictions_dir.glob("fold_*_*.csv"))
    return all_files[0] if all_files else None


def plot_actual_vs_pred(predictions_dir: Path, output_path: Path) -> None:
    file = _get_xgboost_fold_file(predictions_dir)
    if file is None:
        logger.warning("No prediction files found for plotting.")
        return
    model_label = file.stem.split("_", 2)[-1]
    df = pd.read_csv(file)
    plt.figure(figsize=(12, 4))
    plt.plot(df["y_true"].values, label="Actual")
    plt.plot(df["y_pred"].values, label=f"Predicted ({model_label})")
    plt.title(f"Actual vs Predicted — {model_label} (Fold 1)")
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residuals(predictions_dir: Path, output_path: Path) -> None:
    file = _get_xgboost_fold_file(predictions_dir)
    if file is None:
        logger.warning("No prediction files found for plotting.")
        return
    model_label = file.stem.split("_", 2)[-1]
    df = pd.read_csv(file)
    residuals = df["y_true"] - df["y_pred"]
    plt.figure(figsize=(12, 4))
    plt.plot(residuals.values)
    plt.title(f"Residuals Over Time — {model_label} (Fold 1)")
    plt.axhline(0, color="black", linewidth=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_metric_comparison(metrics_path: Path, output_path: Path) -> None:
    if not metrics_path.exists():
        logger.warning("Metrics file not found for plotting.")
        return
    metrics_df = pd.read_csv(metrics_path)
    if metrics_df.empty:
        return
    pivot = metrics_df.groupby("model")[["MAE", "RMSE"]].mean().sort_values("MAE")
    ax = pivot.plot(kind="bar", figsize=(10, 4))
    if "XGBoost" in pivot.index:
        xgb_pos = list(pivot.index).index("XGBoost")
        ax.get_xticklabels()[xgb_pos].set_fontweight("bold")
    plt.title("Metric Comparison Across Models")
    plt.ylabel("Metric Value")
    plt.xticks(rotation=45, ha="right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_importance(models_dir: Path, output_path: Path) -> None:
    model_path = models_dir / CONFIG.final_model_filename
    if not model_path.exists():
        logger.warning("Final model not found; skipping feature importance plot.")
        return
    artifact = joblib.load(model_path)
    model = artifact.get("fit")
    model_name = artifact.get("model", "")
    if model is None or not hasattr(model, "feature_importances_"):
        logger.info(f"Model '{model_name}' does not expose feature_importances_; skipping.")
        return
    importances = pd.Series(model.feature_importances_, index=model.feature_names_in_)
    importances = importances.sort_values(ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", ax=ax)
    ax.set_title(f"Feature Importances — {model_name}")
    ax.set_xlabel("Importance")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_prediction_accuracy(metrics_path: Path, output_path: Path) -> None:
    if not metrics_path.exists():
        logger.warning("Metrics file not found for prediction accuracy plot.")
        return
    metrics_df = pd.read_csv(metrics_path)
    if metrics_df.empty or "MAPE" not in metrics_df.columns:
        return

    accuracy = (
        metrics_df.groupby("model")["MAPE"]
        .mean()
        .apply(lambda x: 100 - x)
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(accuracy.index, accuracy.values, color="#1abc9c", width=0.6)

    for bar, val in zip(bars, accuracy.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title("Validation Prediction Accuracy by Model (higher is better)", fontsize=13)
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%) = 100 - MAPE")
    ax.set_ylim(min(accuracy.values) - 2, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


@app.command()
def main() -> None:
    figures_dir = CONFIG.figures_dir
    predictions_dir = CONFIG.predictions_dir
    metrics_path = CONFIG.reports_dir / CONFIG.fold_metrics_filename

    plot_actual_vs_pred(predictions_dir, figures_dir / "actual_vs_pred.png")
    plot_residuals(predictions_dir, figures_dir / "residuals.png")
    plot_metric_comparison(metrics_path, figures_dir / "metric_comparison.png")
    plot_feature_importance(CONFIG.models_dir, figures_dir / "feature_importance.png")
    plot_prediction_accuracy(metrics_path, figures_dir / "prediction_accuracy.png")
    logger.success("Plots generated.")


if __name__ == "__main__":
    app()
