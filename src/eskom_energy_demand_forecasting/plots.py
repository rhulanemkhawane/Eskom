from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
import typer

from eskom_energy_demand_forecasting.config import CONFIG

app = typer.Typer()


def plot_actual_vs_pred(predictions_dir: Path, output_path: Path) -> None:
    files = sorted(predictions_dir.glob("fold_*_*.csv"))
    if not files:
        logger.warning("No prediction files found for plotting.")
        return
    df = pd.read_csv(files[0])
    plt.figure(figsize=(12, 4))
    plt.plot(df["y_true"], label="Actual")
    plt.plot(df["y_pred"], label="Predicted")
    plt.title("Actual vs Predicted (Sample Fold)")
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residuals(predictions_dir: Path, output_path: Path) -> None:
    files = sorted(predictions_dir.glob("fold_*_*.csv"))
    if not files:
        logger.warning("No prediction files found for plotting.")
        return
    df = pd.read_csv(files[0])
    residuals = df["y_true"] - df["y_pred"]
    plt.figure(figsize=(12, 4))
    plt.plot(residuals)
    plt.title("Residuals Over Time (Sample Fold)")
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
    pivot = metrics_df.groupby("model")[["MAE", "RMSE"]].mean()
    pivot.plot(kind="bar", figsize=(10, 4))
    plt.title("Metric Comparison Across Models")
    plt.ylabel("Metric Value")
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
    logger.success("Plots generated.")


if __name__ == "__main__":
    app()
