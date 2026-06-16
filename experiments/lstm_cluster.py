"""Sweep LSTM-by-cluster precipitation experiments for RS A801.

For each configuration, this script:
1. Builds climate windows and clusters them.
2. Trains one LSTM model per cluster.
3. Saves metrics, predictions, summaries, and diagnostic plots.

After all configurations finish, it also writes a LaTeX table that can be
copied directly into Overleaf.
"""

from __future__ import annotations

import sys
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from climate_cluster.config import DATA_ROOT, OUTPUTS_DIR
from climate_cluster.config_data import load_single_station
from climate_cluster.evaluation.metrics import (
    calculate_regression_metrics,
    calculate_zero_precipitation_metrics,
    create_evaluation_report,
    plot_cluster_performance,
    plot_error_by_magnitude,
    plot_predictions_vs_actual,
    plot_residuals,
)
from climate_cluster.pipeline.lstm import LSTMPrecipitationPredictor
from clustering_protocol import (
    PCA_VARIANCE_THRESHOLD,
    calculate_sigma_values,
    cluster_feature_matrix,
    create_cluster_feature_matrix,
    numeric_feature_columns,
)


# ==================== CONFIGURATION ====================

STATE = "RS"
STATION_ID = "A801"

# Edit these lists to choose the sweep. LSTM sweeps can become expensive quickly.
WINDOW_SIZES = [8, 12, 16, 20, 24, 28]
N_CLUSTERS_LIST = [3,4,5]
CLUSTERING_ALGORITHM = "spectral"  # Options: "kmeans", "spectral"
#SIGMA_VALUES = [0.1]
N_SIGMA_VALUES = 5  # Used only by spectral clustering.
USE_ALL_FEATURES = True

# Metrics included in the cluster-level Overleaf tables.
# Available options from calculate_regression_metrics: MSE, RMSE, MAE, RMSLE, R2, MAPE.
#QUANTITATIVE_METRICS = ["MSE", "RMSE", "MAE", "R2"]
QUANTITATIVE_METRICS = ["MSE"]

LSTM_UNITS = 64
LSTM_UNITS_2 = 32
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001

EPOCHS = 50
BATCH_SIZE = 32
EARLY_STOPPING = True
PATIENCE = 10
VERBOSE_TRAINING = 1

TRAIN_RATIO = 0.6
VAL_RATIO = 0.1
TEST_RATIO = 0.3
RANDOM_STATE = 42

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SWEEP_NAME = f"lstm_cluster_sweep_{STATE}_{STATION_ID}_{RUN_TIMESTAMP}"
SWEEP_DIR = OUTPUTS_DIR / SWEEP_NAME


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters that uniquely define one experiment run."""

    window_size: int
    n_clusters: int
    algorithm: str
    sigma: float | None

    @property
    def name(self) -> str:
        sigma_part = "sigma_na" if self.sigma is None else f"sigma_{self.sigma:g}"
        return (
            f"{STATE}_{STATION_ID}_w{self.window_size:02d}_"
            f"k{self.n_clusters:02d}_{self.algorithm}_{sigma_part}"
        ).replace(".", "p")


def setup_styling() -> None:
    """Configure plotting defaults."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9


def print_section(title: str) -> None:
    """Print a compact section header."""
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def build_configurations(sigmas: list[float | None]) -> list[ExperimentConfig]:
    """Build the sweep grid."""
    #sigmas = SIGMA_VALUES if CLUSTERING_ALGORITHM.lower() == "spectral" else [None]
    return [
        ExperimentConfig(
            window_size=window_size,
            n_clusters=n_clusters,
            algorithm=CLUSTERING_ALGORITHM.lower(),
            sigma=sigma,
        )
        for window_size in WINDOW_SIZES
        for n_clusters in N_CLUSTERS_LIST
        for sigma in sigmas
    ]


def split_by_cluster(
    X: np.ndarray,
    y: np.ndarray,
    cluster_labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, ...]:
    """Split data, stratifying by cluster whenever every cluster has enough rows."""
    test_ratio = 1 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("TRAIN_RATIO + VAL_RATIO must be smaller than 1.")

    counts = pd.Series(cluster_labels).value_counts()
    stratify = cluster_labels if counts.min() >= 3 else None

    X_tv, X_test, y_tv, y_test, c_tv, c_test = train_test_split(
        X,
        y,
        cluster_labels,
        test_size=test_ratio,
        stratify=stratify,
        random_state=random_state,
    )

    tv_counts = pd.Series(c_tv).value_counts()
    stratify_tv = c_tv if len(tv_counts) > 0 and tv_counts.min() >= 2 else None
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val, c_train, c_val = train_test_split(
        X_tv,
        y_tv,
        c_tv,
        test_size=val_ratio_adjusted,
        stratify=stratify_tv,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, c_train, c_val, c_test


def next_day_precipitation_targets(
    df: pd.DataFrame,
    window_size: int,
    n_windows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid window indices and next-day precipitation targets."""
    valid_indices = np.arange(min(n_windows, len(df) - window_size))
    target_indices = valid_indices + window_size
    targets = df.iloc[target_indices]["PRECIPITACAO_TOTAL"].to_numpy(dtype=float)
    return valid_indices, targets


def to_lstm_shape(X: np.ndarray) -> np.ndarray:
    """Represent one flattened window as one LSTM timestep."""
    return X.reshape(X.shape[0], 1, X.shape[1])


def clipped_predictions(model: LSTMPrecipitationPredictor, X: np.ndarray) -> np.ndarray:
    """Predict precipitation and enforce the physical non-negative lower bound."""
    return np.maximum(model.predict(X).ravel(), 0.0)


def train_cluster_models(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    c_train: np.ndarray,
    c_val: np.ndarray,
    c_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, object], dict[int, dict[str, float]]]:
    """Train one LSTM per cluster and return aggregate predictions."""
    X_train_lstm = to_lstm_shape(X_train)
    X_val_lstm = to_lstm_shape(X_val)
    X_test_lstm = to_lstm_shape(X_test)

    y_pred_train = np.zeros_like(y_train, dtype=float)
    y_pred_val = np.zeros_like(y_val, dtype=float)
    y_pred_test = np.zeros_like(y_test, dtype=float)
    histories_by_cluster: dict[int, object] = {}
    metrics_by_cluster: dict[int, dict[str, float]] = {}

    for cluster_id in sorted(np.unique(c_train)):
        tr_mask = c_train == cluster_id
        va_mask = c_val == cluster_id
        te_mask = c_test == cluster_id
        n_tr, n_va, n_te = tr_mask.sum(), va_mask.sum(), te_mask.sum()

        print(f"  Cluster {cluster_id}: train={n_tr}, val={n_va}, test={n_te}")
        if n_tr == 0:
            continue

        model = LSTMPrecipitationPredictor(
            input_shape=(1, X_train_lstm.shape[2]),
            lstm_units=LSTM_UNITS,
            lstm_units_2=LSTM_UNITS_2,
            dropout_rate=DROPOUT_RATE,
            learning_rate=LEARNING_RATE,
            random_state=RANDOM_STATE,
        )
        history = model.fit(
            X_train_lstm[tr_mask],
            y_train[tr_mask],
            X_val=X_val_lstm[va_mask] if n_va > 0 else None,
            y_val=y_val[va_mask] if n_va > 0 else None,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=VERBOSE_TRAINING,
            early_stopping=EARLY_STOPPING and n_va > 0,
            patience=PATIENCE,
        )

        y_pred_train[tr_mask] = clipped_predictions(model, X_train_lstm[tr_mask])
        if n_va > 0:
            y_pred_val[va_mask] = clipped_predictions(model, X_val_lstm[va_mask])
        if n_te > 0:
            y_pred_test[te_mask] = clipped_predictions(model, X_test_lstm[te_mask])
            metrics_by_cluster[int(cluster_id)] = calculate_regression_metrics(
                y_test[te_mask],
                y_pred_test[te_mask],
            )

        histories_by_cluster[int(cluster_id)] = history

    return y_pred_train, y_pred_val, y_pred_test, histories_by_cluster, metrics_by_cluster


def save_training_history_plots(
    histories_by_cluster: dict[int, object],
    output_dir: Path,
) -> None:
    """Save loss and MAE curves for each cluster model."""
    for cluster_id, history in histories_by_cluster.items():
        hist = history.history
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        axes[0].plot(hist.get("loss", []), label="Train", linewidth=2)
        if "val_loss" in hist:
            axes[0].plot(hist["val_loss"], label="Validation", linewidth=2)
        axes[0].set_title(f"Cluster {cluster_id}: Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE")
        axes[0].legend()

        axes[1].plot(hist.get("mae", []), label="Train", linewidth=2)
        if "val_mae" in hist:
            axes[1].plot(hist["val_mae"], label="Validation", linewidth=2)
        axes[1].set_title(f"Cluster {cluster_id}: MAE")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE (mm)")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(output_dir / f"01_training_history_cluster_{cluster_id}.png", dpi=300)
        plt.close(fig)


def save_cluster_precipitation_histograms(
    y_test: np.ndarray,
    c_test: np.ndarray,
    output_dir: Path,
) -> None:
    """Save one actual-precipitation histogram per cluster."""
    hist_dir = output_dir / "cluster_precipitation_histograms"
    hist_dir.mkdir(exist_ok=True)

    for cluster_id in sorted(np.unique(c_test)):
        mask = c_test == cluster_id
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(y_test[mask], bins=25, color="#4C78A8", edgecolor="black", alpha=0.8)
        ax.set_title(f"Cluster {int(cluster_id)}: Precipitation Occurrences")
        ax.set_xlabel("Precipitation (mm)")
        ax.set_ylabel("Number of occurrences")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(
            hist_dir / f"cluster_{int(cluster_id)}_precipitation_histogram.png",
            dpi=300,
        )
        plt.close(fig)


def save_cluster_prediction_histograms(
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    c_test: np.ndarray,
    output_dir: Path,
) -> None:
    """Save actual, predicted, and residual histograms for each test cluster."""
    hist_dir = output_dir / "cluster_prediction_histograms"
    hist_dir.mkdir(exist_ok=True)

    for cluster_id in sorted(np.unique(c_test)):
        mask = c_test == cluster_id
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        residuals = y_test[mask] - y_pred_test[mask]

        axes[0].hist(y_test[mask], bins=25, color="#4C78A8", alpha=0.8)
        axes[0].set_title(f"Cluster {cluster_id}: Actual")
        axes[0].set_xlabel("Precipitation (mm)")

        axes[1].hist(y_pred_test[mask], bins=25, color="#F58518", alpha=0.8)
        axes[1].set_title(f"Cluster {cluster_id}: Predicted")
        axes[1].set_xlabel("Precipitation (mm)")

        axes[2].hist(residuals, bins=25, color="#54A24B", alpha=0.8)
        axes[2].axvline(0, color="black", linestyle="--", linewidth=1)
        axes[2].set_title(f"Cluster {cluster_id}: Residual")
        axes[2].set_xlabel("Actual - predicted (mm)")

        for ax in axes:
            ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(hist_dir / f"cluster_{int(cluster_id)}_prediction_histograms.png", dpi=300)
        plt.close(fig)


def notebook_markdown_cell(source: str) -> dict[str, object]:
    """Create a minimal Jupyter markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def notebook_code_cell(source: str) -> dict[str, object]:
    """Create a minimal Jupyter code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def dataframe_preview_markdown(
    title: str,
    csv_path: Path,
    max_rows: int | None = None,
) -> str:
    """Return a notebook-friendly preview of a CSV artifact."""
    if not csv_path.exists():
        return f"## {title}\n\nMissing file: `{csv_path.name}`\n"

    df = pd.read_csv(csv_path)
    preview_df = df if max_rows is None else df.head(max_rows)
    row_note = "" if max_rows is None or len(df) <= max_rows else f"\n\nShowing {max_rows} of {len(df)} rows."
    return (
        f"## {title}\n\n"
        f"Source: `{csv_path.name}`{row_note}\n\n"
        "```text\n"
        f"{preview_df.to_string(index=False)}\n"
        "```\n"
    )


def text_artifact_markdown(title: str, path: Path) -> str:
    """Return a markdown section containing a text artifact."""
    if not path.exists():
        return f"## {title}\n\nMissing file: `{path.name}`\n"

    return (
        f"## {title}\n\n"
        f"Source: `{path.name}`\n\n"
        "```text\n"
        f"{path.read_text(encoding='utf-8')}\n"
        "```\n"
    )


def relative_notebook_path(path: Path, notebook_dir: Path) -> str:
    """Return a POSIX-style relative path for markdown image links."""
    return path.relative_to(notebook_dir).as_posix()


def save_experiment_notebook(
    config: ExperimentConfig,
    output_dir: Path,
    feature_columns: list[str],
    split_sizes: dict[str, int],
) -> Path:
    """Create a per-experiment notebook containing this run's data and plots."""
    notebook_path = output_dir / f"{config.name}.ipynb"
    image_paths = sorted(output_dir.glob("*.png"))
    image_paths.extend(sorted((output_dir / "cluster_precipitation_histograms").glob("*.png")))
    image_paths.extend(sorted((output_dir / "cluster_prediction_histograms").glob("*.png")))

    cells = [
        notebook_markdown_cell(
            f"# {config.name}\n\n"
            "Per-experiment LSTM cluster outputs generated by `experiments/lstm_cluster.py`.\n\n"
            "## Configuration\n\n"
            f"- Station: `{STATE}/{STATION_ID}`\n"
            f"- Window size: `{config.window_size}`\n"
            f"- Number of clusters: `{config.n_clusters}`\n"
            f"- Clustering algorithm: `{config.algorithm}`\n"
            f"- Sigma: `{config.sigma if config.sigma is not None else 'not used'}`\n"
            f"- PCA variance threshold: `{PCA_VARIANCE_THRESHOLD:.2f}`\n"
            f"- LSTM units: `{LSTM_UNITS}`, `{LSTM_UNITS_2}`\n"
            f"- Dropout rate: `{DROPOUT_RATE}`\n"
            f"- Learning rate: `{LEARNING_RATE}`\n"
            f"- Epochs: `{EPOCHS}`\n"
            f"- Batch size: `{BATCH_SIZE}`\n"
            f"- Splits: `{split_sizes}`\n"
            f"- Features ({len(feature_columns)}): `{', '.join(feature_columns)}`\n"
        ),
        notebook_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n\n"
            "experiment_dir = Path.cwd()\n"
            "metrics = pd.read_csv(experiment_dir / 'metrics_summary.csv')\n"
            "cluster_metrics = pd.read_csv(experiment_dir / 'cluster_model_metrics.csv')\n"
            "predictions = pd.read_csv(experiment_dir / 'test_predictions.csv')\n"
        ),
        notebook_markdown_cell(
            dataframe_preview_markdown("Split-Level Metrics", output_dir / "metrics_summary.csv")
        ),
        notebook_markdown_cell(
            dataframe_preview_markdown(
                "Cluster Model Metrics",
                output_dir / "cluster_model_metrics.csv",
            )
        ),
        notebook_markdown_cell(
            dataframe_preview_markdown(
                "Test Predictions",
                output_dir / "test_predictions.csv",
                max_rows=25,
            )
        ),
        notebook_markdown_cell(
            "## Prediction Summary\n\n"
            "```text\n"
            f"{pd.read_csv(output_dir / 'test_predictions.csv').describe().to_string()}\n"
            "```\n"
        ),
        notebook_markdown_cell(
            text_artifact_markdown("Evaluation Report", output_dir / "evaluation_report.txt")
        ),
        notebook_markdown_cell(text_artifact_markdown("Run Summary", output_dir / "summary.txt")),
    ]

    if image_paths:
        plot_sections = ["# Experiment Plots\n"]
        for image_path in image_paths:
            title = image_path.stem.replace("_", " ").title()
            rel_path = relative_notebook_path(image_path, output_dir)
            plot_sections.append(f"## {title}\n\n![{title}]({rel_path})\n")
        cells.append(notebook_markdown_cell("\n".join(plot_sections)))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    return notebook_path


def save_precipitation_by_cluster_plot(
    y_test: np.ndarray,
    c_test: np.ndarray,
    output_dir: Path,
) -> None:
    """Save a boxplot showing the target distribution by cluster."""
    plot_df = pd.DataFrame({"cluster": c_test, "precipitation_mm": y_test})
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=plot_df, x="cluster", y="precipitation_mm", ax=ax)
    ax.set_title("Test Set: Precipitation Distribution by Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Next-day precipitation (mm)")
    fig.tight_layout()
    fig.savefig(output_dir / "07_precipitation_distribution_by_cluster.png", dpi=300)
    plt.close(fig)


def save_cluster_distribution_plot(c_test: np.ndarray, output_dir: Path) -> None:
    """Save cluster sample counts for the test set."""
    unique_clusters, counts = np.unique(c_test, return_counts=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(unique_clusters, counts, color="#4C78A8", alpha=0.8)
    ax.set_title("Test Set: Cluster Distribution")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Samples")
    ax.bar_label(bars)
    fig.tight_layout()
    fig.savefig(output_dir / "06_cluster_distribution.png", dpi=300)
    plt.close(fig)


def save_visualizations(
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    c_test: np.ndarray,
    histories_by_cluster: dict[int, object],
    output_dir: Path,
) -> None:
    """Save the diagnostic plots for one configuration."""
    save_training_history_plots(histories_by_cluster, output_dir)

    fig, _ = plot_predictions_vs_actual(
        y_test,
        y_pred_test,
        title="Test Set: Predictions vs Actual Precipitation",
    )
    fig.savefig(output_dir / "02_predictions_vs_actual.png", dpi=300)
    plt.close(fig)

    fig, _ = plot_residuals(y_test, y_pred_test, title="Test Set: Residual Analysis")
    fig.savefig(output_dir / "03_residuals_analysis.png", dpi=300)
    plt.close(fig)

    fig, _ = plot_error_by_magnitude(
        y_test,
        y_pred_test,
        n_bins=10,
        title="Test Set: Error by Precipitation Magnitude",
    )
    fig.savefig(output_dir / "04_error_by_magnitude.png", dpi=300)
    plt.close(fig)

    fig, _ = plot_cluster_performance(
        c_test,
        y_test,
        y_pred_test,
        title="Test Set: LSTM Performance by Cluster",
    )
    fig.savefig(output_dir / "05_cluster_performance.png", dpi=300)
    plt.close(fig)

    save_cluster_distribution_plot(c_test, output_dir)
    save_precipitation_by_cluster_plot(y_test, c_test, output_dir)
    save_cluster_precipitation_histograms(y_test, c_test, output_dir)
    save_cluster_prediction_histograms(y_test, y_pred_test, c_test, output_dir)


def metrics_dataframe(
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> pd.DataFrame:
    """Return split-level metrics as a tidy dataframe."""
    rows = []
    for split, metrics in [
        ("Train", train_metrics),
        ("Validation", val_metrics),
        ("Test", test_metrics),
    ]:
        row = {"split": split}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_metrics_dataframe(metrics_by_cluster: dict[int, dict[str, float]]) -> pd.DataFrame:
    """Return test metrics for each cluster-specific model."""
    rows = []
    for cluster_id, metrics in sorted(metrics_by_cluster.items()):
        row = {"cluster": cluster_id}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def save_config_summary(
    config: ExperimentConfig,
    output_dir: Path,
    feature_columns: list[str],
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    zero_metrics: dict[str, float],
    metrics_by_cluster: dict[int, dict[str, float]],
    split_sizes: dict[str, int],
) -> None:
    """Save a compact human-readable summary for one configuration."""
    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("LSTM CLUSTER SWEEP CONFIGURATION\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Run folder: {config.name}\n")
        f.write(f"Station: {STATE}/{STATION_ID}\n")
        f.write(f"Window size: {config.window_size}\n")
        f.write(f"Number of clusters: {config.n_clusters}\n")
        f.write(f"Clustering algorithm: {config.algorithm}\n")
        f.write(f"Sigma: {config.sigma if config.sigma is not None else 'not used'}\n")
        f.write(f"PCA variance threshold: {PCA_VARIANCE_THRESHOLD:.2f}\n")
        f.write(f"Features ({len(feature_columns)}): {', '.join(feature_columns)}\n")
        f.write(f"Splits: {split_sizes}\n\n")

        f.write("Metrics\n")
        f.write("-" * 72 + "\n")
        for split, metrics in [
            ("Train", train_metrics),
            ("Validation", val_metrics),
            ("Test", test_metrics),
        ]:
            f.write(
                f"{split:<10} RMSE={metrics['RMSE']:.4f}  "
                f"MAE={metrics['MAE']:.4f}  R2={metrics['R2']:.4f}  "
                f"RMSLE={metrics['RMSLE']:.4f}\n"
            )

        f.write("\nZero vs rainy days on test set\n")
        f.write("-" * 72 + "\n")
        for key, value in zero_metrics.items():
            f.write(f"{key}: {value}\n")

        f.write("\nTest metrics by cluster\n")
        f.write("-" * 72 + "\n")
        for cluster_id, metrics in sorted(metrics_by_cluster.items()):
            f.write(
                f"Cluster {cluster_id}: MSE={metrics['MSE']:.4f}, "
                f"RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, "
                f"R2={metrics['R2']:.4f}\n"
            )


def save_run_outputs(
    config: ExperimentConfig,
    output_dir: Path,
    feature_columns: list[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    y_pred_train: np.ndarray,
    y_pred_val: np.ndarray,
    y_pred_test: np.ndarray,
    c_test: np.ndarray,
    histories_by_cluster: dict[int, object],
    metrics_by_cluster: dict[int, dict[str, float]],
) -> dict[str, float | int | str | None]:
    """Save all artifacts for one run and return one sweep-level result row."""
    train_metrics = calculate_regression_metrics(y_train, y_pred_train)
    val_metrics = calculate_regression_metrics(y_val, y_pred_val)
    test_metrics = calculate_regression_metrics(y_test, y_pred_test)
    zero_metrics = calculate_zero_precipitation_metrics(y_test, y_pred_test)
    split_sizes = {
        "train": len(y_train),
        "validation": len(y_val),
        "test": len(y_test),
    }

    metrics_dataframe(train_metrics, val_metrics, test_metrics).to_csv(
        output_dir / "metrics_summary.csv",
        index=False,
    )
    cluster_metrics_dataframe(metrics_by_cluster).to_csv(
        output_dir / "cluster_model_metrics.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "actual": y_test,
            "predicted": y_pred_test,
            "residual": y_test - y_pred_test,
            "cluster": c_test,
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)

    with open(output_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(
            create_evaluation_report(
                y_train,
                y_val,
                y_test,
                y_pred_train,
                y_pred_val,
                y_pred_test,
                c_test,
            )
        )

    save_config_summary(
        config=config,
        output_dir=output_dir,
        feature_columns=feature_columns,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        zero_metrics=zero_metrics,
        metrics_by_cluster=metrics_by_cluster,
        split_sizes=split_sizes,
    )
    save_visualizations(
        y_test,
        y_pred_test,
        c_test,
        histories_by_cluster,
        output_dir,
    )
    notebook_path = save_experiment_notebook(
        config=config,
        output_dir=output_dir,
        feature_columns=feature_columns,
        split_sizes=split_sizes,
    )
    print(f"  Notebook: {notebook_path.name}")

    result = {
        "run_name": config.name,
        "window_size": config.window_size,
        "n_clusters": config.n_clusters,
        "algorithm": config.algorithm,
        "sigma": config.sigma,
        "zero_days_ratio": zero_metrics["zero_days_ratio"],
        "rainy_days_rmse": zero_metrics.get("rainy_days_rmse", np.nan),
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
    }

    for split, metrics in [
        ("train", train_metrics),
        ("val", val_metrics),
        ("test", test_metrics),
    ]:
        for metric_name, value in metrics.items():
            result[f"{split}_{metric_name.lower()}"] = value

    for cluster_id, metrics in sorted(metrics_by_cluster.items()):
        for metric_name, value in metrics.items():
            result[f"cluster_{cluster_id}_{metric_name.lower()}"] = value

    return result


def run_configuration(
    df: pd.DataFrame,
    config: ExperimentConfig,
    numeric_cols: list[str],
    output_dir: Path,
) -> dict[str, float | int | str | None]:
    """Run one full LSTM-clustering experiment."""
    print_section(f"Running {config.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = numeric_cols if USE_ALL_FEATURES else None
    windows, windows_flat, _, _, feature_columns = create_cluster_feature_matrix(
        df,
        window_size=config.window_size,
        columns=columns,
        normalize=True,
        variance_threshold=PCA_VARIANCE_THRESHOLD,
    )
    labels = cluster_feature_matrix(
        windows_flat,
        n_clusters=config.n_clusters,
        algorithm=config.algorithm,
        sigma=config.sigma,
        random_state=RANDOM_STATE,
    )
    valid_indices, targets = next_day_precipitation_targets(
        df,
        config.window_size,
        len(windows),
    )

    X = windows_flat[valid_indices]
    c = labels[valid_indices]
    X_train, X_val, X_test, y_train, y_val, y_test, c_train, c_val, c_test = split_by_cluster(
        X,
        targets,
        c,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        random_state=RANDOM_STATE,
    )

    print(
        f"  Windows={len(windows)}, samples={len(targets)}, "
        f"features={X.shape[1]}, clusters={sorted(np.unique(c).tolist())}"
    )
    print(f"  Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    (
        y_pred_train,
        y_pred_val,
        y_pred_test,
        histories_by_cluster,
        metrics_by_cluster,
    ) = train_cluster_models(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        c_train,
        c_val,
        c_test,
    )

    result = save_run_outputs(
        config,
        output_dir,
        feature_columns,
        y_train,
        y_val,
        y_test,
        y_pred_train,
        y_pred_val,
        y_pred_test,
        c_test,
        histories_by_cluster,
        metrics_by_cluster,
    )
    print(
        f"  Test metrics: RMSE={result['test_rmse']:.4f}, "
        f"MAE={result['test_mae']:.4f}, R2={result['test_r2']:.4f}"
    )
    return result


def latex_table(results_df: pd.DataFrame) -> str:
    """Format sweep results as a LaTeX table."""
    table_df = results_df.sort_values(["test_rmse", "test_mae"]).copy()
    metrics = [metric.upper() for metric in QUANTITATIVE_METRICS]
    alignment = "rrr" + "r" * len(metrics) + "r"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{LSTM cluster sweep results for RS A801.}",
        r"\label{tab:lstm_cluster_sweep_rs_a801}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\hline",
        "Window & $K$ & Sigma & "
        + " & ".join(metric_latex_label(metric, "test") for metric in metrics)
        + r" & $N_{test}$ \\",
        r"\hline",
    ]

    for _, row in table_df.iterrows():
        sigma = "N/A" if pd.isna(row["sigma"]) else f"{row['sigma']:.4g}"
        metric_values = [
            format_latex_number(row.get(f"test_{metric_key(metric)}"))
            for metric in metrics
        ]
        lines.append(
            f"{int(row['window_size'])} & "
            f"{int(row['n_clusters'])} & "
            f"{sigma} & "
            + " & ".join(metric_values)
            + " & "
            f"{int(row['n_test'])} \\\\"
        )

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def format_latex_number(value: object, digits: int = 3) -> str:
    """Format numeric values for compact LaTeX tables."""
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def metric_key(metric_name: str) -> str:
    """Return the dataframe key used for a metric name."""
    return metric_name.lower()


def metric_latex_label(metric_name: str, suffix: str) -> str:
    """Return a compact LaTeX-safe metric column label."""
    label = "R^2" if metric_name.upper() == "R2" else metric_name.upper()
    return rf"${label}_{{{suffix}}}$"


def cluster_metric_latex_table(results_df: pd.DataFrame, n_clusters: int) -> str:
    """Format one per-cluster metric table for a fixed number of clusters."""
    table_df = results_df[results_df["n_clusters"] == n_clusters].copy()
    table_df = table_df.sort_values(["window_size", "sigma"])

    metrics = [metric.upper() for metric in QUANTITATIVE_METRICS]
    overall_columns = [(metric, metric_key(metric)) for metric in metrics]
    cluster_columns = [
        (metric, metric_key(metric), cluster_id)
        for metric in metrics
        for cluster_id in range(n_clusters)
    ]
    alignment = "rr" + "r" * (len(overall_columns) + len(cluster_columns))
    metric_headers = [
        metric_latex_label(metric, "all")
        for metric, _ in overall_columns
    ] + [
        metric_latex_label(metric, str(cluster_id))
        for metric, _, cluster_id in cluster_columns
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\hline",
        "Window & Sigma & " + " & ".join(metric_headers) + r" \\",
        r"\hline",
    ]

    for _, row in table_df.iterrows():
        sigma = "N/A" if pd.isna(row["sigma"]) else f"{row['sigma']:.4g}"
        overall_values = [
            format_latex_number(row.get(f"test_{metric_key_name}"))
            for _, metric_key_name in overall_columns
        ]
        cluster_values = [
            format_latex_number(row.get(f"cluster_{cluster_id}_{metric_key}"))
            for _, metric_key, cluster_id in cluster_columns
        ]
        lines.append(
            f"{int(row['window_size'])} & {sigma} & "
            + " & ".join(overall_values + cluster_values)
            + r" \\"
        )

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
            r"}",
            rf"\caption{{Overall and cluster-specific test metrics for $K={n_clusters}$ models.}}",
            rf"\label{{tab:lstm_cluster_metrics_k{n_clusters}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def cluster_metric_latex_tables(results_df: pd.DataFrame) -> str:
    """Format one cluster-metric LaTeX table for each tested K."""
    tables = [
        cluster_metric_latex_table(results_df, int(n_clusters))
        for n_clusters in sorted(results_df["n_clusters"].unique())
    ]
    return "\n".join(tables)


def save_sweep_outputs(results: list[dict[str, float | int | str | None]]) -> None:
    """Save sweep-level CSV, text summary, and LaTeX table."""
    results_df = pd.DataFrame(results).sort_values(["test_rmse", "test_mae"])
    results_df.to_csv(SWEEP_DIR / "sweep_results.csv", index=False)

    with open(SWEEP_DIR / "overleaf_table.txt", "w", encoding="utf-8") as f:
        f.write(latex_table(results_df))
    with open(SWEEP_DIR / "overleaf_cluster_metric_tables.txt", "w", encoding="utf-8") as f:
        f.write(cluster_metric_latex_tables(results_df))

    best = results_df.iloc[0]
    with open(SWEEP_DIR / "sweep_summary.txt", "w", encoding="utf-8") as f:
        f.write("LSTM CLUSTER SWEEP SUMMARY\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Station: {STATE}/{STATION_ID}\n")
        f.write(f"Configurations: {len(results_df)}\n")
        f.write(f"Window sizes: {WINDOW_SIZES}\n")
        f.write(f"Cluster counts: {N_CLUSTERS_LIST}\n")
        f.write(f"Algorithm: {CLUSTERING_ALGORITHM}\n\n")
        f.write("Best configuration by test RMSE\n")
        f.write("-" * 72 + "\n")
        f.write(best.to_string())
        f.write("\n\nFull results are in sweep_results.csv.\n")


def main() -> None:
    """Run the full sweep."""
    setup_styling()
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    print_section("Loading Data")
    df = load_single_station(state=STATE, station_id=STATION_ID, data_root=DATA_ROOT)
    numeric_cols = numeric_feature_columns(df)
    print(f"Loaded {len(df)} rows from {df['Data'].min().date()} to {df['Data'].max().date()}")
    print(f"Numeric features: {numeric_cols}")

    if CLUSTERING_ALGORITHM.lower() == "spectral":
        print_section("Calculating Sigma Values")
        sigma_values = calculate_sigma_values(df, n_values=N_SIGMA_VALUES).tolist()
        print(f"Generated {len(sigma_values)} sigma values: {sigma_values}")
    else:
        sigma_values = [None]

    configurations = build_configurations(sigma_values)
    print_section("LSTM CLUSTER SWEEP")
    print(f"Station: {STATE}/{STATION_ID}")
    print(f"Output directory: {SWEEP_DIR}")
    print(f"Configurations: {len(configurations)}")
    for config in configurations:
        print(f"  - {config.name}: {asdict(config)}")


    results = []
    for index, config in enumerate(configurations, start=1):
        print(f"\nConfiguration {index}/{len(configurations)}")
        results.append(run_configuration(df, config, numeric_cols, SWEEP_DIR / config.name))

    save_sweep_outputs(results)

    print_section("Sweep Complete")
    print(f"Results folder: {SWEEP_DIR}")
    print("Sweep-level files:")
    print("  - sweep_results.csv")
    print("  - sweep_summary.txt")
    print("  - overleaf_table.txt")
    print("  - overleaf_cluster_metric_tables.txt")
    print("Each configuration folder contains metrics, reports, predictions, plots, and a notebook.")


if __name__ == "__main__":
    main()
