"""Run the LSTM-by-cluster precipitation sweep."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from data.load_data import load_station_daily_data
from data.lstm_outputs import save_run_outputs, save_sweep_outputs
from evaluation.metrics import calculate_regression_metrics
from models.lstm import LSTMPrecipitationPredictor
from methods.cluster.cluster_pipeline import (
    PCA_VARIANCE_THRESHOLD,
    cluster_feature_matrix,
    create_cluster_feature_matrix,
    numeric_feature_columns,
)
from methods.lstm_cluster.console import print_info, print_section
from methods.lstm_cluster.report import generate_config_report
from methods.tools.sigma_choosing import calculate_sigma_values


@dataclass(frozen=True)
class ExperimentConfig:
    """One sweep configuration."""

    state: str
    station_id: str
    window_size: int
    n_clusters: int
    algorithm: str
    sigma: float | None

    @property
    def name(self) -> str:
        sigma_part = "sigma_na" if self.sigma is None else f"sigma_{self.sigma:g}"
        return (
            f"{self.state}_{self.station_id}_w{self.window_size:02d}_"
            f"k{self.n_clusters:02d}_{self.algorithm}_{sigma_part}"
        ).replace(".", "p")


DEFAULT_PLOT_STYLE: dict[str, object] = {
    "seaborn": {
        "style": "whitegrid",
        "palette": "deep",
    },
    "rc_params": {
        "figure.facecolor": "white",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    },
}


def _mapping_from_config(value: object) -> Mapping[str, object]:
    """Return nested config dictionaries safely."""
    return value if isinstance(value, Mapping) else {}


def setup_styling(plot_style: Mapping[str, object] | None = None) -> None:
    """Apply shared plotting defaults for generated figures."""
    plot_style = _mapping_from_config(plot_style)
    default_seaborn = _mapping_from_config(DEFAULT_PLOT_STYLE["seaborn"])
    default_rc_params = _mapping_from_config(DEFAULT_PLOT_STYLE["rc_params"])
    seaborn_style = {
        **default_seaborn,
        **_mapping_from_config(plot_style.get("seaborn")),
    }
    rc_params = {
        **default_rc_params,
        **_mapping_from_config(plot_style.get("rc_params")),
    }

    sns.set_theme(
        style=str(seaborn_style["style"]),
        palette=seaborn_style.get("palette"),
    )
    plt.rcParams.update(rc_params)


def build_configurations(
    sigmas: list[float | None],
    state: str,
    station_id: str,
    window_sizes: list[int],
    n_clusters_list: list[int],
    clustering_algorithm: str,
) -> list[ExperimentConfig]:
    """Return every window, cluster-count, and sigma combination."""
    return [
        ExperimentConfig(
            state=state,
            station_id=station_id,
            window_size=window_size,
            n_clusters=n_clusters,
            algorithm=clustering_algorithm.lower(),
            sigma=sigma,
        )
        for window_size in window_sizes
        for n_clusters in n_clusters_list
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
    """Split samples while preserving cluster balance when possible."""
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
    """Align each window with the following day's precipitation."""
    valid_indices = np.arange(min(n_windows, len(df) - window_size))
    target_indices = valid_indices + window_size
    targets = df.iloc[target_indices]["PRECIPITACAO_TOTAL"].to_numpy(dtype=float)
    return valid_indices, targets


def to_lstm_shape(X: np.ndarray) -> np.ndarray:
    """Represent each flattened window as a one-step LSTM sequence."""
    return X.reshape(X.shape[0], 1, X.shape[1])


def clipped_predictions(model: LSTMPrecipitationPredictor, X: np.ndarray) -> np.ndarray:
    """Predict precipitation and clip impossible negative values."""
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
    lstm_units: int,
    lstm_units_2: int,
    dropout_rate: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    early_stopping: bool,
    patience: int,
    verbose_training: int,
    random_state: int,
    show_console_info: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, object], dict[int, dict[str, float]]]:
    """Train cluster-specific LSTMs and merge their predictions."""
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

        print_info(
            f"  Cluster {cluster_id}: train={n_tr}, val={n_va}, test={n_te}",
            show_console_info,
        )
        if n_tr == 0:
            continue

        model = LSTMPrecipitationPredictor(
            input_shape=(1, X_train_lstm.shape[2]),
            lstm_units=lstm_units,
            lstm_units_2=lstm_units_2,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        history = model.fit(
            X_train_lstm[tr_mask],
            y_train[tr_mask],
            X_val=X_val_lstm[va_mask] if n_va > 0 else None,
            y_val=y_val[va_mask] if n_va > 0 else None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose_training if show_console_info else 0,
            early_stopping=early_stopping and n_va > 0,
            patience=patience,
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


def run_configuration(
    df: pd.DataFrame,
    config: ExperimentConfig,
    numeric_cols: list[str],
    output_dir: Path,
    use_all_features: bool,
    train_ratio: float,
    val_ratio: float,
    random_state: int,
    lstm_units: int,
    lstm_units_2: int,
    dropout_rate: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    early_stopping: bool,
    patience: int,
    verbose_training: int,
    show_console_info: bool,
) -> dict[str, float | int | str | None]:
    """Run one sweep configuration and save its artifacts."""
    print_section(f"Running {config.name}", show_console_info)
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = numeric_cols if use_all_features else None
    windows, windows_flat, _, _, feature_columns = create_cluster_feature_matrix(
        df,
        window_size=config.window_size,
        columns=columns,
        normalize=True,
        variance_threshold=PCA_VARIANCE_THRESHOLD,
        verbose=show_console_info,
    )
    labels = cluster_feature_matrix(
        windows_flat,
        n_clusters=config.n_clusters,
        algorithm=config.algorithm,
        sigma=config.sigma,
        random_state=random_state,
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
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_state=random_state,
    )

    print_info(
        f"  Windows={len(windows)}, samples={len(targets)}, "
        f"features={X.shape[1]}, clusters={sorted(np.unique(c).tolist())}",
        show_console_info,
    )
    print_info(
        f"  Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}",
        show_console_info,
    )
    n_samples = len(targets)
    report_config = {
        **asdict(config),
        "name": config.name,
        "dataset_start_date": df["Data"].min().date().isoformat(),
        "dataset_end_date": df["Data"].max().date().isoformat(),
        "features": feature_columns,
        "n_samples": n_samples,
        "splits": {
            "Training": {
                "samples": len(y_train),
                "percent": len(y_train) / n_samples if n_samples else 0,
            },
            "Validation": {
                "samples": len(y_val),
                "percent": len(y_val) / n_samples if n_samples else 0,
            },
            "Test": {
                "samples": len(y_test),
                "percent": len(y_test) / n_samples if n_samples else 0,
            },
        },
        "lstm_units": lstm_units,
        "lstm_units_2": lstm_units_2,
        "dense_units": [16, 8],
        "output_units": 1,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "early_stopping": early_stopping,
        "patience": patience,
        "optimizer": "Adam",
        "loss": "mean_squared_error",
        "metrics": ["mae", "mse"],
    }

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
        lstm_units=lstm_units,
        lstm_units_2=lstm_units_2,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping=early_stopping,
        patience=patience,
        verbose_training=verbose_training,
        random_state=random_state,
        show_console_info=show_console_info,
    )

    result = save_run_outputs(
        config,
        output_dir,
        feature_columns,
        targets,
        c,
        y_train,
        y_val,
        y_test,
        y_pred_train,
        y_pred_val,
        y_pred_test,
        c_test,
        histories_by_cluster,
        metrics_by_cluster,
        state=config.state,
        station_id=config.station_id,
        pca_variance_threshold=PCA_VARIANCE_THRESHOLD,
    )
    tex_path, pdf_path = generate_config_report(output_dir, report_config)
    print_info(f"  Report: {tex_path.name}", show_console_info)
    if pdf_path is not None:
        print_info(f"  Report PDF: {pdf_path.name}", show_console_info)
    print_info(
        f"  Test metrics: RMSE={result['test_rmse']:.4f}, "
        f"MAE={result['test_mae']:.4f}, R2={result['test_r2']:.4f}",
        show_console_info,
    )
    return result


def run_experiment(
    state: str,
    station_id: str,
    window_sizes: list[int],
    n_clusters_list: list[int],
    clustering_algorithm: str,
    n_sigma_values: int,
    use_all_features: bool,
    quantitative_metrics: list[str],
    lstm_units: int,
    lstm_units_2: int,
    dropout_rate: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    early_stopping: bool,
    patience: int,
    verbose_training: int,
    train_ratio: float,
    val_ratio: float,
    random_state: int,
    data_root: Path,
    output_root: Path,
    sweep_name: str | None = None,
    sweep_name_prefix: str = "lstm_cluster_sweep",
    timestamp_format: str = "%Y%m%d_%H%M%S",
    plot_style: Mapping[str, object] | None = None,
    show_console_info: bool = True,
    sigma_values: list[float] | None = None,
) -> Path:
    """Run the configured sweep and return its output directory."""
    timestamp = datetime.now().strftime(timestamp_format)
    if sweep_name is None:
        sweep_name = f"{sweep_name_prefix}_{state}_{station_id}_{timestamp}"
    output_root = Path(output_root)
    sweep_dir = output_root / sweep_name

    setup_styling(plot_style)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print_section("Loading Data", show_console_info)
    df = load_station_daily_data(state=state, station_id=station_id, data_root=data_root)
    numeric_cols = numeric_feature_columns(df)
    print_info(
        f"Loaded {len(df)} rows from {df['Data'].min().date()} to {df['Data'].max().date()}",
        show_console_info,
    )
    print_info(f"Numeric features: {numeric_cols}", show_console_info)

    if clustering_algorithm.lower() == "spectral":
        if sigma_values is None:
            print_section("Calculating Sigma Values", show_console_info)
            selected_sigma_values = calculate_sigma_values(
                df,
                n_values=n_sigma_values,
            ).tolist()
            print_info(
                f"Generated {len(selected_sigma_values)} sigma values: "
                f"{selected_sigma_values}",
                show_console_info,
            )
        else:
            selected_sigma_values = [float(sigma) for sigma in sigma_values]
            if not selected_sigma_values:
                raise ValueError(
                    "sigma_values must contain at least one value in manual mode."
                )
            valid_sigmas = all(
                np.isfinite(sigma) and sigma > 0
                for sigma in selected_sigma_values
            )
            if not valid_sigmas:
                raise ValueError(
                    "Every manual sigma value must be a positive, finite number."
                )
            print_section("Using Manual Sigma Values", show_console_info)
            print_info(
                f"Using {len(selected_sigma_values)} sigma values: "
                f"{selected_sigma_values}",
                show_console_info,
            )
    else:
        selected_sigma_values = [None]

    configurations = build_configurations(
        selected_sigma_values,
        state=state,
        station_id=station_id,
        window_sizes=window_sizes,
        n_clusters_list=n_clusters_list,
        clustering_algorithm=clustering_algorithm,
    )
    print_section("LSTM CLUSTER SWEEP", show_console_info)
    print_info(f"Station: {state}/{station_id}", show_console_info)
    print_info(f"Output directory: {sweep_dir}", show_console_info)
    print_info(f"Configurations: {len(configurations)}", show_console_info)
    for config in configurations:
        print_info(f"  - {config.name}: {asdict(config)}", show_console_info)

    results = []
    for index, config in enumerate(configurations, start=1):
        print_info(f"\nConfiguration {index}/{len(configurations)}", show_console_info)
        results.append(
            run_configuration(
                df,
                config,
                numeric_cols,
                sweep_dir / config.name,
                use_all_features=use_all_features,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                random_state=random_state,
                lstm_units=lstm_units,
                lstm_units_2=lstm_units_2,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                early_stopping=early_stopping,
                patience=patience,
                verbose_training=verbose_training,
                show_console_info=show_console_info,
            )
        )

    save_sweep_outputs(
        results,
        sweep_dir=sweep_dir,
        state=state,
        station_id=station_id,
        window_sizes=window_sizes,
        n_clusters_list=n_clusters_list,
        clustering_algorithm=clustering_algorithm,
        quantitative_metrics=quantitative_metrics,
    )

    print_section("Sweep Complete", show_console_info)
    print_info(f"Results folder: {sweep_dir}", show_console_info)
    print_info("Sweep-level files:", show_console_info)
    print_info("  - sweep_results.csv", show_console_info)
    print_info("  - sweep_summary.txt", show_console_info)
    print_info("  - overleaf_table.txt", show_console_info)
    print_info("  - overleaf_cluster_metric_tables.txt", show_console_info)
    print_info(
        "Each configuration folder contains metrics, reports, predictions, and plots.",
        show_console_info,
    )
    return sweep_dir
