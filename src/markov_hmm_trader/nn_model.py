from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PipelineConfig

_LAST_TRAINING_METRICS: list[dict[str, object]] = []
_LAST_SPLIT_METADATA: list[dict[str, object]] = []


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def make_forward_labels(df: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    out = df.copy()
    if "Close" in out:
        forward = np.log(out["Close"].shift(-horizon_bars) / out["Close"])
    else:
        forward = out["ret_1"].rolling(horizon_bars).sum().shift(-horizon_bars)
    out["target_forward_return"] = forward
    out["target_positive"] = (forward > 0.0).astype(float)
    out.loc[forward.isna(), "target_positive"] = np.nan
    return out


def _set_empty_nn_columns(out: pd.DataFrame, cfg: PipelineConfig, error: str = "") -> pd.DataFrame:
    out["nn_probability_positive"] = np.nan
    out["nn_probability_std"] = np.nan
    out["nn_adjusted_probability"] = np.nan
    out["nn_expected_return_proxy"] = np.nan
    out["nn_signal"] = 0
    out["nn_confidence"] = 0.0
    out["nn_model_type"] = cfg.nn_model_type
    out["nn_error"] = error
    out["nn_window_id"] = out.get("hmm_window_id", 0)
    out["nn_is_oos"] = bool(error == "")
    return out


@dataclass
class SequenceDataset:
    X: np.ndarray
    y: np.ndarray
    positions: np.ndarray
    sample_weight: np.ndarray


def make_time_decay_weights(
    positions: np.ndarray,
    anchor_position: int,
    decay_factor: float,
    enabled: bool = True,
) -> np.ndarray:
    """Return recency weights similar to the local LSTM training pipeline."""
    positions = np.asarray(positions, dtype=float)
    if positions.size == 0 or not enabled:
        return np.ones(positions.shape, dtype=np.float32)
    ages = np.maximum(float(anchor_position) - positions, 0.0)
    weights = np.exp(-float(decay_factor) * ages)
    mean_weight = float(np.nanmean(weights))
    if np.isfinite(mean_weight) and mean_weight > 0.0:
        weights = weights / mean_weight
    return weights.astype(np.float32)


def make_sequence_dataset(
    matrix: np.ndarray,
    labels: np.ndarray,
    positions: np.ndarray,
    lookback: int,
    sample_weights: np.ndarray | None = None,
) -> SequenceDataset:
    X_rows = []
    y_rows = []
    pos_rows = []
    weight_rows = []
    valid_positions = set(int(position) for position in positions)
    for position in positions:
        position = int(position)
        start = position - lookback + 1
        if start < 0 or position not in valid_positions:
            continue
        label = labels[position]
        if not np.isfinite(label):
            continue
        window = matrix[start : position + 1]
        if not np.isfinite(window).all():
            continue
        X_rows.append(window)
        y_rows.append(label)
        pos_rows.append(position)
        weight_rows.append(1.0 if sample_weights is None else sample_weights[position])
    if not X_rows:
        return SequenceDataset(
            np.empty((0, lookback, matrix.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=np.float32),
        )
    return SequenceDataset(
        np.asarray(X_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.float32),
        np.asarray(pos_rows, dtype=int),
        np.asarray(weight_rows, dtype=np.float32),
    )


def _position_date(df: pd.DataFrame, position: object) -> str:
    if position is None or (isinstance(position, float) and not np.isfinite(position)):
        return ""
    try:
        value = df.index[int(position)]
    except (IndexError, TypeError, ValueError):
        return ""
    try:
        return pd.Timestamp(value).date().isoformat()
    except (TypeError, ValueError):
        return str(value)


def _horizon_suffix(horizon_bars: int) -> str:
    mapping = {1: "1d", 5: "1w", 20: "1m", 21: "1m", 60: "3m", 63: "3m", 126: "6m"}
    return mapping.get(int(horizon_bars), f"{int(horizon_bars)}b")


def _torch_modules(cfg: PipelineConfig, n_features: int, lookback: int):
    from torch import nn

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            in_features = n_features * lookback
            for hidden in cfg.nn_hidden_sizes:
                layers.append(nn.Linear(in_features, int(hidden)))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(float(cfg.nn_dropout)))
                in_features = int(hidden)
            layers.append(nn.Linear(in_features, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)

    class ConvLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            channels = int(cfg.nn_conv_channels)
            hidden = int(cfg.nn_hidden_sizes[0] if cfg.nn_hidden_sizes else 32)
            self.conv = nn.Conv1d(n_features, channels, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(float(cfg.nn_dropout))
            self.lstm = nn.LSTM(channels, hidden, batch_first=True)
            self.head = nn.Sequential(
                nn.Dropout(float(cfg.nn_dropout)),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.dropout(self.relu(self.conv(x)))
            x = x.transpose(1, 2)
            output, _ = self.lstm(x)
            return self.head(output[:, -1, :]).squeeze(-1)

    model_type = cfg.nn_model_type.lower()
    if model_type == "conv_lstm":
        return ConvLSTM()
    if model_type != "mlp":
        raise ValueError(f"unsupported_nn_model_type:{model_type}")
    return MLP()


def _train_predict_window(
    matrix: np.ndarray,
    labels: np.ndarray,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    cfg: PipelineConfig,
    fold_id: int,
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, object]]:
    import torch
    from torch import nn

    lookback = max(int(cfg.nn_lookback_bars), 2)
    full_sample_weights = np.ones(len(labels), dtype=np.float32)
    if len(train_positions) > 0:
        full_sample_weights[train_positions] = make_time_decay_weights(
            train_positions,
            anchor_position=int(train_positions.max()),
            decay_factor=float(cfg.nn_time_decay_factor),
            enabled=bool(cfg.nn_use_time_decay_weights),
        )
    train_ds = make_sequence_dataset(
        matrix,
        labels,
        train_positions,
        lookback,
        sample_weights=full_sample_weights,
    )
    test_ds = make_sequence_dataset(matrix, labels, test_positions, lookback)
    metrics: list[dict[str, object]] = []
    split_info: dict[str, object] = {
        "run_id": "",
        "window_id": fold_id,
        "fold_id": fold_id,
        "model_type": cfg.nn_model_type,
        "horizon_bars": int(cfg.nn_horizon_bars),
        "lookback_bars": lookback,
        "raw_train_rows": int(len(train_positions)),
        "raw_oos_rows": int(len(test_positions)),
        "train_rows": 0,
        "validation_rows": 0,
        "oos_rows": int(len(test_ds.positions)),
        "train_start_position": (
            int(train_ds.positions[0]) if len(train_ds.positions) else np.nan
        ),
        "train_end_position": np.nan,
        "validation_start_position": np.nan,
        "validation_end_position": np.nan,
        "oos_start_position": (
            int(test_ds.positions[0]) if len(test_ds.positions) else np.nan
        ),
        "oos_end_position": (
            int(test_ds.positions[-1]) if len(test_ds.positions) else np.nan
        ),
        "strict_oos_enforced": bool(cfg.nn_require_oos_metadata),
        "use_time_decay_weights": bool(cfg.nn_use_time_decay_weights),
        "mean_sample_weight": (
            float(np.mean(train_ds.sample_weight)) if len(train_ds.sample_weight) else np.nan
        ),
        "error": "",
    }
    if len(train_ds.y) < max(10, int(cfg.nn_batch_size // 2)) or len(test_ds.positions) == 0:
        split_info["error"] = "insufficient_oos_training_data"
        return pd.DataFrame(), metrics, split_info

    val_count = max(1, int(round(len(train_ds.y) * float(cfg.nn_validation_split))))
    val_count = min(val_count, max(len(train_ds.y) - 1, 1))
    train_count = len(train_ds.y) - val_count
    split_info.update(
        {
            "train_rows": int(train_count),
            "validation_rows": int(val_count),
            "train_end_position": int(train_ds.positions[train_count - 1]),
            "validation_start_position": int(train_ds.positions[train_count]),
            "validation_end_position": int(train_ds.positions[-1]),
        }
    )
    X_train = torch.tensor(train_ds.X[:train_count], dtype=torch.float32)
    y_train = torch.tensor(train_ds.y[:train_count], dtype=torch.float32)
    w_train = torch.tensor(train_ds.sample_weight[:train_count], dtype=torch.float32)
    X_val = torch.tensor(train_ds.X[train_count:], dtype=torch.float32)
    y_val = torch.tensor(train_ds.y[train_count:], dtype=torch.float32)
    X_test = torch.tensor(test_ds.X, dtype=torch.float32)

    torch.manual_seed(int(cfg.random_seeds[0] if cfg.random_seeds else 7) + fold_id)
    model = _torch_modules(cfg, matrix.shape[1], lookback)
    optimiser = torch.optim.Adam(model.parameters(), lr=float(cfg.nn_learning_rate))
    lr_factor = min(max(float(cfg.nn_lr_factor), 1e-6), 0.999)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="min",
        factor=lr_factor,
        patience=max(int(cfg.nn_lr_patience), 0),
    )
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    best_state = None
    best_val = np.inf
    best_epoch = 0
    batch_size = max(int(cfg.nn_batch_size), 1)
    min_delta = max(float(cfg.nn_min_delta), 0.0)
    early_patience = max(int(cfg.nn_early_stopping_patience), 0)
    epochs_without_improvement = 0

    def weighted_loss(logits, targets, weights=None):
        losses = loss_fn(logits, targets)
        if weights is None:
            return losses.mean()
        return (losses * weights).sum() / weights.sum().clamp_min(1e-8)

    for epoch in range(1, int(cfg.nn_epochs) + 1):
        model.train()
        permutation = torch.randperm(len(X_train))
        train_losses = []
        for start in range(0, len(X_train), batch_size):
            batch = permutation[start : start + batch_size]
            optimiser.zero_grad()
            loss = weighted_loss(model(X_train[batch]), y_train[batch], w_train[batch])
            loss.backward()
            optimiser.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_loss = float(weighted_loss(model(X_val), y_val).detach().cpu())
        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        improved = val_loss < best_val - min_delta
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step(val_loss)
        metrics.append(
            {
                "run_id": "",
                "fold_id": fold_id,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_epoch": best_epoch,
                "early_stop_triggered": False,
                "model_type": cfg.nn_model_type,
                "learning_rate": float(optimiser.param_groups[0]["lr"]),
                "train_rows": int(train_count),
                "validation_rows": int(val_count),
                "oos_rows": int(len(test_ds.positions)),
                "use_time_decay_weights": bool(cfg.nn_use_time_decay_weights),
                "mean_sample_weight": float(np.mean(train_ds.sample_weight[:train_count])),
                "error": "",
            }
        )
        if early_patience > 0 and epochs_without_improvement >= early_patience:
            metrics[-1]["early_stop_triggered"] = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    samples = []
    model.train()
    with torch.no_grad():
        for _ in range(max(int(cfg.nn_mc_dropout_samples), 1)):
            samples.append(torch.sigmoid(model(X_test)).detach().cpu().numpy())
    probabilities = np.vstack(samples)
    mean = probabilities.mean(axis=0)
    std = probabilities.std(axis=0)
    adjusted = mean - float(cfg.nn_uncertainty_penalty) * std
    pred_class = np.select(
        [adjusted >= cfg.nn_positive_threshold, adjusted <= cfg.nn_negative_threshold],
        [1, -1],
        default=0,
    )
    return (
        pd.DataFrame(
            {
                "position": test_ds.positions,
                "nn_probability_positive": mean,
                "nn_probability_std": std,
                "nn_adjusted_probability": adjusted,
                "nn_expected_return_proxy": mean - 0.5,
                "nn_signal": pred_class,
                "nn_confidence": np.abs(adjusted - 0.5) * 2.0,
                "nn_model_type": cfg.nn_model_type,
                "nn_error": "",
                "nn_is_oos": True,
                "nn_window_id": fold_id,
            }
        ),
        metrics,
        split_info,
    )


def add_nn_confirmation(
    df: pd.DataFrame,
    feature_cols: list[str],
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """Add optional PyTorch NN confirmation with strict walk-forward OOS boundaries."""
    global _LAST_TRAINING_METRICS
    _LAST_TRAINING_METRICS = []
    out = make_forward_labels(df, cfg.nn_horizon_bars)
    if not cfg.enable_nn_confirmation:
        return _set_empty_nn_columns(out, cfg)

    if not torch_available():
        print("NN confirmation skipped: install requirements-ml.txt to enable torch.")
        _LAST_TRAINING_METRICS = [
            {
                "run_id": "",
                "fold_id": 0,
                "epoch": 0,
                "train_loss": np.nan,
                "val_loss": np.nan,
                "best_epoch": 0,
                "early_stop_triggered": False,
                "model_type": cfg.nn_model_type,
                "error": "torch_missing",
            }
        ]
        return _set_empty_nn_columns(out, cfg, error="torch_missing")

    values = out[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    labels = out["target_positive"].to_numpy(dtype=float)
    positions = np.arange(len(out))
    predictions = []

    window_ids = out.get("hmm_window_id", pd.Series(np.nan, index=out.index))
    unique_windows = [int(value) for value in sorted(window_ids.dropna().unique())]
    if not unique_windows:
        split = int(round(len(out) * (1.0 - float(cfg.nn_validation_split))))
        unique_windows = [1]
        window_ids = pd.Series(np.where(positions >= split, 1, np.nan), index=out.index)

    for fold_id in unique_windows:
        test_mask = window_ids == fold_id
        test_positions = positions[test_mask.to_numpy()]
        if test_positions.size == 0:
            continue
        test_start = int(test_positions.min())
        train_positions = positions[(positions + int(cfg.nn_horizon_bars)) < test_start]
        if train_positions.size == 0:
            continue
        train_values = values.iloc[train_positions]
        mean = train_values.mean(axis=0)
        std = train_values.std(axis=0).replace(0.0, 1.0).fillna(1.0)
        matrix = ((values - mean) / std).to_numpy(dtype=np.float32)
        try:
            frame, metrics = _train_predict_window(
                matrix,
                labels,
                train_positions,
                test_positions,
                cfg,
                fold_id,
            )
            _LAST_TRAINING_METRICS.extend(metrics)
            if not frame.empty:
                predictions.append(frame)
        except Exception as err:
            _LAST_TRAINING_METRICS.append(
                {
                    "run_id": "",
                    "fold_id": fold_id,
                    "epoch": 0,
                    "train_loss": np.nan,
                    "val_loss": np.nan,
                    "best_epoch": 0,
                    "early_stop_triggered": False,
                    "model_type": cfg.nn_model_type,
                    "error": type(err).__name__,
                }
            )

    out = _set_empty_nn_columns(out, cfg, error="insufficient_oos_training_data")
    if predictions:
        pred = pd.concat(predictions, ignore_index=True).drop_duplicates("position", keep="last")
        for column in [
            "nn_probability_positive",
            "nn_probability_std",
            "nn_adjusted_probability",
            "nn_expected_return_proxy",
            "nn_signal",
            "nn_confidence",
            "nn_model_type",
            "nn_error",
            "nn_is_oos",
            "nn_window_id",
        ]:
            out.iloc[pred["position"].to_numpy(dtype=int), out.columns.get_loc(column)] = pred[
                column
            ].to_numpy()
    if not _LAST_TRAINING_METRICS:
        _LAST_TRAINING_METRICS = [
            {
                "run_id": "",
                "fold_id": 0,
                "epoch": 0,
                "train_loss": np.nan,
                "val_loss": np.nan,
                "best_epoch": 0,
                "early_stop_triggered": False,
                "model_type": cfg.nn_model_type,
                "error": "insufficient_oos_training_data",
            }
        ]
    return out


def nn_predictions_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "nn_probability_positive",
        "nn_probability_std",
        "nn_adjusted_probability",
        "nn_signal",
        "target_forward_return",
        "target_positive",
        "nn_model_type",
        "nn_error",
        "nn_window_id",
        "nn_is_oos",
    ]
    out = df[[column for column in columns if column in df]].copy()
    out["date"] = df.index.date.astype(str)
    out["ticker"] = ""
    out["window_id"] = out.get("nn_window_id", df.get("hmm_window_id", 0))
    out["split"] = np.where(out.get("nn_is_oos", True), "oos", "not_oos")
    out["model_type"] = out.get("nn_model_type", "mlp")
    out["pred_prob"] = out.get("nn_probability_positive", np.nan)
    out["pred_prob_std"] = out.get("nn_probability_std", np.nan)
    out["adjusted_prob"] = out.get("nn_adjusted_probability", np.nan)
    out["pred_class"] = out.get("nn_signal", 0)
    out["label"] = out.get("target_positive", np.nan)
    out["is_oos"] = out.get("nn_is_oos", True)
    if "target_positive" in out and "nn_signal" in out:
        out["prediction_correct"] = ((out["nn_signal"] > 0) & (out["target_positive"] == 1.0)) | (
            (out["nn_signal"] < 0) & (out["target_positive"] == 0.0)
        )
    return out


def nn_training_metrics_frame(cfg: PipelineConfig) -> pd.DataFrame:
    if _LAST_TRAINING_METRICS:
        return pd.DataFrame(_LAST_TRAINING_METRICS)
    return pd.DataFrame(
        [
            {
                "run_id": "",
                "fold_id": 0,
                "epoch": 0,
                "train_loss": np.nan,
                "val_loss": np.nan,
                "best_epoch": 0,
                "early_stop_triggered": False,
                "model_type": cfg.nn_model_type,
                "error": (
                    "" if not cfg.enable_nn_confirmation or torch_available() else "torch_missing"
                ),
            }
        ]
    )
