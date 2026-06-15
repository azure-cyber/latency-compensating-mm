"""
fill_predictor.py
────────────────────────────────────────────────────────────────────
Fill probability models for latency-compensating MM research.

Trains two models as described in Section 4 of the paper:
  1. LSTM baseline
  2. Conv-Transformer main model (with latency conditioning)

Both use survival analysis (Cox partial likelihood loss) to predict
the full time-to-fill distribution for limit orders under retail
execution latency.

Usage
-----
    # Train both models
    python src/fill_predictor.py --model both --epochs 30

    # Train LSTM only
    python src/fill_predictor.py --model lstm --epochs 30

    # Train Conv-Transformer only
    python src/fill_predictor.py --model convtransformer --epochs 30

Output
------
    models/lstm_fill.pt              — saved LSTM weights
    models/convtransformer_fill.pt   — saved Conv-Transformer weights
    results/fill_predictor_results.csv — evaluation metrics

Requirements
------------
    pip install torch pandas numpy pyarrow loguru scikit-learn

Author : Independent Researcher — Carnegie Mellon University
Paper  : Latency-Compensating Market Making (v1, 2026)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# ── Configuration ─────────────────────────────────────────────────────────────

LOOKBACK        = 50        # Snapshots per training window (reduced for speed)
LATENCY_WINDOWS = [200, 500, 1000, 2000]
BATCH_SIZE      = 256
LEARNING_RATE   = 1e-3
PATIENCE        = 5         # Early stopping patience
HIDDEN_DIM      = 64
N_HEADS         = 4
N_TRANSFORMER_LAYERS = 2
DROPOUT         = 0.2
K_SURVIVAL      = 50        # Discretization points for survival function

FEATURES_PATH   = Path("data/features/BTCUSD_features.parquet")
LABELS_PATH     = Path("data/features/BTCUSD_labels.parquet")
MODELS_DIR      = Path("models")
RESULTS_DIR     = Path("results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Feature columns used as model input ───────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns excluding metadata."""
    exclude = {"timestamp_utc", "symbol"}
    return [
        c for c in df.columns
        if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]


# ── Dataset ───────────────────────────────────────────────────────────────────

class FillDataset(Dataset):
    """
    PyTorch Dataset for fill probability training.

    Each sample is:
      x       : (LOOKBACK, n_features) tensor of LOB snapshots
      latency : scalar latency window in ms (normalized)
      t_obs   : observed time (fill time or censoring time) in ms
      delta   : 1 if filled, 0 if censored
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: pd.DataFrame,
        lookback: int = LOOKBACK,
        max_latency: float = 2000.0,
    ):
        self.features   = features.astype(np.float32)
        self.labels     = labels.reset_index(drop=True)
        self.lookback   = lookback
        self.max_latency = max_latency

        # Filter to valid indices (need lookback history)
        self.valid_idx = self.labels[
            self.labels["snapshot_idx"] >= lookback
        ].index.tolist()

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, i):
        row     = self.labels.iloc[self.valid_idx[i]]
        snap_i  = int(row["snapshot_idx"])

        # LOB feature window: (lookback, n_features)
        x = self.features[snap_i - self.lookback : snap_i]

        # Latency window normalized to [0, 1]
        latency_norm = float(row["latency_ms"]) / self.max_latency

        # Survival analysis targets
        fill_time = row["fill_time_ms"]
        filled    = int(row["filled"])

        if filled and fill_time is not None:
            t_obs = float(fill_time)
        else:
            t_obs = float(row["latency_ms"])   # censored at latency window

        # Normalize time to [0, 1]
        t_obs_norm = t_obs / self.max_latency

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(latency_norm, dtype=torch.float32),
            torch.tensor(t_obs_norm, dtype=torch.float32),
            torch.tensor(filled, dtype=torch.float32),
        )


# ── Survival Loss ─────────────────────────────────────────────────────────────

class SurvivalLoss(nn.Module):
    """
    Cox partial likelihood loss for discrete survival analysis.

    L = -1/N * sum[ delta_i * log(h(t_i)) + log(S(t_i)) ]

    where h(t) is the hazard and S(t) is the survival function.
    """

    def forward(
        self,
        log_hazards: torch.Tensor,   # (B, K) log hazard at each time point
        t_obs: torch.Tensor,          # (B,) observed time normalized [0,1]
        delta: torch.Tensor,          # (B,) fill indicator
        k: int = K_SURVIVAL,
    ) -> torch.Tensor:
        B = log_hazards.shape[0]

        # Convert log hazards to hazards and survival
        hazards  = torch.sigmoid(log_hazards)                          # (B, K)
        survival = torch.cumprod(1.0 - hazards + 1e-7, dim=1)        # (B, K)

        # Map observed time to discrete bin index
        t_idx = (t_obs * (k - 1)).long().clamp(0, k - 1)             # (B,)

        # Hazard at observed time
        h_at_t = hazards[torch.arange(B), t_idx].clamp(1e-7, 1.0)   # (B,)

        # Survival at observed time
        s_at_t = survival[torch.arange(B), t_idx].clamp(1e-7, 1.0)  # (B,)

        # Cox partial likelihood
        loss = -(delta * torch.log(h_at_t) + torch.log(s_at_t))
        return loss.mean()


# ── Monotonic Output Layer ────────────────────────────────────────────────────

class MonotonicSurvivalHead(nn.Module):
    """
    Maps a representation vector to a monotonically non-increasing
    survival function S(t_1), ..., S(t_K).

    Monotonicity enforced via cumulative softmax on log hazards.
    S(0) = 1 enforced by construction.
    """

    def __init__(self, input_dim: int, k: int = K_SURVIVAL):
        super().__init__()
        self.k   = k
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, k),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            log_hazards : (B, K) raw log hazard outputs
            survival    : (B, K) monotone survival function
        """
        log_hazards = self.net(x)                                      # (B, K)
        hazards     = torch.sigmoid(log_hazards)
        survival    = torch.cumprod(1.0 - hazards + 1e-7, dim=1)     # (B, K)
        return log_hazards, survival


# ── LSTM Baseline ─────────────────────────────────────────────────────────────

class LSTMFillPredictor(nn.Module):
    """
    LSTM baseline fill probability model (Section 4.3).

    Architecture:
      Input projection → 2-layer LSTM → Latency MLP → Survival head
    """

    def __init__(self, n_features: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden_dim)
        self.lstm       = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=DROPOUT,
        )
        # Latency conditioning MLP
        self.latency_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )
        self.survival_head = MonotonicSurvivalHead(hidden_dim + 16)

    def forward(
        self,
        x: torch.Tensor,           # (B, L, F)
        latency: torch.Tensor,     # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Project features
        x = self.input_proj(x)                    # (B, L, H)

        # LSTM encoding
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]                               # (B, H) — last layer hidden state

        # Latency conditioning
        lat = self.latency_mlp(latency.unsqueeze(-1))   # (B, 16)

        # Concatenate and decode
        combined = torch.cat([h, lat], dim=-1)    # (B, H+16)
        return self.survival_head(combined)


# ── Conv-Transformer Main Model ───────────────────────────────────────────────

class ConvTransformerFillPredictor(nn.Module):
    """
    Conv-Transformer fill probability model (Section 4.4).

    Architecture:
      Conv encoder (spatial) → Transformer encoder (temporal)
      → Latency conditioning → Survival head
    """

    def __init__(self, n_features: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()

        # Convolutional encoder — extracts spatial LOB features
        self.conv_encoder = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, hidden_dim, kernel_size=1),
            nn.ReLU(),
        )

        # Transformer encoder — captures temporal dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=N_HEADS,
            dim_feedforward=128,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=N_TRANSFORMER_LAYERS,
        )

        # Latency conditioning MLP
        self.latency_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        # Survival head
        self.survival_head = MonotonicSurvivalHead(hidden_dim + 32)

    def forward(
        self,
        x: torch.Tensor,           # (B, L, F)
        latency: torch.Tensor,     # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, F = x.shape

        # Conv encoder operates on (B, F, L)
        x_conv = self.conv_encoder(x.permute(0, 2, 1))   # (B, H, L)
        x_conv = x_conv.permute(0, 2, 1)                  # (B, L, H)

        # Transformer encoder
        x_trans = self.transformer(x_conv)                # (B, L, H)

        # Global average pooling over time
        h = x_trans.mean(dim=1)                           # (B, H)

        # Latency conditioning
        lat = self.latency_mlp(latency.unsqueeze(-1))     # (B, 32)

        # Concatenate and decode
        combined = torch.cat([h, lat], dim=-1)            # (B, H+32)
        return self.survival_head(combined)


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    model_name: str,
) -> dict:
    """
    Train a fill probability model with early stopping.

    Parameters
    ----------
    model : nn.Module
        LSTM or Conv-Transformer model
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    epochs : int
        Maximum training epochs
    model_name : str
        Name for logging and saving

    Returns
    -------
    dict
        Training history and best validation loss
    """
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    criterion = SurvivalLoss()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / f"{model_name}_fill.pt"

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    logger.info(f"Training {model_name} on {DEVICE}…")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        train_losses = []

        for x, latency, t_obs, delta in train_loader:
            x       = x.to(DEVICE)
            latency = latency.to(DEVICE)
            t_obs   = t_obs.to(DEVICE)
            delta   = delta.to(DEVICE)

            optimizer.zero_grad()
            log_hazards, _ = model(x, latency)
            loss = criterion(log_hazards, t_obs, delta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_losses = []

        with torch.no_grad():
            for x, latency, t_obs, delta in val_loader:
                x       = x.to(DEVICE)
                latency = latency.to(DEVICE)
                t_obs   = t_obs.to(DEVICE)
                delta   = delta.to(DEVICE)

                log_hazards, _ = model(x, latency)
                loss = criterion(log_hazards, t_obs, delta)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        elapsed    = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        logger.info(
            f"[{model_name}] Epoch {epoch:03d}/{epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.success(f"[{model_name}] New best val loss: {best_val_loss:.4f} — saved.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"[{model_name}] Early stopping at epoch {epoch}.")
                break

    # Load best weights
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    history["best_val_loss"] = best_val_loss
    return history


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    model_name: str,
) -> dict:
    """
    Evaluate a trained fill probability model.

    Computes:
    - Concordance index (C-index)
    - Brier score at each latency window
    - Overall accuracy

    Parameters
    ----------
    model : nn.Module
        Trained model
    test_loader : DataLoader
        Test data loader
    model_name : str
        Name for reporting

    Returns
    -------
    dict
        Evaluation metrics
    """
    model.eval()
    all_preds   = []
    all_t_obs   = []
    all_delta   = []
    all_latency = []

    with torch.no_grad():
        for x, latency, t_obs, delta in test_loader:
            x       = x.to(DEVICE)
            latency = latency.to(DEVICE)

            _, survival = model(x, latency)

            # Fill probability = 1 - S(end of latency window)
            fill_prob = 1.0 - survival[:, -1]

            all_preds.append(fill_prob.cpu().numpy())
            all_t_obs.append(t_obs.numpy())
            all_delta.append(delta.numpy())
            all_latency.append(latency.cpu().numpy())

    preds   = np.concatenate(all_preds)
    t_obs   = np.concatenate(all_t_obs)
    delta   = np.concatenate(all_delta)
    latency = np.concatenate(all_latency)

    # ── Concordance Index (C-index) ───────────────────────────────────────────
    # Fraction of concordant pairs: higher pred → shorter fill time
    filled_mask = delta == 1
    c_index = _concordance_index(preds[filled_mask], t_obs[filled_mask])

    # ── Brier Score ───────────────────────────────────────────────────────────
    brier = np.mean((preds - delta) ** 2)

    # ── Accuracy (threshold at 0.5) ───────────────────────────────────────────
    accuracy = np.mean((preds > 0.5).astype(float) == delta)

    metrics = {
        "model":     model_name,
        "c_index":   round(float(c_index), 4),
        "brier":     round(float(brier), 4),
        "accuracy":  round(float(accuracy), 4),
    }

    logger.success(
        f"[{model_name}] C-index: {c_index:.4f} | "
        f"Brier: {brier:.4f} | Accuracy: {accuracy:.4f}"
    )
    return metrics


def _concordance_index(pred_fill_prob: np.ndarray, fill_times: np.ndarray) -> float:
    """
    Compute concordance index: higher predicted fill probability
    should correspond to shorter actual fill time.
    """
    n = len(pred_fill_prob)
    if n < 2:
        return 0.5

    concordant = 0
    total      = 0

    # Sample pairs for efficiency on large datasets
    max_pairs = 50_000
    if n * (n - 1) // 2 > max_pairs:
        idx = np.random.choice(n, size=int(np.sqrt(max_pairs * 2)), replace=False)
        pred_fill_prob = pred_fill_prob[idx]
        fill_times     = fill_times[idx]
        n = len(idx)

    for i in range(n):
        for j in range(i + 1, n):
            if fill_times[i] != fill_times[j]:
                total += 1
                if (pred_fill_prob[i] > pred_fill_prob[j]) == (fill_times[i] < fill_times[j]):
                    concordant += 1

    return concordant / total if total > 0 else 0.5


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(
    max_labels: int = 200_000,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """
    Load features and labels, subsample if needed.

    Parameters
    ----------
    max_labels : int
        Maximum number of label rows to use (for memory efficiency)

    Returns
    -------
    tuple
        (feature_array, labels_df, feature_cols)
    """
    logger.info("Loading features and labels…")

    features_df = pd.read_parquet(FEATURES_PATH)
    labels_df   = pd.read_parquet(LABELS_PATH)

    feat_cols = get_feature_cols(features_df)
    features  = features_df[feat_cols].values.astype(np.float32)

    # Replace NaN/inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Subsample labels if too large
    if len(labels_df) > max_labels:
        labels_df = labels_df.sample(max_labels, random_state=42).reset_index(drop=True)
        logger.info(f"Subsampled labels to {max_labels:,} rows.")

    logger.success(
        f"Loaded {len(features):,} snapshots × {len(feat_cols)} features, "
        f"{len(labels_df):,} labels."
    )
    return features, labels_df, feat_cols


# ── Main ──────────────────────────────────────────────────────────────────────

def main(model_type: str = "both", epochs: int = 30):
    logger.info(f"Fill Predictor — model={model_type}, epochs={epochs}, device={DEVICE}")

    # ── Load data ─────────────────────────────────────────────────────────────
    features, labels, feat_cols = load_data()
    n_features = len(feat_cols)

    # ── Train/val/test split (temporal — no shuffle) ──────────────────────────
    n = len(labels)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train_labels = labels.iloc[:train_end]
    val_labels   = labels.iloc[train_end:val_end]
    test_labels  = labels.iloc[val_end:]

    logger.info(
        f"Split: train={len(train_labels):,} | "
        f"val={len(val_labels):,} | test={len(test_labels):,}"
    )

    # ── Datasets and loaders ──────────────────────────────────────────────────
    train_ds = FillDataset(features, train_labels)
    val_ds   = FillDataset(features, val_labels)
    test_ds  = FillDataset(features, test_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_metrics = []

    # ── LSTM Baseline ─────────────────────────────────────────────────────────
    if model_type in ("lstm", "both"):
        lstm_model = LSTMFillPredictor(n_features=n_features)
        train_model(lstm_model, train_loader, val_loader, epochs, "lstm")
        metrics = evaluate_model(lstm_model, test_loader, "LSTM")
        all_metrics.append(metrics)

    # ── Conv-Transformer ──────────────────────────────────────────────────────
    if model_type in ("convtransformer", "both"):
        ct_model = ConvTransformerFillPredictor(n_features=n_features)
        train_model(ct_model, train_loader, val_loader, epochs, "convtransformer")
        metrics = evaluate_model(ct_model, test_loader, "ConvTransformer")
        all_metrics.append(metrics)

    # ── Save results ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(all_metrics)
    results_path = RESULTS_DIR / "fill_predictor_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "="*60)
    print("FILL PREDICTOR RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)
    print(f"\nResults saved → {results_path}")
    print(f"Models saved  → {MODELS_DIR}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fill probability models")
    parser.add_argument(
        "--model",
        choices=["lstm", "convtransformer", "both"],
        default="both",
        help="Which model to train (default: both)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum training epochs (default: 30)",
    )
    args = parser.parse_args()
    main(model_type=args.model, epochs=args.epochs)
