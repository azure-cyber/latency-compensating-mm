"""
lob_features.py
────────────────────────────────────────────────────────────────────
LOB feature engineering pipeline for latency-compensating MM research.

Two modes:
  1. SYNTHETIC — generates realistic fake LOB data for development
  2. REAL      — reads Parquet files from the Kraken collector

Produces a feature tensor and fill-label dataset ready for the
fill probability model (fill_predictor.py).

Usage
-----
    # Synthetic mode (development)
    python lob_features.py --mode synthetic --rows 50000

    # Real mode (once collector has data)
    python lob_features.py --mode real --data-dir data/raw

Output
------
    data/features/features.parquet   — feature matrix
    data/features/labels.parquet     — fill labels per latency window

Requirements
------------
    pip install pandas numpy pyarrow loguru

Author : Independent Researcher — Carnegie Mellon University
Paper  : Latency-Compensating Market Making (v1, 2026)
"""

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ── Configuration ─────────────────────────────────────────────────────────────

DEPTH_LEVELS    = 10                        # Number of bid/ask levels
LOOKBACK        = 100                       # Snapshots per training window
LATENCY_WINDOWS = [200, 500, 1000, 2000]   # ms — the retail latency regime
QUOTE_OFFSETS   = [1, 2, 3, 5]             # Tick offsets to simulate orders at
OUTPUT_DIR      = Path("data/features")

# Synthetic data parameters
SYNTHETIC_SYMBOLS  = ["BTCUSD", "ETHUSD", "SOLUSD"]
SYNTHETIC_PRICES   = {"BTCUSD": 65000.0, "ETHUSD": 3500.0, "SOLUSD": 150.0}
SYNTHETIC_TICK     = {"BTCUSD": 0.10,    "ETHUSD": 0.01,   "SOLUSD": 0.001}
SYNTHETIC_INTERVAL = 0.1   # seconds between snapshots (100ms)

# ── Synthetic Data Generator ──────────────────────────────────────────────────

def generate_synthetic_lob(
    symbol: str,
    n_rows: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic LOB snapshot sequence.

    Simulates:
    - Mid-price following a geometric Brownian motion
    - Bid/ask levels with realistic depth profiles
    - Order book imbalance with mean-reverting dynamics
    - Intraday volatility clustering

    Parameters
    ----------
    symbol : str
        Asset symbol (e.g. 'BTCUSD')
    n_rows : int
        Number of snapshots to generate
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        LOB snapshot dataframe matching the collector output format
    """
    rng   = np.random.default_rng(seed)
    price = SYNTHETIC_PRICES.get(symbol, 1000.0)
    tick  = SYNTHETIC_TICK.get(symbol, 0.01)

    logger.info(f"Generating {n_rows:,} synthetic snapshots for {symbol}…")

    # ── Mid-price simulation (GBM with volatility clustering) ────────────────
    dt      = SYNTHETIC_INTERVAL / (252 * 24 * 3600)  # fraction of trading year
    sigma   = 0.35                                      # annual volatility
    returns = rng.normal(0, sigma * np.sqrt(dt), n_rows)

    # Add volatility clustering via GARCH-like effect
    vol_factor = np.ones(n_rows)
    for i in range(1, n_rows):
        vol_factor[i] = 0.95 * vol_factor[i-1] + 0.05 * abs(returns[i-1]) * 20
    returns *= (1 + vol_factor * 0.5)

    mid_prices = price * np.exp(np.cumsum(returns))

    # ── Spread simulation (wider during volatile periods) ─────────────────────
    base_spread_ticks = rng.integers(2, 5, n_rows)
    spread_ticks      = (base_spread_ticks * (1 + vol_factor)).astype(int).clip(2, 20)
    spreads           = spread_ticks * tick

    # ── Order book imbalance (mean-reverting) ─────────────────────────────────
    obi = np.zeros(n_rows)
    for i in range(1, n_rows):
        obi[i] = 0.85 * obi[i-1] + 0.15 * rng.normal(0, 0.3)
    obi = np.clip(obi, -0.95, 0.95)

    # ── Build rows ────────────────────────────────────────────────────────────
    start_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []

    for i in range(n_rows):
        mid    = mid_prices[i]
        spread = spreads[i]
        best_bid = mid - spread / 2
        best_ask = mid + spread / 2

        row = {
            "timestamp_utc": start_ts + timedelta(seconds=i * SYNTHETIC_INTERVAL),
            "symbol":        symbol,
            "mid_price":     mid,
            "spread":        spread,
            "spread_bps":    (spread / mid) * 10_000,
            "obi":           obi[i],
        }

        # Generate depth profile — exponentially decaying volume away from best
        total_bid_qty = 0.0
        total_ask_qty = 0.0

        for lvl in range(1, DEPTH_LEVELS + 1):
            # Price levels
            bid_price = best_bid - (lvl - 1) * tick
            ask_price = best_ask + (lvl - 1) * tick

            # Volume — larger at outer levels, modulated by imbalance
            base_vol     = rng.exponential(10.0 * (1.1 ** (lvl - 1)))
            bid_vol_mult = 1.0 + 0.5 * obi[i]   # more bid volume when OBI > 0
            ask_vol_mult = 1.0 - 0.5 * obi[i]

            bid_qty = max(0.01, base_vol * bid_vol_mult)
            ask_qty = max(0.01, base_vol * ask_vol_mult)

            row[f"bid_price_{lvl}"] = round(bid_price, 8)
            row[f"bid_qty_{lvl}"]   = round(bid_qty, 4)
            row[f"ask_price_{lvl}"] = round(ask_price, 8)
            row[f"ask_qty_{lvl}"]   = round(ask_qty, 4)

            total_bid_qty += bid_qty
            total_ask_qty += ask_qty

        row["depth_imbalance"] = (
            (total_bid_qty - total_ask_qty)
            / (total_bid_qty + total_ask_qty + 1e-9)
        )

        rows.append(row)

    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    logger.success(f"Generated {len(df):,} rows for {symbol}.")
    return df


# ── Real Data Loader ──────────────────────────────────────────────────────────

def load_real_lob(data_dir: Path, symbol: str) -> pd.DataFrame:
    """
    Load and concatenate all Parquet files for a given symbol.

    Parameters
    ----------
    data_dir : Path
        Root data directory (e.g. data/raw)
    symbol : str
        Asset symbol directory name (e.g. BTCUSD)

    Returns
    -------
    pd.DataFrame
        Concatenated LOB snapshots sorted by timestamp
    """
    symbol_dir = data_dir / symbol
    if not symbol_dir.exists():
        raise FileNotFoundError(f"No data directory found for {symbol} at {symbol_dir}")

    parquet_files = sorted(symbol_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found under {symbol_dir}")

    logger.info(f"Loading {len(parquet_files)} Parquet files for {symbol}…")
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df  = pd.concat(dfs, ignore_index=True)
    df  = df.sort_values("timestamp_utc").reset_index(drop=True)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    logger.success(f"Loaded {len(df):,} snapshots for {symbol}.")
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features from raw LOB snapshots.

    Features added:
    - Price normalizations (relative to mid-price)
    - Volume normalizations (rolling z-score)
    - Multi-level imbalance features
    - Lagged mid-price returns
    - Rolling volatility estimate

    Parameters
    ----------
    df : pd.DataFrame
        Raw LOB snapshot dataframe

    Returns
    -------
    pd.DataFrame
        Dataframe with additional feature columns
    """
    logger.info("Engineering features…")
    df = df.copy()

    mid = df["mid_price"]

    # ── Normalize prices relative to mid ──────────────────────────────────────
    for lvl in range(1, DEPTH_LEVELS + 1):
        df[f"bid_price_{lvl}_norm"] = (mid - df[f"bid_price_{lvl}"]) / mid
        df[f"ask_price_{lvl}_norm"] = (df[f"ask_price_{lvl}"] - mid) / mid

    # ── Normalize volumes by rolling mean ─────────────────────────────────────
    roll_window = 10
    for lvl in range(1, DEPTH_LEVELS + 1):
        for side in ("bid", "ask"):
            col      = f"{side}_qty_{lvl}"
            roll_mean = df[col].rolling(roll_window, min_periods=1).mean()
            roll_std  = df[col].rolling(roll_window, min_periods=1).std().clip(lower=1e-6)
            df[f"{col}_norm"] = (df[col] - roll_mean) / roll_std

    # ── Multi-level imbalance features ────────────────────────────────────────
    for lvl in range(1, DEPTH_LEVELS + 1):
        bq = df[f"bid_qty_{lvl}"]
        aq = df[f"ask_qty_{lvl}"]
        df[f"imbalance_{lvl}"] = (bq - aq) / (bq + aq + 1e-9)

    # Cumulative imbalance across levels
    for k in [3, 5, 10]:
        bid_cum = sum(df[f"bid_qty_{lvl}"] for lvl in range(1, k + 1))
        ask_cum = sum(df[f"ask_qty_{lvl}"] for lvl in range(1, k + 1))
        df[f"cum_imbalance_{k}"] = (bid_cum - ask_cum) / (bid_cum + ask_cum + 1e-9)

    # ── Mid-price returns (lagged) ────────────────────────────────────────────
    for lag in [1, 2, 3, 5]:
        df[f"return_lag_{lag}"] = mid.pct_change(lag)

    # ── Rolling volatility ────────────────────────────────────────────────────
    df["return_1"]     = mid.pct_change(1)
    df["vol_20"]       = df["return_1"].rolling(20).std()
    df["vol_100"]      = df["return_1"].rolling(100).std()
    df["vol_ratio"]    = (df["vol_20"] / (df["vol_100"] + 1e-9)).clip(0, 5)

    # ── Spread features ───────────────────────────────────────────────────────
    df["spread_norm"]      = df["spread"] / mid
    df["spread_roll_mean"] = df["spread_bps"].rolling(100).mean()
    df["spread_roll_z"]    = (
        (df["spread_bps"] - df["spread_roll_mean"])
        / (df["spread_bps"].rolling(100).std().clip(lower=1e-6))
    )

    # ── Drop NaN rows from rolling computations ───────────────────────────────
    df = df.fillna(0).reset_index(drop=True)

    logger.success(f"Feature engineering complete. Shape: {df.shape}")
    return df


# ── Fill Label Construction ───────────────────────────────────────────────────

def build_fill_labels(
    df: pd.DataFrame,
    latency_windows_ms: list[int] = LATENCY_WINDOWS,
    quote_offsets: list[int] = QUOTE_OFFSETS,
) -> pd.DataFrame:
    """
    Simulate limit order submissions and compute fill labels.

    For each snapshot and each (latency_window, quote_offset) combination,
    determines whether a simulated limit order would have filled within
    the latency window based on subsequent price movement.

    This implements the fill probability dataset construction described
    in Section 6.2 of the paper.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered LOB dataframe
    latency_windows_ms : list[int]
        Latency windows in milliseconds to label
    quote_offsets : list[int]
        Tick offsets from best bid/ask to simulate orders at

    Returns
    -------
    pd.DataFrame
        Label dataframe with columns:
        snapshot_idx, side, offset, latency_ms, filled, fill_time_ms
    """
    logger.info(
        f"Building fill labels for "
        f"{len(latency_windows_ms)} latency windows × "
        f"{len(quote_offsets)} offsets × 2 sides…"
    )

    mid       = df["mid_price"].values
    spread    = df["spread"].values
    tick_size = spread / 4   # approximate tick from spread
    n         = len(df)

    records = []
    interval_ms = SYNTHETIC_INTERVAL * 1000   # 100ms per snapshot

    for i in range(n - max(latency_windows_ms) // int(interval_ms) - 1):
        mid_i  = mid[i]
        tick_i = tick_size[i]

        for offset in quote_offsets:
            bid_price = mid_i - offset * tick_i
            ask_price = mid_i + offset * tick_i

            for lat_ms in latency_windows_ms:
                # Number of snapshots covered by this latency window
                n_steps = max(1, int(lat_ms / interval_ms))
                end_idx = min(i + n_steps, n - 1)

                future_mids = mid[i+1 : end_idx+1]

                # Bid fills if ask price drops to or below bid_price
                # (i.e. mid drops enough that asks cross our bid)
                bid_filled    = bool(np.any(future_mids <= bid_price + tick_i))
                bid_fill_time = None
                if bid_filled:
                    fill_step     = np.argmax(future_mids <= bid_price + tick_i)
                    bid_fill_time = (fill_step + 1) * interval_ms

                # Ask fills if bid price rises to or above ask_price
                ask_filled    = bool(np.any(future_mids >= ask_price - tick_i))
                ask_fill_time = None
                if ask_filled:
                    fill_step     = np.argmax(future_mids >= ask_price - tick_i)
                    ask_fill_time = (fill_step + 1) * interval_ms

                records.append({
                    "snapshot_idx": i,
                    "side":         "bid",
                    "offset":       offset,
                    "latency_ms":   lat_ms,
                    "filled":       int(bid_filled),
                    "fill_time_ms": bid_fill_time,
                    "mid_at_submit": mid_i,
                    "quote_price":   bid_price,
                })
                records.append({
                    "snapshot_idx": i,
                    "side":         "ask",
                    "offset":       offset,
                    "latency_ms":   lat_ms,
                    "filled":       int(ask_filled),
                    "fill_time_ms": ask_fill_time,
                    "mid_at_submit": mid_i,
                    "quote_price":   ask_price,
                })

    labels = pd.DataFrame(records)
    fill_rate = labels["filled"].mean()
    logger.success(
        f"Built {len(labels):,} label records. "
        f"Overall fill rate: {fill_rate:.1%}"
    )
    return labels


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    mode: str = "synthetic",
    data_dir: Path = Path("data/raw"),
    n_synthetic_rows: int = 50_000,
    symbol: str = "BTCUSD",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full feature engineering pipeline.

    Parameters
    ----------
    mode : str
        'synthetic' or 'real'
    data_dir : Path
        Root data directory (used in real mode only)
    n_synthetic_rows : int
        Number of synthetic snapshots to generate
    symbol : str
        Asset symbol to process

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (features, labels) dataframes
    """
    t0 = time.time()
    logger.info(f"Running pipeline in {mode.upper()} mode for {symbol}.")

    # ── Load or generate raw data ──────────────────────────────────────────────
    if mode == "synthetic":
        raw = generate_synthetic_lob(symbol, n_synthetic_rows)
    elif mode == "real":
        raw = load_real_lob(data_dir, symbol)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'synthetic' or 'real'.")

    # ── Feature engineering ───────────────────────────────────────────────────
    features = engineer_features(raw)

    # ── Fill label construction ───────────────────────────────────────────────
    labels = build_fill_labels(features)

    # ── Save outputs ──────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feat_path  = OUTPUT_DIR / f"{symbol}_features.parquet"
    label_path = OUTPUT_DIR / f"{symbol}_labels.parquet"

    features.to_parquet(feat_path,  index=False, compression="snappy")
    labels.to_parquet(label_path,   index=False, compression="snappy")

    elapsed = time.time() - t0
    logger.success(f"Pipeline complete in {elapsed:.1f}s.")
    logger.success(f"Features → {feat_path}  ({len(features):,} rows)")
    logger.success(f"Labels   → {label_path}  ({len(labels):,} rows)")

    # ── Summary statistics ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)
    print(f"Snapshots  : {len(features):,}")
    print(f"Features   : {features.shape[1]}")
    print(f"Symbol     : {symbol}")
    print(f"Time range : {features['timestamp_utc'].min()} →")
    print(f"             {features['timestamp_utc'].max()}")

    print("\n" + "="*60)
    print("FILL RATE BY LATENCY WINDOW AND OFFSET")
    print("="*60)
    pivot = labels.groupby(["latency_ms", "offset"])["filled"].mean().unstack()
    print(pivot.round(3).to_string())
    print("="*60 + "\n")

    return features, labels


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="LOB feature engineering pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "real"],
        default="synthetic",
        help="Data source mode (default: synthetic)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Root data directory for real mode",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=50_000,
        help="Number of synthetic rows to generate (default: 50000)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSD",
        choices=["BTCUSD", "ETHUSD", "SOLUSD"],
        help="Asset symbol to process (default: BTCUSD)",
    )
    return parser.parse_args()


# ── Unit tests ────────────────────────────────────────────────────────────────

def run_tests():
    """Basic unit tests for core pipeline functions."""
    logger.info("Running unit tests…")

    # Test 1: Synthetic generation produces correct shape
    df = generate_synthetic_lob("BTCUSD", 1000, seed=0)
    assert len(df) == 1000, "Wrong number of rows"
    assert "mid_price" in df.columns, "Missing mid_price"
    assert "bid_price_1" in df.columns, "Missing bid_price_1"
    assert "ask_price_1" in df.columns, "Missing ask_price_1"
    logger.success("Test 1 passed: synthetic generation shape correct.")

    # Test 2: Bid prices always below ask prices
    assert (df["bid_price_1"] < df["ask_price_1"]).all(), "Crossed order book detected"
    logger.success("Test 2 passed: no crossed order book.")

    # Test 3: OBI is bounded [-1, 1]
    assert df["obi"].between(-1, 1).all(), "OBI out of bounds"
    logger.success("Test 3 passed: OBI bounded correctly.")

    # Test 4: Feature engineering runs without error
    features = engineer_features(df)
    assert len(features) > 0, "Feature engineering produced empty dataframe"
    assert "cum_imbalance_10" in features.columns, "Missing cumulative imbalance"
    logger.success("Test 4 passed: feature engineering complete.")

    # Test 5: Fill labels have correct columns
    labels = build_fill_labels(features.head(500), latency_windows_ms=[200, 500])
    assert "filled" in labels.columns, "Missing filled column"
    assert labels["filled"].isin([0, 1]).all(), "Non-binary fill labels"
    assert (labels["fill_time_ms"].dropna() > 0).all(), "Non-positive fill times"
    logger.success("Test 5 passed: fill labels valid.")

    logger.success("All tests passed.")


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "synthetic" and args.rows == 50_000:
        # Run tests first in default mode
        run_tests()

    run_pipeline(
        mode=args.mode,
        data_dir=args.data_dir,
        n_synthetic_rows=args.rows,
        symbol=args.symbol,
    )
