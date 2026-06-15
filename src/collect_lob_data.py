"""
collect_lob_data.py
────────────────────────────────────────────────────────────────────
Binance WebSocket LOB collector for latency-compensating MM research.

Connects to Binance depth streams for BTC/USDT, ETH/USDT, SOL/USDT,
records top-10 bid/ask levels at each update, and saves compressed
Parquet files partitioned by asset / date / hour.

Usage
-----
    python collect_lob_data.py

    # Runs continuously until Ctrl+C.
    # Data lands in:  ./data/raw/<SYMBOL>/<YYYY-MM-DD>/<HH>.parquet

Requirements
------------
    pip install websocket-client pandas pyarrow schedule loguru

Author : Independent Researcher — Carnegie Mellon University
Paper  : Latency-Compensating Market Making (v1, 2026)
"""

import json
import os
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import websocket
from loguru import logger

# ── Configuration ────────────────────────────────────────────────────────────

SYMBOLS = ["btcusdt", "ethusdt", "solusdt"]   # Binance stream symbols (lowercase)
DEPTH_LEVELS = 10                              # Top-N bid/ask levels to record
FLUSH_INTERVAL_SECONDS = 300                  # Write to disk every 5 minutes
DATA_DIR = Path("data/raw")                   # Root output directory
RECONNECT_DELAY_SECONDS = 5                   # Wait before reconnecting on error
MAX_RECONNECT_ATTEMPTS = 20                   # Give up after this many retries

# ── Globals ───────────────────────────────────────────────────────────────────

# Buffer: symbol -> list of snapshot dicts
_buffers: dict[str, list[dict]] = defaultdict(list)
_buffer_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _output_path(symbol: str, ts: datetime) -> Path:
    """Return the Parquet path for a given symbol and UTC timestamp."""
    return (
        DATA_DIR
        / symbol.upper()
        / ts.strftime("%Y-%m-%d")
        / f"{ts.strftime('%H')}.parquet"
    )


def _parse_snapshot(symbol: str, msg: dict) -> dict | None:
    """
    Parse a Binance @depth20 WebSocket message into a flat dict.

    Returns None if the message is malformed or missing required fields.
    """
    try:
        ts_ms = msg.get("T") or msg.get("E")   # transaction time or event time
        if ts_ms is None:
            return None

        bids = msg["bids"][:DEPTH_LEVELS]
        asks = msg["asks"][:DEPTH_LEVELS]

        row: dict = {
            "timestamp_utc": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
            "symbol": symbol.upper(),
        }

        for i, (price, qty) in enumerate(bids, start=1):
            row[f"bid_price_{i}"] = float(price)
            row[f"bid_qty_{i}"]   = float(qty)

        for i, (price, qty) in enumerate(asks, start=1):
            row[f"ask_price_{i}"] = float(price)
            row[f"ask_qty_{i}"]   = float(qty)

        # Derived features (pre-computed to save time during training)
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid      = (best_bid + best_ask) / 2.0
        spread   = best_ask - best_bid

        total_bid_qty = sum(float(q) for _, q in bids)
        total_ask_qty = sum(float(q) for _, q in asks)

        row["mid_price"]       = mid
        row["spread"]          = spread
        row["spread_bps"]      = (spread / mid) * 10_000 if mid > 0 else None
        row["obi"]             = (            # order book imbalance (level 1)
            (float(bids[0][1]) - float(asks[0][1]))
            / (float(bids[0][1]) + float(asks[0][1]) + 1e-9)
        )
        row["depth_imbalance"] = (            # depth imbalance (all levels)
            (total_bid_qty - total_ask_qty)
            / (total_bid_qty + total_ask_qty + 1e-9)
        )

        return row

    except (KeyError, IndexError, ValueError, TypeError) as exc:
        logger.warning(f"[{symbol}] Malformed message skipped: {exc}")
        return None


def _flush_buffer(symbol: str, force: bool = False) -> None:
    """
    Write buffered snapshots for *symbol* to Parquet, partitioned by hour.

    Called from the flush thread every FLUSH_INTERVAL_SECONDS, or on shutdown.
    """
    with _buffer_lock:
        rows = _buffers[symbol]
        if not rows:
            return
        _buffers[symbol] = []   # clear buffer before releasing lock

    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    # Group by hour so each file stays reasonably small
    for hour_ts, group in df.groupby(df["timestamp_utc"].dt.floor("h")):
        path = _output_path(symbol, hour_ts.to_pydatetime())
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            # Append to existing Parquet by reading + concatenating
            existing = pd.read_parquet(path)
            group    = pd.concat([existing, group], ignore_index=True)

        group.to_parquet(path, index=False, compression="snappy")
        logger.info(
            f"[{symbol}] Flushed {len(group):,} rows → {path} "
            f"(buffer was {len(rows):,})"
        )


def _flush_all(force: bool = False) -> None:
    """Flush all symbol buffers."""
    for symbol in SYMBOLS:
        try:
            _flush_buffer(symbol, force=force)
        except Exception as exc:
            logger.error(f"Flush error for {symbol}: {exc}")


# ── WebSocket callbacks ───────────────────────────────────────────────────────

def _make_callbacks(symbol: str):
    """Return (on_message, on_error, on_close, on_open) for a given symbol."""

    def on_message(ws, raw):
        try:
            msg = json.loads(raw)
            row = _parse_snapshot(symbol, msg)
            if row is not None:
                with _buffer_lock:
                    _buffers[symbol].append(row)
        except Exception as exc:
            logger.error(f"[{symbol}] on_message error: {exc}")

    def on_error(ws, error):
        logger.warning(f"[{symbol}] WebSocket error: {error}")

    def on_close(ws, close_status_code, close_msg):
        logger.warning(
            f"[{symbol}] Connection closed "
            f"(code={close_status_code}, msg={close_msg})"
        )

    def on_open(ws):
        logger.success(f"[{symbol}] Connected to Binance depth stream.")

    return on_message, on_error, on_close, on_open


# ── Stream thread ─────────────────────────────────────────────────────────────

def _run_stream(symbol: str) -> None:
    """
    Connect to Binance WebSocket depth stream for *symbol* and run forever,
    reconnecting automatically on disconnection or error.
    """
    url = f"wss://stream.binance.com:9443/ws/{symbol}@depth20@100ms"
    attempts = 0

    while attempts < MAX_RECONNECT_ATTEMPTS:
        try:
            on_message, on_error, on_close, on_open = _make_callbacks(symbol)
            ws = websocket.WebSocketApp(
                url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open,
            )
            ws.run_forever(ping_interval=30, ping_timeout=10)
        except Exception as exc:
            logger.error(f"[{symbol}] Stream crashed: {exc}")

        attempts += 1
        logger.info(
            f"[{symbol}] Reconnecting in {RECONNECT_DELAY_SECONDS}s "
            f"(attempt {attempts}/{MAX_RECONNECT_ATTEMPTS})…"
        )
        time.sleep(RECONNECT_DELAY_SECONDS)

    logger.critical(f"[{symbol}] Max reconnect attempts reached. Giving up.")


# ── Flush thread ──────────────────────────────────────────────────────────────

def _run_flush_loop() -> None:
    """Periodically flush all buffers to disk."""
    while True:
        time.sleep(FLUSH_INTERVAL_SECONDS)
        logger.info("Scheduled flush triggered…")
        _flush_all()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logger.add(
        "data/collector.log",
        rotation="100 MB",
        retention="30 days",
        compression="gz",
        level="INFO",
    )
    logger.info("Starting Binance LOB collector.")
    logger.info(f"Symbols : {SYMBOLS}")
    logger.info(f"Levels  : {DEPTH_LEVELS} per side")
    logger.info(f"Output  : {DATA_DIR.resolve()}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # One thread per symbol
    stream_threads = []
    for symbol in SYMBOLS:
        t = threading.Thread(
            target=_run_stream,
            args=(symbol,),
            name=f"stream-{symbol}",
            daemon=True,
        )
        t.start()
        stream_threads.append(t)
        logger.info(f"Stream thread started for {symbol}.")

    # Flush thread
    flush_thread = threading.Thread(
        target=_run_flush_loop,
        name="flush-loop",
        daemon=True,
    )
    flush_thread.start()
    logger.info("Flush thread started.")

    # Keep main thread alive; flush on exit
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutdown requested — flushing remaining data…")
        _flush_all(force=True)
        logger.success("All data flushed. Collector stopped.")


if __name__ == "__main__":
    main()
