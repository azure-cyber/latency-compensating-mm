"""
collect_lob_data.py
────────────────────────────────────────────────────────────────────
Kraken WebSocket LOB collector for latency-compensating MM research.

Connects to Kraken v2 WebSocket API for BTC/USD, ETH/USD, SOL/USD,
records top-10 bid/ask levels at each update, and saves compressed
Parquet files partitioned by asset / date / hour.

Usage
-----
    python collect_lob_data.py

    # Runs continuously until Ctrl+C.
    # Data lands in:  ./data/raw/<SYMBOL>/<YYYY-MM-DD>/<HH>.parquet

Requirements
------------
    pip install websocket-client pandas pyarrow loguru

Author : Independent Researcher — Carnegie Mellon University
Paper  : Latency-Compensating Market Making (v1, 2026)
"""

import json
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import websocket
from loguru import logger

# ── Configuration ─────────────────────────────────────────────────────────────

SYMBOLS = {
    "BTC/USD": "BTCUSD",
    "ETH/USD": "ETHUSD",
    "SOL/USD": "SOLUSD",
}
DEPTH_LEVELS   = 10
FLUSH_INTERVAL = 300
DATA_DIR       = Path("data/raw")
RECONNECT_DELAY     = 12      # seconds between retries (5 per minute)
MAX_RECONNECT       = 50      # 50 retries × 12s = 10 minutes total
KRAKEN_WS_URL  = "wss://ws.kraken.com/v2"

# ── Globals ───────────────────────────────────────────────────────────────────

_buffers: dict[str, list[dict]] = defaultdict(list)
_buffer_lock = threading.Lock()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _output_path(symbol: str, ts: datetime) -> Path:
    return (
        DATA_DIR
        / symbol
        / ts.strftime("%Y-%m-%d")
        / f"{ts.strftime('%H')}.parquet"
    )


def _parse_snapshot(symbol_key: str, data: dict) -> dict | None:
    try:
        bids = sorted(data.get("bids", []), key=lambda x: -float(x["price"]))[:DEPTH_LEVELS]
        asks = sorted(data.get("asks", []), key=lambda x:  float(x["price"]))[:DEPTH_LEVELS]

        if not bids or not asks:
            return None

        ts_str = data.get("timestamp")
        if ts_str:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        else:
            ts = datetime.now(tz=timezone.utc)

        symbol_out = SYMBOLS[symbol_key]
        row: dict = {
            "timestamp_utc": ts,
            "symbol": symbol_out,
        }

        for i, level in enumerate(bids, start=1):
            row[f"bid_price_{i}"] = float(level["price"])
            row[f"bid_qty_{i}"]   = float(level["qty"])

        for i, level in enumerate(asks, start=1):
            row[f"ask_price_{i}"] = float(level["price"])
            row[f"ask_qty_{i}"]   = float(level["qty"])

        best_bid      = float(bids[0]["price"])
        best_ask      = float(asks[0]["price"])
        best_bid_qty  = float(bids[0]["qty"])
        best_ask_qty  = float(asks[0]["qty"])
        mid           = (best_bid + best_ask) / 2.0
        spread        = best_ask - best_bid
        total_bid_qty = sum(float(l["qty"]) for l in bids)
        total_ask_qty = sum(float(l["qty"]) for l in asks)

        row["mid_price"]       = mid
        row["spread"]          = spread
        row["spread_bps"]      = (spread / mid) * 10_000 if mid > 0 else None
        row["obi"]             = (
            (best_bid_qty - best_ask_qty)
            / (best_bid_qty + best_ask_qty + 1e-9)
        )
        row["depth_imbalance"] = (
            (total_bid_qty - total_ask_qty)
            / (total_bid_qty + total_ask_qty + 1e-9)
        )

        return row

    except (KeyError, IndexError, ValueError, TypeError) as exc:
        logger.warning(f"[{symbol_key}] Malformed message skipped: {exc}")
        return None


def _flush_buffer(symbol_key: str) -> None:
    symbol_out = SYMBOLS[symbol_key]
    with _buffer_lock:
        rows = _buffers[symbol_key]
        if not rows:
            return
        _buffers[symbol_key] = []

    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    for hour_ts, group in df.groupby(df["timestamp_utc"].dt.floor("h")):
        path = _output_path(symbol_out, hour_ts.to_pydatetime())
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = pd.read_parquet(path)
            group    = pd.concat([existing, group], ignore_index=True)

        group.to_parquet(path, index=False, compression="snappy")
        logger.info(f"[{symbol_key}] Flushed {len(group):,} rows → {path}")


def _flush_all() -> None:
    for symbol_key in SYMBOLS:
        try:
            _flush_buffer(symbol_key)
        except Exception as exc:
            logger.error(f"Flush error for {symbol_key}: {exc}")


# ── WebSocket handler ─────────────────────────────────────────────────────────

class KrakenBookHandler:

    def _subscribe_msg(self) -> str:
        return json.dumps({
            "method": "subscribe",
            "params": {
                "channel": "book",
                "symbol": list(SYMBOLS.keys()),
                "depth": DEPTH_LEVELS,
            }
        })

    def on_open(self, ws):
        logger.success("Connected to Kraken WebSocket v2.")
        ws.send(self._subscribe_msg())
        logger.info(f"Subscribed to: {list(SYMBOLS.keys())}")

    def on_message(self, ws, raw):
        try:
            msg = json.loads(raw)
            if msg.get("channel") != "book":
                return
            if msg.get("type") not in ("snapshot", "update"):
                return
            for item in msg.get("data", []):
                symbol_key = item.get("symbol")
                if symbol_key not in SYMBOLS:
                    continue
                row = _parse_snapshot(symbol_key, item)
                if row is not None:
                    with _buffer_lock:
                        _buffers[symbol_key].append(row)
        except Exception as exc:
            logger.error(f"on_message error: {exc}")

    def on_error(self, ws, error):
        logger.warning(f"WebSocket error: {error}")

    def on_close(self, ws, code, msg):
        logger.warning(f"Connection closed (code={code}, msg={msg})")

    def run(self):
        # Retry indefinitely in outer loop — inner loop handles
        # planned Kraken server restarts (up to 4 minutes of retries).
        # After 4 minutes of failed retries, outer loop resets and
        # we try again from scratch rather than giving up permanently.
        while True:
            attempts = 0
            connected_once = False
            while attempts < MAX_RECONNECT:
                try:
                    self._connected = False
                    ws = websocket.WebSocketApp(
                        KRAKEN_WS_URL,
                        on_open=self.on_open,
                        on_message=self.on_message,
                        on_error=self.on_error,
                        on_close=self.on_close,
                    )
                    ws.run_forever(ping_interval=30, ping_timeout=10)
                    connected_once = True
                    attempts = 0   # reset counter after clean disconnect
                except Exception as exc:
                    logger.error(f"Stream crashed: {exc}")
                attempts += 1
                if attempts < MAX_RECONNECT:
                    logger.info(
                        f"Reconnecting in {RECONNECT_DELAY}s "
                        f"(attempt {attempts}/{MAX_RECONNECT} — "
                        f"up to {MAX_RECONNECT * RECONNECT_DELAY // 60}min total)…"
                    )
                    time.sleep(RECONNECT_DELAY)
            logger.warning(
                f"Failed to reconnect after {MAX_RECONNECT} attempts "
                f"({MAX_RECONNECT * RECONNECT_DELAY}s). "
                f"Flushing data and restarting connection loop…"
            )
            _flush_all()
            time.sleep(60)   # wait 1 minute before outer loop retry


# ── Flush thread ──────────────────────────────────────────────────────────────

def _run_flush_loop():
    while True:
        time.sleep(FLUSH_INTERVAL)
        logger.info("Scheduled flush triggered…")
        _flush_all()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    logger.add(
        "data/collector.log",
        rotation="100 MB",
        retention="30 days",
        compression="gz",
        level="INFO",
    )
    logger.info("Starting Kraken LOB collector.")
    logger.info(f"Symbols : {list(SYMBOLS.keys())}")
    logger.info(f"Levels  : {DEPTH_LEVELS} per side")
    logger.info(f"Output  : {DATA_DIR.resolve()}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    flush_thread = threading.Thread(
        target=_run_flush_loop,
        name="flush-loop",
        daemon=True,
    )
    flush_thread.start()
    logger.info("Flush thread started.")

    handler = KrakenBookHandler()
    try:
        handler.run()
    except KeyboardInterrupt:
        logger.info("Shutdown requested — flushing remaining data…")
        _flush_all()
        logger.success("All data flushed. Collector stopped.")


if __name__ == "__main__":
    main()
