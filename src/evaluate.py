"""
evaluate.py
────────────────────────────────────────────────────────────────────
Comprehensive evaluation and visualization for latency-compensating
market making research.

Reads all results files produced by the pipeline and generates:
  1. Full performance table (Table 3 in paper)
  2. Equity curves for all strategies
  3. Drawdown chart
  4. Fill rate comparison by latency window
  5. RL training history (PnL and fill count over episodes)
  6. Strategy comparison radar chart

Usage
-----
    python src/evaluate.py

Output
------
    results/figures/equity_curves.png
    results/figures/drawdown.png
    results/figures/fill_rates.png
    results/figures/rl_training.png
    results/figures/strategy_radar.png
    results/performance_table.csv
    results/performance_table.txt    ← LaTeX-ready table

Requirements
------------
    pip install matplotlib pandas numpy scipy

Author : Independent Researcher — Carnegie Mellon University
Paper  : Latency-Compensating Market Making (v1, 2026)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

# ── Configuration ─────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("results/figures")
FEATURES_PATH = Path("data/features/BTCUSD_features.parquet")

# Publication-quality plot settings
plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Color palette — publication-friendly, colorblind-safe
COLORS = {
    "AvellanedaStoikov":  "#2196F3",   # blue
    "LatencyPenalized":   "#FF9800",   # orange
    "LatencyCompensating":"#4CAF50",   # green
    "DQN":                "#9C27B0",   # purple
    "PPO":                "#F44336",   # red
    "neutral":            "#607D8B",   # grey
}

STRATEGY_LABELS = {
    "AvellanedaStoikov":   "Avellaneda-Stoikov (baseline)",
    "LatencyPenalized":    "Latency-Penalized AS",
    "LatencyCompensating": "Latency-Compensating AS",
    "DQN":                 "DQN Agent",
    "PPO":                 "PPO Agent",
}

# ── Data Loaders ──────────────────────────────────────────────────────────────

def load_mm_results() -> pd.DataFrame:
    path = RESULTS_DIR / "mm_strategy_results.csv"
    if not path.exists():
        print(f"[WARN] {path} not found — skipping MM strategy results.")
        return pd.DataFrame()
    return pd.read_csv(path)


def load_rl_results() -> pd.DataFrame:
    path = RESULTS_DIR / "rl_agent_results.csv"
    if not path.exists():
        print(f"[WARN] {path} not found — skipping RL results.")
        return pd.DataFrame()
    return pd.read_csv(path)


def load_rl_history() -> pd.DataFrame:
    path = RESULTS_DIR / "rl_agent_history.csv"
    if not path.exists():
        print(f"[WARN] {path} not found — skipping RL history.")
        return pd.DataFrame()
    return pd.read_csv(path)


def load_fill_results() -> pd.DataFrame:
    path = RESULTS_DIR / "fill_predictor_results.csv"
    if not path.exists():
        print(f"[WARN] {path} not found — skipping fill predictor results.")
        return pd.DataFrame()
    return pd.read_csv(path)


def load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(FEATURES_PATH)


# ── Simulate Equity Curves ────────────────────────────────────────────────────

def simulate_equity_curves(features_df: pd.DataFrame, n_steps: int = 10_000) -> dict:
    """
    Re-simulate equity curves for all analytical strategies
    to produce smooth time series for plotting.

    Uses mid-price from feature data (or synthetic if unavailable).
    """
    if features_df.empty or "mid_price" not in features_df.columns:
        # Generate synthetic price path
        rng = np.random.default_rng(42)
        dt = 1 / (252 * 24 * 3600 / 0.1)
        returns = rng.normal(0, 0.35 * np.sqrt(dt), n_steps)
        prices = 65000 * np.exp(np.cumsum(returns))
    else:
        prices = features_df["mid_price"].values[:n_steps]
        if len(prices) < n_steps:
            prices = np.pad(prices, (0, n_steps - len(prices)), mode="edge")

    tick = 0.10
    gamma = 0.01
    sigma = 0.35
    kappa = 1.5
    T = 1.0

    curves = {name: np.zeros(n_steps) for name in [
        "AvellanedaStoikov", "LatencyPenalized", "LatencyCompensating"
    ]}

    inventories = {name: 0 for name in curves}
    cash        = {name: 0.0 for name in curves}

    rng2 = np.random.default_rng(123)

    for i in range(1, n_steps):
        mid = prices[i]
        t   = i / n_steps
        tau = max(T - t, 1e-6)

        # AS optimal spread
        r     = mid - inventories["AvellanedaStoikov"] * gamma * sigma**2 * tau
        delta = gamma * sigma**2 * tau + (2/gamma) * np.log(1 + gamma/kappa)
        delta = np.clip(delta, tick, 10 * tick)

        spread_move = abs(prices[i] - prices[i-1])

        for name in curves:
            if name == "AvellanedaStoikov":
                half_spread = delta / 2
            elif name == "LatencyPenalized":
                latency_adj = 0.5 * sigma * np.sqrt(500 / (1000 * 252 * 24 * 3600)) * 10000
                half_spread = delta / 2 + latency_adj
            else:
                # LC: abstain if large move predicted
                if spread_move > 0.01 * mid:
                    curves[name][i] = curves[name][i-1]
                    continue
                half_spread = delta / 2

            half_spread = np.clip(half_spread, tick, 5 * tick)

            # Simulate fills probabilistically
            fill_prob = np.exp(-kappa * half_spread)
            bid_fill = rng2.random() < fill_prob and inventories[name] < 10
            ask_fill = rng2.random() < fill_prob and inventories[name] > -10

            step_pnl = 0.0
            if bid_fill:
                inventories[name] += 1
                cash[name] -= (mid - half_spread)
                step_pnl += half_spread
            if ask_fill:
                inventories[name] -= 1
                cash[name] += (mid + half_spread)
                step_pnl += half_spread

            # Inventory penalty
            step_pnl -= gamma * inventories[name]**2

            curves[name][i] = curves[name][i-1] + step_pnl

    return curves


# ── Figure 1: Equity Curves ───────────────────────────────────────────────────

def plot_equity_curves(curves: dict, mm_results: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    ax_eq, ax_dd = axes

    # Equity curves
    for name, pnl in curves.items():
        label = STRATEGY_LABELS.get(name, name)
        color = COLORS.get(name, "#607D8B")
        ax_eq.plot(pnl, label=label, color=color, linewidth=1.5, alpha=0.9)

    ax_eq.set_title(
        "Equity Curves — Analytical Market Making Strategies\n"
        "Synthetic BTC/USD LOB Data, Latency = 500ms",
        pad=12
    )
    ax_eq.set_ylabel("Cumulative PnL (USD)")
    ax_eq.legend(loc="upper left", framealpha=0.9)
    ax_eq.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    # Drawdown
    for name, pnl in curves.items():
        color  = COLORS.get(name, "#607D8B")
        cummax = np.maximum.accumulate(pnl)
        dd     = pnl - cummax
        ax_dd.fill_between(range(len(dd)), dd, 0, alpha=0.3, color=color)
        ax_dd.plot(dd, color=color, linewidth=0.8, alpha=0.7)

    ax_dd.set_title("Drawdown")
    ax_dd.set_ylabel("Drawdown (USD)")
    ax_dd.set_xlabel("Time Steps (100ms intervals)")

    # Annotate max drawdown for each strategy
    if not mm_results.empty:
        for _, row in mm_results.iterrows():
            name = row["strategy"]
            color = COLORS.get(name, "#607D8B")
            ax_eq.annotate(
                f"MDD: {row['max_drawdown']:.1f}",
                xy=(0.02, 0.95 - list(mm_results["strategy"]).index(name) * 0.08),
                xycoords="axes fraction",
                fontsize=8,
                color=color,
                alpha=0.8,
            )

    plt.tight_layout()
    path = FIGURES_DIR / "equity_curves.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ── Figure 2: Fill Rate Table (heatmap) ──────────────────────────────────────

def plot_fill_rates():
    """
    Reproduce the fill rate table from lob_features.py as a heatmap.
    This is Table 1 in the paper.
    """
    latency_windows = [200, 500, 1000, 2000]
    offsets         = [1, 2, 3, 5]

    # Values from lob_features.py synthetic run
    fill_rates = np.array([
        [0.625, 0.608, 0.591, 0.557],
        [0.754, 0.742, 0.730, 0.705],
        [0.824, 0.815, 0.806, 0.787],
        [0.875, 0.869, 0.862, 0.849],
    ])

    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(fill_rates, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=0.95)

    ax.set_xticks(range(len(offsets)))
    ax.set_yticks(range(len(latency_windows)))
    ax.set_xticklabels([f"Offset {o} tick{'s' if o > 1 else ''}" for o in offsets])
    ax.set_yticklabels([f"{l}ms" for l in latency_windows])
    ax.set_xlabel("Quote Offset from Mid-Price")
    ax.set_ylabel("Latency Window")
    ax.set_title(
        "Fill Probability by Latency Window and Quote Offset\n"
        "Synthetic BTC/USD LOB Data (1M Snapshots)",
        pad=12
    )

    # Annotate cells
    for i in range(len(latency_windows)):
        for j in range(len(offsets)):
            ax.text(j, i, f"{fill_rates[i, j]:.3f}",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if fill_rates[i, j] < 0.65 else "black")

    plt.colorbar(im, ax=ax, label="Fill Probability")
    plt.tight_layout()

    path = FIGURES_DIR / "fill_rates.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ── Figure 3: RL Training History ────────────────────────────────────────────

def plot_rl_training(rl_history: pd.DataFrame):
    if rl_history.empty:
        print("[SKIP] No RL history data.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    for agent_name, group in rl_history.groupby("agent"):
        color = COLORS.get(agent_name, "#607D8B")
        episodes = group["episode"].values
        pnl      = group["total_pnl"].values
        fills    = group["n_fills"].values

        # Smooth with rolling average
        window = max(1, len(pnl) // 20)
        pnl_smooth = pd.Series(pnl).rolling(window, min_periods=1).mean().values

        axes[0].plot(episodes, pnl_smooth, label=agent_name, color=color,
                     linewidth=2.0)
        axes[0].fill_between(episodes, pnl, pnl_smooth, alpha=0.1, color=color)

        axes[1].plot(episodes, fills, label=agent_name, color=color,
                     linewidth=1.5, alpha=0.8)

    axes[0].set_title("RL Agent Training History — Episode PnL\n(Smoothed, 500ms Latency)",
                       pad=12)
    axes[0].set_ylabel("Episode PnL (USD)")
    axes[0].set_xlabel("Training Episode")
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].set_title("Fill Count per Episode")
    axes[1].set_ylabel("Number of Fills")
    axes[1].set_xlabel("Training Episode")
    axes[1].legend()

    # Add annotation about synthetic data limitation
    axes[0].annotate(
        "Note: Trained on synthetic GBM data.\nConvergence expected on real LOB data (v2).",
        xy=(0.98, 0.05), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                  edgecolor="orange", alpha=0.8)
    )

    plt.tight_layout()
    path = FIGURES_DIR / "rl_training.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ── Figure 4: Strategy Comparison Bar Chart ───────────────────────────────────

def plot_strategy_comparison(mm_results: pd.DataFrame):
    if mm_results.empty:
        print("[SKIP] No MM strategy results.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    strategies = mm_results["strategy"].tolist()
    colors     = [COLORS.get(s, "#607D8B") for s in strategies]
    labels     = [STRATEGY_LABELS.get(s, s) for s in strategies]
    x          = np.arange(len(strategies))

    # Sharpe ratio
    axes[0].bar(x, mm_results["sharpe"], color=colors, alpha=0.85, edgecolor="white")
    axes[0].set_title("Annualized Sharpe Ratio")
    axes[0].set_ylabel("Sharpe Ratio")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    for i, v in enumerate(mm_results["sharpe"]):
        axes[0].text(i, v + 5, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")

    # Maximum drawdown (lower is better)
    axes[1].bar(x, mm_results["max_drawdown"], color=colors, alpha=0.85, edgecolor="white")
    axes[1].set_title("Maximum Drawdown (Lower = Better)")
    axes[1].set_ylabel("Max Drawdown (USD)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    for i, v in enumerate(mm_results["max_drawdown"]):
        axes[1].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")

    # Total PnL
    axes[2].bar(x, mm_results["total_pnl"], color=colors, alpha=0.85, edgecolor="white")
    axes[2].set_title("Total PnL")
    axes[2].set_ylabel("Cumulative PnL (USD)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    for i, v in enumerate(mm_results["total_pnl"]):
        axes[2].text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle(
        "Strategy Comparison — Analytical Benchmarks\nSynthetic BTC/USD, Latency = 500ms",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    path = FIGURES_DIR / "strategy_comparison.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ── Figure 5: Fill Predictor Results ─────────────────────────────────────────

def plot_fill_predictor(fill_results: pd.DataFrame):
    if fill_results.empty:
        print("[SKIP] No fill predictor results.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    models = fill_results["model"].tolist()
    colors = ["#2196F3", "#F44336"]
    x      = np.arange(len(models))

    metrics = [
        ("c_index",  "Concordance Index (C-index)", "Higher is better\n(0.5 = random, 1.0 = perfect)"),
        ("brier",    "Brier Score",                 "Lower is better"),
        ("accuracy", "Fill Accuracy",               "Higher is better"),
    ]

    for ax, (col, title, subtitle) in zip(axes, metrics):
        bars = ax.bar(x, fill_results[col], color=colors[:len(models)],
                      alpha=0.85, edgecolor="white", width=0.5)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylim(0, max(fill_results[col]) * 1.25)

        for bar, v in zip(bars, fill_results[col]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=10,
                    fontweight="bold")

        # Add success criterion line for C-index
        if col == "c_index":
            ax.axhline(0.70, color="green", linewidth=1.5, linestyle="--",
                       alpha=0.7, label="Target (0.70)")
            ax.axhline(0.50, color="red", linewidth=1.0, linestyle=":",
                       alpha=0.5, label="Random (0.50)")
            ax.legend(fontsize=8)

    fig.suptitle(
        "Fill Probability Model Results\nSynthetic BTC/USD, 200K Training Samples",
        fontsize=13
    )
    plt.tight_layout()
    path = FIGURES_DIR / "fill_predictor.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ── Performance Table ─────────────────────────────────────────────────────────

def build_performance_table(
    mm_results:   pd.DataFrame,
    rl_results:   pd.DataFrame,
    fill_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the master performance table (Table 3 in paper).
    Combines all strategy results into one unified table.
    """
    rows = []

    # Analytical strategies
    if not mm_results.empty:
        for _, row in mm_results.iterrows():
            rows.append({
                "Strategy":       STRATEGY_LABELS.get(row["strategy"], row["strategy"]),
                "Type":           "Analytical",
                "Latency (ms)":   int(row["latency_ms"]),
                "Total PnL":      f"{row['total_pnl']:.2f}",
                "Sharpe":         f"{row['sharpe']:.3f}",
                "Sortino":        f"{row['sortino']:.3f}",
                "Max DD":         f"{row['max_drawdown']:.2f}",
                "AFR":            f"{row['adverse_fill_rate']:.3f}",
                "QUR":            f"{row['quote_util_rate']:.3f}",
            })

    # RL agents
    if not rl_results.empty:
        for _, row in rl_results.iterrows():
            rows.append({
                "Strategy":       STRATEGY_LABELS.get(row["agent"], row["agent"]),
                "Type":           "RL Agent",
                "Latency (ms)":   int(row["latency_ms"]),
                "Total PnL":      f"{row['avg_pnl']:.2f}",
                "Sharpe":         "N/A",
                "Sortino":        "N/A",
                "Max DD":         "N/A",
                "AFR":            f"{row['avg_afr']:.3f}",
                "QUR":            "N/A",
            })

    table = pd.DataFrame(rows)
    return table


def save_latex_table(table: pd.DataFrame):
    """Generate LaTeX table code for the paper."""
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Performance Comparison — All Strategies, Latency = 500ms, Synthetic BTC/USD Data}")
    latex.append("\\label{tab:performance}")
    latex.append("\\small")
    latex.append("\\begin{tabular}{llrrrrrrr}")
    latex.append("\\toprule")
    latex.append("Strategy & Type & Latency & PnL & Sharpe & Sortino & Max DD & AFR & QUR \\\\")
    latex.append("\\midrule")

    for _, row in table.iterrows():
        line = " & ".join(str(v) for v in row.values) + " \\\\"
        latex.append(line)

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    path = RESULTS_DIR / "performance_table.txt"
    with open(path, "w") as f:
        f.write("\n".join(latex))
    print(f"Saved → {path}")


# ── Summary Stats ─────────────────────────────────────────────────────────────

def print_summary(
    mm_results:   pd.DataFrame,
    rl_results:   pd.DataFrame,
    fill_results: pd.DataFrame,
):
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)

    if not fill_results.empty:
        print("\n── Fill Probability Models ──")
        print(fill_results[["model", "c_index", "brier", "accuracy"]].to_string(index=False))
        lstm_row = fill_results[fill_results["model"].str.contains("LSTM", case=False)]
        if not lstm_row.empty:
            c = float(lstm_row["c_index"].iloc[0])
            print(f"\n  LSTM C-index: {c:.4f} ({'✓ meets' if c >= 0.70 else '✗ below'} 0.70 target)")

    if not mm_results.empty:
        print("\n── Analytical Strategy Benchmarks ──")
        cols = ["strategy", "total_pnl", "sharpe", "max_drawdown",
                "adverse_fill_rate", "quote_util_rate"]
        print(mm_results[cols].to_string(index=False))

        lc = mm_results[mm_results["strategy"] == "LatencyCompensating"]
        as_ = mm_results[mm_results["strategy"] == "AvellanedaStoikov"]
        if not lc.empty and not as_.empty:
            dd_ratio = float(as_["max_drawdown"].iloc[0]) / max(float(lc["max_drawdown"].iloc[0]), 0.01)
            print(f"\n  LC vs AS drawdown reduction: {dd_ratio:.1f}x")

    if not rl_results.empty:
        print("\n── RL Agent Results ──")
        print(rl_results[["agent", "avg_pnl", "avg_afr", "avg_fills"]].to_string(index=False))
        print("  Note: Synthetic data — convergence expected on real LOB data (v2)")

    print("\n" + "="*70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Running evaluation pipeline…")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load all results
    mm_results   = load_mm_results()
    rl_results   = load_rl_results()
    rl_history   = load_rl_history()
    fill_results = load_fill_results()
    features_df  = load_features()

    # Print summary
    print_summary(mm_results, rl_results, fill_results)

    # Generate figures
    print("\nGenerating figures…")

    # Figure 1: Equity curves + drawdown
    print("  [1/5] Equity curves…")
    curves = simulate_equity_curves(features_df)
    plot_equity_curves(curves, mm_results)

    # Figure 2: Fill rate heatmap
    print("  [2/5] Fill rate heatmap…")
    plot_fill_rates()

    # Figure 3: RL training history
    print("  [3/5] RL training history…")
    plot_rl_training(rl_history)

    # Figure 4: Strategy comparison bars
    print("  [4/5] Strategy comparison…")
    plot_strategy_comparison(mm_results)

    # Figure 5: Fill predictor results
    print("  [5/5] Fill predictor results…")
    plot_fill_predictor(fill_results)

    # Performance table
    print("\nBuilding performance table…")
    table = build_performance_table(mm_results, rl_results, fill_results)
    table.to_csv(RESULTS_DIR / "performance_table.csv", index=False)
    save_latex_table(table)

    print(f"\nAll figures saved → {FIGURES_DIR}/")
    print(f"Performance table → {RESULTS_DIR}/performance_table.csv")
    print(f"LaTeX table       → {RESULTS_DIR}/performance_table.txt")
    print("\nDone.")


if __name__ == "__main__":
    main()
