"""
mm_strategy.py
────────────────────────────────────────────────────────────────────
Analytical market making strategies for latency-compensating MM research.

Implements three strategies as described in Sections 3 and 5.8:

  1. AvellanedaStoikov   — classical optimal market making (zero latency)
  2. LatencyPenalized    — AS extended with latency cost term (Section 3.2)
  3. LatencyCompensating — AS + fill probability integration (Section 3.3)

These serve as analytical benchmarks against which the RL agents
(rl_agent.py) are evaluated in Experiment 3 (Section 6.3).

Usage
-----
    # Run all three strategies on synthetic data
    python src/mm_strategy.py --latency 500

    # Run with specific latency
    python src/mm_strategy.py --latency 1000

Output
------
    results/mm_strategy_results.csv — performance metrics per strategy

Requirements
------------
    pip install pandas numpy pyarrow loguru scipy

Author : Independent Researcher — Carnegie Mellon University
Paper  : Latency-Compensating Market Making (v1, 2026)
"""

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import norm

# ── Configuration ─────────────────────────────────────────────────────────────

RESULTS_DIR     = Path("results")
FEATURES_PATH   = Path("data/features/BTCUSD_features.parquet")

# Avellaneda-Stoikov parameters
GAMMA           = 0.01      # Risk aversion parameter
SIGMA           = 0.35      # Asset volatility (annualized)
KAPPA           = 1.5       # Order arrival intensity decay
A_PARAM         = 1.0       # Order arrival base rate
T_HORIZON       = 1.0       # Trading horizon (normalized)
Q_MAX           = 10        # Maximum inventory units
DELTA_MAX       = 10        # Maximum spread in ticks
TICK_SIZE       = 0.10      # BTC tick size (USD)

# Latency penalty parameters (Section 3.2)
ALPHA_LATENCY   = 0.5       # Latency cost scaling factor

# Fill probability threshold (Section 3.3)
TAU_FILL        = 0.25      # Minimum acceptable fill probability
DELTA_SAFE      = 0.10     # Maximum tolerable mid-price move during latency

# Simulation parameters
DT              = 0.1       # Time step in seconds (100ms)
N_STEPS         = 50_000    # Simulation steps

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MarketState:
    """Current state of the market and the market maker's position."""
    mid_price:   float = 100.0
    spread:      float = 0.10
    obi:         float = 0.0
    inventory:   int   = 0
    cash:        float = 0.0
    t:           float = 0.0
    step:        int   = 0


@dataclass
class Quote:
    """A bid/ask quote pair."""
    bid_price:   float
    ask_price:   float
    bid_offset:  float
    ask_offset:  float
    abstain:     bool = False


@dataclass
class Fill:
    """A recorded fill event."""
    step:        int
    side:        str        # 'bid' or 'ask'
    price:       float
    mid_at_decision: float
    adverse:     bool       # True if fill was adverse (latency-induced)
    pnl:         float


@dataclass
class StrategyResult:
    """Full results from a strategy simulation."""
    name:            str
    latency_ms:      float
    pnl_series:      list[float] = field(default_factory=list)
    fills:           list[Fill]  = field(default_factory=list)
    abstentions:     int = 0
    total_steps:     int = 0


# ── Avellaneda-Stoikov Core ───────────────────────────────────────────────────

class AvellanedaStoikov:
    """
    Classical Avellaneda-Stoikov market making model (2008).

    Computes optimal bid and ask quotes as a function of:
    - Current inventory position q
    - Time remaining T - t
    - Asset volatility sigma
    - Risk aversion gamma
    - Order arrival parameters kappa, A

    Reference: Avellaneda & Stoikov (2008), Quantitative Finance.
    """

    def __init__(
        self,
        gamma:  float = GAMMA,
        sigma:  float = SIGMA,
        kappa:  float = KAPPA,
        A:      float = A_PARAM,
        T:      float = T_HORIZON,
        q_max:  int   = Q_MAX,
    ):
        self.gamma  = gamma
        self.sigma  = sigma
        self.kappa  = kappa
        self.A      = A
        self.T      = T
        self.q_max  = q_max

    def reservation_price(self, mid: float, q: int, t: float) -> float:
        """
        Compute the reservation price — the mid-price adjusted
        for inventory risk.

        r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

        A long inventory (q > 0) lowers the reservation price,
        making the market maker more willing to sell. A short
        inventory (q < 0) raises it, making them more willing to buy.
        """
        tau = max(self.T - t, 1e-6)
        return mid - q * self.gamma * (self.sigma ** 2) * tau

    def optimal_spread(self, t: float) -> float:
        """
        Compute the optimal total bid-ask spread.

        delta*(t) = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/kappa)

        Spread widens as risk aversion increases, as volatility
        increases, and as time horizon increases.
        """
        tau = max(self.T - t, 1e-6)
        term1 = self.gamma * (self.sigma ** 2) * tau
        term2 = (2.0 / self.gamma) * np.log(1.0 + self.gamma / self.kappa)
        return term1 + term2

    def compute_quotes(self, mid: float, q: int, t: float) -> Quote:
        """
        Compute optimal bid and ask quotes.

        Parameters
        ----------
        mid : float
            Current mid-price
        q : int
            Current inventory position
        t : float
            Current time (normalized to [0, T])

        Returns
        -------
        Quote
            Optimal bid and ask prices and offsets
        """
        r     = self.reservation_price(mid, q, t)
        delta = self.optimal_spread(t)

        bid_offset = delta / 2.0 - (r - mid)
        ask_offset = delta / 2.0 + (r - mid)

        # Clip to reasonable range
        bid_offset = np.clip(bid_offset, TICK_SIZE, DELTA_MAX * TICK_SIZE)
        ask_offset = np.clip(ask_offset, TICK_SIZE, DELTA_MAX * TICK_SIZE)

        return Quote(
            bid_price  = mid - bid_offset,
            ask_price  = mid + ask_offset,
            bid_offset = bid_offset,
            ask_offset = ask_offset,
            abstain    = False,
        )


# ── Strategy 1: Pure Avellaneda-Stoikov ──────────────────────────────────────

class PureAvellanedaStoikov(AvellanedaStoikov):
    """
    Strategy 1: Classical AS with zero latency assumption.

    This is the benchmark that assumes instant execution — the
    standard model that fails under retail latency conditions.
    Used as the floor benchmark in Experiment 3 (Section 6.3).
    """

    name = "AvellanedaStoikov"

    def get_quote(
        self,
        state: MarketState,
        latency_ms: float = 0.0,
        fill_prob_bid: float = 1.0,
        fill_prob_ask: float = 1.0,
        predicted_mid: Optional[float] = None,
    ) -> Quote:
        return self.compute_quotes(state.mid_price, state.inventory, state.t)


# ── Strategy 2: Latency-Penalized AS ─────────────────────────────────────────

class LatencyPenalizedAS(AvellanedaStoikov):
    """
    Strategy 2: AS extended with latency cost term (Section 3.2).

    Widens the spread proportionally to latency to compensate for
    the increased adverse selection risk during the execution window.
    Does NOT use predictive models — purely reactive compensation.

    The latency penalty term is:
        delta_latency = alpha * sigma * sqrt(ell / 1000)

    where ell is the latency in milliseconds. This captures the
    expected mid-price movement during the latency window under
    a Brownian motion model for the mid-price.
    """

    name = "LatencyPenalized"

    def __init__(self, alpha_latency: float = ALPHA_LATENCY, **kwargs):
        super().__init__(**kwargs)
        self.alpha_latency = alpha_latency

    def latency_spread_adjustment(self, latency_ms: float) -> float:
        """
        Compute additional spread needed to compensate for latency risk.

        Under Brownian motion, expected mid-price move in ell ms is:
            E[|Delta S|] = sigma * sqrt(ell / annualization_factor)

        We use this as the minimum additional spread required to break
        even on adverse fills induced by the latency window.
        """
        ell_years = latency_ms / (1000 * 252 * 24 * 3600)
        return self.alpha_latency * self.sigma * np.sqrt(ell_years) * 10_000

    def get_quote(
        self,
        state: MarketState,
        latency_ms: float = 500.0,
        fill_prob_bid: float = 1.0,
        fill_prob_ask: float = 1.0,
        predicted_mid: Optional[float] = None,
    ) -> Quote:
        base_quote = self.compute_quotes(state.mid_price, state.inventory, state.t)
        adjustment = self.latency_spread_adjustment(latency_ms)

        # Widen spread symmetrically by latency adjustment
        return Quote(
            bid_price  = base_quote.bid_price  - adjustment,
            ask_price  = base_quote.ask_price  + adjustment,
            bid_offset = base_quote.bid_offset + adjustment,
            ask_offset = base_quote.ask_offset + adjustment,
            abstain    = False,
        )


# ── Strategy 3: Latency-Compensating AS ──────────────────────────────────────

class LatencyCompensatingAS(AvellanedaStoikov):
    """
    Strategy 3: AS + fill probability integration (Section 3.3).

    This is the analytical implementation of the latency-compensating
    framework. It uses predicted fill probabilities and predicted
    mid-price direction to:

    1. Adjust quote prices based on predicted future mid-price
    2. Widen spreads when fill probability is low
    3. ABSTAIN entirely when fill probability or price conditions
       do not meet the viability thresholds

    This is the key strategy that demonstrates the latency-compensating
    principle analytically, before the full RL implementation.
    """

    name = "LatencyCompensating"

    def __init__(
        self,
        tau_fill:   float = TAU_FILL,
        delta_safe: float = DELTA_SAFE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tau_fill   = tau_fill
        self.delta_safe = delta_safe

    def get_quote(
        self,
        state: MarketState,
        latency_ms: float = 500.0,
        fill_prob_bid: float = 0.7,
        fill_prob_ask: float = 0.7,
        predicted_mid: Optional[float] = None,
    ) -> Quote:
        """
        Compute latency-compensating quotes.

        Parameters
        ----------
        state : MarketState
            Current market and position state
        latency_ms : float
            Execution latency in milliseconds
        fill_prob_bid : float
            Predicted fill probability for bid quote from fill predictor
        fill_prob_ask : float
            Predicted fill probability for ask quote from fill predictor
        predicted_mid : float, optional
            Predicted mid-price at execution time (t + ell)

        Returns
        -------
        Quote
            Latency-compensating quote, or abstain=True if conditions
            do not meet viability thresholds (Section 3.3)
        """
        mid = state.mid_price

        # ── Abstention check (Section 3.3) ────────────────────────────────────
        # Check predicted price movement constraint
        if predicted_mid is not None:
            predicted_move = abs(predicted_mid - mid) / mid
            if predicted_move > self.delta_safe:
                return Quote(
                    bid_price=mid, ask_price=mid,
                    bid_offset=0, ask_offset=0,
                    abstain=True,
                )

        # Check fill probability constraint
        if fill_prob_bid < self.tau_fill and fill_prob_ask < self.tau_fill:
            return Quote(
                bid_price=mid, ask_price=mid,
                bid_offset=0, ask_offset=0,
                abstain=True,
            )

        # ── Quote computation on predicted mid-price ───────────────────────────
        # Use predicted mid-price if available, otherwise current mid
        effective_mid = predicted_mid if predicted_mid is not None else mid
        base_quote    = self.compute_quotes(effective_mid, state.inventory, state.t)

        # ── Spread adjustment based on fill probability ────────────────────────
        # Lower fill probability → wider spread to compensate
        # At tau_fill=0.4: spread multiplier = 1 / 0.4 = 2.5x
        # At fill_prob=1.0: spread multiplier = 1.0x (no adjustment)
        bid_fp_mult = 1.0 / max(fill_prob_bid, 0.1)
        ask_fp_mult = 1.0 / max(fill_prob_ask, 0.1)

        bid_offset = base_quote.bid_offset * bid_fp_mult
        ask_offset = base_quote.ask_offset * ask_fp_mult

        bid_offset = np.clip(bid_offset, TICK_SIZE, DELTA_MAX * TICK_SIZE)
        ask_offset = np.clip(ask_offset, TICK_SIZE, DELTA_MAX * TICK_SIZE)

        # Only quote sides with sufficient fill probability
        abstain_bid = fill_prob_bid < self.tau_fill
        abstain_ask = fill_prob_ask < self.tau_fill

        if abstain_bid and abstain_ask:
            return Quote(
                bid_price=mid, ask_price=mid,
                bid_offset=0, ask_offset=0,
                abstain=True,
            )

        return Quote(
            bid_price  = effective_mid - bid_offset if not abstain_bid else mid,
            ask_price  = effective_mid + ask_offset if not abstain_ask else mid,
            bid_offset = bid_offset,
            ask_offset = ask_offset,
            abstain    = False,
        )


# ── Simulator ─────────────────────────────────────────────────────────────────

class MarketSimulator:
    """
    Realistic market making simulator with latency modeling.

    Replays LOB feature data (or generates synthetic price paths)
    and evaluates analytical market making strategies under
    configurable execution latency.

    Implements the simulation environment described in Section 5.5,
    adapted for analytical (non-RL) strategies.
    """

    def __init__(
        self,
        latency_ms:    float,
        n_steps:       int   = N_STEPS,
        tick_size:     float = TICK_SIZE,
        q_max:         int   = Q_MAX,
        seed:          int   = 42,
    ):
        self.latency_ms  = latency_ms
        self.n_steps     = n_steps
        self.tick_size   = tick_size
        self.q_max       = q_max
        self.rng         = np.random.default_rng(seed)

    def _load_or_generate_prices(self) -> np.ndarray:
        """Load real features or generate synthetic mid-price path."""
        if FEATURES_PATH.exists():
            logger.info("Loading real LOB features for simulation…")
            df = pd.read_parquet(FEATURES_PATH)
            prices = df["mid_price"].values[:self.n_steps]
            if len(prices) < self.n_steps:
                logger.warning(
                    f"Only {len(prices):,} real snapshots available, "
                    f"padding with synthetic."
                )
                extra = self._synthetic_prices(self.n_steps - len(prices), prices[-1])
                prices = np.concatenate([prices, extra])
            return prices
        else:
            logger.info("No real data found — using synthetic price path.")
            return self._synthetic_prices(self.n_steps, start_price=65_000.0)

    def _synthetic_prices(self, n: int, start_price: float = 65_000.0) -> np.ndarray:
        """Generate GBM price path."""
        dt_year = DT / (252 * 24 * 3600)
        returns = self.rng.normal(0, SIGMA * np.sqrt(dt_year), n)
        return start_price * np.exp(np.cumsum(returns))

    def _simulate_fill(
        self,
        quote_price: float,
        side: str,
        mid_at_execution: float,
        spread_at_execution: float,
    ) -> bool:
        """
        Determine if a limit order fills at execution time.

        A bid fills if the ask at execution crosses below the bid price.
        An ask fills if the bid at execution crosses above the ask price.
        """
        half_spread = spread_at_execution / 2.0
        if side == "bid":
            return mid_at_execution - half_spread <= quote_price
        else:
            return mid_at_execution + half_spread >= quote_price

    def _is_adverse_fill(
        self,
        side: str,
        quote_price: float,
        mid_at_decision: float,
    ) -> bool:
        """
        Classify a fill as adverse if the market moved against
        the market maker during the latency window.

        Adverse bid fill: market maker bought above the mid-price
        at decision time (price fell during latency window).

        Adverse ask fill: market maker sold below the mid-price
        at decision time (price rose during latency window).
        """
        if side == "bid":
            return quote_price > mid_at_decision
        else:
            return quote_price < mid_at_decision

    def _get_fill_probabilities(
        self,
        mid: float,
        spread: float,
        obi: float,
    ) -> tuple[float, float]:
        """
        Simple analytical fill probability estimate based on
        order book imbalance and spread.

        In the full system these come from the trained fill predictor
        (fill_predictor.py). Here we use a tractable approximation
        for the analytical strategy benchmarks.

        Fill probability increases with:
        - Higher OBI (for bids) — more buying pressure
        - Tighter spread — more liquid market
        - Longer latency window — more time to fill
        """
        latency_factor = np.clip(self.latency_ms / 2000.0, 0.1, 1.0)
        spread_factor  = np.clip(1.0 - spread / (mid * 0.001), 0.1, 1.0)

        base_fp = 0.6 + 0.3 * latency_factor * spread_factor

        fp_bid = np.clip(base_fp * (1.0 + 0.3 * obi),  0.05, 0.95)
        fp_ask = np.clip(base_fp * (1.0 - 0.3 * obi),  0.05, 0.95)

        return float(fp_bid), float(fp_ask)

    def run(self, strategy) -> StrategyResult:
        """
        Run a full simulation of a market making strategy.

        Parameters
        ----------
        strategy : PureAvellanedaStoikov | LatencyPenalizedAS | LatencyCompensatingAS
            The strategy to evaluate

        Returns
        -------
        StrategyResult
            Complete simulation results including PnL series and fills
        """
        prices = self._load_or_generate_prices()
        n      = min(self.n_steps, len(prices))

        # Latency in steps
        latency_steps = max(1, int(self.latency_ms / (DT * 1000)))

        result  = StrategyResult(
            name       = strategy.name,
            latency_ms = self.latency_ms,
        )

        state = MarketState()
        state.mid_price = prices[0]

        pending_orders = []   # (step_to_execute, side, quote_price, mid_at_decision)
        cumulative_pnl = 0.0

        for step in range(n - latency_steps - 1):
            state.mid_price = prices[step]
            state.spread    = max(TICK_SIZE * 2, state.mid_price * 0.0002)
            state.t         = step / n
            state.step      = step

            # ── Execute pending orders ────────────────────────────────────────
            still_pending = []
            for (exec_step, side, quote_price, mid_at_decision) in pending_orders:
                if step >= exec_step:
                    # Order arrives at exchange — check if it fills
                    mid_exec    = prices[step]
                    spread_exec = max(TICK_SIZE * 2, mid_exec * 0.0002)

                    if self._simulate_fill(quote_price, side, mid_exec, spread_exec):
                        adverse = self._is_adverse_fill(side, quote_price, mid_at_decision)

                        # PnL from fill
                        if side == "bid":
                            fill_pnl = mid_exec - quote_price   # bought below current mid
                            state.inventory += 1
                            state.cash      -= quote_price
                        else:
                            fill_pnl = quote_price - mid_exec   # sold above current mid
                            state.inventory -= 1
                            state.cash      += quote_price

                        # Inventory penalty
                        inv_penalty = GAMMA * (state.inventory ** 2)
                        net_pnl     = fill_pnl - inv_penalty

                        # Adverse fill penalty (Section 5.4)
                        if adverse:
                            adverse_penalty = 2.0 * abs(mid_exec - mid_at_decision)
                            net_pnl        -= adverse_penalty

                        cumulative_pnl += net_pnl

                        result.fills.append(Fill(
                            step=step, side=side,
                            price=quote_price,
                            mid_at_decision=mid_at_decision,
                            adverse=adverse,
                            pnl=net_pnl,
                        ))
                    # else: order didn't fill, silently dropped
                else:
                    still_pending.append((exec_step, side, quote_price, mid_at_decision))

            pending_orders = still_pending

            # ── Inventory limit check ─────────────────────────────────────────
            if abs(state.inventory) >= Q_MAX:
                result.pnl_series.append(cumulative_pnl)
                result.total_steps += 1
                continue

            # ── Compute fill probabilities ────────────────────────────────────
            obi = getattr(state, "obi", 0.0)
            fp_bid, fp_ask = self._get_fill_probabilities(
                state.mid_price, state.spread, obi
            )

            # Predicted mid-price (simple linear extrapolation)
            if step > 0:
                price_trend  = prices[step] - prices[step - 1]
                predicted_mid = prices[step] + price_trend * latency_steps
            else:
                predicted_mid = prices[step]

            # ── Get quote from strategy ───────────────────────────────────────
            quote = strategy.get_quote(
                state         = state,
                latency_ms    = self.latency_ms,
                fill_prob_bid = fp_bid,
                fill_prob_ask = fp_ask,
                predicted_mid = predicted_mid,
            )

            if quote.abstain:
                result.abstentions += 1
            else:
                # Submit orders — they arrive after latency_steps
                exec_step = step + latency_steps
                mid_at_decision = state.mid_price

                if quote.bid_price < state.mid_price:
                    pending_orders.append(
                        (exec_step, "bid", quote.bid_price, mid_at_decision)
                    )
                if quote.ask_price > state.mid_price:
                    pending_orders.append(
                        (exec_step, "ask", quote.ask_price, mid_at_decision)
                    )

            result.pnl_series.append(cumulative_pnl)
            result.total_steps += 1

        return result


# ── Performance Metrics ───────────────────────────────────────────────────────

def compute_metrics(result: StrategyResult) -> dict:
    """
    Compute performance metrics from a strategy simulation result.

    Metrics match those defined in Section 6.4:
    - Sharpe ratio (annualized)
    - Sortino ratio (annualized)
    - Maximum drawdown
    - Adverse fill rate
    - Quote utilization rate
    - Fill rate conditional on quoting
    """
    pnl  = np.array(result.pnl_series)
    rets = np.diff(pnl) if len(pnl) > 1 else np.array([0.0])

    # Annualization factor: steps per year at 100ms per step
    steps_per_year = 252 * 24 * 3600 / DT

    # Sharpe ratio
    mu    = np.mean(rets)
    sigma = np.std(rets) + 1e-9
    sharpe = (mu / sigma) * np.sqrt(steps_per_year)

    # Sortino ratio
    downside = rets[rets < 0]
    sigma_down = np.std(downside) + 1e-9 if len(downside) > 0 else 1e-9
    sortino = (mu / sigma_down) * np.sqrt(steps_per_year)

    # Maximum drawdown
    cummax = np.maximum.accumulate(pnl)
    drawdowns = cummax - pnl
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Fill metrics
    n_fills   = len(result.fills)
    n_adverse = sum(1 for f in result.fills if f.adverse)
    afr       = n_adverse / n_fills if n_fills > 0 else 0.0

    # Quote utilization
    n_quoted = result.total_steps - result.abstentions
    qur      = n_quoted / result.total_steps if result.total_steps > 0 else 0.0

    # Fill rate conditional on quoting
    frq = n_fills / (n_quoted * 2) if n_quoted > 0 else 0.0  # *2 for bid+ask

    # Total PnL
    total_pnl = float(pnl[-1]) if len(pnl) > 0 else 0.0

    return {
        "strategy":         result.name,
        "latency_ms":       result.latency_ms,
        "total_pnl":        round(total_pnl, 4),
        "sharpe":           round(float(sharpe), 4),
        "sortino":          round(float(sortino), 4),
        "max_drawdown":     round(max_dd, 4),
        "adverse_fill_rate": round(afr, 4),
        "quote_util_rate":  round(qur, 4),
        "fill_rate_quoted": round(frq, 4),
        "n_fills":          n_fills,
        "n_adverse":        n_adverse,
        "abstentions":      result.abstentions,
        "total_steps":      result.total_steps,
    }


# ── Unit Tests ────────────────────────────────────────────────────────────────

def run_tests():
    """Basic unit tests for strategy components."""
    logger.info("Running unit tests…")

    # Test 1: AS reservation price moves correctly with inventory
    as_model = AvellanedaStoikov()
    r_long  = as_model.reservation_price(100.0, q=5,  t=0.0)
    r_short = as_model.reservation_price(100.0, q=-5, t=0.0)
    assert r_long < 100.0,  "Long inventory should lower reservation price"
    assert r_short > 100.0, "Short inventory should raise reservation price"
    logger.success("Test 1 passed: reservation price moves correctly with inventory.")

    # Test 2: Optimal spread is positive and decreases as T-t → 0
    spread_early = as_model.optimal_spread(t=0.0)
    spread_late  = as_model.optimal_spread(t=0.99)
    assert spread_early > 0,            "Spread must be positive"
    assert spread_early > spread_late,  "Spread should decrease near horizon"
    logger.success("Test 2 passed: optimal spread is positive and time-varying.")

    # Test 3: Latency-compensating strategy abstains when fill prob is low
    lc = LatencyCompensatingAS(tau_fill=0.5)
    state = MarketState(mid_price=100.0, inventory=0, t=0.5)
    quote = lc.get_quote(state, latency_ms=500, fill_prob_bid=0.1, fill_prob_ask=0.1)
    assert quote.abstain, "Should abstain when fill prob below threshold"
    logger.success("Test 3 passed: abstention triggered correctly.")

    # Test 4: Latency-compensating strategy quotes when fill prob is high
    quote2 = lc.get_quote(state, latency_ms=500, fill_prob_bid=0.8, fill_prob_ask=0.8)
    assert not quote2.abstain, "Should quote when fill prob above threshold"
    assert quote2.bid_price < state.mid_price, "Bid must be below mid"
    assert quote2.ask_price > state.mid_price, "Ask must be above mid"
    logger.success("Test 4 passed: valid quote when fill prob sufficient.")

    # Test 5: Latency penalty increases with latency
    lp = LatencyPenalizedAS()
    adj_low  = lp.latency_spread_adjustment(200)
    adj_high = lp.latency_spread_adjustment(2000)
    assert adj_high > adj_low, "Latency adjustment should increase with latency"
    logger.success("Test 5 passed: latency spread adjustment scales correctly.")

    logger.success("All unit tests passed.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(latency_ms: float = 500.0, n_steps: int = N_STEPS):
    logger.info(f"Running MM strategy benchmarks at latency={latency_ms}ms.")

    run_tests()

    # Instantiate strategies
    strategies = [
        PureAvellanedaStoikov(),
        LatencyPenalizedAS(),
        LatencyCompensatingAS(),
    ]

    # Run simulations
    simulator = MarketSimulator(latency_ms=latency_ms, n_steps=n_steps)
    all_metrics = []

    for strategy in strategies:
        logger.info(f"Simulating {strategy.name}…")
        t0     = time.time()
        result = simulator.run(strategy)
        elapsed = time.time() - t0
        metrics = compute_metrics(result)
        all_metrics.append(metrics)
        logger.success(
            f"[{strategy.name}] PnL={metrics['total_pnl']:.2f} | "
            f"Sharpe={metrics['sharpe']:.3f} | "
            f"AFR={metrics['adverse_fill_rate']:.3f} | "
            f"QUR={metrics['quote_util_rate']:.3f} | "
            f"Time={elapsed:.1f}s"
        )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df   = pd.DataFrame(all_metrics)
    results_path = RESULTS_DIR / "mm_strategy_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "="*70)
    print(f"MM STRATEGY RESULTS — Latency: {latency_ms}ms")
    print("="*70)
    cols = ["strategy", "total_pnl", "sharpe", "sortino",
            "max_drawdown", "adverse_fill_rate", "quote_util_rate"]
    print(results_df[cols].to_string(index=False))
    print("="*70)
    print(f"\nResults saved → {results_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MM strategy benchmarks")
    parser.add_argument(
        "--latency", type=float, default=500.0,
        help="Execution latency in milliseconds (default: 500)"
    )
    parser.add_argument(
        "--steps", type=int, default=N_STEPS,
        help=f"Simulation steps (default: {N_STEPS})"
    )
    args = parser.parse_args()
    main(latency_ms=args.latency, n_steps=args.steps)
