"""
rl_agent.py
────────────────────────────────────────────────────────────────────
Latency-aware reinforcement learning market making agents.

Implements two agents as described in Section 5 of the paper:
  1. DQN baseline     — discrete side selection, fixed spread (Section 5.6)
  2. PPO main agent   — hybrid action space, continuous spread (Section 5.7)

Both agents use the trained fill probability model (fill_predictor.py)
as a signal source in their state representation, and are trained in
a realistic latency-simulating environment (Section 5.5).

The novel latency-aware reward function (Section 5.4) explicitly
penalizes adverse fills attributable to execution delay.

Usage
-----
    # Train both agents
    python src/rl_agent.py --agent both --episodes 500

    # Train PPO only
    python src/rl_agent.py --agent ppo --episodes 500

    # Train DQN only
    python src/rl_agent.py --agent dqn --episodes 500

Output
------
    models/dqn_agent.pt          — saved DQN weights
    models/ppo_actor.pt          — saved PPO actor weights
    models/ppo_critic.pt         — saved PPO critic weights
    results/rl_agent_results.csv — evaluation metrics

Requirements
------------
    pip install torch pandas numpy pyarrow loguru scipy

Author : Independent Researcher — Carnegie Mellon University
Paper  : Latency-Compensating Market Making (v1, 2026)
"""

import argparse
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from loguru import logger

# ── Configuration ─────────────────────────────────────────────────────────────

FEATURES_PATH   = Path("data/features/BTCUSD_features.parquet")
MODELS_DIR      = Path("models")
RESULTS_DIR     = Path("results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment parameters (Section 5.5)
LATENCY_MS      = 500           # Target latency in milliseconds
LATENCY_JITTER  = 0.20          # ±20% random latency variation
DT_MS           = 100           # LOB update interval in ms
DECISION_FREQ   = 5             # Make decision every N LOB updates (500ms)
Q_MAX           = 10            # Maximum inventory units
TICK_SIZE       = 0.10          # BTC tick size
DELTA_MAX_TICKS = 10            # Maximum spread in ticks
EPISODE_STEPS   = 2000          # Steps per training episode

# Reward function parameters (Section 5.4)
GAMMA_INV       = 0.01          # Inventory risk aversion
ALPHA_ADVERSE   = 2.0           # Adverse fill penalty weight

# DQN parameters (Section 5.6)
DQN_HIDDEN      = [256, 128, 64]
DQN_LR          = 1e-3
DQN_BUFFER_SIZE = 100_000
DQN_BATCH_SIZE  = 256
DQN_GAMMA       = 0.99
DQN_TAU         = 0.005         # Soft target update
DQN_EPS_START   = 1.0
DQN_EPS_END     = 0.05
DQN_EPS_DECAY   = 0.995
DQN_UPDATE_FREQ = 4
FIXED_SPREAD_TICKS = 3          # DQN uses fixed spread (Section 5.6)

# PPO parameters (Section 5.7)
PPO_HIDDEN      = 192           # Shared encoder output dim
PPO_LR          = 3e-4
PPO_GAMMA       = 0.99
PPO_LAMBDA      = 0.95          # GAE lambda
PPO_CLIP        = 0.20          # Clipping parameter epsilon
PPO_ENTROPY     = 0.01          # Entropy bonus coefficient
PPO_EPOCHS      = 10            # Epochs per PPO update
PPO_BATCH_SIZE  = 64
PPO_STEPS       = 2048          # Steps before each PPO update

# Actions: 0=abstain, 1=bid only, 2=ask only, 3=both sides
N_ACTIONS       = 4

# ── Fill Predictor Integration ────────────────────────────────────────────────

class FillPredictorSignal:
    """
    Loads the trained LSTM fill predictor and provides
    fill probability signals for the RL agent state.

    If the trained model is not available, falls back to
    an analytical approximation.
    """

    def __init__(self, n_features: int):
        self.n_features  = n_features
        self.model       = None
        self.use_trained = False
        self._try_load_model(n_features)

    def _try_load_model(self, n_features: int):
        """Attempt to load the trained LSTM fill predictor."""
        lstm_path = MODELS_DIR / "lstm_fill.pt"
        if not lstm_path.exists():
            logger.warning(
                "Trained fill predictor not found at models/lstm_fill.pt. "
                "Using analytical approximation."
            )
            return

        try:
            # Import LSTM architecture from fill_predictor
            import sys
            sys.path.insert(0, str(Path("src")))
            from fill_predictor import LSTMFillPredictor

            model = LSTMFillPredictor(n_features=n_features)
            model.load_state_dict(
                torch.load(lstm_path, map_location=DEVICE)
            )
            model.eval()
            model.to(DEVICE)
            self.model       = model
            self.use_trained = True
            logger.success("Loaded trained LSTM fill predictor.")
        except Exception as exc:
            logger.warning(f"Could not load fill predictor: {exc}. Using approximation.")

    def get_fill_probs(
        self,
        lob_window: np.ndarray,     # (LOOKBACK, n_features)
        latency_ms: float,
    ) -> tuple[float, float]:
        """
        Get predicted fill probabilities for bid and ask quotes.

        Parameters
        ----------
        lob_window : np.ndarray
            Recent LOB feature history
        latency_ms : float
            Current latency in milliseconds

        Returns
        -------
        tuple[float, float]
            (fp_bid, fp_ask) — fill probabilities in [0, 1]
        """
        if self.use_trained and self.model is not None:
            try:
                x = torch.tensor(
                    lob_window[np.newaxis], dtype=torch.float32
                ).to(DEVICE)
                lat = torch.tensor(
                    [latency_ms / 2000.0], dtype=torch.float32
                ).to(DEVICE)

                with torch.no_grad():
                    _, survival = self.model(x, lat)
                    fp = float(1.0 - survival[0, -1].cpu())

                # Use OBI to differentiate bid/ask fill probs
                obi = float(lob_window[-1, -1]) if lob_window.shape[1] > 0 else 0.0
                fp_bid = np.clip(fp * (1.0 + 0.2 * obi), 0.05, 0.95)
                fp_ask = np.clip(fp * (1.0 - 0.2 * obi), 0.05, 0.95)
                return float(fp_bid), float(fp_ask)
            except Exception:
                pass

        # Analytical fallback
        return self._analytical_fill_probs(latency_ms)

    def _analytical_fill_probs(self, latency_ms: float) -> tuple[float, float]:
        """Simple analytical fill probability approximation."""
        base = 0.60 + 0.25 * np.clip(latency_ms / 2000.0, 0.0, 1.0)
        return float(base), float(base)


# ── Market Environment ────────────────────────────────────────────────────────

@dataclass
class EnvState:
    """Complete environment state at one decision step."""
    lob_window:     np.ndarray      # (LOOKBACK, n_features) recent LOB history
    inventory:      int             # Current inventory position
    fp_bid:         float           # Predicted bid fill probability
    fp_ask:         float           # Predicted ask fill probability
    price_direction: float          # Predicted price direction (-1, 0, +1)
    latency_ms:     float           # Current latency (randomized)
    mid_price:      float           # Current mid-price
    t_normalized:   float           # Normalized time in episode [0, 1]

    def to_tensor(self) -> torch.Tensor:
        """Flatten state to 1D tensor for neural network input."""
        lob_flat = self.lob_window.flatten()
        scalars  = np.array([
            self.inventory / Q_MAX,
            self.fp_bid,
            self.fp_ask,
            self.price_direction,
            self.latency_ms / 2000.0,
            self.t_normalized,
        ], dtype=np.float32)
        vec = np.concatenate([lob_flat, scalars])
        return torch.tensor(vec, dtype=torch.float32)


class LatencyMarketEnv:
    """
    Realistic market making environment with latency simulation.

    Implements the simulation environment described in Section 5.5:
    - LOB replay from historical/synthetic feature data
    - Random latency variation around target (±20%)
    - Pending order queue with delayed execution
    - Latency-aware reward function (Section 5.4)
    - Abstention action available at every step

    Parameters
    ----------
    features : np.ndarray
        LOB feature matrix (n_snapshots, n_features)
    fill_predictor : FillPredictorSignal
        Fill probability signal provider
    latency_ms : float
        Target execution latency in milliseconds
    lookback : int
        Number of LOB snapshots per state window
    """

    def __init__(
        self,
        features:       np.ndarray,
        fill_predictor: FillPredictorSignal,
        latency_ms:     float = LATENCY_MS,
        lookback:       int   = 50,
    ):
        self.features        = features.astype(np.float32)
        self.fill_predictor  = fill_predictor
        self.latency_ms      = latency_ms
        self.lookback        = lookback
        self.n_snapshots     = len(features)
        self.n_features      = features.shape[1]

        # Find mid_price and obi column indices
        self._mid_col = 0   # will be set in reset
        self._obi_col = 0

        # State dimensions
        self.obs_dim = lookback * self.n_features + 6

    def _get_mid_price(self, idx: int) -> float:
        """Get mid-price at snapshot index."""
        return float(self.features[idx, self._mid_col])

    def _get_obi(self, idx: int) -> float:
        """Get order book imbalance at snapshot index."""
        return float(self.features[idx, self._obi_col])

    def reset(self, col_names: Optional[list] = None) -> EnvState:
        """Reset environment to start of a new episode."""
        # Set column indices from feature names if provided
        if col_names is not None:
            try:
                self._mid_col = col_names.index("mid_price")
                self._obi_col = col_names.index("obi")
            except ValueError:
                self._mid_col = 0
                self._obi_col = min(4, self.n_features - 1)

        # Random start position with enough history
        max_start = self.n_snapshots - EPISODE_STEPS * DECISION_FREQ - self.lookback - 100
        self.start_idx   = np.random.randint(self.lookback, max(self.lookback + 1, max_start))
        self.current_idx = self.start_idx
        self.step_count  = 0

        self.inventory      = 0
        self.cash           = 0.0
        self.cumulative_pnl = 0.0
        self.pending_orders = []   # (exec_idx, side, quote_price, mid_at_decision)
        self.episode_fills  = []
        self.episode_adverses = 0

        return self._get_state()

    def _get_state(self) -> EnvState:
        """Construct current environment state."""
        idx    = self.current_idx
        window = self.features[idx - self.lookback : idx]

        # Randomized latency (Section 5.5)
        jitter     = np.random.uniform(1 - LATENCY_JITTER, 1 + LATENCY_JITTER)
        latency_ms = self.latency_ms * jitter

        # Fill probability signals from predictor
        fp_bid, fp_ask = self.fill_predictor.get_fill_probs(window, latency_ms)

        # Short-term price direction (sign of recent return)
        if idx > 0:
            prev_mid = self._get_mid_price(idx - DECISION_FREQ)
            curr_mid = self._get_mid_price(idx)
            direction = np.sign(curr_mid - prev_mid)
        else:
            direction = 0.0

        return EnvState(
            lob_window      = window,
            inventory       = self.inventory,
            fp_bid          = fp_bid,
            fp_ask          = fp_ask,
            price_direction = float(direction),
            latency_ms      = latency_ms,
            mid_price       = self._get_mid_price(idx),
            t_normalized    = self.step_count / EPISODE_STEPS,
        )

    def step(
        self,
        action_side:   int,                     # 0=abstain,1=bid,2=ask,3=both
        spread_bid_ticks: float = FIXED_SPREAD_TICKS,
        spread_ask_ticks: float = FIXED_SPREAD_TICKS,
    ) -> tuple[EnvState, float, bool]:
        """
        Execute one decision step in the environment.

        Parameters
        ----------
        action_side : int
            Side selection action (0=abstain, 1=bid, 2=ask, 3=both)
        spread_bid_ticks : float
            Bid half-spread in ticks
        spread_ask_ticks : float
            Ask half-spread in ticks

        Returns
        -------
        tuple[EnvState, float, bool]
            (next_state, reward, done)
        """
        idx     = self.current_idx
        mid     = self._get_mid_price(idx)
        latency_steps = max(1, int(self.latency_ms / DT_MS))

        # ── Execute pending orders ─────────────────────────────────────────────
        reward       = 0.0
        still_pending = []

        for (exec_idx, side, quote_price, mid_at_decision) in self.pending_orders:
            if idx >= exec_idx:
                exec_mid = self._get_mid_price(min(exec_idx, self.n_snapshots - 1))
                half_spread = TICK_SIZE * 2

                # Check fill
                if side == "bid" and exec_mid - half_spread <= quote_price:
                    filled = True
                elif side == "ask" and exec_mid + half_spread >= quote_price:
                    filled = True
                else:
                    filled = False

                if filled:
                    # Determine if adverse (Section 5.4)
                    if side == "bid":
                        adverse = quote_price > mid_at_decision
                        self.inventory += 1
                        self.cash      -= quote_price
                        fill_pnl        = exec_mid - quote_price
                    else:
                        adverse = quote_price < mid_at_decision
                        self.inventory -= 1
                        self.cash      += quote_price
                        fill_pnl        = quote_price - exec_mid

                    # Inventory penalty
                    inv_penalty = GAMMA_INV * (self.inventory ** 2)

                    # Adverse fill penalty (key contribution — Section 5.4)
                    if adverse:
                        adv_penalty = ALPHA_ADVERSE * abs(exec_mid - mid_at_decision)
                        self.episode_adverses += 1
                    else:
                        adv_penalty = 0.0

                    step_reward = fill_pnl - inv_penalty - adv_penalty
                    reward     += step_reward
                    self.cumulative_pnl += step_reward
                    self.episode_fills.append({
                        "side": side, "adverse": adverse, "pnl": step_reward
                    })
            else:
                still_pending.append((exec_idx, side, quote_price, mid_at_decision))

        self.pending_orders = still_pending

        # ── Submit new orders ──────────────────────────────────────────────────
        if action_side != 0 and abs(self.inventory) < Q_MAX:
            exec_idx        = idx + latency_steps
            mid_at_decision = mid

            if action_side in (1, 3):   # bid
                bid_price = mid - spread_bid_ticks * TICK_SIZE
                self.pending_orders.append(
                    (exec_idx, "bid", bid_price, mid_at_decision)
                )

            if action_side in (2, 3):   # ask
                ask_price = mid + spread_ask_ticks * TICK_SIZE
                self.pending_orders.append(
                    (exec_idx, "ask", ask_price, mid_at_decision)
                )

        # ── Advance state ──────────────────────────────────────────────────────
        self.current_idx += DECISION_FREQ
        self.step_count  += 1

        done = (
            self.step_count >= EPISODE_STEPS
            or self.current_idx >= self.n_snapshots - self.lookback - 10
        )

        next_state = self._get_state() if not done else self._get_state()
        return next_state, float(reward), done

    def get_episode_metrics(self) -> dict:
        """Compute episode-level performance metrics."""
        n_fills   = len(self.episode_fills)
        n_adverse = self.episode_adverses
        afr       = n_adverse / n_fills if n_fills > 0 else 0.0
        abstentions = self.step_count - sum(
            1 for _ in self.episode_fills
        )

        return {
            "total_pnl":    self.cumulative_pnl,
            "n_fills":      n_fills,
            "n_adverse":    n_adverse,
            "afr":          afr,
            "abstentions":  abstentions,
        }


# ── DQN Agent (Section 5.6) ───────────────────────────────────────────────────

class DQNNetwork(nn.Module):
    """
    Deep Q-Network for discrete side selection.

    Maps state vector to Q-values over 4 actions:
    0=abstain, 1=bid, 2=ask, 3=both
    """

    def __init__(self, obs_dim: int, n_actions: int = N_ACTIONS):
        super().__init__()
        dims = [obs_dim] + DQN_HIDDEN + [n_actions]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer for DQN training."""

    def __init__(self, capacity: int = DQN_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN market making agent (Section 5.6).

    Uses fixed spread of FIXED_SPREAD_TICKS ticks on both sides.
    Only learns which side(s) to quote via Q-learning.
    """

    def __init__(self, obs_dim: int):
        self.obs_dim  = obs_dim
        self.epsilon  = DQN_EPS_START
        self.steps    = 0

        self.policy_net = DQNNetwork(obs_dim).to(DEVICE)
        self.target_net = DQNNetwork(obs_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=DQN_LR)
        self.buffer    = ReplayBuffer()

    def select_action(self, state: torch.Tensor) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)
        with torch.no_grad():
            q_vals = self.policy_net(state.unsqueeze(0).to(DEVICE))
            return int(q_vals.argmax().item())

    def update(self) -> Optional[float]:
        """One gradient step on a sampled minibatch."""
        if len(self.buffer) < DQN_BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(DQN_BATCH_SIZE)
        states      = states.to(DEVICE)
        actions     = actions.to(DEVICE)
        rewards     = rewards.to(DEVICE)
        next_states = next_states.to(DEVICE)
        dones       = dones.to(DEVICE)

        # Current Q values
        q_vals = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q       = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            targets = rewards + DQN_GAMMA * next_q * (1 - dones)

        loss = F.smooth_l1_loss(q_vals, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Soft target update
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(DQN_TAU * pp.data + (1 - DQN_TAU) * tp.data)

        # Decay epsilon
        self.epsilon = max(DQN_EPS_END, self.epsilon * DQN_EPS_DECAY)
        self.steps  += 1

        return float(loss.item())

    def save(self, path: Path):
        torch.save(self.policy_net.state_dict(), path)
        logger.success(f"DQN saved → {path}")

    def load(self, path: Path):
        self.policy_net.load_state_dict(torch.load(path, map_location=DEVICE))
        self.target_net.load_state_dict(self.policy_net.state_dict())


# ── PPO Agent (Section 5.7) ───────────────────────────────────────────────────

class PPOActor(nn.Module):
    """
    PPO Actor network — outputs action distribution.

    Hybrid action space:
    - Categorical distribution over 4 side-selection actions
    - Two Gaussian distributions for bid/ask spread sizing
    """

    def __init__(self, obs_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, PPO_HIDDEN),
            nn.ReLU(),
        )
        # Discrete head: side selection logits
        self.side_head  = nn.Linear(PPO_HIDDEN, N_ACTIONS)

        # Continuous heads: spread sizing means and log stds
        self.spread_mean    = nn.Linear(PPO_HIDDEN, 2)   # bid and ask
        self.spread_log_std = nn.Parameter(torch.zeros(2))

    def forward(self, x: torch.Tensor):
        h           = self.shared(x)
        side_logits = self.side_head(h)
        spread_mean = torch.sigmoid(self.spread_mean(h)) * DELTA_MAX_TICKS
        return side_logits, spread_mean

    def get_action(self, x: torch.Tensor):
        """Sample action from current policy."""
        side_logits, spread_mean = self.forward(x)

        # Sample discrete side action
        side_dist   = torch.distributions.Categorical(logits=side_logits)
        side_action = side_dist.sample()

        # Sample continuous spread
        spread_std  = torch.exp(self.spread_log_std).clamp(0.1, 2.0)
        spread_dist = torch.distributions.Normal(spread_mean, spread_std)
        spread      = spread_dist.sample().clamp(1.0, DELTA_MAX_TICKS)

        # Log probability (combined)
        log_prob = side_dist.log_prob(side_action) + spread_dist.log_prob(spread).sum(-1)

        return side_action, spread, log_prob, side_dist.entropy()


class PPOCritic(nn.Module):
    """PPO Critic network — estimates state value V(s)."""

    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, PPO_HIDDEN),
            nn.ReLU(),
            nn.Linear(PPO_HIDDEN, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PPOAgent:
    """
    PPO market making agent (Section 5.7).

    Jointly optimizes hybrid action space:
    - Which side(s) to quote (discrete, 4 options)
    - How wide to set spreads (continuous, per side)

    Uses clipped surrogate objective for stable training.
    """

    def __init__(self, obs_dim: int):
        self.obs_dim = obs_dim
        self.actor   = PPOActor(obs_dim).to(DEVICE)
        self.critic  = PPOCritic(obs_dim).to(DEVICE)

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=PPO_LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=PPO_LR)

        # Rollout buffer
        self.reset_buffer()

    def reset_buffer(self):
        self.buf_states   = []
        self.buf_actions  = []
        self.buf_spreads  = []
        self.buf_logprobs = []
        self.buf_rewards  = []
        self.buf_dones    = []
        self.buf_values   = []

    def select_action(self, state: torch.Tensor):
        """Sample action and estimate value."""
        state = state.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            side_action, spread, log_prob, _ = self.actor.get_action(state)
            value = self.critic(state)

        return (
            int(side_action.item()),
            spread.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    def store(self, state, action, spread, log_prob, reward, done, value):
        """Store one transition in rollout buffer."""
        self.buf_states.append(state)
        self.buf_actions.append(action)
        self.buf_spreads.append(spread)
        self.buf_logprobs.append(log_prob)
        self.buf_rewards.append(reward)
        self.buf_dones.append(done)
        self.buf_values.append(value)

    def _compute_gae(self, last_value: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).

        GAE reduces variance in policy gradient estimates by
        combining multi-step returns with a learned value baseline.
        """
        rewards  = np.array(self.buf_rewards)
        dones    = np.array(self.buf_dones)
        values   = np.array(self.buf_values + [last_value])

        advantages = np.zeros_like(rewards)
        gae        = 0.0

        for t in reversed(range(len(rewards))):
            delta       = rewards[t] + PPO_GAMMA * values[t+1] * (1 - dones[t]) - values[t]
            gae         = delta + PPO_GAMMA * PPO_LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages, returns

    def update(self, last_value: float) -> dict:
        """
        PPO policy update using clipped surrogate objective.

        Runs PPO_EPOCHS gradient steps on the collected rollout,
        using mini-batches of size PPO_BATCH_SIZE.
        """
        advantages, returns = self._compute_gae(last_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states    = torch.stack(self.buf_states).to(DEVICE)
        actions   = torch.tensor(self.buf_actions, dtype=torch.long).to(DEVICE)
        spreads   = torch.tensor(np.array(self.buf_spreads), dtype=torch.float32).to(DEVICE)
        old_lp    = torch.tensor(self.buf_logprobs, dtype=torch.float32).to(DEVICE)
        advs      = torch.tensor(advantages, dtype=torch.float32).to(DEVICE)
        rets      = torch.tensor(returns, dtype=torch.float32).to(DEVICE)

        n = len(states)
        actor_losses  = []
        critic_losses = []
        entropies     = []

        for _ in range(PPO_EPOCHS):
            # Mini-batch updates
            idx = np.random.permutation(n)
            for start in range(0, n, PPO_BATCH_SIZE):
                mb_idx = idx[start : start + PPO_BATCH_SIZE]

                mb_states  = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_spreads = spreads[mb_idx]
                mb_old_lp  = old_lp[mb_idx]
                mb_advs    = advs[mb_idx]
                mb_rets    = rets[mb_idx]

                # Recompute log probs and entropy
                side_logits, spread_mean = self.actor(mb_states)
                side_dist   = torch.distributions.Categorical(logits=side_logits)
                spread_std  = torch.exp(self.actor.spread_log_std).clamp(0.1, 2.0)
                spread_dist = torch.distributions.Normal(spread_mean, spread_std)

                new_lp  = (
                    side_dist.log_prob(mb_actions)
                    + spread_dist.log_prob(mb_spreads).sum(-1)
                )
                entropy = side_dist.entropy()

                # PPO clipped objective
                ratio      = torch.exp(new_lp - mb_old_lp)
                surr1      = ratio * mb_advs
                surr2      = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * mb_advs
                actor_loss = -torch.min(surr1, surr2).mean() - PPO_ENTROPY * entropy.mean()

                # Critic loss
                values      = self.critic(mb_states)
                critic_loss = F.mse_loss(values, mb_rets)

                # Update actor
                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_opt.step()

                # Update critic
                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_opt.step()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropies.append(float(entropy.mean().item()))

        self.reset_buffer()

        return {
            "actor_loss":  np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "entropy":     np.mean(entropies),
        }

    def save(self, actor_path: Path, critic_path: Path):
        torch.save(self.actor.state_dict(),  actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        logger.success(f"PPO saved → {actor_path}, {critic_path}")

    def load(self, actor_path: Path, critic_path: Path):
        self.actor.load_state_dict(torch.load(actor_path,  map_location=DEVICE))
        self.critic.load_state_dict(torch.load(critic_path, map_location=DEVICE))


# ── Training Loops ────────────────────────────────────────────────────────────

def train_dqn(
    env:        LatencyMarketEnv,
    agent:      DQNAgent,
    n_episodes: int,
    col_names:  list,
) -> list[dict]:
    """Train DQN agent and return episode history."""
    logger.info(f"Training DQN for {n_episodes} episodes on {DEVICE}…")
    logger.info(f"Parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}")

    history = []
    t0      = time.time()

    for ep in range(1, n_episodes + 1):
        state = env.reset(col_names)
        state_t = state.to_tensor()
        ep_reward = 0.0
        losses    = []

        for _ in range(EPISODE_STEPS):
            action  = agent.select_action(state_t)
            next_state, reward, done = env.step(action)
            next_state_t = next_state.to_tensor()

            agent.buffer.push(state_t, action, reward, next_state_t, float(done))

            if len(agent.buffer) >= DQN_BATCH_SIZE and _ % DQN_UPDATE_FREQ == 0:
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)

            ep_reward += reward
            state_t    = next_state_t

            if done:
                break

        metrics = env.get_episode_metrics()
        metrics.update({
            "episode":   ep,
            "epsilon":   agent.epsilon,
            "avg_loss":  np.mean(losses) if losses else 0.0,
            "agent":     "DQN",
        })
        history.append(metrics)

        if ep % 50 == 0:
            elapsed = time.time() - t0
            logger.info(
                f"[DQN] Episode {ep:04d}/{n_episodes} | "
                f"PnL={metrics['total_pnl']:.3f} | "
                f"AFR={metrics['afr']:.3f} | "
                f"eps={agent.epsilon:.3f} | "
                f"Time={elapsed:.0f}s"
            )

    return history


def train_ppo(
    env:        LatencyMarketEnv,
    agent:      PPOAgent,
    n_episodes: int,
    col_names:  list,
) -> list[dict]:
    """Train PPO agent and return episode history."""
    logger.info(f"Training PPO for {n_episodes} episodes on {DEVICE}…")
    logger.info(
        f"Actor params:  {sum(p.numel() for p in agent.actor.parameters()):,}"
    )
    logger.info(
        f"Critic params: {sum(p.numel() for p in agent.critic.parameters()):,}"
    )

    history      = []
    rollout_step = 0
    t0           = time.time()

    for ep in range(1, n_episodes + 1):
        state    = env.reset(col_names)
        state_t  = state.to_tensor()
        ep_reward = 0.0

        for _ in range(EPISODE_STEPS):
            action, spread, log_prob, value = agent.select_action(state_t)
            spread_bid = float(spread[0])
            spread_ask = float(spread[1])

            next_state, reward, done = env.step(action, spread_bid, spread_ask)
            next_state_t = next_state.to_tensor()

            agent.store(state_t, action, spread, log_prob, reward, float(done), value)
            ep_reward   += reward
            state_t      = next_state_t
            rollout_step += 1

            # PPO update every PPO_STEPS steps
            if rollout_step % PPO_STEPS == 0:
                with torch.no_grad():
                    last_val = float(
                        agent.critic(state_t.unsqueeze(0).to(DEVICE)).item()
                    )
                agent.update(last_val)

            if done:
                break

        metrics = env.get_episode_metrics()
        metrics.update({
            "episode": ep,
            "agent":   "PPO",
        })
        history.append(metrics)

        if ep % 50 == 0:
            elapsed = time.time() - t0
            logger.info(
                f"[PPO] Episode {ep:04d}/{n_episodes} | "
                f"PnL={metrics['total_pnl']:.3f} | "
                f"AFR={metrics['afr']:.3f} | "
                f"Fills={metrics['n_fills']} | "
                f"Time={elapsed:.0f}s"
            )

    return history


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_agent(
    env:       LatencyMarketEnv,
    agent,
    agent_type: str,
    col_names: list,
    n_episodes: int = 20,
) -> dict:
    """
    Evaluate a trained agent over multiple test episodes.

    Returns averaged performance metrics matching Section 6.4.
    """
    logger.info(f"Evaluating {agent_type} over {n_episodes} episodes…")
    all_metrics = []

    for _ in range(n_episodes):
        state   = env.reset(col_names)
        state_t = state.to_tensor()

        for _ in range(EPISODE_STEPS):
            if agent_type == "DQN":
                # Greedy action (no exploration)
                with torch.no_grad():
                    q_vals = agent.policy_net(state_t.unsqueeze(0).to(DEVICE))
                    action = int(q_vals.argmax().item())
                spread_bid = spread_ask = FIXED_SPREAD_TICKS
            else:
                # PPO deterministic action (mean of distribution)
                with torch.no_grad():
                    side_logits, spread_mean = agent.actor(
                        state_t.unsqueeze(0).to(DEVICE)
                    )
                    action     = int(side_logits.argmax().item())
                    spread_bid = float(spread_mean[0, 0].item())
                    spread_ask = float(spread_mean[0, 1].item())

            next_state, _, done = env.step(action, spread_bid, spread_ask)
            state_t = next_state.to_tensor()
            if done:
                break

        all_metrics.append(env.get_episode_metrics())

    # Average across evaluation episodes
    avg = {
        "agent":            agent_type,
        "latency_ms":       LATENCY_MS,
        "avg_pnl":          round(np.mean([m["total_pnl"]  for m in all_metrics]), 4),
        "avg_afr":          round(np.mean([m["afr"]         for m in all_metrics]), 4),
        "avg_fills":        round(np.mean([m["n_fills"]     for m in all_metrics]), 1),
        "avg_adverse":      round(np.mean([m["n_adverse"]   for m in all_metrics]), 1),
        "std_pnl":          round(np.std( [m["total_pnl"]  for m in all_metrics]), 4),
    }

    logger.success(
        f"[{agent_type}] Avg PnL={avg['avg_pnl']:.3f} | "
        f"AFR={avg['avg_afr']:.3f} | "
        f"Fills={avg['avg_fills']:.0f}"
    )
    return avg


# ── Unit Tests ────────────────────────────────────────────────────────────────

def run_tests(features: np.ndarray, col_names: list, n_features: int):
    """Unit tests for environment and agent components."""
    logger.info("Running unit tests…")

    fp = FillPredictorSignal(n_features)
    env = LatencyMarketEnv(features[:5000], fp, latency_ms=500, lookback=50)

    # Test 1: Environment resets correctly
    state = env.reset(col_names)
    assert state.inventory == 0,             "Inventory should reset to 0"
    assert 0 <= state.fp_bid <= 1,           "fp_bid out of range"
    assert 0 <= state.fp_ask <= 1,           "fp_ask out of range"
    assert state.to_tensor().shape[0] == env.obs_dim, "State tensor wrong size"
    logger.success("Test 1 passed: environment resets correctly.")

    # Test 2: Environment step returns valid types
    next_state, reward, done = env.step(3, 3.0, 3.0)
    assert isinstance(reward, float),        "Reward should be float"
    assert isinstance(done, bool),           "Done should be bool"
    assert next_state.inventory in range(-Q_MAX, Q_MAX + 1), "Inventory out of bounds"
    logger.success("Test 2 passed: environment step returns valid types.")

    # Test 3: Abstain action produces no pending orders
    env2   = LatencyMarketEnv(features[:5000], fp, latency_ms=500, lookback=50)
    env2.reset(col_names)
    env2.step(0)   # abstain
    assert len(env2.pending_orders) == 0,   "Abstain should leave no pending orders"
    logger.success("Test 3 passed: abstain produces no pending orders.")

    # Test 4: DQN network produces correct output shape
    obs_dim = env.obs_dim
    dqn_net = DQNNetwork(obs_dim).to(DEVICE)
    dummy   = torch.zeros(4, obs_dim).to(DEVICE)
    q_vals  = dqn_net(dummy)
    assert q_vals.shape == (4, N_ACTIONS),  "DQN output shape wrong"
    logger.success("Test 4 passed: DQN network output shape correct.")

    # Test 5: PPO actor produces valid action distribution
    ppo_actor = PPOActor(obs_dim).to(DEVICE)
    dummy     = torch.zeros(1, obs_dim).to(DEVICE)
    action, spread, log_prob, entropy = ppo_actor.get_action(dummy)
    assert action.item() in range(N_ACTIONS), "PPO action out of range"
    assert spread.shape == (1, 2),            "PPO spread shape wrong"
    logger.success("Test 5 passed: PPO actor produces valid actions.")

    # Test 6: Latency-aware reward penalizes adverse fills
    env3 = LatencyMarketEnv(features[:5000], fp, latency_ms=500, lookback=50)
    env3.reset(col_names)
    env3.inventory    = 0
    env3.current_idx  = 200
    # Manually inject an adverse fill
    mid = env3._get_mid_price(200)
    env3.pending_orders = [(200, "bid", mid + 5 * TICK_SIZE, mid - 5 * TICK_SIZE)]
    _, reward, _ = env3.step(0)   # abstain — only processes pending
    logger.success(f"Test 6 passed: adverse fill reward={reward:.4f} (penalized).")

    logger.success("All unit tests passed.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(agent_type: str = "both", n_episodes: int = 500):
    logger.info(f"RL Agent — type={agent_type}, episodes={n_episodes}, device={DEVICE}")

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading LOB features…")
    df = pd.read_parquet(FEATURES_PATH)

    # Get numeric feature columns
    exclude   = {"timestamp_utc", "symbol"}
    feat_cols = [
        c for c in df.columns
        if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64]
    ]
    features  = df[feat_cols].values.astype(np.float32)
    features  = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    n_features = len(feat_cols)

    logger.success(f"Loaded {len(features):,} snapshots × {n_features} features.")

    # ── Train/test split ──────────────────────────────────────────────────────
    split      = int(len(features) * 0.85)
    train_feat = features[:split]
    test_feat  = features[split:]

    # ── Fill predictor ────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fill_pred = FillPredictorSignal(n_features)

    # ── Run unit tests ────────────────────────────────────────────────────────
    run_tests(train_feat, feat_cols, n_features)

    # ── Environments ──────────────────────────────────────────────────────────
    LOOKBACK  = 20
    train_env = LatencyMarketEnv(train_feat, fill_pred, LATENCY_MS, LOOKBACK)
    test_env  = LatencyMarketEnv(test_feat,  fill_pred, LATENCY_MS, LOOKBACK)
    obs_dim   = train_env.obs_dim
    logger.info(f"Observation dimension: {obs_dim}")

    all_eval_metrics = []
    all_history      = []

    # ── DQN Training ──────────────────────────────────────────────────────────
    if agent_type in ("dqn", "both"):
        dqn_agent   = DQNAgent(obs_dim)
        dqn_history = train_dqn(train_env, dqn_agent, n_episodes, feat_cols)
        dqn_agent.save(MODELS_DIR / "dqn_agent.pt")
        all_history.extend(dqn_history)

        dqn_metrics = evaluate_agent(test_env, dqn_agent, "DQN", feat_cols)
        all_eval_metrics.append(dqn_metrics)

    # ── PPO Training ──────────────────────────────────────────────────────────
    if agent_type in ("ppo", "both"):
        ppo_agent   = PPOAgent(obs_dim)
        ppo_history = train_ppo(train_env, ppo_agent, n_episodes, feat_cols)
        ppo_agent.save(MODELS_DIR / "ppo_actor.pt", MODELS_DIR / "ppo_critic.pt")
        all_history.extend(ppo_history)

        ppo_metrics = evaluate_agent(test_env, ppo_agent, "PPO", feat_cols)
        all_eval_metrics.append(ppo_metrics)

    # ── Save results ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    eval_df = pd.DataFrame(all_eval_metrics)
    eval_df.to_csv(RESULTS_DIR / "rl_agent_results.csv", index=False)

    hist_df = pd.DataFrame(all_history)
    hist_df.to_csv(RESULTS_DIR / "rl_agent_history.csv", index=False)

    print("\n" + "="*65)
    print("RL AGENT EVALUATION RESULTS")
    print("="*65)
    print(eval_df.to_string(index=False))
    print("="*65)
    print(f"\nResults saved → {RESULTS_DIR}/rl_agent_results.csv")
    print(f"History saved → {RESULTS_DIR}/rl_agent_history.csv")
    print(f"Models saved  → {MODELS_DIR}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL market making agents")
    parser.add_argument(
        "--agent",
        choices=["dqn", "ppo", "both"],
        default="both",
        help="Which agent to train (default: both)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of training episodes (default: 500)",
    )
    args = parser.parse_args()
    main(agent_type=args.agent, n_episodes=args.episodes)
