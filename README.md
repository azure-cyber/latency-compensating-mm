# Latency-Compensating Market Making
## An AI Framework for Retail Liquidity Provision

**Author:** Jude Kriel Ramcharitar
**SSRN:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6954278         Currently being published
**Version:** v1.0 — Working Paper (June 2026)

---

## Abstract

High-frequency market makers profit by being faster than everyone else. For retail traders operating under execution delays of 200ms to 2 seconds, market making has been effectively impossible. This paper proposes latency-compensating market making — an AI framework in which predictive models trained on limit order book data substitute for execution speed by anticipating market conditions within the trader's latency window.

---

## Repository Structure

- src/collect_lob_data.py — Kraken WebSocket LOB collector (24/7 on Azure)
- src/lob_features.py — Feature engineering pipeline
- src/fill_predictor.py — LSTM + Conv-Transformer fill probability models
- src/mm_strategy.py — Avellaneda-Stoikov analytical benchmarks
- src/rl_agent.py — DQN + PPO latency-aware RL agents
- src/evaluate.py — Performance evaluation and figures

## Key Results (Synthetic BTC/USD, 500ms Latency)

| Strategy | Sharpe | Max Drawdown |
|---|---|---|
| Avellaneda-Stoikov (baseline) | 158.6 | 42.5 |
| Latency-Penalized AS | 343.1 | 35.2 |
| Latency-Compensating AS | 245.3 | 6.1 |

Key finding: 7x drawdown reduction. Full results on real Kraken data in v2.

## Installation

pip install -r requirements.txt
python src/lob_features.py --mode synthetic --rows 50000
python src/fill_predictor.py --model both --epochs 30
python src/mm_strategy.py --latency 500
python src/rl_agent.py --agent ppo --episodes 500
python src/evaluate.py

## Citation

Ramcharitar, J.K. (2026). Latency-Compensating Market Making:
An AI Framework for Retail Liquidity Provision.
SSRN Working Paper. Abstract ID: 6954278.
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6954278

## Roadmap

- v1 (current): Theoretical framework + synthetic data results
- v2 (July 2026): Real Kraken LOB data + RELAVER comparison
- v3 (September 2026): Multi-asset + journal submission
