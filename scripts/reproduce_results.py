"""
Reproduce headline results with fixed seeds.

Usage:
    python scripts/reproduce_results.py

Sections:
  1. Black-Scholes spot check  -- known ATM prices and Greeks
  2. Monte Carlo convergence   -- GBM pricer vs. closed form (seed=42)
  3. Walk-forward CV results   -- SPY 2019-2024, horizon=5d (uses cached data)
"""

from __future__ import annotations

import numpy as np

from volforecast.pricing.black_scholes import bsm_price
from volforecast.pricing.greeks import delta, vega
from volforecast.montecarlo.gbm import mc_price
from volforecast.montecarlo.variance_reduction import (
    mc_price_antithetic,
    mc_price_control_variate,
)
from volforecast.forecast.metrics import run_and_print

# ── reference parameters: S=K=100, T=1y, r=5%, σ=20%, q=0 ─────────────────
S0, K0, T0, r0, sig0, q0 = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def main() -> None:
    # ── 1. Black-Scholes spot check ─────────────────────────────────────────
    section("1. Black-Scholes spot check  (S=K=100, T=1y, r=5%, vol=20%)")

    bs_call = float(bsm_price(S0, K0, T0, r0, sig0, q0, "call"))
    bs_put  = float(bsm_price(S0, K0, T0, r0, sig0, q0, "put"))
    d_call  = float(delta(S0, K0, T0, r0, sig0, q0, "call"))
    d_put   = float(delta(S0, K0, T0, r0, sig0, q0, "put"))
    v       = float(vega(S0, K0, T0, r0, sig0, q0))

    print(f"  Call price  : {bs_call:.4f}   (reference: 10.4506)")
    print(f"  Put  price  : {bs_put:.4f}   (reference:  5.5735)")
    print(f"  Call delta  : {d_call:.4f}   (reference:  0.6368)")
    print(f"  Put  delta  : {d_put:.4f}  (reference: -0.3632)")
    print(f"  Vega        : {v:.4f}  (reference: 37.524)")

    # Put-call parity check
    pcp = abs((bs_call - bs_put) - (S0 - K0 * np.exp(-r0 * T0)))
    print(f"  Put-call parity residual: {pcp:.2e}  (should be ~0)")

    # ── 2. Monte Carlo convergence ──────────────────────────────────────────
    section("2. Monte Carlo convergence  (seed=42, 200k paths)")

    mc_p, mc_se = mc_price(S0, K0, T0, r0, sig0, q0, "call", n_paths=200_000, seed=42)
    print(f"  Naive MC price    : {mc_p:.4f} ± {mc_se:.4f}")
    print(f"  BS closed form    : {bs_call:.4f}")
    print(f"  Error / stderr    : {abs(mc_p - bs_call) / mc_se:.2f}σ  (should be < 3)")

    anti = mc_price_antithetic(S0, K0, T0, r0, sig0, q0, n_paths=100_000, seed=42)
    cv   = mc_price_control_variate(S0, K0, T0, r0, sig0, q0, n_paths=100_000, seed=42)
    print(f"\n  Variance-reduction summary (100k paths each):")
    print(f"    Antithetic variates : {anti['reduction_pct']:.1f}% variance reduction")
    print(f"    Control variate     : {cv['reduction_pct']:.1f}% variance reduction")

    # ── 3. Walk-forward CV results ──────────────────────────────────────────
    section("3. Walk-forward CV  (SPY 2019-2024, horizon=5d, min_train=252, step=21)")
    print("  Loading cached data and running CV (may take ~30s) …\n")

    run_and_print(
        ticker="SPY",
        horizon=5,
        start="2019-01-01",
        end="2024-12-31",
        min_train=252,
        step=21,
    )

    print("All sections complete.")


if __name__ == "__main__":
    main()
