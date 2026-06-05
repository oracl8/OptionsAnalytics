# Volatility Forecasting & Options Analytics

A Python library for options pricing, risk sensitivities (Greeks), Monte Carlo simulation,
implied-volatility solving, a real-data volatility surface, and an ML model that forecasts
realized volatility with lookahead-free time-series validation.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

## Project structure

```
volforecast/
  pricing/      black_scholes.py, greeks.py, implied_vol.py
  montecarlo/   gbm.py, variance_reduction.py, exotic.py
  surface/      build_surface.py, plot.py
  vol/          realized_vol.py, estimators.py, garch.py
  forecast/     features.py, models.py, validation.py, metrics.py
  data/         loader.py, cache/
notebooks/      demo.ipynb
tests/
```

## Running tests

```bash
pytest
```

## Milestones

1. Scaffold (this milestone)
2. Black-Scholes + Greeks
3. Implied volatility
4. Data layer
5. Volatility surface
6. Monte Carlo + variance reduction
7. Realized vol + baselines
8. ML forecaster + walk-forward validation
9. Implied vs. forecast + demo notebook
10. Polish + reproducibility

## Methodology

- **Pricing:** Black-Scholes-Merton with full analytical Greeks.
- **Implied vol:** Newton-Raphson (vega-based) with bisection fallback; vectorized over chains.
- **Monte Carlo:** geometric Brownian motion; antithetic and control variates for variance reduction.
- **Realized vol target:** annualized std of log returns over a forward horizon (no lookahead).
- **ML validation:** walk-forward / purged + embargoed cross-validation. Scalers fit inside each
  fold only. Standard k-fold is never used on time-series data here.
- **Baselines:** historical vol, EWMA, GARCH(1,1). The ML model is compared honestly against these;
  failing to beat GARCH is a valid and reported outcome.

## Limitations & assumptions

- Market data sourced from yfinance (free tier): survivorship bias, delayed quotes, gaps possible.
- No transaction costs, bid-ask spread, or slippage modeled in the signal.
- Black-Scholes assumes constant vol and no dividends by default (dividend yield optional).
- Results are for research/educational purposes only — not investment advice.
