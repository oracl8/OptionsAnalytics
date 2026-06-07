"""
Milestone 4 verification: fetch live SPY data and print shapes/sample rows.
Results are cached in volforecast/data/cache/ — subsequent runs are instant.

Run from project root:
    .venv/Scripts/python scripts/fetch_demo.py
"""

from volforecast.data.loader import YFinanceSource

src = YFinanceSource()

# --- OHLCV ---
print("=" * 60)
print("SPY OHLCV  (2024-01-01 to 2024-04-01)")
print("=" * 60)
ohlcv = src.get_ohlcv("SPY", start="2024-01-01", end="2024-04-01")
print(f"Shape: {ohlcv.shape}")
print(ohlcv.tail(3).to_string())

# --- Available expiries ---
print("\n" + "=" * 60)
print("SPY available option expiries")
print("=" * 60)
expiries = src.get_available_expiries("SPY")
print(f"Total: {len(expiries)}   First 5: {expiries[:5]}")

# --- Option chain (nearest expiry) ---
print("\n" + "=" * 60)
print("SPY option chain (nearest expiry)")
print("=" * 60)
chain = src.get_option_chain("SPY")
print(f"Expiry : {chain['expiry']}")
print(f"Calls  : {chain['calls'].shape}   Puts: {chain['puts'].shape}")
print("\nCalls sample:")
print(chain["calls"].head(4).to_string(index=False))
print("\nPuts sample:")
print(chain["puts"].head(4).to_string(index=False))
