"""
Verify the implied-vol surface for a liquid ticker.

Usage:
    python scripts/verify_surface.py [TICKER]   (default: SPY)

Produces four PNG files in the current directory:
    surface_smile.png, surface_term.png, surface_heatmap.png, surface_3d.png
"""

import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # headless — write to file

from volforecast.surface.build_surface import build_surface
from volforecast.surface.plot import (
    plot_smile,
    plot_surface_3d,
    plot_surface_heatmap,
    plot_term_structure,
)

ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"

print(f"Building IV surface for {ticker} …")
surface = build_surface(ticker, max_expiries=20)

if surface.empty:
    print("ERROR: surface is empty — check data availability.")
    sys.exit(1)

n = len(surface)
iv_min = surface["iv"].min()
iv_max = surface["iv"].max()
n_expiries = surface["expiry"].nunique()
print(f"  {n} data points across {n_expiries} expiries")
print(f"  IV range: {iv_min*100:.1f}% – {iv_max*100:.1f}%")
print(f"  Maturity range: {surface['T'].min():.3f}y – {surface['T'].max():.3f}y")

# Nearest expiry for smile plot
nearest_expiry = surface.sort_values("T")["expiry"].iloc[0]
print(f"  Nearest expiry: {nearest_expiry}")

print("Generating plots …")
plot_smile(surface, nearest_expiry, option_type="call", save_path="surface_smile.png")
print("  surface_smile.png")

plot_term_structure(surface, option_type="call", save_path="surface_term.png")
print("  surface_term.png")

plot_surface_heatmap(surface, option_type="call", save_path="surface_heatmap.png")
print("  surface_heatmap.png")

plot_surface_3d(surface, option_type="call", save_path="surface_3d.png")
print("  surface_3d.png")

print("Done.")
