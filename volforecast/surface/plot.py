from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _filter_type(df: pd.DataFrame, option_type: str) -> pd.DataFrame:
    if option_type in ("call", "put"):
        return df[df["option_type"] == option_type]
    return df


def plot_smile(
    surface_df: pd.DataFrame,
    expiry: str,
    option_type: str = "call",
    spot: float | None = None,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Axes:
    """IV vs strike for one expiry — the volatility smile/skew."""
    data = _filter_type(surface_df, option_type)
    data = data[data["expiry"] == expiry].sort_values("strike")
    if data.empty:
        raise ValueError(f"No data for expiry={expiry!r}, option_type={option_type!r}")

    T = data["T"].iloc[0]
    fig_created = ax is None
    if fig_created:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(data["strike"], data["iv"] * 100, marker="o", markersize=3, linewidth=1.5)

    # ATM marker: closest-to-spot strike (or log_moneyness ≈ 0)
    atm_idx = data["log_moneyness"].abs().idxmin()
    atm_strike = data.loc[atm_idx, "strike"]
    ax.axvline(atm_strike, color="grey", linestyle="--", linewidth=0.9, label="ATM")

    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title(f"Vol Smile — {expiry} (T={T:.3f}y, {option_type})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if fig_created:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()

    return ax


def plot_term_structure(
    surface_df: pd.DataFrame,
    option_type: str = "call",
    moneyness_band: float = 0.03,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Axes:
    """Near-ATM implied vol vs time to expiry (the term structure)."""
    data = _filter_type(surface_df, option_type)
    # Keep strikes within moneyness_band of ATM (|K/S - 1| < band)
    atm = data[data["moneyness"].between(1 - moneyness_band, 1 + moneyness_band)]
    if atm.empty:
        raise ValueError(f"No near-ATM data within moneyness_band={moneyness_band}")

    ts = atm.groupby("T")["iv"].median().reset_index().sort_values("T")

    fig_created = ax is None
    if fig_created:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(ts["T"], ts["iv"] * 100, marker="o", markersize=5, linewidth=1.5)
    ax.set_xlabel("Time to Expiry (years)")
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title(f"Vol Term Structure ({option_type}, ATM ±{moneyness_band*100:.0f}%)")
    ax.grid(True, alpha=0.3)

    if fig_created:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()

    return ax


def plot_surface_heatmap(
    surface_df: pd.DataFrame,
    option_type: str = "call",
    n_moneyness_bins: int = 30,
    figsize: tuple = (11, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Heatmap of IV over log-moneyness × maturity.

    Log-moneyness is binned into *n_moneyness_bins* equal-width buckets so the
    colour grid is regular even though raw strikes don't align across expiries.
    """
    data = _filter_type(surface_df, option_type).copy()
    if data.empty:
        raise ValueError(f"No data for option_type={option_type!r}")

    # Bin log-moneyness
    lo, hi = data["log_moneyness"].min(), data["log_moneyness"].max()
    bins = np.linspace(lo, hi, n_moneyness_bins + 1)
    data["lm_bin"] = pd.cut(data["log_moneyness"], bins=bins, labels=False)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    pivot = (
        data.groupby(["T", "lm_bin"])["iv"]
        .mean()
        .unstack("lm_bin")  # columns = moneyness bins, index = T
    )
    # Ensure all bin columns are present
    pivot = pivot.reindex(columns=range(n_moneyness_bins))

    T_labels = [f"{t:.2f}" for t in pivot.index]

    fig, ax = plt.subplots(figsize=figsize)
    mesh = ax.pcolormesh(
        np.arange(n_moneyness_bins),
        np.arange(len(pivot)),
        pivot.values * 100,
        cmap="viridis",
        shading="auto",
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Implied Volatility (%)")

    # Label a subset of moneyness ticks to avoid crowding
    n_xticks = min(7, n_moneyness_bins)
    xtick_idx = np.linspace(0, n_moneyness_bins - 1, n_xticks, dtype=int)
    ax.set_xticks(xtick_idx)
    ax.set_xticklabels([f"{bin_centers[i]:.2f}" for i in xtick_idx], rotation=30, ha="right")

    ax.set_yticks(np.arange(len(T_labels)))
    ax.set_yticklabels(T_labels)

    ax.set_xlabel("Log-Moneyness ln(K/S)")
    ax.set_ylabel("Time to Expiry (years)")
    ax.set_title(f"Implied Vol Surface Heatmap ({option_type})")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

    return fig


def plot_surface_3d(
    surface_df: pd.DataFrame,
    option_type: str = "call",
    figsize: tuple = (10, 7),
    save_path: str | None = None,
) -> plt.Figure:
    """3D scatter of the IV surface: log-moneyness × maturity × IV."""
    data = _filter_type(surface_df, option_type)
    if data.empty:
        raise ValueError(f"No data for option_type={option_type!r}")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        data["T"],
        data["log_moneyness"],
        data["iv"] * 100,
        c=data["iv"] * 100,
        cmap="viridis",
        s=8,
        alpha=0.7,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label("IV (%)")

    ax.set_xlabel("T (years)", labelpad=8)
    ax.set_ylabel("ln(K/S)", labelpad=8)
    ax.set_zlabel("IV (%)", labelpad=8)
    ax.set_title(f"Implied Vol Surface ({option_type})")
    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

    return fig
