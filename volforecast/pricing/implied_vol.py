import numpy as np

from .black_scholes import bsm_price
from .greeks import vega as bsm_vega

# Switch from Newton to bisection when vega falls below this threshold.
# Tiny vega makes the NR step numerically unreliable (deep ITM/OTM options).
_VEGA_MIN = 1e-10


def implied_vol(
    market_price,
    S,
    K,
    T,
    r,
    q=0.0,
    option_type="call",
    tol=1e-8,
    max_iter_newton=50,
    max_iter_bisect=100,
    sigma_lo=1e-6,
    sigma_hi=10.0,
):
    """
    Implied volatility via Newton-Raphson with bisection fallback.

    Parameters
    ----------
    market_price : observed option price (scalar or ndarray)
    S, K, T, r, q : same conventions as bsm_price
    option_type   : "call" or "put"
    tol           : convergence criterion |p(σ) − market_price| < tol
    sigma_lo/hi   : bisection bracket bounds

    Returns
    -------
    ndarray  — NaN where no solution exists (price below intrinsic, T≤0, etc.)
    """
    mp, S, K, T, r, q = (np.asarray(x, dtype=float) for x in (market_price, S, K, T, r, q))
    mp, S, K, T, r, q = np.broadcast_arrays(mp, S, K, T, r, q)
    out_shape = mp.shape

    n = mp.size
    mp = mp.ravel().copy()
    S = S.ravel().copy()
    K = K.ravel().copy()
    T = T.ravel().copy()
    r = r.ravel().copy()
    q = q.ravel().copy()

    iv = np.full(n, np.nan)

    # No-arb bounds: price must lie in (intrinsic, upper_bound).
    fwd_S = S * np.exp(-q * T)
    fwd_K = K * np.exp(-r * T)
    ot = option_type.lower()
    if ot == "call":
        intrinsic = np.maximum(fwd_S - fwd_K, 0.0)
        upper_bound = fwd_S
    elif ot == "put":
        intrinsic = np.maximum(fwd_K - fwd_S, 0.0)
        upper_bound = fwd_K
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    eps = 1e-8
    valid = (T > 0) & (mp >= intrinsic - eps) & (mp <= upper_bound + eps)

    if not valid.any():
        return iv.reshape(out_shape)

    idx = np.where(valid)[0]
    mp_v = mp[idx]
    S_v = S[idx]; K_v = K[idx]; T_v = T[idx]; r_v = r[idx]; q_v = q[idx]
    nv = idx.size

    sigma = np.full(nv, 0.2)
    converged = np.zeros(nv, dtype=bool)
    use_bisect = np.zeros(nv, dtype=bool)

    # ------------------------------------------------------------------ #
    # Phase 1: Newton-Raphson                                              #
    # ------------------------------------------------------------------ #
    for _ in range(max_iter_newton):
        active = ~converged & ~use_bisect
        if not active.any():
            break
        ia = np.where(active)[0]

        p = np.asarray(bsm_price(S_v[ia], K_v[ia], T_v[ia], r_v[ia], sigma[ia], q_v[ia], option_type), dtype=float).ravel()
        v = np.asarray(bsm_vega(S_v[ia], K_v[ia], T_v[ia], r_v[ia], sigma[ia], q_v[ia]), dtype=float).ravel()
        diff = p - mp_v[ia]

        conv = np.abs(diff) < tol
        converged[ia[conv]] = True

        still = ~conv
        if not still.any():
            continue

        ia_s = ia[still]
        v_s = v[still]
        diff_s = diff[still]

        # Low vega → NR step is meaningless; route to bisection
        low_vega = np.abs(v_s) < _VEGA_MIN
        safe_v = np.where(low_vega, 1.0, v_s)
        new_sigma = sigma[ia_s] - diff_s / safe_v

        oob = low_vega | (new_sigma <= sigma_lo) | (new_sigma >= sigma_hi)
        use_bisect[ia_s[oob]] = True
        sigma[ia_s[~oob]] = new_sigma[~oob]

    # ------------------------------------------------------------------ #
    # Phase 2: Bisection for anything not yet converged                   #
    # ------------------------------------------------------------------ #
    needs_bisect = ~converged
    if needs_bisect.any():
        ib = np.where(needs_bisect)[0]
        mp_b = mp_v[ib]

        lo = np.full(ib.size, sigma_lo)
        hi = np.full(ib.size, sigma_hi)

        p_lo = np.asarray(bsm_price(S_v[ib], K_v[ib], T_v[ib], r_v[ib], lo, q_v[ib], option_type), dtype=float).ravel()
        p_hi = np.asarray(bsm_price(S_v[ib], K_v[ib], T_v[ib], r_v[ib], hi, q_v[ib], option_type), dtype=float).ravel()

        # Bracketed only where p(lo) ≤ mp ≤ p(hi); others stay NaN
        bracketed = (p_lo <= mp_b + eps) & (mp_b <= p_hi + eps)
        bisect_done = ~bracketed  # unbracketed ones are already NaN, skip them

        mid = (lo + hi) / 2.0

        for _ in range(max_iter_bisect):
            active_b = bracketed & ~bisect_done
            if not active_b.any():
                break
            ia = np.where(active_b)[0]

            p = np.asarray(bsm_price(S_v[ib[ia]], K_v[ib[ia]], T_v[ib[ia]], r_v[ib[ia]], mid[ia], q_v[ib[ia]], option_type), dtype=float).ravel()
            diff = p - mp_b[ia]

            conv = np.abs(diff) < tol
            bisect_done[ia[conv]] = True
            converged[ib[ia[conv]]] = True
            sigma[ib[ia[conv]]] = mid[ia[conv]]

            still = ~conv
            if not still.any():
                continue
            ia_s = ia[still]
            diff_s = diff[still]

            # price monotone increasing in sigma: price too high → sigma too high → shrink hi
            above = diff_s > 0
            hi[ia_s[above]] = mid[ia_s[above]]
            lo[ia_s[~above]] = mid[ia_s[~above]]
            mid[ia_s] = (lo[ia_s] + hi[ia_s]) / 2.0

    iv[idx[converged]] = sigma[converged]
    return iv.reshape(out_shape)
