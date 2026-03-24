"""
CosmiKAI — Preprocessing Pipeline
=====================================
Transforms a raw (time, flux) light curve into the 512-point standardised
vector expected by TransitCNN.

Pipeline (must match training exactly)
---------------------------------------
1. ``flatten_flux``      — Remove instrumental/stellar trends (SG filter)
2. ``normalize_flux``    — Scale flux so out-of-transit baseline ≈ 1
3. ``run_bls``           — Box Least Squares period search → best period + metadata
4. ``phase_fold``        — Fold and sort by orbital phase
5. ``bin_to_fixed_length`` — Median-bin to exactly N_BINS = 512 points
6. Standardise           — (x − median) / std  [done in server.py, not here]

⚠ CRITICAL: Step 6 (standardisation) is performed in server.py after
``bin_to_fixed_length``.  Without it, the sigmoid output collapses to a
constant ≈ 0.5035 for *all* inputs.  Do not omit it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.timeseries import BoxLeastSquares
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BLSResult:
    """
    Best-fit parameters returned by the BLS period search.

    Attributes
    ----------
    period : float
        Best-fit orbital period in days.
    t0 : float
        Transit reference time (mid-transit epoch) in days.
    duration : float
        Transit duration in days.
    depth : float
        Fractional transit depth (dimensionless, positive ≈ dimming).
    power : float
        Peak BLS power at the best period.
    """

    period: float
    t0: float
    duration: float
    depth: float
    power: float


# ---------------------------------------------------------------------------
# Step 1 — Detrending
# ---------------------------------------------------------------------------

def flatten_flux(
    flux: np.ndarray,
    window_length: int = 301,
    sigma: int = 2,
) -> np.ndarray:
    """
    Remove long-term stellar/instrumental trends with a Savitzky-Golay filter.

    The flux is divided by a smoothed trend so that the resulting array is
    centred around 1 with short-timescale astrophysical signals preserved.

    Parameters
    ----------
    flux : np.ndarray
        Raw flux values (float32 or float64).
    window_length : int
        Length of the SG filter window in cadences.  Must be odd and ≤ len(flux).
        Typical value: 301 cadences (~6 h for 30-min Kepler LC).
    sigma : int
        (Reserved for future iterative sigma-clipping during trend fitting.)
        Currently unused; present to match the training call signature.

    Returns
    -------
    np.ndarray
        Detrended flux (float32), flux / trend.
    """
    n = len(flux)
    # Enforce odd window length; clamp to array length - 1 (must be < n)
    wl = window_length if window_length % 2 == 1 else window_length + 1
    wl = min(wl, n - 1 if n % 2 == 0 else n)
    wl = max(wl, 5)  # SG filter requires at least polyorder + 2 points

    trend = savgol_filter(flux.astype(np.float64), window_length=wl, polyorder=2)

    # Guard against near-zero trend values
    trend = np.where(np.abs(trend) < 1e-10, 1.0, trend)

    return (flux / trend).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 2 — Normalisation
# ---------------------------------------------------------------------------

def normalize_flux(
    flux: np.ndarray,
    method: str = "median",
) -> np.ndarray:
    """
    Scale flux so that the out-of-transit baseline is ≈ 1.

    Parameters
    ----------
    flux : np.ndarray
        Flux array (typically already flattened).
    method : str
        Normalisation strategy:

        * ``"median"`` — divide by the median (robust to transits).
        * ``"mean"`` — divide by the mean.

    Returns
    -------
    np.ndarray
        Normalised flux (float32).

    Raises
    ------
    ValueError
        If ``method`` is not recognised.
    """
    if method == "median":
        centre = float(np.nanmedian(flux))
    elif method == "mean":
        centre = float(np.nanmean(flux))
    else:
        raise ValueError(
            f"Unknown normalisation method '{method}'. "
            "Supported values: 'median', 'mean'."
        )

    if not np.isfinite(centre) or centre == 0.0:
        return flux.astype(np.float32)

    return (flux / centre).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 3 — BLS period search
# ---------------------------------------------------------------------------

def run_bls(
    time: np.ndarray,
    flux: np.ndarray,
    pmin: float = 0.6,
    pmax: float = 12.0,
    n_periods: int = 5000,
) -> BLSResult:
    """
    Box Least Squares (BLS) period search over a uniform grid of trial periods.

    Uses astropy's ``BoxLeastSquares`` implementation, which is the same
    algorithm used during training-label generation.

    Parameters
    ----------
    time : np.ndarray
        Time array in days (float32 or float64).
    flux : np.ndarray
        Normalised flux array (float32 or float64).
    pmin : float
        Minimum trial period in days.
    pmax : float
        Maximum trial period in days.
    n_periods : int
        Number of uniformly spaced trial periods in [pmin, pmax].

    Returns
    -------
    BLSResult
        Dataclass containing the best-fit period, epoch, duration, depth,
        and peak BLS power.

    Notes
    -----
    Transit durations of 0.03–0.3 days (≈ 0.7–7.2 hours) cover the
    physical range for hot-Jupiter to super-Earth transits.
    """
    # Use float64 for numerical stability inside BLS
    t = np.asarray(time, dtype=np.float64)
    f = np.asarray(flux, dtype=np.float64)

    # Zero-centre flux for BLS (improves numerical conditioning)
    f_centred = f - np.nanmedian(f)

    bls = BoxLeastSquares(t, f_centred)

    periods = np.linspace(pmin, pmax, n_periods, dtype=np.float64)
    durations = np.linspace(0.03, 0.3, 20, dtype=np.float64)  # days

    power = bls.power(periods, durations)

    best_idx = int(np.nanargmax(power.power))

    return BLSResult(
        period=float(power.period[best_idx]),
        t0=float(power.transit_time[best_idx]),
        duration=float(power.duration[best_idx]),
        depth=float(power.depth[best_idx]),
        power=float(power.power[best_idx]),
    )


# ---------------------------------------------------------------------------
# Step 4 — Phase folding
# ---------------------------------------------------------------------------

def phase_fold(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Phase-fold a light curve to the best-fit orbital period.

    The transit is centred at phase 0.5 (i.e. the phase wraps such that
    the transit dip appears in the middle of the folded array).

    Parameters
    ----------
    time : np.ndarray
        Time array in days.
    flux : np.ndarray
        Normalised flux array.
    period : float
        Orbital period in days.
    t0 : float | None
        Transit mid-time epoch.  If ``None``, ``time[0]`` is used as the
        reference epoch.

    Returns
    -------
    phase : np.ndarray
        Phase values in [0, 1), sorted ascending.
    folded_flux : np.ndarray
        Flux values sorted to match ``phase``.
    """
    if t0 is None:
        t0 = float(time[0])

    # Centre the transit at phase 0.5 (standard lightkurve convention)
    phase = ((time - t0 + 0.5 * period) % period) / period

    sort_idx = np.argsort(phase)
    return phase[sort_idx].astype(np.float32), flux[sort_idx].astype(np.float32)


# ---------------------------------------------------------------------------
# Step 5 — Fixed-length binning
# ---------------------------------------------------------------------------

def bin_to_fixed_length(
    phase: np.ndarray,
    flux: np.ndarray,
    n_bins: int = 512,
) -> np.ndarray:
    """
    Median-bin a phase-folded light curve into exactly ``n_bins`` bins.

    Empty bins (no data points) are filled by linear interpolation from
    neighbouring bins so that the output is always NaN-free.

    Parameters
    ----------
    phase : np.ndarray
        Phase values in [0, 1), sorted ascending (output of ``phase_fold``).
    flux : np.ndarray
        Corresponding flux values, same length as ``phase``.
    n_bins : int
        Number of output bins.  Must match the value used during training
        (default 512).

    Returns
    -------
    np.ndarray
        Array of shape ``(n_bins,)`` containing the binned flux (float32).
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(phase, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    binned = np.full(n_bins, np.nan, dtype=np.float32)
    for k in range(n_bins):
        vals = flux[bin_indices == k]
        if vals.size > 0:
            binned[k] = float(np.nanmedian(vals))

    # Fill empty bins by linear interpolation
    nan_mask = ~np.isfinite(binned)
    if nan_mask.any():
        x = np.arange(n_bins, dtype=np.float32)
        valid_x = x[~nan_mask]
        valid_y = binned[~nan_mask]
        if len(valid_x) >= 2:
            binned[nan_mask] = np.interp(x[nan_mask], valid_x, valid_y)
        else:
            # Edge case: almost all bins empty — fill with median of what we have
            binned[nan_mask] = float(np.nanmedian(binned[~nan_mask])) if len(valid_x) > 0 else 0.0

    return binned.astype(np.float32)
