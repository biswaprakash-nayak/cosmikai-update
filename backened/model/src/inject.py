"""
CosmiKAI — Synthetic Transit Injection
=========================================
Utilities for injection-recovery experiments: inject a synthetic box
transit into a real light curve and measure whether TransitCNN recovers it.

This module is used for model sensitivity analysis and ground-truthing
before satellite deployment.  It is *not* called by the production server.

Typical workflow
----------------
1. Fetch a quiet, transit-free light curve with ``mast_fetch.fetch_lightcurve``.
2. Inject a synthetic transit with ``inject_transit``.
3. Run the full preprocessing pipeline on the injected light curve.
4. Call TransitCNN and compare the returned score to the injected parameters.
5. Repeat over a grid of (period, depth, duration) to build a sensitivity map.

Example
-------
    from backened.model.src.mast_fetch import fetch_lightcurve
    from backened.model.src.inject import inject_transit, InjectionParams
    from backened.model.src.preprocess import flatten_flux, normalize_flux, run_bls, phase_fold, bin_to_fixed_length
    from backened.model.src.model import load_model
    import numpy as np, torch

    time, flux = fetch_lightcurve("KIC 3351888", "Kepler")  # quiet star

    params = InjectionParams(period=3.5, t0=time[0] + 1.0, depth=0.01, duration=0.1)
    _, injected_flux = inject_transit(time, flux, params)

    injected_flux = flatten_flux(injected_flux)
    injected_flux = normalize_flux(injected_flux)
    bls = run_bls(time, injected_flux)
    phase, folded = phase_fold(time, injected_flux, bls.period, bls.t0)
    binned = bin_to_fixed_length(phase, folded)

    med, std = np.median(binned), np.std(binned)
    standardised = (binned - med) / std if std > 1e-10 else binned - med

    model = load_model("backened/model/models/best_model.pt")
    with torch.no_grad():
        score = torch.sigmoid(model(torch.from_numpy(standardised).unsqueeze(0).float())).item()

    print(f"Injected depth={params.depth:.4f}  Recovered score={score:.4f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

log = logging.getLogger("cosmikai.inject")


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InjectionParams:
    """
    Parameters describing a synthetic box transit.

    Attributes
    ----------
    period : float
        Orbital period in days.
    t0 : float
        Mid-transit epoch in the same time system as the light-curve array.
    depth : float
        Fractional transit depth (0–1).  A value of 0.01 means a 1 % dimming.
    duration : float
        Full transit duration (T₁₄) in days.
    """

    period: float
    t0: float
    depth: float
    duration: float


@dataclass
class InjectionResult:
    """
    Result of a single injection-recovery trial.

    Attributes
    ----------
    params : InjectionParams
        The injected signal parameters.
    recovered_score : float
        TransitCNN sigmoid score (0–1) on the injected light curve.
    recovered_period : float
        Best BLS period found during preprocessing.
    period_error_frac : float
        |recovered_period - injected_period| / injected_period.
    recovered : bool
        True if ``recovered_score`` ≥ ``threshold``.
    """

    params: InjectionParams
    recovered_score: float
    recovered_period: float
    period_error_frac: float
    recovered: bool


# ---------------------------------------------------------------------------
# Core injection function
# ---------------------------------------------------------------------------

def inject_transit(
    time: np.ndarray,
    flux: np.ndarray,
    params: InjectionParams,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inject a synthetic box-shaped transit into an existing light curve.

    The transit model is a simple box: flux is multiplied by
    ``(1 - depth)`` during in-transit cadences and left unchanged otherwise.

    Parameters
    ----------
    time : np.ndarray
        Time array in days (float32 or float64).
    flux : np.ndarray
        Flux array to inject into.  Should be trend-free and close to 1.0
        out-of-transit for the injection to be physically meaningful.
    params : InjectionParams
        Transit parameters to inject.

    Returns
    -------
    time : np.ndarray
        Unchanged time array (same object as input).
    injected_flux : np.ndarray
        New flux array with the synthetic transit applied (float32).

    Notes
    -----
    The box model does not include limb darkening or ingress/egress slopes.
    For more realistic injection, replace the box with a trapezoidal or
    Mandel-Agol model.
    """
    injected = flux.copy().astype(np.float32)

    # Phase of each cadence relative to transit centre
    phase = (time - params.t0) % params.period
    # Centre at 0 (transit centre)
    phase = np.where(phase > params.period / 2, phase - params.period, phase)

    # In-transit cadences: |phase| ≤ duration/2
    in_transit = np.abs(phase) <= (params.duration / 2.0)
    injected[in_transit] *= (1.0 - params.depth)

    n_injected = int(in_transit.sum())
    log.debug(
        "Injected transit: period=%.3f d  depth=%.4f  duration=%.3f d  "
        "n_cadences_affected=%d",
        params.period,
        params.depth,
        params.duration,
        n_injected,
    )

    return time, injected


# ---------------------------------------------------------------------------
# Grid sweep helper
# ---------------------------------------------------------------------------

def param_grid(
    periods: list[float],
    depths: list[float],
    durations: list[float],
    t0_offset: float = 0.5,
) -> Iterator[InjectionParams]:
    """
    Generate ``InjectionParams`` instances for a grid of (period, depth, duration).

    Parameters
    ----------
    periods : list[float]
        Trial orbital periods in days.
    depths : list[float]
        Trial transit depths (fractional).
    durations : list[float]
        Trial transit durations in days.
    t0_offset : float
        Fixed offset from time zero for the first mid-transit epoch.

    Yields
    ------
    InjectionParams
        One parameter set per grid point.
    """
    for period in periods:
        for depth in depths:
            for duration in durations:
                yield InjectionParams(
                    period=period,
                    t0=t0_offset,
                    depth=depth,
                    duration=duration,
                )
