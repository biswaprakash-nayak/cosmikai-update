"""
CosmiKAI — MAST Light Curve Fetcher
======================================
Downloads stellar photometry from the Mikulski Archive for Space Telescopes
(MAST) via the ``lightkurve`` library.

Primary strategy: search for preprocessed SAP/PDCSAP light-curve products.
Fallback strategy: download Target Pixel Files (TPFs) and perform simple
aperture photometry if no preprocessed light curves are available.

This module intentionally does *not* flatten or normalise the flux — those
steps belong to the explicit preprocessing pipeline so that they can be
tuned and validated independently.
"""

from __future__ import annotations

import logging

import lightkurve as lk
import numpy as np

log = logging.getLogger("cosmikai.mast_fetch")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_lightcurve(
    target_name: str,
    mission: str,
    author: str | None = None,
    sigma_clip: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Download and minimally clean a stellar light curve from MAST.

    The returned arrays are *not* detrended or normalised; callers should
    apply ``flatten_flux`` and ``normalize_flux`` from ``preprocess.py``
    before running BLS or CNN inference.

    Parameters
    ----------
    target_name : str
        Stellar identifier accepted by MAST (e.g. ``"Kepler-10"``,
        ``"TOI-700"``, ``"KIC 11904151"``).
    mission : str
        Photometry mission: ``"Kepler"``, ``"K2"``, or ``"TESS"``.
    author : str | None
        Restrict to a specific pipeline author (e.g. ``"Kepler"``,
        ``"SPOC"``, ``"QLP"``).  ``None`` searches all available authors.
    sigma_clip : float
        Sigma threshold used for outlier removal after stitching.

    Returns
    -------
    time : np.ndarray
        Barycentric-corrected time values (float32, sorted ascending).
    flux : np.ndarray
        Relative flux values (float32, NaN-free, outlier-cleaned,
        roughly median-normalised to ≈ 1).

    Raises
    ------
    ValueError
        If no data can be found for the requested target and mission.
    RuntimeError
        If the download succeeds but produces an unusable light curve.
    """
    log.info("Fetching light curve — target=%s  mission=%s", target_name, mission)

    # --- Strategy 1: preprocessed light-curve products ---
    try:
        time, flux = _fetch_from_lightcurve(target_name, mission, author, sigma_clip)
        log.info(
            "LC fetch succeeded — %d points for %s/%s", len(time), target_name, mission
        )
        return time, flux
    except _NoDataError:
        log.info(
            "No preprocessed LC found for %s/%s — trying TPF fallback",
            target_name,
            mission,
        )
    except Exception as exc:
        log.warning(
            "Preprocessed LC fetch failed for %s/%s (%s) — trying TPF fallback",
            target_name,
            mission,
            exc,
        )

    # --- Strategy 2: Target Pixel File → aperture photometry ---
    try:
        time, flux = _fetch_from_tpf(target_name, mission, sigma_clip)
        log.info(
            "TPF fallback succeeded — %d points for %s/%s",
            len(time),
            target_name,
            mission,
        )
        return time, flux
    except _NoDataError:
        pass
    except Exception as exc:
        log.warning("TPF fallback also failed for %s/%s: %s", target_name, mission, exc)

    raise ValueError(
        f"No light curve data found for '{target_name}' in mission '{mission}'. "
        "Verify the star name and mission are correct and that data exists in MAST."
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

class _NoDataError(Exception):
    """Raised when a MAST search returns zero results."""


def _fetch_from_lightcurve(
    target_name: str,
    mission: str,
    author: str | None,
    sigma_clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Download preprocessed light curves (SAP/PDCSAP) from MAST.

    Parameters are the same as ``fetch_lightcurve``.

    Raises
    ------
    _NoDataError
        If the MAST search returns zero results.
    """
    kwargs: dict = {"mission": mission}
    if author is not None:
        kwargs["author"] = author

    search = lk.search_lightcurve(target_name, **kwargs)
    if len(search) == 0:
        raise _NoDataError(f"No preprocessed LC results for {target_name}/{mission}")

    log.debug("Found %d LC products, downloading all", len(search))
    lcs = search.download_all()

    # Stitch multi-quarter/sector observations into a single light curve
    lc = lcs.stitch() if hasattr(lcs, "stitch") else lcs

    lc = lc.remove_nans().remove_outliers(sigma=sigma_clip)

    return _lc_to_arrays(lc)


def _fetch_from_tpf(
    target_name: str,
    mission: str,
    sigma_clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Download the first available Target Pixel File and extract a light curve
    via simple aperture photometry using the pipeline mask.

    Parameters are the same as ``fetch_lightcurve``.

    Raises
    ------
    _NoDataError
        If the MAST search returns zero TPF results.
    """
    search = lk.search_targetpixelfile(target_name, mission=mission)
    if len(search) == 0:
        raise _NoDataError(f"No TPF results for {target_name}/{mission}")

    log.debug("Found %d TPF products, downloading first", len(search))
    tpf = search[0].download()

    lc = tpf.to_lightcurve(aperture_mask="pipeline")
    lc = lc.remove_nans().remove_outliers(sigma=sigma_clip)

    return _lc_to_arrays(lc)


def _lc_to_arrays(lc) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a ``lightkurve.LightCurve`` object to clean float32 numpy arrays.

    Handles the two common flux unit conventions used across Kepler, K2, and
    TESS pipelines:

    * **Electrons / second** (or arbitrary counts) — median-normalised to ≈ 1.
    * **Parts-per-million (ppm)** — converted to relative flux (1 + ppm/1e6).

    Parameters
    ----------
    lc : lightkurve.LightCurve
        A cleaned (NaN-free, outlier-removed) light curve.

    Returns
    -------
    time, flux : tuple[np.ndarray, np.ndarray]
        Both float32, NaN/Inf-free, sorted by time.

    Raises
    ------
    RuntimeError
        If the resulting arrays are empty after cleaning.
    """
    time = lc.time.value.astype(np.float32)
    raw_flux = lc.flux.value.astype(np.float32)

    # Detect ppm units used by some TESS pipelines
    unit_str = str(getattr(lc.flux, "unit", "")).lower()
    if "ppm" in unit_str:
        flux = 1.0 + raw_flux / 1.0e6
    else:
        # Normalise by median so flux ≈ 1 out-of-transit
        med = float(np.nanmedian(raw_flux))
        if np.isfinite(med) and med != 0.0:
            flux = raw_flux / med
        else:
            flux = raw_flux

    flux = flux.astype(np.float32)

    # Remove any surviving NaN / Inf
    finite_mask = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[finite_mask], flux[finite_mask]

    if len(time) == 0:
        raise RuntimeError("Light curve is empty after cleaning — cannot proceed.")

    # Ensure chronological order
    order = np.argsort(time)
    return time[order], flux[order]
