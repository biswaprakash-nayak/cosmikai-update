# This modules data downloading and basic data processing to clean up the lightcurve data

# last updated: 5-April-2026
# updated by: Biswaprakash Nayak
# changes made: added logging mainly

# imports
# lightkurve for the file reading/downloading
# numpy for array manipulation
# astropy for handling units
# logging for logging messages
# concurrent.futures for handling timeouts during download and processing
# warnings for suppressing non-fatal warnings during processing
# 
import logging
import time as _time
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Callable
import lightkurve as lk
import numpy as np
import astropy.units as u

# set up logging
LOG = logging.getLogger("cosmikai.data_ingestion")

# this is the function to download lightcurve data from MAST database
# by default, no author means search for all authors
# input:
# target_name: the name of the target star (e.g., "Kepler-10")
# mission: the name of the mission (e.g., "Kepler", "TESS
# author: the name of the data author (e.g., "Kepler", "TESS", "SPOC", "QLP", etc.) - optional
# download_all: whether to download all lightcurves found (default True)
# download_timeout_seconds: soft time budget for downloading all products.
# if exceeded, proceed with whatever has already been downloaded.
# max_products: cap how many products to attempt when download_all=True.
# per_product_timeout_seconds: timeout for each individual product download attempt when download_all=True.
# progress_callback: optional callback function to report progress, called as progress_callback(stage: str, pct: int, msg: str)
# output:
# lightcurves: a LightCurveCollection object containing the downloaded lightcurves
def download_lightcurve_data(
    target_name: str,
    mission: str,
    author: str | None = None,
    download_all: bool = True,
    download_timeout_seconds: float | None = 180.0,
    max_products: int | None = None,
    per_product_timeout_seconds: float = 25.0,
    progress_callback: Callable[[str, int, str], None] | None = None,
):

    kwargs = {"mission": mission}
    # download from specified author if mentioned
    if author is not None and author != "None":
        kwargs["author"] = author

    # downloading data from lightkurve library (which uses MAST database)
    search_result = lk.search_lightcurve(target_name, **kwargs)

    # check if the data actually exists (returns a search result)
    if len(search_result) == 0:
        # raise error if no data found
        raise ValueError(f"No lightcurve data found for target '{target_name}' in mission '{mission}'")

    # single best product only
    if not download_all:
        lightcurve = search_result.download()
        if lightcurve is None:
            raise RuntimeError("Lightkurve returned no downloadable product.")
        return lightcurve

    # keep whatever is already fetched when budget is exhausted
    total_available = len(search_result)
    total_to_try = total_available if max_products is None else min(total_available, max_products)
    started = _time.monotonic()
    downloaded: list = []

    # log the download plan with timeouts and product limits
    LOG.info(
        "Downloading up to %s lightcurve products for %s (%s) with budget=%ss, per_product_timeout=%ss",
        total_to_try,
        target_name,
        mission,
        "None" if download_timeout_seconds is None else f"{download_timeout_seconds:.0f}",
        f"{per_product_timeout_seconds:.0f}",
    )

    # attempt to download each product with per-product timeout
    for idx in range(total_to_try):
        current_timeout = per_product_timeout_seconds
        # check if overall download budget is exhausted before starting next download
        if download_timeout_seconds is not None:
            elapsed = _time.monotonic() - started
            if elapsed >= download_timeout_seconds:
                LOG.warning(
                    "Download budget reached after %.1fs. Proceeding with %s/%s products.",
                    elapsed,
                    len(downloaded),
                    total_to_try,
                )
                break
        # adjust current product timeout if approaching overall budget limit
        try:
            if download_timeout_seconds is not None:
                remaining_budget = max(0.1, download_timeout_seconds - (_time.monotonic() - started))
                current_timeout = min(per_product_timeout_seconds, remaining_budget)
        # log the attempt with the current timeout
            LOG.info("Downloading product %s/%s (timeout %.1fs)", idx + 1, total_to_try, current_timeout)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(search_result[idx].download)
                lc = future.result(timeout=current_timeout)
        # if download is successful, add to downloaded list; otherwise log the failure
            if lc is not None:
                downloaded.append(lc)
                LOG.info("Downloaded product %s/%s", idx + 1, total_to_try)
            else:
                LOG.warning("Product %s/%s returned no data", idx + 1, total_to_try)
        except FuturesTimeoutError:
            LOG.warning("Timed out product %s/%s after %.1fs; moving on.", idx + 1, total_to_try, current_timeout)
        except Exception as exc:
            LOG.warning("Failed to download product %s/%s: %s", idx + 1, total_to_try, exc)
        finally:
            progress_pct = int(((idx + 1) * 100) / total_to_try)
            LOG.info(
                "Download progress: %s%% (%s/%s attempted, %s downloaded)",
                progress_pct,
                idx + 1,
                total_to_try,
                len(downloaded),
            )
            if progress_callback is not None:
                progress_callback(
                    "download",
                    progress_pct,
                    f"attempted {idx + 1}/{total_to_try}, downloaded {len(downloaded)}",
                )
    # check if we downloaded anything at all
    if not downloaded:
        raise TimeoutError(
            "No lightcurve products were downloaded before timeout/failure. "
            "Try increasing download_timeout_seconds or narrowing mission/author."
        )
    # log the final download summary
    if progress_callback is not None:
        progress_callback("preprocess", 2, "finalizing downloaded products")
    with warnings.catch_warnings(record=True):
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in cast",
            category=RuntimeWarning,
        )
        return lk.LightCurveCollection(downloaded)


# this is the function to process lightcurve data
# by default, it removes outliers beyond 5 sigma which should be enough
# input: 
# lightcurves: a LightCurveCollection object from lightkurve library
# sigma: the threshold for outlier removal (default 5.0)
# flatten_window_length: the window length for flattening the lightcurve (optional)
# flatten_timeout_seconds: soft time budget for flattening the lightcurve
# progress_callback: optional callback function to report progress, called as progress_callback(stage: str,
# output
# processed_lightcurves: a cleaned and flattened LightCurveCollection object
def process_lightcurve_data(
    lightcurves,
    sigma: float = 5.0,
    flatten_window_length: int | None = None,
    flatten_timeout_seconds: float = 120.0,
    progress_callback: Callable[[str, int, str], None] | None = None,
):
    # helper function to report progress during processing
    def _report(pct: int, msg: str) -> None:
        if progress_callback is not None:
            progress_callback("preprocess", pct, msg)
    # processing the lightcurve data can take a while
    _report(10, "preprocess started")
    # it stiches multiple lightcuves (.sticth())
    if hasattr(lightcurves, "stitch"):
        _report(25, "stitching lightcurves")
        # Astropy may emit a non-fatal metadata cast RuntimeWarning when combining products.
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.filterwarnings(
                    "ignore",
                    message="invalid value encountered in cast",
                    category=RuntimeWarning,
                )
                processed_lightcurves = lightcurves.stitch()
            if caught:
                LOG.info("Suppressed %s non-fatal metadata warning(s) during stitch.", len(caught))
        except Exception as stitch_exc:
            # Some MAST products have incompatible metadata/column dtypes (e.g. quality field).
            # Fallback: use the longest valid single product so inference can proceed.
            LOG.warning("Stitch failed (%s). Falling back to best single product.", stitch_exc)
            best_lc = None
            best_n = -1
            for idx, lc_piece in enumerate(lightcurves):
                try:
                    t = np.asarray(lc_piece.time.value, dtype=np.float64)
                    f = np.asarray(lc_piece.flux.value, dtype=np.float64)
                    m = np.isfinite(t) & np.isfinite(f)
                    n = int(np.count_nonzero(m))
                    if n > best_n:
                        best_n = n
                        best_lc = lc_piece
                except Exception as piece_exc:
                    LOG.warning("Skipping incompatible lightcurve piece %s during fallback: %s", idx, piece_exc)

            if best_lc is None:
                raise RuntimeError("Unable to build fallback lightcurve after stitch failure.") from stitch_exc
            processed_lightcurves = best_lc
    else:
        processed_lightcurves = lightcurves
    # removing NaNs 
    _report(45, "removing NaNs")
    # removes NaNs from the lightcurve data
    processed_lightcurves = processed_lightcurves.remove_nans()
    # removing outliers beyond sigma
    _report(60, f"removing outliers sigma={sigma}")
    # removes outliers from the lightcurve data
    processed_lightcurves = processed_lightcurves.remove_outliers(sigma=sigma)
    # flattening the lightcurve
    _report(75, "flattening lightcurve")
    # optional (probably for future) to prevent very large data from being over flattened and normalize
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        if flatten_window_length is None:
            future = executor.submit(processed_lightcurves.flatten)
        else:
            future = executor.submit(processed_lightcurves.flatten, window_length=flatten_window_length)
        processed_lightcurves = future.result(timeout=flatten_timeout_seconds)
    except FuturesTimeoutError:
        _report(85, f"flatten timeout after {int(flatten_timeout_seconds)}s; using unflattened signal")
        LOG.warning(
            "Flatten timed out after %.1fs. Continuing without flatten for this request.",
            flatten_timeout_seconds,
        )
    finally:
        # Do not wait for timed-out worker to finish; continue request flow.
        executor.shutdown(wait=False, cancel_futures=True)
    _report(90, "flatten complete")
    # normalize it, handled in get_time_flux(), uncomment if that doesn't work well
    # processed_lightcurves = processed_lightcurves.normalize()
    return processed_lightcurves


# this is the main function to get time and flux arrays from the target name and mission
# this is essntially the conversion from the previous two functions to get the final time 
# and flux arrays that will be used for the rest of the pipeline
# inputs:
# target_name: the name of the target star 
# mission: the name of the mission 
# author: the name of the data author - optional
# sigma: the threshold for outlier removal (default 5.0)
# download_all: whether to download all lightcurves found 
# flatten_window_length: the window length for flattening the lightcurve (optional)
# download_timeout_seconds: soft time budget for data download when download_all=True
# max_products: cap number of products attempted when download_all=True
# per_product_timeout_seconds: timeout for each individual product download attempt when download_all=True
# flatten_timeout_seconds: soft time budget for flattening the lightcurve during processing
# progress_callback: optional callback function to report progress
# outputs:
# time: np.ndarray float32 (sorted, finite)
# flux: np.ndarray float32 (cleaned, flattened)
def get_time_flux(target_name: str, mission: str, author: str = "None",
                    sigma: float = 5.0, download_all: bool = False, flatten_window_length: int | None = None,
                    flatten_timeout_seconds: float = 120.0,
                    download_timeout_seconds: float | None = 180.0, max_products: int | None = None,
                    per_product_timeout_seconds: float = 25.0,
                    progress_callback: Callable[[str, int, str], None] | None = None):
    # helper function to report progress during the entire get_time_flux process
    def _report(stage: str, pct: int, msg: str) -> None:
        if progress_callback is not None:
            progress_callback(stage, pct, msg)
    # Download lightcurve data
    lcs = download_lightcurve_data(
        target_name,
        mission,
        author=author,
        download_all=download_all,
        download_timeout_seconds=download_timeout_seconds,
        max_products=max_products,
        per_product_timeout_seconds=per_product_timeout_seconds,
        progress_callback=progress_callback,
    )
    # log the completion of download and the start of processing
    _report("preprocess", 5, "download complete; preparing stitch")
    # process can take a while on large collections; emit stage milestones
    _report("preprocess", 20, "stitch/remove_nans starting")
    # Process lightcurve data *using the second function*
    lc = process_lightcurve_data(
        lcs,
        sigma=sigma,
        flatten_window_length=flatten_window_length,
        flatten_timeout_seconds=flatten_timeout_seconds,
        progress_callback=progress_callback,
    )
    _report("preprocess", 80, "stitch/clean complete; converting arrays")
    # Convert to numpy arrays
    time = lc.time.value.astype(np.float32)
    # Handle flux units and normalization
    unit_str = str(getattr(lc.flux, "unit", "")).lower()
    # If the flux is in parts per million (ppm), convert it to relative flux
    if "ppm" in unit_str:
        flux = (1.0 + lc.flux.value.astype(np.float32) / 1e6)
    else:
        # normalize flux by median if it's not in ppm
        flux = lc.flux.value.astype(np.float32)
        med = np.nanmedian(flux)
        if np.isfinite(med) and med != 0:
            flux = flux / med
    # Remove any remaining NaNs or Infs
    m = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[m], flux[m]
    # Sort by time
    order = np.argsort(time)
    time, flux = time[order], flux[order]
    _report("preprocess", 100, f"ready with {len(time)} datapoints")
    # return the time and flux arrays
    return time, flux
