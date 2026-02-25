# This modules data downloading and basic data processing to clean up the lightcurve data

# last updated: 18-feb-2026
# updated by: Biswaprakash Nayak
# changes made: implementing changes made by arpon and adding comments

# imports
# lightkurve for the file reading/downloading
# numpy for array manipulation
import lightkurve as lk
import numpy as np
import astropy.units as u

# this is the function to download lightcurve data from MAST database
# by default, no author means search for all authors
# input:
# target_name: the name of the target star (e.g., "Kepler-10")
# mission: the name of the mission (e.g., "Kepler", "TESS
# author: the name of the data author (e.g., "Kepler", "TESS", "SPOC", "QLP", etc.) - optional
# download_all: whether to download all lightcurves found (default True)
# output:
# lightcurves: a LightCurveCollection object containing the downloaded lightcurves
def download_lightcurve_data(target_name: str, mission: str, author: str | None = None, download_all: bool = True):

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

    # downloads all the lightcurves found
    lightcurves = search_result.download_all() if download_all else search_result.download()

    return lightcurves


# this is the function to process lightcurve data
# by default, it removes outliers beyond 5 sigma which should be enough
# input: 
# lightcurves: a LightCurveCollection object from lightkurve library
# sigma: the threshold for outlier removal (default 5.0)
# output
# processed_lightcurves: a cleaned and flattened LightCurveCollection object
def process_lightcurve_data(lightcurves, sigma: float = 5.0, flatten_window_length: int | None = None):

    # it stiches multiple lightcuves (.sticth())
    processed_lightcurves = lightcurves.stitch() if hasattr(lightcurves, "stitch") else lightcurves

    # removes NaNs from the lightcurve data
    processed_lightcurves = processed_lightcurves.remove_nans()

    # removes outliers from the lightcurve data
    processed_lightcurves = processed_lightcurves.remove_outliers(sigma=sigma)

    # optional (probably for future) to prevent very large data from being over flattened and normalize
    if flatten_window_length is None:
        processed_lightcurves = processed_lightcurves.flatten()
    else:
        processed_lightcurves = processed_lightcurves.flatten(window_length=flatten_window_length)

    # don't delete for now, might be broken
    # should normalize it
    processed_lightcurves = processed_lightcurves.normalize()

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
# outputs:
# time: np.ndarray float32 (sorted, finite)
# flux: np.ndarray float32 (cleaned, flattened)
def get_time_flux(target_name: str, mission: str, author: str = "None",
                    sigma: float = 5.0, download_all: bool = False, flatten_window_length: int | None = None):
    
    # Download lightcurve data *using the first function*
    lcs = download_lightcurve_data(target_name, mission, author=author, download_all=download_all)
    # Process lightcurve data *using the second function*
    lc = process_lightcurve_data(lcs, sigma=sigma, flatten_window_length=flatten_window_length)

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

    # return the time and flux arrays
    return time, flux
