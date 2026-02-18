# This modules data downloading and basic data processing to clean up the lightcurve data

# last updated: 18-feb-2026
# updated by: Arpon Deb
# changes made: edits to process_lightcurve_data and get_time_flux to fix data_ingestion.py

import numpy as np
import lightkurve as lk
from astropy import units as u

def download_lightcurve_data(target_name: str, mission: str, author: str | None = None,
                             download_all: bool = True):
    kwargs = {"mission": mission}
    if author not in (None, "None", ""):
        kwargs["author"] = author

    # IMPORTANT: use keyword args so mission doesn't get treated as radius
    search_result = lk.search_lightcurve(target_name, **kwargs)

    if len(search_result) == 0:
        raise ValueError(f"No lightcurve data found for target '{target_name}' in mission '{mission}'")

    return search_result.download_all() if download_all else search_result.download()


def process_lightcurve_data(lightcurves, sigma: float = 5.0, flatten_window_length: int | None = None):
    # If it's a collection, stitch it. If it's already a single LightCurve, use it directly.
    lc = lightcurves.stitch() if hasattr(lightcurves, "stitch") else lightcurves

    lc = lc.remove_nans()
    lc = lc.remove_outliers(sigma=sigma)

    if flatten_window_length is None:
        lc = lc.flatten()
    else:
        lc = lc.flatten(window_length=flatten_window_length)

    return lc



def get_time_flux(target_name: str, mission: str, author: str = "None",
                  sigma: float = 5.0, download_all: bool = False):
    lcs = download_lightcurve_data(target_name, mission, author=author, download_all=download_all)
    lc = process_lightcurve_data(lcs, sigma=sigma)

    time = lc.time.value.astype(np.float32)

    unit_str = str(getattr(lc.flux, "unit", "")).lower()
    if "ppm" in unit_str:
        flux = (1.0 + lc.flux.value.astype(np.float32) / 1e6)
    else:
        flux = lc.flux.value.astype(np.float32)
        med = np.nanmedian(flux)
        if np.isfinite(med) and med != 0:
            flux = flux / med


    # Final cleanup
    m = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[m], flux[m]
    order = np.argsort(time)
    return time[order], flux[order]




