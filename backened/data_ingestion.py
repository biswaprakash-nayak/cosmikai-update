# This modules data downloading and basic data processing to clean up the lightcurve data

# last updated: 17-feb-2026
# updated by: Arpon Deb
# changes made: edits to all functions due to errors in process_lightcurve_data()

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


def process_lightcurve_data(lightcurves, sigma: float = 5.0):
    # Avoid normalize() here because some products are zero-centered (ppm)
    lc = lightcurves.stitch().remove_nans()
    lc = lc.remove_outliers(sigma=sigma)
    lc = lc.flatten()
    return lc


def get_time_flux(target_name: str, mission: str, author: str = "None",
                  sigma: float = 5.0, download_all: bool = False):
    lcs = download_lightcurve_data(target_name, mission, author=author, download_all=download_all)
    lc = process_lightcurve_data(lcs, sigma=sigma)

    time = lc.time.value.astype(np.float32)

    # Convert flux safely
    if hasattr(lc.flux, "unit") and lc.flux.unit == u.ppm:
        # ppm is usually zero-centered: convert to relative flux around 1
        flux = (1.0 + lc.flux.value.astype(np.float32) / 1e6)
    else:
        flux = lc.flux.value.astype(np.float32)
        med = np.nanmedian(flux)
        if np.isfinite(med) and med != 0:
            flux = flux / med  # baseline ~1

    # Final cleanup
    m = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[m], flux[m]
    order = np.argsort(time)
    return time[order], flux[order]




