# This modules data downloading and basic data processing to clean up the lightcurve data
# last updated: 6-jan-2026
# updated by: Biswaprakash Nayak
# changes made: add the initial 2 functions to download and process lightcurve data

import lightkurve as lk


# this is the function to download lightcurve data from MAST database
# by default, no author means search for all authors
def download_lightcurve_data(target_name: str, mission: str, author: str = "None"):

    # download from specified author 
    if author != "None":
        search_result = lk.search_lightcurve(target_name, mission, author=author)
    # download from all authors
    else:
        search_result = lk.search_lightcurve(target_name, mission)

    # check if the data actually exists (returns a search result)
    if len(search_result) == 0:
        # raise error if no data found
        raise ValueError(f"No lightcurve data found for target '{target_name}' in mission '{mission}'")

    # downloads all the lightcurves found
    lightcurves = search_result.download_all()

    return lightcurves


# this is the function to process lightcurve data
# by default, it removes outliers beyond 5 sigma which should be enough
def process_lightcurve_data(lightcurves, sigma: float = 5.0):

    # it stiches multiple lightcuves (.sticth())
    # then remove empy data points (.remove_nans())
    # then normalize the flux values according to the sigma value (.normalize())
    processed_lightcurves = lightcurves.stitch().remove_nans().normalize().remove_outliers(sigma=sigma)

    # this will be flatten the lightcurve to remove long term trends
    processed_lightcurves = processed_lightcurves.flatten()

    return processed_lightcurves




