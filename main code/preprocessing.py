# To do the preprocessing of the light curve data and running the bls algorithm

# last updated: 23-feb-2026
# updated by: Biswaprakash Nayak
# changes made: creating the file and adding the code

# imports
# numpy for array manipulation so far
# astropy for the bls algorithm and time manipulation
# candidates is the class code I made and calling on that
import numpy as np
from astropy.timeseries import BoxLeastSquares
from candidates import Candidate

# this takes the flux inport which is normalized and makes it have a zero mean so its easier for bls
# input: flux array
# output: normalized flux array with zero mean
def prep_flux_for_bls(flux: np.ndarray) -> np.ndarray:
    # very simple, this does all the conversion
    return (flux - np.nanmedian(flux)).astype(np.float32)

# this does the bls algorithm for the top k candidates,
# input:
# time: array of time values
# flux: array of flux values
# k: number of candidates to return
# pmin: minimum period to search for
# pmax: maximum period to search for
# n_periods: number of periods to evaluate between pmin and pmax
# durations: array of durations to evaluate (if None, uses default range)
# sig_th: significance threshold for period selection (default 0.01) to prevent duplicates (or very similar periods)
def bls_topk(
    time: np.ndarray,
    flux: np.ndarray,
    k: int = 15,
    pmin: float = 0.5,
    pmax: float | None = None,
    n_periods: int = 20000,
    durations: np.ndarray | None = None,
    sig_th: float = 0.02,
) -> list[Candidate]:

    # convert to numpy arrays and prepare flux for bls
    time = np.asarray(time, dtype=np.float64)
    y = prep_flux_for_bls(np.asarray(flux, dtype=np.float32))

    # set pmax based on the time span of the data if not provided
    baseline = float(time.max() - time.min())

    # if pmax is not provided, set it to a value that allows for at least a few transits in the data span (e.g., 80% of the baseline)
    if pmax is None:
        # making it like 80% so it doesn't treat the whole data span as one period or something
        pmax = max(1.0, 0.8 * baseline)

    # if durations is not provided, use a default range of durations (in days)
    if durations is None:
        # 0.03 to 0.3 days seems reasonable for a transit so it doesn't treat small fluctuations as transits and doesn't treat the whole data span as one transit
        durations = np.linspace(0.03, 0.3, 20).astype(np.float64)  # 0.03 - 0.3 days is 0.72 - 7.2 hours

    # create a logarithmic grid of periods to search, supposedly better for bls
    periods = np.exp(np.linspace(np.log(pmin), np.log(pmax), n_periods)).astype(np.float64)

    # runs the bls algorithm using astropy's BoxLeastSquares class
    bls = BoxLeastSquares(time, y)
    # computes the bls power for the specified periods and durations
    power = bls.power(periods, durations)

    # Sort by descending power
    order = np.argsort(power.power)[::-1]

    # creates the list where the candidates will be stored
    picked: list[Candidate] = []

    # checks if the period is a duplicate
    def is_duplicate(p: float) -> bool:
        return any(abs(p - c.period) / c.period < sig_th for c in picked)

    # iterates through the sorted periods and picks the top k candidates that are not duplicates based on the significance threshold
    for idx in order:
        p = float(power.period[idx])
        if is_duplicate(p):
            continue

        cand = Candidate(
            period=p,
            t0=float(power.transit_time[idx]),
            duration=float(power.duration[idx]),
            depth=float(power.depth[idx]),
            power=float(power.power[idx]),
        )
        picked.append(cand)

        if len(picked) >= k:
            break

    # returns the list of candidates
    return picked

# folds the time and flux arrays into a phase folded light curve from the bls candidates
# input:
# time_array: array of time values
# flux_array: array of flux values
# period: period of the candidate
# t0: transit time of the candidate
# nbins: number of bins to fold into (default 512)
# output:
# pf: array of folded and binned flux values, standardized to have zero mean and unit variance
def fold_to_bins(time_array, flux_array, period, t0, nbins=512):
    # getting the phase and binning the flux values
    phase = ((time_array - t0 + 0.5 * period) % period) / period
    bins = np.linspace(0, 1, nbins + 1)
    idx = np.digitize(phase, bins) - 1

    # compute the median flux in each bin, ignoring empty bins
    pf = np.full(nbins, np.nan, dtype=np.float32)
    for k in range(nbins):
        vals = flux_array[idx == k]
        if vals.size > 0:
            pf[k] = np.nanmedian(vals)

    # Fill empty bins with the median of the non-empty bins
    med = np.nanmedian(pf)
    pf = np.where(np.isfinite(pf), pf, med).astype(np.float32)

    # Standardize the flux and return it
    pf = (pf - np.nanmedian(pf)) / (np.nanstd(pf) + 1e-6)
    return pf