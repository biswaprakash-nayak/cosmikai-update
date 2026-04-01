# To do the preprocessing of the light curve data and running the bls algorithm

# last updated: 1-Apr-2026
# updated by: Biswaprakash Nayak
# changes made: creating the file and adding the code
# GPU BLS added

# imports
# numpy for array manipulation so far
# astropy for the bls algorithm and time manipulation
# candidates is the class code I made and calling on that
# torch for GPU-accelerated BLS computation
import numpy as np
import torch
from astropy.timeseries import BoxLeastSquares
from candidates import Candidate

# this takes the flux inport which is normalized and makes it have a zero mean so its easier for bls
# input: flux array
# output: normalized flux array with zero mean
def prep_flux_for_bls(flux: np.ndarray) -> np.ndarray:
    # very simple, this does all the conversion
    return (flux - np.nanmedian(flux)).astype(np.float32)

# GPU-accelerated BLS using PyTorch
# input:
# time: time values (float64)
# y: flux values (float32)
# periods: periods to search
# durations: transit durations to search
# output:
# power_vals: power values for each (period, duration) pair
# transit_times: optimal transit time (t0) for each period
# depths: transit depth for each (period, duration) pair
def _bls_gpu(
    time: np.ndarray,
    y: np.ndarray,
    periods: np.ndarray,
    durations: np.ndarray,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # detects the cuda device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move data to GPU
    time_t = torch.from_numpy(time.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y.astype(np.float32)).to(device)
    periods_t = torch.from_numpy(periods.astype(np.float32)).to(device)
    durations_t = torch.from_numpy(durations.astype(np.float32)).to(device)
    
    n_periods = len(periods)
    n_durations = len(durations)
    n_times = len(time_t)
    
    # Initialize output arrays
    power_vals = np.zeros((n_periods, n_durations), dtype=np.float32)
    transit_times = np.zeros((n_periods, n_durations), dtype=np.float32)
    depths = np.zeros((n_periods, n_durations), dtype=np.float32)
    
    # Process periods in batches to avoid GPU memory issues (currently 100)
    batch_size = min(100, n_periods)  # 
    
    # Loop over period batches
    for period_start in range(0, n_periods, batch_size):
        period_end = min(period_start + batch_size, n_periods)
        periods_batch = periods_t[period_start:period_end]  # (B,)
        batch_len = len(periods_batch)
        
        # For each period, compute phase (vectorized)
        # phase shape: (B, n_times)
        phases = (time_t - time_t[0]).unsqueeze(0) / periods_batch.unsqueeze(1)  
        phases = phases - torch.floor(phases)  # normalize to [0, 1)
        
        # For each duration in batch
        for dur_idx, duration in enumerate(durations_t):
            # Create transit box model for this duration
            # transit occurs when phase is in [0.5 - dur/2/period, 0.5 + dur/2/period]
            transit_phase_width = (duration / periods_batch).clamp(0, 1)  # (B,)
            
            # Try all transit times by sliding window
            # Simplified: compute power at high-resolution transit phases
            n_phases = 100
            phase_offsets = torch.linspace(0, 1, n_phases, device=device)
            
            # Track best for each period in batch independently
            best_power = torch.full((batch_len,), -np.inf, device=device)
            best_t0 = torch.zeros(batch_len, device=device)
            best_depth = torch.zeros(batch_len, device=device)
            
            for phase_offset in phase_offsets:
                transit_start = phase_offset - transit_phase_width / 2  # (B,)
                transit_end = phase_offset + transit_phase_width / 2  # (B,)
                
                # Create mask for transit region (vectorized over period batch)
                # in_transit shape: (B, n_times)
                in_transit = ((phases >= transit_start.unsqueeze(1)) & 
                                (phases <= transit_end.unsqueeze(1))).float()
                
                # Compute statistics
                flux_in = torch.where(in_transit.bool(), y_t, torch.tensor(0.0, device=device))
                flux_out = torch.where((1 - in_transit).bool(), y_t, torch.tensor(0.0, device=device))
                
                count_in = in_transit.sum(dim=1).clamp(min=1)  # (B,)
                count_out = (1 - in_transit).sum(dim=1).clamp(min=1)  # (B,)
                
                mean_in = flux_in.sum(dim=1) / count_in  # (B,)
                mean_out = flux_out.sum(dim=1) / count_out  # (B,)
                
                # Compute depth as difference
                depth = (mean_out - mean_in)  # (B,)
                
                # Compute chi-squared like statistic (power is inverse)
                var_in = (flux_in ** 2).sum(dim=1) / count_in - mean_in ** 2
                var_out = (flux_out ** 2).sum(dim=1) / count_out - mean_out ** 2
                
                # Power: signal-to-noise like metric
                power = (depth ** 2) / (var_in / count_in + var_out / count_out + 1e-8)  # (B,)
                
                # Update best for each period in batch where power is better
                mask = power > best_power
                best_power = torch.where(mask, power, best_power)
                best_t0 = torch.where(mask, phase_offset, best_t0)
                best_depth = torch.where(mask, depth, best_depth)
            
            # Store results
            for b in range(batch_len):
                period_idx = period_start + b
                power_vals[period_idx, dur_idx] = best_power[b].item()
                transit_times[period_idx, dur_idx] = best_t0[b].item() * periods[period_idx] + time[0]
                depths[period_idx, dur_idx] = abs(best_depth[b].item())
    
    return power_vals, transit_times, depths

# CPU fallback for BLS (original astropy implementation)
def _bls_cpu(
    time: np.ndarray,
    y: np.ndarray,
    periods: np.ndarray,
    durations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # uses the cpu instead
    bls = BoxLeastSquares(time, y)
    power_result = bls.power(periods, durations)
    
    # Reshape to match GPU output: (n_periods, n_durations)
    n_periods = len(periods)
    n_durations = len(durations)
    
    power_vals = power_result.power.reshape(n_periods, n_durations)
    transit_times = power_result.transit_time.reshape(n_periods, n_durations)
    depths = power_result.depth.reshape(n_periods, n_durations)
    
    return power_vals, transit_times, depths

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
# use_gpu: whether to use GPU-accelerated BLS (default True if CUDA available)
def bls_topk(
    time: np.ndarray,
    flux: np.ndarray,
    k: int = 15,
    pmin: float = 0.5,
    pmax: float | None = None,
    n_periods: int = 20000,
    durations: np.ndarray | None = None,
    sig_th: float = 0.02,
    use_gpu: bool = True,
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

    # runs the bls algorithm using GPU if available and requested, otherwise CPU
    try:
        if use_gpu and torch.cuda.is_available():
            print(f"Using GPU-accelerated BLS on {torch.cuda.get_device_name(0)}")
            power_vals, transit_times, depths = _bls_gpu(time, y, periods, durations)
        else:
            if use_gpu:
                print("GPU requested but not available, falling back to CPU BLS")
            power_vals, transit_times, depths = _bls_cpu(time, y, periods, durations)
    except Exception as e:
        print(f"GPU BLS failed: {e}, falling back to CPU BLS")
        power_vals, transit_times, depths = _bls_cpu(time, y, periods, durations)

    # Flatten and sort by descending power
    power_flat = power_vals.flatten()
    order = np.argsort(power_flat)[::-1]

    # creates the list where the candidates will be stored
    picked: list[Candidate] = []

    # checks if the period is a duplicate
    def is_duplicate(p: float) -> bool:
        return any(abs(p - c.period) / c.period < sig_th for c in picked)

    # iterates through the sorted periods and picks the top k candidates that are not duplicates based on the significance threshold
    for flat_idx in order:
        period_idx = flat_idx // len(durations)
        duration_idx = flat_idx % len(durations)
        
        p = float(periods[period_idx])
        if is_duplicate(p):
            continue

        cand = Candidate(
            period=p,
            t0=float(transit_times[period_idx, duration_idx]),
            duration=float(durations[duration_idx]),
            depth=float(depths[period_idx, duration_idx]),
            power=float(power_vals[period_idx, duration_idx]),
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