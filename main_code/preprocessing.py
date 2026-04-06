# To do the preprocessing of the light curve data and running the bls algorithm

# last updated: 5-Apr-2026
# updated by: Biswaprakash Nayak
# changes made: GPU-accelerated BLS and folding reimplemented (used ai for this)

import logging
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from astropy.timeseries import BoxLeastSquares

from candidates import Candidate

LOG = logging.getLogger("cosmikai.preprocessing")


def prep_flux_for_bls(flux: np.ndarray) -> np.ndarray:
    # Normalize around median for robust BLS-like scoring.
    return (flux - np.nanmedian(flux)).astype(np.float32)


def _standardize_rowwise(x: torch.Tensor) -> torch.Tensor:
    med = torch.median(x, dim=1, keepdim=True).values
    std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (x - med) / std


def _fold_profiles_for_period_chunk(
    time_t: torch.Tensor,
    flux_t: torch.Tensor,
    periods_chunk: torch.Tensor,
    n_phase_bins: int,
) -> torch.Tensor:
    # Build folded phase profiles for a batch of periods in one pass.
    bsz = periods_chunk.shape[0]
    base_t = time_t.min()

    phase = ((time_t.unsqueeze(0) - base_t) / periods_chunk.unsqueeze(1))
    phase = phase - torch.floor(phase)
    bin_idx = (phase * n_phase_bins).long().clamp(0, n_phase_bins - 1)

    sums = torch.zeros((bsz, n_phase_bins), device=time_t.device, dtype=torch.float32)
    cnts = torch.zeros((bsz, n_phase_bins), device=time_t.device, dtype=torch.float32)
    flux_expand = flux_t.unsqueeze(0).expand(bsz, -1)
    ones_expand = torch.ones_like(flux_expand)

    sums.scatter_add_(1, bin_idx, flux_expand)
    cnts.scatter_add_(1, bin_idx, ones_expand)

    profile = sums / cnts.clamp_min(1.0)
    profile = torch.where(cnts > 0, profile, torch.nan)

    row_med = torch.nanmedian(profile, dim=1).values
    row_med = torch.where(torch.isfinite(row_med), row_med, torch.zeros_like(row_med))
    profile = torch.where(torch.isfinite(profile), profile, row_med.unsqueeze(1))

    return profile


def _bls_gpu(
    time: np.ndarray,
    y: np.ndarray,
    periods: np.ndarray,
    durations: np.ndarray,
    device: torch.device,
    n_phase_bins: int = 512,
    period_chunk_size: int = 96,
    progress_callback: Callable[[str, int, str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # GPU approximation of BLS: fold each period, scan box widths, maximize SNR depth.
    LOG.info("GPU BLS: periods=%s durations=%s bins=%s", len(periods), len(durations), n_phase_bins)

    time_t = torch.as_tensor(time, dtype=torch.float32, device=device)
    flux_t = torch.as_tensor(y, dtype=torch.float32, device=device)

    n_periods = len(periods)
    n_durations = len(durations)

    power_vals = torch.zeros((n_periods, n_durations), device=device, dtype=torch.float32)
    transit_times = torch.zeros((n_periods, n_durations), device=device, dtype=torch.float32)
    depths = torch.zeros((n_periods, n_durations), device=device, dtype=torch.float32)

    periods_t = torch.as_tensor(periods, dtype=torch.float32, device=device)
    durations_t = torch.as_tensor(durations, dtype=torch.float32, device=device)
    t0 = time_t.min()
    next_progress_pct = 10

    for start in range(0, n_periods, period_chunk_size):
        end = min(start + period_chunk_size, n_periods)
        p_chunk = periods_t[start:end]
        profile = _fold_profiles_for_period_chunk(time_t, flux_t, p_chunk, n_phase_bins)

        baseline = profile.mean(dim=1)
        sigma = profile.std(dim=1).clamp_min(1e-6)

        for d_idx in range(n_durations):
            width_bins = ((durations_t[d_idx] / p_chunk) * n_phase_bins).round().long()
            width_bins = width_bins.clamp(1, max(2, n_phase_bins // 2))

            unique_w = torch.unique(width_bins)
            for w in unique_w.tolist():
                sel = width_bins == int(w)
                if not torch.any(sel):
                    continue

                prof_sel = profile[sel]
                kernel = torch.ones((1, 1, int(w)), device=device, dtype=torch.float32) / float(w)
                if int(w) > 1:
                    ext = torch.cat([prof_sel, prof_sel[:, : int(w) - 1]], dim=1)
                else:
                    ext = prof_sel

                roll = F.conv1d(ext.unsqueeze(1), kernel).squeeze(1)
                min_vals, min_idx = torch.min(roll, dim=1)

                base_sel = baseline[sel]
                sigma_sel = sigma[sel]
                depth = (base_sel - min_vals).clamp_min(0.0)
                snr = depth / sigma_sel

                local_idx = torch.where(sel)[0]
                global_idx = start + local_idx

                power_vals[global_idx, d_idx] = snr
                depths[global_idx, d_idx] = depth
                transit_times[global_idx, d_idx] = t0 + (min_idx.float() / n_phase_bins) * p_chunk[local_idx]

        progress_pct = int((end * 100) / n_periods)
        if progress_pct >= next_progress_pct or end == n_periods:
            LOG.info("GPU BLS progress: %s%% (%s/%s periods)", progress_pct, end, n_periods)
            if progress_callback is not None:
                progress_callback("bls", progress_pct, f"processed {end}/{n_periods} periods")
            while next_progress_pct <= progress_pct:
                next_progress_pct += 10

    return (
        power_vals.detach().cpu().numpy().astype(np.float32),
        transit_times.detach().cpu().numpy().astype(np.float32),
        depths.detach().cpu().numpy().astype(np.float32),
    )


def _bls_cpu(
    time: np.ndarray,
    y: np.ndarray,
    periods: np.ndarray,
    durations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bls = BoxLeastSquares(time, y)
    power_result = bls.power(periods, durations)

    n_periods = len(periods)
    n_durations = len(durations)

    power_vals = np.asarray(power_result.power, dtype=np.float32).reshape(n_periods, n_durations)
    transit_times = np.asarray(power_result.transit_time, dtype=np.float32).reshape(n_periods, n_durations)
    depths = np.asarray(power_result.depth, dtype=np.float32).reshape(n_periods, n_durations)

    return power_vals, transit_times, depths


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
    progress_callback: Callable[[str, int, str], None] | None = None,
) -> list[Candidate]:
    time = np.asarray(time, dtype=np.float64)
    y = prep_flux_for_bls(np.asarray(flux, dtype=np.float32))

    baseline = float(time.max() - time.min())
    if pmax is None:
        pmax = max(1.0, 0.8 * baseline)

    if durations is None:
        durations = np.linspace(0.03, 0.3, 20).astype(np.float64)

    periods = np.exp(np.linspace(np.log(pmin), np.log(pmax), n_periods)).astype(np.float64)

    try:
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            LOG.info("=== BLS Phase: GPU mode (%s) ===", torch.cuda.get_device_name(0))
            power_vals, transit_times, depths = _bls_gpu(
                time,
                y,
                periods,
                durations,
                device=device,
                progress_callback=progress_callback,
            )
        else:
            if use_gpu:
                LOG.info("GPU requested but not available, falling back to CPU BLS")
            LOG.info("=== BLS Phase: CPU mode ===")
            power_vals, transit_times, depths = _bls_cpu(time, y, periods, durations)
            if progress_callback is not None:
                progress_callback("bls", 100, "CPU BLS complete")
    except Exception as exc:
        LOG.error("GPU BLS failed: %s. Falling back to CPU BLS", exc)
        power_vals, transit_times, depths = _bls_cpu(time, y, periods, durations)

    power_flat = power_vals.flatten()
    order = np.argsort(power_flat)[::-1]

    picked: list[Candidate] = []

    def is_duplicate(p: float) -> bool:
        return any(abs(p - c.period) / c.period < sig_th for c in picked)

    n_durations = len(durations)
    for flat_idx in order:
        period_idx = flat_idx // n_durations
        duration_idx = flat_idx % n_durations

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

    return picked


def _fold_to_bins_gpu(
    time_array: np.ndarray,
    flux_array: np.ndarray,
    period: float,
    t0: float,
    nbins: int,
    device: torch.device,
) -> np.ndarray:
    time_t = torch.as_tensor(time_array, dtype=torch.float32, device=device)
    flux_t = torch.as_tensor(flux_array, dtype=torch.float32, device=device)

    phase = ((time_t - t0 + 0.5 * period) % period) / period
    idx = torch.clamp((phase * nbins).long(), 0, nbins - 1)

    sums = torch.zeros(nbins, device=device, dtype=torch.float32)
    cnts = torch.zeros(nbins, device=device, dtype=torch.float32)
    sums.scatter_add_(0, idx, flux_t)
    cnts.scatter_add_(0, idx, torch.ones_like(flux_t))

    pf = sums / cnts.clamp_min(1.0)
    pf = torch.where(cnts > 0, pf, torch.nan)

    med = torch.nanmedian(pf)
    med = torch.where(torch.isfinite(med), med, torch.tensor(0.0, device=device))
    pf = torch.where(torch.isfinite(pf), pf, med)

    pf = (pf - torch.median(pf)) / pf.std().clamp_min(1e-6)
    return pf.detach().cpu().numpy().astype(np.float32)


def fold_to_bins(
    time_array,
    flux_array,
    period,
    t0,
    nbins: int = 512,
    use_gpu: bool = True,
    device: torch.device | None = None,
    progress_callback: Callable[[str, int, str], None] | None = None,
):
    # GPU path uses mean-per-bin aggregation; CPU path preserves median behavior.
    if use_gpu and torch.cuda.is_available():
        gpu_device = device if device is not None else torch.device("cuda")
        return _fold_to_bins_gpu(
            np.asarray(time_array, dtype=np.float32),
            np.asarray(flux_array, dtype=np.float32),
            float(period),
            float(t0),
            int(nbins),
            gpu_device,
        )

    phase = ((time_array - t0 + 0.5 * period) % period) / period
    bins = np.linspace(0, 1, nbins + 1)
    idx = np.digitize(phase, bins) - 1

    pf = np.full(nbins, np.nan, dtype=np.float32)
    for k in range(nbins):
        vals = flux_array[idx == k]
        if vals.size > 0:
            pf[k] = np.nanmedian(vals)

    med = np.nanmedian(pf)
    pf = np.where(np.isfinite(pf), pf, med).astype(np.float32)
    pf = (pf - np.nanmedian(pf)) / (np.nanstd(pf) + 1e-6)
    return pf
