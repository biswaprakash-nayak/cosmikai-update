from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.config import load_config
from src.mast_fetch import fetch_lightcurve
from src.preprocess import (
    bin_to_fixed_length,
    flatten_flux,
    normalize_flux,
    phase_fold,
    run_bls,
)
from src.inject import inject_box_transit


def load_targets(target_file: str) -> List[str]:
    """Load target list from a text file, one target per line."""
    p = Path(target_file)
    if not p.exists():
        raise FileNotFoundError(f"Target file not found: {target_file}")
    targets = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    return targets


def main():
    cfg = load_config("configs/default.yaml")
    np.random.seed(cfg.get("seed", 42))

    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load target list
    target_file = cfg["data"].get("target_file", "data/raw/kepler_targets_1922.txt")
    targets = load_targets(target_file)
    print(f"[INFO] Loaded {len(targets)} targets from {target_file}")

    # Injection config
    n_inj = cfg["inject"].get("n_injections_per_star", 3)
    depth_min = cfg["inject"].get("depth_ppm_min", 200)
    depth_max = cfg["inject"].get("depth_ppm_max", 5000)
    dur_min = cfg["inject"].get("duration_frac_min", 0.015)
    dur_max = cfg["inject"].get("duration_frac_max", 0.10)
    n_bins = cfg["data"].get("n_bins", 512)

    # Checkpoint setup - save progress every N stars
    checkpoint_every = 50
    checkpoint_file = processed_dir / "checkpoint.json"

    # Resume from checkpoint if it exists
    start_idx = 0
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    meta: List[Dict] = []

    if checkpoint_file.exists():
        print("[INFO] Found checkpoint, resuming...")
        cp = json.loads(checkpoint_file.read_text())
        start_idx = cp["next_idx"]
        # Load existing partial data
        partial_X = np.load(processed_dir / "X_partial.npy")
        partial_y = np.load(processed_dir / "y_partial.npy")
        X_list = list(partial_X)
        y_list = list(partial_y)
        meta = cp["meta"]
        print(f"[INFO] Resuming from star {start_idx}, {len(X_list)} samples so far")

    skipped = 0
    processed = 0

    for i in range(start_idx, len(targets)):
        target = targets[i]
        star_num = i + 1

        try:
            time, flux = fetch_lightcurve(target, mission="Kepler")
        except Exception as e:
            print(f"[{star_num}/{len(targets)}] SKIP {target}: {e}")
            skipped += 1
            continue

        try:
            flux = flatten_flux(
                flux,
                cfg["preprocess"]["flatten_window"],
                cfg["preprocess"]["flatten_polyorder"],
            )
            flux = normalize_flux(flux, cfg["preprocess"]["normalize"])

            period = run_bls(
                time, flux,
                cfg["bls"]["period_min"],
                cfg["bls"]["period_max"],
                cfg["bls"]["n_periods"],
            )

            phase, folded = phase_fold(time, flux, period)
            binned = bin_to_fixed_length(phase, folded, n_bins)

            # --- Negative sample: real LC folded on BLS period (no injection) ---
            X_list.append(binned.copy())
            y_list.append(0)
            meta.append({
                "star": target, "label": 0, "period": float(period), "type": "negative"
            })

            # --- Positive samples: inject synthetic transits ---
            for j in range(n_inj):
                depth_ppm = np.random.uniform(depth_min, depth_max)
                dur_frac = np.random.uniform(dur_min, dur_max)
                injected, t0 = inject_box_transit(phase, folded, depth_ppm, dur_frac)
                inj_binned = bin_to_fixed_length(phase, injected, n_bins)
                X_list.append(inj_binned)
                y_list.append(1)
                meta.append({
                    "star": target, "label": 1, "period": float(period),
                    "type": "injected", "depth_ppm": float(depth_ppm),
                    "duration_frac": float(dur_frac), "t0": float(t0),
                })

            processed += 1
            total_samples = len(X_list)
            if processed % 10 == 0:
                print(f"[{star_num}/{len(targets)}] OK {target} | {processed} stars done, {total_samples} samples, {skipped} skipped")

        except Exception as e:
            print(f"[{star_num}/{len(targets)}] PREPROCESS FAIL {target}: {e}")
            skipped += 1
            continue

        # Clear lightkurve cache to save disk space
        cache_dir = Path.home() / ".lightkurve" / "cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)

        # Checkpoint
        if processed % checkpoint_every == 0 and processed > 0:
            print(f"[CHECKPOINT] Saving at {processed} stars ({len(X_list)} samples)...")
            np.save(processed_dir / "X_partial.npy", np.stack(X_list))
            np.save(processed_dir / "y_partial.npy", np.array(y_list, dtype=np.int64))
            checkpoint_file.write_text(json.dumps({
                "next_idx": i + 1,
                "processed": processed,
                "skipped": skipped,
                "meta": meta,
            }))
            print(f"[CHECKPOINT] Saved.")

    # Final save
    if len(X_list) == 0:
        print("[ERROR] No samples created! All targets were skipped.")
        return

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    np.save(processed_dir / "X.npy", X)
    np.save(processed_dir / "y.npy", y)
    (processed_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Clean up checkpoint files
    for f in [processed_dir / "X_partial.npy", processed_dir / "y_partial.npy", checkpoint_file]:
        if f.exists():
            f.unlink()

    print("=" * 60)
    print("DATASET BUILD COMPLETE")
    print(f"Stars processed: {processed}")
    print(f"Stars skipped:   {skipped}")
    print(f"Total samples:   {len(X)}")
    print(f"Positives:       {int((y == 1).sum())}")
    print(f"Negatives:       {int((y == 0).sum())}")
    print(f"Saved to:        {processed_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
