# This is the main server inference code that runs the model.

# last updated: 5-April-2026
# updated by: Biswaprakash Nayak
# changes made: added logging


# imports 
# torch for the model and inference
# numpy for array manipulation
# pathlib for file path handling
# dataclasses for converting Candidate objects to dicts for easier output formatting
# logging for logging progress and information during inference
# time for measuring inference time
# ------------------------------------
# below are imports from other files in this project
# candidates for the Candidate class definition
# data_ingestion for getting time and flux arrays from the lightcurve data
# preprocessing for the bls algorithm and folding the lightcurve data into bins

# from __future__ import annotations  # can help with older Python versions, uncomment if needed

# imported external libraries
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import logging
import time as timer
from typing import Callable
# imported internal modules
from candidates import Candidate
from data_ingestion import get_time_flux
from preprocessing import bls_topk, fold_to_bins

# set up logging for this module
LOG = logging.getLogger("cosmikai.model_inference")

# this function checks if two periods are related by being harmonics or subharmonics of each other, which helps in filtering out duplicate candidates that are just harmonics of the same signal
# inputs:
# p1: the first period to compare
# p2: the second period to compare
# rel_tol: the relative tolerance to consider for determining if the periods are related 
# output:
# a boolean indicating whether the two periods are related (True) or not (False)
def _periods_are_related(p1: float, p2: float, rel_tol: float = 0.04) -> bool:
    # checks if either period is non-positive, which is invalid for periods, and returns False in that case
    if p1 <= 0 or p2 <= 0:
        return False
    # calculates the ratio of the two periods and checks if it is close to an integer value 
    ratio = max(p1, p2) / min(p1, p2)
    # gets the absolute value of the difference
    for n in (1.0, 1.5, 2.0, 2.5, 3.0):
        if abs(ratio - n) <= rel_tol * n:
            return True
    return False

# this function computes a dynamic cutoff score for determining which candidates are considered 
# inputs:
# sorted_scores: a list of scores for the candidates, sorted in descending order
# threshold: the base score threshold to consider for detection
# cutoff_mode: the mode to use for computing the cutoff, either "relative" or "elbow"
# confidence_drop_fraction: the fraction by which the score can drop from the best score to still be considered a detection
# output:
# a tuple containing the computed cutoff score, the source of the cutoff (for logging), and the index of the elbow if elbow mode is used
def _compute_dynamic_cutoff(
    sorted_scores: list[float],
    threshold: float,
    cutoff_mode: str,
    confidence_drop_fraction: float,
) -> tuple[float, str, int | None]:
    # if there are no scores, return the threshold as the cutoff and indicate that the source is "empty-scores"
    if not sorted_scores:
        return float(threshold), "empty-scores", None
    # the best score is the first one in the sorted list
    best_score = float(sorted_scores[0])
    # the relative cutoff is calculated as the maximum of the threshold and the best score multiplied by (1 - confidence_drop_fraction)
    rel_cutoff = max(float(threshold), best_score * (1.0 - float(confidence_drop_fraction)))
    mode = (cutoff_mode or "relative").lower().strip()
    # if the mode is not "elbow", return the relative cutoff and indicate that the source is "relative"
    if mode != "elbow":
        return rel_cutoff, "relative", None
    if len(sorted_scores) < 3:
        return rel_cutoff, "elbow-fallback-short", None
    # compute the gaps between consecutive scores to find the elbow point
    gaps = [float(sorted_scores[i] - sorted_scores[i + 1]) for i in range(len(sorted_scores) - 1)]
    elbow_idx = int(np.argmax(gaps))
    elbow_gap = gaps[elbow_idx]
    # If the score drop is too small, elbow is ambiguous; fall back to relative mode.
    if elbow_gap < 0.015:
        return rel_cutoff, "elbow-fallback-weak-gap", None
    upper = float(sorted_scores[elbow_idx])
    lower = float(sorted_scores[elbow_idx + 1])
    elbow_cutoff = max(float(threshold), 0.5 * (upper + lower))
    return elbow_cutoff, f"elbow-gap@{elbow_idx}:{elbow_gap:.4f}", elbow_idx

# path to model weights
DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parent / "model" / "best_model.pt"
)

# it helps fix the older models not being compatitble by changing the dictionary to standardize it
# input:
# raw_state: the raw state dictionary loaded from the checkpoint
# output:
# a normalized state dictionary that can be loaded into the model regardless of the training version
def _normalize_checkpoint_state_dict(raw_state: dict) -> dict:
    state = raw_state
    # some older checkpoints might have the state dict nested under "state_dict" or "model_state_dict", this checks for both
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    elif "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
        state = state["model_state_dict"]
    # this remaps the keys in the dictionary
    remapped = {}
    legacy_to_current_prefixes = {
        "conv.0": "feature_extractor.0",
        "conv.2": "feature_extractor.3",
        "conv.4": "feature_extractor.6",
        "fc.1": "classifier.1",
        "fc.4": "classifier.4",
    }
    # it iterates through the state dictionary and remaps the keys based on the legacy prefixes to the current model architecture prefixes
    for key, value in state.items():
        new_key = key
        for legacy_prefix, current_prefix in legacy_to_current_prefixes.items():
            if key.startswith(f"{legacy_prefix}."):
                new_key = key.replace(legacy_prefix, current_prefix, 1)
                break
        remapped[new_key] = value
    # returns the remapped state dictionary that can be loaded into the model
    return remapped

# checks the device for cuda availability and returns the appropriate torch device
def resolve_torch_device(preferred_device: str | None = None) -> torch.device:
    # If a preferred device is specified, attempt to use it
    if preferred_device:
        # Validate the preferred device string
        dev = torch.device(preferred_device)
        # If CUDA is not there, raise an error
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device was requested but is not available.")
        return dev
    # returns the device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# the main model class definition for the CNN architecture used for transit detection
# copied from model.py (by pranav)
# ask him for any details
class TransitCNN(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        features = self.feature_extractor(x)
        return self.classifier(features).squeeze(1)

# loads the model from the disk and preps for inference
# inputs:
# weights_path: the path to the model weights (default is the constant defined above)
# device: the device to run inference on (default is None, which means it will automatically choose cuda if available)
# output:
# model: the loaded TransitCNN model with the weights loaded and set to eval mode
# resolved_device: the torch device that will be used for inference
def load_trained_model(
    # the model path (the constant)
    weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
    # the device used for inference (cpu or gpu)
    device: str | torch.device | None = None,
    # converts it to a tuple
) -> tuple[TransitCNN, torch.device]:    
    # checkst the device
    resolved_device = resolve_torch_device(str(device) if device is not None else None)
    # loads the model architecture
    model = TransitCNN()
    # loads the model weights from the specified path to the specified device into a state dictionary
    state_dict = torch.load(Path(weights_path), map_location=resolved_device, weights_only=True)
    state_dict = _normalize_checkpoint_state_dict(state_dict)
    # loads the state dictionary into the model
    model.load_state_dict(state_dict)
    # moves the model to the resolved device
    model.to(resolved_device)
    # puts it in evaluation mode 
    model.eval()
    # returns the model and the device it will run on
    return model, resolved_device



# this is the function to convert the BLS candidates into a matrix of folded lightcurves that can be fed into the model for inference
# inputs:
# time: the time array of the lightcurve data
# flux: the flux array of the lightcurve data
# candidates: a list of Candidate objects that are the output of the BLS algorithm
# nbins: the number of bins to fold the lightcurve data into (default 512)
# output:
# X: a numpy array of shape (N, nbins) where N is the number of candidates
# folded_curves: list of folded arrays for each candidate (for visualization)
def build_candidate_matrix(
    time: np.ndarray,
    flux: np.ndarray,
    candidates: list[Candidate],
    nbins: int = 512,
    use_gpu: bool = False,
    device: torch.device | None = None,
    progress_callback: Callable[[str, int, str], None] | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    # checks if there are any candidates, if not raises an error
    if not candidates:
        raise ValueError("No BLS candidates were provided.")
    # for each candidate, it folds the lightcurve data into the bins
    # then it returns a numpy array with all the converted candidates
    total_candidates = len(candidates)
    next_progress_pct = 10
    LOG.info("Folding progress: 0%% (0/%s candidates)", total_candidates)
    rows = []
    folded_curves = []
    for idx, cand in enumerate(candidates, start=1):
        folded = fold_to_bins(
            time,
            flux,
            cand.period,
            cand.t0,
            nbins=nbins,
            use_gpu=use_gpu,
            device=device,
            progress_callback=progress_callback,
        )
        rows.append(folded)
        folded_curves.append(folded)
        progress_pct = int((idx * 100) / total_candidates)
        if progress_pct >= next_progress_pct or idx == total_candidates:
            LOG.info("Folding progress: %s%% (%s/%s candidates)", progress_pct, idx, total_candidates)
            if progress_callback is not None:
                progress_callback("folding", progress_pct, f"folded {idx}/{total_candidates} candidates")
            while next_progress_pct <= progress_pct:
                next_progress_pct += 10
    # stacks the folded lightcurves into a 2D array and converts to float32 for the model
    X = np.stack(rows, axis=0).astype(np.float32)
    # checks for infnite matrix
    if not np.isfinite(X).all():
        raise ValueError("Candidate matrix contains NaN/Inf after preprocessing.")
    # returns the final candidate matrix and list of folded curves
    return X, folded_curves

# this scores the candidates
# inputs:
# model: the loaded TransitCNN model
# X: the candidate matrix of shape (N, nbins) that is the output of the previous function
# device: the torch device to run inference on
# output:
# scores: a numpy array of shape (N,) with the sigmoid scores for each candidate
def score_candidates(model: TransitCNN, X: np.ndarray, device: torch.device) -> np.ndarray:
    # converts the candidate matrix to a torch tensor and moves it to the device
    x_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    # runs inference
    with torch.no_grad():
        logits = model(x_t)
        probs = torch.sigmoid(logits)
    # converts the output probabilities to a numpy array and returns it
    return probs.cpu().numpy().astype(np.float32)

# this is the main function that runs the entire inference pipeline
# inputs:
# target_name: the name of the target star to run inference on
# mission: the name of the mission 
# author: the name of the data author - optional
# threshold: the score threshold to determine if a transit is detected 
# k_candidates: the number of top BLS candidates to consider for scoring 
# model_weights_path: the path to the model weights 
# device: the device to run inference on 
# output:
# a list of dictionaries with inference results for each detected candidate (up to 5),
# sorted by score descending
def predict_star_transit(
    target_name: str,
    mission: str,
    author: str = "None",
    threshold: float = 0.5,
    k_candidates: int = 15,
    cutoff_mode: str = "relative",
    confidence_drop_fraction: float = 0.10,
    elbow_plus_extra: int = 1,
    model_weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
    device: str | None = None,
    progress_callback: Callable[[str, int, str], None] | None = None,
) -> list[dict]:
    t_total = timer.time()
    LOG.info(f"=== INFERENCE START: {target_name} on {mission} ===")
    if progress_callback is not None:
        progress_callback("pipeline", 0, "inference started")
    
    # Step 1: Load model
    LOG.info("Step 1/5: Loading model...")
    t1 = timer.time()
    model, resolved_device = load_trained_model(model_weights_path, device=device)
    LOG.info(f"  Model loaded on {resolved_device} in {timer.time()-t1:.2f}s")
    if progress_callback is not None:
        progress_callback("model", 100, f"loaded on {resolved_device}")
    
    # Step 2: Download lightcurve
    LOG.info("Step 2/5: Downloading lightcurve from MAST...")
    t2 = timer.time()
    time, flux = get_time_flux(
        target_name=target_name,
        mission=mission,
        author=author,
        download_all=True,
        progress_callback=progress_callback,
    )
    LOG.info(f"  Downloaded {len(time)} datapoints in {timer.time()-t2:.2f}s")
    
    # Step 3: Run BLS
    LOG.info(f"Step 3/5: Running BLS to find top {k_candidates} candidates...")
    t3 = timer.time()
    candidates = bls_topk(
        time,
        flux,
        k=k_candidates,
        use_gpu=resolved_device.type == "cuda",
        progress_callback=progress_callback,
    )
    LOG.info(f"  BLS completed in {timer.time()-t3:.2f}s, found {len(candidates)} candidates")
    
    # Step 4: Build candidate matrix and score
    LOG.info("Step 4/5: Folding lightcurves and scoring candidates...")
    t4 = timer.time()
    X, folded_curves = build_candidate_matrix(
        time,
        flux,
        candidates,
        nbins=512,
        use_gpu=resolved_device.type == "cuda",
        device=resolved_device,
        progress_callback=progress_callback,
    )
    LOG.info(f"  Built candidate matrix {X.shape} in {timer.time()-t4:.2f}s")
    LOG.info("Step 4/5 (continued): Running model inference on GPU...")
    t4b = timer.time()
    scores = score_candidates(model, X, resolved_device)
    LOG.info(f"  Model inference completed in {timer.time()-t4b:.2f}s")
    if progress_callback is not None:
        progress_callback("model_inference", 100, f"scored {len(scores)} candidates")
    
    # Step 5: Filter and format results
    LOG.info("Step 5/5: Filtering results and preparing output...")
    
    # Create list of (index, score) tuples and sort by score descending
    scored_candidates = list(enumerate(scores))
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    sorted_scores = [float(s) for _, s in scored_candidates]
    dynamic_cutoff, cutoff_source, elbow_idx = _compute_dynamic_cutoff(
        sorted_scores=sorted_scores,
        threshold=float(threshold),
        cutoff_mode=cutoff_mode,
        confidence_drop_fraction=float(confidence_drop_fraction),
    )
    best_score = sorted_scores[0] if sorted_scores else 0.0
    LOG.info(
        "Dynamic confidence cutoff: mode=%s source=%s best_score=%.4f cutoff=%.4f fraction=%.2f",
        cutoff_mode,
        cutoff_source,
        best_score,
        dynamic_cutoff,
        confidence_drop_fraction,
    )
    
    # Filter and keep top detections that pass threshold, with period de-duplication.
    results = []
    max_detections = 5
    picked_periods: list[float] = []
    # If elbow mode is active and the detector found an elbow, optionally include
    # a small number of extra candidates beyond the elbow before applying cutoff.
    min_results_required = 1
    if cutoff_mode.lower().strip() == "elbow" and elbow_idx is not None:
        min_results_required = min(max_detections, max(1, (elbow_idx + 1) + max(0, int(elbow_plus_extra))))
        LOG.info(
            "Elbow retention rule: elbow_idx=%s plus_extra=%s => min_results_required=%s",
            elbow_idx,
            elbow_plus_extra,
            min_results_required,
        )

    for idx, score in scored_candidates:
        candidate = candidates[idx]

        # Always keep the highest-scoring candidate as baseline output.
        if len(results) == 0:
            folded = folded_curves[idx]
            results.append({
                "target_name": target_name,
                "mission": mission,
                "threshold": float(threshold),
                "best_score": float(score),
                "verdict": "TRANSIT_DETECTED" if score >= threshold else "NO_TRANSIT",
                "best_candidate": asdict(candidate),
                "num_candidates": len(candidates),
                "device": str(resolved_device),
                "all_scores": [float(s) for s in scores],
                "folded_lightcurve": [float(x) for x in folded],
            })
            picked_periods.append(float(candidate.period))
            if len(results) >= max_detections:
                break
            continue

        # Additional planets must remain near the top score and not be harmonic duplicates.
        if score < dynamic_cutoff and len(results) >= min_results_required:
            break
        if any(_periods_are_related(float(candidate.period), p) for p in picked_periods):
            continue

        folded = folded_curves[idx]
        results.append({
            "target_name": target_name,
            "mission": mission,
            "threshold": float(threshold),
            "best_score": float(score),
            "verdict": "TRANSIT_DETECTED",
            "best_candidate": asdict(candidate),
            "num_candidates": len(candidates),
            "device": str(resolved_device),
            "all_scores": [float(s) for s in scores],
            "folded_lightcurve": [float(x) for x in folded],
        })
        picked_periods.append(float(candidate.period))
        if len(results) >= max_detections:
            break
    
    # If nothing passed threshold, still return the best one
    if not results:
        best_idx = int(np.argmax(scores))
        best_candidate = candidates[best_idx]
        best_folded = folded_curves[best_idx]
        results = [{
            "target_name": target_name,
            "mission": mission,
            "threshold": float(threshold),
            "best_score": float(scores[best_idx]),
            "verdict": "NO_TRANSIT",
            "best_candidate": asdict(best_candidate),
            "num_candidates": len(candidates),
            "device": str(resolved_device),
            "all_scores": [float(s) for s in scores],
            "folded_lightcurve": [float(x) for x in best_folded],
        }]
    
    total_time = timer.time() - t_total
    LOG.info(f"  Results: {len(results)} detections")
    LOG.info(f"=== INFERENCE COMPLETE in {total_time:.2f}s ===")
    if progress_callback is not None:
        progress_callback("pipeline", 100, "inference complete")
    
    return results