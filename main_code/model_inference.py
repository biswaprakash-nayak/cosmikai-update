# This is the main server inference code that runs the model.

# last updated: 28-March-2026
# updated by: Biswaprakash Nayak
# changes made: made this code and added all the neccesary functions.


# imports 
# torch for the model and inference
# numpy for array manipulation
# pathlib for file path handling
# dataclasses for converting Candidate objects to dicts for easier output formatting
#------------------------------------
# below are imports from other files in this project
# candidates for the Candidate class definition
# data_ingestion for getting time and flux arrays from the lightcurve data
# preprocessing for the bls algorithm and folding the lightcurve data into bins

#from __future__ import annotations  # can help with older Python versions, uncomment if needed

# imported external libraries
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
# imported internal modules
from candidates import Candidate
from data_ingestion import get_time_flux
from preprocessing import bls_topk, fold_to_bins

# path to model weights
DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parent / "model" / "best_model.pt"
)

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
def build_candidate_matrix(
    time: np.ndarray,
    flux: np.ndarray,
    candidates: list[Candidate],
    nbins: int = 512,
) -> np.ndarray:
    # checks if there are any candidates, if not raises an error
    if not candidates:
        raise ValueError("No BLS candidates were provided.")
    # for each candidate, it folds the lightcurve data into the bins
    # then it returns a numpy array with all the converted candidates
    rows = []
    for cand in candidates:
        folded = fold_to_bins(time, flux, cand.period, cand.t0, nbins=nbins)
        rows.append(folded)
    # stacks the folded lightcurves into a 2D array and converts to float32 for the model
    X = np.stack(rows, axis=0).astype(np.float32)
    # checks for infnite matrix
    if not np.isfinite(X).all():
        raise ValueError("Candidate matrix contains NaN/Inf after preprocessing.")
    # returns the final candidate matrix
    return X

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
# a dictionary with the inference results, including the best candidate and its score, the verdict, and other relevant information
def predict_star_transit(
    target_name: str,
    mission: str,
    author: str = "None",
    threshold: float = 0.5,
    k_candidates: int = 15,
    model_weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
    device: str | None = None,
) -> dict:
    # loads the model and resolves the device for inference
    model, resolved_device = load_trained_model(model_weights_path, device=device)
    # gets the time and flux arrays for the target star and mission using the data ingestion function
    time, flux = get_time_flux(
        target_name=target_name,
        mission=mission,
        author=author,
        download_all=False,
    )
    # runs the BLS algorithm to get the top k candidates and builds the candidate matrix for the model
    candidates = bls_topk(time, flux, k=k_candidates)
    # builds the candidate matrix for the model using the previous function
    X = build_candidate_matrix(time, flux, candidates, nbins=512)
    # scores the candidates using the model and the previous function
    scores = score_candidates(model, X, resolved_device)
    # finds the best candidate and its score
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_candidate = candidates[best_idx]
    # returns the inference results dict
    return {
        "target_name": target_name,
        "mission": mission,
        "threshold": float(threshold),
        "best_score": best_score,
        "verdict": "TRANSIT_DETECTED" if best_score >= threshold else "NO_TRANSIT",
        "best_candidate": asdict(best_candidate),
        "num_candidates": len(candidates),
        "device": str(resolved_device),
        "all_scores": [float(s) for s in scores],
    }