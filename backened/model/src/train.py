import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import average_precision_score
from pathlib import Path

from src.model import TransitCNN


def main():
    device = torch.device("cpu")
    print("=" * 60)
    print("EXOPLANET TRANSIT DETECTOR - TRAINING")
    print("=" * 60)

    # Load dataset
    processed = Path("data/processed")
    X = np.load(processed / "X.npy")
    y = np.load(processed / "y.npy").astype(np.float32)
    meta = json.loads((processed / "meta.json").read_text())

    print(f"[INFO] Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"[INFO] Positives: {int(y.sum())}, Negatives: {int((1-y).sum())}")

    # Extract star IDs for group splitting (no data leakage)
    star_ids = np.array([m["star"] for m in meta])
    unique_stars = len(set(star_ids))
    print(f"[INFO] Unique stars: {unique_stars}")

    # Expand dims for Conv1d: (N, 512) -> (N, 1, 512)
    X = np.expand_dims(X, axis=1)

    # Group split by star
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups=star_ids))

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print(f"[INFO] Train: {len(X_train)} samples | Val: {len(X_val)} samples")
    print(f"[INFO] Train positives: {int(y_train.sum())} | Train negatives: {int((1-y_train).sum())}")

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    )

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    # Model
    model = TransitCNN()
    model.to(device)

    # Weighted loss to handle class imbalance
    pos_frac = float(y_train.sum()) / len(y_train)
    pos_weight = torch.tensor([(1 - pos_frac) / max(pos_frac, 1e-6)], device=device)
    print(f"[INFO] Class weight: pos_weight={pos_weight.item():.3f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Create models directory
    Path("models").mkdir(exist_ok=True)

    best_ap = 0
    patience = 8
    no_improve = 0

    print(f"\n[INFO] Starting training for 50 epochs (early stopping patience: {patience})")
    print("-" * 60)

    for epoch in range(50):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(device))
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.extend(probs.flatten())
                targets.extend(yb.numpy().flatten())

        ap = average_precision_score(targets, preds)
        print(f"Epoch {epoch+1:02d} | Loss {total_loss:.4f} | Val AUPRC {ap:.4f}", end="")

        if ap > best_ap:
            best_ap = ap
            no_improve = 0
            torch.save(model.state_dict(), "models/best_model.pt")
            print(" *saved*")
        else:
            no_improve += 1
            print()

        if no_improve >= patience:
            print(f"\n[INFO] Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    print("-" * 60)
    print(f"[RESULT] Best Validation AUPRC: {best_ap:.4f}")
    print(f"[INFO] Model saved to: models/best_model.pt")

    # Save training summary
    summary = {
        "best_auprc": float(best_ap),
        "total_samples": len(y),
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "unique_stars": unique_stars,
        "epochs_trained": epoch + 1,
        "pos_weight": float(pos_weight.item()),
    }
    (Path("models") / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[INFO] Training summary saved to: models/training_summary.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
