import numpy as np
from data_ingestion import get_time_flux
from preprocessing import bls_topk  
from preprocessing import fold_to_bins  

t, f = get_time_flux("Pi Mensae", "TESS", author="SPOC", download_all=False)

cands = bls_topk(t, f, k=10, sig_th=0.02)

X = np.stack([fold_to_bins(t, f, c.period, c.t0, nbins=512) for c in cands], axis=0)

print("X shape:", X.shape)
print("finite:", np.isfinite(X).all())
print("min/max:", float(X.min()), float(X.max()))