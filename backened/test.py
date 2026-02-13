from data_ingestion import get_time_flux
import numpy as np

t, f = get_time_flux("Pi Mensae", "TESS")
print("N =", len(t))
print("time range:", float(t.min()), "to", float(t.max()))
print("flux median/std:", float(np.median(f)), float(np.std(f)))
print("NaNs:", int(np.isnan(t).sum()), int(np.isnan(f).sum()))
print("Infs:", int(np.isinf(t).sum()), int(np.isinf(f).sum()))

