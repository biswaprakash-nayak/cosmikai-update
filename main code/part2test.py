from data_ingestion import get_time_flux
from preprocessing import bls_topk

t, f = get_time_flux("Pi Mensae", "TESS", author="SPOC", download_all=False)

cands = bls_topk(t, f, k=10)
for i, c in enumerate(cands, 1):
    print(i, "P=", c.period, "dur=", c.duration, "depth=", c.depth, "power=", c.power)


