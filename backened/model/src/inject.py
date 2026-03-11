import numpy as np

def inject_box_transit(phase, flux, depth_ppm, duration_frac, t0=None):
    """
    Simple box-shaped transit injection in phase space.
    depth_ppm: 200..5000
    duration_frac: 0.015..0.10 (fraction of phase)
    """
    depth = depth_ppm * 1e-6
    if t0 is None:
        t0 = np.random.uniform(0, 1)

    # compute phase distance with wrap-around
    d = np.abs((phase - t0 + 0.5) % 1.0 - 0.5)
    in_transit = d < (duration_frac / 2.0)

    injected = flux.copy()
    injected[in_transit] *= (1.0 - depth)
    return injected, t0
