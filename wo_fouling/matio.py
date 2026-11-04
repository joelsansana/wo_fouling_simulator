"""MAT-file readers for the upstream reference data.

``sim_results.mat`` (model_simulink_closed_loop/) is a MATLAB v7.3 (HDF5)
file containing a single serialised ``timeseries`` object ``sim_data``
with 11 channels on a 5 s grid (1080 samples, 0..5395 s — a 90 min
closed-loop run). The channel order (identified from the recorded
magnitudes against the harness wiring, 2026-11-03):

    1 FA, 2 FB, 3 TA, 4 TB, 5 Tcin, 6 Tcout, 7 LR, 8 TR, 9 QR,
    10 Qcin, 11 DeltaP

Requires ``h5py`` (test-only optional dependency).
``model_ic.mat`` is v7 (scipy-readable): ``xFinal`` = 209-vector
``[L, ma..mp, mt, Tr, Tc(1..200)]`` — a 200-cell golden snapshot.
"""

from __future__ import annotations

import numpy as np

CHANNEL_NAMES = ["FA", "FB", "TA", "TB", "Tcin", "Tcout", "LR", "TR",
                 "QR", "Qcin", "DeltaP"]


def load_sim_results(path: str) -> dict:
    """Load ``sim_results.mat`` into ``{name: array}`` + ``Time``."""
    import h5py
    with h5py.File(path, "r") as f:
        refs = f["#refs#"]
        t = np.asarray(refs["p"][:]).ravel()          # Time
        z = np.asarray(refs["z"][:])                  # (1080, 11) Data
    out = {"Time": t}
    for i, name in enumerate(CHANNEL_NAMES):
        out[name] = z[:, i]
    return out


def load_model_ic(path: str) -> dict:
    """Load ``model_ic.mat`` (v7) -> dict with L, masses..., mt, Tr, Tc."""
    from scipy.io import loadmat
    d = loadmat(path)
    xf = np.asarray(d["xFinal"], dtype=float).ravel()
    n = len(xf) - 9
    return {
        "L": xf[0],
        "masses": xf[1:7],
        "mt": xf[7],
        "Tr": xf[8],
        "Tc": xf[9:],
        "n": n,
        "xFinal": xf,
    }
