"""Open-loop step-test system identification — partial port of ``OL_sysid.m``.

``OL_sysid.m`` loads eight step-test .mat files from João's OneDrive path
(``step_test_clean/foul{1,2}_{up,down}.mat``) which are NOT part of this
repository, plots them, computes static gains and gain ratios, and then —
after an early ``return`` — fits second-order-plus-dead-time models with
the MATLAB ``procest``/``idproc`` toolboxes.

What the port provides:

* :func:`static_gains` — the arithmetic of OL_sysid.m lines 64-85
  (steady-state gains and clean/fouled ratios), given the recorded
  steady-state values;
* :func:`recorded_reference_values` — the hard numbers embedded in the
  script (TR0 = 366.7929 K clean / 366.48 K fouled, Qcin0 = 1.5e-3 /
  2.119e-3 m3/s, dt = 5 s, steps 1.5e-5 / 3e-5 m3/s);
* :func:`fit_sopdt` — a SciPy least-squares fit of a
  second-order-plus-dead-time-plus-underdamping model
  ``K*wc^2/(s^2 + 2*zeta*wc*s + wc^2) * exp(-L*s)`` as the documented
  substitute for ``idproc('P2DU')`` + ``procest`` (risk recorded in
  PORT_NOTES.md: different objective/optimiser than MATLAB's prediction-
  error method, so fitted parameters may differ from MATLAB's for the
  same data).

The full script cannot be reproduced bit-for-bit because its input data
is not in the repository; verdict: TRANSCRIBED (arithmetic) with a
RECONSTRUCTED model-fit substitute.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


RECORDED = {
    "TR0_clean": 366.7929,        # K
    "TR0_foul": 366.48,           # K (Rf = 3e-4)
    "Qcin0_clean": 1.5e-3,        # m3/s
    "Qcin0_foul": 2.119e-3,       # m3/s
    "deltat": 5.0,                # s
    "delta_u1": 1.5e-5,           # m3/s (1% step)
    "delta_u2": 3.0e-5,           # m3/s (2% step)
    "delta_ufoul": 2.119e-3 * 0.01,
}


def static_gains(TR_end_clean_up: float, TR_end_clean_down: float,
                 TR_end_foul_up: float, TR_end_foul_down: float,
                 TR0_clean: float | None = None,
                 TR0_foul: float | None = None) -> dict:
    """OL_sysid.m lines 64-85: steady-state gains for the ±1% Qcin steps,
    clean and fouled, and their ratios."""
    TR0_c = RECORDED["TR0_clean"] if TR0_clean is None else TR0_clean
    TR0_f = RECORDED["TR0_foul"] if TR0_foul is None else TR0_foul
    du = RECORDED["delta_u1"]
    duf = RECORDED["delta_ufoul"]
    Kp_clean1_up = (TR_end_clean_up - TR0_c) / (-du)
    Kp_clean1_down = (TR_end_clean_down - TR0_c) / du
    Kp_foul1_up = (TR_end_foul_up - TR0_f) / duf
    Kp_foul1_down = (TR_end_foul_down - TR0_f) / (-duf)
    return {
        "Kp_clean1_up": Kp_clean1_up,
        "Kp_clean1_down": Kp_clean1_down,
        "Kp_foul1_up": Kp_foul1_up,
        "Kp_foul1_down": Kp_foul1_down,
        "Kp_ratio_up_down": Kp_clean1_down / Kp_clean1_up,
        "Kp_ratio_clean_foul_up": Kp_clean1_up / Kp_foul1_up,
        "Kp_ratio_foul_clean_up": Kp_foul1_up / Kp_clean1_up,
        "Kp_ratio_clean_foul_down": Kp_clean1_down / Kp_foul1_down,
        "Kp_ratio_foul_clean_down": Kp_foul1_down / Kp_clean1_down,
    }


def _sopdt_step_response(theta, t):
    K, wn, zeta, L = theta
    # second-order step response (zero initial state), delayed by L
    tt = np.maximum(t - L, 0.0)
    if abs(zeta - 1.0) < 1e-9:
        y = 1.0 - (1.0 + wn * tt) * np.exp(-wn * tt)
    elif zeta < 1.0:
        wd = wn * np.sqrt(1.0 - zeta ** 2)
        phi = np.arccos(zeta)
        y = 1.0 - np.exp(-zeta * wn * tt) * (
            np.cos(wd * tt) + (zeta / np.sqrt(1 - zeta ** 2)) * np.sin(wd * tt))
        y = np.where(tt == 0.0, 0.0, y)
    else:
        s = np.sqrt(zeta ** 2 - 1.0)
        r1 = -wn * (zeta - s)
        r2 = -wn * (zeta + s)
        y = 1.0 + (r2 * np.exp(r1 * tt) - r1 * np.exp(r2 * tt)) / (r1 - r2)
        y = np.where(tt == 0.0, 0.0, y)
    return K * y


def fit_sopdt(t: np.ndarray, y: np.ndarray, u_step: float,
              theta0=(1.0, 0.01, 0.9, 30.0)):
    """Fit ``K, wn, zeta, L`` of a delayed second-order model to a step
    response (documented substitute for MATLAB ``idproc('P2DU')`` +
    ``procest``). ``y`` is the output deviation, ``u_step`` the input step
    (the fitted K is the gain per unit u)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    def resid(theta):
        return _sopdt_step_response(theta, t) * u_step - y

    sol = least_squares(resid, theta0, method="lm", max_nfev=20000)
    K, wn, zeta, L = sol.x
    return dict(K=float(K), wn=float(wn), zeta=float(zeta), L=float(L),
                tau1=1.0 / float(wn), cost=float(sol.cost))
