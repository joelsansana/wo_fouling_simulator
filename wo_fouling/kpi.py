"""KPI / CPI pipeline — port of ``CPI_calc.m``.

Computes:

* the LMTD-based fouling KPI from measured channels
  ``KPI = dTR / dTml`` with
  ``dTR = TR - 0.5*(TA+TB)``,
  ``dT1 = 0.5*(TA+TB) - Tcin``,
  ``dT2 = TR - Tcout``,
  ``dTml = (dT1 - dT2)/log(dT1/dT2)``;
* the EWMA filter ``KPI_filt`` (alpha = 0.01, seed value 6 or 7 upstream —
  parameterised here);
* max-scaling of Rf / KPI / KPI_filt and the Spearman rank correlations
  ``rho(Rf, KPI_filt)`` and ``rho(CPI, KPI_filt)``;
* controller-performance indices against a historical benchmark window:
  ``CPI_1 = IAE_bench/IAE_window`` and ``CPI_2 = ISEu_bench/ISEu_window``
  computed over consecutive ``time_bench`` windows, then restructured to
  zero-order-hold at the 30 s sample grid with a leading 1 (as saved to
  the CL CSVs upstream);
* the fouling-threshold detector (indices where KPI_filt departs the
  final value by more than ±thresh).

Upstream consumed 30 s-sampled tables (``CL_*.csv``) produced by the
Simulink runs; ``sim_results.mat`` is the same data in timeseries form.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def lmtd_kpi(TR, TA, TB, Tcin, Tcout) -> np.ndarray:
    """KPI = dTR / dTml (CPI_calc.m lines 69-74 / 201-207)."""
    TR = np.asarray(TR, dtype=float)
    TA = np.asarray(TA, dtype=float)
    TB = np.asarray(TB, dtype=float)
    Tcin = np.asarray(Tcin, dtype=float)
    Tcout = np.asarray(Tcout, dtype=float)
    deltaTR = TR - 0.5 * (TA + TB)
    deltaT1 = 0.5 * (TA + TB) - Tcin
    deltaT2 = TR - Tcout
    deltaTml = (deltaT1 - deltaT2) / np.log(deltaT1 / deltaT2)
    return deltaTR / deltaTml


def ewma_filter(kpi: np.ndarray, alpha: float = 0.01,
                seed: float = 6.0) -> np.ndarray:
    """EWMA filter (CPI_calc.m lines 77-82 / 211-217): KPI_filt(1) = seed
    (6 or 7 upstream); KPI_filt(k) = alpha*KPI(k) + (1-alpha)*KPI_filt(k-1)."""
    kpi = np.asarray(kpi, dtype=float)
    out = np.empty_like(kpi)
    out[0] = seed
    for k in range(1, len(kpi)):
        out[k] = alpha * kpi[k] + (1.0 - alpha) * out[k - 1]
    return out


def scale_by_max(x: np.ndarray) -> np.ndarray:
    """x/max(x) — MATLAB parity: all-zero input gives NaN (0/0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(x, dtype=float) / np.max(np.asarray(x, dtype=float))


def spearman(x, y) -> float:
    return float(stats.spearmanr(np.asarray(x, dtype=float).ravel(),
                                 np.asarray(y, dtype=float).ravel()).statistic)


def controller_performance_indices(TR, TR_sp, Qcin, deltat: float,
                                   time_bench: float = 4 * 3600.0,
                                   u0: float = 0.0):
    """CPI_1 (IAE-based) and CPI_2 (ISEu-based) over benchmark windows.

    Port of CPI_calc.m lines 285-337 (windowed evaluation) plus the
    ZOH restructuring to the 30 s grid (lines 398-407).

    Returns ``(CPI_1_zoh, CPI_2_zoh, info)`` where the CPI arrays start
    with 1 (benchmark window) as saved upstream, and ``info`` carries the
    raw windowed values.
    """
    TR = np.asarray(TR, dtype=float)
    TR_sp = np.asarray(TR_sp, dtype=float)
    Qcin = np.asarray(Qcin, dtype=float)

    error = TR_sp - TR
    N = int(round(time_bench / deltat))
    Ntotal = len(TR) - 1

    if Ntotal < N:
        raise ValueError("not enough samples for one benchmark window")

    TR_bench = TR[:N]
    u_bench = Qcin[:N]
    error_bench = error[:N]
    IAE_bench = float(np.sum(np.abs(error_bench)) * deltat)
    ISEu_bench = float(np.sum((u_bench - u0) ** 2) * deltat)

    # du (move penalty, computed but only used for Ju upstream)
    du_bench = np.zeros(len(Qcin))
    du_bench[1:N] = np.abs(np.diff(u_bench[:N]))
    Ju_bench = float(np.sum(du_bench))

    Timetotal = deltat * (len(TR) - 1)
    Step = int(Timetotal / time_bench)
    IAE = np.zeros(max(Step - 1, 0))
    ISEu = np.zeros(max(Step - 1, 0))
    CPI_1 = np.zeros(max(Step - 1, 0))
    CPI_2 = np.zeros(max(Step - 1, 0))
    # MATLAB `for i = 1:Step-1`: windows AFTER the benchmark window
    for i in range(1, Step):
        init = N * i
        end = N * i + N
        e1 = error[init:end]
        uu = Qcin[init:end]
        IAE[i - 1] = np.sum(np.abs(e1)) * deltat
        ISEu[i - 1] = np.sum((uu - u0) ** 2) * deltat
        # plain division, MATLAB parity: 0/0 -> NaN, x/0 -> inf
        with np.errstate(divide="ignore", invalid="ignore"):
            CPI_1[i - 1] = IAE_bench / IAE[i - 1]
            CPI_2[i - 1] = ISEu_bench / ISEu[i - 1]

    CPI_1 = np.concatenate([[1.0], CPI_1])
    CPI_2 = np.concatenate([[1.0], CPI_2])

    # ZOH restructure to the 30 s grid (CPI_calc.m lines 398-407):
    # y = repmat(CPI', time_bench, 1); y = y(:); y = y(1:30:end)
    def zoh(cpi):
        y = np.repeat(cpi[:, None], int(time_bench), axis=1).ravel()
        y = y[::30]
        return np.concatenate([[1.0], y])

    info = dict(IAE_bench=IAE_bench, ISEu_bench=ISEu_bench, Ju_bench=Ju_bench,
                N=N, Step=Step, IAE=IAE, ISEu=ISEu)
    return zoh(CPI_1), zoh(CPI_2), info


def fouling_threshold_indices(kpi_filt: np.ndarray, thresh: float = 0.1):
    """Indices where KPI_filt has departed the final value by more than
    ``thresh`` (CPI_calc.m lines 223-227)."""
    kpi_filt = np.asarray(kpi_filt, dtype=float)
    return np.where((kpi_filt - kpi_filt[-1]) <= thresh * kpi_filt[-1])[0]


def kpi_pipeline(data: dict, ewma_seed: float = 6.0,
                 time_bench: float = 4 * 3600.0):
    """Full offline pipeline over a measured channel dict with keys
    ``Time, FA, FB, TA, TB, Tcin, Tcout, LR, TR, QR, Qcin, DeltaP, Rf,
    TR_sp`` (30 s grid). Returns a dict with KPI, KPI_filt, scaled series,
    Spearman correlations and CPI arrays."""
    TR, TA, TB = data["TR"], data["TA"], data["TB"]
    Tcin, Tcout = data["Tcin"], data["Tcout"]
    t = np.asarray(data["Time"], dtype=float)
    deltat = float(t[1] - t[0])

    kpi = lmtd_kpi(TR, TA, TB, Tcin, Tcout)
    kpi_filt = ewma_filter(kpi, alpha=0.01, seed=ewma_seed)

    out = dict(KPI=kpi, KPI_filt=kpi_filt)
    if "Rf" in data and data["Rf"] is not None:
        rf_scaled = scale_by_max(data["Rf"])
        out["Rf_scaled"] = rf_scaled
        out["rho_Rf_KPI"] = spearman(rf_scaled, scale_by_max(kpi_filt))
    if "CPI" in data and data["CPI"] is not None:
        out["rho_CPI_KPI"] = spearman(data["CPI"], scale_by_max(kpi_filt))
    if "TR_sp" in data and data["TR_sp"] is not None:
        cpi1, cpi2, info = controller_performance_indices(
            data["TR"], data["TR_sp"], data["Qcin"], deltat, time_bench)
        out["CPI_1"] = cpi1
        out["CPI_2"] = cpi2
        out["CPI_info"] = info
    out["fouling_idx"] = fouling_threshold_indices(kpi_filt)
    return out
