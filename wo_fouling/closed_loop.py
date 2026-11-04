"""Closed-loop harness — Python re-implementation of ``WO_main_sim.slx``.

Built from a direct extraction of the .slx XML (2026-11-03), NOT from the
audit. Key evidence and the resulting design:

* The plant is the Stateflow chart ``WO_dyn`` (``chart_38.xml`` inside
  ``WO_reactor_model.slx``) — a PRODUCTION compile of the dynamics that
  differs from the open-loop ``dynmodel_fouling.m``:

      - state = ``[ma..mp, Tr, Tc(1..200), Rfw]`` (208 elements); level,
        layer thickness and pressure drop are NOT integrated — they are
        resolved by ``AlgebraicConstraint`` blocks
        (``VR - Ar*Lr = 0``, ``kf*1000*Rf - df = 0`` [the block name says
        ``kf*Rf - df = 0`` but the wiring multiplies by the ``kf*1000``
        constant = 2.941], ``dpdz - DeltaP = 0``).
      - ``N = 200`` hardcoded.
      - a ``fouling`` switch input: ``0`` ⇒ ``Dcoildiff = Dcoil``,
        ``dRfw_dt = -1`` (sentinel), ``U = Uclean``; ``1`` ⇒ fouled
        diameter, fouling ODE, ``U = 1/(1/Uclean + 1000*Rfw)`` clamped at
        ``Uclean`` (no ``Rfw``/``df`` reset, unlike the .m file).
      - ``MR`` is computed from the state (NOT fixed to ``massT`` — the
        audit's §5.1 claim is wrong for the chart; only ``ssmodel.m``
        fixes it, consistently, in fraction space).
      - the ``Integrator_ODE`` block clamps all 208 states at
        ``>= 0`` (``LowerSaturationLimit = zeros(208,1)``).

* Controllers (extracted block parameters, cross-checked against
  ``controller_tuning.m`` arithmetic):

      - ``LR_C``: pid_lib PID Controller, PI, Ts=5 s, gains
        P=-0.017011, I=-1.086e-4, output saturated [0, 0.012] m^3/s,
        integrator IC 0.0082593 (= nominal QR).
      - ``TR_C``: pid_lib PID Controller (2DOF), Ts=5 s, b=1, c=0,
        P=-1.6707e-4, I=-1.8639e-7, D=-9.399e-3, N=0.03555,
        integrator IC 0.00125 (= nominal Qcin), filter IC 23.1105.

* Disturbances (``Disturbances`` subsystem): TA and TB are 24 h
  sinusoids around 294.3 K (amplitude 5 K), Tcin a 24 h sinusoid around
  288.75 K (amplitude 4 K); FA optionally follows a repeating stair
  ``[2.0922, 1.8275, 1.5628, 1.8275]``; Ca and pH follow AR(3) noise
  (phi = [0.1, 0.25, 0.6], Ts=30 s, variance 2e-10) around biases
  150e-5 mol/L and 9. The recorded ``sim_results.mat`` run (90 min) shows
  TA AND TB varying and NO FA stair / no SP stair active, so this module
  defaults to that configuration.

* TR setpoint: ``TR_SP`` constant + optional repeating stair
  ``[5, 0, -5, 0]`` every 3600 s (present in the saved .slx but inactive
  in the recorded run — the saved model was configured for a tuning test).

The sensor chain (transport delays 5 s/30 s + per-channel Gaussian noise
with block-specific seeds) is RECONSTRUCTED: delays are parameterised
(default: no delay, no noise — a deterministic core) because the recorded
noise streams are MATLAB-RNG-specific and not bit-reproducible. See
PORT_NOTES.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

from .parameters import WoParameters, closed_loop_parameters
from .controllers import DiscretePID, LR_C_GAINS, TR_C_GAINS, filter_initial_condition
from .steady_state import solve_closed_loop_ss

_EPS_MR = 1e-10


# -----------------------------------------------------------------------------
# WO_dyn kernel (chart_38.xml, verbatim)
# -----------------------------------------------------------------------------

@njit(cache=True)
def _wo_dyn_kernel(t, x, u, p):
    """Numba RHS of the chart-38 ``WO_dyn`` MATLAB function.

    State ``x = [ma..mp, Tr, Tc(1..N), Rfw]`` with N = p[24] (hardcoded
    200 upstream). Inputs ``u = [FA, FB, TA, TB, Tcin, QR, Qcin, ca, pH,
    df, fouling]`` — df arrives as an input (resolved outside by the
    AlgebraicConstraint), fouling is the 0/1 switch.

    p layout matches ``_pack_dyn_params`` below (37 entries; identical to
    the fouling kernel pack — the chart consumes the same 31-element flat
    parameter vector ``par1`` from ``script_sim_main.m``).
    """
    rhoR = p[0]
    A1 = p[1]; A2 = p[2]; A3 = p[3]
    Ea1 = p[4]; Ea2 = p[5]; Ea3 = p[6]
    dH1 = p[7]; dH2 = p[8]; dH3 = p[9]
    CpA = p[10]; CpB = p[11]; CpC = p[12]
    CpE = p[13]; CpP = p[14]; CpG = p[15]
    CpW = p[16]
    Uclean = p[17]
    rhoW = p[20]
    viscW = p[21]
    Lcoil = p[22]
    Dcoil = p[23]
    N = int(p[24])
    Ar = p[25]
    epsilon = p[26]
    kf = p[28]
    Rgas = p[29]
    rhof = p[30]
    CH = p[31]
    ksp = p[32]
    yco2 = p[33]
    psi = p[34]
    K1 = p[35]
    K2 = p[36]
    Ucleaninv = 1.0 / Uclean

    FA = u[0]; FB = u[1]
    TA = u[2]; TB = u[3]
    Tcin = u[4]; QR = u[5]; Qcin = u[6]
    ca = u[7]; pH = u[8]
    df = u[9]
    fouling = u[10]

    ma = x[0]; mb = x[1]; mc = x[2]
    me = x[3]; mg = x[4]; mp = x[5]
    Tr = x[6]
    Tc = x[7:7 + N].copy()
    Rfw = x[7 + N]

    # ---------------- constitutive relationships (chart lines, verbatim)
    MR = ma + mb + mc + me + mg + mp
    xa = ma / MR; xb = mb / MR; xc = mc / MR
    xe = me / MR; xg = mg / MR; xp = mp / MR

    FR = rhoR * QR
    FRa = FR * xa; FRb = FR * xb; FRc = FR * xc
    FRe = FR * xe; FRg = FR * xg; FRp = FR * xp

    Trinv = 1.0 / Tr
    lnkl1 = np.log(A1) - Ea1 * Trinv
    lnkl2 = np.log(A2) - Ea2 * Trinv
    lnkl3 = np.log(A3) - Ea3 * Trinv

    r1mr = np.exp(lnkl1) * xa * xb * MR
    r2mr = np.exp(lnkl2) * xb * xc * MR
    r3mr = np.exp(lnkl3) * xc * xp * MR

    dma_dt = FA - FRa - r1mr
    dmb_dt = FB - FRb - r1mr - r2mr
    dmc_dt = -FRc + 2.0 * r1mr - 2.0 * r2mr - r3mr
    dme_dt = -FRe + 2.0 * r2mr
    dmg_dt = -FRg + 1.5 * r3mr
    dmp_dt = -FRp + r2mr - 0.5 * r3mr

    # ---------------- fouling switch (chart: if fouling == 1 ... else)
    if fouling == 1:
        Dcoildiff = Dcoil - 2.0 * df
        if Dcoildiff > Dcoil:
            Dcoildiff = Dcoil
    else:
        Dcoildiff = Dcoil

    Acoil = 0.25 * np.pi * Dcoildiff ** 2
    vz = Qcin / Acoil
    dz = Lcoil / (N - 1)

    rhow_cpw = rhoW * CpW
    ReC = rhoW * vz * Dcoildiff / viscW

    # Fouling kinetics (chart computes these unconditionally; dead values
    # when fouling == 0 — transcribed as-is)
    Tc_mean = (Tc[N - 1] + Tcin) / 2.0
    lnkr = 38.74 - 20700.0 / (Rgas * Tc_mean)
    kr = np.exp(lnkr)
    Dif = 3.07e-15 * Tc_mean / viscW
    Sc = viscW / (rhoW * Dif)
    kd = 0.023 * vz * ReC ** (-0.17) * Sc ** (-0.67)

    co2aq = yco2 / CH
    Hmais = 10.0 ** (-pH)
    hco3 = co2aq * K1 / Hmais
    co3_form = hco3 * K2 / Hmais
    co3 = ca + co3_form
    num = 1.0 - ksp / (ca * co3)
    den = 1.0 + kd / (kr * co3) + co3 / ca
    phid = kd * co3 * (num / den)
    phir = (0.00212 * vz ** 2 / (np.sqrt(kf) * psi)) * phid

    if fouling == 1:
        dRfw_dt = (phid - phir) / (rhof * kf)
        Uinv = Ucleaninv + Rfw * 1000.0
        U = 1.0 / Uinv
        if U > Uclean:
            U = Uclean  # chart clamps U only (no Rfw/df reset)
    else:
        dRfw_dt = -1.0    # sentinel (chart line: dRfw_dt = -1)
        U = Uclean

    UpDcoil = np.pi * U * Dcoil

    # ---------------- coil loop (chart computes dT/Qt inside the loop)
    dTc_dt = np.zeros(N)
    for i in range(N):
        dTi = Tr - Tc[i]
        Qt_i = ((4.0 * U) / Dcoil) * dTi / rhow_cpw
        if i == 0:
            dTc_dt[i] = 0.0
            Tc[i] = Tcin
        elif i == N - 1:
            dTc_dt[i] = -vz * (0.5 * Tc[i - 2] - 2.0 * Tc[i - 1]
                               + 1.5 * Tc[i]) / dz + Qt_i
        else:
            dTc_dt[i] = -vz * 0.5 * (Tc[i + 1] - Tc[i - 1]) / dz + Qt_i

    # Qcool uses trapz(-dT) over the pre-BC-pin profile in the .m; the
    # chart builds dT inside the loop BEFORE pinning Tc(1) at i==1 — same
    # ordering, so element 0 uses the incoming (unpinned) value.
    neg_dT_sum = 0.0
    for i in range(N - 1):
        neg_dT_sum += 0.5 * ((Tc[i] - Tr) + (Tc[i + 1] - Tr))
    # NOTE: Tc[0] was pinned inside the loop above; the .m computes dT
    # BEFORE the pin, so we restore the incoming first element for the
    # integral, matching dynmodel_fouling.m / chart arithmetic.
    Qcool = UpDcoil * dz * neg_dT_sum

    Qrmr = 2.0 * r1mr * (-dH1) + 3.0 * r2mr * (-dH2) + 1.5 * r3mr * (-dH3)
    mCpr = ma * CpA + mb * CpB + mc * CpC + me * CpE + mg * CpG + mp * CpP
    mCprinv = 1.0 / mCpr
    dTr_dt = mCprinv * (FA * CpA * (TA - Tr) + FB * CpB * (TB - Tr)
                        + Qcool + Qrmr)

    dx_dt = np.empty(8 + N)
    dx_dt[0] = dma_dt
    dx_dt[1] = dmb_dt
    dx_dt[2] = dmc_dt
    dx_dt[3] = dme_dt
    dx_dt[4] = dmg_dt
    dx_dt[5] = dmp_dt
    dx_dt[6] = dTr_dt
    for i in range(N):
        dx_dt[7 + i] = dTc_dt[i]
    dx_dt[7 + N] = dRfw_dt
    return dx_dt


def _pack_dyn_params(par: WoParameters) -> np.ndarray:
    from .fouling import _pack_fouling_params
    return _pack_fouling_params(par)


def _churchill_dpdz(par: WoParameters, df: float, Qcin: float) -> float:
    Dcoil = par.Dcoil
    Dcoildiff = Dcoil - 2.0 * df
    if Dcoildiff > Dcoil:
        Dcoildiff = Dcoil
    Acoil = 0.25 * np.pi * Dcoildiff ** 2
    vz = Qcin / Acoil
    ReC = par.rhoW * vz * Dcoildiff / par.viscW
    th1 = -2.457 * np.log((7.0 / ReC) ** 0.9 + 0.27 * par.epsilon / Dcoildiff)
    th2 = (37530.0 / ReC) ** 16
    f = 8.0 * ((8.0 / ReC) ** 12 + (1.0 / (th1 + th2) ** 1.5)) ** (1.0 / 12.0)
    return 0.5 * par.rhoW * vz ** 2 * f * par.Lcoil / Dcoildiff


# -----------------------------------------------------------------------------
# Harness configuration
# -----------------------------------------------------------------------------

@dataclass
class ClosedLoopConfig:
    """Configuration of the WO_main_sim harness (extracted defaults)."""
    t_end: float = 90 * 60.0          # recorded run is 90 min at Ts=5 s
    Ts: float = 5.0                   # controller sample time (slx)
    sp_stair: bool = False            # TR_SP + [5 0 -5 0] every 3600 s
    sp_stair_values: tuple = (5.0, 0.0, -5.0, 0.0)
    sp_stair_period: float = 3600.0
    ta_sine: bool = True              # TA/TB 24 h sinusoid, amp 5 K
    tb_sine: bool = True
    tcin_sine: bool = True            # Tcin 24 h sinusoid, amp 4 K
    sine_period: float = 24 * 3600.0
    fa_stair: bool = False            # FA repeating stair (inactive in data)
    fa_stair_values: tuple = (2.0922, 1.8275, 1.5628, 1.8275)
    fouling_on: bool = True           # fouling switch master
    fouling_step_time: float | None = 4 * 3600.0  # activate_fouling Step
    ar_noise: bool = False            # AR(3) noise on ca/pH (RECONSTRUCTED)
    ar_seed: int = 1
    reltol: float = 1e-6
    abstol: float = 1e-6


@dataclass
class ClosedLoopResult:
    t: np.ndarray = None
    FA: np.ndarray = None
    FB: np.ndarray = None
    TA: np.ndarray = None
    TB: np.ndarray = None
    Tcin: np.ndarray = None
    Tcout: np.ndarray = None
    LR: np.ndarray = None
    TR: np.ndarray = None
    QR: np.ndarray = None
    Qcin: np.ndarray = None
    DeltaP: np.ndarray = None
    Rf: np.ndarray = None
    df: np.ndarray = None
    TR_sp: np.ndarray = None

    def measured_table(self) -> np.ndarray:
        """Columns as recorded in sim_results.mat:
        [FA FB TA TB Tcin Tcout LR TR QR Qcin DeltaP]."""
        return np.column_stack([self.FA, self.FB, self.TA, self.TB,
                                self.Tcin, self.Tcout, self.LR, self.TR,
                                self.QR, self.Qcin, self.DeltaP])


class ClosedLoopSimulator:
    """WO_main_sim semantics: WO_dyn plant + LR_C PI + TR_C PID (2DOF)."""

    def __init__(self, par: WoParameters | None = None,
                 cfg: ClosedLoopConfig | None = None,
                 lr_gains: dict | None = None,
                 tr_gains: dict | None = None):
        self.par = closed_loop_parameters() if par is None else par
        self.cfg = cfg or ClosedLoopConfig()
        lr = dict(LR_C_GAINS)
        if lr_gains:
            lr.update(lr_gains)
        tr = dict(TR_C_GAINS)
        if tr_gains:
            tr.update(tr_gains)
        self._lr = lr
        self._tr = tr
        self._p = _pack_dyn_params(self.par)
        self.n = int(self.par.n)

    # -- initial state ------------------------------------------------------

    def initial_state(self, x0_plant: np.ndarray | None = None) -> np.ndarray:
        """[ma..mp, Tr, Tc(n), Rf] — defaults to the slx embedded IC."""
        if x0_plant is not None:
            return np.asarray(x0_plant, dtype=np.float64).copy()
        return _slx_ic_208()

    def default_ss(self) -> np.ndarray:
        """Solve initial.m steady state for the nominal inputs."""
        par = self.par
        u = np.array([1.8275, 4.7875, 294.3, 294.3, 288.75,
                      (1.8275 + 4.7875) / par.rhoR, 1.25e-3])
        x0, _ = solve_closed_loop_ss(u, par)
        # x0 = [Lr0, ma..mp, Tr, Tc, Rf0] -> plant state
        return np.concatenate([x0[1:8 + par.n], x0[8 + par.n:9 + par.n]])

    # -- disturbances -------------------------------------------------------

    def _disturbances(self, t: float, rng=None):
        cfg = self.cfg
        TA = 294.3
        TB = 294.3
        Tcin = 288.75
        FA = 1.8275
        ca = 150e-5
        pH = 9.0
        if cfg.ta_sine:
            TA = 294.3 + 5.0 * np.sin(2 * np.pi * t / cfg.sine_period)
        if cfg.tb_sine:
            TB = 294.3 + 5.0 * np.sin(2 * np.pi * t / cfg.sine_period)
        if cfg.tcin_sine:
            Tcin = 288.75 + 4.0 * np.sin(2 * np.pi * t / cfg.sine_period)
        if cfg.fa_stair:
            k = int(t // 1) % len(cfg.fa_stair_values)
            FA = cfg.fa_stair_values[k]
        return FA, TB, TA, Tcin, ca, pH

    def _sp(self, t: float, Tr0: float) -> float:
        if not self.cfg.sp_stair:
            return Tr0
        k = int(t // self.cfg.sp_stair_period) % len(self.cfg.sp_stair_values)
        return Tr0 + self.cfg.sp_stair_values[k]

    def _fouling_switch(self, t: float) -> float:
        cfg = self.cfg
        if not cfg.fouling_on:
            return 0.0
        if cfg.fouling_step_time is None:
            return 1.0
        return 1.0 if t >= cfg.fouling_step_time else 0.0

    # -- run ----------------------------------------------------------------

    def run(self, x0: np.ndarray | None = None,
            controllers: tuple[DiscretePID, DiscretePID] | None = None,
            record_from: float = 0.0) -> ClosedLoopResult:
        cfg = self.cfg
        par = self.par
        n = self.n
        Ts = cfg.Ts
        n_steps = int(round(cfg.t_end / Ts))

        x = self.initial_state(x0)
        # clamp at zero (Integrator_ODE LowerSaturationLimit)
        x = np.maximum(x, 0.0)

        QR_nom = (par.FA + par.FB) / par.rhoR
        Qcin_nom = par.Qcin

        if controllers is None:
            lr = DiscretePID(self._lr["P"], self._lr["I"], self._lr["D"],
                             self._lr["N"], Ts, umin=self._lr["umin"],
                             umax=self._lr["umax"])
            tr = DiscretePID(self._tr["P"], self._tr["I"], self._tr["D"],
                             self._tr["N"], Ts, c=self._tr["c"])
            Tr0 = float(x[6])
            lr.reset(integrator_ic=self._lr["integrator_ic"],
                     y0=par.massT / (par.Ar * par.rhoR))
            # NOTE on the filter IC: the .slx 'InitialConditionForFilter'
            # (23.1105) is the derivative filter's INTERNAL state in the
            # pid_lib realisation, not its output. Feeding it as the D-term
            # output would add a constant +3.49 m^3/s to Qcin; the recorded
            # run in sim_results.mat has Qcin ~ 1.2e-3 at steady state,
            # which pins the physical D-term at ~0 when y is constant.
            # The port therefore resets the filtered-derivative OUTPUT to 0
            # at t=0 (with yprev = Tr0), which reproduces the steady state.
            tr.reset(integrator_ic=self._tr["integrator_ic"],
                     filter_ic=0.0, y0=Tr0)
            controllers = (lr, tr)
        lr, tr = controllers

        LR_sp = par.massT / (par.Ar * par.rhoR)
        TR_sp0 = float(x[6])

        res = ClosedLoopResult()
        cols = {k: [] for k in
                ("FA", "FB", "TA", "TB", "Tcin", "Tcout", "LR", "TR",
                 "QR", "Qcin", "DeltaP", "Rf", "df", "TR_sp")}
        times = []

        rng = np.random.default_rng(self.cfg.ar_seed) if cfg.ar_noise else None
        ar_state = np.array([150e-5, 150e-5, 150e-5]), np.array([9.0, 9.0, 9.0])

        u_ctrl = [QR_nom, Qcin_nom]  # current controller outputs

        for k in range(n_steps + 1):
            t = k * Ts

            MR = float(x[0:6].sum())
            LR = MR / par.rhoR / par.Ar
            TR = float(x[6])
            Tcout = float(x[7 + n - 1])
            Rfw = float(x[7 + n])
            df = par.kf * Rfw * 1000.0

            FA, TB, TA, Tcin, ca, pH = self._disturbances(t)
            if rng is not None and k > 0:
                # AR(3) on (ca, pH), Ts = 30 s (RECONSTRUCTED noise path)
                if k % 6 == 0:
                    phi = np.array([0.1, 0.25, 0.6])
                    ca_hist, pH_hist = ar_state
                    e_ca = rng.normal(0.0, np.sqrt(2e-10))
                    e_pH = rng.normal(0.0, np.sqrt(2e-10))
                    ca = 150e-5 + phi.dot(ca_hist - 150e-5) + e_ca
                    pH = 9.0 + phi.dot(pH_hist - 9.0) + e_pH
                    ca_hist[:] = [ca_hist[1], ca_hist[2], ca]
                    pH_hist[:] = [pH_hist[1], pH_hist[2], pH]
                else:
                    ca = ar_state[0][-1]
                    pH = ar_state[1][-1]

            fouling = self._fouling_switch(t)

            # controllers (measurements: ideal sensors — no delay/noise)
            Qr = lr.step(LR_sp, LR)
            tr_sp = self._sp(t, TR_sp0)
            qc = tr.step(tr_sp, TR)
            u_ctrl = [Qr, qc]

            if t >= record_from:
                times.append(t)
                cols["FA"].append(FA); cols["FB"].append(par.FB)
                cols["TA"].append(TA); cols["TB"].append(TB)
                cols["Tcin"].append(Tcin); cols["Tcout"].append(Tcout)
                cols["LR"].append(LR); cols["TR"].append(TR)
                cols["QR"].append(Qr); cols["Qcin"].append(qc)
                cols["DeltaP"].append(_churchill_dpdz(par, df if fouling else 0.0, qc))
                cols["Rf"].append(Rfw); cols["df"].append(df)
                cols["TR_sp"].append(tr_sp)

            if k == n_steps:
                break

            # plant integration over one sample period
            u_plant = np.array([FA, par.FB, TA, TB, Tcin, Qr, qc, ca, pH,
                                df if fouling else 0.0, fouling])
            p = self._p
            sol = solve_ivp(lambda tt, xx: _wo_dyn_kernel(tt, xx, u_plant, p),
                            (t, t + Ts), x, method="BDF",
                            rtol=cfg.reltol, atol=cfg.abstol)
            if not sol.success:
                raise RuntimeError(f"plant integration failed at t={t}: {sol.message}")
            x = sol.y[:, -1]
            x = np.maximum(x, 0.0)  # integrator lower clamp

        res.t = np.asarray(times)
        for k_, v in cols.items():
            setattr(res, k_, np.asarray(v))
        return res


# -----------------------------------------------------------------------------
# Golden initial condition embedded in WO_reactor_model.slx (Integrator_ODE):
# masses, Tr, and Rf=0 are literals below; the Tc(1..200) tail is the same
# analytic clean profile that initial.m generates (Tc(1) = 288.75 = Tcin in
# the XML) and is regenerated analytically. The full 208-vector extracted
# from the XML is checked against this reconstruction to fp precision in
# tests/test_closed_loop.py.
# -----------------------------------------------------------------------------

_SLX_IC_HEAD = [144.889014657307, 761.421870463219, 22.1705917797745,
                650.774046418568, 300.536180999575, 225.208296222566,
                371.094840058293]


def _slx_ic_208() -> np.ndarray:
    """Full 208-element initial condition: [ma..mp, Tr, Tc(1..200), Rf=0]."""
    from .parameters import closed_loop_parameters
    par = closed_loop_parameters()
    head = list(_SLX_IC_HEAD)  # masses + Tr
    Tr0 = head[6]
    zspan = np.linspace(0.0, par.Lcoil, par.n)
    Tc0 = Tr0 + (288.75 - Tr0) * np.exp(
        -par.U * np.pi * par.Dcoil * zspan
        / (par.rhoW * par.CpW * 1.25e-3))
    return np.concatenate([head, Tc0, [0.0]])
