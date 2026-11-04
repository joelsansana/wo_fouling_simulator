"""Williams-Otto reactor with CaCO3 fouling — open-loop RHS.

Port of ``model_matlab_open_loop/dynmodel_fouling.m`` (version 2021.09.22.1,
authors Coutinho / Sansana / Granjo, CIEPQPF-Coimbra + CERENA-IST, 2021).

Two layers are provided:

1. :func:`dynmodel_fouling` — the *parity* function: it takes the full
   ``11 + n`` state vector used by the MATLAB ``ode15s`` call
   (``[L, ma..mp, Tr, Tc(1..n), Rfw, df, dpdz]``) and returns the same
   vector of residuals/derivatives, with the three algebraic rows ``g1``
   (level), ``g2`` (layer thickness) and ``g3`` (pressure drop) in their
   MATLAB positions. Use this to check the port against the .m file
   equation-by-equation.

2. :class:`FoulingReactor` — the *integrated* form. MATLAB handled the three
   algebraic rows with a singular Mass matrix (``M(1,1)=M(end-1,end-1)=
   M(end,end)=0``); ``scipy.integrate.solve_ivp`` has no mass-matrix
   support, so — exactly like the Simulink realisation, which resolves the
   same constraints through ``AlgebraicConstraint`` blocks — the port
   resolves ``L``, ``df`` and ``dpdz`` algebraically inside the RHS from the
   differential states (``L = MR/(rhoR*Ar)``, ``df = kf*Rfw*1000``,
   ``dpdz`` from the Darcy-Churchill expression) and integrates only the
   ``8 + n`` differential states ``[ma..mp, Tr, Tc(1..n), Rfw]``.

Fouling kinetics: precipitation–removal model of Souza & Costa (2019)
[3]; removal term per Cresmaschi & Wu (2015) [5]. Units are transcribed
verbatim from the .m file — in particular ``kf`` is in kW/(m·K) and the
``*1000`` factors convert the fouling resistance between kW- and W-bases.

Execution order inside the RHS matches the .m file line-for-line, including
the quirk that ``dRfw_dt`` is computed from the pre-clamp ``Rfw`` and that
the ``U > Uclean`` clamp additionally resets ``Rfw = 1e-10`` and
``df = 1e-8`` *after* ``dRfw_dt`` and *before* ``g2`` (making ``g2``
inconsistent — nonzero residual — when the clamp fires; dead code in
practice because ``Rfw >= 0`` implies ``U <= Uclean``).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .parameters import WoParameters

_EPS_MR = 1e-10


# -----------------------------------------------------------------------------
# Numba kernel over the differential states [ma..mp, Tr, Tc(1..n), Rfw]
# -----------------------------------------------------------------------------

@njit(cache=True)
def _fouling_kernel(t, x, u, p):
    """Numba-compiled differential RHS of the fouling WO reactor.

    Parameter layout (index → meaning), see ``_pack_fouling_params``:

        0  rhoR       10 CpA        20 rhoW       28 kf
        1  A1         11 CpB        21 viscW      29 Rgas
        2  A2         12 CpC        22 Lcoil      30 rhof
        3  A3         13 CpE        23 Dcoil      31 CH
        4  Ea1        14 CpP        24 n          32 ksp
        5  Ea2        15 CpG        25 Ar         33 yco2
        6  Ea3        16 CpW        26 epsilon    34 psi
        7  DeltaH1    17 Uclean     27 (spare)    35 K1
        8  DeltaH2    18 rhoW  (dup of 20, unused)  36 K2
        9  DeltaH3    19 (spare)
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
    n = int(p[24])
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

    ma = x[0]; mb = x[1]; mc = x[2]
    me = x[3]; mg = x[4]; mp = x[5]
    Tr = x[6]
    Tc = x[7:7 + n].copy()  # MATLAB value semantics (BC pin must not leak)
    Rfw = x[7 + n]

    # Sanity clamps (upstream warnings, transcribed)
    if Tr < 0.0: Tr = 0.0
    if ma < 0.0: ma = 0.0
    if mb < 0.0: mb = 0.0
    if mc < 0.0: mc = 0.0
    if me < 0.0: me = 0.0
    if mg < 0.0: mg = 0.0
    if mp < 0.0: mp = 0.0
    for i in range(n):
        if Tc[i] < 0.0:
            Tc[i] = 0.0

    # ------------------------------------------------ mass / level (AE g1)
    MR = ma + mb + mc + me + mg + mp
    if MR < _EPS_MR:
        MR = _EPS_MR
    xa = ma / MR; xb = mb / MR; xc = mc / MR
    xe = me / MR; xg = mg / MR; xp = mp / MR

    FR = rhoR * QR
    if FR < 0.0: FR = 0.0
    FRa = FR * xa; FRb = FR * xb; FRc = FR * xc
    FRe = FR * xe; FRg = FR * xg; FRp = FR * xp

    # Algebraic g2 resolved: df = kf*Rfw*1000 (consistent state)
    df = kf * Rfw * 1000.0

    # ------------------------------------------------------- reaction rates
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

    # ------------------------------------------------ coil hydraulics (g3)
    Dcoildiff = Dcoil - 2.0 * df
    if Dcoildiff > Dcoil:
        Dcoildiff = Dcoil
    Acoil = 0.25 * np.pi * Dcoildiff ** 2
    vz = Qcin / Acoil

    dz = Lcoil / (n - 1)

    rhow_cpw = rhoW * CpW

    ReC = rhoW * vz * Dcoildiff / viscW

    # Darcy friction factor — Churchill correlation (verbatim)
    theta1 = -2.457 * np.log((7.0 / ReC) ** 0.9 + 0.27 * epsilon / Dcoildiff)
    theta2 = (37530.0 / ReC) ** 16
    f = 8.0 * ((8.0 / ReC) ** 12 + (1.0 / (theta1 + theta2) ** 1.5)) ** (1.0 / 12.0)

    # Mean coil temperature (inlet/outlet mean)
    Tc_mean = (Tc[n - 1] + Tcin) / 2.0

    # Precipitation rate coefficient (Arrhenius fit)
    lnkr = 38.74 - 20700.0 / (Rgas * Tc_mean)
    kr = np.exp(lnkr)

    # Schmidt number
    Dif = 3.07e-15 * Tc_mean / viscW
    Sc = viscW / (rhoW * Dif)

    # Transport coefficient
    kd = 0.023 * vz * ReC ** (-0.17) * Sc ** (-0.67)

    # Fouling formation rate
    co2aq = yco2 / CH
    Hmais = 10.0 ** (-pH)
    hco3 = co2aq * K1 / Hmais
    co3_form = hco3 * K2 / Hmais
    co3 = ca + co3_form
    num = 1.0 - ksp / (ca * co3)
    den = 1.0 + kd / (kr * co3) + co3 / ca
    phid = kd * co3 * (num / den)
    phir = (0.00212 * vz ** 2 / (np.sqrt(kf) * psi)) * phid

    dRfw_dt = (phid - phir) / (rhof * kf)

    # Fouled heat transfer coefficient + clamp (verbatim order)
    Uinv = Ucleaninv + Rfw * 1000.0
    U = 1.0 / Uinv
    if U > Uclean:
        U = Uclean
        Rfw = 1e-10
        df = 1e-8

    UpDcoil = np.pi * U * Dcoil

    # ------------------------------------------------------ coil energy
    # dT computed BEFORE the BC pin — same order as the .m file.
    dT = Tr - Tc
    Qt = ((4.0 * U) / Dcoil) * dT / rhow_cpw

    dTc_dt = np.zeros(n)
    dTc_dt[0] = 0.0
    Tc[0] = Tcin
    if n > 1:
        for i in range(1, n - 1):
            dTc_dt[i] = -vz * 0.5 * (Tc[i + 1] - Tc[i - 1]) / dz + Qt[i]
        dTc_dt[n - 1] = (
            -vz * (0.5 * Tc[n - 3] - 2.0 * Tc[n - 2] + 1.5 * Tc[n - 1]) / dz
            + Qt[n - 1]
        )

    neg_dT_sum = 0.0
    for i in range(n - 1):
        neg_dT_sum += 0.5 * (-dT[i] + -dT[i + 1])
    Qcool = UpDcoil * dz * neg_dT_sum

    # -------------------------------------------------- reactor energy
    Qrmr = 2.0 * r1mr * (-dH1) + 3.0 * r2mr * (-dH2) + 1.5 * r3mr * (-dH3)

    mCpr = ma * CpA + mb * CpB + mc * CpC + me * CpE + mg * CpG + mp * CpP
    mCprinv = 1.0 / mCpr

    dTr_dt = mCprinv * (
        FA * CpA * (TA - Tr) + FB * CpB * (TB - Tr) + Qcool + Qrmr
    )

    dx_dt = np.empty(8 + n)
    dx_dt[0] = dma_dt
    dx_dt[1] = dmb_dt
    dx_dt[2] = dmc_dt
    dx_dt[3] = dme_dt
    dx_dt[4] = dmg_dt
    dx_dt[5] = dmp_dt
    dx_dt[6] = dTr_dt
    for i in range(n):
        dx_dt[7 + i] = dTc_dt[i]
    dx_dt[7 + n] = dRfw_dt
    return dx_dt


def _pack_fouling_params(par: WoParameters) -> np.ndarray:
    p = np.zeros(37, dtype=np.float64)
    p[0] = par.rhoR
    p[1] = par.A1; p[2] = par.A2; p[3] = par.A3
    p[4] = par.Ea1; p[5] = par.Ea2; p[6] = par.Ea3
    p[7] = par.DeltaH1; p[8] = par.DeltaH2; p[9] = par.DeltaH3
    p[10] = par.CpA; p[11] = par.CpB; p[12] = par.CpC
    p[13] = par.CpE; p[14] = par.CpP; p[15] = par.CpG
    p[16] = par.CpW
    p[17] = par.U
    p[20] = par.rhoW
    p[21] = par.viscW
    p[22] = par.Lcoil
    p[23] = par.Dcoil
    p[24] = float(par.n)
    p[25] = par.Ar
    p[26] = par.epsilon
    p[28] = par.kf
    p[29] = par.Rgas
    p[30] = par.rhof
    p[31] = par.CH
    p[32] = par.ksp
    p[33] = par.yco2
    p[34] = par.psi
    p[35] = par.K1
    p[36] = par.K2
    return p


# -----------------------------------------------------------------------------
# Parity layer — full 11+n MATLAB state with AE rows in MATLAB positions
# -----------------------------------------------------------------------------

def dynmodel_fouling(t, x, u, par: WoParameters) -> np.ndarray:
    """Full-parity RHS mirroring ``dynmodel_fouling.m`` exactly.

    ``x = [L, ma, mb, mc, me, mg, mp, Tr, Tc(1..n), Rfw, df, dpdz]``
    (length ``11 + n``). Returns
    ``[g1, dma_dt..dmp_dt, dTr_dt, dTc_dt, dRfw_dt, g2, g3]``.

    Implemented by delegating the differential rows to the kernel with the
    algebraic state values resolved from ``x`` itself, then recomputing the
    three residuals. The numeric path is shared with the integrated form so
    the two can never drift apart.
    """
    n = int(par.n)
    x = np.asarray(x, dtype=np.float64)
    L = x[0]
    ma, mb, mc, me, mg, mp = x[1:7]
    Tr = x[7]
    Tc = x[8:8 + n]
    Rfw = x[8 + n]
    df = x[9 + n]
    dpdz = x[10 + n]

    MR = float(ma + mb + mc + me + mg + mp)
    if MR < _EPS_MR:
        MR = _EPS_MR
    g1 = MR / par.rhoR - par.Ar * L

    # Differential rows (kernel treats df as kf*Rfw*1000; if the caller
    # passes a df that violates g2, g2 below reports the inconsistency,
    # matching the .m behaviour).
    xdiff = np.concatenate([x[1:8 + n], [Rfw]])
    p = _pack_fouling_params(par)
    d = _fouling_kernel(t, xdiff, np.asarray(u, dtype=np.float64), p)

    # Recompute the AE residuals from the *caller's* df/dpdz values using
    # the same constitutive expressions as the .m.
    Dcoil = par.Dcoil
    kf = par.kf
    Dcoildiff = Dcoil - 2.0 * df
    if Dcoildiff > Dcoil:
        Dcoildiff = Dcoil
    Acoil = 0.25 * np.pi * Dcoildiff ** 2
    vz = u[6] / Acoil
    ReC = par.rhoW * vz * Dcoildiff / par.viscW
    theta1 = -2.457 * np.log((7.0 / ReC) ** 0.9 + 0.27 * par.epsilon / Dcoildiff)
    theta2 = (37530.0 / ReC) ** 16
    f = 8.0 * ((8.0 / ReC) ** 12 + (1.0 / (theta1 + theta2) ** 1.5)) ** (1.0 / 12.0)
    g2 = kf * Rfw * 1000.0 - df
    g3 = 0.5 * par.rhoW * vz ** 2 * f * par.Lcoil / Dcoildiff - dpdz

    out = np.empty(11 + n)
    out[0] = g1
    out[1:8 + n] = d[:7 + n]
    out[8 + n] = d[7 + n]          # dRfw_dt
    out[9 + n] = g2
    out[10 + n] = g3
    return out


def mass_matrix(n_states: int, n: int) -> np.ndarray:
    """The MATLAB ``M`` for the fouling model (documentational helper).

    Identity except ``M(0,0) = M(n_diff-1, n_diff-1) = M(n_diff, n_diff) = 0``
    where ``n_diff = 8 + n`` differential states precede the two AE rows —
    i.e. rows for ``L`` (index 0), ``df`` and ``dpdz`` (last two).
    """
    M = np.eye(n_states)
    M[0, 0] = 0.0
    M[n_states - 2, n_states - 2] = 0.0
    M[n_states - 1, n_states - 1] = 0.0
    return M


# -----------------------------------------------------------------------------
# Integrated wrapper
# -----------------------------------------------------------------------------

class FoulingReactor:
    """Callable wrapper for ``scipy.integrate`` over the differential states.

    State layout (length ``8 + n``)::

        x[0:6]   ma..mp      kg
        x[6]     Tr          K
        x[7:7+n] Tc(z)       K
        x[7+n]   Rfw         m^2 K / kW

    Inputs ``u`` (length 9): ``[FA, FB, TA, TB, Tcin, QR, Qcin, ca, pH]``.

    ``algebraic(x, u)`` returns the resolved AE outputs ``L``, ``df``,
    ``dpdz`` (and derived quantities) for inspection/logging.
    """

    def __init__(self, par: WoParameters):
        self.par = par
        self._p = _pack_fouling_params(par)
        self.n = int(par.n)

    def state_length(self) -> int:
        return 8 + self.n

    def initial_state(self, ma, mb, mc, me, mg, mp, Tr, Tc, Rfw=0.0) -> np.ndarray:
        x = np.empty(self.state_length(), dtype=np.float64)
        x[0:6] = (ma, mb, mc, me, mg, mp)
        x[6] = Tr
        Tc = np.asarray(Tc, dtype=np.float64)
        if len(Tc) < self.n:
            Tc = np.concatenate([Tc, np.full(self.n - len(Tc), Tc[-1])])
        elif len(Tc) > self.n:
            Tc = Tc[:self.n]
        x[7:7 + self.n] = Tc
        x[7 + self.n] = Rfw
        return x

    def nominal_inputs(
        self,
        FA=None, FB=None, TA=None, TB=None,
        Tcin=None, QR=None, Qcin=None, ca=None, pH=None,
    ) -> np.ndarray:
        """9-element input vector. Defaults mirror ``script_sim_main.m``:
        TA=TB=294.3, Tcin=288.75, Qcin=1.25e-3, ca=150e-5, pH=9,
        QR=(FA+FB)/rhoR."""
        par = self.par
        FA = par.FA if FA is None else FA
        FB = par.FB if FB is None else FB
        TA = 294.3 if TA is None else TA
        TB = 294.3 if TB is None else TB
        Tcin = 288.75 if Tcin is None else Tcin
        QR = (FA + FB) / par.rhoR if QR is None else QR
        Qcin = par.Qcin if Qcin is None else Qcin
        ca = par.ca if ca is None else ca
        pH = par.pH if pH is None else pH
        return np.array([FA, FB, TA, TB, Tcin, QR, Qcin, ca, pH], dtype=np.float64)

    def f(self, u: np.ndarray):
        u = np.asarray(u, dtype=np.float64)
        p = self._p

        def rhs(t, x):
            return _fouling_kernel(t, x, u, p)
        return rhs

    def algebraic(self, x: np.ndarray, u: np.ndarray) -> dict:
        """Resolve the three AE outputs plus derived fouling diagnostics."""
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        par = self.par
        n = self.n
        MR = float(x[0:6].sum())
        L = MR / par.rhoR / par.Ar
        Rfw = float(x[7 + n])
        df = par.kf * Rfw * 1000.0
        Dcoildiff = par.Dcoil - 2.0 * df
        if Dcoildiff > par.Dcoil:
            Dcoildiff = par.Dcoil
        Acoil = 0.25 * np.pi * Dcoildiff ** 2
        vz = u[6] / Acoil
        ReC = par.rhoW * vz * Dcoildiff / par.viscW
        theta1 = -2.457 * np.log((7.0 / ReC) ** 0.9 + 0.27 * par.epsilon / Dcoildiff)
        theta2 = (37530.0 / ReC) ** 16
        f = 8.0 * ((8.0 / ReC) ** 12 + (1.0 / (theta1 + theta2) ** 1.5)) ** (1.0 / 12.0)
        dpdz = 0.5 * par.rhoW * vz ** 2 * f * par.Lcoil / Dcoildiff
        Uinv = 1.0 / par.U + Rfw * 1000.0
        return {
            "L": L, "df": df, "dpdz": dpdz, "MR": MR,
            "Dcoildiff": Dcoildiff, "vz": vz, "ReC": ReC, "fanning8": f,
            "U": 1.0 / Uinv,
        }


if __name__ == "__main__":
    from .parameters import closed_loop_parameters
    par = closed_loop_parameters()
    reactor = FoulingReactor(par)
    u = reactor.nominal_inputs()
    # Clean steady-state-ish probe state
    Tr0 = 371.0948
    zspan = np.linspace(0, par.Lcoil, par.n)
    Tc0 = Tr0 + (u[4] - Tr0) * np.exp(
        -par.U * np.pi * par.Dcoil * zspan / (par.rhoW * par.CpW * u[6]))
    x0 = reactor.initial_state(144.889, 761.422, 22.171, 650.774, 300.536,
                               225.208, Tr0, Tc0, Rfw=0.0)
    dx = reactor.f(u)(0.0, x0)
    print("state length:", reactor.state_length())
    print("max |d mass/dt|:", np.abs(dx[0:6]).max())
    print("dTr/dt:", dx[6], " K/s")
    print("dRfw/dt:", dx[7 + par.n], " m^2.K/(kJ.s)")
    print("algebraic:", {k: round(v, 6) for k, v in reactor.algebraic(x0, u).items()})
