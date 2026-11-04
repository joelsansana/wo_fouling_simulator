"""Williams-Otto reactor — open-loop RHS.

Port of ``model_matlab_open_loop/dynmodel.m``.

This module exposes the right-hand side ``f(t, x, u, par) -> dx/dt`` for
the open-loop Williams-Otto reactor with a discretised cooling coil. It is
a faithful line-by-line port of the upstream MATLAB; see PORT_NOTES.md
for the source mapping, gotchas, and validation contract.

State vector layout (length ``11 + N`` where ``N = par.n``):

    x[0]    = L          reactor level, m                       (AE row)
    x[1]    = ma         kg A in reactor                        (DE)
    x[2]    = mb         kg B                                   (DE)
    x[3]    = mc         kg C                                   (DE)
    x[4]    = me         kg E                                   (DE)
    x[5]    = mg         kg G                                   (DE)
    x[6]    = mp         kg P                                   (DE)
    x[7]    = Tr         reactor temperature, K                (DE)
    x[8:8+N] = Tc        cooling-water temperature along z, K   (DE, with BC)

Note: the upstream ``sim_main.m`` declares ``M(1,1) = 0`` (mass matrix),
turning the first equation (level) into an algebraic constraint. The RHS
still produces a value for ``dx/dt[0]`` — the constraint is enforced by
the integrator, not by us. See AUDIT §2.5.

Input vector (length 7):

    u[0] = FA       kg/s — A feed
    u[1] = FB       kg/s — B feed
    u[2] = TA       K    — A feed temperature
    u[3] = TB       K    — B feed temperature
    u[4] = Tcin     K    — cooling-water inlet temperature
    u[5] = QR       m^3/s — outlet volumetric flow (rhoR * QR = mass out)
    u[6] = Qcin     m^3/s — cooling-water volumetric flow

The recycle stream ``FL`` is wired to 0 by upstream design; we keep the
same convention and ignore ``FL`` entirely.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .parameters import WoParameters


# Numerical floor for negative states — matches the upstream warning pattern
# in dynmodel.m. We clamp silently (the integrator never sees the warning)
# to avoid log-of-zero / sqrt-of-negative pathologies.
_EPS_STATE = 0.0
_EPS_MR = 1e-10


# -----------------------------------------------------------------------------
# Numba-friendly core. Arrays in, array out — no Python objects in the hot
# path. We use this for ``scipy.integrate.solve_ivp(method='BDF')`` calls
# later in the simulation layer.
# -----------------------------------------------------------------------------

@njit(cache=True)
def _dynmodel_kernel(
    t: float,
    x: np.ndarray,
    u: np.ndarray,
    # Parameters flattened into a single 1-D array in the order produced by
    # ``WoParameters.to_flat()`` plus a few extras appended (n, Ar, Vr,
    # Uclean, etc.) — see ``_pack_params`` for the exact layout.
    p: np.ndarray,
) -> np.ndarray:
    """Numba-compiled RHS of the open-loop WO reactor.

    Parameter layout (index → meaning):

        0  rhoR       kg/m^3
        1  A1         s^-1 kg
        2  A2         s^-1 kg
        3  A3         s^-1 kg
        4  Ea1        K
        5  Ea2        K
        6  Ea3        K
        7  DeltaH1    kJ/kg A
        8  DeltaH2    kJ/kg B
        9  DeltaH3    kJ/kg C
        10 CpA        kJ/(kg K)
        11 CpB
        12 CpC
        13 CpE
        14 CpP
        15 CpG
        16 CpW
        17 Uclean     kW/(m^2 K)
        18 rhoW       kg/m^3
        19 Lcoil      m
        20 Dcoil      m
        21 Acoil      m^2
        22 n          int — number of coil cells
        23 Ar         m^2
        24 Vr         m^3
    """
    # ------------------------------------------------------------------ unpack
    rhoR = p[0]
    A1 = p[1]; A2 = p[2]; A3 = p[3]
    Ea1 = p[4]; Ea2 = p[5]; Ea3 = p[6]
    dH1 = p[7]; dH2 = p[8]; dH3 = p[9]
    CpA = p[10]; CpB = p[11]; CpC = p[12]
    CpE = p[13]; CpP = p[14]; CpG = p[15]
    CpW = p[16]
    Uclean = p[17]
    rhoW = p[18]
    Lcoil = p[19]
    Dcoil = p[20]
    Acoil = p[21]
    n = int(p[22])
    Ar = p[23]
    # p[24] (Vr) is exposed through ``WoParameters.Vr`` for callers; the open-
    # loop RHS doesn't consume it directly because the level constraint is
    # expressed via ``Ar`` (see ``g1`` below). Keep the unpack here for
    # parity with the packed layout — discard with ``_``.
    _Vr = p[24]  # noqa: F841
    FA = u[0]
    FB = u[1]
    TA = u[2]
    TB = u[3]
    Tcin = u[4]
    QR = u[5]
    Qcin = u[6]

    # Recycle streams forced to 0 (matches dynmodel.m). Kept as named locals
    # for parity with the upstream RHS even though they're not consumed
    # downstream (the material-balance code inlines ``FL* = 0``).
    FLa = 0.0  # noqa: F841
    FLb = 0.0  # noqa: F841
    FLe = 0.0  # noqa: F841
    FLg = 0.0  # noqa: F841
    FLp = 0.0  # noqa: F841

    # ------------------------------------------------------------------ state
    L  = x[0]
    ma = x[1]
    mb = x[2]
    mc = x[3]
    me = x[4]
    mg = x[5]
    mp = x[6]
    Tr = x[7]
    # Tc as a COPY, not a view: dynmodel.m has MATLAB value semantics —
    # writing Tc(1) = Tcin inside the RHS must NOT leak into the caller's
    # state vector (a view here would corrupt the integrator's y).
    Tc = x[8:8 + n].copy()

    # Sanity-clamp negatives (matches the upstream warnings in dynmodel.m,
    # but we don't emit warnings; the integrator never sees the originals).
    if L  < 0.0: L  = 0.0
    if Tr < 0.0: Tr = 0.0
    if ma < 0.0: ma = 0.0
    if mb < 0.0: mb = 0.0
    if mc < 0.0: mc = 0.0
    if me < 0.0: me = 0.0
    if mg < 0.0: mg = 0.0
    if mp < 0.0: mp = 0.0
    # Clamp negative Tc cells to 0.
    for i in range(n):
        if Tc[i] < 0.0:
            Tc[i] = 0.0

    # ---------------------------------------------------- constitutive (mass)
    MR = ma + mb + mc + me + mg + mp
    if MR < _EPS_MR:
        MR = _EPS_MR

    xa = ma / MR
    xb = mb / MR
    xc = mc / MR
    xe = me / MR
    xg = mg / MR
    xp = mp / MR

    FR = rhoR * QR
    if FR < 0.0: FR = 0.0
    FRa = FR * xa
    FRb = FR * xb
    FRc = FR * xc
    FRe = FR * xe
    FRg = FR * xg
    FRp = FR * xp

    # ------------------------------------------- algebraic: level constraint
    # This is the AE row. sim_main.m uses Mass(1,1)=0 to enforce it; we
    # still write the value because solvers with mass-matrix support expect
    # a row to exist.
    g1 = (MR / rhoR) - Ar * L

    # --------------------------------------- reaction rates (Arrhenius mass)
    # Inlined log_rate_constants + mass_rates for Numba performance.
    Trinv = 1.0 / Tr
    lnkl1 = np.log(A1) - Ea1 * Trinv
    lnkl2 = np.log(A2) - Ea2 * Trinv
    lnkl3 = np.log(A3) - Ea3 * Trinv
    r1mr = np.exp(lnkl1) * xa * xb * MR
    r2mr = np.exp(lnkl2) * xb * xc * MR
    r3mr = np.exp(lnkl3) * xc * xp * MR

    # -------------------------------------------- material balance residuals
    dma_dt = FA + FLa - FRa -      r1mr +         0.0
    dmb_dt = FB + FLb - FRb -      r1mr -      r2mr
    dmc_dt = 0.0 - FRc + 2.0 * r1mr - 2.0 * r2mr -       r3mr
    dme_dt = 0.0 - FRe +         0.0 + 2.0 * r2mr
    dmg_dt = 0.0 - FRg +         0.0 +         0.0 + 1.5 * r3mr
    dmp_dt = 0.0 - FRp +         0.0 +      r2mr - 0.5 * r3mr
    # NOTE: FL* terms collapsed to 0 since FL is forced to 0 (see top).

    # --------------------------------------- cooling coil (finite difference)
    # Cooling-water velocity (m/s). Acoil == 0 ⇒ div-by-zero, guard with 1e-30.
    if Acoil <= 0.0:
        vz = 0.0
    else:
        vz = Qcin / Acoil
    dz = (Lcoil - 0.0) / (n - 1)

    rhow_cpw = rhoW * CpW
    UpDcoil = Uclean * np.pi * Dcoil

    dT = Tr - Tc                       # shape (n,)
    Qt = ((4.0 * Uclean) / Dcoil) * dT / rhow_cpw

    dTc_dt = np.zeros(n)
    # BC at i=1 (z=0): Tc(0,t) = Tcin  ⇒  residual 0 and Tc[0]=Tcin
    dTc_dt[0] = 0.0
    Tc[0] = Tcin
    if n == 1:
        # Degenerate 1-cell coil: just BC.
        pass
    else:
        # Interior: centred differences.
        for i in range(1, n - 1):
            dTc_dt[i] = -vz * 0.5 * (Tc[i + 1] - Tc[i - 1]) / dz + Qt[i]
        # Outlet (i=n): backward difference with second-order correction.
        dTc_dt[n - 1] = (
            -vz * (0.5 * Tc[n - 3] - 2.0 * Tc[n - 2] + 1.5 * Tc[n - 1]) / dz
            + Qt[n - 1]
        )

    # ----------------------------------------- integrated heat (trapz on -dT)
    # MATLAB: Qcool = UpDcoil * dz * trapz(-dT)  (unit spacing, NEGATED dT:
    # positive dT = reactor hotter than coil = Qcool negative = cooling).
    # np.trapz was removed in NumPy 2.0 and np.trapezoid is not Numba-typed,
    # so the trapezoid rule is written out explicitly. Identical arithmetic.
    neg_dT_sum = 0.0
    for i in range(n - 1):
        neg_dT_sum += 0.5 * ((Tc[i] - Tr) + (Tc[i + 1] - Tr))
    Qcool = UpDcoil * dz * neg_dT_sum

    # ----------------------------------------- reactor energy balance
    Qrmr = (
        2.0 * r1mr * (-dH1)
        + 3.0 * r2mr * (-dH2)
        + 1.5 * r3mr * (-dH3)
    )

    mCpr = (
        ma * CpA + mb * CpB + mc * CpC
        + me * CpE + mg * CpG + mp * CpP
    )
    if mCpr <= 0.0:
        mCprinv = 0.0
    else:
        mCprinv = 1.0 / mCpr

    dTr_dt = mCprinv * (
        FA * CpA * (TA - Tr)
        + FB * CpB * (TB - Tr)
        + Qcool
        + Qrmr
    )

    # --------------------------------------------------- assemble dx/dt
    dx_dt = np.empty(8 + n)
    dx_dt[0] = g1
    dx_dt[1] = dma_dt
    dx_dt[2] = dmb_dt
    dx_dt[3] = dmc_dt
    dx_dt[4] = dme_dt
    dx_dt[5] = dmg_dt
    dx_dt[6] = dmp_dt
    dx_dt[7] = dTr_dt
    for i in range(n):
        dx_dt[8 + i] = dTc_dt[i]
    return dx_dt


def _pack_params(par: WoParameters) -> np.ndarray:
    """Pack ``WoParameters`` into the flat array consumed by the kernel.

    Layout (25 entries; see _dynmodel_kernel docstring):

        rhoR, A1, A2, A3, Ea1, Ea2, Ea3, dH1, dH2, dH3,
        CpA..CpG (6), CpW,
        Uclean, rhoW,
        Lcoil, Dcoil, Acoil,
        n, Ar, Vr
    """
    return np.array([
        par.rhoR,
        par.A1, par.A2, par.A3,
        par.Ea1, par.Ea2, par.Ea3,
        par.DeltaH1, par.DeltaH2, par.DeltaH3,
        par.CpA, par.CpB, par.CpC, par.CpE, par.CpP, par.CpG,
        par.CpW,
        par.U,
        par.rhoW,
        par.Lcoil, par.Dcoil, par.Acoil,
        float(par.n),
        par.Ar,
        par.Vr,
    ], dtype=np.float64)


# -----------------------------------------------------------------------------
# Python-facing wrapper
# -----------------------------------------------------------------------------

class OpenLoopReactor:
    """Callable wrapper around the open-loop RHS for ``scipy.integrate``.

    Usage::

        reactor = OpenLoopReactor(par)
        rhs = reactor.f(u)             # u is a 1-D array of length 7
        dx_dt = rhs(t, x)

    The instance caches the packed parameter vector so it isn't reallocated
    on every integrator step. The underlying Numba kernel is jit-compiled
    on first call.
    """

    def __init__(self, par: WoParameters):
        self.par = par
        self._p = _pack_params(par)
        self.n = int(par.n)

    def state_length(self) -> int:
        """Return the full state length ``8 + n``."""
        return 8 + self.n

    def initial_state(
        self,
        L: float,
        ma: float, mb: float, mc: float, me: float, mg: float, mp: float,
        Tr: float,
        Tc: np.ndarray,
    ) -> np.ndarray:
        """Build a state vector of length ``8 + n``.

        Parameters
        ----------
        L : float
            Initial level, m.
        ma, mb, mc, me, mg, mp : float
            Initial mass hold-ups, kg.
        Tr : float
            Initial reactor temperature, K.
        Tc : (n,) array
            Initial cooling-water temperature profile along z, K. If shorter
            than ``n`` the tail is padded with the last value; if longer it
            is truncated.
        """
        if len(Tc) < self.n:
            Tc = np.concatenate([Tc, np.full(self.n - len(Tc), Tc[-1])])
        elif len(Tc) > self.n:
            Tc = Tc[: self.n]
        x = np.empty(self.state_length(), dtype=np.float64)
        x[0] = L
        x[1] = ma
        x[2] = mb
        x[3] = mc
        x[4] = me
        x[5] = mg
        x[6] = mp
        x[7] = Tr
        x[8:] = Tc
        return x

    def nominal_inputs(
        self,
        FA: float | None = None,
        FB: float | None = None,
        TA: float | None = None,
        TB: float | None = None,
        Tcin: float | None = None,
        Qcin: float | None = None,
    ) -> np.ndarray:
        """Build the 7-element input vector with sensible defaults.

        Defaults mirror ``sim_main.m``:
            TA = TB = 294.3 K, Tcin = 288.75 K, Qcin = 1.5e-3 m^3/s,
            FA, FB from ``WoParameters`` (1.8275 / 4.7875 kg/s),
            QR = (FA + FB) / rhoR (steady-state volumetric outlet flow).
        """
        FA_eff = self.par.FA if FA is None else FA
        FB_eff = self.par.FB if FB is None else FB
        TA_eff = 294.3 if TA is None else TA
        TB_eff = 294.3 if TB is None else TB
        Tcin_eff = 288.75 if Tcin is None else Tcin
        Qcin_eff = 1.5e-3 if Qcin is None else Qcin
        QR_eff = (FA_eff + FB_eff) / self.par.rhoR
        return np.array([
            FA_eff, FB_eff,
            TA_eff, TB_eff,
            Tcin_eff,
            QR_eff,
            Qcin_eff,
        ], dtype=np.float64)

    def f(self, u: np.ndarray):
        """Return a callable ``(t, x) -> dx/dt`` for the given input vector.

        The returned function is suitable as the ``fun`` argument to
        ``scipy.integrate.solve_ivp``.
        """
        u = np.asarray(u, dtype=np.float64)
        p = self._p
        # Bind into locals so the Numba kernel gets flat scalars/arrays.
        def rhs(t: float, x: np.ndarray) -> np.ndarray:
            return _dynmodel_kernel(t, x, u, p)
        return rhs


# -----------------------------------------------------------------------------
# CLI sanity check — call the kernel at the upstream steady state and verify
# every derivative is effectively zero (steady state).
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    from .parameters import default_parameters

    par = default_parameters()
    reactor = OpenLoopReactor(par)
    u = reactor.nominal_inputs()
    rhs = reactor.f(u)

    # Build a synthetic SS initial state at sim_main.m's published SS values.
    # xi0 = [0.073, 0.37, 0.012, 0.31, 0.13, 0.1, 368.81]
    # (level in m, mass-fractions dimensionless, Tr in K)
    L0 = 0.073
    xi0 = np.array([0.37, 0.012, 0.31, 0.13, 0.1])  # ma..mg mass fractions
    mp_mass_frac = 1.0 - xi0.sum()                  # closure
    Tr0 = 368.81

    massT = par.massT
    masses = xi0 * massT
    mp = mp_mass_frac * massT
    # Coil analytic SS profile (z=0..Lcoil).
    zspan = np.linspace(0, par.Lcoil, par.n)
    Tc0 = Tr0 + (u[4] - Tr0) * np.exp(
        -par.U * np.pi * par.Dcoil * zspan / (par.rhoW * par.CpW * u[6])
    )

    x0 = reactor.initial_state(L0, *masses, mp, Tr0, Tc0)
    dx = rhs(0.0, x0)

    print(f"State length: {reactor.state_length()}")
    print("At the upstream steady-state guess:")
    print(f"  L  = {x0[0]:.4f} m     (dx/dt[0] = {dx[0]:+.3e}  AE)")
    for i, name in enumerate("ma mb mc me mg mp".split()):
        print(f"  {name} = {x0[i + 1]:7.2f} kg   (dx/dt[{i + 1}] = {dx[i + 1]:+.3e})")
    print(f"  Tr = {x0[7]:7.3f} K    (dx/dt[7] = {dx[7]:+.3e})")
    print(f"  Tc[0]   = {x0[8]:.3f} K  (dx/dt[8] = {dx[8]:+.3e})")
    print(f"  Tc[-1]  = {x0[-1]:.3f} K  (dx/dt[-1] = {dx[-1]:+.3e})")
    print(f"  max |dx/dt[Tr]| = {max(abs(dx[7]), abs(dx[8:]).max()):.3e}")
