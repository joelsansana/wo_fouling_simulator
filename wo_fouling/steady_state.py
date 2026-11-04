"""Steady-state initialisation — ports of ``init.m``, ``ssmodel.m``, ``initial.m``.

Two solvers, both using the upstream 4-stage sequential homotopy (each
stage's solution warm-starts the next, because a cold fsolve on the full
``12 + n`` system does not converge):

* :func:`solve_open_loop_ss` — port of the ``sim_main.m`` initialisation
  over ``init.m``. Works in MASS space: stage vector
  ``[L, ma..mp, Trinv, lnkl1..3, Tr, Tc(1..n)]``.
* :func:`solve_closed_loop_ss` — port of ``initial.m`` over ``ssmodel.m``.
  Works in MASS-FRACTION space with ``MR = par.massT`` fixed (consistent
  there because the six fractions are the unknowns and ``massT`` is the
  scale), then converts back to masses.

Upstream starting point (``sim_main.m`` line 69):
``x0 = [0.073; 0.37; 0.012; 0.31; 0.13; 0.1; 368.81]`` — SIX mass
fractions (A,B,C,E,G,P) followed by Tr — with masses = massT * fractions.

Solver note (port substitution): MATLAB used ``fsolve`` (trust-region
dogleg, central differences in ``initial.m``); the port uses
``scipy.optimize.fsolve`` (MINPACK hybrid Powell). Both drive the same
residuals to the same tolerance; the substitution is recorded in
PORT_NOTES.md.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve

from .parameters import WoParameters

_EPS_MR = 1e-10


def _coil_analytic_profile(par: WoParameters, u: np.ndarray, Tr0: float) -> np.ndarray:
    zspan = np.linspace(0.0, par.Lcoil, int(par.n))
    return Tr0 + (u[4] - Tr0) * np.exp(
        -par.U * np.pi * par.Dcoil * zspan / (par.rhoW * par.CpW * u[6]))


# -----------------------------------------------------------------------------
# init.m residual (open-loop, mass space)
# -----------------------------------------------------------------------------

def init_residual(x, u, par: WoParameters, tcopt: int, ctx: dict) -> np.ndarray:
    """Port of ``init.m`` — residual for the given homotopy stage ``tcopt``.

    x layout: ``[L, ma..mp]`` (7) for tcopt=-2; plus
    ``[Trinv, lnkl1..3, Tr]`` (12) for -1; plus ``Tc(1..n)`` for 0/1.
    ``ctx`` carries the stage-0 constants (``Trinv0, lnkl10..30, Tr0, Tc0``).
    """
    A1, A2, A3 = par.A1, par.A2, par.A3
    Ea1, Ea2, Ea3 = par.Ea1, par.Ea2, par.Ea3
    DeltaH1, DeltaH2, DeltaH3 = par.DeltaH1, par.DeltaH2, par.DeltaH3
    CpA, CpB, CpC = par.CpA, par.CpB, par.CpC
    CpE, CpG, CpP, Cpw = par.CpE, par.CpG, par.CpP, par.CpW
    U, Lcoil, Dcoil = par.U, par.Lcoil, par.Dcoil
    Acoil, rhoR, rhoW, Ar = par.Acoil, par.rhoR, par.rhoW, par.Ar

    FA, FB, TA, TB, Tcin, QR, Qcin = u[0:7]

    L = x[0]
    ma, mb, mc, me, mg, mp = x[1:7]
    if tcopt == -2:
        Trinv = ctx["Trinv0"]
        lnkl1, lnkl2, lnkl3 = ctx["lnkl10"], ctx["lnkl20"], ctx["lnkl30"]
        Tr = ctx["Tr0"]
        Tc = ctx["Tc0"]
    else:
        Trinv, lnkl1, lnkl2, lnkl3, Tr = x[7], x[8], x[9], x[10], x[11]
        Tc = ctx["Tc0"] if tcopt == -1 else np.asarray(x[12:], dtype=np.float64)
    n = len(Tc)

    # Sanity clamps (upstream warnings, transcribed)
    if L < 0: L = 0.0
    if Tr < 0: Tr = 0.0
    ma = max(ma, 0.0); mb = max(mb, 0.0); mc = max(mc, 0.0)
    me = max(me, 0.0); mg = max(mg, 0.0); mp = max(mp, 0.0)
    Tc = np.where(Tc < 0, 0.0, Tc)

    MR = ma + mb + mc + me + mg + mp
    if MR < _EPS_MR:
        MR = _EPS_MR
    xa, xb, xc = ma / MR, mb / MR, mc / MR
    xe, xg, xp = me / MR, mg / MR, mp / MR

    FR = rhoR * QR
    if FR < 0: FR = 0.0
    FRa, FRb, FRc = FR * xa, FR * xb, FR * xc
    FRe, FRg, FRp = FR * xe, FR * xg, FR * xp

    g1 = MR / rhoR - Ar * L

    g1aux0 = 1.0 - Trinv * Tr
    g1aux1 = lnkl1 - (np.log(A1) - Ea1 * Trinv)
    g1aux2 = lnkl2 - (np.log(A2) - Ea2 * Trinv)
    g1aux3 = lnkl3 - (np.log(A3) - Ea3 * Trinv)

    r1mr = np.exp(lnkl1) * xa * xb * MR
    r2mr = np.exp(lnkl2) * xb * xc * MR
    r3mr = np.exp(lnkl3) * xc * xp * MR

    g3 = FA - FRa - r1mr
    g4 = FB - FRb - r1mr - r2mr
    g5 = -FRc + 2 * r1mr - 2 * r2mr - r3mr
    g6 = -FRe + 2 * r2mr
    g7 = -FRg + 1.5 * r3mr
    g8 = -FRp + r2mr - 0.5 * r3mr

    vz = Qcin / Acoil
    dz = Lcoil / (n - 1)
    rhow_cpw = rhoW * Cpw
    UpDcoil = U * np.pi * Dcoil

    dT = Tr - Tc
    Qt = ((4 * U) / Dcoil) * dT / rhow_cpw

    g9 = np.zeros(n)
    if tcopt == 0:
        zspan = np.arange(0.0, Lcoil + 0.5 * dz, dz)[:n]
        Tcan = Tr + (Tcin - Tr) * np.exp(-UpDcoil * zspan / (rhow_cpw * Qcin))
        g9 = Tcan - Tc
    elif tcopt == 1:
        Tc = Tc.copy()
        g9[0] = 0.0
        Tc[0] = Tcin
        if n > 2:
            g9[1:n - 1] = -vz * 0.5 * (Tc[2:] - Tc[:n - 2]) / dz + Qt[1:n - 1]
        g9[n - 1] = (
            -vz * (0.5 * Tc[n - 3] - 2 * Tc[n - 2] + 1.5 * Tc[n - 1]) / dz
            + Qt[n - 1]
        )

    neg_dT_sum = float(np.sum(0.5 * (-dT[:-1] + -dT[1:])))
    Qcool = UpDcoil * dz * neg_dT_sum

    Qrmr = 2 * r1mr * (-DeltaH1) + 3 * r2mr * (-DeltaH2) + 1.5 * r3mr * (-DeltaH3)
    mCpr = ma * CpA + mb * CpB + mc * CpC + me * CpE + mg * CpG + mp * CpP
    g2 = (FA * CpA * (TA - Tr) + FB * CpB * (TB - Tr) + Qcool + Qrmr) / mCpr

    if tcopt == -2:
        return np.array([g1, g3, g4, g5, g6, g7, g8])
    if tcopt == -1:  # material + reactor energy only (init.m line 254-255)
        return np.array([g1, g3, g4, g5, g6, g7, g8,
                         g1aux0, g1aux1, g1aux2, g1aux3, g2])
    return np.concatenate([[g1, g3, g4, g5, g6, g7, g8],
                           [g1aux0, g1aux1, g1aux2, g1aux3, g2], g9])


# Upstream initial guess: six mass fractions (A,B,C,E,G,P) then Tr.
UPSTREAM_XI0 = np.array([0.073, 0.37, 0.012, 0.31, 0.13, 0.1])
UPSTREAM_TR0 = 368.81


def solve_open_loop_ss(u: np.ndarray, par: WoParameters,
                       xi0: np.ndarray | None = None,
                       Tr0: float | None = None,
                       xtol: float = 1e-13, verbose: bool = False):
    """4-stage homotopy port of the ``sim_main.m`` initialisation over
    ``init.m``. Returns ``(x_ss, info)`` with
    ``x_ss = [L, ma..mp, Tr, Tc(1..n)]`` (length ``8 + n``)."""
    u = np.asarray(u, dtype=np.float64)
    xi0 = UPSTREAM_XI0 if xi0 is None else np.asarray(xi0, dtype=np.float64)
    Tr0 = UPSTREAM_TR0 if Tr0 is None else float(Tr0)
    Trinv0 = 1.0 / Tr0
    ctx = {
        "Trinv0": Trinv0,
        "lnkl10": np.log(par.A1) - par.Ea1 * Trinv0,
        "lnkl20": np.log(par.A2) - par.Ea2 * Trinv0,
        "lnkl30": np.log(par.A3) - par.Ea3 * Trinv0,
        "Tr0": Tr0,
        "Tc0": _coil_analytic_profile(par, u, Tr0),
    }

    masses0 = xi0 * par.massT
    L0 = par.Vr / par.Ar

    opts = dict(xtol=xtol, maxfev=int(1e6), full_output=True)

    sol7, _, ier7, msg7 = fsolve(
        lambda x: init_residual(x, u, par, -2, ctx),
        np.concatenate([[L0], masses0]), **opts)
    sol12, _, ier12, msg12 = fsolve(
        lambda x: init_residual(x, u, par, -1, ctx),
        np.concatenate([sol7, [ctx["Trinv0"], ctx["lnkl10"],
                               ctx["lnkl20"], ctx["lnkl30"], ctx["Tr0"]]]),
        **opts)
    sol0, _, ier0, msg0 = fsolve(
        lambda x: init_residual(x, u, par, 0, ctx),
        np.concatenate([sol12, ctx["Tc0"]]), **opts)
    sol1, _, ier1, msg1 = fsolve(
        lambda x: init_residual(x, u, par, 1, ctx), sol0, **opts)

    if verbose:
        for name, ier, msg, sol, tc in (("stage-2", ier7, msg7, sol7, -2),
                                        ("stage-1", ier12, msg12, sol12, -1),
                                        ("stage0", ier0, msg0, sol0, 0),
                                        ("stage1", ier1, msg1, sol1, 1)):
            res = init_residual(sol, u, par, tc, ctx)
            print(f"{name}: ier={ier} max|g|={np.abs(res).max():.3e} ({msg})")

    x_ss = np.concatenate([sol1[0:7], sol1[11:]])
    info = dict(stage_solutions=(sol7, sol12, sol0, sol1),
                ier=(ier7, ier12, ier0, ier1),
                ctx=ctx,
                max_residual=float(np.abs(
                    init_residual(sol1, u, par, 1, ctx)).max()))
    return x_ss, info


# -----------------------------------------------------------------------------
# ssmodel.m residual (closed-loop, mass-fraction space) + initial.m driver
# -----------------------------------------------------------------------------

def ssmodel_residual(x, u, par: WoParameters, tcopt: int, ctx: dict) -> np.ndarray:
    """Port of ``ssmodel.m`` — fraction-space steady-state residual.

    x layout: ``[xa..xp]`` (6) for tcopt=-2; plus ``[Trinv, lnkl1..3, Tr]``
    (11) for -1; plus ``Tc(1..n)`` for 0/1. ``MR = par.massT`` is FIXED
    here (upstream choice — consistent in fraction space).
    """
    A1, A2, A3 = par.A1, par.A2, par.A3
    Ea1, Ea2, Ea3 = par.Ea1, par.Ea2, par.Ea3
    DeltaH1, DeltaH2, DeltaH3 = par.DeltaH1, par.DeltaH2, par.DeltaH3
    CpA, CpB, CpC = par.CpA, par.CpB, par.CpC
    CpE, CpG, CpP, Cpw = par.CpE, par.CpG, par.CpP, par.CpW
    U, Lcoil, Dcoil = par.U, par.Lcoil, par.Dcoil
    Acoil, rhoR, rhoW = par.Acoil, par.rhoR, par.rhoW
    MR = par.massT

    FA, FB, TA, TB, Tcin, QR, Qcin = u[0:7]

    xa, xb, xc, xe, xg, xp = x[0:6]
    if tcopt == -2:
        Trinv, lnkl1, lnkl2, lnkl3, Tr = (
            ctx["Trinv0"], ctx["lnkl10"], ctx["lnkl20"],
            ctx["lnkl30"], ctx["Tr0"])
        Tc = ctx["Tc0"]
    else:
        Trinv, lnkl1, lnkl2, lnkl3, Tr = x[6], x[7], x[8], x[9], x[10]
        Tc = ctx["Tc0"] if tcopt == -1 else np.asarray(x[11:], dtype=np.float64)
    n = len(Tc)

    Tc = np.where(Tc < 0, 0.0, Tc)

    FR = rhoR * QR
    if FR < 0: FR = 0.0
    FRa, FRb, FRc = FR * xa, FR * xb, FR * xc
    FRe, FRg = FR * xe, FR * xg

    g1aux0 = 1.0 - Trinv * Tr
    g1aux1 = lnkl1 - (np.log(A1) - Ea1 * Trinv)
    g1aux2 = lnkl2 - (np.log(A2) - Ea2 * Trinv)
    g1aux3 = lnkl3 - (np.log(A3) - Ea3 * Trinv)

    r1mr = np.exp(lnkl1) * xa * xb * MR
    r2mr = np.exp(lnkl2) * xb * xc * MR
    r3mr = np.exp(lnkl3) * xc * xp * MR

    g1 = FA - FRa - r1mr
    g2 = FB - FRb - r1mr - r2mr
    g3 = -FRc + 2 * r1mr - 2 * r2mr - r3mr
    g4 = -FRe + 2 * r2mr
    g5 = -FRg + 1.5 * r3mr
    g6 = 1.0 - (xa + xb + xc + xe + xg) - xp

    vz = Qcin / Acoil
    dz = Lcoil / (n - 1)
    rhow_cpw = rhoW * Cpw
    UpDcoil = U * np.pi * Dcoil

    dT = Tr - Tc
    Qt = ((4 * U) / Dcoil) * dT / rhow_cpw

    g8 = np.zeros(n)
    if tcopt == 0:
        zspan = np.arange(0.0, Lcoil + 0.5 * dz, dz)[:n]
        Tcan = Tr + (Tcin - Tr) * np.exp(-UpDcoil * zspan / (rhow_cpw * Qcin))
        g8 = Tcan - Tc
    elif tcopt == 1:
        Tc = Tc.copy()
        g8[0] = 0.0
        Tc[0] = Tcin
        if n > 2:
            g8[1:n - 1] = -vz * 0.5 * (Tc[2:] - Tc[:n - 2]) / dz + Qt[1:n - 1]
        g8[n - 1] = (
            -vz * (0.5 * Tc[n - 3] - 2 * Tc[n - 2] + 1.5 * Tc[n - 1]) / dz
            + Qt[n - 1]
        )

    neg_dT_sum = float(np.sum(0.5 * (-dT[:-1] + -dT[1:])))
    Qcool = UpDcoil * dz * neg_dT_sum

    Qrmr = 2 * r1mr * (-DeltaH1) + 3 * r2mr * (-DeltaH2) + 1.5 * r3mr * (-DeltaH3)
    mCpr = xa * CpA + xb * CpB + xc * CpC + xe * CpE + xg * CpG + xp * CpP
    g7 = (FA * CpA * (TA - Tr) + FB * CpB * (TB - Tr) + Qcool + Qrmr) / mCpr

    if tcopt == -2:
        return np.array([g1, g2, g3, g4, g5, g6])
    if tcopt == -1:  # material + reactor energy only (ssmodel.m line 216-217)
        return np.array([g1, g2, g3, g4, g5, g6,
                         g1aux0, g1aux1, g1aux2, g1aux3, g7])
    return np.concatenate([[g1, g2, g3, g4, g5, g6],
                           [g1aux0, g1aux1, g1aux2, g1aux3, g7], g8])


def solve_closed_loop_ss(u: np.ndarray, par: WoParameters,
                         x_guess: np.ndarray | None = None,
                         xtol: float = 1e-13):
    """Port of ``initial.m``: 4-stage fsolve over ``ssmodel_residual``.

    ``x_guess`` default = ``[1/6; ...; 1/6; 370]`` from ``script_sim_main.m``.
    Returns ``(x0, info)`` with
    ``x0 = [Lr0, ma..mp, Tr, Tc(1..n), Rf0=0]`` (length ``9 + n``).
    """
    u = np.asarray(u, dtype=np.float64)
    if x_guess is None:
        x_guess = np.concatenate([np.full(6, 1.0 / 6.0), [370.0]])
    x_guess = np.asarray(x_guess, dtype=np.float64)

    xi0 = x_guess[:6]
    Tr0 = float(x_guess[6])
    Trinv0 = 1.0 / Tr0
    ctx = {
        "Trinv0": Trinv0,
        "lnkl10": np.log(par.A1) - par.Ea1 * Trinv0,
        "lnkl20": np.log(par.A2) - par.Ea2 * Trinv0,
        "lnkl30": np.log(par.A3) - par.Ea3 * Trinv0,
        "Tr0": Tr0,
        "Tc0": _coil_analytic_profile(par, u, Tr0),
    }

    opts = dict(xtol=xtol, maxfev=int(1e6), full_output=True)

    sol6, _, ier1, _ = fsolve(
        lambda x: ssmodel_residual(x, u, par, -2, ctx), xi0, **opts)
    sol11, _, ier2, _ = fsolve(
        lambda x: ssmodel_residual(x, u, par, -1, ctx),
        np.concatenate([sol6, [Trinv0, ctx["lnkl10"], ctx["lnkl20"],
                               ctx["lnkl30"], Tr0]]), **opts)
    sol_a, _, ier3, _ = fsolve(
        lambda x: ssmodel_residual(x, u, par, 0, ctx),
        np.concatenate([sol11, ctx["Tc0"]]), **opts)
    sol_b, _, ier4, _ = fsolve(
        lambda x: ssmodel_residual(x, u, par, 1, ctx), sol_a, **opts)

    Lr0 = par.massT / (par.Ar * par.rhoR)
    # sol_b = [xa..xp, Trinv, lnkl1..3, Tr, Tc(1..n)] -> [Lr0] + masses + Tr + Tc + Rf0
    x0 = np.concatenate([[Lr0], sol_b[0:6] * par.massT, [sol_b[10]],
                         sol_b[11:], [0.0]])

    info = dict(ier=(ier1, ier2, ier3, ier4),
                fractions=sol_b[0:6],
                stage_solutions=(sol6, sol11, sol_a, sol_b),
                ctx=ctx,
                max_residual=float(np.abs(
                    ssmodel_residual(sol_b, u, par, 1, ctx)).max()))
    return x0, info


if __name__ == "__main__":
    from .parameters import closed_loop_parameters, open_loop_parameters
    from .reactor import OpenLoopReactor

    # Open-loop: sim_main.m inputs (Qcin = 1.5e-3 there)
    par_o = open_loop_parameters()
    u_o = np.array([1.8275, 4.7875, 294.3, 294.3, 288.75,
                    (1.8275 + 4.7875) / par_o.rhoR, 1.5e-3])
    x_ss, info = solve_open_loop_ss(u_o, par_o)
    print("open-loop SS: ier:", info["ier"], "max|g|:", info["max_residual"])
    print("  L=%.4f m  mt=%.2f kg  Tr=%.4f K" % (
        x_ss[0], x_ss[1:7].sum(), x_ss[7]))
    rhs = OpenLoopReactor(par_o).f(u_o)
    d = rhs(0.0, x_ss)
    print("  RHS check at SS: |dm|max=%.2e  |dTr|=%.2e  |dTc|max=%.2e" % (
        np.abs(d[1:7]).max(), abs(d[7]), np.abs(d[8:]).max()))

    # Closed-loop: script_sim_main.m inputs
    par_c = closed_loop_parameters()
    u_c = np.array([1.8275, 4.7875, 294.3, 294.3, 288.75,
                    (1.8275 + 4.7875) / par_c.rhoR, 1.25e-3, 150e-5, 9.0])
    x0, info_c = solve_closed_loop_ss(u_c, par_c)
    print("closed-loop SS: ier:", info_c["ier"], "max|g|:", info_c["max_residual"])
    print("  Lr0=%.4f m  mt=%.2f kg  Tr0=%.4f K  Rf0=%.1f" % (
        x0[0], x0[1:7].sum(), x0[7], x0[-1]))
