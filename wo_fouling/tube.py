"""Isolated fouling-tube simulator — port of ``simple_fouling_tube_sim/``.

Port of ``tubo_fouling.m`` (RHS) and ``sim_tubo_fouling.m`` (driver).
A single tube with the SAME CaCO3 fouling kinetics as the reactor coil,
driven by a fixed ambient temperature ``Tamb`` instead of the reactor
temperature ``Tr``. Useful for validating the fouling RHS in isolation.

State layout (length ``n + 3``), mirroring the .m::

    x[0:n]     Tc(z)   cooling-water temperature profile, K
    x[n]       Rfw     fouling thermal resistance, m^2 K / kW
    x[n+1]     df      fouling layer thickness, m              (AE row g10)
    x[n+2]     dpdz    pressure drop, Pa                       (AE row g11)

Inputs ``u`` (length 5): ``[Tcin, Qcin, Tamb, ca, pH]``.

``fator`` multiplies ``Dcoil`` (the .m's "FATOR A MULTIPLICAR" experiment
knob); ``fator=1`` reproduces the default run.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .parameters import WoParameters


@njit(cache=True)
def _tube_kernel(t, x, u, p):
    """Numba RHS. p layout:

        0 CpW     3 Lcoil   6  kf    9  rhof  12 yco2  15 K2    18 epsilon
        1 Uclean  4 rhoW    7  Rgas  10 CH    13 psi   16 n
        2 Dcoil   5 viscW   8  ksp   11 ksp?  14 K1    17 fator
    """
    CpW = p[0]
    Uclean = p[1]
    Dcoil = p[2]
    Lcoil = p[3]
    rhoW = p[4]
    viscW = p[5]
    kf = p[6]
    Rgas = p[7]
    rhof = p[8]
    CH = p[9]
    ksp = p[10]
    yco2 = p[11]
    psi = p[12]
    K1 = p[13]
    K2 = p[14]
    n = int(p[15])
    fator = p[16]
    epsilon = p[17]
    Ucleaninv = 1.0 / Uclean

    Tcin = u[0]; Qcin = u[1]; Tamb = u[2]; ca = u[3]; pH = u[4]

    Tc = x[0:n].copy()
    Rfw = x[n]
    df = x[n + 1]
    dpdz = x[n + 2]

    for i in range(n):
        if Tc[i] < 0.0:
            Tc[i] = 0.0

    # Hydraulics
    Dcoildiff = Dcoil - 2.0 * df
    if Dcoildiff > Dcoil:
        Dcoildiff = Dcoil
    Acoil = 0.25 * np.pi * Dcoildiff ** 2
    vz = Qcin / Acoil

    dz = Lcoil / (n - 1)
    rhow_cpw = rhoW * CpW

    ReC = rhoW * vz * Dcoildiff / viscW

    theta1 = -2.457 * np.log((7.0 / ReC) ** 0.9 + 0.27 * epsilon / Dcoildiff)
    theta2 = (37530.0 / ReC) ** 16
    f = 8.0 * ((8.0 / ReC) ** 12 + (1.0 / (theta1 + theta2) ** 1.5)) ** (1.0 / 12.0)

    g11 = 0.5 * rhoW * vz ** 2 * f * Lcoil / Dcoildiff - dpdz

    # Fouling kinetics (identical to dynmodel_fouling.m)
    Tc_mean = (Tc[n - 1] + Tcin) / 2.0
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

    dRfw_dt = (phid - phir) / (rhof * kf)

    Uinv = Ucleaninv + Rfw * 1000.0
    U = 1.0 / Uinv
    if U > Uclean:
        U = Uclean  # .m tube version does NOT reset Rfw/df here
    g10 = kf * Rfw * 1000.0 - df

    # Coil energy balance driven by Tamb
    dT = Tamb - Tc
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

    out = np.empty(n + 3)
    for i in range(n):
        out[i] = dTc_dt[i]
    out[n] = dRfw_dt
    out[n + 1] = g10
    out[n + 2] = g11
    return out


def _pack_tube_params(par: WoParameters, fator: float) -> np.ndarray:
    return np.array([
        par.CpW, par.U, par.Dcoil * fator, par.Lcoil,
        par.rhoW, par.viscW, par.kf, par.Rgas, par.rhof, par.CH,
        par.ksp, par.yco2, par.psi, par.K1, par.K2,
        float(par.n), float(fator), par.epsilon,
    ], dtype=np.float64)


def tubo_fouling(t, x, u, par: WoParameters, fator: float = 1.0) -> np.ndarray:
    """Parity RHS: returns ``[dTc_dt, dRfw_dt, g10, g11]`` (length n+3)."""
    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    return _tube_kernel(t, x, u, _pack_tube_params(par, fator))


def tube_initial_state(par: WoParameters, u: np.ndarray, fator: float = 1.0) -> np.ndarray:
    """Initial state per ``sim_tubo_fouling.m``:

    analytic clean steady profile Tc0(z) + ``Rfw=0, df=0`` and the CLEAN
    pressure drop ``dpdz0`` computed on the unfouled diameter.
    """
    u = np.asarray(u, dtype=np.float64)
    Tcin, Qcin, Tamb = u[0], u[1], u[2]
    n = int(par.n)
    zspan = np.linspace(0.0, par.Lcoil, n)
    Tc0 = Tamb + (Tcin - Tamb) * np.exp(
        -par.U * np.pi * fator * par.Dcoil * zspan
        / (par.rhoW * par.CpW * Qcin))
    vz = Qcin / par.Acoil
    ReC = par.rhoW * vz * par.Dcoil / par.viscW
    epsilon = par.epsilon
    theta1 = -2.457 * np.log((7.0 / ReC) ** 0.9 + 0.27 * epsilon / par.Dcoil)
    theta2 = (37530.0 / ReC) ** 16
    f = 8.0 * ((8.0 / ReC) ** 12 + (1.0 / (theta1 + theta2) ** 1.5)) ** (1.0 / 12.0)
    dpdz0 = 0.5 * par.rhoW * vz ** 2 * f * par.Lcoil / par.Dcoil
    return np.concatenate([Tc0, [0.0, 0.0, dpdz0]])


class TubeSimulator:
    """Integrated wrapper: differential states ``[Tc(n), Rfw]`` with ``df``
    and ``dpdz`` resolved algebraically (MATLAB used M(end-1,end-1)=0,
    M(end,end)=0)."""

    def __init__(self, par: WoParameters, fator: float = 1.0):
        self.par = par
        self.fator = float(fator)
        self._p = _pack_tube_params(par, fator)
        self.n = int(par.n)

    def state_length(self) -> int:
        return self.n + 1

    def f(self, u: np.ndarray):
        u = np.asarray(u, dtype=np.float64)
        n = self.n
        p = self._p

        def rhs(t, x):
            # Resolve df/dpdz algebraically, call the parity kernel, strip
            # the AE rows.
            full = np.empty(n + 3)
            full[:n + 1] = x
            df = p[6] * x[n] * 1000.0  # kf * Rfw * 1000
            Dcoil = p[2]
            Dcoildiff = Dcoil - 2.0 * df
            if Dcoildiff > Dcoil:
                Dcoildiff = Dcoil
            Acoil = 0.25 * np.pi * Dcoildiff ** 2
            vz = u[1] / Acoil
            ReC = p[4] * vz * Dcoildiff / p[5]
            th1 = -2.457 * np.log((7.0 / ReC) ** 0.9 + 0.27 * p[17] / Dcoildiff)
            th2 = (37530.0 / ReC) ** 16
            ff = 8.0 * ((8.0 / ReC) ** 12 + (1.0 / (th1 + th2) ** 1.5)) ** (1.0 / 12.0)
            dpdz = 0.5 * p[4] * vz ** 2 * ff * p[3] / Dcoildiff
            full[n + 1] = df
            full[n + 2] = dpdz
            out = _tube_kernel(t, full, u, p)
            return out[:n + 1]
        return rhs

    def initial_state(self, u: np.ndarray) -> np.ndarray:
        x0 = tube_initial_state(self.par, u, self.fator)
        return x0[:self.n + 1]

    def outputs(self, x: np.ndarray, u: np.ndarray) -> dict:
        """Recover df / dpdz / Tcout from the reduced state."""
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        par = self.par
        n = self.n
        Rfw = float(x[n])
        df = par.kf * Rfw * 1000.0
        Dcoil = par.Dcoil * self.fator
        Dcoildiff = Dcoil - 2.0 * df
        if Dcoildiff > Dcoil:
            Dcoildiff = Dcoil
        Acoil = 0.25 * np.pi * Dcoildiff ** 2
        vz = u[1] / Acoil
        ReC = par.rhoW * vz * Dcoildiff / par.viscW
        th1 = -2.457 * np.log((7.0 / ReC) ** 0.9 + 0.27 * par.epsilon / Dcoildiff)
        th2 = (37530.0 / ReC) ** 16
        f = 8.0 * ((8.0 / ReC) ** 12 + (1.0 / (th1 + th2) ** 1.5)) ** (1.0 / 12.0)
        dpdz = 0.5 * par.rhoW * vz ** 2 * f * par.Lcoil / Dcoildiff
        return {"Tcout": float(x[n - 1]), "Rf": Rfw, "df": df, "dpdz": dpdz}


if __name__ == "__main__":
    from .parameters import closed_loop_parameters
    par = closed_loop_parameters()
    # sim_tubo_fouling.m defaults
    u = np.array([288.75, 0.0013 * 0.1, 370.0, 50e-3 / 100.0, 9.0])
    sim = TubeSimulator(par, fator=1.0)
    x0 = sim.initial_state(u)
    dx = sim.f(u)(0.0, x0)
    print("reduced state length:", sim.state_length())
    print("max |dTc/dt|:", np.abs(dx[:par.n]).max(), "K/s")
    print("dRfw/dt:", dx[par.n], "m^2.K/(kJ.s)")
    print("outputs:", {k: round(v, 8) for k, v in sim.outputs(x0, u).items()})
