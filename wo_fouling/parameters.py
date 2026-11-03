"""
WO reactor parameters — port of ``modelparameters.m``.

The defaults reproduce exactly the values from the upstream ``modelparameters.m``
(open-loop and closed-loop versions are bit-identical except for the closed-loop
alias ``MR = par.massT`` which we ignore — see docs/AUDIT.md §5.1).

References:
[1] TJ Williams, RE Otto, Trans Amer IEE,
    Part I: Communication and Electronics 79, 458-473, 1960.
[3] ARC Souza, ALH Costa, Chem Eng Res Des 121, 15-31, 2019.  -- fouling kinetics
[5] L Cresmaschi, X Wu, Heat Transfer Engineering 36, 663-675, 2015.  -- psi
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# -----------------------------------------------------------------------------
# Reaction chemistry (Williams-Otto, 1960)
# -----------------------------------------------------------------------------

@dataclass
class ReactionKinetics:
    """Arrhenius kinetics for the three WO reactions.

    A + B -> C         (rate r1)
    C + B -> E + P     (rate r2)
    C + P -> G         (rate r3)
    """

    # Frequency factors, s^-1 kg
    A1: float = 1.6599e6
    A2: float = 7.2117e8
    A3: float = 2.6745e12
    # Activation energies, K  (note: original .m uses K, not J/mol; Rgas below is in cal/(K mol))
    Ea1: float = 6666.7
    Ea2: float = 8333.3
    Ea3: float = 11111.0
    # Heats of reaction, kJ/kg-X (negative = exothermic)
    DeltaH1: float = -263.8   # per kg A consumed
    DeltaH2: float = -158.3   # per kg B consumed
    DeltaH3: float = -226.3   # per kg C consumed


# -----------------------------------------------------------------------------
# Heat capacities and densities
# -----------------------------------------------------------------------------

@dataclass
class Thermo:
    """Species-specific heat capacities and phase densities."""

    # Heat capacities, kJ/(kg K). Sourced from KUMMER 2020 via upstream.
    CpA: float = 321.204 / 142
    CpB: float = 127.14 / 60
    CpC: float = 352.288 / 202
    CpE: float = 166.212 / 81
    CpP: float = 844.132 / 383
    CpG: float = 426.617 / 181
    CpW: float = 4.186  # water

    # Densities, kg/m^3
    rhoR: float = 800.92   # reacting mixture (assumed constant)
    rhoW: float = 1000.0   # cooling water


# -----------------------------------------------------------------------------
# Reactor and coil geometry
# -----------------------------------------------------------------------------

@dataclass
class Geometry:
    """Reactor and cooling coil geometric constants."""

    Uclean: float = 0.800          # kW/(m^2 K) — clean overall heat-transfer coefficient
    Lcoil: float = 116.4336        # m — total coil length
    Dcoil: float = 0.0254          # m — coil inner tube diameter
    viscW: float = 8e-4            # Pa s — water dynamic viscosity (treated as constant)
    massT: float = 2105.0          # kg — total reactor hold-up
    epsilon: float = 0.0015        # pipe roughness (Churchill friction factor)
    n: int = 200                   # coil cells (default matches WO_dyn hardcoded N)

    @property
    def Acoil(self) -> float:
        """Coil cross-section area, m^2."""
        return 0.25 * np.pi * self.Dcoil ** 2


# -----------------------------------------------------------------------------
# Fouling kinetics — Souza & Costa (2019)
# -----------------------------------------------------------------------------

@dataclass
class FoulingChemistry:
    """CaCO3 precipitation–removal kinetics.

    All parameters from [3] Souza & Costa, ChemEngResDes 121, 15-31, 2019,
    except psi which is from [5] Cresmaschi & Wu, Heat Transfer Engineering 36,
    663-675, 2015.
    """

    kf: float = 2.941e-3           # kW/(m K) — fouling-layer thermal conductivity
    Rgas: float = 1.98720425864083 # cal/(K mol) — gas constant (note units: cal!)
    rhof: float = 2710.0           # kg/m^3 — fouling-layer density (CaCO3)
    CH: float = 29.5               # atm L/mol — Henry's constant for CO2
    ksp: float = 4.9e-9            # mol^2/L^2 — CaCO3 solubility product
    yco2: float = 0.0314e-2        # mol fraction CO2 in water
    psi: float = 2.5               # mechanical resistance parameter [5]
    K1: float = 4.47e-7            # carbonate equilibrium 1
    K2: float = 4.68e-11           # carbonate equilibrium 2

    # Convenience: -log10 of the equilibria, used internally by the kinetics.
    @property
    def pK1(self) -> float:
        return -np.log10(self.K1)

    @property
    def pK2(self) -> float:
        return -np.log10(self.K2)

    @property
    def pCH(self) -> float:
        return -np.log10(self.CH)

    @property
    def pksp(self) -> float:
        return -np.log10(self.ksp)


# -----------------------------------------------------------------------------
# Top-level parameter container
# -----------------------------------------------------------------------------

@dataclass
class WoParameters:
    """All parameters for the Williams-Otto reactor with CaCO3 fouling.

    Defaults reproduce the upstream MATLAB reference exactly. Override
    fields to study parametric variations.
    """

    kinetics: ReactionKinetics = field(default_factory=ReactionKinetics)
    thermo: Thermo = field(default_factory=Thermo)
    geometry: Geometry = field(default_factory=Geometry)
    fouling: FoulingChemistry = field(default_factory=FoulingChemistry)

    # Operating-point initial conditions (used by sim_main.m as nominal inputs)
    FA: float = 1.8275         # kg/s — A feed mass flow
    FB: float = 4.7875         # kg/s — B feed mass flow
    TA: float = 294.3          # K — A feed temperature
    TB: float = 294.3          # K — B feed temperature
    Tcin: float = 288.75       # K — cooling water inlet temperature
    Qcin: float = 1.25e-3      # m^3/s — cooling water volumetric flow (closed-loop)
    # Open-loop sim_main.m uses Qcin = 1.5e-3 m^3/s; closed-loop uses 1.25e-3.
    ca: float = 150e-5         # mol/L — Ca2+ concentration in cooling water
    pH: float = 9.0            # — cooling water pH

    # Cross-section-dependent reactor geometry (Vr, Ar) live on the top-level
    # container because they need `rhoR` from Thermo, not just Geometry.

    @property
    def Vr(self) -> float:
        return self.geometry.massT / self.thermo.rhoR

    @property
    def Ar(self) -> float:
        Vt = self.Vr + self.geometry.Acoil * self.geometry.Lcoil
        Dr = (2.0 * Vt / np.pi) ** (1.0 / 3.0)
        return 0.25 * np.pi * Dr ** 2

    # ------------------------------------------------------------------------
    # Convenience properties matching the upstream `par.<field>` access pattern
    # ------------------------------------------------------------------------
    @property
    def A1(self) -> float: return self.kinetics.A1
    @property
    def A2(self) -> float: return self.kinetics.A2
    @property
    def A3(self) -> float: return self.kinetics.A3
    @property
    def Ea1(self) -> float: return self.kinetics.Ea1
    @property
    def Ea2(self) -> float: return self.kinetics.Ea2
    @property
    def Ea3(self) -> float: return self.kinetics.Ea3
    @property
    def DeltaH1(self) -> float: return self.kinetics.DeltaH1
    @property
    def DeltaH2(self) -> float: return self.kinetics.DeltaH2
    @property
    def DeltaH3(self) -> float: return self.kinetics.DeltaH3

    @property
    def CpA(self) -> float: return self.thermo.CpA
    @property
    def CpB(self) -> float: return self.thermo.CpB
    @property
    def CpC(self) -> float: return self.thermo.CpC
    @property
    def CpE(self) -> float: return self.thermo.CpE
    @property
    def CpP(self) -> float: return self.thermo.CpP
    @property
    def CpG(self) -> float: return self.thermo.CpG
    @property
    def CpW(self) -> float: return self.thermo.CpW
    @property
    def rhoR(self) -> float: return self.thermo.rhoR
    @property
    def rhoW(self) -> float: return self.thermo.rhoW

    @property
    def U(self) -> float: return self.geometry.Uclean
    @property
    def Lcoil(self) -> float: return self.geometry.Lcoil
    @property
    def Dcoil(self) -> float: return self.geometry.Dcoil
    @property
    def Acoil(self) -> float: return self.geometry.Acoil
    @property
    def viscW(self) -> float: return self.geometry.viscW
    @property
    def massT(self) -> float: return self.geometry.massT
    @property
    def n(self) -> int: return self.geometry.n
    @property
    def epsilon(self) -> float: return self.geometry.epsilon

    @property
    def kf(self) -> float: return self.fouling.kf
    @property
    def Rgas(self) -> float: return self.fouling.Rgas
    @property
    def rhof(self) -> float: return self.fouling.rhof
    @property
    def CH(self) -> float: return self.fouling.CH
    @property
    def ksp(self) -> float: return self.fouling.ksp
    @property
    def yco2(self) -> float: return self.fouling.yco2
    @property
    def psi(self) -> float: return self.fouling.psi
    @property
    def K1(self) -> float: return self.fouling.K1
    @property
    def K2(self) -> float: return self.fouling.K2

    def to_flat(self) -> np.ndarray:
        """Pack all 31 parameters into the flat vector the WO_dyn Simulink
        chart consumes. Positions match chart_38.xml order.

        Returns
        -------
        np.ndarray, shape (31,), float64
            [rhoR, A1, A2, A3, Ea1, Ea2, Ea3, dH1, dH2, dH3,
             CpA..CpP, CpG, CpW, Uclean, rhoW, viscW,
             Lcoil, Dcoil, kf, Rgas, rhof, CH, ksp, yco2, psi, K1, K2]
        """
        return np.array([
            self.rhoR,
            self.A1, self.A2, self.A3,
            self.Ea1, self.Ea2, self.Ea3,
            self.DeltaH1, self.DeltaH2, self.DeltaH3,
            self.CpA, self.CpB, self.CpC, self.CpE, self.CpP, self.CpG,
            self.CpW,
            self.U,         # Uclean position 18 in WO_dyn par vector
            self.rhoW,
            self.viscW,
            self.Lcoil,
            self.Dcoil,
            self.kf,
            self.Rgas,
            self.rhof,
            self.CH,
            self.ksp,
            self.yco2,
            self.psi,
            self.K1,
            self.K2,
        ], dtype=np.float64)


# -----------------------------------------------------------------------------
# Default accessor
# -----------------------------------------------------------------------------

def default_parameters() -> WoParameters:
    """Return WoParameters() with upstream defaults. Equivalent to
    ``[par, massT, Vr] = modelparameters()`` in the MATLAB reference."""
    return WoParameters()


if __name__ == "__main__":
    # Quick CLI sanity check — `python -m wo_fouling.parameters`.
    par = default_parameters()
    print("WoParameters default loadout:")
    print(f"  rhoR  = {par.rhoR} kg/m^3")
    print(f"  Vr    = {par.Vr:.4f} m^3")
    print(f"  Ar    = {par.Ar:.6f} m^2")
    print(f"  Acoil = {par.Acoil:.6e} m^2")
    print(f"  A1..A3 = {par.A1:.3e}, {par.A2:.3e}, {par.A3:.3e}")
    print(f"  Ea1..Ea3 = {par.Ea1}, {par.Ea2}, {par.Ea3}  K")
    print(f"  flat par shape = {par.to_flat().shape}")
