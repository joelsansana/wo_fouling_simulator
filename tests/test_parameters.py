"""Smoke tests for wo_fouling.parameters.

Run: ``pytest tests/test_parameters.py -v`` from the repo root.
"""

from __future__ import annotations

import numpy as np
import pytest

from wo_fouling.parameters import (
    WoParameters,
    ReactionKinetics,
    Thermo,
    Geometry,
    FoulingChemistry,
    default_parameters,
)


class TestWoParametersDefaults:
    """The defaults must match the upstream MATLAB reference bit-for-bit."""

    def test_returns_instance(self):
        par = default_parameters()
        assert isinstance(par, WoParameters)

    def test_reactor_holdup(self):
        # massT == 2105 kg, rhoR == 800.92 kg/m^3, Vr = massT / rhoR ~= 2.6288 m^3
        par = default_parameters()
        assert par.massT == pytest.approx(2105.0)
        assert par.rhoR == pytest.approx(800.92)
        assert par.Vr == pytest.approx(2105.0 / 800.92, rel=1e-12)

    def test_arrhenius_factors(self):
        # A1..A3 from upstream modelparameters.m
        par = default_parameters()
        assert par.A1 == pytest.approx(1.6599e6, rel=1e-12)
        assert par.A2 == pytest.approx(7.2117e8, rel=1e-12)
        assert par.A3 == pytest.approx(2.6745e12, rel=1e-12)

    def test_activation_energies(self):
        par = default_parameters()
        assert par.Ea1 == pytest.approx(6666.7)
        assert par.Ea2 == pytest.approx(8333.3)
        assert par.Ea3 == pytest.approx(11111.0)

    def test_heats_of_reaction(self):
        par = default_parameters()
        assert par.DeltaH1 == pytest.approx(-263.8)
        assert par.DeltaH2 == pytest.approx(-158.3)
        assert par.DeltaH3 == pytest.approx(-226.3)

    def test_heat_capacities(self):
        # CpA = 321.204 / 142, etc.
        par = default_parameters()
        assert par.CpA == pytest.approx(321.204 / 142, rel=1e-12)
        assert par.CpB == pytest.approx(127.14 / 60, rel=1e-12)
        assert par.CpC == pytest.approx(352.288 / 202, rel=1e-12)
        assert par.CpE == pytest.approx(166.212 / 81, rel=1e-12)
        assert par.CpP == pytest.approx(844.132 / 383, rel=1e-12)
        assert par.CpG == pytest.approx(426.617 / 181, rel=1e-12)
        assert par.CpW == pytest.approx(4.186)

    def test_coil_geometry(self):
        par = default_parameters()
        assert par.Lcoil == pytest.approx(116.4336)
        assert par.Dcoil == pytest.approx(0.0254)
        assert par.Acoil == pytest.approx(0.25 * np.pi * 0.0254 ** 2, rel=1e-12)

    def test_clean_overall_htc(self):
        par = default_parameters()
        assert par.U == pytest.approx(0.800)

    def test_fouling_chemistry_defaults(self):
        par = default_parameters()
        assert par.kf == pytest.approx(2.941e-3, rel=1e-12)
        assert par.rhof == pytest.approx(2710.0)
        assert par.CH == pytest.approx(29.5)
        assert par.ksp == pytest.approx(4.9e-9, rel=1e-12)
        assert par.yco2 == pytest.approx(0.0314e-2)
        assert par.psi == pytest.approx(2.5)
        assert par.K1 == pytest.approx(4.47e-7, rel=1e-12)
        assert par.K2 == pytest.approx(4.68e-11, rel=1e-12)


class TestDerivedProperties:
    """Computed reactor / coil geometry should match upstream exactly."""

    def test_acoil_computed(self):
        par = default_parameters()
        assert par.Acoil == pytest.approx(0.25 * np.pi * 0.0254 ** 2, rel=1e-12)

    def test_reactor_cross_section_area(self):
        # Ar derives from Vr + Acoil*Lcoil treated as cylindrical.
        # Vt = Vr + Acoil * Lcoil = 2.6288 + 5.0671e-4 * 116.4336 ~= 2.6878
        # Dr = (2 Vt / pi)^(1/3) ~= 1.1902
        # Ar = pi/4 Dr^2 ~= 1.1129
        par = default_parameters()
        Vt = par.Vr + par.Acoil * par.Lcoil
        Dr = (2.0 * Vt / np.pi) ** (1.0 / 3.0)
        Ar_expected = 0.25 * np.pi * Dr ** 2
        assert par.Ar == pytest.approx(Ar_expected, rel=1e-12)

    def test_pK1_pK2(self):
        par = default_parameters()
        assert par.fouling.pK1 == pytest.approx(-np.log10(4.47e-7), rel=1e-12)
        assert par.fouling.pK2 == pytest.approx(-np.log10(4.68e-11), rel=1e-12)


class TestMutableDefaults:
    """Default factories must give each instance its own dict/array (no aliasing)."""

    def test_default_factory_safety(self):
        p1 = default_parameters()
        p2 = default_parameters()
        # Independent containers — mutating one must not affect the other.
        assert p1.kinetics is not p2.kinetics
        assert p1.thermo is not p2.thermo


class TestFlatParameterVector:
    """The flat 31-vector must match the WO_dyn chart positions."""

    def test_shape(self):
        par = default_parameters()
        flat = par.to_flat()
        assert flat.shape == (31,)
        assert flat.dtype == np.float64

    def test_positions_match_chart_38(self):
        # chart_38.xml documents the parameter positions explicitly:
        # par(1)=rhoR, par(2..4)=A1..A3, par(5..7)=Ea1..Ea3, par(8..10)=dH1..dH3,
        # par(11..16)=CpA..CpG, par(17)=CpW, par(18)=Uclean, par(19)=rhoW,
        # par(20)=viscW, par(21)=Lcoil, par(22)=Dcoil,
        # par(23..31)=kf, Rgas, rhof, CH, ksp, yco2, psi, K1, K2
        par = default_parameters()
        flat = par.to_flat()
        assert flat[0] == pytest.approx(par.rhoR)
        assert flat[1] == pytest.approx(par.A1)
        assert flat[4] == pytest.approx(par.Ea1)
        assert flat[7] == pytest.approx(par.DeltaH1)
        # CpA..CpP,CpG at indices 10..15 (par(11..16) 1-based)
        assert flat[10] == pytest.approx(par.CpA)
        assert flat[15] == pytest.approx(par.CpG)
        # CpW at index 16 (par(17))
        assert flat[16] == pytest.approx(par.CpW)
        # Uclean at index 17 (par(18))
        assert flat[17] == pytest.approx(par.U)
        # rhoW at index 18 (par(19))
        assert flat[18] == pytest.approx(par.rhoW)
        # viscW at index 19 (par(20))
        assert flat[19] == pytest.approx(par.viscW)
        # Lcoil at index 20 (par(21))
        assert flat[20] == pytest.approx(par.Lcoil)
        # Dcoil at index 21 (par(22))
        assert flat[21] == pytest.approx(par.Dcoil)
        # fouling block at 22..30 (par(23..31))
        assert flat[22] == pytest.approx(par.kf)
        assert flat[23] == pytest.approx(par.Rgas)
        assert flat[30] == pytest.approx(par.K2)


class TestSubgroupDefaults:
    """Each subgroup dataclass should be constructible with no args."""

    def test_kinetics_defaults(self):
        k = ReactionKinetics()
        assert k.A1 == pytest.approx(1.6599e6, rel=1e-12)

    def test_thermo_defaults(self):
        t = Thermo()
        assert t.rhoR == pytest.approx(800.92)

    def test_geometry_defaults(self):
        g = Geometry()
        assert g.n == 200
        assert g.epsilon == pytest.approx(0.0015)

    def test_fouling_defaults(self):
        f = FoulingChemistry()
        assert f.psi == pytest.approx(2.5)


if __name__ == "__main__":
    # Allow `python tests/test_parameters.py` for ad-hoc runs.
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
