"""Smoke tests for wo_fouling.kinetics.

Run: ``pytest tests/test_kinetics.py -v`` from the repo root.
"""

from __future__ import annotations

import numpy as np
import pytest

from wo_fouling.kinetics import (
    log_rate_constants,
    mass_rates,
    reaction_heat,
)
from wo_fouling.parameters import default_parameters


class TestLogRateConstants:
    """Verify ln(kj) matches the upstream ``dynmodel.m`` block, line by line."""

    def test_arrays_in_arrays_out(self):
        par = default_parameters()
        Tr = np.array([330.0, 368.81, 400.0])
        lnkl1, lnkl2, lnkl3 = log_rate_constants(Tr, par)
        assert lnkl1.shape == (3,)
        assert lnkl2.shape == (3,)
        assert lnkl3.shape == (3,)

    def test_scalar_in_scalar_out(self):
        par = default_parameters()
        # At Tr = 368.81 K (the upstream nominal SS), the three log-rate
        # constants should match a manual hand calculation.
        lnkl1, lnkl2, lnkl3 = log_rate_constants(368.81, par)
        # Manual: log(A1) - Ea1/Tr, etc.
        Trinv = 1.0 / 368.81
        assert float(lnkl1) == pytest.approx(np.log(par.A1) - par.Ea1 * Trinv)
        assert float(lnkl2) == pytest.approx(np.log(par.A2) - par.Ea2 * Trinv)
        assert float(lnkl3) == pytest.approx(np.log(par.A3) - par.Ea3 * Trinv)

    def test_increases_with_temperature(self):
        """Arrhenius: kj grows monotonically with Tr (positive Ea)."""
        par = default_parameters()
        Tr = np.array([300.0, 350.0, 400.0, 450.0])
        lnkl1, lnkl2, lnkl3 = log_rate_constants(Tr, par)
        # Each log k is monotonically increasing.
        assert np.all(np.diff(lnkl1) > 0)
        assert np.all(np.diff(lnkl2) > 0)
        assert np.all(np.diff(lnkl3) > 0)

    def test_relative_ordering(self):
        """At a given Tr, lnkl1 < lnkl2 < lnkl3 (matches A1 < A2 < A3 ordering
        with Ea1 < Ea2 < Ea3 — but the A_j grows faster than Ea_j, so k3 > k2 > k1)."""
        par = default_parameters()
        lnkl1, lnkl2, lnkl3 = log_rate_constants(368.81, par)
        assert lnkl1 < lnkl2 < lnkl3

    def test_floors_negative_temperature(self):
        """Defensive floor: Tr -> 0 K is floored to 1 K to avoid div-by-zero."""
        par = default_parameters()
        lnkl1, _, _ = log_rate_constants(0.0, par)
        # At Tr=1K (the floor), lnkl1 = log(A1) - Ea1*1.0
        assert np.isfinite(lnkl1)


class TestMassRates:
    """Verify mass-basis rates match the upstream ``dynmodel.m`` block."""

    def test_zero_for_zero_mass_fraction(self):
        par = default_parameters()
        # If any reactant is absent, the corresponding rate is zero.
        r1mr, r2mr, r3mr = mass_rates(0.0, 0.5, 0.5, 0.1, 1.0, 368.81, par)
        assert r1mr == pytest.approx(0.0)
        # r2mr = k2 * xb * xc * MR — nonzero if xb, xc are nonzero
        # r3mr = k3 * xc * xp * MR — nonzero if xc, xp are nonzero
        assert r2mr > 0
        assert r3mr > 0

    def test_zero_for_empty_reactor(self):
        par = default_parameters()
        # MR -> 0 should be floored to 1e-10, producing a tiny but nonzero rate.
        r1mr, r2mr, r3mr = mass_rates(0.5, 0.5, 0.0, 0.0, 0.0, 368.81, par)
        # r1 is the only nonzero; with xc=0, r2 = 0 exactly; with xp=0, r3 = 0.
        assert np.isfinite(r1mr)
        assert r2mr == pytest.approx(0.0)
        assert r3mr == pytest.approx(0.0)

    def test_reaction_rates_dimensions(self):
        """Verify rates are in kg/s — i.e. scale linearly with MR."""
        par = default_parameters()
        MR1, MR2 = 100.0, 200.0
        r1_a, _, _ = mass_rates(0.5, 0.5, 0.0, 0.0, MR1, 368.81, par)
        r1_b, _, _ = mass_rates(0.5, 0.5, 0.0, 0.0, MR2, 368.81, par)
        assert r1_b == pytest.approx(2.0 * r1_a, rel=1e-12)

    def test_matches_manual_calculation(self):
        """Hand calculation at Tr = 368.81 K, MR = 2105 kg, xa=xb=0.5."""
        par = default_parameters()
        Tr = 368.81
        MR = 2105.0
        xa = xb = 0.5
        xc = xp = 0.0
        Trinv = 1.0 / Tr
        k1 = np.exp(np.log(par.A1) - par.Ea1 * Trinv)
        r1mr_expected = k1 * xa * xb * MR
        r1mr, r2mr, r3mr = mass_rates(xa, xb, xc, xp, MR, Tr, par)
        assert r1mr == pytest.approx(r1mr_expected, rel=1e-12)
        # r2 and r3 are zero (no C or P)
        assert r2mr == pytest.approx(0.0)
        assert r3mr == pytest.approx(0.0)


class TestReactionHeat:
    """Verify the heat-release expression matches ``dynmodel.m`` exactly."""

    def test_known_case(self):
        """Hand-calculate the heat release for a given (r1, r2, r3)."""
        par = default_parameters()
        r1mr, r2mr, r3mr = 1.0, 1.0, 1.0
        # Qrmr = 2 * 1 * (-dH1) + 3 * 1 * (-dH2) + 1.5 * 1 * (-dH3)
        expected = (
            2.0 * (-par.DeltaH1)
            + 3.0 * (-par.DeltaH2)
            + 1.5 * (-par.DeltaH3)
        )
        assert reaction_heat(r1mr, r2mr, r3mr, par) == pytest.approx(expected, rel=1e-12)

    def test_sign_convention(self):
        """DeltaH is negative (exothermic). Qrmr should be positive."""
        par = default_parameters()
        Qrmr = reaction_heat(1.0, 1.0, 1.0, par)
        assert Qrmr > 0

    def test_array_input(self):
        par = default_parameters()
        r1 = np.array([0.5, 1.0, 1.5])
        r2 = np.array([0.5, 1.0, 1.5])
        r3 = np.array([0.5, 1.0, 1.5])
        Qrmr = reaction_heat(r1, r2, r3, par)
        assert Qrmr.shape == (3,)
        # Linear in r1+r2+r3 with the same coefficients.
        expected = (
            2.0 * r1 * (-par.DeltaH1)
            + 3.0 * r2 * (-par.DeltaH2)
            + 1.5 * r3 * (-par.DeltaH3)
        )
        assert np.allclose(Qrmr, expected, rtol=1e-12)


class TestParityWithAudit:
    """Cross-checks against the values documented in the prior audit
    (removed with the MATLAB phase-out; see PORT_NOTES.md §4)."""

    def test_arrhenius_factors_audit(self):
        """A1..A3 and Ea1..Ea3 are the values cited in the prior audit's
        physics walkthrough (git history)."""
        par = default_parameters()
        # These exact numbers appear in the prior audit's physics walkthrough.
        assert par.A1 == pytest.approx(1.6599e6)
        assert par.A2 == pytest.approx(7.2117e8)
        assert par.A3 == pytest.approx(2.6745e12)
        assert par.Ea1 == pytest.approx(6666.7)
        assert par.Ea2 == pytest.approx(8333.3)
        assert par.Ea3 == pytest.approx(11111.0)
