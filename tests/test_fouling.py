"""Golden + smoke tests for wo_fouling.fouling — the dynmodel_fouling.m port.

Validation stance (no MATLAB on this machine — see PORT_NOTES.md):
* hand-computed fouling-kinetics values at the slx-embedded steady state
  (arithmetic independently done from the .m equations);
* consistency between the parity layer (11+n MATLAB state) and the
  integrated layer (8+n differential states);
* the algebraic constraints g1/g2/g3 are identically zero at a consistent
  state and df/dpdz track Rfw as the .m prescribes;
* a 24 h fouling run shows monotone Rf growth, the expected U decay, and
  pressure-drop growth.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from wo_fouling.parameters import closed_loop_parameters, open_loop_parameters
from wo_fouling.fouling import (
    FoulingReactor, dynmodel_fouling, mass_matrix, _pack_fouling_params,
    _fouling_kernel,
)
from wo_fouling.closed_loop import _SLX_IC_HEAD


PAR = closed_loop_parameters()


def _steady_state_state():
    """Plant state at the slx-embedded steady state (masses, Tr, analytic
    Tc profile, Rf = 0)."""
    r = FoulingReactor(PAR)
    Tr0 = _SLX_IC_HEAD[6]
    zspan = np.linspace(0, PAR.Lcoil, PAR.n)
    Tc0 = Tr0 + (288.75 - Tr0) * np.exp(
        -PAR.U * np.pi * PAR.Dcoil * zspan / (PAR.rhoW * PAR.CpW * 1.25e-3))
    x = r.initial_state(*_SLX_IC_HEAD[:6], Tr0, Tc0, Rfw=0.0)
    return x, r


class TestFoulingKinetics:
    """Hand-computed fouling chemistry at the clean steady state."""

    def setup_method(self):
        self.x0, self.r = _steady_state_state()
        self.u = self.r.nominal_inputs()

    def test_chemistry_hand_computed(self):
        """co3, phid, phir, dRfw/dt by hand from the .m equations.

        ca = 150e-5, pH = 9, yco2 = 0.0314e-2, CH = 29.5, K1 = 4.47e-7,
        K2 = 4.68e-11, Tc_mean = (Tcout + Tcin)/2 with Tcout = Tc(200).
        """
        u = self.u
        Tc_mean = (self.x0[7 + PAR.n - 1] + u[4]) / 2.0
        co2aq = 0.0314e-2 / 29.5
        H = 10.0 ** (-9.0)
        hco3 = co2aq * 4.47e-7 / H
        co3f = hco3 * 4.68e-11 / H
        ca = 150e-5
        co3 = ca + co3f

        kf, psi, rhof = 2.941e-3, 2.5, 2710.0
        Dcoildiff = 0.0254 - 2 * (kf * 0.0 * 1000)
        Acoil = 0.25 * np.pi * Dcoildiff ** 2
        vz = 1.25e-3 / Acoil
        ReC = 1000.0 * vz * Dcoildiff / 8e-4
        kr = np.exp(38.74 - 20700.0 / (1.98720425864083 * Tc_mean))
        Dif = 3.07e-15 * Tc_mean / 8e-4
        Sc = 8e-4 / (1000.0 * Dif)
        kd = 0.023 * vz * ReC ** -0.17 * Sc ** -0.67
        num = 1 - 4.9e-9 / (ca * co3)
        den = 1 + kd / (kr * co3) + co3 / ca
        phid = kd * co3 * num / den
        phir = (0.00212 * vz ** 2 / (np.sqrt(kf) * psi)) * phid
        dRfw = (phid - phir) / (rhof * kf)

        # kernel value at the same state
        d = self.r.f(u)(0.0, self.x0)
        assert d[7 + PAR.n] == pytest.approx(dRfw, rel=1e-12)

    def test_U_decreases_with_Rfw(self):
        """U = 1/(1/Uclean + 1000 Rf) at Rf = 3e-4 (the fouled sysid point)."""
        Rf = 3e-4
        U = 1.0 / (1.0 / 0.8 + Rf * 1000.0)
        assert U == pytest.approx(0.8 / (1.0 + 0.8 * Rf * 1000.0))
        assert U < 0.8
        alg = self.r.algebraic(
            np.concatenate([self.x0[:7 + PAR.n], [Rf]]), self.u)
        assert alg["U"] == pytest.approx(U, rel=1e-12)

    def test_df_tracks_Rfw(self):
        for Rf in (0.0, 1e-4, 3e-4, 1e-3):
            alg = self.r.algebraic(
                np.concatenate([self.x0[:7 + PAR.n], [Rf]]), self.u)
            assert alg["df"] == pytest.approx(2.941e-3 * Rf * 1000.0,
                                              rel=1e-12)

    def test_dpdz_matches_recorded_run(self):
        """dpdz at the clean steady state matches the DeltaP channel of
        sim_results.mat (~8.2e7 Pa mean of the noisy recording)."""
        alg = self.r.algebraic(self.x0, self.u)
        assert alg["dpdz"] == pytest.approx(8.2e7, rel=0.05)


class TestParityLayer:
    """dynmodel_fouling parity: full 11+n state with AE rows."""

    def setup_method(self):
        self.x0, self.r = _steady_state_state()
        self.u = self.r.nominal_inputs()

    def test_shapes(self):
        n = PAR.n
        x_full = np.concatenate([[2.3395], self.x0, [0.0, 0.0, 8.3e7]])
        out = dynmodel_fouling(0.0, x_full, self.u, PAR)
        assert out.shape == (11 + n,)
        M = mass_matrix(11 + n, n)
        # rows 0, end-1, end are AE (zero mass rows)
        assert M[0, 0] == 0.0 and M[-1, -1] == 0.0 and M[-2, -2] == 0.0
        assert M[4, 4] == 1.0

    def test_ae_residuals_zero_at_consistent_state(self):
        alg = self.r.algebraic(self.x0, self.u)
        x_full = np.concatenate([[alg["L"]], self.x0,
                                 [alg["df"], alg["dpdz"]]])
        out = dynmodel_fouling(0.0, x_full, self.u, PAR)
        assert out[0] == pytest.approx(0.0, abs=1e-9)
        assert out[9 + PAR.n] == pytest.approx(0.0, abs=1e-6)
        assert out[10 + PAR.n] == pytest.approx(0.0, rel=1e-9)

    def test_differential_rows_match_kernel(self):
        alg = self.r.algebraic(self.x0, self.u)
        x_full = np.concatenate([[alg["L"]], self.x0,
                                 [alg["df"], alg["dpdz"]]])
        out = dynmodel_fouling(0.0, x_full, self.u, PAR)
        d = self.r.f(self.u)(0.0, self.x0)
        np.testing.assert_allclose(out[1:9 + PAR.n], d, rtol=1e-12, atol=1e-14)

    def test_kernel_does_not_mutate_state(self):
        x = self.x0.copy()
        self.r.f(self.u)(0.0, x)
        np.testing.assert_array_equal(x, self.x0)


class TestIntegration:
    def test_24h_fouling_growth(self):
        """24 h fouling run from the clean SS: Rf grows monotonically,
        dpdz(24h)/dpdz(0) > 1."""
        r = FoulingReactor(PAR)
        u = r.nominal_inputs()
        sol = solve_ivp(r.f(u), (0.0, 24 * 3600.0), _steady_state_state()[0],
                        method="BDF", rtol=1e-6, atol=1e-8)
        assert sol.success
        Rf = sol.y[7 + PAR.n]
        assert Rf[-1] > Rf[0]
        assert np.all(np.diff(Rf) >= -1e-15)  # monotone (no removal flip)
        alg0 = r.algebraic(sol.y[:, 0], u)
        alg1 = r.algebraic(sol.y[:, -1], u)
        assert alg1["dpdz"] > alg0["dpdz"]
        assert alg1["U"] < alg0["U"]
