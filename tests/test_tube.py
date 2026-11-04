"""Tests for the tube sim (tubo_fouling.m port) and steady-state solvers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from wo_fouling.parameters import closed_loop_parameters
from wo_fouling.tube import TubeSimulator, tube_initial_state, tubo_fouling

PAR = closed_loop_parameters()


class TestTube:
    def setup_method(self):
        # sim_tubo_fouling.m defaults
        self.u = np.array([288.75, 0.0013 * 0.1, 370.0, 50e-3 / 100.0, 9.0])
        self.sim = TubeSimulator(PAR, fator=1.0)

    def test_parity_rhs_shapes(self):
        x0 = tube_initial_state(PAR, self.u, 1.0)
        out = tubo_fouling(0.0, x0, self.u, PAR, 1.0)
        assert out.shape == (PAR.n + 3,)
        # clean start: g10 = kf*0*1000 - 0 = 0
        assert out[PAR.n + 1] == pytest.approx(0.0, abs=1e-15)
        # g11 = dpdz_clean - dpdz0 = 0 by construction of the IC
        assert out[PAR.n + 2] == pytest.approx(0.0, rel=1e-12)

    def test_analytic_profile_is_steady(self):
        """At the analytic clean profile with vz>0 the coil residuals
        dTc/dt are small. NOT zero: the centered difference of an
        exponential has O((k dz)^2) truncation, ~2e-3 K/s here — which is
        exactly why upstream stage 1 re-solves the coil numerically after
        the analytic stage 0."""
        x0 = tube_initial_state(PAR, self.u, 1.0)
        out = tubo_fouling(0.0, x0, self.u, PAR, 1.0)
        assert np.abs(out[:PAR.n]).max() < 5e-3

    def test_one_month_fouling(self):
        """1 month: Rf grows, df = kf*Rf*1000, dpdz ratio grows.
        (Monotonicity is checked at solver-tolerance level: BDF with
        rtol=1e-6 jitters ~1e-10 on a slowly-growing Rf.)"""
        sol = solve_ivp(self.sim.f(self.u), (0.0, 60 * 24 * 3600.0),
                        self.sim.initial_state(self.u),
                        method="BDF", rtol=1e-6, atol=1e-6)
        assert sol.success
        n = PAR.n
        Rf = sol.y[n]
        assert Rf[-1] > 0.0
        assert np.all(np.diff(Rf) >= -1e-8)   # no physical decline
        assert Rf[-1] - Rf[1] > 0.5 * (Rf[np.argmin(np.abs(sol.t - 86400))]
                                       - Rf[1])  # sustained growth
        o0 = self.sim.outputs(sol.y[:, 0], self.u)
        o1 = self.sim.outputs(sol.y[:, -1], self.u)
        assert o1["df"] == pytest.approx(2.941e-3 * Rf[-1] * 1000.0, rel=1e-9)
        assert o1["dpdz"] / o0["dpdz"] > 1.0
