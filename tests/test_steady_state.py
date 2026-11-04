"""Tests for the steady-state solvers (init.m / ssmodel.m / initial.m)."""

from __future__ import annotations

import numpy as np
import pytest

from wo_fouling.parameters import closed_loop_parameters, open_loop_parameters
from wo_fouling.steady_state import (
    solve_open_loop_ss, solve_closed_loop_ss, init_residual, ssmodel_residual,
    UPSTREAM_XI0, UPSTREAM_TR0,
)
from wo_fouling.reactor import OpenLoopReactor
from wo_fouling.fouling import FoulingReactor


class TestOpenLoopSS:
    def setup_method(self):
        self.par = open_loop_parameters()
        self.u = np.array([1.8275, 4.7875, 294.3, 294.3, 288.75,
                           (1.8275 + 4.7875) / self.par.rhoR, 1.5e-3])

    def test_converges_to_residual_tolerance(self):
        x_ss, info = solve_open_loop_ss(self.u, self.par)
        assert info["max_residual"] < 1e-10

    def test_rhs_zero_at_ss(self):
        """The ported dynmodel.m RHS is ~0 at the solved steady state —
        the self-consistency check that replaces a MATLAB cross-run."""
        x_ss, _ = solve_open_loop_ss(self.u, self.par)
        d = OpenLoopReactor(self.par).f(self.u)(0.0, x_ss)
        assert np.abs(d[1:7]).max() < 1e-8     # kg/s
        assert abs(d[7]) < 1e-7                # K/s
        assert np.abs(d[8:]).max() < 1e-5      # K/s
        # level constraint satisfied: L = MR/(rhoR*Ar)
        MR = x_ss[1:7].sum()
        assert x_ss[0] == pytest.approx(MR / self.par.rhoR / self.par.Ar,
                                        rel=1e-10)

    def test_upstream_guess_shape(self):
        assert UPSTREAM_XI0.shape == (6,)
        assert UPSTREAM_TR0 == 368.81


class TestClosedLoopSS:
    """Cross-validation: the solved closed-loop steady state against the
    golden IC embedded in WO_reactor_model.slx (Integrator_ODE)."""

    def setup_method(self):
        self.par = closed_loop_parameters()
        self.u = np.array([1.8275, 4.7875, 294.3, 294.3, 288.75,
                           (1.8275 + 4.7875) / self.par.rhoR,
                           1.25e-3, 150e-5, 9.0])

    def test_converges_all_stages(self):
        x0, info = solve_closed_loop_ss(self.u, self.par)
        assert all(i == 1 for i in info["ier"])
        assert info["max_residual"] < 1e-10

    def test_matches_slx_embedded_ic(self):
        """initial.m reproduces the .slx steady state:
        Lr0 = 2.3395, Tr0 ~ 371.0948, masses = slx masses."""
        from wo_fouling.closed_loop import _slx_ic_208
        x0, _ = solve_closed_loop_ss(self.u, self.par)
        slx = _slx_ic_208()
        assert x0[0] == pytest.approx(2.3395, abs=1e-4)       # LR_SP const
        assert x0[7] == pytest.approx(slx[6], abs=2e-3)       # Tr0
        np.testing.assert_allclose(x0[1:7], slx[:6], rtol=2e-3)

    def test_rhs_zero_at_ss(self):
        x0, _ = solve_closed_loop_ss(self.u, self.par)
        xd = np.concatenate([x0[1:8 + self.par.n],
                             x0[8 + self.par.n:9 + self.par.n]])
        r = FoulingReactor(self.par)
        d = r.f(self.u[:7])(0.0, xd)
        assert np.abs(d[0:6]).max() < 1e-6
        assert abs(d[6]) < 1e-6
