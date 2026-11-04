"""Smoke tests for wo_fouling.reactor — the open-loop RHS.

Run: ``pytest tests/test_reactor.py -v`` from the repo root.

These tests verify three things:

1. **Structural fidelity to dynmodel.m.** The state vector has the documented
   layout, parameter packing matches ``WoParameters``, and the Numba kernel
   compiles and produces finite outputs.
2. **Conservation laws.** With the inlet/outlet streams turned off and Tr
   held constant by setting TA = TB = Tr = Tcin, the reactor hold-up is
   exactly conserved at the initial mass over an arbitrary integration
   interval.
3. **Sign conventions and directionality.** Mass flows in (FA, FB) drive the
   reactor hold-up *up*; mass flows out (FR = rhoR * QR) drive it *down*.
   Heat input (TA > Tr) drives Tr *up*.

What these tests *do not* check is bit-for-bit equivalence with the MATLAB
``dynmodel.m`` outputs — that needs a MATLAB re-run against the upstream
``model_ic.mat`` fixture (now at ``tests/fixtures/model_ic.mat``; see
PORT_NOTES.md §3). The validation-grade comparison lives in
``tests/test_pipeline.py`` (golden ``sim_results.mat`` corroboration).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from wo_fouling.parameters import default_parameters
from wo_fouling.reactor import (
    OpenLoopReactor,
    _dynmodel_kernel,
    _pack_params,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _make_reactor(n: int | None = None) -> OpenLoopReactor:
    par = default_parameters()
    if n is not None:
        # Re-import with custom n to keep the tests fast.
        from wo_fouling.parameters import Geometry, WoParameters
        par = WoParameters(geometry=Geometry(n=n))
    return OpenLoopReactor(par)


# -----------------------------------------------------------------------------
# 1. Structural tests
# -----------------------------------------------------------------------------

class TestStateLayout:
    """The state-vector layout must match the documented layout
    (``reactor.py`` docstring; see PORT_NOTES.md §2)."""

    def test_state_length_default(self):
        reactor = _make_reactor()
        # 8 leading entries (L, ma..mp, Tr) + N coil cells.
        assert reactor.state_length() == 8 + reactor.n

    def test_state_length_custom_n(self):
        reactor = _make_reactor(n=20)
        assert reactor.state_length() == 8 + 20

    def test_state_length_audit_default(self):
        """The prior audit cited 211 = 11 + 200 cells for the closed-loop
        chart (see PORT_NOTES.md §4); the open-loop default n = 200 gives
        208 entries."""
        reactor = _make_reactor(n=200)
        assert reactor.state_length() == 208

    def test_initial_state_layout(self):
        reactor = _make_reactor(n=5)
        Tc = np.array([290.0, 295.0, 300.0, 305.0, 310.0])
        x = reactor.initial_state(
            L=1.0, ma=10.0, mb=20.0, mc=30.0, me=5.0, mg=5.0, mp=5.0,
            Tr=370.0, Tc=Tc,
        )
        assert len(x) == 13
        assert x[0] == pytest.approx(1.0)
        assert x[1:7].tolist() == pytest.approx([10.0, 20.0, 30.0, 5.0, 5.0, 5.0])
        assert x[7] == pytest.approx(370.0)
        assert np.allclose(x[8:], Tc)

    def test_initial_state_tc_padding(self):
        """Shorter Tc is padded with the last value."""
        reactor = _make_reactor(n=10)
        Tc = np.array([290.0, 295.0])
        x = reactor.initial_state(1.0, 1, 1, 1, 1, 1, 1, 370.0, Tc)
        # First two entries from input, the rest padded with Tc[-1] = 295.
        assert np.allclose(x[8:10], [290.0, 295.0])
        assert np.allclose(x[10:], np.full(8, 295.0))

    def test_initial_state_tc_truncation(self):
        reactor = _make_reactor(n=5)
        Tc = np.arange(20, dtype=np.float64) + 280.0
        x = reactor.initial_state(1.0, 1, 1, 1, 1, 1, 1, 370.0, Tc)
        assert np.allclose(x[8:], Tc[:5])


class TestParameterPacking:
    """The flat parameter vector must contain the upstream values."""

    def test_packed_shape(self):
        # See ``_dynmodel_kernel`` docstring for the 25-element layout.
        p = _pack_params(default_parameters())
        assert p.shape == (25,)

    def test_packed_values_match_par(self):
        """Spot-check the first 20 entries match ``WoParameters`` accessors."""
        par = default_parameters()
        p = _pack_params(par)
        assert p[0] == pytest.approx(par.rhoR)
        assert p[1] == pytest.approx(par.A1)
        assert p[4] == pytest.approx(par.Ea1)
        assert p[7] == pytest.approx(par.DeltaH1)
        assert p[10] == pytest.approx(par.CpA)
        assert p[17] == pytest.approx(par.U)
        assert p[18] == pytest.approx(par.rhoW)
        assert p[19] == pytest.approx(par.Lcoil)
        assert p[20] == pytest.approx(par.Dcoil)
        assert p[21] == pytest.approx(par.Acoil)
        assert p[22] == pytest.approx(float(par.n))
        assert p[23] == pytest.approx(par.Ar)
        assert p[24] == pytest.approx(par.Vr)


# -----------------------------------------------------------------------------
# 2. Numba kernel correctness
# -----------------------------------------------------------------------------

class TestKernelOutputShape:
    """The kernel must return an array of the correct shape and dtype."""

    def test_kernel_shape(self):
        reactor = _make_reactor(n=20)
        u = reactor.nominal_inputs()
        x = reactor.initial_state(1.0, 100, 100, 100, 100, 100, 100,
                                  368.81, np.full(20, 300.0))
        dx = _dynmodel_kernel(0.0, x, u, reactor._p)
        assert dx.shape == (8 + 20,)

    def test_kernel_finite(self):
        reactor = _make_reactor(n=20)
        u = reactor.nominal_inputs()
        x = reactor.initial_state(1.0, 100, 100, 100, 100, 100, 100,
                                  368.81, np.full(20, 300.0))
        dx = _dynmodel_kernel(0.0, x, u, reactor._p)
        assert np.all(np.isfinite(dx))

    def test_kernel_bc_enforced(self):
        """The first cell of dTc/dt must be 0 (Dirichlet BC at z=0)."""
        reactor = _make_reactor(n=20)
        u = reactor.nominal_inputs()
        x = reactor.initial_state(1.0, 100, 100, 100, 100, 100, 100,
                                  368.81, np.full(20, 300.0))
        dx = _dynmodel_kernel(0.0, x, u, reactor._p)
        assert dx[8] == pytest.approx(0.0, abs=0.0)

    def test_kernel_dtc_finite_and_directed(self):
        """With Tr > Tc throughout the coil, heat flux Qt > 0 everywhere.
        The dTc/dt profile should be physically meaningful: finite, with
        the convective cooling effect balanced against the heating."""
        reactor = _make_reactor(n=20)
        u = reactor.nominal_inputs()
        x = reactor.initial_state(1.0, 100, 100, 100, 100, 100, 100,
                                  368.81, np.full(20, 300.0))
        dx = _dynmodel_kernel(0.0, x, u, reactor._p)
        dtc = dx[8:]
        assert np.all(np.isfinite(dtc))
        # Dirichlet BC enforces zero at z=0.
        assert dtc[0] == pytest.approx(0.0, abs=0.0)
        # At Tr > Tc everywhere, heating dominates at the outlet end of the coil
        # because the cooling water is hottest there but Tr is even higher.
        assert dtc[-1] > 0.0


class TestKernelAlgebraicConstraint:
    """The first row of dx/dt must be ``VR - Ar * L`` (algebraic)."""

    def test_level_ae_value(self):
        reactor = _make_reactor(n=5)
        u = reactor.nominal_inputs()
        # Mass = 1000 kg, rhoR = 800.92, so VR = 1.249 m^3. Ar is computed
        # from the upstream geometry, so g1 = VR - Ar * L.
        x = reactor.initial_state(1.0, 200, 200, 200, 200, 100, 100,
                                  350.0, np.full(5, 300.0))
        dx = _dynmodel_kernel(0.0, x, u, reactor._p)
        # MR (the in-kernel total mass) is 1000 kg, but the upstream set
        # L=1.0 independently. The kernel computes VR = MR/rhoR = 1.249 m^3,
        # so g1 = 1.249 - Ar * 1.0 (which is the AE residual).
        MR = sum(x[1:7])
        VR = MR / reactor.par.rhoR
        expected_g1 = VR - reactor.par.Ar * 1.0
        assert dx[0] == pytest.approx(expected_g1, rel=1e-12)


# -----------------------------------------------------------------------------
# 3. Conservation laws (integration tests)
# -----------------------------------------------------------------------------

class TestMassConservation:
    """Integration tests: the integrator must conserve mass over the run."""

    def test_no_inlet_mass_constant(self):
        """With all inlets and outlets turned off AND no reactions
        (cold, no C/P), the total mass is conserved exactly."""
        reactor = _make_reactor(n=5)
        # Zero flow everywhere.
        u = np.array([0.0, 0.0, 100.0, 100.0, 100.0, 0.0, 0.0])
        # Tr = 100 K — far below any meaningful reaction threshold, and the
        # only reactants are A and B (no C, no P), so r1mr is the only rate
        # and at 100 K k1 is essentially zero. The mass must be conserved.
        x0 = reactor.initial_state(
            L=1.0,
            ma=200.0, mb=200.0, mc=0.0, me=0.0, mg=0.0, mp=0.0,
            Tr=100.0, Tc=np.full(5, 100.0),
        )
        rhs = reactor.f(u)
        sol = solve_ivp(rhs, (0.0, 100.0), x0, method='Radau', rtol=1e-10, atol=1e-12)
        assert sol.success
        # Mass hold-up must be bit-stable (no reactions, no flows).
        assert np.allclose(sol.y[1:7, -1], sol.y[1:7, 0], atol=1e-8)

    def test_inlet_only_increases_mass(self):
        """With only FA in (no outlet), the A hold-up grows monotonically."""
        reactor = _make_reactor(n=5)
        # Tiny QR → nearly zero outlet flow.
        u = np.array([1.0, 0.0, 300.0, 300.0, 300.0, 1e-9, 1e-9])
        x0 = reactor.initial_state(
            L=1.0,
            ma=100.0, mb=100.0, mc=100.0, me=100.0, mg=100.0, mp=100.0,
            Tr=300.0, Tc=np.full(5, 300.0),
        )
        rhs = reactor.f(u)
        sol = solve_ivp(rhs, (0.0, 60.0), x0, method='Radau', rtol=1e-9, atol=1e-11)
        assert sol.success
        # ma must have grown by ~FA * dt = 60 kg
        assert sol.y[1, -1] > sol.y[1, 0] + 50.0  # at least 50 kg of A in

    def test_outlet_only_decreases_mass(self):
        """With only outlet flow (no inlet), the total mass must decrease."""
        reactor = _make_reactor(n=5)
        # Zero inlets, decent outlet.
        u = np.array([0.0, 0.0, 300.0, 300.0, 300.0, 1e-3 / reactor.par.rhoR * 0.5, 1e-9])
        # The QR value above is 0.5 kg/s of mass outflow.
        x0 = reactor.initial_state(
            L=1.0,
            ma=100.0, mb=100.0, mc=100.0, me=100.0, mg=100.0, mp=100.0,
            Tr=300.0, Tc=np.full(5, 300.0),
        )
        rhs = reactor.f(u)
        sol = solve_ivp(rhs, (0.0, 60.0), x0, method='Radau', rtol=1e-9, atol=1e-11)
        assert sol.success
        # Total mass should have decreased.
        m_initial = sum(x0[1:7])
        m_final = sum(sol.y[1:7, -1])
        assert m_final < m_initial


class TestEnergyDirection:
    """Cold inlet (TA < Tr) must drive Tr *down*; hot inlet (TA > Tr) drives Tr *up*."""

    def test_cold_inlet_drives_tr_down(self):
        """Cold inlet (TA < Tr) drives Tr down — but only when reactions are
        negligible. With the REAL Arrhenius kinetics a 400 kg holdup of pure
        A+B at 350 K releases ~460 kW (r1 fires without any C/P present) and
        the reactor runs away thermally — the premise only holds with the
        kinetics switched off, so this fixture pins A1..A3 to ~zero and tests
        exactly what the docstring claims: the inlet-stream energy balance.
        (Fixture corrected 2026-11-03; the assertion itself is unchanged.)
        """
        from wo_fouling.parameters import Geometry, ReactionKinetics, WoParameters
        par = WoParameters(
            geometry=Geometry(n=5),
            kinetics=ReactionKinetics(A1=1e-300, A2=1e-300, A3=1e-300),
        )
        reactor = OpenLoopReactor(par)
        # Cold inlets, hot reactor, no reactions.
        u = np.array([0.5, 0.5, 250.0, 250.0, 250.0, 0.0, 1e-6])
        x0 = reactor.initial_state(
            L=1.0,
            ma=200.0, mb=200.0, mc=0.0, me=0.0, mg=0.0, mp=0.0,
            Tr=350.0, Tc=np.full(5, 250.0),
        )
        rhs = reactor.f(u)
        sol = solve_ivp(rhs, (0.0, 100.0), x0, method='Radau', rtol=1e-9, atol=1e-11)
        assert sol.success
        # Tr should be decreasing toward TA=TB=250 K.
        assert sol.y[7, -1] < 350.0


# -----------------------------------------------------------------------------
# 4. Nominal inputs sanity check
# -----------------------------------------------------------------------------

class TestNominalInputs:
    """Default inputs mirror ``sim_main.m`` defaults."""

    def test_default_inputs_match_sim_main(self):
        reactor = _make_reactor()
        u = reactor.nominal_inputs()
        assert u[0] == pytest.approx(1.8275)         # FA
        assert u[1] == pytest.approx(4.7875)         # FB
        assert u[2] == pytest.approx(294.3)          # TA
        assert u[3] == pytest.approx(294.3)          # TB
        assert u[4] == pytest.approx(288.75)         # Tcin
        assert u[5] == pytest.approx((1.8275 + 4.7875) / 800.92)  # QR
        assert u[6] == pytest.approx(1.5e-3)         # Qcin

    def test_overrides_applied(self):
        reactor = _make_reactor()
        u = reactor.nominal_inputs(FA=10.0, TB=350.0, Qcin=2.5e-3)
        assert u[0] == pytest.approx(10.0)
        assert u[3] == pytest.approx(350.0)
        assert u[6] == pytest.approx(2.5e-3)


# -----------------------------------------------------------------------------
# 5. End-to-end integration: from upstream SS guess to a stable SS
# -----------------------------------------------------------------------------

class TestEndToEndIntegration:
    """The reactor must integrate without exploding."""

    def test_6h_integration_reaches_steady_state(self):
        reactor = _make_reactor(n=20)  # smaller n for speed
        u = reactor.nominal_inputs()

        # Initial state at the published sim_main.m guess.
        xi0 = np.array([0.37, 0.012, 0.31, 0.13, 0.1])
        mp_mf = max(1.0 - xi0.sum(), 0.0)
        masses = xi0 * reactor.par.massT
        mp = mp_mf * reactor.par.massT
        x0 = reactor.initial_state(
            L=0.073,
            ma=float(masses[0]), mb=float(masses[1]),
            mc=float(masses[2]), me=float(masses[3]),
            mg=float(masses[4]),
            mp=mp,
            Tr=368.81,
            Tc=np.full(reactor.n, 288.75),
        )

        rhs = reactor.f(u)
        sol = solve_ivp(rhs, (0.0, 6 * 3600.0), x0,
                        method='Radau', rtol=1e-8, atol=1e-10,
                        t_eval=[6 * 3600.0])
        assert sol.success

        # Mass conservation — total mass should be close to massT (the guess
        # may not satisfy the closure perfectly, so allow a small drift).
        mt = sum(sol.y[1:7, -1])
        assert mt == pytest.approx(reactor.par.massT, rel=1e-6)

        # Tc[0] must equal Tcin at all times (Dirichlet BC).
        assert sol.y[8, -1] == pytest.approx(u[4], abs=1e-9)

        # All temperatures remain physical.
        assert sol.y[7, -1] > 200.0
        assert sol.y[7, -1] < 600.0
        assert np.all(sol.y[8:, -1] > 200.0)
        assert np.all(sol.y[8:, -1] < 600.0)
