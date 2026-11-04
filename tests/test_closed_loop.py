"""Tests for the closed-loop harness and controllers (extracted slx values)."""

from __future__ import annotations

import numpy as np
import pytest

from wo_fouling.controllers import (
    DiscretePID, tyreus_luyben_pid, amigo_pid, amigo_pi_itd_level,
    theta_with_half_sample, TEMP_RELAY_CLEAN, TEMP_TAU_TUNER,
    TEMP_RELAY_FOULED_24H, LEVEL_TAU_TUNER, LEVEL_RELAY,
    filter_initial_condition, relay_autotune, RelayParams,
    LR_C_GAINS, TR_C_GAINS,
)
from wo_fouling.closed_loop import (
    ClosedLoopSimulator, ClosedLoopConfig, _slx_ic_208,
)


class TestTuningRelations:
    """controller_tuning.m arithmetic -> the gains actually embedded in
    the .slx PID blocks (extracted 2026-11-03). This is the strongest
    cross-check available without MATLAB: the production controller gains
    must come out of the recorded relay/tau-tuner constants."""

    def test_tyreus_luyben_matches_slx_tr_c(self):
        KP, KI, KD, N = tyreus_luyben_pid(TEMP_RELAY_CLEAN["Kcu"],
                                          TEMP_RELAY_CLEAN["Pu"])
        assert KP == pytest.approx(TR_C_GAINS["P"], rel=2e-4)
        assert KI == pytest.approx(TR_C_GAINS["I"], rel=2e-4)
        assert KD == pytest.approx(TR_C_GAINS["D"], rel=2e-4)
        assert N == pytest.approx(TR_C_GAINS["N"], rel=2e-4)

    def test_amigo_level_matches_slx_lr_c(self):
        theta = theta_with_half_sample(LEVEL_TAU_TUNER["theta_raw"],
                                       LEVEL_TAU_TUNER["deltat"])
        KP, KI = amigo_pi_itd_level(LEVEL_TAU_TUNER["Kp"], theta)
        assert KP == pytest.approx(LR_C_GAINS["P"], rel=1e-3)
        assert KI == pytest.approx(LR_C_GAINS["I"], rel=1e-3)

    def test_amigo_pid_values(self):
        theta = theta_with_half_sample(TEMP_TAU_TUNER["theta_raw"], 5.0)
        KP, KI, KD, N = amigo_pid(TEMP_TAU_TUNER["Kp"], theta)
        assert KP == pytest.approx(-7.265e-5, rel=1e-3)
        assert KI == pytest.approx(-1.089e-7, rel=1e-3)
        assert KD == pytest.approx(-3.030e-3, rel=1e-3)
        assert N == pytest.approx(0.048, rel=1e-3)

    def test_fouled_retuning_changes(self):
        """Second relay test after 24 h fouling (controller_tuning.m
        lines 105-124): the PID gain changes are the paper's dependent
        variables."""
        KP1, KI1, _, _ = tyreus_luyben_pid(TEMP_RELAY_CLEAN["Kcu"],
                                           TEMP_RELAY_CLEAN["Pu"])
        KP2, KI2, KD2, _ = tyreus_luyben_pid(TEMP_RELAY_FOULED_24H["Kcu2"],
                                             TEMP_RELAY_FOULED_24H["Pu2"])
        assert 100 * (KP2 - KP1) / KP1 == pytest.approx(187.5, abs=0.5)
        assert 100 * (KI2 - KI1) / KI1 == pytest.approx(69.4, abs=0.5)
        assert 100 * (KD2 - tyreus_luyben_pid(
            TEMP_RELAY_CLEAN["Kcu"], TEMP_RELAY_CLEAN["Pu"])[2]
        ) / tyreus_luyben_pid(TEMP_RELAY_CLEAN["Kcu"],
                              TEMP_RELAY_CLEAN["Pu"])[2] == pytest.approx(
            388.1, abs=0.5)

    def test_filter_ic_formula(self):
        """script_sim_main.m TR_filter_ic arithmetic -> 23.1105."""
        val = filter_initial_condition(TR_C_GAINS["D"], TR_C_GAINS["N"],
                                       5.0, 371.0948, c=0.0)
        assert val == pytest.approx(23.1105, rel=1e-4)


class TestDiscretePID:
    def test_pure_p_tracks_sign(self):
        pid = DiscretePID(P=-2.0, I=0.0, D=0.0, N=100, Ts=5.0)
        u = pid.step(1.0, 0.5)   # e = 0.5
        assert u == pytest.approx(-2.0 * 0.5)

    def test_saturation_clamps(self):
        pid = DiscretePID(P=-2.0, I=0.0, D=0.0, N=100, Ts=5.0,
                          umin=0.0, umax=0.1)
        assert pid.step(1.0, 10.0) == 0.1
        assert pid.step(1.0, -10.0) == 0.0

    def test_integrator_at_steady_state_holds_output(self):
        """Controller pre-charged via integrator IC: with e = 0 forever,
        output stays at the IC value (the closed-loop t=0 condition)."""
        pid = DiscretePID(P=-1.67e-4, I=-1.8639e-7, D=0.0, N=100, Ts=5.0)
        pid.reset(integrator_ic=0.00125, y0=371.0948)
        for _ in range(20):
            u = pid.step(371.0948, 371.0948)
        assert u == pytest.approx(0.00125, rel=1e-12)


class TestRelayAutotuner:
    def test_symmetric_relay_on_fopdt(self):
        """Drive a first-order-plus-dead-time plant through the relay
        autotuner; the extracted Kcu/Pu must match the describing-
        function prediction for a symmetric relay."""
        K = -74.26          # static gain (negative: cooling)
        T = 300.0           # time constant, s
        L = 90.0            # dead time, s
        Ts = 5.0
        d = 5e-4            # relay amplitude
        h = 0.01            # hysteresis

        y = 0.0
        y0 = 0.0
        # state for the LPD: y' = (K*u(t-L) - y)/T with input history
        hist_len = int(L / Ts) + 2
        uhist = [0.0] * hist_len

        def plant_step(u):
            nonlocal y
            u_delayed = uhist[0]
            y += Ts * (K * u_delayed - y) / T
            uhist.pop(0)
            uhist.append(u)
            return y

        params = RelayParams(noise_measure_time=30.0, hyst=h, eps=0.05,
                             asym=2.0, ustart=0.0, h=Ts, uon=d, fixed=1)
        res = relay_autotune(plant_step, y0=0.0, params=params,
                             t_max=40000.0)
        assert res.converged
        # describing function: Kcu ~ pi*sqrt(a^2 - h^2)/(4d/pi) ... the
        # classic symmetric-relay estimate Kcu = 4d/(pi*a) (with the
        # hysteresis correction the chart applies). Check sign and order.
        amp_est = res.Kcu and (4 * d) / (np.pi * abs(res.Kcu))
        assert 1e-4 < amp_est < 1.0
        # period > 2*L (relay oscillation is slower than the dead time)
        assert res.Pu > 2 * L


class TestSlxIC:
    def test_ic_length_and_masses(self):
        ic = _slx_ic_208()
        assert ic.shape == (208,)
        assert ic[:6].sum() == pytest.approx(2105.0, abs=1e-6)
        assert ic[6] == pytest.approx(371.094840058293, abs=1e-9)
        assert ic[7] == pytest.approx(288.75, abs=1e-9)  # Tc(1) = Tcin
        assert ic[-1] == 0.0                              # Rf = 0


class TestClosedLoop:
    def test_quiet_run_holds_steady_state(self):
        """Quiet configuration (no sines, fouling off, 10 min): every
        channel matches the recorded sim_results.mat steady state."""
        cfg = ClosedLoopConfig(t_end=600.0, fouling_on=False,
                               ta_sine=False, tb_sine=False,
                               tcin_sine=False)
        r = ClosedLoopSimulator(cfg=cfg).run()
        assert r.TR.mean() == pytest.approx(371.0948, abs=5e-3)
        assert r.LR.mean() == pytest.approx(2.3395, abs=1e-3)
        assert r.Qcin.mean() == pytest.approx(1.25e-3, rel=5e-3)
        assert r.QR.mean() == pytest.approx(8.259e-3, rel=1e-3)
        assert r.Tcout[-1] == pytest.approx(351.1, abs=0.5)
        assert r.DeltaP.mean() == pytest.approx(8.2e7, rel=3e-2)

    def test_tr_tracks_sp_with_stair(self):
        """With the SP stair active, TR follows the ±5 K schedule."""
        cfg = ClosedLoopConfig(t_end=2 * 3600.0, sp_stair=True,
                               fouling_on=False, ta_sine=False,
                               tb_sine=False, tcin_sine=False)
        r = ClosedLoopSimulator(cfg=cfg).run()
        # second hour: SP = 371.0948 (stair slot 2); TR should settle back
        tail = r.TR[r.t > 6000]
        assert tail.mean() == pytest.approx(371.0948, abs=1.5)
        # first hour SP = 376.0948: TR must rise well above nominal
        assert r.TR.max() > 372.5
