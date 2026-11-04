"""Controller tuning, discrete PID blocks, and the relay autotuner.

Ports of:

* ``controller_tuning.m`` — SIMC / AMIGO (PI, PID, integrating) /
  Tyreus-Luyben tuning relations and the recorded experiment constants
  (tau-tuner and relay results hard-coded upstream).
* the Simulink PID blocks — ``WO_main_sim/WO_plant/LR_C`` (pid_lib/PID
  Controller, PI, Ts=5 s, output saturated to [0, 0.012]) and
  ``WO_main_sim/WO_plant/TRC/PID/TR_C`` (pid_lib/PID Controller (2DOF),
  Ts=5 s, b=1, c=0), both extracted from the .slx XML (2026-11-03).
* the Stateflow chart ``relay_exp`` (chart_40.xml) — Åström-Hägglund relay
  autotuner: noise measurement phase, exponential startup, hysteresis relay
  with asymmetry management, period-convergence test, then symmetric-relay
  ultimate-gain extraction ``Kcu = 4*d1/(pi*sqrt(amp^2 - h^2))`` and
  Tyreus-Luyben PID tuning.

The extracted production gains (see PORT_NOTES.md §closed-loop):

    LR_C : P=-0.017011, I=-1.086e-4, D=0, N=100, Ts=5 s,
           integrator IC=0.0082593 (= QR nominal), limits [0, 0.012]
    TR_C : P=-1.6707e-4, I=-1.8639e-7, D=-9.399e-3, N=0.03555, c=0, Ts=5 s,
           integrator IC=0.00125 (= Qcin nominal), filter IC=23.1105

Both cross-check against the AMIGO/TL arithmetic in ``controller_tuning.m``
(see ``test_controllers.py`` golden tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

# -----------------------------------------------------------------------------
# Tuning relations (controller_tuning.m)
# -----------------------------------------------------------------------------

def simc_pi(Kp: float, theta: float, tauC: float | None = None):
    """SIMC PI tuning (commented block in controller_tuning.m):
    tauC = theta; KP = (1/Kp)/(tauC + theta); tauI = 4*(tauC + theta)."""
    if tauC is None:
        tauC = theta
    KP = (1.0 / Kp) * (1.0 / (tauC + theta))
    tauI = 4.0 * (tauC + theta)
    return KP, KP / tauI


def amigo_pi_itd_level(Kp: float, theta: float) -> tuple[float, float]:
    """AMIGO-ITD PI for the LEVEL loop (controller_tuning.m lines 29-31):
    KP = 0.35/(Kp*theta); Ti = 13.4*theta."""
    KP = 0.35 / (Kp * theta)
    Ti = 13.4 * theta
    return KP, KP / Ti


def amigo_pid(Kp: float, theta: float):
    """AMIGO PID (controller_tuning.m lines 54-59). Returns (KP, KI, KD, N)
    with Ti = 8*theta, Td = 0.5*theta, N = 2/Td."""
    KP = 0.45 / (Kp * theta)
    Ti = 8.0 * theta
    Td = 0.5 * theta
    return KP, KP / Ti, KP * Td, 2.0 / Td


def tyreus_luyben_pid(Kcu: float, Pu: float):
    """Tyreus-Luyben PID (controller_tuning.m lines 68-74):
    Kc=0.45*Kcu, tauI=2.2*Pu, tauD=Pu/6.3,
    KP = Kc*(1+tauD/tauI), KI = Kc/tauI, KD = Kc*tauD,
    N = 2/(tauD/(1+tauD/tauI))."""
    Kc = 0.45 * Kcu
    tauI = Pu * 2.2
    tauD = Pu / 6.3
    KP = Kc * (1.0 + tauD / tauI)
    KI = Kc / tauI
    KD = Kc * tauD
    N = 2.0 / (tauD / (1.0 + tauD / tauI))
    return KP, KI, KD, N


# Recorded experiment results (controller_tuning.m hard values) -------------

LEVEL_TAU_TUNER = {"Kp": -1.76, "theta_raw": 9.19, "deltat": 5.0}
TEMP_TAU_TUNER = {"Kp": -74.26, "theta_raw": 80.906, "deltat": 5.0}
LEVEL_RELAY = {"Kcu": -0.0520, "Pu": 100.0}
TEMP_RELAY_CLEAN = {"Kcu": -3.4628e-4, "Pu": 380.0}
TEMP_RELAY_FOULED_24H = {"Kcu2": -9.9569e-4, "Pu2": 645.0}


def theta_with_half_sample(theta_raw: float, deltat: float) -> float:
    """controller_tuning.m: theta = theta_raw + 0.5*deltat (sample-time
    compensation applied to the tau-tuner dead time)."""
    return theta_raw + 0.5 * deltat


# -----------------------------------------------------------------------------
# Discrete PID blocks (pid_lib semantics, discrete Ts, backward-Euler)
# -----------------------------------------------------------------------------

@dataclass
class PIDState:
    """Discrete pid_lib state: integral accumulator, derivative filter,
    previous measurement."""
    i_acc: float = 0.0        # integral accumulator (error·s)
    dfilt: float = 0.0        # filtered derivative output D_k
    yprev: float = 0.0        # previous measurement
    uprev_sat: float = 0.0    # previous saturated output


class DiscretePID:
    """pid_lib 'PID Controller' / 'PID Controller (2DOF)' discrete form.

    Matches the Simulink blocks (sample time Ts, backward-Euler integrator,
    filtered derivative with coefficient N):

        u_k = P*(b*r_k - y_k) + I*i_acc_k + Dfilt_k
        i_acc_{k+1} = i_acc_k + Ts*e_k                      (clamped)
        Dfilt_k = a*(Dfilt_{k-1} - D*N*(y_k - y_{k-1})),
        a = Ts*N/(1+Ts*N)                                   (c=0: deriv on y)

    The block gains P/I/D are the PARALLEL coefficients (P=K, I=K/Ti,
    D=K*Td). Output saturation [umin, umax] with clamping anti-windup
    (integrator freezes when saturated and the error pushes further out).
    """

    def __init__(self, P: float, I: float, D: float, N: float,
                 Ts: float, umin: float | None = None,
                 umax: float | None = None,
                 b: float = 1.0, c: float = 0.0):
        self.P, self.I, self.D, self.N = P, I, D, N
        self.Ts = Ts
        self.umin, self.umax = umin, umax
        self.b, self.c = b, c
        self.state = PIDState()

    def reset(self, integrator_ic: float = 0.0, filter_ic: float = 0.0,
              y0: float = 0.0):
        """Seed the state. ``integrator_ic`` reproduces the pid_lib
        'InitialConditionForIntegrator' (the I-branch's output value at
        k=0, i.e. a pre-charged controller output at steady state);
        ``filter_ic`` the 'InitialConditionForFilter' (D_k at k=0);
        ``y0`` the initial measurement (keeps the first derivative kick at
        zero)."""
        s = PIDState()
        s.i_acc = integrator_ic / self.I if self.I != 0.0 else 0.0
        s.dfilt = filter_ic
        s.yprev = y0
        self.state = s

    def step(self, r: float, y: float) -> float:
        s = self.state
        Ts = self.Ts
        e = r - y

        # filtered derivative on measurement (c = 0)
        if self.D != 0.0:
            a = Ts * self.N / (1.0 + Ts * self.N)
            s.dfilt = a * (s.dfilt - self.D * self.N * (y - s.yprev))
        dterm = s.dfilt

        u_unsat = self.P * (self.b * r - y) + self.I * s.i_acc + dterm

        u = u_unsat
        if self.umin is not None and u_unsat < self.umin:
            u = self.umin
        if self.umax is not None and u_unsat > self.umax:
            u = self.umax
        saturated = u != u_unsat
        drives_out = np.sign(e * self.I) == np.sign(u_unsat - u)
        if not (saturated and drives_out):
            s.i_acc += Ts * e

        s.yprev = y
        s.uprev_sat = u
        return u


# Extracted production controllers (WO_main_sim.slx) --------------------------

LR_C_GAINS = dict(P=-0.017011, I=-0.0001086, D=0.0, N=100.0, Ts=5.0,
                  umin=0.0, umax=0.012, integrator_ic=0.0082593)
TR_C_GAINS = dict(P=-0.00016707, I=-1.8639e-07, D=-0.009399, N=0.03555,
                  Ts=5.0, c=0.0, integrator_ic=0.00125, filter_ic=23.1105)


def filter_initial_condition(KD: float, N: float, deltat: float,
                             Tr0: float, c: float = 0.0) -> float:
    """Port of the TR_filter_ic formula in ``script_sim_main.m`` /
    ``controller_tuning.m``:

        b1 = 1; b2 = -1/(1+N*deltat);
        a1 = 1; a2 = -1; y1 = 0; c = 0;
        u_filter = KD_TR*(c*Tr0 - Tr0);
        TR_filter_ic = -(y1 - (b1/a1)*u_filter) / (b2 - (a2*b1)/a1);
    """
    b1 = 1.0
    b2 = -1.0 / (1.0 + N * deltat)
    a1 = 1.0
    a2 = -1.0
    y1 = 0.0
    u_filter = KD * (c * Tr0 - Tr0)
    return -(y1 - (b1 / a1) * u_filter) / (b2 - (a2 * b1) / a1)


# -----------------------------------------------------------------------------
# Relay autotuner (Stateflow chart_40 'relay_exp', symmetric-relay branch)
# -----------------------------------------------------------------------------

@dataclass
class RelayParams:
    """Relay experiment parameters (script_sim_main.m 'params' struct)."""
    noise_measure_time: float = 90.0   # s
    hyst: float = 0.1                  # default hysteresis (def_h)
    eps: float = 0.01                  # convergence limit (relative)
    asym: float = 2.0                  # gamma
    ustart: float = 0.0                # nominal control signal
    low_limit_f: float = 2.0           # factor * h
    upp_limit_f: float = 12.0          # factor * h  (dev_limit upstream)
    h: float = 5.0                     # sampling time (s)
    uon: float = 5e-4                  # relayMaxDev — max control deviation
    fixed: int = 1                     # 1 = symmetric relay


@dataclass
class RelayResult:
    Pu: float = 0.0
    Kcu: float = 0.0
    ton: float = 0.0
    toff: float = 0.0
    h: float = 0.0
    tau: float = 0.0
    tstop: float = 0.0
    converged: bool = False


def relay_autotune(plant_step, y0: float, params: RelayParams,
                   t_max: float = 40000.0) -> RelayResult:
    """Symmetric-relay autotuner — port of ``relay_exp`` (chart_40.xml).

    ``plant_step(u) -> y'': a ONE-STEP callable that applies control
    ``u`` for one sample period ``params.h`` and returns the new measured
    output (the Simulink loop drove the closed loop from the chart; here
    the caller supplies the plant interface).

    Phases (transcribed from the chart script):

    1. noise measurement (``noiseMeasureTime``): hold u = ustart, track
       yh/yl, then h = max(2*noise, def_h).
    2. exponential startup: u = uref - c + c*b^(exp_t/ttf) with b=40,
       c = relayMaxDev/(b-1), ttf = 5 s; until |y - yref| > h.
    3. hysteresis relay u = uon / uoff around yref ± h with symmetric
       amplitudes (fixAmp=1), until two consecutive full periods agree
       within ``eps`` (relative).
    4. extraction: Pu = ton + toff;
       Kcu = Ksign*4*d1/(pi*sqrt(amp^2 - h^2)) with amp = (ymax-ymin)/2
       of the last full period; then Tyreus-Luyben PID.
    """
    t_sample = params.h
    gamma = params.asym
    epsilon = params.eps
    relayMaxDev = params.uon
    def_h = params.hyst
    fixAmp = params.fixed
    uref = params.ustart

    # relay amplitude check (asymmetry clamp, fixAmp branch)
    d1 = d2 = relayMaxDev if fixAmp == 1 else gamma

    sample = 0
    t = 0.0
    y = y0
    yref = y0
    yh = yl = y0
    measure_noise = True
    startup = False
    done = False
    Ksign = 1.0
    u = uref
    h = def_h

    # startup constants
    ttf = 5.0 / t_sample  # in samples
    b = 40.0
    c_amp = relayMaxDev / (b - 1.0)
    a_exp = uref - c_amp
    exp_t0 = 0.0

    uvec = [u]
    yvec = [y]
    tvec = [0.0]
    switching_times: list[float] = []
    last_switch_sample = 0
    tstop = 0.0

    noise_samples = int(round(params.noise_measure_time / t_sample))

    while t < t_max and not done:
        sample += 1
        t = sample * t_sample
        y = plant_step(u)
        yvec.append(y)
        tvec.append(t)

        if measure_noise:
            yh = max(yh, y)
            yl = min(yl, y)
            u = uref
            if sample >= noise_samples:
                noise = (yh - yl) / 2.0
                yref = (yh + yl) / 2.0
                h = max(2.0 * noise, def_h)
                measure_noise = False
                startup = True
                exp_t0 = sample
        elif startup:
            exp_t = sample - exp_t0
            u = a_exp + c_amp * b ** (exp_t / ttf)
            if u > relayMaxDev + uref:
                u = relayMaxDev + uref
            if exp_t > 1:
                ydev = y - yref
                if abs(ydev) > h:
                    Ksign = ydev / abs(ydev)
                    startup = False
        elif not done:
            u_on = uref + Ksign * d1
            u_off = uref - Ksign * d2
            # rel_func
            if y <= yref - h:
                u = u_on
            elif yref - h < y < yref + h:
                u = uvec[-1]
            else:
                u = u_off

        uvec.append(u)

        # switching detection
        if sample > 1 and (uvec[-1] - uref) * (uvec[-2] - uref) < 0:
            switching_times.append(t)
            if len(switching_times) > 3:
                tp = switching_times[-1] - switching_times[-3]
                tp_old = switching_times[-2] - switching_times[-4]
                if abs(tp - tp_old) < tp_old * epsilon:
                    done = True
                    tstop = t
            last_switch_sample = sample

    if not done or len(switching_times) < 4:
        return RelayResult(converged=False)

    # ton/toff from the last two half-periods
    u_on = uref + Ksign * d1
    tail = uvec[-2:]
    if tail[0] == u_on:
        ton = switching_times[-1] - switching_times[-2]
        toff = switching_times[-2] - switching_times[-3]
    else:
        toff = switching_times[-1] - switching_times[-2]
        ton = switching_times[-2] - switching_times[-3]

    Pu = ton + toff
    # last full period of the relay response
    i1 = last_switch_sample
    i0 = last_switch_sample - int(round(Pu / t_sample)) - 1
    if i0 < 0:
        i0 = 0
    yp = np.asarray(yvec[i0:i1 + 1]) - yref
    ymax, ymin = float(yp.max()), float(yp.min())
    amp = (ymax - ymin) / 2.0
    if amp ** 2 - h ** 2 <= 0:
        return RelayResult(converged=False)
    Kcu = (Ksign * 4.0 * d1) / (np.pi * np.sqrt(amp ** 2 - h ** 2))

    return RelayResult(Pu=float(Pu), Kcu=float(Kcu), ton=float(ton),
                       toff=float(toff), h=float(h), tstop=float(tstop),
                       converged=True)


def relay_pid_from_result(res: RelayResult):
    """Tyreus-Luyben PID from a converged relay result (chart_40 tail)."""
    if not res.converged:
        raise ValueError("relay experiment did not converge")
    return tyreus_luyben_pid(res.Kcu, res.Pu)


if __name__ == "__main__":
    # controller_tuning.m arithmetic reproduction (golden values in tests)
    theta = theta_with_half_sample(TEMP_TAU_TUNER["theta_raw"],
                                   TEMP_TAU_TUNER["deltat"])
    KP, KI, KD, N = amigo_pid(TEMP_TAU_TUNER["Kp"], theta)
    print(f"AMIGO PID (T):  KP={KP:.6e}  KI={KI:.6e}  KD={KD:.6e}  N={N:.4f}")

    Wu = 2 * np.pi / TEMP_RELAY_CLEAN["Pu"]
    theta_r = np.pi / (2 * Wu)
    print(f"relay theta (from Pu=380 s): {theta_r:.3f} s")
    KP, KI, KD, N = tyreus_luyben_pid(TEMP_RELAY_CLEAN["Kcu"],
                                      TEMP_RELAY_CLEAN["Pu"])
    print(f"TL PID (T):     KP={KP:.6e}  KI={KI:.6e}  KD={KD:.6e}  N={N:.5f}")
    print("extracted TR_C: KP=-1.6707e-04  KI=-1.8639e-07  KD=-9.399e-03  N=0.03555")

    KP2, KI2, KD2, N2 = tyreus_luyben_pid(TEMP_RELAY_FOULED_24H["Kcu2"],
                                          TEMP_RELAY_FOULED_24H["Pu2"])
    print(f"TL PID fouled:  KP={KP2:.6e}  KI={KI2:.6e}  KD={KD2:.6e}  N={N2:.5f}")
    print(f"KP change: {100*(KP2-KP)/KP:.2f}%  KI change: {100*(KI2-KI)/KI:.2f}%  "
          f"KD change: {100*(KD2-KD)/KD:.2f}%")
