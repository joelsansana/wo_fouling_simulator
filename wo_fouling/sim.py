"""Scenario runners — ports of the simulation drivers.

* :func:`run_open_loop` — ``sim_main.m``: 4-stage steady state then
  ``ode15s``-equivalent integration of ``dynmodel.m`` (10 h default).
* :func:`run_open_loop_fouling` — the fouling leg of
  ``script_sim_main.m``: ``dynmodel_fouling.m`` integration with the
  three AE rows resolved algebraically.
* :func:`run_tube_fouling` — ``sim_tubo_fouling.m`` (1 month default).
* :func:`run_closed_loop` — ``WO_main_sim.slx`` semantics via
  :class:`wo_fouling.closed_loop.ClosedLoopSimulator`.

Solver substitution: MATLAB ``ode15s`` (variable-order BDF) → SciPy
``solve_ivp(method='BDF')`` with the same tolerances where recorded
(RelTol 1e-6; AbsTol 1e-8 open-loop clean, 1e-6 fouling/closed-loop).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .parameters import WoParameters, open_loop_parameters, closed_loop_parameters
from .reactor import OpenLoopReactor
from .fouling import FoulingReactor
from .tube import TubeSimulator, tube_initial_state
from .steady_state import solve_open_loop_ss


def open_loop_inputs(par: WoParameters) -> np.ndarray:
    """sim_main.m nominal inputs (Qcin = 1.5e-3)."""
    FA, FB = 1.8275, 4.7875
    return np.array([FA, FB, 294.3, 294.3, 288.75,
                     (FA + FB) / par.rhoR, 1.5e-3])


def run_open_loop(t_end: float = 10 * 3600.0, par: WoParameters | None = None,
                  x0: np.ndarray | None = None, t_eval=None):
    """sim_main.m: steady-state init + clean dynamics."""
    par = open_loop_parameters() if par is None else par
    u = open_loop_inputs(par)
    if x0 is None:
        x_ss, info = solve_open_loop_ss(u, par)
        x0 = x_ss
    reactor = OpenLoopReactor(par)
    sol = solve_ivp(reactor.f(u), (0.0, t_end), x0, method="BDF",
                    rtol=1e-6, atol=1e-8, t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol, u, par


def run_open_loop_fouling(t_end: float = 4 * 3600.0,
                          par: WoParameters | None = None,
                          x0: np.ndarray | None = None,
                          u: np.ndarray | None = None, t_eval=None):
    """script_sim_main.m fouling leg: dynmodel_fouling with AE rows
    resolved (Mass-matrix semantics). IC: closed-loop steady state by
    default (the script chains the clean run then fouling)."""
    par = closed_loop_parameters() if par is None else par
    if u is None:
        FA, FB = 1.8275, 4.7875
        u = np.array([FA, FB, 294.3, 294.3, 288.75,
                      (FA + FB) / par.rhoR, 1.25e-3, 150e-5, 9.0])
    reactor = FoulingReactor(par)
    if x0 is None:
        from .steady_state import solve_closed_loop_ss
        x0_ss, _ = solve_closed_loop_ss(u, par)
        # [Lr0, ma..mp, Tr, Tc, Rf0] -> differential state
        x0 = np.concatenate([x0_ss[1:8 + par.n], x0_ss[8 + par.n:9 + par.n]])
    sol = solve_ivp(reactor.f(u), (0.0, t_end), x0, method="BDF",
                    rtol=1e-6, atol=1e-6, t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol, u, par


def run_tube_fouling(t_end: float = 60 * 24 * 3600.0,
                     par: WoParameters | None = None,
                     fator: float = 1.0, fator_Qc: float = 0.1,
                     fator_Ca: float = 1.0, Tcin: float = 288.75,
                     Qcin: float = 0.0013, Tamb: float = 370.0,
                     Ca: float = 50e-3 / 100.0, pH: float = 9.0,
                     t_eval=None):
    """sim_tubo_fouling.m defaults (Qcin scaled by fator_Qc=0.1,
     Ca = 50 mg/L / 100)."""
    par = closed_loop_parameters() if par is None else par
    u = np.array([Tcin, Qcin * fator_Qc, Tamb, Ca * fator_Ca, pH])
    sim = TubeSimulator(par, fator=fator)
    x0 = sim.initial_state(u)
    sol = solve_ivp(sim.f(u), (0.0, t_end), x0, method="BDF",
                    rtol=1e-6, atol=1e-6, t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol, u, par, sim


def run_closed_loop(t_end: float = 90 * 60.0, **cfg_kwargs):
    """WO_main_sim semantics (see wo_fouling.closed_loop)."""
    from .closed_loop import ClosedLoopSimulator, ClosedLoopConfig
    cfg = ClosedLoopConfig(t_end=t_end, **cfg_kwargs)
    sim = ClosedLoopSimulator(cfg=cfg)
    return sim.run()
