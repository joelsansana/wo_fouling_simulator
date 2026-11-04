# wo_fouling — Williams-Otto reactor with CaCO₃ fouling (Python port)

A complete open-source Python reimplementation of a MATLAB/Simulink
simulation apparatus for studying cooling-water CaCO₃ fouling and its
impact on closed-loop PID control of a Williams-Otto CSTR reactor:

- **Open-loop reactor dynamics** (`dynmodel.m`, `dynmodel_fouling.m`) —
  six-component reaction network with Arrhenius kinetics, a discretised
  cooling-coil temperature profile, and CaCO₃ fouling chemistry coupled
  to the coil heat transfer.
- **Closed-loop Simulink harness** (`WO_main_sim.slx`) — the production
  Stateflow-compiled plant (`WO_dyn`, 208-state) with two PID loops
  (level, temperature), setpoint/disturbance schedules, and the extracted
  production controller gains.
- **Controller-performance pipeline** (`CPI_calc.m`) — LMTD-based fouling
  KPI, EWMA filtering, and IAE/ISEu controller-performance indices against
  a benchmark window.
- **Clustering gain scheduling** (`clustering_gain_scheduling.m`) — Ward
  clustering into clean/fouled operating regimes plus a k-NN scheduler
  classifier.
- **Open-loop system identification** (`OL_sysid.m`, partial) — recorded
  static gains and SciPy second-order-plus-dead-time fits.
- **Isolated fouling-tube model** (`tubo_fouling.m`) — the same fouling
  kinetics on a single tube for validation in isolation.

## Provenance

Original MATLAB/Simulink work (2021): João Coutinho and Joel Sansana
(CIEPQPF, University of Coimbra), with José F.O. Granjo (CERENA, IST,
ULisboa). Fouling kinetics from Souza & Costa, *Chem Eng Res Des* 121,
15-31, 2019.

The MATLAB/Simulink sources were transcribed into Python on
`feature/audit-and-python-port` and then phased out of the working tree;
they remain retrievable from git history (baseline commit `4ba7d4b`) and
from the upstream repository,
https://github.com/joaoplcoutinho/wo_fouling_simulator.

**Read [PORT_NOTES.md](PORT_NOTES.md) first** — it maps every MATLAB
subsystem to its Python counterpart with a VERIFIED / TRANSCRIBED /
RECONSTRUCTED verdict, lists all port substitutions and their risks,
documents the validation against the collected `sim_results.mat` (all
physical channels of a 90-min closed-loop run land within 0.11 σ of the
recording), and records where the prior code audit was wrong (§4).
A module-by-module map of the Python package is in
[docs/architecture.md](docs/architecture.md).

## Install

Requires Python ≥ 3.10. Using [uv](https://docs.astral.sh/uv/):

```bash
uv venv .venv --python 3.12
uv pip install --python .venv -e ".[test]" scikit-learn h5py
```

(`scikit-learn` for the clustering pipeline; `h5py` only for reading the
v7.3 reference `sim_results.mat` in the golden tests. `skopt` is optional
— without it the k-NN hyperparameter search uses the documented
grid-search fallback, see PORT_NOTES §5.)

With plain pip:

```bash
python -m venv .venv && .venv/bin/pip install -e ".[test]" scikit-learn h5py
```

## Quickstart

Every runner lives in `wo_fouling.sim`; the package root re-exports the
common API. Full signatures: `help(wo_fouling)`.

```python
import wo_fouling as wf

# 1. Steady-state initialisation (initial.m / ssmodel.m) — reproduces the
#    production initial condition embedded in WO_reactor_model.slx.
#    Input vector: [FA, FB, TA, TB, Tcin, QR, Qcin, ca, pH]
import numpy as np
par = wf.default_parameters()      # closed-loop preset: psi=2.5, massT=2105, n=200
u = np.array([1.8275, 4.7875, 294.3, 294.3, 288.75,
              (1.8275 + 4.7875) / par.rhoR, 1.25e-3, 150e-5, 9.0])
x0, info = wf.solve_closed_loop_ss(u, par)
print(x0[0], x0[7])   # 2.3395 m, 371.095 K — the .slx golden IC is 371.0948 K

# 2. Open loop, clean (sim_main.m): 4-stage steady state + 10 h trajectory
sol, u, par = wf.sim.run_open_loop()

# 3. Open loop with fouling (dynmodel_fouling.m), 4 h from the closed-loop SS
sol, u, par = wf.sim.run_open_loop_fouling()

# 4. Isolated fouling tube (sim_tubo_fouling.m), 1 month — Rf grows,
#    df = kf*Rf*1000, and the pressure-drop ratio rises above 1
sol, u, par, sim = wf.sim.run_tube_fouling()

# 5. Closed loop (WO_main_sim.slx semantics): deterministic 90-min quiet
#    run — the configuration corroborated against sim_results.mat
res = wf.sim.run_closed_loop(t_end=90 * 60, fouling_on=False,
                             ta_sine=False, tb_sine=False, tcin_sine=False)
print(res.TR[-1], res.LR[-1])   # ≈ 371.095 K, ≈ 2.3395 m (recorded op. point)
# Disturbances are ON by default (24 h TA/TB/Tcin sines, fouling from 4 h):
# see wo_fouling.closed_loop.ClosedLoopConfig for every switch.
```

Offline analysis on the measured channels (both take a dict of channel
arrays on a 30 s grid, the upstream convention). The CPI comparison needs
at least one 4 h benchmark window, so run a 6 h campaign:

```python
from wo_fouling.kpi import kpi_pipeline

# 6-h campaign: quiet start, fouling activates at 4 h (production default)
res = wf.sim.run_closed_loop(t_end=6 * 3600, fouling_on=True,
                             ta_sine=False, tb_sine=False, tcin_sine=False)
data = {k: getattr(res, k)[::6] for k in
        ("t", "TR", "TA", "TB", "Tcin", "Tcout", "Qcin", "Rf", "TR_sp")}
data["Time"] = data.pop("t")
out = kpi_pipeline(data)    # KPI, KPI_filt, CPI_1/CPI_2 vs the 4-h benchmark,
print(out["KPI_filt"][-1])  # Spearman rho(Rf, KPI_filt), fouling indices
                           # (≈ 6.47 on the fouled leg; ≈ 6.83 clean)

from wo_fouling.clustering import clustering_pipeline
labels, model, info = clustering_pipeline(data)   # 2 clusters: clean/fouled
```

Parameter presets: `wf.open_loop_parameters()` (psi=1, massT=2105.2,
n=2000) and `wf.closed_loop_parameters()` (psi=2.5, massT=2105, n=200 —
matches the production .slx; this is also `wf.default_parameters()`).
The closed-loop controller gains embedded in the production .slx are
exposed as `wf.controllers.LR_C_GAINS` / `TR_C_GAINS`.

## Run the tests

```bash
uv run pytest            # or: .venv/bin/python -m pytest tests/ -q
```

96 tests covering every subsystem: kinetics, reactor, fouling, tube,
steady-state solvers, controllers/autotuner, closed-loop harness, KPI/CPI
arithmetic, clustering, and the golden-data corroboration against
`tests/fixtures/sim_results.mat` and the .slx-embedded initial condition.
Full run ≈ 4 min (the n=2000 steady-state and the 1-month tube cases are
the slow items). The `sim_results.mat` tests are skipped automatically
when `h5py` is missing.

## Layout

```
wo_fouling/                 the port — 13 modules (see docs/architecture.md)
tests/                      96 tests incl. golden-data corroboration
tests/fixtures/             reference recordings: sim_results.mat, model_ic.mat
docs/architecture.md        module-by-module map, data flow, test coverage
PORT_NOTES.md               port record: mapping, verdicts, tolerances, risks
WO_fouling_simulator.pptx   original project presentation (kept as-is)
```

License: GPL-3.0-or-later (matches the original work's licensing intent).
