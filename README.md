# wo_fouling — Williams-Otto reactor with CaCO₃ fouling (Python port)

Python port of the MATLAB/Simulink simulation apparatus for studying
cooling-water CaCO₃ fouling and its impact on closed-loop PID control of
the Williams-Otto reactor, with the KPI / controller-performance /
clustering gain-scheduling pipeline. Original MATLAB work (2021):
João Coutinho, Joel Sansana (CIEPQPF, U. Coimbra), José F.O. Granjo
(CERENA, IST). Fouling kinetics from Souza & Costa, *Chem Eng Res Des*
121, 15-31, 2019.

The original MATLAB/Simulink sources were phased out of the working tree on
this branch (retrievable from git history, baseline commit `4ba7d4b`, and
from the upstream repository). Everything under `wo_fouling/` is the port.

**Read [PORT_NOTES.md](PORT_NOTES.md) first** — it maps every MATLAB
subsystem to its Python counterpart with a VERIFIED / TRANSCRIBED /
RECONSTRUCTED verdict, lists all port substitutions and their risks,
documents the validation against the collected `sim_results.mat`, and
records where the prior code audit was wrong (§4).

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

With plain pip: `python -m venv .venv && .venv/bin/pip install -e ".[test]" scikit-learn h5py`.

## Run the simulations

```python
import wo_fouling as wf

# open-loop clean (sim_main.m): steady state + 10 h trajectory
sol, u, par = wf.sim.run_open_loop()

# open-loop with fouling (dynmodel_fouling.m), 4 h from the closed-loop SS
sol, u, par = wf.sim.run_open_loop_fouling()

# isolated fouling tube (sim_tubo_fouling.m), 1 month
sol, u, par, sim = wf.sim.run_tube_fouling()

# closed loop (WO_main_sim.slx semantics): 90 min quiet run
res = wf.sim.run_closed_loop(t_end=90*60, fouling_on=False,
                             ta_sine=False, tb_sine=False, tcin_sine=False)

# KPI/CPI pipeline on a measured-channel dict (30 s grid)
from wo_fouling.kpi import kpi_pipeline
out = kpi_pipeline({"Time": res.t, "TR": res.TR, "TA": res.TA,
                    "TB": res.TB, "Tcin": res.Tcin, "Tcout": res.Tcout,
                    "Qcin": res.Qcin, "Rf": res.Rf, "TR_sp": res.TR_sp})

# clustering / gain scheduling
from wo_fouling.clustering import clustering_pipeline
labels, model, info = clustering_pipeline(data_30s_grid)
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
`sim_results.mat` and the .slx-embedded initial condition. Full run
≈ 4 min (the n=2000 steady-state and the 1-month tube cases are the slow
items).

## Layout

```
wo_fouling/            the port (parameters, kinetics, reactor, fouling,
                       tube, steady_state, controllers, closed_loop,
                       kpi, clustering, ol_sysid, matio, sim)
tests/                 96 tests incl. golden-data tests (reference .mat
                       recordings in tests/fixtures/)
PORT_NOTES.md          port record: mapping, verdicts, tolerances, risks
```

License: GPL-3.0-or-later (matches the original work's licensing intent).
