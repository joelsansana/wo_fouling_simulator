# Architecture — `wo_fouling`

Module-by-module map of the Python port, the data flow between modules,
and what each test file covers. Every upstream reference and verdict below
is recorded per subsystem in [PORT_NOTES.md](../PORT_NOTES.md) §2; API
names are the actual module contents (see each module docstring).

The package has 13 modules in four layers:

1. **Physics kernels** — `parameters`, `kinetics`, `reactor`, `fouling`, `tube`
2. **Initialisation** — `steady_state`
3. **Closed loop** — `controllers`, `closed_loop`
4. **Offline analysis & I/O** — `kpi`, `clustering`, `ol_sysid`, `matio`,
   plus `sim` (scenario runners) at the top.

## Module map

| Module | Role | Upstream source | Key API | Port verdict |
|---|---|---|---|---|
| `parameters.py` | Parameter container, both presets | the two `modelparameters.m` | `WoParameters`, `open_loop_parameters()` (psi=1, massT=2105.2, n=2000), `closed_loop_parameters()` (psi=2.5, massT=2105, n=200; also `default_parameters()`) | TRANSCRIBED / VERIFIED (closed-loop flat vector matches the .slx Constant element-by-element) |
| `kinetics.py` | Arrhenius rate constants, mass-basis reaction rates, heat release | `dynmodel.m` kinetics block | `log_rate_constants()`, `mass_rates()`, `reaction_heat()` | TRANSCRIBED |
| `reactor.py` | Open-loop clean RHS: state `11 + N` (level AE row, 6 masses, Tr, coil profile) | `dynmodel.m` | `OpenLoopReactor` (`.f()`, `.initial_state()`, `.nominal_inputs()`) | TRANSCRIBED |
| `fouling.py` | Open-loop fouling RHS: adds `Rfw` state + layer/pressure-drop AE rows; CaCO₃ chemistry | `dynmodel_fouling.m` | `FoulingReactor`, parity function `dynmodel_fouling()` | TRANSCRIBED + VERIFIED dpdz anchor (8.354e7 Pa vs recorded 8.32e7 ± 0.27e7) |
| `tube.py` | Isolated single-tube fouling rig, same kinetics, driven by `Tamb` | `tubo_fouling.m` + `sim_tubo_fouling.m` | `TubeSimulator`, `tubo_fouling()`, `tube_initial_state()` | TRANSCRIBED |
| `steady_state.py` | 4-stage homotopy steady-state solvers; the closed-loop solver reproduces the .slx-embedded IC to 6e-4 K | `init.m`, `ssmodel.m`, `initial.m` | `solve_open_loop_ss()`, `solve_closed_loop_ss()` | TRANSCRIBED / VERIFIED |
| `controllers.py` | Discrete PID (backward-Euler + filtered derivative + anti-windup), SIMC/AMIGO/Tyreus-Luyben tuning, relay autotuner; production gains extracted from the .slx XML | `controller_tuning.m`, PID blocks, Stateflow `relay_exp` | `DiscretePID`, `relay_autotune()`, `tyreus_luyben_pid()`, `amigo_pid()`, `simc_pi()`; constants `LR_C_GAINS`, `TR_C_GAINS` | VERIFIED gains / TRANSCRIBED relations / RECONSTRUCTED autotuner bookkeeping |
| `closed_loop.py` | Closed-loop harness: plant = Stateflow `WO_dyn` chart (208-state production compile), PID loops, SP/disturbance schedules | `WO_main_sim.slx`, `WO_reactor_model.slx` | `ClosedLoopSimulator`, `ClosedLoopConfig` (sines, SP stair, FA stair, `fouling_on`, AR noise), `ClosedLoopResult` (`.measured_table()`) | TRANSCRIBED chart / VERIFIED harness (90-min run within 0.11 σ of `sim_results.mat` on all channels) |
| `kpi.py` | LMTD fouling KPI, EWMA filter, IAE/ISEu windowed CPIs vs benchmark window, threshold detector | `CPI_calc.m` | `lmtd_kpi()`, `ewma_filter()`, `controller_performance_indices()`, `fouling_threshold_indices()`, `kpi_pipeline()` | TRANSCRIBED |
| `clustering.py` | Ward linkage → 2 operating regimes → k-NN gain-scheduling classifier | `clustering_gain_scheduling.m` | `ward_two_clusters()`, `fit_knn_classifier()`, `clustering_pipeline()`; `FEATURE_COLUMNS` | TRANSCRIBED + RECONSTRUCTED hyperparameter search (skopt or documented grid fallback) |
| `ol_sysid.py` | Static-gain arithmetic from recorded step tests + SciPy SOPTD fits | `OL_sysid.m` | `static_gains()`, `fit_sopdt()`; `RECORDED` constants | RECONSTRUCTED (partial — step-test .mat inputs absent from the repo) |
| `matio.py` | Readers for the reference recordings (v7.3 HDF5 timeseries; v7 snapshot) | — | `load_sim_results()`, `load_model_ic()`; `CHANNEL_NAMES` | — (I/O only) |
| `sim.py` | Thin scenario runners wiring the above (ode15s→BDF tolerances preserved) | `sim_main.m`, `script_sim_main.m`, `sim_tubo_fouling.m` | `run_open_loop()`, `run_open_loop_fouling()`, `run_tube_fouling()`, `run_closed_loop()` | TRANSCRIBED |

## Data flow

```
                parameters.py   WoParameters (open-loop / closed-loop presets)
                       │
   ┌───────────────────┼───────────────────────────┐
   ▼                   ▼                           ▼
kinetics.py      reactor.py / fouling.py        tube.py
(rate laws)      open-loop RHS, clean/fouled    isolated tube rig
   │                   │                           │
   │  steady_state.py: 4-stage initialisation (feeds both loops)
   ▼                   ▼
   │            closed_loop.py  ClosedLoopSimulator
   │              + controllers.py (DiscretePID ×2, schedules)
   │                       │
   │                       ▼  ClosedLoopResult  (FA..DeltaP, Rf, TR_sp, 5 s grid)
   │           ┌───────────┴────────────┐
   │           ▼                        ▼
   │       kpi.py                  clustering.py
   │   KPI/CPI offline          2-regime k-NN classifier
   │                                     │
   │                                 ol_sysid.py (SOPTD step fits)
   │
   └── sim.py: scenario runners over every path above
       matio.py: reads tests/fixtures/{sim_results.mat, model_ic.mat}
       (golden reference for the tests; not used by the solvers)
```

Two state layouts matter when moving between layers: the open-loop
kernels use `11 + n` (`[L, ma..mp, Tr, Tc(1..n)]`, plus `Rfw, df, dpdz`
rows in the fouling parity function), while the closed-loop plant uses the
chart's hardcoded 208-state form (`[ma..mp, Tr, Tc(1..200), Rfw]`, with
level/df/DeltaP resolved by AlgebraicConstraints). `closed_loop.py`
converts between them via its own initial-state helpers.

## Test coverage

96 tests (`pytest tests/`, ≈ 4 min; slow items are the n=2000 open-loop
steady state and the 1-month tube run).

| Test file | Covers |
|---|---|
| `test_parameters.py` | both `modelparameters.m` presets; the closed-loop flat vector vs the .slx Constant |
| `test_kinetics.py` | Arrhenius constants vs the prior audit's values; rate and heat-release arithmetic |
| `test_reactor.py` | state layout, Numba kernel, hold-up conservation, sign conventions |
| `test_fouling.py` | hand-computed fouling chemistry at the clean SS; parity layer vs integrated layer; the dpdz anchor vs the recorded DeltaP |
| `test_tube.py` | analytic IC; 1-month run: Rf monotone, `df = kf·Rf·1000` exact, ΔP ratio > 1 |
| `test_steady_state.py` | 4-stage convergence (residual < 1e-10); RHS ≈ 0 at the solution; closed-loop SS vs the .slx-embedded golden IC (Tr within 6e-4 K) |
| `test_closed_loop.py` | harness semantics; extracted PID gains reproduce `controller_tuning.m` constants (Tyreus-Luyben, AMIGO); DiscretePID step behaviour incl. the filter-IC interpretation |
| `test_pipeline.py` | KPI/CPI hand-computed goldens (incl. the 0/0→NaN and window-indexing parity cases); clustering on synthetic two-regime data; golden corroboration: `sim_results.mat` structure + quiet-run config + 90-min closed-loop run within 0.11 σ per channel; `model_ic.mat` state-layout pinning |

## Where the ground truth lives

- `tests/fixtures/sim_results.mat` — the collected 90-min, 11-channel,
  5 s recording used for the closed-loop corroboration (PORT_NOTES §3, §6).
- `tests/fixtures/model_ic.mat` — the 209-vector open-loop snapshot used to
  pin the state layout (PORT_NOTES §3).
- The .slx-embedded constants (initial condition, PID gains, SP values) —
  quoted in PORT_NOTES §2 and asserted in the tests.
- The original MATLAB/Simulink tree — removed from the working tree;
  retrievable in git history at baseline commit `4ba7d4b` and from the
  upstream repository.
