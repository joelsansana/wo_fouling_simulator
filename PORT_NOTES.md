# PORT_NOTES — wo_fouling Python port of the MATLAB/Simulink apparatus

Port executed 2026-11-03/04 by Joel Sansana (with Hermes agent support),
branch `feature/audit-and-python-port`. MATLAB tree is READ-ONLY (untouched,
`git diff` against the baseline commit shows no changes under
`model_matlab_open_loop/`, `model_simulink_closed_loop/`,
`simple_fouling_tube_sim/`).

`docs/AUDIT.md` (2026-07-20, Eusébio) was treated as **additional
information only**. Every claim used below was re-verified against the
actual `.m` sources and the extracted `.slx` XML; discrepancies are listed
in §4. Where this document and the audit disagree, this document wins.

---

## 1. Environment

```
uv venv .venv --python 3.12
uv pip install --python .venv -e ".[test]" scikit-learn h5py
.venv/bin/python -m pytest tests/ -q      # or: uv run pytest
```

Versions at validation time: Python 3.12, numpy 2.5.4, scipy 1.18.1,
numba 0.68.1, scikit-learn 1.9.1, h5py (latest), pytest 8.x.
No MATLAB and no Octave exist on this machine — reference re-runs are
impossible; validation is line-by-line transcription plus the collected
reference data (`sim_results.mat`, the .slx-embedded initial condition,
and the hard numbers inside the .m scripts). `skopt` is NOT installed;
the clustering hyperparameter search uses the documented fallback (§5).

## 2. .m → .py mapping and subsystem verdicts

Verdict scale: **VERIFIED** = checked against collected data (not just
transcription). **TRANSCRIBED** = line-by-line port, checked by
hand-computed values / self-consistency, no external reference run.
**RECONSTRUCTED** = exact upstream logic not fully recoverable; stated
assumptions.

| MATLAB/Simulink source | Python | Verdict | Evidence |
|---|---|---|---|
| `model_matlab_open_loop/modelparameters.m` | `parameters.py` (`open_loop_parameters()`) | TRANSCRIBED | all literals line-checked; psi=1, massT=2105.2, n=2000, pin=50e5 |
| `model_simulink_closed_loop/modelparameters.m` | `parameters.py` (`closed_loop_parameters()`, default) | VERIFIED | the 31-element flat vector matches the `model_parameters` Constant embedded in `WO_reactor_model.slx` element-by-element (psi=2.5, massT=2105, n=200) |
| `dynmodel.m` (kinetics + balances) | `kinetics.py`, `reactor.py` | TRANSCRIBED | rate laws, balances, coil scheme (centred interior, 2nd-order one-sided outlet, Dirichlet inlet), `trapz(-dT)` sign — all line-checked; RHS = 0 at the solved steady state to 1e-10 (test) |
| `dynmodel.m` solver wiring (`sim_main.m`) | `sim.py::run_open_loop` | TRANSCRIBED | ode15s→`solve_ivp(BDF)`, rtol 1e-6 / atol 1e-8 as in `sim_main.m`; AE row (level) resolved algebraically instead of Mass-matrix (§5) |
| `dynmodel_fouling.m` | `fouling.py` (`FoulingReactor` + parity `dynmodel_fouling`) | TRANSCRIBED + one VERIFIED anchor | fouling chemistry hand-checked at the clean SS (test); execution order preserved (dRfw/dt from pre-clamp Rfw; U-clamp resets Rfw=1e-10, df=1e-8 making g2 inconsistent — dead in practice); **dpdz at the .slx SS = 8.354e7 Pa vs recorded DeltaP 8.32e7 ± 0.27e7** (test) |
| `init.m` + the 4-stage block of `sim_main.m` | `steady_state.py::solve_open_loop_ss` | TRANSCRIBED | stage vectors/residuals line-checked; converges to max&#124;g&#124; = 1.5e-12; RHS≈0 at the solution. NOTE: the open-loop SS is a one-parameter family (QR = balanced flow ⇒ total mass conserved), so the solver returns the member nearest the upstream guess (mt=2366.9 kg from the upstream guess); `model_ic.mat`'s mt=2261.6 is a different member from a different history — documented, not an error |
| `ssmodel.m` + `initial.m` | `steady_state.py::solve_closed_loop_ss` | **VERIFIED** | reproduces the golden IC embedded in `WO_reactor_model.slx` `Integrator_ODE`: Lr0 = 2.3395 (= slx `LR_SP` const), Tr0 = 371.0954 vs 371.0948 (Δ 6e-4 K), masses match to <0.1% (test) |
| `simple_fouling_tube_sim/tubo_fouling.m` + `sim_tubo_fouling.m` | `tube.py` | TRANSCRIBED | analytic IC + clean dpdz IC reproduced; 1-month run: Rf monotone (≥ −1e-8 solver jitter), df = kf·Rf·1000 exact, ΔP ratio > 1 (test) |
| `WO_reactor_model.slx` chart_38 `WO_dyn` + AlgebraicConstraints | `closed_loop.py::_wo_dyn_kernel` | TRANSCRIBED (chart script transcribed verbatim from the XML, incl. the `fouling` switch, `dRfw_dt = -1` sentinel, no-Rfw-reset U-clamp, fouling-kinetics computed-then-discarded when off, N=200) |
| `WO_main_sim.slx` harness (PID blocks, SP/disturbances, integrator clamp) | `closed_loop.py::ClosedLoopSimulator` | **VERIFIED** | 90-min deterministic run vs `sim_results.mat` (1080 samples @5 s): **all 8 physical channels within 0.11 σ of the noisy recording** — TR 371.095 (z=0.01), LR 2.33947 (z=0.01), Qcin 1.250e-3 (z=0.03), QR 8.259e-3 (z=0.01), Tcout 351.20 (z=0.00), DeltaP 8.354e7 (z=0.11), FA/FB (z=0.07); KPI at the port SS 6.83 vs recorded 6.96 ± 0.53 (test + §6) |
| `WO_main_sim.slx` PID block parameters | `controllers.py` (`LR_C_GAINS`, `TR_C_GAINS`, `DiscretePID`) | **VERIFIED** | gains extracted from the .slx XML reproduce EXACTLY from `controller_tuning.m`'s recorded constants: TL(Kcu=−3.4628e-4, Pu=380) → P=−1.67069e-4 / I=−1.86395e-7 / D=−9.39903e-3 / N=0.03555 vs extracted −1.6707e-4 / −1.8639e-7 / −9.399e-3 / 0.03555; AMIGO-ITD(Kp=−1.76, θ=11.69) reproduces LR_C (tests) |
| `controller_tuning.m` tuning relations + 24 h retuning | `controllers.py` (`amigo_*`, `tyreus_luyben_pid`, `simc_pi`) | TRANSCRIBED | arithmetic reproduced (KP +187.5 %, KI +69.4 %, KD +388.1 % for the fouled 24 h relay retest — the paper's degradation deltas) |
| relay autotuner (Stateflow chart_40 `relay_exp` + chart_31) | `controllers.py::relay_autotune` | TRANSCRIBED (symmetric-relay branch: noise phase, exponential startup b=40, hysteresis relay, period-convergence ε, `Kcu = 4·d1/(π·√(amp²−h²))`, `Pu = ton+toff`) + RECONSTRUCTED details: the amplitude-management bookkeeping (`check_amplitudes`) is simplified for the fixed-amplitude symmetric case that produced the recorded tunings; validated only on a synthetic FOPDT plant (test) |
| `OL_sysid.m` | `ol_sysid.py` | RECONSTRUCTED (partial) | static-gain arithmetic + the recorded constants ported; the step-test `.mat` inputs live on João's OneDrive (absent from the repo) and `procest`/`idproc` are replaced by a SciPy SOPTD least-squares fit (§5) |
| `CPI_calc.m` | `kpi.py` | TRANSCRIBED | LMTD KPI, EWMA (α=0.01, seed 6/7), scaling, Spearman ρ, IAE/ISEu windowed CPIs with the ZOH restructure, threshold detector — line-checked + hand-computed goldens; two real port bugs found and fixed during testing (window indexing `1:Step-1`; 0/0→NaN parity) |
| `clustering_gain_scheduling.m` | `clustering.py` | TRANSCRIBED + RECONSTRUCTED search | Ward linkage (`scipy`, same algorithm, deterministic), MaxClust=2 cut, k-NN via sklearn; the Bayesian 'auto' search → documented fallback (§5) |
| `kpi_test.slx` | — (dropped) | DROPPED (verified) | extracted the XML: 4 channels of Backlash→Saturate[0,6]→RateLimiter reading `KPI_data.mat` — a scope rig, no dynamics. Its saturation bounds [0,6] independently corroborate the KPI ≈ 6–7 scale |
| KPI/KPI_fouling subsystems in `WO_main_sim.slx` | — (dropped, confirmed commented out) | — | `Commented="on"` in the XML; the KPI computation lives offline in `CPI_calc.m` (ported as `kpi.py`) |

## 3. Validation method and tolerances

No MATLAB exists here, so **no cross-run trajectory comparison is
possible**. Three evidence classes were used, in decreasing strength:

1. **Collected-data corroboration (VERIFIED).**
   `sim_results.mat` (v7.3/HDF5; single 11-channel `timeseries`,
   1080 samples @ 5 s = 90 min; channel order
   `FA FB TA TB Tcin Tcout LR TR QR Qcin DeltaP` identified from the
   magnitudes and harness wiring) — the port's deterministic 90-min
   closed-loop run must land inside the recording's noise band.
   **Metric: |port − recorded_mean| < 4·σ_recorded per channel; achieved
   max |z| = 0.11 (DeltaP), all others ≤ 0.07.** (test
   `test_port_steady_state_within_recorded_noise_band` +
   the full-scale run in §6.)
   The .slx-embedded `Integrator_ODE` initial condition (208 literals)
   — golden for the closed-loop SS solver: Tr within 6e-4 K, Lr0 within
   1e-4 m, masses within 0.1 %.
   The .slx PID block parameters — golden for the tuning relations:
   match to 4–5 significant figures.
2. **Hand-computed goldens.** Fouling chemistry, LMTD KPI, EWMA, CPI
   windows, AMIGO/TL arithmetic — each test carries the hand-derived
   number from the .m equations.
3. **Self-consistency.** RHS ≈ 0 at solved steady states; AE residuals
   g1/g2/g3 = 0 at consistent states; parity layer vs integrated layer
   agree to 1e-12; kernels never mutate the integrator state.

Tolerance summary: steady-state solver residual < 1e-10 (fsolve,
xtol 1e-13); SS RHS < 1e-6 kg/s and K/s; corroboration band 4σ (achieved
0.11σ); integration rtol 1e-6 / atol 1e-8 (clean) and 1e-6/1e-6
(fouling/closed loop), matching the .m `odeset` values.

**model_ic.mat** loads (209-vector, n=200 layout) and is used in tests to
pin the state layout — but it comes from a *different* configuration
(mt≈2261.6 kg, Tr≈363.95 K, unknown exact inputs), so it cannot serve as
a trajectory golden; this is stated explicitly in the test.

## 4. Audit-vs-reality discrepancies (all re-verified on the artifacts)

1. **"n = 200 default" (audit §2.2/§3.1) — FALSE for the open-loop
   tree.** `model_matlab_open_loop/modelparameters.m` has `n = 2000`;
   the closed-loop file has `n = 200`. The audit's §2.4 parameter table
   silently used the closed-loop values as canonical (see also 2).
2. **The two `modelparameters.m` are NOT "bit-identical" (audit implied
   one parameter set).** They differ in psi (1 vs 2.5), massT (2105.2 vs
   2105.0), n (2000 vs 200), and the closed-loop tree stores `par.Vr`.
   The port exposes both presets explicitly.
3. **"`modelparameters.m:25` defines `MR = par.massT`" (audit §5.1) —
   wrong location and half-wrong substance.** Neither `modelparameters.m`
   defines `MR`. `ssmodel.m:68` does (`MR = par.massT`) — consistent
   there, because `ssmodel` solves in mass-fraction space where massT is
   the scale. Crucially, the **chart-38 `WO_dyn` computes
   `MR = ma+mb+…+mp` from the state** (verified in the XML) — the audit's
   claim that the production closed-loop dynamics force MR = massT is
   FALSE, and its "port decision" (prefer the open-loop definition) is
   moot: both do the same thing.
4. **The central chart claim is TRUE.** `chart_38.xml` inside
   `WO_reactor_model.slx` does contain a production `WO_dyn` MATLAB-
   function compile distinct from the open-loop .m files: state
   `[ma..mp, Tr, Tc(1..200), Rf]` (208; L/df/dpdz via AlgebraicConstraint
   blocks — the `kf*Rf−df` block is actually wired with a `kf*1000`
   = 2.941 constant), hardcoded N=200, `fouling` 0/1 switch with
   `dRfw_dt = −1` sentinel, and a U-clamp that resets only U (no
   `Rfw=1e-10/df=1e-8` reset, unlike `dynmodel_fouling.m`). Additional
   detail the audit missed: `Integrator_ODE` lower-saturates all 208
   states at 0 (`LowerSaturationLimit = zeros(208,1)`), which replaces
   the .m files' warning-clamps. The port follows the chart version for
   the closed loop, as ordered.
5. **Audit §5.9 mischaracterises stage 0.** `tcopt=0` does not merely
   "check" the analytic profile: it solves the same 12+n unknowns as
   stage 1, with the coil residual `g9 = Tcan − Tc` (distance to the
   analytic profile) instead of the discretised balance. Both stages
   share the unknown set; the residual differs.
6. **`sim_results.mat` is not "pre-collected closed-loop results used as
   input to clustering" in the CSV sense (audit §3.2).** It is a single
   90-min, 11-channel, 5 s timeseries (the `Meas_var` recording of a
   quiet, noise-on run: no SP stair, no FA stair — verified from the
   data). The CL CSVs the clustering script consumed are indeed absent
   (audit §9 correct on that).
7. Minor: audit §2.2 says "11+N = 211 for the default config" — wrong
   under its own (n=200) default (would be 211 only for n=200 —
   actually consistent; but the *open-loop* default n=2000 gives 2011).
   Audit §2.4 lists ψ=2.5 and massT=2105 without noting they are the
   closed-loop values only.
8. Confirmed-true audit items worth keeping: KPI/KPI_fouling subsystems
   commented out; recycle streams are dead code; the Tc outlet uses the
   0.5/−2/1.5 second-order one-sided scheme; `kpi_test.slx` is a
   post-processing rig; no CSV fixtures / no unit tests upstream; the
   `.slx` SP constants (LR_SP 2.3395, TR_SP 371.0948) — the latter pair
   *re-derived* by the ported `initial.m` to 6e-4 K.

## 5. Port substitutions and their risks

| MATLAB | Python | Risk |
|---|---|---|
| `fsolve` (trust-region dogleg; central FD in `initial.m`) | `scipy.optimize.fsolve` (MINPACK hybrid) | different step heuristic; both drive the same residuals to <1e-10. Observed: open-loop stages −2/1 return ier=5 ("not making good progress") **at residual ≤ 1.5e-12** — accepted, tolerance met |
| `ode15s` (variable-order BDF) + singular Mass matrix | `solve_ivp(BDF)` + AE rows resolved algebraically inside the RHS (exactly what the Simulink AlgebraicConstraint blocks do) | none structural; solver order/adaptivity differ so *trajectories* would differ at tolerance level from ode15s — no MATLAB exists to quantify; tolerances matched (rtol 1e-6) |
| pid_lib discrete PID (2DOF) | `DiscretePID` (backward-Euler integrator, filtered derivative `a = TsN/(1+TsN)`, clamping anti-windup, parallel gains) | the .slx `InitialConditionForFilter = 23.1105` is the filter's internal state, not its output; feeding it as output adds +3.49 m³/s to Qcin and wrecks the loop (reproduced, then fixed). The port resets the D-output to 0 at t=0 — pinned by the recorded Qcin ≈ 1.25e-3. RESIDUAL RISK: pid_lib's exact internal discretisation is not documented in the XML; step-response-level equivalence to Simulink is not bit-proven |
| `bayesopt` (expected improvement) in `fitcknn('OptimizeHyperparameters','auto')` | `skopt.BayesSearchCV` if skopt is installed, else deterministic GridSearchCV (k∈1..20 × {euclidean, cityblock, cosine}, 5-fold) | fallback may pick a different hyperparameter point than bayesopt; cluster labels (Ward cut) unaffected; which search ran is recorded in the result info |
| `procest`/`idproc('P2DU')` | SciPy `least_squares` SOPTD step fit | different objective (simulation-error LS vs prediction-error); fitted (K, ωn, ζ, L) may differ from MATLAB's on the same data; input .mat files absent anyway |
| AR(3) disturbance blocks (RandomNumber seeds 1..11, Ts=30 s, φ=[0.1,0.25,0.6], var 2e-10) | optional seeded numpy AR(3), OFF by default | MATLAB RNG streams are not bit-reproducible; the block-level wiring (which φ multiplies which delay) was not fully recoverable from the XML — RECONSTRUCTED, deterministic core defaults to the bias path (matches the recorded run's inactive switches) |
| Sensor transport delays (5 s / 30 s) + Gaussian noise | ideal sensors by default | same class as above; the deterministic core is what the recorded run corroborates |

Known non-portable: MATLAB `warning()` calls (dropped — behaviour
identical, clamps retained), `pin=50e5` (unused downstream), plots.

## 6. sim_results.mat usage (done-criterion 3)

Used in THREE golden tests (`tests/test_pipeline.py`):

1. structure/channel-mapping test (11 channels, 1080 samples @5 s);
2. configuration pinning: the recording is a *quiet* run (TR span < 1 K,
   FA span < 0.2) — this is what justified the port's deterministic
   default configuration;
3. **steady-state corroboration**: the ported closed loop's 90-min run
   lands within 0.11 σ of the recording on all physical channels:

```
channel   recorded mean  recorded sd    port value   |diff|/sd
TR          371.095        0.105        371.095      0.01
LR            2.33954     0.01028        2.33947     0.01
Qcin       0.00124899    3.588e-05     0.00124999    0.03
QR         0.0082562    0.0002063     0.00825925     0.01
Tcout       351.197        1.032        351.201      0.00
DeltaP     8.32499e+07    2.727e+06    8.35374e+07   0.11
FA           1.82981      0.03089        1.8275      0.07
FB           4.78041       0.1005        4.7875      0.07
KPI(recorded) = 6.96 ± 0.53   vs  KPI(port SS) = 6.83
```

What it cannot corroborate: fouling dynamics (the recording is clean —
no Rf channel, 90 min ≪ fouling timescale), transient trajectories
(no MATLAB for cross-runs), and the clustering CSVs (absent).

## 7. What Paper C can now claim (factual, one paragraph)

The complete simulation apparatus — Williams-Otto reactor dynamics with
CaCO3 fouling (Souza–Costa kinetics), steady-state initialisation,
closed-loop PID control with the production chart-compiled plant, relay
autotuning, the LMTD-based fouling KPI, IAE/ISEu controller-performance
indices, and the Ward-clustering + k-NN gain-scheduling pipeline — is
reproducible end-to-end in open-source Python (NumPy/SciPy/scikit-learn,
GPL-3.0), with 96 passing tests. The closed-loop core is validated
against the collected 90-minute plant recording: every physical channel
reproduces the recorded operating point within 0.11 standard deviations
of the measurement noise, the steady-state initialisation reproduces the
production initial condition to 6e-4 K, and the deployed PID gains are
re-derived exactly from the recorded relay/tau-tuner constants via the
AMIGO and Tyreus–Luyben relations, including the +187 %/+69 %/+388 %
gain changes measured after 24 h of fouling. Subsystems without
collected reference data (open-loop trajectories, tube rig, KPI/clustering
pipeline internals) are line-by-line transcriptions with hand-checked
goldens, and every such judgement is documented per subsystem in
PORT_NOTES.md. (The manuscript rewrite itself is out of scope here.)

## 8. Test-suite summary

Baseline (before this port): 52 tests, 10 failing (NumPy-2 `np.trapz`
removal inside the Numba kernel) + 1 false-premise test fixture.
Final: **96 tests, all passing** (`uv run pytest`, ~3.7 min; the slow
items are the n=2000 open-loop steady state and the 1-month tube run).

Commits on `feature/audit-and-python-port`:
`4ba7d4b` baseline snapshot → `2f757d9` port fixes (trapz, state
mutation, presets, test fixture) → `69acfa5` Qcool sign → `643e140`
fouling/tube/steady-state/controllers → `d786ef1` closed-loop harness
(+filter-IC fix) → `2060c67` KPI/CPI + clustering + readers + golden
tests → final docs commit.
