"""Pipeline tests: KPI/CPI (CPI_calc.m), clustering (clustering_gain_
scheduling.m), and the golden-data corroboration against sim_results.mat.

The sim_results.mat tests are skipped automatically when h5py is missing.
"""

from __future__ import annotations

import os
import numpy as np
import pytest

from wo_fouling.kpi import (
    lmtd_kpi, ewma_filter, controller_performance_indices, kpi_pipeline,
    fouling_threshold_indices, scale_by_max, spearman,
)
from wo_fouling.clustering import ward_two_clusters, clustering_pipeline
from wo_fouling.matio import load_sim_results, load_model_ic

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_RESULTS = os.path.join(REPO, "model_simulink_closed_loop", "sim_results.mat")
MODEL_IC = os.path.join(REPO, "model_matlab_open_loop", "model_ic.mat")

h5py = pytest.importorskip("h5py", reason="h5py not installed")


# -----------------------------------------------------------------------------
# KPI arithmetic (hand-computed golden values)
# -----------------------------------------------------------------------------

class TestKPI:
    def test_lmtd_kpi_hand_computed(self):
        # single-point hand computation:
        # TR=371.0948, TA=TB=294.5, Tcin=288.9, Tcout=351.1
        # dTR = 76.5948, dT1 = 5.6, dT2 = 19.9948
        # ln(5.6/19.9948) = -1.27267 -> dTml = 11.3109
        # KPI = 76.5948/11.3109 = 6.7721
        kpi = lmtd_kpi([371.0948], [294.5], [294.5], [288.9], [351.1])
        assert kpi[0] == pytest.approx(6.7721, abs=1e-4)

    def test_ewma_filter(self):
        kpi = np.array([6.0, 10.0, 10.0])
        f = ewma_filter(kpi, alpha=0.01, seed=6.0)
        assert f[0] == 6.0
        assert f[1] == pytest.approx(0.01 * 10 + 0.99 * 6)
        assert f[2] == pytest.approx(0.01 * 10 + 0.99 * f[1])

    def test_ewma_seed_6_and_7_upstream(self):
        """CPI_calc.m seeds the filter with 7 (fouling block) and 6
        (assessment block) — both must be accepted."""
        kpi = np.full(50, 6.8)
        assert ewma_filter(kpi, seed=7.0)[0] == 7.0
        assert ewma_filter(kpi, seed=6.0)[0] == 6.0

    def test_cpi_constant_for_steady_data(self):
        """Constant error and input: IAE windows all zero -> CPI_1 = 0/0
        = NaN (MATLAB parity); ISEu windows all equal -> CPI_2 exactly 1."""
        n = 4 * 480  # 4 windows of 4h at 30 s
        TR_sp = np.full(n, 371.0)
        TR = np.full(n, 371.0)
        Qcin = np.full(n, 1.25e-3)
        c1, c2, info = controller_performance_indices(TR, TR_sp, Qcin, 30.0)
        # 1441 = [1] + ZOH blocks of 480: benchmark block 1.0, then NaN
        assert np.allclose(c1[:481], 1.0)
        assert np.isnan(c1[481:]).all()    # 0/0 -> NaN, as in MATLAB
        assert np.allclose(c2, 1.0)
        assert info["IAE_bench"] == 0.0

    def test_cpi_drops_when_error_grows(self):
        """Tripled error in later windows -> CPI_1 ~ 1/3 there. With
        4x480 samples: Step = int(30*1919/14400) = 3, so 2 evaluated
        windows; the ZOH restructure yields [1] + [1.0]*480 + [1/3]*480."""
        n = 4 * 480
        TR_sp = np.full(n, 371.0)
        err = np.concatenate([np.full(2 * 480, 0.1), np.full(2 * 480, 0.3)])
        TR = TR_sp - err
        Qcin = np.full(n, 1.25e-3)
        c1, _, _ = controller_performance_indices(TR, TR_sp, Qcin, 30.0)
        assert c1[0] == 1.0
        # 1441 = [1] + 3 ZOH blocks of 480: windows 1-2 at 1.0, window 3 at 1/3
        assert np.allclose(c1[1:961], 1.0)
        assert np.allclose(c1[961:], 1.0 / 3.0, rtol=1e-9)

    def test_fouling_threshold_indices(self):
        kpi_filt = np.concatenate([np.full(10, 5.0), np.linspace(5.0, 7.0, 10)])
        idx = fouling_threshold_indices(kpi_filt, thresh=0.1)
        assert len(idx) >= 1
        assert idx[-1] == len(kpi_filt) - 1


# -----------------------------------------------------------------------------
# Clustering
# -----------------------------------------------------------------------------

class TestClustering:
    def _two_regime_table(self, n=40):
        """Two well-separated operating regimes (clean / fouled) with the
        measured-channel + CPI columns, 30 s grid."""
        rng = np.random.default_rng(1)
        clean = dict(
            FA=np.full(n, 1.8275), FB=np.full(n, 4.7875),
            TA=294.3 + 0.1 * rng.standard_normal(n),
            TB=294.3 + 0.1 * rng.standard_normal(n),
            Tcin=np.full(n, 288.75), Tcout=351.0 + 0.05 * rng.standard_normal(n),
            LR=np.full(n, 2.3395), TR=np.full(n, 371.0),
            QR=np.full(n, 8.26e-3), Qcin=np.full(n, 1.25e-3),
            DeltaP=np.full(n, 8.2e7), Rf=np.full(n, 1e-5),
            TR_sp=np.full(n, 371.0), CPI=np.full(n, 1.0))
        fouled = dict(
            FA=np.full(n, 1.8275), FB=np.full(n, 4.7875),
            TA=294.3 + 0.1 * rng.standard_normal(n),
            TB=294.3 + 0.1 * rng.standard_normal(n),
            Tcin=np.full(n, 288.75), Tcout=344.0 + 0.05 * rng.standard_normal(n),
            LR=np.full(n, 2.3395), TR=np.full(n, 371.5),
            QR=np.full(n, 8.26e-3), Qcin=np.full(n, 2.1e-3),
            DeltaP=np.full(n, 9.8e7), Rf=np.full(n, 3e-4),
            TR_sp=np.full(n, 371.0), CPI=np.full(n, 0.55))
        data = {k: np.concatenate([clean[k], fouled[k]]) for k in clean}
        data["Time"] = np.arange(2 * n) * 30.0
        return data

    def test_ward_separates_regimes(self):
        data = self._two_regime_table()
        from wo_fouling.clustering import FEATURE_COLUMNS
        X = np.column_stack([data[c] for c in FEATURE_COLUMNS])
        labels = ward_two_clusters(X)
        # each true regime is exactly one cluster
        assert len(set(labels[:40])) == 1
        assert len(set(labels[40:])) == 1
        assert labels[0] != labels[-1]

    def test_pipeline_end_to_end(self):
        data = self._two_regime_table(20)
        labels, model, info = clustering_pipeline(data)
        assert labels[0] != labels[-1]
        assert info["fit"]["best_score"] > 0.95
        # classifier reproduces the cluster assignment on the training data
        from wo_fouling.clustering import FEATURE_COLUMNS
        X = np.column_stack([data[c] for c in FEATURE_COLUMNS])
        assert (model.predict(X) == labels).mean() > 0.99

    def test_spearman(self):
        x = np.arange(10.0)
        assert spearman(x, x) == pytest.approx(1.0)
        assert spearman(x, -x) == pytest.approx(-1.0)


# -----------------------------------------------------------------------------
# Golden data: sim_results.mat / model_ic.mat
# -----------------------------------------------------------------------------

class TestSimResultsMat:
    """sim_results.mat (v7.3 HDF5, single timeseries, 11 channels,
    1080 samples at 5 s) — decoded via the wo_fouling.matio loader."""

    def test_loads_and_channel_mapping(self):
        from wo_fouling.matio import load_sim_results, CHANNEL_NAMES
        d = load_sim_results(SIM_RESULTS)
        assert set(CHANNEL_NAMES).issubset(d.keys())
        assert len(d["Time"]) == 1080
        assert d["Time"][1] - d["Time"][0] == pytest.approx(5.0)
        for name, lo, hi in [("FA", 1.7, 2.0), ("FB", 4.5, 5.1),
                             ("TA", 292, 297), ("TB", 292, 297),
                             ("Tcin", 287, 292), ("Tcout", 348, 354),
                             ("LR", 2.30, 2.37), ("TR", 370.5, 371.5),
                             ("QR", 7.4e-3, 9.0e-3), ("Qcin", 1.1e-3, 1.4e-3),
                             ("DeltaP", 7.0e7, 9.5e7)]:
            assert lo < d[name].mean() < hi, name

    def test_recorded_run_is_quiet_closed_loop(self):
        """The recording shows NO SP stair (TR stays within ~0.7 K) and
        no FA stair — this pins the configuration the port must
        reproduce with its deterministic defaults."""
        d = load_sim_results(SIM_RESULTS)
        assert d["TR"].max() - d["TR"].min() < 1.0
        assert d["FA"].max() - d["FA"].min() < 0.2

    def test_port_steady_state_within_recorded_noise_band(self):
        """GOLDEN corroboration: the ported closed loop (quiet config)
        settles inside mean ± 4σ of the recorded noisy run on every
        physical channel."""
        d = load_sim_results(SIM_RESULTS)
        from wo_fouling.closed_loop import (ClosedLoopSimulator,
                                            ClosedLoopConfig)
        cfg = ClosedLoopConfig(t_end=600.0, fouling_on=False,
                               ta_sine=False, tb_sine=False, tcin_sine=False)
        r = ClosedLoopSimulator(cfg=cfg).run()
        for name, port in [("TR", r.TR.mean()), ("LR", r.LR.mean()),
                           ("Qcin", r.Qcin.mean()), ("QR", r.QR.mean()),
                           ("Tcout", r.Tcout[-1]),
                           ("DeltaP", r.DeltaP.mean())]:
            rec = d[name]
            mu, sd = rec.mean(), rec.std()
            assert abs(port - mu) < 4 * sd + 1e-12 * abs(mu), (
                f"{name}: port {port} vs recorded {mu}±{4 * sd}")

    def test_kpi_on_recorded_data(self):
        """KPI computed from the recorded channels is ~6-7 (the same
        magnitude as the EWMA seeds 6/7 upstream)."""
        d = load_sim_results(SIM_RESULTS)
        kpi = lmtd_kpi(d["TR"], d["TA"], d["TB"], d["Tcin"], d["Tcout"])
        assert 5.5 < kpi.mean() < 7.5

    def test_model_ic_mat(self):
        from wo_fouling.matio import load_model_ic
        ic = load_model_ic(MODEL_IC)
        assert ic["n"] == 200
        assert len(ic["xFinal"]) == 209
        assert ic["Tc"][0] == pytest.approx(288.75, abs=0.5)
        # documented in PORT_NOTES: snapshot from a DIFFERENT configuration
        # (mt ~ 2261.6 kg, Tr ~ 363.95 K) — not the closed-loop SS.
        assert abs(ic["mt"] - 2261.6) < 1.0
