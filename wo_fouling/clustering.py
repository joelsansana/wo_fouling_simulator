"""Clustering + gain-scheduling classifier — port of
``clustering_gain_scheduling.m``.

Upstream pipeline:

1. read a CL CSV table (Time + measured channels + CPI), drop the Time
   column;
2. compute the LMTD KPI from the measured channels;
3. Ward linkage (euclidean) on the raw feature table;
4. cut into 2 clusters (``rng(1)`` for reproducibility — the cut itself is
   deterministic; the seed guards MATLAB's internal tie-breaking);
5. ``fitcknn`` with ``'OptimizeHyperparameters','auto'`` and
   expected-improvement acquisition — a Bayesian optimisation over the
   k-NN hyperparameters (MATLAB optimises ``NumNeighbors`` and
   ``Distance`` in 'auto' mode for k-NN).

Port substitutions (documented in PORT_NOTES.md):

* Ward linkage + maxclust cut: ``scipy.cluster.hierarchy`` — the linkage
  algorithm is the same (Ward's minimum variance, euclidean); results are
  deterministic in both.
* ``fitcknn``: ``sklearn.neighbors.KNeighborsClassifier`` with z-score
  standardisation off by default (MATLAB 'auto' may standardise).
* Bayesian hyperparameter search: MATLAB's ``bayesopt`` with
  expected-improvement. The closest faithful Python equivalent is
  ``skopt.BayesSearchCV`` (which also uses expected improvement by
  default). skopt is NOT a dependency of this package; when it is not
  installed the module falls back to a deterministic grid search over
  ``n_neighbors`` x ``metric`` with 5-fold CV, and records which search
  ran in the returned info dict. RISK: the fallback can pick a different
  (slightly suboptimal) hyperparameter point than bayesopt would; the
  cluster labels themselves (step 4) are unaffected.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

from .kpi import lmtd_kpi

FEATURE_COLUMNS = ["FA", "FB", "TA", "TB", "Tcin", "Tcout", "LR", "TR",
                   "QR", "Qcin", "DeltaP", "Rf", "TR_sp", "CPI"]


def ward_two_clusters(features: np.ndarray) -> np.ndarray:
    """Ward linkage (euclidean) + cut into 2 clusters.

    Mirrors ``Z = linkage(X,'ward','euclidean'); clust = cluster(Z,
    'MaxClust',2)``. Returns labels in {1, 2}.
    """
    features = np.asarray(features, dtype=float)
    Z = linkage(features, method="ward", metric="euclidean")
    return fcluster(Z, t=2, criterion="maxclust")


def _search_space():
    n_neighbors = list(range(1, 21))
    metrics = ["euclidean", "cityblock", "cosine"]
    return n_neighbors, metrics


def fit_knn_classifier(features: np.ndarray, labels: np.ndarray,
                       use_skopt: bool | None = None,
                       random_state: int = 1):
    """Fit the k-NN classifier with hyperparameter optimisation.

    Returns ``(model, info)``. ``info["search"]`` records which search
    strategy ran ("skopt-bayes" or "grid-cv-fallback") and the chosen
    hyperparameters.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV, cross_val_score

    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)

    n_neighbors, metrics = _search_space()

    if use_skopt is None:
        try:
            import skopt  # noqa: F401
            use_skopt = True
        except ImportError:
            use_skopt = False

    if use_skopt:
        from skopt import BayesSearchCV
        search = BayesSearchCV(
            KNeighborsClassifier(),
            {"n_neighbors": n_neighbors, "metric": metrics},
            n_iter=25, cv=5, random_state=random_state,
        )
        search.fit(features, labels)
        model = search.best_estimator_
        info = dict(search="skopt-bayes",
                    params=dict(n_neighbors=int(model.n_neighbors),
                                metric=model.metric),
                    best_score=float(search.best_score_))
        return model, info

    param_grid = [{"n_neighbors": [k], "metric": [m]}
                  for k in n_neighbors for m in metrics]
    gs = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    gs.fit(features, labels)
    model = gs.best_estimator_
    info = dict(search="grid-cv-fallback",
                params=dict(n_neighbors=int(model.n_neighbors),
                            metric=model.metric),
                best_score=float(gs.best_score_))
    return model, info


def clustering_pipeline(data: dict, feature_columns=None,
                        use_skopt: bool | None = None,
                        random_state: int = 1):
    """Full pipeline: KPI + Ward 2-cluster + k-NN classifier.

    ``data`` is a dict of 30 s-grid channel arrays with at least the keys
    used for the KPI (TR, TA, TB, Tcin, Tcout); the feature table is
    assembled from ``feature_columns`` (default: the measured channels +
    CPI, mirroring the CSV columns minus Time; missing keys are skipped).

    Returns ``(labels, model, info)``.
    """
    kpi = lmtd_kpi(data["TR"], data["TA"], data["TB"],
                   data["Tcin"], data["Tcout"])
    cols = feature_columns or FEATURE_COLUMNS
    features = np.column_stack([
        np.asarray(data[c], dtype=float)
        for c in cols if c in data and data[c] is not None
    ])

    labels = ward_two_clusters(features)
    model, fit_info = fit_knn_classifier(features, labels,
                                         use_skopt=use_skopt,
                                         random_state=random_state)
    info = dict(feature_columns=[c for c in cols
                                 if c in data and data[c] is not None],
                kpi=kpi, fit=fit_info,
                cluster_sizes={int(v): int(np.sum(labels == v))
                               for v in np.unique(labels)})
    return labels, model, info
