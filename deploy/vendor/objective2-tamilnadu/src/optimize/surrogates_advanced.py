"""
src/optimize/surrogates_advanced.py
======================================
Two additional surrogate families, added to the three in `surrogates.py`
(nn_ensemble / xgboost / rsm_quadratic) because the review asked for
something more advanced than XGBoost.

ADDITIVE ON PURPOSE. Her three families stay exactly as they are and keep
competing, because the only honest way to answer "is this better than
XGBoost" is to measure it against XGBoost on the same folds of the same
data. An earlier attempt at this swapped XGBoost out on the strength of a
SYNTHETIC benchmark, where a GP beat it by 11.2x on RMSE. The lesson is
kept here as a rule -- the bake-off decides, not the model's reputation.

CORRECTED 2026-11-09. This docstring previously claimed "on the real DOE,
XGBoost won 21 of 27 groups and the GP came third". That is NOT what the
current results say, and it should not be quoted. Measured from
results/g8_s3_comparison.csv (360 rows, 9 pairs x 4 targets x 5 families
x 2 validation schemes), winner = lowest RMSE per group:

  per (pair, target) group, 36 groups per validation scheme:
    designs_5fold        stacked 14, gp_ard 13, rsm_quadratic 5, xgboost 4
    leave_one_year_out   stacked 14, gp_ard 14, rsm_quadratic 4, xgboost 4
  bases only (stacked removed), 72 groups pooled:
    gp_ard 42, rsm_quadratic 18, xgboost 12

So XGBoost wins 4 of 36, not 21 of 27, and the GP is FIRST among the base
families, not third. The "27 groups" figure came from an older 3-target
DOE and never applied to this 4-target one. Overall per-pair selection
(results/g8_s3_selection.csv) picks `stacked` in all 9 pairs.

  gp_ard        Gaussian-process regression with an ANISOTROPIC (ARD)
                Matern-5/2 kernel plus a fitted white-noise term, one model
                per target. Why this and not RBF: RBF assumes an infinitely
                smooth response, which over-smooths the kink where the PCM
                enters its melt band; Matern-5/2 is twice differentiable --
                smooth enough for an optimiser to follow, not so smooth
                that it erases structure. Two things it gives that no tree
                can: an exact closed-form posterior standard deviation
                (so active learning can sample where the model is really
                unsure, instead of where a tree committee happens to
                disagree), and a learned length scale per input, which
                doubles as a sensitivity readout cross-checkable against
                the Taguchi ANOVA.

  stacked       Out-of-fold stacked ensemble: each base family (GP, XGB,
                RSM, NN) predicts the held-out folds, and a non-negative
                ridge meta-learner is fitted on those out-of-fold
                predictions, per target. This is strictly more capable than
                picking one winner -- if the families make different errors
                it beats all of them, and if one dominates the meta-learner
                collapses onto it. Non-negative weights keep it from
                "fixing" one model with the negative of another, which
                overfits badly at n~128.

Both expose `fit(X, Y)` / `predict(X)` with the same shapes as her
classes, so `g8_s3_compare.py`-style code can use them interchangeably.
Feature matrix and TARGETS are imported from her module so the two sets
can never drift apart.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.optimize.surrogates import ARR, TARGETS, feature_matrix, NNEnsemble, XGB, RSMQuadratic


class GPARD:
    """Anisotropic Matern-5/2 GP, one model per target."""
    name = "gp_ard"

    def __init__(self, seed=0, n_restarts=4):
        self.seed, self.n_restarts = seed, n_restarts

    def fit(self, X, Y):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        self.xs = StandardScaler().fit(X)
        Xs = self.xs.transform(X)
        d = Xs.shape[1]
        self.models = []
        for j in range(Y.shape[1]):
            # Upper length-scale bound deliberately generous: an input the
            # response does not depend on wants an effectively infinite
            # length scale, and a tight bound turns that correct answer
            # into a convergence warning on every fit.
            kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                      * Matern(length_scale=np.ones(d),
                               length_scale_bounds=(1e-2, 1e4), nu=2.5)
                      + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1)))
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                          n_restarts_optimizer=self.n_restarts,
                                          random_state=self.seed + j)
            gp.fit(Xs, Y[:, j])
            self.models.append(gp)
        return self

    def predict(self, X, return_std=False):
        Xs = self.xs.transform(np.asarray(X, float))
        if return_std:
            out = [m.predict(Xs, return_std=True) for m in self.models]
            return (np.column_stack([o[0] for o in out]),
                    np.column_stack([o[1] for o in out]))
        return np.column_stack([m.predict(Xs) for m in self.models])

    def length_scales(self):
        """Learned length scale per input, per target. LARGE means that
        design variable barely matters for that target."""
        out = []
        for m in self.models:
            ls = getattr(getattr(m.kernel_, "k1", None), "k2", None)
            out.append(np.atleast_1d(getattr(ls, "length_scale", np.nan)))
        return np.array(out, dtype=object)


class Stacked:
    """Out-of-fold stacked ensemble over the base families."""
    name = "stacked"

    def __init__(self, seed=0, n_splits=5, bases=None):
        self.seed, self.n_splits = seed, n_splits
        self._base_factories = bases or [
            lambda s: GPARD(seed=s, n_restarts=2),
            lambda s: XGB(seed=s),
            lambda s: RSMQuadratic(),
            lambda s: NNEnsemble(seed=s),
        ]

    def fit(self, X, Y):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        n, n_t = len(X), Y.shape[1]
        n_b = len(self._base_factories)
        kf = KFold(n_splits=min(self.n_splits, n), shuffle=True, random_state=self.seed)

        # out-of-fold base predictions: (n, n_targets, n_bases)
        oof = np.full((n, n_t, n_b), np.nan)
        for tr, te in kf.split(X):
            for b, make in enumerate(self._base_factories):
                try:
                    m = make(self.seed).fit(X[tr], Y[tr])
                    oof[te, :, b] = np.asarray(m.predict(X[te]), float).reshape(len(te), n_t)
                except Exception:
                    oof[te, :, b] = np.nanmean(Y[tr], axis=0)   # degrade, do not crash

        # meta-learner per target, on the out-of-fold predictions only
        self.meta = []
        for j in range(n_t):
            A = oof[:, j, :]
            keep = np.isfinite(A).all(axis=1)
            r = Ridge(alpha=1e-6, positive=True, fit_intercept=True)
            r.fit(A[keep], Y[keep, j])
            self.meta.append(r)

        # refit the bases on ALL the data for prediction time
        self.bases = []
        for make in self._base_factories:
            try:
                self.bases.append(make(self.seed).fit(X, Y))
            except Exception:
                self.bases.append(None)
        self._n_t = n_t
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        preds = []
        for m in self.bases:
            if m is None:
                preds.append(np.zeros((len(X), self._n_t)))
            else:
                preds.append(np.asarray(m.predict(X), float).reshape(len(X), self._n_t))
        out = np.empty((len(X), self._n_t))
        for j in range(self._n_t):
            A = np.column_stack([p[:, j] for p in preds])
            out[:, j] = self.meta[j].predict(A)
        return out

    def weights(self):
        """Meta-learner weight per base family, per target -- reportable as
        'which surrogate actually carried each objective'."""
        names = [f(0).name for f in self._base_factories]
        return names, np.array([m.coef_ for m in self.meta])


ADVANCED_FAMILIES = (GPARD, Stacked)
