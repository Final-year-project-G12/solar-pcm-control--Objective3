"""
src/optimize/surrogates.py
=============================
O2 G8 / S3 (plan WP5 5.5) - the three surrogate families compared on the DOE of ONE (regime, O1 PCM) pair:

  nn_ensemble  5 small feed-forward networks (32, 32), bootstrap resamples, different seeds, scikit-learn
               MLPRegressor (no PyTorch, per the brief); mean = prediction, std across members = uncertainty
  xgboost      one gradient-boosted tree model per target (Yan et al. 2025's best model)
  rsm_quadratic  full second-order polynomial + ridge, one model per target (Li et al. 2020)

Inputs (all families): inner radius [mm], PCM fraction, flow [kg/s], one-hot arrangement (3) -- PCM identity is not an
input (one model per pair). Standardised on the training fold only. Nothing here replaces the simulator: a surrogate
only ranks candidates; every reported number is a simulator run.
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from xgboost import XGBRegressor

ARR = ("single-layer", "staggered", "radial")
TARGETS = ("g_window", "g_annual", "d_overheat", "t_pcm_max")


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    onehot = np.column_stack([(df["arrangement"] == a).astype(float) for a in ARR])
    return np.column_stack([df["R_in_m"] * 1000.0, df["fraction"], df["flow"], onehot])


class NNEnsemble:
    name = "nn_ensemble"

    def __init__(self, n_members=5, hidden=(32, 32), seed=0):
        self.n_members, self.hidden, self.seed = n_members, hidden, seed

    def fit(self, X, Y):
        self.xs, self.ys = StandardScaler().fit(X), StandardScaler().fit(Y)
        Xs, Ys = self.xs.transform(X), self.ys.transform(Y)
        rng = np.random.default_rng(self.seed)
        self.members = []
        for m in range(self.n_members):
            idx = rng.integers(0, len(Xs), len(Xs))                 # bootstrap resample
            net = MLPRegressor(hidden_layer_sizes=self.hidden, activation="tanh", solver="adam",
                               learning_rate_init=0.01, alpha=1e-3, max_iter=3000, tol=1e-6,
                               n_iter_no_change=60, early_stopping=False, random_state=self.seed * 100 + m)
            self.members.append(net.fit(Xs[idx], Ys[idx]))
        return self

    def predict(self, X, return_std=False):
        Xs = self.xs.transform(X)
        P = np.stack([m.predict(Xs) for m in self.members])        # members x n x targets
        mean = self.ys.inverse_transform(P.mean(axis=0))
        if return_std:
            return mean, P.std(axis=0) * self.ys.scale_
        return mean


class XGB:
    name = "xgboost"

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, Y):
        self.models = [XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=1.0, reg_lambda=1.0, min_child_weight=2,
                                    random_state=self.seed, n_jobs=1).fit(X, Y[:, j]) for j in range(Y.shape[1])]
        return self

    def predict(self, X):
        return np.column_stack([m.predict(X) for m in self.models])


class RSMQuadratic:
    name = "rsm_quadratic"

    def fit(self, X, Y):
        self.xs = StandardScaler().fit(X)
        self.poly = PolynomialFeatures(2, include_bias=False)
        self.model = Ridge(alpha=1e-3).fit(self.poly.fit_transform(self.xs.transform(X)), Y)
        return self

    def predict(self, X):
        return self.model.predict(self.poly.transform(self.xs.transform(X)))


FAMILIES = {"nn_ensemble": NNEnsemble, "xgboost": XGB, "rsm_quadratic": RSMQuadratic}
