"""
src/optimize/nsga2.py
========================
O2 G9 / S4 - NSGA-II (Deb et al. 2002) in numpy, because pymoo and DEAP are not installed and the brief says to
ask before any pip install. All objectives are MINIMISED. Genes live in [0, 1]^n (the caller decodes them, including
the categorical arrangement and the integer capsule count). Constraint handling is Deb's feasibility-first rule:
  feasible beats infeasible; two infeasible -> smaller violation wins; two feasible -> Pareto dominance.
Operators: simulated-binary crossover (eta 15, p 0.9) and polynomial mutation (eta 20, p 1/n); binary tournament on
(rank, crowding). Also: exact hypervolume for 2 and 3 objectives and the additive epsilon indicator, used to compare
fronts. Tested on ZDT1 and on a constrained problem in tests/test_nsga2.py.
"""

import numpy as np


# ---------------------------------------------------------------- dominance
def dominates(a, b):
    return bool(np.all(a <= b) and np.any(a < b))


def constrained_dominates(fa, ga, fb, gb):
    if ga <= 0 and gb > 0:
        return True
    if ga > 0 and gb <= 0:
        return False
    if ga > 0 and gb > 0:
        return ga < gb
    return dominates(fa, fb)


def non_dominated_mask(F):
    F = np.asarray(F, float)
    n = len(F)
    keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]:
            continue
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if dom.any():
            keep[i] = False
    return keep


def fast_non_dominated_sort(F, G):
    n = len(F)
    S = [[] for _ in range(n)]
    n_dom = np.zeros(n, int)
    for p in range(n):
        for q in range(p + 1, n):
            if constrained_dominates(F[p], G[p], F[q], G[q]):
                S[p].append(q)
                n_dom[q] += 1
            elif constrained_dominates(F[q], G[q], F[p], G[p]):
                S[q].append(p)
                n_dom[p] += 1
    fronts, cur = [], [i for i in range(n) if n_dom[i] == 0]
    while cur:
        fronts.append(np.array(cur))
        nxt = []
        for p in cur:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    nxt.append(q)
        cur = nxt
    return fronts


def crowding_distance(F):
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        d[:] = np.inf
        return d
    for j in range(m):
        o = np.argsort(F[:, j])
        d[o[0]] = d[o[-1]] = np.inf
        span = F[o[-1], j] - F[o[0], j]
        if span <= 0:
            continue
        d[o[1:-1]] += (F[o[2:], j] - F[o[:-2], j]) / span
    return d


# ---------------------------------------------------------------- operators
def sbx(p1, p2, rng, eta=15.0, pc=0.9):
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > pc:
        return c1, c2
    for i in range(len(p1)):
        if rng.random() < 0.5 and abs(p1[i] - p2[i]) > 1e-12:
            u = rng.random()
            beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
            c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
            c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)


def poly_mutation(x, rng, eta=20.0, pm=None):
    pm = 1.0 / len(x) if pm is None else pm
    y = x.copy()
    for i in range(len(x)):
        if rng.random() < pm:
            u = rng.random()
            d = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            y[i] = min(max(y[i] + d, 0.0), 1.0)
    return y


def _rank_and_crowd(F, G):
    fronts = fast_non_dominated_sort(F, G)
    rank = np.zeros(len(F), int)
    crowd = np.zeros(len(F))
    for r, fr in enumerate(fronts):
        rank[fr] = r
        crowd[fr] = crowding_distance(F[fr])
    return fronts, rank, crowd


def nsga2(evaluate, n_var, n_pop=100, n_gen=100, seed=0, X0=None):
    """evaluate(X: n x n_var) -> (F: n x m, G: n) with G <= 0 meaning feasible (G = total violation otherwise).
    Returns dict(X, F, G) of the final population."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_pop, n_var))
    if X0 is not None:                                   # seed part of the population (e.g. known good designs)
        X0 = np.asarray(X0, float)[: n_pop // 2]
        X[: len(X0)] = X0
    F, G = evaluate(X)
    for _ in range(n_gen):
        _, rank, crowd = _rank_and_crowd(F, G)
        def pick():
            a, b = rng.integers(0, n_pop, 2)
            return a if (rank[a], -crowd[a]) < (rank[b], -crowd[b]) else b
        kids = []
        while len(kids) < n_pop:
            c1, c2 = sbx(X[pick()], X[pick()], rng)
            kids += [poly_mutation(c1, rng), poly_mutation(c2, rng)]
        Xc = np.array(kids[:n_pop])
        Fc, Gc = evaluate(Xc)
        XA, FA, GA = np.vstack([X, Xc]), np.vstack([F, Fc]), np.concatenate([G, Gc])
        fronts = fast_non_dominated_sort(FA, GA)
        sel = []
        for fr in fronts:
            if len(sel) + len(fr) <= n_pop:
                sel += fr.tolist()
            else:
                cd = crowding_distance(FA[fr])
                sel += fr[np.argsort(-cd)[: n_pop - len(sel)]].tolist()
                break
        X, F, G = XA[sel], FA[sel], GA[sel]
    return {"X": X, "F": F, "G": G}


def pareto_set(F, G=None):
    """Indices of the non-dominated FEASIBLE rows."""
    F = np.asarray(F, float)
    idx = np.arange(len(F))
    if G is not None:
        idx = idx[np.asarray(G) <= 0]
    if len(idx) == 0:
        return idx
    return idx[non_dominated_mask(F[idx])]


# ---------------------------------------------------------------- indicators
def _hv2(F, ref):
    P = F[np.all(F < ref, axis=1)]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0])]
    hv, best_y = 0.0, ref[1]
    for x, y in P:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y)
            best_y = y
    return hv


def hypervolume(F, ref):
    """Exact dominated hypervolume for 2 or 3 minimised objectives w.r.t. reference point `ref`."""
    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    if F.shape[1] == 2:
        return _hv2(F, ref)
    if F.shape[1] != 3:
        raise NotImplementedError("hypervolume implemented for 2 or 3 objectives")
    F = F[np.argsort(F[:, 2])]
    hv = 0.0
    for i in range(len(F)):
        z_hi = F[i + 1, 2] if i + 1 < len(F) else ref[2]
        if z_hi > F[i, 2]:
            hv += _hv2(F[: i + 1, :2], ref[:2]) * (z_hi - F[i, 2])
    return hv


def additive_epsilon(A, B):
    """Smallest eps (per the max over objectives) such that every point of B is weakly dominated by some point of A
    shifted by eps: eps = max_b min_a max_k (a_k - b_k). Negative means A dominates B outright. Also returns the
    per-objective worst shortfall max_b min_a (a_k - b_k) at the minimising a for each b."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    worst = np.full(B.shape[1], -np.inf)
    eps = -np.inf
    for b in B:
        diff = A - b                                   # a - b, per objective
        j = int(np.argmin(diff.max(axis=1)))
        eps = max(eps, diff[j].max())
        worst = np.maximum(worst, diff[j])
    return float(eps), worst
