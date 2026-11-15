"""
src/optimize/anova.py
========================
O2 G8 / S1 - one-way-per-factor ANOVA for a balanced orthogonal array (Taguchi L9), after Chen et al. 2025
(Tables 13/17). Contribution % = SS_factor / SS_total. The error term comes from the weather-year replicates of each
run, so it is the year-to-year spread and NOT independent replication of the design: p-values are optimistic
(Chen's own five "repeat" columns have the same property) and are reported as a screening aid only.
"""

import pandas as pd
from scipy import stats

# Taguchi L9 (3^4): level indices 0..2 for factors A..D
L9 = [(0, 0, 0, 0), (0, 1, 1, 1), (0, 2, 2, 2),
      (1, 0, 1, 2), (1, 1, 2, 0), (1, 2, 0, 1),
      (2, 0, 2, 1), (2, 1, 0, 2), (2, 2, 1, 0)]


def anova(df: pd.DataFrame, factors, response: str) -> pd.DataFrame:
    y = df[response].astype(float)
    grand, n = y.mean(), len(y)
    sst = float(((y - grand) ** 2).sum())
    rows, ss_model, dof_model = [], 0.0, 0
    for f in factors:
        g = y.groupby(df[f])
        ss = float((g.count() * (g.mean() - grand) ** 2).sum())
        dof = g.ngroups - 1
        rows.append([f, ss, dof])
        ss_model += ss
        dof_model += dof
    sse, dfe = sst - ss_model, n - 1 - dof_model
    out = []
    for f, ss, dof in rows:
        F = (ss / dof) / (sse / dfe) if sse > 1e-300 and dfe > 0 else float("inf")
        p = float(stats.f.sf(F, dof, dfe)) if dfe > 0 and F != float("inf") else 0.0
        out.append({"factor": f, "response": response, "SS": ss, "dof": dof,
                    "contribution_pct": 100.0 * ss / sst if sst > 0 else 0.0, "F": F, "p": p})
    out.append({"factor": "error(year replicates)", "response": response, "SS": sse, "dof": dfe,
                "contribution_pct": 100.0 * sse / sst if sst > 0 else 0.0, "F": float("nan"), "p": float("nan")})
    return pd.DataFrame(out)
