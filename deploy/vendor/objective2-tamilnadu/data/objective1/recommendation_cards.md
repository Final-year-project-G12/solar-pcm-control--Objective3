# Objective 1 — Recommendation Cards (Tamil Nadu)

Generated from 3 climate regimes (GMM clustering, 133 population points).

**Physics validation summary (Phase 7):** mean Spearman rho across clusters = 0.169 (MCDM consensus rank vs. simulated annual solar fraction, grey-box lumped-enthalpy tank model driven by each cluster's medoid point's real 10-year daily climate data). See `10_physics_validation.py`'s docstring for the full stated assumption list (tank size, collector efficiency, draw schedule) before quoting this number without qualification.


## Cluster 0

- **Points in regime:** 46
- **Population covered:** 20,468,850
- **Medoid point (highest membership confidence):** TNP_0006 (10.875, 78.625)

**Climate signature (population-weighted mean):**

| Index | Value |
|---|---|
| GHI_daily_kWh | 5.318 |
| Ta_mean | 29.672 |
| DTR_true | 9.915 |
| kt_daily_mean | 0.811 |
| cloudy_frac | 0.011 |
| CCI | 2.739 |
| HDD18 | 0.000 |
| CDD24 | 1282.748 |
| RH_sunrise_mean | 81.417 |
| HSI_sunrise | 24.517 |
| monsoon_index | 0.980 |

**Derived targets:** Tm_target = 67.0 C, L_required = 406 kJ/kg

**Candidates screened:** 8 survived Phase 5 feasibility filtering (melting window, absolute band, latent-heat floor, cycling, supercooling, corrosion veto, safety exclusion)

**Top-3 PCM candidates (Borda consensus of TOPSIS + GRA + PROMETHEE II + VIKOR):**

| Rank | PCM | Family | Tm (C) | Latent heat (kJ/kg) | TOPSIS | GRA | PROMETHEE | VIKOR_Q | MC Top-3 % |
|---|---|---|---|---|---|---|---|---|---|
| 1 | RT57HC |  | 56.5 | 240 | 0.633 | 0.117 | +0.258 | 0.085 | 86.5% |
| 2 | n-Hexacosane (C26) |  | 56.5 | 256 | 0.600 | 0.115 | +0.272 | 0.096 | 75.4% |
| 3 | PureTemp 58 |  | 58.0 | 225 | 0.546 | 0.113 | +0.135 | 0.271 | 36.1% |

*Kendall's W (4-method concordance) = 0.848 (strong agreement)*

**Phase 7 — simulated annual performance (grey-box lumped-enthalpy tank, real climate data):**

| PCM | Consensus rank | Simulated solar fraction | In 54-84% benchmark band? | Complete cycles/yr |
|---|---|---|---|---|
| RT57HC | 1 | 55.4% | Yes | 167 |
| n-Hexacosane (C26) | 2 | 52.1% | No | 167 |
| PureTemp 58 | 3 | 51.9% | No | 143 |
| PlusICE A58 | 4 | 51.9% | No | 143 |
| CrodaTherm 60 | 5 | 38.3% | No | 110 |

*Spearman rho (MCDM rank vs. simulated solar fraction) for this cluster: 0.381 — weak agreement — diagnose before trusting the MCDM ranking here*

**Caveats:** thermal conductivity / density / specific heat not reported in the source data for the literature-added candidates (see 06_build_pcm_database.py); Phase 7's tank/collector parameters are stated assumptions, not measurements (see 10_physics_validation.py's docstring).


## Cluster 1

- **Points in regime:** 41
- **Population covered:** 20,422,159
- **Medoid point (highest membership confidence):** TNP_0007 (10.875, 76.875)

**Climate signature (population-weighted mean):**

| Index | Value |
|---|---|
| GHI_daily_kWh | 5.238 |
| Ta_mean | 27.105 |
| DTR_true | 11.107 |
| kt_daily_mean | 0.803 |
| cloudy_frac | 0.011 |
| CCI | 2.629 |
| HDD18 | 0.006 |
| CDD24 | 830.192 |
| RH_sunrise_mean | 83.769 |
| HSI_sunrise | 22.437 |
| monsoon_index | 0.912 |

**Derived targets:** Tm_target = 67.0 C, L_required = 438 kJ/kg

**Candidates screened:** 10 survived Phase 5 feasibility filtering (melting window, absolute band, latent-heat floor, cycling, supercooling, corrosion veto, safety exclusion)

**Top-3 PCM candidates (Borda consensus of TOPSIS + GRA + PROMETHEE II + VIKOR):**

| Rank | PCM | Family | Tm (C) | Latent heat (kJ/kg) | TOPSIS | GRA | PROMETHEE | VIKOR_Q | MC Top-3 % |
|---|---|---|---|---|---|---|---|---|---|
| 1 | RT57HC |  | 56.5 | 240 | 0.626 | 0.117 | +0.162 | 0.000 | 75.2% |
| 2 | n-Hexacosane (C26) |  | 56.5 | 256 | 0.602 | 0.114 | +0.163 | 0.018 | 53.2% |
| 3 | n-Pentacosane (C25) |  | 54.0 | 238 | 0.483 | 0.115 | +0.221 | 0.460 | 37.8% |

*Kendall's W (4-method concordance) = 0.477 (weak agreement — this regime's PCM choice is genuinely ambiguous)*

**Phase 7 — simulated annual performance (grey-box lumped-enthalpy tank, real climate data):**

| PCM | Consensus rank | Simulated solar fraction | In 54-84% benchmark band? | Complete cycles/yr |
|---|---|---|---|---|
| RT57HC | 1 | 44.4% | No | 73 |
| n-Hexacosane (C26) | 2 | 44.4% | No | 73 |
| n-Pentacosane (C25) | 3 | 53.3% | No | 105 |
| savE® OM55 | 4 | 42.7% | No | 90 |
| Palmitic-stearic acid/Expanded graphite | 5 | 42.8% | No | 88 |

*Spearman rho (MCDM rank vs. simulated solar fraction) for this cluster: 0.103 — weak agreement — diagnose before trusting the MCDM ranking here*

**Caveats:** thermal conductivity / density / specific heat not reported in the source data for the literature-added candidates (see 06_build_pcm_database.py); Phase 7's tank/collector parameters are stated assumptions, not measurements (see 10_physics_validation.py's docstring).


## Cluster 2

- **Points in regime:** 46
- **Population covered:** 30,335,761
- **Medoid point (highest membership confidence):** TNP_0001 (13.125, 80.125)

**Climate signature (population-weighted mean):**

| Index | Value |
|---|---|
| GHI_daily_kWh | 5.228 |
| Ta_mean | 29.706 |
| DTR_true | 8.201 |
| kt_daily_mean | 0.817 |
| cloudy_frac | 0.026 |
| CCI | 4.417 |
| HDD18 | 0.000 |
| CDD24 | 1476.884 |
| RH_sunrise_mean | 82.670 |
| HSI_sunrise | 25.126 |
| monsoon_index | 0.970 |

**Derived targets:** Tm_target = 67.0 C, L_required = 406 kJ/kg

**Candidates screened:** 8 survived Phase 5 feasibility filtering (melting window, absolute band, latent-heat floor, cycling, supercooling, corrosion veto, safety exclusion)

**Top-3 PCM candidates (Borda consensus of TOPSIS + GRA + PROMETHEE II + VIKOR):**

| Rank | PCM | Family | Tm (C) | Latent heat (kJ/kg) | TOPSIS | GRA | PROMETHEE | VIKOR_Q | MC Top-3 % |
|---|---|---|---|---|---|---|---|---|---|
| 1 | RT57HC |  | 56.5 | 240 | 0.623 | 0.114 | +0.192 | 0.000 | 84.2% |
| 2 | n-Pentacosane (C25) |  | 54.0 | 238 | 0.538 | 0.112 | +0.260 | 0.446 | 59.0% |
| 2 | n-Hexacosane (C26) |  | 56.5 | 256 | 0.604 | 0.110 | +0.199 | 0.013 | 71.9% |

*Kendall's W (4-method concordance) = 0.780 (moderate agreement — discuss the disagreement)*

**Phase 7 — simulated annual performance (grey-box lumped-enthalpy tank, real climate data):**

| PCM | Consensus rank | Simulated solar fraction | In 54-84% benchmark band? | Complete cycles/yr |
|---|---|---|---|---|
| RT57HC | 1 | 53.6% | No | 117 |
| n-Pentacosane (C25) | 2 | 52.0% | No | 166 |
| n-Hexacosane (C26) | 2 | 53.6% | No | 117 |
| PureTemp 58 | 4 | 48.3% | No | 98 |
| PlusICE A58 | 5 | 48.3% | No | 98 |

*Spearman rho (MCDM rank vs. simulated solar fraction) for this cluster: 0.024 — weak agreement — diagnose before trusting the MCDM ranking here*

**Caveats:** thermal conductivity / density / specific heat not reported in the source data for the literature-added candidates (see 06_build_pcm_database.py); Phase 7's tank/collector parameters are stated assumptions, not measurements (see 10_physics_validation.py's docstring).
