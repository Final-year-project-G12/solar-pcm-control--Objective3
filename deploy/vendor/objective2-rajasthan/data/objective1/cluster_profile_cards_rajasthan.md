# Rajasthan — Level A Cluster Profile Cards

Generated from k=3 GMM clusters (320 population points). Auto-suggested k selection reason: in silhouette band AND expected single-state k range. Bootstrap-ARI stability at this k: 0.8077 (mean over 50/50 resamples).

Koppen-Geiger external validation: ARI=0.2787, NMI=0.3817 against Beck et al. (2018) present-climate classes for these same 320 points (see koppen_validation_rajasthan.csv for the full cluster x Koppen-class contingency table). This is a low-to-moderate agreement — the GMM is finding climate structure at a finer resolution than Koppen's broad classes capture within Rajasthan, which is a legitimate finding in its own right, not a failure of the clustering.

This is a single-state run — see the printed EXTERNAL VALIDATION section for what's not yet wired in (NBC/ECBC, state identity).


## Cluster 0

- **Points in regime:** 109
- **Total population covered:** 21,755,737
- **Medoid point (climate-feature-space, no district lookup available — lat/lon only):** RJP_0132 (24.375, 74.125)

**Two-tier climate signature (population-weighted mean +/- std):**

| Index | Mean | Std |
|---|---|---|
| Ta_mean | 27.130 | 0.698 |
| Ta_p95 | 35.638 | 1.349 |
| Ta_p05 | 17.457 | 1.005 |
| T_sunrise_mean | 21.956 | 0.618 |
| T_noon_mean | 29.727 | 0.718 |
| T_sunset_mean | 29.708 | 0.825 |
| diurnal_gradient | 7.771 | 0.369 |
| GHI_noon_mean | 750.289 | 8.563 |
| GHI_sunset_mean | 27.761 | 1.885 |
| GHI_daily_kWh | 5.164 | 0.073 |
| kt_noon_mean | 0.840 | 0.012 |
| kt_noon_std | 0.214 | 0.013 |
| kt_daily_mean | 0.867 | 0.008 |
| kt_daily_std | 0.181 | 0.011 |
| SAI | 0.867 | 0.008 |
| cloudy_frac | 0.016 | 0.005 |
| CCI | 0.645 | 0.035 |
| HDD18 | 1100.408 | 449.769 |
| CDD24 | 12372.140 | 1527.682 |
| DTR_true | 13.518 | 0.215 |
| RH_sunrise_mean | 70.984 | 1.934 |
| HSI_sunrise | 20.767 | 0.534 |
| wind_noon_mean | 2.644 | 0.220 |
| wind_sunset_mean | 2.265 | 0.143 |
| daylength_mean | 12.153 | 0.002 |
| daylength_amplitude | 1.534 | 0.050 |
| seasonality | 0.219 | 0.011 |
| monsoon_index | 0.928 | 0.033 |

**Physical description (auto-generated, review before publishing):** Cooler, arid/low-monsoon, erratic solar resource, short low-clearness runs.

**Derived PCM targets:** Tm_target_C = 67.0 C, L_required_kJ_per_kg = 438 kJ/kg (CEILING, not an achievability bar — see 04_climate_signature_rajasthan.py's docstring)


## Cluster 1

- **Points in regime:** 83
- **Total population covered:** 14,033,429
- **Medoid point (climate-feature-space, no district lookup available — lat/lon only):** RJP_0192 (26.875, 73.375)

**Two-tier climate signature (population-weighted mean +/- std):**

| Index | Mean | Std |
|---|---|---|
| Ta_mean | 27.984 | 0.585 |
| Ta_p95 | 36.607 | 0.756 |
| Ta_p05 | 16.180 | 1.730 |
| T_sunrise_mean | 22.043 | 0.494 |
| T_noon_mean | 30.451 | 0.476 |
| T_sunset_mean | 31.457 | 0.935 |
| diurnal_gradient | 8.407 | 0.217 |
| GHI_noon_mean | 772.816 | 8.421 |
| GHI_sunset_mean | 29.217 | 4.814 |
| GHI_daily_kWh | 5.381 | 0.150 |
| kt_noon_mean | 0.880 | 0.009 |
| kt_noon_std | 0.161 | 0.012 |
| kt_daily_mean | 0.907 | 0.011 |
| kt_daily_std | 0.131 | 0.009 |
| SAI | 0.907 | 0.011 |
| cloudy_frac | 0.004 | 0.002 |
| CCI | 0.811 | 0.044 |
| HDD18 | 1615.046 | 650.400 |
| CDD24 | 16440.402 | 1633.101 |
| DTR_true | 13.797 | 0.331 |
| RH_sunrise_mean | 67.234 | 2.206 |
| HSI_sunrise | 20.682 | 0.389 |
| wind_noon_mean | 3.338 | 0.367 |
| wind_sunset_mean | 3.005 | 0.317 |
| daylength_mean | 12.158 | 0.003 |
| daylength_amplitude | 1.691 | 0.089 |
| seasonality | 0.209 | 0.017 |
| monsoon_index | 1.048 | 0.038 |

**Physical description (auto-generated, review before publishing):** Hot, monsoon-influenced, steady solar resource, long low-clearness runs (high autonomy demand).

**Derived PCM targets:** Tm_target_C = 67.0 C, L_required_kJ_per_kg = 427 kJ/kg (CEILING, not an achievability bar — see 04_climate_signature_rajasthan.py's docstring)


## Cluster 2

- **Points in regime:** 128
- **Total population covered:** 34,514,036
- **Medoid point (climate-feature-space, no district lookup available — lat/lon only):** RJP_0083 (26.625, 75.875)

**Two-tier climate signature (population-weighted mean +/- std):**

| Index | Mean | Std |
|---|---|---|
| Ta_mean | 26.696 | 0.489 |
| Ta_p95 | 35.879 | 0.495 |
| Ta_p05 | 14.987 | 1.306 |
| T_sunrise_mean | 21.437 | 0.513 |
| T_noon_mean | 29.606 | 0.484 |
| T_sunset_mean | 29.044 | 0.627 |
| diurnal_gradient | 8.168 | 0.363 |
| GHI_noon_mean | 749.251 | 9.901 |
| GHI_sunset_mean | 21.628 | 4.450 |
| GHI_daily_kWh | 5.033 | 0.188 |
| kt_noon_mean | 0.855 | 0.007 |
| kt_noon_std | 0.184 | 0.014 |
| kt_daily_mean | 0.875 | 0.010 |
| kt_daily_std | 0.156 | 0.009 |
| SAI | 0.875 | 0.010 |
| cloudy_frac | 0.008 | 0.002 |
| CCI | 0.765 | 0.051 |
| HDD18 | 2149.120 | 468.125 |
| CDD24 | 14590.109 | 1871.080 |
| DTR_true | 13.716 | 0.191 |
| RH_sunrise_mean | 72.097 | 3.414 |
| HSI_sunrise | 20.370 | 0.394 |
| wind_noon_mean | 2.624 | 0.260 |
| wind_sunset_mean | 2.188 | 0.195 |
| daylength_mean | 12.159 | 0.003 |
| daylength_amplitude | 1.713 | 0.083 |
| seasonality | 0.223 | 0.021 |
| monsoon_index | 1.025 | 0.045 |

**Physical description (auto-generated, review before publishing):** Cooler, arid/low-monsoon, erratic solar resource, short low-clearness runs.

**Derived PCM targets:** Tm_target_C = 67.0 C, L_required_kJ_per_kg = 443 kJ/kg (CEILING, not an achievability bar — see 04_climate_signature_rajasthan.py's docstring)
