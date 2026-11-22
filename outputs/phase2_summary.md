# Phase-2 results (revised, leakage-safe)

phase2 revised: leakage-safe per-window cohort; seeds aggregated within window; descriptive min-max band over windows; no significance test (n_windows small).
Target shortfall <= 0.05. Provenance: REAL:Alibaba-microservices-v2022-MSMetrics.
Windows: 5; per-window cohort turnover 0.651 (union 83, intersection 29); missing-minute rate 0.0.


## Lead time H = 60 s

| Method | Cost mean (min-max) | Test shortfall | Feasible windows | #svc>5% |
|---|---|---|---|---|
| Oracle (perfect) | 213.466 (153.576-311.566) | 0.0 | 5/5 | 0.0 |
| Delayed persistence | 230.072 (169.337-327.508) | 0.0233 | 5/5 | 7.0 |
| Holt (ES) | 232.391 (168.81-327.223) | 0.0258 | 5/5 | 7.4 |
| LSTM | 235.37 (176.49-328.561) | 0.0198 | 5/5 | 6.4 |
| GBDT | 239.331 (178.099-331.27) | 0.0236 | 5/5 | 6.0 |
| Moving average | 248.888 (186.825-343.945) | 0.0178 | 5/5 | 4.8 |

*Paired vs delayed-persistence (per-window, descriptive; no p-value):*
| Method | mean Δcost % | per-window Δ% | #windows more expensive |
|---|---|---|---|
| Holt (ES) | 0.79 | [-0.15, -0.31, -0.05, 4.55, -0.09] | 1/5 |
| LSTM | 2.42 | [-0.81, 4.22, 3.9, 4.49, 0.32] | 4/5 |
| GBDT | 4.4 | [4.42, 5.17, 6.46, 4.82, 1.15] | 5/5 |
| Moving average | 8.65 | [1.25, 10.33, 17.58, 9.06, 5.02] | 5/5 |

## Lead time H = 180 s

| Method | Cost mean (min-max) | Test shortfall | Feasible windows | #svc>5% |
|---|---|---|---|---|
| Oracle (perfect) | 213.466 (153.576-311.566) | 0.0 | 5/5 | 0.0 |
| Delayed persistence | 238.59 (177.749-327.722) | 0.0273 | 5/5 | 7.4 |
| LSTM | 240.191 (185.395-329.738) | 0.031 | 4/5 | 8.4667 |
| Holt (ES) | 244.513 (175.976-327.023) | 0.0364 | 4/5 | 10.6 |
| GBDT | 246.026 (200.671-336.136) | 0.0205 | 5/5 | 5.4 |
| Moving average | 247.343 (191.656-344.221) | 0.0195 | 5/5 | 4.2 |

*Paired vs delayed-persistence (per-window, descriptive; no p-value):*
| Method | mean Δcost % | per-window Δ% | #windows more expensive |
|---|---|---|---|
| LSTM | 0.72 | [-1.73, 4.3, -0.72, 1.13, 0.62] | 3/5 |
| Holt (ES) | 2.7 | [-0.62, -1.0, 15.29, 0.06, -0.21] | 2/5 |
| GBDT | 3.8 | [8.2, 14.13, -7.23, 1.32, 2.57] | 4/5 |
| Moving average | 4.07 | [1.02, 14.53, -0.13, -0.08, 5.03] | 3/5 |

## Lead time H = 300 s

| Method | Cost mean (min-max) | Test shortfall | Feasible windows | #svc>5% |
|---|---|---|---|---|
| Oracle (perfect) | 213.397 (153.457-311.341) | 0.0 | 5/5 | 0.0 |
| LSTM | 239.306 (186.148-330.054) | 0.0234 | 5/5 | 6.8 |
| Delayed persistence | 244.424 (186.004-327.949) | 0.0296 | 4/5 | 8.4 |
| GBDT | 246.052 (202.329-334.616) | 0.032 | 4/5 | 9.6 |
| Holt (ES) | 249.225 (188.617-342.712) | 0.0555 | 3/5 | 13.8 |
| Moving average | 249.526 (201.454-344.338) | 0.0165 | 5/5 | 4.4 |

*Paired vs delayed-persistence (per-window, descriptive; no p-value):*
| Method | mean Δcost % | per-window Δ% | #windows more expensive |
|---|---|---|---|
| LSTM | -2.03 | [1.72, 0.08, -13.92, 1.32, 0.64] | 4/5 |
| GBDT | 1.28 | [6.19, 9.81, -13.71, 2.07, 2.03] | 4/5 |
| Holt (ES) | 1.91 | [-1.01, 6.24, -0.24, 0.06, 4.5] | 3/5 |
| Moving average | 2.43 | [5.73, 9.99, -13.2, 4.64, 5.0] | 4/5 |

*Five windows cannot support a paired significance test (exact two-sided sign-test floor 2/32 = 0.0625); estimates are descriptive.*
