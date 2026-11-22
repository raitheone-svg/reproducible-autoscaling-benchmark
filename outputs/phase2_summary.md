# Phase-2 results (robustness)

Protocol: phase2: validation-frozen; mean±95% CI over window×seed; paired bootstrap over windows. Target SLO ≤ 0.05. Provenance: REAL:Alibaba-microservices-v2022-MSMetrics.
Seeds [7, 17, 23]; lead times [60, 180, 300] s; 5 window(s) (separated=True).


## Lead time H = 60 s

| Method | Cost (mean±95%CI) | Test SLO | Worst-service viol. | #svc > target |
|---|---|---|---|---|
| Oracle (perfect) | 201.014±29.324 (-10.8%) | 0.0 | 0.0 | 0.0 |
| Holt (ES) | 223.166±33.081 (-0.9%) | 0.029 | 0.311 | 9.0 |
| Delayed persistence | 225.231±32.492 | 0.019 | 0.222 | 7.2 |
| LSTM | 227.305±31.438 (+0.9%) | 0.016 | 0.27 | 6.6 |
| GBDT | 231.841±30.789 (+2.9%) | 0.017 | 0.278 | 7.0 |
| Moving average | 242.634±28.768 (+7.7%) | 0.015 | 0.267 | 5.2 |

*Paired vs delayed-persistence baseline (window-level bootstrap):*
| Method | Δcost % | 95% CI | p | #windows |
|---|---|---|---|---|
| Holt (ES) | -1.04 | [-2.88, -0.08] | 0.0 | 5 |
| LSTM | 1.22 | [-0.22, 2.96] | 0.1392 | 5 |
| GBDT | 3.49 | [0.48, 7.36] | 0.0 | 5 |
| Moving average | 9.18 | [1.51, 19.92] | 0.0 | 5 |

## Lead time H = 180 s

| Method | Cost (mean±95%CI) | Test SLO | Worst-service viol. | #svc > target |
|---|---|---|---|---|
| Oracle (perfect) | 201.014±29.324 (-14.8%) | 0.0 | 0.0 | 0.0 |
| Delayed persistence | 235.896±28.439 | 0.026 | 0.267 | 8.4 |
| Holt (ES) | 236.514±27.518 (+0.3%) | 0.033 | 0.378 | 8.8 |
| GBDT | 239.76±26.966 (+1.6%) | 0.019 | 0.367 | 6.2 |
| LSTM | 239.851±26.634 (+1.7%) | 0.014 | 0.248 | 5.4 |
| Moving average | 241.478±25.146 (+2.4%) | 0.017 | 0.311 | 5.2 |

*Paired vs delayed-persistence baseline (window-level bootstrap):*
| Method | Δcost % | 95% CI | p | #windows |
|---|---|---|---|---|
| Holt (ES) | 0.45 | [-0.45, 1.99] | 0.6588 | 5 |
| GBDT | 1.97 | [-0.87, 4.82] | 0.1624 | 5 |
| LSTM | 2.11 | [0.12, 4.63] | 0.0332 | 5 |
| Moving average | 3.16 | [0.13, 8.76] | 0.0072 | 5 |

## Lead time H = 300 s

| Method | Cost (mean±95%CI) | Test SLO | Worst-service viol. | #svc > target |
|---|---|---|---|---|
| Oracle (perfect) | 200.938±29.304 (-16.5%) | 0.0 | 0.0 | 0.0 |
| Delayed persistence | 240.572±25.331 | 0.021 | 0.318 | 6.8 |
| LSTM | 241.358±25.318 (+0.3%) | 0.019 | 0.314 | 6.067 |
| Moving average | 241.873±25.004 (+0.5%) | 0.02 | 0.376 | 5.8 |
| Holt (ES) | 248.408±20.62 (+3.3%) | 0.036 | 0.4 | 8.4 |
| GBDT | 248.502±26.338 (+3.3%) | 0.014 | 0.294 | 4.0 |

*Paired vs delayed-persistence baseline (window-level bootstrap):*
| Method | Δcost % | 95% CI | p | #windows |
|---|---|---|---|---|
| LSTM | 0.36 | [-0.23, 1.13] | 0.3592 | 5 |
| Moving average | 0.61 | [0.24, 1.04] | 0.0 | 5 |
| GBDT | 3.3 | [0.83, 5.76] | 0.0004 | 5 |
| Holt (ES) | 4.08 | [-0.34, 9.58] | 0.1556 | 5 |

*Oracle = perfect-forecast lower bound on cost at the SLO. ⚠ marks a frozen operating point whose mean test SLO exceeds the target.*
