# Phase-2 results (robustness)

Protocol: phase2: validation-frozen; mean±95% CI over window×seed. Target SLO ≤ 0.05. Provenance: REAL:Alibaba-microservices-v2022-MSMetrics.
Seeds [7, 17, 23]; lead times [60, 180, 300] s; 1 window(s).


## Lead time H = 60 s

| Method | Cost (mean±95%CI) | Test SLO | Worst-service viol. | #svc > target |
|---|---|---|---|---|
| Oracle (perfect) | 167.72±0.0 (-9.5%) | 0.0 | 0.0 | 0.0 |
| GBDT | 178.553±0.0 (-3.6%) | 0.033 | 0.182 | 9.0 |
| Moving average | 179.98±0.0 (-2.8%) | 0.03 | 0.273 | 9.0 |
| LSTM | 181.301±10.283 (-2.1%) | 0.025 | 0.227 | 7.333 |
| Holt (ES) | 184.483±0.0 (-0.4%) | 0.026 | 0.273 | 7.0 |
| Delayed persistence | 185.225±0.0 | 0.017 | 0.182 | 5.0 |

## Lead time H = 180 s

| Method | Cost (mean±95%CI) | Test SLO | Worst-service viol. | #svc > target |
|---|---|---|---|---|
| Oracle (perfect) | 167.72±0.0 (-10.2%) | 0.0 | 0.0 | 0.0 |
| LSTM | 176.086±0.715 (-5.7%) | 0.069 ⚠ | 0.364 | 20.333 |
| Moving average | 181.412±0.0 (-2.9%) | 0.028 | 0.273 | 8.0 |
| GBDT | 182.584±0.0 (-2.2%) | 0.028 | 0.273 | 7.0 |
| Delayed persistence | 186.744±0.0 | 0.028 | 0.318 | 8.0 |
| Holt (ES) | 192.774±0.0 (+3.2%) | 0.035 | 0.409 | 8.0 |

## Lead time H = 300 s

| Method | Cost (mean±95%CI) | Test SLO | Worst-service viol. | #svc > target |
|---|---|---|---|---|
| Oracle (perfect) | 167.429±0.0 (-6.5%) | 0.0 | 0.0 | 0.0 |
| Delayed persistence | 179.067±0.0 | 0.032 | 0.381 | 9.0 |
| Moving average | 182.441±0.0 (+1.9%) | 0.025 | 0.238 | 8.0 |
| LSTM | 183.89±0.672 (+2.7%) | 0.017 | 0.238 | 6.0 |
| GBDT | 188.475±0.0 (+5.3%) | 0.011 | 0.191 | 3.0 |
| Holt (ES) | 200.274±0.0 (+11.8%) | 0.024 | 0.381 | 6.0 |

*Oracle = perfect-forecast lower bound on cost at the SLO. ⚠ marks a frozen operating point whose mean test SLO exceeds the target.*
