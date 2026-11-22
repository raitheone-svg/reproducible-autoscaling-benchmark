Provenance: REAL:Alibaba-microservices-v2022-MSMetrics
Protocol: validation-frozen headroom; test reported.

Table I - forecast accuracy (test split; computed before headroom)
method | MAE | RMSE | sMAPE%
Delayed persistence | 5.3748 | 10.6095 | 3.7
Moving average | 6.9551 | 10.209 | 4.8
Holt (ES) | 6.5682 | 12.9433 | 4.4
GBDT | 8.3872 | 13.0103 | 6.7
LSTM | 5.2171 | 9.7536 | 4.0

Table II - validation-frozen deployment result (target <= 5% under-provisioning)
method | test capacity | headroom | test under-prov % | vs delayed persistence
Delayed persistence | 186.7444 | 0.1 | 2.8 | +0.0%
Moving average | 181.4124 | 0.05 | 2.8 | +2.9%
Holt (ES) | 192.7735 | 0.15 | 3.5 | -3.2%
GBDT | 182.5842 | 0.05 | 2.8 | +2.2%
LSTM | 175.815 | 0.05 | 6.9 | +5.9%
