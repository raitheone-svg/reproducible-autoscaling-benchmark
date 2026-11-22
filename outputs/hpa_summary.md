# HPA baseline comparison

Provenance: REAL:Alibaba-microservices-v2022-MSMetrics. Target SLO $\le$ 0.05. per_replica_capacity = 9.8082 (mean demand / 20).

| Policy | Actuation | Cost | Under-prov % | Operating point |
|---|---|---|---|---|
| Delayed persistence | continuous allocation | 303.1605 | 3.0 | headroom 0.15 |
| HPA (reactive) | integer replicas, tolerance/stabilisation/rate-limited | 327.5736 | 0.6 | target_util 0.8, 36.04 actions/svc |

At comparable reliability the replica HPA provisions **+8.1%** versus the idealised continuous baseline -- the cost of integer replicas, the tolerance band, and downscale stabilisation that the continuous abstraction omits. `per_replica_capacity` is a documented modelling choice (one replica serves the mean busy-service demand divided by 20); the HPA cost scales with it, but the sign of the overhead does not.
