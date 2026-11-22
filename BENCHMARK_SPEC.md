# Project Spec — A Reproducible Benchmark of Proactive Autoscaling & Workload-Prediction Methods on Production Microservice Traces

This is a **benchmark / reproducibility study**, not a novel-algorithm proposal.
Its contribution is a fair, reproducible comparison of *existing* methods on a
public production trace, under one protocol, with an open artifact. Scope below
describes **what this repository implements today**; aspirational extensions are
listed separately under *Planned / future work*.

## One-line statement
We replay the public Alibaba microservice trace and compare representative
autoscaling / workload-prediction methods under a single protocol, reporting the
**SLO-violation vs. cost** trade-off and which methods are efficient at 60 s
resolution.

## The benchmark (implemented)
**Task:** per microservice, forecast near-future CPU demand, provision capacity
to the forecast under a **scaling delay**, and measure outcomes against a
reactive baseline — all under one protocol.

**Methods compared (all pre-existing; this repo implements and benchmarks them):**
- **Reactive (HPA-style)** — provisions to the last observed demand under a
  scaling delay of H steps (the industry-default baseline).
- **Moving average** — mean of the recent window.
- **Holt (double exponential smoothing)** — trend-aware extrapolation.
- **GBDT** — gradient-boosted regression on window features.
- **LSTM** — recurrent network over the demand window (optional; needs PyTorch).

**Metrics (implemented, on the test split):**
- Forecast accuracy: **MAE, RMSE, sMAPE**.
- Autoscaling outcome: **SLO-violation rate** (under-provision), **over-provisioning %**
  (waste), **scaling-action count**, a **cost proxy** (mean provisioned capacity).
- The **SLO–cost Pareto frontier**, swept over a provisioning-headroom grid.

**Protocol:** leakage-safe **chronological per-service split** with a purge gap of
(window + horizon) at each boundary; fixed seed; the same features and headroom
sweep for every method; results on a held-out test split.

## Findings (from the real-trace run in `outputs/`)
At 60 s resolution on the Alibaba trace: lightweight proactive forecasting
(moving average, GBDT, LSTM) provisions modestly cheaper than reactive HPA at a
fixed SLO, while a **simple moving average sits on the efficient frontier** — the
deep model does not buy cheaper provisioning, only marginally better worst-case
SLO. Practical takeaway: at this timescale, prefer a simple forecaster over a
heavyweight ML autoscaler. (Effect sizes are small; see Limitations.)

## Data
Alibaba **cluster-trace-microservices-v2022** (`MSMetrics` table), obtained from
the official source under its terms; **not redistributed here**. See the README
for download and citation.

## Limitations (stated honestly)
- The SLO signal is a **CPU-capacity proxy** (under-provision = capacity < demand),
  not measured latency.
- Results here use a limited slice (first ~2 h, 50 busiest services); effect
  sizes are small and should be firmed up with more hours/services.
- 60 s sampling: fast, sub-minute dynamics are not observed.

## Planned / future work (NOT yet in this repo)
- Additional baselines: **ARIMA**, **Prophet**.
- **Response-time SLOs** from the `MSRTMCR` table (real latency, not a CPU proxy).
- A **call-graph sub-study**: does `MSCallGraph` fan-out improve forecasting?
- Extra metrics: **surge recall & lead time**, per-characteristic breakdowns
  (call rate, burstiness, call-graph position).
- Scaling to more hours and services; a second trace (Google 2011, CC-BY).

## Related work (benchmarked / positioned against)
- S. Luo et al., "The Power of Prediction: Microservice Auto Scaling via Workload
  Learning" (Madu), ACM SoCC, 2022.
- S. Luo et al., "Characterizing Microservice Dependency and Performance: Alibaba
  Trace Analysis," ACM SoCC, 2021. *(dataset)*
- C. Meng et al., "DeepScaler: Holistic Autoscaling for Microservices Based on
  Spatiotemporal GNN with Adaptive Graph Learning," IEEE/ACM ASE, 2023.
- H. X. Nguyen et al., "Graph-PHPA: Graph-based Proactive Horizontal Pod
  Autoscaling for Microservices using LSTM-GNN," IEEE CloudNet, 2022.
- K. Rzadca et al., "Autopilot: Workload Autoscaling at Google Scale," EuroSys, 2020.

## Target venues (benchmark/repro-friendly)
ICPE (best fit), MASCOTS, IEEE CLOUD, CCGrid, or a workshop for a faster first
publication.
