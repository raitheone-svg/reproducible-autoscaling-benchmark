# Reproducible Autoscaling Benchmark

A **benchmark / reproducibility study** (not a new-algorithm proposal) that fairly
compares proactive vs. reactive microservice autoscaling on the **real Alibaba
cluster-trace-microservices-v2022** trace under one protocol: forecast
per-microservice CPU demand, provision to it under a scaling delay, and measure
the **SLO-violation vs. provisioning-cost** trade-off.

Companion code for the paper *"Does Prediction Pay Off? A Reproducible Benchmark
of Proactive vs. Reactive Microservice Autoscaling on Production Traces."*

**Methods compared:** Reactive HPA (delay-lagged persistence), Moving average,
Holt exponential smoothing, Gradient-boosted trees, LSTM (optional, needs PyTorch).

**Headline findings.**
- **The aggregate SLO hides per-service failure.** Although every method meets the
  5% aggregate under-provisioning target, 8 of 50 services still miss it, with
  worst-service violation of 27–41% (`phase2_perservice.png`, `phase2_summary.md`).
  This aggregate-vs-per-service reliability gap is the most robust result here and
  is independent of the small cost differences below.
- At 60-second resolution a simple moving average sits on the efficient frontier:
  **the learned sequence models evaluated here (GBDT, LSTM) do not provision more
  cheaply** than the moving average at equal reliability.
- The advantage of prediction is lead-time-dependent (present at 1–3 min, reversed
  at 5 min), and forecast accuracy does not predict deployment cost.

> **Scope of the complexity claim.** This benchmark evaluates reactive, classical,
> and *sequence* learned models (GBDT, LSTM). It does **not** evaluate
> graph-/call-graph-based autoscalers (e.g. GNNs); the "complexity does not pay
> off" statement is therefore scoped to the models above, not to graph models,
> which are future work (see `BENCHMARK_SPEC.md`).

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # numpy pandas scikit-learn matplotlib pyyaml
pip install torch                   # optional: enables the LSTM method
```

## Self-test (no data needed, ~3 s)

```bash
python run_benchmark.py --config config_synth.yaml
```
Confirms the toolchain end-to-end and writes `outputs/`. **These numbers are
synthetic — not results.**

## Run on the real Alibaba trace

1. Obtain the per-container CPU metrics table from the
   [official Alibaba v2022 trace](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2022)
   under its terms (not redistributed here). Place the `MSMetrics*.csv` files
   under `./data/`.
   > The loader globs `*MSMetrics*.csv` / `*MSResource*.csv`. Keep **one** copy of
   > the table under `data/` — don't mix an extracted `.tar.gz` with the
   > `MSMetricsUpdate_*.csv`, or rows are double-counted.
2. Run the benchmark and the robustness study:
   ```bash
   python run_benchmark.py --config config.yaml         # Table I & II, pareto.png
   python run_phase2.py   --config config_phase2.yaml   # lead-time sweep, CIs,
                                                        # per-service gap, bootstrap
   ```

## Results (real trace, seed 7, 50 busiest services, H = 3 min)

**Table II — validation-frozen deployment cost (≤ 5% under-provisioning target):**

| Method | test capacity | headroom | under-prov % |
|---|---|---|---|
| Delayed persistence | 186.74 | 0.10 | 2.8 |
| Moving average | **181.41** | 0.05 | 2.8 |
| GBDT | 182.58 | 0.05 | 2.8 |
| Holt (ES) | 192.77 | 0.15 | 3.5 |
| LSTM | 175.82 | 0.05 | 6.9 ⚠ (breaches target) |

Full outputs are committed under `outputs/` as reference values.

## Robustness & significance (`run_phase2.py`)

Sweeps lead time (1/3/5 min) and seeds, and reports:
- **Window-level paired bootstrap** of each method's cost vs the baseline, with a
  two-sided p-value — the defensible significance test for the small (~3%) effects.
  It needs **several windows** to be meaningful; a 2-hour slice yields one window,
  so the CI/p are reported as `n/a` until you load more trace hours.
- Per-service tail reliability and the **aggregate-vs-per-service gap figure**.
- A perfect-forecast oracle (cost lower bound at the SLO).

Config knobs (`config_phase2.yaml` → `phase2:`): `seeds`, `lead_times_seconds`,
`n_windows` (auto-capped to what the loaded span supports), `separated_windows`
(non-adjacent windows across busy/quiet regimes). **To get a significance
statement on the cost effects, load ≥ ~12 h of the trace and set `n_windows: 8`.**

## Reproducibility

The deterministic methods (Delayed persistence, Moving average, Holt) and the data
aggregation reproduce **bit-for-bit** on any NumPy/pandas — independently verified
in a clean environment (186.74 / 181.41 / 192.77). GBDT and LSTM reproduce exactly
only with matched library versions; `outputs/manifest.json` stamps the seed, git
commit, and package versions on every run. Step-by-step instructions are in
**[`REPLICATE.md`](REPLICATE.md)**.

## Repository layout

```
run_benchmark.py        driver (Table I & II)
run_phase2.py           robustness: lead-time × seed × window sweep, CIs,
                        paired bootstrap, per-service gap, oracle
config*.yaml            real / self-test / phase-2 configs
src/io_demand.py        load per-microservice demand from MSMetrics (chunked, leakage-safe cohort)
src/features.py         lead-correct forecasting features
src/split.py            leakage-safe chronological split with purge gap
src/methods.py          reactive / MA / Holt / GBDT / LSTM
src/metrics.py          accuracy + SLO/cost metrics, validation-frozen headroom
src/report.py           tables, figures, results.json, manifest.json
outputs/                committed reference results (figures, tables, manifest)
BENCHMARK_SPEC.md       full scope, protocol, and planned extensions
REPLICATE.md            exact local-replication steps
```

## Data & citation

Uses the Alibaba microservice trace; please cite the dataset papers and respect
the trace's usage terms (the data itself is not redistributed here):

- S. Luo et al., "Characterizing Microservice Dependency and Performance: Alibaba
  Trace Analysis," ACM SoCC, 2021.
- S. Luo et al., "The Power of Prediction: Microservice Auto Scaling via Workload
  Learning," ACM SoCC, 2022.

To cite this benchmark, see `CITATION.cff` (GitHub's "Cite this repository").

## License

MIT — see `LICENSE`.
