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

**Headline findings.** At 60-second resolution a simple moving average sits on the
efficient frontier — the learned models do not provision more cheaply at equal
reliability; the advantage of prediction is lead-time-dependent (present at 1–3
min, reversed at 5 min); forecast accuracy does not predict deployment cost; and
although the aggregate SLO is met, per-service tails are heavy (8 of 50 services
miss the target). See `outputs/tables.md` and `outputs/phase2_summary.md`.

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
2. Run the benchmark (≈1.5 min for the committed ~2-hour slice):
   ```bash
   python run_benchmark.py --config config.yaml      # Table I & II, pareto.png
   python run_phase2.py   --config config_phase2.yaml # lead-time sweep + CIs
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
run_phase2.py           robustness: lead-time × seed sweep, CIs, per-service tails
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
