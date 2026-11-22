# Replicate & verify — local macOS steps

Verified reproduction of the autoscaling benchmark on the real Alibaba
microservice v2022 trace. These are the exact steps to reproduce from a clean
environment on your Mac.

## 0. What you need

- Python 3.11+ (your committed run used 3.14; any recent 3.x works).
- The repo (this folder) and the trace already in `data/`:
  `MSMetricsUpdate_0.csv … MSMetricsUpdate_3.csv` (the ~2-hour slice, ~4.5 GB).
- ~2 GB RAM free; the real run takes ≈1.5 min after dependencies are installed.

> **Data gotcha.** The loader globs `*MSMetrics*.csv` (and `*MSResource*.csv`)
> recursively under `data/`. The committed results use the four
> `MSMetricsUpdate_*.csv` files. Do **not** also extract `MSMetrics_*.tar.gz`
> into `data/` — the glob would then read both sets and change the numbers.
> Keep exactly the four `MSMetricsUpdate_*.csv` in `data/`.

## 1. Create an isolated environment

```bash
cd ~/Downloads/autoscaling_benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt     # numpy pandas scikit-learn matplotlib pyyaml
pip install torch                   # optional: enables the LSTM method
```

## 2. Self-test on synthetic data (no trace needed, ~3 s)

```bash
python run_benchmark.py --config config_synth.yaml
```
Confirms the toolchain end-to-end and writes `outputs/`. **These numbers are
synthetic — not results.** (Verified: runs clean and writes
`results.json`, `tables.md`, `pareto.png`, `accuracy.png`.)

## 3. Reproduce the real-data results (Table I & II, ~1.5 min)

```bash
python run_benchmark.py --config config.yaml
```
This loads the real trace (50 busiest services, ~2 h, seed 7, H = 3 min),
sweeps the headroom, and writes `outputs/`.

**Acceptance check — `outputs/tables.md` Table II must read:**

| method | test capacity | headroom | under-prov % |
|---|---|---|---|
| Delayed persistence | 186.7444 | 0.10 | 2.8 |
| Moving average | 181.4124 | 0.05 | 2.8 |
| GBDT | 182.5842 | 0.05 | 2.8 |
| Holt (ES) | 192.7735 | 0.15 | 3.5 |
| LSTM | 175.8150 | 0.05 | 6.9 |

> **Independently verified in a clean environment:** the three deterministic
> methods (Delayed persistence, Moving average, Holt) reproduce the above values
> **exactly** (186.7444 / 181.4124 / 192.7735). GBDT and LSTM reproduce exactly
> only with the same library versions you used — see §5.

## 4. Reproduce the robustness / lead-time sweep (Phase 2 → paper Fig. 2)

```bash
python run_phase2.py --config config_phase2.yaml
```
Writes `outputs/phase2_results.json`, `phase2_summary.md`,
`phase2_leadtime.png`, `phase2_cost_ci.png` — the H = 1/3/5 min sweep over
seeds [7, 17, 23], including the oracle and the per-service tail columns.

**Acceptance check (from `phase2_summary.md`, H = 180 s):** LSTM 176.086 with
test SLO 0.069 ⚠ (frozen point breaches target); Moving average 181.412 (−2.9%);
GBDT 182.584 (−2.2%); Delayed persistence 186.744; Holt 192.774 (+3.2%);
Oracle 167.72 (−10.2%).

## 5. Exact-reproduction notes (reviewer item 7)

- Classical methods and the data aggregation are deterministic — they reproduce
  bit-for-bit on any NumPy/pandas.
- GBDT (scikit-learn) and LSTM (PyTorch) can drift across library versions. For a
  bit-exact artifact, pin the versions your committed run used and record them in
  `requirements.txt`:
  `numpy==2.2.*  pandas==2.3.*  scikit-learn==1.9.*  torch==2.15.*` (adjust to
  what `outputs/manifest.json` records on your machine).
- `outputs/manifest.json` already stamps seed, git commit, Python, platform, and
  package versions on every run — keep it with the results.

## 6. One-liner (after §1)

```bash
python run_benchmark.py --config config_synth.yaml && \
python run_benchmark.py --config config.yaml && \
python run_phase2.py --config config_phase2.yaml
```
