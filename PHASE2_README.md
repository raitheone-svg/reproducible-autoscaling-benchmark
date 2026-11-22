# Phase 2 — Robustness evidence

`run_phase2.py` adds the robustness the IEEE TAI reviewer asked for, **without
changing any Phase-1 result**. It reuses the exact Phase-1 modules
(`io_demand`, `features`, `split`, `methods`, `metrics`) and the same
validation-frozen protocol, then repeats it across more conditions.

## What it produces
- **Lead-time sweep** — H ∈ {1, 3, 5} min (configurable), so the "simple ≥ complex"
  finding is tested at several forecast horizons.
- **Multiple seeds + windows → 95% confidence intervals** on cost and SLO, so the
  differences between methods can be read as real or as noise.
- **Perfect-forecast oracle** — a lower bound on cost at the SLO (what cost would be
  if demand were known exactly). The gap between a method and the oracle is the
  headroom that *better* forecasting could still recover.
- **Per-service tail reliability** — at each method's frozen operating point, the
  distribution of per-service violation: worst service, p95, and how many services
  exceed the 5% target. (Aggregate SLO can look fine while a few services suffer.)

## Run it
Drop this file next to `run_benchmark.py` in the repo (it imports `src/…`). With
your real Alibaba data in `./data`:

```bash
python run_phase2.py --config config_phase2.yaml
```

`config_phase2.yaml` is `config.yaml` plus a `phase2:` block:

```yaml
phase2:
  seeds: [7, 17, 23]
  lead_times_seconds: [60, 180, 300]   # H = 1, 3, 5 min
  n_windows: 2                          # contiguous slices of the trace
```

Grid size = `n_windows × len(lead_times) × len(seeds)` runs (default 2×3×3 = 18).
Each run trains GBDT + LSTM, so expect a few minutes on the ~2 h slice.

## Outputs (in `outputs/`)
- `phase2_results.json` — full aggregates (mean, 95% CI, tail stats) per lead time.
- `phase2_summary.md` — the tables to paste into the paper.
- `phase2_cost_ci.png` — cost mean ± 95% CI per method (SLO-violating bars in red).
- `phase2_leadtime.png` — cost vs. lead time per method.

## Notes
- **Confidence intervals need variety to be meaningful.** Deterministic methods
  (delayed persistence, moving average, Holt) have zero run-to-run variance within a
  fixed window; their CI comes only from *windows*. To tighten and populate the CIs,
  load more hours of trace and raise `n_windows`. GBDT/LSTM also vary across `seeds`.
- The oracle and per-service tail are new diagnostics; they do not alter the
  Phase-1 Table I / Table II numbers.
- If `torch` is absent the LSTM is skipped (everything else still runs); install it
  to include the LSTM, as in Phase 1.
