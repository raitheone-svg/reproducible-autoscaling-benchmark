#!/usr/bin/env python3
"""Regenerate the manuscript's phase-2 numbers from the released run matrix.

Reads outputs/phase2_runmatrix.json (the disaggregated per-(window, method,
lead, seed) matrix) and produces, with seeds aggregated WITHIN each window first:

  * Table 5  -- descriptive mean Δcost% vs the reactive baseline per method and
                lead time, the per-window differences, and how many windows each
                method is more expensive in (no p-value: five windows cannot
                support a paired test, floor 2/32 = 0.0625).
  * Fig. 2   -- per method and lead, the mean and the [min, max] band of the five
                seed-averaged window costs.
  * Table 6  -- per window and method (at the middle lead), aggregate test
                shortfall, #services over target, worst-service and p95 rates,
                and the per-service test denominator.

Writes outputs/analyze_phase2_tables.json and prints a summary.
"""
import json, os
from collections import defaultdict
import numpy as np

OUT = "outputs"
M = json.load(open(os.path.join(OUT, "phase2_runmatrix.json")))
leads = sorted({r["lead_s"] for r in M})
methods = ["reactive", "moving_average", "holt", "gbdt", "lstm"]
DISP = {"reactive": "Delayed persistence", "moving_average": "Moving average",
        "holt": "Holt (ES)", "gbdt": "GBDT", "lstm": "LSTM"}


def window_mean_cost(lead, method):
    by_win = defaultdict(list)
    for r in M:
        if r["lead_s"] == lead and r["method"] == method:
            by_win[r["window"]].append(r["cost"])
    return {w: float(np.mean(v)) for w, v in by_win.items()}


out = {"table5": {}, "fig2_band": {}, "table6": {}, "notes":
       "seeds aggregated within window; descriptive; 5 windows, no significance test"}

# ---- Table 5 + Fig 2 band ----
for h in leads:
    base = window_mean_cost(h, "reactive")
    out["table5"][h] = {}
    out["fig2_band"][h] = {}
    for m in methods:
        wm = window_mean_cost(h, m)
        if not wm:
            continue
        vals = np.array([wm[w] for w in sorted(wm)])
        out["fig2_band"][h][m] = dict(mean=round(float(vals.mean()), 2),
                                      lo=round(float(vals.min()), 2),
                                      hi=round(float(vals.max()), 2))
        if m == "reactive":
            continue
        wins = sorted(set(wm) & set(base))
        deltas = [round((wm[w] - base[w]) / base[w] * 100, 2) for w in wins if base[w]]
        a = np.array(deltas) if deltas else np.array([0.0])
        out["table5"][h][m] = dict(mean_delta_pct=round(float(a.mean()), 2),
                                   per_window=deltas,
                                   n_more_expensive=int((a > 0).sum()),
                                   n_windows=len(deltas))

# ---- Table 6: per window x method at the middle lead ----
hmid = leads[len(leads) // 2]
t6 = defaultdict(dict)
agg_by = defaultdict(lambda: defaultdict(list))
for r in M:
    if r["lead_s"] != hmid:
        continue
    k = (r["window"], r["method"])
    agg_by[k]["slo"].append(r["test_slo"])
    agg_by[k]["over"].append(r["svc_over_target"])
    agg_by[k]["worst"].append(r["worst_svc"])
    agg_by[k]["p95"].append(r["p95_svc"])
    agg_by[k]["denom"].append(r["denom_median"])
    agg_by[k]["ntot"].append(r["svc_total"])
for (w, m), d in agg_by.items():
    t6[m][w] = dict(agg_shortfall=round(float(np.mean(d["slo"])), 4),
                    svc_over=round(float(np.mean(d["over"])), 1),
                    svc_total=int(np.median(d["ntot"])),
                    worst=round(float(np.mean(d["worst"])), 3),
                    p95=round(float(np.mean(d["p95"])), 3),
                    denom=int(np.median(d["denom"])),
                    agg_pass=bool(np.mean(d["slo"]) <= 0.05))
out["table6"] = {m: {int(w): v for w, v in wd.items()} for m, wd in t6.items()}

json.dump(out, open(os.path.join(OUT, "analyze_phase2_tables.json"), "w"), indent=2)

# ---- print ----
print("=== Table 5 (descriptive Δcost% vs reactive; seeds within window) ===")
for h in leads:
    print(f"H={h}s: " + ", ".join(
        f"{DISP[m]} {out['table5'][h][m]['mean_delta_pct']:+.1f}%"
        f"(more in {out['table5'][h][m]['n_more_expensive']}/{out['table5'][h][m]['n_windows']})"
        for m in methods if m != "reactive" and m in out["table5"].get(h, {})))
print(f"\n=== Table 6 (per-window service breach, H={hmid}s) ===")
for m in methods:
    if m not in out["table6"]:
        continue
    rows = out["table6"][m]
    over = [rows[w]["svc_over"] for w in sorted(rows)]
    passes = sum(rows[w]["agg_pass"] for w in sorted(rows))
    print(f"  {DISP[m]:<20} agg_pass {passes}/{len(rows)} windows; "
          f"svc>5% per window {over}; worst "
          f"{max(rows[w]['worst'] for w in rows):.2f}")
print("\nWrote outputs/analyze_phase2_tables.json")
