#!/usr/bin/env python3
"""Phase-2 benchmark: robustness evidence for the peer review.

Extends the Phase-1 validation-frozen pipeline along the axes the reviewer asked
for, without changing any Phase-1 result:

  * lead-time sweep        H in {1, 3, 5} min (configurable)
  * multiple seeds         run-to-run variability of the learned models
  * multiple windows       slices of the trace -> data variability
  * 95% confidence intervals on cost and SLO (mean over window x seed)
  * window-level PAIRED BOOTSTRAP of (method - baseline) cost, with a two-sided
    bootstrap p-value -- the defensible significance test for the small effects
    (needs several windows to be meaningful; load more trace hours to raise it)
  * perfect-forecast ORACLE: a lower bound on cost at the SLO (true demand known)
  * per-service tail reliability: the distribution of per-service violation
    at each method's frozen operating point (worst service, p95, #services > target),
    plus a figure elevating the aggregate-vs-per-service reliability GAP

Usage:
  python run_phase2.py --config config_phase2.yaml

Reads an optional `phase2:` block in the config:
  phase2:
    seeds: [7, 17, 23]
    lead_times_seconds: [60, 180, 300]
    n_windows: 8            # capped to what the loaded trace span supports
    separated_windows: true # non-adjacent windows across regimes (vs contiguous)
Defaults are used if the block is absent. Writes to <run.out_dir>:
  phase2_results.json, phase2_summary.md, phase2_cost_ci.png, phase2_leadtime.png,
  phase2_perservice.png
"""
import argparse, json, os, copy, math
import numpy as np
import yaml
from src import io_demand, features, split, methods, metrics, report

DISP = {"reactive": "Delayed persistence", "moving_average": "Moving average",
        "holt": "Holt (ES)", "gbdt": "GBDT", "lstm": "LSTM", "oracle": "Oracle (perfect)"}
SLO_TARGET = getattr(report, "SLO_TARGET", 0.05)


def _run_methods(b, tr, which, seed):
    """Like methods.run but threads `seed` into the LSTM and adds the oracle."""
    out = {}
    for name in which:
        if name in methods.REGISTRY:
            out[name] = methods.REGISTRY[name](b)
        elif name == "gbdt":
            out[name] = methods.gbdt(b, tr)
        elif name == "lstm":
            try:
                out[name] = methods.lstm(b, tr, seed=seed)
            except Exception as e:
                print(f"        [lstm skipped: {type(e).__name__}]")
    # perfect-forecast oracle: forecast == true demand -> cost lower bound at SLO
    out["oracle"] = b["y"].astype(float)
    return out


def _per_service_tail(b, yhat_all, te, h, cfg):
    """Per-service SLO-violation rate at the frozen headroom h, on the test split."""
    y = b["y"][te]; svc = b["svc"][te]; yhat = yhat_all[te] * (1.0 + h)
    viol = (yhat < y).astype(float)
    rates = {}
    for s in np.unique(svc):
        m = svc == s
        if m.sum():
            rates[s] = float(viol[m].mean())
    r = np.array(list(rates.values())) if rates else np.array([0.0])
    return dict(median=round(float(np.median(r)), 4),
                p90=round(float(np.quantile(r, 0.90)), 4),
                p95=round(float(np.quantile(r, 0.95)), 4),
                worst=round(float(r.max()), 4),
                n_services=int(len(r)),
                n_over_target=int((r > SLO_TARGET).sum()),
                rates=[round(float(x), 4) for x in r])   # raw vector for the gap figure


def _one_run(df, dt, cfg, h_seconds, seed, tmin, tmax):
    """Build -> split -> evaluate a single (window, lead-time, seed) configuration."""
    c = copy.deepcopy(cfg)
    c["features"]["horizon_seconds"] = h_seconds
    np.random.seed(seed)
    b = features.build(df, dt, c)
    if len(b["y"]) < 50:
        return None
    tr, va, te = split.blocked_split(b["meta"], c)
    if not (tr.sum() and va.sum() and te.sum()):
        return None
    which = [m for m in c["methods"]]
    fc = _run_methods(b, tr, which, seed)
    res = metrics.evaluate(b, fc, va, te, c, slo_target=SLO_TARGET)
    rec = {}
    for name, mm in res.items():
        rec[name] = dict(cost=mm["test_cost"], slo=mm["test_slo"],
                         headroom=mm["frozen_headroom"], smape=mm["accuracy"]["smape"],
                         tail=_per_service_tail(b, fc[name], te, mm["frozen_headroom"], c))
    return rec


def _ci(vals):
    """Mean and 95% half-width (t-interval; falls back to normal)."""
    a = np.asarray([v for v in vals if v is not None], float)
    n = len(a)
    if n == 0:
        return (float("nan"), float("nan"), 0)
    mean = float(a.mean())
    if n == 1:
        return (round(mean, 3), 0.0, 1)
    sd = float(a.std(ddof=1))
    se = sd / math.sqrt(n)
    try:
        from scipy import stats
        t = float(stats.t.ppf(0.975, n - 1))
    except Exception:
        t = 1.96
    return (round(mean, 3), round(t * se, 3), n)


def _paired_bootstrap(method_by_win, base_by_win, B=5000, seed=0):
    """Window-level paired bootstrap of (method - baseline) cost, in %.
    Inputs map window_idx -> mean cost over seeds for that window.
    Returns dict(delta_pct, ci_lo, ci_hi, p_value, n_windows). With one window
    the interval is undefined (n too small) and p is NaN -- load more trace hours
    for a real test. This is the significance statement the reviewer asked for."""
    wins = sorted(set(method_by_win) & set(base_by_win))
    deltas = np.array([(method_by_win[w] - base_by_win[w]) / base_by_win[w] * 100.0
                       for w in wins if base_by_win[w]])
    k = len(deltas)
    if k == 0:
        return dict(delta_pct=float("nan"), ci_lo=float("nan"),
                    ci_hi=float("nan"), p_value=float("nan"), n_windows=0)
    if k == 1:
        return dict(delta_pct=round(float(deltas[0]), 2), ci_lo=float("nan"),
                    ci_hi=float("nan"), p_value=float("nan"), n_windows=1)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(deltas, k, replace=True).mean() for _ in range(B)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2.0 * min((boot >= 0).mean(), (boot <= 0).mean())
    return dict(delta_pct=round(float(deltas.mean()), 2), ci_lo=round(float(lo), 2),
                ci_hi=round(float(hi), 2), p_value=round(float(min(p, 1.0)), 4),
                n_windows=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    p2 = cfg.get("phase2", {})
    seeds = p2.get("seeds", [7, 17, 23])
    leads = p2.get("lead_times_seconds", [60, 180, 300])
    n_windows = int(p2.get("n_windows", 2))
    separated = bool(p2.get("separated_windows", False))
    out_dir = cfg["run"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"[phase2] loading demand (mode={cfg['data']['mode']}) ...")
    df, dt = io_demand.load(cfg)
    tmin, tmax = int(df["t_step"].min()), int(df["t_step"].max())
    span = tmax - tmin + 1
    W = max(1, int(cfg["features"]["window_seconds"] / dt))
    maxH = max(1, int(max(leads) / dt))
    min_win = max(100, (W + maxH) * 5)   # steps below this starve the chronological split
    n_eff = max(1, min(n_windows, span // min_win))
    if n_eff != n_windows:
        print(f"[phase2] note: trace span {span} steps supports only {n_eff} window(s) "
              f"of >={min_win} steps (asked {n_windows}). Load more trace hours to raise "
              f"this -- the paired bootstrap needs several windows to be meaningful.")
    n_windows = n_eff
    if separated and n_windows > 1:
        # non-adjacent windows of length min_win, spread across the span (busy/quiet regimes)
        starts = np.linspace(tmin, tmax + 1 - min_win, n_windows).astype(int)
        windows = [(int(s), int(s + min_win)) for s in starts]
    else:
        edges = np.linspace(tmin, tmax + 1, n_windows + 1).astype(int)
        windows = [(int(edges[i]), int(edges[i + 1])) for i in range(n_windows)]
    print(f"         services={df['service'].nunique()} rows={len(df)} dt={dt}s "
          f"t_step in [{tmin},{tmax}]  windows={windows}  (separated={separated})")
    print(f"[phase2] grid: {n_windows} window(s) x {len(leads)} lead-time(s) x "
          f"{len(seeds)} seed(s) = {n_windows*len(leads)*len(seeds)} runs")

    # runs[lead][method] -> pooled lists; cells[lead][method][win_idx] -> seed costs
    runs = {h: {} for h in leads}
    cells = {h: {} for h in leads}
    per_service = {h: {} for h in leads}
    total = n_windows * len(leads) * len(seeds); done = 0
    for w_idx, (w0, w1) in enumerate(windows):
        dfw = df[(df["t_step"] >= w0) & (df["t_step"] < w1)]
        for h_sec in leads:
            for seed in seeds:
                done += 1
                rec = _one_run(dfw, dt, cfg, h_sec, seed, w0, w1)
                tag = f"win{w_idx}[{w0},{w1}) H={h_sec}s seed={seed}"
                if rec is None:
                    print(f"  ({done}/{total}) {tag}: skipped (too little data)")
                    continue
                for name, r in rec.items():
                    runs[h_sec].setdefault(name, {"cost": [], "slo": []})
                    runs[h_sec][name]["cost"].append(r["cost"])
                    runs[h_sec][name]["slo"].append(r["slo"])
                    cells[h_sec].setdefault(name, {}).setdefault(w_idx, []).append(r["cost"])
                    per_service[h_sec].setdefault(name, []).append(r["tail"])
                print(f"  ({done}/{total}) {tag}: ok")

    # ---- aggregate ----
    agg = {"protocol": "phase2: validation-frozen; mean±95% CI over window×seed; "
                        "paired bootstrap over windows",
           "slo_target": SLO_TARGET, "seeds": seeds, "lead_times_seconds": leads,
           "n_windows": n_windows, "separated_windows": separated,
           "provenance": cfg["data"]["provenance"],
           "lead_times": {}, "paired_vs_baseline": {}}
    for h_sec in leads:
        block = {}
        for name, d in runs[h_sec].items():
            cost_m, cost_ci, n = _ci(d["cost"])
            slo_m, slo_ci, _ = _ci(d["slo"])
            tails = per_service[h_sec].get(name, [])
            worst = _ci([t["worst"] for t in tails])[0] if tails else float("nan")
            p95 = _ci([t["p95"] for t in tails])[0] if tails else float("nan")
            p90 = _ci([t["p90"] for t in tails])[0] if tails and "p90" in tails[0] else float("nan")
            med = _ci([t["median"] for t in tails])[0] if tails else float("nan")
            over = _ci([t["n_over_target"] for t in tails])[0] if tails else float("nan")
            nsvc = tails[0]["n_services"] if tails else 0
            block[name] = dict(display=DISP.get(name, name), n_runs=n,
                               cost_mean=cost_m, cost_ci95=cost_ci,
                               slo_mean=round(slo_m, 4), slo_ci95=round(slo_ci, 4),
                               tail_median_mean=med, tail_p90_mean=p90,
                               tail_worst_mean=worst, tail_p95_mean=p95,
                               tail_services_over_target_mean=over, n_services=nsvc)
        agg["lead_times"][h_sec] = block
        # window-level paired bootstrap vs the reactive baseline
        base_by_win = {w: float(np.mean(c)) for w, c in cells[h_sec].get("reactive", {}).items()}
        pb = {}
        for name, wd in cells[h_sec].items():
            if name in ("reactive", "oracle"):
                continue
            method_by_win = {w: float(np.mean(c)) for w, c in wd.items()}
            pb[name] = _paired_bootstrap(method_by_win, base_by_win)
        agg["paired_vs_baseline"][h_sec] = pb

    with open(os.path.join(out_dir, "phase2_results.json"), "w") as f:
        json.dump(agg, f, indent=2)
    _write_summary(out_dir, agg)
    _plots(out_dir, agg, leads)
    _print(agg, leads)
    print(f"\nWrote {out_dir}/phase2_results.json, phase2_summary.md, "
          f"phase2_cost_ci.png, phase2_leadtime.png, phase2_perservice.png")


def _print(agg, leads):
    print("\nPhase-2 validation-frozen cost (mean ± 95% CI) by lead time:")
    for h in leads:
        print(f"  H = {h}s:")
        blk = agg["lead_times"][h]
        base = blk.get("reactive", {}).get("cost_mean")
        pb = agg["paired_vs_baseline"].get(h, {})
        for name, m in sorted(blk.items(), key=lambda kv: kv[1]["cost_mean"]):
            vs = ""
            if base and name != "reactive":
                vs = f"  ({(m['cost_mean']-base)/base*100:+.1f}% vs base)"
            sig = ""
            if name in pb and pb[name]["n_windows"] >= 2:
                sig = f"  [boot {pb[name]['delta_pct']:+.1f}% CI({pb[name]['ci_lo']},{pb[name]['ci_hi']}) p={pb[name]['p_value']}]"
            flag = "  <SLO VIOLATED>" if m["slo_mean"] > agg["slo_target"] else ""
            print(f"    {m['display']:<20} cost={m['cost_mean']:>8} ±{m['cost_ci95']:<6} "
                  f"slo={m['slo_mean']:.3f}{flag}{vs}{sig}")


def _write_summary(out_dir, agg):
    L = ["# Phase-2 results (robustness)\n",
         f"Protocol: {agg['protocol']}. Target SLO ≤ {agg['slo_target']}. "
         f"Provenance: {agg['provenance']}.",
         f"Seeds {agg['seeds']}; lead times {agg['lead_times_seconds']} s; "
         f"{agg['n_windows']} window(s) (separated={agg.get('separated_windows', False)}).\n"]
    for h, blk in agg["lead_times"].items():
        L.append(f"\n## Lead time H = {h} s\n")
        L.append("| Method | Cost (mean±95%CI) | Test SLO | Worst-service viol. | #svc > target |")
        L.append("|---|---|---|---|---|")
        base = blk.get("reactive", {}).get("cost_mean")
        for name, m in sorted(blk.items(), key=lambda kv: kv[1]["cost_mean"]):
            vs = "" if (not base or name == "reactive") else \
                f" ({(m['cost_mean']-base)/base*100:+.1f}%)"
            viol = " ⚠" if m["slo_mean"] > agg["slo_target"] else ""
            L.append(f"| {m['display']} | {m['cost_mean']}±{m['cost_ci95']}{vs} | "
                     f"{m['slo_mean']}{viol} | {m['tail_worst_mean']} | "
                     f"{m['tail_services_over_target_mean']} |")
        # paired bootstrap significance vs baseline
        pb = agg["paired_vs_baseline"].get(h, {})
        if pb:
            L.append("\n*Paired vs delayed-persistence baseline (window-level bootstrap):*")
            L.append("| Method | Δcost % | 95% CI | p | #windows |")
            L.append("|---|---|---|---|---|")
            for name, d in sorted(pb.items(), key=lambda kv: kv[1]["delta_pct"]):
                ci = "n/a" if d["n_windows"] < 2 else f"[{d['ci_lo']}, {d['ci_hi']}]"
                p = "n/a" if d["n_windows"] < 2 else f"{d['p_value']}"
                L.append(f"| {DISP.get(name, name)} | {d['delta_pct']} | {ci} | {p} | {d['n_windows']} |")
    if agg["n_windows"] < 2:
        L.append("\n> **Note:** only one window fits the loaded trace span, so the paired "
                 "bootstrap CI/p are undefined. Load more trace hours and set a higher "
                 "`phase2.n_windows` to obtain a significance statement on these effects.")
    L.append("\n*Oracle = perfect-forecast lower bound on cost at the SLO. "
             "⚠ marks a frozen operating point whose mean test SLO exceeds the target.*\n")
    open(os.path.join(out_dir, "phase2_summary.md"), "w").write("\n".join(L))


def _plots(out_dir, agg, leads):
    try:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["font.family"] = "DejaVu Serif"
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [plots skipped: {e}]")
        return
    hmid = leads[len(leads) // 2]
    blk = agg["lead_times"][hmid]
    items = [kv for kv in sorted(blk.items(), key=lambda kv: kv[1]["cost_mean"])
             if np.isfinite(kv[1]["cost_mean"])]
    if not items:
        print("  [plots skipped: no successful runs to plot — see the note above]")
        return
    names = [m["display"] for _, m in items]
    means = [m["cost_mean"] for _, m in items]
    cis = [m["cost_ci95"] for _, m in items]
    viol = [m["slo_mean"] > agg["slo_target"] for _, m in items]
    fig, ax = plt.subplots(figsize=(7, 4))
    cols = ["#b23b3b" if v else "#3b6db2" for v in viol]
    ax.bar(range(len(names)), means, yerr=cis, capsize=4, color=cols)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Provisioning cost (mean ± 95% CI)")
    ax.set_title(f"Validation-frozen cost by method (H = {hmid} s)")
    ax.set_ylim(min(means) * 0.95, max(means) * 1.03)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "phase2_cost_ci.png"), dpi=140)
    plt.close(fig)
    # lead-time sweep
    fig, ax = plt.subplots(figsize=(7, 4))
    allm = sorted({n for h in leads for n in agg["lead_times"][h]})
    for name in allm:
        ys = [agg["lead_times"][h].get(name, {}).get("cost_mean", float("nan")) for h in leads]
        es = [agg["lead_times"][h].get(name, {}).get("cost_ci95", 0) for h in leads]
        ax.errorbar(leads, ys, yerr=es, marker="o", capsize=3, label=DISP.get(name, name))
    ax.set_xlabel("Lead time H (s)"); ax.set_ylabel("Provisioning cost (mean ± 95% CI)")
    ax.set_title("Lead-time sweep"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "phase2_leadtime.png"), dpi=140)
    plt.close(fig)
    # per-service reliability GAP: aggregate SLO vs worst-service violation
    gitems = [kv for kv in items if kv[0] != "oracle"]
    gnames = [m["display"] for _, m in gitems]
    agg_slo = [m["slo_mean"] * 100 for _, m in gitems]
    worst = [(m["tail_worst_mean"] or 0) * 100 for _, m in gitems]
    over = [m["tail_services_over_target_mean"] for _, m in gitems]
    nsvc = next((m["n_services"] for _, m in gitems if m.get("n_services")), 0)
    x = np.arange(len(gnames)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.bar(x - w/2, agg_slo, w, label="Aggregate SLO", color="#3b6db2")
    ax.bar(x + w/2, worst, w, label="Worst-service violation", color="#b23b3b")
    ax.axhline(agg["slo_target"] * 100, ls="--", lw=1, color="#444",
               label=f"{int(agg['slo_target']*100)}% target")
    ytop = max(worst) if worst else 1.0
    ax.set_ylim(0, ytop * 1.20)
    for xi, wv, o in zip(x, worst, over):
        ax.annotate(f"{o:.0f}/{nsvc} svc", (xi + w/2, wv),
                    xytext=(xi + w/2, wv + ytop * 0.025),
                    fontsize=7, ha="center", color="#b23b3b")
    ax.set_xticks(x); ax.set_xticklabels(gnames, rotation=18, ha="right")
    ax.set_ylabel("CPU under-provisioning (%)")
    ax.set_title(f"Aggregate vs per-service reliability gap (H = {hmid} s)", pad=26)
    ax.legend(fontsize=8, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "phase2_perservice.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
