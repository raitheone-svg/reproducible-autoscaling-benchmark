#!/usr/bin/env python3
"""Phase-2 benchmark (robustness evidence), revised after pre-submission review.

Changes vs. the previous version:
  * PROSPECTIVE, leakage-safe cohort: each window's services are selected from
    that window's OWN training period (io_demand.load_phase2), not a global
    pre-window cutoff. Per-window cohorts and turnover are recorded.
  * DENSE minute grid in feature building, so the H-step lead is true minutes.
  * AUDITABLE export: the full disaggregated per-(window, method, lead, seed)
    run matrix and per-service diagnostics are written to disk, so every
    reported number is reproducible from released files.
  * DESCRIPTIVE statistics: seeds are aggregated WITHIN each window first; the
    cross-window summary reports the mean and the min-max band of the five
    seed-averaged window costs, and the per-window paired differences. No
    bootstrap p-value is reported -- five windows cannot support a paired
    significance test (exact two-sided sign-test floor 2/32 = 0.0625).

Usage:
  python run_phase2.py --config config_phase2.yaml
"""
import argparse, json, os, copy, hashlib, platform, subprocess, sys
import numpy as np
import yaml
from src import io_demand, features, split, methods, metrics, report

DISP = {"reactive": "Delayed persistence", "moving_average": "Moving average",
        "holt": "Holt (ES)", "gbdt": "GBDT", "lstm": "LSTM", "oracle": "Oracle (perfect)"}
SLO_TARGET = getattr(report, "SLO_TARGET", 0.05)


def _run_methods(b, tr, which, seed):
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
    out["oracle"] = b["y"].astype(float)       # perfect-forecast lower bound
    return out


def _per_service_tail(b, yhat_all, te, h):
    """Per-service shortfall rate at the frozen headroom h, on the test split,
    with the per-service test-cell denominator."""
    y = b["y"][te]; svc = b["svc"][te]; yhat = yhat_all[te] * (1.0 + h)
    viol = (yhat < y).astype(float)
    rates, denom = {}, {}
    for s in np.unique(svc):
        m = svc == s
        if m.sum():
            rates[s] = float(viol[m].mean()); denom[s] = int(m.sum())
    ids = list(rates.keys())
    r = np.array([rates[s] for s in ids]) if ids else np.array([0.0])
    dn = np.array([denom[s] for s in ids]) if ids else np.array([0])
    return dict(worst=round(float(r.max()), 4),
                p95=round(float(np.quantile(r, 0.95)), 4),
                p90=round(float(np.quantile(r, 0.90)), 4),
                median=round(float(np.median(r)), 4),
                n_services=int(len(r)),
                n_over_target=int((r > SLO_TARGET).sum()),
                denom_min=int(dn.min()), denom_median=int(np.median(dn)),
                svc_ids=[str(s) for s in ids],
                denoms=[int(denom[s]) for s in ids],
                rates=[round(float(x), 4) for x in r])


def _one_run(df, dt, cfg, h_seconds, seed):
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
        h = mm["frozen_headroom"]
        pt = next((p for p in mm["pareto_test"] if p["headroom"] == h), {})
        rec[name] = dict(
            cost=mm["test_cost"], slo=mm["test_slo"], headroom=h,
            frozen_val_slo=mm["frozen_val_slo"],
            val_feasible=bool(mm["frozen_val_slo"] <= SLO_TARGET),
            test_feasible=bool(mm["test_slo"] <= SLO_TARGET),
            overprov_pct=mm["test_overprov_pct"], actions=pt.get("actions"),
            mae=mm["accuracy"]["mae"], rmse=mm["accuracy"]["rmse"],
            smape=mm["accuracy"]["smape"], usable_cells=int(te.sum()),
            tail=_per_service_tail(b, fc[name], te, h))
    return rec, int(b["grid_missing"]), int(b["grid_total"])


def _window_mean(cells):
    """cells: {w_idx: [values over seeds]} -> {w_idx: seed-mean}."""
    return {w: float(np.mean(v)) for w, v in cells.items() if len(v)}


def _paired_descriptive(method_wm, base_wm):
    wins = sorted(set(method_wm) & set(base_wm))
    deltas = [round((method_wm[w] - base_wm[w]) / base_wm[w] * 100.0, 2)
              for w in wins if base_wm[w]]
    if not deltas:
        return dict(delta_pct_mean=float("nan"), per_window_delta=[], n_windows=0,
                    n_positive=0, n_negative=0)
    a = np.array(deltas)
    return dict(delta_pct_mean=round(float(a.mean()), 2),
                per_window_delta=deltas, n_windows=len(deltas),
                n_positive=int((a > 0).sum()), n_negative=int((a < 0).sum()))


def _data_fingerprint(cfg):
    try:
        files = io_demand._find_files(cfg)
    except Exception:
        return []
    out = []
    for f in files:
        try:
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(blk)
            out.append(dict(file=os.path.basename(f), bytes=os.path.getsize(f),
                            sha256=h.hexdigest()))
        except Exception as e:
            out.append(dict(file=os.path.basename(f), error=str(e)))
    return out


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _versions():
    v = {}
    for m in ("numpy", "pandas", "sklearn", "scipy", "torch", "matplotlib", "yaml"):
        try:
            v[m] = __import__(m).__version__
        except Exception:
            v[m] = "absent"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--skip-checksums", action="store_true",
                    help="skip sha256 of raw trace files (faster)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    p2 = cfg.get("phase2", {})
    seeds = p2.get("seeds", [7, 17, 23])
    leads = p2.get("lead_times_seconds", [60, 180, 300])
    out_dir = cfg["run"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"[phase2] leakage-safe load (mode={cfg['data']['mode']}) ...")
    df, dt, windows, cohorts, info = io_demand.load_phase2(cfg)
    n_windows = len(windows)
    print(f"         windows={windows}")
    print(f"         per-window cohort sizes={info['cohort_sizes']} "
          f"union={info['cohort_union']} turnover={info['cohort_turnover']}")
    print(f"[phase2] grid: {n_windows} window(s) x {len(leads)} lead(s) x "
          f"{len(seeds)} seed(s) = {n_windows*len(leads)*len(seeds)} runs")

    cells = {h: {} for h in leads}          # cost:   [lead][method][w] -> [seeds]
    slo_cells = {h: {} for h in leads}      # test slo
    tail_cells = {h: {} for h in leads}     # tail dicts
    matrix = []                             # disaggregated per-(w,lead,seed,method)
    gmiss = gtot = 0
    total = n_windows * len(leads) * len(seeds); done = 0
    for w_idx, (w0, w1) in enumerate(windows):
        coh = cohorts.get(w_idx, set())
        dfw = df[(df["t_step"] >= w0) & (df["t_step"] < w1) & (df["service"].isin(coh))]
        for h_sec in leads:
            for seed in seeds:
                done += 1
                out = _one_run(dfw, dt, cfg, h_sec, seed)
                tag = f"win{w_idx}[{w0},{w1}) H={h_sec}s seed={seed}"
                if out is None:
                    print(f"  ({done}/{total}) {tag}: skipped (too little data)")
                    continue
                rec, gm, gt = out; gmiss += gm; gtot += gt
                for name, r in rec.items():
                    cells[h_sec].setdefault(name, {}).setdefault(w_idx, []).append(r["cost"])
                    slo_cells[h_sec].setdefault(name, {}).setdefault(w_idx, []).append(r["slo"])
                    tail_cells[h_sec].setdefault(name, {}).setdefault(w_idx, []).append(r["tail"])
                    matrix.append(dict(window=w_idx, w0=w0, w1=w1, lead_s=h_sec, seed=seed,
                                       method=name, cohort_size=len(coh),
                                       headroom=r["headroom"], frozen_val_slo=r["frozen_val_slo"],
                                       val_feasible=r["val_feasible"], test_slo=r["slo"],
                                       test_feasible=r["test_feasible"], cost=r["cost"],
                                       overprov_pct=r["overprov_pct"], actions=r["actions"],
                                       mae=r["mae"], rmse=r["rmse"], smape=r["smape"],
                                       usable_cells=r["usable_cells"],
                                       svc_over_target=r["tail"]["n_over_target"],
                                       svc_total=r["tail"]["n_services"],
                                       worst_svc=r["tail"]["worst"], p95_svc=r["tail"]["p95"],
                                       denom_median=r["tail"]["denom_median"]))
                print(f"  ({done}/{total}) {tag}: ok")

    # ---- window-level descriptive aggregation (seeds within window first) ----
    agg = {"protocol": "phase2 revised: leakage-safe per-window cohort; "
                        "seeds aggregated within window; descriptive min-max band "
                        "over windows; no significance test (n_windows small)",
           "slo_target": SLO_TARGET, "seeds": seeds, "lead_times_seconds": leads,
           "n_windows": n_windows, "windows": windows,
           "provenance": cfg["data"]["provenance"],
           "cohort": {"sizes": info["cohort_sizes"], "union": info["cohort_union"],
                      "intersection": info["cohort_intersection"],
                      "turnover": info["cohort_turnover"]},
           "grid_missing_minute_rate": round(gmiss / max(1, gtot), 5),
           "lead_times": {}, "paired_vs_baseline": {}}
    for h in leads:
        block = {}
        for name, wd in cells[h].items():
            wm = _window_mean(wd)                       # window -> seed-mean cost
            vals = np.array(list(wm.values()))
            slo_wm = _window_mean(slo_cells[h].get(name, {}))
            slo_vals = np.array(list(slo_wm.values())) if slo_wm else np.array([np.nan])
            # tail: average the per-seed tail stats within window, then over windows
            tw = tail_cells[h].get(name, {})
            def _tmean(key):
                per_win = [np.mean([t[key] for t in seeds_list]) for seeds_list in tw.values() if seeds_list]
                return round(float(np.mean(per_win)), 4) if per_win else float("nan")
            block[name] = dict(
                display=DISP.get(name, name), n_windows=int(len(wm)),
                cost_mean=round(float(vals.mean()), 3) if len(vals) else float("nan"),
                cost_min=round(float(vals.min()), 3) if len(vals) else float("nan"),
                cost_max=round(float(vals.max()), 3) if len(vals) else float("nan"),
                window_costs={int(w): round(c, 3) for w, c in wm.items()},
                slo_mean=round(float(np.nanmean(slo_vals)), 4),
                test_feasible_windows=int((slo_vals <= SLO_TARGET).sum()),
                tail_worst_mean=_tmean("worst"), tail_p95_mean=_tmean("p95"),
                tail_services_over_target_mean=_tmean("n_over_target"),
                n_services=next((t[0]["n_services"] for t in tw.values() if t), 0))
        agg["lead_times"][h] = block
        base_wm = _window_mean(cells[h].get("reactive", {}))
        pb = {}
        for name, wd in cells[h].items():
            if name in ("reactive", "oracle"):
                continue
            pb[name] = _paired_descriptive(_window_mean(wd), base_wm)
        agg["paired_vs_baseline"][h] = pb

    # ---- write outputs ----
    json.dump(agg, open(os.path.join(out_dir, "phase2_results.json"), "w"), indent=2)
    json.dump(matrix, open(os.path.join(out_dir, "phase2_runmatrix.json"), "w"), indent=2)
    # per-service rates per (window, method) at the middle lead, for the gap table
    hmid = leads[len(leads) // 2]
    ps = []
    for name, wd in tail_cells[hmid].items():
        for w, seeds_list in wd.items():
            for si, t in enumerate(seeds_list):
                ps.append(dict(window=w, lead_s=hmid, method=name, seed=seeds[si] if si < len(seeds) else si,
                               n_services=t["n_services"], n_over_target=t["n_over_target"],
                               worst=t["worst"], p95=t["p95"], denom_median=t["denom_median"],
                               svc_ids=t.get("svc_ids", []), denoms=t.get("denoms", []),
                               rates=t["rates"]))
    json.dump(ps, open(os.path.join(out_dir, "phase2_perservice.json"), "w"), indent=2)

    run_manifest = dict(git_commit=_git_commit(), python=platform.python_version(),
                        platform=platform.platform(), packages=_versions(),
                        config=cfg, windows=windows, cohort=agg["cohort"],
                        cohorts_by_window=info["cohorts"],
                        grid_missing_minute_rate=agg["grid_missing_minute_rate"],
                        figure_commands=["python run_phase2.py --config config_phase2.yaml",
                                         "python analyze_phase2.py"],
                        data_files=[] if args.skip_checksums else _data_fingerprint(cfg))
    json.dump(run_manifest, open(os.path.join(out_dir, "phase2_manifest.json"), "w"), indent=2)

    _write_summary(out_dir, agg)
    _print(agg, leads)
    print(f"\nWrote {out_dir}/phase2_results.json, phase2_runmatrix.json, "
          f"phase2_perservice.json, phase2_manifest.json, phase2_summary.md")


def _print(agg, leads):
    print("\nPhase-2 (revised) window-level cost by lead time "
          "[mean over windows, (min, max) band]:")
    for h in leads:
        print(f"  H = {h}s:")
        blk = agg["lead_times"][h]
        base = blk.get("reactive", {}).get("cost_mean")
        pb = agg["paired_vs_baseline"].get(h, {})
        for name, m in sorted(blk.items(), key=lambda kv: kv[1]["cost_mean"]):
            vs = ""
            if name in pb and pb[name]["n_windows"]:
                d = pb[name]
                vs = (f"  [dcost {d['delta_pct_mean']:+.1f}% "
                      f"pos {d['n_positive']}/{d['n_windows']}]")
            print(f"    {m['display']:<20} cost={m['cost_mean']:>8} "
                  f"({m['cost_min']}, {m['cost_max']}) slo={m['slo_mean']:.3f}"
                  f" feas_win={m['test_feasible_windows']}/{m['n_windows']}{vs}")


def _write_summary(out_dir, agg):
    L = ["# Phase-2 results (revised, leakage-safe)\n", f"{agg['protocol']}.",
         f"Target shortfall <= {agg['slo_target']}. Provenance: {agg['provenance']}.",
         f"Windows: {agg['n_windows']}; per-window cohort turnover "
         f"{agg['cohort']['turnover']} (union {agg['cohort']['union']}, "
         f"intersection {agg['cohort']['intersection']}); "
         f"missing-minute rate {agg['grid_missing_minute_rate']}.\n"]
    for h, blk in agg["lead_times"].items():
        L.append(f"\n## Lead time H = {h} s\n")
        L.append("| Method | Cost mean (min-max) | Test shortfall | Feasible windows | #svc>5% |")
        L.append("|---|---|---|---|---|")
        for name, m in sorted(blk.items(), key=lambda kv: kv[1]["cost_mean"]):
            L.append(f"| {m['display']} | {m['cost_mean']} ({m['cost_min']}-{m['cost_max']}) | "
                     f"{m['slo_mean']} | {m['test_feasible_windows']}/{m['n_windows']} | "
                     f"{m['tail_services_over_target_mean']} |")
        pb = agg["paired_vs_baseline"].get(h, {})
        if pb:
            L.append("\n*Paired vs delayed-persistence (per-window, descriptive; no p-value):*")
            L.append("| Method | mean Δcost % | per-window Δ% | #windows more expensive |")
            L.append("|---|---|---|---|")
            for name, d in sorted(pb.items(), key=lambda kv: kv[1]["delta_pct_mean"]):
                L.append(f"| {DISP.get(name, name)} | {d['delta_pct_mean']} | "
                         f"{d['per_window_delta']} | {d['n_positive']}/{d['n_windows']} |")
    L.append("\n*Five windows cannot support a paired significance test "
             "(exact two-sided sign-test floor 2/32 = 0.0625); estimates are descriptive.*\n")
    open(os.path.join(out_dir, "phase2_summary.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()
