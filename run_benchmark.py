#!/usr/bin/env python3
"""Autoscaling benchmark driver.

  python run_benchmark.py --config config.yaml

Set data.mode: synthetic for a self-test (no data needed), or alibaba to run on
the real MSMetrics trace in data/.
"""
import argparse, yaml
import numpy as np
from src import io_demand, features, split, methods, metrics, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    np.random.seed(cfg["run"]["seed"])

    print(f"[1/5] loading demand (mode={cfg['data']['mode']}) ...")
    df, dt = io_demand.load(cfg)
    print(f"      services={df['service'].nunique()}  rows={len(df)}  dt={dt}s")

    print("[2/5] building forecasting samples ...")
    b = features.build(df, dt, cfg)
    tr, va, te = split.blocked_split(b["meta"], cfg)
    print(f"      samples={len(b['y'])}  train={tr.sum()} val={va.sum()} test={te.sum()}"
          f"  (W={b['W']} H={b['H']})")

    print("[3/5] running methods (learned models train on TRAIN only) ...")
    forecasts = methods.run(b, tr, cfg["methods"])   # train on train; val is a true holdout
    print(f"      methods: {', '.join(forecasts)}")

    print("[4/5] evaluating (validation-frozen headroom; test reported) ...")
    results = metrics.evaluate(b, forecasts, va, te, cfg, slo_target=report.SLO_TARGET)

    print("[5/5] reporting ...")
    manifest = _manifest(cfg)
    summary = report.write(cfg["run"]["out_dir"], results, cfg, cfg["data"]["provenance"], manifest)
    print(f"\nValidation-frozen result (headroom chosen on val, reported on test):")
    for m, mm in summary["methods"].items():
        print(f"      {mm['display']:<20} sMAPE={mm['accuracy']['smape']:>5}  "
              f"h*={mm['frozen_headroom']}  test_cap={mm['test_cost']:>8}  "
              f"test_underprov={round(mm['test_slo']*100,1)}%")
    print(f"\nWrote {cfg['run']['out_dir']}/results.json, manifest.json, tables.md, "
          "pareto.png, accuracy.png")
    print(f"Provenance: {summary['provenance']}")


def _manifest(cfg):
    import platform, sys, importlib, subprocess
    vers = {}
    for p in ("numpy", "pandas", "sklearn", "scipy", "matplotlib", "torch"):
        try:
            vers[p] = importlib.import_module(p).__version__
        except Exception:
            vers[p] = "absent"
    commit = "unknown"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass
    return {"seed": cfg["run"]["seed"], "git_commit": commit,
            "python": sys.version.split()[0], "platform": platform.platform(),
            "packages": vers,
            "config": {k: cfg[k] for k in ("data", "features", "split", "provision", "methods")}}


if __name__ == "__main__":
    main()
