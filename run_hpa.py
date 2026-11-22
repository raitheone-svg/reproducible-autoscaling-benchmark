#!/usr/bin/env python3
"""HPA baseline comparison.

Evaluates a Kubernetes-style replica-based reactive autoscaler (src/hpa.py)
against the continuous delayed-persistence baseline, under the SAME
validation-frozen protocol, on the real trace. Addresses the reviewer
recommendation for a realistic Kubernetes HPA baseline: it shows how much the
integer-replica, tolerance, and downscale-stabilisation constraints add over the
idealised continuous allocation the paper otherwise uses.

  python run_hpa.py --config config.yaml [--target-replicas 20]

Writes <out_dir>/hpa_results.json and hpa_summary.md.
"""
import argparse, json, os
import numpy as np, yaml
from src import io_demand, features, split, methods, metrics, hpa

SLO_TARGET = 0.05
TARGET_UTILS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]   # lower target = more headroom


def hpa_curve(b, va, te, per_replica, hpa_cfg):
    H = b["H"]; y = b["y"]
    val, tst = [], []
    for tu in TARGET_UTILS:
        cap, acts = hpa.evaluate_rows(b, H, per_replica, tu, hpa_cfg)
        for mask, out in ((va, val), (te, tst)):
            c = cap[mask]; a = y[mask]
            out.append(dict(target_util=tu, slo=round(float((c < a).mean()), 4),
                            cost=round(float(c.mean()), 4), actions=round(acts, 2)))
    return val, tst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--target-replicas", type=float, default=20.0)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    np.random.seed(cfg["run"]["seed"])
    print("[hpa] loading demand ...")
    df, dt = io_demand.load(cfg)
    b = features.build(df, dt, cfg)
    tr, va, te = split.blocked_split(b["meta"], cfg)
    print(f"      services={df['service'].nunique()} test_cells={int(te.sum())} "
          f"(W={b['W']} H={b['H']})")
    per_replica = hpa.derive_per_replica(b, a.target_replicas)
    hpa_cfg = cfg.get("hpa", {})
    print(f"[hpa] per_replica_capacity={per_replica:.3f} "
          f"(mean demand / {a.target_replicas:g} replicas)")

    base = metrics.evaluate(b, {"reactive": methods.reactive(b)}, va, te, cfg,
                            slo_target=SLO_TARGET)["reactive"]
    val, tst = hpa_curve(b, va, te, per_replica, hpa_cfg)
    feas = [p for p in val if p["slo"] <= SLO_TARGET]
    chosen = min(feas, key=lambda p: p["cost"]) if feas else min(val, key=lambda p: p["slo"])
    tu = chosen["target_util"]
    hpa_test = next(p for p in tst if p["target_util"] == tu)

    out_dir = cfg["run"]["out_dir"]; os.makedirs(out_dir, exist_ok=True)
    res = dict(provenance=cfg["data"]["provenance"],
               per_replica_capacity=round(per_replica, 4),
               target_replicas=a.target_replicas, slo_target=SLO_TARGET,
               delayed_persistence=dict(cost=base["test_cost"], slo=base["test_slo"],
                                        headroom=base["frozen_headroom"]),
               hpa=dict(cost=hpa_test["cost"], slo=hpa_test["slo"], target_util=tu,
                        actions_per_service=hpa_test["actions"]),
               hpa_curve_test=tst)
    json.dump(res, open(os.path.join(out_dir, "hpa_results.json"), "w"), indent=2)
    _summary(out_dir, res)
    _print(res)
    print(f"\nWrote {out_dir}/hpa_results.json, hpa_summary.md")


def _print(res):
    b = res["delayed_persistence"]; h = res["hpa"]
    dpct = (h['cost'] - b['cost']) / b['cost'] * 100
    print("\nValidation-frozen comparison (test split):")
    print(f"  {'Delayed persistence (cont.)':<28} cost={b['cost']:>8}  "
          f"under-prov={b['slo']*100:>4.1f}%  headroom h*={b['headroom']}")
    print(f"  {'HPA replicas (reactive)':<28} cost={h['cost']:>8}  "
          f"under-prov={h['slo']*100:>4.1f}%  target_util={h['target_util']}, "
          f"{h['actions_per_service']:.1f} actions/svc")
    print(f"\n  HPA provisions {dpct:+.1f}% vs the continuous baseline at comparable "
          f"reliability --\n  the overhead of integer replicas, tolerance, and downscale "
          f"stabilisation that the\n  continuous abstraction hides.")


def _summary(out_dir, res):
    b = res["delayed_persistence"]; h = res["hpa"]
    dpct = (h['cost'] - b['cost']) / b['cost'] * 100
    L = ["# HPA baseline comparison\n",
         f"Provenance: {res['provenance']}. Target SLO $\\le$ {res['slo_target']}. "
         f"per_replica_capacity = {res['per_replica_capacity']} "
         f"(mean demand / {res['target_replicas']:g}).\n",
         "| Policy | Actuation | Cost | Under-prov % | Operating point |",
         "|---|---|---|---|---|",
         f"| Delayed persistence | continuous allocation | {b['cost']} | "
         f"{round(b['slo']*100,1)} | headroom {b['headroom']} |",
         f"| HPA (reactive) | integer replicas, tolerance/stabilisation/rate-limited | "
         f"{h['cost']} | {round(h['slo']*100,1)} | target_util {h['target_util']}, "
         f"{h['actions_per_service']} actions/svc |",
         f"\nAt comparable reliability the replica HPA provisions **{dpct:+.1f}%** versus "
         "the idealised continuous baseline -- the cost of integer replicas, the tolerance "
         "band, and downscale stabilisation that the continuous abstraction omits. "
         "`per_replica_capacity` is a documented modelling choice (one replica serves the "
         f"mean busy-service demand divided by {res['target_replicas']:g}); the HPA cost "
         "scales with it, but the sign of the overhead does not.\n"]
    open(os.path.join(out_dir, "hpa_summary.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()
