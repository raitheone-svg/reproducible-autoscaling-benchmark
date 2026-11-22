#!/usr/bin/env python3
"""Small, hand-checkable tests for the benchmark harness.
Run:  python tests/test_harness.py   (no pytest needed)

Covers the reviewer's requested checks: delay shift, headroom math, cost
aggregation, split disjointness / no-leakage, and synthetic trend recovery.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import features, split, methods, metrics

PASS = 0; FAIL = 0
def check(name, cond):
    global PASS, FAIL
    print(("PASS " if cond else "FAIL ") + name)
    PASS += cond; FAIL += (not cond)

CFG = {"data": {"dt_seconds": 60.0}, "features": {"window_seconds": 600.0, "horizon_seconds": 180.0},
       "split": {"val_fraction": 0.2, "test_fraction": 0.2}, "provision": {"deadband": 0.05},
       "run": {"seed": 0}}

# a small deterministic two-service frame
def make_df():
    rows = []
    for s in ("A", "B"):
        for t in range(120):
            rows.append((s, t, 10.0 + 0.1 * t + (2.0 if s == "B" else 0.0), 1.0))
    return pd.DataFrame(rows, columns=["service", "t_step", "demand", "mem"])

b = features.build(make_df(), 60.0, CFG)
H = b["H"]

# 1) Delay shift: reactive forecast for target t equals demand at t-H.
yhat_react = methods.reactive(b)
ok = all(abs(yhat_react[i] - b["series"][b["svc"][i]][b["pos"][i] - H]) < 1e-9 for i in range(len(yhat_react)))
check("reactive forecast = demand[t-H] (delay shift)", ok)

# 2) Headroom math + cost aggregation: capacity = yhat*(1+h); cost = mean(capacity).
y = np.array([10., 20., 30.]); yhat = np.array([10., 10., 10.]); svc = np.array(["A", "A", "A"]); ts = np.array([0, 1, 2])
pt = metrics.provision_point(y, yhat, 0.5, svc, ts, 0.05)
check("capacity cost = mean(yhat*(1+h))", abs(pt["cost"] - 15.0) < 1e-9)
# under-provision when capacity(15) < demand(20,30) -> 2 of 3
check("under-provision rate counts capacity<demand", abs(pt["slo"] - 2/3) < 1e-3)

# 3) Split: masks disjoint, all non-empty, and no feature window crosses a split boundary.
tr, va, te = split.blocked_split(b["meta"], CFG)
check("splits are disjoint", not ((tr & va).any() or (tr & te).any() or (va & te).any()))
check("splits all non-empty", tr.sum() > 0 and va.sum() > 0 and te.sum() > 0)
# leakage: a test target's history window [t-H-W, t-H] must not contain a train target's t
W = b["W"]; meta = b["meta"]
leak = False
train_t = {(b["svc"][i], int(meta["t_step"].iloc[i])) for i in np.where(tr)[0]}
for i in np.where(te)[0]:
    s = b["svc"][i]; p = b["pos"][i]
    for q in range(p - H - W, p - H + 1):
        if (s, q) in train_t:
            leak = True; break
check("no train target inside a test feature window (no leakage)", not leak)

# 4) Synthetic trend recovery: on a pure linear ramp, Holt's H-step forecast is near-exact.
ramp = np.arange(200, dtype=float)
lvl, trd = methods._holt(ramp, alpha=0.5, beta=0.5)
pred = lvl[100] + H * trd[100]
check("Holt recovers a linear trend (|err|<0.5)", abs(pred - ramp[100 + H]) < 0.5)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
