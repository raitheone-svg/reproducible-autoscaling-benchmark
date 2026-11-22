"""Leakage-safe chronological split, per service.

For a continuous forecasting target, a simple chronological holdout is the right
choice (unlike rare-event detection). We split each service's timeline into
train / val / test in order, and purge a gap of (W+H) samples at each boundary
so a later split's feature window never overlaps an earlier split's targets.
This keeps ~80% of samples (the earlier blocked split wasted most of them on
short series).
"""
import numpy as np


def blocked_split(meta, cfg):
    dt = cfg["data"]["dt_seconds"]
    W = max(1, int(cfg["features"]["window_seconds"] / dt))
    H = max(1, int(cfg["features"]["horizon_seconds"] / dt))
    gap = W + H
    vf = cfg["split"]["val_fraction"]; tf = cfg["split"]["test_fraction"]
    tr = np.zeros(len(meta), bool); va = np.zeros(len(meta), bool); te = np.zeros(len(meta), bool)
    for svc, idx in meta.groupby("service").groups.items():
        idx = np.array(sorted(idx, key=lambda k: meta.loc[k, "t_step"]))
        n = len(idx)
        a = int(n * (1 - vf - tf)); b = int(n * (1 - tf))
        tr[idx[0: max(0, a - gap)]] = True      # train, minus gap before val
        va[idx[a: max(a, b - gap)]] = True       # val,   minus gap before test
        te[idx[b: n]] = True                     # test
    return tr, va, te
