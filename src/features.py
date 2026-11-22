"""Build forecasting samples. Each sample is a decision to provision for a
target time t, using only information available H steps earlier (window ending
at t-H) — so every method, classical or learned, forecasts t from the same
lead-correct history. No leakage.

Returns a bundle:
  X      : feature matrix (for learned methods)
  y      : actual demand at target t
  yprev  : demand at t-1 (for the reactive baseline)
  meta   : DataFrame[service, t_step]
  series : {service: demand ndarray}
  pos    : target index p within each service's series (parallel to rows)
  svc    : service id per row (parallel)
  names  : feature names
"""
import numpy as np
import pandas as pd


def _feat(w, p, day):
    mean = w.mean(); std = w.std(); last = w[-1]
    slope = (w[-1] - w[0]) / max(1, len(w) - 1)
    cv = std / (mean + 1e-9)
    tod = (p % day) / day
    return [mean, std, w.min(), w.max(), last, slope, cv,
            np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod)]

NAMES = ["mean", "std", "min", "max", "last", "slope", "cv", "tod_sin", "tod_cos"]


def build(df, dt, cfg):
    W = max(1, int(cfg["features"]["window_seconds"] / dt))
    H = max(1, int(cfg["features"]["horizon_seconds"] / dt))
    day = 24 * 3600 / dt
    X, y, yprev, svc, pos, mt = [], [], [], [], [], []
    series = {}
    for s, g in df.sort_values("t_step").groupby("service"):
        d = g["demand"].to_numpy(dtype=float)
        tsteps = g["t_step"].to_numpy()
        series[s] = d
        n = len(d)
        for p in range(H + W, n):              # target index p
            w = d[p - H - W + 1: p - H + 1]     # window ending H before target
            X.append(_feat(w, p, day))
            y.append(d[p]); yprev.append(d[p - 1])
            svc.append(s); pos.append(p); mt.append((s, int(tsteps[p])))
    return {
        "X": np.asarray(X, float), "y": np.asarray(y, float),
        "yprev": np.asarray(yprev, float),
        "meta": pd.DataFrame(mt, columns=["service", "t_step"]),
        "series": series, "pos": np.asarray(pos), "svc": np.asarray(svc, dtype=object),
        "names": NAMES, "W": W, "H": H,
    }
