"""Build forecasting samples. Each sample is a decision to provision for a
target time t, using only information available H steps earlier (window ending
at t-H) -- so every method, classical or learned, forecasts t from the same
lead-correct history. No leakage.

Each service's series is reindexed onto a DENSE per-minute grid before windows
are cut, so the H-step lead is a true H minutes even when telemetry has gaps
(missing minutes are forward/back-filled and counted). This fixes the effective
forecast lead flagged in review.

Returns a bundle:
  X      : feature matrix (for learned methods)
  y      : actual demand at target t
  yprev  : demand at t-1 (for the reactive baseline)
  meta   : DataFrame[service, t_step]
  series : {service: dense demand ndarray}
  pos    : target index p within each service's DENSE series (parallel to rows)
  svc    : service id per row (parallel)
  names  : feature names
  grid_missing / grid_total : filled vs total minute-cells (missing-minute rate)
"""
import numpy as np
import pandas as pd


def _feat(w, tstep_abs, day):
    mean = w.mean(); std = w.std(); last = w[-1]
    slope = (w[-1] - w[0]) / max(1, len(w) - 1)
    cv = std / (mean + 1e-9)
    tod = (tstep_abs % day) / day           # absolute time-of-day, not row index
    return [mean, std, w.min(), w.max(), last, slope, cv,
            np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod)]

NAMES = ["mean", "std", "min", "max", "last", "slope", "cv", "tod_sin", "tod_cos"]


def build(df, dt, cfg):
    W = max(1, int(cfg["features"]["window_seconds"] / dt))
    H = max(1, int(cfg["features"]["horizon_seconds"] / dt))
    day = 24 * 3600 / dt
    X, y, yprev, svc, pos, mt = [], [], [], [], [], []
    series = {}
    grid_missing = 0; grid_total = 0
    for s, g in df.sort_values("t_step").groupby("service"):
        g = g.drop_duplicates("t_step", keep="last")
        t0, t1 = int(g["t_step"].min()), int(g["t_step"].max())
        full = np.arange(t0, t1 + 1)                       # dense minute grid
        dd = g.set_index("t_step")["demand"].reindex(full)
        grid_missing += int(dd.isna().sum()); grid_total += len(full)
        d = dd.ffill().bfill().to_numpy(dtype=float)       # fill missing minutes
        if not np.isfinite(d).all():
            continue
        series[s] = d
        n = len(d)
        for p in range(H + W, n):              # target index p (dense)
            w = d[p - H - W + 1: p - H + 1]     # window ending H before target
            X.append(_feat(w, int(full[p]), day))
            y.append(d[p]); yprev.append(d[p - 1])
            svc.append(s); pos.append(p); mt.append((s, int(full[p])))
    return {
        "X": np.asarray(X, float), "y": np.asarray(y, float),
        "yprev": np.asarray(yprev, float),
        "meta": pd.DataFrame(mt, columns=["service", "t_step"]),
        "series": series, "pos": np.asarray(pos), "svc": np.asarray(svc, dtype=object),
        "names": NAMES, "W": W, "H": H,
        "grid_missing": grid_missing, "grid_total": grid_total,
    }
