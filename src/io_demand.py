"""Load per-microservice CPU demand from the Alibaba MSMetrics table (or a
synthetic sample). Streams large CSVs in chunks; keeps the busiest services.

Output: tidy DataFrame [service, t_step, demand, mem] at fixed dt, plus dt.
`demand` = aggregate CPU utilization of the microservice (sum over its
containers) — the quantity an autoscaler must provision for.
"""
import os, glob
import numpy as np
import pandas as pd

MSRES_COLS = ["timestamp", "msname", "msinstanceid", "nodeid",
              "cpu_utilization", "memory_utilization"]
ALIAS = {"cpu_utilization": "cpu", "memory_utilization": "mem", "node_id": "nodeid"}
CHUNK = 1_000_000


def _find(dd, *names, exclude=()):
    out = []
    for name in names:
        out += glob.glob(os.path.join(dd, "**", f"*{name}*.csv"), recursive=True)
        out += glob.glob(os.path.join(dd, "**", f"*{name}*.CSV"), recursive=True)
    return [f for f in sorted(set(out)) if os.path.isfile(f)
            and not any(x in os.path.basename(f) for x in exclude)]


def _detect(path):
    head = pd.read_csv(path, nrows=1, header=None)
    v = head.iloc[0, 0]
    has_header = isinstance(v, str) and not str(v).replace(".", "", 1).isdigit()
    return has_header, head.shape[1]


def _iter(path, schema):
    has_header, ncols = _detect(path)
    if has_header:
        header, names = 0, None
    else:
        names = (schema + [f"x{i}" for i in range(max(0, ncols - len(schema)))])[:ncols]
        header = None
    for ch in pd.read_csv(path, header=header, names=names, chunksize=CHUNK):
        yield ch.rename(columns=ALIAS)


def _load_real(cfg):
    dd = cfg["data"]["data_dir"]
    dt = cfg["data"]["dt_seconds"]
    maxN = int(cfg["data"]["max_services"])
    files = _find(dd, "MSMetrics", "MSResource", exclude=("MSRTMCR", "CallGraph", "Node"))
    if not files:
        raise FileNotFoundError(
            f"No MSMetrics/MSResource CSV under {dd}. Extract the downloaded "
            "MSMetrics .tar.gz into data/ (see README).")
    # --- prospective service selection (no leakage): choose the cohort using
    # ONLY the training period, defined by a global timestamp cutoff, so the
    # busiest-N selection cannot use test-period activity. ---
    vf = cfg["split"]["val_fraction"]; tf = cfg["split"]["test_fraction"]
    train_frac = 1.0 - vf - tf
    # pass 0: global max t_step -> training cutoff
    tmax = 0
    for p in files:
        for ch in _iter(p, MSRES_COLS):
            m = int((ch["timestamp"] / 1000.0 / dt).max())
            if m > tmax:
                tmax = m
    train_cutoff = int(train_frac * tmax)
    # pass 1: busiest services by peak aggregate CPU WITHIN the training period
    peak = {}
    for p in files:
        for ch in _iter(p, MSRES_COLS):
            ch = ch.assign(t_step=(ch["timestamp"] / 1000.0 / dt).astype(int))
            ch = ch[ch["t_step"] < train_cutoff]
            if not len(ch):
                continue
            s = ch.groupby(["msname", "t_step"])["cpu"].sum().groupby("msname").max()
            for sv, v in s.items():
                if v > peak.get(sv, -1.0):
                    peak[sv] = v
    top = set(sorted(peak, key=peak.get, reverse=True)[:maxN])
    # pass 2: aggregate demand for the top services
    parts = []
    for p in files:
        for ch in _iter(p, MSRES_COLS):
            ch = ch[ch["msname"].isin(top)]
            if len(ch):
                ch = ch.assign(t_step=(ch["timestamp"] / 1000.0 / dt).astype(int))
                parts.append(ch.groupby(["msname", "t_step"])
                               .agg(demand=("cpu", "sum"), mem=("mem", "sum")).reset_index())
    g = (pd.concat(parts, ignore_index=True).groupby(["msname", "t_step"])
           .agg(demand=("demand", "sum"), mem=("mem", "sum")).reset_index())
    return g.rename(columns={"msname": "service"})[["service", "t_step", "demand", "mem"]], dt


def _load_synth(cfg):
    d = cfg["data"]; dt = d["dt_seconds"]
    rng = np.random.default_rng(cfg["run"]["seed"])
    n_steps = int(d["duration_hours"] * 3600 / dt)
    t = np.arange(n_steps)
    day = 24 * 3600 / dt
    rows = []
    for s in range(d["n_services"]):
        base = rng.uniform(1, 5)
        amp = rng.uniform(0.3, 1.2) * base
        phase = rng.uniform(0, 2 * np.pi)
        # daily sinusoid + AR(1) noise + occasional bursts  -> forecastable demand
        season = base + amp * (0.5 + 0.5 * np.sin(2 * np.pi * t / day + phase))
        noise = np.zeros(n_steps); e = 0.0
        for i in range(n_steps):
            e = 0.8 * e + rng.normal(0, 0.15 * base); noise[i] = e
        dem = np.clip(season + noise, 0.05, None)
        for _ in range(rng.integers(2, 6)):          # bursts
            c = rng.integers(0, n_steps); w = rng.uniform(2, 6)
            dem += rng.uniform(0.5, 1.5) * base * np.exp(-((t - c) ** 2) / (2 * w ** 2))
        for i in range(n_steps):
            rows.append((f"svc_{s:03d}", int(t[i]), float(dem[i]), float(dem[i] * 0.3)))
    g = pd.DataFrame(rows, columns=["service", "t_step", "demand", "mem"])
    return g, dt


def load(cfg):
    return _load_synth(cfg) if cfg["data"]["mode"] == "synthetic" else _load_real(cfg)
