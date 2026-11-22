"""Load per-microservice CPU demand from the Alibaba MSMetrics table (or a
synthetic sample). Streams large CSVs in chunks.

`demand` = aggregate CPU utilization of the microservice (sum over its
containers) -- the quantity an autoscaler must provision for.

Two entry points:
  * load(cfg)        -- single committed window (phase-1). Cohort selected from
                        the training period only; service-minute totals are
                        aggregated EXACTLY across chunk boundaries.
  * load_phase2(cfg) -- multi-window (phase-2). Each window's cohort is selected
                        PROSPECTIVELY from that window's OWN training period, so
                        a later window's validation/test activity can never
                        influence which services are studied (fixes the
                        sample-selection leakage flagged in review). Returns the
                        per-window cohorts and turnover so the correction is
                        auditable.
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


def _find_files(cfg):
    dd = cfg["data"]["data_dir"]
    files = _find(dd, "MSMetrics", "MSResource", exclude=("MSRTMCR", "CallGraph", "Node"))
    if not files:
        raise FileNotFoundError(
            f"No MSMetrics/MSResource CSV under {dd}. Extract the downloaded "
            "MSMetrics .tar.gz into data/ (see README).")
    return files


def _exact_peak(parts):
    """Given per-chunk aggregated frames with columns [key..., msname, t_step, cpu],
    sum the SAME (msname, t_step) across chunks (exact), then take each service's
    peak service-minute. `parts` already carry any grouping key columns."""
    allt = pd.concat(parts, ignore_index=True)
    keys = [c for c in allt.columns if c not in ("cpu",)]
    gm = [c for c in keys if c != "t_step"]
    allt = allt.groupby(keys, as_index=False)["cpu"].sum()       # exact minute totals
    return allt.groupby(gm, as_index=False)["cpu"].max()          # per-service peak


def _load_real(cfg):
    """Single-window loader: prospective cohort from the training period, with
    exact cross-chunk service-minute aggregation."""
    dt = cfg["data"]["dt_seconds"]
    maxN = int(cfg["data"]["max_services"])
    files = _find_files(cfg)
    vf = cfg["split"]["val_fraction"]; tf = cfg["split"]["test_fraction"]
    train_frac = 1.0 - vf - tf
    # pass 0: global max t_step -> training cutoff
    tmax = 0
    for p in files:
        for ch in _iter(p, MSRES_COLS):
            m = int((ch["timestamp"] / 1000.0 / dt).max())
            tmax = max(tmax, m)
    train_cutoff = int(train_frac * tmax)
    # pass 1: EXACT busiest-service selection within the training period
    parts = []
    for p in files:
        for ch in _iter(p, MSRES_COLS):
            ch = ch.assign(t_step=(ch["timestamp"] / 1000.0 / dt).astype(int))
            ch = ch[ch["t_step"] < train_cutoff]
            if len(ch):
                parts.append(ch.groupby(["msname", "t_step"], as_index=False)["cpu"].sum())
    peak = _exact_peak(parts) if parts else pd.DataFrame(columns=["msname", "cpu"])
    top = set(peak.sort_values("cpu", ascending=False)["msname"].head(maxN))
    # pass 2: EXACT aggregate demand for the top services
    dparts = []
    for p in files:
        for ch in _iter(p, MSRES_COLS):
            ch = ch[ch["msname"].isin(top)]
            if len(ch):
                ch = ch.assign(t_step=(ch["timestamp"] / 1000.0 / dt).astype(int))
                dparts.append(ch.groupby(["msname", "t_step"])
                               .agg(demand=("cpu", "sum"), mem=("mem", "sum")).reset_index())
    g = (pd.concat(dparts, ignore_index=True).groupby(["msname", "t_step"])
           .agg(demand=("demand", "sum"), mem=("mem", "sum")).reset_index())
    return g.rename(columns={"msname": "service"})[["service", "t_step", "demand", "mem"]], dt


def _window_bounds(tmin, tmax, cfg):
    """Replicate the phase-2 window layout so cohort selection and evaluation
    use identical windows."""
    dt = cfg["data"]["dt_seconds"]
    p2 = cfg.get("phase2", {})
    leads = p2.get("lead_times_seconds", [60, 180, 300])
    W = max(1, int(cfg["features"]["window_seconds"] / dt))
    maxH = max(1, int(max(leads) / dt))
    min_win = max(100, (W + maxH) * 5)
    span = tmax - tmin + 1
    n_req = int(p2.get("n_windows", 2))
    separated = bool(p2.get("separated_windows", False))
    n_eff = max(1, min(n_req, span // min_win))
    if separated and n_eff > 1:
        starts = np.linspace(tmin, tmax + 1 - min_win, n_eff).astype(int)
        windows = [(int(s), int(s + min_win)) for s in starts]
    else:
        edges = np.linspace(tmin, tmax + 1, n_eff + 1).astype(int)
        windows = [(int(edges[i]), int(edges[i + 1])) for i in range(n_eff)]
    return windows, int(min_win), int(n_eff)


def _cohorts_synth(df, windows, train_frac, maxN):
    cohorts = {}
    for wi, (w0, w1) in enumerate(windows):
        cut = w0 + int(train_frac * (w1 - w0))
        tr = df[(df["t_step"] >= w0) & (df["t_step"] < cut)]
        peak = tr.groupby("service")["demand"].max().sort_values(ascending=False)
        cohorts[wi] = set(peak.index[:maxN])
    return cohorts


def load_phase2(cfg):
    """Leakage-safe multi-window loader. Returns (df, dt, windows, cohorts, info).
    `df` holds exact per-(service,t_step) demand for the union of per-window
    cohorts; `cohorts[w]` is the set of services selected from window w's own
    training period."""
    dt = cfg["data"]["dt_seconds"]
    maxN = int(cfg["data"]["max_services"])
    vf = cfg["split"]["val_fraction"]; tf = cfg["split"]["test_fraction"]
    train_frac = 1.0 - vf - tf

    if cfg["data"]["mode"] == "synthetic":
        df, dt = _load_synth(cfg)
        tmin, tmax = int(df["t_step"].min()), int(df["t_step"].max())
        windows, min_win, n_eff = _window_bounds(tmin, tmax, cfg)
        cohorts = _cohorts_synth(df, windows, train_frac, maxN)
    else:
        files = _find_files(cfg)
        # pass 0: global t_step range
        tmin, tmax = None, None
        for p in files:
            for ch in _iter(p, MSRES_COLS):
                st = (ch["timestamp"] / 1000.0 / dt).astype(int)
                lo, hi = int(st.min()), int(st.max())
                tmin = lo if tmin is None else min(tmin, lo)
                tmax = hi if tmax is None else max(tmax, hi)
        windows, min_win, n_eff = _window_bounds(tmin, tmax, cfg)
        cuts = [(w0, w0 + int(train_frac * (w1 - w0)), w1) for (w0, w1) in windows]
        # timestamp assertion: each window's cohort is selected strictly from its
        # training period, which ends before that window's validation/test region.
        for (w0, ct, w1) in cuts:
            val_start = w0 + int(train_frac * (w1 - w0))
            assert ct <= val_start, (
                f"cohort cutoff {ct} must not reach window {w0}-{w1} val/test "
                f"(starts at {val_start})")
        # pass 1: EXACT per-(window, service, minute) totals over TRAINING periods only
        tparts = []
        for p in files:
            for ch in _iter(p, MSRES_COLS):
                ch = ch.assign(t_step=(ch["timestamp"] / 1000.0 / dt).astype(int))
                for wi, (w0, ct, w1) in enumerate(cuts):
                    sub = ch[(ch["t_step"] >= w0) & (ch["t_step"] < ct)]
                    if len(sub):
                        gg = sub.groupby(["msname", "t_step"], as_index=False)["cpu"].sum()
                        gg["w"] = wi
                        tparts.append(gg)
        cohorts = {}
        if tparts:
            peak = _exact_peak(tparts)                     # keys: w, msname -> peak cpu
            for wi in range(len(windows)):
                pw = peak[peak["w"] == wi].sort_values("cpu", ascending=False)
                cohorts[wi] = set(pw["msname"].head(maxN))
        else:
            cohorts = {wi: set() for wi in range(len(windows))}
        # pass 2: EXACT demand for the union of per-window cohorts, over the window span
        union = set().union(*cohorts.values()) if cohorts else set()
        wlo = min(w0 for w0, _ in windows); whi = max(w1 for _, w1 in windows)
        dparts = []
        for p in files:
            for ch in _iter(p, MSRES_COLS):
                ch = ch[ch["msname"].isin(union)]
                if not len(ch):
                    continue
                ch = ch.assign(t_step=(ch["timestamp"] / 1000.0 / dt).astype(int))
                ch = ch[(ch["t_step"] >= wlo) & (ch["t_step"] < whi)]
                if len(ch):
                    dparts.append(ch.groupby(["msname", "t_step"])
                                   .agg(demand=("cpu", "sum"), mem=("mem", "sum")).reset_index())
        if dparts:
            df = (pd.concat(dparts, ignore_index=True).groupby(["msname", "t_step"])
                    .agg(demand=("demand", "sum"), mem=("mem", "sum")).reset_index())
        else:
            df = pd.DataFrame(columns=["msname", "t_step", "demand", "mem"])
        df = df.rename(columns={"msname": "service"})[["service", "t_step", "demand", "mem"]]

    # cohort turnover (auditable)
    present = [cohorts[wi] for wi in sorted(cohorts) if cohorts[wi]]
    union_n = len(set().union(*present)) if present else 0
    inter_n = len(set.intersection(*present)) if len(present) == len(windows) and present else 0
    info = dict(tmin=int(tmin), tmax=int(tmax), windows=windows, min_win=min_win,
                n_windows=n_eff, train_frac=train_frac,
                cohort_sizes={wi: len(cohorts[wi]) for wi in cohorts},
                cohort_union=union_n, cohort_intersection=inter_n,
                cohort_turnover=round(1.0 - inter_n / max(1, union_n), 3),
                cohorts={wi: sorted(cohorts[wi]) for wi in cohorts})
    return df, dt, windows, cohorts, info


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
        season = base + amp * (0.5 + 0.5 * np.sin(2 * np.pi * t / day + phase))
        noise = np.zeros(n_steps); e = 0.0
        for i in range(n_steps):
            e = 0.8 * e + rng.normal(0, 0.15 * base); noise[i] = e
        dem = np.clip(season + noise, 0.05, None)
        for _ in range(rng.integers(2, 6)):
            c = rng.integers(0, n_steps); w = rng.uniform(2, 6)
            dem += rng.uniform(0.5, 1.5) * base * np.exp(-((t - c) ** 2) / (2 * w ** 2))
        for i in range(n_steps):
            rows.append((f"svc_{s:03d}", int(t[i]), float(dem[i]), float(dem[i] * 0.3)))
    g = pd.DataFrame(rows, columns=["service", "t_step", "demand", "mem"])
    return g, dt


def load(cfg):
    return _load_synth(cfg) if cfg["data"]["mode"] == "synthetic" else _load_real(cfg)
