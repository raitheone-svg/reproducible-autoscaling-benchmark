"""Forecasting methods. Each returns a per-sample forecast aligned to the
feature bundle's rows. Classical methods are dependency-free; GBDT uses
scikit-learn; LSTM uses PyTorch and is skipped automatically if torch is absent.
"""
import numpy as np


# ---------------- classical (no training) ----------------
def reactive(b):
    """Reactive HPA under a scaling delay of H steps: a decision made at t-H to
    'match the load I see now' only takes effect at t, so capacity tracks
    demand[t-H]. It cannot anticipate ramps within the delay window."""
    H = b["H"]
    return np.array([b["series"][s][p - H] for s, p in zip(b["svc"], b["pos"])])


def moving_average(b):
    H, W = b["H"], b["W"]
    out = []
    for s, p in zip(b["svc"], b["pos"]):
        d = b["series"][s]
        out.append(d[p - H - W + 1: p - H + 1].mean())
    return np.asarray(out)


def _holt(d, alpha=0.5, beta=0.3):
    n = len(d); level = np.empty(n); trend = np.empty(n)
    level[0] = d[0]; trend[0] = 0.0
    for i in range(1, n):
        level[i] = alpha * d[i] + (1 - alpha) * (level[i - 1] + trend[i - 1])
        trend[i] = beta * (level[i] - level[i - 1]) + (1 - beta) * trend[i - 1]
    return level, trend


def holt(b):
    H = b["H"]
    cache = {}
    out = []
    for s, p in zip(b["svc"], b["pos"]):
        if s not in cache:
            cache[s] = _holt(b["series"][s])
        level, trend = cache[s]
        out.append(max(0.0, level[p - H] + H * trend[p - H]))
    return np.asarray(out)


# ---------------- learned ----------------
def gbdt(b, tr):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.08,
                                      max_iter=300, l2_regularization=1.0)
    m.fit(b["X"][tr], b["y"][tr])
    return np.clip(m.predict(b["X"]), 0, None)


def _sequences(b):
    H, W = b["H"], b["W"]
    seq = np.empty((len(b["pos"]), W), dtype=np.float32)
    for r, (s, p) in enumerate(zip(b["svc"], b["pos"])):
        seq[r] = b["series"][s][p - H - W + 1: p - H + 1]
    return seq


def lstm(b, tr, seed=7):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    seq = _sequences(b)
    sc = seq[tr].std() + 1e-6; mu = seq[tr].mean()
    xs = (seq - mu) / sc
    y = b["y"]; ysc = y[tr].std() + 1e-6; ymu = y[tr].mean()

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.lstm = nn.LSTM(1, 32, batch_first=True)
            s.head = nn.Linear(32, 1)
        def forward(s, x):
            o, _ = s.lstm(x.unsqueeze(-1))
            return s.head(o[:, -1]).squeeze(-1)

    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    lossf = nn.L1Loss()
    xt = torch.tensor(xs[tr]); yt = torch.tensor(((y[tr] - ymu) / ysc).astype(np.float32))
    for _ in range(80):
        opt.zero_grad(); loss = lossf(net(xt), yt); loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(torch.tensor(xs)).numpy() * ysc + ymu
    return np.clip(pred, 0, None)


REGISTRY = {"reactive": reactive,
            "moving_average": moving_average, "holt": holt}
LEARNED = {"gbdt": gbdt, "lstm": lstm}


def run(b, tr, which):
    out = {}
    for name in which:
        if name in REGISTRY:
            out[name] = REGISTRY[name](b)
        elif name == "gbdt":
            out[name] = gbdt(b, tr)
        elif name == "lstm":
            try:
                out[name] = lstm(b, tr)
            except Exception as e:
                print(f"      [lstm skipped: {type(e).__name__} — install torch to enable]")
    return out
