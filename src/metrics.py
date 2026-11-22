"""Forecast-accuracy and autoscaling-outcome metrics, evaluated on the test
split. For each method we report headroom-independent accuracy and a
headroom-swept SLO-vs-cost curve (the Pareto frontier).
"""
import numpy as np
import pandas as pd


def accuracy(y, yhat):
    e = yhat - y
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    smape = float(np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(yhat) + 1e-9)) * 100)
    return dict(mae=round(mae, 4), rmse=round(rmse, 4), smape=round(smape, 1))


def _actions(c, svc, tstep, deadband):
    """Average number of capacity changes per service over the test window."""
    dfm = pd.DataFrame({"c": c, "svc": svc, "t": tstep}).sort_values(["svc", "t"])
    total = 0; nsvc = 0
    for _, g in dfm.groupby("svc"):
        cc = g["c"].to_numpy()
        if len(cc) < 2:
            continue
        thr = deadband * (cc.mean() + 1e-9)
        total += int(np.sum(np.abs(np.diff(cc)) > thr)); nsvc += 1
    return total / max(1, nsvc)


def provision_point(y, yhat, headroom, svc, tstep, deadband):
    c = yhat * (1 + headroom); a = y
    slo = float(np.mean(c < a))                       # under-provision rate
    waste = float(np.mean(np.maximum(0, c - a)) / (np.mean(a) + 1e-9) * 100)
    cost = float(np.mean(c))                          # provisioned resource (lower better)
    return dict(headroom=headroom, slo=round(slo, 4), overprov_pct=round(waste, 1),
                cost=round(cost, 4), actions=round(_actions(c, svc, tstep, deadband), 2))


def _curve(bundle, yhat_all, mask, cfg):
    y = bundle["y"][mask]; svc = bundle["svc"][mask]
    tstep = bundle["meta"]["t_step"].to_numpy()[mask]
    db = cfg["provision"]["deadband"]
    return [provision_point(y, yhat_all[mask], h, svc, tstep, db)
            for h in cfg["provision"]["headrooms"]]


def evaluate(bundle, forecasts, va, te, cfg, slo_target=0.05):
    """Validation-frozen protocol: the operating headroom is chosen on the
    VALIDATION split (cheapest headroom achieving the SLO target there), frozen,
    and then the TEST cost and achieved violation at that headroom are reported.
    The full test curve is also returned but is a RETROSPECTIVE frontier."""
    tmeta = bundle["meta"]["t_step"].to_numpy()
    results = {}
    for name, yhat_all in forecasts.items():
        acc = accuracy(bundle["y"][te], yhat_all[te])
        test_curve = _curve(bundle, yhat_all, te, cfg)
        val_curve = _curve(bundle, yhat_all, va, cfg)
        # freeze headroom on validation
        feas = [p for p in val_curve if p["slo"] <= slo_target]
        chosen = (min(feas, key=lambda p: p["cost"]) if feas
                  else min(val_curve, key=lambda p: p["slo"]))
        h = chosen["headroom"]
        test_at = next(p for p in test_curve if p["headroom"] == h)
        results[name] = dict(
            accuracy=acc,
            frozen_headroom=h,
            frozen_val_slo=chosen["slo"],
            test_cost=test_at["cost"],
            test_slo=test_at["slo"],
            test_overprov_pct=test_at["overprov_pct"],
            pareto_test=test_curve,       # retrospective frontier (figure only)
            pareto_val=val_curve)
    return results
