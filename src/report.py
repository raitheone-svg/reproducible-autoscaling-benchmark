"""Write results.json, tables.md, and figures (Pareto + accuracy)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SLO_TARGET = 0.05            # headline: cost to hold SLO violations <= 5%
COLORS = ["#B2182B", "#8c6bb1", "#1b7837", "#2166AC", "#d95f0e", "#444444"]


def _cost_at_target(curve, target=SLO_TARGET):
    ok = [p for p in curve if p["slo"] <= target]
    if ok:
        best = min(ok, key=lambda p: p["cost"])
        return best["cost"], best["headroom"], best["slo"]
    best = min(curve, key=lambda p: p["slo"])     # none reach target
    return best["cost"], best["headroom"], best["slo"]


def _order(results):
    pref = ["reactive", "persistence", "moving_average", "holt", "gbdt", "lstm"]
    return [m for m in pref if m in results] + [m for m in results if m not in pref]


DISP = {"reactive": "Delayed persistence", "persistence": "Persistence",
        "moving_average": "Moving average", "holt": "Holt (ES)",
        "gbdt": "GBDT", "lstm": "LSTM"}


def write(out_dir, results, cfg, provenance, manifest=None):
    os.makedirs(out_dir, exist_ok=True)
    order = _order(results); disp = DISP

    # ---- Pareto figure (RETROSPECTIVE test frontier) ----
    plt.rcParams.update({"font.family": "DejaVu Serif"})
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for i, m in enumerate(order):
        cur = sorted(results[m]["pareto_test"], key=lambda p: p["cost"])
        ax.plot([p["cost"] for p in cur], [p["slo"] * 100 for p in cur],
                "-o", ms=3, lw=1.3, color=COLORS[i % len(COLORS)], label=disp.get(m, m))
    ax.axhline(SLO_TARGET * 100, ls=":", color="#888", lw=1)
    ax.text(ax.get_xlim()[1], SLO_TARGET * 100, " 5% target", fontsize=7, va="bottom", ha="right", color="#888")
    ax.set_xlabel("Mean provisioned capacity"); ax.set_ylabel("CPU under-provisioning rate (%)")
    ax.set_title("Retrospective SLO–cost frontier (test)", fontsize=10)
    ax.legend(fontsize=7, frameon=False); ax.grid(True, lw=0.4, alpha=0.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "pareto.png"), dpi=200); plt.close(fig)

    # ---- forecast accuracy figure ----
    fig, ax = plt.subplots(figsize=(5, 3))
    smape = [results[m]["accuracy"]["smape"] for m in order]
    ax.bar([disp.get(m, m) for m in order], smape,
           color=[COLORS[i % len(COLORS)] for i in range(len(order))])
    ax.set_ylabel("sMAPE (%)"); ax.set_title("Forecast error (lower is better)", fontsize=10)
    plt.xticks(rotation=25, ha="right", fontsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "accuracy.png"), dpi=200); plt.close(fig)

    # ---- results.json ----
    summary = {"provenance": provenance, "slo_target": SLO_TARGET,
               "protocol": "validation-frozen headroom; test reported",
               "manifest": manifest or {}, "methods": {}}
    for m in order:
        r = results[m]
        summary["methods"][m] = {
            "display": disp.get(m, m), "accuracy": r["accuracy"],
            "frozen_headroom": r["frozen_headroom"], "frozen_val_slo": round(r["frozen_val_slo"], 4),
            "test_cost": round(r["test_cost"], 4), "test_slo": round(r["test_slo"], 4),
            "test_overprov_pct": r["test_overprov_pct"],
            "pareto_test": r["pareto_test"], "pareto_val": r["pareto_val"]}
    json.dump(summary, open(os.path.join(out_dir, "results.json"), "w"), indent=2)
    if manifest:
        json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)

    # ---- tables.md ----
    base = summary["methods"].get("reactive", {}).get("test_cost")
    with open(os.path.join(out_dir, "tables.md"), "w") as fh:
        fh.write(f"Provenance: {provenance}\nProtocol: validation-frozen headroom; test reported.\n\n")
        fh.write("Table I - forecast accuracy (test split; computed before headroom)\n")
        fh.write("method | MAE | RMSE | sMAPE%\n")
        for m in order:
            a = results[m]["accuracy"]
            fh.write(f"{disp.get(m,m)} | {a['mae']} | {a['rmse']} | {a['smape']}\n")
        fh.write(f"\nTable II - validation-frozen deployment result (target <= {int(SLO_TARGET*100)}% under-provisioning)\n")
        fh.write("method | test capacity | headroom | test under-prov % | vs delayed persistence\n")
        for m in order:
            mm = summary["methods"][m]
            sv = f"{(1 - mm['test_cost']/base)*100:+.1f}%" if base else "-"
            fh.write(f"{mm['display']} | {mm['test_cost']} | {mm['frozen_headroom']} "
                     f"| {round(mm['test_slo']*100,1)} | {sv}\n")
    return summary
