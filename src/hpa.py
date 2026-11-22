"""Replica-based reactive autoscaler that resembles the Kubernetes Horizontal
Pod Autoscaler (HPA v2), for use as a baseline in the benchmark.

This is still an OFFLINE replay -- no cluster -- but unlike the continuous
delayed-persistence baseline it actuates in INTEGER replicas under the real HPA
control constraints: a utilisation target, the tolerance band, min/max replicas,
per-direction scale-rate limits, and a downscale stabilisation window. Capacity
provisioned at step t is replicas(t) * per_replica_capacity, in the same CPU
units as demand, so its cost and shortfall are directly comparable to the other
policies. Decisions use demand observed one scaling delay ago (d[t-H]).

The utilisation target plays the role the headroom sweep plays for the other
methods: a lower target provisions more and is safer, so sweeping it traces an
SLO--cost curve that the validation-frozen protocol selects an operating point
from. Integer replicas, tolerance, and the downscale lag make HPA provision
somewhat MORE than the idealised continuous baseline at equal reliability -- the
overhead the continuous abstraction hides.

HPA v2 control law: desiredReplicas = ceil(currentReplicas * metric/target),
with the tolerance band, bounds, rate limits, and downscale stabilisation
(defaults match Kubernetes ~1.27; pin and cite the version you emulate).
"""
import numpy as np


def replica_trajectory(demand, H, per_replica, target_util,
                       min_rep=1, max_rep=100000, tol=0.10,
                       up_rate=4, down_rate=2, stab=3):
    """Return (capacity[T], scaling_actions) for one service's demand series."""
    T = len(demand)
    rep = np.empty(T, dtype=int)
    cur = max(min_rep, 1)
    desired_hist = []
    actions = 0
    denom = per_replica * target_util + 1e-9
    for t in range(T):
        obs = demand[t - H] if t >= H else demand[t]      # observed one delay ago
        desired_raw = obs / denom
        ratio = desired_raw / max(cur, 1)
        desired = cur if abs(ratio - 1.0) <= tol else int(np.ceil(desired_raw))
        desired = int(np.clip(desired, min_rep, max_rep))
        desired_hist.append(desired)
        # downscale stabilisation: don't drop below the max desired over the window
        if desired < cur:
            stabilized = max(desired_hist[-stab:])
            desired = min(cur, stabilized) if stabilized < cur else cur
        # per-direction rate limits
        if desired > cur:
            desired = min(desired, cur + up_rate)
        elif desired < cur:
            desired = max(desired, cur - down_rate)
        if desired != cur:
            actions += 1
        cur = desired
        rep[t] = cur
    return rep.astype(float) * per_replica, actions


def evaluate_rows(bundle, H, per_replica, target_util, hpa_cfg):
    """Map the per-service replica trajectories onto the sample rows of `bundle`.
    Returns (capacity_per_row[N], mean_actions_per_service)."""
    series = bundle["series"]
    traj, acts = {}, {}
    for s, d in series.items():
        cap, a = replica_trajectory(d, H, per_replica, target_util,
                                    min_rep=hpa_cfg.get("min_replicas", 1),
                                    max_rep=hpa_cfg.get("max_replicas", 100000),
                                    tol=hpa_cfg.get("tolerance", 0.10),
                                    up_rate=hpa_cfg.get("scale_up_max_per_step", 4),
                                    down_rate=hpa_cfg.get("scale_down_max_per_step", 2),
                                    stab=hpa_cfg.get("downscale_stabilization_steps", 3))
        traj[s] = cap
        acts[s] = a
    cap_row = np.array([traj[s][p] for s, p in zip(bundle["svc"], bundle["pos"])])
    return cap_row, float(np.mean(list(acts.values()))) if acts else 0.0


def derive_per_replica(bundle, target_replicas=20.0):
    """A documented modelling choice: size one replica so the average busy
    service runs ~`target_replicas` replicas. per_replica = mean demand /
    target_replicas. Report this value; the HPA cost scales with it."""
    alld = np.concatenate([bundle["series"][s] for s in bundle["series"]])
    return max(1e-6, float(alld.mean()) / target_replicas)
