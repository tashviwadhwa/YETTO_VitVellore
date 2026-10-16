"""What-if simulator: try a decision on the virtual patient before the real one.

A scenario describes what happens in the next few hours:
    meal        - which Indian dish, how big a portion, how many minutes from now
    walk        - minutes of walking, and when it starts
    glimepiride - taken as prescribed, skipped, or at a different dose

The calibrated twin simulates each scenario from the patient's current
internal state, so the answer is personal: the same plate of rice gives a
different curve for every patient.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import twin
from .lifestyle import KA_SCALE, MENU

DISHES = {name: (carbs, ka) for meal in MENU.values() for opts in meal.values()
          for name, carbs, ka in opts}
STEPS_PER_MIN_WALK = 100


@dataclass
class Scenario:
    name: str = "As planned"
    dish: str | None = None          # None = no new meal
    portion: float = 1.0
    meal_in_min: int = 0
    walk_min: int = 0
    walk_starts_in_min: int = 30
    glimepiride: str = "as prescribed"   # "as prescribed" | "skip next dose" | "half dose" | "double dose"


def simulate_scenarios(T: dict, patient: str, minute: int, scenarios: list[Scenario],
                       horizon: int = 360) -> dict:
    """Return {scenario name: glucose curve (horizon,)} plus the time axis."""
    k = int(np.where(T["pids"] == patient)[0][0])
    j = minute // 15
    n = len(scenarios)
    rows = np.full(n, k)
    start = np.full(n, minute)
    states = {key: np.full(n, T["states"][key][k, j]) for key in ["G", "X", "I", "Q1", "Q2", "load"]}

    carbs = np.zeros((n, horizon))
    kabs = np.repeat(T["inp"]["kabs"][k, min(minute, T["inp"]["kabs"].shape[1] - 1)], n * horizon).reshape(n, horizon)
    steps = np.zeros((n, horizon))
    theta = {key: v.copy() for key, v in T["theta"].items()}
    # one theta row per scenario (so the dose can differ by scenario)
    theta = {key: np.concatenate([v, np.repeat(v[k], n)]) for key, v in theta.items()}
    sc_rows = np.arange(len(T["pids"]), len(T["pids"]) + n)
    pri = T["pri"]
    pri_ext = pri.iloc[np.concatenate([np.arange(len(pri)), rows])].reset_index(drop=True)

    effects = {}
    for i, s in enumerate(scenarios):
        if s.dish:
            c, ka = DISHES[s.dish]
            o = int(np.clip(s.meal_in_min, 0, horizon - 1))
            carbs[i, o] = c * s.portion
            kabs[i, o:] = ka * KA_SCALE
        if s.walk_min > 0:
            o = int(np.clip(s.walk_starts_in_min, 0, horizon - 1))
            steps[i, o:o + s.walk_min] = STEPS_PER_MIN_WALK
        skip = ("sulfonylurea",) if s.glimepiride == "skip next dose" else ()
        e = twin.future_effects(T["inp"], rows[i:i + 1], start[i:i + 1], horizon, skip=skip)
        for cls, v in e.items():
            effects.setdefault(cls, []).append(v[0])
        gain = {"half dose": 0.5, "double dose": 2.0}.get(s.glimepiride, 1.0)
        theta["log_drug_gain"][sc_rows[i]] += np.log(gain)
    effects = {cls: np.vstack(v) for cls, v in effects.items()}

    sim = twin.forecast_from_states(theta, pri_ext, states, sc_rows, start, T["inp"],
                                    horizon=horizon, carbs=carbs, steps=steps, kabs=kabs,
                                    effects=effects,
                                    sleep_hours=np.repeat(T["inp"]["sleep_hours"][k:k + 1], n, 0))
    return {"minutes": np.arange(1, horizon + 1),
            "curves": {s.name: sim["G"][i] for i, s in enumerate(scenarios)}}


def summarise(curve: np.ndarray) -> dict:
    """Clinically meaningful numbers for one scenario curve (1-minute samples)."""
    return {"Peak glucose (mg/dL)": float(curve.max()),
            "Minutes above 180": int((curve > 180).sum()),
            "Lowest glucose (mg/dL)": float(curve.min()),
            "Minutes below 70": int((curve < 70).sum()),
            "Glucose after 2 h (mg/dL)": float(curve[min(119, len(curve) - 1)])}
