"""Daily life of each virtual patient: Indian meals, walking, sleep and medicine doses.

Everything is produced at 1-minute resolution so the physiology simulator
can react to it. Meals and medicine doses are also written to event logs,
like the food/medication diary in a diabetes app (e.g. Apple Health,
Google Fit, or a CGM companion app).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ehr_generator import DRUGS

# name -> (carbohydrate grams, absorption speed: higher = faster spike)
# Absorption speed is the gut absorption rate (1/min); white rice and idli
# are high glycaemic index (fast), millets, dal and oats are slower.
MENU = {
    "breakfast": {
        "South": [("Idli (4) with sambar", 62, 0.050), ("Masala dosa", 58, 0.045),
                  ("Pongal", 55, 0.048), ("Upma", 50, 0.042), ("Ragi dosa", 45, 0.030)],
        "North": [("Aloo paratha with curd", 66, 0.038), ("Poha", 54, 0.046),
                  ("Bread omelette", 36, 0.040), ("Besan chilla", 34, 0.030), ("Oats porridge", 40, 0.028)],
    },
    "lunch": {
        "South": [("Rice, sambar and poriyal", 92, 0.048), ("Curd rice", 76, 0.046),
                  ("Lemon rice", 80, 0.050), ("Chicken biryani", 96, 0.045), ("Millet meals", 62, 0.030)],
        "North": [("Roti (3), dal and sabzi", 72, 0.034), ("Rajma chawal", 86, 0.040),
                  ("Chole with rice", 88, 0.042), ("Veg pulao with raita", 82, 0.044), ("Roti (2) with paneer", 52, 0.030)],
    },
    "snack": {
        "any": [("Masala chai with biscuits", 26, 0.060), ("Samosa", 32, 0.045),
                ("Fruit (banana)", 24, 0.040), ("Sprouts chaat", 16, 0.025), ("Murukku", 28, 0.050)],
    },
    "dinner": {
        "South": [("Chapati (2) with kurma", 54, 0.034), ("Dosa (2) with chutney", 56, 0.045),
                  ("Rice with rasam", 78, 0.048), ("Idiyappam with stew", 60, 0.046)],
        "North": [("Roti (3) with dal", 66, 0.034), ("Khichdi", 64, 0.036),
                  ("Rice with kadhi", 76, 0.044), ("Roti (2) with chicken curry", 46, 0.030)],
    },
    "sweet": {
        "any": [("Gulab jamun (2)", 34, 0.065), ("Jalebi", 40, 0.070), ("Payasam", 38, 0.055)],
    },
}

# Menu speeds are relative glycaemic indices; this converts them to a gut
# absorption rate giving a realistic post-meal peak at ~45-75 minutes.
KA_SCALE = 0.45

STAGE = {"awake": 0, "light": 1, "deep": 2, "rem": 3}


def _region_key(region: str) -> str:
    return "South" if region in ("South", "East") else "North"


def _sleep_night(rng, minutes, onset, wake, stage_arr, asleep_arr):
    """Fill one night with 90-minute sleep cycles (light -> deep -> light -> REM)."""
    t = onset
    cycle = 0
    deep_len = [32, 26, 16, 8, 4, 0, 0]
    rem_len = [8, 14, 20, 26, 30, 30, 30]
    while t < wake:
        seq = [("light", int(rng.uniform(18, 30))),
               ("deep", max(0, int(deep_len[min(cycle, 6)] + rng.normal(0, 4)))),
               ("light", int(rng.uniform(8, 16))),
               ("rem", max(4, int(rem_len[min(cycle, 6)] + rng.normal(0, 4))))]
        if rng.random() < 0.6:
            seq.append(("awake", int(rng.uniform(3, 15))))
        for stage, length in seq:
            end = min(t + length, wake, minutes)
            if t >= minutes:
                return
            stage_arr[t:end] = STAGE[stage]
            asleep_arr[t:end] = stage != "awake"
            t = end
            if t >= wake:
                return
        cycle += 1


def generate_lifestyle(patients: pd.DataFrame, twins: pd.DataFrame, meds: pd.DataFrame,
                       n_days: int, rng: np.random.Generator, start_date: str):
    """Return 1-minute input arrays for the simulator plus event logs."""
    n = len(patients)
    T = n_days * 1440
    carbs = np.zeros((n, T))            # grams eaten in that minute
    kabs = np.full((n, T), 0.04 * KA_SCALE)        # gut absorption rate in force
    steps = np.zeros((n, T))            # steps in that minute
    stage = np.zeros((n, T), dtype=np.int8)
    asleep = np.zeros((n, T), dtype=bool)
    dose = {c: np.zeros((n, T)) for c in ["metformin", "sulfonylurea", "dpp4",
                                           "sglt2", "basal_insulin"]}
    sleep_hours = np.zeros((n, n_days))  # sleep obtained the night ending on day d

    start = pd.Timestamp(start_date)
    meal_log, med_log, walk_log = [], [], []
    regimen_by_pid = meds.groupby("patient_id")["drug_class"].apply(set).to_dict()

    for k, p in patients.iterrows():
        pid, region = p.patient_id, _region_key(p.region)
        adherence = twins.loc[k, "adherence"]
        regimen = regimen_by_pid.get(pid, set())

        # Personal habits
        base_steps = float(np.clip(rng.lognormal(np.log(5200), 0.4), 1500, 14000))
        morning_walker = rng.random() < 0.35
        post_meal_walk_p = float(np.clip(rng.beta(1.5, 4), 0, 0.8))
        bedtime_mean = rng.normal(23.1, 0.6)       # hours
        sleep_need = rng.normal(6.8, 0.6)
        portion = float(np.clip(rng.normal(1.0, 0.15), 0.7, 1.4))
        sweet_tooth = float(np.clip(rng.beta(2, 8), 0, 0.6))
        times = dict(breakfast=rng.normal(8.2, 0.4), lunch=rng.normal(13.3, 0.4),
                     snack=rng.normal(17.0, 0.4), dinner=rng.normal(20.9, 0.5))

        for d in range(n_days):
            day0 = d * 1440
            weekend = (start + pd.Timedelta(days=d)).dayofweek >= 5

            # ---- Sleep for the night starting this evening -------------------
            bt = bedtime_mean + rng.normal(0, 0.5) + (0.6 if weekend else 0)
            dur = np.clip(sleep_need + rng.normal(0, 0.7) - (1.8 if rng.random() < 0.12 else 0), 3.5, 9.5)
            onset = int(day0 + bt * 60)
            wake = int(onset + dur * 60)
            _sleep_night(rng, T, onset, wake, stage[k], asleep[k])
            if d + 1 < n_days:
                sleep_hours[k, d + 1] = dur
            # first night (before the recording starts): assume usual sleep
            if d == 0:
                sleep_hours[k, 0] = sleep_need + rng.normal(0, 0.5)
                prev_wake = int(day0 + (bt + dur - 24) * 60)
                stage[k, :max(prev_wake, 0)] = STAGE["light"]
                asleep[k, :max(prev_wake, 0)] = True

            # ---- Meals --------------------------------------------------------
            for meal in ["breakfast", "lunch", "snack", "dinner"]:
                if meal == "breakfast" and rng.random() < 0.08:
                    continue                      # skipped breakfast
                if meal == "lunch" and rng.random() < 0.04:
                    continue
                if meal == "snack" and rng.random() < 0.4:
                    continue
                opts = MENU[meal].get(region, MENU[meal].get("any"))
                name, c, ka = opts[rng.integers(len(opts))]
                g = c * portion * rng.uniform(0.85, 1.2) * (1.1 if weekend else 1.0)
                t = int(day0 + (times[meal] + rng.normal(0, 0.35) + (0.5 if weekend else 0)) * 60)
                if 0 <= t < T:
                    carbs[k, t] += g
                    kabs[k, t:t + 300] = ka * KA_SCALE * rng.uniform(0.9, 1.1)
                    meal_log.append(dict(patient_id=pid, timestamp=start + pd.Timedelta(minutes=t),
                                         meal=meal, food=name, carbs_g=round(g, 1)))
                    # post-meal walk
                    if meal in ("lunch", "dinner") and rng.random() < post_meal_walk_p:
                        w0, wl = t + int(rng.uniform(10, 30)), int(rng.uniform(10, 30))
                        steps[k, w0:w0 + wl] += rng.normal(100, 8, size=len(steps[k, w0:w0 + wl])).clip(60)
                        walk_log.append(dict(patient_id=pid, timestamp=start + pd.Timedelta(minutes=w0),
                                             activity="post-meal walk", minutes=wl))
            if rng.random() < sweet_tooth:
                name, c, ka = MENU["sweet"]["any"][rng.integers(3)]
                t = int(day0 + rng.uniform(16, 21.5) * 60)
                carbs[k, t] += c
                kabs[k, t:t + 240] = ka * KA_SCALE
                meal_log.append(dict(patient_id=pid, timestamp=start + pd.Timedelta(minutes=t),
                                     meal="sweet", food=name, carbs_g=float(c)))

            # ---- Background activity while awake -----------------------------
            awake_idx = np.arange(day0, min(day0 + 1440, T))
            awake_idx = awake_idx[~asleep[k, awake_idx]]
            day_target = base_steps * rng.lognormal(0, 0.25) * (0.8 if weekend else 1.0)
            # bursty movement: active 15-min blocks
            level = rng.gamma(0.6, 1.0, size=len(awake_idx) // 15 + 1).repeat(15)[:len(awake_idx)]
            moving = rng.random(len(awake_idx)) < 0.35
            raw = level * moving * rng.uniform(5, 60, size=len(awake_idx))
            if raw.sum() > 0:
                raw *= day_target / raw.sum()
            steps[k, awake_idx] += np.minimum(raw, 130)

            if morning_walker and rng.random() < 0.75:
                w0 = int(day0 + rng.normal(6.6, 0.3) * 60)
                wl = int(rng.uniform(25, 45))
                steps[k, w0:w0 + wl] += rng.normal(108, 8, size=wl).clip(70)
                walk_log.append(dict(patient_id=pid, timestamp=start + pd.Timedelta(minutes=w0),
                                     activity="morning walk", minutes=wl))

            # ---- Medicines ----------------------------------------------------
            for cls in dose:
                if cls not in regimen:
                    continue
                for h in DRUGS[cls][2]:
                    t = int(day0 + (h + rng.normal(0, 0.3)) * 60)
                    taken = rng.random() < adherence
                    if taken and 0 <= t < T:
                        dose[cls][k, t] = 1.0
                    med_log.append(dict(patient_id=pid, timestamp=start + pd.Timedelta(minutes=max(t, 0)),
                                        drug_class=cls, drug=DRUGS[cls][0], taken=taken))

    steps = np.clip(steps, 0, 140)
    steps[asleep] = 0
    logs = dict(meals=pd.DataFrame(meal_log), medications=pd.DataFrame(med_log),
                activity=pd.DataFrame(walk_log))
    inputs = dict(carbs=carbs, kabs=kabs, steps=steps, stage=stage, asleep=asleep,
                  dose=dose, sleep_hours=sleep_hours)
    return inputs, logs
