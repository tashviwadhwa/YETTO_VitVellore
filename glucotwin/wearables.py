"""Smartwatch signals: heart rate, heart rate variability (HRV), steps and sleep stages.

These are not random numbers: they are coupled to the patient's body.
    * Heart rate rises with walking, falls in deep sleep, and jumps when
      glucose goes low (the adrenaline response to hypoglycaemia).
    * HRV (RMSSD) is lower in older patients, in those with nerve damage
      (diabetic autonomic neuropathy) or hypertension, after short sleep,
      and drops sharply during hypoglycaemia.
That coupling is what lets the ML model learn real early-warning patterns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STAGE_NAMES = np.array(["awake", "light", "deep", "rem"])


def _block(arr: np.ndarray, k: int, how: str = "mean") -> np.ndarray:
    N, T = arr.shape
    a = arr[:, : T // k * k].reshape(N, T // k, k)
    return a.sum(2) if how == "sum" else a.mean(2)


def smartwatch_signals(patients: pd.DataFrame, inputs: dict, G: np.ndarray,
                       rng: np.random.Generator, interval: int = 5):
    """Return 5-minute heart rate, HRV, steps and sleep stage arrays (N, S)."""
    steps = inputs["steps"]; stage = inputs["stage"]; asleep = inputs["asleep"]
    N, T = steps.shape
    n_days = T // 1440
    mean_steps = steps.sum(1) / n_days

    age = patients.age.values; bmi = patients.bmi.values
    htn = patients.hypertension.values.astype(float)
    neuro = patients.neuropathy.values.astype(float)
    smoker = patients.smoker.values.astype(float)

    # Resting heart rate: older, unfit, hypertensive, neuropathy -> higher
    rhr = (70 + 0.08 * (age - 50) + 3.5 * htn + 4 * neuro + 0.25 * (bmi - 26)
           + 2.5 * smoker - 2.5 * (mean_steps - 6000) / 3000 + rng.normal(0, 3.5, N))
    # Baseline HRV (RMSSD, ms): falls with age, neuropathy, hypertension
    hrv0 = np.clip(44 - 0.45 * (age - 40) - 8 * neuro - 3.5 * htn - 3 * smoker
                   + rng.normal(0, 5, N), 10, 80)

    # Sleep debt per minute (hours below 7 the previous night)
    day_idx = np.minimum(np.arange(T) // 1440, inputs["sleep_hours"].shape[1] - 1)
    debt = np.maximum(0, 7 - inputs["sleep_hours"][:, day_idx])

    stage_hr = np.array([0, -7, -11, -4])[stage]
    hypo = np.clip((72 - G) / 20, 0, 1.5)
    hyper = np.clip((G - 180) / 100, 0, 1.5)
    hour = (np.arange(T) % 1440) / 60
    circ = 3 * np.sin((hour - 9) / 24 * 2 * np.pi)

    # AR(1) physiological noise
    noise = np.empty((N, T)); e = rng.normal(0, 1, (N, T)); noise[:, 0] = e[:, 0]
    phi = 0.97
    for t in range(1, T):
        noise[:, t] = phi * noise[:, t - 1] + np.sqrt(1 - phi ** 2) * e[:, t]

    hr = (rhr[:, None] + 0.33 * steps + stage_hr + 13 * hypo + 3 * hyper
          + 1.2 * debt + np.where(asleep, 0, circ) + 3 * noise)
    hr = np.clip(hr, 40, 185)

    stage_hrv = np.array([1.0, 1.12, 1.35, 0.95])[stage]
    hrv = (hrv0[:, None] * stage_hrv * (1 - 0.06 * debt) * (1 - 0.30 * np.minimum(hypo, 1))
           * (1 - 0.08 * hyper) * np.exp(-steps / 120) * np.exp(0.12 * noise))
    hrv = np.clip(hrv, 5, 150)

    hr5 = _block(hr, interval).round(0)
    hrv5 = _block(hrv, interval).round(1)
    steps5 = _block(steps, interval, "sum").round(0)
    stage5 = stage[:, interval // 2::interval][:, : hr5.shape[1]]
    # Watches only report HRV when the wearer is still
    hrv5[steps5 > 20] = np.nan
    # Watch off the wrist (charging) for ~1 hour a day at a random time
    for k in range(N):
        for d in range(n_days):
            s0 = d * 288 + rng.integers(7 * 12, 21 * 12)
            hr5[k, s0:s0 + 12] = np.nan
            hrv5[k, s0:s0 + 12] = np.nan
    return dict(heart_rate=hr5, hrv_rmssd=hrv5, steps=steps5, sleep_stage=stage5,
                rhr=rhr, hrv0=hrv0)


def nightly_sleep_summary(patient_ids, stage: np.ndarray, hrv5: np.ndarray, hr5: np.ndarray,
                          start: pd.Timestamp, interval: int = 5) -> pd.DataFrame:
    """One row per night, like the sleep page of a fitness app.

    A "night" runs from 6 PM to noon the next day, and the summary is dated
    by the morning the patient woke up.
    """
    N, T = stage.shape
    rows = []
    n_days = T // 1440
    for k, pid in enumerate(patient_ids):
        for d in range(1, n_days):
            a, b = (d - 1) * 1440 + 18 * 60, d * 1440 + 12 * 60
            st = stage[k, a:b]
            asleep = st > 0
            if asleep.sum() == 0:
                continue
            idx = np.where(asleep)[0]
            onset, wake = idx[0], idx[-1]
            window = st[onset:wake + 1]
            s5a, s5b = (a + onset) // interval, (a + wake) // interval
            rows.append(dict(
                patient_id=pid, date=(start + pd.Timedelta(days=d)).date().isoformat(),
                sleep_onset=(start + pd.Timedelta(minutes=int(a + onset))).strftime("%Y-%m-%d %H:%M"),
                wake_time=(start + pd.Timedelta(minutes=int(a + wake))).strftime("%Y-%m-%d %H:%M"),
                total_sleep_min=int((window > 0).sum()),
                deep_min=int((window == 2).sum()), rem_min=int((window == 3).sum()),
                light_min=int((window == 1).sum()), awake_min=int((window == 0).sum()),
                sleep_efficiency=round(float((window > 0).mean()), 3),
                avg_sleep_hrv=round(float(np.nanmean(hrv5[k, s5a:s5b + 1])), 1),
                avg_sleep_hr=round(float(np.nanmean(hr5[k, s5a:s5b + 1])), 1)))
    return pd.DataFrame(rows)
