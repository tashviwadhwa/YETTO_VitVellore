"""Day 1: generate the full synthetic dataset for GlucoTwin.

Run from the project root:
    python scripts/01_generate_data.py

Creates:
    data/ehr/          static stream  - patients, conditions, labs, medications, genetics
    data/wearables/    dynamic stream - CGM glucose, smartwatch (HR, HRV, steps, sleep), nightly sleep
    data/logs/         app diary      - meals, medicine doses, walks
    data/ground_truth/ hidden twin parameters (for evaluation only, never used as model input)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glucotwin import config as C
from glucotwin.ehr_generator import generate_cohort, make_lab_observations
from glucotwin.lifestyle import generate_lifestyle
from glucotwin.physiology import cgm_readings, drug_effects, simulate
from glucotwin.wearables import STAGE_NAMES, nightly_sleep_summary, smartwatch_signals

PARAM_COLS = ["SI", "beta", "p1", "gb_untreated", "dawn_amp", "sleep_sensitivity",
              "body_weight_kg", "su_push", "basal_dose"]


def long_format(pids, start, values: dict, interval: int) -> pd.DataFrame:
    N, S = next(iter(values.values())).shape
    ts = pd.date_range(start, periods=S, freq=f"{interval}min")
    df = pd.DataFrame({"patient_id": np.repeat(pids, S), "timestamp": np.tile(ts, N)})
    for name, arr in values.items():
        df[name] = arr.reshape(-1)
    return df


def main():
    t0 = time.time()
    rng = np.random.default_rng(C.SEED)
    for d in [C.EHR_DIR, C.WEARABLE_DIR, C.LOG_DIR, C.TRUTH_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # One extra warm-up day lets the physiology settle; it is discarded.
    sim_start = pd.Timestamp(C.START_DATE) - pd.Timedelta(days=1)
    start = pd.Timestamp(C.START_DATE)
    n_sim_days = C.N_DAYS + 1

    print(f"[1/5] Creating {C.N_PATIENTS} synthetic Indian T2D patients...")
    patients, conditions, meds, genetics, twins = generate_cohort(C.N_PATIENTS, rng, C.START_DATE)

    print("[2/5] Simulating daily life (meals, walking, sleep, medicines)...")
    inputs, logs = generate_lifestyle(patients, twins, meds, n_sim_days, rng, str(sim_start.date()))

    print("[3/5] Running the glucose-insulin physiology for every patient...")
    T = inputs["carbs"].shape[1]
    effects = drug_effects(inputs["dose"], T)
    params = {c: twins[c].values.astype(float) for c in PARAM_COLS}
    sim = simulate(params, inputs["carbs"], inputs["kabs"], inputs["steps"],
                   inputs["sleep_hours"], effects)

    # Drop the warm-up day
    W = 1440
    G = sim["G"][:, W:]
    trimmed = dict(steps=inputs["steps"][:, W:], stage=inputs["stage"][:, W:],
                   asleep=inputs["asleep"][:, W:], sleep_hours=inputs["sleep_hours"][:, 1:])

    print("[4/5] Producing sensor readings (CGM + smartwatch)...")
    cgm = cgm_readings(G, rng, C.CGM_INTERVAL_MIN)
    watch = smartwatch_signals(patients, trimmed, G, rng, C.CGM_INTERVAL_MIN)
    pids = patients.patient_id.values

    cgm_df = long_format(pids, start, {"glucose_mgdl": cgm}, C.CGM_INTERVAL_MIN)
    watch_df = long_format(pids, start, {"heart_rate_bpm": watch["heart_rate"],
                                         "hrv_rmssd_ms": watch["hrv_rmssd"],
                                         "steps": watch["steps"]}, C.CGM_INTERVAL_MIN)
    watch_df["sleep_stage"] = STAGE_NAMES[watch["sleep_stage"].reshape(-1)]
    sleep_df = nightly_sleep_summary(pids, trimmed["stage"], watch["hrv_rmssd"],
                                     watch["heart_rate"], start)

    print("[5/5] Writing clinic lab results and saving files...")
    mean_g = G.mean(1)
    fasting_idx = np.arange(7 * 60, G.shape[1], 1440)
    fasting = G[:, fasting_idx].mean(1)
    labs = make_lab_observations(patients, twins, mean_g, fasting, rng, C.START_DATE)

    # Logs: keep only the recorded period
    for name, df in logs.items():
        df = df[df.timestamp >= start].sort_values(["patient_id", "timestamp"])
        df.to_csv(C.LOG_DIR / f"{name}.csv", index=False)

    patients.to_csv(C.EHR_DIR / "patients.csv", index=False)
    conditions.to_csv(C.EHR_DIR / "conditions.csv", index=False)
    labs.to_csv(C.EHR_DIR / "observations.csv", index=False)
    meds.to_csv(C.EHR_DIR / "medications.csv", index=False)
    genetics.to_csv(C.EHR_DIR / "genetics.csv", index=False)

    cgm_df.to_csv(C.WEARABLE_DIR / "cgm.csv.gz", index=False)
    watch_df.to_csv(C.WEARABLE_DIR / "smartwatch.csv.gz", index=False)
    sleep_df.to_csv(C.WEARABLE_DIR / "sleep_nightly.csv", index=False)

    truth = twins.copy()
    truth["resting_hr"] = watch["rhr"].round(1)
    truth["baseline_hrv"] = watch["hrv0"].round(1)
    truth["true_mean_glucose"] = mean_g.round(1)
    truth.to_csv(C.TRUTH_DIR / "twin_parameters.csv", index=False)
    np.savez_compressed(C.TRUTH_DIR / "true_plasma_glucose_1min.npz",
                        glucose=G.astype(np.float32), patient_ids=pids)

    # ---- Summary ----------------------------------------------------------
    a1c = labs[labs.test == "HbA1c"].groupby("patient_id").value.last()
    valid = cgm[~np.isnan(cgm)]
    print("\nDone in %.0f s" % (time.time() - t0))
    print(f"Patients: {len(patients)} | CGM readings: {len(cgm_df):,} | Smartwatch rows: {len(watch_df):,}")
    print(f"Meals logged: {len(logs['meals']):,} | Medicine doses logged: {len(logs['medications']):,}")
    print(f"HbA1c median {a1c.median():.1f}% (range {a1c.min():.1f}-{a1c.max():.1f})")
    print(f"Time in range 70-180: {((valid >= 70) & (valid <= 180)).mean():.1%} | "
          f"above 180: {(valid > 180).mean():.1%} | below 70: {(valid < 70).mean():.1%}")


if __name__ == "__main__":
    main()
