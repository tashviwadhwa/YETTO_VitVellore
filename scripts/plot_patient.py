"""Plot both data streams for one virtual patient, to check the data looks real.

Usage:
    python scripts/plot_patient.py GT-0012 --days 2 --start-day 3
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glucotwin import config as C

STAGE_Y = {"awake": 3, "rem": 2, "light": 1, "deep": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patient_id")
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--start-day", type=int, default=3)
    a = ap.parse_args()

    t0 = pd.Timestamp(C.START_DATE) + pd.Timedelta(days=a.start_day)
    t1 = t0 + pd.Timedelta(days=a.days)

    def load(path, col="timestamp"):
        df = pd.read_csv(path, parse_dates=[col])
        return df[(df.patient_id == a.patient_id) & (df[col] >= t0) & (df[col] < t1)]

    cgm = load(C.WEARABLE_DIR / "cgm.csv.gz")
    watch = load(C.WEARABLE_DIR / "smartwatch.csv.gz")
    meals = load(C.LOG_DIR / "meals.csv")
    doses = load(C.LOG_DIR / "medications.csv")
    pat = pd.read_csv(C.EHR_DIR / "patients.csv").set_index("patient_id").loc[a.patient_id]
    meds = pd.read_csv(C.EHR_DIR / "medications.csv")
    meds = meds[meds.patient_id == a.patient_id].drug.tolist()
    labs = pd.read_csv(C.EHR_DIR / "observations.csv")
    a1c = labs[(labs.patient_id == a.patient_id) & (labs.test == "HbA1c")].value.iloc[0]

    fig, ax = plt.subplots(5, 1, figsize=(13, 11), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 1.2, 1.2, 1.2, 1]))
    title = (f"{a.patient_id} | {pat['name']}, {pat.age}{pat.sex}, {pat.region} India | "
             f"BMI {pat.bmi} | T2D {pat.diabetes_duration_yrs} yrs | HbA1c {a1c}%\n"
             f"Medicines: {', '.join(meds)}")
    fig.suptitle(title, fontsize=11, x=0.01, ha="left")

    ax[0].axhspan(70, 180, color="#2e7d32", alpha=0.08)
    ax[0].axhline(180, color="#c62828", ls="--", lw=0.8)
    ax[0].axhline(70, color="#ef6c00", ls="--", lw=0.8)
    ax[0].plot(cgm.timestamp, cgm.glucose_mgdl, color="#1565c0", lw=1.2)
    for _, m in meals.iterrows():
        ax[0].annotate(f"{m.food}\n{m.carbs_g:.0f} g", (m.timestamp, 45), fontsize=6.5,
                       ha="center", color="#5d4037")
        ax[0].axvline(m.timestamp, color="#8d6e63", lw=0.6, alpha=0.6)
    for _, d in doses[doses.taken & doses.drug_class.isin(["sulfonylurea", "basal_insulin"])].iterrows():
        ax[0].scatter(d.timestamp, 390, marker="v", color="#6a1b9a", s=25)
    ax[0].set_ylim(35, 405); ax[0].set_ylabel("CGM glucose\n(mg/dL)")

    ax[1].plot(watch.timestamp, watch.heart_rate_bpm, color="#c62828", lw=0.8)
    ax[1].set_ylabel("Heart rate\n(bpm)")
    ax[2].plot(watch.timestamp, watch.hrv_rmssd_ms, ".", color="#00897b", ms=2)
    ax[2].set_ylabel("HRV RMSSD\n(ms)")
    ax[3].bar(watch.timestamp, watch.steps, width=0.0035, color="#455a64")
    ax[3].set_ylabel("Steps\n(per 5 min)")
    ax[4].step(watch.timestamp, watch.sleep_stage.map(STAGE_Y), where="post", color="#3949ab", lw=0.8)
    ax[4].set_yticks([0, 1, 2, 3], ["deep", "light", "REM", "awake"])
    ax[4].set_ylabel("Sleep")
    ax[4].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    for x in ax:
        x.grid(alpha=0.25)
    fig.tight_layout()
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = C.FIG_DIR / f"patient_{a.patient_id}.png"
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
