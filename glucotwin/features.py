"""Data fusion: turn the two data streams into one table the model can learn from.

Every 15 minutes, for every patient, we stand at a "prediction moment" and ask:
"Using only what is known RIGHT NOW, what happens in the next 2 hours?"

Each row = one patient at one moment, with three kinds of features:
    STATIC   - from the health record (age, HbA1c, medicines, genetics, ...)
    SENSOR   - from the CGM and smartwatch, summarised over the recent past
    DIARY    - from the app diary (meals eaten, medicine doses taken so far)

and the answers (labels), computed from the FUTURE CGM readings:
    spike_2h - glucose goes above 180 mg/dL (for 15+ min) within 2 hours
    hypo_2h  - glucose goes below 70 mg/dL (for 15+ min) within 2 hours
    glucose_+30/+60/+90/+120 - the actual glucose at those horizons

No future information is ever used to build features (no data leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

STEP = C.CGM_INTERVAL_MIN            # 5 minutes per sample
HORIZON = 120 // STEP                 # 24 samples = 2 hours
SPD = 1440 // STEP                    # samples per day
STRIDE = 3                            # a prediction moment every 15 min
MIN_RUN = 3                           # 3 readings in a row = 15 minutes


# ---------------------------------------------------------------------------
# Loading everything into (patients x time) arrays
# ---------------------------------------------------------------------------
def load_arrays():
    cgm = pd.read_csv(C.WEARABLE_DIR / "cgm.csv.gz", parse_dates=["timestamp"])
    watch = pd.read_csv(C.WEARABLE_DIR / "smartwatch.csv.gz", parse_dates=["timestamp"])
    pids = cgm.patient_id.unique()
    S = len(cgm) // len(pids)
    start = cgm.timestamp.min()

    def arr(df, col):
        return df[col].to_numpy(dtype=float).reshape(len(pids), S)

    A = dict(pids=pids, start=start, S=S,
             glucose=arr(cgm, "glucose_mgdl"),
             hr=arr(watch, "heart_rate_bpm"), hrv=arr(watch, "hrv_rmssd_ms"),
             steps=arr(watch, "steps"),
             asleep=(watch.sleep_stage.to_numpy() != "awake").reshape(len(pids), S).astype(float),
             deep=(watch.sleep_stage.to_numpy() == "deep").reshape(len(pids), S).astype(float))

    # Diary events placed into 5-minute slots
    idx = {p: i for i, p in enumerate(pids)}

    def slot(df):
        # Round UP: an event at 07:47 is only known from the 07:50 reading
        # onwards, so it must not appear in the 07:45 features (no leakage).
        secs = (df.timestamp - start).dt.total_seconds()
        s = np.ceil(secs / (STEP * 60)).astype(int)
        return df.patient_id.map(idx).to_numpy(), s.to_numpy()

    meals = pd.read_csv(C.LOG_DIR / "meals.csv", parse_dates=["timestamp"])
    meds = pd.read_csv(C.LOG_DIR / "medications.csv", parse_dates=["timestamp"])
    carbs = np.zeros((len(pids), S))
    i, s = slot(meals); ok = (s >= 0) & (s < S)
    np.add.at(carbs, (i[ok], s[ok]), meals.carbs_g.to_numpy()[ok])
    A["carbs"] = carbs
    for cls in ["sulfonylurea", "basal_insulin", "metformin"]:
        d = meds[(meds.drug_class == cls) & meds.taken]
        a = np.zeros((len(pids), S))
        i, s = slot(d); ok = (s >= 0) & (s < S)
        a[i[ok], s[ok]] = 1
        A[f"dose_{cls}"] = a
    return A


# ---------------------------------------------------------------------------
# Small helpers for "look back over the last k samples" with missing values
# ---------------------------------------------------------------------------
def _cum(x):
    v = np.nan_to_num(x, nan=0.0)
    c = np.isfinite(x).astype(float)
    z = np.zeros((x.shape[0], 1))
    return np.hstack([z, np.cumsum(v, 1)]), np.hstack([z, np.cumsum(c, 1)])


def back_mean(x, k):
    """Mean of x over samples (t-k, t], ignoring missing values."""
    cs, cc = _cum(x)
    S = x.shape[1]
    t = np.arange(S)
    lo = np.maximum(t + 1 - k, 0)
    s = cs[:, t + 1] - cs[:, lo]
    c = cc[:, t + 1] - cc[:, lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan)


def back_sum(x, k):
    cs, _ = _cum(x)
    t = np.arange(x.shape[1])
    return cs[:, t + 1] - cs[:, np.maximum(t + 1 - k, 0)]


def back_std(x, k):
    m = back_mean(x, k)
    m2 = back_mean(x ** 2, k)
    return np.sqrt(np.maximum(m2 - m ** 2, 0))


def back_extreme(x, k, fn):
    """Rolling max/min over the last k samples (NaN-aware)."""
    df = pd.DataFrame(x.T)
    r = df.rolling(k, min_periods=1)
    return (r.max() if fn == "max" else r.min()).to_numpy().T


def time_since(event, cap):
    """Samples since the last time `event` was > 0 (capped)."""
    N, S = event.shape
    out = np.full((N, S), float(cap))
    last = np.full(N, -10 ** 9)
    for t in range(S):
        last = np.where(event[:, t] > 0, t, last)
        out[:, t] = np.minimum(t - last, cap)
    return out


def ffill_limited(x, limit):
    """Fill short sensor gaps (up to `limit` samples) with the last value."""
    df = pd.DataFrame(x.T).ffill(limit=limit)
    return df.to_numpy().T


# ---------------------------------------------------------------------------
# Labels: what happens in the next 2 hours
# ---------------------------------------------------------------------------
def future_event(flag, horizon, run=MIN_RUN):
    """True if `flag` holds for `run` consecutive samples within (t, t+horizon]."""
    N, S = flag.shape
    f = flag.astype(int)
    runs = np.zeros_like(f)
    for j in range(run):
        runs[:, : S - j] += f[:, j:]
    started = (runs == run).astype(float)          # a run starts at this sample
    cs = np.hstack([np.zeros((N, 1)), np.cumsum(started, 1)])
    t = np.arange(S)
    lo = np.minimum(t + 1, S)
    hi = np.minimum(t + 1 + horizon - run + 1, S)
    return (cs[:, hi] - cs[:, lo]) > 0


# ---------------------------------------------------------------------------
# Static (EHR) features
# ---------------------------------------------------------------------------
STATIC_FEATURES = [
    "age", "sex_male", "bmi", "waist_cm", "diabetes_duration_yrs", "family_history_t2d",
    "smoker", "rice_diet", "adherence_score", "egfr", "hypertension", "dyslipidaemia",
    "neuropathy", "retinopathy", "ckd", "hba1c", "hba1c_change_6m", "fasting_glucose",
    "fasting_insulin", "homa_ir", "ldl", "hdl", "triglycerides", "systolic_bp",
    "tcf7l2_risk_alleles", "polygenic_risk_score",
    "on_metformin", "on_sulfonylurea", "on_dpp4", "on_sglt2", "on_basal_insulin",
]


def static_table() -> pd.DataFrame:
    p = pd.read_csv(C.EHR_DIR / "patients.csv").set_index("patient_id")
    labs = pd.read_csv(C.EHR_DIR / "observations.csv", parse_dates=["date"]).sort_values("date")
    meds = pd.read_csv(C.EHR_DIR / "medications.csv")
    gen = pd.read_csv(C.EHR_DIR / "genetics.csv").set_index("patient_id")

    latest = labs.groupby(["patient_id", "test"]).value.last().unstack()
    a1c = labs[labs.test == "HbA1c"].groupby("patient_id").value
    s = pd.DataFrame(index=p.index)
    s["age"] = p.age
    s["sex_male"] = (p.sex == "M").astype(int)
    for c in ["bmi", "waist_cm", "diabetes_duration_yrs", "egfr"]:
        s[c] = p[c]
    for c in ["family_history_t2d", "smoker", "hypertension", "dyslipidaemia",
              "neuropathy", "retinopathy", "ckd"]:
        s[c] = p[c].astype(int)
    s["rice_diet"] = (p.diet_pattern == "rice-predominant").astype(int)
    s["adherence_score"] = p.self_reported_adherence.map({"good": 2, "fair": 1, "poor": 0})
    s["hba1c"] = latest["HbA1c"]
    s["hba1c_change_6m"] = a1c.last() - a1c.first()
    s["fasting_glucose"] = latest["Fasting plasma glucose"]
    s["fasting_insulin"] = latest["Fasting insulin"]
    s["homa_ir"] = latest["HOMA-IR"]
    s["ldl"] = latest["LDL cholesterol"]
    s["hdl"] = latest["HDL cholesterol"]
    s["triglycerides"] = latest["Triglycerides"]
    s["systolic_bp"] = latest["Systolic BP"]
    s["tcf7l2_risk_alleles"] = gen.TCF7L2_rs7903146_risk_alleles
    s["polygenic_risk_score"] = gen.polygenic_risk_score
    for cls in ["metformin", "sulfonylurea", "dpp4", "sglt2", "basal_insulin"]:
        on = meds[meds.drug_class == cls].patient_id.unique()
        s[f"on_{cls}"] = s.index.isin(on).astype(int)
    return s[STATIC_FEATURES]


# ---------------------------------------------------------------------------
# Dynamic features
# ---------------------------------------------------------------------------
def dynamic_features(A) -> tuple[dict, list, list]:
    g_raw = A["glucose"]
    g = ffill_limited(g_raw, 6)               # bridge gaps up to 30 min
    F = {}
    h = lambda m: m // STEP                   # minutes -> samples

    # ---- CGM -------------------------------------------------------------
    F["glucose_now"] = g
    for m in [15, 30, 60]:
        F[f"glucose_lag_{m}"] = np.hstack([np.full((g.shape[0], h(m)), np.nan), g[:, :-h(m)]])
        F[f"glucose_delta_{m}"] = g - F[f"glucose_lag_{m}"]          # trend (mg/dL per window)
    F["glucose_accel"] = F["glucose_delta_15"] - (F["glucose_lag_15"] - F["glucose_lag_30"])
    for m in [60, 180]:
        F[f"glucose_mean_{m}"] = back_mean(g, h(m))
        F[f"glucose_std_{m}"] = back_std(g, h(m))
    F["glucose_max_180"] = back_extreme(g, h(180), "max")
    F["glucose_min_180"] = back_extreme(g, h(180), "min")
    F["glucose_mean_24h"] = back_mean(g, SPD)
    F["glucose_cv_24h"] = back_std(g, SPD) / F["glucose_mean_24h"]
    F["time_above_180_24h"] = back_mean(np.where(np.isfinite(g), (g > 180).astype(float), np.nan), SPD)
    F["time_below_70_24h"] = back_mean(np.where(np.isfinite(g), (g < 70).astype(float), np.nan), SPD)
    F["cgm_missing_1h"] = 1 - back_mean(np.isfinite(g_raw).astype(float), h(60))

    # ---- Smartwatch ----------------------------------------------------------
    F["hr_15"] = back_mean(A["hr"], h(15))
    F["hr_60"] = back_mean(A["hr"], h(60))
    F["hr_vs_24h"] = F["hr_15"] - back_mean(A["hr"], SPD)    # elevated vs personal norm
    F["hr_rise_30"] = F["hr_15"] - np.hstack([np.full((g.shape[0], h(30)), np.nan), F["hr_15"][:, :-h(30)]])
    F["hrv_60"] = back_mean(A["hrv"], h(60))
    F["hrv_vs_24h"] = F["hrv_60"] - back_mean(A["hrv"], SPD)
    for m in [15, 60, 180]:
        F[f"steps_{m}"] = back_sum(A["steps"], h(m))
    F["steps_24h"] = back_sum(A["steps"], SPD)
    F["asleep_now"] = A["asleep"]
    F["sleep_last_12h_hrs"] = back_sum(A["asleep"], h(720)) * STEP / 60
    F["deep_sleep_last_12h_hrs"] = back_sum(A["deep"], h(720)) * STEP / 60

    # ---- App diary (only events that already happened) -----------------------
    F["carbs_30"] = back_sum(A["carbs"], h(30))
    F["carbs_60"] = back_sum(A["carbs"], h(60))
    F["carbs_120"] = back_sum(A["carbs"], h(120))
    F["carbs_240"] = back_sum(A["carbs"], h(240))
    F["mins_since_meal"] = time_since(A["carbs"], h(720)) * STEP
    F["carbs_today_24h"] = back_sum(A["carbs"], SPD)
    F["mins_since_sulfonylurea"] = time_since(A["dose_sulfonylurea"], h(1440)) * STEP
    F["mins_since_basal_insulin"] = time_since(A["dose_basal_insulin"], h(1440)) * STEP
    F["mins_since_metformin"] = time_since(A["dose_metformin"], h(1440)) * STEP

    # ---- Time context -------------------------------------------------------
    S = g.shape[1]
    mod = (np.arange(S) % SPD) * STEP / 60
    F["hour_sin"] = np.broadcast_to(np.sin(2 * np.pi * mod / 24), g.shape)
    F["hour_cos"] = np.broadcast_to(np.cos(2 * np.pi * mod / 24), g.shape)
    dow = (A["start"] + pd.to_timedelta(np.arange(S) * STEP, unit="min")).dayofweek
    F["weekend"] = np.broadcast_to((dow >= 5).astype(float), g.shape)

    cgm_cols = [k for k in F if k.startswith(("glucose", "time_", "cgm_"))]
    watch_cols = [k for k in F if k.startswith(("hr", "steps", "asleep", "sleep", "deep"))]
    diary_cols = [k for k in F if k.startswith(("carbs", "mins_since"))]
    time_cols = ["hour_sin", "hour_cos", "weekend"]
    groups = dict(cgm=cgm_cols, watch=watch_cols, diary=diary_cols, time=time_cols)
    return F, g_raw, groups


def build_dataset() -> tuple[pd.DataFrame, dict]:
    A = load_arrays()
    F, g_raw, groups = dynamic_features(A)
    N, S = g_raw.shape

    # Labels from the raw (un-filled) future CGM
    hi = np.nan_to_num(g_raw, nan=0) > C.HYPER_THRESHOLD
    lo = np.nan_to_num(g_raw, nan=999) < C.HYPO_THRESHOLD
    spike = future_event(hi, HORIZON)
    hypo = future_event(lo, HORIZON)
    fut = {}
    for m in [30, 60, 90, 120]:
        k = m // STEP
        f = np.full((N, S), np.nan)
        f[:, : S - k] = g_raw[:, k:]
        fut[f"glucose_plus_{m}"] = f

    # Prediction moments: after 1 day of history, every 15 min, with enough future
    t_idx = np.arange(SPD, S - HORIZON, STRIDE)
    now = F["glucose_now"][:, t_idx]
    rows = {"patient_id": np.repeat(A["pids"], len(t_idx)),
            "timestamp": np.tile(A["start"] + pd.to_timedelta(t_idx * STEP, unit="min"), N)}
    for k, v in F.items():
        rows[k] = np.asarray(v)[:, t_idx].reshape(-1)
    rows["spike_2h"] = spike[:, t_idx].reshape(-1).astype(int)
    rows["hypo_2h"] = hypo[:, t_idx].reshape(-1).astype(int)
    for k, v in fut.items():
        rows[k] = v[:, t_idx].reshape(-1)
    df = pd.DataFrame(rows)
    df = df[np.isfinite(now.reshape(-1))]                 # need a current reading

    # Who is "eligible" for each warning: you can only predict an event that
    # has not started yet.
    lag = df.glucose_lag_15.fillna(df.glucose_now)
    df["spike_eligible"] = ((df.glucose_now < C.HYPER_THRESHOLD) & (lag < C.HYPER_THRESHOLD)).astype(int)
    df["hypo_eligible"] = ((df.glucose_now >= C.HYPO_THRESHOLD) & (lag >= C.HYPO_THRESHOLD)).astype(int)

    static = static_table()
    df = df.merge(static, left_on="patient_id", right_index=True, how="left")
    groups["static"] = STATIC_FEATURES
    return df.reset_index(drop=True), groups
