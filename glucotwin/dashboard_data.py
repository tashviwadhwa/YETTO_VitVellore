"""Data access and simulation layer for the GlucoTwin Doctor Dashboard.

Every number the dashboard shows comes from the real pipeline outputs:
    health record      data/ehr
    wearables + diary  data/wearables, data/logs
    calibrated twins   data/twins (parameters + saved internal state every 15 min)
    spike models       models/*.json, explained with TreeSHAP (glucotwin.explain)
    hypo engine        the calibrated twin's 2-hour forecast < threshold (model card)
    what-if            glucotwin.whatif.simulate_scenarios, started from the twin's
                       real saved state at the chosen moment
"""
from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config as C
from . import twin
from .explain import contributions, source_of, top_reasons
from .hybrid import load_twin
from .whatif import DISHES, Scenario, simulate_scenarios

START = pd.Timestamp(C.START_DATE)
STATE_KEYS = ["G", "X", "I", "Q1", "Q2", "load"]
_CACHE: Dict[str, Any] = {}


def _twin() -> dict:
    """Calibrated twins for all patients (loaded once, ~3 s)."""
    if "twin" not in _CACHE:
        _CACHE["twin"] = load_twin()
    return _CACHE["twin"]


def _read(name: str, folder) -> pd.DataFrame:
    """Read a large file once and keep it in memory (parquet if available)."""
    if name not in _CACHE:
        pq = folder / f"{name}.parquet"
        df = pd.read_parquet(pq) if pq.exists() else pd.read_csv(folder / f"{name}.csv.gz")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        _CACHE[name] = df
    return _CACHE[name]


# ---------------------------------------------------------------------------
# 1. Cohort & Patient Static Data Loader
# ---------------------------------------------------------------------------
RISK_PROFILE_NOTE = ("Rule of thumb, not a validated score: one point each for HbA1c 7.5-8.4% "
                     "(two for 8.5% or more), BMI 25 or more, 8+ years of diabetes, and taking "
                     "glimepiride or insulin.")


def load_cohort_metadata() -> pd.DataFrame:
    """Load and merge complete metadata for all 200 patients in the cohort."""
    patients = pd.read_csv(C.EHR_DIR / "patients.csv")
    obs = pd.read_csv(C.EHR_DIR / "observations.csv")
    meds = pd.read_csv(C.EHR_DIR / "medications.csv")
    conds = pd.read_csv(C.EHR_DIR / "conditions.csv")
    genetics = pd.read_csv(C.EHR_DIR / "genetics.csv")
    twins = pd.read_csv(C.DATA_DIR / "twins" / "twin_params.csv")

    obs_sorted = obs.sort_values("date")
    latest_obs = obs_sorted.groupby(["patient_id", "test"]).last().reset_index()
    obs_pivot = latest_obs.pivot(index="patient_id", columns="test", values="value").reset_index()
    first_a1c = obs_sorted[obs_sorted.test == "HbA1c"].groupby("patient_id").first()["value"].rename("hba1c_baseline")
    obs_pivot = obs_pivot.merge(first_a1c, on="patient_id", how="left")
    obs_pivot["hba1c_change"] = obs_pivot["HbA1c"] - obs_pivot["hba1c_baseline"]

    med_list = meds.groupby("patient_id")["drug"].apply(lambda s: ", ".join(sorted(set(s)))).reset_index(name="medications")
    drug_classes = meds.groupby("patient_id")["drug_class"].apply(lambda s: sorted(set(s))).reset_index(name="drug_classes")
    cond_list = conds.groupby("patient_id")["condition"].apply(lambda s: sorted(set(s))).reset_index(name="conditions")

    df = patients.merge(obs_pivot, on="patient_id", how="left")
    df = df.merge(med_list, on="patient_id", how="left")
    df = df.merge(drug_classes, on="patient_id", how="left")
    df = df.merge(cond_list, on="patient_id", how="left")
    df = df.merge(genetics, on="patient_id", how="left")
    df = df.merge(twins, on="patient_id", how="left")
    df["medications"] = df["medications"].fillna("None")
    df["drug_classes"] = df["drug_classes"].apply(lambda v: v if isinstance(v, list) else [])
    df["conditions"] = df["conditions"].apply(lambda v: v if isinstance(v, list) else [])
    df["on_glimepiride"] = df["drug_classes"].apply(lambda v: "sulfonylurea" in v)

    # Asian-Indian BMI classes (Misra et al. 2009 consensus; WHO Asia-Pacific)
    def classify_bmi(bmi):
        if pd.isna(bmi):
            return "Unknown"
        if bmi < 18.5:
            return "Underweight (<18.5)"
        if bmi < 23.0:
            return "Normal (18.5–22.9)"
        if bmi < 25.0:
            return "Overweight (23.0–24.9)"
        return "Obese (≥25.0)"

    df["asian_bmi_category"] = df["bmi"].apply(classify_bmi)

    # Glycaemic control against the ADA HbA1c target (<7.0%)
    def classify_a1c(a1c):
        if pd.isna(a1c):
            return "Unknown"
        if a1c < 7.0:
            return "At target (<7.0%)"
        if a1c <= 8.5:
            return "Above target (7.0–8.5%)"
        return "Well above target (>8.5%)"

    df["glycemic_status"] = df["HbA1c"].apply(classify_a1c)

    # Rule-of-thumb profile (see RISK_PROFILE_NOTE) - not used by any model
    def calc_profile(row):
        score = 0
        a1c = row.get("HbA1c", 0) or 0
        score += 2 if a1c >= 8.5 else (1 if a1c >= 7.5 else 0)
        score += 1 if (row.get("bmi", 0) or 0) >= 25.0 else 0
        score += 1 if (row.get("diabetes_duration_yrs", 0) or 0) >= 8.0 else 0
        classes = row.get("drug_classes", [])
        score += 1 if ("sulfonylurea" in classes or "basal_insulin" in classes) else 0
        if score >= 3:
            return "Higher-risk profile"
        if score >= 2:
            return "Moderate profile"
        return "Lower-risk profile"

    df["risk_tier"] = df.apply(calc_profile, axis=1)
    return df


# ---------------------------------------------------------------------------
# 2. Patient Longitudinal Time-Series Loader
# ---------------------------------------------------------------------------
def load_patient_sensors(patient_id: str) -> Dict[str, Any]:
    """14-day CGM, smartwatch, diary and the calibrated twin's reconstruction."""
    cgm = _read("cgm", C.WEARABLE_DIR)
    watch = _read("smartwatch", C.WEARABLE_DIR)
    cgm = cgm[cgm.patient_id == patient_id].sort_values("timestamp")
    watch = watch[watch.patient_id == patient_id].sort_values("timestamp")

    sleep = pd.read_csv(C.WEARABLE_DIR / "sleep_nightly.csv", parse_dates=["date"])
    sleep = sleep[sleep.patient_id == patient_id].sort_values("date").reset_index(drop=True)
    meals = pd.read_csv(C.LOG_DIR / "meals.csv", parse_dates=["timestamp"])
    meals = meals[meals.patient_id == patient_id].sort_values("timestamp").reset_index(drop=True)
    doses = pd.read_csv(C.LOG_DIR / "medications.csv", parse_dates=["timestamp"])
    doses = doses[doses.patient_id == patient_id].sort_values("timestamp").reset_index(drop=True)
    activity = pd.read_csv(C.LOG_DIR / "activity.csv", parse_dates=["timestamp"])
    activity = activity[activity.patient_id == patient_id].sort_values("timestamp").reset_index(drop=True)

    merged = pd.merge(cgm[["timestamp", "glucose_mgdl"]],
                      watch[["timestamp", "heart_rate_bpm", "hrv_rmssd_ms", "steps", "sleep_stage"]],
                      on="timestamp", how="outer").sort_values("timestamp").reset_index(drop=True)

    T = _twin()
    k = int(np.where(T["pids"] == patient_id)[0][0])
    twin_g = np.asarray(T["glucose_5min"][k], dtype=float)
    merged["twin_glucose_mgdl"] = twin_g[: len(merged)] if len(twin_g) >= len(merged) else np.nan

    merged["day"] = ((merged["timestamp"] - START).dt.total_seconds() // 86400).astype(int)
    merged["time_str"] = merged["timestamp"].dt.strftime("%H:%M")
    return dict(timeseries=merged, meals=meals, doses=doses, activity=activity, sleep=sleep, start_date=START)


# ---------------------------------------------------------------------------
# 3. Clinical Glycemic Metrics (Ambulatory Glucose Profile)
# ---------------------------------------------------------------------------
def compute_agp_metrics(glucose_series: pd.Series) -> Dict[str, Any]:
    """International consensus CGM metrics (Battelino et al., Diabetes Care 2019)."""
    g = glucose_series.dropna()
    if len(g) == 0:
        return {}
    n = len(g)
    mean_g, sd_g = g.mean(), g.std()
    return dict(
        tir=g.between(70, 180).sum() / n * 100,   # target > 70%
        tar=(g > 180).sum() / n * 100,            # target < 25%
        tar_very_high=(g > 250).sum() / n * 100,  # target < 5%
        tbr=(g < 70).sum() / n * 100,             # target < 4%
        tbr_very_low=(g < 54).sum() / n * 100,    # target < 1%
        mean_glucose=mean_g,
        std_glucose=sd_g,
        cv=(sd_g / mean_g * 100) if mean_g > 0 else 0.0,   # target <= 36%
        gmi=3.31 + 0.02392 * mean_g,              # Glucose Management Indicator
        readings_count=n,
    )


# ---------------------------------------------------------------------------
# 4. Models & Explainability Engine
# ---------------------------------------------------------------------------
SPIKE_LIKELY, SPIKE_POSSIBLE = 0.70, 0.40      # same cut-offs as the model card / observatory


def twin_forecast(patient_id: str, ts: pd.Timestamp, horizon: int = 120) -> np.ndarray:
    """The calibrated twin's forecast from its saved state at a 15-minute moment
    (assumes no new meal or walk; scheduled medicines taken as prescribed)."""
    T = _twin()
    k = int(np.where(T["pids"] == patient_id)[0][0])
    minute = int((pd.Timestamp(ts).floor("15min") - START) / pd.Timedelta(minutes=1))
    st = {key: np.array([T["states"][key][k, minute // 15]]) for key in STATE_KEYS}
    return twin.forecast_from_states(T["theta"], T["pri"], st, np.array([k]), np.array([minute]),
                                     T["inp"], horizon=horizon)["G"][0]


class ModelService:
    def __init__(self):
        self.card = json.loads((C.ROOT / "models" / "model_card.json").read_text())
        self.m_cgm = xgb.XGBClassifier()
        self.m_cgm.load_model(str(C.ROOT / "models" / "spike_cgm.json"))
        self.m_nocgm = xgb.XGBClassifier()
        self.m_nocgm.load_model(str(C.ROOT / "models" / "spike_nocgm.json"))
        self._fused_df = None
        self._week2_df = None

    def get_fused_data(self) -> pd.DataFrame:
        if self._fused_df is None:
            pq_path = C.DATA_DIR / "processed" / "fused_dataset.parquet"
            self._fused_df = (pd.read_parquet(pq_path) if pq_path.exists()
                              else pd.read_csv(C.DATA_DIR / "processed" / "fused_dataset.csv.gz"))
            self._fused_df["timestamp"] = pd.to_datetime(self._fused_df["timestamp"])
        return self._fused_df

    def get_week2_data(self) -> pd.DataFrame:
        if self._week2_df is None:
            fused = self.get_fused_data()
            tf_path = C.DATA_DIR / "processed" / "twin_features_week2.parquet"
            tf = (pd.read_parquet(tf_path) if tf_path.exists()
                  else pd.read_csv(C.DATA_DIR / "processed" / "twin_features_week2.csv.gz"))
            tf["timestamp"] = pd.to_datetime(tf["timestamp"])
            self._week2_df = fused.merge(tf, on=["patient_id", "timestamp"])
        return self._week2_df

    def get_patient_moments(self, patient_id: str, mode: str = "cgm") -> pd.DataFrame:
        df = self.get_fused_data() if mode == "cgm" else self.get_week2_data()
        return df[df.patient_id == patient_id].sort_values("timestamp").reset_index(drop=True)

    def predict_moment(self, row: pd.Series, mode: str = "cgm") -> Dict[str, Any]:
        """Spike score + TreeSHAP reasons, and the twin's hypo check, for one moment."""
        key = "spike_cgm" if mode == "cgm" else "spike_nocgm"
        model = self.m_cgm if mode == "cgm" else self.m_nocgm
        features = self.card[key]["features"]
        X = pd.DataFrame([row[features]]).astype(float)
        score = float(model.predict_proba(X)[0, 1])
        contrib_df = contributions(model, X)
        reasons = top_reasons(contrib_df.iloc[0], row, k=8)

        # Hypo engine = the calibrated twin's 2-hour forecast (works in both modes)
        hypo_th = float(self.card["hypo_twin"]["threshold_mgdl"])
        fc = twin_forecast(row["patient_id"], row["timestamp"])
        hypo_nadir = float(fc.min())
        hypo_time = pd.Timestamp(row["timestamp"]) + pd.Timedelta(minutes=int(np.argmin(fc)) + 1)
        perf = [t for t in self.card["hypo_twin"]["threshold_table"] if t["threshold_mgdl"] == hypo_th][0]

        contrib_no_base = contrib_df.drop(columns="_baseline").iloc[0]
        imp_by_source = contrib_no_base.abs().groupby(contrib_no_base.index.map(source_of)).sum()
        total_imp = imp_by_source.sum()
        source_pct = (imp_by_source / total_imp * 100).to_dict() if total_imp > 0 else {}

        level = "likely" if score >= SPIKE_LIKELY else ("possible" if score >= SPIKE_POSSIBLE else "unlikely")
        return dict(
            spike_probability=score,          # kept for compatibility: a model SCORE, not a calibrated probability
            spike_score=score,
            spike_level=level,
            spike_alert=score >= SPIKE_LIKELY,
            hypo_alert=hypo_nadir < hypo_th,
            hypo_nadir=hypo_nadir,
            hypo_nadir_time=hypo_time,
            hypo_threshold=hypo_th,
            hypo_rule_sensitivity=float(perf["sensitivity"]),
            hypo_rule_precision=float(perf["precision"]),
            twin_forecast=fc,
            reasons=reasons,
            source_breakdown=source_pct,
            features_used=features,
            all_contributions=contrib_no_base.to_dict(),
        )


_model_service = None


def get_model_service() -> ModelService:
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service


# ---------------------------------------------------------------------------
# 5. Interactive "What-If" simulation on the calibrated twin
# ---------------------------------------------------------------------------
NO_MEAL = "No meal"
GLIMEPIRIDE_OPTIONS = ["as prescribed", "skip next dose", "half dose", "double dose"]


def dish_carbs(dish: str) -> float:
    return float(DISHES[dish][0]) if dish in DISHES else 0.0


def moment_state(patient_id: str, ts: pd.Timestamp) -> Dict[str, Any]:
    """Glucose at a moment: the observed CGM reading and the twin's own estimate."""
    T = _twin()
    k = int(np.where(T["pids"] == patient_id)[0][0])
    ts = pd.Timestamp(ts).floor("15min")
    minute = int((ts - START) / pd.Timedelta(minutes=1))
    cgm = _read("cgm", C.WEARABLE_DIR)
    row = cgm[(cgm.patient_id == patient_id) & (cgm.timestamp <= ts) & (cgm.timestamp > ts - pd.Timedelta(minutes=30))]
    obs = float(row.glucose_mgdl.dropna().iloc[-1]) if row.glucose_mgdl.notna().any() else np.nan
    meals = pd.read_csv(C.LOG_DIR / "meals.csv", parse_dates=["timestamp"])
    meals = meals[(meals.patient_id == patient_id) & (meals.timestamp <= ts)]
    last_meal = meals.iloc[-1] if len(meals) else None
    return dict(timestamp=ts, cgm=obs, twin=float(T["states"]["G"][k, minute // 15]),
                food_in_gut_g=float((T["states"]["Q1"][k, minute // 15] + T["states"]["Q2"][k, minute // 15]) / 1000),
                last_meal=None if last_meal is None else
                f"{last_meal.food}, {last_meal.carbs_g:.0f} g, {last_meal.timestamp:%a %H:%M}")


def _scenario(name: str, dish: str, carbs_g: float, meal_in_min: int, walk_min: int,
              walk_delay_min: int, glimepiride: str) -> Scenario:
    has_meal = dish and dish != NO_MEAL and carbs_g > 0
    portion = float(np.clip(carbs_g / dish_carbs(dish), 0.1, 3.0)) if has_meal else 1.0
    meal_t = int(meal_in_min) if has_meal else 0
    return Scenario(name=name, dish=dish if has_meal else None, portion=portion, meal_in_min=meal_t,
                    walk_min=int(walk_min), walk_starts_in_min=meal_t + int(walk_delay_min),
                    glimepiride=glimepiride)


def simulate_whatif(patient_id: str, start_ts: pd.Timestamp,
                    a: Dict[str, Any], b: Dict[str, Any], horizon_minutes: int = 240) -> Dict[str, Any]:
    """Compare two scenarios on the patient's calibrated twin, starting from the twin's
    real saved state at `start_ts` (glucose, insulin action, food still in the gut,
    recent exercise, medicines already taken).

    a / b: dict(name, dish, carbs_g, meal_in_min, walk_min, walk_delay_min, glimepiride)
    """
    T = _twin()
    ts = pd.Timestamp(start_ts).floor("15min")
    minute = int((ts - START) / pd.Timedelta(minutes=1))
    plan = Scenario(name="As planned (no new meal or walk)")
    sa, sb = _scenario(**a), _scenario(**b)
    if sa.name == sb.name:
        sb.name = sb.name + " (B)"
    res = simulate_scenarios(T, patient_id, minute, [plan, sa, sb], horizon=horizon_minutes)
    g_plan, g_a, g_b = (res["curves"][s.name] for s in (plan, sa, sb))

    k = int(np.where(T["pids"] == patient_id)[0][0])
    planned = T["inp"]["planned"]["sulfonylurea"][k, minute + 1: minute + 1 + horizon_minutes]
    su_due = bool(planned.sum() > 0)

    peak_a, peak_b = float(g_a.max()), float(g_b.max())
    above_a, above_b = int((g_a > 180).sum()), int((g_b > 180).sum())
    min_a, min_b = float(g_a.min()), float(g_b.min())
    below_a, below_b = int((g_a < 70).sum()), int((g_b < 70).sum())
    t_peak_b = ts + pd.Timedelta(minutes=int(np.argmax(g_b)) + 1)

    notes = []
    d = peak_a - peak_b
    if d > 5:
        notes.append(f"Scenario B peaks {d:.0f} mg/dL lower than A ({peak_a:.0f} → {peak_b:.0f} mg/dL, "
                     f"at {t_peak_b:%H:%M}).")
    elif d < -5:
        notes.append(f"Scenario B peaks {-d:.0f} mg/dL higher than A ({peak_a:.0f} → {peak_b:.0f} mg/dL).")
    else:
        notes.append(f"The two scenarios reach a similar peak ({peak_a:.0f} vs {peak_b:.0f} mg/dL).")
    if above_a or above_b:
        notes.append(f"Time above 180 mg/dL: {above_a} min in A, {above_b} min in B.")
    if below_a or below_b:
        notes.append(f"⚠️ Glucose falls below 70 mg/dL for {below_a} min in A and {below_b} min in B "
                     f"(lowest {min_a:.0f} and {min_b:.0f} mg/dL).")
    else:
        notes.append(f"Neither scenario goes below 70 mg/dL (lowest {min(min_a, min_b):.0f} mg/dL).")
    if not su_due and any(s.glimepiride == "skip next dose" for s in (sa, sb)):
        notes.append("No glimepiride dose is due in this window, so 'skip next dose' changes nothing yet.")
    if any(s.glimepiride in ("half dose", "double dose") for s in (sa, sb)):
        notes.append("A half or double dose also scales glimepiride already taken today "
                     "('what if today's dose had been different').")
    notes.append("Hypothetical: simulated on this patient's calibrated twin; not a recorded outcome.")

    minutes = res["minutes"]
    return dict(
        minutes=minutes,
        time_labels=[(ts + pd.Timedelta(minutes=int(m))).strftime("%H:%M") for m in minutes],
        timestamps=[ts + pd.Timedelta(minutes=int(m)) for m in minutes],
        glucose_plan=g_plan, glucose_baseline=g_a, glucose_whatif=g_b,
        name_a=sa.name, name_b=sb.name,
        peak_baseline=peak_a, peak_whatif=peak_b, peak_reduction=peak_a - peak_b,
        min_baseline=min_a, min_whatif=min_b,
        hyper_min_baseline=above_a, hyper_min_whatif=above_b, hyper_min_saved=above_a - above_b,
        hypo_min_baseline=below_a, hypo_min_whatif=below_b,
        hypo_risk_baseline=below_a > 0, hypo_risk_whatif=below_b > 0,
        glimepiride_due=su_due,
        clinical_note=" ".join(notes),
    )
