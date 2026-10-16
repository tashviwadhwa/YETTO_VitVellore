"""Explain every alert in words a doctor understands.

XGBoost can split each prediction into per-feature contributions (TreeSHAP:
SHapley Additive exPlanations, computed exactly for tree models). A positive
contribution pushes the risk up, a negative one pulls it down. We turn the
top contributions into sentences such as
    "High-carb meal 40 min ago (92 g)  -> raises risk"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

# feature -> (readable name, data source, how to show the value)
FEATURES = {
    # CGM
    "glucose_now": ("Current glucose", "CGM", "{:.0f} mg/dL"),
    "glucose_lag_15": ("Glucose 15 min ago", "CGM", "{:.0f} mg/dL"),
    "glucose_lag_30": ("Glucose 30 min ago", "CGM", "{:.0f} mg/dL"),
    "glucose_lag_60": ("Glucose 1 h ago", "CGM", "{:.0f} mg/dL"),
    "glucose_delta_15": ("Glucose trend (last 15 min)", "CGM", "{:+.0f} mg/dL"),
    "glucose_delta_30": ("Glucose trend (last 30 min)", "CGM", "{:+.0f} mg/dL"),
    "glucose_delta_60": ("Glucose trend (last 1 h)", "CGM", "{:+.0f} mg/dL"),
    "glucose_accel": ("Glucose acceleration", "CGM", "{:+.0f} mg/dL"),
    "glucose_mean_60": ("Average glucose (last 1 h)", "CGM", "{:.0f} mg/dL"),
    "glucose_std_60": ("Glucose swings (last 1 h)", "CGM", "{:.0f} mg/dL"),
    "glucose_mean_180": ("Average glucose (last 3 h)", "CGM", "{:.0f} mg/dL"),
    "glucose_std_180": ("Glucose swings (last 3 h)", "CGM", "{:.0f} mg/dL"),
    "glucose_max_180": ("Highest glucose (last 3 h)", "CGM", "{:.0f} mg/dL"),
    "glucose_min_180": ("Lowest glucose (last 3 h)", "CGM", "{:.0f} mg/dL"),
    "glucose_mean_24h": ("Average glucose (24 h)", "CGM", "{:.0f} mg/dL"),
    "glucose_cv_24h": ("Glucose variability (24 h)", "CGM", "{:.0%}"),
    "time_above_180_24h": ("Time above 180 (24 h)", "CGM", "{:.0%}"),
    "time_below_70_24h": ("Time below 70 (24 h)", "CGM", "{:.1%}"),
    "cgm_missing_1h": ("CGM signal gaps (1 h)", "CGM", "{:.0%}"),
    # Smartwatch
    "hr_15": ("Heart rate (last 15 min)", "Smartwatch", "{:.0f} bpm"),
    "hr_60": ("Heart rate (last 1 h)", "Smartwatch", "{:.0f} bpm"),
    "hr_vs_24h": ("Heart rate vs personal normal", "Smartwatch", "{:+.0f} bpm"),
    "hr_rise_30": ("Heart rate change (30 min)", "Smartwatch", "{:+.0f} bpm"),
    "hrv_60": ("Heart rate variability (1 h)", "Smartwatch", "{:.0f} ms"),
    "hrv_vs_24h": ("HRV vs personal normal", "Smartwatch", "{:+.0f} ms"),
    "steps_15": ("Steps (last 15 min)", "Smartwatch", "{:.0f}"),
    "steps_60": ("Steps (last 1 h)", "Smartwatch", "{:.0f}"),
    "steps_180": ("Steps (last 3 h)", "Smartwatch", "{:.0f}"),
    "steps_24h": ("Steps (24 h)", "Smartwatch", "{:.0f}"),
    "asleep_now": ("Asleep now", "Smartwatch", "yesno"),
    "sleep_last_12h_hrs": ("Sleep last night", "Smartwatch", "{:.1f} h"),
    "deep_sleep_last_12h_hrs": ("Deep sleep last night", "Smartwatch", "{:.1f} h"),
    # App diary
    "carbs_30": ("Carbs eaten (last 30 min)", "Diary", "{:.0f} g"),
    "carbs_60": ("Carbs eaten (last 1 h)", "Diary", "{:.0f} g"),
    "carbs_120": ("Carbs eaten (last 2 h)", "Diary", "{:.0f} g"),
    "carbs_240": ("Carbs eaten (last 4 h)", "Diary", "{:.0f} g"),
    "mins_since_meal": ("Time since last meal", "Diary", "{:.0f} min"),
    "carbs_today_24h": ("Carbs (24 h)", "Diary", "{:.0f} g"),
    "mins_since_sulfonylurea": ("Time since glimepiride dose", "Diary", "{:.0f} min"),
    "mins_since_basal_insulin": ("Time since insulin dose", "Diary", "{:.0f} min"),
    "mins_since_metformin": ("Time since metformin dose", "Diary", "{:.0f} min"),
    # Time
    "hour_sin": ("Time of day", "Time", "{:.2f}"),
    "hour_cos": ("Time of day", "Time", "{:.2f}"),
    "weekend": ("Weekend", "Time", "yesno"),
    # Health record
    "age": ("Age", "EHR", "{:.0f} yrs"),
    "sex_male": ("Sex (male)", "EHR", "yesno"),
    "bmi": ("BMI", "EHR", "{:.1f}"),
    "waist_cm": ("Waist", "EHR", "{:.0f} cm"),
    "diabetes_duration_yrs": ("Years with diabetes", "EHR", "{:.1f} yrs"),
    "family_history_t2d": ("Family history of diabetes", "EHR", "yesno"),
    "smoker": ("Smoker", "EHR", "yesno"),
    "rice_diet": ("Rice-based diet", "EHR", "yesno"),
    "adherence_score": ("Medicine adherence (self-report)", "EHR", "adherence"),
    "egfr": ("Kidney function (eGFR)", "EHR", "{:.0f}"),
    "hypertension": ("High blood pressure", "EHR", "yesno"),
    "dyslipidaemia": ("High cholesterol", "EHR", "yesno"),
    "neuropathy": ("Diabetic nerve damage", "EHR", "yesno"),
    "retinopathy": ("Diabetic eye disease", "EHR", "yesno"),
    "ckd": ("Chronic kidney disease", "EHR", "yesno"),
    "hba1c": ("HbA1c", "EHR", "{:.1f}%"),
    "hba1c_change_6m": ("HbA1c change (6 months)", "EHR", "{:+.1f}"),
    "fasting_glucose": ("Fasting glucose (lab)", "EHR", "{:.0f} mg/dL"),
    "fasting_insulin": ("Fasting insulin (lab)", "EHR", "{:.1f}"),
    "homa_ir": ("Insulin resistance (HOMA-IR)", "EHR", "{:.1f}"),
    "ldl": ("LDL cholesterol", "EHR", "{:.0f}"),
    "hdl": ("HDL cholesterol", "EHR", "{:.0f}"),
    "triglycerides": ("Triglycerides", "EHR", "{:.0f}"),
    "systolic_bp": ("Systolic BP", "EHR", "{:.0f}"),
    "tcf7l2_risk_alleles": ("TCF7L2 risk gene copies", "EHR", "{:.0f}"),
    "polygenic_risk_score": ("Genetic risk score", "EHR", "{:+.1f}"),
    "on_metformin": ("On metformin", "EHR", "yesno"),
    "on_sulfonylurea": ("On glimepiride", "EHR", "yesno"),
    "on_dpp4": ("On sitagliptin", "EHR", "yesno"),
    "on_sglt2": ("On dapagliflozin", "EHR", "yesno"),
    "on_basal_insulin": ("On basal insulin", "EHR", "yesno"),
    # Digital twin
    "twin_glucose_now": ("Twin's estimate of glucose now", "Twin", "{:.0f} mg/dL"),
    "twin_glucose_plus_30": ("Twin forecast in 30 min", "Twin", "{:.0f} mg/dL"),
    "twin_glucose_plus_60": ("Twin forecast in 1 h", "Twin", "{:.0f} mg/dL"),
    "twin_glucose_plus_120": ("Twin forecast in 2 h", "Twin", "{:.0f} mg/dL"),
    "twin_max_2h": ("Twin's predicted peak (2 h)", "Twin", "{:.0f} mg/dL"),
    "twin_min_2h": ("Twin's predicted low (2 h)", "Twin", "{:.0f} mg/dL"),
    "twin_slope_30": ("Twin's predicted trend (30 min)", "Twin", "{:+.0f} mg/dL"),
    "twin_gut_glucose": ("Food still being absorbed", "Twin", "{:.0f} g"),
    "twin_insulin_action": ("Insulin currently acting", "Twin", "{:.1f}"),
    "twin_log_SI": ("Personal insulin sensitivity", "Twin", "{:.2f}"),
    "twin_log_DI": ("Personal disposition index", "Twin", "{:.2f}"),
    "twin_gb": ("Personal liver glucose drive", "Twin", "{:.0f} mg/dL"),
    "twin_drug_gain": ("Personal medicine response", "Twin", "{:.2f}"),
}


def contributions(model: xgb.XGBClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """Exact TreeSHAP contributions (log-odds units). Last column = baseline."""
    dm = xgb.DMatrix(X, feature_names=list(X.columns))
    c = model.get_booster().predict(dm, pred_contribs=True)
    return pd.DataFrame(c, columns=list(X.columns) + ["_baseline"], index=X.index)


def top_reasons(contrib_row: pd.Series, x_row: pd.Series, k: int = 4):
    """Top-k human-readable reasons for one prediction.

    Time-of-day features (sin and cos) are merged into one reason.
    """
    c = contrib_row.drop("_baseline").copy()
    if "hour_sin" in c and "hour_cos" in c:
        c["hour_sin"] = c["hour_sin"] + c["hour_cos"]
        c = c.drop("hour_cos")
    order = c.abs().sort_values(ascending=False).index[:k]
    out = []
    for f in order:
        name, source, fmt = FEATURES.get(f, (f, "", "{}"))
        if f == "hour_sin":
            h = (np.degrees(np.arctan2(x_row["hour_sin"], x_row["hour_cos"])) / 15) % 24
            mins = int(round(h * 60 / 5) * 5) % 1440
            value = f"{mins // 60:02d}:{mins % 60:02d}"
        else:
            v = x_row[f]
            if pd.isna(v):
                value = "n/a"
            elif fmt == "yesno":
                value = "Yes" if v >= 0.5 else "No"
            elif fmt == "adherence":
                value = {0: "poor", 1: "fair", 2: "good"}.get(int(v), str(v))
            else:
                value = fmt.format(v)
        out.append(dict(feature=f, name=name, source=source, value=value,
                        impact=float(c[f]), direction="raises risk" if c[f] > 0 else "lowers risk"))
    return out


def source_of(feature: str) -> str:
    return FEATURES.get(feature, ("", "Other", ""))[1]
