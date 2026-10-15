"""Synthetic Electronic Health Records for Indian Type 2 Diabetes patients.

Why not Synthea? Synthea's disease modules are calibrated on the US
(Massachusetts) population. Indian T2D patients differ in important ways:
diabetes appears at a younger age and at a lower BMI (the "thin-fat"
phenotype), diets are carbohydrate heavy, and medicine use differs. So we
generate an India-calibrated cohort ourselves, in a Synthea-like table
layout (patients / conditions / observations / medications / genetics).

Each patient also gets hidden "twin parameters" (insulin sensitivity,
insulin secretion, etc.). These drive the physiology simulator and are
saved only as ground truth: the ML model never sees them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIONS = ["South", "North", "West", "East"]
REGION_P = [0.30, 0.30, 0.22, 0.18]

FIRST_NAMES_M = ["Arjun", "Ravi", "Suresh", "Vikram", "Karthik", "Rahul", "Amit",
                 "Sanjay", "Prakash", "Manoj", "Ganesh", "Rajesh", "Anil", "Deepak",
                 "Srinivas", "Harish", "Naveen", "Mohan", "Venkat", "Imran"]
FIRST_NAMES_F = ["Lakshmi", "Priya", "Anita", "Sunita", "Kavya", "Meena", "Divya",
                 "Pooja", "Geeta", "Revathi", "Shalini", "Asha", "Nirmala", "Farah",
                 "Rekha", "Usha", "Swati", "Padma", "Sneha", "Radha"]
SURNAMES = {"South": ["Iyer", "Reddy", "Nair", "Rao", "Pillai", "Subramanian", "Krishnan"],
            "North": ["Sharma", "Singh", "Gupta", "Verma", "Kapoor", "Malhotra", "Khan"],
            "West": ["Patel", "Shah", "Desai", "Joshi", "Kulkarni", "Mehta", "Pawar"],
            "East": ["Das", "Banerjee", "Mukherjee", "Bose", "Ghosh", "Sen", "Mohanty"]}

# Drug catalogue: class -> (generic name, typical dose, daily schedule in hours)
DRUGS = {
    "metformin": ("Metformin", "500 mg", [8.5, 20.5]),
    "sulfonylurea": ("Glimepiride", "2 mg", [8.5]),
    "dpp4": ("Sitagliptin", "100 mg", [8.5]),
    "sglt2": ("Dapagliflozin", "10 mg", [8.5]),
    "basal_insulin": ("Insulin glargine", "14 units", [22.0]),
    "statin": ("Atorvastatin", "10 mg", [21.5]),
    "ace_arb": ("Telmisartan", "40 mg", [8.5]),
}


def _clip(x, lo, hi):
    return float(np.clip(x, lo, hi))


def generate_cohort(n: int, rng: np.random.Generator, start_date: str):
    """Create the static EHR tables and hidden twin parameters."""
    start = pd.Timestamp(start_date)
    patients, conditions, meds, genetics, twins = [], [], [], [], []

    for i in range(n):
        pid = f"GT-{i + 1:04d}"
        sex = rng.choice(["M", "F"], p=[0.55, 0.45])
        region = rng.choice(REGIONS, p=REGION_P)
        age = int(round(_clip(rng.normal(52, 10), 28, 76)))
        first = rng.choice(FIRST_NAMES_M if sex == "M" else FIRST_NAMES_F)
        name = f"{first} {rng.choice(SURNAMES[region])}"

        height = _clip(rng.normal(168 if sex == "M" else 155, 6.5), 145, 190)
        bmi = _clip(rng.normal(26.3, 3.8), 19.0, 40.0)   # lower BMI than Western T2D
        weight = bmi * (height / 100) ** 2
        waist = _clip((88 if sex == "M" else 84) + 2.1 * (bmi - 25) + rng.normal(0, 4), 70, 130)

        # Years since diagnosis, roughly grows with age
        duration = _clip(rng.gamma(2.0, 3.2) * (age / 52), 0.5, 28)

        # ---- Genetics ------------------------------------------------------
        fam_hist = rng.random() < 0.62
        # TCF7L2 rs7903146 T allele: strongest known T2D risk variant, ~0.3 in South Asians
        tcf7l2 = int(rng.binomial(2, 0.30))
        prs = float(rng.normal(0.25 * fam_hist, 1.0))      # polygenic risk score (z)

        # ---- Hidden disease severity (drives physiology and treatment) ----
        severity = (0.55 * rng.normal() + 0.05 * (duration - 6)
                    + 0.08 * (bmi - 26) + 0.15 * tcf7l2 + 0.1 * prs)

        # ---- Hidden twin parameters ----------------------------------------
        SI = 3.2e-4 * np.exp(-0.45 * severity) * (26 / bmi) ** 0.8 * rng.lognormal(0, 0.18)
        beta = 0.028 * np.exp(-0.035 * duration) * np.exp(-0.30 * severity) * rng.lognormal(0, 0.18)
        p1 = 0.022 * rng.lognormal(0, 0.15)                   # glucose effectiveness
        gb_untreated = _clip(156 + 30 * severity + rng.normal(0, 8), 105, 260)
        dawn = _clip(rng.normal(0.08, 0.04), 0.0, 0.2)        # early-morning glucose rise
        adherence = _clip(rng.beta(8, 2), 0.4, 1.0)            # fraction of doses taken
        sleep_sens = _clip(rng.normal(0.05, 0.015), 0.01, 0.1) # SI loss per hour of sleep debt
        twin = dict(patient_id=pid, SI=SI, beta=beta, p1=p1, gb_untreated=gb_untreated,
                    dawn_amp=dawn, adherence=adherence, sleep_sensitivity=sleep_sens,
                    severity=severity, body_weight_kg=weight)

        # ---- Comorbidities -------------------------------------------------
        cond = [("Type 2 diabetes mellitus", "E11", start - pd.Timedelta(days=int(duration * 365)))]
        htn = rng.random() < (0.30 + 0.006 * (age - 40) + 0.015 * (bmi - 25))
        dyslip = rng.random() < 0.55
        neuropathy = rng.random() < (0.05 + 0.02 * duration + 0.04 * max(severity, 0))
        retinopathy = rng.random() < (0.03 + 0.015 * duration)
        ckd = rng.random() < (0.04 + 0.012 * duration + 0.08 * htn)
        obesity = bmi >= 27.5   # Asian-Indian obesity cut-off
        for flag, label, code in [(htn, "Essential hypertension", "I10"),
                                  (dyslip, "Dyslipidaemia", "E78.5"),
                                  (neuropathy, "Diabetic peripheral neuropathy", "E11.42"),
                                  (retinopathy, "Non-proliferative diabetic retinopathy", "E11.329"),
                                  (ckd, "Chronic kidney disease stage 2-3", "N18.3"),
                                  (obesity, "Obesity (Asian BMI cut-off)", "E66.9")]:
            if flag:
                onset = start - pd.Timedelta(days=int(rng.uniform(0.2, max(duration, 0.5)) * 365))
                cond.append((label, code, onset))
        for label, code, onset in cond:
            conditions.append(dict(patient_id=pid, condition=label, icd10=code,
                                   onset_date=onset.date().isoformat()))

        # ---- Treatment plan (chosen by "doctor" from prior control) --------
        prior_a1c = 7.4 + 1.1 * severity + rng.normal(0, 0.4)
        egfr = _clip(rng.normal(92 - 0.6 * (age - 40) - (22 if ckd else 0), 10), 25, 120)
        regimen = {}
        if egfr >= 30 and rng.random() < 0.88:   # some patients cannot tolerate metformin
            regimen["metformin"] = True
        if prior_a1c > 7.8 and rng.random() < 0.6:
            regimen["sulfonylurea"] = True
        if rng.random() < 0.25 + 0.1 * (prior_a1c > 8):
            regimen["dpp4"] = True
        if rng.random() < 0.18 and egfr >= 45:
            regimen["sglt2"] = True
        if prior_a1c > 8.6 and duration > 5 and rng.random() < 0.7:
            regimen["basal_insulin"] = True
        if dyslip or rng.random() < 0.3:
            regimen["statin"] = True
        if htn:
            regimen["ace_arb"] = True
        for cls in regimen:
            gname, dose, sched = DRUGS[cls]
            meds.append(dict(patient_id=pid, drug_class=cls, drug=gname, dose=dose,
                             schedule_hours=";".join(f"{h:g}" for h in sched),
                             start_date=(start - pd.Timedelta(days=int(rng.uniform(60, 900)))).date().isoformat()))
        twin["regimen"] = ";".join(regimen.keys())
        # Treatment calibration: the doctor titrated these drugs to bring glucose
        # down, so the untreated liver glucose drive is higher than what we see.
        # Drug strength is expressed relative to the patient's own insulin
        # sensitivity: the dose was titrated to give a meaningful effect.
        twin["su_push"] = 1.7 * p1 / SI * 0.12 * rng.lognormal(0, 0.2) if "sulfonylurea" in regimen else 0.0
        twin["basal_dose"] = 0.9 * p1 / SI * rng.lognormal(0, 0.2) if "basal_insulin" in regimen else 0.0
        twin["su_push"] = min(twin["su_push"], 9.0)
        twin["basal_dose"] = min(twin["basal_dose"], 75.0)
        if "sulfonylurea" in regimen:
            twin["gb_untreated"] *= 1.06
        if "basal_insulin" in regimen:
            twin["gb_untreated"] *= 1.12
        twin["gb_untreated"] = min(twin["gb_untreated"], 235.0)

        # Self-reported adherence is a noisy view of true adherence
        sr = adherence + rng.normal(0, 0.08)
        adherence_label = "good" if sr > 0.85 else ("fair" if sr > 0.7 else "poor")

        # Lifestyle habits recorded at the clinic visit
        diet = "rice-predominant" if region in ("South", "East") else "wheat-predominant"
        smoker = rng.random() < (0.18 if sex == "M" else 0.02)
        patients.append(dict(
            patient_id=pid, name=name, sex=sex, age=age, region=region,
            height_cm=round(height, 1), weight_kg=round(weight, 1), bmi=round(bmi, 1),
            waist_cm=round(waist, 1), diabetes_duration_yrs=round(duration, 1),
            family_history_t2d=fam_hist, smoker=smoker, diet_pattern=diet,
            self_reported_adherence=adherence_label, egfr=round(egfr, 0),
            hypertension=htn, dyslipidaemia=dyslip, neuropathy=neuropathy,
            retinopathy=retinopathy, ckd=ckd))
        genetics.append(dict(patient_id=pid, TCF7L2_rs7903146_risk_alleles=tcf7l2,
                             polygenic_risk_score=round(prs, 3)))
        twins.append(twin)

    return (pd.DataFrame(patients), pd.DataFrame(conditions), pd.DataFrame(meds),
            pd.DataFrame(genetics), pd.DataFrame(twins))


def make_lab_observations(patients: pd.DataFrame, twins: pd.DataFrame,
                          sim_mean_glucose: np.ndarray, sim_fasting: np.ndarray,
                          rng: np.random.Generator, start_date: str) -> pd.DataFrame:
    """Lab results from the last clinic visit (1-3 months before the wearable period).

    HbA1c is derived from the twin's own simulated average glucose using the
    ADAG formula (eAG = 28.7 x HbA1c - 46.7), so the EHR and the sensor data
    tell a consistent physiological story.
    """
    start = pd.Timestamp(start_date)
    rows = []
    for k, p in patients.iterrows():
        pid = p.patient_id
        visit = start - pd.Timedelta(days=int(rng.uniform(25, 90)))
        a1c = (sim_mean_glucose[k] + 46.7) / 28.7 + rng.normal(0, 0.25)
        fpg = sim_fasting[k] + rng.normal(0, 8)
        ldl = rng.normal(118 if p.dyslipidaemia else 96, 22)
        hdl = rng.normal(40 if p.sex == "M" else 46, 7)
        tg = rng.lognormal(np.log(170 if p.dyslipidaemia else 130), 0.3)
        sbp = rng.normal(142 if p.hypertension else 124, 10)
        dbp = rng.normal(88 if p.hypertension else 79, 7)
        creat = 90 / max(p.egfr, 20) * (1.0 if p.sex == "M" else 0.8)
        uacr = rng.lognormal(np.log(45 if p.ckd else 12), 0.5)
        # Fasting insulin & HOMA-IR reflect hidden insulin resistance (noisy)
        si = twins.loc[k, "SI"]
        f_ins = float(np.clip(8 * (3e-4 / si) ** 0.6 * rng.lognormal(0, 0.2), 2, 45))
        homa = fpg * f_ins / 405
        for test, val, unit in [("HbA1c", a1c, "%"), ("Fasting plasma glucose", fpg, "mg/dL"),
                                ("Fasting insulin", f_ins, "uIU/mL"), ("HOMA-IR", homa, ""),
                                ("LDL cholesterol", ldl, "mg/dL"), ("HDL cholesterol", hdl, "mg/dL"),
                                ("Triglycerides", tg, "mg/dL"), ("Serum creatinine", creat, "mg/dL"),
                                ("eGFR", p.egfr, "mL/min/1.73m2"), ("Urine albumin/creatinine ratio", uacr, "mg/g"),
                                ("Systolic BP", sbp, "mmHg"), ("Diastolic BP", dbp, "mmHg")]:
            rows.append(dict(patient_id=pid, date=visit.date().isoformat(), test=test,
                             value=round(float(val), 2), unit=unit))
        # An older HbA1c, 6 months earlier, gives the doctor a trend
        rows.append(dict(patient_id=pid, date=(visit - pd.Timedelta(days=180)).date().isoformat(),
                         test="HbA1c", value=round(float(a1c + rng.normal(0.2, 0.4)), 2), unit="%"))
    return pd.DataFrame(rows)
