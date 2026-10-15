"""Global settings for the GlucoTwin data generation.

All data in this project is synthetic. No real patient data is used,
in line with the DPDP Act 2023 and HIPAA sandbox rules of the competition.
"""
from pathlib import Path

# ---- Reproducibility -------------------------------------------------------
SEED = 42

# ---- Cohort size and simulation length ------------------------------------
N_PATIENTS = 200          # virtual Type 2 Diabetes patients
N_DAYS = 14               # days of wearable data per patient
START_DATE = "2026-09-01" # first simulated day (00:00 IST)

# ---- Time resolution ------------------------------------------------------
SIM_DT_MIN = 1            # physiology is integrated every 1 minute
CGM_INTERVAL_MIN = 5      # CGM / smartwatch readings every 5 minutes

# ---- Clinical thresholds (mg/dL) ------------------------------------------
HYPER_THRESHOLD = 180     # post-meal spike (ADA time-in-range upper limit)
HYPO_THRESHOLD = 70       # hypoglycaemia (ADA level 1)

# ---- Paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EHR_DIR = DATA_DIR / "ehr"
WEARABLE_DIR = DATA_DIR / "wearables"
LOG_DIR = DATA_DIR / "logs"
TRUTH_DIR = DATA_DIR / "ground_truth"
FIG_DIR = ROOT / "docs" / "figures"
