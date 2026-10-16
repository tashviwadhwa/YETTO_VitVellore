"""GlucoTwin - clinical decision dashboard for the Type 2 diabetes digital twin.

Run from the project root:  streamlit run app.py

Pages
    Patient   Overview . Digital Twin . Predictive Alerts . What-If Simulator
    Population Cohort Explorer . Model Evidence

Every number shown comes from glucotwin/dashboard_data.py (the real pipeline
outputs). This file is presentation only; visual tokens live in ui_kit.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from glucotwin import config as C                      # noqa: E402
from glucotwin import dashboard_data as dd             # noqa: E402
import ui_kit as ui                                    # noqa: E402
from ui_kit import esc                                  # noqa: E402

st.set_page_config(page_title="GlucoTwin", page_icon=":material/monitor_heart:", layout="wide",
                   initial_sidebar_state="expanded")
ui.css()


# ---------------------------------------------------------------------------
# Cached data access (thin wrappers around dashboard_data)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cohort() -> pd.DataFrame:
    return dd.load_cohort_metadata()


@st.cache_data(show_spinner=False)
def get_patient_data(pid: str) -> dict:
    return dd.load_patient_sensors(pid)


@st.cache_resource(show_spinner=False)
def get_models():
    return dd.get_model_service()


@st.cache_data(show_spinner=False)
def get_labs(pid: str) -> pd.DataFrame:
    obs = pd.read_csv(C.EHR_DIR / "observations.csv")
    obs = obs[obs.patient_id == pid].sort_values("date")
    first = obs.groupby("test").first()
    last = obs.groupby("test").last()
    out = pd.DataFrame({"first": first["value"], "first_date": first["date"], "latest": last["value"],
                        "latest_date": last["date"], "unit": last["unit"]})
    return out


@st.cache_data(show_spinner=False)
def get_conditions(pid: str) -> pd.DataFrame:
    c = pd.read_csv(C.EHR_DIR / "conditions.csv")
    return c[c.patient_id == pid].sort_values("onset_date")


@st.cache_data(show_spinner=False)
def get_prescriptions(pid: str) -> pd.DataFrame:
    m = pd.read_csv(C.EHR_DIR / "medications.csv")
    return m[m.patient_id == pid]


@st.cache_data(show_spinner=False)
def run_whatif(pid: str, start_ts: pd.Timestamp, a: tuple, b: tuple) -> dict:
    return dd.simulate_whatif(pid, start_ts, dict(a), dict(b), horizon_minutes=240)


@st.cache_data(show_spinner=False)
def get_moment_state(pid: str, ts: pd.Timestamp) -> dict:
    return dd.moment_state(pid, ts)


@st.cache_data(show_spinner=False)
def read_json(name: str) -> dict:
    return json.loads((C.ROOT / "results" / name).read_text())


cohort = get_cohort()
PIDS = cohort["patient_id"].tolist()
if "pid" not in st.session_state:
    st.session_state.pid = "GT-0029" if "GT-0029" in PIDS else PIDS[0]


def patient() -> pd.Series:
    return cohort[cohort.patient_id == st.session_state.pid].iloc[0]


def fmt(v, spec: str, dash: str = "—") -> str:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return dash
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


PROFILE_KIND = {"Higher-risk profile": "alert", "Moderate profile": "caution", "Lower-risk profile": "good"}


def context_chip(p: pd.Series) -> str:
    """Breadcrumb for patient pages: which patient every number on the page belongs to."""
    return f"Patient &nbsp;/&nbsp; <b>{esc(p['patient_id'])}</b> &nbsp;{esc(p['name'])}"


POP_CRUMB = "Population &nbsp;/&nbsp; <b>200 synthetic patients</b>"


# ---------------------------------------------------------------------------
# Sidebar: brand, patient, identity, navigation
# ---------------------------------------------------------------------------
def sidebar(pages: dict, current) -> None:
    with st.sidebar:
        st.markdown(f"<div class='gt-brand'>{ui.LOGO}<span>GlucoTwin</span></div>", unsafe_allow_html=True)
        st.selectbox("Patient", PIDS, key="pid", label_visibility="collapsed",
                     format_func=lambda x: f"{x}  ·  {cohort.loc[cohort.patient_id == x, 'name'].iloc[0]}")
        p = patient()
        a1c_col = ui.ALERT if p["HbA1c"] > 8.5 else (ui.CAUTION if p["HbA1c"] >= 7.0 else ui.GOOD)
        chg = p.get("hba1c_change", np.nan)
        prof = p["risk_tier"]
        prof_col = {"alert": ui.ALERT, "caution": ui.CAUTION, "good": ui.GOOD}[PROFILE_KIND.get(prof, "good")]
        st.markdown(
            f"<div class='gt-id'><div class='gt-id-name'>{esc(p['name'])}</div>"
            f"<div class='gt-id-meta'>{p['age']} y &middot; {esc(p['sex'])} &middot; {esc(p['region'])} India</div>"
            f"<div class='gt-kv'>"
            f"<span class='k'>HbA1c</span><span class='v'><b style='color:{a1c_col}'>{p['HbA1c']:.1f}%</b>"
            f" <span style='color:{ui.MUTED}'>({fmt(chg, '+.1f')})</span></span>"
            f"<span class='k'>Diabetes</span><span class='v'>{p['diabetes_duration_yrs']:.1f} years</span>"
            f"<span class='k'>BMI</span><span class='v'>{p['bmi']:.1f} kg/m²</span>"
            f"<span class='k'>Profile</span><span class='v' title='{esc(dd.RISK_PROFILE_NOTE)}' "
            f"style='color:{prof_col};cursor:help'>{esc(prof.replace(' profile', ''))}</span>"
            f"</div><div class='gt-meds'>{esc(p.get('medications', 'None'))}</div></div>", unsafe_allow_html=True)

        for group, items in pages.items():
            st.markdown(f"<div class='gt-navlabel'>{group}</div>", unsafe_allow_html=True)
            for pg in items:
                if pg.url_path == current.url_path:      # active page: teal bar + tint, not a link
                    icon = str(pg.icon or "").replace(":material/", "").strip(":")
                    st.markdown(f"<div class='gt-nav-active' aria-current='page'>"
                                f"<span style=\"position:absolute;left:0.65rem;top:0.32rem;font-family:'Material Symbols Rounded';"
                                f"font-size:1.15rem;color:{ui.TEAL};font-weight:400\" aria-hidden='true'>{icon}</span>"
                                f"{esc(pg.title)}</div>", unsafe_allow_html=True)
                else:
                    st.page_link(pg)
        st.markdown("<div class='gt-foot'>Synthetic cohort only. No real patient data (DPDP Act 2023, HIPAA).<br>"
                    "Team YETTO &middot; VIT Vellore</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Shared chart builders
# ---------------------------------------------------------------------------
def day_window(key: str, default=(9, 10)) -> tuple[int, int]:
    return st.select_slider("Days shown", options=list(range(14)), value=default, key=key,
                            format_func=lambda d: (dd.START + pd.Timedelta(days=d)).strftime("%a %d %b"))


def glucose_trace_fig(df: pd.DataFrame, meals: pd.DataFrame, doses: pd.DataFrame, height: int = 360) -> go.Figure:
    fig = go.Figure()
    ui.thresholds(fig)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["glucose_mgdl"], name="CGM (observed)", mode="lines",
                             line=dict(color=ui.NAVY, width=1.6),
                             hovertemplate="%{y:.0f} mg/dL<extra>CGM</extra>"))
    if df["twin_glucose_mgdl"].notna().any():
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["twin_glucose_mgdl"], name="Calibrated twin",
                                 mode="lines", line=dict(color=ui.TEAL, width=1.6, dash="dot"),
                                 hovertemplate="%{y:.0f} mg/dL<extra>Twin</extra>"))
    if len(meals):
        fig.add_trace(go.Scatter(x=meals["timestamp"], y=[48] * len(meals), mode="markers", name="Meal (diary)",
                                 marker=dict(symbol="triangle-up", size=9, color="#A0527A"),
                                 customdata=np.c_[meals["food"], meals["carbs_g"]],
                                 hovertemplate="%{customdata[0]}, %{customdata[1]:.0f} g carbs<extra>Meal</extra>"))
    key_doses = doses[doses["drug_class"].isin(["sulfonylurea", "basal_insulin"])]
    if len(key_doses):
        fig.add_trace(go.Scatter(x=key_doses["timestamp"], y=[42] * len(key_doses), mode="markers",
                                 name="Glimepiride / insulin dose",
                                 marker=dict(symbol=["circle" if t else "circle-open" for t in key_doses["taken"]],
                                             size=7, color=ui.CAUTION, line=dict(width=1.5, color=ui.CAUTION)),
                                 customdata=np.c_[key_doses["drug"], np.where(key_doses["taken"], "taken", "missed")],
                                 hovertemplate="%{customdata[0]}: %{customdata[1]}<extra>Dose</extra>"))
    ui.plotly_base(fig, height, "Glucose (mg/dL)")
    top = np.nanmax([df["glucose_mgdl"].max(), df["twin_glucose_mgdl"].max(), 200])
    fig.update_yaxes(range=[35, top + 20])
    fig.update_xaxes(tickformat="%a %H:%M")
    return fig


# ---------------------------------------------------------------------------
# Page 1 - Patient Overview
# ---------------------------------------------------------------------------
LAB_ORDER = ["HbA1c", "Fasting plasma glucose", "Fasting insulin", "HOMA-IR", "Systolic BP", "Diastolic BP",
             "LDL cholesterol", "HDL cholesterol", "Triglycerides", "Serum creatinine", "eGFR",
             "Urine albumin/creatinine ratio"]
LAB_FMT = {"HbA1c": ".1f", "Serum creatinine": ".2f", "HOMA-IR": ".1f", "Fasting insulin": ".1f"}


def range_bar(agp: dict) -> str:
    vlow = agp["tbr_very_low"]
    low = agp["tbr"] - vlow
    vhigh = agp["tar_very_high"]
    high = agp["tar"] - vhigh
    tir = agp["tir"]
    parts = [("Very low <54", vlow, "#8E2A20"), ("Low 54–69", low, ui.ALERT), ("In range 70–180", tir, ui.GOOD),
             ("High 181–250", high, "#D9A441"), ("Very high >250", vhigh, ui.CAUTION)]
    bar = "".join(f"<span style='width:{max(v, 0):.2f}%;background:{c}' title='{k}: {v:.1f}%'></span>"
                  for k, v, c in parts)
    leg = "".join(f"<span><i style='background:{c}'></i>{k} <b class='num' style='color:{ui.NAVY}'>{v:.1f}%</b></span>"
                  for k, v, c in parts)
    return f"<div class='gt-rangebar' role='img' aria-label='Time in ranges'>{bar}</div><div class='gt-rangelegend'>{leg}</div>"


def page_overview() -> None:
    p = patient()
    ui.page_header("Patient overview", "Health record, recent glucose control and the data this twin is built on.",
                   context_chip(p))
    data = get_patient_data(p["patient_id"])

    # Glucose control first: it is what a clinician looks at before anything else.
    ui.section("Glucose control", "Continuous glucose monitor readings over the selected days. "
               "Targets from the international CGM consensus (Battelino 2019).")
    win = day_window("ov_days", (0, 13))
    ts = data["timeseries"]
    w = ts[(ts.day >= win[0]) & (ts.day <= win[1])]
    agp = dd.compute_agp_metrics(w["glucose_mgdl"])
    if not agp:
        st.info("No CGM readings in this window.")
    else:
        tir_kind = ui.GOOD if agp["tir"] >= 70 else (ui.CAUTION if agp["tir"] >= 50 else ui.ALERT)
        c1, c2 = st.columns([1.15, 1], gap="large")
        with c1:
            st.markdown(
                f"<div class='gt-signal' style='--c:{tir_kind}'><div class='lbl'>Time in range 70–180 mg/dL</div>"
                f"<div class='val'>{agp['tir']:.0f}<small>%</small></div>"
                f"<div class='txt'>Consensus target: more than 70%. Below 70 mg/dL {agp['tbr']:.1f}% "
                f"(target under 4%).</div></div>", unsafe_allow_html=True)
            st.markdown(range_bar(agp), unsafe_allow_html=True)
        with c2:
            ui.figs([
                ("Mean glucose", f"{agp['mean_glucose']:.0f} <small>mg/dL</small>", None),
                ("GMI (estimated A1c)", f"{agp['gmi']:.1f}<small>%</small>", None),
                ("Variability (CV)", f"{agp['cv']:.0f}<small>%</small>", "target 36% or less"),
            ])
    meals, doses = data["meals"], data["doses"]
    t0, t1 = w["timestamp"].min(), w["timestamp"].max()
    st.plotly_chart(glucose_trace_fig(w, meals[meals.timestamp.between(t0, t1)], doses[doses.timestamp.between(t0, t1)],
                                      height=300), config=ui.CONFIG, width="stretch")

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        ui.section("Laboratory results", "First and latest values in the health record.")
        labs = get_labs(p["patient_id"])
        rows = []
        for t in LAB_ORDER:
            if t not in labs.index:
                continue
            r = labs.loc[t]
            f = LAB_FMT.get(t, ".0f")
            changed = r["first_date"] != r["latest_date"]
            delta = r["latest"] - r["first"] if changed else np.nan
            unit = "" if pd.isna(r["unit"]) else r["unit"]
            rows.append(
                f"<tr{' class=primary' if t == 'HbA1c' else ''}><td>{esc(t)}</td>"
                f"<td class='n'>{format(r['latest'], f)} <span class='ref'>{esc(unit)}</span></td>"
                f"<td class='n ref'>{format(r['first'], f) if changed else '—'}</td>"
                f"<td class='n ref'>{fmt(delta, '+' + f)}</td>"
                f"<td class='n ref'>{esc(r['latest_date'])}</td></tr>")
        st.markdown("<table class='gt-table'><tr><th>Test</th><th class='n'>Latest</th><th class='n'>First</th>"
                    "<th class='n'>Change</th><th class='n'>Date</th></tr>" + "".join(rows) + "</table>",
                    unsafe_allow_html=True)
        st.caption("HbA1c target used across the dashboard: below 7.0% (ADA). "
                   f"Status: {p['glycemic_status']}.")
    with right:
        ui.section("Diagnoses")
        cond = get_conditions(p["patient_id"])
        st.markdown("<table class='gt-table'>" + "".join(
            f"<tr><td>{esc(r.condition)}</td><td class='n ref'>{esc(r.icd10)}</td><td class='n ref'>since {esc(r.onset_date[:4])}</td></tr>"
            for r in cond.itertuples()) + "</table>", unsafe_allow_html=True)
        ui.section("Prescriptions")
        rx = get_prescriptions(p["patient_id"])
        if len(rx):
            def sched(s):
                return ", ".join(f"{int(float(h)):02d}:{int(round(float(h) % 1 * 60)):02d}" for h in str(s).split(";"))
            st.markdown("<table class='gt-table'>" + "".join(
                f"<tr><td>{esc(r.drug)}</td><td class='n'>{esc(r.dose)}</td><td class='n ref'>{sched(r.schedule_hours)}</td></tr>"
                for r in rx.itertuples()) + "</table>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='gt-note'>No glucose-lowering medicines prescribed.</div>", unsafe_allow_html=True)
        ui.section("Lifestyle and genetics")
        yn = lambda v: "Yes" if bool(v) else "No"  # noqa: E731
        st.markdown(
            "<table class='gt-table'>"
            f"<tr><td>Diet pattern</td><td class='n'>{esc(str(p['diet_pattern']).replace('-', ' '))}</td></tr>"
            f"<tr><td>Self-reported adherence</td><td class='n'>{esc(p['self_reported_adherence'])}</td></tr>"
            f"<tr><td>Family history of T2D</td><td class='n'>{yn(p['family_history_t2d'])}</td></tr>"
            f"<tr><td>Smoker</td><td class='n'>{yn(p['smoker'])}</td></tr>"
            f"<tr><td>Waist</td><td class='n'>{p['waist_cm']:.0f} cm</td></tr>"
            f"<tr><td>BMI (Asian cut-offs)</td><td class='n'>{p['bmi']:.1f} &middot; {esc(p['asian_bmi_category'])}</td></tr>"
            f"<tr><td>TCF7L2 rs7903146 risk alleles</td><td class='n'>{int(p['TCF7L2_rs7903146_risk_alleles'])}</td></tr>"
            f"<tr><td>Polygenic risk score</td><td class='n'>{p['polygenic_risk_score']:+.2f} SD</td></tr>"
            "</table>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page 2 - Digital Twin
# ---------------------------------------------------------------------------
def page_twin() -> None:
    p = patient()
    ui.page_header("Digital twin", "The patient's personal physiology model, calibrated on week 1 of CGM and checked "
                   "on week 2 without it.", context_chip(p))
    tw = read_json("twin_results.json")
    rec = tw["parameter_recovery_spearman"]

    left, right = st.columns([1.3, 1], gap="large")
    with left:
        ui.section("Calibrated parameters", "Extended Bergman minimal model. Prior = estimate from the health "
                   "record alone, before any CGM data.")
        rows = [
            ("Insulin sensitivity (SI)", f"{p['SI'] * 1e4:.2f}", f"{p['prior_SI'] * 1e4:.2f}", "×10⁻⁴ per µU/mL/min",
             "How strongly insulin lowers glucose"),
            ("Insulin secretion (β)", f"{p['beta']:.3f}", f"{p['prior_beta']:.3f}", "",
             "Pancreatic response to glucose"),
            ("Basal glucose drive (Gb)", f"{p['gb']:.0f}", f"{p['prior_gb']:.0f}", "mg/dL",
             "Liver glucose output at rest"),
            ("Drug response gain", f"{p['drug_gain']:.2f}", "1.00", "×", "Personal scaling of medicine effects"),
        ]
        st.markdown("<table class='gt-table'><tr><th>Parameter</th><th class='n'>Calibrated</th><th class='n'>Prior</th>"
                    "<th>Unit</th><th>Meaning</th></tr>" + "".join(
                        f"<tr><td>{a}</td><td class='n'><b>{b}</b></td><td class='n ref'>{c}</td><td class='ref'>{d}</td>"
                        f"<td class='ref'>{e}</td></tr>" for a, b, c, d, e in rows) + "</table>", unsafe_allow_html=True)
        st.caption(f"β on its own is weakly identifiable from CGM; the disposition index (SI × β) is recovered well "
                   f"(rank correlation with the simulator's truth {rec['Disposition index (SI x beta)']['cal']:.2f}).")
    with right:
        ui.section("Fit quality", "Mean absolute error between twin and CGM.")
        w1, w2 = p["fit_error_week1"], p["error_week2"]
        med = tw["week2_median_error"]["Calibrated twin (GlucoTwin)"]
        ui.figs([("Week 1 · calibration", f"{w1:.1f} <small>mg/dL</small>", "fitted with CGM"),
                 ("Week 2 · validation", f"{w2:.1f} <small>mg/dL</small>", f"CGM hidden · cohort median {med:.1f}")])
        ui.figs([("Disposition index", f"{p['disposition_index'] * 1e6:.1f} <small>×10⁻⁶</small>",
                  "SI × β, insulin action relative to need")])

    ui.section("Fourteen days of fused data", "CGM against the twin's own reconstruction, with smartwatch and diary "
               "streams on the same time axis.")
    data = get_patient_data(p["patient_id"])
    win = day_window("tw_days", (9, 10))
    ts = data["timeseries"]
    w = ts[(ts.day >= win[0]) & (ts.day <= win[1])]
    t0, t1 = w["timestamp"].min(), w["timestamp"].max()
    meals = data["meals"][data["meals"].timestamp.between(t0, t1)]
    doses = data["doses"][data["doses"].timestamp.between(t0, t1)]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035, row_heights=[0.48, 0.2, 0.16, 0.16],
                        subplot_titles=["Glucose (mg/dL)", "Heart rate (bpm) and HRV (ms)", "Steps per 5 min",
                                        "Sleep stage"])
    g = glucose_trace_fig(w, meals, doses)
    for tr in g.data:
        fig.add_trace(tr, row=1, col=1)
    ui.thresholds(fig, row=1, col=1, label=False)
    fig.add_trace(go.Scatter(x=w["timestamp"], y=w["heart_rate_bpm"], name="Heart rate", line=dict(color=ui.ALERT, width=1),
                             hovertemplate="%{y:.0f} bpm<extra>HR</extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=w["timestamp"], y=w["hrv_rmssd_ms"], name="HRV (RMSSD)",
                             line=dict(color=ui.SCEN_A, width=1, dash="dot"),
                             hovertemplate="%{y:.0f} ms<extra>HRV</extra>"), row=2, col=1)
    fig.add_trace(go.Bar(x=w["timestamp"], y=w["steps"], name="Steps", marker_color="#9AA5B1",
                         hovertemplate="%{y:.0f} steps<extra></extra>"), row=3, col=1)
    sleep_map = {"awake": 3, "rem": 2, "light": 1, "deep": 0}
    fig.add_trace(go.Scatter(x=w["timestamp"], y=w["sleep_stage"].map(sleep_map), name="Sleep stage",
                             line=dict(color="#3D6FB6", width=1.2, shape="hv"), connectgaps=False,
                             hovertemplate="%{y}<extra>Sleep</extra>"), row=4, col=1)
    ui.plotly_base(fig, 720)
    fig.update_layout(margin=dict(l=8, r=8, t=30, b=8))
    fig.update_annotations(font=dict(size=12, color=ui.INK_2), x=0, xanchor="left")
    fig.update_yaxes(range=[35, np.nanmax([w["glucose_mgdl"].max(), 200]) + 20], row=1, col=1)
    fig.update_yaxes(tickvals=[0, 1, 2, 3], ticktext=["Deep", "Light", "REM", "Awake"], row=4, col=1)
    fig.update_xaxes(tickformat="%a %H:%M", row=4, col=1)
    st.plotly_chart(fig, config=ui.CONFIG, width="stretch")


# ---------------------------------------------------------------------------
# Page 3 - Predictive Alerts
# ---------------------------------------------------------------------------
MODES = {"Smartwatch + twin (no CGM)": "nocgm", "CGM worn": "cgm"}


def page_alerts() -> None:
    p = patient()
    ui.page_header("Predictive alerts", "Standing at a chosen moment and using only what was known then: "
                   "will glucose go above 180 or below 70 mg/dL in the next 2 hours?", context_chip(p))
    svc = get_models()

    c1, c2, c3 = st.columns([1.45, 1.5, 1], vertical_alignment="bottom")
    with c1:
        mode_lbl = st.segmented_control("Data available", list(MODES), default=list(MODES)[0], key="al_mode",
                                        help="No-CGM mode uses smartwatch, diary, health record and the calibrated "
                                             "twin. It covers week 2, after the twin was calibrated.")
        mode = MODES[mode_lbl or list(MODES)[0]]
    moments = svc.get_patient_moments(p["patient_id"], mode=mode)
    if len(moments) == 0:
        st.info("No prediction moments for this patient in this mode.")
        return
    has_truth = "spike_2h" in moments.columns
    with c3:
        only_spikes = st.toggle("Only moments before a spike", value=False, key="al_only",
                                help="Uses the replayed outcome to find interesting moments quickly.") if has_truth else False
    idx = moments.index[moments["spike_2h"] == 1].tolist() if (has_truth and only_spikes) else list(moments.index)
    if not idx:
        st.info("No moment in this mode was followed by a spike.")
        return
    labels = moments["timestamp"].dt.strftime("%a %d %b · %H:%M")
    default = moments.index[moments["spike_2h"] == 1][0] if has_truth and (moments["spike_2h"] == 1).any() else idx[0]
    with c2:
        sel = st.selectbox("Prediction moment", idx, index=idx.index(default) if default in idx else 0,
                           format_func=lambda i: labels[i], key=f"al_moment_{p['patient_id']}_{mode}_{only_spikes}")
    row = moments.loc[sel]
    with st.spinner("Scoring this moment"):
        r = svc.predict_moment(row, mode=mode)

    # --- the two signals and the twin forecast, side by side -------------
    s1, s2, s3 = st.columns([1, 1, 1.15], gap="large")
    score = r["spike_score"]
    col = ui.ALERT if r["spike_level"] == "likely" else (ui.CAUTION if r["spike_level"] == "possible" else ui.GOOD)
    title = {"likely": "Spike likely", "possible": "Spike possible", "unlikely": "Spike unlikely"}[r["spike_level"]]
    truth = ""
    if has_truth:
        truth = ("<br><span style='color:%s'>Replay check, not known at the time: %s.</span>"
                 % (ui.MUTED, "a spike did follow" if int(row["spike_2h"]) == 1 else "no spike followed"))
    th = r["hypo_threshold"]
    with s1:
        ui.section("Above 180 mg/dL")
        st.markdown(
            f"<div class='gt-signal' style='--c:{col}'><div class='lbl'>{title}</div>"
            f"<div class='val'>{score:.0%} <small>spike score</small></div>"
            f"<div class='txt'>{dd.SPIKE_LIKELY:.0%}+ likely, {dd.SPIKE_POSSIBLE:.0%}–{dd.SPIKE_LIKELY:.0%} possible. "
            f"A ranking score from XGBoost, not a calibrated probability.{truth}</div></div>", unsafe_allow_html=True)
    with s2:
        ui.section("Below 70 mg/dL")
        hcol = ui.ALERT if r["hypo_alert"] else ui.GOOD
        head = (f"Hypo predicted, lowest at {r['hypo_nadir_time']:%H:%M}" if r["hypo_alert"] else "No hypo predicted")
        st.markdown(
            f"<div class='gt-signal' style='--c:{hcol}'><div class='lbl'>{head}</div>"
            f"<div class='val'>{r['hypo_nadir']:.0f} <small>mg/dL forecast low</small></div>"
            f"<div class='txt'>Alerts when the twin's 2-hour forecast drops below {th:.0f} mg/dL. In backtesting it "
            f"caught {r['hypo_rule_sensitivity']:.0%} of hypos; {r['hypo_rule_precision']:.0%} of alerts were "
            f"followed by one.</div></div>", unsafe_allow_html=True)
    with s3:
        ui.section("Twin forecast, next 2 h")
        fc = r["twin_forecast"]
        x = [row["timestamp"] + pd.Timedelta(minutes=i + 1) for i in range(len(fc))]
        f2 = go.Figure()
        f2.add_hline(y=th, line=dict(color=ui.ALERT, width=1, dash="dot"), annotation_text=f"{th:.0f}",
                     annotation_position="right", annotation_font=dict(size=11, color=ui.ALERT))
        f2.add_hline(y=180, line=dict(color=ui.CAUTION, width=1, dash="dot"), annotation_text="180",
                     annotation_position="right", annotation_font=dict(size=11, color=ui.CAUTION))
        f2.add_trace(go.Scatter(x=x, y=fc, mode="lines", line=dict(color=ui.TEAL, width=2.2), name="Twin forecast",
                                hovertemplate="%{x|%H:%M}  %{y:.0f} mg/dL<extra></extra>"))
        ui.plotly_base(f2, 165, None, legend=False)
        f2.update_layout(hovermode="closest", margin=dict(l=4, r=36, t=6, b=4))
        f2.update_yaxes(range=[min(50, fc.min() - 10), max(200, fc.max() + 10)], nticks=4)
        f2.update_xaxes(tickformat="%H:%M", nticks=5)
        st.plotly_chart(f2, config=ui.CONFIG, width="stretch")
        st.caption("Assumes no new meal or walk; scheduled medicines taken.")

    # --- why --------------------------------------------------------------
    ui.section("What drives the spike score", "TreeSHAP contributions for this moment. These are associations the "
               "model learned, not proven causes.")
    rl, rr = st.columns([1, 1.15], gap="large")
    reasons = r["reasons"]
    with rl:
        html = ""
        for x_ in reasons:
            c = ui.SOURCE_COLORS.get(x_["source"], ui.MUTED)
            up = x_["direction"] == "raises risk"
            html += (f"<div class='gt-reason'><div><span class='nm'>{esc(x_['name'])}</span>"
                     f"<span class='src' style='color:{c}'>{esc(x_['source'])}</span>"
                     f"<div class='vl'>value: {esc(x_['value'])}</div></div>"
                     f"<div class='dir' style='color:{ui.ALERT if up else ui.GOOD}'>{'raises' if up else 'lowers'}"
                     f"<br><span style='color:{ui.MUTED}'>{x_['impact']:+.2f}</span></div></div>")
        st.markdown(html, unsafe_allow_html=True)
    with rr:
        df = pd.DataFrame(reasons).iloc[::-1]
        f3 = go.Figure(go.Bar(x=df["impact"], y=df["name"], orientation="h",
                              marker_color=[ui.ALERT if v > 0 else ui.GOOD for v in df["impact"]],
                              customdata=np.c_[df["source"], df["value"]],
                              hovertemplate="%{y}<br>%{customdata[0]} · value %{customdata[1]}<br>"
                                            "contribution %{x:+.2f}<extra></extra>"))
        ui.plotly_base(f3, 330, legend=False)
        f3.update_layout(hovermode="closest")
        f3.add_vline(x=0, line=dict(color=ui.LINE_STRONG, width=1))
        f3.update_xaxes(title=dict(text="Contribution to spike score (log-odds)", font=dict(size=11)), showgrid=True,
                        gridcolor="#EEF0F3")
        f3.update_yaxes(showgrid=False)
        st.plotly_chart(f3, config=ui.CONFIG, width="stretch")
        src = r["source_breakdown"]
        if src:
            st.markdown(f"<div class='gt-colhead'>Share of the explanation by data stream</div>", unsafe_allow_html=True)
            items = sorted(src.items(), key=lambda kv: -kv[1])
            bar = "".join(f"<span style='width:{v:.2f}%;background:{ui.SOURCE_COLORS.get(k, ui.MUTED)}' "
                          f"title='{k} {v:.0f}%'></span>" for k, v in items)
            leg = "".join(f"<span><i style='background:{ui.SOURCE_COLORS.get(k, ui.MUTED)}'></i>{k} "
                          f"<b class='num' style='color:{ui.NAVY}'>{v:.0f}%</b></span>" for k, v in items)
            st.markdown(f"<div class='gt-rangebar'>{bar}</div><div class='gt-rangelegend'>{leg}</div>",
                        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page 4 - What-If Simulator (core experience)
# ---------------------------------------------------------------------------
DISH_OPTIONS = [dd.NO_MEAL] + sorted(dd.DISHES)
DEFAULTS = {
    "a": dict(name="Usual lunch", dish="Rice, sambar and poriyal", walk=0, delay=15, meal_in=15, glim="as prescribed"),
    "b": dict(name="Millet + walk", dish="Millet meals", walk=20, delay=15, meal_in=15, glim="as prescribed"),
}
DAYS = list(pd.date_range(dd.START + pd.Timedelta(days=1), periods=13, freq="D"))
SLOTS = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]
WI_PREFIX = ("wi_a_", "wi_b_")


def _reset_scenarios() -> None:
    for k in list(st.session_state.keys()):
        if k.startswith(WI_PREFIX):
            del st.session_state[k]


def _plan_row(title: str, unit: str | None, modified: bool, render) -> dict:
    """One setting, compared across the two plans: label line, then A | B controls."""
    dot = "<i class='mod' title='Changed from the default'></i>" if modified else ""
    st.markdown(f"<div class='gt-rowlbl'>{esc(title)}" + (f"<small>{esc(unit)}</small>" if unit else "") + f"{dot}</div>",
                unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        va = render("a")
    with cb:
        vb = render("b")
    return {"a": va, "b": vb}


def _short(s: str, n: int = 22) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _end_labels(fig: go.Figure, x_end, items, span: float) -> None:
    """Label each curve at its right end instead of a legend; nudge labels apart."""
    items = sorted(items, key=lambda t: t[0])
    gap = span * 0.075
    ys = []
    for y, _, _ in items:
        ys.append(max(y, ys[-1] + gap) if ys else y)
    for (y0, text, color), y in zip(items, ys):
        fig.add_annotation(x=x_end, y=y, text=text, showarrow=False, xanchor="left", xshift=8,
                           font=dict(size=12, color=color, family="Instrument Sans, sans-serif"), align="left")


def page_whatif() -> None:
    p = patient()
    pid = p["patient_id"]
    on_su = bool(p.get("on_glimepiride", False))
    ui.page_header("What-if simulator", "Compare two plans for the next 4 hours on this patient's calibrated twin.",
                   context_chip(p))

    # ---- 1. starting point: one quiet strip ---------------------------------
    with st.container(key="wi_context"):
        c1, c2, c3 = st.columns([0.95, 0.75, 3.6], gap="medium", vertical_alignment="bottom")
        with c1:
            day = st.selectbox("Start day", DAYS, index=9, format_func=lambda d: d.strftime("%a %d %b"), key="wi_day")
        with c2:
            clock = st.selectbox("Start time", SLOTS, index=SLOTS.index("12:15"), key="wi_time")
        start_ts = pd.Timestamp(f"{day.date()} {clock}")
        s = get_moment_state(pid, start_ts)
        with c3:
            cgm_txt = f"{s['cgm']:.0f} <small>mg/dL</small>" if np.isfinite(s["cgm"]) else "<small>no reading</small>"
            ui.figs([("CGM at start", cgm_txt, None),
                     ("Twin estimate", f"{s['twin']:.0f} <small>mg/dL</small>", None),
                     ("Carbs still absorbing", f"{s['food_in_gut_g']:.0f} <small>g</small>", None),
                     ("Last logged meal", f"<span style='font-size:0.88rem;font-weight:500'>"
                                          f"{esc(s['last_meal'] or 'none logged')}</span>", None)])

    ws = st.container(key="wi_ws")
    left, right = ws.columns([1, 1.6], gap="large")

    # ---- 2. the two plans ---------------------------------------------------
    with left:
        with st.container(key="wi_plans"):
            ui.label("Plans")
            names = {}
            h_a, h_b = st.columns(2)
            for col, sc in ((h_a, "a"), (h_b, "b")):
                with col:
                    st.markdown(f"<div class='gt-planhead'><span class='gt-letter {sc}'>{sc.upper()}</span>"
                                f"{'Comparison' if sc == 'a' else 'Alternative'}</div>", unsafe_allow_html=True)
                    names[sc] = st.text_input(f"Name of plan {sc.upper()}", DEFAULTS[sc]["name"], max_chars=40,
                                              key=f"wi_{sc}_name", label_visibility="collapsed")
            ss = st.session_state

            dish = _plan_row("Meal", None,
                             any(ss.get(f"wi_{x}_dish", DEFAULTS[x]["dish"]) != DEFAULTS[x]["dish"] for x in "ab"),
                             lambda x: st.selectbox(f"Meal, plan {x.upper()}", DISH_OPTIONS,
                                                    index=DISH_OPTIONS.index(DEFAULTS[x]["dish"]), key=f"wi_{x}_dish",
                                                    label_visibility="collapsed"))

            def carbs_input(x):
                d = dish[x]
                dflt = dd.dish_carbs(d)
                return st.number_input(f"Carbohydrate, plan {x.upper()} (g)", 0.0, 200.0, float(dflt), 5.0,
                                       key=f"wi_{x}_carbs_{d}", disabled=d == dd.NO_MEAL, format="%.0f",
                                       label_visibility="collapsed",
                                       help=f"Standard portion of this dish: {dflt:.0f} g. Allowed 0–200 g.")
            carbs = _plan_row("Carbohydrate", "g",
                              any(abs(ss.get(f"wi_{x}_carbs_{dish[x]}", dd.dish_carbs(dish[x])) - dd.dish_carbs(dish[x]))
                                  > 0.01 for x in "ab"), carbs_input)

            meal_in = _plan_row("Meal time", "min after start",
                                any(ss.get(f"wi_{x}_mealin", DEFAULTS[x]["meal_in"]) != DEFAULTS[x]["meal_in"] for x in "ab"),
                                lambda x: st.number_input(f"Meal time, plan {x.upper()} (min after start)", 0, 120,
                                                          DEFAULTS[x]["meal_in"], 15, key=f"wi_{x}_mealin",
                                                          disabled=dish[x] == dd.NO_MEAL, label_visibility="collapsed",
                                                          help="0–120 minutes, in 15-minute steps."))

            walk = _plan_row("Walk after meal", "min · 0 = none",
                             any(ss.get(f"wi_{x}_walk", DEFAULTS[x]["walk"]) != DEFAULTS[x]["walk"] for x in "ab"),
                             lambda x: st.number_input(f"Walk duration, plan {x.upper()} (min)", 0, 60, DEFAULTS[x]["walk"],
                                                       5, key=f"wi_{x}_walk", label_visibility="collapsed",
                                                       help="0–60 minutes. The twin converts walking to about "
                                                            "100 steps a minute."))

            glim = _plan_row("Glimepiride", None if on_su else "not prescribed",
                             any(ss.get(f"wi_{x}_glim", "as prescribed") != "as prescribed" for x in "ab"),
                             lambda x: st.selectbox(f"Glimepiride, plan {x.upper()}", dd.GLIMEPIRIDE_OPTIONS,
                                                    key=f"wi_{x}_glim", disabled=not on_su,
                                                    label_visibility="collapsed"))

            with st.expander("Walk timing"):
                delay = _plan_row("Walk starts", "min after the meal (after start if no meal)",
                                  any(ss.get(f"wi_{x}_delay", DEFAULTS[x]["delay"]) != DEFAULTS[x]["delay"] for x in "ab"),
                                  lambda x: st.number_input(f"Walk start delay, plan {x.upper()} (min)", 0, 60,
                                                            DEFAULTS[x]["delay"], 5, key=f"wi_{x}_delay",
                                                            disabled=walk[x] == 0, label_visibility="collapsed"))

            cfg = {}
            for x in "ab":
                has_meal = dish[x] != dd.NO_MEAL
                cfg[x] = dict(name=(names[x].strip() or f"Plan {x.upper()}"), dish=dish[x],
                              carbs_g=float(carbs[x]) if has_meal else 0.0, meal_in_min=int(meal_in[x]),
                              walk_min=int(walk[x]), walk_delay_min=int(delay[x]),
                              glimepiride=glim[x] if on_su else "as prescribed")
            current = dict(pid=pid, start=start_ts, a=tuple(cfg["a"].items()), b=tuple(cfg["b"].items()))
            applied = ss.get("wi_applied")
            if applied is None or applied["pid"] != pid:      # first visit or new patient: show a result at once
                ss.wi_applied = applied = current

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            b1, b2 = st.columns([1.5, 1], vertical_alignment="center")
            with b1:
                run = st.button("Run simulation", type="primary", key="wi_run", width="stretch",
                                icon=":material/play_arrow:")
            with b2:
                st.button("Reset to defaults", type="tertiary", key="wi_reset", on_click=_reset_scenarios)
            if run:
                ss.wi_applied = applied = current
            stale = applied != current
            if stale:
                st.markdown(f"<div class='gt-status'><i style='background:{ui.CAUTION}'></i>Settings changed. "
                            "Run the simulation to update the prediction.</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='gt-status'><i style='background:{ui.GOOD}'></i>Prediction matches these "
                            "settings.</div>", unsafe_allow_html=True)

    # ---- 3. the prediction ----------------------------------------------------
    try:
        with st.spinner("Running the twin forward 4 hours"):
            res = run_whatif(applied["pid"], applied["start"], applied["a"], applied["b"])
    except Exception as e:  # keep the page usable and say what went wrong
        with right:
            st.error(f"The simulation could not run for these settings: {e}")
        return
    A, B = dict(applied["a"]), dict(applied["b"])
    na, nb = res["name_a"], res["name_b"]
    pa, pb = res["peak_baseline"], res["peak_whatif"]
    ha_, hb_ = res["hyper_min_baseline"], res["hyper_min_whatif"]
    la, lb = res["min_baseline"], res["min_whatif"]
    ya, yb = res["hypo_min_baseline"], res["hypo_min_whatif"]

    # focal message: time below 70 outranks the peak whenever either plan goes low
    if ya or yb:
        focus = "hypo"
        if yb != ya:
            better = yb < ya
            head = (f"{esc(nb)} is predicted to spend <em>{abs(ya - yb)} {'fewer' if better else 'more'} minutes "
                    f"below 70 mg/dL</em> than {esc(na)}.")
            hc = ui.GOOD if better else ui.ALERT
        else:
            head, hc = f"Both plans are predicted to spend <em>{ya} minutes below 70 mg/dL</em>.", ui.ALERT
    else:
        focus = "peak"
        d = pa - pb
        if abs(d) > 5:
            head = (f"{esc(nb)} is predicted to peak <em>{abs(d):.0f} mg/dL {'lower' if d > 0 else 'higher'}</em> "
                    f"than {esc(na)}.")
            hc = ui.GOOD if d > 0 else ui.ALERT
        else:
            head, hc = f"Both plans are predicted to reach <em>a similar peak</em> ({pa:.0f} vs {pb:.0f} mg/dL).", ui.NAVY

    def diff_cell(a, b, unit, lower_is_better):
        d_ = b - a
        if abs(d_) < 0.5:
            return "<td class='n ref'>same</td>"
        good = (d_ < 0) == lower_is_better
        return (f"<td class='n' style='color:{ui.GOOD if good else ui.ALERT};font-weight:600'>{d_:+.0f} "
                f"<span class='ref'>{unit}</span></td>")

    rows = [("peak", "Peak glucose", f"{pa:.0f}", f"{pb:.0f}", "mg/dL", diff_cell(pa, pb, "mg/dL", True)),
            ("tar", "Time above 180 mg/dL", f"{ha_}", f"{hb_}", "min", diff_cell(ha_, hb_, "min", True)),
            ("low", "Lowest glucose", f"{la:.0f}", f"{lb:.0f}", "mg/dL", diff_cell(la, lb, "mg/dL", False)),
            ("hypo", "Time below 70 mg/dL", f"{ya}", f"{yb}", "min", diff_cell(ya, yb, "min", True))]
    tbl = ("<table class='gt-table'><tr><th>Next 4 hours</th>"
           f"<th class='n'><span class='gt-key dash' style='border-color:{ui.SCEN_A}'></span>A</th>"
           f"<th class='n'><span class='gt-key' style='border-color:{ui.SCEN_B}'></span>B</th><th class='n'>B vs A</th></tr>")
    for key, lbl, va, vb, unit, dc in rows:
        cls = " class='primary'" if key == focus else ""
        tbl += (f"<tr{cls}><td>{lbl}</td><td class='n'>{va} <span class='ref'>{unit}</span></td>"
                f"<td class='n'>{vb} <span class='ref'>{unit}</span></td>{dc}</tr>")
    tbl += "</table>"

    # chart: three curves, the gap between A and B shaded, labels at the line ends
    x = res["timestamps"]
    ga, gb, gp = (np.asarray(res[k]) for k in ("glucose_baseline", "glucose_whatif", "glucose_plan"))
    top = float(max(ga.max(), gb.max(), gp.max()))
    low = float(min(ga.min(), gb.min(), gp.min()))
    y0, y1 = min(50, low - 10), max(200, top + 25)
    fig = go.Figure()
    ui.thresholds(fig, band=False)
    fig.add_trace(go.Scatter(x=x, y=gp, name="No new meal or walk", mode="lines",
                             line=dict(color=ui.PLAN, width=1.4, dash="dot"),
                             hovertemplate="%{y:.0f} mg/dL<extra>No new meal or walk</extra>"))
    fig.add_trace(go.Scatter(x=x, y=ga, name=f"A · {na}", mode="lines", line=dict(color=ui.SCEN_A, width=2, dash="dash"),
                             hovertemplate="%{y:.0f} mg/dL<extra>A</extra>"))
    fig.add_trace(go.Scatter(x=x, y=gb, name=f"B · {nb}", mode="lines", line=dict(color=ui.SCEN_B, width=2.8),
                             fill="tonexty", fillcolor="rgba(14,133,132,0.09)",
                             hovertemplate="%{y:.0f} mg/dL<extra>B</extra>"))
    for sc, g_, color in (("a", ga, ui.SCEN_A), ("b", gb, ui.SCEN_B)):
        c_ = A if sc == "a" else B
        has_meal = c_["dish"] != dd.NO_MEAL and c_["carbs_g"] > 0
        meal_t = c_["meal_in_min"] if has_meal else 0
        ev = []
        if has_meal:
            ev.append((min(max(meal_t - 1, 0), len(g_) - 1), "triangle-up"))
        if c_["walk_min"] > 0 and meal_t + c_["walk_delay_min"] < len(g_):
            ev.append((max(meal_t + c_["walk_delay_min"] - 1, 0), "square"))
        k = int(np.argmax(g_))
        peak_inside = k < len(g_) - 20          # a peak at the very end is just "still rising"; the end label covers it
        if peak_inside:
            ev.append((k, "circle"))
        fig.add_trace(go.Scatter(x=[x[i] for i, _ in ev], y=[g_[i] for i, _ in ev], mode="markers", showlegend=False,
                                 hoverinfo="skip", marker=dict(symbol=[m for _, m in ev], size=8, color="white",
                                                               line=dict(color=color, width=1.8))))
        if peak_inside:
            fig.add_annotation(x=x[k], y=g_[k], text=f"{g_[k]:.0f}", showarrow=False, yshift=12,
                               font=dict(size=11, color=color))
    _end_labels(fig, x[-1], [(ga[-1], f"A {ga[-1]:.0f} · {esc(_short(na, 16))}", ui.SCEN_A),
                             (gb[-1], f"B {gb[-1]:.0f} · {esc(_short(nb, 16))}", ui.SCEN_B),
                             (gp[-1], f"{gp[-1]:.0f} · no new meal", ui.MUTED)], y1 - y0)
    ui.plotly_base(fig, 360, None, legend=False)
    fig.update_layout(margin=dict(l=4, r=150, t=22, b=4))
    fig.update_yaxes(range=[y0, y1], dtick=50, ticksuffix="  ")
    fig.update_xaxes(range=[x[0], x[-1]], tickformat="%H:%M", showspikes=True, spikemode="across", spikethickness=1,
                     spikecolor=ui.LINE_STRONG, spikedash="solid")

    with right:
        with st.container(key="wi_out_stale" if stale else "wi_out"):
            st.markdown(f"<div class='gt-label'>Hypothetical prediction &nbsp;·&nbsp; "
                        f"4 h from {applied['start']:%a %d %b, %H:%M}</div>"
                        f"<div class='gt-headline' style='--hc:{hc}'>{head}</div>", unsafe_allow_html=True)
            st.plotly_chart(fig, config=ui.CONFIG, width="stretch")
            st.markdown(f"<div class='gt-note' style='font-size:0.76rem;color:{ui.MUTED};margin:-0.4rem 0 0.4rem 0'>"
                        "mg/dL &nbsp;·&nbsp; △ meal &nbsp; □ walk starts &nbsp; ○ peak &nbsp;·&nbsp; "
                        "shading marks the gap between A and B</div>",
                        unsafe_allow_html=True)
            st.markdown(tbl, unsafe_allow_html=True)

    # ---- 4. interpretation: only what the table and chart do not already say ----------
    diffs = []

    def meal_txt(c_):
        return "no meal" if c_["dish"] == dd.NO_MEAL or c_["carbs_g"] <= 0 else f"{c_['dish']}, {c_['carbs_g']:.0f} g"
    if meal_txt(A) != meal_txt(B):
        diffs.append(f"Meal: {meal_txt(A)} (A) vs {meal_txt(B)} (B)")
    if A["meal_in_min"] != B["meal_in_min"] and dd.NO_MEAL not in (A["dish"], B["dish"]):
        diffs.append(f"Meal time: +{A['meal_in_min']} vs +{B['meal_in_min']} min")

    def walk_txt(c_):
        return "none" if not c_["walk_min"] else f"{c_['walk_min']} min, starting {c_['walk_delay_min']} min after the meal"
    if walk_txt(A) != walk_txt(B):
        diffs.append(f"Walk: {walk_txt(A)} (A) vs {walk_txt(B)} (B)")
    if A["glimepiride"] != B["glimepiride"]:
        diffs.append(f"Glimepiride: {A['glimepiride']} (A) vs {B['glimepiride']} (B)")
    caveats = [s_.strip() for s_ in res["clinical_note"].replace("⚠️ ", "").split(". ")
               if "glimepiride" in s_.lower()]
    n1, n2 = st.columns([1, 1.6], gap="large")
    with n1:
        ui.section("What differs between the plans")
        st.markdown("<div class='gt-note'>" + ("<ul>" + "".join(f"<li>{esc(d_)}</li>" for d_ in diffs) + "</ul>"
                                               if diffs else "The two plans are identical.") + "</div>",
                    unsafe_allow_html=True)
    with n2:
        ui.section("Reading this prediction")
        extra = "".join(f"<li>{esc(c.rstrip('.'))}.</li>" for c in caveats)
        st.markdown("<div class='gt-note'><ul>"
                    "<li>Simulated on this patient's calibrated twin from its saved state; not a recorded outcome "
                    "and not a treatment recommendation.</li>"
                    "<li>Assumes nothing else is eaten and no other walk happens in the 4 hours; scheduled medicines "
                    "are taken as prescribed unless a plan changes glimepiride.</li>"
                    f"{extra}<li>The twin predicts; it does not prove cause.</li></ul></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page 5 - Cohort Explorer
# ---------------------------------------------------------------------------
BMI_COLORS = {"Normal (18.5–22.9)": ui.GOOD, "Overweight (23.0–24.9)": ui.CAUTION, "Obese (≥25.0)": ui.ALERT,
              "Underweight (<18.5)": ui.MUTED}


def page_cohort() -> None:
    ui.page_header("Cohort explorer", "All 200 synthetic Indian patients with type 2 diabetes: phenotype, "
                   "treatment and how well each twin fits.", POP_CRUMB)
    df = cohort
    with st.expander("Filters", icon=":material/tune:"):
        f1, f2, f3, f4 = st.columns(4)
        regions = sorted(df["region"].unique())
        with f1:
            region_f = st.multiselect("Region", regions, default=regions, key="co_region")
        with f2:
            sex_f = st.multiselect("Sex", sorted(df["sex"].unique()), default=sorted(df["sex"].unique()), key="co_sex")
        with f3:
            lo, hi = float(np.floor(df["HbA1c"].min() * 10) / 10), float(np.ceil(df["HbA1c"].max() * 10) / 10)
            a1c = st.slider("HbA1c (%)", lo, hi, (lo, hi), step=0.1, key="co_a1c")
        with f4:
            blo, bhi = float(np.floor(df["bmi"].min())), float(np.ceil(df["bmi"].max()))
            bmi = st.slider("BMI (kg/m²)", blo, bhi, (blo, bhi), step=0.5, key="co_bmi")
    f = df[df["region"].isin(region_f) & df["sex"].isin(sex_f) & df["HbA1c"].between(*a1c) & df["bmi"].between(*bmi)]
    if len(f) == 0:
        st.info("No patients match these filters.")
        return

    ui.figs([("Patients shown", f"{len(f)} <small>of {len(df)}</small>", None),
             ("Median HbA1c", f"{f['HbA1c'].median():.1f}<small>%</small>", f"SD {f['HbA1c'].std():.1f}" if len(f) > 1 else None),
             ("Median BMI", f"{f['bmi'].median():.1f} <small>kg/m²</small>", "Asian cut-offs 23 / 25"),
             ("Women / men", f"{(f['sex'] == 'F').mean() * 100:.0f} / {(f['sex'] == 'M').mean() * 100:.0f} <small>%</small>", None),
             ("Median twin error", f"{f['error_week2'].median():.1f} <small>mg/dL</small>", "week 2, CGM hidden")])

    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.section("BMI and HbA1c", "Many patients have diabetes below the general BMI cut-off of 25.")
        fig = go.Figure()
        for cat, color in BMI_COLORS.items():
            s = f[f["asian_bmi_category"] == cat]
            if len(s):
                fig.add_trace(go.Scatter(x=s["bmi"], y=s["HbA1c"], mode="markers", name=cat,
                                         marker=dict(color=color, size=8, opacity=0.8, line=dict(color="white", width=0.5)),
                                         customdata=np.c_[s["patient_id"], s["name"], s["age"]],
                                         hovertemplate="%{customdata[1]} (%{customdata[0]}), %{customdata[2]} y<br>"
                                                       "BMI %{x:.1f} · HbA1c %{y:.1f}%<extra></extra>"))
        for xv in (23, 25):
            fig.add_vline(x=xv, line=dict(color=ui.LINE_STRONG, width=1, dash="dot"))
        fig.add_hline(y=7.0, line=dict(color=ui.GOOD, width=1, dash="dot"), annotation_text="HbA1c 7.0% target",
                      annotation_position="bottom right", annotation_font=dict(size=11, color=ui.GOOD))
        ui.plotly_base(fig, 360, "HbA1c (%)")
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(title=dict(text="BMI (kg/m²)", font=dict(size=12)))
        st.plotly_chart(fig, config=ui.CONFIG, width="stretch")
    with c2:
        ui.section("Disposition index", "SI × β from each calibrated twin: insulin action relative to need.")
        fig = go.Figure(go.Histogram(x=f["disposition_index"] * 1e6, nbinsx=25, marker_color=ui.TEAL,
                                     marker_line=dict(color="white", width=1),
                                     hovertemplate="%{x} ×10⁻⁶: %{y} patients<extra></extra>"))
        sel = patient()
        if sel["patient_id"] in set(f["patient_id"]):
            fig.add_vline(x=sel["disposition_index"] * 1e6, line=dict(color=ui.NAVY, width=1.5),
                          annotation_text=sel["patient_id"], annotation_position="top",
                          annotation_font=dict(size=11, color=ui.NAVY))
        ui.plotly_base(fig, 360, "Patients", legend=False)
        fig.update_layout(hovermode="closest", bargap=0.02)
        fig.update_xaxes(title=dict(text="Disposition index (×10⁻⁶)", font=dict(size=12)))
        st.plotly_chart(fig, config=ui.CONFIG, width="stretch")

    ui.section("Patients")
    cols = ["patient_id", "name", "age", "sex", "region", "bmi", "asian_bmi_category", "diabetes_duration_yrs", "HbA1c",
            "glycemic_status", "Fasting plasma glucose", "HOMA-IR", "eGFR", "medications", "fit_error_week1", "error_week2"]
    st.dataframe(
        f[cols], width="stretch", height=360, hide_index=True,
        column_config={
            "patient_id": "ID", "name": "Name", "age": "Age", "sex": "Sex", "region": "Region",
            "bmi": st.column_config.NumberColumn("BMI", format="%.1f"),
            "asian_bmi_category": "BMI class (Asian)",
            "diabetes_duration_yrs": st.column_config.NumberColumn("T2D years", format="%.1f"),
            "HbA1c": st.column_config.NumberColumn("HbA1c %", format="%.1f"),
            "glycemic_status": "Glycaemic status",
            "Fasting plasma glucose": st.column_config.NumberColumn("FPG mg/dL", format="%.0f"),
            "HOMA-IR": st.column_config.NumberColumn("HOMA-IR", format="%.1f"),
            "eGFR": st.column_config.NumberColumn("eGFR", format="%.0f"),
            "medications": "Medicines",
            "fit_error_week1": st.column_config.NumberColumn("Twin error wk 1", format="%.1f"),
            "error_week2": st.column_config.NumberColumn("Twin error wk 2", format="%.1f"),
        })
    st.download_button("Download filtered cohort (CSV)", f.to_csv(index=False).encode("utf-8"),
                       file_name="glucotwin_cohort_export.csv", mime="text/csv", icon=":material/download:")


# ---------------------------------------------------------------------------
# Page 6 - Model Evidence
# ---------------------------------------------------------------------------
def page_evidence() -> None:
    ui.page_header("Model evidence", "How the models perform on patients they never saw during training "
                   "(5-fold patient-level cross-validation, or 50 held-out patients).", POP_CRUMB)
    base = read_json("baseline_results.json")
    hyb = read_json("hybrid_results.json")
    deep = read_json("deep_results.json")
    twin_r = read_json("twin_results.json")

    cls, fc = base["classification"], base["forecast"]
    sw_no, sw_ehr = "Smartwatch-only mode | without EHR", "Smartwatch-only mode | with EHR"
    hs, hh = hyb["Glucose spike in 2 h"]["results"], hyb["Hypoglycaemia in 2 h"]["results"]
    gt, cgm = "Smartwatch + EHR + twin (GlucoTwin)", "CGM + EHR (reference, CGM worn)"
    tw = twin_r["week2_median_error"]
    findings = [
        ("Health record + smartwatch fusion",
         f"Without a CGM, adding the health record raises hypo AUPRC from {cls['Hypoglycaemia in 2 h'][sw_no]['auprc']:.2f} "
         f"to {cls['Hypoglycaemia in 2 h'][sw_ehr]['auprc']:.2f} and cuts the 30-minute forecast error from "
         f"{fc['+30 min'][sw_no]['mae']:.1f} to {fc['+30 min'][sw_ehr]['mae']:.1f} mg/dL."),
        ("Calibrated twin accuracy",
         f"In week 2 with the CGM hidden, the calibrated twin tracks glucose with a median error of "
         f"{tw['Calibrated twin (GlucoTwin)']:.1f} mg/dL, against {tw['Population twin']:.1f} for a one-size-fits-all twin."),
        ("Physics + machine learning",
         f"Smartwatch + health record + twin reaches spike AUPRC {hs[gt]['auprc']:.2f}, compared with "
         f"{hs[cgm]['auprc']:.2f} for a model using a worn CGM."),
        ("Deep fusion ensemble",
         f"Averaging XGBoost with a GRU fusion network gives hypo AUPRC {deep['Ensemble (average)']['Hypo AUPRC']:.2f} "
         f"(XGBoost alone {deep['XGBoost']['Hypo AUPRC']:.2f})."),
    ]
    st.markdown("<table class='gt-table'>" + "".join(
        f"<tr><td style='width:2rem;color:{ui.TEAL};font-weight:650' class='num'>{i}</td>"
        f"<td style='width:16rem;font-weight:600'>{esc(t)}</td><td>{esc(d)}</td></tr>"
        for i, (t, d) in enumerate(findings, 1)) + "</table>", unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["Benchmark tables", "Figures", "Ethics and limitations"])
    with t1:
        ui.section("Data fusion ablation", "XGBoost, 5-fold patient-split cross-validation. AUPRC: higher is better; "
                   "MAE: lower is better.")
        rows = []
        for k in ["EHR only", "CGM mode | without EHR", "CGM mode | with EHR", sw_no, sw_ehr]:
            rows.append({"Inputs": k.replace(" | ", ", ").replace("EHR", "health record"),
                         "Spike AUPRC": cls["Glucose spike in 2 h"][k]["auprc"],
                         "Hypo AUPRC": cls["Hypoglycaemia in 2 h"][k]["auprc"],
                         "MAE +30 min (mg/dL)": (f"{fc['+30 min'][k]['mae']:.1f}" if k in fc["+30 min"]
                                                 else "—")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.2f")
                                    for c in ["Spike AUPRC", "Hypo AUPRC"]})
        ui.section("Hybrid model, week 2 with CGM hidden", "Twin forecasts used as features for XGBoost.")
        rows = [{"Model": k, "Spike AUPRC": hs[k]["auprc"], "Hypo AUPRC": hh[k]["auprc"],
                 "MAE +60 min (mg/dL)": hyb["Glucose at +60 min"]["results"][k]["mae"],
                 "MAE +120 min (mg/dL)": hyb["Glucose at +120 min"]["results"][k]["mae"]} for k in hs]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.2f" if "AUPRC" in c else "%.1f")
                                    for c in rows[0] if c != "Model"})
        st.caption(f"Hypo events are rare ({hyb['Hypoglycaemia in 2 h']['n_events']} in week 2), so hypo metrics carry "
                   "wide uncertainty.")
        ui.section("Deep fusion network vs XGBoost", "CGM mode, held-out patients.")
        rows = [dict(Model=k, **v) for k, v in deep.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.2f" if ("AUPRC" in c or "AUROC" in c) else "%.1f")
                                    for c in rows[0] if c != "Model"})
    with t2:
        figs = {
            "Data fusion ablation": C.ROOT / "results" / "fusion_ablation.png",
            "Hybrid model (twin + ML)": C.ROOT / "results" / "hybrid_results.png",
            "Deep fusion net vs XGBoost": C.ROOT / "results" / "deep_vs_xgboost.png",
            "Twin validation (week 2)": C.ROOT / "results" / "twin_validation.png",
            "Twin parameter recovery": C.ROOT / "results" / "twin_recovery.png",
            "TreeSHAP explanations": C.ROOT / "results" / "explainability.png",
            "ROC and PR curves": C.ROOT / "results" / "roc_curves.png",
            "Example patient, GT-0029": C.ROOT / "docs" / "figures" / "patient_GT-0029.png",
        }
        choice = st.selectbox("Figure", list(figs), key="ev_fig")
        path = figs[choice]
        if path.exists():
            st.image(str(path), width="stretch")
        else:
            st.info(f"{path.name} was not found. Run the pipeline scripts to create it.")
    with t3:
        st.markdown(f"""
<div class='gt-note' style='font-size:0.92rem;max-width:80ch'>
<p><b>Synthetic data only.</b> All 200 patients and their 14-day wearable records come from an India-calibrated
physiological simulator, as the competition's sandbox rules require (DPDP Act 2023, HIPAA). No real patient data was used.</p>
<p><b>Decision support, not a device.</b> GlucoTwin is framed as clinical decision support: early warning of
post-meal spikes and hypoglycaemia, and simulation of possible changes. It does not diagnose, prescribe or dose.</p>
<p><b>Known limitations.</b> The twin's equations share their form with the simulator that generated the data,
which flatters the twin; real-world errors would be larger. Hypo events are rare, so hypo metrics are uncertain.
Twin forecasts assume no new meal or walk in the next 2 hours.</p>
<p><b>Why it matters in India.</b> A CGM sensor costs roughly ₹4,000–5,000 every 14 days. The design calibrates a
personal twin during a one-off sensor week, then keeps monitoring with an affordable smartwatch.</p>
</div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
PAGES = {
    "Patient": [
        st.Page(page_overview, title="Patient overview", icon=":material/person:", default=True),
        st.Page(page_twin, title="Digital twin", icon=":material/stacked_line_chart:", url_path="twin"),
        st.Page(page_alerts, title="Predictive alerts", icon=":material/notifications:", url_path="alerts"),
        st.Page(page_whatif, title="What-if simulator", icon=":material/compare_arrows:", url_path="what-if"),
    ],
    "Population": [
        st.Page(page_cohort, title="Cohort explorer", icon=":material/groups:", url_path="cohort"),
        st.Page(page_evidence, title="Model evidence", icon=":material/fact_check:", url_path="evidence"),
    ],
}
current_page = st.navigation(PAGES, position="hidden")
sidebar(PAGES, current_page)
current_page.run()
