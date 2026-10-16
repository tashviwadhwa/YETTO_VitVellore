"""GlucoTwin visual system: design tokens, CSS and small presentation helpers.

Direction: "clinical instrument". Warm off-white page, midnight-navy type, one
muted teal accent reserved for GlucoTwin's own signals (the twin, scenario B,
the active page), and clinical colours (green / amber / red) used only where
they carry meaning. Hierarchy comes from type, spacing and hairlines; the only
bordered surfaces are the places you *work* (the scenario panel) and the
places you *read a result* (the prediction panel).

Theme basics (font, widget borders, radius) live in .streamlit/config.toml.
Only stable hooks are styled here: data-testid attributes and the st-key-<key>
classes Streamlit adds to keyed containers.
"""
from __future__ import annotations

import html as _html

import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
NAVY = "#0E1B2C"        # primary text, key UI
INK_2 = "#4A5768"       # secondary text
MUTED = "#7D8897"       # labels, captions
PAGE = "#FAFAF7"        # warm white page
SURFACE = "#FFFFFF"
LINE = "#E6E8EC"        # hairlines
LINE_STRONG = "#CDD3DA"
TEAL = "#0E8584"        # GlucoTwin accent
TEAL_TINT = "#E7F2F1"
GOOD = "#2F8556"
CAUTION = "#B7791F"
ALERT = "#C0392B"
GOOD_TINT, CAUTION_TINT, ALERT_TINT = "#E8F3EC", "#FBF1E1", "#F9E6E3"

SCEN_A = "#5B6B80"      # Scenario A: slate, dashed
SCEN_B = TEAL           # Scenario B: teal, solid
PLAN = "#AEB6C1"        # "no new meal or walk" reference: light grey, dotted
FONT = "'Instrument Sans', -apple-system, 'Segoe UI', sans-serif"

SOURCE_COLORS = {"CGM": "#0E8584", "Smartwatch": "#B7791F", "Diary": "#A0527A",
                 "Twin": "#3D6FB6", "EHR": "#C0602B", "Time": "#5B6B80"}


def esc(x) -> str:
    return _html.escape(str(x))


def css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&display=swap');

html, body, .stApp {{ background: {PAGE}; }}
html, body, p, li, label, input, textarea, button, select,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] {{ font-family: {FONT}; }}
[data-testid="stIconMaterial"] {{ font-family: 'Material Symbols Rounded' !important; }}
p, li {{ color: {NAVY}; }}
.num {{ font-variant-numeric: tabular-nums; }}
/* Streamlit pulls every markdown block up by 1rem; our blocks own their spacing instead */
[data-testid="stMarkdownContainer"]:has(> div[class^="gt-"]) {{ margin-bottom: 0 !important; }}

/* ---- chrome ---- */
[data-testid="stToolbar"], [data-testid="stDecoration"], footer {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; height: 2.25rem; }}
.block-container {{ padding: 1.6rem 2.5rem 3rem 2.5rem !important; max-width: 1360px; }}
[data-testid="stVerticalBlock"] {{ gap: 0.7rem; }}

/* ---- sidebar ---- */
[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {LINE}; }}
[data-testid="stSidebarHeader"] {{ height: 2.4rem; padding-top: 0.6rem; padding-bottom: 0; }}
[data-testid="stSidebarUserContent"] {{ padding-top: 0 !important; }}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 0.35rem; }}
.gt-brand {{ display: flex; align-items: center; gap: 0.5rem; font-weight: 700; font-size: 1.02rem; color: {NAVY};
            letter-spacing: -0.01em; margin: 0 0 0.9rem 0; }}
.gt-id {{ border-top: 1px solid {LINE}; border-bottom: 1px solid {LINE}; padding: 0.75rem 0 0.8rem 0; margin-top: 0.35rem; }}
.gt-id-name {{ font-size: 0.98rem; font-weight: 650; color: {NAVY}; line-height: 1.25; }}
.gt-id-meta {{ font-size: 0.8rem; color: {MUTED}; margin-top: 0.1rem; }}
.gt-kv {{ display: grid; grid-template-columns: auto 1fr; gap: 0.2rem 0.8rem; font-size: 0.82rem; margin-top: 0.6rem; }}
.gt-kv .k {{ color: {MUTED}; }}
.gt-kv .v {{ color: {NAVY}; font-variant-numeric: tabular-nums; text-align: right; }}
.gt-meds {{ font-size: 0.8rem; color: {INK_2}; margin-top: 0.5rem; line-height: 1.45; }}
.gt-navlabel {{ font-size: 0.7rem; font-weight: 600; color: {MUTED}; letter-spacing: 0.06em; text-transform: uppercase;
               margin: 1rem 0 0.15rem 0.65rem; }}
[data-testid="stPageLink-NavLink"] {{ border-radius: 6px; padding-top: 0.2rem; padding-bottom: 0.2rem; }}
[data-testid="stPageLink-NavLink"]:hover {{ background: #F3F5F6; }}
[data-testid="stPageLink-NavLink"] span {{ color: {INK_2} !important; }}
.gt-nav-active {{ position: relative; background: {TEAL_TINT}; border-radius: 6px; font-size: 0.95rem; font-weight: 600;
                 color: {NAVY}; padding: 0.3rem 0.6rem 0.3rem calc(0.6rem + 23px); }}
.gt-nav-active::before {{ content: ""; position: absolute; left: 0; top: 6px; bottom: 6px; width: 3px; border-radius: 2px;
                         background: {TEAL}; }}
.gt-foot {{ font-size: 0.72rem; color: {MUTED}; margin-top: 1.4rem; line-height: 1.5; }}

/* ---- page header ---- */
.gt-crumb {{ font-size: 0.8rem; color: {MUTED}; margin-bottom: 0.3rem; }}
.gt-crumb b {{ color: {TEAL}; font-weight: 600; }}
.gt-title {{ font-size: 1.55rem; font-weight: 650; color: {NAVY}; line-height: 1.2; letter-spacing: -0.015em; margin: 0; }}
.gt-sub {{ font-size: 0.95rem; color: {INK_2}; margin-top: 0.3rem; max-width: 78ch; }}
.gt-header {{ border-bottom: 1px solid {LINE}; padding-bottom: 1rem; margin-bottom: 0.4rem; }}

/* ---- sections ---- */
.gt-section {{ font-size: 1.02rem; font-weight: 650; color: {NAVY}; margin: 1.3rem 0 0.1rem 0; letter-spacing: -0.005em; }}
.gt-section-sub {{ font-size: 0.85rem; color: {MUTED}; margin-bottom: 0.35rem; }}
.gt-label {{ line-height: 1.4; font-size: 0.72rem; font-weight: 600; color: {MUTED}; letter-spacing: 0.06em; text-transform: uppercase; }}

/* ---- pills ---- */
.gt-pill {{ display: inline-block; font-size: 0.74rem; font-weight: 600; padding: 0.1rem 0.5rem; border-radius: 999px;
           white-space: nowrap; }}
.gt-pill.good {{ background: {GOOD_TINT}; color: {GOOD}; }}
.gt-pill.caution {{ background: {CAUTION_TINT}; color: {CAUTION}; }}
.gt-pill.alert {{ background: {ALERT_TINT}; color: {ALERT}; }}
.gt-pill.teal {{ background: {TEAL_TINT}; color: {TEAL}; }}
.gt-pill.quiet {{ background: #EEF1F4; color: {INK_2}; }}

/* ---- tables ---- */
table.gt-table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; border: none !important; margin: 0; }}
table.gt-table th, table.gt-table td {{ border-left: none !important; border-right: none !important;
    border-top: none !important; background: transparent !important; }}
table.gt-table th {{ text-align: left; font-weight: 600; color: {MUTED}; font-size: 0.74rem; padding: 0.3rem 0.4rem;
                    border-bottom: 1px solid {LINE_STRONG}; }}
table.gt-table td {{ padding: 0.45rem 0.4rem; border-bottom: 1px solid {LINE}; color: {NAVY}; vertical-align: baseline; }}
table.gt-table td.n, table.gt-table th.n {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
table.gt-table .ref {{ color: {MUTED}; font-size: 0.78rem; }}
table.gt-table tr.primary td {{ font-weight: 650; }}
table.gt-table tr:last-child td {{ border-bottom: none; }}

/* ---- signal panels ---- */
.gt-signal {{ border-left: 3px solid var(--c); padding: 0.1rem 0 0.1rem 0.9rem; }}
.gt-signal .lbl {{ font-size: 0.82rem; font-weight: 600; color: var(--c); }}
.gt-signal .val {{ font-size: 1.9rem; font-weight: 650; color: {NAVY}; font-variant-numeric: tabular-nums; line-height: 1.15; }}
.gt-signal .val small {{ font-size: 0.9rem; font-weight: 500; color: {INK_2}; }}
.gt-signal .txt {{ font-size: 0.84rem; color: {INK_2}; margin-top: 0.3rem; line-height: 1.5; }}

/* ---- reasons ---- */
.gt-reason {{ display: grid; grid-template-columns: 1fr auto; gap: 0.1rem 1rem; padding: 0.45rem 0; border-bottom: 1px solid {LINE}; }}
.gt-reason .nm {{ font-size: 0.88rem; color: {NAVY}; font-weight: 550; }}
.gt-reason .src {{ font-size: 0.72rem; font-weight: 600; margin-left: 0.4rem; }}
.gt-reason .vl {{ font-size: 0.8rem; color: {MUTED}; }}
.gt-reason .dir {{ font-size: 0.8rem; text-align: right; font-variant-numeric: tabular-nums; }}

/* ---- range bar ---- */
.gt-rangebar {{ display: flex; height: 12px; border-radius: 3px; overflow: hidden; margin: 0.5rem 0 0.4rem 0; gap: 2px; }}
.gt-rangebar span {{ display: block; height: 100%; }}
.gt-rangelegend {{ display: flex; flex-wrap: wrap; gap: 0.2rem 1.1rem; font-size: 0.78rem; color: {INK_2}; }}
.gt-rangelegend i {{ display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 0.35rem; }}

/* ---- key figures ---- */
.gt-figs {{ display: flex; flex-wrap: wrap; gap: 0.5rem 2.4rem; padding: 0.2rem 0; }}
.gt-fig .k {{ font-size: 0.76rem; color: {MUTED}; }}
.gt-fig .v {{ font-size: 1.25rem; font-weight: 600; color: {NAVY}; font-variant-numeric: tabular-nums; line-height: 1.3; }}
.gt-fig .v small {{ font-size: 0.78rem; color: {INK_2}; font-weight: 500; }}
.gt-fig .t {{ font-size: 0.74rem; color: {MUTED}; }}

/* ---- what-if workspace ---- */
.st-key-wi_context {{ border-bottom: 1px solid {LINE}; padding-bottom: 0.7rem; }}
.st-key-wi_plans {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 10px; padding: 1rem 1.1rem 1.1rem 1.1rem; }}
.st-key-wi_plans [data-testid="stVerticalBlock"] {{ gap: 0.3rem; }}
.st-key-wi_plans [data-testid="stHorizontalBlock"] {{ gap: 0.6rem; }}
.st-key-wi_plans button[data-testid="stNumberInputStepUp"],
.st-key-wi_plans button[data-testid="stNumberInputStepDown"] {{ display: none; }}
.st-key-wi_plans [data-testid="stExpander"] details {{ border: none !important; background: transparent; }}
.st-key-wi_plans [data-testid="stExpander"] summary {{ padding-left: 0; font-size: 0.82rem; color: {INK_2}; }}
.gt-rowlbl {{ font-size: 0.8rem; font-weight: 600; color: {INK_2}; padding: 0.35rem 0 0 0; display: flex; gap: 0.4rem;
             align-items: baseline; line-height: 1.3; }}
.gt-rowlbl small {{ font-weight: 400; color: {MUTED}; font-size: 0.76rem; }}
.gt-rowlbl .mod {{ width: 6px; height: 6px; border-radius: 50%; background: {TEAL}; display: inline-block; align-self: center; }}
.gt-planhead {{ padding: 0.35rem 0 0 0; line-height: 1.4; display: flex; align-items: center; gap: 0.5rem; font-size: 0.8rem; font-weight: 600; color: {INK_2}; }}
.gt-letter {{ display: inline-flex; width: 1.35rem; height: 1.35rem; border-radius: 5px; align-items: center;
             justify-content: center; font-size: 0.75rem; font-weight: 700; flex: none; }}
.gt-letter.a {{ border: 1.5px dashed {SCEN_A}; color: {SCEN_A}; }}
.gt-letter.b {{ background: {SCEN_B}; color: white; }}
.st-key-wi_plans input[aria-label^="Name of plan"] {{ font-weight: 600; }}
.gt-status {{ font-size: 0.8rem; color: {MUTED}; display: flex; align-items: center; gap: 0.4rem; }}
.gt-status i {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}
.st-key-wi_out {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 10px; padding: 1.1rem 1.3rem 0.8rem 1.3rem;
                 transition: opacity 120ms ease; }}
.st-key-wi_out_stale {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 10px; padding: 1.1rem 1.3rem 0.8rem 1.3rem;
                       opacity: 0.5; }}
.gt-headline {{ font-size: 1.32rem; font-weight: 650; color: {NAVY}; line-height: 1.3; letter-spacing: -0.01em;
               margin: 0.25rem 0 0.1rem 0; max-width: 46ch; }}
.gt-headline em {{ font-style: normal; color: var(--hc); }}
.gt-note {{ font-size: 0.85rem; color: {INK_2}; line-height: 1.55; }}
.gt-note ul {{ margin: 0.2rem 0 0 1rem; padding: 0; }}
.gt-note li {{ color: {INK_2}; margin: 0.1rem 0; font-size: 0.85rem; }}
.gt-key {{ display: inline-block; width: 18px; height: 0; border-top: 2.5px solid; vertical-align: middle; margin-right: 0.4rem; }}
.gt-key.dash {{ border-top-style: dashed; }}
.gt-key.dot {{ border-top-style: dotted; }}

/* ---- widgets ---- */
div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-baseweb="input"] > div,
div[data-baseweb="input"] input, [data-testid="stNumberInputContainer"] {{ background-color: {SURFACE} !important; }}
.stButton button, .stDownloadButton button {{ border-radius: 6px !important; font-weight: 600 !important; }}
.stButton button[kind="primary"] {{ background: {NAVY} !important; border-color: {NAVY} !important; }}
.stButton button[kind="primary"] p, .stButton button[kind="primary"] span {{ color: white !important; }}
.stButton button[kind="primary"]:hover {{ background: #1B2D45 !important; }}
.stButton button[kind="tertiary"] p {{ color: {INK_2} !important; font-size: 0.85rem; text-decoration: underline;
                                      text-decoration-color: {LINE_STRONG}; text-underline-offset: 3px; }}
.stButton button[kind="tertiary"]:hover p {{ color: {NAVY} !important; }}
.stButton button:focus-visible, input:focus-visible, a:focus-visible, [role="tab"]:focus-visible {{
    outline: 2px solid {TEAL} !important; outline-offset: 2px; }}
[data-testid="stWidgetLabel"] p {{ font-size: 0.8rem !important; color: {INK_2} !important; font-weight: 500; }}
[data-testid="stExpander"] details {{ border-radius: 8px !important; border-color: {LINE} !important; background: {SURFACE}; }}
[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 8px; }}
[data-testid="stCaptionContainer"] p {{ color: {MUTED} !important; font-size: 0.8rem; }}
@media (max-width: 1150px) {{
  .block-container {{ padding: 1.2rem 1.2rem 2rem 1.2rem !important; }}
  /* stack plans above prediction when there is no room for both */
  .st-key-wi_ws [data-testid="stHorizontalBlock"]:has(.st-key-wi_plans) {{ flex-wrap: wrap; }}
  .st-key-wi_ws [data-testid="stHorizontalBlock"]:has(.st-key-wi_plans) > [data-testid="stColumn"] {{
      min-width: 100%; flex: 1 1 100%; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>""", unsafe_allow_html=True)


LOGO = (f"<svg width='22' height='22' viewBox='0 0 24 24' aria-hidden='true'>"
        f"<circle cx='12' cy='12' r='10.5' fill='none' stroke='{TEAL}' stroke-width='1.6'/>"
        f"<path d='M4 13h3.2l1.8-4 2.6 8 2.2-6 1.4 2H20' fill='none' stroke='{NAVY}' stroke-width='1.6' "
        f"stroke-linecap='round' stroke-linejoin='round'/></svg>")


def page_header(title: str, sub: str, crumb_html: str) -> None:
    """Breadcrumb (section and patient), title, one line of purpose."""
    st.markdown(f"<div class='gt-header'><div class='gt-crumb'>{crumb_html}</div>"
                f"<div class='gt-title' role='heading' aria-level='1'>{esc(title)}</div>"
                f"<div class='gt-sub'>{esc(sub)}</div></div>", unsafe_allow_html=True)


def section(title: str, sub: str | None = None) -> None:
    st.markdown(f"<div class='gt-section'>{esc(title)}</div>"
                + (f"<div class='gt-section-sub'>{esc(sub)}</div>" if sub else ""), unsafe_allow_html=True)


def label(text: str) -> None:
    st.markdown(f"<div class='gt-label'>{esc(text)}</div>", unsafe_allow_html=True)


def pill(text: str, kind: str = "quiet") -> str:
    return f"<span class='gt-pill {kind}'>{esc(text)}</span>"


def figs(items) -> None:
    """A row of a few key figures: [(label, value_html, note)]."""
    st.markdown("<div class='gt-figs'>" + "".join(
        f"<div class='gt-fig'><div class='k'>{esc(k)}</div><div class='v'>{v}</div>"
        + (f"<div class='t'>{esc(t)}</div>" if t else "") + "</div>" for k, v, t in items) + "</div>",
        unsafe_allow_html=True)


def plotly_base(fig: go.Figure, height: int, ytitle: str | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, template="none", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Instrument Sans, -apple-system, Segoe UI, sans-serif", size=12, color=INK_2),
        margin=dict(l=8, r=64, t=8, b=8), hovermode="x unified", showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=12, color=NAVY),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", bordercolor=LINE_STRONG, font=dict(color=NAVY, size=12)),
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE_STRONG, ticks="outside", tickcolor=LINE_STRONG, ticklen=4,
                     tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor="#EEF0F3", zeroline=False, showline=False,
                     tickfont=dict(color=MUTED), title=dict(text=ytitle, font=dict(size=12, color=INK_2)))
    return fig


def thresholds(fig: go.Figure, row=None, col=None, label=True, band=True) -> None:
    """70 and 180 mg/dL lines (and optionally a faint target band)."""
    kw = dict(row=row, col=col) if row else {}
    if band:
        fig.add_hrect(y0=70, y1=180, fillcolor="rgba(47,133,86,0.05)", line_width=0, layer="below", **kw)
    for y, txt, color in ((180, "180", CAUTION), (70, "70", ALERT)):
        ann = dict(annotation_text=txt, annotation_position="right",
                   annotation_font=dict(size=11, color=color)) if label else {}
        fig.add_hline(y=y, line=dict(color=color, width=1, dash="dot"), layer="below", **kw, **ann)


CONFIG = {"displayModeBar": False, "scrollZoom": False}
