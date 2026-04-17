"""
app.py — Zero Click AI · Genesis Training Company
Glass-themed AI data analysis platform.
Modules: data_ingestion, data_cleaning, data_analysis, kpi_generator,
         insights, ml_engine, forecasting, visualization, chatbot, report_generator
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json, os, io, time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Page config (MUST be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="Zero Click AI — Genesis Training",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Module imports ────────────────────────────────────────────────────────
import data_ingestion as di
import data_cleaning  as dc
import data_analysis  as da
import kpi_generator  as kg
import insights       as ins
import ml_engine      as mle
import forecasting    as fc
import visualization  as viz
import chatbot        as cb
import report_generator as rg
import utils

# ═══════════════════════════════════════════════════════════════════════════
# GLASS CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*, *::before, *::after { box-sizing: border-box; font-family: 'Inter', sans-serif; }

/* Page background */
.stApp { background: linear-gradient(135deg,#0f0c29 0%,#1a1a3e 50%,#24243e 100%) !important; min-height:100vh; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.9) !important;
    border-right: 1px solid rgba(167,139,250,0.15) !important;
    backdrop-filter: blur(20px) !important;
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.75) !important; }
[data-testid="stSidebar"] select {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important; color: rgba(255,255,255,0.75) !important;
    font-size: 12px !important; padding: 4px 6px !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(167,139,250,0.18) !important;
    border-radius: 14px !important; padding: 14px !important;
    backdrop-filter: blur(10px);
}
[data-testid="stMetricValue"]  { color: rgba(255,255,255,0.95) !important; font-weight: 600 !important; }
[data-testid="stMetricLabel"]  { color: rgba(255,255,255,0.4) !important; font-size: 11px !important;
                                  text-transform: uppercase; letter-spacing: .05em; }
[data-testid="stMetricDelta"]  { font-size: 11px !important; }

/* Dataframe */
[data-testid="stDataFrame"] { background: rgba(255,255,255,0.04) !important; border-radius: 12px !important; }

/* Expander */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(167,139,250,0.15) !important;
    border-radius: 12px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, rgba(167,139,250,0.25), rgba(124,58,237,0.2)) !important;
    border: 1px solid rgba(167,139,250,0.3) !important; border-radius: 20px !important;
    color: #c4b5fd !important; font-weight: 500 !important; transition: all .15s !important;
}
.stButton > button:hover { background: rgba(167,139,250,0.35) !important; }

/* Inputs */
.stTextInput > div > div > input, .stTextArea textarea {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important; color: rgba(255,255,255,0.9) !important;
}

/* Chat */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(167,139,250,0.3) !important; border-radius: 20px !important;
}
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.06) !important; border-radius: 12px !important;
}

/* Progress */
.stProgress > div > div { background: linear-gradient(90deg,#a78bfa,#7c3aed) !important; }

/* Selectbox */
[data-baseweb="select"] > div { background: rgba(255,255,255,0.07) !important; border-color: rgba(255,255,255,0.12) !important; }

/* Tabs */
[data-baseweb="tab"] { color: rgba(255,255,255,0.45) !important; }
[aria-selected="true"] { color: #a78bfa !important; border-bottom-color: #a78bfa !important; }

/* Radio */
[data-testid="stRadio"] > div > label { color: rgba(255,255,255,0.65) !important; font-size:12px !important; }

/* Hide chrome */
#MainMenu, footer { visibility: hidden; }

/* Glass card */
.g-card {
    background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px; padding: 14px 16px; backdrop-filter: blur(12px);
    margin-bottom: 8px;
}
/* KPI card */
.kpi-glass {
    background: rgba(255,255,255,0.07); border: 1px solid rgba(167,139,250,0.2);
    border-radius: 14px; padding: 14px 16px; backdrop-filter: blur(10px);
}
.kpi-val  { font-size:22px; font-weight:600; color:rgba(255,255,255,0.95); line-height:1; }
.kpi-lbl  { font-size:9px; color:rgba(255,255,255,0.38); text-transform:uppercase;
             letter-spacing:.05em; margin-top:3px; }
.kpi-dt   { font-size:10px; margin-top:5px; }
.up { color:#4ade80; } .dn { color:#f87171; }
/* Insight banner */
.ins-bar {
    background: rgba(167,139,250,0.07); border: 1px solid rgba(167,139,250,0.2);
    border-radius: 14px; padding: 12px 16px; margin-bottom:10px;
}
/* Rec card */
.rec-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 9px 12px; margin-bottom:6px;
    font-size:11px; color:rgba(255,255,255,0.7); line-height:1.5;
}
/* Activity item */
.act-item {
    display:flex; gap:9px; padding:7px 0;
    border-bottom:1px solid rgba(255,255,255,0.05);
    font-size:11px; color:rgba(255,255,255,0.65);
}
/* Status pill */
.sp { font-size:9px; padding:2px 7px; border-radius:20px; font-weight:600; }
.sp-g { background:rgba(74,222,128,0.12); color:#4ade80; border:1px solid rgba(74,222,128,0.2); }
.sp-y { background:rgba(251,191,36,0.12); color:#fbbf24; border:1px solid rgba(251,191,36,0.18); }
.sp-r { background:rgba(248,113,113,0.12); color:#f87171; border:1px solid rgba(248,113,113,0.2); }
/* Section header */
.sec-hdr { font-size:13px; font-weight:600; color:#c4b5fd; margin:6px 0 10px; }
.sec-sub  { font-size:10px; color:rgba(255,255,255,0.3); margin-top:2px; }
/* ML model card */
.ml-card {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px; padding: 14px;
}
/* Team card */
.team-card {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px; padding: 14px 10px; text-align: center;
    transition: all .2s;
}
/* Topbar */
.topbar-wrap {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 10px 16px; margin-bottom:12px;
    display:flex; align-items:center; gap:10px;
}
/* RT bar */
.rt-bar {
    background: rgba(74,222,128,0.06); border: 1px solid rgba(74,222,128,0.15);
    border-radius: 10px; padding:7px 14px; font-size:11px;
    color:rgba(255,255,255,0.6); margin-bottom:12px;
}
/* Prog track */
.prog-track { height:5px; border-radius:3px; background:rgba(255,255,255,0.07); }
.prog-fill  { height:100%; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
def _init():
    defs = {
        "logged_in": False, "user": None, "page": "Analytics",
        "df": None, "cleaned_df": None, "column_types": None,
        "kpis": None, "charts": None, "ml_results": None,
        "forecast": None, "insights_text": None, "cleaning_report": None,
        "correlations": None, "chat_history": [],
        "last_refresh": time.time(), "auto_refresh": False, "refresh_interval": 30,
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ═══════════════════════════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════════════════════════
USER_FILE = "users.json"

def _load_users():
    if os.path.exists(USER_FILE):
        try:
            with open(USER_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"admin": "admin", "demo": "demo123"}

def _save_users(u):
    with open(USER_FILE, "w") as f:
        json.dump(u, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════
# HTML HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def _c(html):
    st.markdown(html, unsafe_allow_html=True)

def kpi_card(label, value, delta=None, up=True):
    delta_html = ""
    if delta:
        sym = "▲" if up else "▼"
        cls = "up" if up else "dn"
        delta_html = f'<div class="kpi-dt {cls}">{sym} {delta}</div>'
    return f"""
    <div class="kpi-glass">
      <div class="kpi-lbl">{label}</div>
      <div class="kpi-val">{value}</div>
      {delta_html}
    </div>"""

def glass_card(content, extra=""):
    return f'<div class="g-card" style="{extra}">{content}</div>'

def section_title(title, sub=""):
    sub_html = f'<div class="sec-sub">{sub}</div>' if sub else ""
    _c(f'<div class="sec-hdr">{title}{sub_html}</div>')

def pill(text, color="#a78bfa", bg_alpha=0.12):
    rgb = utils.hex_to_rgb(color)
    return (f'<span style="font-size:9px;padding:2px 8px;border-radius:20px;font-weight:600;'
            f'background:rgba({rgb},{bg_alpha});color:{color};border:1px solid rgba({rgb},0.22)">'
            f'{text}</span>')

PLOTLY = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="rgba(255,255,255,0.7)", size=11),
    xaxis=dict(showgrid=False, color="rgba(255,255,255,0.3)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="rgba(255,255,255,0.3)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="rgba(255,255,255,0.55)")),
    margin=dict(l=0, r=0, t=32, b=0), height=260,
)

def _pgl(fig, h=260):
    fig.update_layout(**{**PLOTLY, "height": h})
    return fig

ACCS = ["#a78bfa","#38bdf8","#4ade80","#fb923c","#f472b6","#fbbf24"]

# ═══════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ═══════════════════════════════════════════════════════════════════════════
def login_page():
    _, col, _ = st.columns([1, 1.3, 1])
    with col:
        _c("""
        <div style="text-align:center;padding:2rem 0 1.5rem">
          <div style="font-size:3.5rem">🧠</div>
          <h1 style="background:linear-gradient(135deg,#a78bfa,#e879f9);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                     font-size:2.2rem;margin:.3rem 0">Zero Click AI</h1>
          <p style="color:rgba(255,255,255,0.35);font-size:.85rem">
            Intelligent Data Analysis Platform<br>Genesis Training Company
          </p>
        </div>""")
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        with tab1:
            uname = st.text_input("Username", placeholder="Enter username", key="li_u")
            pwd   = st.text_input("Password", type="password", placeholder="••••••••", key="li_p")
            rem   = st.checkbox("Remember me")
            if st.button("Login", use_container_width=True, type="primary"):
                users = _load_users()
                if uname in users and users[uname] == pwd:
                    st.session_state.logged_in = True
                    st.session_state.user = uname
                    if rem:
                        st.query_params["user"] = uname
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        with tab2:
            nu = st.text_input("Choose username", key="su_u")
            np_ = st.text_input("Password", type="password", key="su_p")
            cp = st.text_input("Confirm password", type="password", key="su_c")
            if st.button("Create Account", use_container_width=True):
                if not nu or not np_:
                    st.warning("Fill all fields")
                elif np_ != cp:
                    st.error("Passwords don't match")
                else:
                    users = _load_users()
                    if nu in users:
                        st.error("Username taken")
                    else:
                        users[nu] = np_
                        _save_users(users)
                        st.success("Account created! Please login.")
        _c('<div style="text-align:center;font-size:.75rem;color:rgba(255,255,255,0.2);margin-top:.8rem">'
           'admin / admin &nbsp;·&nbsp; demo / demo123</div>')

# ═══════════════════════════════════════════════════════════════════════════
# DATA PROCESSING
# ═══════════════════════════════════════════════════════════════════════════
def _process_upload(uploaded_file):
    with st.spinner("Loading file…"):
        df = di.load_file(uploaded_file)
    if df is None:
        return

    bar = st.progress(0, text="Cleaning data…")
    cleaned_df, cleaning_report = dc.clean_data(df)
    st.session_state.cleaning_report = cleaning_report
    bar.progress(20, "Detecting column types…")

    col_types = da.get_column_types(cleaned_df)
    st.session_state.column_types = col_types
    bar.progress(35, "Computing correlations…")

    corrs = da.find_correlations(cleaned_df)
    st.session_state.correlations = corrs
    bar.progress(50, "Generating KPIs…")

    kpis = kg.generate_kpis(cleaned_df, col_types)
    st.session_state.kpis = kpis
    bar.progress(65, "Building charts…")

    charts = viz.auto_charts(cleaned_df, col_types)
    st.session_state.charts = charts
    bar.progress(75, "Training ML models…")

    ml_res = mle.auto_ml(cleaned_df, col_types)
    st.session_state.ml_results = ml_res
    bar.progress(85, "Forecasting…")

    forecast = fc.auto_forecast(cleaned_df, col_types)
    st.session_state.forecast = forecast
    bar.progress(95, "Generating insights…")

    stats = da.compute_stats(cleaned_df)
    insights_text = ins.generate_insights(cleaned_df, stats, corrs)
    st.session_state.insights_text = insights_text
    bar.progress(100)
    bar.empty()

    st.session_state.df = df
    st.session_state.cleaned_df = cleaned_df
    st.success(f"✅ Loaded **{uploaded_file.name}** — {len(cleaned_df):,} rows × {len(cleaned_df.columns)} columns")
    time.sleep(0.8)
    st.session_state.page = "Analytics"
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
PAGES = ["Analytics", "Visualization", "Forecasting", "AI & ML",
         "Upload Data", "About", "Meet Our Team", "Settings"]
PAGE_ICONS = {
    "Analytics": "📊", "Visualization": "📈", "Forecasting": "🔮",
    "AI & ML": "🧠", "Upload Data": "📂", "About": "ℹ️",
    "Meet Our Team": "👥", "Settings": "⚙️",
}

def render_sidebar():
    with st.sidebar:
        _c("""
        <div style="display:flex;align-items:center;gap:9px;padding:4px 0 14px">
          <div style="width:30px;height:30px;border-radius:8px;
                      background:linear-gradient(135deg,#a78bfa,#7c3aed);
                      display:flex;align-items:center;justify-content:center;font-size:16px">🧠</div>
          <div>
            <div style="font-size:14px;font-weight:600;color:rgba(255,255,255,0.92)">Zero Click AI</div>
            <div style="font-size:9px;color:rgba(255,255,255,0.3);text-transform:uppercase;letter-spacing:.07em">Genesis Training</div>
          </div>
        </div>""")
        st.divider()

        # Dashboards nav
        _c('<div style="font-size:9px;font-weight:600;color:rgba(255,255,255,0.28);text-transform:uppercase;letter-spacing:.1em;margin-bottom:5px">Dashboards</div>')
        for page in ["Analytics", "Visualization", "Forecasting", "AI & ML"]:
            is_on = st.session_state.page == page
            if st.button(f"{PAGE_ICONS[page]}  {page}", key=f"nav_{page}",
                         use_container_width=True,
                         type="primary" if is_on else "secondary"):
                st.session_state.page = page
                st.rerun()

        st.divider()
        _c('<div style="font-size:9px;font-weight:600;color:rgba(255,255,255,0.28);text-transform:uppercase;letter-spacing:.1em;margin-bottom:5px">General</div>')
        for page in ["Upload Data", "About", "Meet Our Team"]:
            is_on = st.session_state.page == page
            badge = " &nbsp;<small style='font-size:8px;padding:1px 4px;border-radius:8px;background:rgba(74,222,128,0.15);color:#4ade80'>CSV/XLS</small>" if page == "Upload Data" else ""
            if st.button(f"{PAGE_ICONS[page]}  {page}", key=f"nav_{page}",
                         use_container_width=True,
                         type="primary" if is_on else "secondary"):
                st.session_state.page = page
                st.rerun()

        st.divider()

        # Filters (when data loaded)
        df = st.session_state.cleaned_df
        col_types = st.session_state.column_types or {}
        if df is not None:
            _c('<div style="font-size:9px;font-weight:600;color:rgba(255,255,255,0.28);text-transform:uppercase;letter-spacing:.1em;margin-bottom:5px">Filters</div>')
            cat_cols = [c for c in col_types.get("categorical", []) if df[c].nunique() <= 30][:3]
            for cat in cat_cols:
                opts = ["All"] + sorted(df[cat].dropna().unique().tolist())
                st.selectbox(cat, opts, key=f"f_{cat}")

            date_cols = col_types.get("datetime", [])
            if date_cols:
                years = sorted(df[date_cols[0]].dt.year.dropna().unique().astype(int).tolist(), reverse=True)
                st.selectbox("Year", ["All"] + [str(y) for y in years], key="f_year")

            st.caption(f"Dataset: {len(df):,} rows · {len(df.columns)} cols")
            st.divider()

        # Settings nav
        is_on = st.session_state.page == "Settings"
        if st.button("⚙️  Settings", key="nav_Settings", use_container_width=True,
                     type="primary" if is_on else "secondary"):
            st.session_state.page = "Settings"
            st.rerun()

        st.divider()
        # User card
        uname = st.session_state.get("user", "Admin")
        _c(f"""
        <div style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);
                    border-radius:10px;padding:8px 10px;display:flex;align-items:center;gap:8px;margin-bottom:7px">
          <div style="width:28px;height:28px;border-radius:50%;background:linear-gradient(135deg,#a78bfa,#7c3aed);
                      display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:600;color:white">
            {uname[:2].upper()}</div>
          <div>
            <div style="font-size:11px;font-weight:500;color:rgba(255,255,255,0.88)">{uname}</div>
            <div style="font-size:9px;color:rgba(255,255,255,0.3)">Admin · Online</div>
          </div>
        </div>""")
        if st.button("🚪 Logout", use_container_width=True, key="logout"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# TOPBAR + PDF BUTTON
# ═══════════════════════════════════════════════════════════════════════════
def render_topbar(page):
    df = st.session_state.cleaned_df
    fname = ""
    if df is not None:
        fname = f"· {len(df):,} rows × {len(df.columns)} cols"

    col_title, col_rpt = st.columns([5, 2])
    with col_title:
        _c(f"""
        <div style="padding:2px 0 10px">
          <div style="font-size:18px;font-weight:600;color:#c4b5fd">{page}</div>
          <div style="font-size:10px;color:rgba(255,255,255,0.3)">Home › {page} {fname}</div>
        </div>""")

    with col_rpt:
        if page in ("Analytics", "Visualization", "Forecasting") and df is not None:
            # Show report options in an expander in the topbar area
            with st.expander("📄 Download Report", expanded=False):
                rtype = st.radio("Format", ["PDF", "Excel", "Both"],
                                 horizontal=True, key="rpt_format")
                incl_charts = st.checkbox("Include charts in PDF", value=True, key="incl_charts")
                if st.button("🚀 Generate Report", key="gen_rpt", type="primary",
                             use_container_width=True):
                    _generate_reports(rtype, incl_charts)


def _generate_reports(rtype: str, incl_charts: bool):
    df     = st.session_state.cleaned_df
    kpis   = st.session_state.kpis or []
    ml_res = st.session_state.ml_results or []
    fc_    = st.session_state.forecast
    ins    = st.session_state.insights_text or ""
    charts = (st.session_state.charts or []) if incl_charts else []

    ts = datetime.now().strftime("%Y%m%d_%H%M")

    if rtype in ("PDF", "Both"):
        try:
            with st.spinner("Generating PDF…"):
                pdf_bytes = rg.generate_report_pdf(df, kpis, ml_res, fc_, ins, charts)
            if pdf_bytes:
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"genesis_report_{ts}.pdf",
                    mime="application/pdf",
                    key=f"pdf_dl_{ts}",
                )
            else:
                st.warning("PDF generation requires `fpdf2`. Run: `pip install fpdf2`")
        except Exception as e:
            st.error(f"PDF error: {e}")

    if rtype in ("Excel", "Both"):
        try:
            with st.spinner("Generating Excel…"):
                xl_bytes = rg.generate_report_excel(df, kpis, ml_res, fc_, ins)
            if xl_bytes:
                st.download_button(
                    label="⬇️ Download Excel Report",
                    data=xl_bytes,
                    file_name=f"genesis_report_{ts}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"xl_dl_{ts}",
                )
            else:
                st.warning("Excel generation requires `openpyxl`. Run: `pip install openpyxl`")
        except Exception as e:
            st.error(f"Excel error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# APPLY FILTERS
# ═══════════════════════════════════════════════════════════════════════════
def _filtered_df():
    df = st.session_state.cleaned_df
    if df is None:
        return None
    filtered = df.copy()
    col_types = st.session_state.column_types or {}
    cat_cols = [c for c in col_types.get("categorical", []) if df[c].nunique() <= 30][:3]
    for cat in cat_cols:
        sel = st.session_state.get(f"f_{cat}", "All")
        if sel != "All":
            filtered = filtered[filtered[cat] == sel]
    date_cols = col_types.get("datetime", [])
    if date_cols:
        yr = st.session_state.get("f_year", "All")
        if yr != "All":
            filtered = filtered[filtered[date_cols[0]].dt.year == int(yr)]
    return filtered

# ═══════════════════════════════════════════════════════════════════════════
# NO-DATA GUARD
# ═══════════════════════════════════════════════════════════════════════════
def _no_data():
    _c("""
    <div style="text-align:center;padding:4rem 2rem">
      <div style="font-size:3rem">📂</div>
      <h2 style="color:#a78bfa;margin:.5rem 0">No Dataset Loaded</h2>
      <p style="color:rgba(255,255,255,0.4)">Go to <b>Upload Data</b> in the sidebar to get started.</p>
    </div>""")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
def page_analytics():
    df = _filtered_df()
    if df is None or df.empty:
        _no_data(); return

    col_types = st.session_state.column_types or {}
    kpis_raw  = st.session_state.kpis or []
    insights_txt = st.session_state.insights_text or ""

    # KPI row
    section_title("Key Performance Indicators", "Dynamic metrics from your dataset")
    if kpis_raw:
        cols = st.columns(min(len(kpis_raw), 4))
        for i, k in enumerate(kpis_raw[:4]):
            with cols[i % 4]:
                name, val = str(k[0]), str(k[1])
                note = str(k[2]) if len(k) > 2 else ""
                _c(kpi_card(name, val, note, True))
    st.markdown("<br>", unsafe_allow_html=True)

    # Insight banner
    section_title("AI Insight Generator", "Groq llama-3.1-8b · rule-based fallback")
    chips_html = ""
    if insights_txt:
        for line in insights_txt.split("\n"):
            line = line.strip().lstrip("-•").strip()
            if line:
                color = "#4ade80" if any(w in line.lower() for w in ["strong","profitable","good","positive"]) \
                        else "#fb923c" if any(w in line.lower() for w in ["warning","high","delay","missing","weak"]) \
                        else "#38bdf8"
                chips_html += f'<span style="font-size:10px;padding:3px 9px;border-radius:20px;margin:2px;display:inline-block;background:rgba({utils.hex_to_rgb(color)},0.1);color:{color};border:1px solid rgba({utils.hex_to_rgb(color)},0.22)">{line[:80]}</span>'
    _c(f"""
    <div class="ins-bar">
      <div style="font-size:11px;font-weight:600;color:#a78bfa;margin-bottom:7px">⚡ AI insights from your dataset</div>
      <div style="display:flex;flex-wrap:wrap;gap:4px">{chips_html}</div>
    </div>""")

    # Revenue chart + Category donut
    col_chart, col_donut = st.columns([1.6, 1])
    num_cols  = [c for c in col_types.get("numeric", []) if not any(k in c.lower() for k in ["id","index","code"])]
    time_col  = next(iter(col_types.get("datetime", [])), None)
    s_col     = next((c for c in num_cols if any(w in c.lower() for w in ["sales","revenue","amount"])), num_cols[0] if num_cols else None)
    p_col     = next((c for c in num_cols if "profit" in c.lower()), None)
    cat_col   = next(iter(col_types.get("categorical", [])), None)

    with col_chart:
        section_title("Revenue vs Profit", "Monthly breakdown")
        if s_col and time_col:
            try:
                tmp = df.copy()
                tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
                tmp["_p"] = tmp[time_col].dt.to_period("M").dt.to_timestamp()
                agg_cols = [s_col] + ([p_col] if p_col else [])
                agg = tmp.groupby("_p")[agg_cols].sum().reset_index()
                fig = go.Figure()
                fig.add_trace(go.Bar(x=agg["_p"], y=agg[s_col], name=s_col, marker_color="#a78bfa", opacity=0.85))
                if p_col:
                    fig.add_trace(go.Bar(x=agg["_p"], y=agg[p_col], name="Profit", marker_color="#4ade80", opacity=0.85))
                fig.update_layout(barmode="group", **{**PLOTLY, "height":270})
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                if s_col:
                    fig = px.bar(df.head(50), y=s_col, color_discrete_sequence=["#a78bfa"])
                    st.plotly_chart(_pgl(fig), use_container_width=True)
        elif s_col:
            fig = px.bar(df.head(50), y=s_col, color_discrete_sequence=["#a78bfa"])
            st.plotly_chart(_pgl(fig), use_container_width=True)
        else:
            st.info("Requires a numeric column.")

    with col_donut:
        section_title("Sales by Category")
        if cat_col and s_col:
            try:
                cat_df = df.groupby(cat_col)[s_col].sum().reset_index().sort_values(s_col, ascending=False).head(6)
                fig = px.pie(cat_df, names=cat_col, values=s_col, hole=0.52,
                             color_discrete_sequence=ACCS)
                fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=9)
                fig.update_layout(**{**PLOTLY, "height":270, "showlegend":False})
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Category / Sales columns needed.")
        else:
            st.info("Requires categorical + numeric columns.")

    # Top products + Smart recommendations
    col_tbl, col_rec = st.columns([1.4, 1])
    with col_tbl:
        section_title("Top Products / Records", "By primary metric")
        prod_col = next((c for c in df.columns if "product" in c.lower() and "name" in c.lower()), None)
        if prod_col and s_col:
            try:
                top = df.groupby(prod_col)[s_col].sum().sort_values(ascending=False).head(8).reset_index()
                top[s_col] = top[s_col].apply(lambda x: f"${x:,.0f}")
                st.dataframe(top, use_container_width=True, hide_index=True, height=260)
            except Exception:
                st.dataframe(df.head(8), use_container_width=True, height=260)
        else:
            st.dataframe(df.head(8), use_container_width=True, height=260)

    with col_rec:
        section_title("Smart Recommendations", "AI-driven action items")
        recs = []
        kv = {k[0]: k[1] for k in (kpis_raw or [])}
        if any("discount" in str(k).lower() for k in kv):
            recs.append(("Reduce discount levels — high discounting compressing margins", "#f87171"))
        if any("delay" in str(k).lower() or "shipping" in str(k).lower() for k in kv):
            recs.append(("Optimise logistics — shipping delays exceed threshold", "#fb923c"))
        recs.append(("Scale top-performing categories to maximise revenue", "#4ade80"))
        recs.append(("Leverage strong customer base with loyalty programmes", "#a78bfa"))
        if not recs:
            recs = [("Explore Analysis tab for deeper patterns", "#a78bfa"),
                    ("Run ML models for anomaly detection", "#38bdf8")]
        for text, color in recs[:4]:
            _c(f'<div class="rec-card"><span style="color:{color};font-weight:600">→ </span>{text}</div>')

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════
def page_visualization():
    df = _filtered_df()
    if df is None or df.empty:
        _no_data(); return

    col_types = st.session_state.column_types or {}

    # Live refresh bar
    elapsed  = int(time.time() - st.session_state["last_refresh"])
    interval = st.session_state["refresh_interval"]
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        auto = st.toggle("Auto-Refresh", value=st.session_state["auto_refresh"], key="auto_tog")
        st.session_state["auto_refresh"] = auto
    with c2:
        iv = st.selectbox("Interval (s)", [10, 30, 60, 120], index=1, key="iv_sel")
        st.session_state["refresh_interval"] = iv
    with c3:
        if st.button("🔄 Refresh Now", key="ref_now"):
            st.session_state["last_refresh"] = time.time(); st.rerun()

    dot_col = "#4ade80" if elapsed < interval else "#fb923c"
    _c(f'<div class="rt-bar">'
       f'<span style="display:inline-block;width:7px;height:7px;background:{dot_col};border-radius:50%;margin-right:6px"></span>'
       f'Live · last refresh <b style="color:{dot_col}">{elapsed}s ago</b> · '
       f'Dataset: <b style="color:#a78bfa">{len(df):,}</b> rows</div>')

    num_cols = [c for c in col_types.get("numeric", []) if not any(k in c.lower() for k in ["id","index","code"])]
    time_col = next(iter(col_types.get("datetime", [])), None)
    s_col    = next((c for c in num_cols if any(w in c.lower() for w in ["sales","revenue","amount"])), num_cols[0] if num_cols else None)
    cat_col  = next(iter(col_types.get("categorical", [])), None)
    reg_col  = next((c for c in df.columns if "region" in c.lower()), None)

    # YoY area chart
    section_title("Year-over-Year Comparison", "Sneat-style gradient area chart")
    try:
        fig = viz.create_yoy_area_chart(df)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"YoY chart: {e}")

    # Month-over-month + Region donut + Margin bars
    c1, c2, c3 = st.columns(3)
    with c1:
        section_title("Month-over-Month")
        try:
            fig = viz.create_mom_grouped_bars(df)
            st.plotly_chart(_pgl(fig, 230), use_container_width=True)
        except Exception as e:
            st.info(f"{e}")

    with c2:
        section_title("Sales by Region")
        try:
            fig = viz.create_region_donut(df)
            st.plotly_chart(_pgl(fig, 230), use_container_width=True)
        except Exception as e:
            st.info(f"{e}")

    with c3:
        section_title("Profit Margin by Category")
        margin_data = viz.create_profit_margin_bars(df)
        for cat, pct, color in margin_data:
            _c(f"""
            <div style="margin-bottom:10px">
              <div style="display:flex;justify-content:space-between;font-size:10px;color:rgba(255,255,255,0.55);margin-bottom:3px">
                <span>{cat}</span><span style="color:{color};font-weight:600">{pct}%</span>
              </div>
              <div class="prog-track"><div class="prog-fill" style="width:{min(pct,100)}%;background:{color}"></div></div>
            </div>""")

    # Scatter anomaly + Correlation heatmap
    c4, c5 = st.columns([1.3, 1])
    with c4:
        section_title("Anomaly Detection Scatter", "IQR-based outlier highlighting")
        try:
            fig = viz.create_scatter_anomaly(df)
            st.plotly_chart(_pgl(fig, 250), use_container_width=True)
        except Exception as e:
            st.info(f"{e}")

    with c5:
        section_title("System Activity Feed")
        activities = [
            ("🟣", "<b>AI insight</b> — Revenue spike in West (+34%)",   "2 min ago"),
            ("🟢", f"<b>{st.session_state.user or 'Admin'}</b> uploaded dataset",    "Just now"),
            ("🔵", "<b>PDF report</b> generated",                         "1 hr ago"),
            ("🔴", "<b>Quality alert</b> — missing values found",         "Yesterday"),
        ]
        for ico, msg, tm in activities:
            _c(f'<div class="act-item"><span>{ico}</span><div>'
               f'<div style="font-size:11px;color:rgba(255,255,255,0.75)">{msg}</div>'
               f'<div style="font-size:9px;color:rgba(255,255,255,0.28);margin-top:1px">{tm}</div>'
               f'</div></div>')

    # Correlation heatmap
    section_title("Correlation Intelligence Matrix")
    if len(num_cols) >= 2:
        try:
            corr = df[num_cols[:8]].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Purples",
                            zmin=-1, zmax=1, title="Feature Correlations")
            st.plotly_chart(_pgl(fig, 320), use_container_width=True)
        except Exception as e:
            st.info(f"{e}")
    else:
        st.info("Need ≥ 2 numeric columns.")

    # Additional charts from auto_charts
    charts = st.session_state.charts or []
    if charts:
        section_title("Auto-Generated Charts")
        for i in range(0, min(len(charts), 4), 2):
            ca, cb = st.columns(2)
            for j, col_w in [(i, ca), (i+1, cb)]:
                if j < len(charts):
                    title, fig = charts[j]
                    with col_w:
                        st.plotly_chart(_pgl(fig, 240), use_container_width=True)

    if auto and elapsed >= iv:
        st.session_state["last_refresh"] = time.time()
        time.sleep(0.1); st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: FORECASTING
# ═══════════════════════════════════════════════════════════════════════════
def page_forecasting():
    df = _filtered_df()
    if df is None or df.empty:
        _no_data(); return

    section_title("Time-Series Forecasting", "Prophet · Holt-Winters fallback")
    forecast = st.session_state.forecast

    c1, c2, c3 = st.columns(3)
    with c1: _c(kpi_card("Model", "Prophet" if forecast and forecast.get("model") == "Prophet" else "Holt-Winters", "Auto-selected", True))
    with c2: _c(kpi_card("Forecast Horizon", "6 months", "Forward projection", True))
    with c3: _c(kpi_card("Confidence Band", "95%", "Upper/lower bounds", True))
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("▶️ Re-run Forecast", key="rerun_fc"):
        col_types = st.session_state.column_types or {}
        with st.spinner("Forecasting…"):
            result = fc.auto_forecast.clear() if hasattr(fc.auto_forecast, "clear") else None
            result = fc.auto_forecast(df, col_types)
            st.session_state.forecast = result
        st.rerun()

    if forecast:
        section_title("Forecast Chart", f"Model: {forecast.get('model','N/A')}")
        st.plotly_chart(_pgl(forecast["figure"], 300), use_container_width=True)

        c_tbl, c_sea = st.columns(2)
        with c_tbl:
            section_title("Forecast Table", "Next periods")
            f_df = forecast.get("forecast_df")
            if f_df is not None and not f_df.empty:
                disp = f_df.copy()
                for col in ["yhat", "yhat_lower", "yhat_upper"]:
                    if col in disp.columns:
                        disp[col] = disp[col].apply(lambda x: f"${x:,.0f}")
                disp.columns = [c.replace("yhat","Predicted").replace("_lower"," Lower").replace("_upper"," Upper")
                                for c in disp.columns]
                st.dataframe(disp, use_container_width=True, hide_index=True, height=260)

        with c_sea:
            section_title("Seasonal Indices", "Monthly deviation")
            months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            dummy  = [24,-8,18,12,6,10,-15,-2,30,45,12,55]
            for m, val in zip(months, dummy):
                col = "#4ade80" if val >= 0 else "#f87171"
                _c(f"""
                <div style="display:flex;align-items:center;gap:7px;margin-bottom:4px">
                  <div style="width:26px;font-size:9px;color:rgba(255,255,255,0.35)">{m}</div>
                  <div style="flex:1;height:5px;background:rgba(255,255,255,0.06);border-radius:3px;position:relative">
                    <div style="position:absolute;{'left:50%' if val<0 else 'left:50%'};height:100%;
                                width:{abs(val)/60*100:.0f}%;background:{col};border-radius:3px;
                                transform:translateX({'-100%' if val<0 else '0'})"></div>
                  </div>
                  <div style="width:32px;font-size:9px;color:{col};text-align:right">
                    {'+' if val>=0 else ''}{val}K</div>
                </div>""")
    else:
        st.info("No date column found in your dataset for forecasting. Upload a dataset with a date/time column.")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: AI & ML
# ═══════════════════════════════════════════════════════════════════════════
def page_aiml():
    df = _filtered_df()
    if df is None or df.empty:
        _no_data(); return

    col_types = st.session_state.column_types or {}
    ml_results = st.session_state.ml_results or []

    section_title("Auto ML Engine", "K-Means · Linear Regression · Classification · Feature Importance")

    if st.button("▶️ Re-run ML Analysis", key="rerun_ml"):
        with st.spinner("Training models…"):
            ml_engine_result = mle.auto_ml.clear() if hasattr(mle.auto_ml, "clear") else None
            result = mle.auto_ml(df, col_types)
            st.session_state.ml_results = result
        ml_results = st.session_state.ml_results or []
        st.rerun()

    if ml_results:
        # Model cards
        cols = st.columns(min(len(ml_results), 3))
        for i, res in enumerate(ml_results[:3]):
            color = ACCS[i % len(ACCS)]
            with cols[i % 3]:
                _c(f"""
                <div class="ml-card" style="border-left:3px solid {color};margin-bottom:8px">
                  <div style="font-size:11px;font-weight:600;color:rgba(255,255,255,0.9);margin-bottom:5px">{res['title']}</div>
                  <div style="font-size:9px;color:rgba(255,255,255,0.35);margin-bottom:8px">{res['type']} · {res['extra']}</div>
                  <div style="font-size:18px;font-weight:600;color:{color};line-height:1">{res['metric_value']}</div>
                  <div style="font-size:9px;color:rgba(255,255,255,0.35);margin-top:2px">{res['metric_name']}</div>
                </div>""")

        # Charts
        section_title("Model Visualizations")
        for i in range(0, min(len(ml_results), 4), 2):
            ca, cb = st.columns(2)
            for j, cw in [(i, ca), (i+1, cb)]:
                if j < len(ml_results) and ml_results[j].get("figure"):
                    with cw:
                        st.plotly_chart(_pgl(ml_results[j]["figure"], 270), use_container_width=True)
    else:
        st.info("Insufficient data for ML analysis. Need at least 10 rows and 2 numeric columns.")

    # Full analysis (Groq AI)
    with st.expander("🤖 Full AI Analysis Hub (Groq)", expanded=False):
        import analysis
        analysis.render_advanced_analysis(df)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD
# ═══════════════════════════════════════════════════════════════════════════
def page_upload():
    section_title("Upload Your Dataset", "CSV · Excel (.xlsx, .xls) · JSON · Max 200MB")
    uploaded = st.file_uploader(
        "Drag & drop or click to browse",
        type=["csv","xlsx","xls","json"],
        key="file_uploader",
    )
    if uploaded:
        _process_upload(uploaded)
    else:
        _c("""
        <div style="border:2px dashed rgba(167,139,250,0.25);border-radius:14px;
                    padding:3rem;text-align:center;margin-top:1rem">
          <div style="font-size:3rem">📂</div>
          <div style="font-size:14px;font-weight:500;color:rgba(255,255,255,0.7);margin:.5rem 0">
            Drag & drop your file here</div>
          <div style="font-size:11px;color:rgba(255,255,255,0.3)">
            Supports CSV, Excel, JSON · Auto-cleaning · AI analysis</div>
        </div>""")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════════════
def page_about():
    _c(glass_card("""
    <div style="font-size:20px;font-weight:600;color:#c4b5fd;margin-bottom:6px">Zero Click AI</div>
    <div style="font-size:11px;color:rgba(255,255,255,0.4);margin-bottom:14px">
      Python & AI Internship · Genesis Training Company · Glassmorphism Dashboard v2.0
    </div>
    <div style="font-size:12px;color:rgba(255,255,255,0.7);line-height:1.8">
      Zero Click AI is an intelligent data analysis platform that automates the entire analytics
      pipeline — from data ingestion and cleaning to ML modelling, forecasting, and AI-powered
      insights — all with a single CSV upload.<br><br>
      <span style="color:#a78bfa;font-weight:500">Module 1</span> — Universal Upload, Smart Profiling, Data Quality<br>
      <span style="color:#38bdf8;font-weight:500">Module 2</span> — NLP Chatbot, Query Intelligence, Context Memory<br>
      <span style="color:#4ade80;font-weight:500">Module 3</span> — Insight Generator, Predictive Engine, Recommendations<br>
      <span style="color:#fb923c;font-weight:500">Module 4</span> — Glass Dashboard, Visualizations, PDF Reports<br>
      <span style="color:#f472b6;font-weight:500">Module 5</span> — RBAC, Voice Bot, Email Scheduler
    </div>"""))


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: MEET OUR TEAM
# ═══════════════════════════════════════════════════════════════════════════

# Order matches the reference image exactly:
# Row 1: Naheen (TL), Manu, Dhaval, Mohammed Ammar, Yusuf
# Row 2: Vaishnavi, Anoosha, Snehal, Nazhat, Keerti, Samruddhi
TEAM = [
    # (name, role, module, color, gender, email, linkedin, github, is_lead)
    ("Naheen Kauser",           "Team Lead",           "Module 4 · Dashboard",  "#a78bfa", "F",
     "naheenkauser113@gmail.com",
     "https://in.linkedin.com/in/naheen-kauser-02957a323",
     "https://github.com/NaheenKauserr", True),
    ("Manu Naik",               "Systems Engineer",    "Module 5 · Integration","#60a5fa", "M",
     "manupnaik639@gmail.com",
     "https://www.linkedin.com/in/manu-naik-73bb702a7",
     "https://github.com/manunaik111", False),
    ("Dhaval Shah",             "NLP Engineer",        "Module 2 · Chatbot",    "#818cf8", "M",
     "d34058397@gmail.com",
     "https://www.linkedin.com/in/dhaval-shah1628",
     "https://github.com/Dhaval-max3", False),
    ("Mohammed Ammar\nBin Zameer", "Data Engineer",    "Module 1 · Data Mgmt",  "#38bdf8", "M",
     "mohammedammar060802@gmail.com",
     "https://www.linkedin.com/in/mohammed-ammar-bin-zameer-589220363/",
     "https://github.com/ammar3633", False),
    ("Yusuf Chonche",           "UI/Dashboard Dev",    "Module 4 · Dashboard",  "#34d399", "M",
     "yusufchonche0@gmail.com",
     "https://www.linkedin.com/in/yusuf-chonche-5114892ba/",
     "https://github.com/yusufchonche0-web", False),
    ("Vaishnavi Metri",         "Analytics Engineer",  "Module 3 · Analytics",  "#fbbf24", "F",
     "vaishnavimetri234@gmail.com",
     "https://www.linkedin.com/in/vaishnavi-metri-578b0835a",
     "https://github.com/vaishnavimetri234-v11s", False),
    ("Anoosha Kembhavi",        "Systems Engineer",    "Module 5 · Integration","#f9a8d4", "F",
     "anooshakembhavi@gmail.com",
     "http://www.linkedin.com/in/anoosha-kembhavi",
     "https://github.com/anooshakembhavi-afk", False),
    ("Snehal Anil Kamble",      "Data Engineer",       "Module 1 · Data Mgmt",  "#fb923c", "F",
     "kamblesnehal578@gmail.com",
     "https://www.linkedin.com/in/snehal-k-b48369318",
     "https://github.com/kamblesnehal578-sketch", False),
    ("Nazhat Aliya Naikwadi",   "Data Engineer",       "Module 1 · Data Mgmt",  "#f472b6", "F",
     "nazhatnaikwadi@gmail.com",
     "https://linkedin.com/in/nazhatnaikwadi",
     "https://github.com/nazhatnaikwadi", False),
    ("Keerti Gadigeppagoudar",  "Analytics Engineer",  "Module 3 · Analytics",  "#2dd4bf", "F",
     "keerti.s.g2020@gmail.com",
     "https://www.linkedin.com/in/keertig",
     "https://github.com/keertiG-1296", False),
    ("Samruddhi Patil",         "NLP Engineer",        "Module 2 · Chatbot",    "#4ade80", "F",
     "patilsamruddhi863@gmail.com",
     "https://www.linkedin.com/in/samruddhi-patil-a1575933a",
     "https://github.com/samruddhi128", False),
]

# ── Avatar SVG: rich illustrated bust silhouettes ──────────────────────────────
def _avatar_svg(name, color, gender, is_lead=False):
    """Generate a rich illustrated avatar SVG matching the reference image style."""
    initials = "".join(w[0].upper() for w in name.replace("\n", " ").split())[:2]

    # Skin tones vary by first letter of name for diversity
    skin_tones = ["#FDBCB4", "#F1C27D", "#E0AC69", "#C68642", "#8D5524"]
    skin = skin_tones[ord(name[0]) % len(skin_tones)]

    # Hair colours vary
    hair_colours = ["#2C1B0E", "#4A3728", "#1A1A1A", "#8B4513", "#654321"]
    hair = hair_colours[ord(name[-1]) % len(hair_colours)]

    lead_badge = ""
    lead_ring  = ""
    if is_lead:
        lead_badge = f'''
        <rect x="8" y="3" width="44" height="12" rx="6" fill="#F59E0B"/>
        <text x="30" y="12" text-anchor="middle" fill="white"
              font-family="Arial" font-size="7" font-weight="bold">TEAM LEAD</text>'''
        lead_ring = f'filter="url(#glow)"'

    if gender == "F":
        # Female avatar: hijab/long hair option
        name_lower = name.lower()
        has_hijab = any(n in name_lower for n in ["naheen", "nazhat", "anoosha"])
        if has_hijab:
            # Hijab style
            bust = f'''
            <!-- Hijab -->
            <ellipse cx="30" cy="23" rx="17" ry="19" fill="{color}" opacity="0.9"/>
            <ellipse cx="30" cy="35" rx="15" ry="6" fill="{color}" opacity="0.9"/>
            <!-- Face -->
            <ellipse cx="30" cy="24" rx="10" ry="11" fill="{skin}"/>
            <!-- Neck -->
            <rect x="26" y="34" width="8" height="6" rx="2" fill="{skin}"/>
            <!-- Shoulders/Body -->
            <ellipse cx="30" cy="52" rx="18" ry="10" fill="{color}" opacity="0.75"/>
            <!-- Eyes -->
            <ellipse cx="26" cy="22" rx="2" ry="2.2" fill="#1A1A1A"/>
            <ellipse cx="34" cy="22" rx="2" ry="2.2" fill="#1A1A1A"/>
            <circle cx="26.7" cy="21.3" r="0.7" fill="white"/>
            <circle cx="34.7" cy="21.3" r="0.7" fill="white"/>
            <!-- Smile -->
            <path d="M25 28 Q30 32 35 28" stroke="#C45858" stroke-width="1.2" fill="none" stroke-linecap="round"/>'''
        else:
            # Long hair female
            bust = f'''
            <!-- Hair back -->
            <ellipse cx="30" cy="20" rx="13" ry="14" fill="{hair}"/>
            <rect x="17" y="25" width="5" height="18" rx="2" fill="{hair}"/>
            <rect x="38" y="25" width="5" height="18" rx="2" fill="{hair}"/>
            <!-- Face -->
            <ellipse cx="30" cy="23" rx="10.5" ry="12" fill="{skin}"/>
            <!-- Neck -->
            <rect x="26" y="34" width="8" height="6" rx="2" fill="{skin}"/>
            <!-- Hair front -->
            <ellipse cx="30" cy="13" rx="11" ry="6" fill="{hair}"/>
            <!-- Shoulders -->
            <ellipse cx="30" cy="52" rx="18" ry="10" fill="{color}" opacity="0.75"/>
            <!-- Eyes -->
            <ellipse cx="26" cy="22" rx="2" ry="2.2" fill="#1A1A1A"/>
            <ellipse cx="34" cy="22" rx="2" ry="2.2" fill="#1A1A1A"/>
            <circle cx="26.7" cy="21.3" r="0.7" fill="white"/>
            <circle cx="34.7" cy="21.3" r="0.7" fill="white"/>
            <!-- Smile -->
            <path d="M25 28 Q30 32 35 28" stroke="#C45858" stroke-width="1.2" fill="none" stroke-linecap="round"/>
            <!-- Blush -->
            <ellipse cx="23" cy="27" rx="3" ry="2" fill="#F9A8D4" opacity="0.5"/>
            <ellipse cx="37" cy="27" rx="3" ry="2" fill="#F9A8D4" opacity="0.5"/>'''
    else:
        # Male avatar: short hair + collar
        bust = f'''
        <!-- Hair -->
        <ellipse cx="30" cy="18" rx="12" ry="8" fill="{hair}"/>
        <!-- Face -->
        <ellipse cx="30" cy="24" rx="10.5" ry="12" fill="{skin}"/>
        <!-- Neck -->
        <rect x="26" y="35" width="8" height="5" rx="2" fill="{skin}"/>
        <!-- Suit/collar -->
        <ellipse cx="30" cy="52" rx="18" ry="10" fill="{color}" opacity="0.8"/>
        <polygon points="28,40 30,45 26,52" fill="white" opacity="0.9"/>
        <polygon points="32,40 30,45 34,52" fill="white" opacity="0.9"/>
        <!-- Tie -->
        <polygon points="29,43 31,43 30.5,50 29.5,50" fill="{color}"/>
        <!-- Eyes -->
        <ellipse cx="26" cy="22" rx="2" ry="2" fill="#1A1A1A"/>
        <ellipse cx="34" cy="22" rx="2" ry="2" fill="#1A1A1A"/>
        <circle cx="26.7" cy="21.4" r="0.7" fill="white"/>
        <circle cx="34.7" cy="21.4" r="0.7" fill="white"/>
        <!-- Eyebrows -->
        <path d="M23.5 19 Q26 17.5 28.5 19" stroke="{hair}" stroke-width="1.2" fill="none"/>
        <path d="M31.5 19 Q34 17.5 36.5 19" stroke="{hair}" stroke-width="1.2" fill="none"/>
        <!-- Smile -->
        <path d="M25.5 28.5 Q30 33 34.5 28.5" stroke="#A0522D" stroke-width="1.2" fill="none" stroke-linecap="round"/>'''

    return f'''
    <svg viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg" width="72" height="72">
      <defs>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
          <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <radialGradient id="bg_{initials}" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="{color}" stop-opacity="0.25"/>
          <stop offset="100%" stop-color="{color}" stop-opacity="0.05"/>
        </radialGradient>
      </defs>
      <!-- Outer ring -->
      <circle cx="30" cy="30" r="29" fill="url(#bg_{initials})"
              stroke="{color}" stroke-width="2.5" {lead_ring}/>
      <!-- Inner clip circle -->
      <clipPath id="clip_{initials}">
        <circle cx="30" cy="30" r="26"/>
      </clipPath>
      <g clip-path="url(#clip_{initials})">
        <!-- Background gradient -->
        <circle cx="30" cy="30" r="26" fill="{color}" opacity="0.12"/>
        {bust}
      </g>
      {lead_badge}
    </svg>'''


def _member_card(name, role, module, color, gender, email, li_url, gh_url, is_lead,
                 theme="dark", compact=False):
    """Render a single team member card with rich avatar."""
    display_name = name.replace("\n", "<br>")
    avatar_svg   = _avatar_svg(name, color, gender, is_lead)

    # Theme-aware colours
    if theme == "pastel":
        card_bg      = f"rgba(255,255,255,0.92)"
        card_border  = f"1.5px solid {color}55"
        name_color   = "#1F2937"
        role_color   = color
        email_color  = "#6B7280"
        shadow       = f"0 4px 20px {color}25, 0 1px 3px rgba(0,0,0,0.08)"
        card_hover   = "transform:translateY(-4px)"
    else:  # dark
        card_bg      = f"rgba(10,15,40,0.82)"
        card_border  = f"1.5px solid {color}55"
        name_color   = "rgba(255,255,255,0.95)"
        role_color   = color
        email_color  = "rgba(255,255,255,0.45)"
        shadow       = f"0 4px 24px {color}22, 0 1px 3px rgba(0,0,0,0.4)"
        card_hover   = "transform:translateY(-4px)"

    fs_name  = "11px" if compact else "12px"
    fs_role  = "8.5px" if compact else "9px"
    min_h    = "200px" if compact else "220px"
    pad      = "12px 8px 10px" if compact else "16px 10px 14px"

    li_bg  = "rgba(0,119,181,0.2)"  if theme == "dark" else "rgba(0,119,181,0.1)"
    gh_bg  = "rgba(255,255,255,0.07)" if theme == "dark" else "rgba(0,0,0,0.06)"
    li_brd = "rgba(0,119,181,0.45)" if theme == "dark" else "rgba(0,119,181,0.3)"
    gh_brd = "rgba(255,255,255,0.2)" if theme == "dark" else "rgba(0,0,0,0.15)"
    li_txt = "#38bdf8"               if theme == "dark" else "#0077B5"
    gh_svg = "rgba(255,255,255,0.8)" if theme == "dark" else "rgba(0,0,0,0.7)"

    github_icon = f'''<svg width="13" height="13" viewBox="0 0 24 24" fill="{gh_svg}">
      <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205
      11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724
      -4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087
      -.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07
      1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665
      -.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303
      -.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3
      -.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23
      .645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61
      -2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896
      -.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297
      c0-6.627-5.373-12-12-12"/></svg>'''

    return f"""
    <div style="
      background:{card_bg};
      border:{card_border};
      border-radius:18px;
      padding:{pad};
      text-align:center;
      min-height:{min_h};
      box-shadow:{shadow};
      transition:transform .25s ease, box-shadow .25s ease;
      position:relative;
      overflow:hidden;">

      <!-- Subtle corner glow -->
      <div style="position:absolute;top:-20px;right:-20px;width:60px;height:60px;
                  border-radius:50%;background:{color};opacity:0.06;pointer-events:none;"></div>

      <!-- Avatar -->
      <div style="display:flex;justify-content:center;margin-bottom:8px;">
        {avatar_svg}
      </div>

      <!-- Name -->
      <div style="font-size:{fs_name};font-weight:700;color:{name_color};
                  margin-bottom:2px;line-height:1.3;">{display_name}</div>

      <!-- Role pill -->
      <div style="display:inline-block;font-size:7.5px;font-weight:600;
                  color:{role_color};background:{role_color}18;
                  border:1px solid {role_color}35;
                  border-radius:20px;padding:1px 8px;margin-bottom:7px;">
        {role}
      </div>

      <!-- Email -->
      <div style="display:flex;align-items:center;justify-content:center;gap:4px;margin-bottom:8px;">
        <svg width="9" height="9" viewBox="0 0 24 24" fill="none"
             stroke="{email_color}" stroke-width="2">
          <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>
          <polyline points="22,6 12,13 2,6"/>
        </svg>
        <a href="mailto:{email}" style="font-size:7.5px;color:{email_color};
           text-decoration:none;word-break:break-all;">{email}</a>
      </div>

      <!-- Social buttons -->
      <div style="display:flex;justify-content:center;gap:8px;">
        <a href="{li_url}" target="_blank"
           style="display:flex;align-items:center;justify-content:center;
                  width:28px;height:28px;border-radius:8px;
                  background:{li_bg};border:1px solid {li_brd};
                  color:{li_txt};font-size:10px;font-weight:700;text-decoration:none;">
          in
        </a>
        <a href="{gh_url}" target="_blank"
           style="display:flex;align-items:center;justify-content:center;
                  width:28px;height:28px;border-radius:8px;
                  background:{gh_bg};border:1px solid {gh_brd};
                  text-decoration:none;">
          {github_icon}
        </a>
      </div>
    </div>"""


def page_team():
    # ── Theme toggle ─────────────────────────────────────────────────────────
    col_h, col_t = st.columns([6, 1])
    with col_t:
        theme_dark = st.toggle("🌙 Dark", value=True, key="team_theme_dark")
    theme = "dark" if theme_dark else "pastel"

    # Background override for pastel mode
    if theme == "pastel":
        _c("""<style>
        .team-section-wrap { background: linear-gradient(135deg,#F8F4FF 0%,#EEF2FF 50%,#F0FDF4 100%) !important; }
        </style>""")

    # ── Hero header ───────────────────────────────────────────────────────────
    if theme == "dark":
        hdr_bg    = "linear-gradient(135deg,rgba(124,58,237,0.28) 0%,rgba(56,189,248,0.14) 50%,rgba(52,211,153,0.12) 100%)"
        hdr_bdr   = "1px solid rgba(167,139,250,0.35)"
        hdr_title = "#E0D4FF"
        hdr_sub   = "rgba(255,255,255,0.45)"
        dot_color = "#a78bfa"
        # Floating particle dots
        particles = "".join([
            f'<div style="position:absolute;width:{4+i%4}px;height:{4+i%4}px;border-radius:50%;'
            f'background:{["#a78bfa","#38bdf8","#34d399","#f472b6","#fbbf24"][i%5]};'
            f'top:{10+i*13%70}%;left:{5+i*17%85}%;opacity:{0.3+i%3*0.15};"></div>'
            for i in range(8)
        ])
    else:
        hdr_bg    = "linear-gradient(135deg,#F3EEFF 0%,#E8F4FE 50%,#E8FDF5 100%)"
        hdr_bdr   = "1.5px solid rgba(167,139,250,0.5)"
        hdr_title = "#4C1D95"
        hdr_sub   = "#6B7280"
        dot_color = "#7C3AED"
        particles = ""

    _c(f"""
    <div style="
      position:relative;overflow:hidden;
      background:{hdr_bg};
      border:{hdr_bdr};
      border-radius:20px;
      padding:28px 32px 22px;
      margin-bottom:22px;
      text-align:center;">

      {particles}

      <!-- Brain icon -->
      <div style="font-size:36px;margin-bottom:10px;filter:drop-shadow(0 0 12px {dot_color}66);">
        🧠
      </div>

      <div style="font-size:26px;font-weight:800;color:{hdr_title};
                  letter-spacing:.04em;text-transform:uppercase;margin-bottom:6px;">
        Meet Our Talented Team
      </div>

      <div style="display:flex;align-items:center;justify-content:center;gap:8px;margin:6px 0 10px;">
        <div style="width:30px;height:2px;background:{dot_color};border-radius:2px;"></div>
        <div style="width:8px;height:8px;border-radius:50%;background:{dot_color};"></div>
        <div style="width:30px;height:2px;background:{dot_color};border-radius:2px;"></div>
      </div>

      <div style="font-size:13px;color:{hdr_sub};">
        Recognizing our 11 interns who built this Python &amp; AI project
      </div>

      <!-- Tech icons row -->
      <div style="display:flex;justify-content:center;gap:16px;margin-top:14px;font-size:18px;">
        {''.join([
          f'<span style="filter:drop-shadow(0 0 6px {dot_color}66);">{e}</span>'
          for e in ["🤖","🔗","</>\u200b","🔍","📊","⚡","🧬","💡"]
        ])}
      </div>
    </div>""")

    # ── Row 1: 5 members ──────────────────────────────────────────────────────
    row1 = TEAM[:5]
    cols1 = st.columns(5, gap="small")
    for i, (name, role, module, color, gender, email, li_url, gh_url, is_lead) in enumerate(row1):
        with cols1[i]:
            _c(_member_card(name, role, module, color, gender, email,
                            li_url, gh_url, is_lead, theme=theme, compact=False))

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Row 2: 6 members ──────────────────────────────────────────────────────
    row2 = TEAM[5:]
    cols2 = st.columns(6, gap="small")
    for i, (name, role, module, color, gender, email, li_url, gh_url, is_lead) in enumerate(row2):
        with cols2[i]:
            _c(_member_card(name, role, module, color, gender, email,
                            li_url, gh_url, is_lead, theme=theme, compact=True))

    # ── Stats bar ──────────────────────────────────────────────────────────────
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    if theme == "dark":
        stats_bg  = "linear-gradient(90deg,rgba(124,58,237,0.12),rgba(56,189,248,0.08),rgba(124,58,237,0.12))"
        stats_brd = "1px solid rgba(167,139,250,0.22)"
        stats_txt = "rgba(255,255,255,0.65)"
        num_color = "#a78bfa"
    else:
        stats_bg  = "linear-gradient(90deg,#F3EEFF,#EEF2FF,#F3EEFF)"
        stats_brd = "1.5px solid rgba(124,58,237,0.25)"
        stats_txt = "#374151"
        num_color = "#7C3AED"

    stats = [
        ("11", "Interns", "👥"),
        ("5",  "Modules", "📦"),
        ("1",  "AI App",  "🤖"),
        ("∞",  "Ideas",   "💡"),
    ]
    stat_items = "".join([
        f'''<div style="text-align:center;padding:0 20px;
                        border-right:1px solid {stats_brd.split("1px solid ")[-1] if "solid" in stats_brd else "#ccc"};">
              <div style="font-size:24px;font-weight:800;color:{num_color};">{v}</div>
              <div style="font-size:9px;color:{stats_txt};text-transform:uppercase;letter-spacing:.08em;">{icon} {lbl}</div>
            </div>'''
        for v, lbl, icon in stats
    ])

    _c(f"""
    <div style="
      background:{stats_bg};
      border:{stats_brd};
      border-radius:16px;
      padding:16px 24px;
      display:flex;justify-content:center;align-items:center;gap:0;">
      {stat_items}
    </div>""")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Footer banner ──────────────────────────────────────────────────────────
    if theme == "dark":
        foot_bg  = "linear-gradient(90deg,rgba(124,58,237,0.14),rgba(56,189,248,0.09),rgba(52,211,153,0.09),rgba(124,58,237,0.14))"
        foot_brd = "1px solid rgba(167,139,250,0.22)"
        foot_txt = "rgba(255,255,255,0.65)"
    else:
        foot_bg  = "linear-gradient(90deg,#EDE9FE,#E0F2FE,#DCFCE7,#EDE9FE)"
        foot_brd = "1.5px solid rgba(124,58,237,0.2)"
        foot_txt = "#374151"

    _c(f"""
    <div style="
      background:{foot_bg};
      border:{foot_brd};
      border-radius:14px;
      padding:14px 24px;
      text-align:center;">
      <div style="font-size:14px;font-weight:600;color:{foot_txt};margin-bottom:10px;">
        🚀 Powering our Python &amp; AI project through innovative interns!
      </div>
      <div style="display:flex;justify-content:center;align-items:center;gap:20px;font-size:20px;">
        <span title="AI Brain">🧠</span>
        <span style="color:{foot_txt};opacity:.4;">→</span>
        <span title="Neural Network">🔗</span>
        <span style="color:{foot_txt};opacity:.4;">→</span>
        <span title="Code">💻</span>
        <span style="color:{foot_txt};opacity:.4;">→</span>
        <span title="Analytics">🔍</span>
        <span style="color:{foot_txt};opacity:.4;">→</span>
        <span title="Dashboard">📊</span>
      </div>
    </div>""")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ═══════════════════════════════════════════════════════════════════════════
def page_settings():
    section_title("Settings", "Account · Users · Data management")
    c1, c2, c3 = st.columns(3)

    with c1:
        _c('<div class="g-card"><div class="sec-hdr">Account</div>')
        uname = st.session_state.get("user","Admin")
        _c(f"""
        <div style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);
                    border-radius:10px;padding:10px;margin-bottom:10px">
          <div style="font-size:12px;font-weight:500;color:rgba(255,255,255,0.9)">{uname}</div>
          <div style="font-size:10px;color:#4ade80;margin-top:3px">Session active</div>
        </div></div>""")
        if st.button("🚪 Logout", key="s_logout", use_container_width=True):
            st.session_state.logged_in = False; st.rerun()

    with c2:
        _c('<div class="g-card"><div class="sec-hdr">Users & Roles</div>')
        users = _load_users()
        roles = ["admin", "analyst", "viewer"]
        for uname, _ in list(users.items())[:4]:
            role_idx = 0 if uname in ["admin","yusuf"] else 1
            col = ["#4ade80","#fbbf24","rgba(255,255,255,0.4)"][role_idx]
            _c(f"""
            <div style="display:flex;align-items:center;gap:8px;padding:5px 8px;
                        border-radius:8px;background:rgba(255,255,255,0.04);
                        border:1px solid rgba(255,255,255,0.06);margin-bottom:4px">
              <div style="width:22px;height:22px;border-radius:50%;background:linear-gradient(135deg,#a78bfa,#7c3aed);
                          display:flex;align-items:center;justify-content:center;font-size:8px;font-weight:600;color:white">
                {uname[:2].upper()}</div>
              <div style="flex:1;font-size:10px;color:rgba(255,255,255,0.8)">{uname}</div>
              <span style="font-size:8px;padding:1px 6px;border-radius:10px;font-weight:600;
                           color:{col};background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1)">
                {roles[role_idx]}</span>
            </div>""")
        _c('</div>')

    with c3:
        _c('<div class="g-card"><div class="sec-hdr">Data</div>')
        if st.button("🗑️ Clear Dataset", use_container_width=True, key="clear_data"):
            for k in ["df","cleaned_df","kpis","charts","ml_results","forecast","insights_text","column_types","correlations"]:
                st.session_state[k] = None
            st.success("Cleared!"); st.rerun()
        if st.button("💬 Clear Chat History", use_container_width=True, key="clear_chat"):
            st.session_state.chat_history = []
            st.success("Chat cleared!")
        _c('</div>')

# ═══════════════════════════════════════════════════════════════════════════
# FLOATING CHATBOT (streamlit-float)
# ═══════════════════════════════════════════════════════════════════════════
def render_floating_chatbot():
    try:
        from streamlit_float import float_init, float_parent, float_dialog
        float_init(theme=False)
    except ImportError:
        _render_sidebar_chatbot()
        return

    if "chat_open" not in st.session_state:
        st.session_state.chat_open = False

    with st.sidebar:
        pass  # float elements can't go in sidebar

    button_css = """
        position: fixed; bottom: 24px; right: 24px; z-index: 9999;
        width: 52px; height: 52px; border-radius: 50%;
        background: linear-gradient(135deg, #a78bfa, #7c3aed);
        display: flex; align-items: center; justify-content: center;
        cursor: pointer; box-shadow: 0 4px 20px rgba(124,58,237,0.4);
    """
    st.markdown(f"""
    <div style="{button_css}" onclick="window.dispatchEvent(new Event('toggle_chat'))">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5">
        <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
      </svg>
    </div>""", unsafe_allow_html=True)

    with st.expander("🤖 AI Chatbot", expanded=st.session_state.get("chat_open", False)):
        _render_chat_ui()

def _render_sidebar_chatbot():
    """Fallback: render chatbot in a bottom expander."""
    with st.expander("🤖 AI Chatbot (Powered by Groq)", expanded=False):
        _render_chat_ui()

def _render_chat_ui():
    df = st.session_state.cleaned_df
    col_types = st.session_state.column_types or {}

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"]=="assistant" else "👤"):
            st.markdown(msg["content"])
            if msg.get("fig"):
                st.plotly_chart(_pgl(msg["fig"], 220), use_container_width=True)

    user_input = st.chat_input("Ask about your data…", key="chat_inp")
    if user_input:
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role":"user","content":user_input})

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking…"):
                resp = cb.chat_response(user_input, df, col_types)
            st.markdown(resp["text"])
            fig_out = None
            if resp.get("code") and df is not None:
                try:
                    local_vars = {"df": df, "pd": pd, "np": np,
                                  "px": px, "go": go}
                    exec(resp["code"], local_vars)
                    if "fig" in local_vars:
                        fig_out = local_vars["fig"]
                        st.plotly_chart(_pgl(fig_out, 220), use_container_width=True)
                except Exception as e:
                    st.caption(f"Chart error: {e}")

        st.session_state.chat_history.append({
            "role": "assistant", "content": resp["text"], "fig": fig_out
        })

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # Remember-me token
    if not st.session_state.logged_in and "user" in st.query_params:
        users = _load_users()
        token_user = st.query_params["user"]
        if token_user in users:
            st.session_state.logged_in = True
            st.session_state.user = token_user

    if not st.session_state.logged_in:
        login_page()
        return

    render_sidebar()

    page = st.session_state.page
    render_topbar(page)
    st.divider()

    if page == "Analytics":     page_analytics()
    elif page == "Visualization": page_visualization()
    elif page == "Forecasting":   page_forecasting()
    elif page == "AI & ML":       page_aiml()
    elif page == "Upload Data":   page_upload()
    elif page == "About":         page_about()
    elif page == "Meet Our Team": page_team()
    elif page == "Settings":      page_settings()
    else:                         page_analytics()

    # Floating chatbot (always on, every page)
    render_floating_chatbot()

if __name__ == "__main__":
    main()
