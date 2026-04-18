"""
app.py — Zero Click AI · Genesis Training Company
Stripe-inspired theme: Clean, professional, off-white with refined accents
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
# STRIPE-INSPIRED CSS THEME
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

/* Main app background - clean off-white like Stripe */
.stApp {
    background-color: #F9FAFB !important;
    background: linear-gradient(to bottom, #FFFFFF, #F9FAFB) !important;
}

/* Sidebar - light gray/white with subtle border */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] * {
    color: #1F2937 !important;
}

[data-testid="stSidebar"] .stMarkdown, 
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #4B5563 !important;
}

[data-testid="stSidebar"] select {
    background-color: #F9FAFB !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    color: #1F2937 !important;
    font-size: 13px !important;
    padding: 6px 10px !important;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: #E5E7EB !important;
    margin: 0.75rem 0 !important;
}

/* Main content area */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Cards / Metric containers - clean white with subtle border and shadow */
[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 12px !important;
    padding: 1rem 1rem !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    transition: all 0.2s ease !important;
}

[data-testid="metric-container"]:hover {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    border-color: #D1D5DB !important;
}

[data-testid="stMetricValue"] {
    color: #111827 !important;
    font-weight: 600 !important;
    font-size: 1.75rem !important;
}

[data-testid="stMetricLabel"] {
    color: #6B7280 !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.03em !important;
}

[data-testid="stMetricDelta"] {
    color: #059669 !important;
    font-size: 0.75rem !important;
}

/* Dataframe - clean table styling */
[data-testid="stDataFrame"] {
    background: #FFFFFF !important;
    border-radius: 12px !important;
    border: 1px solid #E5E7EB !important;
    overflow: hidden !important;
}

.dataframe {
    font-family: 'Inter', monospace !important;
    font-size: 0.8rem !important;
}

.dataframe th {
    background-color: #F9FAFB !important;
    color: #374151 !important;
    font-weight: 600 !important;
    border-bottom: 1px solid #E5E7EB !important;
}

/* Expander - minimalist */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
}

[data-testid="stExpander"] details {
    border-radius: 12px !important;
}

/* Buttons - Stripe-inspired dark primary, subtle secondary */
.stButton > button {
    background-color: #0F172A !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 40px !important;
    padding: 0.5rem 1rem !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

.stButton > button:hover {
    background-color: #1E293B !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* Secondary button style */
.stButton > button[data-baseweb="button"][kind="secondary"] {
    background-color: #FFFFFF !important;
    color: #1F2937 !important;
    border: 1px solid #D1D5DB !important;
}

.stButton > button[data-baseweb="button"][kind="secondary"]:hover {
    background-color: #F9FAFB !important;
    border-color: #9CA3AF !important;
}

/* Input fields - clean borders */
.stTextInput > div > div > input, 
.stTextArea textarea,
.stSelectbox > div > div {
    background-color: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 8px !important;
    color: #111827 !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 0.75rem !important;
}

.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.1) !important;
    outline: none !important;
}

/* Chat elements */
[data-testid="stChatInput"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 24px !important;
}

[data-testid="stChatMessage"] {
    background: #F9FAFB !important;
    border-radius: 16px !important;
    border: 1px solid #E5E7EB !important;
    padding: 0.75rem !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #6366F1, #8B5CF6) !important;
    border-radius: 20px !important;
}

/* Selectbox dropdown */
[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border-color: #D1D5DB !important;
    border-radius: 8px !important;
}

/* Tabs - Stripe style */
[data-baseweb="tab-list"] {
    gap: 1.5rem !important;
    border-bottom: 1px solid #E5E7EB !important;
}

[data-baseweb="tab"] {
    color: #6B7280 !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 0 !important;
}

[aria-selected="true"] {
    color: #0F172A !important;
    border-bottom: 2px solid #0F172A !important;
}

/* Radio buttons */
[data-testid="stRadio"] > div {
    gap: 1rem !important;
}

[data-testid="stRadio"] > div > label {
    color: #374151 !important;
    font-size: 0.875rem !important;
}

/* Checkbox */
[data-testid="stCheckbox"] label {
    color: #374151 !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, .stDeployButton {
    visibility: hidden;
}

/* Custom card components */
.g-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    transition: all 0.2s ease;
}

.g-card:hover {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    border-color: #D1D5DB;
}

/* KPI card */
.kpi-glass {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    transition: all 0.2s ease;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}

.kpi-glass:hover {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    border-color: #D1D5DB;
}

.kpi-val {
    font-size: 1.75rem;
    font-weight: 600;
    color: #111827;
    line-height: 1.2;
    margin-top: 0.25rem;
}

.kpi-lbl {
    font-size: 0.7rem;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    font-weight: 500;
}

.kpi-dt {
    font-size: 0.7rem;
    margin-top: 0.5rem;
}

.up { color: #059669; }
.dn { color: #DC2626; }

/* Insight banner */
.ins-bar {
    background: #F0FDF4;
    border: 1px solid #D1FAE5;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
}

/* Recommendation card */
.rec-card {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.8rem;
    color: #374151;
    line-height: 1.5;
}

/* Activity item */
.act-item {
    display: flex;
    gap: 0.75rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid #F3F4F6;
    font-size: 0.8rem;
    color: #4B5563;
}

/* Status pills */
.sp {
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-weight: 500;
}

.sp-g {
    background: #ECFDF5;
    color: #059669;
    border: 1px solid #D1FAE5;
}

.sp-y {
    background: #FFFBEB;
    color: #D97706;
    border: 1px solid #FEF3C7;
}

.sp-r {
    background: #FEF2F2;
    color: #DC2626;
    border: 1px solid #FEE2E2;
}

/* Section headers */
.sec-hdr {
    font-size: 1rem;
    font-weight: 600;
    color: #111827;
    margin: 1rem 0 0.75rem;
    letter-spacing: -0.01em;
}

.sec-sub {
    font-size: 0.75rem;
    color: #6B7280;
    margin-top: 0.25rem;
}

/* ML model card */
.ml-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1rem;
    transition: all 0.2s ease;
}

.ml-card:hover {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    border-color: #D1D5DB;
}

/* Team card */
.team-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 1rem;
    text-align: center;
    transition: all 0.2s ease;
}

.team-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    border-color: #D1D5DB;
}

/* Top bar */
.topbar-wrap {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Real-time bar */
.rt-bar {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.75rem;
    color: #4B5563;
    margin-bottom: 1rem;
}

/* Progress track */
.prog-track {
    height: 4px;
    border-radius: 2px;
    background: #E5E7EB;
    overflow: hidden;
}

.prog-fill {
    height: 100%;
    border-radius: 2px;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #111827 !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
}

h1 {
    font-size: 1.875rem !important;
}

h2 {
    font-size: 1.5rem !important;
}

/* Links */
a {
    color: #4F46E5 !important;
    text-decoration: none !important;
}

a:hover {
    color: #6366F1 !important;
    text-decoration: underline !important;
}

/* Code blocks */
code {
    background-color: #F3F4F6 !important;
    color: #1F2937 !important;
    border-radius: 6px !important;
    padding: 0.2rem 0.4rem !important;
    font-size: 0.8rem !important;
}

/* Alert boxes */
.stAlert {
    background-color: #FEFCE8 !important;
    border-left: 4px solid #EAB308 !important;
    border-radius: 8px !important;
    padding: 0.75rem 1rem !important;
}

.stAlert .stMarkdown p {
    color: #854D0E !important;
}

/* Success message */
.element-container div[data-testid="stAlert"]:has(svg[data-icon="check-circle"]) {
    background-color: #ECFDF5 !important;
    border-left-color: #10B981 !important;
}

.element-container div[data-testid="stAlert"]:has(svg[data-icon="check-circle"]) p {
    color: #065F46 !important;
}

/* Error message */
.element-container div[data-testid="stAlert"]:has(svg[data-icon="alert-octagon"]) {
    background-color: #FEF2F2 !important;
    border-left-color: #EF4444 !important;
}

/* Info message */
.element-container div[data-testid="stAlert"]:has(svg[data-icon="info-circle"]) {
    background-color: #EFF6FF !important;
    border-left-color: #3B82F6 !important;
}

/* Toggle switches */
.stToggle {
    background-color: #F3F4F6 !important;
    border-radius: 32px !important;
}

.stToggle[data-baseweb="toggle"] {
    background-color: #E5E7EB !important;
}

/* Slider */
.stSlider > div > div {
    background-color: #E5E7EB !important;
}

.stSlider > div > div > div {
    background-color: #0F172A !important;
}

/* Select slider labels */
.stSlider label {
    color: #374151 !important;
}
</style>
""", unsafe_allow_html=True)

# Update plotly theme to match Stripe's clean aesthetic
PLOTLY = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#F9FAFB",
    font=dict(color="#374151", size=11, family="Inter, sans-serif"),
    xaxis=dict(
        showgrid=False, 
        color="#9CA3AF",
        title_font=dict(size=11, color="#6B7280"),
        tickfont=dict(size=10, color="#6B7280")
    ),
    yaxis=dict(
        gridcolor="#E5E7EB", 
        color="#9CA3AF",
        title_font=dict(size=11, color="#6B7280"),
        tickfont=dict(size=10, color="#6B7280")
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)", 
        font=dict(color="#374151", size=10),
        bordercolor="#E5E7EB",
        borderwidth=1
    ),
    margin=dict(l=0, r=0, t=32, b=0),
    height=280,
    hoverlabel=dict(bgcolor="#FFFFFF", font_size=10, font_color="#1F2937"),
    plot_bgcolor="#FFFFFF",
)

# Stripe-inspired color palette
STRIPE_COLORS = ["#0F172A", "#3B82F6", "#10B981", "#8B5CF6", "#F59E0B", "#EF4444", "#06B6D4", "#EC4899"]

def _pgl(fig, h=280):
    fig.update_layout(**{**PLOTLY, "height": h})
    return fig

ACCS = STRIPE_COLORS

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE (unchanged)
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
# AUTH (unchanged)
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
# HTML HELPERS (updated for Stripe theme)
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

def pill(text, color="#0F172A", bg_alpha=0.08):
    rgb = utils.hex_to_rgb(color)
    return (f'<span style="font-size:0.7rem;padding:0.2rem 0.7rem;border-radius:20px;font-weight:500;'
            f'background:rgba({rgb},{bg_alpha});color:{color};border:1px solid rgba({rgb},0.15)">'
            f'{text}</span>')

# ═══════════════════════════════════════════════════════════════════════════
# LOGIN PAGE (Stripe-inspired styling)
# ═══════════════════════════════════════════════════════════════════════════
def login_page():
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        _c("""
        <div style="text-align: center; padding: 2rem 0 1.5rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🧠</div>
            <h1 style="font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; background: linear-gradient(135deg, #0F172A, #3B82F6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0.5rem 0;">
                Zero Click AI
            </h1>
            <p style="color: #6B7280; font-size: 0.85rem; margin-top: 0.25rem;">
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
        _c('<div style="text-align:center; font-size:0.7rem; color:#9CA3AF; margin-top:1rem;">admin / admin · demo / demo123</div>')

# ═══════════════════════════════════════════════════════════════════════════
# DATA PROCESSING (unchanged)
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
# SIDEBAR (Stripe-inspired)
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
        <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.5rem 0 1rem; border-bottom: 1px solid #E5E7EB; margin-bottom: 1rem;">
            <div style="width: 32px; height: 32px; border-radius: 8px; background: linear-gradient(135deg, #0F172A, #3B82F6); display: flex; align-items: center; justify-content: center; font-size: 1rem;">🧠</div>
            <div>
                <div style="font-size: 1rem; font-weight: 600; color: #111827;">Zero Click AI</div>
                <div style="font-size: 0.65rem; color: #6B7280; text-transform: uppercase; letter-spacing: 0.03em;">Genesis Training</div>
            </div>
        </div>""")
        
        # Dashboards nav
        _c('<div style="font-size: 0.7rem; font-weight: 600; color: #6B7280; text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 0.5rem;">Dashboards</div>')
        for page in ["Analytics", "Visualization", "Forecasting", "AI & ML"]:
            is_on = st.session_state.page == page
            if st.button(f"{PAGE_ICONS[page]}  {page}", key=f"nav_{page}",
                         use_container_width=True,
                         type="primary" if is_on else "secondary"):
                st.session_state.page = page
                st.rerun()

        st.divider()
        _c('<div style="font-size: 0.7rem; font-weight: 600; color: #6B7280; text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 0.5rem;">General</div>')
        for page in ["Upload Data", "About", "Meet Our Team"]:
            is_on = st.session_state.page == page
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
            _c('<div style="font-size: 0.7rem; font-weight: 600; color: #6B7280; text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 0.5rem;">Filters</div>')
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
        <div style="background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 10px; padding: 0.6rem 0.75rem; display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem;">
            <div style="width: 28px; height: 28px; border-radius: 50%; background: linear-gradient(135deg, #0F172A, #3B82F6); display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: 600; color: white;">
                {uname[:2].upper()}
            </div>
            <div>
                <div style="font-size: 0.8rem; font-weight: 500; color: #111827;">{uname}</div>
                <div style="font-size: 0.65rem; color: #6B7280;">Admin · Online</div>
            </div>
        </div>""")
        if st.button("🚪 Logout", use_container_width=True, key="logout"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# TOPBAR + PDF BUTTON (unchanged)
# ═══════════════════════════════════════════════════════════════════════════
def render_topbar(page):
    df = st.session_state.cleaned_df
    fname = ""
    if df is not None:
        fname = f"· {len(df):,} rows × {len(df.columns)} cols"

    col_title, col_rpt = st.columns([5, 2])
    with col_title:
        _c(f"""
        <div style="padding: 0.25rem 0 0.5rem;">
            <div style="font-size: 1.25rem; font-weight: 600; color: #111827; letter-spacing: -0.01em;">{page}</div>
            <div style="font-size: 0.7rem; color: #6B7280;">Home › {page} {fname}</div>
        </div>""")

    with col_rpt:
        if page in ("Analytics", "Visualization", "Forecasting") and df is not None:
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
# APPLY FILTERS (unchanged)
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
# NO-DATA GUARD (unchanged)
# ═══════════════════════════════════════════════════════════════════════════
def _no_data():
    _c("""
    <div style="text-align: center; padding: 3rem 2rem;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📂</div>
        <h2 style="color: #111827; margin: 0.5rem 0;">No Dataset Loaded</h2>
        <p style="color: #6B7280;">Go to <strong>Upload Data</strong> in the sidebar to get started.</p>
    </div>""")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS (Stripe-inspired)
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
                color = "#10B981" if any(w in line.lower() for w in ["strong","profitable","good","positive"]) \
                        else "#F59E0B" if any(w in line.lower() for w in ["warning","high","delay","missing","weak"]) \
                        else "#3B82F6"
                chips_html += f'<span style="font-size:0.7rem;padding:0.25rem 0.75rem;border-radius:20px;margin:0.2rem;display:inline-block;background:rgba({utils.hex_to_rgb(color)},0.08);color:{color};border:1px solid rgba({utils.hex_to_rgb(color)},0.15)">{line[:80]}</span>'
    _c(f"""
    <div class="ins-bar">
        <div style="font-size:0.75rem;font-weight:600;color:#0F172A;margin-bottom:0.5rem;">⚡ AI insights from your dataset</div>
        <div style="display:flex;flex-wrap:wrap;gap:0.3rem;">{chips_html}</div>
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
                fig.add_trace(go.Bar(x=agg["_p"], y=agg[s_col], name=s_col, marker_color="#0F172A", opacity=0.85))
                if p_col:
                    fig.add_trace(go.Bar(x=agg["_p"], y=agg[p_col], name="Profit", marker_color="#10B981", opacity=0.85))
                fig.update_layout(barmode="group", **{**PLOTLY, "height":280})
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                if s_col:
                    fig = px.bar(df.head(50), y=s_col, color_discrete_sequence=["#0F172A"])
                    st.plotly_chart(_pgl(fig), use_container_width=True)
        elif s_col:
            fig = px.bar(df.head(50), y=s_col, color_discrete_sequence=["#0F172A"])
            st.plotly_chart(_pgl(fig), use_container_width=True)
        else:
            st.info("Requires a numeric column.")

    with col_donut:
        section_title("Sales by Category")
        if cat_col and s_col:
            try:
                cat_df = df.groupby(cat_col)[s_col].sum().reset_index().sort_values(s_col, ascending=False).head(6)
                fig = px.pie(cat_df, names=cat_col, values=s_col, hole=0.5,
                             color_discrete_sequence=STRIPE_COLORS)
                fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=9)
                fig.update_layout(**{**PLOTLY, "height":280, "showlegend":False})
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
            recs.append(("Reduce discount levels — high discounting compressing margins", "#EF4444"))
        if any("delay" in str(k).lower() or "shipping" in str(k).lower() for k in kv):
            recs.append(("Optimise logistics — shipping delays exceed threshold", "#F59E0B"))
        recs.append(("Scale top-performing categories to maximise revenue", "#10B981"))
        recs.append(("Leverage strong customer base with loyalty programmes", "#0F172A"))
        if not recs:
            recs = [("Explore Analysis tab for deeper patterns", "#0F172A"),
                    ("Run ML models for anomaly detection", "#3B82F6")]
        for text, color in recs[:4]:
            _c(f'<div class="rec-card"><span style="color:{color};font-weight:600">→ </span>{text}</div>')

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: VISUALIZATION (Stripe-inspired)
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

    dot_col = "#10B981" if elapsed < interval else "#F59E0B"
    _c(f'<div class="rt-bar">'
       f'<span style="display:inline-block;width:8px;height:8px;background:{dot_col};border-radius:50%;margin-right:8px;"></span>'
       f'Live · last refresh <b style="color:{dot_col}">{elapsed}s ago</b> · '
       f'Dataset: <b style="color:#0F172A">{len(df):,}</b> rows</div>')

    num_cols = [c for c in col_types.get("numeric", []) if not any(k in c.lower() for k in ["id","index","code"])]
    time_col = next(iter(col_types.get("datetime", [])), None)
    s_col    = next((c for c in num_cols if any(w in c.lower() for w in ["sales","revenue","amount"])), num_cols[0] if num_cols else None)
    cat_col  = next(iter(col_types.get("categorical", [])), None)
    reg_col  = next((c for c in df.columns if "region" in c.lower()), None)

    # YoY area chart
    section_title("Year-over-Year Comparison", "Sneat-style gradient area chart")
    try:
        fig = viz.create_yoy_area_chart(df)
        fig.update_layout(**PLOTLY)
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
            <div style="margin-bottom:0.75rem;">
                <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#4B5563;margin-bottom:0.25rem;">
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
            ("🔵", "<b>AI insight</b> — Revenue spike in West (+34%)",   "2 min ago"),
            ("🟢", f"<b>{st.session_state.user or 'Admin'}</b> uploaded dataset",    "Just now"),
            ("⚪", "<b>PDF report</b> generated",                         "1 hr ago"),
            ("🟠", "<b>Quality alert</b> — missing values found",         "Yesterday"),
        ]
        for ico, msg, tm in activities:
            _c(f'<div class="act-item"><span>{ico}</span><div>'
               f'<div style="font-size:0.75rem;color:#374151">{msg}</div>'
               f'<div style="font-size:0.65rem;color:#6B7280;margin-top:0.1rem;">{tm}</div>'
               f'</div></div>')

    # Correlation heatmap
    section_title("Correlation Intelligence Matrix")
    if len(num_cols) >= 2:
        try:
            corr = df[num_cols[:8]].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Blues",
                            zmin=-1, zmax=1, title="Feature Correlations")
            fig.update_layout(**PLOTLY)
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
                        fig.update_layout(**PLOTLY)
                        st.plotly_chart(_pgl(fig, 240), use_container_width=True)

    if auto and elapsed >= iv:
        st.session_state["last_refresh"] = time.time()
        time.sleep(0.1); st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: FORECASTING (Stripe-inspired)
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
        fig = forecast["figure"]
        fig.update_layout(**PLOTLY)
        st.plotly_chart(_pgl(fig, 300), use_container_width=True)

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
                col = "#10B981" if val >= 0 else "#EF4444"
                _c(f"""
                <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.35rem;">
                    <div style="width:30px;font-size:0.7rem;color:#6B7280;">{m}</div>
                    <div style="flex:1;height:4px;background:#E5E7EB;border-radius:2px;position:relative;">
                        <div style="position:absolute;{'left:50%' if val<0 else 'left:50%'};height:100%;
                                    width:{abs(val)/60*100:.0f}%;background:{col};border-radius:2px;
                                    transform:translateX({'-100%' if val<0 else '0'})"></div>
                    </div>
                    <div style="width:35px;font-size:0.7rem;color:{col};text-align:right;">
                        {'+' if val>=0 else ''}{val}K</div>
                </div>""")
    else:
        st.info("No date column found in your dataset for forecasting. Upload a dataset with a date/time column.")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: AI & ML (Stripe-inspired)
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
            color = STRIPE_COLORS[i % len(STRIPE_COLORS)]
            with cols[i % 3]:
                _c(f"""
                <div class="ml-card" style="border-left:3px solid {color};margin-bottom:0.5rem;">
                    <div style="font-size:0.75rem;font-weight:600;color:#111827;margin-bottom:0.25rem;">{res['title']}</div>
                    <div style="font-size:0.65rem;color:#6B7280;margin-bottom:0.5rem;">{res['type']} · {res['extra']}</div>
                    <div style="font-size:1.25rem;font-weight:600;color:{color};line-height:1;">{res['metric_value']}</div>
                    <div style="font-size:0.65rem;color:#6B7280;margin-top:0.2rem;">{res['metric_name']}</div>
                </div>""")

        # Charts
        section_title("Model Visualizations")
        for i in range(0, min(len(ml_results), 4), 2):
            ca, cb = st.columns(2)
            for j, cw in [(i, ca), (i+1, cb)]:
                if j < len(ml_results) and ml_results[j].get("figure"):
                    with cw:
                        fig = ml_results[j]["figure"]
                        fig.update_layout(**PLOTLY)
                        st.plotly_chart(_pgl(fig, 270), use_container_width=True)
    else:
        st.info("Insufficient data for ML analysis. Need at least 10 rows and 2 numeric columns.")

    # Full analysis (Groq AI)
    with st.expander("🤖 Full AI Analysis Hub (Groq)", expanded=False):
        import analysis
        analysis.render_advanced_analysis(df)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD (unchanged)
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
        <div style="border:2px dashed #E5E7EB;border-radius:16px;padding:3rem;text-align:center;margin-top:1rem;background:#FFFFFF;">
            <div style="font-size:3rem;margin-bottom:0.75rem;">📂</div>
            <div style="font-size:0.9rem;font-weight:500;color:#111827;margin:0.5rem 0;">Drag & drop your file here</div>
            <div style="font-size:0.7rem;color:#6B7280;">Supports CSV, Excel, JSON · Auto-cleaning · AI analysis</div>
        </div>""")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT (Stripe-inspired)
# ═══════════════════════════════════════════════════════════════════════════
def page_about():
    _c(glass_card("""
    <div style="font-size:1.25rem;font-weight:600;color:#111827;margin-bottom:0.5rem;">Zero Click AI</div>
    <div style="font-size:0.7rem;color:#6B7280;margin-bottom:1rem;">Python & AI Internship · Genesis Training Company · Stripe-inspired Dashboard v3.0</div>
    <div style="font-size:0.8rem;color:#374151;line-height:1.6;">
        Zero Click AI is an intelligent data analysis platform that automates the entire analytics
        pipeline — from data ingestion and cleaning to ML modelling, forecasting, and AI-powered
        insights — all with a single CSV upload.<br><br>
        <span style="color:#0F172A;font-weight:500">Module 1</span> — Universal Upload, Smart Profiling, Data Quality<br>
        <span style="color:#3B82F6;font-weight:500">Module 2</span> — NLP Chatbot, Query Intelligence, Context Memory<br>
        <span style="color:#10B981;font-weight:500">Module 3</span> — Insight Generator, Predictive Engine, Recommendations<br>
        <span style="color:#8B5CF6;font-weight:500">Module 4</span> — Glass Dashboard, Visualizations, PDF Reports<br>
        <span style="color:#F59E0B;font-weight:500">Module 5</span> — RBAC, Voice Bot, Email Scheduler
    </div>"""))

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: MEET OUR TEAM (Stripe-inspired minimalist)
# ═══════════════════════════════════════════════════════════════════════════

TEAM = [
    ("Naheen Kauser",           "Team Lead",           "Module 4 · Dashboard",  "#0F172A", "F",
     "naheenkauser113@gmail.com",
     "https://in.linkedin.com/in/naheen-kauser-02957a323",
     "https://github.com/NaheenKauserr", True),
    ("Manu Naik",               "Systems Engineer",    "Module 5 · Integration","#3B82F6", "M",
     "manupnaik639@gmail.com",
     "https://www.linkedin.com/in/manu-naik-73bb702a7",
     "https://github.com/manunaik111", False),
    ("Dhaval Shah",             "NLP Engineer",        "Module 2 · Chatbot",    "#8B5CF6", "M",
     "d34058397@gmail.com",
     "https://www.linkedin.com/in/dhaval-shah1628",
     "https://github.com/Dhaval-max3", False),
    ("Mohammed Ammar\nBin Zameer", "Data Engineer",    "Module 1 · Data Mgmt",  "#10B981", "M",
     "mohammedammar060802@gmail.com",
     "https://www.linkedin.com/in/mohammed-ammar-bin-zameer-589220363/",
     "https://github.com/ammar3633", False),
    ("Yusuf Chonche",           "UI/Dashboard Dev",    "Module 4 · Dashboard",  "#F59E0B", "M",
     "yusufchonche0@gmail.com",
     "https://www.linkedin.com/in/yusuf-chonche-5114892ba/",
     "https://github.com/yusufchonche0-web", False),
    ("Vaishnavi Metri",         "Analytics Engineer",  "Module 3 · Analytics",  "#06B6D4", "F",
     "vaishnavimetri234@gmail.com",
     "https://www.linkedin.com/in/vaishnavi-metri-578b0835a",
     "https://github.com/vaishnavimetri234-v11s", False),
    ("Anoosha Kembhavi",        "Systems Engineer",    "Module 5 · Integration","#EC4899", "F",
     "anooshakembhavi@gmail.com",
     "http://www.linkedin.com/in/anoosha-kembhavi",
     "https://github.com/anooshakembhavi-afk", False),
    ("Snehal Anil Kamble",      "Data Engineer",       "Module 1 · Data Mgmt",  "#F97316", "F",
     "kamblesnehal578@gmail.com",
     "https://www.linkedin.com/in/snehal-k-b48369318",
     "https://github.com/kamblesnehal578-sketch", False),
    ("Nazhat Aliya Naikwadi",   "Data Engineer",       "Module 1 · Data Mgmt",  "#A855F7", "F",
     "nazhatnaikwadi@gmail.com",
     "https://linkedin.com/in/nazhatnaikwadi",
     "https://github.com/nazhatnaikwadi", False),
    ("Keerti Gadigeppagoudar",  "Analytics Engineer",  "Module 3 · Analytics",  "#14B8A6", "F",
     "keerti.s.g2020@gmail.com",
     "https://www.linkedin.com/in/keertig",
     "https://github.com/keertiG-1296", False),
    ("Samruddhi Patil",         "NLP Engineer",        "Module 2 · Chatbot",    "#6366F1", "F",
     "patilsamruddhi863@gmail.com",
     "https://www.linkedin.com/in/samruddhi-patil-a1575933a",
     "https://github.com/samruddhi128", False),
]

import base64

def _member_card(name, role, module, color, gender, email, li_url, gh_url, is_lead, compact=False):
    display_name = name.replace("\n", "<br>")
    initials = "".join(w[0].upper() for w in name.replace("\n", " ").split())[:2]
    
    card_bg = "#FFFFFF"
    card_border = f"1px solid #E5E7EB"
    name_color = "#111827"
    role_color = color
    email_color = "#6B7280"
    shadow = "0 1px 2px 0 rgba(0, 0, 0, 0.05)"
    init_text = "#FFFFFF"
    li_bg = "rgba(0,119,181,0.08)"; li_brd = "rgba(0,119,181,0.2)"; li_txt = "#0077B5"
    gh_bg = "rgba(0,0,0,0.04)"; gh_brd = "rgba(0,0,0,0.1)"; gh_txt = "#374151"
    
    fs_name = "0.75rem" if compact else "0.85rem"
    min_h = "185px" if compact else "205px"
    pad = "1rem 0.75rem 0.75rem" if compact else "1.25rem 0.75rem 1rem"
    circ_sz = "48px" if compact else "56px"
    circ_fs = "1rem" if compact else "1.125rem"
    
    lead_html = ""
    ring_style = f"border: 2px solid {color};"
    if is_lead:
        lead_html = f'<div style="position:absolute;top:-8px;left:50%;transform:translateX(-50%);background:#F59E0B;color:#fff;font-size:0.6rem;font-weight:600;padding:0.2rem 0.6rem;border-radius:20px;white-space:nowrap;">LEAD</div>'
        ring_style = "border: 3px solid #F59E0B; box-shadow: 0 0 0 2px rgba(245,158,11,0.2);"
    
    gh_svg_str = f'<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="{gh_txt}" d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>'
    gh_b64 = base64.b64encode(gh_svg_str.encode()).decode()
    gh_img = f'<img src="data:image/svg+xml;base64,{gh_b64}" width="12" height="12" style="display:block;" alt="GitHub"/>'
    
    return f"""
    <div style="background:{card_bg};border:{card_border};border-radius:16px;padding:{pad};text-align:center;min-height:{min_h};box-shadow:{shadow};transition:all 0.2s ease;position:relative;overflow:visible;">
        {lead_html}
        <div style="display:flex;justify-content:center;margin-bottom:0.75rem;margin-top:0.25rem;">
            <div style="width:{circ_sz};height:{circ_sz};border-radius:50%;background:linear-gradient(135deg,{color}, {color}CC);display:flex;align-items:center;justify-content:center;font-size:{circ_fs};font-weight:700;color:{init_text};{ring_style}">
                {initials}
            </div>
        </div>
        <div style="font-size:{fs_name};font-weight:600;color:{name_color};margin-bottom:0.2rem;line-height:1.3;">{display_name}</div>
        <div style="display:inline-block;font-size:0.6rem;font-weight:500;color:{role_color};background:{color}10;border:1px solid {color}30;border-radius:20px;padding:0.2rem 0.6rem;margin-bottom:0.5rem;">{role}</div>
        <div style="font-size:0.6rem;color:{email_color};margin-bottom:0.5rem;word-break:break-all;"><a href="mailto:{email}" style="color:{email_color};text-decoration:none;">✉ {email}</a></div>
        <div style="display:flex;justify-content:center;gap:0.5rem;">
            <a href="{li_url}" target="_blank" style="display:flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:6px;background:{li_bg};border:1px solid {li_brd};color:{li_txt};font-size:0.7rem;font-weight:700;text-decoration:none;">in</a>
            <a href="{gh_url}" target="_blank" style="display:flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:6px;background:{gh_bg};border:1px solid {gh_brd};text-decoration:none;">{gh_img}</a>
        </div>
    </div>"""

def page_team():
    _c("""
    <style>
    .stApp {
        background: #F9FAFB !important;
    }
    </style>""")
    
    # Hero header
    _c("""
    <div style="background: linear-gradient(135deg, #FFFFFF, #F9FAFB); border: 1px solid #E5E7EB; border-radius: 20px; padding: 2rem 1.5rem; margin-bottom: 1.5rem; text-align: center;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🧠</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #111827; margin-bottom: 0.5rem;">Meet Our Talented Team</div>
        <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem; margin: 0.5rem 0;">
            <div style="width: 30px; height: 2px; background: #D1D5DB;"></div>
            <div style="width: 6px; height: 6px; border-radius: 50%; background: #0F172A;"></div>
            <div style="width: 30px; height: 2px; background: #D1D5DB;"></div>
        </div>
        <div style="font-size: 0.8rem; color: #6B7280;">Recognizing our 11 interns who built this Python &amp; AI project</div>
        <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1rem; font-size: 1rem; color: #6B7280;">
            <span>🤖</span> <span>🔗</span> <span>💻</span> <span>🔍</span> <span>📊</span> <span>⚡</span>
        </div>
    </div>""")
    
    # Row 1
    row1 = TEAM[:5]
    cols1 = st.columns(5, gap="small")
    for i, member in enumerate(row1):
        with cols1[i]:
            _c(_member_card(*member, compact=False))
    
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    
    # Row 2
    row2 = TEAM[5:]
    cols2 = st.columns(6, gap="small")
    for i, member in enumerate(row2):
        with cols2[i]:
            _c(_member_card(*member, compact=True))
    
    # Stats bar
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    _c("""
    <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 16px; padding: 1rem 1.5rem; display: flex; justify-content: space-around; text-align: center;">
        <div><div style="font-size: 1.5rem; font-weight: 700; color: #0F172A;">11</div><div style="font-size: 0.7rem; color: #6B7280;">👥 Interns</div></div>
        <div><div style="font-size: 1.5rem; font-weight: 700; color: #0F172A;">5</div><div style="font-size: 0.7rem; color: #6B7280;">📦 Modules</div></div>
        <div><div style="font-size: 1.5rem; font-weight: 700; color: #0F172A;">1</div><div style="font-size: 0.7rem; color: #6B7280;">🤖 AI App</div></div>
        <div><div style="font-size: 1.5rem; font-weight: 700; color: #0F172A;">∞</div><div style="font-size: 0.7rem; color: #6B7280;">💡 Ideas</div></div>
    </div>""")
    
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    _c("""
    <div style="background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 12px; padding: 0.75rem; text-align: center; font-size: 0.7rem; color: #6B7280;">
        🚀 Powering our Python &amp; AI project through innovative interns!
    </div>""")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS (Stripe-inspired)
# ═══════════════════════════════════════════════════════════════════════════
def page_settings():
    section_title("Settings", "Account · Users · Data management")
    c1, c2, c3 = st.columns(3)

    with c1:
        _c('<div class="g-card"><div class="sec-hdr">Account</div>')
        uname = st.session_state.get("user","Admin")
        _c(f"""
        <div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:10px;padding:0.75rem;margin-bottom:0.75rem;">
            <div style="font-size:0.8rem;font-weight:500;color:#111827;">{uname}</div>
            <div style="font-size:0.65rem;color:#10B981;margin-top:0.2rem;">Session active</div>
        </div></div>""")
        if st.button("🚪 Logout", key="s_logout", use_container_width=True):
            st.session_state.logged_in = False; st.rerun()

    with c2:
        _c('<div class="g-card"><div class="sec-hdr">Users & Roles</div>')
        users = _load_users()
        roles = ["admin", "analyst", "viewer"]
        for uname, _ in list(users.items())[:4]:
            role_idx = 0 if uname in ["admin","yusuf"] else 1
            col = ["#10B981","#F59E0B","#6B7280"][role_idx]
            _c(f"""
            <div style="display:flex;align-items:center;gap:0.5rem;padding:0.4rem 0.6rem;border-radius:8px;background:#F9FAFB;border:1px solid #E5E7EB;margin-bottom:0.35rem;">
                <div style="width:24px;height:24px;border-radius:50%;background:linear-gradient(135deg,#0F172A,#3B82F6);display:flex;align-items:center;justify-content:center;font-size:0.6rem;font-weight:600;color:white;">{uname[:2].upper()}</div>
                <div style="flex:1;font-size:0.7rem;color:#374151;">{uname}</div>
                <span style="font-size:0.6rem;padding:0.15rem 0.5rem;border-radius:10px;font-weight:500;color:{col};background:{col}10;border:1px solid {col}20;">{roles[role_idx]}</span>
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
# FLOATING CHATBOT (unchanged)
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
        pass

    button_css = """
        position: fixed; bottom: 24px; right: 24px; z-index: 9999;
        width: 48px; height: 48px; border-radius: 50%;
        background: #0F172A;
        display: flex; align-items: center; justify-content: center;
        cursor: pointer; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    """
    st.markdown(f"""
    <div style="{button_css}" onclick="window.dispatchEvent(new Event('toggle_chat'))">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
            <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
        </svg>
    </div>""", unsafe_allow_html=True)

    with st.expander("🤖 AI Chatbot", expanded=st.session_state.get("chat_open", False)):
        _render_chat_ui()

def _render_sidebar_chatbot():
    with st.expander("🤖 AI Chatbot (Powered by Groq)", expanded=False):
        _render_chat_ui()

def _render_chat_ui():
    df = st.session_state.cleaned_df
    col_types = st.session_state.column_types or {}

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"]=="assistant" else "👤"):
            st.markdown(msg["content"])
            if msg.get("fig"):
                fig = msg["fig"]
                fig.update_layout(**PLOTLY)
                st.plotly_chart(_pgl(fig, 220), use_container_width=True)

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
                        fig_out.update_layout(**PLOTLY)
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

    render_floating_chatbot()

if __name__ == "__main__":
    main()
