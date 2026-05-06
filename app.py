"""
Portfolio Intelligence Platform
Enterprise Financial Analysis System — Professional UI v2
"""

import streamlit as st
import streamlit.components.v1 as stc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.database.database import init_db, SessionLocal
from src.database.models import Portfolio, Holding, CompanyInfo, RiskMetrics
from src.data_pipeline.collector import DataCollector
from src.risk_engine.calculator import RiskCalculator
from src.esg_engine.calculator import ESGCalculator

st.set_page_config(
    page_title="Portfolio Intelligence Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Keyframe animations ── */
@keyframes pageIn       { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
@keyframes fadeIn       { from { opacity:0; } to { opacity:1; } }
@keyframes cardIn       { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
@keyframes topbarScan   { 0% { left:-35%; } 100% { left:120%; } }
@keyframes onlinePulse  { 0%,100% { box-shadow:0 0 0 0 rgba(16,185,129,0.5); } 60% { box-shadow:0 0 0 5px rgba(16,185,129,0); } }
@keyframes barFill      { from { width:0%; } to { width:var(--w); } }
@keyframes sectionReveal{ from { opacity:0; transform:translateX(-8px); } to { opacity:1; transform:translateX(0); } }

/* ── Base ── */
html, body, [class*="css"] { font-family:'Inter',sans-serif !important; }
#MainMenu, footer { visibility:hidden; }
header[data-testid="stHeader"] { display:none !important; }
.block-container { padding:0.75rem 1.5rem 2rem 1.5rem !important; background:#eef2f7 !important; animation:pageIn 0.28s ease forwards; }

::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius:2px; }
::-webkit-scrollbar-thumb:hover { background:#2563eb; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background:#0f172a !important; border-right:1px solid rgba(255,255,255,0.05) !important; padding-top:0 !important; }
section[data-testid="stSidebar"] > div { padding-top:0 !important; }
section[data-testid="stSidebar"] * { color:#7c93b0 !important; }
section[data-testid="stSidebar"] .stRadio label {
    padding:0.48rem 0.9rem !important; border-radius:6px !important; cursor:pointer !important;
    transition:all 0.15s ease !important; font-size:0.79rem !important; font-weight:500 !important;
    width:100% !important; display:block !important;
    border-left:2px solid transparent !important; margin-bottom:1px !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background:rgba(255,255,255,0.06) !important;
    color:#e2e8f0 !important;
    border-left-color:#3b82f6 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] > label { display:none !important; }

/* ── Topbar ── */
.topbar {
    background:linear-gradient(90deg, #0f172a 0%, #1e3a8a 45%, #1d4ed8 80%, #2563eb 100%);
    border-radius:10px; padding:0.7rem 1.4rem;
    display:flex; justify-content:space-between; align-items:center;
    margin-bottom:1.2rem; position:relative; overflow:hidden;
    box-shadow:0 4px 24px rgba(15,23,42,0.35), 0 1px 0 rgba(255,255,255,0.06) inset;
}
.topbar::after {
    content:''; position:absolute; top:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent);
}
.topbar-scan {
    position:absolute; top:0; bottom:0; width:30%;
    background:linear-gradient(90deg, transparent, rgba(255,255,255,0.035), transparent);
    animation:topbarScan 5s ease-in-out infinite;
}
.topbar-brand { font-size:1rem; font-weight:900; color:#ffffff !important; letter-spacing:-0.03em; }
.topbar-sep   { color:rgba(255,255,255,0.25) !important; font-size:1.1rem; margin:0 0.6rem; }
.topbar-page  { font-size:0.88rem; font-weight:600; color:#93c5fd !important; letter-spacing:0.01em; }
.topbar-date  { font-size:0.72rem; color:rgba(255,255,255,0.4) !important; font-weight:500; letter-spacing:0.02em; }
.topbar-badge { background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.2); border-radius:20px; padding:0.28rem 0.85rem; font-size:0.72rem; font-weight:600; color:#ffffff !important; }
.topbar-user  { font-size:0.72rem; color:rgba(255,255,255,0.5) !important; text-align:right; line-height:1.4; }
.topbar-user span { display:block; font-weight:700; color:#ffffff !important; font-size:0.82rem; }
.topbar-right { display:flex; align-items:center; gap:1rem; }
.topbar-left  { display:flex; align-items:center; }

/* ── Section header ── */
.section-header { display:flex; align-items:center; gap:0.6rem; margin:1.4rem 0 0.7rem; animation:sectionReveal 0.25s ease forwards; }
.section-header-line { width:3px; height:17px; background:linear-gradient(180deg,#2563eb,#93c5fd); border-radius:2px; flex-shrink:0; }
.section-header-text { font-size:0.88rem; font-weight:700; color:#0f172a !important; letter-spacing:-0.01em; }

/* ── KPI cards ── */
.kpi-card {
    background:#ffffff; border:1px solid #e2e8f0; border-radius:10px;
    padding:1.15rem 1.25rem; position:relative; overflow:hidden;
    transition:border-color 0.2s, transform 0.22s, box-shadow 0.22s; height:100%;
    box-shadow:0 1px 4px rgba(0,0,0,0.05);
    animation:cardIn 0.3s ease forwards;
}
.kpi-card:hover { border-color:#2563eb; transform:translateY(-2px); box-shadow:0 6px 20px rgba(37,99,235,0.13); }
.kpi-card::after { content:''; position:absolute; top:0; left:0; right:0; height:3px; background:linear-gradient(90deg,#2563eb,#93c5fd); }
.kpi-label { font-size:0.62rem; font-weight:700; color:#94a3b8 !important; text-transform:uppercase; letter-spacing:0.09em; margin-bottom:0.35rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.kpi-value { font-size:1.45rem; font-weight:800; color:#0f172a !important; letter-spacing:-0.03em; line-height:1; margin-bottom:0.4rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.kpi-delta { font-size:0.66rem; font-weight:600; display:inline-flex; align-items:center; gap:0.3rem; padding:0.16rem 0.55rem; border-radius:20px; white-space:nowrap; max-width:100%; }
.kpi-delta-pos { background:rgba(16,185,129,0.08); color:#059669 !important; border:1px solid rgba(16,185,129,0.22); }
.kpi-delta-neg { background:rgba(239,68,68,0.08);  color:#dc2626 !important; border:1px solid rgba(239,68,68,0.22); }
.kpi-delta-neu { background:rgba(37,99,235,0.08);  color:#2563eb !important; border:1px solid rgba(37,99,235,0.22); }

/* ── Metric cards ── */
.metric-card { border-radius:10px; padding:1.15rem 1.25rem; border:1px solid #e2e8f0; border-top-width:3px; position:relative; transition:transform 0.2s, box-shadow 0.2s; background:#ffffff; box-shadow:0 1px 4px rgba(0,0,0,0.05); animation:cardIn 0.3s ease forwards; }
.metric-card:hover { transform:translateY(-2px); box-shadow:0 6px 20px rgba(0,0,0,0.09); }
.metric-card-label { font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.11em; margin-bottom:0.38rem; }
.metric-card-value { font-size:1.7rem; font-weight:800; letter-spacing:-0.03em; line-height:1; margin-bottom:0.28rem; }
.metric-card-sub   { font-size:0.67rem; }

/* ── Risk & signal badges ── */
.risk-badge { display:inline-flex; align-items:center; gap:0.4rem; padding:0.42rem 1rem; border-radius:20px; font-weight:700; font-size:0.76rem; letter-spacing:0.06em; }
.risk-badge::before { content:'●'; font-size:0.48rem; }
.risk-low    { background:rgba(16,185,129,0.08); color:#059669 !important; border:1px solid rgba(16,185,129,0.25); }
.risk-medium { background:rgba(245,158,11,0.08); color:#d97706 !important; border:1px solid rgba(245,158,11,0.25); }
.risk-high   { background:rgba(239,68,68,0.08);  color:#dc2626 !important; border:1px solid rgba(239,68,68,0.25); }

.sig-badge  { display:inline-block; padding:0.26rem 0.75rem; border-radius:20px; font-weight:700; font-size:0.7rem; letter-spacing:0.06em; }
.signal-buy  { background:rgba(16,185,129,0.08); color:#059669 !important; border:1px solid rgba(16,185,129,0.25); }
.signal-hold { background:rgba(37,99,235,0.08);  color:#2563eb !important; border:1px solid rgba(37,99,235,0.25); }
.signal-sell { background:rgba(239,68,68,0.08);  color:#dc2626 !important; border:1px solid rgba(239,68,68,0.25); }

/* ── Tables ── */
.styled-table { width:100%; border-collapse:collapse; font-size:0.81rem; }
.styled-table th { background:#f8fafc; color:#64748b !important; font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.09em; padding:0.7rem 1rem; border-bottom:2px solid #e2e8f0; text-align:left; }
.styled-table td { padding:0.65rem 1rem; color:#1e293b !important; border-bottom:1px solid #f1f5f9; vertical-align:middle; transition:background 0.15s; }
.styled-table tr:hover td { background:#eff6ff; }
.styled-table tr:last-child td { border-bottom:none; }
.table-wrap { background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; overflow:hidden; box-shadow:0 1px 4px rgba(0,0,0,0.05); }

/* ── ESG bars ── */
.esg-bar-wrap   { margin-bottom:1rem; }
.esg-bar-header { display:flex; justify-content:space-between; font-size:0.77rem; margin-bottom:0.32rem; }
.esg-bar-label  { color:#475569 !important; font-weight:500; }
.esg-bar-score  { color:#0f172a !important; font-weight:700; }
.esg-bar-track  { background:#e2e8f0; border-radius:4px; height:6px; overflow:hidden; }
.esg-bar-fill   { height:100%; border-radius:4px; animation:barFill 0.8s cubic-bezier(0.16,1,0.3,1) forwards; }

/* ── Chart frame ── */
.chart-frame { background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:1rem 1rem 0.25rem; margin-top:0.25rem; box-shadow:0 1px 4px rgba(0,0,0,0.05); transition:box-shadow 0.2s; }
.chart-frame:hover { box-shadow:0 4px 16px rgba(0,0,0,0.08); }

/* ── Misc ── */
.info-pill  { display:inline-flex; align-items:center; gap:0.4rem; background:#eff6ff; border:1px solid #bfdbfe; border-radius:20px; padding:0.28rem 0.8rem; font-size:0.7rem; color:#2563eb !important; font-weight:600; }
.pip-divider{ border:none; border-top:1px solid #e2e8f0; margin:1.2rem 0; }

/* ── Login ── */
.login-box { background:#ffffff; border:1px solid #e2e8f0; border-radius:14px; padding:2.8rem 2.5rem 2rem; position:relative; overflow:hidden; box-shadow:0 8px 40px rgba(15,23,42,0.12); }
.login-box::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; background:linear-gradient(90deg,#0f172a,#2563eb,#93c5fd); }
.login-demo { font-size:0.68rem; color:#64748b !important; text-align:center; margin-top:1.5rem; line-height:1.8; background:#f8fafc; border-radius:8px; padding:0.75rem; border:1px solid #e2e8f0; }

/* ── Sidebar branding ── */
.sidebar-brand   { padding:1.35rem 0.9rem 0.85rem; border-bottom:1px solid rgba(255,255,255,0.07); margin-bottom:0.4rem; }
.sidebar-logo    { font-size:1.3rem; font-weight:900; color:#ffffff !important; letter-spacing:-0.04em; }
.sidebar-logo span { color:#3b82f6 !important; }
.sidebar-tagline { font-size:0.58rem; font-weight:600; color:rgba(255,255,255,0.3) !important; text-transform:uppercase; letter-spacing:0.14em; margin-top:0.2rem; }
.sidebar-section { font-size:0.6rem; font-weight:700; color:rgba(255,255,255,0.28) !important; text-transform:uppercase; letter-spacing:0.16em; padding:0.85rem 0.9rem 0.3rem; }
.sidebar-user      { border-top:1px solid rgba(255,255,255,0.07); padding:0.85rem 0.9rem 0.5rem; margin-top:0.5rem; }
.sidebar-user-name { font-size:0.81rem; font-weight:600; color:#ffffff !important; }
.sidebar-user-role { font-size:0.68rem; color:#3b82f6 !important; margin-top:0.1rem; }
.online-dot { display:inline-block; width:7px; height:7px; background:#10b981; border-radius:50%; margin-right:0.35rem; animation:onlinePulse 2.2s ease infinite; }

/* ── Page hero ── */
.page-hero { background:linear-gradient(135deg,#1e3a8a 0%,#1d4ed8 100%); border-radius:10px; padding:1.15rem 1.6rem; margin-bottom:1.4rem; position:relative; overflow:hidden; }
.page-hero::before { content:''; position:absolute; left:0; top:0; bottom:0; width:3px; background:linear-gradient(180deg,#ffffff,#93c5fd); }
.page-hero-title { font-size:1rem; font-weight:700; color:#ffffff !important; margin-bottom:0.25rem; }
.page-hero-desc  { font-size:0.76rem; color:#bfdbfe !important; line-height:1.55; max-width:820px; }

/* ── Insight box ── */
.insight-box { background:#eff6ff; border:1px solid #bfdbfe; border-left:3px solid #2563eb; border-radius:0 8px 8px 0; padding:0.9rem 1.15rem; margin:0.6rem 0; }
.insight-box-title { font-size:0.65rem; font-weight:700; color:#1d4ed8 !important; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.35rem; }
.insight-box-text  { font-size:0.78rem; color:#475569 !important; line-height:1.6; }
.insight-box-text strong { color:#0f172a !important; }

/* ── Streamlit native overrides ── */
div[data-testid="metric-container"] { background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:0.75rem 1rem; box-shadow:0 1px 4px rgba(0,0,0,0.05); transition:box-shadow 0.2s, transform 0.2s; }
div[data-testid="metric-container"]:hover { box-shadow:0 4px 16px rgba(37,99,235,0.1); transform:translateY(-1px); }
div[data-testid="metric-container"] label { color:#94a3b8 !important; font-size:0.67rem !important; text-transform:uppercase; letter-spacing:0.09em; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#0f172a !important; font-size:1.32rem !important; font-weight:800 !important; }
div[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size:0.76rem !important; font-weight:600 !important; }
div[data-testid="metric-container"] [data-testid="stMetricDelta"] svg { display:none; }
.stSelectbox > div > div { background:#ffffff !important; border-color:#e2e8f0 !important; color:#0f172a !important; border-radius:8px !important; }
.stSelectbox > div > div > div { color:#0f172a !important; }
.stTextInput > div > div > input { background:#ffffff !important; border-color:#e2e8f0 !important; color:#0f172a !important; border-radius:8px !important; }
.stButton > button[kind="primary"] { background:linear-gradient(135deg,#0f172a,#2563eb) !important; border:none !important; font-weight:600 !important; border-radius:8px !important; color:#ffffff !important; transition:opacity 0.15s, transform 0.15s !important; }
.stButton > button[kind="primary"]:hover { opacity:0.9; transform:translateY(-1px) !important; }
.stButton > button:not([kind="primary"]) { background:#ffffff !important; border:1px solid #e2e8f0 !important; color:#475569 !important; border-radius:8px !important; transition:all 0.15s !important; }
.stButton > button:not([kind="primary"]):hover { border-color:#2563eb !important; color:#2563eb !important; background:#eff6ff !important; }
.stCaption { color:#94a3b8 !important; font-size:0.68rem !important; }
.stAlert { border-radius:8px !important; }
div[data-testid="stDataFrame"] { background:#ffffff !important; border-radius:10px; border:1px solid #e2e8f0; }
div[data-testid="stDataFrame"] * { color:#1e293b !important; }
div[data-testid="stDataFrame"] th { background:#f8fafc !important; color:#64748b !important; font-size:0.68rem !important; font-weight:700 !important; text-transform:uppercase; letter-spacing:0.07em; }
p, span, div, label, li, td, th, h1, h2, h3, h4 { -webkit-font-smoothing:antialiased; }
.block-container p, .block-container span:not([class*="stMarkdown"]) { color:#1e293b; }

/* ── Column gap tightening ── */
[data-testid="column"] { padding-left:0.4rem !important; padding-right:0.4rem !important; }
[data-testid="stHorizontalBlock"] { gap:0.6rem !important; }

/* ── Streamlit progress / spinner ── */
div[data-testid="stSpinner"] p { color:#475569 !important; font-size:0.82rem !important; }

/* ── Chat input bar ── */
div[data-testid="stChatInput"] { border-color:#e2e8f0 !important; border-radius:10px !important; background:#ffffff !important; }
div[data-testid="stChatInput"] textarea { color:#0f172a !important; font-size:0.88rem !important; }

/* ── Consistent card row heights ── */
.kpi-card, .metric-card { min-height:100px; }

/* ── Better divider ── */
.pip-divider { margin:1rem 0; }
</style>
"""

PLOTLY_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#475569', family='Inter, sans-serif', size=12),
    xaxis=dict(gridcolor='#e2e8f0', linecolor='#e2e8f0', tickcolor='#94a3b8'),
    yaxis=dict(gridcolor='#e2e8f0', linecolor='#e2e8f0', tickcolor='#94a3b8'),
    hoverlabel=dict(bgcolor='#1e3a8a', bordercolor='#2563eb',
                    font=dict(color='#ffffff', family='Inter')),
)

st.markdown(CSS, unsafe_allow_html=True)

from src.auth.password import verify_password

# Passwords stored as bcrypt hashes — never in plaintext
USERS = {
    "analyst@pip.com":  {"password_hash": "$2b$12$76xyyLP.8facMGEosbvsseKYl7YQ7mrf4b5Flf52cFf9jgRNvKpAe", "name": "Sarah Mitchell",   "role": "Portfolio Analyst",      "initials": "SM"},
    "manager@pip.com":  {"password_hash": "$2b$12$76xyyLP.8facMGEosbvsseKYl7YQ7mrf4b5Flf52cFf9jgRNvKpAe", "name": "David Chen",       "role": "Senior Risk Manager",    "initials": "DC"},
    "admin@pip.com":    {"password_hash": "$2b$12$eY81cc2/5Ay7hIa7pvgUZu15i1Rpod.f/hb2RfKlxUDHODQz2UtVO", "name": "Varaalakshime V.", "role": "Platform Administrator", "initials": "VV"},
}

for k, d in [("logged_in", False), ("user", None)]:
    if k not in st.session_state: st.session_state[k] = d

# ── HELPERS ───────────────────────────────────────────────────────────────────
def top_bar(page_name):
    from datetime import datetime
    u    = st.session_state.user
    now  = datetime.now().strftime("%d %b %Y  %H:%M")
    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-scan"></div>
        <div class="topbar-left">
            <span class="topbar-brand">PIP</span>
            <span class="topbar-sep">/</span>
            <span class="topbar-page">{page_name}</span>
        </div>
        <div class="topbar-right">
            <span class="topbar-date">{now}</span>
            <span class="topbar-badge">{u['role']}</span>
            <div class="topbar-user"><span>{u['name']}</span>{u['initials']}</div>
        </div>
    </div>""", unsafe_allow_html=True)

def page_hero(title, description, tags=None):
    tags_html = ""
    if tags:
        tags_html = '<div class="page-hero-tags">' + "".join(
            f'<span class="hero-tag">{t}</span>' for t in tags) + '</div>'
    st.markdown(f"""
    <div class="page-hero">
        <div class="page-hero-title">{title}</div>
        <div class="page-hero-desc">{description}</div>
        {tags_html}
    </div>""", unsafe_allow_html=True)

def insight_box(title, text):
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-box-title">{title}</div>
        <div class="insight-box-text">{text}</div>
    </div>""", unsafe_allow_html=True)

def section_header(text):
    st.markdown(f"""
    <div class="section-header">
        <div class="section-header-line"></div>
        <div class="section-header-text">{text}</div>
    </div>""", unsafe_allow_html=True)

def kpi_card(label, value, delta, delta_type="pos"):
    cls   = f"kpi-delta-{delta_type}"
    arrow = "↑" if delta_type == "pos" else ("↓" if delta_type == "neg" else "→")
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <span class="kpi-delta {cls}">{arrow} {delta}</span>
    </div>""", unsafe_allow_html=True)

def metric_card(label, value, sub, accent):
    st.markdown(f"""
    <div class="metric-card" style="background:#ffffff;border-color:#e2e8f0;border-top:3px solid {accent};">
        <div class="metric-card-label" style="color:#94a3b8;">{label}</div>
        <div class="metric-card-value" style="color:{accent};">{value}</div>
        <div class="metric-card-sub" style="color:#94a3b8;">{sub}</div>
    </div>""", unsafe_allow_html=True)

def esg_bar(label, score, color):
    st.markdown(f"""
    <div class="esg-bar-wrap">
        <div class="esg-bar-header">
            <span class="esg-bar-label">{label}</span>
            <span class="esg-bar-score">{score:.1f}<span style="font-size:0.63rem;color:#94a3b8;font-weight:400;"> / 100</span></span>
        </div>
        <div class="esg-bar-track">
            <div class="esg-bar-fill" style="--w:{score}%;background:{color};"></div>
        </div>
    </div>""", unsafe_allow_html=True)

def divider():
    st.markdown('<hr class="pip-divider">', unsafe_allow_html=True)

def chart_wrap(fig, height=380):
    fig.update_layout(height=height, margin=dict(t=15, b=10, l=10, r=10))
    st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── LOGIN ─────────────────────────────────────────────────────────────────────
def show_login():
    col = st.columns([1, 2, 1])[1]
    with col:
        st.markdown("""
        <div class="login-box">
            <div style='text-align:center;font-size:2rem;font-weight:900;color:#0f172a;
                letter-spacing:-0.04em;margin-bottom:0.3rem;'>
                Port<span style='color:#2563eb;'>.</span>
            </div>
            <div style='text-align:center;font-size:0.68rem;font-weight:600;color:#64748b;
                text-transform:uppercase;letter-spacing:0.15em;margin-bottom:2rem;'>
                Portfolio Intelligence Platform
            </div>
        </div>""", unsafe_allow_html=True)
        email    = st.text_input("Email address", placeholder="you@pip.com")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Sign In  →", type="primary", use_container_width=True):
            if email in USERS and verify_password(password, USERS[email]["password_hash"]):
                st.session_state.logged_in = True
                st.session_state.user = USERS[email]
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")
        st.markdown("""
        <div class="login-demo">
            <strong style="color:#475569;">Demo credentials</strong><br>
            analyst@pip.com &nbsp;/&nbsp; Demo@1234<br>
            manager@pip.com &nbsp;/&nbsp; Demo@1234
        </div>""", unsafe_allow_html=True)

@st.cache_resource
def init_app():
    init_db()
    return {'collector': DataCollector(), 'risk_calc': RiskCalculator(), 'esg_calc': ESGCalculator()}

def load_portfolio_data():
    db = SessionLocal()
    try:
        p = db.query(Portfolio).first()
        if not p: return None
        holdings = db.query(Holding).filter(Holding.portfolio_id == p.id).all()
        return {
            'name': p.name, 'total_value': p.total_value,
            'esg_score': p.esg_score_overall, 'esg_rating': p.esg_rating,
            'environmental_score': p.environmental_score, 'social_score': p.social_score,
            'governance_score': p.governance_score, 'carbon_intensity': p.carbon_intensity,
            'holdings': [{'ticker': h.ticker, 'quantity': h.quantity,
                          'purchase_price': h.purchase_price, 'current_price': h.current_price,
                          'value': h.quantity*(h.current_price or h.purchase_price)} for h in holdings]
        }
    finally: db.close()

def load_risk_metrics():
    db = SessionLocal()
    try:
        r = db.query(RiskMetrics).order_by(RiskMetrics.calculation_date.desc()).first()
        if not r: return None
        return {'var_95_daily': r.var_95_daily, 'var_95_monthly': r.var_95_monthly,
                'sharpe_ratio': r.sharpe_ratio, 'sortino_ratio': r.sortino_ratio,
                'max_drawdown': r.max_drawdown, 'volatility': r.volatility}
    finally: db.close()

def load_company_esg():
    db = SessionLocal()
    try: return db.query(CompanyInfo).all()
    finally: db.close()

def compute_signals(portfolio, companies):
    esg_map    = {c.ticker: c.esg_score or 0 for c in companies}
    sector_map = {c.ticker: c.sector or "—"  for c in companies}
    rows = []
    for h in portfolio['holdings']:
        tkr       = h['ticker']
        esg       = esg_map.get(tkr, 50)
        momentum  = ((h['current_price'] or h['purchase_price']) - h['purchase_price']) / h['purchase_price'] * 100
        composite = (esg/100*40) + (min(max((momentum+20)/40,0),1)*40) + 20
        if composite >= 64:   signal, cls = "BUY",  "signal-buy"
        elif composite >= 60: signal, cls = "HOLD", "signal-hold"
        else:                 signal, cls = "SELL", "signal-sell"
        rows.append({'Ticker': tkr, 'Sector': sector_map.get(tkr,'—'),
                     'ESG Score': round(esg,1), 'Momentum': round(momentum,2),
                     'Composite': round(composite,1), 'Signal': signal, '_cls': cls})
    return pd.DataFrame(rows).sort_values('Composite', ascending=False)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    show_login()
else:
    components = init_app()

    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div style="display:flex; align-items:center; gap:0.65rem;">
                <svg width="38" height="38" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
                    <rect x="2" y="2" width="44" height="44" rx="10" fill="#0f172a" stroke="#1e293b" stroke-width="0.8"/>
                    <rect x="2" y="2" width="44" height="3" fill="#3b82f6"/>
                    <defs>
                        <linearGradient id="logofade" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stop-color="#10b981"/>
                            <stop offset="100%" stop-color="#10b981" stop-opacity="0"/>
                        </linearGradient>
                    </defs>
                    <path d="M9 35 L17 22 L24 28 L32 14 L40 18 L40 38 L9 38 Z" fill="url(#logofade)" opacity="0.4"/>
                    <path d="M9 35 L17 22 L24 28 L32 14 L40 18" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
                    <circle cx="32" cy="14" r="2.5" fill="#ffffff"/>
                </svg>
                <div>
                    <div class="sidebar-logo">Vantage<span>Port</span>.</div>
                    <div class="sidebar-tagline">Portfolio Intelligence</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
        page = st.radio("nav", [
            "Dashboard", "Holdings", "Risk Analysis", "ESG Scores",
            "Trade Signals", "Performance Attribution", "Signal Backtest",
            "AI Assistant", "Price History", "BI Dashboard"
        ], label_visibility="collapsed")
        u = st.session_state.user
        st.markdown(f"""
        <div class="sidebar-user">
            <div class="sidebar-user-name">
                <span class="online-dot"></span>{u['name']}
            </div>
            <div class="sidebar-user-role">{u['role']}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Sign Out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    if page == "Dashboard":
        top_bar("Dashboard")
        portfolio = load_portfolio_data()
        risk      = load_risk_metrics()
        if not portfolio: st.warning("No portfolio data found."); st.stop()
        _cos    = load_company_esg()
        sec_map = {c.ticker: c.sector or '—' for c in _cos}

        _hdf = pd.DataFrame(portfolio['holdings'])
        _ret = (((_hdf['current_price'].fillna(_hdf['purchase_price']) - _hdf['purchase_price'])
                 / _hdf['purchase_price']) * (_hdf['value'] / _hdf['value'].sum())).sum() * 100

        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: kpi_card("Total AUM",     f"${portfolio['total_value']:,.0f}", f"{_ret:+.1f}% return", "pos" if _ret>=0 else "neg")
        with c2: kpi_card("ESG Rating",    f"{portfolio['esg_rating']}  {portfolio['esg_score']:.0f}", "Sustainability score", "neu")
        with c3: kpi_card("Sharpe Ratio",  f"{risk['sharpe_ratio']:.2f}" if risk else "—", "vs S&P 500 ~0.6", "pos")
        with c4: kpi_card("Daily VaR 95%", f"${risk['var_95_daily']*portfolio['total_value']:,.0f}" if risk else "—", "Max 1-day loss", "neg")
        with c5: kpi_card("Holdings",      f"{len(portfolio['holdings'])}", f"{len(set(sec_map.get(h['ticker'],'?') for h in portfolio['holdings']))} sectors", "neu")

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([3, 2])
        with col1:
            section_header("Asset Allocation — Donut Chart")
            hdf = pd.DataFrame(portfolio['holdings'])
            fig = px.pie(hdf, values='value', names='ticker', hole=0.42,
                         color_discrete_sequence=['#2563eb','#3b82f6','#60a5fa','#93c5fd',
                                                   '#7c3aed','#a78bfa','#1d4ed8','#1e40af',
                                                   '#10b981','#0d9488','#6366f1','#8b5cf6',
                                                   '#ec4899','#f43f5e','#475569'])
            fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=10,
                              marker=dict(line=dict(color='#ffffff', width=2)))
            fig.update_layout(**PLOTLY_THEME, height=360, showlegend=True,
                              legend=dict(orientation='v', x=1.02, y=0.5,
                                          font=dict(size=10, color='#475569')),
                              margin=dict(t=10, b=10, l=10, r=10))
            fig.add_annotation(text=f"<b>{len(portfolio['holdings'])}</b><br><span style='font-size:10px'>Holdings</span>",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=14, color='#0f172a'), align='center')
            st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            if risk:
                section_header("Risk Snapshot")
                rc1, rc2 = st.columns(2)
                with rc1: st.metric("Ann. Volatility", f"{risk['volatility']*100:.1f}%")
                with rc2: st.metric("Max Drawdown",    f"{risk['max_drawdown']*100:.1f}%")
                divider()
                section_header("Top 5 Positions by Weight")
                hdf_top = pd.DataFrame(portfolio['holdings'])
                hdf_top['weight'] = hdf_top['value'] / hdf_top['value'].sum() * 100
                hdf_top = hdf_top.nlargest(5, 'weight')
                for _, row in hdf_top.iterrows():
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;
                         padding:0.45rem 0;border-bottom:1px solid #f1f5f9;">
                        <span style="font-size:0.82rem;font-weight:700;color:#0f172a;">{row['ticker']}</span>
                        <span style="font-size:0.82rem;font-weight:600;color:#2563eb;">{row['weight']:.1f}%</span>
                    </div>""", unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: PORTFOLIO
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "Holdings":
        top_bar("Holdings")
        portfolio = load_portfolio_data()
        if not portfolio: st.warning("No data."); st.stop()

        hdf_pre = pd.DataFrame(portfolio['holdings'])
        total_mv  = hdf_pre['value'].sum()
        top_pos   = hdf_pre.nlargest(1,'value').iloc[0]['ticker']
        companies = load_company_esg()
        sec_map   = {c.ticker: c.sector or '—' for c in companies}
        sectors   = len(set(sec_map.get(h['ticker'],'—') for h in portfolio['holdings']))

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Total Positions",   len(portfolio['holdings']))
        with c2: st.metric("Total Market Value", f"${total_mv:,.0f}")
        with c3: st.metric("Largest Position",   top_pos)
        with c4: st.metric("Sectors Covered",    sectors)

        divider()
        col_t, col_c = st.columns([3, 2])
        with col_t:
            section_header("Position Snapshot — All Holdings")
            hdf   = pd.DataFrame(portfolio['holdings'])
            hdf['weight'] = hdf['value'] / hdf['value'].sum()
            hdf['sector'] = hdf['ticker'].map(sec_map)

            table_rows = ""
            for _, row in hdf.sort_values('value', ascending=False).iterrows():
                table_rows += f"""<tr>
                    <td><strong style="color:#0f172a;">{row['ticker']}</strong></td>
                    <td style="color:#64748b;">{row['sector']}</td>
                    <td>{row['quantity']:.2f}</td>
                    <td>${row['purchase_price']:.2f}</td>
                    <td>${(row['current_price'] or row['purchase_price']):.2f}</td>
                    <td><strong>${row['value']:,.0f}</strong></td>
                    <td><strong style="color:#2563eb;">{row['weight']*100:.1f}%</strong></td>
                </tr>"""
            st.markdown(f"""<div class="table-wrap"><table class="styled-table">
                <thead><tr><th>Ticker</th><th>Sector</th><th>Shares</th><th>Avg Cost</th>
                <th>Current Price</th><th>Market Value</th><th>Weight</th></tr></thead>
                <tbody>{table_rows}</tbody></table></div>""", unsafe_allow_html=True)

        with col_c:
            section_header("Sector Allocation")
            hdf2 = pd.DataFrame(portfolio['holdings'])
            hdf2['sector'] = hdf2['ticker'].map(sec_map)
            sec_df = hdf2.groupby('sector')['value'].sum().reset_index()
            sec_df['weight'] = sec_df['value'] / sec_df['value'].sum() * 100
            sec_df = sec_df.sort_values('weight', ascending=False)
            colors_sec = ['#2563eb','#3b82f6','#60a5fa','#1d4ed8','#7c3aed',
                          '#10b981','#0d9488','#f59e0b','#ef4444','#6366f1']
            fig = go.Figure(go.Pie(
                labels=sec_df['sector'], values=sec_df['weight'],
                marker=dict(colors=colors_sec[:len(sec_df)],
                            line=dict(color='#ffffff', width=2)),
                textfont=dict(size=11, color='#ffffff'),
                textinfo='label+percent', hole=0.38,
                hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>'
            ))
            fig.update_layout(**PLOTLY_THEME, height=380, showlegend=False,
                              margin=dict(t=10,b=10,l=10,r=10))
            st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: RISK & ANALYTICS
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "Risk Analysis":
        top_bar("Risk Analysis")
        portfolio = load_portfolio_data()
        risk      = load_risk_metrics()
        if not risk: st.warning("No risk data found."); st.stop()

        var_d = risk['var_95_daily']   * portfolio['total_value']
        var_m = risk['var_95_monthly'] * portfolio['total_value']

        c1,c2,c3,c4 = st.columns(4)
        for col,(lbl,val,sub,accent) in zip([c1,c2,c3,c4],[
            ("Daily VaR (95%)",  f"${var_d:,.0f}",              "Max 1-day loss @ 95% CI",  "#ef4444"),
            ("Monthly VaR",      f"${var_m:,.0f}",              "Max 1-month loss",          "#f59e0b"),
            ("Sharpe Ratio",     f"{risk['sharpe_ratio']:.2f}",  "Risk-adjusted return",      "#10b981"),
            ("Sortino Ratio",    f"{risk['sortino_ratio']:.2f}", "Downside deviation ratio",  "#2563eb"),
        ]):
            with col: metric_card(lbl, val, sub, accent)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            section_header("Return Distribution & VaR Threshold")
            std  = risk['volatility'] / np.sqrt(252)
            x    = np.linspace(-0.07, 0.07, 1000)
            y    = (1/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*(x/std)**2)
            vx   = -risk['var_95_daily']
            mask = x <= vx
            fig  = go.Figure()
            fig.add_trace(go.Scatter(x=x[mask]*100, y=y[mask], fill='tozeroy',
                                     fillcolor='rgba(239,68,68,0.15)',
                                     line=dict(color='rgba(0,0,0,0)'), showlegend=False))
            fig.add_trace(go.Scatter(x=x*100, y=y, line=dict(color='#2563eb', width=2),
                                     fill='tozeroy', fillcolor='rgba(37,99,235,0.07)',
                                     name='Daily Return Distribution'))
            fig.add_vline(x=vx*100, line_dash="dash", line_color="#ef4444", line_width=1.5,
                          annotation_text=f"  VaR: {vx*100:.2f}%",
                          annotation_font=dict(color="#ef4444", size=11))
            fig.update_layout(**PLOTLY_THEME, height=340,
                              xaxis_title="Daily Return (%)", yaxis_title="Probability Density",
                              margin=dict(t=15, b=10, l=10, r=10))
            chart_wrap(fig, height=340)

        with col2:
            section_header("Risk Metrics Overview")
            fig = go.Figure(go.Bar(
                x=['Sharpe', 'Sortino', 'Max Drawdown %', 'Volatility %'],
                y=[risk['sharpe_ratio'], risk['sortino_ratio'],
                   risk['max_drawdown']*100, risk['volatility']*100],
                marker=dict(color=['#2563eb','#0284c7','#ef4444','#f59e0b'],
                            line=dict(color='rgba(0,0,0,0)')),
                text=[f"{risk['sharpe_ratio']:.2f}", f"{risk['sortino_ratio']:.2f}",
                      f"{risk['max_drawdown']*100:.1f}%", f"{risk['volatility']*100:.1f}%"],
                textposition='outside', textfont=dict(color='#475569', size=11)
            ))
            fig.update_layout(**PLOTLY_THEME, height=340,
                              margin=dict(t=15, b=10, l=10, r=10))
            chart_wrap(fig, height=340)

        divider()
        col1, col2 = st.columns(2)
        with col1:
            section_header("Risk Metrics Detail")
            rows_html = "".join(f"<tr><td>{l}</td><td style='color:#0f172a;font-weight:600;text-align:right'>{v}</td></tr>"
                for l,v in [("Daily VaR (95%)", f"${var_d:,.2f}"),
                             ("Monthly VaR (95%)", f"${var_m:,.2f}"),
                             ("Annualized Volatility", f"{risk['volatility']*100:.2f}%"),
                             ("Maximum Drawdown", f"{risk['max_drawdown']*100:.2f}%")])
            st.markdown(f'<div class="table-wrap"><table class="styled-table"><tbody>{rows_html}</tbody></table></div>',
                        unsafe_allow_html=True)
        with col2:
            section_header("Performance Metrics")
            rp    = var_d / portfolio['total_value']
            badge = "risk-low" if rp < 0.02 else ("risk-medium" if rp < 0.05 else "risk-high")
            label = "LOW RISK"  if rp < 0.02 else ("MEDIUM RISK" if rp < 0.05 else "HIGH RISK")
            rows_html = "".join(f"<tr><td>{l}</td><td style='color:#0f172a;font-weight:600;text-align:right'>{v}</td></tr>"
                for l,v in [("Sharpe Ratio", f"{risk['sharpe_ratio']:.3f}"),
                             ("Sortino Ratio", f"{risk['sortino_ratio']:.3f}")])
            st.markdown(f'<div class="table-wrap"><table class="styled-table"><tbody>{rows_html}</tbody></table></div>',
                        unsafe_allow_html=True)
            st.markdown(f"<br><span class='risk-badge {badge}'>{label}</span>", unsafe_allow_html=True)

        divider()
        section_header("Stress Test — Historical Crash Scenarios")
        st.markdown("<br>", unsafe_allow_html=True)

        scenarios = [
            {"name": "2008 Global Financial Crisis", "period": "Sep 2008 – Mar 2009",
             "market_drop": -38.5, "color": "#ef4444",
             "description": "Lehman Brothers collapse, credit freeze, global recession. Worst drawdown since the Great Depression."},
            {"name": "COVID-19 Market Crash", "period": "Feb 20 – Mar 23, 2020",
             "market_drop": -33.9, "color": "#f59e0b",
             "description": "Fastest 30% decline in S&P 500 history. Pandemic-driven liquidity panic across all asset classes."},
            {"name": "2022 Rate Hike Cycle", "period": "Jan 2022 – Oct 2022",
             "market_drop": -19.4, "color": "#8b5cf6",
             "description": "Fed raised rates from 0.25% to 4.0%. Growth and tech stocks hit hardest. Bonds fell simultaneously."},
        ]
        sc1, sc2, sc3 = st.columns(3)
        for col, s in zip([sc1, sc2, sc3], scenarios):
            loss     = portfolio['total_value'] * (s['market_drop'] / 100)
            survival = portfolio['total_value'] + loss
            with col:
                st.markdown(f"""
                <div style="background:#ffffff;border:1px solid {s['color']}33;
                     border-top:3px solid {s['color']};border-radius:10px;padding:1.2rem 1.3rem;
                     box-shadow:0 1px 3px rgba(0,0,0,0.06);">
                    <div style="font-size:0.72rem;font-weight:700;color:{s['color']};
                         text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem;">{s['name']}</div>
                    <div style="font-size:0.68rem;color:#64748b;margin-bottom:0.65rem;">{s['period']}</div>
                    <div style="font-size:1.6rem;font-weight:800;color:{s['color']};letter-spacing:-0.03em;">
                        {s['market_drop']:.1f}%</div>
                    <div style="font-size:0.7rem;color:#94a3b8;margin:0.15rem 0 0.7rem;">S&P 500 peak-to-trough</div>
                    <div style="border-top:1px solid #e2e8f0;padding-top:0.7rem;">
                        <div style="font-size:0.68rem;color:#64748b;">Estimated portfolio loss</div>
                        <div style="font-size:1.1rem;font-weight:700;color:#ef4444;">${loss:,.0f}</div>
                        <div style="font-size:0.68rem;color:#64748b;margin-top:0.35rem;">Surviving portfolio value</div>
                        <div style="font-size:1.1rem;font-weight:700;color:#10b981;">${survival:,.0f}</div>
                    </div>
                    <div style="margin-top:0.7rem;font-size:0.71rem;color:#64748b;line-height:1.55;">
                        {s['description']}</div>
                </div>""", unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: ESG INTELLIGENCE
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "ESG Scores":
        top_bar("ESG Scores")
        portfolio = load_portfolio_data()
        if not portfolio: st.warning("No data."); st.stop()

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Environmental Score", f"{portfolio['environmental_score']:.1f} / 100")
        with c2: st.metric("Social Score",         f"{portfolio['social_score']:.1f} / 100")
        with c3: st.metric("Governance Score",     f"{portfolio['governance_score']:.1f} / 100")
        with c4: st.metric("Carbon Intensity",     f"{portfolio['carbon_intensity']:.1f}")

        divider()
        section_header("ESG Pillar Gauges — Environmental · Social · Governance")
        col1, col2, col3 = st.columns(3)
        for col, (lbl, score, color) in zip([col1,col2,col3],[
            ("Environmental", portfolio['environmental_score'], "#10b981"),
            ("Social",        portfolio['social_score'],        "#3b82f6"),
            ("Governance",    portfolio['governance_score'],    "#8b5cf6"),
        ]):
            with col:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=score,
                    title={'text': lbl, 'font': {'color': '#475569', 'size': 13, 'family': 'Inter'}},
                    number={'font': {'color': '#0f172a', 'family': 'Inter', 'size': 30}},
                    gauge={'axis': {'range': [0,100], 'tickcolor':'#e2e8f0',
                                    'tickfont': {'color':'#94a3b8','size':9}},
                           'bar': {'color': color, 'thickness': 0.22},
                           'bgcolor': 'rgba(0,0,0,0)', 'borderwidth': 0,
                           'steps': [{'range':[0,50],'color':'#f8fafc'},
                                     {'range':[50,75],'color':'#f1f5f9'},
                                     {'range':[75,100],'color':'#eff6ff'}],
                           'threshold': {'line':{'color':color,'width':2},'thickness':0.8,'value':score}}
                ))
                fig.update_layout(**PLOTLY_THEME, height=240, margin=dict(t=30,b=10,l=20,r=20))
                st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        divider()
        section_header("ESG Score Comparison Across Holdings")
        companies = load_company_esg()
        if companies:
            edf = pd.DataFrame([{
                'Ticker': c.ticker,
                'Environmental': c.environmental_score or 0,
                'Social': c.social_score or 0,
                'Governance': c.governance_score or 0,
                'Overall': c.esg_score or 0,
                'Rating': c.esg_rating or 'N/A'
            } for c in companies])

            fig = go.Figure()
            for cn, color in [('Environmental','#10b981'),('Social','#3b82f6'),('Governance','#8b5cf6')]:
                fig.add_trace(go.Bar(name=cn, x=edf['Ticker'], y=edf[cn],
                                     marker_color=color, marker_line=dict(color='rgba(0,0,0,0)')))
            fig.update_layout(**PLOTLY_THEME, barmode='group', height=380,
                              bargap=0.2, bargroupgap=0.05, yaxis_title="ESG Score (0–100)",
                              legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#e2e8f0',
                                          font=dict(size=11, color='#475569')),
                              margin=dict(t=15, b=10, l=10, r=10))
            chart_wrap(fig, height=380)

            divider()
            section_header("Company ESG Detail Table")
            st.dataframe(edf, use_container_width=True, hide_index=True)


    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: INVESTMENT SIGNALS
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "Trade Signals":
        top_bar("Trade Signals")
        portfolio = load_portfolio_data()
        if not portfolio: st.warning("No data."); st.stop()
        companies = load_company_esg()
        sig_df    = compute_signals(portfolio, companies)

        buy_c   = len(sig_df[sig_df['Signal']=='BUY'])
        hold_c  = len(sig_df[sig_df['Signal']=='HOLD'])
        sell_c  = len(sig_df[sig_df['Signal']=='SELL'])
        avg_esg = sig_df['ESG Score'].mean()

        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi_card("BUY Signals",   str(buy_c),       "Strong conviction", "pos")
        with c2: kpi_card("HOLD Signals",  str(hold_c),      "Maintain position", "neu")
        with c3: kpi_card("SELL Signals",  str(sell_c),      "Review required",   "neg")
        with c4: kpi_card("Avg ESG Score", f"{avg_esg:.1f}", "Portfolio quality", "neu")

        divider()
        section_header("Signal Summary by Holding")
        st.markdown("<br>", unsafe_allow_html=True)

        hdrs = ["TICKER","SECTOR","ESG SCORE","MOMENTUM","COMPOSITE SCORE","SIGNAL"]
        cols = st.columns([1.5,2,1.2,1.5,1.5,1.2])
        for c,h in zip(cols,hdrs):
            c.markdown(f'<span style="font-size:0.65rem;font-weight:700;color:#64748b;letter-spacing:0.1em;">{h}</span>',
                       unsafe_allow_html=True)
        st.markdown('<hr style="border-color:#e2e8f0;margin:0.3rem 0 0.5rem;">', unsafe_allow_html=True)
        for _, row in sig_df.iterrows():
            cols = st.columns([1.5,2,1.2,1.5,1.5,1.2])
            cols[0].markdown(f'<strong style="color:#0f172a;">{row["Ticker"]}</strong>', unsafe_allow_html=True)
            cols[1].markdown(f'<span style="color:#64748b;">{row["Sector"]}</span>', unsafe_allow_html=True)
            cols[2].markdown(f'<span style="color:#475569;">{row["ESG Score"]}</span>', unsafe_allow_html=True)
            mc = "#10b981" if row['Momentum']>0 else "#ef4444"
            ms = "+" if row['Momentum']>0 else ""
            cols[3].markdown(f'<span style="color:{mc};font-weight:600;">{ms}{row["Momentum"]:.2f}%</span>',
                             unsafe_allow_html=True)
            cols[4].markdown(f'<strong style="color:#475569;">{row["Composite"]}</strong>', unsafe_allow_html=True)
            cols[5].markdown(f'<span class="sig-badge {row["_cls"]}">{row["Signal"]}</span>',
                             unsafe_allow_html=True)

        divider()
        col1, col2 = st.columns(2)
        with col1:
            section_header("Composite Score Ranking")
            df_s = sig_df.sort_values('Composite', ascending=True)
            fig  = go.Figure(go.Bar(
                x=df_s['Composite'], y=df_s['Ticker'], orientation='h',
                marker=dict(color=df_s['Composite'],
                            colorscale=[[0,'#ef4444'],[0.5,'#f59e0b'],[1,'#10b981']],
                            line=dict(color='rgba(0,0,0,0)')),
                text=[f"{v}" for v in df_s['Composite']],
                textposition='outside', textfont=dict(color='#475569',size=10)
            ))
            fig.update_layout(**PLOTLY_THEME, height=430,
                              xaxis_title="Composite Score",
                              margin=dict(t=15,b=10,l=10,r=50))
            chart_wrap(fig, height=430)
        with col2:
            section_header("Signal Distribution Breakdown")
            sc  = sig_df['Signal'].value_counts().reset_index()
            sc.columns = ['Signal','Count']
            color_map = {'BUY': '#10b981', 'HOLD': '#2563eb', 'SELL': '#ef4444'}
            colors_pie = [color_map.get(s, '#94a3b8') for s in sc['Signal']]
            fig = go.Figure(go.Pie(
                labels=sc['Signal'], values=sc['Count'],
                marker=dict(colors=colors_pie,
                            line=dict(color='#ffffff', width=3)),
                textfont=dict(size=12, color='#ffffff'), hole=0.45,
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>'
            ))
            fig.update_layout(**PLOTLY_THEME, height=430, showlegend=False,
                              margin=dict(t=15,b=10,l=10,r=10))
            chart_wrap(fig, height=430)



    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: PERFORMANCE ATTRIBUTION
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "Performance Attribution":
        top_bar("Performance Attribution")
        portfolio  = load_portfolio_data()
        if not portfolio: st.warning("No data."); st.stop()
        companies  = load_company_esg()
        sector_map = {c.ticker: c.sector or "Unknown" for c in companies}

        hdf = pd.DataFrame(portfolio['holdings'])
        hdf['sector']       = hdf['ticker'].map(sector_map)
        hdf['weight']       = hdf['value'] / hdf['value'].sum()
        hdf['momentum']     = ((hdf['current_price'].fillna(hdf['purchase_price']) -
                                hdf['purchase_price']) / hdf['purchase_price'] * 100).round(2)
        hdf['pnl']          = (hdf['current_price'].fillna(hdf['purchase_price']) -
                                hdf['purchase_price']) * hdf['quantity']
        hdf['contribution'] = (hdf['momentum'] * hdf['weight']).round(3)

        total_return = hdf['contribution'].sum()
        best  = hdf.nlargest(1,'momentum').iloc[0]
        worst = hdf.nsmallest(1,'momentum').iloc[0]

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Weighted Portfolio Return", f"{total_return:.2f}%")
        with c2: st.metric("Best Performer",            best['ticker'],  delta=f"{best['momentum']:+.2f}%")
        with c3: st.metric("Worst Performer",           worst['ticker'], delta=f"{worst['momentum']:+.2f}%")
        with c4: st.metric("Total Unrealized P&L",      f"${hdf['pnl'].sum():,.0f}")

        divider()
        col1, col2 = st.columns(2)
        with col1:
            section_header("Individual Holding Returns")
            hs     = hdf.sort_values('momentum')
            colors = ['#10b981' if m >= 0 else '#ef4444' for m in hs['momentum']]
            fig    = go.Figure(go.Bar(
                x=hs['momentum'], y=hs['ticker'], orientation='h',
                marker=dict(color=colors, line=dict(color='rgba(0,0,0,0)')),
                text=[f"{m:+.2f}%" for m in hs['momentum']],
                textposition='outside', textfont=dict(color='#475569',size=10)
            ))
            fig.update_layout(**PLOTLY_THEME, height=450,
                              xaxis_title="Return vs Cost Basis (%)",
                              margin=dict(t=15,b=10,l=10,r=60))
            chart_wrap(fig, height=450)

        with col2:
            section_header("Sector-Level Contribution to Return")
            sec_df = hdf.groupby('sector').agg(
                Weight=('weight','sum'), Avg_Return=('momentum','mean'),
                Contribution=('contribution','sum')
            ).reset_index().round(3)
            fig = px.bar(sec_df, x='sector', y='Contribution', color='Contribution',
                         color_continuous_scale=[[0,'#ef4444'],[0.5,'#f59e0b'],[1,'#10b981']],
                         text='Contribution')
            fig.update_traces(marker_line_width=0, textfont_color='#475569')
            fig.update_layout(**PLOTLY_THEME, height=450,
                              xaxis_title="Sector", yaxis_title="Weighted Contribution (%)",
                              margin=dict(t=15,b=10,l=10,r=10))
            chart_wrap(fig, height=450)

        divider()
        col1, col2 = st.columns(2)
        with col1:
            section_header("Top 5 Performers")
            top5 = hdf.nlargest(5,'momentum')[['ticker','sector','momentum','pnl','weight']].copy()
            top5['pnl']    = top5['pnl'].apply(lambda x: f"${x:,.0f}")
            top5['weight'] = top5['weight'].apply(lambda x: f"{x*100:.1f}%")
            top5.columns   = ['Ticker','Sector','Return %','Unrealized P&L','Weight']
            st.dataframe(top5, use_container_width=True, hide_index=True)
        with col2:
            section_header("Bottom 5 Performers")
            bot5 = hdf.nsmallest(5,'momentum')[['ticker','sector','momentum','pnl','weight']].copy()
            bot5['pnl']    = bot5['pnl'].apply(lambda x: f"${x:,.0f}")
            bot5['weight'] = bot5['weight'].apply(lambda x: f"{x*100:.1f}%")
            bot5.columns   = ['Ticker','Sector','Return %','Unrealized P&L','Weight']
            st.dataframe(bot5, use_container_width=True, hide_index=True)

        divider()
        section_header("Full Sector Attribution Summary")
        sec_df['Weight'] = sec_df['Weight'].apply(lambda x: f"{x*100:.1f}%")
        sec_df.columns   = ['Sector','Portfolio Weight','Avg Return %','Weighted Contribution %']
        st.dataframe(sec_df, use_container_width=True, hide_index=True)


    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: BACKTEST
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "Signal Backtest":
        top_bar("Signal Backtest")
        portfolio = load_portfolio_data()
        if not portfolio: st.warning("No data."); st.stop()
        companies = load_company_esg()
        collector = components['collector']
        sig_df    = compute_signals(portfolio, companies)

        backtest_rows = []
        progress = st.progress(0, text="Loading historical price data...")

        for i, h in enumerate(portfolio['holdings']):
            tkr = h['ticker']
            try:
                data = collector.get_stock_data(tkr, '6mo')
                if data is None or len(data) < 30:
                    continue
                cp  = data['Close'].iloc[-1]
                p30 = data['Close'].iloc[-30] if len(data) >= 30 else data['Close'].iloc[0]
                p60 = data['Close'].iloc[-60] if len(data) >= 60 else data['Close'].iloc[0]
                p90 = data['Close'].iloc[-90] if len(data) >= 90 else data['Close'].iloc[0]
                r30 = (cp - p30) / p30 * 100
                r60 = (cp - p60) / p60 * 100
                r90 = (cp - p90) / p90 * 100
                sig_row = sig_df[sig_df['Ticker'] == tkr]
                signal  = sig_row['Signal'].values[0] if len(sig_row) > 0 else 'HOLD'
                backtest_rows.append({
                    'Ticker':     tkr,
                    'Signal':     signal,
                    'Fwd 30d %':  round(r30, 2),
                    'Fwd 60d %':  round(r60, 2),
                    'Fwd 90d %':  round(r90, 2),
                    'Correct 30d':(signal=='BUY' and r30>0) or (signal=='SELL' and r30<0),
                    'Correct 60d':(signal=='BUY' and r60>0) or (signal=='SELL' and r60<0),
                    'Correct 90d':(signal=='BUY' and r90>0) or (signal=='SELL' and r90<0),
                })
            except Exception:
                continue
            progress.progress((i+1)/len(portfolio['holdings']), text=f"Loading {tkr}...")

        progress.empty()
        if not backtest_rows:
            st.warning("Insufficient historical data for backtest.")
            st.stop()

        bt_df = pd.DataFrame(backtest_rows)
        directional = bt_df[bt_df['Signal'] != 'HOLD']
        acc_30 = directional['Correct 30d'].mean() * 100 if len(directional) else 0
        acc_60 = directional['Correct 60d'].mean() * 100 if len(directional) else 0
        acc_90 = directional['Correct 90d'].mean() * 100 if len(directional) else 0
        buy_90 = bt_df[bt_df['Signal']=='BUY']['Fwd 90d %'].mean()

        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi_card("30-Day Accuracy", f"{acc_30:.0f}%", "Directional signals", "pos" if acc_30>50 else "neg")
        with c2: kpi_card("60-Day Accuracy", f"{acc_60:.0f}%", "Directional signals", "pos" if acc_60>50 else "neg")
        with c3: kpi_card("90-Day Accuracy", f"{acc_90:.0f}%", "Directional signals", "pos" if acc_90>50 else "neg")
        with c4: kpi_card("Avg BUY Return",  f"{buy_90:+.1f}%" if not np.isnan(buy_90) else "—",
                          "90d forward return", "pos" if not np.isnan(buy_90) and buy_90>0 else "neg")

        divider()
        section_header("Forward Return by Holding & Signal")

        def fmt_ret(val):
            c = "#10b981" if val > 0 else "#ef4444"
            s = "+" if val > 0 else ""
            return f'<span style="color:{c};font-weight:600;">{s}{val:.2f}%</span>'

        def tick(correct, signal):
            if signal == 'HOLD': return '<span style="color:#94a3b8;">—</span>'
            return '<span style="color:#10b981;font-size:1rem;">✓</span>' if correct else '<span style="color:#ef4444;font-size:1rem;">✗</span>'

        table_rows = ""
        for _, row in bt_df.iterrows():
            sc = "signal-buy" if row['Signal']=='BUY' else ("signal-sell" if row['Signal']=='SELL' else "signal-hold")
            table_rows += f"""<tr>
                <td><strong style="color:#0f172a;">{row['Ticker']}</strong></td>
                <td><span class="sig-badge {sc}">{row['Signal']}</span></td>
                <td>{fmt_ret(row['Fwd 30d %'])}</td>
                <td>{tick(row['Correct 30d'], row['Signal'])}</td>
                <td>{fmt_ret(row['Fwd 60d %'])}</td>
                <td>{tick(row['Correct 60d'], row['Signal'])}</td>
                <td>{fmt_ret(row['Fwd 90d %'])}</td>
                <td>{tick(row['Correct 90d'], row['Signal'])}</td>
            </tr>"""
        st.markdown(f"""
        <div class="table-wrap"><table class="styled-table">
            <thead><tr>
                <th>Ticker</th><th>Signal</th>
                <th>Fwd 30d</th><th>✓?</th>
                <th>Fwd 60d</th><th>✓?</th>
                <th>Fwd 90d</th><th>✓?</th>
            </tr></thead>
            <tbody>{table_rows}</tbody>
        </table></div>""", unsafe_allow_html=True)

        divider()
        col1, col2 = st.columns(2)
        with col1:
            section_header("Signal Accuracy by Time Horizon")
            fig = go.Figure(go.Bar(
                x=['30 Days','60 Days','90 Days'],
                y=[acc_30, acc_60, acc_90],
                marker=dict(color=[acc_30, acc_60, acc_90],
                            colorscale=[[0,'#ef4444'],[0.5,'#f59e0b'],[1,'#10b981']],
                            cmin=0, cmax=100, line=dict(color='rgba(0,0,0,0)')),
                text=[f"{v:.0f}%" for v in [acc_30, acc_60, acc_90]],
                textposition='outside', textfont=dict(color='#475569', size=13),
                customdata=[f"{v:.0f}%" for v in [acc_30, acc_60, acc_90]],
                hovertemplate='<b>%{x}</b><br>Accuracy: %{customdata}<extra></extra>'
            ))
            fig.add_hline(y=50, line_dash="dash", line_color="#94a3b8",
                          annotation_text="  Random baseline (50%)",
                          annotation_font=dict(color="#94a3b8", size=11))
            fig.update_layout(**PLOTLY_THEME, height=340,
                              yaxis_range=[0, 105],
                              yaxis_title="Accuracy (%)",
                              margin=dict(t=15,b=10,l=10,r=10))
            chart_wrap(fig, height=340)

        with col2:
            section_header("Avg 90-Day Forward Return by Signal")
            abs_df = bt_df.groupby('Signal')['Fwd 90d %'].mean().reset_index()
            colors_s = ['#10b981' if s=='BUY' else ('#ef4444' if s=='SELL' else '#3b82f6')
                        for s in abs_df['Signal']]
            fig2 = go.Figure(go.Bar(
                x=abs_df['Signal'], y=abs_df['Fwd 90d %'],
                marker=dict(color=colors_s, line=dict(color='rgba(0,0,0,0)')),
                text=[f"{v:+.2f}%" for v in abs_df['Fwd 90d %']],
                textposition='outside', textfont=dict(color='#475569', size=12)
            ))
            fig2.add_hline(y=0, line_color="#e2e8f0", line_width=1)
            fig2.update_layout(**PLOTLY_THEME, height=340,
                               yaxis_title="Avg Forward Return (%)",
                               margin=dict(t=15,b=10,l=10,r=10))
            chart_wrap(fig2, height=340)


    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: AI ANALYST
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "AI Assistant":
        top_bar("AI Assistant")
        @st.cache_resource(show_spinner=False)
        def load_chatbot():
            from src.chatbot.langchain_assistant import LangChainRAGAssistant
            return LangChainRAGAssistant()

        try:
            with st.spinner("Initialising AI Analyst — loading models and knowledge base..."):
                assistant = load_chatbot()

            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            if 'pending_q' not in st.session_state:
                st.session_state.pending_q = None

            # ── Status + Clear Memory ─────────────────────────────────────
            col_pill, col_clear = st.columns([5, 1])
            with col_pill:
                turns = st.session_state.chat_history[-1].get('turns', 0) if st.session_state.chat_history else 0
                st.markdown(
                    f'<span class="info-pill">AI Analyst Online — LangChain RAG · {turns} turn{"s" if turns != 1 else ""} in memory</span>',
                    unsafe_allow_html=True)
            with col_clear:
                if st.button("Clear Memory", use_container_width=True):
                    assistant.clear_memory()
                    st.session_state.chat_history = []
                    st.session_state.pending_q = None
                    st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Quick shortcuts ───────────────────────────────────────────
            section_header("Quick Query Shortcuts")
            c1, c2, c3, c4 = st.columns(4)
            shortcuts = {
                c1: ("Portfolio risk",  "What's my portfolio risk?"),
                c2: ("ESG analysis",    "Explain my ESG score"),
                c3: ("Sharpe ratio",    "What's my Sharpe ratio?"),
                c4: ("Holdings",        "What stocks do I own?"),
            }
            for col, (label, query) in shortcuts.items():
                with col:
                    if st.button(label, use_container_width=True):
                        st.session_state.pending_q = query
                        st.rerun()

            divider()

            def render_bubble(question, answer, sources):
                import html as _html
                src = ", ".join(sources) if sources else "—"
                st.markdown(
                    f'<div style="display:flex;justify-content:flex-end;margin-bottom:0.5rem;">'
                    f'<div style="background:#2563eb;color:#ffffff;padding:0.65rem 1rem;'
                    f'border-radius:18px 18px 4px 18px;max-width:78%;font-size:0.92rem;line-height:1.5;">'
                    f'{_html.escape(question)}</div></div>'
                    f'<div style="display:flex;justify-content:flex-start;margin-bottom:1.1rem;">'
                    f'<div style="background:#f1f5f9;border:1px solid #e2e8f0;color:#1e293b;'
                    f'padding:0.75rem 1rem;border-radius:18px 18px 18px 4px;max-width:85%;'
                    f'font-size:0.92rem;line-height:1.65;">'
                    f'{_html.escape(answer)}'
                    f'<div style="font-size:0.65rem;color:#94a3b8;margin-top:0.5rem;">Sources: {src}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True)

            # ── Chat history ──────────────────────────────────────────────
            for chat in st.session_state.chat_history:
                render_bubble(chat["q"], chat["a"], chat.get("sources", []))

            # ── Process pending shortcut ──────────────────────────────────
            if st.session_state.pending_q:
                query = st.session_state.pending_q
                st.session_state.pending_q = None
                with st.spinner("Analysing your portfolio..."):
                    resp = assistant.query(query)
                render_bubble(query, resp["answer"], resp.get("sources", []))
                st.session_state.chat_history.append({
                    "q": query, "a": resp["answer"],
                    "turns": resp["memory_turns"], "sources": resp["sources"],
                })

            # ── Chat input (auto-clears, anchors to bottom) ───────────────
            if user_q := st.chat_input("Ask about your portfolio..."):
                with st.spinner("Analysing your portfolio..."):
                    resp = assistant.query(user_q)
                render_bubble(user_q, resp["answer"], resp.get("sources", []))
                st.session_state.chat_history.append({
                    "q": user_q, "a": resp["answer"],
                    "turns": resp["memory_turns"], "sources": resp["sources"],
                })

        except Exception as e:
            st.error(f"AI system unavailable: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: MARKET DATA
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "Price History":
        top_bar("Price History")
        collector = components['collector']
        c1, c2    = st.columns([3,1])
        with c1: ticker = st.selectbox("Select Instrument",
                ['AAPL','MSFT','GOOGL','NVDA','TSLA','AMZN','JPM','GS','V','JNJ',
                 'BA','CAT','UNH','WMT','XOM'])
        with c2: period = st.selectbox("Time Period", ['1mo','3mo','6mo','1y'], index=2)

        if ticker:
            data = collector.get_stock_data(ticker, period)
            info = collector.get_company_info(ticker)
            if data is not None:
                if info:
                    st.markdown(f"""
                    <div style="margin:0.5rem 0;">
                        <span style="font-size:1.1rem;font-weight:800;color:#0f172a;">{info['company_name']}</span>
                        <span style="font-size:0.82rem;color:#475569;margin-left:0.5rem;font-weight:600;">({ticker})</span>
                    </div>
                    <span class="info-pill">{info['sector']} · {info['industry']}</span>
                    """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                latest  = data['Close'].iloc[-1]
                prev    = data['Close'].iloc[-2]
                change  = ((latest - prev) / prev) * 100
                vol_avg = data['Volume'].mean()

                c1,c2,c3,c4 = st.columns(4)
                with c1: st.metric("Last Price", f"${latest:.2f}", delta=f"{change:+.2f}%")
                if info:
                    with c2: st.metric("Market Cap", f"${info['market_cap']/1e9:.1f}B")
                    with c3:
                        pe = info.get('pe_ratio')
                        st.metric("P/E Ratio", f"{pe:.2f}" if pe else "N/A")
                    with c4:
                        beta = info.get('beta')
                        st.metric("Beta", f"{beta:.2f}" if beta else "N/A")

                divider()
                section_header(f"{ticker} — OHLCV Candlestick Chart ({period})")
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=data.index, open=data['Open'], high=data['High'],
                    low=data['Low'], close=data['Close'], name=ticker,
                    increasing=dict(line=dict(color='#10b981',width=1), fillcolor='rgba(16,185,129,0.3)'),
                    decreasing=dict(line=dict(color='#ef4444',width=1), fillcolor='rgba(239,68,68,0.3)')
                ))
                fig.update_layout(**PLOTLY_THEME, height=420,
                                  xaxis_title="Date", yaxis_title="Price (USD)",
                                  xaxis_rangeslider_visible=False,
                                  margin=dict(t=15,b=10,l=10,r=10))
                chart_wrap(fig, height=420)

                section_header("Trading Volume — Daily Bar Chart")
                colors_v = ['#10b981' if data['Close'].iloc[i] >= data['Open'].iloc[i] else '#ef4444'
                            for i in range(len(data))]
                fig2 = go.Figure(go.Bar(
                    x=data.index, y=data['Volume'],
                    marker=dict(color=colors_v, opacity=0.7, line=dict(color='rgba(0,0,0,0)'))
                ))
                fig2.update_layout(**PLOTLY_THEME, height=200,
                                   xaxis_title="Date", yaxis_title="Volume",
                                   margin=dict(t=10,b=10,l=10,r=10))
                chart_wrap(fig2, height=200)


    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: BI DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "BI Dashboard":
        top_bar("BI Dashboard")
        st.markdown("<br>", unsafe_allow_html=True)

        stc.html("""
        <div style="width:100%;overflow:hidden;border-radius:10px;border:1px solid #e2e8f0;">
            <iframe
                src="https://public.tableau.com/views/Book4_17718673615800/Dashboard2?:embed=y&:display_count=yes&:showVizHome=no&:toolbar=yes"
                width="100%" height="900px" frameborder="0" scrolling="yes"
                style="display:block;">
            </iframe>
        </div>""", height=920, scrolling=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <a href="https://public.tableau.com/app/profile/varaalakshime.vigneswara.pandiarajan/viz/Book4_17718673615800/Dashboard2"
           target="_blank" style="text-decoration:none;">
           <span class="info-pill">↗ Open full dashboard in Tableau Public</span>
        </a>""", unsafe_allow_html=True)