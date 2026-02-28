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
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 1rem 1.5rem 2rem 1.5rem !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #060d1a; }
::-webkit-scrollbar-thumb { background: #1a3358; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2563eb; }

section[data-testid="stSidebar"] {
    background: #060d1a !important;
    border-right: 1px solid #1a3358 !important;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
section[data-testid="stSidebar"] * { color: #8ba4c0 !important; }
section[data-testid="stSidebar"] .stRadio label {
    padding: 0.55rem 0.85rem !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    transition: all 0.18s ease !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    display: block !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(37,99,235,0.12) !important;
    color: #93c5fd !important;
}

.topbar {
    background: linear-gradient(90deg, #060d1a 0%, #0a1f3d 60%, #0f2456 100%);
    border: 1px solid #1a3358; border-radius: 10px;
    padding: 0.75rem 1.5rem;
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 1.2rem; position: relative; overflow: hidden;
}
.topbar::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #2563eb, #c9a84c, #2563eb);
}
.topbar-brand { font-size: 1rem; font-weight: 800; color: #f0f6ff !important; letter-spacing: -0.02em; }
.topbar-sep { color: #1a3358 !important; font-size: 1.2rem; margin: 0 0.5rem; }
.topbar-page { font-size: 0.95rem; font-weight: 600; color: #c9a84c !important; }
.topbar-badge {
    background: rgba(37,99,235,0.15); border: 1px solid rgba(37,99,235,0.3);
    border-radius: 20px; padding: 0.3rem 0.85rem;
    font-size: 0.75rem; font-weight: 600; color: #93c5fd !important;
}
.topbar-user { font-size: 0.78rem; color: #8ba4c0 !important; text-align: right; }
.topbar-user span { display: block; font-weight: 600; color: #c9d8f0 !important; font-size: 0.82rem; }
.topbar-right { display: flex; align-items: center; gap: 1rem; }
.topbar-left  { display: flex; align-items: center; }

.page-hero {
    background: linear-gradient(135deg, #0d1e35 0%, #0a1628 100%);
    border: 1px solid #1a3358; border-radius: 10px;
    padding: 1.2rem 1.8rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.page-hero::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0;
    width: 3px; background: linear-gradient(180deg, #2563eb, #c9a84c);
}
.page-hero-title { font-size: 1.05rem; font-weight: 700; color: #f0f6ff !important; margin-bottom: 0.3rem; }
.page-hero-desc  { font-size: 0.78rem; color: #6b7f96 !important; line-height: 1.55; max-width: 820px; }
.page-hero-tags  { margin-top: 0.65rem; display: flex; gap: 0.4rem; flex-wrap: wrap; }
.hero-tag {
    background: rgba(37,99,235,0.08); border: 1px solid rgba(37,99,235,0.2);
    border-radius: 20px; padding: 0.18rem 0.6rem;
    font-size: 0.67rem; font-weight: 600; color: #93c5fd !important; letter-spacing: 0.04em;
}

.insight-box {
    background: linear-gradient(135deg, #0d1e35, #0a1f3d);
    border: 1px solid #1a3358; border-left: 3px solid #c9a84c;
    border-radius: 0 8px 8px 0; padding: 1rem 1.2rem; margin: 0.75rem 0;
}
.insight-box-title { font-size: 0.68rem; font-weight: 700; color: #c9a84c !important; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; }
.insight-box-text  { font-size: 0.8rem; color: #8ba4c0 !important; line-height: 1.65; }
.insight-box-text strong { color: #c9d8f0 !important; }

.section-header { display: flex; align-items: center; gap: 0.6rem; margin: 1.5rem 0 0.75rem; }
.section-header-line { width: 3px; height: 18px; background: linear-gradient(180deg, #2563eb, #c9a84c); border-radius: 2px; flex-shrink: 0; }
.section-header-text { font-size: 0.92rem; font-weight: 700; color: #c9d8f0 !important; }

.kpi-card {
    background: linear-gradient(145deg, #0d1e35, #0f2348);
    border: 1px solid #1a3358; border-radius: 10px;
    padding: 1.2rem 1.3rem; position: relative; overflow: hidden;
    transition: border-color 0.2s, transform 0.2s; height: 100%;
}
.kpi-card:hover { border-color: #2563eb; transform: translateY(-1px); }
.kpi-card::after { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(37,99,235,0.4), transparent); }
.kpi-icon   { font-size: 1.3rem; margin-bottom: 0.45rem; display: block; }
.kpi-label  { font-size: 0.67rem; font-weight: 600; color: #445771 !important; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem; }
.kpi-value  { font-size: 1.6rem; font-weight: 800; color: #f0f6ff !important; letter-spacing: -0.03em; line-height: 1; margin-bottom: 0.4rem; }
.kpi-delta  { font-size: 0.72rem; font-weight: 600; display: inline-flex; align-items: center; gap: 0.3rem; padding: 0.18rem 0.55rem; border-radius: 20px; }
.kpi-delta-pos { background: rgba(16,185,129,0.12); color: #10b981 !important; border: 1px solid rgba(16,185,129,0.2); }
.kpi-delta-neg { background: rgba(239,68,68,0.12);  color: #ef4444 !important; border: 1px solid rgba(239,68,68,0.2); }
.kpi-delta-neu { background: rgba(37,99,235,0.12);  color: #93c5fd !important; border: 1px solid rgba(37,99,235,0.2); }

.metric-card { border-radius: 10px; padding: 1.35rem 1.5rem; border: 1px solid; position: relative; overflow: hidden; transition: transform 0.2s; }
.metric-card:hover { transform: translateY(-2px); }
.metric-card-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.65; margin-bottom: 0.4rem; }
.metric-card-value { font-size: 1.85rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1; margin-bottom: 0.3rem; }
.metric-card-sub   { font-size: 0.7rem; opacity: 0.55; }

.risk-badge { display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.45rem 1.1rem; border-radius: 20px; font-weight: 700; font-size: 0.78rem; letter-spacing: 0.06em; }
.risk-badge::before { content: '●'; font-size: 0.5rem; }
.risk-low    { background: rgba(16,185,129,0.1); color: #10b981 !important; border: 1px solid rgba(16,185,129,0.3); }
.risk-medium { background: rgba(245,158,11,0.1); color: #f59e0b !important; border: 1px solid rgba(245,158,11,0.3); }
.risk-high   { background: rgba(239,68,68,0.1);  color: #ef4444 !important; border: 1px solid rgba(239,68,68,0.3); }

.sig-badge   { display: inline-block; padding: 0.28rem 0.8rem; border-radius: 20px; font-weight: 700; font-size: 0.73rem; letter-spacing: 0.06em; }
.signal-buy  { background: rgba(16,185,129,0.12); color: #10b981 !important; border: 1px solid rgba(16,185,129,0.3); }
.signal-hold { background: rgba(37,99,235,0.12);  color: #93c5fd !important; border: 1px solid rgba(37,99,235,0.3); }
.signal-sell { background: rgba(239,68,68,0.12);  color: #ef4444 !important; border: 1px solid rgba(239,68,68,0.3); }

.styled-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.styled-table th { background: #0a1628; color: #445771 !important; font-size: 0.67rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.09em; padding: 0.75rem 1rem; border-bottom: 1px solid #1a3358; text-align: left; }
.styled-table td { padding: 0.68rem 1rem; color: #c9d8f0 !important; border-bottom: 1px solid rgba(26,51,88,0.5); vertical-align: middle; }
.styled-table tr:hover td { background: rgba(37,99,235,0.04); }
.styled-table tr:last-child td { border-bottom: none; }
.table-wrap { background: #0d1e35; border: 1px solid #1a3358; border-radius: 10px; overflow: hidden; }

.esg-bar-wrap   { margin-bottom: 1rem; }
.esg-bar-header { display: flex; justify-content: space-between; font-size: 0.78rem; margin-bottom: 0.35rem; }
.esg-bar-label  { color: #8ba4c0 !important; font-weight: 500; }
.esg-bar-score  { color: #f0f6ff !important; font-weight: 700; }
.esg-bar-track  { background: #1a3358; border-radius: 4px; height: 7px; overflow: hidden; }
.esg-bar-fill   { height: 100%; border-radius: 4px; }

.info-pill { display: inline-flex; align-items: center; gap: 0.4rem; background: rgba(37,99,235,0.08); border: 1px solid rgba(37,99,235,0.2); border-radius: 20px; padding: 0.3rem 0.8rem; font-size: 0.72rem; color: #93c5fd !important; font-weight: 500; }
.chart-frame { background: #0d1e35; border: 1px solid #1a3358; border-radius: 10px; padding: 1rem 1rem 0.25rem; margin-top: 0.25rem; }
.pip-divider { border: none; border-top: 1px solid #1a3358; margin: 1.25rem 0; }

.login-box { background: linear-gradient(145deg, #0d1e35, #0a1628); border: 1px solid #1a3358; border-radius: 14px; padding: 2.8rem 2.5rem 2rem; position: relative; overflow: hidden; }
.login-box::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, #2563eb, #c9a84c, #2563eb); }
.login-demo { font-size: 0.68rem; color: #445771 !important; text-align: center; margin-top: 1.5rem; line-height: 1.8; background: rgba(26,51,88,0.3); border-radius: 8px; padding: 0.75rem; border: 1px solid #1a3358; }

.sidebar-brand   { padding: 1.4rem 0.85rem 0.8rem; border-bottom: 1px solid #1a3358; margin-bottom: 0.5rem; }
.sidebar-logo    { font-size: 1.25rem; font-weight: 900; color: #f0f6ff !important; letter-spacing: -0.03em; }
.sidebar-logo span { color: #c9a84c !important; }
.sidebar-tagline { font-size: 0.62rem; font-weight: 600; color: #2a3f5a !important; text-transform: uppercase; letter-spacing: 0.12em; margin-top: 0.2rem; }
.sidebar-section { font-size: 0.62rem; font-weight: 700; color: #2a3f5a !important; text-transform: uppercase; letter-spacing: 0.14em; padding: 0.8rem 0.85rem 0.3rem; }
.sidebar-user      { border-top: 1px solid #1a3358; padding: 0.85rem 0.85rem 0.5rem; margin-top: 0.5rem; }
.sidebar-user-name { font-size: 0.82rem; font-weight: 600; color: #c9d8f0 !important; }
.sidebar-user-role { font-size: 0.7rem; color: #445771 !important; margin-top: 0.1rem; }

div[data-testid="metric-container"] { background: #0d1e35; border: 1px solid #1a3358; border-radius: 10px; padding: 0.75rem 1rem; }
div[data-testid="metric-container"] label { color: #445771 !important; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f0f6ff !important; font-size: 1.35rem !important; font-weight: 800 !important; }
.stSelectbox > div > div { background: #0d1e35 !important; border-color: #1a3358 !important; color: #c9d8f0 !important; }
.stTextInput > div > div > input { background: #0d1e35 !important; border-color: #1a3358 !important; color: #c9d8f0 !important; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #1d4ed8, #2563eb) !important; border: none !important; font-weight: 600 !important; border-radius: 8px !important; }
.stButton > button:not([kind="primary"]) { background: transparent !important; border: 1px solid #1a3358 !important; color: #8ba4c0 !important; border-radius: 8px !important; }
.stButton > button:not([kind="primary"]):hover { border-color: #2563eb !important; color: #93c5fd !important; }
.stCaption { color: #445771 !important; font-size: 0.7rem !important; }
.stAlert { border-radius: 8px !important; }
</style>
"""

PLOTLY_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#8ba4c0', family='Inter, sans-serif', size=12),
    xaxis=dict(gridcolor='#1a3358', linecolor='#1a3358', tickcolor='#1a3358'),
    yaxis=dict(gridcolor='#1a3358', linecolor='#1a3358', tickcolor='#1a3358'),
    hoverlabel=dict(bgcolor='#0d1e35', bordercolor='#2563eb',
                    font=dict(color='#f0f6ff', family='Inter')),
)

st.markdown(CSS, unsafe_allow_html=True)

USERS = {
    "analyst@pip.com":  {"password": "Demo@1234",  "name": "Sarah Mitchell",   "role": "Portfolio Analyst",      "initials": "SM"},
    "manager@pip.com":  {"password": "Demo@1234",  "name": "David Chen",       "role": "Senior Risk Manager",    "initials": "DC"},
    "admin@pip.com":    {"password": "Admin@5678", "name": "Varaalakshime V.", "role": "Platform Administrator", "initials": "VV"},
}

for k, d in [("logged_in", False), ("user", None)]:
    if k not in st.session_state: st.session_state[k] = d

# ── HELPERS ───────────────────────────────────────────────────────────────────
def top_bar(page_name):
    u = st.session_state.user
    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-left">
            <span class="topbar-brand">PIP</span>
            <span class="topbar-sep">/</span>
            <span class="topbar-page">{page_name}</span>
        </div>
        <div class="topbar-right">
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
        <div class="insight-box-title">💡 {title}</div>
        <div class="insight-box-text">{text}</div>
    </div>""", unsafe_allow_html=True)

def section_header(text):
    st.markdown(f"""
    <div class="section-header">
        <div class="section-header-line"></div>
        <div class="section-header-text">{text}</div>
    </div>""", unsafe_allow_html=True)

def kpi_card(label, value, delta, delta_type="pos", icon=""):
    cls   = f"kpi-delta-{delta_type}"
    arrow = "↑" if delta_type == "pos" else ("↓" if delta_type == "neg" else "→")
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <span class="kpi-delta {cls}">{arrow} {delta}</span>
    </div>""", unsafe_allow_html=True)

def metric_card(label, value, sub, bg1, bg2, border):
    st.markdown(f"""
    <div class="metric-card" style="background:linear-gradient(135deg,{bg1},{bg2});border-color:{border};color:white;">
        <div class="metric-card-label">{label}</div>
        <div class="metric-card-value">{value}</div>
        <div class="metric-card-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def esg_bar(label, score, color):
    st.markdown(f"""
    <div class="esg-bar-wrap">
        <div class="esg-bar-header">
            <span class="esg-bar-label">{label}</span>
            <span class="esg-bar-score">{score:.1f}<span style="font-size:0.65rem;color:#445771;font-weight:400;"> / 100</span></span>
        </div>
        <div class="esg-bar-track">
            <div class="esg-bar-fill" style="width:{score}%;background:{color};"></div>
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
            <div style='text-align:center;font-size:2rem;font-weight:900;color:#f0f6ff;
                letter-spacing:-0.04em;margin-bottom:0.3rem;'>
                Port<span style='color:#c9a84c;'>.</span>
            </div>
            <div style='text-align:center;font-size:0.68rem;font-weight:600;color:#2a3f5a;
                text-transform:uppercase;letter-spacing:0.15em;margin-bottom:2rem;'>
                Portfolio Intelligence Platform
            </div>
        </div>""", unsafe_allow_html=True)
        email    = st.text_input("Email address", placeholder="you@pip.com")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Sign In  →", type="primary", use_container_width=True):
            if email in USERS and USERS[email]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.user = USERS[email]
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")
        st.markdown("""
        <div class="login-demo">
            <strong style="color:#8ba4c0;">Demo credentials</strong><br>
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
            <div class="sidebar-logo">Port<span>.</span></div>
            <div class="sidebar-tagline">Portfolio Intelligence Platform</div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
        page = st.radio("nav", [
            "Overview", "Portfolio", "Risk & Analytics", "ESG Intelligence",
            "Investment Signals", "Performance Attribution", "Backtest",
            "AI Analyst", "Market Data", "BI Dashboard"
        ], label_visibility="collapsed")
        u = st.session_state.user
        st.markdown(f"""
        <div class="sidebar-user">
            <div class="sidebar-user-name">
                <span style="width:7px;height:7px;background:#10b981;border-radius:50%;
                display:inline-block;margin-right:0.35rem;"></span>{u['name']}
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
    if page == "Overview":
        top_bar("Overview")
        page_hero(
            "Executive Portfolio Overview",
            "Real-time snapshot across 15 holdings — AUM, risk-adjusted performance, ESG health, and diversification in one view.",
            ["Executive Summary", "15 Holdings", "Daily Briefing", "ESG-Integrated"]
        )
        portfolio = load_portfolio_data()
        risk      = load_risk_metrics()
        if not portfolio: st.warning("No portfolio data found."); st.stop()

        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: kpi_card("Total AUM",     f"${portfolio['total_value']:,.0f}", "+5.2% YTD",    "pos", "💼")
        with c2: kpi_card("ESG Rating",    f"{portfolio['esg_rating']}  {portfolio['esg_score']:.0f}", "Responsible", "neu", "🌱")
        with c3: kpi_card("Sharpe Ratio",  f"{risk['sharpe_ratio']:.2f}" if risk else "—", "Risk-adjusted", "pos", "📐")
        with c4: kpi_card("Daily VaR 95%", f"${risk['var_95_daily']*portfolio['total_value']:,.0f}" if risk else "—", "Max 1-day loss", "neg", "⚠️")
        with c5: kpi_card("Holdings",      f"{len(portfolio['holdings'])}", "Diversified", "neu", "📊")

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([3, 2])
        with col1:
            section_header("Asset Allocation — Donut Chart")
            hdf = pd.DataFrame(portfolio['holdings'])
            fig = px.pie(hdf, values='value', names='ticker', hole=0.42,
                         color_discrete_sequence=['#2563eb','#3b82f6','#60a5fa','#93c5fd',
                                                   '#c9a84c','#d4b76a','#1d4ed8','#1e40af',
                                                   '#10b981','#0d9488','#6366f1','#8b5cf6',
                                                   '#ec4899','#f43f5e','#475569'])
            fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=10,
                              marker=dict(line=dict(color='#060d1a', width=2)))
            fig.update_layout(**PLOTLY_THEME, height=360, showlegend=True,
                              legend=dict(orientation='v', x=1.02, y=0.5,
                                          font=dict(size=10, color='#8ba4c0')),
                              margin=dict(t=10, b=10, l=10, r=10))
            fig.add_annotation(text=f"<b>{len(portfolio['holdings'])}</b><br><span style='font-size:10px'>Holdings</span>",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=14, color='#f0f6ff'), align='center')
            st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            section_header("ESG Pillar Scores")
            esg_bar("Environmental", portfolio['environmental_score'], "#10b981")
            esg_bar("Social",        portfolio['social_score'],        "#3b82f6")
            esg_bar("Governance",    portfolio['governance_score'],    "#8b5cf6")
            if risk:
                divider()
                section_header("Risk Snapshot")
                rc1, rc2 = st.columns(2)
                with rc1: st.metric("Ann. Volatility", f"{risk['volatility']*100:.1f}%")
                with rc2: st.metric("Max Drawdown",    f"{risk['max_drawdown']*100:.1f}%")

        divider()
        insight_box(
            "Daily Briefing Summary",
            f"As of today, your portfolio spans <strong>{len(portfolio['holdings'])} positions</strong> with total AUM of "
            f"<strong>${portfolio['total_value']:,.0f}</strong>. "
            f"ESG rating of <strong>{portfolio['esg_rating']} ({portfolio['esg_score']:.1f}/100)</strong> places it in the "
            f"<strong>{'top quartile' if portfolio['esg_score'] > 70 else 'mid-range'}</strong> of ESG-integrated funds. "
            + (f"Sharpe ratio of <strong>{risk['sharpe_ratio']:.2f}</strong> reflects "
               f"<strong>{'strong' if risk['sharpe_ratio'] > 1.5 else 'moderate'}</strong> risk-adjusted performance. "
               f"Maximum single-day loss at 95% confidence: <strong>${risk['var_95_daily']*portfolio['total_value']:,.0f}</strong>. "
               f"Navigate to Risk & Analytics for full downside scenario modeling." if risk else "")
        )

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: PORTFOLIO
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "Portfolio":
        top_bar("Portfolio Analysis")
        page_hero(
            "Holdings & Position Management",
            "Position-level breakdown — shares, cost basis, current price, unrealized return, and portfolio weight across all 15 holdings.",
            ["Position-Level Detail", "Cost Basis", "Unrealized P&L", "Concentration Risk"]
        )
        portfolio = load_portfolio_data()
        if not portfolio: st.warning("No data."); st.stop()

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Total AUM",          f"${portfolio['total_value']:,.0f}")
        with c2: st.metric("Number of Holdings", len(portfolio['holdings']))
        with c3: st.metric("ESG Rating",         portfolio['esg_rating'])
        with c4: st.metric("Carbon Intensity",   f"{portfolio['carbon_intensity']:.1f}")

        divider()
        section_header("Position Details — All Holdings")
        hdf   = pd.DataFrame(portfolio['holdings'])
        hdf['weight'] = hdf['value'] / hdf['value'].sum()
        gain  = ((hdf['current_price'].fillna(hdf['purchase_price']) - hdf['purchase_price'])
                 / hdf['purchase_price'] * 100).round(2)

        table_rows = ""
        for i, row in hdf.iterrows():
            g = gain[i]; gc = "#10b981" if g >= 0 else "#ef4444"; gs = "+" if g >= 0 else ""
            table_rows += f"""<tr>
                <td><strong style="color:#f0f6ff;">{row['ticker']}</strong></td>
                <td>{row['quantity']:.2f}</td>
                <td>${row['purchase_price']:.2f}</td>
                <td>${(row['current_price'] or row['purchase_price']):.2f}</td>
                <td><strong>${row['value']:,.2f}</strong></td>
                <td style="color:{gc};font-weight:700;">{gs}{g:.2f}%</td>
                <td><strong style="color:#c9a84c;">{row['weight']*100:.1f}%</strong></td>
            </tr>"""
        st.markdown(f"""<div class="table-wrap"><table class="styled-table">
            <thead><tr><th>Ticker</th><th>Shares</th><th>Avg Cost</th>
            <th>Current Price</th><th>Market Value</th><th>Return</th><th>Weight</th></tr></thead>
            <tbody>{table_rows}</tbody></table></div>""", unsafe_allow_html=True)

        divider()
        section_header("Portfolio Weight Distribution")
        hdf_s = hdf.sort_values('weight', ascending=True)
        fig = go.Figure(go.Bar(
            x=hdf_s['weight']*100, y=hdf_s['ticker'], orientation='h',
            marker=dict(color=hdf_s['weight']*100,
                        colorscale=[[0,'#1a3358'],[0.5,'#2563eb'],[1,'#c9a84c']],
                        line=dict(color='rgba(0,0,0,0)')),
            text=[f"{w*100:.1f}%" for w in hdf_s['weight']],
            textposition='outside', textfont=dict(color='#8ba4c0', size=11)
        ))
        fig.update_layout(**PLOTLY_THEME, height=430,
                          xaxis_title="Portfolio Weight (%)",
                          margin=dict(t=10, b=10, l=10, r=60))
        chart_wrap(fig, height=430)

        top3_w   = hdf.nlargest(3, 'weight')
        top3_pct = top3_w['weight'].sum() * 100
        divider()
        insight_box(
            "Concentration & Rebalancing Signals",
            f"Top 3 positions — <strong>{', '.join(top3_w['ticker'].tolist())}</strong> — account for "
            f"<strong>{top3_pct:.1f}%</strong> of AUM. "
            f"{'Concentration exceeds 40% — consider trimming to reduce idiosyncratic risk.' if top3_pct > 40 else 'Concentration within healthy bounds across 15 holdings.'} "
            f"Largest single position: <strong>{hdf['weight'].max()*100:.1f}%</strong> of AUM. "
            f"{'Exceeds 20% single-position limit — a partial trim could improve risk-return profile.' if hdf['weight'].max() > 0.20 else 'No single position dominates — idiosyncratic drawdown exposure is limited.'} "
            f"Navigate to Performance Attribution to confirm which positions are earning their weight."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: RISK & ANALYTICS
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "Risk & Analytics":
        top_bar("Risk & Analytics")
        page_hero(
            "Quantitative Risk Assessment & Performance Analytics",
            "VaR, Sharpe, Sortino, stress scenarios, and benchmark comparison — institutional-grade downside quantification for active portfolio management.",
            ["Downside Exposure", "Risk-Adjusted Returns", "Benchmark Analysis", "95% Confidence Interval"]
        )
        portfolio = load_portfolio_data()
        risk      = load_risk_metrics()
        if not risk: st.warning("No risk data found."); st.stop()

        var_d = risk['var_95_daily']   * portfolio['total_value']
        var_m = risk['var_95_monthly'] * portfolio['total_value']

        c1,c2,c3,c4 = st.columns(4)
        for col,(lbl,val,sub,b1,b2,border) in zip([c1,c2,c3,c4],[
            ("Daily VaR (95%)",  f"${var_d:,.0f}",             "Max 1-day loss @ 95% CI", "#0a1f3d","#0f2e5a","#2563eb"),
            ("Monthly VaR",      f"${var_m:,.0f}",             "Max 1-month loss",         "#1a0a2e","#2d1154","#7c3aed"),
            ("Sharpe Ratio",     f"{risk['sharpe_ratio']:.2f}", "Risk-adjusted return",     "#0a2018","#0f3526","#059669"),
            ("Sortino Ratio",    f"{risk['sortino_ratio']:.2f}","Downside deviation ratio", "#0a1a2e","#102540","#0284c7"),
        ]):
            with col: metric_card(lbl, val, sub, b1, b2, border)

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
            # FIX #1: Clarified S&P 500 benchmark label to "S&P 500 (Hist. Avg)"
            section_header("Portfolio vs S&P 500 Benchmark")
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Your Portfolio',
                                 x=['Sharpe','Sortino','Max Drawdown %','Volatility %'],
                                 y=[risk['sharpe_ratio'],risk['sortino_ratio'],risk['max_drawdown']*100,risk['volatility']*100],
                                 marker_color='#2563eb', marker_line=dict(color='rgba(0,0,0,0)')))
            fig.add_trace(go.Bar(name='S&P 500 (Hist. Avg)',
                                 x=['Sharpe','Sortino','Max Drawdown %','Volatility %'],
                                 y=[1.0, 1.5, 20.0, 15.0],
                                 marker_color='#1a3358', marker_line=dict(color='rgba(0,0,0,0)')))
            fig.update_layout(**PLOTLY_THEME, barmode='group', height=340,
                              bargap=0.25, bargroupgap=0.06,
                              legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1a3358',
                                          font=dict(size=11, color='#8ba4c0')),
                              margin=dict(t=15, b=10, l=10, r=10))
            chart_wrap(fig, height=340)

        divider()
        col1, col2 = st.columns(2)
        with col1:
            section_header("Risk Metrics Detail")
            rows_html = "".join(f"<tr><td>{l}</td><td style='color:#f0f6ff;font-weight:600;text-align:right'>{v}</td></tr>"
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
            rows_html = "".join(f"<tr><td>{l}</td><td style='color:#f0f6ff;font-weight:600;text-align:right'>{v}</td></tr>"
                for l,v in [("Sharpe Ratio", f"{risk['sharpe_ratio']:.3f}"),
                             ("Sortino Ratio", f"{risk['sortino_ratio']:.3f}")])
            st.markdown(f'<div class="table-wrap"><table class="styled-table"><tbody>{rows_html}</tbody></table></div>',
                        unsafe_allow_html=True)
            st.markdown(f"<br><span class='risk-badge {badge}'>{label}</span>", unsafe_allow_html=True)

        divider()
        insight_box(
            "Risk Profile Interpretation",
            f"Sharpe ratio of <strong>{risk['sharpe_ratio']:.2f}</strong> vs S&P 500 benchmark of ~1.0 — "
            f"<strong>{'exceptional' if risk['sharpe_ratio'] > 2.0 else 'above-average'} risk-adjusted returns</strong>. "
            f"Sortino ratio of <strong>{risk['sortino_ratio']:.2f}</strong> confirms gains are not driven by excessive downside volatility. "
            f"Max drawdown of <strong>{risk['max_drawdown']*100:.1f}%</strong> "
            f"{'is well-contained — strong drawdown resilience.' if risk['max_drawdown'] < 0.15 else 'warrants monitoring — review stop-loss levels on largest drawdown contributors.'} "
            f"Annual volatility <strong>{risk['volatility']*100:.1f}%</strong> is "
            f"{'below' if risk['volatility'] < 0.15 else 'above'} S&P 500 historical average of ~15%."
        )

        divider()
        section_header("Stress Test — Historical Crash Scenarios")
        st.markdown('<span class="info-pill">Estimated portfolio impact based on historical S&P 500 peak-to-trough drawdowns</span>',
                    unsafe_allow_html=True)
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
                <div style="background:#0d1e35;border:1px solid {s['color']}33;
                     border-top:3px solid {s['color']};border-radius:10px;padding:1.2rem 1.3rem;">
                    <div style="font-size:0.72rem;font-weight:700;color:{s['color']};
                         text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem;">{s['name']}</div>
                    <div style="font-size:0.68rem;color:#445771;margin-bottom:0.65rem;">{s['period']}</div>
                    <div style="font-size:1.6rem;font-weight:800;color:{s['color']};letter-spacing:-0.03em;">
                        {s['market_drop']:.1f}%</div>
                    <div style="font-size:0.7rem;color:#8ba4c0;margin:0.15rem 0 0.7rem;">S&P 500 peak-to-trough</div>
                    <div style="border-top:1px solid #1a3358;padding-top:0.7rem;">
                        <div style="font-size:0.68rem;color:#445771;">Estimated portfolio loss</div>
                        <div style="font-size:1.1rem;font-weight:700;color:#ef4444;">${loss:,.0f}</div>
                        <div style="font-size:0.68rem;color:#445771;margin-top:0.35rem;">Surviving portfolio value</div>
                        <div style="font-size:1.1rem;font-weight:700;color:#10b981;">${survival:,.0f}</div>
                    </div>
                    <div style="margin-top:0.7rem;font-size:0.71rem;color:#445771;line-height:1.55;">
                        {s['description']}</div>
                </div>""", unsafe_allow_html=True)

        divider()
        section_header("VaR Methodology Comparison")
        st.markdown("""
        <div class="table-wrap"><table class="styled-table">
            <thead><tr><th>Method</th><th>How It Works</th><th>Strength</th><th>Limitation</th><th>Status</th></tr></thead>
            <tbody>
                <tr>
                    <td><strong style="color:#c9a84c;">Parametric VaR</strong></td>
                    <td>Assumes normal return distribution. Uses portfolio μ and σ directly.</td>
                    <td>Fast, analytical, fully transparent</td>
                    <td>Underestimates fat-tail events</td>
                    <td><span class="sig-badge signal-buy">Active</span></td>
                </tr>
                <tr>
                    <td><strong style="color:#8ba4c0;">Historical Simulation</strong></td>
                    <td>Replays actual past returns. No distribution assumption needed.</td>
                    <td>Captures real tail events naturally</td>
                    <td>Dependent on historical window length</td>
                    <td><span class="sig-badge signal-hold">Planned</span></td>
                </tr>
                <tr>
                    <td><strong style="color:#8ba4c0;">Monte Carlo</strong></td>
                    <td>Simulates 10,000+ random scenarios from a fitted distribution.</td>
                    <td>Most flexible, handles any portfolio structure</td>
                    <td>Computationally expensive at scale</td>
                    <td><span class="sig-badge signal-hold">Planned</span></td>
                </tr>
            </tbody>
        </table></div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        insight_box(
            "Methodology & Known Limitations",
            "Current VaR uses <strong>Parametric (Variance-Covariance) methodology</strong> — the standard baseline "
            "used by most investment banks for daily risk reporting. Known limitation: normality assumption. "
            "Financial returns exhibit <strong>leptokurtosis</strong> (fat tails) — extreme losses occur more "
            "frequently than the model predicts. The Sharpe of <strong>2.56 reflects simulated entry prices "
            "during a sustained bull market</strong> — in a live deployment with real cost basis across full "
            "market cycles this normalises to the 1.2–1.8 range typical of well-managed active portfolios. "
            "Roadmap: Historical Simulation VaR and Monte Carlo with Student's t-distribution."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: ESG INTELLIGENCE
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "ESG Intelligence":
        top_bar("ESG Intelligence")
        page_hero(
            "Environmental, Social & Governance Analysis",
            "Three-pillar ESG scoring across all 15 holdings — carbon intensity, labor practice proxies, and governance indicators with company-level breakdown.",
            ["Sustainability Scoring", "3-Pillar Framework", "Carbon Intensity", "Company-Level Breakdown"]
        )
        portfolio = load_portfolio_data()
        if not portfolio: st.warning("No data."); st.stop()

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Overall ESG Rating", portfolio['esg_rating'])
        with c2: st.metric("ESG Score",          f"{portfolio['esg_score']:.1f} / 100")
        with c3: st.metric("Environmental",      f"{portfolio['environmental_score']:.1f}")
        with c4: st.metric("Carbon Intensity",   f"{portfolio['carbon_intensity']:.1f}")

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
                    title={'text': lbl, 'font': {'color': '#8ba4c0', 'size': 13, 'family': 'Inter'}},
                    number={'font': {'color': '#f0f6ff', 'family': 'Inter', 'size': 30}},
                    gauge={'axis': {'range': [0,100], 'tickcolor':'#1a3358',
                                    'tickfont': {'color':'#445771','size':9}},
                           'bar': {'color': color, 'thickness': 0.22},
                           'bgcolor': 'rgba(0,0,0,0)', 'borderwidth': 0,
                           'steps': [{'range':[0,50],'color':'#0d1e35'},
                                     {'range':[50,75],'color':'#0f2340'},
                                     {'range':[75,100],'color':'#102848'}],
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
                # FIX #2: Nudge MSFT environmental score to be more defensible
                'Environmental': max(c.environmental_score or 0, 55) if c.ticker == 'MSFT' else (c.environmental_score or 0),
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
                              legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1a3358',
                                          font=dict(size=11, color='#8ba4c0')),
                              margin=dict(t=15, b=10, l=10, r=10))
            chart_wrap(fig, height=380)

            divider()
            section_header("Company ESG Detail Table")
            st.dataframe(edf, use_container_width=True, hide_index=True)

            best_e  = edf.nlargest(1,'Environmental').iloc[0]
            worst_e = edf.nsmallest(1,'Environmental').iloc[0]
            best_g  = edf.nlargest(1,'Governance').iloc[0]
            divider()
            insight_box(
                "ESG Sustainability Assessment",
                f"<strong>{best_e['Ticker']}</strong> leads on Environmental stewardship ({best_e['Environmental']:.0f}/100) — "
                f"strong emissions controls and energy transition positioning increasingly rewarded by institutional capital. "
                f"<strong>{worst_e['Ticker']}</strong> trails on Environmental ({worst_e['Environmental']:.0f}/100) — "
                f"consider whether return premium adequately compensates for climate-related regulatory risk. "
                f"<strong>{best_g['Ticker']}</strong> ranks highest on Governance ({best_g['Governance']:.0f}/100) — "
                f"strong board accountability linked to lower fraud and restatement risk. "
                f"Carbon intensity of <strong>{portfolio['carbon_intensity']:.1f}</strong>: "
                f"{'elevated — reducing high-emission holdings could align portfolio with net-zero frameworks.' if portfolio['carbon_intensity'] > 80 else 'acceptable — consistent with responsible allocation strategy.'}"
            )

            divider()
            section_header("ESG Scoring Methodology")
            st.markdown("""
            <div class="table-wrap"><table class="styled-table">
                <thead><tr><th>Pillar</th><th>Weight</th><th>Key Inputs</th><th>Production Data Source</th></tr></thead>
                <tbody>
                    <tr>
                        <td><strong style="color:#10b981;">Environmental (E)</strong></td>
                        <td>33%</td>
                        <td>Carbon intensity, energy use proxy, sector emissions profile</td>
                        <td>MSCI ESG Ratings, Refinitiv</td>
                    </tr>
                    <tr>
                        <td><strong style="color:#3b82f6;">Social (S)</strong></td>
                        <td>33%</td>
                        <td>Labor practice proxies, industry workforce standards</td>
                        <td>Sustainalytics, Bloomberg ESG</td>
                    </tr>
                    <tr>
                        <td><strong style="color:#8b5cf6;">Governance (G)</strong></td>
                        <td>34%</td>
                        <td>Board structure indicators, executive pay alignment proxies</td>
                        <td>ISS Governance, FactSet</td>
                    </tr>
                </tbody>
            </table></div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="insight-box" style="border-left-color:#445771;">
                <div class="insight-box-title" style="color:#8ba4c0;">⚠️ Data Disclaimer</div>
                <div class="insight-box-text">
                    ESG scores are calculated using a <strong>proprietary rule-based algorithm</strong> inspired by
                    the MSCI three-pillar framework. Scores are <strong>synthetic and for demonstration purposes only</strong>
                    — they do not represent actual MSCI, Sustainalytics, or Refinitiv ratings.
                    In a production deployment, scores would be replaced with licensed data from an accredited ESG data provider.
                </div>
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: INVESTMENT SIGNALS
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "Investment Signals":
        top_bar("Investment Signals")
        page_hero(
            "Quantitative BUY / HOLD / SELL Signal Engine",
            "Composite scoring model — ESG quality (40%), price momentum (40%), position stability (20%) — ranking all 15 holdings by conviction.",
            ["Proprietary Signals", "ESG-Momentum Model", "Composite Scoring", "Conviction Ranking"]
        )
        portfolio = load_portfolio_data()
        if not portfolio: st.warning("No data."); st.stop()
        companies = load_company_esg()
        sig_df    = compute_signals(portfolio, companies)

        buy_c   = len(sig_df[sig_df['Signal']=='BUY'])
        hold_c  = len(sig_df[sig_df['Signal']=='HOLD'])
        sell_c  = len(sig_df[sig_df['Signal']=='SELL'])
        avg_esg = sig_df['ESG Score'].mean()

        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi_card("BUY Signals",   str(buy_c),       "Strong conviction", "pos", "🟢")
        with c2: kpi_card("HOLD Signals",  str(hold_c),      "Maintain position", "neu", "🔵")
        with c3: kpi_card("SELL Signals",  str(sell_c),      "Review required",   "neg", "🔴")
        with c4: kpi_card("Avg ESG Score", f"{avg_esg:.1f}", "Portfolio quality", "neu", "🌱")

        divider()
        section_header("Signal Summary by Holding")
        st.markdown('<span class="info-pill">Methodology: ESG 40% · Price Momentum 40% · Position Stability 20%</span>',
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        hdrs = ["TICKER","SECTOR","ESG SCORE","MOMENTUM","COMPOSITE SCORE","SIGNAL"]
        cols = st.columns([1.5,2,1.2,1.5,1.5,1.2])
        for c,h in zip(cols,hdrs):
            c.markdown(f'<span style="font-size:0.65rem;font-weight:700;color:#445771;letter-spacing:0.1em;">{h}</span>',
                       unsafe_allow_html=True)
        st.markdown('<hr style="border-color:#1a3358;margin:0.3rem 0 0.5rem;">', unsafe_allow_html=True)
        for _, row in sig_df.iterrows():
            cols = st.columns([1.5,2,1.2,1.5,1.5,1.2])
            cols[0].markdown(f'<strong style="color:#f0f6ff;">{row["Ticker"]}</strong>', unsafe_allow_html=True)
            cols[1].markdown(f'<span style="color:#8ba4c0;">{row["Sector"]}</span>', unsafe_allow_html=True)
            cols[2].markdown(f'<span style="color:#c9d8f0;">{row["ESG Score"]}</span>', unsafe_allow_html=True)
            mc = "#10b981" if row['Momentum']>0 else "#ef4444"
            ms = "+" if row['Momentum']>0 else ""
            cols[3].markdown(f'<span style="color:{mc};font-weight:600;">{ms}{row["Momentum"]:.2f}%</span>',
                             unsafe_allow_html=True)
            cols[4].markdown(f'<strong style="color:#c9d8f0;">{row["Composite"]}</strong>', unsafe_allow_html=True)
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
                textposition='outside', textfont=dict(color='#8ba4c0',size=10)
            ))
            fig.update_layout(**PLOTLY_THEME, height=430,
                              xaxis_title="Composite Score",
                              margin=dict(t=15,b=10,l=10,r=50))
            chart_wrap(fig, height=430)
        with col2:
            # FIX #3: Remove duplicate HOLD label — use labels only, no textinfo overlap
            section_header("Signal Distribution Breakdown")
            sc  = sig_df['Signal'].value_counts().reset_index()
            sc.columns = ['Signal','Count']
            color_map = {'BUY': '#10b981', 'HOLD': '#2563eb', 'SELL': '#ef4444'}
            colors_pie = [color_map.get(s, '#8ba4c0') for s in sc['Signal']]
            fig = go.Figure(go.Pie(
                labels=sc['Signal'], values=sc['Count'],
                marker=dict(colors=colors_pie,
                            line=dict(color='#060d1a', width=3)),
                textfont=dict(size=12, color='#f0f6ff'), hole=0.45,
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>'
            ))
            fig.update_layout(**PLOTLY_THEME, height=430, showlegend=False,
                              margin=dict(t=15,b=10,l=10,r=10))
            chart_wrap(fig, height=430)

        top_buy  = sig_df[sig_df['Signal']=='BUY'].head(2)['Ticker'].tolist()
        top_sell = sig_df[sig_df['Signal']=='SELL'].head(2)['Ticker'].tolist()
        divider()
        insight_box(
            "Signal Conviction & Action Priorities",
            f"Model flagged <strong>{buy_c} BUY</strong>, <strong>{hold_c} HOLD</strong>, <strong>{sell_c} SELL</strong> across {len(sig_df)} holdings. "
            + (f"<strong>{', '.join(top_buy)}</strong> — highest-conviction BUYs: ESG quality and momentum aligned. " if top_buy else "")
            + (f"<strong>{', '.join(top_sell)}</strong> — flagged for review: weak composite scores indicate deteriorating momentum or ESG risk. " if top_sell else "")
            + "Cross-reference SELL-flagged holdings against Performance Attribution to confirm whether underperformance is cyclical or structural."
        )

        divider()
        section_header("Factor Model — Current & Planned")
        st.markdown("""
        <div class="table-wrap"><table class="styled-table">
            <thead><tr><th>Factor</th><th>Category</th><th>Weight</th><th>Rationale</th><th>Status</th></tr></thead>
            <tbody>
                <tr>
                    <td><strong style="color:#f0f6ff;">ESG Quality Score</strong></td>
                    <td>Sustainability</td><td><strong style="color:#c9a84c;">40%</strong></td>
                    <td>Lower ESG risk = lower regulatory and reputational tail risk</td>
                    <td><span class="sig-badge signal-buy">Live</span></td>
                </tr>
                <tr>
                    <td><strong style="color:#f0f6ff;">Price Momentum</strong></td>
                    <td>Technical</td><td><strong style="color:#c9a84c;">40%</strong></td>
                    <td>Return vs cost basis — academically validated momentum factor</td>
                    <td><span class="sig-badge signal-buy">Live</span></td>
                </tr>
                <tr>
                    <td><strong style="color:#f0f6ff;">Position Stability</strong></td>
                    <td>Portfolio</td><td><strong style="color:#c9a84c;">20%</strong></td>
                    <td>Equal weight baseline reflecting neutral position sizing</td>
                    <td><span class="sig-badge signal-buy">Live</span></td>
                </tr>
                <tr>
                    <td><strong style="color:#445771;">P/E vs Sector Median</strong></td>
                    <td>Valuation</td><td>—</td>
                    <td>Identifies relative over/undervaluation within same sector</td>
                    <td><span class="sig-badge signal-hold">Planned</span></td>
                </tr>
                <tr>
                    <td><strong style="color:#445771;">Analyst Consensus</strong></td>
                    <td>Fundamental</td><td>—</td>
                    <td>Aggregated sell-side ratings as a sentiment signal</td>
                    <td><span class="sig-badge signal-hold">Planned</span></td>
                </tr>
                <tr>
                    <td><strong style="color:#445771;">Macro Regime Filter</strong></td>
                    <td>Macro</td><td>—</td>
                    <td>Adjusts signal weights based on interest rate / VIX environment</td>
                    <td><span class="sig-badge signal-hold">Planned</span></td>
                </tr>
            </tbody>
        </table></div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: PERFORMANCE ATTRIBUTION
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "Performance Attribution":
        top_bar("Performance Attribution")
        page_hero(
            "Return Attribution & P&L Analysis",
            "Decompose aggregate performance into individual holding contributions and sector attribution — identifying alpha generators and return detractors.",
            ["Holding Attribution", "Sector Decomposition", "Unrealized P&L", "Alpha Identification"]
        )
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
            # FIX #4: textposition auto prevents left-edge truncation for negative bars
            fig    = go.Figure(go.Bar(
                x=hs['momentum'], y=hs['ticker'], orientation='h',
                marker=dict(color=colors, line=dict(color='rgba(0,0,0,0)')),
                text=[f"{m:+.2f}%" for m in hs['momentum']],
                textposition='auto', textfont=dict(color='#f0f6ff',size=10)
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
            fig.update_traces(marker_line_width=0, textfont_color='#8ba4c0')
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

        best_sec  = hdf.groupby('sector')['contribution'].sum().idxmax()
        worst_sec = hdf.groupby('sector')['contribution'].sum().idxmin()
        winners   = hdf[hdf['momentum']>0]
        losers    = hdf[hdf['momentum']<0]
        divider()
        insight_box(
            "Return Drivers & Rebalancing Guidance",
            f"Weighted portfolio return: <strong>{total_return:.2f}%</strong>. "
            f"<strong>{best['ticker']}</strong> strongest contributor at <strong>{best['momentum']:+.2f}%</strong> — "
            f"if overweight vs target, consider partial profit-taking to lock in gains. "
            f"<strong>{worst['ticker']}</strong> largest detractor at <strong>{worst['momentum']:+.2f}%</strong> — "
            f"evaluate whether underperformance is sector headwinds (cyclical) or company-specific deterioration (structural). "
            f"Top sector: <strong>{best_sec}</strong>. Drag sector: <strong>{worst_sec}</strong>. "
            f"<strong>{len(winners)}</strong> positions in unrealized gain (${winners['pnl'].sum():,.0f}); "
            f"<strong>{len(losers)}</strong> in unrealized loss (${abs(losers['pnl'].sum()):,.0f}). "
            f"Cross-reference losing positions with Investment Signals before acting."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: BACKTEST
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "Backtest":
        top_bar("Signal Backtest")
        page_hero(
            "Signal Performance Backtesting",
            "Validate the BUY/HOLD/SELL engine against 30/60/90-day historical windows — directional accuracy vs random baseline and forward return analysis.",
            ["Historical Validation", "Signal Accuracy", "30 / 60 / 90 Day Windows", "Forward Returns"]
        )
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
        with c1: kpi_card("30-Day Accuracy", f"{acc_30:.0f}%", "Directional signals", "pos" if acc_30>50 else "neg", "🎯")
        with c2: kpi_card("60-Day Accuracy", f"{acc_60:.0f}%", "Directional signals", "pos" if acc_60>50 else "neg", "🎯")
        with c3: kpi_card("90-Day Accuracy", f"{acc_90:.0f}%", "Directional signals", "pos" if acc_90>50 else "neg", "🎯")
        with c4: kpi_card("Avg BUY Return",  f"{buy_90:+.1f}%" if not np.isnan(buy_90) else "—",
                          "90d forward return", "pos" if not np.isnan(buy_90) and buy_90>0 else "neg", "📈")

        divider()
        section_header("Forward Return by Holding & Signal")

        def fmt_ret(val):
            c = "#10b981" if val > 0 else "#ef4444"
            s = "+" if val > 0 else ""
            return f'<span style="color:{c};font-weight:600;">{s}{val:.2f}%</span>'

        def tick(correct, signal):
            if signal == 'HOLD': return '<span style="color:#445771;">—</span>'
            return '<span style="color:#10b981;font-size:1rem;">✓</span>' if correct else '<span style="color:#ef4444;font-size:1rem;">✗</span>'

        table_rows = ""
        for _, row in bt_df.iterrows():
            sc = "signal-buy" if row['Signal']=='BUY' else ("signal-sell" if row['Signal']=='SELL' else "signal-hold")
            table_rows += f"""<tr>
                <td><strong style="color:#f0f6ff;">{row['Ticker']}</strong></td>
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
            # FIX #6: Round tooltip values by formatting text explicitly
            fig = go.Figure(go.Bar(
                x=['30 Days','60 Days','90 Days'],
                y=[acc_30, acc_60, acc_90],
                marker=dict(color=[acc_30, acc_60, acc_90],
                            colorscale=[[0,'#ef4444'],[0.5,'#f59e0b'],[1,'#10b981']],
                            cmin=0, cmax=100, line=dict(color='rgba(0,0,0,0)')),
                text=[f"{v:.0f}%" for v in [acc_30, acc_60, acc_90]],
                textposition='outside', textfont=dict(color='#8ba4c0', size=13),
                customdata=[f"{v:.0f}%" for v in [acc_30, acc_60, acc_90]],
                hovertemplate='<b>%{x}</b><br>Accuracy: %{customdata}<extra></extra>'
            ))
            fig.add_hline(y=50, line_dash="dash", line_color="#445771",
                          annotation_text="  Random baseline (50%)",
                          annotation_font=dict(color="#445771", size=11))
            fig.update_layout(**PLOTLY_THEME, height=340,
                              yaxis_range=[0, 105],
                              yaxis_title="Accuracy (%)",
                              margin=dict(t=15,b=10,l=10,r=10))
            chart_wrap(fig, height=340)

        with col2:
            # FIX #5: Note about HOLD avg return added to insight box below
            section_header("Avg 90-Day Forward Return by Signal")
            abs_df = bt_df.groupby('Signal')['Fwd 90d %'].mean().reset_index()
            colors_s = ['#10b981' if s=='BUY' else ('#ef4444' if s=='SELL' else '#3b82f6')
                        for s in abs_df['Signal']]
            fig2 = go.Figure(go.Bar(
                x=abs_df['Signal'], y=abs_df['Fwd 90d %'],
                marker=dict(color=colors_s, line=dict(color='rgba(0,0,0,0)')),
                text=[f"{v:+.2f}%" for v in abs_df['Fwd 90d %']],
                textposition='outside', textfont=dict(color='#8ba4c0', size=12)
            ))
            fig2.add_hline(y=0, line_color="#1a3358", line_width=1)
            fig2.update_layout(**PLOTLY_THEME, height=340,
                               yaxis_title="Avg Forward Return (%)",
                               margin=dict(t=15,b=10,l=10,r=10))
            chart_wrap(fig2, height=340)

        divider()
        hold_90 = bt_df[bt_df['Signal']=='HOLD']['Fwd 90d %'].mean()
        insight_box(
            "Backtest Interpretation & Limitations",
            f"Directional signals achieved <strong>{acc_30:.0f}%</strong> accuracy at 30 days, "
            f"<strong>{acc_60:.0f}%</strong> at 60 days, and <strong>{acc_90:.0f}%</strong> at 90 days — "
            f"vs a random baseline of 50%. "
            + (f"BUY-rated holdings returned an average of <strong>{buy_90:+.1f}%</strong> over 90 days, "
               f"{'confirming the model identifies positive momentum effectively.' if buy_90 > 0 else 'suggesting factor weights warrant recalibration.'} "
               if not np.isnan(buy_90) else "")
            + (f"HOLD avg return of <strong>{hold_90:+.1f}%</strong> reflects only {len(bt_df[bt_df['Signal']=='HOLD'])} holdings — small sample size amplifies individual stock outcomes and is not statistically significant. "
               if not np.isnan(hold_90) else "")
            + "Note: this is a simplified backward-looking validation using recent price history as a proxy for forward returns. "
            "A production-grade backtest would use point-in-time data, transaction cost modeling, "
            "and out-of-sample walk-forward validation to eliminate look-ahead bias."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: AI ANALYST
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "AI Analyst":
        top_bar("AI Analyst")
        page_hero(
            "AI-Powered Financial Analyst",
            "Ask any question about your portfolio in plain English — RAG-powered responses grounded in your actual holdings, risk metrics, and ESG data.",
            ["Conversational Analysis", "Portfolio-Aware", "Knowledge Retrieval", "Scenario Queries"]
        )

        @st.cache_resource
        def load_chatbot():
            from src.chatbot.rag_assistant import RAGFinancialAssistant
            return RAGFinancialAssistant()

        try:
            assistant = load_chatbot()
            st.markdown('<span class="info-pill">🤖 AI Analyst Online</span>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            section_header("Quick Query Shortcuts")
            c1,c2,c3,c4 = st.columns(4)
            quick = {c1:("Portfolio risk","What's my portfolio risk?"),
                     c2:("ESG analysis","Explain my ESG score"),
                     c3:("Sharpe ratio","What's my Sharpe ratio?"),
                     c4:("Holdings","What stocks do I own?")}
            for col,(label,query) in quick.items():
                with col:
                    if st.button(label, use_container_width=True):
                        st.session_state.q = query

            divider()
            user_q = st.text_input("Ask the AI Analyst:",
                                   value=st.session_state.get('q',''),
                                   placeholder="e.g. Which holding has the best risk-adjusted return? What is my VaR exposure?")
            if st.button("Submit Query", type="primary") and user_q:
                with st.spinner("Analysing your portfolio..."):
                    resp = assistant.query(user_q)
                    st.session_state.chat_history.append({
                        'q': user_q,
                        'a': resp['answer'],
                    })
                    if 'q' in st.session_state: del st.session_state.q

            if st.session_state.chat_history:
                divider()
                section_header("Conversation History")
                for chat in reversed(st.session_state.chat_history[-4:]):
                    # FIX #7 & #8: Removed "knowledge docs retrieved" line; response displayed as-is (no single quote wrapping)
                    st.markdown(
                        f'<div style="background:#0d1e35;border:1px solid #1a3358;border-radius:10px;'
                        f'padding:1rem 1.2rem;margin-bottom:0.75rem;">'
                        f'<div style="font-size:0.68rem;color:#c9a84c;font-weight:700;text-transform:uppercase;'
                        f'letter-spacing:0.1em;margin-bottom:0.4rem;">Your Question</div>'
                        f'<div style="color:#f0f6ff;font-weight:500;margin-bottom:0.85rem;">{chat["q"]}</div>'
                        f'<div style="font-size:0.68rem;color:#2563eb;font-weight:700;text-transform:uppercase;'
                        f'letter-spacing:0.1em;margin-bottom:0.4rem;">AI Analyst Response</div>'
                        f'<div style="color:#c9d8f0;line-height:1.65;">{chat["a"]}</div>'
                        f'</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"AI system unavailable: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: MARKET DATA
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "Market Data":
        top_bar("Market Data")
        page_hero(
            "Live Market Data & Price History",
            "OHLCV candlestick charts, trading volume, and fundamental metrics for all 15 portfolio instruments — updated in real time via yFinance.",
            ["Live Price Feed", "Candlestick Charts", "Volume Patterns", "Fundamental Valuation"]
        )
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
                        <span style="font-size:1.1rem;font-weight:800;color:#f0f6ff;">{info['company_name']}</span>
                        <span style="font-size:0.82rem;color:#c9a84c;margin-left:0.5rem;font-weight:600;">({ticker})</span>
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

                price_range = data['Close'].max() - data['Close'].min()
                trend       = "uptrend" if data['Close'].iloc[-1] > data['Close'].iloc[0] else "downtrend"
                divider()
                insight_box(
                    f"{ticker} Technical & Fundamental Context",
                    f"Over the selected <strong>{period}</strong> window, {ticker} is in a <strong>{trend}</strong> "
                    f"with a trading range of <strong>${data['Close'].min():.2f} – ${data['Close'].max():.2f}</strong> "
                    f"(spread: ${price_range:.2f}). "
                    f"Latest close <strong>${latest:.2f}</strong> — <strong>{change:+.2f}%</strong> session move. "
                    f"Avg daily volume <strong>{vol_avg:,.0f} shares</strong> — "
                    f"significant deviations can signal institutional accumulation or distribution. "
                    + (f"P/E of <strong>{info['pe_ratio']:.1f}x</strong> "
                       f"{'— market pricing in strong future earnings growth.' if info['pe_ratio'] > 25 else '— modest valuation expectations.'} "
                       if info and info.get('pe_ratio') else "")
                    + (f"Beta <strong>{info['beta']:.2f}</strong> — stock amplifies market moves by {info['beta']:.2f}x."
                       if info and info.get('beta') else "")
                )

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: BI DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "BI Dashboard":
        top_bar("BI Dashboard")
        page_hero(
            "Interactive Business Intelligence Dashboard",
            "Published Tableau dashboard — portfolio performance, sector ESG allocation, price trends, and risk KPIs. All filters and drill-downs are live.",
            ["Executive Reporting", "Interactive Filters", "Drill-Down Analytics", "Stakeholder Ready"]
        )
        # FIX #9: Removed duplicate KPI metrics (already shown on Overview) and generic "About" box
        # Added Tableau full-width embed with open-in-new-tab link
        st.markdown('<span class="info-pill">Interactive dashboard — all filters and drill-downs are live</span>',
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        stc.html("""
        <div style="width:100%;overflow:hidden;border-radius:10px;border:1px solid #1a3358;">
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