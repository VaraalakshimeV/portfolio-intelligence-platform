# 📈 Portfolio Intelligence Platform (PIP)
### *Institutional-Grade Portfolio Analytics | RAG-Powered AI Analyst | Live Market Data*

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.43-FF4B4B.svg)
![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-4285F4.svg)
![Pinecone](https://img.shields.io/badge/Pinecone-RAG-green.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.22-purple.svg)
![Status](https://img.shields.io/badge/Status-Live-brightgreen.svg)

**🚀 [Live Demo](https://portfolio-intelligence-platform-3fhbbrtae9n8wzankmyuhz.streamlit.app) | Demo Login: analyst@pip.com / Demo@1234**

</div>

---

## 💡 The Problem

Portfolio managers and analysts need institutional-grade tools to monitor holdings, quantify risk, assess ESG exposure, and generate investment signals — but enterprise platforms like Bloomberg Terminal cost $25,000+/year and are inaccessible to independent analysts and students learning quantitative finance.

**The Challenge:** Build a full-stack portfolio intelligence system that replicates core institutional capabilities — real-time data, quantitative risk modeling, ESG scoring, AI-powered analysis — in an accessible, deployable platform.

---

## ✨ My Solution

I designed and built an end-to-end portfolio analytics platform that ingests live market data for 15 equities, computes institutional risk metrics, generates ESG scores using an MSCI-inspired algorithm, produces BUY/HOLD/SELL signals via a quantitative factor model, and answers natural language questions about the portfolio via a RAG-powered AI chatbot.

**What It Does:**

**Input:** 15 live equity holdings via yFinance + SQLite portfolio database  
**Process:** Market Data → Risk Engine → ESG Scoring → Signal Engine → RAG Chatbot → Interactive Dashboard  
**Output:** 10-page analytics platform with real-time insights, stress testing, backtesting, and AI analysis

---

## 📊 Business Impact

| Capability | Traditional Approach | PIP | Result |
|-----------|---------------------|-----|--------|
| **Risk Reporting** | Manual Excel models | Automated VaR + Sharpe + Sortino | **Real-time quantification** |
| **ESG Analysis** | $10K+/year data licenses | Rule-based MSCI-inspired algorithm | **Automated scoring** |
| **Investment Signals** | Analyst discretion | 3-factor composite model | **Systematic, auditable** |
| **AI Analysis** | Human analyst hours | RAG chatbot + Gemini 2.5 Flash | **Instant portfolio Q&A** |
| **Stress Testing** | Static scenario models | Dynamic 2008/COVID/2022 simulation | **Automated downside modeling** |
| **Signal Validation** | No backtesting | 30/60/90-day accuracy validation | **Model accountability** |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Frontend                     │
│        10 Pages · Custom CSS Design System              │
│        Dark Navy UI · Inter Font · Gold Accents         │
└──────────────┬──────────────────────────────────────────┘
               │
   ┌───────────┼────────────┬──────────────┐
   │           │            │              │
┌──▼─────┐ ┌──▼──────┐ ┌───▼─────┐ ┌─────▼──────┐
│  Data  │ │   AI    │ │  Risk   │ │   Signal   │
│ Layer  │ │ Layer   │ │ Engine  │ │   Engine   │
│SQLite  │ │ Gemini  │ │Param.   │ │ESG 40%     │
│yFinance│ │Pinecone │ │VaR      │ │Momentum 40%│
│SQLAlch.│ │RAG + ML │ │Monte    │ │Stability   │
│        │ │         │ │Carlo    │ │20%         │
└────────┘ └─────────┘ └─────────┘ └────────────┘
```

---

## 🎬 Application Showcase

### **Executive Overview**
<img width="1617" height="823" alt="image" src="https://github.com/user-attachments/assets/1851d55c-eb67-4dd9-8950-55b440e6e496" />

*5 KPI cards — AUM, ESG Rating, Sharpe Ratio, Daily VaR, Holdings count with donut allocation chart*

---

> 📸 **Want to explore all pages?** [View the Live App →](https://portfolio-intelligence-platform-3fhbbrtae9n8wzankmyuhz.streamlit.app/)
>
> *Includes: Login · Risk & Analytics · ESG Intelligence · Investment Signals · Signal Backtest · AI Analyst · Market Data · BI Dashboard*

## ⚙️ Technical Architecture

### **Core Components**

**1. Data Layer**
- SQLite database with SQLAlchemy ORM — portfolio holdings, risk metrics, ESG scores
- yFinance integration for live price feeds and OHLCV candlestick data
- DataCollector class abstracting all market data retrieval

**2. AI Layer**
- Google Gemini 2.5 Flash — LLM for natural language portfolio analysis
- Pinecone vector database — semantic retrieval of financial knowledge documents
- RAG pipeline — hybrid retrieval grounding every response in actual portfolio data
- Sentence Transformers — text embedding for semantic similarity search

**3. Risk Engine**
- Parametric VaR — 95% confidence interval, daily and monthly
- Monte Carlo simulation — 10,000 scenario return distribution
- Sharpe and Sortino ratio calculation
- Stress testing — 2008 GFC, COVID crash, 2022 rate hike cycle

**4. Signal Engine**
- 3-factor composite model: ESG quality (40%) + price momentum (40%) + position stability (20%)
- BUY/HOLD/SELL classification with configurable thresholds
- 30/60/90-day backtested directional accuracy validation

**5. ESG Scoring**
- MSCI-inspired 3-pillar algorithm (Environmental, Social, Governance)
- Rule-based scoring using sector proxies and industry benchmarks
- Letter rating system (AAA → CCC) with carbon intensity tracking

**6. UI Design System**
- Custom CSS — dark navy (#060d1a), gold accent (#c9a84c), Inter font
- 8 reusable component helpers: `kpi_card()`, `metric_card()`, `insight_box()`, `page_hero()` and more
- Consistent Plotly theme across all 15+ interactive charts

---

## 🛠️ Technology Stack

| Category | Technologies | Purpose |
|----------|-------------|---------|
| **Frontend** | Streamlit, Custom CSS, Plotly | UI, interactive charts, dark theme |
| **AI/LLM** | Google Gemini 2.5 Flash, Pinecone, Sentence Transformers | RAG chatbot, vector search |
| **ML** | Scikit-learn, SciPy, NumPy | Risk predictor (R²=0.66), VaR simulation |
| **Data** | yFinance, Pandas, SQLAlchemy | Live prices, data processing, ORM |
| **Database** | SQLite | Portfolio holdings, risk metrics, ESG scores |
| **BI** | Tableau Public | Embedded executive dashboard |
| **Deployment** | Streamlit Community Cloud, GitHub | Live hosting, version control |

---

## 🎯 Key Features

✅ **10-Page Analytics Platform** — Overview, Portfolio, Risk, ESG, Signals, Attribution, Backtest, AI Analyst, Market Data, BI Dashboard

✅ **Live Market Data** — Real-time prices, OHLCV candlesticks, fundamentals via yFinance

✅ **Institutional Risk Metrics** — Parametric VaR, Sharpe, Sortino, stress test scenarios

✅ **RAG-Powered AI Chatbot** — Portfolio-aware Q&A grounded in actual holdings data

✅ **Quantitative Signal Engine** — 3-factor BUY/HOLD/SELL with backtest validation

✅ **ESG Intelligence** — MSCI-inspired scoring across 15 holdings with methodology transparency

✅ **Professional UI** — JPMorgan-style dark navy design system, Inter font, gold accents

✅ **Login System** — Multi-user authentication (Analyst, Manager, Admin roles)

---

## 📊 Portfolio Universe

**15 equities across 8 sectors — $100,000 simulated AUM, equal-weight construction:**

`AAPL` `MSFT` `GOOGL` `NVDA` `TSLA` `AMZN` `JPM` `GS` `V` `JNJ` `BA` `CAT` `UNH` `WMT` `XOM`

| Sector | Holdings |
|--------|---------|
| Technology | AAPL, MSFT, GOOGL, NVDA |
| Financial Services | JPM, GS, V |
| Healthcare | JNJ, UNH |
| Consumer Cyclical | TSLA, AMZN |
| Industrials | BA, CAT |
| Consumer Defensive | WMT |
| Energy | XOM |

---

## 🔑 Key Design Decisions

**Streamlit over React** — analytical depth and rapid iteration over frontend engineering. Target users are portfolio analysts, not consumers.

**SQLite over PostgreSQL** — SQLAlchemy ORM is database-agnostic. One connection string change switches to PostgreSQL in production.

**Parametric VaR** — standard baseline used by most investment banks for daily risk reporting. Fat-tail limitation (leptokurtosis) documented directly in the platform.

**Synthetic ESG scores** — rule-based MSCI-inspired algorithm. In production, replace with licensed data from MSCI, Sustainalytics, or Refinitiv via a single API key change.

**Equal-weight portfolio** — eliminates allocation bias, allowing the signal engine and attribution models to isolate factor contributions cleanly.

---

## 💻 Local Setup

```bash
# Clone the repository
git clone https://github.com/VaraalakshimeV/portfolio-intelligence-platform.git
cd portfolio-intelligence-platform

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create .env file with:
# GOOGLE_API_KEY=your_gemini_key
# PINECONE_API_KEY=your_pinecone_key
# PINECONE_ENVIRONMENT=your_pinecone_env

# Run the app
streamlit run app.py
```

**Demo Login:** `analyst@pip.com` / `Demo@1234`

---

## 📈 Results

- **10 fully functional analytics pages** deployed on Streamlit Cloud
- **15 live equity holdings** with real-time price feeds
- **Sharpe ratio 2.56** — reflects simulated bull market entry prices; normalizes to 1.2–1.8 in live deployment
- **ML risk predictor R² = 0.66** on held-out validation set
- **58% directional signal accuracy** at 30 and 90-day horizons vs 50% random baseline
- **RAG chatbot** grounded in portfolio-specific knowledge base

---

## 🔮 Future Enhancements

- **Walk-forward backtest** — point-in-time data with transaction cost modeling
- **Historical Simulation VaR** — non-parametric fat-tail modeling
- **Monte Carlo VaR** — Student's t-distribution for extreme event modeling
- **P/E vs Sector Median** — relative valuation signal factor
- **Analyst Consensus** — sell-side sentiment signal integration
- **Macro Regime Filter** — interest rate / VIX environment adjustment

---

## 🤝 Let's Connect

I'm a **Data Analytics Engineering graduate student at Northeastern University** (GPA: 4.0) seeking **co-op / full-time Data Science and Data Engineering roles**.

This project demonstrates my ability to:
- ✅ Build full-stack data applications with production deployment
- ✅ Implement quantitative finance models (VaR, Sharpe, factor models)
- ✅ Design and deploy RAG pipelines with LLM integration
- ✅ Create professional analytics platforms with enterprise-grade UI

<div align="center">

📧 **Email:** vigneswarapandiara.v@northeastern.edu  
💼 **LinkedIn:** [linkedin.com/in/varaalakshime-v](https://www.linkedin.com/in/varaalakshime-v)  
🐙 **GitHub:** [github.com/VaraalakshimeV](https://github.com/VaraalakshimeV)

**Available for Co-op:** May 2025 – December 2025

</div>

---

<div align="center">

**⭐ Built with Streamlit · Gemini 2.5 Flash · Pinecone · Plotly · SQLite ⭐**

*Institutional-Grade Portfolio Intelligence, Accessible to Everyone*

### ⭐ If you found this project helpful, please star the repository!

**Built with ❤️ for quantitative finance and applied AI**

</div>
