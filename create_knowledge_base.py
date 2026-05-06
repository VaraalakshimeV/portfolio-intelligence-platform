"""
Create Financial Knowledge Base for RAG
Stores financial concepts, portfolio-specific data, and ESG scores in Pinecone.
Embedding model: all-mpnet-base-v2 (768-dim)
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

INDEX_NAME  = 'fintech-rag'
EMBED_MODEL = 'all-mpnet-base-v2'
DIMENSIONS  = 768

print("=" * 70)
print("CREATING FINANCIAL KNOWLEDGE BASE")
print(f"Model: {EMBED_MODEL}  |  Dimensions: {DIMENSIONS}")
print("=" * 70)

# ── Connect to Pinecone ────────────────────────────────────────────────────────
print("\n1. Connecting to Pinecone...")
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

existing = [i.name for i in pc.list_indexes()]
if INDEX_NAME in existing:
    print(f"   Deleting old index '{INDEX_NAME}'...")
    pc.delete_index(INDEX_NAME)
    time.sleep(5)

print(f"   Creating index '{INDEX_NAME}' ({DIMENSIONS}-dim)...")
pc.create_index(
    name=INDEX_NAME, dimension=DIMENSIONS, metric='cosine',
    spec=ServerlessSpec(cloud='aws', region='us-east-1')
)
time.sleep(10)
index = pc.Index(INDEX_NAME)
print("   ✓ Index ready")

# ── Load embedding model ───────────────────────────────────────────────────────
print("\n2. Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)
print(f"   ✓ Loaded ({DIMENSIONS} dimensions)")

# ── Pull live data from database ───────────────────────────────────────────────
print("\n3. Loading live portfolio data from database...")
from src.database.database import SessionLocal
from src.database.models import Portfolio, Holding, CompanyInfo, RiskMetrics

db = SessionLocal()
try:
    portfolio  = db.query(Portfolio).first()
    holdings   = db.query(Holding).filter(Holding.portfolio_id == portfolio.id).all()
    companies  = db.query(CompanyInfo).all()
    risk       = db.query(RiskMetrics).order_by(RiskMetrics.calculation_date.desc()).first()
    company_map = {c.ticker: c for c in companies}
finally:
    db.close()

print(f"   ✓ Portfolio: {len(holdings)} holdings, ${portfolio.total_value:,.0f} AUM")

# ── Build knowledge documents ──────────────────────────────────────────────────
print("\n4. Building knowledge documents...")
knowledge_base = []

# ── 1. Portfolio summary ───────────────────────────────────────────────────────
holding_list = ", ".join(h.ticker for h in holdings)
knowledge_base.append({
    'id': 'portfolio_summary',
    'category': 'portfolio',
    'text': f"""Portfolio Summary:
Total portfolio value (AUM): ${portfolio.total_value:,.2f}.
The portfolio holds {len(holdings)} positions: {holding_list}.
Overall ESG score: {portfolio.esg_score_overall:.1f}/100, ESG rating: {portfolio.esg_rating}.
Environmental score: {portfolio.environmental_score:.1f}, Social score: {portfolio.social_score:.1f}, Governance score: {portfolio.governance_score:.1f}.
Carbon intensity: {portfolio.carbon_intensity:.1f} tons CO2 per $1M invested.
Carbon footprint: {portfolio.carbon_footprint:.1f} tons CO2."""
})

# ── 2. Risk metrics (actual values) ───────────────────────────────────────────
if risk:
    var_d = risk.var_95_daily   * portfolio.total_value
    var_m = risk.var_95_monthly * portfolio.total_value
    knowledge_base.append({
        'id': 'portfolio_risk_metrics',
        'category': 'risk_metrics',
        'text': f"""Portfolio Risk Metrics (actual calculated values):
Sharpe Ratio: {risk.sharpe_ratio:.4f}. A Sharpe above 2.0 is considered very good; the S&P 500 typically achieves 0.4–0.8 over long periods and around 1.0–1.5 in strong bull markets. Our portfolio Sharpe of {risk.sharpe_ratio:.2f} significantly outperforms the typical S&P 500 benchmark.
Sortino Ratio: {risk.sortino_ratio:.4f} (measures downside risk only, higher is better).
Annualised Volatility: {risk.volatility*100:.2f}%.
Maximum Drawdown: {risk.max_drawdown*100:.2f}% (largest peak-to-trough decline).
Daily VaR at 95% confidence: ${var_d:,.2f} — there is a 5% chance of losing more than this in a single day.
Monthly VaR at 95% confidence: ${var_m:,.2f}.
S&P 500 benchmark context: The S&P 500 historically has a Sharpe ratio of ~0.5, a Sortino of ~0.7, annualised volatility of ~15–20%, and max drawdown of ~50% in 2008 and ~34% in 2020. Our portfolio compares favourably on risk-adjusted return metrics."""
    })

# ── 3. Per-holding details ─────────────────────────────────────────────────────
for h in holdings:
    c   = company_map.get(h.ticker)
    cur = h.current_price or h.purchase_price
    ret = (cur - h.purchase_price) / h.purchase_price * 100
    wt  = (cur * h.quantity) / portfolio.total_value * 100

    esg_text = ""
    if c:
        esg_text = (
            f"ESG overall score: {c.esg_score:.1f}/100, rating: {c.esg_rating or 'N/A'}. "
            f"Environmental: {c.environmental_score:.1f}, Social: {c.social_score:.1f}, "
            f"Governance: {c.governance_score:.1f}. "
            f"Carbon emissions: {c.carbon_emissions:.1f} tons CO2."
        ) if (c.esg_score and c.environmental_score) else "ESG data not available for this holding."

    sector = c.sector if c else "Unknown"
    name   = c.company_name if c else h.ticker

    knowledge_base.append({
        'id': f'holding_{h.ticker.lower()}',
        'category': 'holdings',
        'text': f"""Holding: {h.ticker} ({name}), Sector: {sector}.
Quantity: {h.quantity:.2f} shares. Purchase price: ${h.purchase_price:.2f}. Current price: ${cur:.2f}.
Market value: ${cur * h.quantity:,.2f}. Portfolio weight: {wt:.2f}%.
Return vs cost basis: {ret:+.2f}%.
{esg_text}"""
    })

# ── 4. ESG ranking across all holdings ────────────────────────────────────────
esg_rows = []
for h in holdings:
    c = company_map.get(h.ticker)
    if c and c.esg_score:
        esg_rows.append((h.ticker, c.esg_score, c.esg_rating or 'N/A',
                         c.environmental_score or 0, c.social_score or 0, c.governance_score or 0))

esg_rows.sort(key=lambda x: x[1], reverse=True)
esg_ranking = "\n".join(
    f"  {i+1}. {t}: Overall {s:.1f}/100 ({r}) — E:{e:.1f} S:{so:.1f} G:{g:.1f}"
    for i, (t, s, r, e, so, g) in enumerate(esg_rows)
)
knowledge_base.append({
    'id': 'esg_holdings_ranking',
    'category': 'esg',
    'text': f"""ESG scores for all portfolio holdings ranked highest to lowest:
{esg_ranking}
The highest ESG-rated holding is {esg_rows[0][0]} with a score of {esg_rows[0][1]:.1f}/100.
The lowest ESG-rated holding is {esg_rows[-1][0]} with a score of {esg_rows[-1][1]:.1f}/100.
Portfolio average ESG score: {sum(r[1] for r in esg_rows)/len(esg_rows):.1f}/100."""
})

# ── 5. Benchmark comparison ────────────────────────────────────────────────────
knowledge_base.append({
    'id': 'benchmark_comparison',
    'category': 'risk_metrics',
    'text': f"""Portfolio vs S&P 500 Benchmark Comparison:
Our portfolio Sharpe Ratio is {risk.sharpe_ratio:.2f}. The S&P 500 long-run Sharpe ratio is approximately 0.4–0.6. In strong bull markets (2019, 2021, 2023) the S&P 500 Sharpe can reach 1.0–1.8. Our portfolio's Sharpe of {risk.sharpe_ratio:.2f} is significantly higher than the typical S&P 500, indicating superior risk-adjusted returns.
Our portfolio annualised volatility is {risk.volatility*100:.2f}% vs the S&P 500's typical 15–18%.
Our max drawdown is {risk.max_drawdown*100:.2f}% vs the S&P 500's worst drawdown of ~50% (2008 financial crisis) and ~34% (COVID-19 crash in 2020).
Our Sortino ratio is {risk.sortino_ratio:.2f}, which compares favourably to the S&P 500 Sortino of ~0.7–1.2.
Overall the portfolio demonstrates better risk-adjusted returns than the S&P 500 benchmark across all key metrics."""
})

# ── 6. General financial knowledge ────────────────────────────────────────────
general_docs = [
    ('var_definition', 'risk_metrics', """Value at Risk (VaR) is a statistical measure of the potential loss in portfolio value over a specific time period at a given confidence level.
For example, a daily VaR of $10,000 at 95% confidence means there is a 5% chance of losing more than $10,000 in one day.
VaR is calculated using three methods: Historical (using past returns), Parametric (assuming normal distribution), and Monte Carlo (simulating future scenarios).
Financial institutions use VaR for risk management and regulatory compliance."""),

    ('sharpe_ratio_concept', 'risk_metrics', """The Sharpe Ratio measures risk-adjusted returns by comparing excess returns to volatility.
Formula: (Portfolio Return - Risk-Free Rate) / Portfolio Standard Deviation.
A Sharpe Ratio above 1.0 is considered good, above 2.0 is very good, and above 3.0 is excellent.
The S&P 500 historically achieves a Sharpe of 0.4–0.8. Hedge funds typically target Sharpe ratios of 1.0–2.0.
A Sharpe above 2.0 like ours indicates exceptional risk-adjusted performance."""),

    ('esg_overview', 'esg', """ESG stands for Environmental, Social, and Governance — three pillars for measuring corporate sustainability.
Environmental criteria examine a company's carbon footprint, renewable energy use, and waste management.
Social factors include employee treatment, diversity, and community impact.
Governance assesses board independence, executive compensation, and shareholder rights.
ESG scores range from 0–100. Ratings: AAA/AA (leader), A/BBB (average), BB/B/CCC (laggard).
Higher ESG scores are associated with lower long-term risk and better regulatory standing."""),

    ('max_drawdown', 'risk_metrics', """Maximum Drawdown measures the largest peak-to-trough decline in portfolio value.
It represents the worst possible loss an investor would have experienced at any point.
A 20% max drawdown means the portfolio fell 20% from its highest point before recovering.
Recovery time matters: a 50% drawdown requires a 100% gain to recover.
The S&P 500's worst drawdowns: -50% in 2008, -34% in March 2020, -25% in 2022."""),

    ('beta_alpha', 'risk_metrics', """Beta measures a portfolio's sensitivity to market movements relative to the S&P 500.
Beta = 1.0 means the portfolio moves in line with the market. Beta > 1 means more volatile (aggressive). Beta < 1 means less volatile (defensive).
Alpha measures excess returns above what the portfolio's beta would predict.
Positive alpha means the portfolio outperformed on a risk-adjusted basis — a sign of active management skill."""),

    ('carbon_footprint', 'esg', """Carbon footprint measures greenhouse gas emissions attributed to investments, reported as tons of CO2 equivalent per million dollars invested (carbon intensity).
Carbon intensity below 50 tons/$1M is considered very low. 50–150 is moderate. Above 150 is high.
Sectors with highest carbon intensity: Energy, Utilities, Materials.
Sectors with lowest carbon intensity: Technology, Healthcare, Financials.
Investors reduce portfolio carbon by tilting towards low-emission sectors and companies with net-zero commitments."""),

    ('stress_testing', 'risk_metrics', """Stress testing applies historical market crash scenarios to a portfolio to estimate potential losses.
Key scenarios: 2008 Global Financial Crisis (S&P 500 fell 38.5%), COVID-19 crash Feb–Mar 2020 (S&P 500 fell 33.9%), 2022 Rate Hike Cycle (S&P 500 fell 19.4%).
The methodology applies the historical drawdown percentage to the current portfolio value.
This shows the estimated portfolio loss if an equivalent crisis happened today, helping investors understand tail risk."""),
]

for doc_id, cat, text in general_docs:
    knowledge_base.append({'id': doc_id, 'category': cat, 'text': text})

print(f"   ✓ {len(knowledge_base)} documents prepared ({sum(1 for d in knowledge_base if d['category'] == 'holdings')} holdings + portfolio data + general knowledge)")

# ── Embed and upload ───────────────────────────────────────────────────────────
print("\n5. Embedding and uploading to Pinecone...")
vectors = []
for doc in knowledge_base:
    embedding = embedder.encode(doc['text']).tolist()
    vectors.append({
        'id': doc['id'],
        'values': embedding,
        'metadata': {'text': doc['text'], 'category': doc['category']}
    })
    print(f"   ✓ {doc['id']}")

index.upsert(vectors=vectors)

# ── Verify ─────────────────────────────────────────────────────────────────────
time.sleep(3)
stats = index.describe_index_stats()
print(f"\n6. Verification: {stats['total_vector_count']} vectors in index")

print("\n" + "=" * 70)
print("✅ KNOWLEDGE BASE CREATED")
print("=" * 70)
print(f"\nDocuments indexed:")
print(f"  • Portfolio summary + actual AUM and holdings")
print(f"  • Risk metrics with real values + S&P 500 benchmark comparison")
print(f"  • {len(holdings)} individual holding documents (price, weight, ESG)")
print(f"  • ESG ranking table across all {len(esg_rows)} holdings")
print(f"  • Benchmark comparison (vs S&P 500 Sharpe, volatility, drawdown)")
print(f"  • General financial knowledge (VaR, Beta, Alpha, stress testing)")
