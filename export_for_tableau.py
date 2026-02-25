"""
Export Portfolio Data to CSV for Tableau
Professional Dashboard - Fintech AI Platform
"""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.database.database import SessionLocal
from src.database.models import Portfolio, Holding, RiskMetrics, CompanyInfo
import pandas as pd

print("=" * 70)
print("EXPORTING DATA FOR TABLEAU - PROFESSIONAL DASHBOARD")
print("=" * 70)

output_dir = Path("tableau_data")
output_dir.mkdir(exist_ok=True)

db = SessionLocal()

# ── 1. Portfolio Overview ────────────────────────────────────
print("\n1. Exporting portfolio overview...")
portfolios = db.query(Portfolio).all()

portfolio_data = []
for p in portfolios:
    portfolio_data.append({
        'Portfolio ID':        p.id,
        'Portfolio Name':      p.name,
        'Total Value':         p.total_value,
        'ESG Score':           p.esg_score_overall,
        'ESG Rating':          p.esg_rating,
        'Environmental Score': p.environmental_score,
        'Social Score':        p.social_score,
        'Governance Score':    p.governance_score,
        'Carbon Intensity':    p.carbon_intensity,
        'Carbon Footprint':    p.carbon_footprint,
        'Created Date':        p.created_at,
        'Last Updated':        p.updated_at,
    })

portfolio_df = pd.DataFrame(portfolio_data)
portfolio_df.to_csv(output_dir / "portfolio_overview.csv", index=False)
print(f"   ✓ portfolio_overview.csv ({len(portfolio_df)} rows)")

# ── 2. Holdings ──────────────────────────────────────────────
print("\n2. Exporting holdings...")
holdings = db.query(Holding).all()

holdings_data = []
for h in holdings:
    curr_price = h.current_price or h.purchase_price
    val = h.quantity * curr_price
    holdings_data.append({
        'Holding ID':       h.id,
        'Portfolio ID':     h.portfolio_id,
        'Ticker':           h.ticker,
        'Asset Type':       h.asset_type,
        'Quantity':         h.quantity,
        'Purchase Price':   h.purchase_price,
        'Current Price':    curr_price,
        'Purchase Date':    h.purchase_date,
        'Current Value':    round(val, 2),
        'Carbon Emissions': h.carbon_emissions,
    })

holdings_df = pd.DataFrame(holdings_data)
total_val = holdings_df['Current Value'].sum()
holdings_df['Weight Pct'] = (holdings_df['Current Value'] / total_val * 100).round(4)
holdings_df['PnL']        = (holdings_df['Current Price'] - holdings_df['Purchase Price']).round(2)
holdings_df['PnL Pct']    = ((holdings_df['PnL'] / holdings_df['Purchase Price']) * 100).round(4)

# Pull ESG scores from company data
company_esg_temp = pd.DataFrame([{
    'Ticker':      c.ticker,
    'ESG Score':   float(c.esg_score) if c.esg_score else 0.0,
    'ESG Rating':  c.esg_rating or 'N/A',
    'Sector':      c.sector or 'Unknown',
} for c in db.query(CompanyInfo).all()])
holdings_df = holdings_df.merge(company_esg_temp, on='Ticker', how='left')

holdings_df.to_csv(output_dir / "holdings.csv", index=False)
print(f"   ✓ holdings.csv ({len(holdings_df)} rows)")

# ── 3. Risk Metrics ──────────────────────────────────────────
print("\n3. Exporting risk metrics...")
risks = db.query(RiskMetrics).all()

risk_data = []
for r in risks:
    risk_data.append({
        'Risk ID':            r.id,
        'Portfolio ID':       r.portfolio_id,
        'Calculation Date':   r.calculation_date,
        'Daily VaR 95%':      r.var_95_daily,
        'Monthly VaR 95%':    r.var_95_monthly,
        'Daily VaR 99%':      r.var_99_daily,
        'CVaR':               r.cvar_95,
        'Sharpe Ratio':       r.sharpe_ratio,
        'Sortino Ratio':      r.sortino_ratio,
        'Max Drawdown':       r.max_drawdown,
        'Volatility':         r.volatility,
        'Beta':               r.beta,
        'Alpha':              r.alpha,
        'ESG Risk Score':     r.esg_risk_score,
        'Environmental Risk': r.environmental_risk,
        'Social Risk':        r.social_risk,
        'Governance Risk':    r.governance_risk,
    })

risk_df = pd.DataFrame(risk_data)
risk_df.to_csv(output_dir / "risk_metrics.csv", index=False)
print(f"   ✓ risk_metrics.csv ({len(risk_df)} rows)")

# ── 4. Company ESG Data ──────────────────────────────────────
print("\n4. Exporting company ESG data...")
companies = db.query(CompanyInfo).all()

company_data = []
for c in companies:
    company_data.append({
        'Ticker':                c.ticker,
        'Company Name':          c.company_name,
        'Sector':                c.sector,
        'Industry':              c.industry,
        'Market Cap':            c.market_cap,
        'Market Cap B':          round((c.market_cap or 0) / 1e9, 2),
        'P/E Ratio':             c.pe_ratio,
        'Beta':                  c.beta,
        'ESG Score':             c.esg_score,
        'ESG Rating':            c.esg_rating,
        'Environmental Score':   c.environmental_score,
        'Social Score':          c.social_score,
        'Governance Score':      c.governance_score,
        'Carbon Emissions':      c.carbon_emissions,
        'Carbon Intensity':      c.carbon_intensity,
        'Renewable Energy %':    c.renewable_energy_pct,
        'Employee Satisfaction': c.employee_satisfaction,
        'Diversity Score':       c.diversity_score,
        'Board Independence':    c.board_independence,
        'Controversies':         c.esg_controversies,
    })

company_df = pd.DataFrame(company_data)
company_df.to_csv(output_dir / "company_esg.csv", index=False)
print(f"   ✓ company_esg.csv ({len(company_df)} rows)")

# ── 5. Summary Statistics ────────────────────────────────────
print("\n5. Creating summary statistics...")
summary_df = pd.DataFrame({
    'Metric': [
        'Total Portfolios', 'Total Holdings', 'Total Value',
        'Average ESG Score', 'Average Sharpe Ratio', 'Total Companies Tracked'
    ],
    'Value': [
        len(portfolio_df),
        len(holdings_df),
        f"${holdings_df['Current Value'].sum():,.0f}",
        f"{portfolio_df['ESG Score'].mean():.1f}",
        f"{risk_df['Sharpe Ratio'].mean():.2f}" if len(risk_df) > 0 else "N/A",
        len(company_df),
    ]
})
summary_df.to_csv(output_dir / "summary_stats.csv", index=False)
print(f"   ✓ summary_stats.csv")

# ── 6. Risk Benchmark Comparison ─────────────────────────────
print("\n6. Creating risk benchmark comparison...")
r = risk_df.iloc[-1]

benchmark_df = pd.DataFrame([
    {'Metric': 'Sharpe Ratio',    'Your Portfolio': round(r['Sharpe Ratio'], 4),         'SP500 Benchmark': 1.0,  'Higher Is Better': True},
    {'Metric': 'Sortino Ratio',   'Your Portfolio': round(r['Sortino Ratio'], 4),        'SP500 Benchmark': 1.5,  'Higher Is Better': True},
    {'Metric': 'Volatility %',    'Your Portfolio': round(r['Volatility'] * 100, 4),     'SP500 Benchmark': 15.0, 'Higher Is Better': False},
    {'Metric': 'Max Drawdown %',  'Your Portfolio': round(r['Max Drawdown'] * 100, 4),   'SP500 Benchmark': 20.0, 'Higher Is Better': False},
    {'Metric': 'Daily VaR 95% %', 'Your Portfolio': round(r['Daily VaR 95%'] * 100, 4), 'SP500 Benchmark': 1.5,  'Higher Is Better': False},
])
benchmark_df.to_csv(output_dir / "risk_benchmark.csv", index=False)
print(f"   ✓ risk_benchmark.csv")

# ── 7. VaR Distribution Curve ────────────────────────────────
print("\n7. Creating VaR distribution curve...")
vol = r['Volatility']
std = vol / np.sqrt(252)
x   = np.linspace(-0.06, 0.06, 1000)
y   = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x) / std) ** 2)
var_threshold = -r['Daily VaR 95%']

var_dist_df = pd.DataFrame({
    'Return Pct':  np.round(x * 100, 6),
    'Probability': np.round(y, 6),
    'Is Tail':     x < var_threshold,
})
var_dist_df.to_csv(output_dir / "var_distribution.csv", index=False)
print(f"   ✓ var_distribution.csv (1000 points)")

# ── 8. Historical Prices ─────────────────────────────────────
print("\n8. Exporting historical prices...")
try:
    from src.data_pipeline.collector import DataCollector
    collector  = DataCollector()
    tickers    = holdings_df['Ticker'].unique().tolist()
    price_rows = []

    for ticker in tickers:
        data = collector.get_stock_data(ticker, '6mo')
        if data is not None:
            start = float(data['Close'].iloc[0])
            for date, row in data.iterrows():
                price_rows.append({
                    'Date':          date.strftime('%Y-%m-%d'),
                    'Ticker':        ticker,
                    'Close':         round(float(row['Close']), 2),
                    'Volume':        int(row['Volume']),
                    'Normalized':    round(float(row['Close']) / start * 100, 4),
                    'Daily Ret Pct': round((float(row['Close']) - float(row['Open'])) / float(row['Open']) * 100, 4),
                })

    price_df = pd.DataFrame(price_rows)
    price_df.to_csv(output_dir / "historical_prices.csv", index=False)
    print(f"   ✓ historical_prices.csv ({len(price_rows)} rows)")
except Exception as e:
    print(f"   ⚠ Skipped historical prices: {e}")

db.close()

# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("✅ TABLEAU EXPORT COMPLETE!")
print("=" * 70)
print(f"\n📊 Files in 'tableau_data/' folder:")
for f in sorted(output_dir.glob("*.csv")):
    print(f"   📄 {f.name:<35} {f.stat().st_size:>8,} bytes")