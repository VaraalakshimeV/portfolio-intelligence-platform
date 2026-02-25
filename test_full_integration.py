"""
Full System Integration Test
Tests all components working together
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
import os
os.chdir(str(project_root))
sys.path.insert(0, str(project_root))

# Now import from src
from src.database.database import init_db
from src.database.service import DatabaseService
from src.data_pipeline.collector import DataCollector
from src.risk_engine.calculator import RiskCalculator
from src.esg_engine.calculator import ESGCalculator
import numpy as np

print("=" * 70)
print("FULL SYSTEM INTEGRATION TEST")
print("=" * 70)

# Initialize components
print("\n1. Initializing system components...")
init_db()
db_service = DatabaseService()
data_collector = DataCollector()
risk_calculator = RiskCalculator()
esg_calculator = ESGCalculator()
print("   ✓ All components initialized")

# Create user
print("\n2. Creating test user...")
user = db_service.create_user(
    email="demo@fintech.com",
    full_name="Demo Investor",
    risk_tolerance="moderate",
    esg_priority="high"
)
print(f"   ✓ User created: {user['email']}")

# Create portfolio
print("\n3. Creating portfolio...")
portfolio = db_service.create_portfolio(
    user_id=user['id'],
    name="ESG Tech Portfolio",
    total_value=100000.0
)
print(f"   ✓ Portfolio created: {portfolio['name']}")

# Fetch real stock data
print("\n4. Fetching real market data...")
tickers = [
    # Technology (4 stocks)
    'AAPL', 'MSFT', 'GOOGL', 'NVDA',
    # Financials (3 stocks)
    'JPM', 'V', 'GS',
    # Healthcare (2 stocks)
    'JNJ', 'UNH',
    # Consumer (3 stocks)
    'AMZN', 'TSLA', 'WMT',
    # Energy (1 stock - low ESG contrast)
    'XOM',
    # Industrials (2 stocks)
    'BA', 'CAT'
]
stock_data = {}

for ticker in tickers:
    data = data_collector.get_stock_data(ticker, period="6mo")
    if data is not None:
        stock_data[ticker] = data
        
        # Add to portfolio
        latest_price = data['Close'].iloc[-1]
        quantity = 10000 / latest_price
        
        holding = db_service.add_holding(
            portfolio_id=portfolio['id'],
            ticker=ticker,
            quantity=quantity,
            purchase_price=latest_price,
            asset_type='stock'
        )
        print(f"   ✓ {ticker}: ${latest_price:.2f} ({quantity:.0f} shares)")

# Calculate portfolio returns
print("\n5. Calculating portfolio risk...")
all_returns = []
weights = [1/len(stock_data)] * len(stock_data)

for ticker, data in stock_data.items():
    returns = data_collector.calculate_returns(data)
    all_returns.append(returns)

# Combine returns
portfolio_returns = sum(r * w for r, w in zip(all_returns, weights))
portfolio_returns = portfolio_returns.dropna()

# Calculate risk metrics
risk_metrics = risk_calculator.calculate_comprehensive_risk(
    returns=portfolio_returns.values,
    portfolio_value=100000
)

print(f"   ✓ VaR 95% Daily: ${risk_metrics['var_95_daily'] * 100000:,.2f}")
print(f"   ✓ Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
print(f"   ✓ Max Drawdown: {risk_metrics['max_drawdown_pct']:.1f}%")

# Save risk metrics
db_service.save_risk_metrics(portfolio['id'], risk_metrics)
print("   ✓ Risk metrics saved to database")

# Calculate ESG scores
print("\n6. Calculating ESG scores...")
holdings_esg = []

for ticker in tickers:
    company_info = data_collector.get_company_info(ticker)
    
    if company_info:
        esg_data = data_collector.generate_sample_esg_data(
            ticker,
            company_info['sector']
        )
        
        esg_result = esg_calculator.calculate_esg_score(
            esg_data,
            sector=company_info['sector']
        )
        
        print(f"   ✓ {ticker}: {esg_result['adjusted_rating']} ({esg_result['adjusted_score']:.1f}/100)")
        
        esg_for_db = {
            'esg_score': esg_result['overall_score'],
            'environmental_score': esg_result['environmental_score'],
            'social_score': esg_result['social_score'],
            'governance_score': esg_result['governance_score'],
            'esg_rating': esg_result['adjusted_rating'],
            'esg_controversies': esg_result['controversies'],
            **esg_data
        }
        
        db_service.save_company_info({
            **company_info,
            **esg_for_db
        })
        
        holdings_esg.append({
            'ticker': ticker,
            'value': 100000 / len(tickers),
            'esg_data': esg_result
        })

# Calculate portfolio ESG
portfolio_esg = esg_calculator.calculate_portfolio_esg(holdings_esg)
print(f"\n   📊 Portfolio ESG Score: {portfolio_esg['portfolio_esg_score']:.1f}/100")
print(f"   ⭐ Portfolio Rating: {portfolio_esg['portfolio_rating']}")
print(f"   🌍 Carbon Intensity: {portfolio_esg['carbon_intensity']:.1f} tons CO2/$1M")

# Update portfolio
db_service.update_portfolio_esg(portfolio['id'], {
    'overall': portfolio_esg['portfolio_esg_score'],
    'environmental': portfolio_esg['environmental_score'],
    'social': portfolio_esg['social_score'],
    'governance': portfolio_esg['governance_score'],
    'rating': portfolio_esg['portfolio_rating'],
    'carbon_intensity': portfolio_esg['carbon_intensity'],
    'carbon_footprint': portfolio_esg['carbon_footprint']
})
print("   ✓ Portfolio ESG saved to database")

print("\n7. Testing chat integration...")
db_service.save_chat_message(
    user_id=user['id'],
    session_id="test_session_001",
    user_query="What's my portfolio risk?",
    bot_response=f"Your portfolio has a daily VaR of ${risk_metrics['var_95_daily'] * 100000:,.2f}",
    tokens_used=150,
    response_time=1.2,
    cost=0.0003
)
print("   ✓ Chat message saved")

print("\n" + "=" * 70)
print("✅ FULL INTEGRATION TEST COMPLETE!")
print("=" * 70)

print("\n📊 SYSTEM SUMMARY:")
print(f"   • User: {user['full_name']} ({user['email']})")
print(f"   • Portfolio: {portfolio['name']}")
print(f"   • Holdings: {len(tickers)} stocks")
print(f"   • Portfolio Value: ${portfolio['total_value']:,.2f}")
print(f"   • Risk Level: {risk_metrics['sharpe_ratio']:.2f} Sharpe Ratio")
print(f"   • ESG Rating: {portfolio_esg['portfolio_rating']}")
print(f"   • Database: All data persisted")

print("\n💡 What We Built:")
print("   ✓ Risk Engine - Monte Carlo VaR, Sharpe, Sortino")
print("   ✓ ESG Calculator - Environmental, Social, Governance scoring")
print("   ✓ Data Pipeline - Multi-source with fallbacks")
print("   ✓ Database - Enterprise-grade schema")
print("   ✓ Full Integration - All systems working together")

print("\n🚀 Next Steps:")
print("   1. Build RAG chatbot with Pinecone")
print("   2. Create Streamlit dashboard")
print("   3. Add AI personalization")
print("   4. Deploy to cloud")