"""
Test Risk Calculator
"""

import numpy as np
from src.risk_engine.calculator import RiskCalculator

print("=" * 70)
print("TESTING RISK CALCULATOR")
print("=" * 70)

# Generate sample portfolio data
np.random.seed(42)
returns = np.random.normal(0.0008, 0.015, 252)  # 1 year daily returns
portfolio_value = 100000

print("\n1. Creating risk calculator...")
calculator = RiskCalculator(confidence_level=0.95, simulations=10000)
print("   ✓ Risk calculator created")

print("\n2. Calculating comprehensive risk metrics...")
risk_metrics = calculator.calculate_comprehensive_risk(
    returns=returns,
    portfolio_value=portfolio_value
)
print("   ✓ Risk calculation complete")

print("\n" + "=" * 70)
print("PORTFOLIO RISK ANALYSIS RESULTS")
print("=" * 70)

print(f"\n📊 Portfolio Value: ${portfolio_value:,.2f}")

print(f"\n⚠️ Value at Risk (VaR) - 95% Confidence:")
print(f"   Daily VaR: ${risk_metrics['var_95_daily'] * portfolio_value:,.2f}")
print(f"   Monthly VaR: ${risk_metrics['var_95_monthly'] * portfolio_value:,.2f}")
print(f"   CVaR (Expected Shortfall): ${risk_metrics['cvar_95'] * portfolio_value:,.2f}")

print(f"\n📈 Performance Metrics:")
print(f"   Sharpe Ratio: {risk_metrics['sharpe_ratio']:.3f}")
print(f"   Sortino Ratio: {risk_metrics['sortino_ratio']:.3f}")
print(f"   Volatility (Annual): {risk_metrics['volatility']*100:.2f}%")
print(f"   Max Drawdown: {risk_metrics['max_drawdown_pct']:.2f}%")

print(f"\n🔥 Stress Test Scenarios:")
for scenario, loss in risk_metrics['stress_tests'].items():
    print(f"   {scenario.replace('_', ' ').title()}: ${loss:,.2f}")

print("\n" + "=" * 70)
print("✅ RISK CALCULATOR TEST COMPLETE!")
print("=" * 70)

print("\n💡 What this means:")
print("   • Your portfolio could lose up to ${:,.2f} in a single day (95% confidence)".format(
    risk_metrics['var_95_daily'] * portfolio_value
))
print("   • Sharpe Ratio of {:.2f} indicates {} risk-adjusted returns".format(
    risk_metrics['sharpe_ratio'],
    "good" if risk_metrics['sharpe_ratio'] > 1 else "moderate"
))
print("   • Maximum historical drawdown: {:.1f}%".format(risk_metrics['max_drawdown_pct']))