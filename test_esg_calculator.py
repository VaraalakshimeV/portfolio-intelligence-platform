"""
Test ESG Calculator
"""

from src.esg_engine.calculator import ESGCalculator

print("=" * 70)
print("TESTING ESG CALCULATOR")
print("=" * 70)

# Sample company data (Tech company with good ESG practices)
tech_company = {
    'carbon_intensity': 45,
    'renewable_energy_pct': 75,
    'water_usage': 400,
    'waste_recycling_pct': 80,
    'environmental_innovations': 8,
    'employee_satisfaction': 82,
    'diversity_score': 70,
    'female_employees_pct': 48,
    'employee_turnover_rate': 8,
    'training_hours_per_employee': 50,
    'community_investment': 8000000,
    'labor_practices_score': 85,
    'human_rights_score': 90,
    'board_independence': 75,
    'board_diversity': 65,
    'female_board_members': 40,
    'executive_compensation_ratio': 95,
    'shareholder_rights_score': 80,
    'anti_corruption_score': 88,
    'tax_transparency_score': 75,
    'esg_controversies': 0
}

print("\n1. Creating ESG calculator...")
calculator = ESGCalculator()
print("   ✓ ESG calculator created")

print("\n2. Calculating ESG scores...")
result = calculator.calculate_esg_score(tech_company, sector='Technology')
print("   ✓ ESG calculation complete")

print("\n" + "=" * 70)
print("ESG ANALYSIS RESULTS")
print("=" * 70)

print(f"\n🌍 Environmental Score: {result['environmental_score']}/100")
print(f"👥 Social Score: {result['social_score']}/100")
print(f"🏛️  Governance Score: {result['governance_score']}/100")

print(f"\n📊 Overall ESG Score: {result['adjusted_score']}/100")
print(f"⭐ ESG Rating: {result['adjusted_rating']}")
print(f"🏢 Sector: {result['sector']}")

print(f"\n⚠️  Controversies: {result['controversies']}")
if result['controversy_penalty'] > 0:
    print(f"   Penalty Applied: -{result['controversy_penalty']} points")

print(f"\n📈 Sector Weights (Technology):")
print(f"   Environmental: {result['weights']['E']*100:.0f}%")
print(f"   Social: {result['weights']['S']*100:.0f}%")
print(f"   Governance: {result['weights']['G']*100:.0f}%")

print("\n3. Calculating ESG Risk...")
risk = calculator.calculate_esg_risk(result['adjusted_score'], result['controversies'])
print("   ✓ Risk calculation complete")

print(f"\n🔥 ESG Risk Assessment:")
print(f"   Overall Risk Score: {risk['esg_risk_score']}/100")
print(f"   Risk Level: {risk['risk_level']}")
print(f"   Environmental Risk: {risk['environmental_risk']}/100")
print(f"   Social Risk: {risk['social_risk']}/100")
print(f"   Governance Risk: {risk['governance_risk']}/100")

print("\n4. Testing Portfolio ESG...")
portfolio = [
    {
        'ticker': 'TECH1',
        'value': 50000,
        'esg_data': result
    },
    {
        'ticker': 'TECH2',
        'value': 30000,
        'esg_data': {
            'environmental_score': 65,
            'social_score': 70,
            'governance_score': 75,
            'carbon_emissions': 1000
        }
    },
    {
        'ticker': 'TECH3',
        'value': 20000,
        'esg_data': {
            'environmental_score': 80,
            'social_score': 75,
            'governance_score': 85,
            'carbon_emissions': 500
        }
    }
]

portfolio_esg = calculator.calculate_portfolio_esg(portfolio)

print(f"\n💼 Portfolio ESG Metrics:")
print(f"   Portfolio Value: ${portfolio_esg['total_value']:,.2f}")
print(f"   Holdings: {portfolio_esg['holdings_count']}")
print(f"   Portfolio ESG Score: {portfolio_esg['portfolio_esg_score']}/100")
print(f"   Portfolio Rating: {portfolio_esg['portfolio_rating']}")
print(f"   Carbon Intensity: {portfolio_esg['carbon_intensity']:.2f} tons CO2/$1M")

print("\n" + "=" * 70)
print("✅ ESG CALCULATOR TEST COMPLETE!")
print("=" * 70)

print("\n💡 What this means:")
print(f"   • Company has '{result['adjusted_rating']}' ESG rating (industry-adjusted)")
print(f"   • {result['environmental_score']:.0f}/100 environmental performance")
print(f"   • {risk['risk_level']} ESG-related risk level")
print(f"   • Portfolio carbon footprint: {portfolio_esg['carbon_footprint']:.0f} tons CO2")