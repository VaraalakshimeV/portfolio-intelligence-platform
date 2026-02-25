"""
Test Data Collector
"""

from src.data_pipeline.collector import DataCollector

print("=" * 70)
print("TESTING DATA COLLECTOR")
print("=" * 70)

print("\n1. Initializing data collector...")
collector = DataCollector()
print("   ✓ Data collector initialized")

print("\n2. Fetching stock data (AAPL)...")
data = collector.get_stock_data("AAPL", period="1mo")
if data is not None:
    print(f"   ✓ Fetched {len(data)} days of data")
    print(f"   Latest close: ${data['Close'].iloc[-1]:.2f}")
    print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
else:
    print("   ✗ Failed to fetch data")

print("\n3. Fetching company information...")
info = collector.get_company_info("AAPL")
if info:
    print(f"   ✓ Company: {info['company_name']}")
    print(f"   Sector: {info['sector']}")
    print(f"   Industry: {info['industry']}")
    print(f"   Market Cap: ${info['market_cap']:,.0f}")
    print(f"   P/E Ratio: {info['pe_ratio']:.2f}" if info['pe_ratio'] else "   P/E Ratio: N/A")
    print(f"   Beta: {info['beta']:.2f}" if info['beta'] else "   Beta: N/A")
else:
    print("   ✗ Failed to fetch company info")

print("\n4. Fetching multiple stocks...")
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
multi_data = collector.get_multiple_stocks(tickers, period="1mo")
print(f"   ✓ Fetched data for {len(multi_data)}/{len(tickers)} stocks")
for ticker in multi_data:
    latest_price = multi_data[ticker]['Close'].iloc[-1]
    print(f"   {ticker}: ${latest_price:.2f}")

print("\n5. Calculating returns...")
if data is not None:
    returns = collector.calculate_returns(data)
    print(f"   ✓ Calculated {len(returns)} daily returns")
    print(f"   Average daily return: {returns.mean()*100:.3f}%")
    print(f"   Volatility (std): {returns.std()*100:.3f}%")
    print(f"   Best day: +{returns.max()*100:.2f}%")
    print(f"   Worst day: {returns.min()*100:.2f}%")

print("\n6. Getting market benchmark (S&P 500)...")
spy_data = collector.get_market_benchmark(period="1mo")
if spy_data is not None:
    print(f"   ✓ Fetched S&P 500 data ({len(spy_data)} days)")
    print(f"   SPY close: ${spy_data['Close'].iloc[-1]:.2f}")

print("\n7. Generating sample ESG data...")
esg_data = collector.generate_sample_esg_data("AAPL", "Technology")
print("   ✓ ESG data generated")
print(f"   Renewable Energy: {esg_data['renewable_energy_pct']:.1f}%")
print(f"   Employee Satisfaction: {esg_data['employee_satisfaction']:.1f}/100")
print(f"   Board Independence: {esg_data['board_independence']:.1f}%")
print(f"   ESG Controversies: {esg_data['esg_controversies']}")

print("\n" + "=" * 70)
print("✅ DATA COLLECTOR TEST COMPLETE!")
print("=" * 70)

print("\n💡 What this demonstrates:")
print("   • Multi-source data pipeline with fallbacks")
print("   • Real-time stock price data")
print("   • Company fundamental metrics")
print("   • ESG data generation (sector-adjusted)")
print("   • Returns calculation for risk analysis")