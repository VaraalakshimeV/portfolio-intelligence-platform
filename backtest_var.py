"""
VaR Backtesting Suite — Kupiec Proportion of Failures (POF) Test
Validates whether the VaR model correctly predicts tail losses.

Interpretation:
  p_value > 0.05 → model is well-calibrated (fail to reject H0)
  actual_rate ≈ 5% → VaR correctly predicts tail losses at 95% confidence
"""

import sys
import json
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from src.risk_engine.calculator import RiskCalculator
from src.data_pipeline.collector import DataCollector

LOOKBACK_DAYS = 252
CONFIDENCE_LEVEL = 0.95
BACKTEST_TICKERS = ['SPY', 'QQQ', 'IWM']


def kupiec_test(n_violations: int, n_total: int, confidence_level: float) -> dict:
    """
    Kupiec Proportion of Failures (POF) test.
    H0: violation rate == 1 - confidence_level (e.g. 5% for 95% VaR)
    Under H0: LR statistic ~ chi2(1)
    """
    expected_rate = 1 - confidence_level
    actual_rate = n_violations / n_total if n_total > 0 else 0.0

    if n_violations == 0:
        lr_stat = -2.0 * n_total * np.log(1 - expected_rate)
    elif n_violations == n_total:
        lr_stat = -2.0 * n_total * np.log(expected_rate)
    else:
        lr_stat = -2.0 * (
            n_violations * np.log(expected_rate / actual_rate) +
            (n_total - n_violations) * np.log((1 - expected_rate) / (1 - actual_rate))
        )

    p_value = float(1 - stats.chi2.cdf(lr_stat, df=1))

    return {
        'n_violations': n_violations,
        'n_total': n_total,
        'expected_rate': round(expected_rate, 4),
        'actual_rate': round(actual_rate, 4),
        'lr_statistic': round(lr_stat, 4),
        'p_value': round(p_value, 4),
        'calibrated': p_value > 0.05,
    }


def backtest_ticker(ticker: str, collector: DataCollector, risk_calc: RiskCalculator) -> dict | None:
    print(f"\nBacktesting {ticker}...")

    data = collector.get_stock_data(ticker, period='5y')
    if data is None:
        print(f"  ! Could not fetch data for {ticker}")
        return None

    returns = collector.calculate_returns(data).dropna().values

    if len(returns) < LOOKBACK_DAYS + 50:
        print(f"  ! Not enough data for {ticker}")
        return None

    violations_hist, violations_param, violations_mc = [], [], []

    for i in range(LOOKBACK_DAYS, len(returns) - 1):
        window = returns[i - LOOKBACK_DAYS:i]
        actual_next = returns[i + 1]

        var_h  = risk_calc.calculate_var(window, method='historical')['var_95_daily']
        var_p  = risk_calc.calculate_var(window, method='parametric')['var_95_daily']
        var_mc = risk_calc.calculate_var(window, method='monte_carlo')['var_95_daily']

        violations_hist.append(actual_next < -var_h)
        violations_param.append(actual_next < -var_p)
        violations_mc.append(actual_next < -var_mc)

    n_test = len(violations_hist)
    results = {
        'ticker': ticker,
        'n_test_days': n_test,
        'historical_var':  kupiec_test(sum(violations_hist),  n_test, CONFIDENCE_LEVEL),
        'parametric_var':  kupiec_test(sum(violations_param), n_test, CONFIDENCE_LEVEL),
        'monte_carlo_var': kupiec_test(sum(violations_mc),    n_test, CONFIDENCE_LEVEL),
    }

    print(f"  Test days: {n_test}")
    for method in ['historical_var', 'parametric_var', 'monte_carlo_var']:
        r = results[method]
        status = "PASS" if r['calibrated'] else "FAIL"
        print(
            f"  {method:<18}: violations {r['n_violations']:>3}/{r['n_total']} "
            f"({r['actual_rate']*100:.1f}% vs expected {r['expected_rate']*100:.0f}%) "
            f"p={r['p_value']:.3f}  [{status}]"
        )

    return results


if __name__ == "__main__":
    print("=" * 72)
    print("VAR BACKTESTING SUITE — KUPIEC PROPORTION OF FAILURES TEST")
    print("=" * 72)
    print(f"  Confidence level : {CONFIDENCE_LEVEL*100:.0f}%")
    print(f"  Lookback window  : {LOOKBACK_DAYS} trading days (~1 year)")
    print(f"  Expected violation rate: {(1-CONFIDENCE_LEVEL)*100:.0f}% per day")

    collector = DataCollector()
    # Use fewer MC simulations per day to keep runtime reasonable
    risk_calc = RiskCalculator(confidence_level=CONFIDENCE_LEVEL, simulations=500)

    all_results = []
    for ticker in BACKTEST_TICKERS:
        result = backtest_ticker(ticker, collector, risk_calc)
        if result:
            all_results.append(result)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for method in ['historical_var', 'parametric_var', 'monte_carlo_var']:
        passes = sum(1 for r in all_results if r[method]['calibrated'])
        print(f"  {method:<18}: {passes}/{len(all_results)} tickers well-calibrated")

    out_path = Path(__file__).parent / 'var_backtest_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_path.name}")

    print("\nINTERPRETATION:")
    print("  PASS (p > 0.05) → violation rate is statistically consistent with 5%")
    print("  FAIL (p < 0.05) → model over- or under-estimates tail risk")
    print("  actual_rate close to 5% → VaR is well-calibrated")
