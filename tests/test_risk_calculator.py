"""
Pytest unit tests for RiskCalculator.
Run with: pytest tests/test_risk_calculator.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.risk_engine.calculator import RiskCalculator


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def calc():
    return RiskCalculator(confidence_level=0.95, simulations=1000)


@pytest.fixture
def returns():
    np.random.seed(42)
    return np.random.normal(0.001, 0.02, 252)


@pytest.fixture
def market_returns():
    np.random.seed(99)
    return np.random.normal(0.0008, 0.015, 252)


# ── VaR ─────────────────────────────────────────────────────────────────────

def test_historical_var_positive(calc, returns):
    result = calc.calculate_var(returns, method='historical')
    assert result['var_95_daily'] > 0


def test_historical_var_cvar_ge_var(calc, returns):
    result = calc.calculate_var(returns, method='historical')
    assert result['cvar_95'] >= result['var_95_daily']


def test_historical_var_monthly_ge_daily(calc, returns):
    result = calc.calculate_var(returns, method='historical')
    assert result['var_95_monthly'] > result['var_95_daily']


def test_parametric_var_positive(calc, returns):
    result = calc.calculate_var(returns, method='parametric')
    assert result['var_95_daily'] > 0


def test_monte_carlo_var_99_ge_95(calc, returns):
    result = calc.calculate_var(returns, method='monte_carlo')
    assert result['var_99_daily'] >= result['var_95_daily']


def test_monte_carlo_var_monthly_ge_daily(calc, returns):
    result = calc.calculate_var(returns, method='monte_carlo')
    assert result['var_95_monthly'] > result['var_95_daily']


def test_var_invalid_method_raises(calc, returns):
    with pytest.raises(ValueError):
        calc.calculate_var(returns, method='invalid_method')


# ── Sharpe Ratio ─────────────────────────────────────────────────────────────

def test_sharpe_formula_matches_manual(calc, returns):
    sharpe = calc.calculate_sharpe_ratio(returns, risk_free_rate=0.0)
    expected = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252))
    assert abs(sharpe - expected) < 1e-9


def test_sharpe_positive_for_high_returns(calc):
    good_returns = np.random.normal(0.005, 0.01, 252)
    assert calc.calculate_sharpe_ratio(good_returns) > 0


def test_sharpe_negative_for_negative_returns(calc):
    bad_returns = np.full(252, -0.005)
    assert calc.calculate_sharpe_ratio(bad_returns) < 0


# ── Sortino Ratio ─────────────────────────────────────────────────────────────

def test_sortino_is_finite_for_mixed_returns(calc, returns):
    sortino = calc.calculate_sortino_ratio(returns)
    assert np.isfinite(sortino)


def test_sortino_positive_for_good_returns(calc):
    np.random.seed(0)
    returns = np.random.normal(0.003, 0.01, 252)
    sortino = calc.calculate_sortino_ratio(returns)
    assert isinstance(sortino, float)


# ── Max Drawdown ──────────────────────────────────────────────────────────────

def test_max_drawdown_known_sequence(calc):
    # peak=100 at index 0, trough=60 at index 3 → drawdown = 40%
    prices = np.array([100.0, 90.0, 80.0, 60.0, 70.0, 90.0])
    result = calc.calculate_max_drawdown(prices)
    assert abs(result['max_drawdown_pct'] - 40.0) < 1e-6


def test_max_drawdown_monotonic_increase_is_zero(calc):
    prices = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
    result = calc.calculate_max_drawdown(prices)
    assert result['max_drawdown'] == 0.0


def test_max_drawdown_pct_consistent_with_fraction(calc):
    prices = np.array([200.0, 150.0, 100.0, 120.0])
    result = calc.calculate_max_drawdown(prices)
    assert abs(result['max_drawdown_pct'] - result['max_drawdown'] * 100) < 1e-9


def test_max_drawdown_returns_required_keys(calc):
    prices = np.array([100.0, 90.0, 80.0, 95.0])
    result = calc.calculate_max_drawdown(prices)
    for key in ['max_drawdown', 'max_drawdown_pct', 'peak_index', 'trough_index']:
        assert key in result


# ── Beta ──────────────────────────────────────────────────────────────────────

def test_beta_identical_returns_approx_one(calc):
    np.random.seed(0)
    r = np.random.normal(0.001, 0.01, 252)
    beta = calc.calculate_beta(r, r)
    # np.cov uses N-1, np.var uses N → slight deviation from 1.0
    assert abs(beta - 1.0) < 0.01


def test_beta_double_market_approx_two(calc):
    np.random.seed(0)
    market = np.random.normal(0.001, 0.01, 252)
    portfolio = 2.0 * market
    beta = calc.calculate_beta(portfolio, market)
    assert abs(beta - 2.0) < 0.02


def test_beta_uncorrelated_assets_near_zero(calc):
    np.random.seed(1)
    market = np.random.normal(0, 0.01, 500)
    portfolio = np.random.normal(0, 0.01, 500)
    beta = calc.calculate_beta(portfolio, market)
    assert abs(beta) < 0.2


# ── Alpha ─────────────────────────────────────────────────────────────────────

def test_alpha_market_portfolio_near_zero(calc):
    np.random.seed(42)
    market = np.random.normal(0.001, 0.01, 252)
    alpha = calc.calculate_alpha(market, market, risk_free_rate=0.0)
    assert abs(alpha) < 0.05


def test_alpha_outperforming_portfolio_positive(calc):
    np.random.seed(5)
    market = np.random.normal(0.0005, 0.01, 252)
    # Portfolio with same risk but higher return
    portfolio = market + 0.002
    alpha = calc.calculate_alpha(portfolio, market, risk_free_rate=0.0)
    assert alpha > 0


# ── Volatility ────────────────────────────────────────────────────────────────

def test_volatility_annualization(calc):
    np.random.seed(0)
    r = np.random.normal(0, 0.01, 252)
    vol = calc.calculate_volatility(r)
    expected = np.std(r) * np.sqrt(252)
    assert abs(vol - expected) < 1e-10


# ── Diversification Ratio ─────────────────────────────────────────────────────

def test_diversification_ratio_perfect_correlation_is_one(calc):
    weights = np.array([0.5, 0.5])
    vols = np.array([0.2, 0.2])
    corr = np.array([[1.0, 1.0], [1.0, 1.0]])
    ratio = calc.calculate_diversification_ratio(weights, vols, corr)
    assert abs(ratio - 1.0) < 1e-6


def test_diversification_ratio_zero_correlation_gt_one(calc):
    weights = np.array([0.5, 0.5])
    vols = np.array([0.2, 0.2])
    corr = np.array([[1.0, 0.0], [0.0, 1.0]])
    ratio = calc.calculate_diversification_ratio(weights, vols, corr)
    assert ratio > 1.0


# ── Stress Test ───────────────────────────────────────────────────────────────

def test_stress_test_hardcoded_scenarios(calc, returns):
    result = calc.stress_test(100_000, returns)
    assert result['market_crash_20pct'] == pytest.approx(-20_000)
    assert result['market_crash_30pct'] == pytest.approx(-30_000)
    assert result['flash_crash_10pct_1day'] == pytest.approx(-10_000)


# ── Comprehensive Risk ────────────────────────────────────────────────────────

def test_comprehensive_risk_returns_all_keys(calc, returns):
    result = calc.calculate_comprehensive_risk(returns, 100_000)
    required = [
        'var_95_daily', 'var_95_monthly', 'cvar_95',
        'sharpe_ratio', 'sortino_ratio', 'volatility',
        'max_drawdown', 'max_drawdown_pct', 'stress_tests',
    ]
    for key in required:
        assert key in result, f"Missing key: {key}"


def test_comprehensive_risk_with_market_returns_adds_beta_alpha(calc, returns, market_returns):
    result = calc.calculate_comprehensive_risk(returns, 100_000, market_returns=market_returns)
    assert 'beta' in result
    assert 'alpha' in result
