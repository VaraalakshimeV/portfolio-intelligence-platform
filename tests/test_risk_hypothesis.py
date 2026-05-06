"""
Hypothesis property-based tests for RiskCalculator.
Verifies mathematical invariants that must hold for ANY valid input —
catches edge cases fixed-seed unit tests can miss.

Run with: pytest tests/test_risk_hypothesis.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from src.risk_engine.calculator import RiskCalculator

CALC = RiskCalculator(confidence_level=0.95, simulations=500)

# ── Shared strategies ────────────────────────────────────────────────────────

# Realistic daily returns: ±20% max
returns_st = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=50, max_value=300),
    elements=st.floats(min_value=-0.20, max_value=0.20,
                       allow_nan=False, allow_infinity=False),
)

# Positive price series
prices_st = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=5, max_value=200),
    elements=st.floats(min_value=1.0, max_value=1e6,
                       allow_nan=False, allow_infinity=False),
)

_NO_SLOW = [HealthCheck.too_slow]


# ── VaR invariants ───────────────────────────────────────────────────────────

@given(returns=returns_st)
@settings(max_examples=100, suppress_health_check=_NO_SLOW)
def test_var_always_non_negative(returns):
    """VaR is a loss measure — daily, monthly, and CVaR must all be >= 0."""
    r = CALC.calculate_var(returns, method='historical')
    assert r['var_95_daily'] >= 0
    assert r['var_95_monthly'] >= 0
    assert r['cvar_95'] >= 0


@given(returns=returns_st)
@settings(max_examples=100, suppress_health_check=_NO_SLOW)
def test_cvar_always_ge_var(returns):
    """CVaR (expected shortfall) is the average of losses beyond VaR — must be >= VaR."""
    r = CALC.calculate_var(returns, method='historical')
    # Allow 1e-12 tolerance: when tail is all identical values, floating-point
    # mean may differ from the percentile value by a few ULPs.
    assert r['cvar_95'] >= r['var_95_daily'] - 1e-12


@given(returns=returns_st)
@settings(max_examples=100, suppress_health_check=_NO_SLOW)
def test_monthly_var_ge_daily_var(returns):
    """Monthly VaR = daily × sqrt(21) — must always exceed daily VaR."""
    for method in ('historical', 'parametric'):
        r = CALC.calculate_var(returns, method=method)
        assert r['var_95_monthly'] >= r['var_95_daily']


@given(returns=returns_st)
@settings(max_examples=50, suppress_health_check=_NO_SLOW)
def test_parametric_var_non_negative(returns):
    assume(np.std(returns) > 1e-10)
    r = CALC.calculate_var(returns, method='parametric')
    assert r['var_95_daily'] >= 0
    assert r['cvar_95'] >= r['var_95_daily']


# ── Sharpe invariants ────────────────────────────────────────────────────────

@given(returns=returns_st, delta=st.floats(min_value=1e-4, max_value=0.01))
@settings(max_examples=100, suppress_health_check=_NO_SLOW)
def test_sharpe_increases_with_mean_shift(returns, delta):
    """Adding a positive constant to every return must strictly increase the Sharpe ratio."""
    assume(np.std(returns) > 1e-10)
    sharpe_base = CALC.calculate_sharpe_ratio(returns, risk_free_rate=0.0)
    sharpe_up = CALC.calculate_sharpe_ratio(returns + delta, risk_free_rate=0.0)
    assert sharpe_up > sharpe_base


@given(returns=returns_st, k=st.floats(min_value=0.1, max_value=10.0))
@settings(max_examples=100, suppress_health_check=_NO_SLOW)
def test_sharpe_invariant_to_return_scaling(returns, k):
    """
    When rf=0, Sharpe = mean/std. Scaling returns by k scales both mean and std
    by k, so the ratio is unchanged.
    """
    assume(np.std(returns) > 1e-10)
    sharpe_base = CALC.calculate_sharpe_ratio(returns, risk_free_rate=0.0)
    sharpe_scaled = CALC.calculate_sharpe_ratio(returns * k, risk_free_rate=0.0)
    assert abs(sharpe_scaled - sharpe_base) < 1e-6


# ── Volatility invariants ────────────────────────────────────────────────────

@given(returns=returns_st)
@settings(max_examples=100, suppress_health_check=_NO_SLOW)
def test_volatility_non_negative(returns):
    """Annualized volatility is a standard deviation — always >= 0."""
    assert CALC.calculate_volatility(returns) >= 0


@given(returns=returns_st, k=st.floats(min_value=0.1, max_value=10.0))
@settings(max_examples=100, suppress_health_check=_NO_SLOW)
def test_volatility_scales_linearly(returns, k):
    """Scaling returns by k must scale annualized volatility by exactly k."""
    assume(np.std(returns) > 1e-10)
    vol = CALC.calculate_volatility(returns)
    vol_scaled = CALC.calculate_volatility(returns * k)
    assert abs(vol_scaled - vol * k) < 1e-8


# ── Max Drawdown invariants ──────────────────────────────────────────────────

@given(prices=prices_st)
@settings(max_examples=100, suppress_health_check=_NO_SLOW)
def test_max_drawdown_in_unit_interval(prices):
    """Max drawdown fraction must lie in [0, 1] for any price series."""
    r = CALC.calculate_max_drawdown(prices)
    assert 0.0 <= r['max_drawdown'] <= 1.0


@given(prices=prices_st)
@settings(max_examples=100, suppress_health_check=_NO_SLOW)
def test_max_drawdown_pct_equals_fraction_times_100(prices):
    """max_drawdown_pct must always equal max_drawdown × 100."""
    r = CALC.calculate_max_drawdown(prices)
    assert abs(r['max_drawdown_pct'] - r['max_drawdown'] * 100) < 1e-9


@given(
    base=st.floats(min_value=1.0, max_value=1000.0),
    increments=arrays(
        np.float64,
        st.integers(min_value=4, max_value=100),
        elements=st.floats(min_value=0.001, max_value=10.0,
                           allow_nan=False, allow_infinity=False),
    ),
)
@settings(max_examples=100, suppress_health_check=_NO_SLOW)
def test_monotonic_prices_zero_drawdown(base, increments):
    """A price series that never falls has zero drawdown."""
    prices = np.concatenate([[base], base + np.cumsum(increments)])
    r = CALC.calculate_max_drawdown(prices)
    assert r['max_drawdown'] == pytest.approx(0.0, abs=1e-10)


@given(prices=prices_st)
@settings(max_examples=100, suppress_health_check=_NO_SLOW)
def test_peak_index_le_trough_index(prices):
    """The peak must always occur at or before the trough."""
    r = CALC.calculate_max_drawdown(prices)
    assert r['peak_index'] <= r['trough_index']


# ── Beta invariants ──────────────────────────────────────────────────────────

@given(
    market=arrays(np.float64, st.integers(50, 252),
                  elements=st.floats(-0.05, 0.05, allow_nan=False, allow_infinity=False)),
    k=st.floats(min_value=0.5, max_value=3.0),
)
@settings(max_examples=100, suppress_health_check=_NO_SLOW)
def test_beta_linear_portfolio(market, k):
    """Portfolio = k × market → beta must equal k exactly."""
    assume(np.var(market) > 1e-12)
    beta = CALC.calculate_beta(k * market, market)
    assert abs(beta - k) < 0.01


@given(
    portfolio=arrays(np.float64, 252,
                     elements=st.floats(-0.05, 0.05, allow_nan=False, allow_infinity=False)),
    market=arrays(np.float64, 252,
                  elements=st.floats(-0.05, 0.05, allow_nan=False, allow_infinity=False)),
    c=st.floats(min_value=0.1, max_value=10.0),
)
@settings(max_examples=100, suppress_health_check=_NO_SLOW)
def test_beta_scales_inversely_with_market_scaling(portfolio, market, c):
    """
    cov(p, c·m) / var(c·m) = cov(p,m) / (c·var(m))
    so beta(p, c·m) = beta(p, m) / c.
    """
    assume(np.var(market, ddof=1) > 1e-12)
    beta_orig = CALC.calculate_beta(portfolio, market)
    beta_scaled = CALC.calculate_beta(portfolio, market * c)
    assert abs(beta_scaled - beta_orig / c) < 1e-6


# ── Diversification Ratio invariants ────────────────────────────────────────

@given(
    n=st.integers(min_value=2, max_value=10),
    vol=st.floats(min_value=0.01, max_value=0.5),
)
@settings(max_examples=100, suppress_health_check=_NO_SLOW)
def test_diversification_ratio_perfect_correlation_is_one(n, vol):
    """Perfectly correlated assets provide no diversification benefit → ratio = 1."""
    weights = np.full(n, 1.0 / n)
    vols = np.full(n, vol)
    corr = np.ones((n, n))
    ratio = CALC.calculate_diversification_ratio(weights, vols, corr)
    assert abs(ratio - 1.0) < 1e-6


@given(
    n=st.integers(min_value=2, max_value=8),
    vol=st.floats(min_value=0.01, max_value=0.5),
)
@settings(max_examples=100, suppress_health_check=_NO_SLOW)
def test_diversification_ratio_uncorrelated_gte_one(n, vol):
    """Uncorrelated equal-weight portfolio must always have diversification ratio >= 1."""
    weights = np.full(n, 1.0 / n)
    vols = np.full(n, vol)
    corr = np.eye(n)
    ratio = CALC.calculate_diversification_ratio(weights, vols, corr)
    assert ratio >= 1.0 - 1e-9
