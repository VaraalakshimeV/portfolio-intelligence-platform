from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.schemas import RiskMetricsOut, VaROut, StressTestOut
from src.api.deps import get_db, get_current_user
from src.database.models import Portfolio, RiskMetrics, Holding
from src.risk_engine.calculator import RiskCalculator

router = APIRouter(prefix="/risk", tags=["risk"])

_VALID_VAR_METHODS = {"historical", "parametric", "monte_carlo"}


def _latest_risk(db: Session) -> RiskMetrics:
    r = db.query(RiskMetrics).order_by(RiskMetrics.calculation_date.desc()).first()
    if not r:
        raise HTTPException(status_code=404, detail="No risk metrics found")
    return r


@router.get("/metrics", response_model=RiskMetricsOut)
def get_risk_metrics(
    db: Session = Depends(get_db),
    _: str = Depends(get_current_user),
):
    r = _latest_risk(db)
    return RiskMetricsOut(
        calculation_date=r.calculation_date,
        var_95_daily=r.var_95_daily,
        var_95_monthly=r.var_95_monthly,
        var_99_daily=r.var_99_daily,
        cvar_95=r.cvar_95,
        sharpe_ratio=r.sharpe_ratio,
        sortino_ratio=r.sortino_ratio,
        volatility=r.volatility,
        max_drawdown=r.max_drawdown,
        max_drawdown_pct=round(r.max_drawdown * 100, 4) if r.max_drawdown else None,
        beta=r.beta,
        alpha=r.alpha,
        diversification_ratio=r.diversification_ratio,
    )


@router.get("/var/{method}", response_model=VaROut)
def get_var(
    method: str,
    db: Session = Depends(get_db),
    _: str = Depends(get_current_user),
):
    if method not in _VALID_VAR_METHODS:
        raise HTTPException(
            status_code=422,
            detail=f"method must be one of {sorted(_VALID_VAR_METHODS)}",
        )
    r = _latest_risk(db)
    p = db.query(Portfolio).first()
    if not p:
        raise HTTPException(status_code=404, detail="No portfolio found")

    # Live VaR calculation using the DB-stored metrics as returns proxy
    calc = RiskCalculator(confidence_level=0.95, simulations=5000)
    import numpy as np
    np.random.seed(42)
    proxy_returns = np.random.normal(
        (r.sharpe_ratio or 1.0) * (r.volatility or 0.12) / 252,
        (r.volatility or 0.12) / 252 ** 0.5,
        252,
    )
    result = calc.calculate_var(proxy_returns, method=method)

    return VaROut(
        method=result["method"],
        var_95_daily=round(result["var_95_daily"] * p.total_value, 2),
        var_95_monthly=round(result["var_95_monthly"] * p.total_value, 2),
        cvar_95=round(result["cvar_95"] * p.total_value, 2),
        var_99_daily=round(result.get("var_99_daily", 0) * p.total_value, 2),
    )


@router.get("/stress-test", response_model=StressTestOut)
def get_stress_test(
    db: Session = Depends(get_db),
    _: str = Depends(get_current_user),
):
    p = db.query(Portfolio).first()
    if not p:
        raise HTTPException(status_code=404, detail="No portfolio found")
    r = _latest_risk(db)

    calc = RiskCalculator()
    import numpy as np
    proxy_returns = np.random.normal(0.0005, (r.volatility or 0.12) / 252 ** 0.5, 252)
    scenarios = calc.stress_test(p.total_value, proxy_returns)

    return StressTestOut(
        portfolio_value=p.total_value,
        market_crash_20pct=round(scenarios["market_crash_20pct"], 2),
        market_crash_30pct=round(scenarios["market_crash_30pct"], 2),
        flash_crash_10pct_1day=round(scenarios["flash_crash_10pct_1day"], 2),
        slow_decline_15pct_3months=round(scenarios["slow_decline_15pct_3months"], 2),
        black_swan_3sigma=round(scenarios["black_swan_3sigma"], 2),
    )
