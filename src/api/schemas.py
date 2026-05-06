from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_name: str
    user_role: str


class HoldingOut(BaseModel):
    ticker: str
    asset_type: str
    quantity: float
    purchase_price: float
    current_price: Optional[float]
    market_value: Optional[float]
    weight_pct: Optional[float]
    esg_score: Optional[float]


class PortfolioOut(BaseModel):
    name: str
    total_value: float
    esg_rating: Optional[str]
    esg_score_overall: Optional[float]
    environmental_score: Optional[float]
    social_score: Optional[float]
    governance_score: Optional[float]
    carbon_intensity: Optional[float]
    holdings_count: int


class RiskMetricsOut(BaseModel):
    calculation_date: Optional[datetime]
    var_95_daily: Optional[float]
    var_95_monthly: Optional[float]
    var_99_daily: Optional[float]
    cvar_95: Optional[float]
    sharpe_ratio: Optional[float]
    sortino_ratio: Optional[float]
    volatility: Optional[float]
    max_drawdown: Optional[float]
    max_drawdown_pct: Optional[float]
    beta: Optional[float]
    alpha: Optional[float]
    diversification_ratio: Optional[float]


class VaROut(BaseModel):
    method: str
    var_95_daily: float
    var_95_monthly: float
    cvar_95: float
    var_99_daily: Optional[float]


class StressTestOut(BaseModel):
    portfolio_value: float
    market_crash_20pct: float
    market_crash_30pct: float
    flash_crash_10pct_1day: float
    slow_decline_15pct_3months: float
    black_swan_3sigma: float


class ESGOut(BaseModel):
    overall_score: Optional[float]
    rating: Optional[str]
    environmental: Optional[float]
    social: Optional[float]
    governance: Optional[float]
    carbon_intensity: Optional[float]
    carbon_footprint: Optional[float]


class ESGHoldingOut(BaseModel):
    ticker: str
    esg_score: Optional[float]
    carbon_emissions: Optional[float]
    weight_pct: Optional[float]


class HealthOut(BaseModel):
    status: str
    version: str
