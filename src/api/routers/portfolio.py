from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from src.api.schemas import PortfolioOut, HoldingOut
from src.api.deps import get_db, get_current_user
from src.database.models import Portfolio, Holding

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


def _get_portfolio(db: Session) -> Portfolio:
    p = db.query(Portfolio).first()
    if not p:
        raise HTTPException(status_code=404, detail="No portfolio found")
    return p


@router.get("", response_model=PortfolioOut)
def get_portfolio(
    db: Session = Depends(get_db),
    _: str = Depends(get_current_user),
):
    p = _get_portfolio(db)
    holdings_count = db.query(Holding).filter(Holding.portfolio_id == p.id).count()
    return PortfolioOut(
        name=p.name,
        total_value=p.total_value,
        esg_rating=p.esg_rating,
        esg_score_overall=p.esg_score_overall,
        environmental_score=p.environmental_score,
        social_score=p.social_score,
        governance_score=p.governance_score,
        carbon_intensity=p.carbon_intensity,
        holdings_count=holdings_count,
    )


@router.get("/holdings", response_model=List[HoldingOut])
def get_holdings(
    db: Session = Depends(get_db),
    _: str = Depends(get_current_user),
):
    p = _get_portfolio(db)
    holdings = db.query(Holding).filter(Holding.portfolio_id == p.id).all()
    result = []
    for h in holdings:
        market_value = (h.current_price or h.purchase_price) * h.quantity
        weight_pct = (market_value / p.total_value * 100) if p.total_value else None
        result.append(HoldingOut(
            ticker=h.ticker,
            asset_type=h.asset_type,
            quantity=h.quantity,
            purchase_price=h.purchase_price,
            current_price=h.current_price,
            market_value=round(market_value, 2),
            weight_pct=round(weight_pct, 2) if weight_pct else None,
            esg_score=h.esg_score,
        ))
    return result
