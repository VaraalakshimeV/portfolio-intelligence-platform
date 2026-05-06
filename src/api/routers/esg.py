from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from src.api.schemas import ESGOut, ESGHoldingOut
from src.api.deps import get_db, get_current_user
from src.database.models import Portfolio, Holding

router = APIRouter(prefix="/esg", tags=["esg"])


@router.get("", response_model=ESGOut)
def get_esg(
    db: Session = Depends(get_db),
    _: str = Depends(get_current_user),
):
    p = db.query(Portfolio).first()
    if not p:
        raise HTTPException(status_code=404, detail="No portfolio found")
    return ESGOut(
        overall_score=p.esg_score_overall,
        rating=p.esg_rating,
        environmental=p.environmental_score,
        social=p.social_score,
        governance=p.governance_score,
        carbon_intensity=p.carbon_intensity,
        carbon_footprint=p.carbon_footprint,
    )


@router.get("/holdings", response_model=List[ESGHoldingOut])
def get_esg_holdings(
    db: Session = Depends(get_db),
    _: str = Depends(get_current_user),
):
    p = db.query(Portfolio).first()
    if not p:
        raise HTTPException(status_code=404, detail="No portfolio found")
    holdings = db.query(Holding).filter(Holding.portfolio_id == p.id).all()
    result = []
    for h in holdings:
        market_value = (h.current_price or h.purchase_price) * h.quantity
        weight_pct = round(market_value / p.total_value * 100, 2) if p.total_value else None
        result.append(ESGHoldingOut(
            ticker=h.ticker,
            esg_score=h.esg_score,
            carbon_emissions=h.carbon_emissions,
            weight_pct=weight_pct,
        ))
    return sorted(result, key=lambda x: x.esg_score or 0, reverse=True)
