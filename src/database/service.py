"""
Database Service Layer
"""

from sqlalchemy.orm import Session
from src.database.models import (
    User, Portfolio, Holding, StockData, CompanyInfo,
    RiskMetrics, ChatHistory, PersonalizationEvent
)
from src.database.database import SessionLocal
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseService:
    """High-level database operations"""
    
    def __init__(self):
        pass
    
    # ============ USER OPERATIONS ============
    
    def create_user(self, email: str, full_name: str, 
                   risk_tolerance: str = 'moderate',
                   esg_priority: str = 'balanced') -> Dict:
        """Create a new user - returns dict instead of object"""
        
        db = SessionLocal()
        try:
            user = User(
                email=email,
                full_name=full_name,
                risk_tolerance=risk_tolerance,
                esg_priority=esg_priority,
                esg_focus=['environmental', 'social', 'governance'],
                exclude_sectors=[],
                investment_goals=['wealth_building']
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            
            # Convert to dict before session closes
            user_dict = {
                'id': user.id,
                'email': user.email,
                'full_name': user.full_name,
                'risk_tolerance': user.risk_tolerance,
                'esg_priority': user.esg_priority
            }
            
            logger.info(f"Created user: {email}")
            return user_dict
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == email).first()
            if user:
                return {
                    'id': user.id,
                    'email': user.email,
                    'full_name': user.full_name,
                    'risk_tolerance': user.risk_tolerance
                }
            return None
        finally:
            db.close()
    
    # ============ PORTFOLIO OPERATIONS ============
    
    def create_portfolio(self, user_id: str, name: str, 
                        total_value: float = 0.0) -> Dict:
        """Create a new portfolio"""
        
        db = SessionLocal()
        try:
            portfolio = Portfolio(
                user_id=user_id,
                name=name,
                total_value=total_value
            )
            db.add(portfolio)
            db.commit()
            db.refresh(portfolio)
            
            portfolio_dict = {
                'id': portfolio.id,
                'user_id': portfolio.user_id,
                'name': portfolio.name,
                'total_value': portfolio.total_value
            }
            
            logger.info(f"Created portfolio: {name}")
            return portfolio_dict
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    def update_portfolio_esg(self, portfolio_id: str, esg_scores: Dict) -> Dict:
        """Update portfolio ESG scores"""
        
        db = SessionLocal()
        try:
            portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
            if not portfolio:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            portfolio.esg_score_overall = esg_scores.get('overall')
            portfolio.environmental_score = esg_scores.get('environmental')
            portfolio.social_score = esg_scores.get('social')
            portfolio.governance_score = esg_scores.get('governance')
            portfolio.esg_rating = esg_scores.get('rating')
            portfolio.carbon_intensity = esg_scores.get('carbon_intensity')
            portfolio.carbon_footprint = esg_scores.get('carbon_footprint')
            portfolio.updated_at = datetime.utcnow()
            
            db.commit()
            
            return {
                'id': portfolio.id,
                'esg_score_overall': portfolio.esg_score_overall,
                'esg_rating': portfolio.esg_rating
            }
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    # ============ HOLDINGS OPERATIONS ============
    
    def add_holding(self, portfolio_id: str, ticker: str, 
                   quantity: float, purchase_price: float,
                   asset_type: str = 'stock') -> Dict:
        """Add a holding to portfolio"""
        
        db = SessionLocal()
        try:
            holding = Holding(
                portfolio_id=portfolio_id,
                ticker=ticker,
                asset_type=asset_type,
                quantity=quantity,
                purchase_price=purchase_price,
                current_price=purchase_price,
                purchase_date=datetime.utcnow()
            )
            db.add(holding)
            db.commit()
            db.refresh(holding)
            
            holding_dict = {
                'id': holding.id,
                'ticker': holding.ticker,
                'quantity': holding.quantity,
                'purchase_price': holding.purchase_price
            }
            
            logger.info(f"Added holding {ticker}")
            return holding_dict
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    # ============ COMPANY INFO OPERATIONS ============
    
    def save_company_info(self, company_data: Dict) -> Dict:
        """Save or update company information"""
        
        db = SessionLocal()
        try:
            company = db.query(CompanyInfo)\
                       .filter(CompanyInfo.ticker == company_data['ticker'])\
                       .first()
            
            if company:
                # Update existing
                for key, value in company_data.items():
                    if hasattr(company, key):
                        setattr(company, key, value)
                company.updated_at = datetime.utcnow()
            else:
                # Create new
                company = CompanyInfo(**company_data)
                db.add(company)
            
            db.commit()
            db.refresh(company)
            
            company_dict = {
                'ticker': company.ticker,
                'company_name': company.company_name,
                'esg_score': company.esg_score
            }
            
            logger.info(f"Saved company info for {company_data['ticker']}")
            return company_dict
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    # ============ RISK METRICS OPERATIONS ============
    
    def save_risk_metrics(self, portfolio_id: str, risk_data: Dict) -> Dict:
        """Save portfolio risk metrics"""
        
        db = SessionLocal()
        try:
            risk_metrics = RiskMetrics(
                portfolio_id=portfolio_id,
                calculation_date=datetime.utcnow(),
                var_95_daily=risk_data.get('var_95_daily'),
                var_95_monthly=risk_data.get('var_95_monthly'),
                var_99_daily=risk_data.get('var_99_daily'),
                cvar_95=risk_data.get('cvar_95'),
                sharpe_ratio=risk_data.get('sharpe_ratio'),
                sortino_ratio=risk_data.get('sortino_ratio'),
                max_drawdown=risk_data.get('max_drawdown'),
                volatility=risk_data.get('volatility'),
                beta=risk_data.get('beta'),
                alpha=risk_data.get('alpha')
            )
            db.add(risk_metrics)
            db.commit()
            db.refresh(risk_metrics)
            
            risk_dict = {
                'id': risk_metrics.id,
                'var_95_daily': risk_metrics.var_95_daily,
                'sharpe_ratio': risk_metrics.sharpe_ratio
            }
            
            logger.info(f"Saved risk metrics")
            return risk_dict
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    # ============ CHAT HISTORY OPERATIONS ============
    
    def save_chat_message(self, user_id: str, session_id: str,
                         user_query: str, bot_response: str,
                         tokens_used: int = 0, response_time: float = 0.0,
                         cost: float = 0.0) -> Dict:
        """Save chat interaction"""
        
        db = SessionLocal()
        try:
            chat = ChatHistory(
                user_id=user_id,
                session_id=session_id,
                timestamp=datetime.utcnow(),
                user_query=user_query,
                bot_response=bot_response,
                tokens_used=tokens_used,
                response_time=response_time,
                cost=cost,
                model_used='gpt-3.5-turbo'
            )
            db.add(chat)
            db.commit()
            
            return {'id': chat.id, 'session_id': session_id}
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()