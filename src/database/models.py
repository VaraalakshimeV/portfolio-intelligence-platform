"""
Enterprise Database Models for Fintech AI Platform
Includes ESG metrics and AI-driven personalization
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    """User accounts with personalization profiles"""
    __tablename__ = 'users'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255))
    
    # Risk Profile
    risk_tolerance = Column(String(20), default='moderate')
    investment_horizon = Column(String(20), default='medium')
    investment_goals = Column(JSON)
    
    # ESG Preferences
    esg_priority = Column(String(20), default='balanced')
    esg_focus = Column(JSON)
    exclude_sectors = Column(JSON)
    minimum_esg_score = Column(Float, default=50.0)
    
    # AI Personalization
    interaction_count = Column(Integer, default=0)
    learning_style = Column(String(20), default='balanced')
    preferred_detail_level = Column(String(20), default='medium')
    favorite_topics = Column(JSON)
    typical_session_length = Column(Integer)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    portfolios = relationship("Portfolio", back_populates="user")
    chat_history = relationship("ChatHistory", back_populates="user")
    personalization_events = relationship("PersonalizationEvent", back_populates="user")
    recommendations = relationship("AIRecommendation", back_populates="user")

class Portfolio(Base):
    """User portfolios with ESG tracking"""
    __tablename__ = 'portfolios'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    name = Column(String(255), nullable=False)
    total_value = Column(Float, default=0.0)
    
    # ESG Scores
    esg_score_overall = Column(Float)
    environmental_score = Column(Float)
    social_score = Column(Float)
    governance_score = Column(Float)
    esg_rating = Column(String(3))
    
    # Carbon Metrics
    carbon_intensity = Column(Float)
    carbon_footprint = Column(Float)
    esg_alignment_score = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="portfolios")
    holdings = relationship("Holding", back_populates="portfolio")
    risk_metrics = relationship("RiskMetrics", back_populates="portfolio")
    
    __table_args__ = (Index('idx_user_portfolio', 'user_id'),)

class Holding(Base):
    """Individual holdings with ESG data"""
    __tablename__ = 'holdings'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    portfolio_id = Column(String(36), ForeignKey('portfolios.id'), nullable=False)
    ticker = Column(String(20), nullable=False)
    asset_type = Column(String(20), nullable=False)
    quantity = Column(Float, nullable=False)
    purchase_price = Column(Float, nullable=False)
    current_price = Column(Float)
    purchase_date = Column(DateTime, nullable=False)
    esg_score = Column(Float)
    carbon_emissions = Column(Float)
    
    portfolio = relationship("Portfolio", back_populates="holdings")
    
    __table_args__ = (Index('idx_portfolio_holding', 'portfolio_id', 'ticker'),)

class StockData(Base):
    """Historical stock data"""
    __tablename__ = 'stock_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False, index=True)
    asset_type = Column(String(20), nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float, nullable=False)
    volume = Column(Integer)
    adjusted_close = Column(Float)
    
    __table_args__ = (Index('idx_ticker_date', 'ticker', 'date'),)

class CompanyInfo(Base):
    """Company information with comprehensive ESG metrics"""
    __tablename__ = 'company_info'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), unique=True, nullable=False, index=True)
    company_name = Column(String(255))
    sector = Column(String(100), index=True)
    industry = Column(String(100), index=True)
    
    # Financial Metrics
    market_cap = Column(Float)
    pe_ratio = Column(Float)
    forward_pe = Column(Float)
    peg_ratio = Column(Float)
    price_to_book = Column(Float)
    dividend_yield = Column(Float)
    profit_margin = Column(Float)
    operating_margin = Column(Float)
    roe = Column(Float)
    roa = Column(Float)
    debt_to_equity = Column(Float)
    current_ratio = Column(Float)
    beta = Column(Float)
    fifty_two_week_high = Column(Float)
    fifty_two_week_low = Column(Float)
    
    # ESG Scores
    esg_score = Column(Float)
    environmental_score = Column(Float)
    social_score = Column(Float)
    governance_score = Column(Float)
    esg_rating = Column(String(3))
    esg_controversies = Column(Integer, default=0)
    
    # Environmental Metrics
    carbon_emissions = Column(Float)
    carbon_intensity = Column(Float)
    renewable_energy_pct = Column(Float)
    water_usage = Column(Float)
    waste_recycling_pct = Column(Float)
    environmental_innovations = Column(Integer)
    
    # Social Metrics
    employee_satisfaction = Column(Float)
    diversity_score = Column(Float)
    female_employees_pct = Column(Float)
    employee_turnover_rate = Column(Float)
    training_hours_per_employee = Column(Float)
    community_investment = Column(Float)
    labor_practices_score = Column(Float)
    human_rights_score = Column(Float)
    
    # Governance Metrics
    board_independence = Column(Float)
    board_diversity = Column(Float)
    female_board_members = Column(Float)
    executive_compensation_ratio = Column(Float)
    shareholder_rights_score = Column(Float)
    anti_corruption_score = Column(Float)
    tax_transparency_score = Column(Float)
    
    business_summary = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ESGNews(Base):
    """ESG-related news and controversies"""
    __tablename__ = 'esg_news'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    headline = Column(Text)
    summary = Column(Text)
    category = Column(String(50))
    sentiment = Column(String(20))
    impact_score = Column(Float)
    source = Column(String(255))
    url = Column(Text)
    
    __table_args__ = (Index('idx_esg_ticker_date', 'ticker', 'date'),)

class ChatHistory(Base):
    """Chat history with personalization tracking"""
    __tablename__ = 'chat_history'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    session_id = Column(String(36), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    user_query = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    
    # AI Metrics
    tokens_used = Column(Integer)
    response_time = Column(Float)
    cost = Column(Float)
    model_used = Column(String(50))
    
    # Personalization
    query_topic = Column(String(100))
    user_satisfaction = Column(Integer)
    detail_level_used = Column(String(20))
    included_esg_info = Column(Boolean, default=False)
    
    # Compliance
    compliance_passed = Column(Boolean, default=True)
    compliance_issues = Column(JSON)
    
    user = relationship("User", back_populates="chat_history")
    
    __table_args__ = (Index('idx_user_session', 'user_id', 'session_id'),)

class PersonalizationEvent(Base):
    """Track user behavior for AI-driven personalization"""
    __tablename__ = 'personalization_events'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    event_type = Column(String(50), nullable=False)
    event_data = Column(JSON)
    
    time_of_day = Column(String(20))
    day_of_week = Column(String(10))
    device_type = Column(String(20))
    
    user = relationship("User", back_populates="personalization_events")
    
    __table_args__ = (Index('idx_user_timestamp', 'user_id', 'timestamp'),)

class RiskMetrics(Base):
    """Portfolio risk metrics including ESG risk"""
    __tablename__ = 'risk_metrics'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    portfolio_id = Column(String(36), ForeignKey('portfolios.id'), nullable=False)
    calculation_date = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Traditional Risk
    var_95_daily = Column(Float)
    var_95_monthly = Column(Float)
    var_99_daily = Column(Float)
    cvar_95 = Column(Float)
    
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    volatility = Column(Float)
    beta = Column(Float)
    alpha = Column(Float)
    
    correlation_score = Column(Float)
    diversification_ratio = Column(Float)
    sector_concentration = Column(JSON)
    geographic_concentration = Column(JSON)
    
    # ESG Risk
    esg_risk_score = Column(Float)
    environmental_risk = Column(Float)
    social_risk = Column(Float)
    governance_risk = Column(Float)
    controversy_risk = Column(Float)
    stranded_assets_risk = Column(Float)
    reputation_risk = Column(Float)
    
    monte_carlo_results = Column(JSON)
    
    # Stress Tests
    stress_test_market_crash = Column(Float)
    stress_test_interest_rate = Column(Float)
    stress_test_esg_crisis = Column(Float)
    
    portfolio = relationship("Portfolio", back_populates="risk_metrics")
    
    __table_args__ = (Index('idx_portfolio_date', 'portfolio_id', 'calculation_date'),)

class AIRecommendation(Base):
    """AI-generated personalized recommendations"""
    __tablename__ = 'ai_recommendations'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    portfolio_id = Column(String(36), ForeignKey('portfolios.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    recommendation_type = Column(String(50))
    priority = Column(String(20))
    title = Column(String(255))
    description = Column(Text)
    reasoning = Column(Text)
    
    suggested_actions = Column(JSON)
    expected_impact = Column(JSON)
    
    personalization_score = Column(Float)
    based_on_preferences = Column(JSON)
    
    status = Column(String(20), default='active')
    user_feedback = Column(String(20))
    implemented_at = Column(DateTime)
    
    user = relationship("User", back_populates="recommendations")

class ComplianceLog(Base):
    """Compliance and audit trail"""
    __tablename__ = 'compliance_logs'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    log_type = Column(String(50), nullable=False)
    user_id = Column(String(36))
    event_data = Column(JSON, nullable=False)
    compliance_status = Column(String(20))
    flagged_issues = Column(JSON)
    reviewed = Column(Boolean, default=False)
    reviewer_notes = Column(Text)
    
    __table_args__ = (
        Index('idx_type_timestamp', 'log_type', 'timestamp'),
        Index('idx_compliance_status', 'compliance_status'),
    )

class PerformanceMetrics(Base):
    """System performance monitoring"""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    endpoint = Column(String(100))
    response_time = Column(Float)
    tokens_used = Column(Integer)
    cost = Column(Float)
    cache_hit = Column(Boolean)
    error_occurred = Column(Boolean)
    error_message = Column(Text)
    user_id = Column(String(36))
    
    __table_args__ = (Index('idx_endpoint_time', 'endpoint', 'timestamp'),)