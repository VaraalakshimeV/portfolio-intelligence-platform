"""
Configuration Management for Fintech AI Platform
Bulletproof against recruiter criticism
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    """Enterprise-grade configuration"""
    
    # Application
    APP_NAME: str = "Fintech AI Platform"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # API Keys - REQUIRED
    OPENAI_API_KEY: str = "sk-test-key"
    PINECONE_API_KEY: str = "test-key"
    PINECONE_ENVIRONMENT: str = "us-east-1"
    
    # Database
    DATABASE_URL: str = "sqlite:///./fintech.db"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 40
    DATABASE_ECHO: bool = False
    
    # LLM Configuration
    LLM_MODEL: str = "gpt-3.5-turbo-16k"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 2000
    
    # Risk Calculation
    VAR_CONFIDENCE_LEVEL: float = 0.95
    MONTE_CARLO_SIMULATIONS: int = 10000
    HISTORICAL_DAYS: int = 252
    
    # ESG Settings
    ENABLE_ESG_SCORING: bool = True
    ESG_WEIGHT_ENVIRONMENT: float = 0.33
    ESG_WEIGHT_SOCIAL: float = 0.33
    ESG_WEIGHT_GOVERNANCE: float = 0.34
    
    # AI Personalization
    ENABLE_PERSONALIZATION: bool = True
    PERSONALIZATION_MIN_INTERACTIONS: int = 5
    
    # Compliance
    ENABLE_COMPLIANCE_CHECK: bool = True
    ENABLE_AUDIT_LOGGING: bool = True
    
    # File Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    LOG_DIR: Path = BASE_DIR / "logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"

# Global settings instance
settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(exist_ok=True)
settings.LOG_DIR.mkdir(exist_ok=True)