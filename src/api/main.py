"""
Portfolio Intelligence Platform — FastAPI Backend
Run with: uvicorn src.api.main:app --reload --port 8000
Docs at:  http://localhost:8000/docs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import auth, portfolio, risk, esg
from src.api.schemas import HealthOut

app = FastAPI(
    title="Portfolio Intelligence Platform API",
    description="REST API for portfolio analytics, risk metrics, and ESG data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit dev origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router,      prefix="/api/v1")
app.include_router(portfolio.router, prefix="/api/v1")
app.include_router(risk.router,      prefix="/api/v1")
app.include_router(esg.router,       prefix="/api/v1")


@app.get("/api/v1/health", response_model=HealthOut, tags=["health"])
def health():
    return HealthOut(status="ok", version="1.0.0")
