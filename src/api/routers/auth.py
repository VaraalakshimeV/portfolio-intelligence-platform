from fastapi import APIRouter, HTTPException
from src.api.schemas import LoginRequest, TokenResponse
from src.api.deps import create_access_token
from src.auth.password import verify_password

router = APIRouter(prefix="/auth", tags=["auth"])

# Same hashed credentials as the Streamlit app
_USERS = {
    "analyst@pip.com":  {"hash": "$2b$12$76xyyLP.8facMGEosbvsseKYl7YQ7mrf4b5Flf52cFf9jgRNvKpAe", "name": "Sarah Mitchell",   "role": "Portfolio Analyst"},
    "manager@pip.com":  {"hash": "$2b$12$76xyyLP.8facMGEosbvsseKYl7YQ7mrf4b5Flf52cFf9jgRNvKpAe", "name": "David Chen",       "role": "Senior Risk Manager"},
    "admin@pip.com":    {"hash": "$2b$12$eY81cc2/5Ay7hIa7pvgUZu15i1Rpod.f/hb2RfKlxUDHODQz2UtVO", "name": "Varaalakshime V.", "role": "Platform Administrator"},
}


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest):
    user = _USERS.get(body.email)
    if not user or not verify_password(body.password, user["hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(body.email)
    return TokenResponse(
        access_token=token,
        user_name=user["name"],
        user_role=user["role"],
    )
