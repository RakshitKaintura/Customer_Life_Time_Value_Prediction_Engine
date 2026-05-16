"""
Authentication Router — Mock auth for demo purposes.

Endpoints:
    POST /auth/signin      Sign in with email/password
    POST /auth/signup      Sign up with email/password/name
    POST /auth/signout     Sign out (clear token)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from loguru import logger

router = APIRouter(prefix="/auth", tags=["auth"])


class SignInRequest(BaseModel):
    email: str
    password: str


class SignUpRequest(BaseModel):
    email: str
    password: str
    name: str


class AuthResponse(BaseModel):
    token: str
    user: dict


# Simple in-memory store for demo (in production, use a database)
_users_db: dict = {}


def _generate_token(email: str) -> str:
    """Generate a simple JWT-like token for demo purposes."""
    # In production, use proper JWT library
    payload = {
        "email": email,
        "iat": datetime.utcnow().isoformat(),
        "exp": (datetime.utcnow() + timedelta(days=7)).isoformat(),
    }
    import base64
    return base64.b64encode(json.dumps(payload).encode()).decode()


@router.post("/signin", response_model=AuthResponse)
async def sign_in(request: SignInRequest) -> AuthResponse:
    """
    Sign in with email and password.
    
    For demo: accepts any email/password combination.
    """
    try:
        # Demo: accept any credentials
        if not request.email or not request.password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email and password required"
            )

        # Get or create user
        if request.email not in _users_db:
            _users_db[request.email] = {
                "email": request.email,
                "name": request.email.split("@")[0],
            }

        user = _users_db[request.email]
        token = _generate_token(request.email)

        logger.info(f"User signed in: {request.email}")
        
        return AuthResponse(
            token=token,
            user=user
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Sign in error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sign in failed"
        )


@router.post("/signup", response_model=AuthResponse)
async def sign_up(request: SignUpRequest) -> AuthResponse:
    """
    Sign up with email, password, and name.
    
    For demo: accepts any credentials.
    """
    try:
        if not request.email or not request.password or not request.name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email, password, and name required"
            )

        # Check if user already exists
        if request.email in _users_db:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User already exists"
            )

        # Create new user
        user = {
            "email": request.email,
            "name": request.name,
            "created_at": datetime.utcnow().isoformat(),
        }
        _users_db[request.email] = user

        token = _generate_token(request.email)

        logger.info(f"New user created: {request.email}")
        
        return AuthResponse(
            token=token,
            user=user
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Sign up error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sign up failed"
        )


@router.post("/signout")
async def sign_out() -> dict:
    """
    Sign out (frontend-only operation, clears token).
    """
    logger.info("User signed out")
    return {"message": "Signed out successfully"}
