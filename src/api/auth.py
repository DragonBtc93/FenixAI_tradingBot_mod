from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.config.database import get_db
from src.models.user import User
import os
import uuid

# Security Config
SECRET_KEY = os.getenv("JWT_SECRET", "your-secret-key-keep-it-safe")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["pbkdf2_sha256", "bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

router = APIRouter(prefix="/api/auth", tags=["auth"])

import logging
logger = logging.getLogger("FenixAuth")

# --- Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    role: str
    is_active: bool
    
class LoginRequest(BaseModel):
    email: str
    password: str

# --- Helpers ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password and return the string.

    This function prefers pbkdf2_sha256 (no external non-py dependency), and falls back
    to the configured pwd_context if necessary. This avoids issues when the system's
    bcrypt library is missing or incompatible (e.g. missing __about__ attribute).
    """
    try:
        # Try with the default context (pbkdf2_sha256 if configured as first scheme).
        return pwd_context.hash(password)
    except Exception as e:
        logger = logging.getLogger("FenixAuth")
        logger.warning(f"Password hashing failed with default scheme: {e}. Trying explicit pbkdf2_sha256.")
        # Explicit fallback to pbkdf2_sha256
        try:
            return pwd_context.hash(password, scheme="pbkdf2_sha256")
        except Exception as ex:
            logger.error(f"Fallback hashing with pbkdf2_sha256 also failed: {ex}")
            # Re-raise to let the caller handle failure and to avoid silently storing plain text
            raise

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
        
    result = await db.execute(select(User).where(User.email == token_data.username))
    user = result.scalar_one_or_none()
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# --- Routes ---

@router.post("/login", response_model=dict) # Changed response model to dict to match frontend expectation
async def login_for_access_token(form_data: LoginRequest, request: Request, db: AsyncSession = Depends(get_db)):
    # Note: Frontend sends JSON body {email, password}, not Form data
    logger.info(f"Login attempt for email={form_data.email} from={request.client.host if request.client else 'unknown'}")
    result = await db.execute(select(User).where(User.email == form_data.email))
    user = result.scalar_one_or_none()
    logger.debug(f"User lookup: {user.email if user else 'not found'}")

    if not user:
        logger.warning(f"Authentication failed: user not found for email={form_data.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(form_data.password, user.hashed_password):
        logger.warning(f"Authentication failed: password verification failed for email={form_data.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email, "role": user.role, "userId": user.id}, 
        expires_delta=access_token_expires
    )
    
    # Return structure matching what frontend expects (referencing api/src/routes/auth.ts lines 100-108)
    return {
        "success": True,
        "token": access_token, # Frontend expects 'token' or 'accessToken' in root or data? Checking authStore.ts line 54: data.token
        "user": {
            "id": user.id,
            "email": user.email,
            "name": user.full_name,
            "role": user.role,
            "is_active": user.is_active
        }
    }

@router.get("/me", response_model=dict)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return {
        "success": True,
        "data": {
            "user": {
                "id": current_user.id,
                "email": current_user.email,
                "name": current_user.full_name,
                "role": current_user.role,
                "is_active": current_user.is_active
            },
             # Mock permissions based on role
            "permissions": ["read:trading", "write:trading"] if current_user.role == "admin" else ["read:trading"]
        }
    }

# Standard OAuth2 route (good for swagger UI)
@router.post("/token", response_model=Token)
async def login_swagger(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    # This endpoint is kept for Swagger UI compatibility
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}
