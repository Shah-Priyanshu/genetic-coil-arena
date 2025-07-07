"""
Phase 3D: Production Features - Authentication & Security
Complete user management system for production deployment
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import secrets
import hashlib

# Security configuration
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-memory user store (in production, this would be a database)
users_db = {
    "admin": {
        "username": "admin",
        "email": "admin@genetic-coil-arena.com",
        "hashed_password": pwd_context.hash("admin123"),
        "role": "admin",
        "created_at": datetime.now(),
        "experiments": []
    },
    "researcher": {
        "username": "researcher",
        "email": "researcher@genetic-coil-arena.com", 
        "hashed_password": pwd_context.hash("research123"),
        "role": "researcher",
        "created_at": datetime.now(),
        "experiments": []
    }
}


class User:
    def __init__(self, username: str, email: str, role: str = "researcher"):
        self.username = username
        self.email = email
        self.role = role
        self.created_at = datetime.now()
        self.experiments = []


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Authenticate user credentials."""
    user = users_db.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = users_db.get(username)
    if user is None:
        raise credentials_exception
    return user


def create_user(username: str, email: str, password: str, role: str = "researcher") -> dict:
    """Create a new user."""
    if username in users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    hashed_password = get_password_hash(password)
    user_data = {
        "username": username,
        "email": email,
        "hashed_password": hashed_password,
        "role": role,
        "created_at": datetime.now(),
        "experiments": []
    }
    users_db[username] = user_data
    return user_data


def get_user_experiments(username: str) -> list:
    """Get experiments for a specific user."""
    user = users_db.get(username)
    if not user:
        return []
    return user.get("experiments", [])


def add_experiment_to_user(username: str, experiment_id: str):
    """Add experiment to user's experiment list."""
    user = users_db.get(username)
    if user:
        if "experiments" not in user:
            user["experiments"] = []
        if experiment_id not in user["experiments"]:
            user["experiments"].append(experiment_id)
