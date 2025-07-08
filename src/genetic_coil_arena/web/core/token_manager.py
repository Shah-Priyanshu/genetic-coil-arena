"""
Advanced JWT Token Management
Addresses senior developer review recommendations for token security.
"""

import time
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, Set, Optional, Any
from jose import JWTError, jwt
import logging

logger = logging.getLogger(__name__)

class TokenManager:
    """
    Advanced JWT token management with revocation and refresh capabilities.
    In production, this should use Redis or a database for persistence.
    """
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        
        # In-memory stores (use Redis in production)
        self.revoked_tokens: Set[str] = set()
        self.refresh_tokens: Dict[str, Dict[str, Any]] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Cleanup old tokens periodically
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # 1 hour
    
    def create_access_token(
        self, 
        data: Dict[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a new access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=15)
        
        # Add standard claims
        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": secrets.token_urlsafe(16),  # JWT ID for revocation
            "type": "access"
        })
        
        token = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        # Track active session
        user_id = data.get("sub")
        if user_id:
            session_id = to_encode["jti"]
            self.active_sessions[session_id] = {
                "user_id": user_id,
                "created_at": datetime.now(timezone.utc),
                "expires_at": expire,
                "token_type": "access"
            }
        
        return token
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create a new refresh token."""
        token_id = secrets.token_urlsafe(32)
        expire = datetime.now(timezone.utc) + timedelta(days=30)
        
        refresh_data = {
            "user_id": user_id,
            "token_id": token_id,
            "created_at": datetime.now(timezone.utc),
            "expires_at": expire,
            "used": False
        }
        
        self.refresh_tokens[token_id] = refresh_data
        
        # Create JWT for the refresh token
        payload = {
            "sub": user_id,
            "jti": token_id,
            "type": "refresh",
            "exp": expire,
            "iat": datetime.now(timezone.utc)
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def refresh_access_token(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """Create a new access token using a refresh token."""
        try:
            # Decode refresh token
            payload = jwt.decode(
                refresh_token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            
            token_id = payload.get("jti")
            user_id = payload.get("sub")
            token_type = payload.get("type")
            
            if token_type != "refresh":
                logger.warning("Invalid token type for refresh")
                return None
            
            # Check if refresh token exists and is valid
            if token_id not in self.refresh_tokens:
                logger.warning("Refresh token not found")
                return None
            
            refresh_data = self.refresh_tokens[token_id]
            
            if refresh_data["used"]:
                logger.warning("Refresh token already used")
                return None
            
            if datetime.now(timezone.utc) > refresh_data["expires_at"]:
                logger.warning("Refresh token expired")
                del self.refresh_tokens[token_id]
                return None
            
            # Mark refresh token as used
            refresh_data["used"] = True
            
            # Create new access token
            access_token = self.create_access_token(
                data={"sub": user_id},
                expires_delta=timedelta(minutes=30)
            )
            
            # Create new refresh token
            new_refresh_token = self.create_refresh_token(user_id)
            
            # Remove old refresh token
            del self.refresh_tokens[token_id]
            
            return {
                "access_token": access_token,
                "refresh_token": new_refresh_token,
                "token_type": "bearer"
            }
            
        except JWTError as e:
            logger.warning(f"Invalid refresh token: {e}")
            return None
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a token."""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            
            token_id = payload.get("jti")
            
            # Check if token is revoked
            if token_id in self.revoked_tokens:
                logger.warning("Token has been revoked")
                return None
            
            # Check if session is still active
            if token_id in self.active_sessions:
                session = self.active_sessions[token_id]
                if datetime.now(timezone.utc) > session["expires_at"]:
                    # Clean up expired session
                    del self.active_sessions[token_id]
                    return None
            
            return payload
            
        except JWTError as e:
            logger.warning(f"Token verification failed: {e}")
            return None
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a specific token."""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options={"verify_exp": False}  # Allow expired tokens for revocation
            )
            
            token_id = payload.get("jti")
            if token_id:
                self.revoked_tokens.add(token_id)
                
                # Remove from active sessions
                if token_id in self.active_sessions:
                    del self.active_sessions[token_id]
                
                logger.info(f"Token {token_id[:8]}... revoked")
                return True
            
        except JWTError as e:
            logger.warning(f"Failed to revoke token: {e}")
        
        return False
    
    def revoke_user_tokens(self, user_id: str) -> int:
        """Revoke all tokens for a specific user."""
        revoked_count = 0
        
        # Revoke active sessions
        sessions_to_remove = []
        for session_id, session in self.active_sessions.items():
            if session["user_id"] == user_id:
                self.revoked_tokens.add(session_id)
                sessions_to_remove.append(session_id)
                revoked_count += 1
        
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
        
        # Revoke refresh tokens
        refresh_tokens_to_remove = []
        for token_id, refresh_data in self.refresh_tokens.items():
            if refresh_data["user_id"] == user_id:
                refresh_tokens_to_remove.append(token_id)
                revoked_count += 1
        
        for token_id in refresh_tokens_to_remove:
            del self.refresh_tokens[token_id]
        
        logger.info(f"Revoked {revoked_count} tokens for user {user_id}")
        return revoked_count
    
    def get_user_sessions(self, user_id: str) -> list:
        """Get all active sessions for a user."""
        sessions = []
        for session_id, session in self.active_sessions.items():
            if session["user_id"] == user_id:
                sessions.append({
                    "session_id": session_id,
                    "created_at": session["created_at"],
                    "expires_at": session["expires_at"],
                    "token_type": session["token_type"]
                })
        return sessions
    
    def cleanup_expired_tokens(self) -> Dict[str, int]:
        """Clean up expired tokens and sessions."""
        current_time = time.time()
        
        # Only run cleanup periodically
        if current_time - self._last_cleanup < self._cleanup_interval:
            return {"message": "Cleanup not needed yet"}
        
        self._last_cleanup = current_time
        current_datetime = datetime.now(timezone.utc)
        
        # Clean up expired sessions
        expired_sessions = []
        for session_id, session in self.active_sessions.items():
            if current_datetime > session["expires_at"]:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        # Clean up expired refresh tokens
        expired_refresh = []
        for token_id, refresh_data in self.refresh_tokens.items():
            if current_datetime > refresh_data["expires_at"]:
                expired_refresh.append(token_id)
        
        for token_id in expired_refresh:
            del self.refresh_tokens[token_id]
        
        # Clean up old revoked tokens (keep for 24 hours)
        # This is a simplified approach - in production, use proper storage
        
        result = {
            "expired_sessions_removed": len(expired_sessions),
            "expired_refresh_tokens_removed": len(expired_refresh),
            "total_active_sessions": len(self.active_sessions),
            "total_refresh_tokens": len(self.refresh_tokens),
            "total_revoked_tokens": len(self.revoked_tokens)
        }
        
        logger.info(f"Token cleanup completed: {result}")
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get token management statistics."""
        return {
            "active_sessions": len(self.active_sessions),
            "refresh_tokens": len(self.refresh_tokens),
            "revoked_tokens": len(self.revoked_tokens),
            "last_cleanup": self._last_cleanup,
            "cleanup_interval": self._cleanup_interval
        }


# Global token manager instance
token_manager: Optional[TokenManager] = None

def get_token_manager() -> TokenManager:
    """Get the global token manager instance."""
    global token_manager
    if token_manager is None:
        raise RuntimeError("Token manager not initialized")
    return token_manager

def init_token_manager(secret_key: str) -> TokenManager:
    """Initialize the global token manager."""
    global token_manager
    token_manager = TokenManager(secret_key)
    return token_manager
