"""
Phase 3D: Production Authentication Routes
Complete authentication system with login, registration, and protected endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm, HTTPAuthorizationCredentials
from datetime import timedelta
from typing import Optional
import logging

try:
    from ..core.auth import (
        authenticate_user, create_access_token, get_current_user, 
        create_user, ACCESS_TOKEN_EXPIRE_MINUTES, users_db,
        revoke_token, refresh_access_token, revoke_user_tokens, security
    )
    from ..core.database import db
    from ..core.token_manager import get_token_manager
    AUTH_AVAILABLE = True
except ImportError as e:
    print(f"Auth imports not available: {e}")
    AUTH_AVAILABLE = False
    # Create fallback security for when auth is not available
    from fastapi.security import HTTPBearer
    security = HTTPBearer()

logger = logging.getLogger(__name__)
auth_router = APIRouter(prefix="/auth", tags=["authentication"])


@auth_router.get("/login")
async def login_page():
    """Serve the login page."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Genetic Coil Arena - Login</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 0; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                min-height: 100vh; 
                display: flex; 
                align-items: center; 
                justify-content: center; 
            }
            .login-container { 
                background: white; 
                border-radius: 20px; 
                padding: 40px; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.1); 
                width: 100%; 
                max-width: 400px; 
            }
            .logo { text-align: center; margin-bottom: 30px; }
            .logo h1 { color: #667eea; margin: 0; font-size: 28px; }
            .logo p { color: #666; margin: 5px 0 0 0; }
            .form-group { margin-bottom: 20px; }
            .form-group label { display: block; margin-bottom: 8px; font-weight: 600; color: #333; }
            .form-group input { 
                width: 100%; 
                padding: 15px; 
                border: 2px solid #e0e0e0; 
                border-radius: 8px; 
                font-size: 16px; 
                transition: border-color 0.3s; 
                box-sizing: border-box;
            }
            .form-group input:focus { border-color: #667eea; outline: none; }
            .btn { 
                width: 100%; 
                padding: 15px; 
                background: linear-gradient(45deg, #667eea, #764ba2); 
                color: white; 
                border: none; 
                border-radius: 8px; 
                font-size: 16px; 
                font-weight: 600; 
                cursor: pointer; 
                transition: transform 0.3s; 
            }
            .btn:hover { transform: translateY(-2px); }
            .links { text-align: center; margin-top: 20px; }
            .links a { color: #667eea; text-decoration: none; }
            .alert { 
                padding: 15px; 
                border-radius: 8px; 
                margin-bottom: 20px; 
                border-left: 4px solid #dc3545; 
                background: #f8d7da; 
                color: #721c24; 
            }
            .demo-credentials {
                background: #d4edda;
                border-left: 4px solid #28a745;
                color: #155724;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="login-container">
            <div class="logo">
                <h1>🧬 Genetic Coil Arena</h1>
                <p>Evolutionary Computing Platform</p>
            </div>
            
            <div class="demo-credentials">
                <strong>Demo Credentials:</strong><br>
                <strong>Admin:</strong> admin / admin123<br>
                <strong>Researcher:</strong> researcher / research123
            </div>
            
            <div id="error-message" style="display: none;" class="alert"></div>
            
            <form id="login-form">
                <div class="form-group">
                    <label for="username">Username:</label>
                    <input type="text" id="username" name="username" required>
                </div>
                <div class="form-group">
                    <label for="password">Password:</label>
                    <input type="password" id="password" name="password" required>
                </div>
                <button type="submit" class="btn">🔐 Login</button>
            </form>
            
            <div class="links">
                <a href="/auth/register">Don't have an account? Register here</a><br>
                <a href="/">← Back to Dashboard</a>
            </div>
        </div>
        
        <script>
            document.getElementById('login-form').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                formData.append('username', document.getElementById('username').value);
                formData.append('password', document.getElementById('password').value);
                
                try {
                    const response = await fetch('/auth/token', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        localStorage.setItem('access_token', data.access_token);
                        localStorage.setItem('username', data.username);
                        localStorage.setItem('role', data.role);
                        
                        alert(`Welcome back, ${data.username}! Redirecting to dashboard...`);
                        window.location.href = '/';
                    } else {
                        const error = await response.json();
                        document.getElementById('error-message').style.display = 'block';
                        document.getElementById('error-message').textContent = error.detail || 'Login failed';
                    }
                } catch (error) {
                    document.getElementById('error-message').style.display = 'block';
                    document.getElementById('error-message').textContent = 'Network error. Please try again.';
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, media_type="text/html")


@auth_router.get("/register")
async def register_page():
    """Serve the registration page."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Genetic Coil Arena - Register</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 0; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                min-height: 100vh; 
                display: flex; 
                align-items: center; 
                justify-content: center; 
            }
            .register-container { 
                background: white; 
                border-radius: 20px; 
                padding: 40px; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.1); 
                width: 100%; 
                max-width: 450px; 
            }
            .logo { text-align: center; margin-bottom: 30px; }
            .logo h1 { color: #667eea; margin: 0; font-size: 28px; }
            .logo p { color: #666; margin: 5px 0 0 0; }
            .form-group { margin-bottom: 20px; }
            .form-group label { display: block; margin-bottom: 8px; font-weight: 600; color: #333; }
            .form-group input, .form-group select { 
                width: 100%; 
                padding: 15px; 
                border: 2px solid #e0e0e0; 
                border-radius: 8px; 
                font-size: 16px; 
                transition: border-color 0.3s; 
                box-sizing: border-box;
            }
            .form-group input:focus, .form-group select:focus { border-color: #667eea; outline: none; }
            .btn { 
                width: 100%; 
                padding: 15px; 
                background: linear-gradient(45deg, #667eea, #764ba2); 
                color: white; 
                border: none; 
                border-radius: 8px; 
                font-size: 16px; 
                font-weight: 600; 
                cursor: pointer; 
                transition: transform 0.3s; 
            }
            .btn:hover { transform: translateY(-2px); }
            .links { text-align: center; margin-top: 20px; }
            .links a { color: #667eea; text-decoration: none; }
            .alert { 
                padding: 15px; 
                border-radius: 8px; 
                margin-bottom: 20px; 
            }
            .alert.error { border-left: 4px solid #dc3545; background: #f8d7da; color: #721c24; }
            .alert.success { border-left: 4px solid #28a745; background: #d4edda; color: #155724; }
        </style>
    </head>
    <body>
        <div class="register-container">
            <div class="logo">
                <h1>🧬 Genetic Coil Arena</h1>
                <p>Create Your Research Account</p>
            </div>
            
            <div id="message" style="display: none;" class="alert"></div>
            
            <form id="register-form">
                <div class="form-group">
                    <label for="username">Username:</label>
                    <input type="text" id="username" name="username" required minlength="3">
                </div>
                <div class="form-group">
                    <label for="email">Email:</label>
                    <input type="email" id="email" name="email" required>
                </div>
                <div class="form-group">
                    <label for="password">Password:</label>
                    <input type="password" id="password" name="password" required minlength="6">
                </div>
                <div class="form-group">
                    <label for="confirm_password">Confirm Password:</label>
                    <input type="password" id="confirm_password" name="confirm_password" required>
                </div>
                <div class="form-group">
                    <label for="role">Role:</label>
                    <select id="role" name="role">
                        <option value="researcher">Researcher</option>
                        <option value="admin">Administrator</option>
                    </select>
                </div>
                <button type="submit" class="btn">🚀 Create Account</button>
            </form>
            
            <div class="links">
                <a href="/auth/login">Already have an account? Login here</a><br>
                <a href="/">← Back to Dashboard</a>
            </div>
        </div>
        
        <script>
            document.getElementById('register-form').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const password = document.getElementById('password').value;
                const confirmPassword = document.getElementById('confirm_password').value;
                
                if (password !== confirmPassword) {
                    showMessage('Passwords do not match', 'error');
                    return;
                }
                
                const formData = {
                    username: document.getElementById('username').value,
                    email: document.getElementById('email').value,
                    password: password,
                    role: document.getElementById('role').value
                };
                
                try {
                    const response = await fetch('/auth/register', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(formData)
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        showMessage('Account created successfully! Redirecting to login...', 'success');
                        setTimeout(() => {
                            window.location.href = '/auth/login';
                        }, 2000);
                    } else {
                        const error = await response.json();
                        showMessage(error.detail || 'Registration failed', 'error');
                    }
                } catch (error) {
                    showMessage('Network error. Please try again.', 'error');
                }
            });
            
            function showMessage(text, type) {
                const messageDiv = document.getElementById('message');
                messageDiv.textContent = text;
                messageDiv.className = `alert ${type}`;
                messageDiv.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, media_type="text/html")


@auth_router.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and return access and refresh tokens."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    
    # Create refresh token
    refresh_token = None
    try:
        from ..core.auth import create_refresh_token
        refresh_token = create_refresh_token(user["username"])
    except Exception as e:
        logger.warning(f"Could not create refresh token: {e}")
    
    # Update last login if database is available
    try:
        if AUTH_AVAILABLE:
            db.update_last_login(user["username"])
    except Exception as e:
        logger.warning(f"Could not update last login: {e}")
    
    response = {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user["username"],
        "role": user["role"]
    }
    
    if refresh_token:
        response["refresh_token"] = refresh_token
    
    return response


@auth_router.post("/register")
async def register_user(user_data: dict):
    """Register a new user."""
    try:
        username = user_data["username"]
        email = user_data["email"]
        password = user_data["password"]
        role = user_data.get("role", "researcher")
        
        # Create user in memory store
        user = create_user(username, email, password, role)
        
        # Also create in database if available
        try:
            if AUTH_AVAILABLE:
                from ..core.auth import get_password_hash
                hashed_password = get_password_hash(password)
                db.create_user(username, email, hashed_password, role)
        except Exception as e:
            logger.warning(f"Could not create user in database: {e}")
        
        return {
            "message": "User created successfully",
            "username": username,
            "role": role
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@auth_router.get("/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    """Get current user information."""
    return {
        "username": current_user["username"],
        "email": current_user["email"],
        "role": current_user["role"],
        "created_at": current_user["created_at"]
    }


@auth_router.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Logout user by revoking the token."""
    if AUTH_AVAILABLE:
        success = revoke_token(credentials.credentials)
        if success:
            return {"message": "Logged out successfully"}
        else:
            return {"message": "Token already revoked or invalid"}
    return {"message": "Logged out successfully"}


@auth_router.post("/refresh")
async def refresh_token(refresh_token: str = Form(...)):
    """Refresh access token using refresh token."""
    if not AUTH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Authentication not available")
    
    tokens = refresh_access_token(refresh_token)
    
    if tokens is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    return tokens


@auth_router.post("/revoke-all")
async def revoke_all_tokens(current_user: dict = Depends(get_current_user)):
    """Revoke all tokens for the current user."""
    if not AUTH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Authentication not available")
    
    count = revoke_user_tokens(current_user["username"])
    
    return {
        "message": f"Revoked {count} tokens",
        "revoked_count": count
    }


@auth_router.get("/sessions")
async def get_user_sessions(current_user: dict = Depends(get_current_user)):
    """Get all active sessions for the current user."""
    if not AUTH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Authentication not available")
    
    token_manager = get_token_manager()
    sessions = token_manager.get_user_sessions(current_user["username"])
    
    return {
        "active_sessions": sessions,
        "session_count": len(sessions)
    }


@auth_router.get("/token-stats")
async def get_token_stats():
    """Get token management statistics (admin only)."""
    if not AUTH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Authentication not available")
    
    token_manager = get_token_manager()
    stats = token_manager.get_stats()
    
    return stats


@auth_router.get("/protected")
async def protected_route(current_user: dict = Depends(get_current_user)):
    """Example protected route."""
    return {
        "message": f"Hello {current_user['username']}, this is a protected route!",
        "user_role": current_user["role"]
    }
