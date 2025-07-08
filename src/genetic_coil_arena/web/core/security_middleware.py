"""
Security and input validation middleware for the Genetic Coil Arena.
Addresses senior developer review recommendations for production security.
"""

import time
import logging
from typing import Dict, Any
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import json
import re

logger = logging.getLogger(__name__)

# Rate limiting storage (in production, use Redis)
rate_limit_storage: Dict[str, Dict[str, Any]] = {}

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for input validation and rate limiting."""
    
    def __init__(
        self, 
        app: ASGIApp,
        max_requests: int = 100,
        time_window: int = 300,  # 5 minutes
        max_body_size: int = 10 * 1024 * 1024,  # 10MB
    ):
        super().__init__(app)
        self.max_requests = max_requests
        self.time_window = time_window
        self.max_body_size = max_body_size
    
    async def dispatch(self, request: Request, call_next):
        """Process request through security middleware."""
        
        # 1. Rate limiting
        client_ip = self._get_client_ip(request)
        if self._is_rate_limited(client_ip):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Too many requests"}
            )
        
        # 2. Content-Length validation
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_body_size:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"detail": "Request body too large"}
            )
        
        # 3. Basic header validation
        if not self._validate_headers(request):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Invalid headers"}
            )
        
        # 4. Process request
        start_time = time.time()
        try:
            response = await call_next(request)
            
            # 5. Log request details
            process_time = time.time() - start_time
            self._log_request(request, response, process_time)
            
            # 6. Add security headers
            self._add_security_headers(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Request processing error: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"}
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client is rate limited."""
        current_time = time.time()
        
        if client_ip not in rate_limit_storage:
            rate_limit_storage[client_ip] = {
                "requests": 1,
                "window_start": current_time
            }
            return False
        
        client_data = rate_limit_storage[client_ip]
        
        # Reset window if time has passed
        if current_time - client_data["window_start"] > self.time_window:
            client_data["requests"] = 1
            client_data["window_start"] = current_time
            return False
        
        # Check if limit exceeded
        if client_data["requests"] >= self.max_requests:
            return True
        
        client_data["requests"] += 1
        return False
    
    def _validate_headers(self, request: Request) -> bool:
        """Validate request headers for security issues."""
        
        # Check for suspicious user agents
        user_agent = request.headers.get("user-agent", "")
        suspicious_patterns = [
            r"(?i)(bot|crawler|spider|scraper)",
            r"(?i)(sqlmap|nmap|nikto)",
            r"(?i)(curl|wget|python-requests)"
        ]
        
        # Allow legitimate tools in development
        if any(re.search(pattern, user_agent) for pattern in suspicious_patterns):
            # Log suspicious activity but don't block in development
            logger.warning(f"Suspicious user agent: {user_agent}")
        
        # Validate content type for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if content_type and not any(ct in content_type.lower() for ct in [
                "application/json",
                "application/x-www-form-urlencoded",
                "multipart/form-data"
            ]):
                logger.warning(f"Unusual content type: {content_type}")
        
        return True
    
    def _log_request(self, request: Request, response: Response, process_time: float):
        """Log request details for monitoring."""
        logger.info(
            f"{request.method} {request.url.path} "
            f"- {response.status_code} "
            f"- {process_time:.3f}s "
            f"- {self._get_client_ip(request)}"
        )
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response."""
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "connect-src 'self'"
            ),
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value


class InputValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for validating request input data."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        """Validate request input."""
        
        # Only validate JSON requests
        if (request.method in ["POST", "PUT", "PATCH"] and 
            "application/json" in request.headers.get("content-type", "")):
            
            # Read and validate JSON body
            try:
                body = await request.body()
                if body:
                    json_data = json.loads(body)
                    if not self._validate_json_input(json_data):
                        return JSONResponse(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            content={"detail": "Invalid input data"}
                        )
                    
                    # Create new request with validated body
                    async def receive():
                        return {"type": "http.request", "body": body}
                    
                    request._receive = receive
                    
            except json.JSONDecodeError:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "Invalid JSON format"}
                )
            except Exception as e:
                logger.error(f"Input validation error: {e}")
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "Request processing error"}
                )
        
        return await call_next(request)
    
    def _validate_json_input(self, data: Any) -> bool:
        """Validate JSON input for security issues."""
        
        def validate_recursive(obj, depth=0):
            # Prevent deeply nested objects (DoS protection)
            if depth > 10:
                return False
            
            if isinstance(obj, dict):
                # Check for suspicious keys
                for key in obj.keys():
                    if not isinstance(key, str):
                        return False
                    if len(key) > 100:  # Prevent extremely long keys
                        return False
                    if any(char in key for char in ['<', '>', '&', '"', "'"]):
                        return False
                
                # Validate values recursively
                for value in obj.values():
                    if not validate_recursive(value, depth + 1):
                        return False
            
            elif isinstance(obj, list):
                # Prevent extremely large lists
                if len(obj) > 1000:
                    return False
                
                # Validate items recursively
                for item in obj:
                    if not validate_recursive(item, depth + 1):
                        return False
            
            elif isinstance(obj, str):
                # Prevent extremely long strings
                if len(obj) > 10000:
                    return False
                
                # Check for basic XSS patterns
                suspicious_patterns = [
                    r'<script[^>]*>',
                    r'javascript:',
                    r'on\w+\s*=',
                    r'expression\s*\(',
                    r'@import',
                    r'eval\s*\('
                ]
                
                for pattern in suspicious_patterns:
                    if re.search(pattern, obj, re.IGNORECASE):
                        logger.warning(f"Suspicious pattern detected: {pattern}")
                        return False
            
            return True
        
        return validate_recursive(data)


def create_security_middleware(app: ASGIApp) -> ASGIApp:
    """Create and configure security middleware stack."""
    
    # Add input validation middleware
    app = InputValidationMiddleware(app)
    
    # Add security middleware
    app = SecurityMiddleware(
        app,
        max_requests=100,  # requests per window
        time_window=300,   # 5 minutes
        max_body_size=10 * 1024 * 1024  # 10MB
    )
    
    return app
