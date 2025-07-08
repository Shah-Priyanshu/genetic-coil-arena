"""
Phase 4C: Team Management UI Router
FastAPI router for serving team management interface.
"""

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
import logging
from typing import Dict, Any, Optional

from .team_management_ui import TeamManagementUI
from ..core.auth import get_current_user

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/teams/ui", tags=["team-ui"])

# Initialize UI generator
team_ui = TeamManagementUI()


@router.get("/", response_class=HTMLResponse, summary="Team Management Dashboard")
async def team_dashboard(request: Request, current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Serve the main team management dashboard.
    
    Provides:
    - Team overview and selection
    - Member management interface
    - Experiment sharing controls
    - Collaboration tools
    """
    try:
        # Generate dashboard HTML
        dashboard_html = team_ui.get_team_dashboard(user_id=current_user.get('id', ''))
        
        logger.info(f"Serving team dashboard for user {current_user.get('username', 'unknown')}")
        return HTMLResponse(content=dashboard_html)
        
    except Exception as e:
        logger.error(f"Error serving team dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to load team dashboard")


@router.get("/team/{team_id}", response_class=HTMLResponse, summary="Team Details Page")
async def team_details(
    team_id: str, 
    request: Request, 
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Serve detailed team view with member management.
    
    Args:
        team_id: Team identifier
        
    Returns:
        HTML page with team details and management tools
    """
    try:
        # Fetch team data from database
        from .team_management import team_manager
        
        team_data = team_manager.get_team(team_id)
        if not team_data:
            raise HTTPException(status_code=404, detail="Team not found")
        
        # Get team members from the team object
        members = team_data.members
        
        # Build member list HTML
        members_html = ""
        for member in members:
            role_badge_color = {
                "owner": "#dc3545",
                "admin": "#fd7e14", 
                "member": "#28a745",
                "viewer": "#6c757d"
            }.get(member.role.value, '#6c757d')
            
            members_html += f"""
            <div class="member-card">
                <div class="member-info">
                    <h4>{member.user_id}</h4>
                    <span class="role-badge" style="background-color: {role_badge_color}">
                        {member.role.value.title()}
                    </span>
                </div>
                <div class="member-meta">
                    <small>Joined: {member.joined_at.strftime('%Y-%m-%d') if member.joined_at else 'Unknown'}</small>
                </div>
            </div>
            """
        
        if not members_html:
            members_html = "<p class='no-data'>No team members found.</p>"
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Team Details - Genetic Coil Arena</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(10px);
                    border-radius: 15px;
                    padding: 30px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                }}
                .header {{
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 2px solid #e1e5e9;
                }}
                .team-title {{
                    color: #764ba2;
                    font-size: 2.5rem;
                    margin-bottom: 10px;
                }}
                .team-description {{
                    color: #666;
                    font-size: 1.1rem;
                    margin-bottom: 20px;
                }}
                .team-meta {{
                    display: flex;
                    gap: 20px;
                    flex-wrap: wrap;
                }}
                .team-meta-item {{
                    background: #f8f9fa;
                    padding: 10px 15px;
                    border-radius: 8px;
                    font-size: 0.9rem;
                }}
                .members-section {{
                    margin-top: 30px;
                }}
                .section-title {{
                    color: #333;
                    font-size: 1.5rem;
                    margin-bottom: 20px;
                    border-left: 4px solid #667eea;
                    padding-left: 15px;
                }}
                .member-card {{
                    background: #f8f9fa;
                    border-radius: 8px;
                    padding: 15px;
                    margin-bottom: 15px;
                    border-left: 4px solid #28a745;
                }}
                .member-info {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                .member-info h4 {{
                    margin: 0;
                    color: #333;
                }}
                .role-badge {{
                    color: white;
                    padding: 4px 12px;
                    border-radius: 15px;
                    font-size: 0.8rem;
                    font-weight: bold;
                }}
                .member-meta {{
                    margin-top: 10px;
                    color: #666;
                }}
                .no-data {{
                    text-align: center;
                    color: #666;
                    font-style: italic;
                    padding: 30px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 class="team-title">{team_data.name}</h1>
                    <div class="team-description">{team_data.description or 'No description available.'}</div>
                    <div class="team-meta">
                        <div class="team-meta-item">
                            <strong>Created:</strong> {team_data.created_at.strftime('%Y-%m-%d') if team_data.created_at else 'Unknown'}
                        </div>
                        <div class="team-meta-item">
                            <strong>Members:</strong> {len(team_data.members)}
                        </div>
                        <div class="team-meta-item">
                            <strong>ID:</strong> {team_data.id}
                        </div>
                    </div>
                </div>
                
                <div class="members-section">
                    <h2 class="section-title">Team Members</h2>
                    {members_html}
                </div>
            </div>
        </body>
        </html>
        """
        
        logger.info(f"Serving team details for team {team_id}")
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"Error serving team details: {e}")
        raise HTTPException(status_code=500, detail="Failed to load team details")


@router.get("/experiment/share", response_class=HTMLResponse, summary="Experiment Sharing Interface")
async def experiment_share_ui(
    request: Request, 
    experiment_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Serve experiment sharing interface.
    
    Args:
        experiment_id: Optional experiment ID to pre-select
        
    Returns:
        HTML interface for sharing experiments with teams
    """
    try:
        # Generate experiment sharing form
        share_form_html = team_ui.get_experiment_share_form()
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Share Experiment - Genetic Coil Arena</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(10px);
                    border-radius: 15px;
                    padding: 30px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .header h1 {{
                    color: #764ba2;
                    margin-bottom: 10px;
                }}
                .form-group {{
                    margin-bottom: 20px;
                }}
                .form-label {{
                    display: block;
                    margin-bottom: 5px;
                    font-weight: 600;
                    color: #333;
                }}
                .form-input, .form-select {{
                    width: 100%;
                    padding: 12px;
                    border: 2px solid #e1e5e9;
                    border-radius: 8px;
                    font-size: 1rem;
                    transition: border-color 0.3s ease;
                }}
                .form-input:focus, .form-select:focus {{
                    outline: none;
                    border-color: #667eea;
                }}
                .btn {{
                    padding: 12px 24px;
                    border: none;
                    border-radius: 25px;
                    cursor: pointer;
                    font-weight: 600;
                    transition: all 0.3s ease;
                }}
                .btn-primary {{
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    color: white;
                }}
                .btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
                }}
                .nav-back {{
                    display: inline-block;
                    margin-bottom: 20px;
                    color: #667eea;
                    text-decoration: none;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <a href="/teams/ui" class="nav-back">← Back to Teams</a>
                
                <div class="header">
                    <h1>🤝 Share Experiment</h1>
                    <p>Collaborate with your team on research experiments</p>
                </div>
                
                {share_form_html}
                
                <script>
                    // Load teams for sharing
                    async function loadTeamsForSharing() {{
                        try {{
                            const response = await fetch('/api/teams');
                            if (response.ok) {{
                                const teams = await response.json();
                                const select = document.querySelector('select[name="team_id"]');
                                
                                teams.forEach(team => {{
                                    const option = document.createElement('option');
                                    option.value = team.id;
                                    option.textContent = team.name;
                                    select.appendChild(option);
                                }});
                            }}
                        }} catch (error) {{
                            console.error('Failed to load teams:', error);
                        }}
                    }}
                    
                    // Handle form submission
                    document.getElementById('share-experiment-form').addEventListener('submit', async (e) => {{
                        e.preventDefault();
                        
                        const formData = new FormData(e.target);
                        const shareData = {{
                            team_id: formData.get('team_id'),
                            share_level: formData.get('share_level'),
                            message: formData.get('message'),
                            notify_team: formData.has('notify_team')
                        }};
                        
                        try {{
                            const response = await fetch('/api/experiments/share', {{
                                method: 'POST',
                                headers: {{ 'Content-Type': 'application/json' }},
                                body: JSON.stringify(shareData)
                            }});
                            
                            if (response.ok) {{
                                alert('Experiment shared successfully!');
                                window.location.href = '/teams/ui';
                            }} else {{
                                const error = await response.json();
                                alert('Failed to share experiment: ' + error.detail);
                            }}
                        }} catch (error) {{
                            console.error('Error sharing experiment:', error);
                            alert('Failed to share experiment');
                        }}
                    }});
                    
                    // Load teams when page loads
                    document.addEventListener('DOMContentLoaded', loadTeamsForSharing);
                </script>
            </div>
        </body>
        </html>
        """
        
        logger.info(f"Serving experiment sharing interface for user {current_user.get('username', 'unknown')}")
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"Error serving experiment sharing interface: {e}")
        raise HTTPException(status_code=500, detail="Failed to load sharing interface")


@router.get("/health", summary="UI Router Health Check")
async def ui_health_check() -> Dict[str, Any]:
    """Check if the team management UI router is working."""
    return {
        "status": "healthy",
        "service": "team_management_ui_router",
        "routes_available": [
            "/teams/ui/",
            "/teams/ui/team/{team_id}",
            "/teams/ui/experiment/share"
        ]
    }
