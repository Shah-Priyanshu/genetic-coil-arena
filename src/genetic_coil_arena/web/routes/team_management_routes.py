"""
Phase 4C: Team Management REST API
FastAPI routes for team operations and collaboration features.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, validator
from datetime import datetime
import logging

from .team_management import (
    team_manager, Team, TeamMember, UserRole, TeamPermission
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/teams", tags=["Team Management"])


# Pydantic models for request/response
class CreateTeamRequest(BaseModel):
    name: str
    description: str
    
    @validator('name')
    def validate_name(cls, v):
        if len(v.strip()) < 3:
            raise ValueError('Team name must be at least 3 characters long')
        return v.strip()


class InviteMemberRequest(BaseModel):
    user_id: str
    username: str
    email: str
    role: str = "researcher"
    
    @validator('role')
    def validate_role(cls, v):
        try:
            UserRole(v.lower())
            return v.lower()
        except ValueError:
            raise ValueError('Invalid role. Must be: viewer, researcher, admin, owner')


class UpdateMemberRoleRequest(BaseModel):
    user_id: str
    new_role: str
    
    @validator('new_role')
    def validate_role(cls, v):
        try:
            UserRole(v.lower())
            return v.lower()
        except ValueError:
            raise ValueError('Invalid role. Must be: viewer, researcher, admin, owner')


class TeamResponse(BaseModel):
    id: str
    name: str
    description: str
    owner_id: str
    created_at: str
    updated_at: str
    member_count: int
    experiment_count: int
    is_active: bool


class MemberResponse(BaseModel):
    user_id: str
    username: str
    email: str
    role: str
    joined_at: str
    last_active: Optional[str] = None
    permissions: List[str]


class TeamDetailResponse(BaseModel):
    team_info: TeamResponse
    members: List[MemberResponse]
    experiments: List[str]
    statistics: Dict[str, Any]


# Mock authentication - replace with real auth
def get_current_user():
    """Mock user authentication."""
    return {
        "id": "user_123",
        "username": "demo_user",
        "email": "demo@example.com"
    }


@router.post("/", response_model=TeamResponse)
async def create_team(
    request: CreateTeamRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a new team."""
    try:
        team_id = f"team_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{current_user['id']}"
        
        team = team_manager.create_team(
            team_id=team_id,
            name=request.name,
            description=request.description,
            owner_id=current_user["id"],
            owner_username=current_user["username"],
            owner_email=current_user["email"]
        )
        
        if not team:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create team"
            )
        
        return TeamResponse(
            id=team.id,
            name=team.name,
            description=team.description,
            owner_id=team.owner_id,
            created_at=team.created_at.isoformat(),
            updated_at=team.updated_at.isoformat(),
            member_count=len(team.members),
            experiment_count=len(team.experiment_ids),
            is_active=team.is_active
        )
        
    except Exception as e:
        logger.error(f"Error creating team: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/", response_model=List[TeamResponse])
async def get_user_teams(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get all teams for the current user."""
    try:
        teams = team_manager.get_user_teams(current_user["id"])
        
        return [
            TeamResponse(
                id=team.id,
                name=team.name,
                description=team.description,
                owner_id=team.owner_id,
                created_at=team.created_at.isoformat(),
                updated_at=team.updated_at.isoformat(),
                member_count=len(team.members),
                experiment_count=len(team.experiment_ids),
                is_active=team.is_active
            )
            for team in teams
        ]
        
    except Exception as e:
        logger.error(f"Error getting user teams: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/{team_id}", response_model=TeamDetailResponse)
async def get_team_details(
    team_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get detailed information about a team."""
    try:
        team = team_manager.get_team(team_id)
        if not team:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Team not found"
            )
        
        # Check if user is a member
        member = team.get_member(current_user["id"])
        if not member:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You are not a member of this team"
            )
        
        # Prepare response
        team_info = TeamResponse(
            id=team.id,
            name=team.name,
            description=team.description,
            owner_id=team.owner_id,
            created_at=team.created_at.isoformat(),
            updated_at=team.updated_at.isoformat(),
            member_count=len(team.members),
            experiment_count=len(team.experiment_ids),
            is_active=team.is_active
        )
        
        members = [
            MemberResponse(
                user_id=m.user_id,
                username=m.username,
                email=m.email,
                role=m.role.value,
                joined_at=m.joined_at.isoformat(),
                last_active=m.last_active.isoformat() if m.last_active else None,
                permissions=[p.value for p in m.permissions]
            )
            for m in team.members
        ]
        
        # Only show experiments if user has permission
        experiments = []
        if member.has_permission(TeamPermission.VIEW_EXPERIMENTS):
            experiments = list(team.experiment_ids)
        
        statistics = team.get_team_stats()
        
        return TeamDetailResponse(
            team_info=team_info,
            members=members,
            experiments=experiments,
            statistics=statistics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting team details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/{team_id}/members")
async def invite_member(
    team_id: str,
    request: InviteMemberRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Invite a new member to the team."""
    try:
        team = team_manager.get_team(team_id)
        if not team:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Team not found"
            )
        
        # Check permissions
        if not team.user_has_permission(current_user["id"], TeamPermission.MANAGE_MEMBERS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You don't have permission to manage members"
            )
        
        # Convert role string to enum
        role = UserRole(request.role)
        
        success = team_manager.join_team(
            team_id=team_id,
            user_id=request.user_id,
            username=request.username,
            email=request.email,
            invited_by=current_user["id"],
            role=role
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to invite member"
            )
        
        return JSONResponse(
            content={
                "message": f"Successfully invited {request.username} to team {team.name}",
                "member": {
                    "user_id": request.user_id,
                    "username": request.username,
                    "role": request.role
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error inviting member: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.delete("/{team_id}/members/{user_id}")
async def remove_member(
    team_id: str,
    user_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Remove a member from the team."""
    try:
        team = team_manager.get_team(team_id)
        if not team:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Team not found"
            )
        
        # Check permissions
        if not team.user_has_permission(current_user["id"], TeamPermission.MANAGE_MEMBERS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You don't have permission to manage members"
            )
        
        member = team.get_member(user_id)
        if not member:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Member not found in team"
            )
        
        success = team.remove_member(user_id, current_user["id"])
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to remove member"
            )
        
        return JSONResponse(
            content={
                "message": f"Successfully removed {member.username} from team {team.name}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing member: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.put("/{team_id}/members/{user_id}/role")
async def update_member_role(
    team_id: str,
    user_id: str,
    request: UpdateMemberRoleRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Update a member's role in the team."""
    try:
        team = team_manager.get_team(team_id)
        if not team:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Team not found"
            )
        
        # Check permissions
        if not team.user_has_permission(current_user["id"], TeamPermission.MANAGE_MEMBERS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You don't have permission to manage members"
            )
        
        # Convert role string to enum
        new_role = UserRole(request.new_role)
        
        success = team.update_member_role(user_id, new_role, current_user["id"])
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update member role"
            )
        
        member = team.get_member(user_id)
        
        return JSONResponse(
            content={
                "message": f"Successfully updated {member.username}'s role to {new_role.value}",
                "member": {
                    "user_id": user_id,
                    "username": member.username,
                    "role": new_role.value
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating member role: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/{team_id}/leave")
async def leave_team(
    team_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Leave a team (user removes themselves)."""
    try:
        team = team_manager.get_team(team_id)
        if not team:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Team not found"
            )
        
        member = team.get_member(current_user["id"])
        if not member:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="You are not a member of this team"
            )
        
        success = team_manager.leave_team(team_id, current_user["id"])
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to leave team"
            )
        
        return JSONResponse(
            content={
                "message": f"Successfully left team {team.name}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error leaving team: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.delete("/{team_id}")
async def delete_team(
    team_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete a team (owner only)."""
    try:
        team = team_manager.get_team(team_id)
        if not team:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Team not found"
            )
        
        success = team_manager.delete_team(team_id, current_user["id"])
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: Only team owner can delete the team"
            )
        
        return JSONResponse(
            content={
                "message": f"Successfully deleted team {team.name}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting team: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/{team_id}/experiments")
async def get_team_experiments(
    team_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get experiments accessible through team membership."""
    try:
        experiments = team_manager.get_team_experiments(team_id, current_user["id"])
        
        return JSONResponse(
            content={
                "team_id": team_id,
                "experiments": experiments,
                "count": len(experiments)
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting team experiments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/{team_id}/experiments/{experiment_id}")
async def add_experiment_to_team(
    team_id: str,
    experiment_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Add an experiment to team's accessible experiments."""
    try:
        team = team_manager.get_team(team_id)
        if not team:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Team not found"
            )
        
        # Check permissions
        if not team.user_has_permission(current_user["id"], TeamPermission.MANAGE_SETTINGS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You don't have permission to manage team experiments"
            )
        
        success = team.add_experiment(experiment_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to add experiment to team"
            )
        
        return JSONResponse(
            content={
                "message": f"Successfully added experiment {experiment_id} to team {team.name}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding experiment to team: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.delete("/{team_id}/experiments/{experiment_id}")
async def remove_experiment_from_team(
    team_id: str,
    experiment_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Remove an experiment from team's accessible experiments."""
    try:
        team = team_manager.get_team(team_id)
        if not team:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Team not found"
            )
        
        # Check permissions
        if not team.user_has_permission(current_user["id"], TeamPermission.MANAGE_SETTINGS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You don't have permission to manage team experiments"
            )
        
        success = team.remove_experiment(experiment_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to remove experiment from team"
            )
        
        return JSONResponse(
            content={
                "message": f"Successfully removed experiment {experiment_id} from team {team.name}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing experiment from team: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/{team_id}/export")
async def export_team_data(
    team_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Export team data for backup/analysis."""
    try:
        team = team_manager.get_team(team_id)
        if not team:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Team not found"
            )
        
        # Check if user is admin or owner
        member = team.get_member(current_user["id"])
        if not member or member.role not in [UserRole.ADMIN, UserRole.OWNER]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: Only admins and owners can export team data"
            )
        
        export_data = team_manager.export_team_data(team_id)
        
        return JSONResponse(content=export_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting team data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Health check for team management service
@router.get("/health")
async def team_management_health():
    """Health check for team management service."""
    return {
        "status": "healthy",
        "service": "Team Management API",
        "timestamp": datetime.now().isoformat(),
        "teams_count": len(team_manager.teams),
        "total_members": sum(len(team.members) for team in team_manager.teams.values())
    }
