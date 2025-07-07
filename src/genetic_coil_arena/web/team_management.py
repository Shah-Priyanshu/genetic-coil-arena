"""
Phase 4C: Team Management System
User organization, role-based access control, and experiment ownership.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles within teams."""
    VIEWER = "viewer"
    RESEARCHER = "researcher"
    ADMIN = "admin"
    OWNER = "owner"


class TeamPermission(Enum):
    """Team-level permissions."""
    VIEW_EXPERIMENTS = "view_experiments"
    CREATE_EXPERIMENTS = "create_experiments"
    EDIT_EXPERIMENTS = "edit_experiments"
    DELETE_EXPERIMENTS = "delete_experiments"
    MANAGE_MEMBERS = "manage_members"
    MANAGE_SETTINGS = "manage_settings"


@dataclass
class TeamMember:
    """Team member with role and permissions."""
    user_id: str
    username: str
    email: str
    role: UserRole
    joined_at: datetime = field(default_factory=datetime.now)
    last_active: Optional[datetime] = None
    permissions: Set[TeamPermission] = field(default_factory=set)
    
    def __post_init__(self):
        """Set default permissions based on role."""
        if not self.permissions:
            self.permissions = self._get_default_permissions()
    
    def _get_default_permissions(self) -> Set[TeamPermission]:
        """Get default permissions for the user's role."""
        base_permissions = {
            UserRole.VIEWER: {
                TeamPermission.VIEW_EXPERIMENTS
            },
            UserRole.RESEARCHER: {
                TeamPermission.VIEW_EXPERIMENTS,
                TeamPermission.CREATE_EXPERIMENTS,
                TeamPermission.EDIT_EXPERIMENTS
            },
            UserRole.ADMIN: {
                TeamPermission.VIEW_EXPERIMENTS,
                TeamPermission.CREATE_EXPERIMENTS,
                TeamPermission.EDIT_EXPERIMENTS,
                TeamPermission.DELETE_EXPERIMENTS,
                TeamPermission.MANAGE_MEMBERS
            },
            UserRole.OWNER: set(TeamPermission)  # All permissions
        }
        return base_permissions.get(self.role, set())
    
    def has_permission(self, permission: TeamPermission) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions
    
    def can_manage_user(self, other_role: UserRole) -> bool:
        """Check if user can manage another user based on role hierarchy."""
        role_hierarchy = {
            UserRole.OWNER: 4,
            UserRole.ADMIN: 3,
            UserRole.RESEARCHER: 2,
            UserRole.VIEWER: 1
        }
        return role_hierarchy.get(self.role, 0) > role_hierarchy.get(other_role, 0)


@dataclass
class Team:
    """Team with members and experiment access control."""
    id: str
    name: str
    description: str
    owner_id: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    members: List[TeamMember] = field(default_factory=list)
    experiment_ids: Set[str] = field(default_factory=set)
    settings: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    
    def add_member(self, user_id: str, username: str, email: str, 
                   role: UserRole = UserRole.RESEARCHER) -> bool:
        """Add a new member to the team."""
        try:
            # Check if user is already a member
            if self.get_member(user_id):
                logger.warning(f"User {user_id} is already a member of team {self.id}")
                return False
            
            member = TeamMember(
                user_id=user_id,
                username=username,
                email=email,
                role=role
            )
            
            self.members.append(member)
            self.updated_at = datetime.now()
            
            logger.info(f"Added user {username} to team {self.name} with role {role.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding member to team: {e}")
            return False
    
    def remove_member(self, user_id: str, removed_by: str) -> bool:
        """Remove a member from the team."""
        try:
            member = self.get_member(user_id)
            if not member:
                logger.warning(f"User {user_id} is not a member of team {self.id}")
                return False
            
            # Check if removing owner
            if member.role == UserRole.OWNER:
                logger.error("Cannot remove team owner")
                return False
            
            # Check permissions of the user doing the removal
            remover = self.get_member(removed_by)
            if not remover or not remover.can_manage_user(member.role):
                logger.error(f"User {removed_by} does not have permission to remove {user_id}")
                return False
            
            self.members = [m for m in self.members if m.user_id != user_id]
            self.updated_at = datetime.now()
            
            logger.info(f"Removed user {member.username} from team {self.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing member from team: {e}")
            return False
    
    def update_member_role(self, user_id: str, new_role: UserRole, updated_by: str) -> bool:
        """Update a member's role."""
        try:
            member = self.get_member(user_id)
            if not member:
                logger.warning(f"User {user_id} is not a member of team {self.id}")
                return False
            
            # Check permissions
            updater = self.get_member(updated_by)
            if not updater or not updater.can_manage_user(member.role):
                logger.error(f"User {updated_by} does not have permission to update {user_id}")
                return False
            
            # Prevent changing owner role
            if member.role == UserRole.OWNER and new_role != UserRole.OWNER:
                logger.error("Cannot change owner role")
                return False
            
            old_role = member.role
            member.role = new_role
            member.permissions = member._get_default_permissions()
            self.updated_at = datetime.now()
            
            logger.info(f"Updated {member.username} role from {old_role.value} to {new_role.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating member role: {e}")
            return False
    
    def get_member(self, user_id: str) -> Optional[TeamMember]:
        """Get team member by user ID."""
        return next((m for m in self.members if m.user_id == user_id), None)
    
    def user_has_permission(self, user_id: str, permission: TeamPermission) -> bool:
        """Check if user has specific permission in this team."""
        member = self.get_member(user_id)
        return bool(member and member.has_permission(permission))
    
    def add_experiment(self, experiment_id: str) -> bool:
        """Add experiment to team's experiments."""
        try:
            self.experiment_ids.add(experiment_id)
            self.updated_at = datetime.now()
            logger.info(f"Added experiment {experiment_id} to team {self.name}")
            return True
        except Exception as e:
            logger.error(f"Error adding experiment to team: {e}")
            return False
    
    def remove_experiment(self, experiment_id: str) -> bool:
        """Remove experiment from team's experiments."""
        try:
            self.experiment_ids.discard(experiment_id)
            self.updated_at = datetime.now()
            logger.info(f"Removed experiment {experiment_id} from team {self.name}")
            return True
        except Exception as e:
            logger.error(f"Error removing experiment from team: {e}")
            return False
    
    def get_team_stats(self) -> Dict[str, Any]:
        """Get team statistics."""
        role_counts = {}
        for role in UserRole:
            role_counts[role.value] = len([m for m in self.members if m.role == role])
        
        active_members = len([m for m in self.members if m.last_active and 
                             (datetime.now() - m.last_active).days <= 30])
        
        return {
            "total_members": len(self.members),
            "active_members": active_members,
            "total_experiments": len(self.experiment_ids),
            "role_distribution": role_counts,
            "created_days_ago": (datetime.now() - self.created_at).days,
            "last_updated": self.updated_at.isoformat()
        }


class TeamManager:
    """Manages teams, members, and permissions."""
    
    def __init__(self):
        self.teams: Dict[str, Team] = {}
        self.user_teams: Dict[str, Set[str]] = {}  # user_id -> set of team_ids
        # Import here to avoid circular imports
        try:
            from .file_storage import file_storage
            self.storage = file_storage
            self._load_teams_from_storage()
        except ImportError:
            logger.warning("File storage not available, using in-memory storage")
            self.storage = None
    
    def _load_teams_from_storage(self):
        """Load teams from file storage."""
        if not self.storage:
            return
        
        try:
            # Load all team files using the file storage system
            for team_file in self.storage.teams_dir.glob("*.json"):
                try:
                    with open(team_file, 'r', encoding='utf-8') as f:
                        team_data = json.load(f)
                    
                    if team_data:
                        team = self._deserialize_team(team_data)
                        if team:
                            self.teams[team.id] = team
                            # Build user-team mapping
                            for member in team.members:
                                if member.user_id not in self.user_teams:
                                    self.user_teams[member.user_id] = set()
                                self.user_teams[member.user_id].add(team.id)
                except Exception as e:
                    logger.error(f"Error loading team file {team_file}: {e}")
            
            logger.info(f"Loaded {len(self.teams)} teams from storage")
        except Exception as e:
            logger.error(f"Error loading teams from storage: {e}")
    
    def _serialize_team(self, team: Team) -> Dict[str, Any]:
        """Convert team object to dictionary for storage."""
        return {
            "id": team.id,
            "name": team.name,
            "description": team.description,
            "owner_id": team.owner_id,
            "created_at": team.created_at.isoformat(),
            "updated_at": team.updated_at.isoformat(),
            "is_active": team.is_active,
            "members": [
                {
                    "user_id": m.user_id,
                    "username": m.username,
                    "email": m.email,
                    "role": m.role.value,
                    "joined_at": m.joined_at.isoformat(),
                    "last_active": m.last_active.isoformat() if m.last_active else None,
                    "permissions": [p.value for p in m.permissions]
                }
                for m in team.members
            ],
            "experiment_ids": list(team.experiment_ids)
        }
    
    def _deserialize_team(self, team_data: Dict[str, Any]) -> Optional[Team]:
        """Convert dictionary from storage to team object."""
        try:
            # Create team members
            members = []
            for member_data in team_data.get("members", []):
                member = TeamMember(
                    user_id=member_data["user_id"],
                    username=member_data["username"],
                    email=member_data["email"],
                    role=UserRole(member_data["role"]),
                    joined_at=datetime.fromisoformat(member_data["joined_at"]),
                    last_active=datetime.fromisoformat(member_data["last_active"]) if member_data.get("last_active") else None,
                    permissions={TeamPermission(p) for p in member_data.get("permissions", [])}
                )
                members.append(member)
            
            # Create team
            team = Team(
                id=team_data["id"],
                name=team_data["name"],
                description=team_data["description"],
                owner_id=team_data["owner_id"],
                created_at=datetime.fromisoformat(team_data["created_at"]),
                updated_at=datetime.fromisoformat(team_data["updated_at"]),
                is_active=team_data.get("is_active", True)
            )
            
            # Set members and experiments
            team.members = members
            team.experiment_ids = set(team_data.get("experiment_ids", []))
            
            return team
        except Exception as e:
            logger.error(f"Error deserializing team {team_data.get('id', 'unknown')}: {e}")
            return None
    
    def _save_team(self, team: Team):
        """Save team to storage."""
        if not self.storage:
            return
        
        try:
            team_data = self._serialize_team(team)
            self.storage.save_team(team_data)
        except Exception as e:
            logger.error(f"Error saving team {team.id}: {e}")
    
    def create_team(self, team_id: str, name: str, description: str, 
                   owner_id: str, owner_username: str, owner_email: str) -> Optional[Team]:
        """Create a new team."""
        try:
            if team_id in self.teams:
                logger.warning(f"Team {team_id} already exists")
                return None
            
            team = Team(
                id=team_id,
                name=name,
                description=description,
                owner_id=owner_id
            )
            
            # Add owner as team member
            team.add_member(owner_id, owner_username, owner_email, UserRole.OWNER)
            
            self.teams[team_id] = team
            
            # Update user-team mapping
            if owner_id not in self.user_teams:
                self.user_teams[owner_id] = set()
            self.user_teams[owner_id].add(team_id)
            
            # Save to storage
            self._save_team(team)
            
            logger.info(f"Created team {name} with owner {owner_username}")
            return team
            
        except Exception as e:
            logger.error(f"Error creating team: {e}")
            return None
    
    def get_team(self, team_id: str) -> Optional[Team]:
        """Get team by ID."""
        return self.teams.get(team_id)
    
    def get_user_teams(self, user_id: str) -> List[Team]:
        """Get all teams for a user."""
        team_ids = self.user_teams.get(user_id, set())
        return [self.teams[tid] for tid in team_ids if tid in self.teams]
    
    def join_team(self, team_id: str, user_id: str, username: str, email: str,
                  invited_by: str, role: UserRole = UserRole.RESEARCHER) -> bool:
        """Add user to team (requires invitation)."""
        try:
            team = self.get_team(team_id)
            if not team:
                logger.warning(f"Team {team_id} not found")
                return False
            
            # Check if inviter has permission
            if not team.user_has_permission(invited_by, TeamPermission.MANAGE_MEMBERS):
                logger.error(f"User {invited_by} does not have permission to invite members")
                return False
            
            # Add member to team
            if team.add_member(user_id, username, email, role):
                # Update user-team mapping
                if user_id not in self.user_teams:
                    self.user_teams[user_id] = set()
                self.user_teams[user_id].add(team_id)
                
                # Save to storage
                self._save_team(team)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error joining team: {e}")
            return False
    
    def leave_team(self, team_id: str, user_id: str) -> bool:
        """Remove user from team."""
        try:
            team = self.get_team(team_id)
            if not team:
                return False
            
            # Check if user is owner
            member = team.get_member(user_id)
            if member and member.role == UserRole.OWNER:
                logger.error("Team owner cannot leave team")
                return False
            
            # Remove from team
            if team.remove_member(user_id, user_id):
                # Update user-team mapping
                if user_id in self.user_teams:
                    self.user_teams[user_id].discard(team_id)
                
                # Save to storage
                self._save_team(team)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error leaving team: {e}")
            return False
    
    def delete_team(self, team_id: str, deleted_by: str) -> bool:
        """Delete a team (owner only)."""
        try:
            team = self.get_team(team_id)
            if not team:
                return False
            
            # Check if user is owner
            member = team.get_member(deleted_by)
            if not member or member.role != UserRole.OWNER:
                logger.error(f"User {deleted_by} is not the team owner")
                return False
            
            # Remove team from all users' mappings
            for member in team.members:
                if member.user_id in self.user_teams:
                    self.user_teams[member.user_id].discard(team_id)
            
            # Delete team from storage
            self.storage.delete_team(team_id)
            
            # Delete team from memory
            del self.teams[team_id]
            
            logger.info(f"Deleted team {team.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting team: {e}")
            return False
    
    def check_experiment_access(self, user_id: str, experiment_id: str, 
                              permission: TeamPermission) -> bool:
        """Check if user has permission to access experiment through team membership."""
        user_teams = self.get_user_teams(user_id)
        
        for team in user_teams:
            if (experiment_id in team.experiment_ids and 
                team.user_has_permission(user_id, permission)):
                return True
        
        return False
    
    def get_team_experiments(self, team_id: str, user_id: str) -> List[str]:
        """Get experiments accessible to user through team."""
        team = self.get_team(team_id)
        if not team:
            return []
        
        if not team.user_has_permission(user_id, TeamPermission.VIEW_EXPERIMENTS):
            return []
        
        return list(team.experiment_ids)
    
    def export_team_data(self, team_id: str) -> Dict[str, Any]:
        """Export team data for backup/analysis."""
        team = self.get_team(team_id)
        if not team:
            return {}
        
        return {
            "team_info": {
                "id": team.id,
                "name": team.name,
                "description": team.description,
                "created_at": team.created_at.isoformat(),
                "updated_at": team.updated_at.isoformat()
            },
            "members": [
                {
                    "user_id": m.user_id,
                    "username": m.username,
                    "email": m.email,
                    "role": m.role.value,
                    "joined_at": m.joined_at.isoformat(),
                    "permissions": [p.value for p in m.permissions]
                }
                for m in team.members
            ],
            "experiments": list(team.experiment_ids),
            "statistics": team.get_team_stats()
        }


# Global team manager instance
team_manager = TeamManager()
