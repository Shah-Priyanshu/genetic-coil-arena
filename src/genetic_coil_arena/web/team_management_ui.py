"""
Phase 4C: Team Management UI
Interactive interface for team collaboration and management.
"""

from typing import Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TeamManagementUI:
    """UI generator for team management interface."""
    
    def __init__(self):
        self.templates: Dict[str, str] = {}
        self._init_templates()
    
    def _init_templates(self):
        """Initialize HTML templates."""
        
        # Main team management dashboard
        self.templates['team_dashboard'] = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Team Management - Genetic Coil Arena</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .header h1 {
            color: #764ba2;
            margin-bottom: 10px;
            font-size: 2.5rem;
        }
        
        .nav-tabs {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        
        .tab-button {
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .tab-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        
        .tab-button.active {
            background: linear-gradient(135deg, #f093fb, #f5576c);
        }
        
        .content-section {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .team-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .team-card {
            background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
            border-radius: 15px;
            padding: 20px;
            border: 2px solid transparent;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .team-card:hover {
            border-color: #667eea;
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
        }
        
        .team-card.selected {
            border-color: #f5576c;
            background: linear-gradient(135deg, #ffeef0, #f5576c20);
        }
        
        .team-name {
            font-size: 1.4rem;
            font-weight: 700;
            color: #764ba2;
            margin-bottom: 10px;
        }
        
        .team-info {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
            font-size: 0.9rem;
            color: #666;
        }
        
        .member-list {
            margin-top: 15px;
        }
        
        .member-item {
            display: flex;
            align-items: center;
            padding: 8px;
            margin-bottom: 5px;
            background: rgba(255, 255, 255, 0.7);
            border-radius: 8px;
            transition: background 0.3s ease;
        }
        
        .member-item:hover {
            background: rgba(255, 255, 255, 0.9);
        }
        
        .member-avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            margin-right: 10px;
        }
        
        .member-info {
            flex: 1;
        }
        
        .member-name {
            font-weight: 600;
            color: #333;
        }
        
        .member-role {
            font-size: 0.8rem;
            color: #666;
        }
        
        .role-badge {
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .role-owner { background: #ff6b6b; color: white; }
        .role-admin { background: #4ecdc4; color: white; }
        .role-researcher { background: #45b7d1; color: white; }
        .role-viewer { background: #96ceb4; color: white; }
        
        .action-buttons {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        
        .btn-success {
            background: linear-gradient(135deg, #4ecdc4, #44a08d);
            color: white;
        }
        
        .btn-danger {
            background: linear-gradient(135deg, #ff6b6b, #ee5a6f);
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(5px);
            z-index: 1000;
        }
        
        .modal-content {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            border-radius: 15px;
            padding: 30px;
            max-width: 500px;
            width: 90%;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #333;
        }
        
        .form-input {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }
        
        .form-input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .form-select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 1rem;
            background: white;
        }
        
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            z-index: 1001;
            transform: translateX(100%);
            transition: transform 0.3s ease;
        }
        
        .notification.show {
            transform: translateX(0);
        }
        
        .notification.success {
            background: linear-gradient(135deg, #4ecdc4, #44a08d);
        }
        
        .notification.error {
            background: linear-gradient(135deg, #ff6b6b, #ee5a6f);
        }
        
        .experiment-shared {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
        }
        
        .shared-experiment {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .experiment-info h4 {
            color: #764ba2;
            margin-bottom: 5px;
        }
        
        .experiment-meta {
            font-size: 0.9rem;
            color: #666;
        }
        
        .comment-section {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #e1e5e9;
        }
        
        .comment {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        
        .comment-author {
            font-weight: 600;
            color: #667eea;
            font-size: 0.9rem;
        }
        
        .comment-text {
            margin-top: 5px;
            color: #333;
        }
        
        .comment-time {
            font-size: 0.8rem;
            color: #666;
            margin-top: 5px;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .team-grid {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .nav-tabs {
                flex-wrap: wrap;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 Team Management</h1>
            <p>Collaborate on evolutionary computing experiments</p>
            
            <div class="nav-tabs">
                <button class="tab-button active" onclick="showTab('teams')">My Teams</button>
                <button class="tab-button" onclick="showTab('shared')">Shared Experiments</button>
                <button class="tab-button" onclick="showTab('invitations')">Invitations</button>
                <button class="tab-button" onclick="showTab('create')">Create Team</button>
            </div>
        </div>
        
        <!-- Teams Tab -->
        <div id="teams-tab" class="content-section">
            <h2>Your Teams</h2>
            <div class="team-grid" id="teams-container">
                <!-- Teams will be loaded here -->
            </div>
            
            <div class="action-buttons">
                <button class="btn btn-primary" onclick="createTeam()">
                    ➕ Create New Team
                </button>
                <button class="btn btn-success" onclick="joinTeam()">
                    🤝 Join Team
                </button>
            </div>
        </div>
        
        <!-- Shared Experiments Tab -->
        <div id="shared-tab" class="content-section" style="display: none;">
            <h2>Shared Experiments</h2>
            <div id="shared-experiments">
                <!-- Shared experiments will be loaded here -->
            </div>
        </div>
        
        <!-- Invitations Tab -->
        <div id="invitations-tab" class="content-section" style="display: none;">
            <h2>Team Invitations</h2>
            <div id="invitations-list">
                <!-- Invitations will be loaded here -->
            </div>
        </div>
        
        <!-- Create Team Tab -->
        <div id="create-tab" class="content-section" style="display: none;">
            <h2>Create New Team</h2>
            <form id="create-team-form" onsubmit="handleCreateTeam(event)">
                <div class="form-group">
                    <label class="form-label">Team Name</label>
                    <input type="text" class="form-input" name="name" required 
                           placeholder="Enter team name">
                </div>
                
                <div class="form-group">
                    <label class="form-label">Description</label>
                    <textarea class="form-input" name="description" rows="3"
                              placeholder="Describe your team's research focus"></textarea>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Privacy</label>
                    <select class="form-select" name="privacy">
                        <option value="private">Private - Invitation only</option>
                        <option value="public">Public - Anyone can join</option>
                    </select>
                </div>
                
                <button type="submit" class="btn btn-primary">Create Team</button>
            </form>
        </div>
    </div>
    
    <!-- Create Team Modal -->
    <div id="team-modal" class="modal">
        <div class="modal-content">
            <h3>Team Details</h3>
            <div id="team-details">
                <!-- Team details will be loaded here -->
            </div>
            
            <div class="action-buttons">
                <button class="btn btn-primary" onclick="editTeam()">Edit Team</button>
                <button class="btn btn-success" onclick="inviteMember()">Invite Member</button>
                <button class="btn btn-danger" onclick="leaveTeam()">Leave Team</button>
                <button class="btn" onclick="closeModal()">Close</button>
            </div>
        </div>
    </div>
    
    <!-- Invite Member Modal -->
    <div id="invite-modal" class="modal">
        <div class="modal-content">
            <h3>Invite Team Member</h3>
            <form id="invite-form" onsubmit="handleInvite(event)">
                <div class="form-group">
                    <label class="form-label">Email Address</label>
                    <input type="email" class="form-input" name="email" required
                           placeholder="colleague@university.edu">
                </div>
                
                <div class="form-group">
                    <label class="form-label">Role</label>
                    <select class="form-select" name="role">
                        <option value="viewer">Viewer - Can view experiments</option>
                        <option value="researcher">Researcher - Can create experiments</option>
                        <option value="admin">Admin - Can manage team</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Personal Message (Optional)</label>
                    <textarea class="form-input" name="message" rows="3"
                              placeholder="Join our research team to collaborate on..."></textarea>
                </div>
                
                <div class="action-buttons">
                    <button type="submit" class="btn btn-primary">Send Invitation</button>
                    <button type="button" class="btn" onclick="closeInviteModal()">Cancel</button>
                </div>
            </form>
        </div>
    </div>
    
    <!-- Notification -->
    <div id="notification" class="notification"></div>
    
    <script>
        // Global state
        let currentUser = null;
        let teams = [];
        let selectedTeam = null;
        
        // Initialize the interface
        async function init() {
            try {
                await loadCurrentUser();
                await loadTeams();
                await loadSharedExperiments();
                await loadInvitations();
            } catch (error) {
                showNotification('Failed to load team data', 'error');
                console.error('Initialization error:', error);
            }
        }
        
        // Load current user information
        async function loadCurrentUser() {
            try {
                const response = await fetch('/api/auth/user');
                if (response.ok) {
                    currentUser = await response.json();
                }
            } catch (error) {
                console.error('Failed to load user:', error);
            }
        }
        
        // Load user's teams
        async function loadTeams() {
            try {
                const response = await fetch('/api/teams');
                if (response.ok) {
                    teams = await response.json();
                    renderTeams();
                }
            } catch (error) {
                console.error('Failed to load teams:', error);
                showNotification('Failed to load teams', 'error');
            }
        }
        
        // Render teams in the UI
        function renderTeams() {
            const container = document.getElementById('teams-container');
            container.innerHTML = '';
            
            teams.forEach(team => {
                const teamCard = createTeamCard(team);
                container.appendChild(teamCard);
            });
        }
        
        // Create a team card element
        function createTeamCard(team) {
            const card = document.createElement('div');
            card.className = 'team-card';
            card.onclick = () => selectTeam(team);
            
            const memberCount = team.members ? team.members.length : 0;
            const userRole = getUserRole(team);
            
            card.innerHTML = `
                <div class="team-name">${team.name}</div>
                <div class="team-info">
                    <span>👥 ${memberCount} members</span>
                    <span class="role-badge role-${userRole}">${userRole}</span>
                </div>
                <p style="color: #666; margin-bottom: 15px;">${team.description || 'No description'}</p>
                
                <div class="member-list">
                    <h4 style="margin-bottom: 10px; color: #764ba2;">Recent Members</h4>
                    ${renderRecentMembers(team.members || [])}
                </div>
            `;
            
            return card;
        }
        
        // Render recent team members
        function renderRecentMembers(members) {
            const recentMembers = members.slice(0, 3);
            return recentMembers.map(member => `
                <div class="member-item">
                    <div class="member-avatar">${member.username[0].toUpperCase()}</div>
                    <div class="member-info">
                        <div class="member-name">${member.username}</div>
                        <div class="member-role">${member.role}</div>
                    </div>
                </div>
            `).join('');
        }
        
        // Get user's role in a team
        function getUserRole(team) {
            if (!currentUser || !team.members) return 'viewer';
            const member = team.members.find(m => m.user_id === currentUser.id);
            return member ? member.role : 'viewer';
        }
        
        // Select a team
        function selectTeam(team) {
            selectedTeam = team;
            
            // Update UI to show selection
            document.querySelectorAll('.team-card').forEach(card => {
                card.classList.remove('selected');
            });
            event.currentTarget.classList.add('selected');
            
            // Show team details modal
            showTeamDetails(team);
        }
        
        // Show team details modal
        function showTeamDetails(team) {
            const modal = document.getElementById('team-modal');
            const detailsDiv = document.getElementById('team-details');
            
            detailsDiv.innerHTML = `
                <h4>${team.name}</h4>
                <p style="margin-bottom: 20px;">${team.description || 'No description'}</p>
                
                <div style="margin-bottom: 20px;">
                    <h5>Team Members (${team.members ? team.members.length : 0})</h5>
                    <div class="member-list">
                        ${renderAllMembers(team.members || [])}
                    </div>
                </div>
                
                <div>
                    <h5>Team Statistics</h5>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 10px;">
                        <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #667eea;">0</div>
                            <div style="font-size: 0.9rem; color: #666;">Experiments</div>
                        </div>
                        <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #4ecdc4;">0</div>
                            <div style="font-size: 0.9rem; color: #666;">Shared</div>
                        </div>
                        <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #f5576c;">0</div>
                            <div style="font-size: 0.9rem; color: #666;">Comments</div>
                        </div>
                    </div>
                </div>
            `;
            
            modal.style.display = 'block';
        }
        
        // Render all team members
        function renderAllMembers(members) {
            return members.map(member => `
                <div class="member-item">
                    <div class="member-avatar">${member.username[0].toUpperCase()}</div>
                    <div class="member-info">
                        <div class="member-name">${member.username}</div>
                        <div class="member-role">${member.email}</div>
                    </div>
                    <span class="role-badge role-${member.role}">${member.role}</span>
                </div>
            `).join('');
        }
        
        // Tab switching
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.content-section').forEach(section => {
                section.style.display = 'none';
            });
            
            // Remove active class from all buttons
            document.querySelectorAll('.tab-button').forEach(button => {
                button.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName + '-tab').style.display = 'block';
            event.target.classList.add('active');
        }
        
        // Handle create team form
        async function handleCreateTeam(event) {
            event.preventDefault();
            
            const formData = new FormData(event.target);
            const teamData = {
                name: formData.get('name'),
                description: formData.get('description'),
                privacy: formData.get('privacy')
            };
            
            try {
                const response = await fetch('/api/teams', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(teamData)
                });
                
                if (response.ok) {
                    showNotification('Team created successfully!', 'success');
                    await loadTeams();
                    event.target.reset();
                    showTab('teams');
                } else {
                    const error = await response.json();
                    showNotification(error.detail || 'Failed to create team', 'error');
                }
            } catch (error) {
                console.error('Error creating team:', error);
                showNotification('Failed to create team', 'error');
            }
        }
        
        // Handle invite member
        async function handleInvite(event) {
            event.preventDefault();
            
            if (!selectedTeam) {
                showNotification('No team selected', 'error');
                return;
            }
            
            const formData = new FormData(event.target);
            const inviteData = {
                email: formData.get('email'),
                role: formData.get('role'),
                message: formData.get('message')
            };
            
            try {
                const response = await fetch(`/api/teams/${selectedTeam.id}/invite`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(inviteData)
                });
                
                if (response.ok) {
                    showNotification('Invitation sent successfully!', 'success');
                    closeInviteModal();
                    event.target.reset();
                } else {
                    const error = await response.json();
                    showNotification(error.detail || 'Failed to send invitation', 'error');
                }
            } catch (error) {
                console.error('Error sending invitation:', error);
                showNotification('Failed to send invitation', 'error');
            }
        }
        
        // Load shared experiments
        async function loadSharedExperiments() {
            try {
                const response = await fetch('/api/experiments/shared');
                if (response.ok) {
                    const experiments = await response.json();
                    renderSharedExperiments(experiments);
                }
            } catch (error) {
                console.error('Failed to load shared experiments:', error);
            }
        }
        
        // Render shared experiments
        function renderSharedExperiments(experiments) {
            const container = document.getElementById('shared-experiments');
            container.innerHTML = '';
            
            if (experiments.length === 0) {
                container.innerHTML = `
                    <div style="text-align: center; padding: 40px; color: #666;">
                        <h3>No shared experiments yet</h3>
                        <p>Experiments shared with your teams will appear here</p>
                    </div>
                `;
                return;
            }
            
            experiments.forEach(exp => {
                const expDiv = document.createElement('div');
                expDiv.className = 'experiment-shared';
                expDiv.innerHTML = `
                    <div class="shared-experiment">
                        <div class="experiment-info">
                            <h4>${exp.name}</h4>
                            <div class="experiment-meta">
                                Shared by ${exp.shared_by} • ${exp.team_name} • ${exp.shared_date}
                            </div>
                        </div>
                        <div class="action-buttons">
                            <button class="btn btn-primary" onclick="viewExperiment('${exp.id}')">
                                View
                            </button>
                            <button class="btn btn-success" onclick="commentOnExperiment('${exp.id}')">
                                Comment
                            </button>
                        </div>
                    </div>
                    
                    <div class="comment-section">
                        <h5>Comments (${exp.comments ? exp.comments.length : 0})</h5>
                        <div id="comments-${exp.id}">
                            ${renderComments(exp.comments || [])}
                        </div>
                        
                        <div style="margin-top: 15px;">
                            <textarea class="form-input" placeholder="Add a comment..." 
                                      id="comment-input-${exp.id}" rows="2"></textarea>
                            <button class="btn btn-primary" style="margin-top: 10px;"
                                    onclick="addComment('${exp.id}')">
                                Add Comment
                            </button>
                        </div>
                    </div>
                `;
                container.appendChild(expDiv);
            });
        }
        
        // Render comments
        function renderComments(comments) {
            return comments.map(comment => `
                <div class="comment">
                    <div class="comment-author">${comment.author}</div>
                    <div class="comment-text">${comment.text}</div>
                    <div class="comment-time">${comment.timestamp}</div>
                </div>
            `).join('');
        }
        
        // Load invitations
        async function loadInvitations() {
            try {
                const response = await fetch('/api/teams/invitations');
                if (response.ok) {
                    const invitations = await response.json();
                    renderInvitations(invitations);
                }
            } catch (error) {
                console.error('Failed to load invitations:', error);
            }
        }
        
        // Render invitations
        function renderInvitations(invitations) {
            const container = document.getElementById('invitations-list');
            container.innerHTML = '';
            
            if (invitations.length === 0) {
                container.innerHTML = `
                    <div style="text-align: center; padding: 40px; color: #666;">
                        <h3>No pending invitations</h3>
                        <p>Team invitations will appear here</p>
                    </div>
                `;
                return;
            }
            
            invitations.forEach(invite => {
                const inviteDiv = document.createElement('div');
                inviteDiv.className = 'team-card';
                inviteDiv.innerHTML = `
                    <div class="team-name">${invite.team_name}</div>
                    <div class="team-info">
                        <span>From: ${invite.inviter}</span>
                        <span class="role-badge role-${invite.role}">${invite.role}</span>
                    </div>
                    <p style="color: #666; margin: 15px 0;">${invite.message || 'No message'}</p>
                    
                    <div class="action-buttons">
                        <button class="btn btn-success" onclick="acceptInvitation('${invite.id}')">
                            Accept
                        </button>
                        <button class="btn btn-danger" onclick="declineInvitation('${invite.id}')">
                            Decline
                        </button>
                    </div>
                `;
                container.appendChild(inviteDiv);
            });
        }
        
        // Modal controls
        function closeModal() {
            document.getElementById('team-modal').style.display = 'none';
        }
        
        function closeInviteModal() {
            document.getElementById('invite-modal').style.display = 'none';
        }
        
        function inviteMember() {
            closeModal();
            document.getElementById('invite-modal').style.display = 'block';
        }
        
        // Notification system
        function showNotification(message, type = 'success') {
            const notification = document.getElementById('notification');
            notification.textContent = message;
            notification.className = `notification ${type} show`;
            
            setTimeout(() => {
                notification.classList.remove('show');
            }, 3000);
        }
        
        // Placeholder functions for future implementation
        function createTeam() { showTab('create'); }
        function joinTeam() { showNotification('Join team feature coming soon!'); }
        function editTeam() { showNotification('Edit team feature coming soon!'); }
        function leaveTeam() { showNotification('Leave team feature coming soon!'); }
        function viewExperiment(id) { showNotification('View experiment feature coming soon!'); }
        function commentOnExperiment(id) { showNotification('Comment feature coming soon!'); }
        function addComment(expId) { showNotification('Add comment feature coming soon!'); }
        function acceptInvitation(id) { showNotification('Accept invitation feature coming soon!'); }        function declineInvitation(id) { showNotification('Decline invitation feature coming soon!'); }
        
        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>
        """
    
    def get_team_dashboard(self, user_id: str = "") -> str:
        """Generate the main team management dashboard."""
        return self.templates['team_dashboard']
    
    def get_team_card_html(self, team: Dict[str, Any]) -> str:
        """Generate HTML for a single team card."""
        member_count = len(team.get('members', []))
        
        return f"""
        <div class="team-card" onclick="selectTeam('{team['id']}')">
            <div class="team-name">{team['name']}</div>
            <div class="team-info">
                <span>👥 {member_count} members</span>
                <span>🔬 {team.get('experiment_count', 0)} experiments</span>
            </div>
            <p style="color: #666; margin-bottom: 15px;">{team.get('description', 'No description')}</p>
            
            <div class="member-list">
                <h4 style="margin-bottom: 10px; color: #764ba2;">Recent Activity</h4>
                <div style="font-size: 0.9rem; color: #666;">
                    Last active: {team.get('last_activity', 'Never')}
                </div>
            </div>
        </div>
        """
    
    def get_experiment_share_form(self) -> str:
        """Generate form for sharing experiments."""
        return """
        <div class="share-form">
            <h3>Share Experiment</h3>
            <form id="share-experiment-form">
                <div class="form-group">
                    <label class="form-label">Select Team</label>
                    <select class="form-select" name="team_id" required>
                        <option value="">Choose a team...</option>
                        <!-- Teams will be populated by JavaScript -->
                    </select>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Share Level</label>
                    <select class="form-select" name="share_level">
                        <option value="view">View Only - Team can view results</option>
                        <option value="comment">Comment - Team can view and comment</option>
                        <option value="edit">Edit - Team can modify parameters</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Message (Optional)</label>
                    <textarea class="form-input" name="message" rows="3"
                              placeholder="Why are you sharing this experiment?"></textarea>
                </div>
                
                <div class="form-group">
                    <label>
                        <input type="checkbox" name="notify_team"> 
                        Notify team members via email
                    </label>
                </div>
                
                <button type="submit" class="btn btn-primary">Share Experiment</button>
            </form>
        </div>
        """


# Team collaboration utilities
class TeamCollaborationUtils:
    """Utilities for team collaboration features."""
    
    @staticmethod
    def format_user_activity(activity_data: Dict[str, Any]) -> str:
        """Format user activity for display."""
        activity_type = activity_data.get('type', 'unknown')
        timestamp = activity_data.get('timestamp', 'Unknown time')
        user = activity_data.get('user', 'Unknown user')
        
        activity_messages = {
            'experiment_created': f"🧪 {user} created a new experiment",
            'experiment_shared': f"📤 {user} shared an experiment",
            'comment_added': f"💬 {user} added a comment",
            'team_joined': f"🤝 {user} joined the team",
            'team_left': f"👋 {user} left the team",
            'role_changed': f"🔄 {user}'s role was updated"
        }
        
        message = activity_messages.get(activity_type, f"📋 {user} performed an action")
        return f"{message} • {timestamp}"
    
    @staticmethod
    def generate_team_stats(team_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate team statistics."""
        members = team_data.get('members', [])
        experiments = team_data.get('experiments', [])
        
        return {
            'member_count': len(members),
            'experiment_count': len(experiments),
            'shared_count': sum(1 for exp in experiments if exp.get('shared', False)),
            'comment_count': sum(len(exp.get('comments', [])) for exp in experiments),
            'active_members': sum(1 for member in members 
                                if member.get('last_active') and 
                                (datetime.now() - datetime.fromisoformat(member['last_active'])).days < 7)
        }
