"""
Phase 4C: Notification System
Email/SMS notifications for team collaboration events.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Protocol
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import ssl

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Types of notifications."""
    EXPERIMENT_SHARED = "experiment_shared"
    EXPERIMENT_COMPLETED = "experiment_completed"
    TEAM_INVITATION = "team_invitation"
    TEAM_JOINED = "team_joined"
    COMMENT_ADDED = "comment_added"
    ROLE_CHANGED = "role_changed"
    SYSTEM_ALERT = "system_alert"


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class NotificationChannel(Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    SMS = "sms"
    IN_APP = "in_app"
    WEBHOOK = "webhook"


@dataclass
class NotificationTemplate:
    """Template for notifications."""
    type: NotificationType
    subject_template: str
    body_template: str
    html_template: Optional[str] = None
    variables: List[str] = field(default_factory=list)


@dataclass
class NotificationPreferences:
    """User notification preferences."""
    user_id: str
    email_enabled: bool = True
    sms_enabled: bool = False
    in_app_enabled: bool = True
    channels_by_type: Dict[NotificationType, List[NotificationChannel]] = field(default_factory=dict)
    quiet_hours_start: Optional[str] = None  # "22:00"
    quiet_hours_end: Optional[str] = None    # "08:00"
    frequency_limit: Dict[NotificationType, int] = field(default_factory=dict)  # max per day


@dataclass
class Notification:
    """Individual notification."""
    id: str
    type: NotificationType
    recipient_id: str
    title: str
    message: str
    html_content: Optional[str] = None
    priority: NotificationPriority = NotificationPriority.MEDIUM
    channels: List[NotificationChannel] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_for: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3


class NotificationProvider(Protocol):
    """Protocol for notification providers."""
    
    async def send(self, notification: Notification) -> bool:
        """Send a notification through this provider."""
        ...
    
    def is_available(self) -> bool:
        """Check if provider is available."""
        ...


class EmailProvider:
    """Email notification provider."""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str, 
                 use_tls: bool = True):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        
    async def send(self, notification: Notification) -> bool:
        """Send email notification."""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = notification.title
            msg['From'] = self.username
            msg['To'] = notification.data.get('email', '')
            
            # Add text part
            text_part = MIMEText(notification.message, 'plain')
            msg.attach(text_part)
            
            # Add HTML part if available
            if notification.html_content:
                html_part = MIMEText(notification.html_content, 'html')
                msg.attach(html_part)
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls(context=context)
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {notification.data.get('email', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if email provider is configured."""
        return all([self.smtp_server, self.smtp_port, self.username, self.password])


class SMSProvider:
    """SMS notification provider (placeholder for integration with services like Twilio)."""
    
    def __init__(self, api_key: str, api_secret: str, from_number: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.from_number = from_number
        
    async def send(self, notification: Notification) -> bool:
        """Send SMS notification."""
        try:
            # In a real implementation, you would integrate with Twilio or similar service
            # For now, this is a placeholder that logs the action
            phone_number = notification.data.get('phone_number')
            if not phone_number:
                logger.error("No phone number provided for SMS notification")
                return False
                
            # Placeholder for actual SMS sending logic
            # In production, you would use something like:
            # client = Client(self.api_key, self.api_secret)
            # message = client.messages.create(
            #     body=notification.message,
            #     from_=self.from_number,
            #     to=phone_number
            # )
            
            logger.info(f"SMS notification sent to {phone_number}: {notification.message}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send SMS notification: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if SMS provider is configured."""
        return all([self.api_key, self.api_secret, self.from_number])


class WebhookProvider:
    """Webhook notification provider."""
    
    def __init__(self, webhook_url: str, secret: Optional[str] = None):
        self.webhook_url = webhook_url
        self.secret = secret
        
    async def send(self, notification: Notification) -> bool:
        """Send webhook notification."""
        try:
            import aiohttp
            import hashlib
            import hmac
            import json
            
            # Prepare payload
            payload = {
                "id": notification.id,
                "type": notification.type.value,
                "title": notification.title,
                "message": notification.message,
                "priority": notification.priority.value,
                "created_at": notification.created_at.isoformat(),
                "data": notification.data
            }
            
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "GeneticCoilArena/1.0"
            }
            
            # Add signature if secret is provided
            if self.secret:
                payload_str = json.dumps(payload, sort_keys=True)
                signature = hmac.new(
                    self.secret.encode(),
                    payload_str.encode(),
                    hashlib.sha256
                ).hexdigest()
                headers["X-Webhook-Signature"] = f"sha256={signature}"
            
            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status < 400:
                        logger.info(f"Webhook notification sent successfully to {self.webhook_url}")
                        return True
                    else:
                        logger.error(f"Webhook returned status {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if webhook provider is configured."""
        return bool(self.webhook_url)


class NotificationManager:
    """Central notification management system."""
    
    def __init__(self):
        self.providers: Dict[NotificationChannel, NotificationProvider] = {}
        self.templates: Dict[NotificationType, NotificationTemplate] = {}
        self.preferences: Dict[str, NotificationPreferences] = {}
        self.pending_notifications: List[Notification] = []
        self.sent_notifications: List[Notification] = []
        self._init_templates()
    
    def _init_templates(self):
        """Initialize notification templates."""
        
        # Experiment shared template
        self.templates[NotificationType.EXPERIMENT_SHARED] = NotificationTemplate(
            type=NotificationType.EXPERIMENT_SHARED,
            subject_template="🧬 Experiment Shared: {experiment_name}",
            body_template="""
Hello {recipient_name},

{sharer_name} has shared an experiment with your team "{team_name}":

Experiment: {experiment_name}
Description: {experiment_description}
Shared on: {shared_date}

Message from {sharer_name}:
{message}

You can view and collaborate on this experiment here:
{experiment_url}

Best regards,
Genetic Coil Arena Team
            """.strip(),
            html_template="""
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px 10px 0 0;">
        <h1>🧬 Experiment Shared</h1>
    </div>
    
    <div style="padding: 20px; background: #f9f9f9;">
        <p>Hello <strong>{recipient_name}</strong>,</p>
        
        <p><strong>{sharer_name}</strong> has shared an experiment with your team <strong>"{team_name}"</strong>:</p>
        
        <div style="background: white; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #667eea;">
            <h3 style="margin: 0 0 10px 0; color: #764ba2;">{experiment_name}</h3>
            <p style="margin: 0; color: #666;">{experiment_description}</p>
            <p style="margin: 10px 0 0 0; font-size: 0.9em; color: #888;">Shared on {shared_date}</p>
        </div>
        
        <div style="background: white; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h4 style="margin: 0 0 10px 0;">Message from {sharer_name}:</h4>
            <p style="margin: 0; font-style: italic;">"{message}"</p>
        </div>
        
        <div style="text-align: center; margin: 20px 0;">
            <a href="{experiment_url}" style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 12px 24px; text-decoration: none; border-radius: 25px; display: inline-block;">
                View Experiment
            </a>
        </div>
        
        <p style="font-size: 0.9em; color: #666;">
            Best regards,<br>
            Genetic Coil Arena Team
        </p>
    </div>
</div>
            """.strip(),
            variables=["recipient_name", "sharer_name", "team_name", "experiment_name", 
                      "experiment_description", "shared_date", "message", "experiment_url"]
        )
        
        # Team invitation template
        self.templates[NotificationType.TEAM_INVITATION] = NotificationTemplate(
            type=NotificationType.TEAM_INVITATION,
            subject_template="🤝 Team Invitation: {team_name}",
            body_template="""
Hello {recipient_name},

{inviter_name} has invited you to join the team "{team_name}" on Genetic Coil Arena.

Team: {team_name}
Role: {role}
Invited by: {inviter_name}

Message from {inviter_name}:
{message}

To accept this invitation, click here:
{invitation_url}

Best regards,
Genetic Coil Arena Team
            """.strip(),
            variables=["recipient_name", "inviter_name", "team_name", "role", "message", "invitation_url"]
        )
        
        # Experiment completed template
        self.templates[NotificationType.EXPERIMENT_COMPLETED] = NotificationTemplate(
            type=NotificationType.EXPERIMENT_COMPLETED,
            subject_template="✅ Experiment Completed: {experiment_name}",
            body_template="""
Hello {recipient_name},

Your experiment "{experiment_name}" has completed successfully!

Results:
- Generations: {generations}
- Best fitness: {best_fitness}
- Duration: {duration}

You can view the detailed results here:
{results_url}

Best regards,
Genetic Coil Arena Team
            """.strip(),
            variables=["recipient_name", "experiment_name", "generations", "best_fitness", 
                      "duration", "results_url"]
        )
    
    def register_provider(self, channel: NotificationChannel, provider: NotificationProvider):
        """Register a notification provider."""
        self.providers[channel] = provider
        logger.info(f"Registered {channel.value} notification provider")
    
    def set_user_preferences(self, user_id: str, preferences: NotificationPreferences):
        """Set notification preferences for a user."""
        self.preferences[user_id] = preferences
        logger.info(f"Updated notification preferences for user {user_id}")
    
    def get_user_preferences(self, user_id: str) -> NotificationPreferences:
        """Get notification preferences for a user."""
        return self.preferences.get(user_id, NotificationPreferences(user_id=user_id))
    
    async def send_notification(self, notification: Notification) -> bool:
        """Send a notification through appropriate channels."""
        user_prefs = self.get_user_preferences(notification.recipient_id)
        
        # Check if we should send during quiet hours
        if self._is_quiet_hours(user_prefs):
            # Schedule for later if not urgent
            if notification.priority != NotificationPriority.URGENT:
                notification.scheduled_for = self._calculate_next_send_time(user_prefs)
                self.pending_notifications.append(notification)
                return True
        
        # Check frequency limits
        if self._exceeds_frequency_limit(notification, user_prefs):
            logger.info(f"Notification {notification.id} skipped due to frequency limit")
            return False
        
        # Determine channels to use
        channels = self._get_channels_for_notification(notification, user_prefs)
        if not channels:
            logger.info(f"No channels available for notification {notification.id}")
            return False
        
        # Send through each channel
        success = False
        for channel in channels:
            if channel in self.providers:
                try:
                    result = await self.providers[channel].send(notification)
                    if result:
                        success = True
                        logger.info(f"Notification {notification.id} sent via {channel.value}")
                except Exception as e:
                    logger.error(f"Failed to send notification via {channel.value}: {e}")
        
        # Update notification status
        if success:
            notification.sent_at = datetime.now()
        else:
            notification.failed_at = datetime.now()
            notification.retry_count += 1
            
            # Schedule retry if under limit
            if notification.retry_count <= notification.max_retries:
                notification.scheduled_for = datetime.now() + timedelta(minutes=5 * notification.retry_count)
                self.pending_notifications.append(notification)
        
        self.sent_notifications.append(notification)
        return success
    
    def _is_quiet_hours(self, preferences: NotificationPreferences) -> bool:
        """Check if current time is within user's quiet hours."""
        if not preferences.quiet_hours_start or not preferences.quiet_hours_end:
            return False
        
        try:
            from datetime import time
            
            now = datetime.now().time()
            start_time = time.fromisoformat(preferences.quiet_hours_start)
            end_time = time.fromisoformat(preferences.quiet_hours_end)
            
            # Handle quiet hours that span midnight
            if start_time <= end_time:
                # Same day quiet hours (e.g., 22:00 to 23:59)
                return start_time <= now <= end_time
            else:
                # Quiet hours span midnight (e.g., 22:00 to 08:00)
                return now >= start_time or now <= end_time
                
        except Exception as e:
            logger.error(f"Error checking quiet hours: {e}")
            return False
    
    def _calculate_next_send_time(self, preferences: NotificationPreferences) -> datetime:
        """Calculate next appropriate send time based on user preferences."""
        if not preferences.quiet_hours_start or not preferences.quiet_hours_end:
            # No quiet hours configured, send immediately
            return datetime.now()
        
        try:
            from datetime import time, timedelta
            
            now = datetime.now()
            current_time = now.time()
            end_time = time.fromisoformat(preferences.quiet_hours_end)
            
            # If we're in quiet hours, schedule for when they end
            if self._is_quiet_hours(preferences):
                # Calculate when quiet hours end
                end_datetime = datetime.combine(now.date(), end_time)
                
                # If end time is before current time, it's tomorrow
                if end_time <= current_time:
                    end_datetime += timedelta(days=1)
                
                return end_datetime
            else:
                # Not in quiet hours, send immediately
                return now
                
        except Exception as e:
            logger.error(f"Error calculating next send time: {e}")
            # Default to 8 hours from now
            from datetime import timedelta
            return datetime.now() + timedelta(hours=8)
    
    def _exceeds_frequency_limit(self, notification: Notification, 
                                preferences: NotificationPreferences) -> bool:
        """Check if sending this notification would exceed frequency limits."""
        limit = preferences.frequency_limit.get(notification.type)
        if not limit:
            return False
        
        # Count notifications of this type sent today
        today = datetime.now().date()
        today_count = sum(1 for n in self.sent_notifications 
                         if n.type == notification.type 
                         and n.sent_at 
                         and n.sent_at.date() == today
                         and n.recipient_id == notification.recipient_id)
        
        return today_count >= limit
    
    def _get_channels_for_notification(self, notification: Notification, 
                                     preferences: NotificationPreferences) -> List[NotificationChannel]:
        """Determine which channels to use for a notification."""
        # Use user preferences for this notification type
        preferred_channels = preferences.channels_by_type.get(notification.type, [])
        
        # Fall back to notification's specified channels
        if not preferred_channels:
            preferred_channels = notification.channels
        
        # Fall back to default channels based on user preferences
        if not preferred_channels:
            default_channels = []
            if preferences.email_enabled:
                default_channels.append(NotificationChannel.EMAIL)
            if preferences.sms_enabled:
                default_channels.append(NotificationChannel.SMS)
            if preferences.in_app_enabled:
                default_channels.append(NotificationChannel.IN_APP)
            preferred_channels = default_channels
        
        # Filter by available providers
        available_channels = [ch for ch in preferred_channels 
                            if ch in self.providers and self.providers[ch].is_available()]
        
        return available_channels
    
    async def process_pending_notifications(self):
        """Process notifications scheduled for delivery."""
        current_time = datetime.now()
        ready_notifications = [n for n in self.pending_notifications 
                              if not n.scheduled_for or n.scheduled_for <= current_time]
        
        for notification in ready_notifications:
            await self.send_notification(notification)
            self.pending_notifications.remove(notification)
    
    def create_notification_from_template(self, 
                                        template_type: NotificationType,
                                        recipient_id: str,
                                        variables: Dict[str, Any],
                                        priority: NotificationPriority = NotificationPriority.MEDIUM) -> Notification:
        """Create a notification from a template."""
        template = self.templates.get(template_type)
        if not template:
            raise ValueError(f"No template found for {template_type}")
        
        # Format subject and body
        subject = template.subject_template.format(**variables)
        body = template.body_template.format(**variables)
        html_content = None
        
        if template.html_template:
            html_content = template.html_template.format(**variables)
        
        return Notification(
            id=f"{template_type.value}_{recipient_id}_{datetime.now().timestamp()}",
            type=template_type,
            recipient_id=recipient_id,
            title=subject,
            message=body,
            html_content=html_content,
            priority=priority,
            data=variables
        )
    
    def get_notification_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get notification statistics."""
        notifications = self.sent_notifications
        if user_id:
            notifications = [n for n in notifications if n.recipient_id == user_id]
        
        total = len(notifications)
        sent = len([n for n in notifications if n.sent_at])
        failed = len([n for n in notifications if n.failed_at])
        pending = len(self.pending_notifications)
        
        by_type = {}
        for notification in notifications:
            type_name = notification.type.value
            if type_name not in by_type:
                by_type[type_name] = 0
            by_type[type_name] += 1
        
        return {
            "total": total,
            "sent": sent,
            "failed": failed,
            "pending": pending,
            "success_rate": (sent / total * 100) if total > 0 else 0,
            "by_type": by_type
        }


# Helper function to get user name
async def _get_user_name(user_id: str) -> str:
    """Get user name by ID, with fallback to generic name."""
    try:
        # Import here to avoid circular imports
        from .database import get_database
        
        db = get_database()
        query = "SELECT username FROM users WHERE id = ?"
        cursor = db.execute(query, (user_id,))
        row = cursor.fetchone()
        
        if row:
            return row[0]
        else:
            return "User"
            
    except Exception as e:
        logger.error(f"Error getting user name for {user_id}: {e}")
        return "User"


# Global notification manager instance
notification_manager = NotificationManager()


# Convenience functions for common notifications
async def notify_experiment_shared(recipient_id: str, sharer_name: str, team_name: str,
                                 experiment_name: str, experiment_description: str,
                                 message: str, experiment_url: str):
    """Send experiment shared notification."""
    recipient_name = await _get_user_name(recipient_id)
    
    variables = {
        "recipient_name": recipient_name,
        "sharer_name": sharer_name,
        "team_name": team_name,
        "experiment_name": experiment_name,
        "experiment_description": experiment_description,
        "shared_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "message": message,
        "experiment_url": experiment_url
    }
    
    notification = notification_manager.create_notification_from_template(
        NotificationType.EXPERIMENT_SHARED,
        recipient_id,
        variables
    )
    
    return await notification_manager.send_notification(notification)


async def notify_team_invitation(recipient_id: str, inviter_name: str, team_name: str,
                                role: str, message: str, invitation_url: str):
    """Send team invitation notification."""
    recipient_name = await _get_user_name(recipient_id)
    
    variables = {
        "recipient_name": recipient_name,
        "inviter_name": inviter_name,
        "team_name": team_name,
        "role": role,
        "message": message,
        "invitation_url": invitation_url
    }
    
    notification = notification_manager.create_notification_from_template(
        NotificationType.TEAM_INVITATION,
        recipient_id,
        variables,
        priority=NotificationPriority.HIGH
    )
    
    return await notification_manager.send_notification(notification)


async def notify_experiment_completed(recipient_id: str, experiment_name: str,
                                    generations: int, best_fitness: float,
                                    duration: str, results_url: str):
    """Send experiment completion notification."""
    recipient_name = await _get_user_name(recipient_id)
    
    variables = {
        "recipient_name": recipient_name,
        "experiment_name": experiment_name,
        "generations": str(generations),
        "best_fitness": f"{best_fitness:.6f}",
        "duration": duration,
        "results_url": results_url
    }
    
    notification = notification_manager.create_notification_from_template(
        NotificationType.EXPERIMENT_COMPLETED,
        recipient_id,
        variables
    )
    
    return await notification_manager.send_notification(notification)
