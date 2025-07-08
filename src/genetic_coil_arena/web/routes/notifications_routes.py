"""
Phase 4C: Notification System Routes
FastAPI routes for managing notifications and preferences.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any
import logging
from datetime import datetime

from .notifications import (
    notification_manager, NotificationType,
    NotificationChannel, NotificationPreferences, EmailProvider
)
from ..core.auth import get_current_user

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get("/", summary="Get User Notifications")
async def get_notifications(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    unread_only: bool = Query(False),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get notifications for the current user.
    
    Args:
        limit: Maximum number of notifications to return
        offset: Number of notifications to skip
        unread_only: Only return unread notifications
        
    Returns:
        Paginated list of notifications
    """
    try:
        user_id = current_user['id']
        
        # Filter notifications for this user
        all_notifications = [
            n for n in notification_manager.sent_notifications 
            if n.recipient_id == user_id
        ]
        
        if unread_only:
            all_notifications = [n for n in all_notifications if not n.read_at]
        
        # Sort by created date (newest first)
        all_notifications.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        total = len(all_notifications)
        notifications = all_notifications[offset:offset + limit]
          # Convert to dict format
        notification_data: list[Dict[str, Any]] = []
        for notification in notifications:
            notification_data.append({
                "id": notification.id,
                "type": notification.type.value,
                "title": notification.title,
                "message": notification.message,
                "priority": notification.priority.value,
                "created_at": notification.created_at.isoformat(),
                "sent_at": notification.sent_at.isoformat() if notification.sent_at else None,
                "read_at": notification.read_at.isoformat() if notification.read_at else None,
                "data": notification.data
            })
        
        return {
            "notifications": notification_data,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        }
        
    except Exception as e:
        logger.error(f"Error getting notifications: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve notifications")


@router.post("/{notification_id}/read", summary="Mark Notification as Read")
async def mark_notification_read(
    notification_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Mark a notification as read.
    
    Args:
        notification_id: ID of the notification to mark as read
        
    Returns:
        Success message
    """
    try:
        user_id = current_user['id']
        
        # Find the notification
        notification = None
        for n in notification_manager.sent_notifications:
            if n.id == notification_id and n.recipient_id == user_id:
                notification = n
                break
        
        if not notification:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        # Mark as read
        notification.read_at = datetime.now()
        
        logger.info(f"Notification {notification_id} marked as read by user {user_id}")
        return {"message": "Notification marked as read"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking notification as read: {e}")
        raise HTTPException(status_code=500, detail="Failed to mark notification as read")


@router.get("/preferences", summary="Get Notification Preferences")
async def get_notification_preferences(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get notification preferences for the current user.
    
    Returns:
        User's notification preferences
    """
    try:
        user_id = current_user['id']
        preferences = notification_manager.get_user_preferences(user_id)
        
        return {
            "user_id": preferences.user_id,
            "email_enabled": preferences.email_enabled,
            "sms_enabled": preferences.sms_enabled,
            "in_app_enabled": preferences.in_app_enabled,
            "quiet_hours_start": preferences.quiet_hours_start,
            "quiet_hours_end": preferences.quiet_hours_end,
            "channels_by_type": {
                k.value: [ch.value for ch in v] 
                for k, v in preferences.channels_by_type.items()
            },
            "frequency_limit": {
                k.value: v for k, v in preferences.frequency_limit.items()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting notification preferences: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve preferences")


@router.put("/preferences", summary="Update Notification Preferences")
async def update_notification_preferences(
    preferences_data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Update notification preferences for the current user.
    
    Args:
        preferences_data: New notification preferences
        
    Returns:
        Success message
    """
    try:
        user_id = current_user['id']
        
        # Create preferences object
        preferences = NotificationPreferences(user_id=user_id)
        
        # Update basic settings
        if 'email_enabled' in preferences_data:
            preferences.email_enabled = preferences_data['email_enabled']
        if 'sms_enabled' in preferences_data:
            preferences.sms_enabled = preferences_data['sms_enabled']
        if 'in_app_enabled' in preferences_data:
            preferences.in_app_enabled = preferences_data['in_app_enabled']
        if 'quiet_hours_start' in preferences_data:
            preferences.quiet_hours_start = preferences_data['quiet_hours_start']
        if 'quiet_hours_end' in preferences_data:
            preferences.quiet_hours_end = preferences_data['quiet_hours_end']
        
        # Update channels by type
        if 'channels_by_type' in preferences_data:
            channels_by_type = {}
            for type_str, channel_list in preferences_data['channels_by_type'].items():
                try:
                    notification_type = NotificationType(type_str)
                    channels = [NotificationChannel(ch) for ch in channel_list]
                    channels_by_type[notification_type] = channels
                except ValueError:
                    logger.warning(f"Invalid notification type or channel: {type_str}")
            preferences.channels_by_type = channels_by_type
        
        # Update frequency limits
        if 'frequency_limit' in preferences_data:
            frequency_limit = {}
            for type_str, limit in preferences_data['frequency_limit'].items():
                try:
                    notification_type = NotificationType(type_str)
                    frequency_limit[notification_type] = int(limit)
                except ValueError:
                    logger.warning(f"Invalid notification type: {type_str}")
            preferences.frequency_limit = frequency_limit
        
        # Save preferences
        notification_manager.set_user_preferences(user_id, preferences)
        
        logger.info(f"Updated notification preferences for user {user_id}")
        return {"message": "Notification preferences updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating notification preferences: {e}")
        raise HTTPException(status_code=500, detail="Failed to update preferences")


@router.post("/test", summary="Send Test Notification")
async def send_test_notification(
    notification_type: str = "system_alert",
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Send a test notification to the current user.
    
    Args:
        notification_type: Type of test notification to send
        
    Returns:
        Success message
    """
    try:
        user_id = current_user['id']
        username = current_user.get('username', 'User')
        
        # Create test notification based on type
        if notification_type == "experiment_shared":
            from .notifications import notify_experiment_shared
            await notify_experiment_shared(
                recipient_id=user_id,
                sharer_name="Test User",
                team_name="Test Team",
                experiment_name="Test Experiment",
                experiment_description="This is a test experiment for notification testing.",
                message="This is a test notification to verify the system is working correctly.",
                experiment_url="/experiments/test"
            )
        elif notification_type == "team_invitation":
            from .notifications import notify_team_invitation
            await notify_team_invitation(
                recipient_id=user_id,
                inviter_name="Test Inviter",
                team_name="Test Research Team",
                role="researcher",
                message="This is a test invitation to verify the notification system.",
                invitation_url="/teams/invitations/test"
            )
        elif notification_type == "experiment_completed":
            from .notifications import notify_experiment_completed
            await notify_experiment_completed(
                recipient_id=user_id,
                experiment_name="Test Experiment",
                generations=100,
                best_fitness=0.95,
                duration="5 minutes",
                results_url="/experiments/test/results"
            )
        else:
            # Generic test notification
            from .notifications import Notification, NotificationType, NotificationPriority
            test_notification = Notification(
                id=f"test_{user_id}_{datetime.now().timestamp()}",
                type=NotificationType.SYSTEM_ALERT,
                recipient_id=user_id,
                title=f"🔔 Test Notification for {username}",
                message="This is a test notification to verify the system is working correctly.",
                priority=NotificationPriority.LOW,
                data={"test": True, "timestamp": datetime.now().isoformat()}
            )
            await notification_manager.send_notification(test_notification)
        
        logger.info(f"Sent test notification ({notification_type}) to user {user_id}")
        return {"message": f"Test notification ({notification_type}) sent successfully"}
        
    except Exception as e:
        logger.error(f"Error sending test notification: {e}")
        raise HTTPException(status_code=500, detail="Failed to send test notification")


@router.get("/stats", summary="Get Notification Statistics")
async def get_notification_stats(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get notification statistics for the current user.
    
    Returns:
        Notification statistics and metrics
    """
    try:
        user_id = current_user['id']
        stats = notification_manager.get_notification_stats(user_id)
        
        # Add user-specific metrics
        user_notifications = [
            n for n in notification_manager.sent_notifications 
            if n.recipient_id == user_id
        ]
        
        unread_count = len([n for n in user_notifications if not n.read_at])
        read_count = len([n for n in user_notifications if n.read_at])
        
        stats.update({
            "unread_count": unread_count,
            "read_count": read_count,
            "read_rate": (read_count / (read_count + unread_count) * 100) 
                        if (read_count + unread_count) > 0 else 0
        })
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting notification stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@router.post("/setup-email", summary="Setup Email Provider")
async def setup_email_provider(
    email_config: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Setup email provider for notifications (admin only).
    
    Args:
        email_config: Email provider configuration
        
    Returns:
        Success message
    """
    try:
        # TODO: Add admin role check
        
        # Extract configuration
        smtp_server = email_config.get('smtp_server')
        smtp_port = email_config.get('smtp_port', 587)
        username = email_config.get('username')
        password = email_config.get('password')
        use_tls = email_config.get('use_tls', True)
        
        if not all([smtp_server, username, password]):
            raise HTTPException(
                status_code=400, 
                detail="smtp_server, username, and password are required"
            )
          # Create and register email provider
        email_provider = EmailProvider(
            smtp_server=str(smtp_server),
            smtp_port=int(smtp_port),
            username=str(username),
            password=str(password),
            use_tls=use_tls
        )
        
        notification_manager.register_provider(NotificationChannel.EMAIL, email_provider)
        
        logger.info(f"Email provider configured by user {current_user.get('username', 'unknown')}")
        return {"message": "Email provider configured successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting up email provider: {e}")
        raise HTTPException(status_code=500, detail="Failed to setup email provider")


@router.get("/health", summary="Notification System Health Check")
async def notification_health_check() -> Dict[str, Any]:
    """
    Check the health of the notification system.
    
    Returns:
        Health status and provider availability
    """
    try:
        # Check provider availability
        providers_status = {}
        for channel, provider in notification_manager.providers.items():
            providers_status[channel.value] = {
                "available": provider.is_available(),
                "type": type(provider).__name__
            }
        
        # Get basic stats
        total_sent = len(notification_manager.sent_notifications)
        total_pending = len(notification_manager.pending_notifications)
        total_templates = len(notification_manager.templates)
        
        return {
            "status": "healthy",
            "service": "notification_system",
            "providers": providers_status,
            "stats": {
                "total_sent": total_sent,
                "total_pending": total_pending,
                "total_templates": total_templates
            }
        }
        
    except Exception as e:
        logger.error(f"Error checking notification system health: {e}")
        return {
            "status": "unhealthy",
            "service": "notification_system",
            "error": str(e)
        }
