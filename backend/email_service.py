from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage
from html import escape

logger = logging.getLogger(__name__)


def _clean(value: object, limit: int = 500) -> str:
    return escape(str(value or '').strip())[:limit]


def send_password_reset_email(recipient_email: str, reset_url: str, user_name: str) -> bool:
    """Send a password reset link using the backend's existing SMTP configuration."""
    host = os.getenv('SMTP_HOST')
    try:
        port = int(os.getenv('SMTP_PORT', '587'))
    except (TypeError, ValueError):
        logger.error('Password reset email configuration has an invalid SMTP_PORT.')
        return False
    sender = os.getenv('SMTP_FROM')
    username = os.getenv('SMTP_USERNAME')
    password = os.getenv('SMTP_PASSWORD')
    missing = [name for name, value in {
        'SMTP_HOST': host, 'SMTP_FROM': sender, 'SMTP_USERNAME': username, 'SMTP_PASSWORD': password,
    }.items() if not value]
    if missing:
        logger.error('Password reset email configuration is missing %s.', ', '.join(missing))
        return False

    message = EmailMessage()
    message['Subject'] = 'ATAS - Password Reset Request'
    message['From'] = sender
    message['To'] = recipient_email
    message.set_content(
        f"Hello {user_name},\n\n"
        'A password reset was requested for your ATAS account.\n\n'
        f'Create a new password using this temporary link:\n{reset_url}\n\n'
        'This link expires shortly and can only be used once. If you did not request this, '
        'you can safely ignore this email. Do not share this link with anyone.\n\n'
        'ATAS Security Team\n'
    )
    try:
        with smtplib.SMTP(host, port, timeout=15) as smtp:
            smtp.starttls()
            smtp.login(username, password)
            smtp.send_message(message)
        logger.info('Password reset email accepted by SMTP for the registered account.')
        return True
    except Exception as error:
        logger.error('Password reset email delivery failed (%s).', type(error).__name__)
        return False


def send_threat_notification(
    recipient_email: str,
    risk_score: float,
    threat_level: str,
    reason: str,
    timestamp: str,
    alert_id: str,
    *,
    threat_type: str = 'Threat detected',
    location: str | None = None,
) -> bool:
    host = os.getenv('SMTP_HOST')
    port = int(os.getenv('SMTP_PORT', '587'))
    sender = os.getenv('SMTP_FROM')
    username = os.getenv('SMTP_USERNAME')
    password = os.getenv('SMTP_PASSWORD')
    missing = [name for name, value in {
        'SMTP_HOST': host, 'SMTP_FROM': sender, 'SMTP_USERNAME': username, 'SMTP_PASSWORD': password,
    }.items() if not value]
    if missing:
        logger.error('SMTP configuration is missing %s for alert %s.', ', '.join(missing), alert_id)
        return False
    if not recipient_email or '@' not in recipient_email:
        logger.error('ALERT_EMAIL is missing or invalid for alert %s.', alert_id)
        return False

    score = max(0.0, min(float(risk_score), 100.0))
    level = _clean(threat_level, 32).upper()
    safe_reason = _clean(reason)
    safe_timestamp = _clean(timestamp, 80)
    safe_alert_id = _clean(alert_id, 80)
    safe_threat_type = _clean(threat_type, 120)
    safe_location = _clean(location or 'Not available', 200)

    message = EmailMessage()
    message['Subject'] = 'HIGH THREAT ALERT - Threat Detection System'
    message['From'] = sender
    message['To'] = recipient_email
    message.set_content(
        'HIGH THREAT ALERT\n\n'
        f'Threat: {safe_threat_type}\n'
        f'Confidence: {score:.2f}%\n'
        f'Threat Level: {level}\n\n'
        f'Location: {safe_location}\n'
        f'Detected At: {safe_timestamp}\n\n'
        f'Reason:\n{safe_reason}\n\n'
        f'Alert ID:\n{safe_alert_id}\n\n'
        'Please check the ATAS monitoring dashboard immediately.\n'
    )

    try:
        with smtplib.SMTP(host, port, timeout=15) as smtp:
            smtp.starttls()
            smtp.login(username, password)
            smtp.send_message(message)
        return True
    except (OSError, smtplib.SMTPException) as error:
        logger.error('Email notification failed for alert %s (%s).', safe_alert_id, type(error).__name__)
        return False
