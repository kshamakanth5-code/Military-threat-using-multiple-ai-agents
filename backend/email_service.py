from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage
from html import escape

logger = logging.getLogger(__name__)


def _clean(value: object, limit: int = 500) -> str:
    return escape(str(value or '').strip())[:limit]


def send_threat_notification(
    recipient_email: str,
    risk_score: float,
    threat_level: str,
    reason: str,
    timestamp: str,
    alert_id: str,
) -> bool:
    host = os.getenv('SMTP_HOST')
    port = int(os.getenv('SMTP_PORT', '587'))
    sender = os.getenv('SMTP_FROM')
    username = os.getenv('SMTP_USERNAME')
    password = os.getenv('SMTP_PASSWORD')
    if not all((host, sender, username, password)):
        return False

    score = max(0.0, min(float(risk_score), 100.0))
    level = _clean(threat_level, 32).upper()
    safe_reason = _clean(reason)
    safe_timestamp = _clean(timestamp, 80)
    safe_alert_id = _clean(alert_id, 80)

    message = EmailMessage()
    message['Subject'] = f'ATAS CRITICAL THREAT ALERT - Risk {score:.0f}%'
    message['From'] = sender
    message['To'] = recipient_email
    message.set_content(
        'ATAS THREAT ALERT\n\n'
        'A high-risk event has been detected.\n\n'
        f'Risk Score: {score:.0f}%\n'
        f'Threat Level: {level}\n\n'
        f'Reason:\n{safe_reason}\n\n'
        f'Time:\n{safe_timestamp}\n\n'
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
        logger.error('Email notification failed for alert %s: %s', safe_alert_id, error)
        return False
