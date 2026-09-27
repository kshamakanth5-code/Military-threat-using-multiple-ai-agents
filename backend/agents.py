from __future__ import annotations

import json
import logging
import os
import sqlite3
import urllib.request
import uuid
from contextlib import closing
from datetime import datetime, timezone
from backend.email_service import send_threat_notification


logger = logging.getLogger(__name__)


def _supabase_request(method: str, path: str, body: dict | None = None, *, prefer: str | None = None):
    """Make a server-side Supabase REST request without exposing service credentials."""
    url = os.getenv('SUPABASE_URL', '').rstrip('/')
    key = os.getenv('SUPABASE_SERVICE_ROLE_KEY', '')
    if not url or not key:
        raise RuntimeError('Supabase server credentials are not configured.')
    data = json.dumps(body).encode('utf-8') if body is not None else None
    request = urllib.request.Request(
        f'{url}/rest/v1/{path}',
        data=data,
        method=method,
        headers={
            'apikey': key,
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json',
            **({'Prefer': prefer} if prefer else {}),
        },
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = response.read()
        return json.loads(payload) if payload else None


def _deliver_queued_alert(alert_id: str, supabase_id: str, confidence: float, threat_level: str,
                          threat_type: str, location: str | None, message: str, detected_at: str) -> dict:
    """Claim a Supabase alert, send with server-side SMTP, then record acceptance."""
    recipient = os.getenv('ALERT_EMAIL', '').strip()
    if not recipient or '@' not in recipient:
        logger.error('ALERT_EMAIL is missing or invalid for alert %s.', alert_id)
        status, error = 'failed', 'ALERT_EMAIL is missing or invalid.'
    else:
        claim_path = f'threat_alerts?id=eq.{supabase_id}&email_status=eq.pending&select=id'
        try:
            claimed = _supabase_request('PATCH', claim_path, {'email_status': 'processing'}, prefer='return=representation')
        except Exception as error:
            logger.exception('Could not claim alert %s for SMTP delivery.', alert_id)
            return {'ok': False, 'emailStatus': 'PENDING', 'error': f'Could not claim alert ({type(error).__name__}).'}
        if not claimed:
            return {'ok': True, 'emailStatus': 'PENDING', 'skipped': True, 'reason': 'duplicate_or_already_processing'}

        accepted = send_threat_notification(
            recipient, confidence, threat_level, message, detected_at, alert_id,
            threat_type=threat_type, location=location,
        )
        if accepted:
            status, error = 'sent', None
            logger.info('SMTP server accepted threat alert %s for %s.', alert_id, recipient)
        else:
            status, error = 'failed', 'SMTP server did not accept the threat alert.'
            logger.error('SMTP delivery failed for threat alert %s.', alert_id)

    update = {
        'email_status': status,
        'sent_at': datetime.now(timezone.utc).isoformat() if status == 'sent' else None,
        'sent_to': recipient if status == 'sent' else None,
        'email_recipient': recipient if status == 'sent' else None,
    }
    try:
        _supabase_request(
            'PATCH',
            f'threat_alerts?id=eq.{supabase_id}&email_status=eq.processing' if recipient else f'threat_alerts?id=eq.{supabase_id}&email_status=eq.pending',
            update,
            prefer='return=minimal',
        )
    except Exception as update_error:
        logger.exception('SMTP result could not be recorded for threat alert %s.', alert_id)
        return {'ok': False, 'emailStatus': 'FAILED' if status != 'sent' else 'PROCESSING',
                'error': f'Email status update failed ({type(update_error).__name__}).'}
    return {'ok': status == 'sent', 'emailStatus': status.upper(), 'recipient': recipient if status == 'sent' else None,
            **({'error': error} if error else {})}


def send_test_threat_email() -> dict:
    """Queue and deliver a clearly labeled test message through the real SMTP path."""
    alert_id = str(uuid.uuid4())
    detected_at = datetime.now(timezone.utc).isoformat()
    _supabase_request('POST', 'threat_alerts', {
        'id': alert_id,
        'threat_type': 'Email delivery test',
        'threat_level': 'HIGH',
        'confidence': 75,
        'location': 'Test mode',
        'detected_at': detected_at,
        'message': 'This is a test message. No real threat was detected.',
        'email_status': 'pending',
    }, prefer='return=minimal')
    return _deliver_queued_alert(
        f'ATAS-TEST-{alert_id[:8]}', alert_id, 100, 'HIGH', 'Email delivery test',
        'Test mode', 'This is a test message. No real threat was detected.', detected_at,
    )


def process_threat_event(event: dict, user: dict, database_path) -> dict:
    """Queue a qualifying alert in Supabase and send it through configured SMTP."""
    cooldown_minutes = max(0, int(os.getenv('ALERT_COOLDOWN_MINUTES', '5')))
    risk_score = max(0.0, min(float(event.get('riskScore', 0)), 100.0))
    confidence = risk_score if risk_score <= 1 else risk_score / 100
    stored_confidence = round(confidence * 100, 2)
    user_id = str(user.get('userId', ''))
    email = str(user.get('email', '')).strip().lower()
    result = {
        'alertId': None,
        'userId': user_id,
        'riskScore': stored_confidence,
        'threatLevel': str(event.get('threatLevel', 'LOW')).upper(),
        'emailSent': False,
        'emailStatus': 'NOT_ELIGIBLE',
        'emailThreshold': 0.75,
    }
    if confidence < 0.75:
        return result

    now = datetime.now(timezone.utc)
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute(
                'CREATE TABLE IF NOT EXISTS alerts ('
                'alert_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, email TEXT NOT NULL, '
                'risk_score REAL NOT NULL, threat_level TEXT NOT NULL, reason TEXT NOT NULL, '
                'timestamp TEXT NOT NULL, email_sent INTEGER NOT NULL, email_status TEXT NOT NULL, '
                'created_at TEXT NOT NULL)'
            )
            recent = connection.execute(
                "SELECT 1 FROM alerts WHERE user_id = ? "
                "AND julianday(created_at) >= julianday('now', ?) LIMIT 1",
                (user_id, f'-{cooldown_minutes} minutes'),
            ).fetchone()
            alert_id = f"ATAS-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
            result['alertId'] = alert_id
            if recent:
                result['emailStatus'] = 'COOLDOWN'
                return result
            threat_level = result['threatLevel']
            threat_type = str(event.get('threatType') or event.get('activity') or event.get('source') or 'Threat detected')[:120]
            location = event.get('location')
            detected_at = str(event.get('timestamp') or now.isoformat())
            message = str(event.get('reason') or 'Threat confidence exceeded the configured email threshold.')[:1000]
            supabase_id = str(uuid.uuid4())
            try:
                _supabase_request(
                    'POST',
                    'threat_alerts',
                    {
                        'id': supabase_id,
                        'threat_type': threat_type,
                        'threat_level': threat_level if threat_level in {'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'} else 'HIGH',
                        'confidence': stored_confidence,
                        'location': location,
                        'detected_at': detected_at,
                        'message': message,
                        'email_recipient': None,
                        'email_status': 'pending',
                    },
                    prefer='return=minimal',
                )
            except Exception as error:
                result['emailStatus'] = 'FAILED'
                result['error'] = f'Supabase alert insert failed ({type(error).__name__}).'
                logger.error('Supabase alert insert failed for alert %s (%s).', alert_id, type(error).__name__)
                connection.execute(
                    'INSERT INTO alerts (alert_id, user_id, email, risk_score, threat_level, reason, timestamp, email_sent, email_status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)',
                    (alert_id, user_id, email, risk_score, threat_level, message[:500], detected_at, result['emailStatus'], now.isoformat()),
                )
                return result

            delivery = _deliver_queued_alert(alert_id, supabase_id, stored_confidence, threat_level,
                                             threat_type, location, message, detected_at)
            result['emailStatus'] = delivery['emailStatus']
            result['emailSent'] = delivery['emailStatus'] == 'SENT'
            if delivery.get('error'):
                result['error'] = delivery['error']
            connection.execute(
                'INSERT INTO alerts (alert_id, user_id, email, risk_score, threat_level, reason, timestamp, email_sent, email_status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (alert_id, user_id, email, stored_confidence, result['threatLevel'], str(event.get('reason', ''))[:500], event.get('timestamp', now.isoformat()), int(result['emailSent']), result['emailStatus'], now.isoformat()),
            )
    return result


def build_agent_mesh(dataset_info: dict, threat_level: str = 'MEDIUM', action: str = 'perimeter scan'):
    normal_count = next((item['sample_count'] for item in dataset_info.get('classes', []) if item['label'] == 'Normal Class'), 0)
    crossing_count = next((item['sample_count'] for item in dataset_info.get('classes', []) if item['label'] == 'Wall crossing'), 0)

    threat = str(threat_level).upper()
    is_high = threat in {'HIGH', 'CRITICAL', 'CRIME'}
    is_medium = threat in {'MEDIUM', 'SUSPICIOUS'}

    detection_status = 'Alert' if is_high else 'Scanning'
    pose_status = 'Crouched posture' if is_high else 'Steady posture'
    motion_status = 'Concealment pattern' if is_high else 'Loitering pattern' if is_medium else 'Routine movement'
    intent_status = 'Covert intrusion' if is_high else 'Suspicious intent' if is_medium else 'Authorized patrol'
    threat_status = 'High risk' if is_high else 'Medium risk' if is_medium else 'Low risk'
    explanation_status = 'Explaining causal chain' if is_high else 'Reasoning over scene'
    response_status = 'Alarm triggered' if is_high else 'Operator warning' if is_medium else 'Monitor only'

    return [
        {
            'id': 'D1',
            'name': 'Detection Agent (YOLOv8)',
            'status': detection_status,
            'color': '#06b6d4',
            'payload': {
                'model': 'YOLOv8',
                'entity': 'Person',
                'confidence': 97 if is_high else 84,
                'bbox': [24, 18, 52, 73],
                'classes_detected': ['Person', 'Vehicle', 'Firearm', 'Package'],
                'flags': ['perimeter scan', 'person presence', action],
                'dataset': {'normal_frames': normal_count, 'wall_crossing_frames': crossing_count},
            },
        },
        {
            'id': 'D2',
            'name': 'Pose Extraction Agent (YOLO11 Pose)',
            'status': pose_status,
            'color': '#06b6d4',
            'payload': {
                'model': 'YOLO11 pose checkpoint',
                'keypoints': 17,
                'pose': 'crouched' if is_high else 'upright',
                'orientation': 'north-east',
                'posture': pose_status,
                'body_joints': ['Head', 'Shoulders', 'Elbows', 'Hands', 'Hips', 'Knees', 'Feet'],
            },
        },
        {
            'id': 'D3',
            'name': 'Activity Recognition Agent',
            'status': 'Analyzing motion',
            'color': '#22c55e',
            'payload': {
                'model': 'Pose geometry + temporal tracking heuristic',
                'activity': 'concealment' if is_high else 'loitering' if is_medium else 'walking',
                'motion': motion_status,
                'confidence': 0.93 if is_high else 0.79,
                'states': ['standing', 'walking', 'running', 'crawling', 'sleeping'],
            },
        },
        {
            'id': 'D4',
            'name': 'Intent Prediction Agent',
            'status': 'Predicting intent',
            'color': '#f59e0b',
            'payload': {
                'model': 'Trajectory predictor',
                'trajectory': 'restricted approach' if is_high else 'perimeter drift',
                'confidence': 0.89 if is_high else 0.74,
                'intent': intent_status,
                'dwell': 2.6,
                'targets': ['Perimeter Approach', 'Infiltration', 'Concealment'],
            },
        },
        {
            'id': 'D5',
            'name': 'Threat Analysis Agent (LLM Engine)',
            'status': threat_status,
            'color': '#ef4444',
            'payload': {
                'model': 'LLM engine',
                'threatLevel': 'HIGH' if is_high else 'MEDIUM' if is_medium else 'LOW',
                'zone': 'Zone A',
                'rulesTriggered': ['restricted boundary', 'concealment', 'weapon detection'] if is_high else ['restricted boundary', 'loitering'] if is_medium else ['routine transit'],
                'context_summary': 'Context, zone policies, and intent history were fused to assign the threat level.',
            },
        },
        {
            'id': 'D6',
            'name': 'Explainability Agent (XAI)',
            'status': explanation_status,
            'color': '#06b6d4',
            'payload': {
                'model': 'XAI narrative engine',
                'rationale': 'Object tracking indicates a person approaching a restricted boundary with concealment and hostile intent patterns. The model combines posture, motion, environment, and dwell time.',
                'confidence': 0.91 if is_high else 0.78,
                'reasoning': 'Human-readable justification generated for operator review.',
            },
        },
        {
            'id': 'D7',
            'name': 'Alert & Learning Agent',
            'status': response_status,
            'color': '#ef4444',
            'payload': {
                'model': 'Response policy + feedback loop',
                'action': 'alarm_triggered' if is_high else 'notify_operator' if is_medium else 'monitor_only',
                'modelUpdate': 'retention_window_2m',
                'objectFocus': ['Pen', 'Weapon', 'Vehicle'] if is_high else ['Person', 'Package'],
                'learning': 'State outputs are logged for model refinement and continuous retraining.',
            },
        },
    ]


def agent_catalog():
    return [
        'Detection Agent (YOLOv8)',
        'Pose Extraction Agent (MediaPipe)',
        'Activity Recognition Agent',
        'Intent Prediction Agent',
        'Threat Analysis Agent (LLM Engine)',
        'Explainability Agent (XAI)',
        'Alert & Learning Agent',
    ]
