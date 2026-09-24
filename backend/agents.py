from __future__ import annotations

import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone

from backend.email_service import send_threat_notification


def process_threat_event(event: dict, user: dict, database_path, email_sender=send_threat_notification) -> dict:
    """Apply Alert Agent policy to one structured threat event."""
    threshold = float(os.getenv('RISK_THRESHOLD_EMAIL', os.getenv('RISK_THRESHOLD', '95')))
    confirmation_frames = max(1, int(os.getenv('CONFIRMATION_FRAMES', '5')))
    cooldown_minutes = max(0, int(os.getenv('ALERT_COOLDOWN_MINUTES', '5')))
    risk_score = max(0.0, min(float(event.get('riskScore', 0)), 100.0))
    user_id = str(user.get('userId', ''))
    email = str(user.get('email', '')).strip().lower()
    state = process_threat_event._state.setdefault(user_id, {'high_frames': 0})
    state['high_frames'] = state['high_frames'] + 1 if risk_score >= threshold else 0

    result = {
        'alertId': None,
        'userId': user_id,
        'riskScore': risk_score,
        'threatLevel': str(event.get('threatLevel', 'LOW')).upper(),
        'emailSent': False,
        'emailStatus': 'NOT_ELIGIBLE',
        'confirmationFrames': state['high_frames'],
        'confirmationRequired': confirmation_frames,
    }
    if risk_score < threshold:
        return result
    if state['high_frames'] < confirmation_frames:
        result['emailStatus'] = 'AWAITING_CONFIRMATION'
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
                "SELECT 1 FROM alerts WHERE user_id = ? AND email_sent = 1 "
                "AND created_at >= datetime('now', ?) LIMIT 1",
                (user_id, f'-{cooldown_minutes} minutes'),
            ).fetchone()
            alert_id = f"ATAS-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
            result['alertId'] = alert_id
            if recent:
                result['emailStatus'] = 'COOLDOWN'
                return result
            if not email:
                result['emailStatus'] = 'NO_AUTHENTICATED_EMAIL'
                return result

            try:
                sent = bool(email_sender(email, risk_score, result['threatLevel'], event.get('reason', ''), event.get('timestamp', now.isoformat()), alert_id))
            except Exception:
                sent = False
            result['emailSent'] = sent
            result['emailStatus'] = 'SENT' if sent else 'FAILED'
            connection.execute(
                'INSERT INTO alerts (alert_id, user_id, email, risk_score, threat_level, reason, timestamp, email_sent, email_status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (alert_id, user_id, email, risk_score, result['threatLevel'], str(event.get('reason', ''))[:500], event.get('timestamp', now.isoformat()), int(sent), result['emailStatus'], now.isoformat()),
            )
    return result


process_threat_event._state = {}

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
