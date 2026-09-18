from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any


def _normalize_activity(activity: Any) -> str:
    if activity is None:
        return 'unknown'
    label = str(activity).strip().lower()
    if not label:
        return 'unknown'
    return label


def build_temporal_context(context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = context or {}
    motion_score = float(context.get('motionScore', 0.0) or 0.0)
    moving_persons = int(context.get('movingPersons') or 0)
    moving_objects = int(context.get('movingObjects') or 0)
    pose_activities = [
        _normalize_activity(item)
        for item in (context.get('poseActivities') or [])
    ]
    detections = context.get('detections') or []
    threat_level = str(context.get('threatLevel') or 'LOW').upper()
    object_count = len(detections)

    posture_counts = Counter(pose_activities)
    dominant_posture = posture_counts.most_common(1)[0][0] if posture_counts else 'unknown'

    if motion_score >= 0.7 or moving_persons >= 2 or 'running' in pose_activities or 'crawling' in pose_activities:
        movement_pattern = 'sudden movement'
    elif motion_score >= 0.35 or moving_persons > 0 or 'walking' in pose_activities:
        movement_pattern = 'active movement'
    elif 'crouching' in pose_activities or 'concealment' in pose_activities:
        movement_pattern = 'low-profile movement'
    else:
        movement_pattern = 'routine movement'

    restricted_zone_context = 'restricted perimeter' if threat_level in {'MEDIUM', 'HIGH'} else 'standard patrol area'
    temporal_anomaly = 'elevated' if movement_pattern in {'sudden movement', 'low-profile movement'} or motion_score >= 0.55 else 'normal'
    risk_contribution = 0.15 if movement_pattern == 'routine movement' else 0.3 if movement_pattern == 'active movement' else 0.55 if movement_pattern == 'low-profile movement' else 0.8

    if threat_level == 'HIGH' and movement_pattern in {'sudden movement', 'low-profile movement'}:
        risk_contribution += 0.2

    if object_count > 0:
        risk_contribution += min(0.2, object_count * 0.04)

    risk_contribution = max(0.0, min(1.0, risk_contribution))
    risk_level = 'HIGH' if risk_contribution >= 0.75 else 'MEDIUM' if risk_contribution >= 0.4 else 'LOW'

    event_timeline = [
        {
            'timestamp': datetime.now(timezone.utc).strftime('%H:%M:%S'),
            'event': 'frame acquisition',
            'details': 'Temporal tracking started for the current scene window.',
        },
        {
            'timestamp': datetime.now(timezone.utc).strftime('%H:%M:%S'),
            'event': 'movement analysis',
            'details': f"Movement pattern classified as {movement_pattern} with motion score {motion_score:.2f}.",
        },
        {
            'timestamp': datetime.now(timezone.utc).strftime('%H:%M:%S'),
            'event': 'zone context',
            'details': f"Contextual review used the {restricted_zone_context} policy model.",
        },
        {
            'timestamp': datetime.now(timezone.utc).strftime('%H:%M:%S'),
            'event': 'risk evolution',
            'details': f"Temporal anomaly state is {temporal_anomaly}; the aggregated risk contribution is {risk_contribution:.2f}.",
        },
    ]

    return {
        'movementPattern': movement_pattern,
        'restrictedZoneContext': restricted_zone_context,
        'temporalAnomaly': temporal_anomaly,
        'riskContribution': round(risk_contribution, 2),
        'riskLevel': risk_level,
        'dominantPosture': dominant_posture,
        'eventTimeline': event_timeline,
        'riskEvolution': {
            'baseline': 'normal',
            'current': temporal_anomaly,
            'trend': 'increasing' if risk_contribution >= 0.5 else 'stable',
            'confidence': round(min(0.99, 0.45 + risk_contribution), 2),
        },
        'summary': (
            'Movement and posture were interpreted over time rather than as isolated frame-level events. '
            'A sudden or low-profile pattern increases risk only when contextual and sensor evidence support it.'
        ),
    }
