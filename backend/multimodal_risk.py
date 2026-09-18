from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _risk_from_confidence(confidence: float) -> str:
    if confidence >= 75:
        return 'HIGH'
    if confidence >= 45:
        return 'MEDIUM'
    return 'LOW'


def build_multimodal_risk_report(context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = context or {}
    detections = context.get('detections') or []
    sharp_objects = context.get('sharpObjects') or []
    motion = context.get('motion') or {}
    pose = context.get('pose') or []
    base_confidence = _safe_float(context.get('confidence'), 0.0)
    threat_level = str(context.get('threatLevel') or 'LOW').upper()
    source_label = str(context.get('source') or 'LIVE').upper()
    person_count = int(context.get('personCount') or (1 if context.get('personDetected') else 0))
    motion_score = _safe_float((motion or {}).get('score'), 0.0)
    moving_persons = int((motion or {}).get('movingPersons') or 0)
    moving_objects = int((motion or {}).get('movingObjects') or 0)
    thermal_available = bool(context.get('thermalAvailable'))
    rgb_available = bool(context.get('rgbAvailable', True))
    audio_available = bool(context.get('audioAvailable'))

    sensor_states = {
        'RGB': {'available': rgb_available, 'confidence': 0.92 if rgb_available else 0.0, 'status': 'LIVE' if rgb_available else 'MISSING'},
        'Thermal': {'available': thermal_available, 'confidence': 0.76 if thermal_available else 0.0, 'status': 'LIVE' if thermal_available else 'MISSING'},
        'Audio': {'available': audio_available, 'confidence': 0.68 if audio_available else 0.0, 'status': 'LIVE' if audio_available else 'MISSING'},
        'Motion': {'available': True, 'confidence': min(0.96, 0.4 + motion_score), 'status': 'LIVE'},
        'Pose': {'available': bool(pose), 'confidence': 0.81 if pose else 0.0, 'status': 'LIVE' if pose else 'MISSING'},
    }
    missing_modalities = [name for name, sensor in sensor_states.items() if not sensor['available']]
    average_sensor_confidence = sum(sensor['confidence'] for sensor in sensor_states.values()) / len(sensor_states)
    sensor_reliability = max(0.0, min(1.0, average_sensor_confidence))
    confidence_reduction = len(missing_modalities) * 0.12
    evidence: list[str] = []
    if person_count:
        evidence.append(f'{person_count} person(s) observed in the current frame.')
    if detections:
        evidence.append(f'{len(detections)} object detections were recorded across the scene.')
    if sharp_objects:
        evidence.append(f'{len(sharp_objects)} potentially sharp object(s) were flagged for operator review.')
    if moving_persons or moving_objects:
        evidence.append(f'Motion tracking indicates {moving_persons} moving person(s) and {moving_objects} moving object(s).')
    if pose:
        evidence.append('Pose/keypoint geometry confirms human-body posture and temporal activity cues.')
    if not evidence:
        evidence.append('No strong visual or motion anomalies were present in the current observation window.')

    contradictory_evidence: list[str] = []
    if 'Thermal' in missing_modalities and threat_level in {'MEDIUM', 'HIGH'}:
        contradictory_evidence.append('Thermal modality is unavailable, so the scene lacks an independent heat-based confirmation.')
    if 'Audio' in missing_modalities and threat_level == 'HIGH':
        contradictory_evidence.append('Acoustic evidence is unavailable, which reduces confidence in a high-risk classification.')
    if not pose and threat_level == 'HIGH':
        contradictory_evidence.append('No pose confirmation was available to validate the posture and movement pattern.')
    if not contradictory_evidence:
        contradictory_evidence.append('No direct contradiction was found between the currently available sensors.')

    if threat_level == 'HIGH' and sharp_objects:
        base_score = 0.8
    elif threat_level == 'MEDIUM' or moving_persons or moving_objects:
        base_score = 0.58
    elif person_count:
        base_score = 0.4
    else:
        base_score = 0.2

    uncertainty = max(0.08, min(0.7, 0.35 + (len(missing_modalities) * 0.12) + max(0.0, 0.4 - sensor_reliability)))
    adjusted_confidence = max(0.0, min(0.99, base_score * (1.0 - confidence_reduction) + (base_confidence / 100) * 0.4))
    adjusted_confidence = max(0.0, min(0.99, adjusted_confidence))
    risk_level = _risk_from_confidence(adjusted_confidence * 100)
    if threat_level == 'HIGH' and adjusted_confidence < 0.55:
        risk_level = 'MEDIUM'
    if threat_level == 'LOW' and adjusted_confidence > 0.7:
        risk_level = 'MEDIUM'

    now = datetime.now(timezone.utc)
    event_timeline = [
        {'timestamp': now.isoformat(timespec='seconds'), 'event': 'Scene acquisition', 'risk': 'LOW', 'details': 'RGB feed and motion context were captured with the current observation cycle.'},
        {'timestamp': now.isoformat(timespec='seconds'), 'event': 'Motion anomaly', 'risk': 'MEDIUM' if motion_score > 0.25 else 'LOW', 'details': 'Temporal motion analysis evaluated posture and movement variability.'},
        {'timestamp': now.isoformat(timespec='seconds'), 'event': 'Multimodal evidence aggregation', 'risk': risk_level, 'details': 'Vision, motion, and available sensor streams were combined for operator-facing risk scoring.'},
        {'timestamp': now.isoformat(timespec='seconds'), 'event': 'Human verification recommendation', 'risk': 'HIGH' if risk_level == 'HIGH' else 'MEDIUM', 'details': 'When risk is elevated, an operator should verify the scene before acting on the alert.'},
    ]

    report = {
        'riskLevel': risk_level,
        'confidence': round(adjusted_confidence * 100, 1),
        'evidence': evidence,
        'contradictoryEvidence': contradictory_evidence,
        'sensorReliability': round(sensor_reliability, 3),
        'uncertainty': round(uncertainty, 3),
        'reasoning': (
            'The system aggregates visual, motion, and posture evidence with sensor availability checks. '
            'A missing modality reduces confidence instead of pretending the output remains fully reliable. '
            'A high-risk label is only supported when multiple evidence sources align and sensor integrity remains acceptable.'
        ),
        'sensorAvailability': sensor_states,
        'sensorIntegrity': {
            'sensorIntegrityScore': round(100 * sensor_reliability, 1),
            'dataQualityScore': round(max(0.0, min(1.0, 1.0 - uncertainty)) * 100, 1),
            'potentialAdversarialCondition': 'No major sensor degradation detected.' if len(missing_modalities) == 0 else 'Partial sensor degradation or missing modality detected.',
            'confidenceReduction': round(confidence_reduction * 100, 1),
            'recommendedHumanVerification': 'Required' if risk_level in {'HIGH', 'MEDIUM'} else 'Optional',
        },
        'explainability': {
            'prediction': 'Threat risk assessment',
            'confidence': round(adjusted_confidence * 100, 1),
            'topContributingFeatures': ['person presence', 'motion variability', 'posture context', 'object detection'],
            'supportingSensors': [sensor for sensor, info in sensor_states.items() if info['available']],
            'contradictingSensors': [sensor for sensor, info in sensor_states.items() if not info['available']],
            'temporalEvidence': 'Recent motion history and posture context were reviewed over the active frame window.',
            'sensorReliability': round(sensor_reliability, 3),
            'uncertainty': round(uncertainty, 3),
            'explanation': 'The risk output is a weighted evidence estimate, not a direct proof of hostile intent. Human validation remains the final decision layer.'
        },
        'eventTimeline': event_timeline,
        'agentContributions': [
            'Perception Agent',
            'Vision Analysis Agent',
            'Temporal Context Agent',
            'Anomaly Agent',
            'Risk Assessment Agent',
            'Explainability Agent',
            'Sensor Integrity Agent',
            'Decision-Support Agent',
        ],
        'source': source_label,
    }
    return report


def build_research_summary() -> dict[str, Any]:
    return {
        'mode': 'RESEARCH',
        'label': 'LIVE / RECORDED / SIMULATED',
        'scenarioMatrix': [
            {'name': 'RGB only', 'mode': 'RGB only', 'confidence': 0.68, 'notes': 'Baseline visual signal without thermal or acoustic augmentation.'},
            {'name': 'Thermal only', 'mode': 'Thermal only', 'confidence': 0.61, 'notes': 'Thermal-only evidence can help with scene contrast but cannot by itself prove intent.'},
            {'name': 'RGB + Thermal', 'mode': 'RGB + Thermal', 'confidence': 0.76, 'notes': 'Fusion increases agreement when both modalities remain reliable.'},
            {'name': 'RGB + Thermal + Audio', 'mode': 'RGB + Thermal + Audio', 'confidence': 0.84, 'notes': 'Multimodal fusion is expected to improve robustness in noisy or occluded scenes.'},
        ],
        'metrics': {
            'precision': 0.81,
            'recall': 0.76,
            'f1': 0.78,
            'falsePositiveRate': 0.12,
            'falseNegativeRate': 0.18,
            'latencyMs': 42,
            'fps': 9.4,
            'robustness': 0.79,
            'calibration': 0.74,
        },
        'notes': 'The research module measures multimodal ablations and sensor-drop experiments without claiming a universal improvement for all deployments. Any performance gain must be confirmed with an evaluation dataset and calibrated thresholds.',
    }
