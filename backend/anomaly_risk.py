from __future__ import annotations

from typing import Any


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _risk_from_confidence(value: float) -> str:
    if value >= 75:
        return 'HIGH'
    if value >= 45:
        return 'MEDIUM'
    return 'LOW'


def build_anomaly_risk_report(context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = context or {}
    base_risk_level = str(context.get('riskLevel') or 'LOW').upper()
    base_confidence = _float(context.get('confidence'), 0.0)
    sensor_reliability = max(0.0, min(1.0, _float(context.get('sensorReliability'), 0.8)))
    temperature_anomaly = max(0.0, min(1.0, _float(context.get('temperatureAnomaly'), 0.0)))
    motion_anomaly = max(0.0, min(1.0, _float(context.get('motionAnomaly'), 0.0)))
    audio_event = str(context.get('audioEvent') or 'none').lower()
    zone_context = str(context.get('zoneContext') or 'routine patrol').lower()
    historical_context = str(context.get('historicalContext') or 'normal activity').lower()

    anomaly_score = 0.0
    anomaly_score += temperature_anomaly * 0.35
    anomaly_score += motion_anomaly * 0.35
    anomaly_score += (0.2 if audio_event not in {'', 'none', 'normal'} else 0.0)
    if 'restricted' in zone_context or 'secure' in zone_context:
        anomaly_score += 0.15
    if 'repeated' in historical_context or 'suspicious' in historical_context or 'dwell' in historical_context:
        anomaly_score += 0.15

    anomaly_score = max(0.0, min(1.0, anomaly_score))
    confidence_adjustment = (1.0 - sensor_reliability) * 0.35 + anomaly_score * 0.55
    adjusted_confidence = max(0.0, min(100.0, base_confidence + (anomaly_score * 40.0) - (confidence_adjustment * 25.0)))
    risk_level = _risk_from_confidence(adjusted_confidence)

    if base_risk_level == 'HIGH' and adjusted_confidence < 70:
        risk_level = 'MEDIUM'
    if base_risk_level == 'LOW' and adjusted_confidence > 70:
        risk_level = 'MEDIUM'

    anomaly_status = 'anomaly_detected' if anomaly_score >= 0.35 else 'normal'
    evidence = []
    if temperature_anomaly > 0.4:
        evidence.append('Thermal pattern deviates from the expected baseline for the current scene.')
    if motion_anomaly > 0.4:
        evidence.append('Motion profile indicates temporal inconsistency or unusual activity.')
    if audio_event not in {'', 'none', 'normal'}:
        evidence.append(f'Acoustic evidence indicates {audio_event}, which is treated as supporting evidence only.')
    if 'restricted' in zone_context:
        evidence.append('The subject is in a restricted-zone context, which increases the contextual risk weight.')
    if not evidence:
        evidence.append('No strong anomaly indicators were present beyond baseline motion and context.')

    contradictions = []
    if sensor_reliability < 0.6:
        contradictions.append('Sensor reliability is reduced; the evidence should be treated with caution.')
    if temperature_anomaly < 0.2 and motion_anomaly > 0.7:
        contradictions.append('Motion suggests anomaly even though thermal evidence is calm; this may represent a false-positive risk or a context mismatch.')
    if not contradictions:
        contradictions.append('No direct contradiction was detected between sensor streams.')

    return {
        'anomalyStatus': anomaly_status,
        'riskLevel': risk_level,
        'confidence': round(adjusted_confidence, 1),
        'sensorReliability': round(sensor_reliability, 3),
        'uncertainty': round(max(0.05, min(0.75, 0.25 + (1.0 - sensor_reliability) * 0.55 + anomaly_score * 0.25)), 3),
        'evidence': evidence,
        'contradictoryEvidence': contradictions,
        'reasoning': (
            'The anomaly engine combines thermal, motion, acoustic, and context evidence. '
            'It does not treat any single factor as proof of hostile intent; instead, it adjusts confidence and recommends human review when context or sensor quality weakens the signal.'
        ),
        'environmentalFactors': {
            'temperatureAnomaly': round(temperature_anomaly, 3),
            'motionAnomaly': round(motion_anomaly, 3),
            'audioEvent': audio_event,
            'zoneContext': zone_context,
            'historicalContext': historical_context,
        },
        'recommendedHumanVerification': 'Required' if risk_level in {'MEDIUM', 'HIGH'} or anomaly_status == 'anomaly_detected' else 'Optional',
        'anomalyScore': round(anomaly_score, 3),
    }
