from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image

EMOTION_MODEL_ID = os.getenv('EMOTION_MODEL_ID', 'dima806/facial_emotions_image_detection')
emotion_classifier = None
emotion_classifier_attempted = False


def _empty_report(reason: str = 'No face detected') -> dict[str, Any]:
    return {
        'expression': 'unknown',
        'emotion': 'unknown',
        'confidence': 0,
        'emotionConfidence': 0,
        'emotionScores': [],
        'faceDetected': False,
        'faceCount': 0,
        'faceBox': None,
        'method': 'unavailable',
        'emotionModel': EMOTION_MODEL_ID,
        'reason': reason,
        'limitations': 'Expression is contextual evidence, not proof of intent or future behavior.',
    }


def analyze_facial_expression(image: Image.Image) -> dict[str, Any]:
    """Estimate visible expression with an auditable OpenCV baseline.

    The bundled cascades can detect faces and smiles, but they are not a
    general emotion classifier. Unknown is returned when the signal is weak.
    """
    try:
        import cv2
    except ImportError:
        return _empty_report('OpenCV is unavailable')

    if not all(hasattr(cv2, name) for name in ('CascadeClassifier', 'cvtColor', 'data')):
        return _empty_report('OpenCV cascade support is unavailable')

    frame = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(36, 36))
    if len(faces) == 0:
        return _empty_report()

    x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
    face_rgb = np.asarray(image.convert('RGB'))[y:y + height, x:x + width]
    face = frame[y:y + height, x:x + width]
    smiles = smile_cascade.detectMultiScale(
        face,
        scaleFactor=1.7,
        minNeighbors=20,
        minSize=(max(12, width // 5), max(8, height // 10)),
    )
    face_box = [
        round(x / frame.shape[1] * 100, 2),
        round(y / frame.shape[0] * 100, 2),
        round(width / frame.shape[1] * 100, 2),
        round(height / frame.shape[0] * 100, 2),
    ]
    classifier = _load_emotion_classifier()
    emotion_scores = []
    if classifier is not None:
        try:
            predictions = classifier(Image.fromarray(face_rgb), top_k=7)
            emotion_scores = [
                {'label': str(item['label']).lower(), 'confidence': round(float(item['score']) * 100, 1)}
                for item in predictions
            ]
        except Exception:
            emotion_scores = []
    if emotion_scores:
        emotion = emotion_scores[0]['label']
        confidence = round(emotion_scores[0]['confidence'])
        expression = emotion
        method = f'huggingface:{EMOTION_MODEL_ID}'
        reason = 'Emotion classifier evaluated the largest detected face crop.'
    elif len(smiles) > 0:
        emotion = 'happy'
        expression = emotion
        confidence = min(82, 56 + len(smiles) * 8)
        method = 'opencv-haar-smile-fallback'
        reason = 'Emotion model unavailable; smile cascade detected a positive facial cue.'
    else:
        emotion = 'unknown'
        expression = emotion
        confidence = 0
        method = 'opencv-haar-face-only'
        reason = 'Face detected, but the emotion model is unavailable and no smile cue was found.'

    return {
        'expression': expression,
        'emotion': emotion,
        'confidence': confidence,
        'emotionConfidence': confidence,
        'emotionScores': emotion_scores,
        'faceDetected': True,
        'faceCount': len(faces),
        'faceBox': face_box,
        'method': method,
        'emotionModel': EMOTION_MODEL_ID,
        'reason': reason,
        'limitations': 'Emotion predictions are probabilistic and affected by lighting, pose, occlusion, dataset bias, and cultural context. They must not be treated as proof of intent.',
    }


def predict_next_actions(expression_report: dict[str, Any], activity: str, threat_level: str, sharp_object_count: int) -> dict[str, Any]:
    """Rank observable next-action hypotheses from multimodal context."""
    expression = expression_report.get('emotion') or expression_report.get('expression', 'unknown')
    threat = str(threat_level).upper()
    activity_label = str(activity or 'unknown')
    scores = {
        'continue_current_activity': 0.46,
        'approach_or_cross_boundary': 0.18,
        'pause_or_change_direction': 0.18,
        'flee_or_run': 0.10,
        'interact_with_object': 0.08,
    }
    evidence = []
    if expression == 'happy':
        scores['continue_current_activity'] += 0.14
        scores['pause_or_change_direction'] -= 0.04
        evidence.append('positive facial cue')
    elif expression == 'neutral':
        evidence.append('neutral facial cue')
    else:
        evidence.append('facial expression unavailable')
    if activity_label in {'walking', 'running', 'crawling'}:
        scores['approach_or_cross_boundary'] += 0.12
    if activity_label == 'running':
        scores['flee_or_run'] += 0.22
    if activity_label in {'standing', 'sitting'}:
        scores['pause_or_change_direction'] += 0.08
    if threat in {'HIGH', 'CRITICAL'}:
        scores['approach_or_cross_boundary'] += 0.16
    if sharp_object_count:
        scores['interact_with_object'] += 0.18
    total = sum(max(value, 0.01) for value in scores.values())
    ranked = [
        {'action': action, 'probability': round(max(value, 0.01) / total, 2)}
        for action, value in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]
    return {
        'topPrediction': ranked[0]['action'],
        'confidence': round(ranked[0]['probability'] * 100),
        'rankedActions': ranked,
        'evidence': evidence,
        'decisionPolicy': 'Decision support only; verify trajectory and context before intervention.',
    }


def _load_emotion_classifier():
    global emotion_classifier, emotion_classifier_attempted
    if emotion_classifier_attempted:
        return emotion_classifier
    emotion_classifier_attempted = True
    try:
        from transformers import pipeline
        emotion_classifier = pipeline('image-classification', model=EMOTION_MODEL_ID, device=-1)
    except Exception:
        emotion_classifier = None
    return emotion_classifier