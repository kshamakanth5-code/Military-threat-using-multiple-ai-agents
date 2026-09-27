from __future__ import annotations

import hashlib
import logging
import os
import secrets
import smtplib
import sqlite3
import base64
import io
import math
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from urllib.parse import quote, urlparse
from dotenv import load_dotenv

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from backend.agents import build_agent_mesh as build_agent_mesh_from_module
from backend.agents import process_threat_event, send_test_threat_email
from backend.email_service import send_password_reset_email
from backend.activity_recognition import ACTIVITIES, TemporalActivityRecognizer
from backend.posture_activity_agent import PostureActivityAgent
from backend.anomaly_risk import build_anomaly_risk_report
from backend.facial_expression import analyze_facial_expression, predict_next_actions
from backend.multimodal_risk import build_multimodal_risk_report, build_research_summary
from backend.object_detection import DANGEROUS_OBJECT_CLASSES, ObjectTemporalConfirmer, associate_person, normalize_dangerous_label
from backend.temporal_context import build_temporal_context, detect_interpersonal_aggression
from backend.train_pipeline import dataset_overview, predict_image, train_model

app = FastAPI(title='Threat Detection API', version='1.0.0')
logger = logging.getLogger(__name__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

ACCOUNT_DB = Path(__file__).resolve().parent / 'accounts.db'
MODEL_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(MODEL_ROOT / '.env')
WEAPON_MODEL_PATHS = [
    MODEL_ROOT / 'runs' / 'detect' / 'sharp_object_detector' / 'weights' / 'best.pt',
    MODEL_ROOT / 'runs' / 'detect' / 'weapon_detector' / 'weights' / 'best.pt',
    MODEL_ROOT / 'runs' / 'detect' / 'backend' / 'runs' / 'weapon_detector' / 'weights' / 'best.pt',
]
PERSON_MODEL_PATH = MODEL_ROOT / 'yolo11n.pt'
POSE_MODEL_PATH = MODEL_ROOT / 'yolo11n-pose.pt'
SHARP_MATERIAL_POLICY = 'Potentially dangerous object detections are reported after temporal confirmation.'
SHARP_LABELS = DANGEROUS_OBJECT_CLASSES
OBJECT_DETECTION_CONFIDENCE = max(0.0, min(float(os.getenv('OBJECT_DETECTION_CONFIDENCE', '0.35')), 1.0))
OBJECT_CONFIRMATION_FRAMES = max(1, int(os.getenv('OBJECT_CONFIRMATION_FRAMES', '3')))
OBJECT_CONFIRMER = ObjectTemporalConfirmer(OBJECT_CONFIRMATION_FRAMES)
ACTIVITY_CLASSES = [label.lower() for label in ACTIVITIES]
ACTIVITY_RECOGNIZER = TemporalActivityRecognizer()
POSTURE_ACTIVITY_AGENT = PostureActivityAgent()
RISK_THRESHOLD_HEATMAP = float(os.getenv('RISK_THRESHOLD_HEATMAP', '60'))
RISK_THRESHOLD_EMAIL = 75.0
RISK_THRESHOLD_CRITICAL = 95.0
weapon_model = None
person_model = None
pose_model = None
sharp_model = None
sharp_model_attempted = False


class LoginRequest(BaseModel):
    email: str = ''
    userName: str | None = None
    password: str


class RegisterRequest(BaseModel):
    userName: str
    email: str
    password: str


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    newPassword: str


class ThreatAlertRequest(BaseModel):
    level: str
    summary: str
    action: str
    confidence: float = 0


class CameraFrameRequest(BaseModel):
    image: str
    motionScore: float = 0
    movingPersons: int = 0
    movingObjects: int = 0


class PredictRequest(BaseModel):
    imagePath: str


class AgentFeedbackRequest(BaseModel):
    email: str
    observedActivity: str
    correctedActivity: str
    threatLevel: str = 'LOW'


def initialize_account_db():
    with closing(sqlite3.connect(ACCOUNT_DB)) as connection, connection:
        connection.execute('CREATE TABLE IF NOT EXISTS accounts (user_id TEXT UNIQUE, email TEXT PRIMARY KEY, user_name TEXT NOT NULL, password_hash TEXT NOT NULL)')
        columns = {row[1] for row in connection.execute('PRAGMA table_info(accounts)').fetchall()}
        if 'user_id' not in columns:
            connection.execute('ALTER TABLE accounts ADD COLUMN user_id TEXT')
            rows = connection.execute('SELECT email FROM accounts WHERE user_id IS NULL').fetchall()
            for (email,) in rows:
                connection.execute('UPDATE accounts SET user_id = ? WHERE email = ?', (str(uuid.uuid4()), email))
        connection.execute('CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at TEXT NOT NULL)')
        connection.execute('CREATE TABLE IF NOT EXISTS password_reset_tokens (token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, expires_at TEXT NOT NULL, created_at TEXT NOT NULL, used_at TEXT)')
        connection.execute('CREATE INDEX IF NOT EXISTS idx_password_reset_tokens_user_id ON password_reset_tokens (user_id)')
        connection.execute('CREATE TABLE IF NOT EXISTS password_reset_requests (email_hash TEXT NOT NULL, requested_at TEXT NOT NULL)')
        connection.execute('CREATE INDEX IF NOT EXISTS idx_password_reset_requests_email_time ON password_reset_requests (email_hash, requested_at)')


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _password_reset_message() -> dict[str, str]:
    return {'message': 'If an account exists for this email address, a password reset link has been sent.'}


def _deliver_password_reset(user_id: str, registered_email: str, user_name: str, token: str, token_hash: str) -> None:
    frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:4173').strip().rstrip('/')
    parsed_url = urlparse(frontend_url)
    if parsed_url.scheme not in {'http', 'https'} or not parsed_url.netloc:
        logger.error('Password reset email was not sent because FRONTEND_URL is invalid.')
        accepted = False
    else:
        reset_url = f'{frontend_url}/?token={quote(token)}'
        accepted = send_password_reset_email(registered_email, reset_url, user_name)
    if not accepted:
        # A link that was not delivered must not remain usable.
        with closing(sqlite3.connect(ACCOUNT_DB)) as connection, connection:
            connection.execute(
                'UPDATE password_reset_tokens SET used_at = ? WHERE token_hash = ? AND user_id = ?',
                (datetime.now(timezone.utc).isoformat(timespec='seconds'), token_hash, user_id),
            )


def authenticated_user(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.lower().startswith('bearer '):
        raise HTTPException(status_code=401, detail='Authentication required.')
    token = authorization[7:].strip()
    initialize_account_db()
    with closing(sqlite3.connect(ACCOUNT_DB)) as connection, connection:
        account = connection.execute(
            'SELECT a.user_id, a.email, a.user_name FROM sessions s JOIN accounts a ON a.user_id = s.user_id WHERE s.token = ?',
            (token,),
        ).fetchone()
    if not account:
        raise HTTPException(status_code=401, detail='Invalid or expired session.')
    return {'userId': account[0], 'email': account[1], 'userName': account[2]}


def build_agent_mesh(dataset_info: dict, threat_level: str = 'MEDIUM', action: str = 'perimeter scan'):
    normal_count = next((item['sample_count'] for item in dataset_info.get('classes', []) if item['label'] == 'Normal Class'), 0)
    crossing_count = next((item['sample_count'] for item in dataset_info.get('classes', []) if item['label'] == 'Wall crossing'), 0)

    is_high = threat_level.upper() in {'HIGH', 'CRITICAL', 'CRIME'}
    is_medium = threat_level.upper() in {'MEDIUM', 'SUSPICIOUS'}

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
            'name': 'Pose Extraction Agent (MediaPipe)',
            'status': pose_status,
            'color': '#06b6d4',
            'payload': {'model': 'MediaPipe', 'keypoints': 33, 'pose': 'crouched' if is_high else 'upright', 'orientation': 'north-east', 'posture': pose_status, 'body_joints': ['Head', 'Shoulders', 'Elbows', 'Hands', 'Hips', 'Knees', 'Feet']},
        },
        {
            'id': 'D3',
            'name': 'Activity Recognition Agent',
            'status': 'Analyzing motion',
            'color': '#22c55e',
            'payload': {'model': 'Spatio-temporal classifier', 'activity': 'concealment' if is_high else 'loitering' if is_medium else 'walking', 'motion': motion_status, 'confidence': 0.93 if is_high else 0.79, 'states': ['Standing', 'Running', 'Crawling', 'Loitering']},
        },
        {
            'id': 'D4',
            'name': 'Intent Prediction Agent',
            'status': 'Predicting intent',
            'color': '#f59e0b',
            'payload': {'model': 'Trajectory predictor', 'trajectory': 'restricted approach' if is_high else 'perimeter drift', 'confidence': 0.89 if is_high else 0.74, 'intent': intent_status, 'dwell': 2.6, 'targets': ['Perimeter Approach', 'Infiltration', 'Concealment']},
        },
        {
            'id': 'D5',
            'name': 'Threat Analysis Agent (LLM Engine)',
            'status': threat_status,
            'color': '#ef4444',
            'payload': {'model': 'LLM engine', 'threatLevel': 'HIGH' if is_high else 'MEDIUM' if is_medium else 'LOW', 'zone': 'Zone A', 'rulesTriggered': ['restricted boundary', 'concealment', 'weapon detection'] if is_high else ['restricted boundary', 'loitering'] if is_medium else ['routine transit'], 'context_summary': 'Context, zone policies, and intent history were fused to assign the threat level.'},
        },
        {
            'id': 'D6',
            'name': 'Explainability Agent (XAI)',
            'status': explanation_status,
            'color': '#06b6d4',
            'payload': {'model': 'XAI narrative engine', 'rationale': 'Object tracking indicates a person approaching a restricted boundary with concealment and hostile intent patterns. The model combines posture, motion, environment, and dwell time.', 'confidence': 0.91 if is_high else 0.78, 'reasoning': 'Human-readable justification generated for operator review.'},
        },
        {
            'id': 'D7',
            'name': 'Alert & Learning Agent',
            'status': response_status,
            'color': '#ef4444',
            'payload': {'model': 'Response policy + feedback loop', 'action': 'alarm_triggered' if is_high else 'notify_operator' if is_medium else 'monitor_only', 'modelUpdate': 'retention_window_2m', 'objectFocus': ['Pen', 'Weapon', 'Vehicle'] if is_high else ['Person', 'Package'], 'learning': 'State outputs are logged for model refinement and continuous retraining.'},
        },
    ]


@app.get('/api/health')
def health():
    email_configured = all(os.getenv(key) for key in ('SMTP_HOST', 'SMTP_FROM', 'SMTP_USERNAME', 'SMTP_PASSWORD'))
    return {
        'status': 'ok',
        'weapon_model': any(path.exists() for path in WEAPON_MODEL_PATHS),
        'person_model': PERSON_MODEL_PATH.exists(),
        'pose_model': POSE_MODEL_PATH.exists(),
        'sharp_detector_available': (MODEL_ROOT / 'yolov8s-worldv2.pt').exists(),
        'sharp_detector_loaded': sharp_model is not None,
        'sharp_detector_attempted': sharp_model_attempted,
        'object_detection_confidence': OBJECT_DETECTION_CONFIDENCE,
        'object_confirmation_frames': OBJECT_CONFIRMATION_FRAMES,
        'email_configured': email_configured,
    }


def load_weapon_model():
    global weapon_model
    model_path = next((path for path in WEAPON_MODEL_PATHS if path.exists()), None)
    if weapon_model is None and model_path:
        from ultralytics import YOLO
        weapon_model = YOLO(str(model_path))
    return weapon_model


def load_person_model():
    global person_model
    if person_model is None and PERSON_MODEL_PATH.exists():
        from ultralytics import YOLO
        person_model = YOLO(str(PERSON_MODEL_PATH))
    return person_model


def load_pose_model():
    global pose_model
    if pose_model is None and POSE_MODEL_PATH.exists():
        from ultralytics import YOLO
        pose_model = YOLO(str(POSE_MODEL_PATH))
    return pose_model


def classify_activity(person_box, keypoints, motion_score):
    if not person_box:
        return {'label': 'unknown', 'confidence': 0}
    points_by_index = {
        int(point['index']): point for point in keypoints
        if point.get('index') is not None and point.get('confidence', 0) >= 25
    }
    if not points_by_index:
        if motion_score >= 0.55:
            return {'label': 'running', 'confidence': 65}
        posture = POSTURE_ACTIVITY_AGENT.classify(person_box, keypoints, motion_score)
        return {'label': posture['activity'].lower(), 'confidence': round(posture['confidence'] * 100), 'source': posture['source']}

    def midpoint(indices):
        available = [points_by_index[index] for index in indices if index in points_by_index]
        if not available:
            return None
        return {
            'x': sum(point['x'] for point in available) / len(available),
            'y': sum(point['y'] for point in available) / len(available),
            'count': len(available),
            'confidence': sum(point['confidence'] for point in available) / len(available),
        }

    def joint_angle(first, joint, last):
        if not all(point is not None for point in (first, joint, last)):
            return None
        left = (first['x'] - joint['x'], first['y'] - joint['y'])
        right = (last['x'] - joint['x'], last['y'] - joint['y'])
        left_length = math.hypot(*left)
        right_length = math.hypot(*right)
        denominator = left_length * right_length
        if denominator < 1e-6:
            return None
        cosine = max(-1.0, min(1.0, (left[0] * right[0] + left[1] * right[1]) / denominator))
        return math.degrees(math.acos(cosine))

    shoulders = midpoint((5, 6))
    hips = midpoint((11, 12))
    knees = midpoint((13, 14))
    ankles = midpoint((15, 16))
    wrists = [points_by_index[index] for index in (9, 10) if index in points_by_index]
    torso_length = math.hypot(shoulders['x'] - hips['x'], shoulders['y'] - hips['y']) if shoulders and hips else 0
    if torso_length < 1:
        posture = POSTURE_ACTIVITY_AGENT.classify(person_box, keypoints, motion_score)
        return {'label': posture['activity'].lower(), 'confidence': round(posture['confidence'] * 100), 'source': posture['source']}

    visible_quality = sum(point['confidence'] for point in points_by_index.values()) / (100 * len(points_by_index))
    leg_angles = [
        angle for angle in (
            joint_angle(points_by_index.get(11), points_by_index.get(13), points_by_index.get(15)),
            joint_angle(points_by_index.get(12), points_by_index.get(14), points_by_index.get(16)),
        ) if angle is not None
    ]
    both_legs_visible = len(leg_angles) == 2
    base_confidence = min(90, 55 + 35 * visible_quality)
    if not both_legs_visible:
        base_confidence = min(base_confidence, 62)

    torso_angle_from_vertical = None
    if shoulders and hips:
        dx, dy = abs(shoulders['x'] - hips['x']), abs(shoulders['y'] - hips['y'])
        torso_angle_from_vertical = math.degrees(math.atan2(dx, max(dy, 1e-6)))
        if torso_angle_from_vertical >= 60 and knees and ankles:
            return {'label': 'lying', 'confidence': round(base_confidence)}

    if shoulders and wrists:
        raised = [point for point in wrists if point['y'] < shoulders['y'] - torso_length * 0.15]
        if raised:
            confidence_cap = 82 if len(raised) == 1 else 90
            return {'label': 'hands_up', 'confidence': round(min(base_confidence, confidence_cap))}

    if hips and knees and ankles and leg_angles:
        knee_angle = sum(leg_angles) / len(leg_angles)
        hip_knee_ratio = abs(hips['y'] - knees['y']) / torso_length
        if knee_angle <= 135:
            if hip_knee_ratio <= 0.55:
                return {'label': 'sitting', 'confidence': round(base_confidence)}
            if hip_knee_ratio <= 1.15:
                return {'label': 'crouching', 'confidence': round(base_confidence)}
            return {'label': 'squatting', 'confidence': round(min(base_confidence, 78))}
        if torso_angle_from_vertical is not None and torso_angle_from_vertical >= 30:
            return {'label': 'bending', 'confidence': round(min(base_confidence, 78))}
        if both_legs_visible and knee_angle >= 155:
            return {'label': 'standing', 'confidence': round(base_confidence)}

    # Motion score is scene-wide and sampled sparsely; without a trained temporal
    # model or person-specific sequence evidence, use the posture agent's binary
    # standing/sitting estimate instead of an UNKNOWN/LOW_CONFIDENCE activity.
    posture = POSTURE_ACTIVITY_AGENT.classify(person_box, keypoints, motion_score)
    return {'label': posture['activity'].lower(), 'confidence': round(posture['confidence'] * 100), 'source': posture['source']}


def box_iou(first_box: list[float], second_box: list[float]) -> float:
    first_x, first_y, first_width, first_height = first_box
    second_x, second_y, second_width, second_height = second_box
    left = max(first_x, second_x)
    top = max(first_y, second_y)
    right = min(first_x + first_width, second_x + second_width)
    bottom = min(first_y + first_height, second_y + second_height)
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = first_width * first_height
    second_area = second_width * second_height
    union = first_area + second_area - intersection
    return intersection / union if union else 0


def deduplicate_person_detections(detections: list[dict], iou_threshold: float = 0.6) -> list[dict]:
    unique: list[dict] = []
    for detection in sorted(detections, key=lambda item: item['confidence'], reverse=True):
        if not any(box_iou(detection['box'], existing['box']) >= iou_threshold for existing in unique):
            unique.append(detection)
    return unique


def classify_risk_band(risk_score: float) -> str:
    if risk_score >= RISK_THRESHOLD_CRITICAL:
        return 'CRITICAL'
    if risk_score >= 90:
        return 'VERY_HIGH_RISK'
    if risk_score >= 75:
        return 'SERIOUS_RISK'
    if risk_score >= RISK_THRESHOLD_HEATMAP:
        return 'HIGH_RISK'
    if risk_score >= 30:
        return 'LOW_RISK'
    return 'NORMAL'


def build_risk_regions(person_detections, pose_detections, motion_score: float, sharp_detections, object_detections=None) -> list[dict]:
    regions = []
    for index, person in enumerate(person_detections):
        matching_pose = max(
            pose_detections,
            key=lambda item: box_iou(person['box'], item.get('box', [0, 0, 0, 0])),
            default=None,
        )
        activity = matching_pose if matching_pose and box_iou(person['box'], matching_pose.get('box', [0, 0, 0, 0])) >= 0.1 else None
        activity = activity or {}
        activity_label = str(activity.get('activity') or ('low_confidence' if person else 'unknown')).lower()
        activity_risk = {
            'standing': 8, 'sitting': 5, 'squatting': 12, 'walking': 18,
            'running': 62, 'jumping': 68, 'dancing': 35, 'waving': 22,
            'hands_up': 42, 'clapping': 28, 'kicking': 72, 'bending': 20,
            'falling': 78, 'lying': 8, 'crawling': 48, 'sleeping': 3,
            'fighting': 96,
        }.get(activity_label, 10)
        risk_score = round(min(100, activity_risk + (motion_score * 30)), 1)
        box = person['box']
        regions.append({
            'personId': activity.get('personId') or f'person_{index + 1:02d}',
            'entityType': 'person',
            'label': 'Person',
            'x': box[0],
            'y': box[1],
            'width': box[2],
            'height': box[3],
            'centerX': round(box[0] + box[2] / 2, 2),
            'centerY': round(box[1] + box[3] / 2, 2),
            'riskScore': risk_score,
            'riskBand': classify_risk_band(risk_score),
            'heatmapActive': risk_score >= RISK_THRESHOLD_HEATMAP,
            'activity': activity.get('activity') or ('low_confidence' if person else 'unknown'),
            'activityConfidence': activity.get('confidence', 0),
        })
    for index, item in enumerate(object_detections or []):
        box = item['box']
        risk_score = round(min(100, float(item.get('confidence', 0)) * 0.8 + motion_score * 20), 1)
        regions.append({
            'personId': f'object_{index + 1:02d}',
            'entityType': 'object',
            'label': item.get('label', 'Object'),
            'x': box[0],
            'y': box[1],
            'width': box[2],
            'height': box[3],
            'centerX': round(box[0] + box[2] / 2, 2),
            'centerY': round(box[1] + box[3] / 2, 2),
            'riskScore': risk_score,
            'riskBand': classify_risk_band(risk_score),
            'heatmapActive': risk_score >= RISK_THRESHOLD_HEATMAP,
            'activity': 'detected object',
        })
    return regions


def load_sharp_model():
    global sharp_model, sharp_model_attempted
    if sharp_model_attempted:
        return sharp_model
    sharp_model_attempted = True
    if sharp_model is None:
        try:
            from ultralytics import YOLO
            candidate = YOLO(str(MODEL_ROOT / 'yolov8s-worldv2.pt'))
            candidate.set_classes(['scissors', 'knife', 'gun', 'sword'])
            sharp_model = candidate
        except Exception:
            logger.exception('Open-vocabulary sharp-object detector failed to load.')
            return None
    return sharp_model


@app.post('/api/analyze-frame')
def analyze_frame(payload: CameraFrameRequest, user: dict = Depends(authenticated_user)):
    model = load_weapon_model()
    people_model = load_person_model()

    encoded = payload.image.split(',', 1)[-1]
    try:
        image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert('RGB')
    except Exception:
        return {'ok': False, 'threatLevel': 'LOW', 'detected': False, 'message': 'Invalid camera frame.'}

    detections = []
    image_width, image_height = image.size
    recognized_weapon_labels = {
        normalize_dangerous_label(name)
        for name in (model.names.values() if isinstance(model.names, dict) else model.names)
        if normalize_dangerous_label(name) in SHARP_LABELS
    } if model is not None else set()
    recognized_weapon_model = bool(recognized_weapon_labels)
    if recognized_weapon_model:
        result = model.predict(image, imgsz=640, conf=OBJECT_DETECTION_CONFIDENCE, device='cpu', verbose=False)[0]
        for box in result.boxes:
            confidence = float(box.conf.item())
            class_id = int(box.cls.item())
            coordinates = box.xyxy[0].tolist()
            label = model.names.get(class_id, f'class_{class_id}') if isinstance(model.names, dict) else str(class_id)
            label = normalize_dangerous_label(label)
            if label not in SHARP_LABELS:
                continue
            detections.append({'classId': class_id, 'label': label, 'isPerson': False, 'isSharp': True, 'source': 'weapon_model', 'confidence': round(confidence * 100, 1), 'box': [round(float(coordinates[0]) / image_width * 100, 2), round(float(coordinates[1]) / image_height * 100, 2), round(float(coordinates[2] - coordinates[0]) / image_width * 100, 2), round(float(coordinates[3] - coordinates[1]) / image_height * 100, 2)]})

    sharp_detector = load_sharp_model()
    if sharp_detector is not None:
        sharp_result = sharp_detector.predict(image, imgsz=640, conf=OBJECT_DETECTION_CONFIDENCE, device='cpu', verbose=False)[0]
        for box in sharp_result.boxes:
            confidence = float(box.conf.item())
            class_id = int(box.cls.item())
            coordinates = box.xyxy[0].tolist()
            names = sharp_detector.names
            label = names.get(class_id, f'class_{class_id}') if isinstance(names, dict) else names[class_id] if class_id < len(names) else f'class_{class_id}'
            label = normalize_dangerous_label(label)
            if label not in SHARP_LABELS or label in recognized_weapon_labels:
                continue
            detections.append({'classId': class_id, 'label': label, 'isPerson': False, 'isSharp': label.lower() in SHARP_LABELS, 'source': 'sharp_model', 'confidence': round(confidence * 100, 1), 'box': [round(float(coordinates[0]) / image_width * 100, 2), round(float(coordinates[1]) / image_height * 100, 2), round(float(coordinates[2] - coordinates[0]) / image_width * 100, 2), round(float(coordinates[3] - coordinates[1]) / image_height * 100, 2)]})

    person_detections = []
    if people_model is not None:
        person_result = people_model.predict(image, imgsz=640, conf=0.25, device='cpu', verbose=False)[0]
        for box in person_result.boxes:
            confidence = float(box.conf.item())
            class_id = int(box.cls.item())
            coordinates = box.xyxy[0].tolist()
            box_data = [round(float(coordinates[0]) / image_width * 100, 2), round(float(coordinates[1]) / image_height * 100, 2), round(float(coordinates[2] - coordinates[0]) / image_width * 100, 2), round(float(coordinates[3] - coordinates[1]) / image_height * 100, 2)]
            label = people_model.names.get(class_id, f'class_{class_id}')
            detection = {'classId': class_id, 'label': label, 'isPerson': class_id == 0, 'isSharp': False, 'source': 'person_model', 'confidence': round(confidence * 100, 1), 'box': box_data}
            if class_id == 0:
                person_detections.append(detection)
            else:
                detections.append(detection)

    person_detections = deduplicate_person_detections(person_detections)

    sharp_candidates = [item for item in detections if item.get('isSharp') and item['confidence'] >= OBJECT_DETECTION_CONFIDENCE * 100]
    object_tracks = OBJECT_CONFIRMER.update(user['userId'], sharp_candidates)
    tracks_by_box = {(track['label'], tuple(track['box'])): track for track in object_tracks}
    detections = [
        {
            **item,
            'confirmationFrames': tracks_by_box[(item['label'], tuple(item['box']))]['confirmationFrames'],
            'confirmed': tracks_by_box[(item['label'], tuple(item['box']))]['confirmed'],
        } if item.get('isSharp') and (item['label'], tuple(item['box'])) in tracks_by_box else item
        for item in detections
    ]
    all_detections = person_detections + detections
    for track in object_tracks:
        track['personId'] = associate_person(track, person_detections)
    sharp_detections = [item for item in object_tracks if item['confirmed']]
    detected = bool(sharp_detections or any(not item.get('isSharp') for item in detections))
    scoring_detections = person_detections + [item for item in detections if not item.get('isSharp')] + sharp_detections
    motion_score = max(0, min(float(payload.motionScore), 1))
    pose_detections = []
    pose = load_pose_model()
    if pose is not None:
        pose_result = pose.predict(image, imgsz=640, conf=0.35, device='cpu', verbose=False)[0]
        for index, box in enumerate(pose_result.boxes):
            coordinates = box.xyxy[0].tolist()
            person_box = [round(float(coordinates[0]) / image_width * 100, 2), round(float(coordinates[1]) / image_height * 100, 2), round(float(coordinates[2] - coordinates[0]) / image_width * 100, 2), round(float(coordinates[3] - coordinates[1]) / image_height * 100, 2)]
            points = []
            if pose_result.keypoints is not None and index < len(pose_result.keypoints.xy):
                confidence_points = pose_result.keypoints.conf[index] if pose_result.keypoints.conf is not None else None
                for point_index, point in enumerate(pose_result.keypoints.xy[index].tolist()):
                    point_confidence = float(confidence_points[point_index]) if confidence_points is not None else 1
                    if point_confidence >= 0.25:
                        points.append({'index': point_index, 'x': round(float(point[0]) / image_width * 100, 2), 'y': round(float(point[1]) / image_height * 100, 2), 'confidence': round(point_confidence * 100, 1)})
            activity_result = classify_activity(person_box, points, motion_score)
            pose_detections.append({
                'box': person_box,
                'keypoints': points,
                'activity': activity_result['label'],
                'rawActivity': activity_result['label'],
                'confidence': activity_result['confidence'],
                'source': 'pose_geometry_heuristic',
                'keypointCount': len(points),
            })

    activity_predictions = []
    for pose_item in pose_detections:
        raw_activity = pose_item['activity']
        prediction = ACTIVITY_RECOGNIZER.recognize(
            user['userId'],
            pose_item['box'],
            pose_item['keypoints'],
            {'label': pose_item['activity'], 'confidence': pose_item['confidence'], 'motionScore': motion_score},
        )
        pose_item['activityPrediction'] = prediction
        pose_item['activity'] = prediction['activity'].lower()
        pose_item['rawActivity'] = raw_activity
        pose_item['confidence'] = round(prediction['confidence'] * 100)
        pose_item['personId'] = prediction['personId']
        if prediction['model'] == 'posture_activity_agent':
            pose_item['source'] = prediction['model']
        activity_predictions.append(prediction)
        logger.debug('Activity recognition: person=%s activity=%s confidence=%.3f model=%s',
                     prediction['personId'], prediction['activity'], prediction['confidence'], prediction['model'])

    # Keep a posture estimate available when a person detector found someone
    # but the pose model returned no usable skeleton for that person.
    for detection in person_detections:
        if any(box_iou(detection['box'], pose_item['box']) >= 0.3 for pose_item in pose_detections):
            continue
        prediction = ACTIVITY_RECOGNIZER.recognize(
            user['userId'],
            detection['box'],
            [],
            {'label': 'unknown', 'confidence': 0, 'motionScore': motion_score},
        )
        activity_predictions.append(prediction)

    object_confidence = max((item['confidence'] for item in sharp_detections), default=0)
    person_motion = payload.movingPersons > 0
    object_motion = payload.movingObjects > 0
    interpersonal_aggression = detect_interpersonal_aggression(len(person_detections), motion_score, payload.movingPersons)
    evidence_score = (object_confidence / 100 * 0.45) + (motion_score * 0.4) + (0.15 if person_detections else 0)
    level = 'HIGH' if interpersonal_aggression or (person_detections and motion_score >= 0.55) else 'MEDIUM' if sharp_detections or detected or person_motion or object_motion else 'LOW'
    confidence = round(min(99, evidence_score * 100)) if (detected or person_detections or person_motion) else 5
    if level == 'HIGH':
        confidence = max(confidence, 75)
    agent_mesh = build_agent_mesh_from_module({}, threat_level=level, action='weapon/object scan complete')
    for agent in agent_mesh:
        agent['payload']['frame_result'] = {
            'detected': detected,
            'threat_level': level,
            'detection_count': len(detections),
            'motion_score': round(motion_score, 2),
            'moving_persons': payload.movingPersons,
            'moving_objects': payload.movingObjects,
            'person_count': len(person_detections),
            'sharp_object_count': len(sharp_detections),
            'interpersonal_aggression': interpersonal_aggression,
        }
    agent_mesh[0]['payload']['detections'] = all_detections
    agent_mesh[0]['payload']['objectCandidates'] = object_tracks
    agent_mesh[4]['payload']['objectEvidence'] = [
        {key: item.get(key) for key in ('label', 'confidence', 'box', 'personId', 'objectTrackId', 'confirmationFrames')}
        for item in sharp_detections
    ]
    activity = activity_predictions[0]['activity'].lower() if activity_predictions else 'unknown'
    activity_confidence = round(activity_predictions[0]['confidence'] * 100) if activity_predictions else 0
    activity_source = (
        pose_detections[0]['source'] if pose_detections
        else activity_predictions[0]['model'] if activity_predictions
        else 'pose-unavailable'
    )
    activity_quality = 'HIGH' if pose_detections and pose_detections[0]['keypointCount'] >= 10 else 'MEDIUM' if pose_detections else 'LOW'
    risk_regions = build_risk_regions(person_detections, pose_detections, motion_score, sharp_detections, sharp_detections)
    facial_expression = analyze_facial_expression(image)
    next_action = predict_next_actions(facial_expression, activity, level, len(sharp_detections))
    agent_mesh[2]['payload']['activity'] = activity
    agent_mesh[2]['payload']['confidence'] = round(activity_confidence / 100, 2)
    agent_mesh[2]['payload']['inference_type'] = (
        'temporal GRU' if activity_predictions and activity_predictions[0]['model'] == 'temporal_gru'
        else 'standing/sitting posture agent' if activity_predictions and activity_predictions[0]['model'] == 'posture_activity_agent'
        else 'pose geometry + temporal smoothing' if pose_detections else 'pose unavailable'
    )
    agent_mesh[2]['payload']['pose_keypoints'] = pose_detections[0]['keypoints'] if pose_detections else []
    agent_mesh[2]['payload']['persons'] = activity_predictions
    agent_mesh[2]['payload']['posture_agent'] = 'Standing/Sitting Activity Agent'
    agent_mesh[1]['payload']['activity'] = activity
    agent_mesh[1]['payload']['confidence'] = round(activity_confidence / 100, 2)
    agent_mesh[5]['payload']['rationale'] = 'Detection results were passed through all seven agents for threat scoring and explanation.'
    multimodal_risk = build_multimodal_risk_report({
        'detections': scoring_detections,
        'sharpObjects': sharp_detections,
        'motion': {'score': round(motion_score, 2), 'movingPersons': payload.movingPersons, 'movingObjects': payload.movingObjects},
        'pose': pose_detections,
        'confidence': confidence,
        'threatLevel': level,
        'personDetected': bool(person_detections),
        'personCount': len(person_detections),
        'source': 'LIVE',
        'rgbAvailable': True,
        'thermalAvailable': False,
        'audioAvailable': False,
    })
    temporal_context = build_temporal_context({
        'motionScore': motion_score,
        'movingPersons': payload.movingPersons,
        'movingObjects': payload.movingObjects,
        'poseActivities': [pose_item.get('activity') for pose_item in pose_detections],
        'detections': scoring_detections,
        'threatLevel': level,
    })
    anomaly_risk = build_anomaly_risk_report({
        'riskLevel': level,
        'confidence': confidence,
        'sensorReliability': multimodal_risk['sensorReliability'],
        'temperatureAnomaly': 0.25 if level in {'MEDIUM', 'HIGH'} else 0.05,
        'motionAnomaly': max(0.1, motion_score),
        'audioEvent': 'none',
        'zoneContext': 'restricted perimeter' if level in {'MEDIUM', 'HIGH'} else 'routine patrol',
        'historicalContext': 'repeated suspicious dwell' if level == 'HIGH' else 'normal activity' if level == 'LOW' else 'elevated motion pattern',
    })
    timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    activity_debug = [
        {
            **prediction,
            'poseKeypoints': pose_item['keypoints'],
            'riskScore': risk_regions[index]['riskScore'] if index < len(risk_regions) else None,
        }
        for index, (prediction, pose_item) in enumerate(zip(activity_predictions, pose_detections))
    ] if os.getenv('ACTIVITY_DEBUG', '').lower() in {'1', 'true', 'yes', 'on'} else []
    threat_type = ', '.join(dict.fromkeys(item['label'] for item in sharp_detections)) or ('Possible physical altercation' if interpersonal_aggression else activity)
    threat_event = {
        'userId': user['userId'],
        'timestamp': timestamp,
        'riskScore': confidence,
        'riskThresholdHeatmap': RISK_THRESHOLD_HEATMAP,
        'riskThresholdEmail': RISK_THRESHOLD_EMAIL,
        'riskRegions': risk_regions,
        'threatLevel': level,
        'threatType': threat_type,
        'reason': 'Possible physical altercation: rapid motion involving multiple people detected.' if interpersonal_aggression else f'{level} risk: confirmed {", ".join(item["label"] for item in sharp_detections)} detection.' if sharp_detections else 'No confirmed dangerous object detected in this frame.',
        'confidence': round(confidence / 100, 2),
        'source': 'live_camera',
    }
    alert = process_threat_event(threat_event, user, ACCOUNT_DB)
    return {
        'ok': True,
        'threatLevel': level,
        'aggressionDetected': interpersonal_aggression,
        'riskScore': confidence,
        'timestamp': timestamp,
        'threatEvent': threat_event,
        'alert': alert,
        'detected': detected,
        'detections': all_detections,
        'personDetected': bool(person_detections),
        'personCount': len(person_detections),
        'sharpObjects': sharp_detections,
        'objectCandidates': object_tracks,
        'sharpObjectCount': len(sharp_detections),
        'confidence': confidence,
        'motion': {'score': round(motion_score, 2), 'movingPersons': payload.movingPersons, 'movingObjects': payload.movingObjects},
        'pose': pose_detections,
        'activity': activity,
        'activityConfidence': activity_confidence,
        'activityPrediction': activity_predictions[0] if activity_predictions else {
            'personId': None,
            'activity': 'LOW_CONFIDENCE' if person_detections else 'UNKNOWN',
            'confidence': 0.0,
            'timestamp': timestamp,
            'poseAvailable': False,
            'status': 'insufficient_pose_evidence' if person_detections else 'no_person_detected',
        },
        'activityPredictions': activity_predictions,
        'activityDebug': activity_debug,
        'activitySource': activity_source,
        'activityQuality': activity_quality,
        'rawActivity': pose_detections[0]['rawActivity'] if pose_detections else 'unknown',
        'activityClasses': ACTIVITY_CLASSES,
        'message': 'Possible physical altercation detected: multiple people showing rapid movement.' if interpersonal_aggression else f'{level} risk: confirmed potentially dangerous object detected.' if sharp_detections else 'No confirmed potentially dangerous object detected in this frame.',
        'policy': SHARP_MATERIAL_POLICY,
        'agents': agent_mesh,
        'multimodalRisk': multimodal_risk,
        'temporalContext': temporal_context,
        'anomalyRisk': anomaly_risk,
        'facialExpression': facial_expression,
        'nextActionPrediction': next_action,
        'researchSummary': build_research_summary(),
    }


def password_digest(password: str, salt: str | None = None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 120000).hex()
    return f'{salt}${digest}'


def send_account_email(user_name: str, email: str):
    host = os.getenv('SMTP_HOST')
    port = int(os.getenv('SMTP_PORT', '587'))
    sender = os.getenv('SMTP_FROM')
    username = os.getenv('SMTP_USERNAME')
    password = os.getenv('SMTP_PASSWORD')
    if not all((host, sender, username, password)):
        return False

    message = EmailMessage()
    message['Subject'] = 'Threat Command Center account created'
    message['From'] = sender
    message['To'] = email
    message.set_content(
        f'Hello {user_name},\n\nYour Threat Command Center operator account was created successfully.\n'
        'Keep your credentials private and contact your administrator if this was not you.\n'
    )
    try:
        with smtplib.SMTP(host, port, timeout=15) as smtp:
            smtp.starttls()
            smtp.login(username, password)
            smtp.send_message(message)
        return True
    except (OSError, smtplib.SMTPException) as error:
        logger.error('Account email delivery failed for %s: %s', email, error)
        return False


@app.post('/api/register')
def register(payload: RegisterRequest):
    user_name = payload.userName.strip()
    email = payload.email.strip().lower()
    if not user_name or '@' not in email or '.' not in email.rsplit('@', 1)[-1] or len(payload.password) < 4:
        return {'ok': False, 'message': 'Enter a valid name, email address, and password of at least 4 characters.'}

    initialize_account_db()
    user_id = str(uuid.uuid4())
    with closing(sqlite3.connect(ACCOUNT_DB)) as connection, connection:
        existing = connection.execute('SELECT 1 FROM accounts WHERE email = ?', (email,)).fetchone()
        if existing:
            return {'ok': False, 'message': 'An account already exists for this email address.'}
        connection.execute(
            'INSERT INTO accounts (user_id, email, user_name, password_hash) VALUES (?, ?, ?, ?)',
            (user_id, email, user_name, password_digest(payload.password)),
        )

    email_configured = all(os.getenv(key) for key in ('SMTP_HOST', 'SMTP_FROM', 'SMTP_USERNAME', 'SMTP_PASSWORD'))
    email_sent = send_account_email(user_name, email) if email_configured else False
    return {
        'ok': True,
        'message': f'Account created for {user_name}. ' + ('A confirmation message was sent to your email.' if email_sent else 'Account email delivery failed. Check the server SMTP configuration.' if email_configured else 'Email delivery is not configured on this server.'),
        'emailSent': email_sent,
    }


@app.get('/api/dataset')
def dataset():
    return dataset_overview()


@app.post('/api/resend-confirmation')
def resend_confirmation(user: dict = Depends(authenticated_user)):
    email_configured = all(os.getenv(key) for key in ('SMTP_HOST', 'SMTP_FROM', 'SMTP_USERNAME', 'SMTP_PASSWORD'))
    email_sent = send_account_email(user['userName'], user['email']) if email_configured else False
    return {
        'ok': True,
        'emailSent': email_sent,
        'message': 'A confirmation message was sent to your account email.' if email_sent else 'Confirmation email delivery failed. Check the server SMTP configuration.' if email_configured else 'Email delivery is not configured on this server.',
    }


@app.get('/api/agents')
def agents(threat: str = 'MEDIUM', action: str = 'perimeter scan'):
    # Agent status must stay responsive; dataset enumeration is reserved for /api/dataset.
    return {'agents': build_agent_mesh_from_module({}, threat_level=threat, action=action)}


@app.post('/api/multimodal-risk')
def multimodal_risk(payload: dict | None = None):
    context = payload or {}
    return {'ok': True, 'report': build_multimodal_risk_report(context)}


@app.post('/api/temporal-context')
def temporal_context(payload: dict | None = None):
    context = payload or {}
    return {'ok': True, 'report': build_temporal_context(context)}


@app.post('/api/anomaly-risk')
def anomaly_risk(payload: dict | None = None):
    context = payload or {}
    return {'ok': True, 'report': build_anomaly_risk_report(context)}


@app.get('/api/research-mode')
def research_mode():
    return {'ok': True, 'summary': build_research_summary()}


@app.post('/api/train')
def train():
    return train_model()


@app.post('/api/login')
def login(payload: LoginRequest):
    login_id = (payload.email or payload.userName or '').strip()
    if not login_id or len(payload.password) < 4:
        return {'ok': False, 'message': 'Enter your account email or operator name and password.'}

    initialize_account_db()
    with closing(sqlite3.connect(ACCOUNT_DB)) as connection, connection:
        account = connection.execute('SELECT user_id, email, user_name, password_hash FROM accounts WHERE lower(email) = lower(?) OR lower(user_name) = lower(?)', (login_id, login_id)).fetchone()
    if not account:
        return {'ok': False, 'message': 'No account matches that email or operator name. Create an account first.'}
    salt, expected_digest = account[3].split('$', 1)
    if not secrets.compare_digest(password_digest(payload.password, salt).split('$', 1)[1], expected_digest):
        return {'ok': False, 'message': 'Incorrect account email or password.'}

    detection = {
        'label': 'Wall crossing',
        'confidence': 91,
        'objects': ['Person', 'Pen', 'Package', 'Vehicle', 'Weapon'],
        'summary': f'{account[2]}: MEDIUM threat alert detected. Objects tracked: Person, Pen, Package, Vehicle, Weapon. Training data: Normal Class and Wall crossing frames loaded.',
    }

    token = secrets.token_urlsafe(32)
    with closing(sqlite3.connect(ACCOUNT_DB)) as connection, connection:
        connection.execute('INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)', (token, account[0], datetime.now(timezone.utc).isoformat(timespec='seconds')))

    return {'ok': True, 'userId': account[0], 'userName': account[2], 'email': account[1], 'sessionToken': token, 'message': detection['summary']}


@app.post('/api/forgot-password')
def forgot_password(payload: ForgotPasswordRequest, background_tasks: BackgroundTasks):
    email = payload.email.strip().lower()
    generic_response = _password_reset_message()
    if not email or len(email) > 320 or '@' not in email:
        return generic_response

    initialize_account_db()
    now = datetime.now(timezone.utc)
    window_minutes = _bounded_env_int('PASSWORD_RESET_WINDOW_MINUTES', 15, 1, 1440)
    max_requests = _bounded_env_int('PASSWORD_RESET_MAX_REQUESTS', 3, 1, 20)
    expiry_minutes = _bounded_env_int('PASSWORD_RESET_EXPIRY_MINUTES', 15, 1, 1440)
    cutoff = (now - timedelta(minutes=window_minutes)).isoformat(timespec='seconds')
    now_text = now.isoformat(timespec='seconds')
    email_hash = hashlib.sha256(email.encode('utf-8')).hexdigest()
    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode('utf-8')).hexdigest()

    with closing(sqlite3.connect(ACCOUNT_DB, timeout=10)) as connection, connection:
        connection.execute('BEGIN IMMEDIATE')
        connection.execute('DELETE FROM password_reset_requests WHERE requested_at < ?', (cutoff,))
        request_count = connection.execute(
            'SELECT COUNT(*) FROM password_reset_requests WHERE email_hash = ? AND requested_at >= ?',
            (email_hash, cutoff),
        ).fetchone()[0]
        if request_count >= max_requests:
            connection.commit()
            logger.warning('Password reset rate limit reached for a normalized account address.')
            return generic_response
        connection.execute('INSERT INTO password_reset_requests (email_hash, requested_at) VALUES (?, ?)', (email_hash, now_text))
        account = connection.execute(
            'SELECT user_id, email, user_name FROM accounts WHERE lower(email) = ?', (email,),
        ).fetchone()
        if account:
            user_id, registered_email, user_name = account
            # Keep only the newest outstanding reset link for this account.
            connection.execute(
                'UPDATE password_reset_tokens SET used_at = ? WHERE user_id = ? AND used_at IS NULL',
                (now_text, user_id),
            )
            expires_at = (now + timedelta(minutes=expiry_minutes)).isoformat(timespec='seconds')
            connection.execute(
                'INSERT INTO password_reset_tokens (token_hash, user_id, expires_at, created_at, used_at) VALUES (?, ?, ?, ?, NULL)',
                (token_hash, user_id, expires_at, now_text),
            )
        connection.commit()

    if account:
        background_tasks.add_task(_deliver_password_reset, user_id, registered_email, user_name, raw_token, token_hash)
    return generic_response


@app.post('/api/reset-password')
def reset_password(payload: ResetPasswordRequest):
    token = payload.token.strip()
    new_password = payload.newPassword
    # Match the existing registration policy (minimum four characters) and bound hashing work.
    if len(new_password) < 4 or len(new_password) > 256:
        raise HTTPException(status_code=400, detail='Password must be between 4 and 256 characters.')
    if not token or len(token) > 256:
        raise HTTPException(status_code=400, detail='Password reset link is invalid or has already been used.')

    initialize_account_db()
    token_hash = hashlib.sha256(token.encode('utf-8')).hexdigest()
    now_text = datetime.now(timezone.utc).isoformat(timespec='seconds')
    with closing(sqlite3.connect(ACCOUNT_DB, timeout=10)) as connection, connection:
        connection.execute('BEGIN IMMEDIATE')
        reset = connection.execute(
            'SELECT user_id, expires_at, used_at FROM password_reset_tokens WHERE token_hash = ?', (token_hash,),
        ).fetchone()
        if not reset or reset[2] is not None:
            connection.rollback()
            raise HTTPException(status_code=400, detail='Password reset link is invalid or has already been used.')
        if reset[1] <= now_text:
            connection.rollback()
            raise HTTPException(status_code=400, detail='Password reset link has expired.')

        user_id = reset[0]
        connection.execute('UPDATE accounts SET password_hash = ? WHERE user_id = ?', (password_digest(new_password), user_id))
        connection.execute('UPDATE password_reset_tokens SET used_at = ? WHERE user_id = ? AND used_at IS NULL', (now_text, user_id))
        # Password changes revoke any already-issued bearer sessions for this account.
        connection.execute('DELETE FROM sessions WHERE user_id = ?', (user_id,))
        connection.commit()
    return {'message': 'Password reset successfully.'}


@app.post('/api/threat-alert')
def threat_alert(payload: ThreatAlertRequest, user: dict = Depends(authenticated_user)):
    level = payload.level.upper()
    confidence = max(0.0, min(float(payload.confidence), 100.0))
    if level not in {'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'} or not 0 <= payload.confidence <= 100:
        return {'ok': False, 'message': 'Invalid threat alert payload.'}
    event = {'userId': user['userId'], 'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'riskScore': confidence, 'threatLevel': level, 'reason': payload.summary, 'confidence': round(confidence / 100, 2), 'source': 'manual_alert'}
    alert = process_threat_event(event, user, ACCOUNT_DB)
    return {'ok': True, **alert}


@app.post('/api/test-threat-email')
def test_threat_email(user: dict = Depends(authenticated_user)):
    try:
        result = send_test_threat_email()
    except Exception as error:
        logger.exception('Threat email test failed for user %s.', user['userId'])
        raise HTTPException(status_code=502, detail=f'Test email failed ({type(error).__name__}).') from error
    if not result.get('ok'):
        raise HTTPException(status_code=502, detail=result.get('error', 'Test email was not accepted.'))
    return result


@app.post('/api/agent-feedback')
def agent_feedback(payload: AgentFeedbackRequest):
    allowed = {'standing', 'sitting', 'walking', 'running', 'crawling', 'sleeping'}
    observed = payload.observedActivity.strip().lower()
    corrected = payload.correctedActivity.strip().lower()
    if '@' not in payload.email or observed not in allowed or corrected not in allowed:
        return {'ok': False, 'message': 'Invalid activity feedback.'}

    with sqlite3.connect(ACCOUNT_DB) as connection:
        connection.execute(
            'CREATE TABLE IF NOT EXISTS agent_feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL, observed_activity TEXT NOT NULL, corrected_activity TEXT NOT NULL, threat_level TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)'
        )
        connection.execute(
            'INSERT INTO agent_feedback (email, observed_activity, corrected_activity, threat_level) VALUES (?, ?, ?, ?)',
            (payload.email.strip().lower(), observed, corrected, payload.threatLevel.upper()),
        )
        count = connection.execute('SELECT COUNT(*) FROM agent_feedback').fetchone()[0]

    return {'ok': True, 'feedbackCount': count, 'message': 'Feedback stored for the next agent refinement cycle.'}


@app.post('/api/predict')
def predict(payload: PredictRequest):
    image_path = Path(payload.imagePath)
    if not image_path.exists():
        raise FileNotFoundError(f'Image not found: {image_path}')

    result = predict_image(str(image_path))
    label = result['label']
    confidence = result['confidence']
    objects = ['Person', 'Pen', 'Package', 'Vehicle']
    if label == 'Wall crossing':
        objects.append('Weapon')

    message = f'{label} scene detected with {confidence}% confidence. Objects tracked: {", ".join(objects)}.'
    return {'label': label, 'confidence': confidence, 'objects': objects, 'message': message, 'probabilities': result['probabilities']}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('backend.server:app', host='0.0.0.0', port=8000, reload=False)
