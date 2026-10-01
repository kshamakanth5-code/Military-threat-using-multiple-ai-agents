from __future__ import annotations

from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from functools import wraps
import os
from pathlib import Path
from threading import RLock
import time

import numpy as np
import torch
from torch import nn
from backend.posture_activity_agent import PostureActivityAgent


ACTIVITIES = (
    'STANDING', 'SITTING', 'WALKING', 'RUNNING', 'JOGGING', 'CRAWLING',
    'FALLING', 'LYING', 'BENDING', 'SQUATTING', 'KNEELING', 'JUMPING',
    'WAVING', 'RAISING_HANDS', 'TURNING', 'STOPPING', 'STARTING_TO_WALK',
    'STARTING_TO_RUN', 'CLIMBING', 'CROUCHING', 'SLEEPING', 'DANCING',
    'CLAPPING', 'KICKING', 'FIGHTING', 'HOLDING_OBJECT', 'USING_OBJECT',
    'ABNORMAL_MOVEMENT',
)
ACTIVITY_INDEX = {label: index for index, label in enumerate(ACTIVITIES)}
POSE_KEYPOINTS = 17
FEATURE_SIZE = POSE_KEYPOINTS * 3 + 4
SEQUENCE_LENGTH = 24
PREDICTION_WINDOW = 5
ACTIVITY_REQUIRED_SECONDS = 3.0
MIN_ACTIVITY_CONFIDENCE = max(0.0, min(float(os.getenv('ACTIVITY_MIN_CONFIDENCE', '0.50')), 1.0))
TEMPORAL_ONLY_ACTIVITIES = {
    'WALKING', 'RUNNING', 'JOGGING', 'CRAWLING', 'FALLING', 'JUMPING',
    'WAVING', 'TURNING', 'STOPPING', 'STARTING_TO_WALK', 'STARTING_TO_RUN',
    'CLIMBING', 'DANCING', 'CLAPPING', 'KICKING', 'FIGHTING',
    'HOLDING_OBJECT', 'USING_OBJECT', 'ABNORMAL_MOVEMENT',
}
MODEL_PATH = Path(__file__).resolve().parent / 'activity_model.pth'


def _serialized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapped


class TemporalActivityModel(nn.Module):
    """Small GRU for per-person pose feature sequences."""

    def __init__(self, input_size: int = FEATURE_SIZE, hidden_size: int = 64, classes: int = len(ACTIVITIES)):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden_size, classes))

    def forward(self, sequence):
        encoded, _ = self.gru(sequence)
        return self.head(encoded[:, -1])


def pose_feature_vector(person_box: list[float], keypoints: list[dict]) -> np.ndarray:
    """Encode COCO-17 points relative to the person box plus box geometry."""
    x, y, width, height = (float(value) / 100 for value in person_box)
    width = max(width, 1e-4)
    height = max(height, 1e-4)
    points = np.zeros((POSE_KEYPOINTS, 3), dtype=np.float32)
    for point in keypoints:
        index = point.get('index')
        if index is None or not 0 <= int(index) < POSE_KEYPOINTS:
            continue
        points[int(index)] = (
            (float(point.get('x', 0)) / 100 - x) / width,
            (float(point.get('y', 0)) / 100 - y) / height,
            float(point.get('confidence', 0)) / 100,
        )
    geometry = np.asarray([x + width / 2, y + height / 2, width, height], dtype=np.float32)
    return np.concatenate((points.flatten(), geometry))


def _load_model(path: Path = MODEL_PATH):
    if not path.is_file():
        return None
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location='cpu')
    activities = checkpoint.get('activities') if isinstance(checkpoint, dict) else None
    state_dict = checkpoint.get('state_dict') if isinstance(checkpoint, dict) else checkpoint
    if activities != list(ACTIVITIES) or not isinstance(state_dict, dict):
        return None
    model = TemporalActivityModel(classes=len(activities))
    model.load_state_dict(state_dict)
    model.eval()
    return model


class TemporalActivityRecognizer:
    """Track-level pose windows, model inference, and short-window label smoothing."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self._lock = RLock()
        self.model_path = Path(model_path)
        self.model = _load_model(self.model_path)
        self.posture_agent = PostureActivityAgent()
        self._features = defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH))
        self._predictions = defaultdict(lambda: deque(maxlen=PREDICTION_WINDOW))
        self._tracks = defaultdict(dict)
        self._next_track_id = defaultdict(int)

    def _track_id(self, scope: str, box: list[float]) -> str:
        tracks = self._tracks[scope]
        center_x = float(box[0]) + float(box[2]) / 2
        center_y = float(box[1]) + float(box[3]) / 2
        best_id, best_distance = None, float('inf')
        for track_id, previous in tracks.items():
            if previous['age'] > 8:
                continue
            distance = ((center_x - previous['x']) ** 2 + (center_y - previous['y']) ** 2) ** 0.5
            # Camera frames arrive every two seconds, so a moving person can
            # shift farther than a tight frame-to-frame IoU tracker allows.
            gate = min(45.0, max(15.0, float(box[3]) * 2.2))
            if distance <= gate and distance < best_distance:
                best_id, best_distance = track_id, distance
        if best_id is None:
            self._next_track_id[scope] += 1
            best_id = f'person_{self._next_track_id[scope]:02d}'
        for state in tracks.values():
            state['age'] += 1
        state = tracks.setdefault(best_id, {
            'previousActivity': 'LOW_CONFIDENCE',
            'currentActivity': 'LOW_CONFIDENCE',
            'centerTrajectory': deque(maxlen=12),
            'movementHistory': deque(maxlen=12),
            'observationTrajectory': deque(maxlen=8),
            'observationCount': 0,
            'observationStartedAt': None,
        })
        previous_center = (state.get('x', center_x), state.get('y', center_y))
        state['lastDisplacement'] = float(np.hypot(center_x - previous_center[0], center_y - previous_center[1]))
        state.update({'x': center_x, 'y': center_y, 'age': 0})
        for stale_id in [key for key, value in tracks.items() if value['age'] > 30]:
            del tracks[stale_id]
            self._features.pop((scope, stale_id), None)
            self._predictions.pop((scope, stale_id), None)
        return best_id

    def _model_prediction(self, key) -> tuple[str, float] | None:
        sequence = self._features[key]
        if self.model is None or len(sequence) < 8:
            return None
        values = np.stack(sequence).astype(np.float32)
        if len(values) < SEQUENCE_LENGTH:
            values = np.concatenate((np.repeat(values[:1], SEQUENCE_LENGTH - len(values), axis=0), values))
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(torch.from_numpy(values[-SEQUENCE_LENGTH:][None, ...])), dim=1)[0]
        index = int(torch.argmax(probabilities).item())
        return ACTIVITIES[index], float(probabilities[index].item())

    @staticmethod
    def _trajectory_motion(state: dict) -> tuple[str, float, float] | None:
        """Estimate sustained person movement from the tracked center over time.

        Speed is normalized by the person's visible height so the thresholds do
        not depend directly on camera resolution or distance from the camera.
        Require a sustained, mostly consistent trajectory to avoid labeling box
        jitter as movement.
        """
        observations = list(state.get('observationTrajectory', ()))
        if len(observations) < 3:
            return None
        observations = observations[-6:]
        elapsed = observations[-1][0] - observations[0][0]
        if elapsed < ACTIVITY_REQUIRED_SECONDS:
            return None
        height = sum(item[3] for item in observations) / len(observations)
        if height < 1:
            return None
        path_length = sum(
            float(np.hypot(current[1] - previous[1], current[2] - previous[2]))
            for previous, current in zip(observations, observations[1:])
        )
        net_displacement = float(np.hypot(
            observations[-1][1] - observations[0][1],
            observations[-1][2] - observations[0][2],
        ))
        speed = net_displacement / height / elapsed
        straightness = net_displacement / max(path_length, 1e-6)
        if straightness < 0.55 or net_displacement / height < 0.45:
            return None
        if speed >= 0.2:
            return 'MOVING', min(0.82, 0.55 + speed * 0.3), speed
        return None

    @_serialized
    def recognize(self, scope: str, person_box: list[float], keypoints: list[dict], fallback: dict, timestamp: str | None = None) -> dict:
        person_id = self._track_id(scope, person_box)
        key = (scope, person_id)
        state = self._tracks[scope][person_id]
        observed_at = time.monotonic()
        trajectory = state['observationTrajectory']
        if trajectory and observed_at - trajectory[-1][0] > 4.0:
            trajectory.clear()
            state['observationCount'] = 0
            state['observationStartedAt'] = None
            self._predictions[key].clear()
        if state.get('observationStartedAt') is None:
            state['observationStartedAt'] = observed_at
        center_x = float(person_box[0]) + float(person_box[2]) / 2
        center_y = float(person_box[1]) + float(person_box[3]) / 2
        trajectory.append((
            observed_at,
            center_x,
            center_y,
            max(float(person_box[3]), 1.0),
        ))
        state['observationCount'] += 1
        orientation = self._body_orientation(keypoints)
        self._features[key].append(pose_feature_vector(person_box, keypoints))
        prediction = self._model_prediction(key)
        if prediction is None:
            label = str(fallback.get('label', 'UNKNOWN')).upper().replace(' ', '_')
            label = {'LYING_DOWN': 'LYING', 'HANDS_UP': 'RAISING_HANDS'}.get(label, label)
            confidence = max(0.0, min(float(fallback.get('confidence', 0)) / 100, 1.0))
            if label in TEMPORAL_ONLY_ACTIVITIES:
                label, confidence = 'UNKNOWN', 0.0
        else:
            label, confidence = prediction

        if label not in ACTIVITY_INDEX or confidence < MIN_ACTIVITY_CONFIDENCE:
            label = 'UNKNOWN'
        history = self._predictions[key]
        history.append((label, confidence))
        counts = Counter(item[0] for item in history)
        selected = max(counts, key=lambda name: (counts[name], sum(score for act, score in history if act == name)))
        selected_confidence = sum(score for act, score in history if act == selected) / max(counts[selected], 1)
        if selected == 'UNKNOWN' or selected_confidence < MIN_ACTIVITY_CONFIDENCE:
            selected = 'UNKNOWN'
        posture_fallback = selected == 'UNKNOWN'
        if posture_fallback:
            posture = self.posture_agent.classify(
                person_box,
                keypoints,
                float(fallback.get('motionScore', 0.0) or 0.0),
            )
            selected = posture['activity']
            selected_confidence = posture['confidence']
        trajectory_motion = self._trajectory_motion(state) if prediction is None else None
        if trajectory_motion is not None:
            selected, selected_confidence, movement_speed = trajectory_motion
            posture_fallback = False
        else:
            movement_speed = 0.0
        activity_label = selected
        stamp = timestamp or datetime.now(timezone.utc).isoformat(timespec='milliseconds')
        previous = state.get('currentActivity', activity_label)
        state['previousActivity'] = activity_label if previous in {'LOW_CONFIDENCE', 'UNKNOWN'} else previous
        state['currentActivity'] = activity_label
        state['activityConfidence'] = round(selected_confidence, 3)
        state['poseLandmarks'] = list(keypoints)
        state['bodyOrientation'] = orientation
        state['centerTrajectory'].append([round(state['x'], 2), round(state['y'], 2)])
        state['movementHistory'].append({
            'displacement': round(state.get('lastDisplacement', 0.0), 3),
            'motionScore': round(float(fallback.get('motionScore', 0.0) or 0.0), 3),
            'activity': activity_label,
        })
        elapsed_seconds = max(0.0, observed_at - state['observationStartedAt'])
        activity_ready = elapsed_seconds >= ACTIVITY_REQUIRED_SECONDS
        result = {
            'personId': person_id,
            'activity': activity_label,
            'previousActivity': state['previousActivity'],
            'confidence': round(selected_confidence, 3),
            'timestamp': stamp,
            'poseAvailable': bool(keypoints),
            'status': 'classified' if activity_ready else 'observing',
            'observationCount': state['observationCount'],
            'observationElapsedSeconds': round(elapsed_seconds, 2),
            'requiredSeconds': ACTIVITY_REQUIRED_SECONDS,
            'poseLandmarks': list(keypoints),
            'movementHistory': list(state['movementHistory']),
            'bodyOrientation': orientation,
            'centerTrajectory': list(state['centerTrajectory']),
            'model': 'temporal_motion_heuristic' if trajectory_motion is not None else 'posture_activity_agent' if posture_fallback else 'temporal_gru' if self.model is not None and len(self._features[key]) >= 8 else 'temporal_pose_fallback',
            'activityAgent': self.posture_agent.name if posture_fallback else None,
            'movementSpeedBodyLengthsPerSecond': round(movement_speed, 3),
        }
        return result

    @staticmethod
    def _body_orientation(keypoints: list[dict]) -> str:
        points = {
            int(point['index']): point for point in keypoints
            if point.get('index') is not None and point.get('confidence', 0) >= 25
        }
        shoulders = [points[index] for index in (5, 6) if index in points]
        hips = [points[index] for index in (11, 12) if index in points]
        if not shoulders or not hips:
            return 'unknown'
        sx = sum(point['x'] for point in shoulders) / len(shoulders)
        sy = sum(point['y'] for point in shoulders) / len(shoulders)
        hx = sum(point['x'] for point in hips) / len(hips)
        hy = sum(point['y'] for point in hips) / len(hips)
        dx, dy = sx - hx, sy - hy
        length = float(np.hypot(dx, dy))
        if length < 1:
            return 'unknown'
        horizontal_ratio = abs(dx) / length
        if horizontal_ratio >= 0.82:
            return 'horizontal'
        if horizontal_ratio >= 0.42:
            return 'leaning'
        return 'upright'

    @_serialized
    def reset(self, scope: str | None = None):
        if scope is None:
            self._features.clear()
            self._predictions.clear()
            self._tracks.clear()
            self._next_track_id.clear()
        else:
            for key in [key for key in self._features if key[0] == scope]:
                self._features.pop(key, None)
                self._predictions.pop(key, None)
            self._tracks.pop(scope, None)
            self._next_track_id.pop(scope, None)
