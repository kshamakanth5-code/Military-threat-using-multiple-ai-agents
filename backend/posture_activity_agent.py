"""Fallback agent that always resolves uncertain posture to standing or sitting."""
from __future__ import annotations

import math


class PostureActivityAgent:
    """Infer a coarse standing/sitting label from COCO-17 pose geometry.

    The existing YOLO pose checkpoint supplies keypoints. This agent is a
    deterministic posture fallback; it does not require or overwrite a new
    activity-training checkpoint.
    """

    name = "Standing/Sitting Activity Agent"

    @staticmethod
    def _angle(first: dict, joint: dict, last: dict) -> float | None:
        left = (first['x'] - joint['x'], first['y'] - joint['y'])
        right = (last['x'] - joint['x'], last['y'] - joint['y'])
        denominator = math.hypot(*left) * math.hypot(*right)
        if denominator < 1e-6:
            return None
        cosine = max(-1.0, min(1.0, (left[0] * right[0] + left[1] * right[1]) / denominator))
        return math.degrees(math.acos(cosine))

    def classify(self, person_box: list[float], keypoints: list[dict], motion_score: float = 0.0) -> dict:
        points = {
            int(point['index']): {'x': float(point['x']), 'y': float(point['y']), 'confidence': float(point.get('confidence', 0))}
            for point in keypoints
            if point.get('index') is not None and float(point.get('confidence', 0)) >= 25
        }
        shoulders = [points[index] for index in (5, 6) if index in points]
        hips = [points[index] for index in (11, 12) if index in points]

        if hips:
            hip_y = sum(point['y'] for point in hips) / len(hips)
            torso = 0.0
            if shoulders:
                shoulder_y = sum(point['y'] for point in shoulders) / len(shoulders)
                torso = abs(hip_y - shoulder_y)
            knees = [points[index] for index in (13, 14) if index in points]
            ankles = [points[index] for index in (15, 16) if index in points]
            angles = [
                angle for angle in (
                    self._angle(points[hip_index], points[knee_index], points[ankle_index])
                    for hip_index, knee_index, ankle_index in ((11, 13, 15), (12, 14, 16))
                    if hip_index in points and knee_index in points and ankle_index in points
                ) if angle is not None
            ]
            if angles:
                mean_angle = sum(angles) / len(angles)
                knee_y = sum(point['y'] for point in knees) / len(knees)
                hip_knee_ratio = abs(hip_y - knee_y) / max(torso, 1.0)
                if mean_angle <= 138 and hip_knee_ratio <= 0.8:
                    return {
                        'activity': 'SITTING',
                        'confidence': round(min(0.92, 0.65 + (138 - mean_angle) / 250 + (0.8 - hip_knee_ratio) / 4), 3),
                        'source': 'pose_joint_geometry',
                    }
                if mean_angle >= 153:
                    return {
                        'activity': 'STANDING',
                        'confidence': round(min(0.92, 0.62 + (mean_angle - 153) / 100), 3),
                        'source': 'pose_joint_geometry',
                    }
                if hip_knee_ratio <= 0.4:
                    return {'activity': 'SITTING', 'confidence': 0.62, 'source': 'pose_hip_knee_geometry'}
                return {'activity': 'STANDING', 'confidence': 0.58, 'source': 'pose_joint_geometry'}

        # If lower-body landmarks are missing, use the person crop's shape as a
        # weak tie-breaker. Movement generally implies an upright posture.
        width = float(person_box[2]) if len(person_box) > 2 else 0.0
        height = float(person_box[3]) if len(person_box) > 3 else 0.0
        aspect = width / height if height > 1e-6 else 0.0
        if motion_score >= 0.25 or aspect < 0.48:
            activity = 'STANDING'
        else:
            activity = 'SITTING'
        has_box = width > 0 and height > 0
        return {
            'activity': activity,
            'confidence': 0.42 if has_box else 0.30,
            'source': 'person_box_posture_estimate' if has_box else 'default_upright_estimate',
        }
