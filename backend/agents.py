from __future__ import annotations


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
