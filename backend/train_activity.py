from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from backend.activity_recognition import (
    ACTIVITIES,
    ACTIVITY_INDEX,
    FEATURE_SIZE,
    MODEL_PATH,
    SEQUENCE_LENGTH,
    TemporalActivityModel,
    pose_feature_vector,
)


BASE_DIR = Path(__file__).resolve().parent.parent
ACTIVITY_DATASET_DIR = BASE_DIR / 'activity_dataset'
METRICS_PATH = BASE_DIR / 'backend' / 'activity_model_metrics.json'
POSE_MODEL_PATH = BASE_DIR / 'yolo11n-pose.pt'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MIN_SUBJECTS_PER_ACTIVITY = 3
MAX_EPOCHS = 60
PATIENCE = 8
SEED = 42


def discover_sequences(root: Path = ACTIVITY_DATASET_DIR) -> list[dict]:
    """Expected layout: <ACTIVITY>/<subject-id>/<clip-folder-or-video>/."""
    rows = []
    for label in ACTIVITIES:
        class_dir = root / label
        if not class_dir.is_dir():
            continue
        for subject_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            for source in sorted(subject_dir.iterdir()):
                if source.is_dir() and any(item.suffix.lower() in IMAGE_EXTENSIONS for item in source.rglob('*')):
                    rows.append({'activity': label, 'subject': subject_dir.name, 'source': source})
                elif source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS | {'.mp4', '.avi', '.mov', '.mkv', '.webm'}:
                    rows.append({'activity': label, 'subject': subject_dir.name, 'source': source})
    return rows


def dataset_report(rows: list[dict]) -> dict:
    by_activity = Counter(row['activity'] for row in rows)
    by_subject = defaultdict(set)
    for row in rows:
        by_subject[row['activity']].add(row['subject'])
    return {
        'root': str(ACTIVITY_DATASET_DIR),
        'sequences': len(rows),
        'activities': {
            activity: {'sequences': by_activity[activity], 'subjects': len(by_subject[activity])}
            for activity in ACTIVITIES
        },
        'missingActivities': [activity for activity in ACTIVITIES if by_activity[activity] == 0],
    }


def _split_by_subject(rows: list[dict]):
    subjects = sorted({row['subject'] for row in rows})
    random.Random(SEED).shuffle(subjects)
    if len(subjects) < 3:
        raise ValueError('At least three distinct subject IDs are required for train/validation/test splitting.')
    train_end = max(1, int(len(subjects) * 0.70))
    validation_size = max(1, int(len(subjects) * 0.15))
    if train_end + validation_size >= len(subjects):
        train_end = len(subjects) - 2
    val_end = train_end + validation_size
    partitions = {
        'train': set(subjects[:train_end]),
        'validation': set(subjects[train_end:val_end]),
        'test': set(subjects[val_end:]),
    }
    split_rows = {name: [row for row in rows if row['subject'] in ids] for name, ids in partitions.items()}
    missing = {
        name: [activity for activity in ACTIVITIES if not any(row['activity'] == activity for row in split_rows[name])]
        for name in split_rows
    }
    invalid = {name: labels for name, labels in missing.items() if labels}
    if invalid:
        details = '; '.join(f'{name} missing {", ".join(labels)}' for name, labels in invalid.items())
        raise ValueError(f'Subject-separated split lacks activity coverage ({details}). Collect more labeled subjects.')
    return split_rows


def _source_frames(source: Path, maximum: int = SEQUENCE_LENGTH):
    if source.is_dir():
        paths = sorted(path for path in source.rglob('*') if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            return []
        indices = np.linspace(0, len(paths) - 1, min(maximum, len(paths))).astype(int)
        return [cv2.imread(str(paths[index])) for index in indices]

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f'Could not open video: {source}')
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(np.ceil(frame_count / maximum))) if frame_count else 1
    frames, frame_index = [], 0
    while len(frames) < maximum:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % stride == 0:
            frames.append(frame)
        frame_index += 1
    capture.release()
    return frames


def _pose_sequence(row: dict, pose_model) -> np.ndarray:
    features = []
    for frame in _source_frames(row['source']):
        if frame is None:
            continue
        height, width = frame.shape[:2]
        result = pose_model.predict(frame, imgsz=640, conf=0.25, device='cpu', verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
            continue
        best = int(torch.argmax(result.boxes.conf).item())
        x1, y1, x2, y2 = result.boxes.xyxy[best].tolist()
        box = [x1 / width * 100, y1 / height * 100, (x2 - x1) / width * 100, (y2 - y1) / height * 100]
        points = []
        xy = result.keypoints.xy[best].tolist()
        conf = result.keypoints.conf[best].tolist() if result.keypoints.conf is not None else [1.0] * len(xy)
        for index, (point, score) in enumerate(zip(xy, conf)):
            points.append({'index': index, 'x': point[0] / width * 100, 'y': point[1] / height * 100, 'confidence': score * 100})
        features.append(pose_feature_vector(box, points))
    if not features:
        raise ValueError(f'No person poses found in labeled sequence: {row["source"]}')
    values = np.stack(features).astype(np.float32)
    sample_at = np.linspace(0, len(values) - 1, SEQUENCE_LENGTH).astype(int)
    return values[sample_at]


def _augment(sequence: torch.Tensor) -> torch.Tensor:
    sequence = sequence.clone()
    if random.random() < 0.5:
        points = sequence[:, :51].reshape(SEQUENCE_LENGTH, 17, 3)
        points[:, :, 0] = 1.0 - points[:, :, 0]
        sequence[:, -4] = 1.0 - sequence[:, -4]
    points = sequence[:, :51].reshape(SEQUENCE_LENGTH, 17, 3)
    if random.random() < 0.5:
        points[:, :, :2] = ((points[:, :, :2] - 0.5) * random.uniform(0.92, 1.08) + 0.5).clamp(0, 1)
    if random.random() < 0.35:
        start = random.randrange(0, SEQUENCE_LENGTH // 4 + 1)
        end = random.randrange(SEQUENCE_LENGTH - SEQUENCE_LENGTH // 4, SEQUENCE_LENGTH + 1)
        cropped = sequence[start:end]
        indexes = torch.linspace(0, len(cropped) - 1, SEQUENCE_LENGTH).long()
        sequence = cropped[indexes]
    return sequence


def _metrics(labels: list[int], predictions: list[int]):
    confusion = [[0 for _ in ACTIVITIES] for _ in ACTIVITIES]
    for actual, predicted in zip(labels, predictions):
        confusion[actual][predicted] += 1
    per_class = {}
    for index, name in enumerate(ACTIVITIES):
        tp = confusion[index][index]
        fp = sum(confusion[row][index] for row in range(len(ACTIVITIES)) if row != index)
        fn = sum(confusion[index][col] for col in range(len(ACTIVITIES)) if col != index)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': sum(confusion[index])}
    count = max(len(labels), 1)
    return {
        'accuracy': sum(actual == predicted for actual, predicted in zip(labels, predictions)) / count,
        'macroPrecision': sum(row['precision'] for row in per_class.values()) / len(ACTIVITIES),
        'macroRecall': sum(row['recall'] for row in per_class.values()) / len(ACTIVITIES),
        'macroF1': sum(row['f1'] for row in per_class.values()) / len(ACTIVITIES),
        'perClass': per_class,
        'confusionMatrix': confusion,
    }


def train_activity_model():
    rows = discover_sequences()
    report = dataset_report(rows)
    if report['missingActivities']:
        missing = ', '.join(report['missingActivities'])
        raise FileNotFoundError(
            f'No complete activity-labeled sequence dataset found at {ACTIVITY_DATASET_DIR}. '
            f'Missing labeled classes: {missing}. No existing dataset was modified.'
        )
    for activity in ACTIVITIES:
        subjects = {row['subject'] for row in rows if row['activity'] == activity}
        if len(subjects) < MIN_SUBJECTS_PER_ACTIVITY:
            raise ValueError(f'{activity} has {len(subjects)} subjects; need at least {MIN_SUBJECTS_PER_ACTIVITY} for a subject-separated split.')
    splits = _split_by_subject(rows)

    from ultralytics import YOLO
    pose_model = YOLO(str(POSE_MODEL_PATH))
    encoded = {}
    for split, split_rows in splits.items():
        encoded[split] = [
            (_pose_sequence(row, pose_model), ACTIVITY_INDEX[row['activity']])
            for row in split_rows
        ]

    counts = Counter(label for _, label in encoded['train'])
    class_weights = torch.tensor(
        [len(encoded['train']) / (len(ACTIVITIES) * max(counts[index], 1)) for index in range(len(ACTIVITIES))],
        dtype=torch.float32,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TemporalActivityModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    best_validation, best_state, stale_epochs = float('inf'), None, 0

    for _epoch in range(MAX_EPOCHS):
        model.train()
        order = list(range(len(encoded['train'])))
        random.shuffle(order)
        for start in range(0, len(order), 32):
            batch = [encoded['train'][index] for index in order[start:start + 32]]
            sequences = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32)
            labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
            sequences = torch.stack([_augment(sequence) for sequence in sequences]).to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(sequences), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        losses = []
        with torch.inference_mode():
            for start in range(0, len(encoded['validation']), 32):
                batch = encoded['validation'][start:start + 32]
                sequences = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32, device=device)
                labels = torch.tensor([item[1] for item in batch], dtype=torch.long, device=device)
                losses.append(float(criterion(model(sequences), labels).item()))
        validation_loss = float(np.mean(losses))
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {name: value.cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError('Training did not produce a validation checkpoint.')
    model.load_state_dict(best_state)
    model.to(device).eval()
    actual, predicted = [], []
    with torch.inference_mode():
        for sequence, label in encoded['test']:
            logits = model(torch.tensor(sequence, dtype=torch.float32, device=device).unsqueeze(0))[0]
            actual.append(label)
            predicted.append(int(torch.argmax(logits).item()))
    test_metrics = _metrics(actual, predicted)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.cpu().state_dict(), 'activities': list(ACTIVITIES)}, MODEL_PATH)
    output = {
        'dataset': report,
        'splitSequences': {name: len(items) for name, items in encoded.items()},
        'test': test_metrics,
        'validationLoss': best_validation,
        'device': str(device),
    }
    METRICS_PATH.write_text(json.dumps(output, indent=2), encoding='utf-8')
    return output


if __name__ == '__main__':
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(json.dumps(train_activity_model(), indent=2))
