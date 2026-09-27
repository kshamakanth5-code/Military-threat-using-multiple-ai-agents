"""Train a separate walking-vs-other pose model from CMU video clips.

This script leaves the existing activity dataset, model, and metrics untouched.
It writes a new checkpoint and metrics file alongside the existing backend files.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from backend.activity_recognition import SEQUENCE_LENGTH, TemporalActivityModel
from backend.train_activity import POSE_MODEL_PATH, _augment, _pose_sequence


BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_ROOT = BASE_DIR / "activity_dataset" / "cmu_motion_performance" / "mpg" / "mpg"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "cmu_binary_activity_model.pth"
METRICS_PATH = Path(__file__).resolve().parent / "cmu_binary_activity_metrics.json"
CLASSES = ("WALKING", "OTHER_MOVEMENT")
SEED = 42
MAX_EPOCHS = 80
PATIENCE = 12

# CMU trial descriptions identify these clips as walking. All other available
# clips are grouped into the deliberately broad OTHER_MOVEMENT label.
def _label(subject: int, trial: int) -> int:
    is_walking = (
        (subject == 2 and trial == 2)
        or (subject == 3 and trial in {1, 2, 3})
        or (subject == 5 and trial == 1)
        or (subject == 6 and trial == 1)
        or (subject == 8 and 1 <= trial <= 11)
        or (subject == 9 and trial == 12)
    )
    return 0 if is_walking else 1


def _discover() -> list[dict]:
    rows = []
    for path in sorted(VIDEO_ROOT.rglob("*.mpg")):
        match = re.fullmatch(r"(?P<subject>\d+)_(?P<trial>\d+)", path.stem)
        if not match:
            continue
        subject, trial = int(match.group("subject")), int(match.group("trial"))
        rows.append({"source": path, "subject": f"{subject:02d}", "label": _label(subject, trial)})
    return rows


def _scores(actual: list[int], predicted: list[int]) -> dict:
    matrix = [[0, 0], [0, 0]]
    for truth, guess in zip(actual, predicted):
        matrix[truth][guess] += 1
    per_class = {}
    for index, name in enumerate(CLASSES):
        tp = matrix[index][index]
        fp = matrix[1 - index][index]
        fn = matrix[index][1 - index]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": sum(matrix[index]),
        }
    return {
        "accuracy": sum(a == p for a, p in zip(actual, predicted)) / max(len(actual), 1),
        "macroF1": sum(item["f1"] for item in per_class.values()) / len(CLASSES),
        "perClass": per_class,
        "confusionMatrix": matrix,
    }


def train() -> dict:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(torch.get_num_threads(), 4))

    rows = _discover()
    if not rows:
        raise FileNotFoundError(f"No CMU MPG videos found under {VIDEO_ROOT}")
    split_subjects = {
        "train": {"01", "02", "03"},
        "validation": {"05"},
        "test": {"06", "08", "09"},
    }
    splits = {name: [row for row in rows if row["subject"] in subjects] for name, subjects in split_subjects.items()}
    for name, items in splits.items():
        if not items or {row["label"] for row in items} != {0, 1}:
            raise ValueError(f"{name} split does not contain both binary classes")

    from ultralytics import YOLO

    pose_model = YOLO(str(POSE_MODEL_PATH))
    encoded = {name: [] for name in splits}
    for name, items in splits.items():
        for index, row in enumerate(items, start=1):
            sequence = _pose_sequence(row, pose_model)
            encoded[name].append((sequence, row["label"], row["subject"], row["source"].name))
            print(f"pose extraction {name}: {index}/{len(items)} ({row['source'].name})", flush=True)

    train_counts = Counter(label for _, label, _, _ in encoded["train"])
    weights = torch.tensor(
        [len(encoded["train"]) / (2 * max(train_counts[index], 1)) for index in range(2)],
        dtype=torch.float32,
    )
    model = TemporalActivityModel(classes=len(CLASSES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)
    best_loss, best_state, stale = float("inf"), None, 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        # Balance batches by resampling clips, while keeping every split
        # person-separated. This avoids the train set's strong class skew.
        counts = [max(train_counts[index], 1) for index in range(2)]
        epoch_size = 2 * max(counts)
        order = random.choices(
            list(range(len(encoded["train"]))),
            weights=[1.0 / counts[item[1]] for item in encoded["train"]],
            k=epoch_size,
        )
        random.shuffle(order)
        for start in range(0, len(order), 8):
            batch = [encoded["train"][i] for i in order[start:start + 8]]
            sequences = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32)
            labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
            sequences = torch.stack([_augment(sequence) for sequence in sequences])
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(sequences), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        validation_losses = []
        with torch.inference_mode():
            for start in range(0, len(encoded["validation"]), 8):
                batch = encoded["validation"][start:start + 8]
                x = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32)
                y = torch.tensor([item[1] for item in batch], dtype=torch.long)
                validation_losses.append(float(criterion(model(x), y).item()))
        validation_loss = float(np.mean(validation_losses))
        print(f"epoch {epoch + 1}: validation_loss={validation_loss:.4f}", flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    actual, predicted, test_probabilities = [], [], []
    with torch.inference_mode():
        for sequence, label, _, _ in encoded["test"]:
            logits = model(torch.tensor(sequence, dtype=torch.float32).unsqueeze(0))[0]
            actual.append(label)
            test_probabilities.append(float(torch.softmax(logits, dim=0)[0].item()))
            predicted.append(int(logits.argmax().item()))

    result = {
        "classes": list(CLASSES),
        "labelSource": "CMU motion index descriptions keyed by subject_trial filename",
        "videoCount": len(rows),
        "subjects": sorted({row["subject"] for row in rows}),
        "splitSubjects": {name: sorted(values) for name, values in split_subjects.items()},
        "splitCounts": {
            name: {
                "clips": len(items),
                "classes": {CLASSES[index]: sum(item[1] == index for item in items) for index in range(2)},
            }
            for name, items in encoded.items()
        },
        "validationLoss": best_loss,
        "test": _scores(actual, predicted),
        "testWalkingProbabilities": test_probabilities,
        "poseCheckpoint": POSE_MODEL_PATH.name,
        "device": "cpu",
    }
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "activities": list(CLASSES)}, CHECKPOINT_PATH)
    METRICS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
