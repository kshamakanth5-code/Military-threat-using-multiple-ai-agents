"""Train an isolated classifier on ATAS synthetic human-motion images."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset

from backend.train_har_starter import HARImageModel, _metrics


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "activity_dataset" / "ATAS_HUMAN_MOTION_DATASET_FINAL" / "splits"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "atas_human_motion_image_model.pth"
METRICS_PATH = Path(__file__).resolve().parent / "atas_human_motion_image_metrics.json"
IMAGE_SIZE = 128
SEED = 42
EPOCHS = 50
PATIENCE = 8


class MotionImages(Dataset):
    def __init__(self, records: list[tuple[Path, str]], class_to_index: dict[str, int], augment: bool = False):
        self.records = records
        self.class_to_index = class_to_index
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        with Image.open(path) as source:
            image = source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        if self.augment and random.random() < 0.5:
            image = ImageOps.mirror(image)
        if self.augment and random.random() < 0.35:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
        return tensor, self.class_to_index[label]


def _collect(split: str) -> list[tuple[Path, str]]:
    records = []
    for class_dir in sorted(path for path in (DATASET_DIR / split).iterdir() if path.is_dir()):
        records.extend((path, class_dir.name) for path in sorted(class_dir.iterdir()) if path.is_file())
    return records


def train() -> dict:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(torch.get_num_threads(), 4))

    splits = {name: _collect(name) for name in ("train", "val", "test")}
    classes = sorted({label for records in splits.values() for _, label in records})
    class_to_index = {name: index for index, name in enumerate(classes)}
    if not classes or any(not records for records in splits.values()):
        raise ValueError("Expected labeled train, val, and test folders under the dataset splits directory.")
    for name, records in splits.items():
        missing = set(classes) - {label for _, label in records}
        if missing:
            raise ValueError(f"{name} split is missing classes: {', '.join(sorted(missing))}")

    loaders = {
        "train": DataLoader(MotionImages(splits["train"], class_to_index, augment=True), batch_size=32, shuffle=True, num_workers=0),
        "val": DataLoader(MotionImages(splits["val"], class_to_index), batch_size=32, shuffle=False, num_workers=0),
        "test": DataLoader(MotionImages(splits["test"], class_to_index), batch_size=32, shuffle=False, num_workers=0),
    }
    model = HARImageModel(len(classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.03)
    best_accuracy, best_state, stale = -1.0, None, 0

    for epoch in range(EPOCHS):
        model.train()
        for images, labels in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        actual, predicted = [], []
        with torch.inference_mode():
            for images, labels in loaders["val"]:
                actual.extend(labels.tolist())
                predicted.extend(model(images).argmax(dim=1).tolist())
        accuracy = sum(a == b for a, b in zip(actual, predicted)) / max(len(actual), 1)
        print(f"epoch {epoch + 1}: val_accuracy={accuracy:.4f}", flush=True)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    actual, predicted = [], []
    with torch.inference_mode():
        for images, labels in loaders["test"]:
            actual.extend(labels.tolist())
            predicted.extend(model(images).argmax(dim=1).tolist())

    result = {
        "dataset": "activity_dataset/ATAS_HUMAN_MOTION_DATASET_FINAL/splits",
        "datasetNote": "AI-generated 640x640 JPEG motion images; isolated image-classification baseline.",
        "classes": classes,
        "splitCounts": {name: len(records) for name, records in splits.items()},
        "bestValidationAccuracy": best_accuracy,
        "test": _metrics(actual, predicted, classes),
        "imageSize": IMAGE_SIZE,
        "device": "cpu",
    }
    torch.save({"state_dict": model.state_dict(), "classes": classes, "image_size": IMAGE_SIZE}, CHECKPOINT_PATH)
    METRICS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
