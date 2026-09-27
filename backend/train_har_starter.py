"""Train an isolated image-classification baseline on the provided HAR set."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "activity_dataset" / "HAR"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "har_starter_image_model.pth"
METRICS_PATH = Path(__file__).resolve().parent / "har_starter_image_metrics.json"
IMAGE_SIZE = 128
SEED = 42
EPOCHS = 50
PATIENCE = 8


class HARImageDataset(Dataset):
    def __init__(self, rows: list[dict], class_to_index: dict[str, int], augment: bool = False):
        self.rows = rows
        self.class_to_index = class_to_index
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = DATASET_DIR / row["filepath"]
        with Image.open(path) as source:
            image = source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        if self.augment and random.random() < 0.5:
            image = ImageOps.mirror(image)
        if self.augment and random.random() < 0.35:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
        return tensor, self.class_to_index[row["label"]]


class HARImageModel(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(96, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(128, classes))

    def forward(self, images):
        return self.classifier(self.features(images))


def _metrics(actual: list[int], predicted: list[int], classes: list[str]) -> dict:
    matrix = [[0 for _ in classes] for _ in classes]
    for truth, guess in zip(actual, predicted):
        matrix[truth][guess] += 1
    per_class = {}
    for index, name in enumerate(classes):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(len(classes)) if row != index)
        fn = sum(matrix[index][col] for col in range(len(classes)) if col != index)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": sum(matrix[index]),
        }
    return {
        "accuracy": sum(a == b for a, b in zip(actual, predicted)) / max(len(actual), 1),
        "macroF1": sum(item["f1"] for item in per_class.values()) / max(len(classes), 1),
        "perClass": per_class,
        "confusionMatrix": matrix,
    }


def train() -> dict:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(torch.get_num_threads(), 4))

    with (DATASET_DIR / "labels.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    classes = sorted({row["label"] for row in rows})
    class_to_index = {name: index for index, name in enumerate(classes)}
    splits = {split: [row for row in rows if row["split"] == split] for split in ("train", "val", "test")}
    if not classes or any(not split_rows for split_rows in splits.values()):
        raise ValueError("The HAR labels file must provide classes and train/val/test rows.")
    for split, split_rows in splits.items():
        missing = set(classes) - {row["label"] for row in split_rows}
        if missing:
            raise ValueError(f"{split} split is missing classes: {', '.join(sorted(missing))}")

    loaders = {
        "train": DataLoader(HARImageDataset(splits["train"], class_to_index, augment=True), batch_size=32, shuffle=True, num_workers=0),
        "val": DataLoader(HARImageDataset(splits["val"], class_to_index), batch_size=32, shuffle=False, num_workers=0),
        "test": DataLoader(HARImageDataset(splits["test"], class_to_index), batch_size=32, shuffle=False, num_workers=0),
    }
    model = HARImageModel(len(classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    best_accuracy, best_state, stale = -1.0, None, 0

    for epoch in range(EPOCHS):
        model.train()
        for images, labels in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_actual, val_predicted = [], []
        with torch.inference_mode():
            for images, labels in loaders["val"]:
                logits = model(images)
                val_actual.extend(labels.tolist())
                val_predicted.extend(logits.argmax(dim=1).tolist())
        val_accuracy = sum(a == b for a, b in zip(val_actual, val_predicted)) / max(len(val_actual), 1)
        print(f"epoch {epoch + 1}: val_accuracy={val_accuracy:.4f}", flush=True)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
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
    test_actual, test_predicted = [], []
    with torch.inference_mode():
        for images, labels in loaders["test"]:
            logits = model(images)
            test_actual.extend(labels.tolist())
            test_predicted.extend(logits.argmax(dim=1).tolist())

    result = {
        "dataset": "activity_dataset/HAR",
        "datasetNote": "Synthetic stick-figure images; baseline pipeline training only.",
        "classes": classes,
        "splitCounts": {name: len(items) for name, items in splits.items()},
        "bestValidationAccuracy": best_accuracy,
        "test": _metrics(test_actual, test_predicted, classes),
        "imageSize": IMAGE_SIZE,
        "device": "cpu",
    }
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes, "image_size": IMAGE_SIZE}, CHECKPOINT_PATH)
    METRICS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
