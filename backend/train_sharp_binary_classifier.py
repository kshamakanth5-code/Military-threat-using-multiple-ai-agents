"""Train and evaluate an image classifier on the local sharp/non-sharp set.

This dataset has image-level labels only. The resulting classifier identifies
whether an image is sharp or non-sharp; it does not localize individual objects.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path(r"C:\Users\ksham\Downloads\sharp_vs_nonsharp_dataset\sharp_vs_nonsharp_dataset")
RUNS = ROOT / "runs" / "classify"
RUN_NAME = "sharp_vs_nonsharp_classifier"
PRETRAINED = "yolo11n-cls.pt"


def split_counts() -> dict[str, dict[str, int]]:
    counts = {}
    for split in ("train", "val", "test"):
        counts[split] = {}
        for class_name in ("sharp", "non_sharp"):
            folder = SOURCE / split / class_name
            if not folder.is_dir():
                raise FileNotFoundError(f"Expected class folder is missing: {folder}")
            counts[split][class_name] = sum(
                item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                for item in folder.iterdir()
            )
            if counts[split][class_name] == 0:
                raise ValueError(f"No supported images found in {folder}")
    return counts


def evaluate(checkpoint: Path, counts: dict[str, dict[str, int]]) -> dict:
    best_model = YOLO(str(checkpoint))
    metrics = best_model.val(
        data=str(SOURCE),
        split="test",
        imgsz=224,
        batch=16,
        workers=0,
        device="cpu",
        project=str(RUNS),
        name=f"{RUN_NAME}_test",
        exist_ok=True,
        plots=True,
    )

    class_names = best_model.names
    class_ids = {str(name): int(class_id) for class_id, name in class_names.items()}
    matrix = [[0 for _ in class_ids] for _ in class_ids]
    test_images = sorted((SOURCE / "test").glob("*/*"))
    predictions = best_model.predict(
        source=[str(path) for path in test_images if path.is_file()],
        imgsz=224,
        device="cpu",
        verbose=False,
    )
    for prediction in predictions:
        true_name = Path(prediction.path).parent.name
        true_id = class_ids[true_name]
        predicted_id = int(prediction.probs.top1)
        matrix[true_id][predicted_id] += 1

    per_class = {}
    for class_id, name in class_names.items():
        class_id = int(class_id)
        true_positive = float(matrix[class_id][class_id])
        false_positive = float(sum(row[class_id] for row in matrix) - true_positive)
        false_negative = float(sum(matrix[class_id]) - true_positive)
        precision = true_positive / max(true_positive + false_positive, 1e-12)
        recall = true_positive / max(true_positive + false_negative, 1e-12)
        per_class[str(name)] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        }

    summary = {
        "task": "image_classification",
        "dataset": str(SOURCE),
        "splitCounts": counts,
        "checkpoint": str(checkpoint),
        "test": {
            "accuracy": float(metrics.top1),
            "top5Accuracy": float(metrics.top5),
            "perClass": per_class,
            "confusionMatrix": matrix,
            "confusionMatrixClassOrder": [name for name in class_names.values()],
        },
        "limitation": "Image-level sharp/non-sharp classification only; no object boxes or object names are provided.",
    }
    report = RUNS / RUN_NAME / "test_metrics.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report"] = str(report)
    return summary


def train() -> dict:
    counts = split_counts()
    model = YOLO(PRETRAINED)
    model.train(
        data=str(SOURCE),
        epochs=30,
        imgsz=224,
        batch=16,
        workers=0,
        device="cpu",
        patience=8,
        project=str(RUNS),
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        seed=42,
        val=True,
    )
    return evaluate(RUNS / RUN_NAME / "weights" / "best.pt", counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate the existing best checkpoint without training again.")
    arguments = parser.parse_args()
    dataset_counts = split_counts()
    checkpoint_path = RUNS / RUN_NAME / "weights" / "best.pt"
    result = evaluate(checkpoint_path, dataset_counts) if arguments.evaluate_only else train()
    print(json.dumps(result, indent=2))
