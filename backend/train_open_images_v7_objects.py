"""Download a balanced Open Images V7 subset and fine-tune a detector on it.

Open Images supplies object-level bounding boxes. This script keeps the source
dataset untouched and writes a class-balanced YOLO subset under dataset/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset" / "open_images_v7_objects"
CACHE = DATASET / "fiftyone_cache"
DATA_YAML = DATASET / "data.yaml"
RUNS = ROOT / "runs" / "detect"
RUN_NAME = "open_images_v7_objects"
INITIAL_MODEL = ROOT / "yolo11n.pt"

DANGEROUS_CLASSES = [
    "Axe",
    "Bomb",
    "Dagger",
    "Handgun",
    "Kitchen knife",
    "Knife",
    "Scissors",
    "Shotgun",
    "Sword",
]
NORMAL_CLASSES = [
    "Backpack",
    "Book",
    "Bottle",
    "Bowl",
    "Camera",
    "Coffee cup",
    "Computer keyboard",
    "Computer mouse",
    "Handbag",
    "Headphones",
    "Laptop",
    "Mobile phone",
    "Mug",
    "Pen",
    "Pencil case",
    "Spoon",
    "Tablet computer",
    "Television",
    "Toy",
    "Umbrella",
    "Watch",
]
CLASSES = DANGEROUS_CLASSES + NORMAL_CLASSES
CLASS_IDS = {name: class_id for class_id, name in enumerate(CLASSES)}
SAMPLES_PER_CLASS = {"train": 50, "validation": 20, "test": 20}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _reset_export_dirs() -> None:
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            folder = DATASET / kind / split
            folder.mkdir(parents=True, exist_ok=True)
            for item in folder.iterdir():
                if item.is_file():
                    item.unlink()


def download_subset() -> dict:
    import fiftyone as fo
    import fiftyone.zoo as foz

    CACHE.mkdir(parents=True, exist_ok=True)
    fo.config.dataset_zoo_dir = str(CACHE / "datasets")
    fo.config.database_dir = str(CACHE / "database")
    _reset_export_dirs()

    summary = {
        "source": "Open Images V7",
        "sourceUrl": "https://storage.googleapis.com/openimages/web/index.html",
        "classes": CLASSES,
        "dangerousClasses": DANGEROUS_CLASSES,
        "normalClasses": NORMAL_CLASSES,
        "perClassSampleCaps": SAMPLES_PER_CLASS,
        "splits": {},
    }
    metadata_rows = []

    for source_split, output_split in (("train", "train"), ("validation", "val"), ("test", "test")):
        image_dir = DATASET / "images" / output_split
        label_dir = DATASET / "labels" / output_split
        merged_labels: dict[str, set[str]] = {}
        image_sources: dict[str, Path] = {}
        class_sample_counts = {}

        for class_name in CLASSES:
            dataset_name = f"oiv7_{source_split}_{_slug(class_name)}"
            samples = foz.load_zoo_dataset(
                "open-images-v7",
                split=source_split,
                label_types=["detections"],
                classes=[class_name],
                only_matching=False,
                load_hierarchy=False,
                shuffle=True,
                seed=42,
                max_samples=SAMPLES_PER_CLASS[source_split],
                dataset_name=dataset_name,
                drop_existing_dataset=True,
                persistent=True,
            )
            count = 0
            for sample in samples.iter_samples(progress=True):
                source_image = Path(sample.filepath)
                image_id = source_image.stem
                destination_name = image_id + source_image.suffix.lower()
                image_sources.setdefault(destination_name, source_image)
                merged_labels.setdefault(destination_name, set())
                count += 1

                detections = getattr(sample, "ground_truth", None)
                for detection in getattr(detections, "detections", []) or []:
                    class_id = CLASS_IDS.get(detection.label)
                    if class_id is None:
                        continue
                    x, y, width, height = (float(value) for value in detection.bounding_box)
                    x1, y1 = max(0.0, x), max(0.0, y)
                    x2, y2 = min(1.0, x + width), min(1.0, y + height)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    line = f"{class_id} {center_x:.7f} {center_y:.7f} {x2 - x1:.7f} {y2 - y1:.7f}"
                    merged_labels[destination_name].add(line)

            class_sample_counts[class_name] = count
            fo.delete_dataset(samples.name)

        annotated_images = 0
        annotated_objects = 0
        for image_name, source_image in image_sources.items():
            destination_image = image_dir / image_name
            if not destination_image.is_file():
                shutil.copy2(source_image, destination_image)
            lines = sorted(merged_labels[image_name])
            (label_dir / f"{Path(image_name).stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            annotated_images += bool(lines)
            annotated_objects += len(lines)
            metadata_rows.append({"split": output_split, "image": image_name, "source": "Open Images V7"})

        summary["splits"][output_split] = {
            "images": len(image_sources),
            "imagesWithSelectedBoxes": int(annotated_images),
            "annotatedObjects": annotated_objects,
            "classSampleCounts": class_sample_counts,
            "classBoxCounts": {
                name: sum(line.startswith(f"{class_id} ") for labels in merged_labels.values() for line in labels)
                for name, class_id in CLASS_IDS.items()
            },
        }
        print(json.dumps({"split": output_split, **summary["splits"][output_split]}, indent=2), flush=True)

    names_yaml = "\n".join(f"  {index}: '{name}'" for index, name in enumerate(CLASSES))
    DATA_YAML.write_text(
        f"path: '{DATASET.resolve().as_posix()}'\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names:\n{names_yaml}\n",
        encoding="utf-8",
    )
    (DATASET / "source_images.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata_rows),
        encoding="utf-8",
    )
    (DATASET / "download_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train() -> dict:
    from ultralytics import YOLO

    summary = download_subset()
    model = YOLO(str(INITIAL_MODEL))
    model.train(
        data=str(DATA_YAML),
        epochs=6,
        imgsz=256,
        batch=64,
        workers=0,
        device="cpu",
        freeze=10,
        patience=4,
        project=str(RUNS),
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        seed=42,
        val=False,
    )

    checkpoint = RUNS / RUN_NAME / "weights" / "best.pt"
    best_model = YOLO(str(checkpoint))
    metrics = best_model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=320,
        batch=32,
        workers=0,
        device="cpu",
        project=str(RUNS),
        name=f"{RUN_NAME}_test",
        exist_ok=True,
        plots=True,
    )
    per_class = {
        str(name): {
            "precision": float(metrics.box.p[index]),
            "recall": float(metrics.box.r[index]),
            "f1": float(
                2 * metrics.box.p[index] * metrics.box.r[index]
                / max(metrics.box.p[index] + metrics.box.r[index], 1e-12)
            ),
            "mAP50": float(metrics.box.ap50[index]),
        }
        for index, name in enumerate(metrics.names.values())
    }
    macro_f1 = sum(item["f1"] for item in per_class.values()) / max(len(per_class), 1)
    result = {
        "dataset": summary,
        "checkpoint": str(checkpoint),
        "test": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "macroF1": macro_f1,
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "perClass": per_class,
        },
    }
    (RUNS / RUN_NAME / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
