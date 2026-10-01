"""Convert and train the user's local COCO guns/knives dataset."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path(r"C:\Users\ksham\Downloads\archive\guns-knives-coco\guns-knives-coco")
DATASET = ROOT / "dataset" / "guns_knives_coco"
SHARP_OBJECTS = ROOT / "dataset"
DATA_YAML = DATASET / "data.yaml"
RUNS = ROOT / "runs" / "detect"
RUN_NAME = "knife_blade_detector"
INITIAL_MODEL = RUNS / "dangerous_object_detector" / "weights" / "best.pt"
CLASS_IDS = {"knife": 0, "pistol": 2}
CLASS_NAMES = {0: "knife", 1: "blade", 2: "gun"}


def link_or_copy(source: Path, destination: Path):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_dataset():
    """Convert COCO boxes and add source-labeled blade examples in YOLO format."""
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {SOURCE}")

    # Recreate only this dedicated conversion directory, never the source data
    # or the existing shared `dataset/` training folders.
    for split in ("train", "valid", "test"):
        split_dir = DATASET / split
        for kind in ("images", "labels"):
            output_dir = split_dir / kind
            output_dir.mkdir(parents=True, exist_ok=True)
            for entry in output_dir.iterdir():
                if entry.is_file():
                    entry.unlink()

    summary = {"source": str(SOURCE), "classes": CLASS_NAMES, "splits": {}}
    for split in ("train", "valid", "test"):
        source_dir = SOURCE / split
        annotation_file = source_dir / "_annotations.coco.json"
        if not annotation_file.is_file():
            raise FileNotFoundError(annotation_file)
        coco = json.loads(annotation_file.read_text(encoding="utf-8"))
        categories = {item["id"]: item["name"].strip().lower() for item in coco["categories"]}
        category_map = {category_id: CLASS_IDS[name] for category_id, name in categories.items() if name in CLASS_IDS}
        if not {CLASS_IDS["knife"], CLASS_IDS["pistol"]}.issubset(category_map.values()):
            raise ValueError(f"{split} does not include both knife and pistol categories: {categories}")

        images = {item["id"]: item for item in coco["images"]}
        annotations = {image_id: [] for image_id in images}
        for item in coco["annotations"]:
            class_id = category_map.get(item["category_id"])
            if class_id is None:
                continue  # Ignore the unused generic parent category `weapon`.
            image = images[item["image_id"]]
            x, y, width, height = (float(value) for value in item["bbox"])
            image_width, image_height = float(image["width"]), float(image["height"])
            if image_width <= 0 or image_height <= 0 or width <= 0 or height <= 0:
                raise ValueError(f"Invalid dimensions/bounding box in {annotation_file}: {item}")
            # Clip boxes to image boundaries before conversion to normalized xywh.
            x1, y1 = max(0.0, x), max(0.0, y)
            x2 = min(image_width, x + width)
            y2 = min(image_height, y + height)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Bounding box falls outside its image: {annotation_file}: {item}")
            box = ((x1 + x2) / 2 / image_width, (y1 + y2) / 2 / image_height,
                   (x2 - x1) / image_width, (y2 - y1) / image_height)
            annotations[item["image_id"]].append((class_id, *box))

        image_output = DATASET / split / "images"
        label_output = DATASET / split / "labels"
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)
        for image in coco["images"]:
            source_image = source_dir / image["file_name"]
            if not source_image.is_file():
                raise FileNotFoundError(source_image)
            destination_image = image_output / Path(image["file_name"]).name
            link_or_copy(source_image, destination_image)
            lines = [
                f"{class_id} {xc:.8f} {yc:.8f} {width:.8f} {height:.8f}"
                for class_id, xc, yc, width, height in annotations[image["id"]]
            ]
            (label_output / f"{destination_image.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
        summary["splits"][split] = {
            "images": len(images),
            "annotated_objects": sum(map(len, annotations.values())),
            "empty_images": sum(not boxes for boxes in annotations.values()),
        }

    # The existing object-detection dataset has source-labeled knife and sword
    # boxes. Treat its sword class as the available blade example class and
    # map its several firearm labels to the existing broad gun class.
    source_to_output_class = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2}
    sharp_summary = {}
    for source_split, output_split in (("train", "train"), ("val", "valid"), ("test", "test")):
        image_dir = SHARP_OBJECTS / "images" / source_split
        label_dir = SHARP_OBJECTS / "labels" / source_split
        added_images = added_objects = 0
        for source_image in sorted(image_dir.iterdir()):
            if not source_image.is_file():
                continue
            source_label = label_dir / f"{source_image.stem}.txt"
            if not source_label.is_file():
                raise FileNotFoundError(source_label)
            remapped = []
            for line in source_label.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"Malformed sharp-object annotation: {source_label}: {line}")
                source_class = int(fields[0])
                if source_class not in source_to_output_class:
                    raise ValueError(f"Unknown sharp-object class {source_class} in {source_label}")
                remapped.append(f"{source_to_output_class[source_class]} {' '.join(fields[1:])}")
            if not remapped:
                continue
            destination_name = f"sharp_{source_image.name}"
            destination_image = DATASET / output_split / "images" / destination_name
            destination_label = DATASET / output_split / "labels" / f"{Path(destination_name).stem}.txt"
            link_or_copy(source_image, destination_image)
            destination_label.write_text("\n".join(remapped) + "\n", encoding="utf-8")
            added_images += 1
            added_objects += len(remapped)
        sharp_summary[output_split] = {"images": added_images, "annotated_objects": added_objects}
        summary["splits"][output_split]["images"] += added_images
        summary["splits"][output_split]["annotated_objects"] += added_objects
    summary["additional_sharp_source"] = str(SHARP_OBJECTS / "images")
    summary["sharp_source_summary"] = sharp_summary

    names = "\n".join(f"  {class_id}: {name}" for class_id, name in CLASS_NAMES.items())
    root_path = DATASET.resolve().as_posix()
    DATA_YAML.write_text(
        f"path: '{root_path}'\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    (DATASET / "conversion_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train():
    summary = prepare_dataset()
    checkpoint = INITIAL_MODEL if INITIAL_MODEL.is_file() else ROOT / "yolo11n.pt"
    model = YOLO(str(checkpoint))
    result = model.train(
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
        pretrained=True,
        seed=42,
        val=False,
    )
    model_path = RUNS / RUN_NAME / "weights" / "best.pt"
    best_model = YOLO(str(model_path))
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
    precision, recall = float(metrics.box.mp), float(metrics.box.mr)
    macro_f1 = sum(item["f1"] for item in per_class.values()) / max(len(per_class), 1)
    return {
        "dataset": summary,
        "checkpoint": str(model_path),
        "test": {
            "precision": precision,
            "recall": recall,
            "macroF1": macro_f1,
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "perClass": per_class,
        },
    }


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
