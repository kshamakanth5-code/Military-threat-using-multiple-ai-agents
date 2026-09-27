"""Create a consolidated train/test dataset tree using space-efficient hard links."""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "dataset"
ACTIVITY = ROOT / "activity_dataset"
FRAMES = ROOT / "dataset_frames"


def link_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except FileExistsError:
        raise RuntimeError(f"Refusing to replace existing file: {target}")


def add_tree(source: Path, target: Path) -> int:
    count = 0
    for path in source.rglob("*"):
        if path.is_file():
            link_file(path, target / path.relative_to(source))
            count += 1
    return count


def add_har() -> int:
    source = ACTIVITY / "HAR"
    count = 0
    with (source / "labels.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        split = row["split"]
        # Keep the project's requested top-level train/test structure while
        # retaining validation as a visibly separate subset under train.
        parent = "train/validation" if split in {"val", "validation"} else split
        src = source / row["filepath"]
        link_file(src, DEST / parent / "human_activity" / "HAR" / row["label"] / src.name)
        count += 1
    for split in ("train", "test"):
        selected = [row for row in rows if row["split"] == split]
        out = DEST / split / "human_activity" / "HAR" / "labels.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filepath", "label", "split"])
            writer.writeheader()
            for row in selected:
                writer.writerow({"filepath": f"{row['label']}/{Path(row['filepath']).name}", "label": row["label"], "split": split})
    return count


def add_atas_motion_images() -> int:
    source = ACTIVITY / "ATAS_HUMAN_MOTION_DATASET_FINAL" / "splits"
    count = 0
    for split in ("train", "val", "test"):
        parent = "train/validation" if split == "val" else split
        count += add_tree(source / split, DEST / parent / "human_activity" / "ATAS_HUMAN_MOTION")
    return count


def _cmu_split(path: Path) -> str:
    parts = path.parts
    subject = None
    if "subjects" in parts:
        index = parts.index("subjects")
        if index + 1 < len(parts):
            subject = parts[index + 1]
    if subject is None:
        match = re.match(r"(?P<subject>\d+)_\d+", path.stem)
        if match:
            subject = match.group("subject")
    if subject in {"05"}:
        return "train/validation"
    if subject in {"06", "08", "09"}:
        return "test"
    if subject in {"01", "02", "03"}:
        return "train"
    if subject and subject.isdigit() and int(subject) % 5 == 0:
        return "test"
    return "train"


def add_cmu_motion() -> int:
    source = ACTIVITY / "cmu_motion_performance"
    count = 0
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        split = _cmu_split(path)
        relative = path.relative_to(source)
        link_file(path, DEST / split / "human_activity" / "CMU_Motion" / relative)
        count += 1
    return count


def add_sequence_detection_data() -> int:
    count = 0
    for class_name in ("Normal Class", "Wall crossing"):
        source = FRAMES / class_name
        for sequence_dir in (path for path in source.iterdir() if path.is_dir()):
            try:
                sequence_id = int(sequence_dir.name)
            except ValueError:
                # Keep non-numeric folders in training and note their origin.
                sequence_id = 1
            split = "test" if sequence_id % 5 == 0 else "train"
            count += add_tree(sequence_dir, DEST / split / "scene_frames" / class_name / sequence_dir.name)
    return count


def add_yolo_detection_data() -> int:
    count = 0
    for name in ("sharp_object_detection", "weapon_detection"):
        source = FRAMES / name
        for source_split, target_split in (("train", "train"), ("val", "test")):
            split_source = source / source_split
            if split_source.exists():
                count += add_tree(split_source, DEST / target_split / "object_detection" / name)
        data_yaml = source / "data.yaml"
        if data_yaml.is_file():
            link_file(data_yaml, DEST / "train" / "object_detection" / name / "source_data.yaml")
    return count


def build() -> dict:
    if DEST.exists():
        raise FileExistsError(f"Destination already exists; refusing to overwrite: {DEST}")
    DEST.mkdir()
    (DEST / "train").mkdir()
    (DEST / "test").mkdir()

    counts = {
        "HAR": add_har(),
        "ATAS_HUMAN_MOTION": add_atas_motion_images(),
        "CMU_Motion": add_cmu_motion(),
        "sequence_scene_frames": add_sequence_detection_data(),
        "sharp_and_weapon_detection": add_yolo_detection_data(),
    }
    manifest = {
        "layout": "dataset/{train,test}; pre-existing validation data is stored under dataset/train/validation",
        "storage": "Dataset files are hard links to the original files where supported; originals remain in place.",
        "notes": [
            "Human-action image datasets and object/scene datasets remain in separate task folders.",
            "HAR and ATAS validation samples remain under train/validation, not mixed into train or test examples.",
            "CMU subject files are grouped without splitting a subject: subjects 01/02/03 train, 06/08/09 test, 05 validation; other subjects use a stable subject-ID split.",
            "Normal Class and Wall crossing are split by numbered sequence directory (IDs divisible by five go to test) to keep frames from a sequence together.",
            "Sharp-object and weapon-detection source validation folders are mapped to the consolidated test folder.",
        ],
        "linkedFileCounts": counts,
        "totalLinkedFiles": sum(counts.values()),
    }
    (DEST / "README.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
