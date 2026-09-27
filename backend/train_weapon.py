from pathlib import Path
import shutil

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DATASET = BASE_DIR / 'dataset_frames' / 'weapon_detection'
SHARP_DATASET = BASE_DIR / 'dataset_frames' / 'sharp_object_detection'
DATA_FILE = SHARP_DATASET / 'data.yaml'
OUTPUT_DIR = BASE_DIR / 'runs' / 'detect'
SHARP_CLASSES = {'Knife': 0, 'Sword': 1}


def prepare_sharp_dataset():
    """Create a two-class dataset from the existing knife and sword annotations.

    The source export has generic class names and mixes category names into the
    image filenames. Its object boxes are retained, while their class IDs are
    remapped from the image's labeled category into explicit sharp-object names.
    """
    if not SOURCE_DATASET.is_dir():
        raise FileNotFoundError(f'Weapon dataset not found: {SOURCE_DATASET}')

    for split in ('train', 'val'):
        image_dir = SOURCE_DATASET / split / 'images'
        label_dir = SOURCE_DATASET / split / 'labels'
        output_images = SHARP_DATASET / split / 'images'
        output_labels = SHARP_DATASET / split / 'labels'
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        for image_path in image_dir.iterdir():
            category = image_path.stem.rsplit('_', 1)[0]
            if category not in SHARP_CLASSES:
                continue
            label_path = label_dir / f'{image_path.stem}.txt'
            if not label_path.is_file():
                raise FileNotFoundError(f'Missing annotation for {image_path.name}')
            remapped = []
            for line in label_path.read_text(encoding='utf-8').splitlines():
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f'Invalid YOLO annotation in {label_path}: {line}')
                remapped.append(f"{SHARP_CLASSES[category]} {' '.join(fields[1:])}")
            if not remapped:
                raise ValueError(f'No bounding boxes in {label_path}')
            shutil.copy2(image_path, output_images / image_path.name)
            (output_labels / label_path.name).write_text('\n'.join(remapped) + '\n', encoding='utf-8')

        for category in SHARP_CLASSES:
            count = sum(1 for path in output_images.iterdir() if path.stem.startswith(f'{category}_'))
            if count == 0:
                raise ValueError(f'No {category} examples found in {split} split.')

    root = SHARP_DATASET.resolve().as_posix()
    DATA_FILE.write_text(
        f"path: '{root}'\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n"
        "  0: knife\n"
        "  1: sword\n",
        encoding='utf-8',
    )
    return DATA_FILE


if __name__ == '__main__':
    data_file = prepare_sharp_dataset()
    model = YOLO('yolo11n.pt')
    model.train(
        data=str(data_file),
        epochs=30,
        imgsz=640,
        batch=8,
        workers=0,
        device='cpu',
        project=str(OUTPUT_DIR),
        name='sharp_object_detector',
        exist_ok=True,
        plots=True,
    )
