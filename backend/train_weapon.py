from pathlib import Path
import shutil

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DATASET = BASE_DIR / 'dataset_frames' / 'weapon_detection'
HF_DATASET = BASE_DIR / 'activity_dataset' / 'cctv-weapon-dataset' / 'samples'
DANGEROUS_DATASET = BASE_DIR / 'dataset_frames' / 'dangerous_object_detection'
DATA_FILE = DANGEROUS_DATASET / 'data.yaml'
OUTPUT_DIR = BASE_DIR / 'runs' / 'detect'
INITIAL_CHECKPOINT = OUTPUT_DIR / 'dangerous_object_detector' / 'weights' / 'best.pt'
DANGEROUS_CLASSES = {
    'Knife': ('knife', 0), 'Sword': ('knife', 0),
    'Automatic Rifle': ('gun', 1), 'Handgun': ('gun', 1),
    'Shotgun': ('gun', 1), 'SMG': ('gun', 1), 'Sniper': ('gun', 1),
    'Bazooka': ('launcher', 2), 'Grenade Launcher': ('launcher', 2),
}
HF_WEAPON_CLASS_ID = 3


def prepare_dangerous_dataset():
    """Create a filtered dangerous-object set and optionally add CCTV weapon boxes.

    The source export has generic class names and mixes category names into the
    image filenames. Its object boxes are retained, while their class IDs are
    remapped from the image's labeled category into explicit sharp-object names.
    """
    if not SOURCE_DATASET.is_dir():
        raise FileNotFoundError(f'Weapon dataset not found: {SOURCE_DATASET}')

    for split in ('train', 'val'):
        image_dir = SOURCE_DATASET / split / 'images'
        label_dir = SOURCE_DATASET / split / 'labels'
        output_images = DANGEROUS_DATASET / split / 'images'
        output_labels = DANGEROUS_DATASET / split / 'labels'
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        for image_path in image_dir.iterdir():
            category = image_path.stem.rsplit('_', 1)[0]
            if category not in DANGEROUS_CLASSES:
                continue
            label_path = label_dir / f'{image_path.stem}.txt'
            if not label_path.is_file():
                raise FileNotFoundError(f'Missing annotation for {image_path.name}')
            remapped = []
            for line in label_path.read_text(encoding='utf-8').splitlines():
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f'Invalid YOLO annotation in {label_path}: {line}')
                class_id = DANGEROUS_CLASSES[category][1]
                remapped.append(f"{class_id} {' '.join(fields[1:])}")
            if not remapped:
                raise ValueError(f'No bounding boxes in {label_path}')
            shutil.copy2(image_path, output_images / image_path.name)
            (output_labels / label_path.name).write_text('\n'.join(remapped) + '\n', encoding='utf-8')

    hf_image_dir = HF_DATASET / 'images'
    hf_label_dir = HF_DATASET / 'labels'
    include_hf = hf_image_dir.is_dir() and hf_label_dir.is_dir()
    if include_hf:
        for image_path in sorted(hf_image_dir.iterdir()):
            if not image_path.is_file():
                continue
            label_path = hf_label_dir / f'{image_path.stem}.txt'
            if not label_path.is_file():
                raise FileNotFoundError(f'Missing Hugging Face annotation for {image_path.name}')
            scene = image_path.stem.split('_', 1)[0]
            split = 'val' if scene in {'Scene5', 'Scene6'} else 'train'
            remapped = []
            for line in label_path.read_text(encoding='utf-8').splitlines():
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f'Invalid YOLO annotation in {label_path}: {line}')
                # The source class 0 is person and is deliberately excluded;
                # only its generic weapon class is added to the detector.
                if fields[0] == '1':
                    remapped.append(f"{HF_WEAPON_CLASS_ID} {' '.join(fields[1:])}")
            destination_name = f'hf_{image_path.name}'
            output_images = DANGEROUS_DATASET / split / 'images'
            output_labels = DANGEROUS_DATASET / split / 'labels'
            shutil.copy2(image_path, output_images / destination_name)
            (output_labels / f'{Path(destination_name).stem}.txt').write_text(
                '\n'.join(remapped) + '\n', encoding='utf-8'
            )

    class_definitions = [('knife', 0), ('gun', 1), ('launcher', 2)]
    if include_hf:
        class_definitions.append(('weapon', HF_WEAPON_CLASS_ID))

    for split in ('train', 'val'):
        output_labels = DANGEROUS_DATASET / split / 'labels'
        for class_name, class_id in class_definitions:
            count = sum(
                1 for label_path in output_labels.iterdir()
                if any(line.split() and int(line.split()[0]) == class_id
                       for line in label_path.read_text(encoding='utf-8').splitlines())
            )
            if count == 0:
                raise ValueError(f'No {class_name} examples found in {split} split.')

    root = DANGEROUS_DATASET.resolve().as_posix()
    names = ''.join(f'  {class_id}: {class_name}\n' for class_name, class_id in class_definitions)
    DATA_FILE.write_text(
        f"path: '{root}'\n"
        "train: train/images\n"
        "val: val/images\n"
        f"names:\n{names}",
        encoding='utf-8',
    )
    return DATA_FILE


if __name__ == '__main__':
    data_file = prepare_dangerous_dataset()
    initial_model = INITIAL_CHECKPOINT if INITIAL_CHECKPOINT.is_file() else BASE_DIR / 'yolo11n.pt'
    model = YOLO(str(initial_model))
    model.train(
        data=str(data_file),
        epochs=6,
        imgsz=320,
        batch=16,
        workers=0,
        device='cpu',
        project=str(OUTPUT_DIR),
        name='dangerous_object_detector_hf',
        exist_ok=True,
        plots=True,
    )
