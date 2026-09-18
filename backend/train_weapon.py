from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / 'dataset_frames' / 'weapon_detection' / 'data.yaml'
OUTPUT_DIR = BASE_DIR / 'runs' / 'detect'


if __name__ == '__main__':
    model = YOLO('yolo11n.pt')
    model.train(
        data=str(DATA_FILE),
        epochs=30,
        imgsz=640,
        batch=8,
        workers=0,
        device='cpu',
        project=str(OUTPUT_DIR),
        name='weapon_detector',
        exist_ok=True,
        plots=True,
    )
