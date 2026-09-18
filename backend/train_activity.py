from __future__ import annotations

import json
from pathlib import Path

ACTIVITY_CLASSES = ('crawling', 'sleeping', 'running', 'walking', 'standing')
BASE_DIR = Path(__file__).resolve().parent.parent
ACTIVITY_DATASET_DIR = BASE_DIR / 'activity_dataset'
ACTIVITY_MODEL_PATH = BASE_DIR / 'backend' / 'activity_model.pth'


def find_sequences():
    return {
        label: sorted(path for path in (ACTIVITY_DATASET_DIR / label).glob('*') if path.is_dir())
        for label in ACTIVITY_CLASSES
    }


def train_activity_model():
    sequences = find_sequences()
    missing = [label for label, paths in sequences.items() if not paths]
    if missing:
        raise FileNotFoundError(
            'Activity training needs labeled sequence folders: '
            + ', '.join(f'{label}={ACTIVITY_DATASET_DIR / label}' for label in missing)
        )

    raise NotImplementedError(
        'Activity sequence extraction is intentionally blocked until labeled data is present; '
        'do not train on guessed labels because that would make accuracy claims invalid.'
    )


if __name__ == '__main__':
    print(json.dumps(train_activity_model(), indent=2))
