from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / 'dataset_frames'
MODEL_PATH = BASE_DIR / 'backend' / 'threat_model.pth'
MODEL_METADATA_PATH = BASE_DIR / 'backend' / 'model_meta.json'

CLASS_NAMES = ['Normal Class', 'Wall crossing']
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASS_NAMES)}
ACTIVITY_CLASSES = ['crawling', 'sleeping', 'running', 'walking', 'standing']


@lru_cache(maxsize=None)
def image_files_from_dir(folder: Path):
    if not folder.exists():
        return []
    results = []
    for image_path in sorted(folder.rglob('*')):
        if image_path.is_file() and image_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            results.append(str(image_path))
    return results


class ThreatDataset(Dataset):
    def __init__(self, root_dir: Path, class_names: list[str], max_per_class: int | None = None):
        self.samples = []
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        for label in class_names:
            class_dir = root_dir / label
            files = image_files_from_dir(class_dir)
            selected_files = files if max_per_class is None else files[:max_per_class]
            for path in selected_files:
                self.samples.append((path, CLASS_TO_INDEX[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, label


def build_model(device: torch.device):
    model = models.resnet18(weights=None, num_classes=len(CLASS_NAMES))
    model.to(device)
    return model


def train_model():
    dataset = ThreatDataset(DATASET_DIR, CLASS_NAMES)
    if len(dataset) == 0:
        raise FileNotFoundError(f'No usable training images were found under {DATASET_DIR}')

    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(3):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    metadata = {
        'classes': CLASS_NAMES,
        'trained_samples': len(dataset),
        'max_samples_per_class': None,
        'activity_classes': [],
        'activity_training_available': False,
        'device': str(device),
        'threshold': 0.7,
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return {'status': 'trained', 'samples': len(dataset), 'classes': CLASS_NAMES}


def load_or_train_model():
    if MODEL_PATH.exists():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = build_model(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model

    return train_model()


def predict_image(image_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(device)
    state = torch.load(MODEL_PATH, map_location=device) if MODEL_PATH.exists() else None
    if state is None:
        train_model()
        state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    label = CLASS_NAMES[pred_idx]
    return {'label': label, 'confidence': round(confidence * 100, 2), 'probabilities': {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(len(CLASS_NAMES))}}


@lru_cache(maxsize=1)
def dataset_overview():
    overview = []
    for label in CLASS_NAMES:
        class_dir = DATASET_DIR / label
        files = image_files_from_dir(class_dir)
        overview.append({'label': label, 'sample_count': len(files), 'folder': str(class_dir)})
    return {
        'dataset_root': str(DATASET_DIR),
        'classes': overview,
        'special_objects': ['Person', 'Pen', 'Package', 'Vehicle', 'Weapon'],
        'activity_classes': ACTIVITY_CLASSES,
        'activity_training_available': False,
        'training_note': 'Threat model uses all available Normal Class and Wall crossing frames. Activity labels require labeled video sequences.',
    }
