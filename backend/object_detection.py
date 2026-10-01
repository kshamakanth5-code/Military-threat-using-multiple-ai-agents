"""Temporal confirmation helpers for potentially dangerous object detections."""
from __future__ import annotations

from threading import RLock
from uuid import uuid4


DANGEROUS_OBJECT_CLASSES = frozenset({'scissors', 'knife', 'blade', 'gun', 'sword', 'bomb', 'launcher', 'weapon'})
SHARP_OBJECT_CLASSES = frozenset({'scissors', 'knife', 'blade', 'sword'})


def normalize_dangerous_label(label: object) -> str:
    normalized = str(label or '').strip().lower().replace('_', ' ').replace('-', ' ')
    aliases = {
        'kitchen knife': 'knife',
        'pocket knife': 'knife',
        'dagger': 'knife',
        'sharp blade': 'blade',
        'shiny sharp object': 'blade',
        'shiny blade': 'blade',
        'metal blade': 'blade',
        'sharp metal object': 'blade',
        'metallic blade': 'blade',
        'sharp object': 'blade',
        'razor blade': 'blade',
        'box cutter': 'blade',
        'utility knife': 'knife',
        'paring knife': 'knife',
        'chef knife': 'knife',
        'scissor': 'scissors',
        'firearm': 'gun',
        'handgun': 'gun',
        'automatic rifle': 'gun',
        'shotgun': 'gun',
        'smg': 'gun',
        'sniper': 'gun',
        'bazooka': 'launcher',
        'grenade launcher': 'launcher',
        'grenade': 'bomb',
        'explosive': 'bomb',
        'explosive device': 'bomb',
    }
    return aliases.get(normalized, normalized)


def box_iou(left: list[float], right: list[float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    ix1, iy1 = max(lx, rx), max(ly, ry)
    ix2, iy2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def box_overlap_smaller(left: list[float], right: list[float]) -> float:
    """Return intersection area divided by the smaller box area."""
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    intersection = max(0.0, min(lx + lw, rx + rw) - max(lx, rx)) * max(0.0, min(ly + lh, ry + rh) - max(ly, ry))
    smaller_area = min(lw * lh, rw * rh)
    return intersection / smaller_area if smaller_area > 0 else 0.0


class ObjectTemporalConfirmer:
    """Confirms matching class/box detections across consecutive frames per user."""

    def __init__(self, required_frames: int = 3, iou_threshold: float = 0.2):
        self.required_frames = max(1, int(required_frames))
        self.iou_threshold = iou_threshold
        self._tracks: dict[str, list[dict]] = {}
        self._lock = RLock()

    def update(self, user_id: str, candidates: list[dict]) -> list[dict]:
        with self._lock:
            previous = self._tracks.get(user_id, [])
            used: set[int] = set()
            current = []
            output = []
            for candidate in candidates:
                label = normalize_dangerous_label(candidate.get('label'))
                if label not in DANGEROUS_OBJECT_CLASSES:
                    continue
                matching_index = next((index for index, track in enumerate(previous)
                    if index not in used and track['label'] == label
                    and box_iou(track['box'], candidate['box']) >= self.iou_threshold), None)
                if matching_index is None:
                    track_id, count = str(uuid4()), 1
                else:
                    used.add(matching_index)
                    track_id = previous[matching_index]['objectTrackId']
                    count = previous[matching_index]['confirmationFrames'] + 1
                track = {**candidate, 'label': label, 'objectTrackId': track_id,
                         'confirmationFrames': count, 'confirmed': count >= self.required_frames}
                current.append(track)
                output.append(track)
            self._tracks[user_id] = current
            return output


def associate_person(object_detection: dict, person_detections: list[dict]) -> str | None:
    """Associate object center with containing person, then best box overlap."""
    if not person_detections:
        return None
    x, y, width, height = object_detection['box']
    center_x, center_y = x + width / 2, y + height / 2
    containing = []
    for index, person in enumerate(person_detections):
        px, py, pw, ph = person['box']
        if px <= center_x <= px + pw and py <= center_y <= py + ph:
            containing.append(index)
    if containing:
        return f'person_{containing[0] + 1:02d}'
    best_index = max(range(len(person_detections)), key=lambda i: box_iou(object_detection['box'], person_detections[i]['box']))
    return f'person_{best_index + 1:02d}' if box_iou(object_detection['box'], person_detections[best_index]['box']) > 0 else None
