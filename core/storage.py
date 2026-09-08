"""
IdentiScan - Local persistent storage.

People:  data/faces/<person_id>/profile_image.jpg + metadata.json
History: data/history.csv (append-only recognition event log)

Plain files on disk - no extra database beyond the ChromaDB vector store
that already exists in this project.
"""
import csv
import json
import os
import shutil
import threading
from datetime import datetime

import cv2

DATA_DIR = "data"
FACES_DIR = os.path.join(DATA_DIR, "faces")
HISTORY_PATH = os.path.join(DATA_DIR, "history.csv")
HISTORY_FIELDS = ["timestamp", "person_id", "name", "distance", "similarity", "status"]

_history_lock = threading.Lock()

os.makedirs(FACES_DIR, exist_ok=True)


def _person_dir(person_id):
    return os.path.join(FACES_DIR, person_id)


def person_exists(person_id):
    return os.path.isdir(_person_dir(person_id))


def save_person(person_id, image_bgr, metadata):
    """Persist a person's profile image and metadata to disk."""
    folder = _person_dir(person_id)
    os.makedirs(folder, exist_ok=True)

    image_path = os.path.join(folder, "profile_image.jpg")
    cv2.imwrite(image_path, image_bgr)

    metadata = dict(metadata)
    metadata["person_id"] = person_id
    metadata["registered_at"] = datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return image_path


def load_person(person_id):
    folder = _person_dir(person_id)
    meta_path = os.path.join(folder, "metadata.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        metadata = json.load(f)
    metadata["image_path"] = os.path.join(folder, "profile_image.jpg")
    return metadata


def list_people():
    """Return metadata for every registered person, newest first."""
    people = []
    if not os.path.isdir(FACES_DIR):
        return people
    for person_id in os.listdir(FACES_DIR):
        metadata = load_person(person_id)
        if metadata:
            people.append(metadata)
    people.sort(key=lambda p: p.get("registered_at", ""), reverse=True)
    return people


def delete_person(person_id):
    """Remove a person's stored image + metadata from disk."""
    folder = _person_dir(person_id)
    if os.path.isdir(folder):
        shutil.rmtree(folder)
        return True
    return False


def count_people():
    if not os.path.isdir(FACES_DIR):
        return 0
    return len([d for d in os.listdir(FACES_DIR) if os.path.isdir(_person_dir(d))])


# ---------------------------------------------------------------------------
# Recognition history
# ---------------------------------------------------------------------------

def log_recognition(person_id, name, distance, similarity, status):
    """Append one recognition event. Thread-safe - the live recognition video
    processor runs on a background thread, separate from the Streamlit script."""
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "person_id": person_id or "",
        "name": name or "Unknown",
        "distance": f"{distance:.4f}" if distance is not None else "",
        "similarity": f"{similarity:.1f}" if similarity is not None else "",
        "status": status,
    }
    with _history_lock:
        is_new = not os.path.exists(HISTORY_PATH)
        with open(HISTORY_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)


def read_history(limit=None):
    if not os.path.exists(HISTORY_PATH):
        return []
    with _history_lock:
        with open(HISTORY_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
    rows.reverse()  # most recent first
    if limit:
        rows = rows[:limit]
    return rows


def clear_history():
    with _history_lock:
        if os.path.exists(HISTORY_PATH):
            os.remove(HISTORY_PATH)


def history_stats():
    if not os.path.exists(HISTORY_PATH):
        return {"total": 0, "known": 0, "unknown": 0}
    with _history_lock:
        with open(HISTORY_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
    known = sum(1 for r in rows if r["status"] == "known")
    unknown = sum(1 for r in rows if r["status"] == "unknown")
    return {"total": len(rows), "known": known, "unknown": unknown}
