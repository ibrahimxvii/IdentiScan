"""
IdentiScan - Core recognition engine.

Wraps the existing detection / embedding / vector-search stack:
    MediaPipe (face detection) -> FaceNet, via facenet-pytorch (embeddings) -> ChromaDB (similarity search)

Models and the database client are loaded once via st.cache_resource so every
page shares the same instances instead of reloading them on each navigation.

Note: the embedding backend uses facenet-pytorch (PyTorch) rather than
keras-facenet (TensorFlow) because TensorFlow does not yet publish wheels for
newer Python releases (3.13+), while PyTorch does. Same pretrained FaceNet
architecture (InceptionResnetV1, trained on VGGFace2), same 512-d embedding
output, same role in the pipeline - only the runtime differs.
"""
import os

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import torch
import chromadb
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization

from utils import ensure_models_downloaded

COLLECTION_NAME = "face_database"
CHROMA_PATH = os.path.join("data", "chroma")

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_EMBED_SIZE = 160  # InceptionResnetV1 expects 160x160 face crops

# ---------------------------------------------------------------------------
# Recognition threshold - single source of truth used everywhere in the app.
# ChromaDB returns a distance (lower = more similar). A face is only treated
# as "known" if the closest distance is below this value.
# ---------------------------------------------------------------------------
RECOGNITION_THRESHOLD = 1.0

# Minimum face bounding-box size (pixels) accepted during registration.
MIN_FACE_SIZE = 40


@st.cache_resource(show_spinner="Loading face detector...")
def load_detector():
    detector_path = ensure_models_downloaded()
    base_options = mp_python.BaseOptions(model_asset_path=detector_path)
    options = mp_vision.FaceDetectorOptions(base_options=base_options)
    return mp_vision.FaceDetector.create_from_options(options)


@st.cache_resource(show_spinner="Loading FaceNet embedder...")
def load_embedder():
    model = InceptionResnetV1(pretrained="vggface2").eval().to(_DEVICE)
    return model


@st.cache_resource(show_spinner=False)
def get_chroma_collection():
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def get_engine_status():
    """Best-effort readiness check for the dashboard status pills.
    Never raises - returns what succeeded and the first error, if any."""
    status = {"detector": False, "embedder": False, "database": False, "error": None}
    try:
        load_detector()
        status["detector"] = True
        load_embedder()
        status["embedder"] = True
        get_chroma_collection()
        status["database"] = True
    except Exception as exc:  # defensive: surface a friendly message, not a traceback
        status["error"] = str(exc)
    return status


def detect_faces(image_bgr):
    """Run face detection on a BGR numpy image.
    Returns a list of (left, top, right, bottom) boxes clipped to the image bounds."""
    detector = load_detector()
    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = detector.detect(mp_image)

    h, w = image_bgr.shape[:2]
    boxes = []
    for detection in result.detections:
        bbox = detection.bounding_box
        left = max(0, bbox.origin_x)
        top = max(0, bbox.origin_y)
        right = min(w, bbox.origin_x + bbox.width)
        bottom = min(h, bbox.origin_y + bbox.height)
        if right > left and bottom > top:
            boxes.append((left, top, right, bottom))
    return boxes


def get_embedding(face_crop_bgr):
    """Generate a FaceNet embedding for a cropped face (BGR numpy array)."""
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return None
    model = load_embedder()

    rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (_EMBED_SIZE, _EMBED_SIZE))
    tensor = torch.from_numpy(np.ascontiguousarray(resized)).float()
    tensor = fixed_image_standardization(tensor)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        embedding = model(tensor)[0].cpu().numpy()
    return embedding


def validate_registration_image(image_bgr):
    """Validate an uploaded image before registration.
    Returns (ok, message, box) where box is (left, top, right, bottom) or None."""
    if image_bgr is None or image_bgr.size == 0:
        return False, "The uploaded file could not be read as an image.", None

    boxes = detect_faces(image_bgr)
    if len(boxes) == 0:
        return False, "No face was detected in this image. Please upload a clear, front-facing photo.", None
    if len(boxes) > 1:
        return False, "Multiple faces were detected. Please upload an image containing exactly one face.", None

    left, top, right, bottom = boxes[0]
    if (right - left) < MIN_FACE_SIZE or (bottom - top) < MIN_FACE_SIZE:
        return False, "The detected face is too small. Please upload a closer, higher-resolution photo.", None

    return True, "Face detected successfully.", boxes[0]


def register_face_embedding(person_id, face_crop_bgr):
    """Compute and store a person's embedding in ChromaDB."""
    embedding = get_embedding(face_crop_bgr)
    if embedding is None:
        return False, "Could not generate a facial embedding for this image."

    collection = get_chroma_collection()
    collection.add(
        ids=[person_id],
        embeddings=[embedding.tolist()],
        metadatas=[{"person_id": person_id}],
    )
    return True, "Embedding stored successfully."


def remove_face_embedding(person_id):
    """Delete a person's embedding from ChromaDB. Safe to call even if absent."""
    collection = get_chroma_collection()
    try:
        collection.delete(ids=[person_id])
    except Exception:
        pass


def match_embedding(embedding):
    """Query ChromaDB for the closest match. Returns (person_id, distance) or (None, None)."""
    collection = get_chroma_collection()
    try:
        if collection.count() == 0:
            return None, None
        result = collection.query(query_embeddings=[embedding.tolist()], n_results=1)
    except Exception:
        return None, None

    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]
    if not ids:
        return None, None
    return ids[0], distances[0]


def distance_to_similarity(distance):
    """Convert a ChromaDB distance into a 0-100% similarity figure for display.
    This is a linear heuristic (100% at distance 0, 0% at 2x the recognition
    threshold), not a calibrated probability - it exists purely to make the
    real distance value easier to read at a glance."""
    if distance is None:
        return 0.0
    span = RECOGNITION_THRESHOLD * 2
    similarity = (1 - (distance / span)) * 100
    return max(0.0, min(100.0, similarity))


def is_known(distance):
    return distance is not None and distance < RECOGNITION_THRESHOLD
