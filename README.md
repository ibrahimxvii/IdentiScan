# IdentiScan

**Intelligent Face Recognition & Identity Management System**

A local, real-time facial recognition dashboard built with Streamlit. Register
people through a simple upload form, then recognize them live through your
webcam - no manual retraining step required.

## Features

- **Register Person** - upload a photo, enter metadata (name, ID, email, role, department, notes), and the app validates the face, generates a FaceNet embedding, and stores everything persistently.
- **Live Recognition** - real-time webcam face detection with bounding boxes, recognized names, and similarity scores; unknown faces are clearly labeled.
- **People** - browse, inspect, and delete registered people; deleting a person also removes their stored embedding.
- **History** - a lightweight log of recognition events (known and unknown).
- **System Info** - explains the detection/embedding/vector-search pipeline and distinguishes the pretrained model from your custom dataset.

## Architecture

```
Webcam / Upload
      |
MediaPipe (Face Detection)
      |
FaceNet (Embedding Generation)
      |
ChromaDB (Vector Similarity Search)
      |
Threshold Decision -> Known / Unknown
      |
Metadata Display
```

- **Face Detection:** MediaPipe BlazeFace (short-range)
- **Feature Extraction:** FaceNet (`facenet-pytorch`, InceptionResnetV1 pretrained on VGGFace2), pretrained - used only to generate embeddings, not retrained
- **Vector Database:** ChromaDB (persistent, local)
- **UI:** Streamlit + `streamlit-webrtc` for the live camera feed

## Folder Structure

```
app.py                     Dashboard (entry point)
pages/
  1_Live_Recognition.py    Real-time webcam recognition
  2_Register_Person.py     Upload + register a new person
  3_People.py               Browse / delete registered people
  4_History.py              Recognition event log
  5_System_Info.py          Pipeline & model explanation
core/
  engine.py                 Detection, embedding, ChromaDB matching, threshold config
  storage.py                 Persistent storage for people + history
  ui.py                      Shared glassmorphism UI helpers
assets/style.css             Dashboard theme
utils.py                     Downloads the MediaPipe detector model on first run
data/                        Created at runtime: faces/<person_id>/, chroma/, history.csv
```

## Requirements

- **Python 3.10+ (including 3.13/3.14).** The embedding model runs on
  PyTorch (`facenet-pytorch`), which publishes wheels for current Python
  releases, so no specific Python version is required.

## Installation

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The MediaPipe face detector model is downloaded automatically on first run.
The first run also downloads the pretrained FaceNet weights (via
`facenet-pytorch`) - this requires an internet connection once.

## Running

```bash
streamlit run app.py
```

Streamlit will show all pages (Live Recognition, Register Person, People,
History, System Info) in the sidebar automatically.

## Usage

1. **Register a person:** open *Register Person*, upload a clear photo containing exactly one face, fill in Full Name and Person ID (required), then submit.
2. **Recognize live:** open *Live Recognition* and start the camera. Recognized people are shown with a name and similarity score; unrecognized faces are labeled "Unknown".
3. **Manage people:** open *People* to view or delete anyone in the dataset. Deleting removes both their stored image/metadata and their embedding.

## Model Notes

FaceNet is a pretrained embedding model - it is not trained or fine-tuned by
this project. The "dataset" is entirely made of the people you register:
their image, metadata, and embedding are what the recognition threshold
compares new faces against.

## Limitations

- Recognition quality depends on lighting, angle, and image quality at registration time.
- The recognition threshold is a fixed distance value (configurable in `core/engine.py`), not a calibrated confidence score.
- Designed for local, single-user use on CPU - not intended for production security deployments.
