import streamlit as st

from core.ui import inject_css, page_header, status_pill
from core.engine import RECOGNITION_THRESHOLD, get_engine_status
from core.storage import count_people

inject_css()
page_header("AI Model & System Information", "How IdentiScan detects and recognizes faces.", icon="info")

status = get_engine_status()
st.markdown(
    status_pill("Face Detector", status["detector"])
    + status_pill("FaceNet Embedder", status["embedder"])
    + status_pill("Vector Database", status["database"]),
    unsafe_allow_html=True,
)
if status["error"]:
    st.warning(f"System initialization issue: {status['error']}")

st.write("")

st.markdown('<div class="fg-card">', unsafe_allow_html=True)
st.subheader("Pipeline")
st.markdown(
    """
1. **Face Detection** &rarr; MediaPipe (BlazeFace short-range model)
2. **Feature Extraction** &rarr; FaceNet (`facenet-pytorch`, InceptionResnetV1 pretrained on VGGFace2), a pretrained model that turns a face crop into a 512-dimension embedding
3. **Vector Database** &rarr; ChromaDB, storing one embedding per registered person
4. **Recognition** &rarr; the incoming face's embedding is compared against every stored embedding via similarity search; the closest match is accepted only if its distance is below the recognition threshold
    """
)
st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.markdown('<div class="fg-card">', unsafe_allow_html=True)
st.subheader("Pretrained model vs. your dataset")
st.markdown(
    """
IdentiScan does **not** train FaceNet from scratch. FaceNet is a pretrained
model used only for **feature extraction** - turning a face image into a
numeric embedding.

What *is* custom is the **dataset you build**: every person you register
through the Register Person page contributes their own image, metadata, and
embedding. No retraining step is required - a newly registered person can be
recognized immediately because their embedding is added straight to the
vector database.
    """
)
st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.markdown('<div class="fg-card">', unsafe_allow_html=True)
st.subheader("Configuration")
st.write(f"Recognition threshold (distance): `{RECOGNITION_THRESHOLD}`")
st.write(f"Registered people: `{count_people()}`")
st.caption("The threshold is defined in one place - `core/engine.py` - and used by both Live Recognition and Register Person.")
st.markdown("</div>", unsafe_allow_html=True)