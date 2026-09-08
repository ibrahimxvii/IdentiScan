import time

import av
import cv2
import streamlit as st
import streamlit_webrtc as webrtc

from core.ui import inject_css, page_header
from core.engine import (
    RECOGNITION_THRESHOLD,
    detect_faces,
    distance_to_similarity,
    get_embedding,
    is_known,
    load_detector,
    load_embedder,
    get_chroma_collection,
    match_embedding,
)
from core.storage import load_person, log_recognition

inject_css()
page_header("Live Recognition", "Real-time face detection and identity matching from your webcam.", icon="videocam")

# Pre-warm models/db on the main thread before the background video thread starts.
try:
    load_detector()
    load_embedder()
    get_chroma_collection()
except Exception as exc:
    st.error(f"Could not initialize the recognition engine: {exc}")
    st.stop()

RECOGNITION_INTERVAL = 10  # process every Nth frame (keeps CPU-only usage smooth)
LOG_DEBOUNCE_SECONDS = 5   # avoid flooding the history log with duplicate events


class RecognitionProcessor:
    def __init__(self):
        self.frame_counter = 0
        self.last_faces = []  # (box, name, person_id, distance, similarity, known)
        self._last_logged_key = None
        self._last_logged_time = 0.0

    def _maybe_log(self, person_id, name, distance, similarity, known):
        key = person_id if known else "unknown"
        now = time.time()
        if key != self._last_logged_key or (now - self._last_logged_time) > LOG_DEBOUNCE_SECONDS:
            log_recognition(
                person_id if known else None,
                name,
                distance,
                similarity,
                "known" if known else "unknown",
            )
            self._last_logged_key = key
            self._last_logged_time = now

    def process_frame(self, image_bgr):
        results = []
        try:
            boxes = detect_faces(image_bgr)
        except Exception:
            return results

        for (left, top, right, bottom) in boxes:
            face_crop = image_bgr[top:bottom, left:right]
            if face_crop.size == 0:
                continue

            try:
                embedding = get_embedding(face_crop)
            except Exception:
                embedding = None
            if embedding is None:
                continue

            person_id, distance = match_embedding(embedding)
            known = is_known(distance)
            similarity = distance_to_similarity(distance) if distance is not None else None

            name = "Unknown"
            if known and person_id:
                metadata = load_person(person_id)
                name = metadata.get("full_name", person_id) if metadata else person_id

            self._maybe_log(person_id, name, distance, similarity, known)
            results.append(((left, top, right, bottom), name, person_id if known else None, distance, similarity, known))
        return results

    @staticmethod
    def draw(image_bgr, faces):
        for (left, top, right, bottom), name, person_id, distance, similarity, known in faces:
            color = (166, 219, 84) if known else (110, 110, 240)  # BGR: green-ish / red-ish
            cv2.rectangle(image_bgr, (left, top), (right, bottom), color, 2)
            label = f"{name} ({similarity:.0f}%)" if known and similarity is not None else "Unknown"
            cv2.putText(
                image_bgr, label, (left + 6, max(15, bottom - 6)),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1,
            )
        return image_bgr

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image_bgr = frame.to_ndarray(format="bgr24")
        if self.frame_counter % RECOGNITION_INTERVAL == 0:
            try:
                self.last_faces = self.process_frame(image_bgr)
            except Exception:
                self.last_faces = []
        self.frame_counter += 1
        image_bgr = self.draw(image_bgr, self.last_faces)
        return av.VideoFrame.from_ndarray(image_bgr, format="bgr24")


col_video, col_info = st.columns([2, 1])

with col_video:
    ctx = webrtc.webrtc_streamer(
        key="face-recognition",
        video_processor_factory=RecognitionProcessor,
        media_stream_constraints={
            # Cap resolution: smaller frames mean faster detection + embedding
            # per recognized frame, which is what keeps this smooth on CPU.
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
            "audio": False,
        },
        async_processing=True,
    )
    st.caption(":material/videocam: If the camera does not start, check your browser's camera permissions.")

with col_info:
    st.markdown(
        '<div class="fg-card"><span class="material-symbols-outlined fg-header-icon">rule</span>'
        f'<b>Recognition threshold</b><br>'
        f'Distance below <code>{RECOGNITION_THRESHOLD}</code> is a known match.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    info_placeholder = st.empty()

    if ctx.state.playing:
        while ctx.state.playing:
            faces = ctx.video_processor.last_faces if ctx.video_processor else []
            with info_placeholder.container():
                st.markdown("**:material/visibility: Currently detected**")
                if not faces:
                    st.caption("No face currently in view.")
                for (_, name, person_id, distance, similarity, known) in faces:
                    st.markdown('<div class="fg-card">', unsafe_allow_html=True)
                    if known:
                        metadata = load_person(person_id) or {}
                        st.success(f":material/check_circle: **{name}**  \nSimilarity: {similarity:.1f}%")
                        if metadata.get("role"):
                            st.write(f"Role: {metadata['role']}")
                        if metadata.get("department"):
                            st.write(f"Department: {metadata['department']}")
                        if metadata.get("email"):
                            st.write(f"Email: {metadata['email']}")
                    else:
                        st.error(":material/help: **Unknown**  \nUnrecognized face")
                    st.markdown("</div>", unsafe_allow_html=True)
            time.sleep(0.5)
    else:
        with info_placeholder.container():
            st.caption(":material/play_circle: Start the camera above to see recognition results here.")
