import streamlit as st

from core.ui import inject_css, page_header, metric_card, status_pill, empty_state
from core.storage import count_people, history_stats, read_history
from core.engine import get_engine_status

inject_css()
page_header("IdentiScan", "Intelligent Face Recognition & Identity Management System", icon="space_dashboard")

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

stats = history_stats()
cols = st.columns(4)
with cols[0]:
    metric_card("Registered People", count_people())
with cols[1]:
    metric_card("Recognitions", stats["known"], "Known-face matches logged")
with cols[2]:
    metric_card("Unknown Detections", stats["unknown"])
with cols[3]:
    metric_card("Total Events", stats["total"])

st.write("")
st.subheader("Recent Activity")
history = read_history(limit=10)
if not history:
    empty_state(
        "No recognition activity yet",
        "Register a person and start Live Recognition to see events here.",
    )
else:
    st.dataframe(history, width="stretch", hide_index=True)

st.write("")
st.caption("Use the sidebar to register people, run live recognition, and manage your dataset.")
