import streamlit as st

from core.ui import inject_css, page_header, metric_card, empty_state
from core.storage import clear_history, history_stats, read_history

inject_css()
page_header("Recognition History", "A log of recognition events from Live Recognition.", icon="history")

stats = history_stats()
cols = st.columns(3)
with cols[0]:
    metric_card("Total Events", stats["total"])
with cols[1]:
    metric_card("Known Matches", stats["known"])
with cols[2]:
    metric_card("Unknown Detections", stats["unknown"])

st.write("")

history = read_history()
if not history:
    empty_state("No recognition events yet", "Events will appear here once you run Live Recognition.")
else:
    st.dataframe(history, width="stretch", hide_index=True)

    with st.expander("Clear history"):
        st.caption("This permanently deletes the recognition event log. Registered people are not affected.")
        if st.button("Clear all history"):
            clear_history()
            st.rerun()
