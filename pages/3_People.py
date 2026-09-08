import os

import streamlit as st

from core.ui import inject_css, page_header, empty_state
from core.engine import remove_face_embedding
from core.storage import delete_person, list_people

inject_css()
page_header("People", "Everyone currently registered in the IdentiScan dataset.", icon="group")

people = list_people()

if not people:
    empty_state("No people registered yet", "Go to Register Person to add your first entry.")
    st.stop()

if "pending_delete" not in st.session_state:
    st.session_state.pending_delete = None

st.caption(f"{len(people)} registered {'person' if len(people) == 1 else 'people'}")
st.write("")

COLUMNS = 3
rows = [people[i:i + COLUMNS] for i in range(0, len(people), COLUMNS)]

for row in rows:
    cols = st.columns(COLUMNS)
    for col, person in zip(cols, row):
        with col:
            st.markdown('<div class="fg-person-card">', unsafe_allow_html=True)
            image_path = person.get("image_path")
            if image_path and os.path.exists(image_path):
                st.image(image_path, width="stretch")
            st.markdown(f'<div class="fg-person-name">{person.get("full_name", "Unnamed")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="fg-person-id">ID: {person.get("person_id", "")}</div>', unsafe_allow_html=True)

            meta_lines = []
            if person.get("role"):
                meta_lines.append(f"Role: {person['role']}")
            if person.get("department"):
                meta_lines.append(f"Department: {person['department']}")
            if person.get("registered_at"):
                meta_lines.append(f"Registered: {person['registered_at'][:10]}")
            if meta_lines:
                st.markdown(f'<div class="fg-person-meta">{"<br>".join(meta_lines)}</div>', unsafe_allow_html=True)

            with st.expander("Details"):
                if person.get("email"):
                    st.write(f"Email: {person['email']}")
                if person.get("phone"):
                    st.write(f"Phone: {person['phone']}")
                if person.get("notes"):
                    st.write(f"Notes: {person['notes']}")
                if not any([person.get("email"), person.get("phone"), person.get("notes")]):
                    st.caption("No additional details provided.")

            pid = person.get("person_id")
            if st.session_state.pending_delete == pid:
                st.warning(f"Delete {person.get('full_name', pid)}? This cannot be undone.")
                cc1, cc2 = st.columns(2)
                with cc1:
                    if st.button("Confirm", key=f"confirm_{pid}", width="stretch"):
                        remove_face_embedding(pid)
                        delete_person(pid)
                        st.session_state.pending_delete = None
                        st.rerun()
                with cc2:
                    if st.button("Cancel", key=f"cancel_{pid}", width="stretch"):
                        st.session_state.pending_delete = None
                        st.rerun()
            else:
                if st.button("Delete", key=f"delete_{pid}", width="stretch"):
                    st.session_state.pending_delete = pid
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)
