import re

import cv2
import numpy as np
import streamlit as st

from core.ui import inject_css, page_header
from core.engine import register_face_embedding, remove_face_embedding, validate_registration_image
from core.storage import delete_person, person_exists, save_person

inject_css()
page_header("Register Person", "Add a new person to the IdentiScan dataset.", icon="person_add")

VALID_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def read_image(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    return cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)


col_upload, col_preview = st.columns([1, 1])

with col_upload:
    st.markdown("**Step 1 - Upload a photo**")
    uploaded_file = st.file_uploader("JPG, JPEG or PNG, exactly one face", type=["jpg", "jpeg", "png"])

image_bgr = None
validation_ok = False
face_box = None

if uploaded_file is not None:
    image_bgr = read_image(uploaded_file)
    with col_preview:
        st.markdown("**Preview**")
        if image_bgr is None:
            st.error("The uploaded file could not be read as an image.")
        else:
            validation_ok, message, face_box = validate_registration_image(image_bgr)
            preview = image_bgr.copy()
            if face_box:
                left, top, right, bottom = face_box
                cv2.rectangle(preview, (left, top), (right, bottom), (166, 219, 84), 2)
            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), width="stretch")
            if validation_ok:
                st.success(message)
            else:
                st.error(message)

st.write("")
st.markdown("**Step 2 - Person information**")

with st.form("register_form", clear_on_submit=False):
    c1, c2 = st.columns(2)
    with c1:
        full_name = st.text_input("Full Name *")
        person_id = st.text_input("Person ID *", help="Unique identifier, letters/numbers/-/_ only")
        email = st.text_input("Email (optional)")
    with c2:
        role = st.text_input("Role / Designation (optional)")
        department = st.text_input("Department (optional)")
        phone = st.text_input("Phone (optional)")
    notes = st.text_area("Notes (optional)")

    submitted = st.form_submit_button("Register Person", width="stretch")

if submitted:
    if uploaded_file is None or image_bgr is None:
        st.error("Please upload a photo before registering.")
    elif not full_name.strip():
        st.error("Full Name is required.")
    elif not person_id.strip():
        st.error("Person ID is required.")
    elif not VALID_ID_RE.match(person_id.strip()):
        st.error("Person ID may only contain letters, numbers, hyphens and underscores.")
    elif person_exists(person_id.strip()):
        st.error(f"Person ID '{person_id.strip()}' is already registered. Please choose a different ID.")
    else:
        # Re-validate at submission time in case the preview state is stale.
        validation_ok, message, face_box = validate_registration_image(image_bgr)
        if not validation_ok:
            st.error(message)
        else:
            pid = person_id.strip()
            left, top, right, bottom = face_box
            face_crop = image_bgr[top:bottom, left:right]

            embed_ok, embed_message = register_face_embedding(pid, face_crop)
            if not embed_ok:
                st.error(embed_message)
            else:
                metadata = {
                    "full_name": full_name.strip(),
                    "email": email.strip(),
                    "phone": phone.strip(),
                    "role": role.strip(),
                    "department": department.strip(),
                    "notes": notes.strip(),
                }
                try:
                    save_person(pid, image_bgr, metadata)
                    st.success(f"{full_name.strip()} was registered successfully as '{pid}'.")
                    st.balloons()
                except Exception as exc:
                    # Roll back the embedding so we don't leave an orphaned vector.
                    remove_face_embedding(pid)
                    delete_person(pid)
                    st.error(f"Registration failed while saving data: {exc}")
