import streamlit as st

st.set_page_config(page_title="IdentiScan", layout="wide")

pages = [
    st.Page("pages/0_Dashboard.py", title="Dashboard", icon=":material/space_dashboard:", default=True),
    st.Page("pages/1_Live_Recognition.py", title="Live Recognition", icon=":material/videocam:"),
    st.Page("pages/2_Register_Person.py", title="Register Person", icon=":material/person_add:"),
    st.Page("pages/3_People.py", title="People", icon=":material/group:"),
    st.Page("pages/4_History.py", title="History", icon=":material/history:"),
    st.Page("pages/5_System_Info.py", title="System Info", icon=":material/info:"),
]

nav = st.navigation(pages)
nav.run()
