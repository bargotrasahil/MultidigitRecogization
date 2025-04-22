import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(
    page_title="Multi-Digit Recognition ",
    layout="wide"
)

from canvas_recognition import show_canvas_app
from camera_recognition import show_camera_app
from upload_recognition import show_upload_app
from sidebar import show_sidebar

mode = st.sidebar.selectbox("Choose Mode", ["Canvas Input", "Camera Input", "By uploading"])

show_sidebar(mode)

if mode == "Canvas Input":
    show_canvas_app()
elif mode == "Camera Input":
    show_camera_app()
else:
    show_upload_app()
