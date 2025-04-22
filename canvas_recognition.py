import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
import random

@st.cache_resource
def load_digit_model():
    return load_model("model/mnist.h5")

model = load_digit_model()

def show_canvas_app():
    st.markdown("""
        <style>
            ::-webkit-scrollbar { display: none; }
            .block-container { padding: 2rem 3rem; }
            .stButton > button {
                border-radius: 10px;
                height: 2.8em;
                font-weight: bold;
                background-color: #7289DA;
                color: white;
                border: none;
                margin-bottom: 10px;
            }
            .stButton > button:hover {
                background-color: #5b6eae;
            }
            canvas {
                border: 2px solid #7289DA;
                border-radius: 10px;
            }
            .toolbar, .upper-canvas ~ div {
                display: none !important;
            }
        </style>
    """, unsafe_allow_html=True)

    
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = "canvas1"
    if "selected_index" not in st.session_state:
        st.session_state.selected_index = None
    if "predictions" not in st.session_state:
        st.session_state.predictions = []

    
    def random_pastel():
        h = random.random()
        s = 0.5 + random.random() * 0.5
        v = 0.7 + random.random() * 0.3
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.3)"

    
    def smooth_line(x, y, num=300):
        spline = make_interp_spline(x, y)
        x_smooth = np.linspace(min(x), max(x), num)
        y_smooth = spline(x_smooth)
        return x_smooth, y_smooth

    
    def predict_digit(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        sorted_contours = sorted(zip(contours, bounding_boxes), key=lambda b: b[1][0])
        predictions = []
        for cnt, (x, y, w, h) in sorted_contours:
            if w < 10 or h < 10:
                continue
            digit = th[y:y+h, x:x+w]
            resized_digit = cv2.resize(digit, (18, 18))
            padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
            input_digit = padded_digit.reshape(1, 28, 28, 1).astype("float32") / 255.0
            pred = model.predict(input_digit, verbose=0)[0]
            predictions.append((np.argmax(pred), pred))
        return predictions

    
    left_col, right_col = st.columns([3, 2], gap="medium")

    with left_col:
        st.markdown("## Multi-Digit Recognition")

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=14,
            stroke_color="#000000",
            background_color="#FFFFFF",
            width=500,
            height=450,
            drawing_mode="freedraw",
            key=st.session_state.canvas_key,
            display_toolbar=False,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            predict_button = st.button("Predict", use_container_width=True)
        with col2:
            if st.button("Clear", use_container_width=True):
                st.session_state.canvas_key = "canvas" + str(np.random.randint(10000))
                st.session_state.predictions = []
                st.session_state.selected_index = None
                st.rerun()
        with col3:
            if canvas_result.image_data is not None:
                img = Image.fromarray(canvas_result.image_data.astype("uint8"))
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="PNG")
                st.download_button("Download", data=img_byte_arr.getvalue(), file_name="drawing.png", mime="image/png", use_container_width=True)

    with right_col:
        st.markdown("##")

        if predict_button:
            if canvas_result.image_data is not None:
                img = canvas_result.image_data.astype("uint8")
                st.session_state.predictions = predict_digit(img)
                st.session_state.selected_index = None
            else:
                st.warning("Please draw something first..")

        if st.session_state.predictions:
            full_number = ''.join(str(d) for d, _ in st.session_state.predictions)
            st.markdown(f"### Predicted Number: `{full_number}`")

            dropdown_options = ["All Digits"] + [
                f"Digit {i+1}: {digit} ({int(np.max(probs)*100)}%)"
                for i, (digit, probs) in enumerate(st.session_state.predictions)
            ]
            selected_option = st.selectbox("Select Digit View", options=dropdown_options)

            fig = go.Figure()
            x_raw = list(range(10))

            if selected_option == "All Digits":
                for i, (digit, probs) in enumerate(st.session_state.predictions):
                    color = random_pastel()
                    x_smooth, y_smooth = smooth_line(x_raw, probs)
                    fig.add_trace(go.Scatter(
                        x=x_smooth,
                        y=y_smooth,
                        name=f"Digit {i+1}: {digit}",
                        mode="lines",
                        line=dict(color=color, width=3),
                        fill='tozeroy',
                        fillcolor=color
                    ))
            else:
                selected_index = dropdown_options.index(selected_option) - 1
                sel_digit, sel_probs = st.session_state.predictions[selected_index]
                color = random_pastel()
                x_smooth, y_smooth = smooth_line(x_raw, sel_probs)
                fig.add_trace(go.Scatter(
                    x=x_smooth,
                    y=y_smooth,
                    name=f"Digit {selected_index+1}: {sel_digit}",
                    mode="lines",
                    line=dict(color=color, width=4),
                    fill='tozeroy',
                    fillcolor=color
                ))
                st.markdown(f"### Selected Digit: `{sel_digit}`")

            fig.update_layout(
                title="Digit Confidence",
                xaxis=dict(
                    title="Digit",
                    tickmode='linear',
                    tick0=0,
                    dtick=1,
                    range=[-0.5, 9.5],
                    showgrid=False
                ),
                yaxis=dict(
                    title="Confidence",
                    range=[0, 1],
                    tickformat=".0%",
                    showgrid=False
                ),
                height=340,
                margin=dict(l=40, r=35, t=40, b=30),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=False)


