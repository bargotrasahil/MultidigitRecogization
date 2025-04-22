import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

@st.cache_resource
def load_digit_model():
    return load_model("model/mnist_model.h5")

model = load_digit_model()

def show_camera_app():
    st.markdown("Show multiple digits to the webcam and let the model predict each!")

    
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False

    
    if st.button("Start Camera" if not st.session_state.camera_running else "Stop Camera"):
        st.session_state.camera_running = not st.session_state.camera_running

    FRAME_WINDOW = st.image([])

    if st.session_state.camera_running:
        cap = cv2.VideoCapture(0)

        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                st.error("Couldn't access the camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  # sort left-to-right

            predicted_digits = []

            for cnt in contours:
                if 1000 < cv2.contourArea(cnt) < 10000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = w / float(h)
                    if 0.2 < aspect_ratio < 1.2:
                        digit = thresh[y:y+h, x:x+w]
                        digit = cv2.resize(digit, (18, 18))
                        digit = np.pad(digit, ((5, 5), (5, 5)), mode='constant', constant_values=0)
                        digit = digit.reshape(1, 28, 28, 1).astype('float32') / 255.0

                        prediction = model.predict(digit)
                        pred_class = np.argmax(prediction)
                        confidence = np.max(prediction)
                        predicted_digits.append(str(pred_class))

                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        label = f"{pred_class} ({confidence*100:.1f}%)"
                        cv2.putText(frame, label, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        top_preds = prediction[0].argsort()[-3:][::-1]
                        for i, p in enumerate(top_preds):
                            prob = prediction[0][p]
                            text = f"{p}: {prob*100:.1f}%"
                            cv2.putText(frame, text, (x, y + h + 20 + i * 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            if predicted_digits:
                number_str = ''.join(predicted_digits)
                text_size = cv2.getTextSize(number_str, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                frame_center = (frame.shape[1] - text_size[0]) // 2
                cv2.putText(frame, number_str, (frame_center, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)

        cap.release()

show_camera_app()
