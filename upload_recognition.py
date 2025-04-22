import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def show_upload_app():
    
    model = load_model("model/digit_classifier.h5")    
    st.markdown(
        """
        <style>
        .streamlit-expanderHeader {
            font-size: 18px;
        }
        .css-1e2nbp9 {
            height: 150px;
            padding-top: 10px;
        }
        .css-1y4opfs {
            padding-top: 5px;
        }
        .css-1e2nbp9 .css-1q4w4cz {
            padding: 10px 20px;
        }
        .css-18e3th9 {
            display: flex;
            justify-content: space-between;
        }
        </style>
        """, unsafe_allow_html=True
    )

    
    uploaded_file = st.file_uploader("Upload image...", type=["png", "jpg", "jpeg"])

    def preprocess_for_segmentation(image):
        image = image.convert("L")
        img_array = np.array(image)
        img_array = cv2.bitwise_not(img_array)
        _, thresh = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def extract_digits(thresh_img):
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        digit_imgs = []
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 5 and h > 10:
                roi = thresh_img[y:y+h, x:x+w]
                resized = cv2.resize(roi, (18, 18), interpolation=cv2.INTER_AREA)
                padded = np.pad(resized, ((5,5),(5,5)), mode='constant', constant_values=0)
                digit_imgs.append(padded)
                boxes.append((x, roi))

        sorted_digits = [img for _, img in sorted(zip(boxes, digit_imgs), key=lambda b: b[0][0])]
        return sorted_digits

    def predict_digits(digit_imgs):
        results = []
        for digit in digit_imgs:
            digit = digit.astype("float32") / 255.0
            digit = digit.reshape(1, 28, 28, 1)
            pred = model.predict(digit)
            results.append((np.argmax(pred), pred[0]))
        return results

    def plot_combined_confidence(predictions):
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, (_, conf) in enumerate(predictions):
            ax.bar(np.arange(10) + i*11, conf, width=0.8, label=f'Digit {i}')
        ax.set_xticks([i*11 + 4.5 for i in range(len(predictions))])
        ax.set_xticklabels([f'D{i}' for i in range(len(predictions))])
        ax.set_ylabel('Confidence')
        ax.set_title('Confidence for Each Digit')
        st.pyplot(fig)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            with st.spinner("Segmenting and predicting..."):
                thresh = preprocess_for_segmentation(image)
                digit_imgs = extract_digits(thresh)

                if not digit_imgs:
                    st.error("No digits found. Please upload a clearer image.")
                else:
                    predictions = predict_digits(digit_imgs)
                    predicted_number = ''.join(str(d[0]) for d in predictions)
                    st.markdown(f"### Predicted Number: **{predicted_number}**")
                    plot_combined_confidence(predictions)


