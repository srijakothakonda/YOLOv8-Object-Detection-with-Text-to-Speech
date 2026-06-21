import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from gtts import gTTS
import tempfile
import os

st.set_page_config(page_title="YOLOv8 Object Detection + TTS", layout="wide")
st.title("YOLOv8 Object Detection with Text-to-Speech")
st.markdown("---")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()
st.success("YOLOv8 Nano model loaded!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with st.spinner("Running detection..."):
        results = model(image_bgr)

    output_img = cv2.cvtColor(image_bgr.copy(), cv2.COLOR_BGR2RGB)
    detected_labels = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[class_id]
            detected_labels.append(label)
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(output_img, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    with col2:
        st.subheader("Detected Objects")
        st.image(output_img, use_container_width=True)

    st.markdown("---")
    unique_labels = list(set(detected_labels))

    if unique_labels:
        st.subheader("Detection Results")
        cols = st.columns(min(len(unique_labels), 4))
        for i, lbl in enumerate(unique_labels):
            cols[i % 4].metric(label="Object", value=lbl, delta=f"{detected_labels.count(lbl)}x")

        speech_text = "Detected objects are: " + ", ".join(unique_labels)
        st.info(f"Speech: {speech_text}")

        with st.spinner("Generating speech..."):
            tts = gTTS(speech_text, lang="en")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                tts.save(f.name)
                with open(f.name, "rb") as audio_file:
                    audio_bytes = audio_file.read()
            os.unlink(f.name)

        st.audio(audio_bytes, format="audio/mp3")
    else:
        st.warning("No objects detected.")