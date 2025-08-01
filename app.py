import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import requests
import tensorflow as tf

MODEL_PATH = "mask_model_int8.tflite"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Download model from GitHub release if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("📥 Downloading quantized model...")
        url = "https://github.com/Hariom-Nagar211/Face-Mask-Detection/releases/download/v2.0/mask_model_int8.tflite"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.success("✅ Model downloaded!")

download_model()

# Load TFLite model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load Haar Cascade
haar = cv2.CascadeClassifier(CASCADE_PATH)
if haar.empty():
    st.error("❌ Haar cascade loading failed.")
    st.stop()

# Predict mask status: 0 = Mask, 1 = No Mask
def detect_face_mask(img):
    try:
        img_resized = cv2.resize(img, (224, 224))
        img_input = np.expand_dims(img_resized.astype(np.uint8), axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0][0]

        return 0 if output <= 0.5 else 1
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return -1

# Draw label box
def draw_label(img, text, pos, bg_color):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, cv2.FILLED)[0]
    end_x = pos[0] + text_size[0] + 10
    end_y = pos[1] - text_size[1] - 10
    cv2.rectangle(img, pos, (end_x, pos[1]), bg_color, cv2.FILLED)
    cv2.putText(img, text, (pos[0] + 5, pos[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

# Detect face
def detect_face(img_gray):
    faces = haar.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    return faces

# Process image: detect face, predict mask, draw label
def process_image(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = detect_face(gray)

    if len(faces) == 0:
        st.warning("⚠️ No faces detected.")
        return img_bgr

    for (x, y, w, h) in faces:
        face_roi = img_bgr[y:y+h, x:x+w]
        if face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
            continue

        label = detect_face_mask(face_roi)
        if label == -1:
            continue

        color = (0, 255, 0) if label == 0 else (0, 0, 255)
        text = "Mask" if label == 0 else "No Mask"
        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), color, 2)
        draw_label(img_bgr, text, (x, y), color)

    return img_bgr

# Streamlit UI
st.set_page_config(page_title="Face Mask Detection")
st.title("😷 Face Mask Detection Web App")
st.markdown("Upload or capture an image to check for face mask compliance.")

uploaded = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])
camera_img = st.camera_input("📸 Or take a photo")

if uploaded or camera_img:
    image = Image.open(uploaded if uploaded else camera_img).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    st.write("🔍 Processing...")
    processed = process_image(img_bgr.copy())

    st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
             caption="📸 Detection Result", use_container_width=True)
