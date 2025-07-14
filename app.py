import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("mask_model.h5")

# Load Haar Cascade
haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if haar.empty():
    st.error("❌ Failed to load Haar cascade file.")
    st.stop()

# Predict mask status: 0 = Mask, 1 = No Mask
def detect_face_mask(img):
    try:
        img_resized = cv2.resize(img, (224, 224))
        img_norm = img_resized.astype("float32") / 255.0
        y_pred = model.predict(img_norm.reshape(1, 224, 224, 3))[0][0]
        return 0 if y_pred <= 0.5 else 1
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

# Your original face detection function
def detect_face(img_gray):
    faces = haar.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces

# Process image: detect face, predict mask, draw label
def process_image(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = detect_face(gray)

    if len(faces) == 0:
        st.warning("⚠️ No faces detected.")
        return img_bgr

    for (x, y, w, h) in faces:
        x, y = max(0, x), max(0, y)
        w = min(w, img_bgr.shape[1] - x)
        h = min(h, img_bgr.shape[0] - y)

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
