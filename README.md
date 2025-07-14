# 😷 Face Mask Detection Web App

A real-time **face mask detection system** built using **TensorFlow**, **OpenCV**, and **Streamlit**.  
It detects human faces in uploaded or webcam images and predicts whether each face is wearing a mask.

---

## 📸 Features

- ✅ Upload an image or capture one from webcam
- 🧠 Predicts whether each detected face has a mask or not
- 🎯 Color-coded bounding boxes:
  - 🟢 Green = Mask
  - 🔴 Red = No Mask
- 🖼️ Label text with matching background and text color
- 🌐 Clean web interface powered by Streamlit

---

## 🧠 Model Details

- **Architecture:** VGG16 (pre-trained base) + custom classifier
- **Input Size:** 224 × 224 × 3
- **Output:** Binary classification (Mask / No Mask)
- **Framework:** TensorFlow / Keras
- **File:** `mask_model.h5` (included in the project folder)

---

## 🗂 Project Structure

```text
.
├── app.py                             # Main Streamlit app
├── mask_model.h5                      # Trained Keras model
├── haarcascade_frontalface_default.xml # Haar cascade file for face detection
├── requirements.txt                   # All required libraries
└── README.md                          # This file
```

---

## ⚙️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/face-mask-detection-app.git
cd face-mask-detection-app
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit opencv-python tensorflow pillow numpy
```

### 3. Run the App

```bash
streamlit run app.py
```

---

## 🧩 Dependencies

- `streamlit`
- `tensorflow`
- `numpy`

Ensure you also have the following files in the root directory:

- `mask_model.h5` – your trained Keras model
- `haarcascade_frontalface_default.xml` – Haar cascade file  

---

## 🚀 Future Improvements

- Add real-time webcam support (OpenCV-based)
- Show confidence scores (e.g. 92% Mask)
- Deploy online via Streamlit Cloud or HuggingFace Spaces
- Allow download of annotated image

---

## 🙌 Credits

- **OpenCV** – for real-time face detection  
- **TensorFlow/Keras** – for building and training the mask classifier  
- **Streamlit** – for the interactive web UI  
- Inspired by real-world safety use-cases in the COVID era

---

> Made with ❤️ by Hariom Nagar
