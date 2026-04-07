import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("mask_detector.keras")

labels = ["Mask", "No Mask"]

st.title("😷 Face Mask Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100)) / 255.0
        face = np.reshape(face, (1, 100, 100, 3))

        prediction = model.predict(face, verbose=0)
        label = labels[np.argmax(prediction)]

        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    st.image(img, caption="Result", use_column_width=True)
