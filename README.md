# 😷 Face Mask Detection using CNN

## 📌 Project Overview

This project detects whether a person is wearing a face mask or not using a Convolutional Neural Network (CNN) and OpenCV.

## 🎯 Features

* Detects mask and no-mask faces
* Real-time webcam detection
* Image-based prediction
* Uses deep learning (CNN)

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy

## 📊 Model Details

* CNN with Conv2D, MaxPooling, Dense layers
* Input size: 100x100 images
* Output classes: Mask / No Mask

## ⚠️ Limitations

* Trained for only 1 epoch
* No validation dataset
* Needs better generalization

## 🚀 Future Improvements

* Increase epochs (20–30)
* Add data augmentation
* Use transfer learning (MobileNetV2)
* Deploy using Streamlit

## ▶️ How to Run

### Install dependencies

pip install -r requirements.txt

### Run main file

python main.py

### Run web app

streamlit run app.py

## 👨‍💻 Author

Rakesh V
