# Driver-Drowsiness-Computer-Vision

# Driver Drowsiness Detection

Driver Drowsiness Detection is a real-time project built using **Dlib** and **OpenCV** with **Python** as the backend. It detects whether a driver is **active**, **drowsy**, or **sleepy** using facial landmarks and eye aspect ratio analysis.

[![Python](https://img.shields.io/badge/Python-3.6%2B-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-brightgreen?logo=opencv)](https://opencv.org/)
[![Dlib](https://img.shields.io/badge/Dlib-ML--based%20Landmarks-purple)](http://dlib.net/)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🔧 How it Works

- The project uses **Dlib’s 68 facial landmark detector** and its face detection algorithm.
- A specific calculation based on the **Eye Aspect Ratio (EAR)** is used:
  
  > EAR = (Vertical Eye Distances) / (Horizontal Eye Distance)

- Based on this EAR value, thresholds are set to detect:
  - **Active**: Eyes open
  - **Drowsy**: Eyes partially closed

---

## 📸 Demo Screenshots

Facial landmark detection in action:

### Detection States:

| Active | Drowsy |
|--------|--------|
| <img alt="Screenshot 2025-04-07 at 2 13 16 PM" src="https://github.com/user-attachments/assets/745736d1-fa01-404e-971d-44ba33da8daf" height="200"> | <img alt="Screenshot 2025-04-07 at 2 13 26 PM" src="https://github.com/user-attachments/assets/af25201a-0685-40fa-b057-fb6e00704009" height="200"> 

---

## Download Model File
Download the facial landmark predictor from dlib’s website, extract it, and place shape_predictor_68_face_landmarks.dat in the project root.

## Run the Code

python drowsiness_detector.py
Press ctrl + c to exit the video stream.


## 🙌 Credits

Dlib and OpenCV community
EAR concept by Soukupová & Čech
Original implementation reference by @infoaryan

