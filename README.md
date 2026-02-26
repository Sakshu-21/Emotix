#  Real-Time Emotion Detector
A real-time facial emotion detection app using your webcam, with a live updating graph that tracks emotion changes over time.
## Features
- 🎥 Live webcam feed with face detection
- 😄 Detects 7 emotions — Happy, Sad, Angry, Disgust, Fear, Surprise, Neutral
- 🎨 Color-coded face rectangle based on dominant emotion
- 📊 Live emotion graph that updates in real-time (runs in separate thread)
- 📈 Mini emotion confidence bars displayed next to each detected face
- 🔲 Clean dark UI overlay
##  Built With
- Python
- OpenCV — webcam feed & face annotations
- FER (Facial Expression Recognition) with MTCNN
- Matplotlib — live animated emotion graph
- Threading — graph runs parallel to webcam
## How to Run
Install dependencies:
```bash
pip install opencv-python fer matplotlib
```

Run:
```bash
python emotion_detector.py
```
Press **Q** to quit.

## Made by
Sakshi — built as a real-time computer vision project 🎓
