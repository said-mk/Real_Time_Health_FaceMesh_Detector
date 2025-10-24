# Real-Time Health FaceMesh Detector

A Python-based system that tracks **468 facial landmarks** in real time using **MediaPipe** and **OpenCV**.  
It monitors live facial geometry, estimates basic emotional states (neutral, fatigue, confusion).
And allows annotated screenshots to be saved with a single keypress.


## Features

- Real-time facial landmark detection (468 points)
- Live FPS (frames per second) overlay
- Emotion estimation (Neutral, Fatigue, Confusion)
- Save annotated screenshots by pressing `s`
- Exit the live stream by pressing `q`
- Modular architecture (easily extendable to more health or emotion metrics)

## Tech Stack

- **Python 3.11.4**
- **OpenCV** for image processing
- **MediaPipe** for facial landmark extraction
- **NumPy** for numerical operations


- **PyTest** for testing
