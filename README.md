markdown
# Driver Drowsiness Detection System

A real-time driver drowsiness detection system using **OpenCV** and **MediaPipe Face Mesh**. It monitors eye closure using the Eye Aspect Ratio (EAR) and alerts the driver when drowsiness is detected.

## Features

- Real-time eye tracking with MediaPipe Face Mesh
- Eye Aspect Ratio (EAR) based drowsiness detection
- Visual alert on screen
- Audio alarm (beep sound on Windows)
- FPS display and status indicator
- Works with webcam

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy

### Installation

```bash
pip install opencv-python mediapipe numpy


## Usage

```bash
python main.py
```

Press `q` to quit.

## Customization

You can adjust these parameters in the code:

```python
EYE_AR_THRESH = 0.21          # Eye closure threshold
EYE_AR_CONSEC_FRAMES = 20     # Consecutive frames for alert
```

## How It Works

The system calculates the Eye Aspect Ratio (EAR) using 6 landmarks per eye. If EAR stays below the threshold for consecutive frames, it triggers a drowsiness alert.
