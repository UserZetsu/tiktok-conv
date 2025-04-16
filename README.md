# MediaPipe Human Pose Detection with YOLO

This project uses YOLO and MediaPipe for accurate human pose detection.

## Requirements

- Python 3.6+
- OpenCV
- MediaPipe

Install the required packages:

```bash
pip install mediapipe opencv-python
```

## Available Scripts

### 1. YOLO-Based Pose Detection (`yolo_pose_detection.py`)

This script uses the powerful YOLO object detection model combined with MediaPipe:

```bash
python yolo_pose_detection.py
```

#### Features:
- Uses YOLOv4-tiny to accurately detect people in the frame
- Applies MediaPipe pose estimation on each detected person
- Shows confidence scores for each detection
- Uses non-maximum suppression to eliminate duplicate detections
- Falls back to full-frame pose detection if YOLO doesn't detect anyone
- Press `ESC` to exit the application

**Note:** The first time you run this script, it will automatically download the required YOLO model files.

### 2. Landmark Data Collection (`pose_landmarks_extraction.py`)

This script detects human poses and saves the landmark coordinates to a CSV file:

```bash
python pose_landmarks_extraction.py
```

- Data is saved to the `pose_data` directory with a timestamp in the filename
- Press `ESC` to stop recording

## Understanding MediaPipe Pose Landmarks

MediaPipe Pose provides 33 pose landmarks:

- 0-10: Face landmarks
- 11-22: Upper body landmarks (shoulders, elbows, wrists)
- 23-32: Lower body landmarks (hips, knees, ankles)

Each landmark contains:
- x, y: Normalized coordinates (0.0-1.0) 
- z: Relative depth where smaller values are closer to the camera
- visibility: Confidence score (0.0-1.0)

## Customization

You can adjust the MediaPipe pose model parameters:

```python
mp_pose.Pose(
    min_detection_confidence=0.5,  # Adjust this value (0.0-1.0)
    min_tracking_confidence=0.5,   # Adjust this value (0.0-1.0)
    model_complexity=1             # 0, 1, or 2 (higher is more accurate but slower)
)
```

For the YOLO-based detection, you can adjust:
- Confidence threshold for person detection (default 0.5)
- Non-maximum suppression threshold (default 0.4)
- YOLO model input size (default 416x416)
- Padding around detected persons (default 10% of box size)

## How It Works

The system uses a two-stage detection approach:

1. **Person Detection**: YOLO identifies people in the frame with bounding boxes
2. **Pose Estimation**: MediaPipe extracts detailed pose landmarks for each detected person
3. **Visualization**: Results are displayed with bounding boxes, confidence scores, and pose skeletons 