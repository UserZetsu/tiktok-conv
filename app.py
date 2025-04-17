import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import time
from collections import deque
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import savgol_filter
import base64
from io import BytesIO
from PIL import Image
import torch
from ultralytics import YOLO
import os
import pandas as pd

# Page config
st.set_page_config(
    page_title="Rat Dance Detector",
    page_icon="🕺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Modern Dark Theme */
    .main {
        padding: 2rem;
        background-color: #0e1117;
        color: #f0f2f6;
    }
    
    /* Glass Morphism Cards */
    .glass-card {
        background: rgba(38, 39, 48, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.45);
    }
    
    /* Video Container */
    .video-container {
        position: relative;
        border-radius: 12px;
        overflow: hidden;
        background: rgba(0, 0, 0, 0.2);
        margin: 1rem 0;
    }
    
    .video-container img {
        width: 100%;
        height: auto;
        display: block;
    }
    
    /* Metrics Grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-box {
        background: rgba(46, 49, 56, 0.7);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        transform: translateY(-3px);
        background: rgba(56, 59, 66, 0.8);
    }
    
    /* Movement Graph */
    .movement-graph {
        background: rgba(46, 49, 56, 0.7);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Session Info */
    .session-info {
        background: rgba(46, 49, 56, 0.7);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .info-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .info-item:last-child {
        border-bottom: none;
    }
    
    .info-label {
        color: #aaa;
        font-size: 0.9rem;
    }
    
    .info-value {
        font-weight: 500;
        color: #f0f2f6;
    }
    
    /* Status Badge */
    .badge {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        box-shadow: 0 2px 10px rgba(76, 175, 80, 0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.4);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(76, 175, 80, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(76, 175, 80, 0);
        }
    }
    
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background: rgba(38, 39, 48, 0.95);
        color: #fff;
        text-align: center;
        border-radius: 8px;
        padding: 0.75rem;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        font-size: 0.9rem;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Headers */
    .header-text {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 10px rgba(76, 175, 80, 0.3);
    }
    
    .subheader-text {
        font-size: 1.2rem;
        color: #aaa;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* Controls */
    .controls-container {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
        padding: 1rem;
        background: rgba(46, 49, 56, 0.7);
        border-radius: 12px;
    }
    
    .control-label {
        font-size: 0.9rem;
        color: #aaa;
        margin-bottom: 0.5rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .glass-card {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(46, 49, 56, 0.7);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4CAF50;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = deque(maxlen=30)
if 'prob_history' not in st.session_state:
    st.session_state.prob_history = deque(maxlen=30)
if 'timestamp_history' not in st.session_state:
    st.session_state.timestamp_history = deque(maxlen=30)
if 'top_poses' not in st.session_state:
    st.session_state.top_poses = []
if 'top_pose_images' not in st.session_state:
    st.session_state.top_pose_images = deque(maxlen=3)
if 'detected_people' not in st.session_state:
    st.session_state.detected_people = []
if 'yolo_model' not in st.session_state:
    st.session_state.yolo_model = None
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'recorded_frames' not in st.session_state:
    st.session_state.recorded_frames = []
if 'show_heatmap' not in st.session_state:
    st.session_state.show_heatmap = False
if 'heatmap_data' not in st.session_state:
    st.session_state.heatmap_data = np.zeros((100, 100))
if 'detection_mode' not in st.session_state:
    st.session_state.detection_mode = "Standard"
if 'movement_history' not in st.session_state:
    st.session_state.movement_history = []
if 'detection_metrics' not in st.session_state:
    st.session_state.detection_metrics = {
        'total_detections': 0,
        'avg_confidence': 0,
        'peak_confidence': 0,
        'session_duration': 0
    }
if 'session_start_time' not in st.session_state:
    st.session_state.session_start_time = time.time()
if 'confidence_threshold' not in st.session_state:
    st.session_state.confidence_threshold = 0.65  # Lowered from 0.75

# Load models
@st.cache_resource
def load_models():
    model = load_model('models/pklfiles/nn_model.h5')
    scaler = joblib.load('models/pklfiles/scaler.pkl')
    return model, scaler

model, scaler = load_models()

# Load YOLO model
@st.cache_resource
def load_yolo_model():
    # Check if model exists, if not download it
    model_path = 'models/yolov8n.pt'
    if not os.path.exists(model_path):
        st.info("Downloading YOLOv8 model... This may take a moment.")
        yolo_model = YOLO('yolov8n.pt')
        # Save the model
        yolo_model.save(model_path)
    else:
        yolo_model = YOLO(model_path)
    return yolo_model

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Key landmarks
key_pose_landmarks = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE
]

key_hand_landmarks = [
    mp_hands.HandLandmark.WRIST,
    mp_hands.HandLandmark.THUMB_TIP,
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP
]

def smooth_probability(prob_history, window_length=5):
    if len(prob_history) < window_length:
        return list(prob_history)
    return list(savgol_filter(list(prob_history), window_length, 2))

def extract_live_landmarks(frame, pose_model, hands_model, bbox=None):
    # If bbox is provided, crop the frame to that region
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        # Add padding to ensure we capture the full person
        h, w = y2 - y1, x2 - x1
        x1 = max(0, x1 - int(w * 0.1))
        y1 = max(0, y1 - int(h * 0.1))
        x2 = min(frame.shape[1], x2 + int(w * 0.1))
        y2 = min(frame.shape[0], y2 + int(h * 0.1))
        cropped_frame = frame[y1:y2, x1:x2]
    else:
        cropped_frame = frame
    
    img_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    pose_result = pose_model.process(img_rgb)
    hands_result = hands_model.process(img_rgb)
    landmark_data = []

    # Pose
    if pose_result.pose_landmarks:
        for l in key_pose_landmarks:
            pos = pose_result.pose_landmarks.landmark[l]
            landmark_data.extend([pos.x, pos.y, pos.visibility])
    else:
        landmark_data.extend([0, 0, 0] * len(key_pose_landmarks))

    # Hands
    hands_detected = 0
    if hands_result.multi_hand_landmarks:
        for hand_landmarks in hands_result.multi_hand_landmarks:
            if hands_detected < 2:
                for l in key_hand_landmarks:
                    pos = hand_landmarks.landmark[l]
                    landmark_data.extend([pos.x, pos.y, pos.z])
                hands_detected += 1
    while hands_detected < 2:
        landmark_data.extend([0, 0, 0] * len(key_hand_landmarks))
        hands_detected += 1

    return np.array(landmark_data).reshape(1, -1), pose_result, hands_result

def create_skeleton_image(landmarks, width=200, height=200):
    # Create a blank image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image.fill(240)  # Light gray background
    
    # Scale landmarks to image size
    scaled_landmarks = []
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        scaled_landmarks.append((x, y))
    
    # Draw connections
    connections = mp_pose.POSE_CONNECTIONS
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        if start_idx < len(scaled_landmarks) and end_idx < len(scaled_landmarks):
            cv2.line(image, scaled_landmarks[start_idx], scaled_landmarks[end_idx], (0, 128, 255), 2)
    
    # Draw landmarks
    for landmark in scaled_landmarks:
        cv2.circle(image, landmark, 4, (0, 0, 255), -1)
    
    return image

def process_frame(frame, pose_model, hands_model, yolo_model):
    frame_copy = frame.copy()
    results = yolo_model(frame_copy, classes=0)
    detected_people = []
    prob = 0.0  # Initialize prob with a default value
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            
            if conf > 0.5:
                features, pose_result, hands_result = extract_live_landmarks(
                    frame_copy, pose_model, hands_model, [x1, y1, x2, y2]
                )
                
                features_scaled = scaler.transform(features)
                person_prob = model.predict(features_scaled, verbose=0)[0][0]
                
                # Track movement
                if pose_result.pose_landmarks:
                    movement = calculate_movement(pose_result.pose_landmarks.landmark)
                    st.session_state.movement_history.append({
                        'timestamp': time.time(),
                        'movement': movement,
                        'probability': person_prob
                    })
                
                # Update metrics
                update_detection_metrics(person_prob)
                
                # Enhanced visualization
                draw_enhanced_detection(
                    frame_copy, 
                    [x1, y1, x2, y2], 
                    person_prob,
                    pose_result,
                    hands_result
                )
                
                person_data = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'probability': person_prob,
                    'pose_result': pose_result,
                    'hands_result': hands_result
                }
                detected_people.append(person_data)
                
                # Update the overall probability to the highest one detected
                prob = max(prob, person_prob)
    
    st.session_state.detected_people = detected_people
    
    # Apply light green filter if probability meets or exceeds threshold
    if prob >= st.session_state.confidence_threshold:
        # Create a green overlay
        green_overlay = np.zeros_like(frame_copy)
        green_overlay[:, :, 1] = 50  # Green channel
        
        # Blend the overlay with the original frame
        alpha = 0.3  # Transparency factor
        frame_copy = cv2.addWeighted(frame_copy, 1, green_overlay, alpha, 0)
    
    return frame_copy, prob

def calculate_movement(landmarks):
    """Calculate overall movement intensity from landmarks."""
    total_movement = 0
    for landmark in landmarks:
        total_movement += landmark.visibility * (landmark.x**2 + landmark.y**2)
    return total_movement / len(landmarks)

def draw_enhanced_detection(frame, bbox, prob, pose_result, hands_result):
    """Draw enhanced detection visualization."""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Always draw bounding box with rounded corners
    box_color = (0, int(255 * prob), int(255 * (1 - prob)))
    box_thickness = 3
    
    # Draw main rectangle with rounded corners
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
    
    # Draw corner accents with smooth corners
    corner_length = 30
    corner_radius = 10
    
    # Top-left corner
    cv2.ellipse(frame, (x1 + corner_radius, y1 + corner_radius), 
               (corner_radius, corner_radius), 180, 0, 90, box_color, box_thickness)
    cv2.line(frame, (x1 + corner_radius, y1), (x1 + corner_length, y1), box_color, box_thickness)
    cv2.line(frame, (x1, y1 + corner_radius), (x1, y1 + corner_length), box_color, box_thickness)
    
    # Top-right corner
    cv2.ellipse(frame, (x2 - corner_radius, y1 + corner_radius), 
               (corner_radius, corner_radius), 270, 0, 90, box_color, box_thickness)
    cv2.line(frame, (x2 - corner_radius, y1), (x2 - corner_length, y1), box_color, box_thickness)
    cv2.line(frame, (x2, y1 + corner_radius), (x2, y1 + corner_length), box_color, box_thickness)
    
    # Bottom-left corner
    cv2.ellipse(frame, (x1 + corner_radius, y2 - corner_radius), 
               (corner_radius, corner_radius), 90, 0, 90, box_color, box_thickness)
    cv2.line(frame, (x1 + corner_radius, y2), (x1 + corner_length, y2), box_color, box_thickness)
    cv2.line(frame, (x1, y2 - corner_radius), (x1, y2 - corner_length), box_color, box_thickness)
    
    # Bottom-right corner
    cv2.ellipse(frame, (x2 - corner_radius, y2 - corner_radius), 
               (corner_radius, corner_radius), 0, 0, 90, box_color, box_thickness)
    cv2.line(frame, (x2 - corner_radius, y2), (x2 - corner_length, y2), box_color, box_thickness)
    cv2.line(frame, (x2, y2 - corner_radius), (x2, y2 - corner_length), box_color, box_thickness)
    
    # Draw gradient background for probability display
    gradient_height = 40
    gradient_y = y1 - gradient_height - 10
    if gradient_y > 0:
        for i in range(gradient_height):
            alpha = 1 - (i / gradient_height)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, gradient_y + i), (x2, gradient_y + i + 1), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, alpha * 0.6, frame, 1 - alpha * 0.6, 0, frame)
    
    # Draw probability bar with gradient
    bar_width = int((x2 - x1) * prob)
    bar_color = (0, int(255 * prob), int(255 * (1 - prob)))
    cv2.rectangle(frame, (x1, gradient_y), (x1 + bar_width, gradient_y + 10), 
                 bar_color, -1)
    
    # Add glowing effect for high probability
    if prob > 0.65:  # Lowered threshold for glow effect
        blur_factor = int(10 * prob)
        glow = frame.copy()
        cv2.rectangle(glow, (x1, gradient_y), (x1 + bar_width, gradient_y + 10), 
                     bar_color, -1)
        glow = cv2.GaussianBlur(glow, (blur_factor * 2 + 1, blur_factor * 2 + 1), 0)
        cv2.addWeighted(frame, 1, glow, 0.3, 0, frame)
    
    # Draw skeleton with style
    if pose_result.pose_landmarks:
        # Draw connections with gradient color
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            start_point = pose_result.pose_landmarks.landmark[start_idx]
            end_point = pose_result.pose_landmarks.landmark[end_idx]
            
            start_pos = (int(start_point.x * frame.shape[1]), int(start_point.y * frame.shape[0]))
            end_pos = (int(end_point.x * frame.shape[1]), int(end_point.y * frame.shape[0]))
            
            # Create gradient color based on confidence
            color = (int(255 * (1 - prob)), int(255 * prob), 0)
            cv2.line(frame, start_pos, end_pos, color, 2)
        
        # Draw landmarks with glow effect
        for landmark in pose_result.pose_landmarks.landmark:
            pos = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
            # Outer glow
            cv2.circle(frame, pos, 6, (255, 255, 255), -1)
            # Inner point
            cv2.circle(frame, pos, 4, (0, 255, 0), -1)
    
    # Add floating labels with animation
    label_y = gradient_y - 25
    label_text = f"Confidence: {prob:.2f}"
    cv2.putText(frame, label_text, (x1 + 5, label_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def create_pose_heatmap(poses, width=400, height=400):
    """Create an enhanced pose heatmap with anatomical regions and better visualization."""
    heatmap = np.zeros((height, width))
    region_labels = {
        'Upper Body': [(0.3, 0.1, 0.7, 0.3), '#FF6B6B'],
        'Core': [(0.3, 0.3, 0.7, 0.5), '#4ECDC4'],
        'Lower Body': [(0.3, 0.5, 0.7, 0.9), '#45B7D1']
    }
    
    for pose in poses:
        if pose['probability'] > 0.75:
            landmarks = pose['landmarks']
            for landmark in landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                if 0 <= x < width and 0 <= y < height:
                    # Add Gaussian blur for smoother heatmap
                    sigma = 10
                    x_grid, y_grid = np.meshgrid(
                        np.arange(-3*sigma, 3*sigma+1),
                        np.arange(-3*sigma, 3*sigma+1)
                    )
                    gaussian = np.exp(-(x_grid**2 + y_grid**2)/(2*sigma**2))
                    
                    # Add the Gaussian to the heatmap
                    y_start = max(0, y-3*sigma)
                    y_end = min(height, y+3*sigma+1)
                    x_start = max(0, x-3*sigma)
                    x_end = min(width, x+3*sigma+1)
                    
                    g_y_start = max(0, 3*sigma-y)
                    g_y_end = gaussian.shape[0] - max(0, y+3*sigma+1-height)
                    g_x_start = max(0, 3*sigma-x)
                    g_x_end = gaussian.shape[1] - max(0, x+3*sigma+1-width)
                    
                    heatmap[y_start:y_end, x_start:x_end] += \
                        gaussian[g_y_start:g_y_end, g_x_start:g_x_end]
    
    return heatmap, region_labels

def analyze_movement_patterns(poses):
    """Analyze movement patterns in different body regions."""
    movement_data = {
        'Upper Body': {'intensity': 0, 'description': ''},
        'Core': {'intensity': 0, 'description': ''},
        'Lower Body': {'intensity': 0, 'description': ''}
    }
    
    # Calculate movement intensity for each region
    for pose in poses:
        landmarks = pose['landmarks']
        
        # Upper body movement (shoulders, arms)
        upper_landmarks = [0, 1, 2, 3, 4, 5, 6, 7]
        core_landmarks = [11, 12, 23, 24]
        lower_landmarks = [25, 26, 27, 28, 29, 30, 31, 32]
        
        for region, landmark_indices in [
            ('Upper Body', upper_landmarks),
            ('Core', core_landmarks),
            ('Lower Body', lower_landmarks)
        ]:
            movement = sum(
                landmarks[i].visibility * (landmarks[i].x**2 + landmarks[i].y**2)
                for i in landmark_indices
            ) / len(landmark_indices)
            
            movement_data[region]['intensity'] = min(100, movement * 100)
    
    # Add descriptions based on intensity
    for region in movement_data:
        intensity = movement_data[region]['intensity']
        if intensity > 75:
            movement_data[region]['description'] = 'High activity detected'
        elif intensity > 50:
            movement_data[region]['description'] = 'Moderate activity detected'
        elif intensity > 25:
            movement_data[region]['description'] = 'Low activity detected'
        else:
            movement_data[region]['description'] = 'Minimal activity detected'
    
    return movement_data

def update_detection_metrics(prob):
    """Update detection metrics with new probability."""
    metrics = st.session_state.detection_metrics
    metrics['total_detections'] += 1
    metrics['avg_confidence'] = (metrics['avg_confidence'] * (metrics['total_detections'] - 1) + prob) / metrics['total_detections']
    metrics['peak_confidence'] = max(metrics['peak_confidence'], prob)
    metrics['session_duration'] = time.time() - st.session_state.session_start_time

# Main UI
st.markdown('<h1 class="header-text">Rat Dance Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader-text">Advanced real-time pose detection and classification for rat dance movements.</p>', unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.markdown('<h2 class="sidebar-header">System Controls</h2>', unsafe_allow_html=True)
    
    # Camera control buttons - always visible
    col1, col2 = st.columns(2)
    
    with col1:
        init_button = st.button("Initialize Camera", key="init_button", help="Start the camera for pose detection")
        if init_button and not st.session_state.is_running:
            st.session_state.is_running = True
            st.success("Camera initialized successfully")
            st.experimental_rerun()
    
    with col2:
        term_button = st.button("Terminate Camera", key="term_button", help="Stop the camera")
        if term_button and st.session_state.is_running:
            st.session_state.is_running = False
            st.info("Camera terminated")
            st.experimental_rerun()
    
    # Show camera status
    if st.session_state.is_running:
        st.markdown('<div style="color: #4CAF50; font-weight: 500; margin-top: 10px;">Camera: Active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color: #F44336; font-weight: 500; margin-top: 10px;">Camera: Inactive</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Detection mode selection
    st.markdown('<h2 class="sidebar-header">Detection Settings</h2>', unsafe_allow_html=True)
    detection_mode = st.selectbox(
        "Detection Mode",
        ["Standard", "High Precision", "Fast"],
        index=0,
        help="Standard: Balanced detection, High Precision: More accurate but slower, Fast: Faster but less accurate"
    )
    st.session_state.detection_mode = detection_mode
    
    # Recording controls
    if st.session_state.is_running:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<h2 class="sidebar-header">Recording</h2>', unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        with col3:
            record_button = st.button("Record", key="record_button", help="Start recording detected poses")
            if record_button and not st.session_state.recording:
                st.session_state.recording = True
                st.session_state.recorded_frames = []
                st.success("Recording started")
                st.experimental_rerun()
        
        with col4:
            stop_record_button = st.button("Stop Recording", key="stop_record_button", help="Stop recording and save")
            if stop_record_button and st.session_state.recording:
                st.session_state.recording = False
                st.info("Recording stopped")
                st.experimental_rerun()
    
    # Heatmap toggle
    if st.session_state.is_running:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<h2 class="sidebar-header">Visualization</h2>', unsafe_allow_html=True)
        show_heatmap = st.checkbox("Show Heatmap", value=False, help="Display heatmap of detected poses")
        st.session_state.show_heatmap = show_heatmap
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="sidebar-header">System Information</h2>', unsafe_allow_html=True)
    st.markdown("""
        <div style="font-size: 0.9rem; color: #aaa;">
            <p><strong>Framework:</strong> TensorFlow</p>
            <p><strong>Pose Detection:</strong> MediaPipe</p>
            <p><strong>Person Detection:</strong> YOLOv8</p>
            <strong>Interface:</strong> Streamlit</p>
            <p><strong>Processing:</strong> Real-time</p>
        </div>
    """, unsafe_allow_html=True)

# Main content area
tab1, tab2, tab3 = st.tabs(["Live Detection", "Analysis", "Settings"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Enhanced video feed container
        st.markdown("""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h3 style="margin: 0;">Live Detection Feed</h3>
                    <div class="badge" id="status-badge">Active</div>
                </div>
                <div class="video-container">
        """, unsafe_allow_html=True)
        
        video_placeholder = st.empty()
        
        st.markdown("""
                </div>
                <div class="controls-container">
                    <div class="tooltip">
                        <span class="tooltiptext">Adjust detection sensitivity</span>
                        <div class="control-label">Detection Threshold</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Add back the confidence threshold slider
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            0.0, 1.0, st.session_state.confidence_threshold, 0.05,
            help="Adjust the sensitivity of the detection. Lower values make it easier to detect the rat dance."
        )
        st.session_state.confidence_threshold = confidence_threshold
        
        # Initialize camera and process frames
        if st.session_state.is_running:
            # Load YOLO model if not already loaded
            if st.session_state.yolo_model is None:
                st.session_state.yolo_model = load_yolo_model()
            
            cap = cv2.VideoCapture(0)
            
            with mp_pose.Pose(static_image_mode=False, model_complexity=2) as pose_model, \
                 mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5) as hands_model:
                
                while st.session_state.is_running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame with YOLO
                    processed_frame, prob = process_frame(frame, pose_model, hands_model, st.session_state.yolo_model)
                    
                    # Update prediction history
                    st.session_state.prob_history.append(prob)
                    st.session_state.timestamp_history.append(time.time())
                    st.session_state.prediction_history.append(1 if prob > st.session_state.confidence_threshold else 0)
                    
                    # Record frame if recording is active
                    if st.session_state.recording and prob > st.session_state.confidence_threshold:
                        st.session_state.recorded_frames.append(processed_frame.copy())
                    
                    # Convert to RGB for Streamlit
                    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display frame
                    video_placeholder.image(processed_frame_rgb, channels="RGB")
                    
                    # Add small delay to prevent overwhelming the system
                    time.sleep(0.01)
            
            cap.release()
    
    with col2:
        # Real-time metrics dashboard
        st.markdown("""
            <div class="glass-card">
                <h3 style="margin-top: 0;">Live Metrics</h3>
                <div class="metrics-grid">
        """, unsafe_allow_html=True)
        
        metrics = st.session_state.detection_metrics
        
        # Display metrics in a grid
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{metrics['avg_confidence']:.2f}</div>
                    <div class="metric-label">Avg Confidence</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{metrics['peak_confidence']:.2f}</div>
                    <div class="metric-label">Peak Confidence</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Movement intensity graph
        if len(st.session_state.movement_history) > 0:
            movement_data = pd.DataFrame(st.session_state.movement_history)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=movement_data['movement'].rolling(window=5).mean(),
                mode='lines',
                name='Movement',
                line=dict(color='#4CAF50', width=2)
            ))
            
            fig.update_layout(
                title="Movement Intensity",
                height=200,
                margin=dict(l=20, r=20, t=30, b=20),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Session info
        st.markdown(f"""
            <div class="session-info">
                <div class="info-item">
                    <span class="info-label">Session Duration:</span>
                    <span class="info-value">{int(metrics['session_duration'])}s</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Total Detections:</span>
                    <span class="info-value">{metrics['total_detections']}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

with tab2:
    # Analysis tab content
    st.markdown('<h2 class="sidebar-header">Detailed Analysis</h2>', unsafe_allow_html=True)
    
    # Ensure we have data to display
    if len(st.session_state.prob_history) == 0:
        st.markdown("""
            <div class="glass-card">
                <h3 style="margin-top: 0;">No Data Available</h3>
                <p style="color: #aaa; font-size: 0.9rem;">
                    Start the camera and perform some movements to generate analysis data.
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Probability history graph
        # Create time-based x-axis
        start_time = st.session_state.timestamp_history[0]
        time_data = [t - start_time for t in st.session_state.timestamp_history]
        
        # Smooth the probability history
        smoothed_prob = smooth_probability(st.session_state.prob_history)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_data,
            y=smoothed_prob,
            mode='lines',
            name='Confidence',
            line=dict(color='#4CAF50', width=2)
        ))
        fig.add_hline(y=st.session_state.confidence_threshold, line_dash="dash", line_color="#F44336",
                     annotation_text="Threshold", annotation_position="right")
        
        fig.update_layout(
            title="Confidence History",
            xaxis_title="Time (seconds)",
            yaxis_title="Confidence Score",
            yaxis_range=[0, 1],
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        positive_predictions = sum(st.session_state.prediction_history)
        total_predictions = len(st.session_state.prediction_history)
        positive_percentage = (positive_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        
        # Create a more sophisticated statistics display
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-value">{positive_predictions}</div>
                    <div class="metric-label">Rat Dance Detections</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col_b:
            st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-value">{positive_percentage:.1f}%</div>
                    <div class="metric-label">Detection Rate</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Add movement pattern analysis
        if len(st.session_state.movement_history) > 0:
            st.markdown('<h3 style="margin-top: 2rem;">Movement Pattern Analysis</h3>', unsafe_allow_html=True)
            
            # Convert movement history to DataFrame
            movement_df = pd.DataFrame(st.session_state.movement_history)
            
            # Calculate movement statistics
            avg_movement = movement_df['movement'].mean()
            max_movement = movement_df['movement'].max()
            movement_std = movement_df['movement'].std()
            
            # Create movement statistics display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                    <div class="glass-card">
                        <div class="metric-value">{avg_movement:.2f}</div>
                        <div class="metric-label">Average Movement</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="glass-card">
                        <div class="metric-value">{max_movement:.2f}</div>
                        <div class="metric-label">Peak Movement</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class="glass-card">
                        <div class="metric-value">{movement_std:.2f}</div>
                        <div class="metric-label">Movement Variability</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Movement intensity graph
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=movement_df['movement'].rolling(window=5).mean(),
                mode='lines',
                name='Movement',
                line=dict(color='#4CAF50', width=2)
            ))
            
            fig.update_layout(
                title="Movement Intensity Over Time",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add correlation analysis
            if len(movement_df) > 1:
                correlation = movement_df['movement'].corr(movement_df['probability'])
                
                st.markdown(f"""
                    <div class="glass-card">
                        <h4 style="margin-top: 0;">Movement-Confidence Correlation</h4>
                        <div class="metric-value">{correlation:.2f}</div>
                        <div class="metric-label">Correlation Coefficient</div>
                        <p style="color: #aaa; font-size: 0.9rem; margin-top: 1rem;">
                            This value indicates how strongly movement intensity correlates with detection confidence.
                            Values closer to 1.0 indicate a stronger positive correlation.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="glass-card">
                    <h3 style="margin-top: 0;">No Movement Data Available</h3>
                    <p style="color: #aaa; font-size: 0.9rem;">
                        Perform some movements in front of the camera to generate movement analysis data.
                    </p>
                </div>
            """, unsafe_allow_html=True)

    # Enhanced heatmap visualization
    if st.session_state.show_heatmap and len(st.session_state.top_poses) > 0:
        st.markdown("""
            <div class="glass-card">
                <h3 style="margin-top: 0;">Movement Heatmap Analysis</h3>
                <p style="color: #aaa; font-size: 0.9rem;">
                    This heatmap shows the intensity of movement across different body regions. 
                    Brighter areas indicate more frequent movement detection.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        heatmap, region_labels = create_pose_heatmap(st.session_state.top_poses)
        
        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # Create heatmap figure with anatomical regions
        fig = go.Figure()
        
        # Add the heatmap
        fig.add_trace(go.Heatmap(
            z=heatmap,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title='Movement Intensity',
                titleside='right',
                thickness=15,
                len=0.75,
                ticktext=['Low', 'Medium', 'High'],
                tickvals=[0, 0.5, 1],
                tickmode='array'
            )
        ))
        
        # Add anatomical regions
        for region, (coords, color) in region_labels.items():
            x0, y0, x1, y1 = coords
            fig.add_shape(
                type="rect",
                x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color=color, width=2),
                fillcolor="rgba(0,0,0,0)",
                name=region
            )
            # Add region labels
            fig.add_annotation(
                x=(x0+x1)/2,
                y=y0-0.02,
                text=region,
                showarrow=False,
                font=dict(size=12, color=color)
            )
        
        fig.update_layout(
            title=dict(
                text="Body Movement Analysis",
                font=dict(size=20)
            ),
            height=500,
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12, color='#f0f2f6'),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add movement insights
        if len(st.session_state.top_poses) > 0:
            movement_data = analyze_movement_patterns(st.session_state.top_poses)
            
            st.markdown("""
                <div class="glass-card">
                    <h4 style="margin-top: 0;">Movement Insights</h4>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
            """, unsafe_allow_html=True)
            
            for region, stats in movement_data.items():
                st.markdown(f"""
                    <div class="metric-card">
                        <h5>{region}</h5>
                        <div class="metric-value">{stats['intensity']:.1f}%</div>
                        <div class="metric-label">Movement Intensity</div>
                        <div style="font-size: 0.8rem; color: #aaa; margin-top: 0.5rem;">
                            {stats['description']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Recorded frames
    if len(st.session_state.recorded_frames) > 0:
        st.markdown('<h3 style="margin-top: 2rem;">Recorded Frames</h3>', unsafe_allow_html=True)
        
        # Create a grid of recorded frames
        cols = st.columns(3)
        for i, frame in enumerate(st.session_state.recorded_frames[:9]):  # Show up to 9 frames
            with cols[i % 3]:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        # Download button for recorded frames
        if len(st.session_state.recorded_frames) > 0:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<h3 style="margin-top: 1rem;">Download Recording</h3>', unsafe_allow_html=True)
            
            # Create a video from recorded frames
            if st.button("Generate Video"):
                with st.spinner("Generating video..."):
                    # Create a temporary directory for the video
                    os.makedirs("temp", exist_ok=True)
                    
                    # Save frames as images
                    for i, frame in enumerate(st.session_state.recorded_frames):
                        cv2.imwrite(f"temp/frame_{i:04d}.jpg", frame)
                    
                    # Create video from frames
                    os.system("ffmpeg -framerate 10 -i temp/frame_%04d.jpg -c:v libx264 -pix_fmt yuv420p temp/output.mp4")
                    
                    # Read the video file
                    with open("temp/output.mp4", "rb") as file:
                        video_bytes = file.read()
                    
                    # Create download button
                    st.download_button(
                        label="Download Video",
                        data=video_bytes,
                        file_name="rat_dance_recording.mp4",
                        mime="video/mp4"
                    )
                    
                    # Clean up temporary files
                    for i in range(len(st.session_state.recorded_frames)):
                        os.remove(f"temp/frame_{i:04d}.jpg")
                    os.remove("temp/output.mp4")

with tab3:
    # Settings tab content
    st.markdown('<h2 class="sidebar-header">Application Settings</h2>', unsafe_allow_html=True)
    
    # Visualization settings
    st.markdown('<h3 style="margin-top: 1rem;">Visualization Settings</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        show_landmarks = st.checkbox("Show Landmarks", value=True, help="Display pose landmarks on video")
    
    with col2:
        show_bounding_boxes = st.checkbox("Show Bounding Boxes", value=True, help="Display bounding boxes around detected people")
    
    # Performance settings
    st.markdown('<h3 style="margin-top: 2rem;">Performance Settings</h3>', unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    with col3:
        frame_skip = st.slider(
            "Frame Skip",
            0, 5, 0,
            help="Process every Nth frame (0 = process all frames)"
        )
    
    with col4:
        history_length = st.slider(
            "History Length",
            10, 100, 30,
            help="Number of frames to keep in history"
        )
    
    # Save settings button
    if st.button("Save Settings"):
        st.success("Settings saved successfully!") 