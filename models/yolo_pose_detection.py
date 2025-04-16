import cv2
import mediapipe as mp
import numpy as np
import os
import time
from datetime import datetime

# Initialize MediaPipe pose components
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize YOLO for person detection
print("Setting up YOLO model for person detection...")

# YOLO model files
yolo_cfg = "yolov4-tiny.cfg"
yolo_weights = "yolov4-tiny.weights"
coco_names = "coco.names"

# Check if YOLO files exist
if not (os.path.exists(yolo_cfg) and os.path.exists(yolo_weights)):
    print("Downloading YOLO model files...")
    # Download YOLOv4-tiny model files
    import urllib.request
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
        yolo_cfg)
    urllib.request.urlretrieve(
        "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
        yolo_weights)
    print("YOLO model downloaded successfully!")

# Download COCO class names if needed
if not os.path.exists(coco_names):
    print("Downloading COCO class names...")
    import urllib.request
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
        coco_names)
    print("Class names downloaded successfully!")

# Load COCO class names
with open(coco_names, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Load YOLO network
net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)

# Set preferred backend and target
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use DNN_TARGET_OPENCL for GPU

# Get output layer names
layer_names = net.getLayerNames()
try:
    # OpenCV 4.5.4+
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except:
    # Older OpenCV versions
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Set full-frame pose model for fallback
full_pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
    smooth_landmarks=True)

# Use a higher detection confidence for ROI-based detection
# This helps avoid false positives in the smaller regions
roi_pose = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1,
    static_image_mode=False,  # Set to false for better tracking
    smooth_landmarks=True)

# Frame counter for pose estimation strategy
frame_count = 0
fallback_interval = 10  # Check full frame every N frames even if YOLO finds people

# For webcam input:
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
        
    # Get frame dimensions
    h, w, c = image.shape
    
    # Convert the BGR image to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a working copy of the image for drawing
    display_image = image.copy()
    
    # Prepare image for YOLO
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    # Set input to the network
    net.setInput(blob)
    
    # Run forward pass
    outs = net.forward(output_layers)
    
    # Lists to store detection results
    class_ids = []
    confidences = []
    boxes = []
    
    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filter detections: only persons (class_id 0) with confidence > 0.5
            if class_id == 0 and confidence > 0.5:  # class_id 0 = person in COCO dataset
                # YOLO returns center (x, y) with width and height
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)
                
                # Rectangle coordinates
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression to remove overlapping bounding boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # List to store person boxes
    person_boxes = []
    
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, width, height = boxes[i]
            confidence = confidences[i]
            
            # Add padding to make sure whole body is included
            padding_x = int(width * 0.1)
            padding_y = int(height * 0.1)
            start_x = max(0, x - padding_x)
            start_y = max(0, y - padding_y)
            end_x = min(w, x + width + padding_x)
            end_y = min(h, y + height + padding_y)
            
            # Store person box
            person_boxes.append((start_x, start_y, end_x, end_y, confidence))
    
    # Process each detected person with MediaPipe Pose
    for i, (start_x, start_y, end_x, end_y, confidence) in enumerate(person_boxes):
        # Extract person ROI
        person_roi = image_rgb[start_y:end_y, start_x:end_x]
        
        if person_roi.size > 0 and person_roi.shape[0] > 0 and person_roi.shape[1] > 0:
            # Process ROI with MediaPipe Pose
            results = roi_pose.process(person_roi)
            
            # Draw bounding box around person
            cv2.rectangle(display_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            
            # Add person label with confidence
            label = f"Person {i+1}: {confidence:.2f}"
            cv2.putText(display_image, label, (start_x, start_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # If pose landmarks detected, visualize
            if results.pose_landmarks:
                # Draw pose landmarks
                mp_drawing.draw_landmarks(
                    display_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS)
    
    # Try a full-frame pose detection as fallback if no people detected by YOLO
    if len(person_boxes) == 0:
        # Process whole image with MediaPipe Pose
        results = full_pose.process(image_rgb)
        
        # Draw the pose landmarks on the image
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                display_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS)
            
            # Label as fallback detection
            cv2.putText(display_image, "Fallback Detection", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Increment frame counter
    frame_count += 1
    if frame_count > 100:  # Reset to avoid potential overflow
        frame_count = 0
    
    # Add UI elements
    cv2.putText(display_image, "Multi-Person Pose Detection (YOLO + MediaPipe)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Show count of detected people
    cv2.putText(display_image, f"Detected: {len(person_boxes)} people", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display the resulting frame
    cv2.imshow('MediaPipe Pose Landmark Recording', display_image)
    
    if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows() 