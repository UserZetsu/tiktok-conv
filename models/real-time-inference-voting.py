import cv2
import mediapipe as mp
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from collections import deque  

# Load Keras model and scaler
model = load_model('models/pklfiles/rnn_quarter_model.keras')
scaler = joblib.load('models/pklfiles/rnn_quarter_scaler.pkl')

# Define timesteps (should match training)
timesteps = 30
sequence_buffer = deque(maxlen=timesteps)

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# KEY LANDMARKS (same as training)
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

def extract_live_landmarks(frame, pose_model, hands_model):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

    # Hands (up to 2 hands, pad with zeros if less)
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

    return np.array(landmark_data).reshape(1, -1)

# Real-time video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)

# Add prediction buffer
prediction_buffer = deque(maxlen=30)

with mp_pose.Pose(static_image_mode=False, model_complexity=2) as pose_model, \
     mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5) as hands_model:

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Data needs to look like what jackson did

        # I'm gonna extract it :O
        features = extract_live_landmarks(frame, pose_model, hands_model)
        sequence_buffer.append(features[0])  # <- collect sequence of frames

        # Keras model prediction (sigmoid output)
        if len(sequence_buffer) == timesteps:
            sequence_array = np.array(sequence_buffer)  # shape (timesteps, features)
            sequence_scaled = scaler.transform(sequence_array)  # normalize
            sequence_scaled = sequence_scaled.reshape(1, timesteps, -1)  # RNN shape

            prob = model.predict(sequence_scaled)[0][0]
            predicted_label = 1 if prob > 0.65 else 0
        else:
            prob = 0.0
            predicted_label = 0

        # Add current prediction to buffer
        prediction_buffer.append(predicted_label)

        # Bounding box around body
        pose_result = pose_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if pose_result.pose_landmarks:
            h, w, _ = frame.shape
            x_coords = [int(lm.x * w) for lm in pose_result.pose_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in pose_result.pose_landmarks.landmark]

            x_min = max(0, min(x_coords) - 20)
            x_max = min(w, max(x_coords) + 20)
            y_min = max(0, min(y_coords) - 150)  # extend above head or ielse it will be eye level
            y_max = min(h, max(y_coords) + 20)

            # Box color red when not dancing else green
            box_color = (0, 255, 0) if predicted_label == 1 else (0, 0, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 5)
            
            # Fun labels
            if predicted_label == 0: 
                label_text = f'YOU ARE NOT RAT DANCING. Prob of rat dancing: {prob:.2f}'
            else: 
                label_text = f'Rat dancing god: {prob:.2f}'

            # Box dimensions etc
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x_min, y_min - th - 10), (x_min + tw + 10, y_min), box_color, cv2.FILLED)
            cv2.putText(frame, label_text, (x_min + 5, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

            # Show RAT DANCE if majority of buffer == 1
            if len(prediction_buffer) == prediction_buffer.maxlen:
                if prediction_buffer.count(1) > prediction_buffer.count(0):
                    cv2.putText(frame, "RAT DANCE ACTIVATED!", (x_min, y_min - 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        # Show video
        cv2.imshow("Real-Time Pose Inference", frame)
        if cv2.waitKey(10) & 0xFF in [ord('q'), 27]:  # 27 = ESC
            break

cap.release()
cv2.destroyAllWindows()
