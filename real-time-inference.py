import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained model 
model = joblib.load("my_trained_model.pkl")

# Setup MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Key landmarks
key_pose_landmarks = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST
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
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_result = pose_model.process(img)
    hands_result = hands_model.process(img)
    landmark_data = []

    # Pose
    if pose_result.pose_landmarks:
        for l in key_pose_landmarks:
            pos = pose_result.pose_landmarks.landmark[l]
            landmark_data.extend([pos.x, pos.y, pos.visibility])
    else:
        landmark_data.extend([0, 0, 0] * len(key_pose_landmarks))

    # Hands
    hands_count = 0
    if hands_result.multi_hand_landmarks:
        for hand_landmarks in hands_result.multi_hand_landmarks:
            if hands_count < 2:
                for l in key_hand_landmarks:
                    pos = hand_landmarks.landmark[l]
                    landmark_data.extend([pos.x, pos.y, pos.z])
                hands_count += 1
    while hands_count < 2:
        landmark_data.extend([0, 0, 0] * len(key_hand_landmarks))
        hands_count += 1

    return np.array(landmark_data).reshape(1, -1)

# Real-time video
cap = cv2.VideoCapture(0)

with mp_pose.Pose(static_image_mode=False, model_complexity=2) as pose_model, \
     mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5) as hands_model:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract landmarks
        features = extract_live_landmarks(frame, pose_model, hands_model)

        # Predict using your ML model
        prediction = model.predict(features)[0]  # For sklearn-type model
        # For Keras: prediction = model.predict(features).argmax()

        # Display prediction on frame
        cv2.putText(frame, f'Prediction: {prediction}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Real-Time Pose Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
