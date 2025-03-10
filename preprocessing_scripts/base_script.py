import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

# Initialize models separately
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

def estimate_landmarks(frame, pose_model, hands_model):
    """Extracts key upper-body pose and important hand landmarks."""
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process pose & hand models separately
    pose_result = pose_model.process(img)
    hands_result = hands_model.process(img)

    landmark_data = []

    # **Extract Pose Landmarks**
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

    if pose_result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        # Running this mf loop with no Z cause mad inaccurate according to google.
        for landmark in key_pose_landmarks:
            x = pose_result.pose_landmarks.landmark[landmark].x 
            y = pose_result.pose_landmarks.landmark[landmark].y
            viz = pose_result.pose_landmarks.landmark[landmark].visibility            
            
            landmark_data.extend([x, y, viz])
    else: 
        for landmark in key_hand_landmarks:
            landmark_data.extend([0, 0, 0])
    
    # **Extract Hand Landmarks**
    key_hand_landmarks = [
        mp_hands.HandLandmark.WRIST,
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]

    # BOTH HANDS
    hands_detected = 0
    if hands_result.multi_hand_landmarks:
        for hand_landmarks in hands_result.multi_hand_landmarks:
            if hands_detected < 2:  
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for landmark in key_hand_landmarks:
                    x = hand_landmarks.landmark[landmark].x  
                    y = hand_landmarks.landmark[landmark].y
                    z = hand_landmarks.landmark[landmark].z        
                    landmark_data.extend([x, y, z])
                hands_detected += 1

    # If fewer than 2 hands detected, fill missing slots with zeros
    while hands_detected < 2:
        landmark_data.extend([0, 0, 0] * len(key_hand_landmarks))
        hands_detected += 1

    return landmark_data

def video_to_dataframe(path):
    """Processes a video and returns a Pandas DataFrame of pose & hand landmarks for each frame."""
    
    video = cv2.VideoCapture(path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_landmarks = []

    # Initialize MediaPipe models separately
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2) as pose_model, \
         mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5) as hands_model:

        for i in tqdm(range(total_frames)):
            ok, frame = video.read()
            if not ok:
                break
            
            # Extract landmarks for the current frame + index it 
            landmarks = estimate_landmarks(frame, pose_model, hands_model)
            frames_landmarks.append([i] + landmarks)  

    video.release()

    # Define column names
    pose_columns = [f"{axis}_pose_{j}" for j in range(10) for axis in ['x', 'y', 'visibility']]
    hand_columns1 = [f"{axis}_hand1_{j}" for j in range(6) for axis in ['x', 'y', 'z']]
    hand_columns2 = [f"{axis}_hand2_{j}" for j in range(6) for axis in ['x', 'y', 'z']]
    
    column_names = ["frame"] + pose_columns + hand_columns1 + hand_columns2
    
    # Convert to DataFrame
    df = pd.DataFrame(frames_landmarks, columns=column_names)
    return df


# Process video and return DataFrame, 
# TO USE THIS JUST PROVIDE THE PATH TO THE CLIP!!!
df = video_to_dataframe('clip_1.mp4')

# Save to CSV
df.to_csv('landmarks.csv', index=False)
print("Landmark data saved to landmarks.csv !!!!!!!!!!!!")
