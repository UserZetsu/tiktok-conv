import cv2
import mediapipe as mp
import pandas as pd
import os
import re
from tqdm import tqdm

# Initialize models separately
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

def estimate_landmarks(frame, pose_model, hands_model):
    """Extracts key upper-body pose and important hand landmarks."""
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_result = pose_model.process(img)
    hands_result = hands_model.process(img)

    landmark_data = []

    # Pose landmarks
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
        for landmark in key_pose_landmarks:
            x = pose_result.pose_landmarks.landmark[landmark].x 
            y = pose_result.pose_landmarks.landmark[landmark].y
            viz = pose_result.pose_landmarks.landmark[landmark].visibility            
            landmark_data.extend([x, y, viz])
    else: 
        landmark_data.extend([0, 0, 0] * len(key_pose_landmarks))
    
    # Hand landmarks
    key_hand_landmarks = [
        mp_hands.HandLandmark.WRIST,
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    
    hands_detected = 0
    if hands_result.multi_hand_landmarks:
        for hand_landmarks in hands_result.multi_hand_landmarks:
            if hands_detected < 2:  
                for landmark in key_hand_landmarks:
                    x = hand_landmarks.landmark[landmark].x  
                    y = hand_landmarks.landmark[landmark].y
                    z = hand_landmarks.landmark[landmark].z        
                    landmark_data.extend([x, y, z])
                hands_detected += 1

    while hands_detected < 2:
        landmark_data.extend([0, 0, 0] * len(key_hand_landmarks))
        hands_detected += 1

    return landmark_data

def video_to_dataframe(path):
    """Processes a video and returns a Pandas DataFrame of pose & hand landmarks for each frame."""
    video = cv2.VideoCapture(path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_landmarks = []

    with mp_pose.Pose(min_detection_confidence=0.5) as pose_model, \
         mp_hands.Hands(min_detection_confidence=0.5) as hands_model:
        for i in tqdm(range(total_frames)):
            ok, frame = video.read()
            if not ok:
                break
            landmarks = estimate_landmarks(frame, pose_model, hands_model)
            frames_landmarks.append([i] + landmarks)  
    
    video.release()

    pose_columns = [f"{axis}_pose_{j}" for j in range(10) for axis in ['x', 'y', 'visibility']]
    hand_columns1 = [f"{axis}_hand1_{j}" for j in range(6) for axis in ['x', 'y', 'z']]
    hand_columns2 = [f"{axis}_hand2_{j}" for j in range(6) for axis in ['x', 'y', 'z']]
    
    column_names = ["frame"] + pose_columns + hand_columns1 + hand_columns2
    
    return pd.DataFrame(frames_landmarks, columns=column_names)

def process_videos_in_folder(folder, prefix, output_folder):
    """ processes all videos in a given folder and saves them as CSV files in output folder """
    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    video_files.sort()
    
    for idx, video_file in enumerate(video_files, start=1):
        video_path = os.path.join(folder, video_file)
        print(f"Processing {video_file}...")
        df = video_to_dataframe(video_path)
        match = re.search(r'_(\d+)', video_file)
        file_number = match.group(1) if match else 'unknown'
        csv_filename = os.path.join(output_folder, f"{prefix}{idx}_{file_number}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"Saved {csv_filename}")

# Process for ratdanceCSV
# process_videos_in_folder('ratdance/train', 'train', 'ratdanceCSV/train')
# process_videos_in_folder('ratdance/val', 'val', 'ratdanceCSV/val')
# process_videos_in_folder('ratdance/test', 'test', 'ratdanceCSV/test')

# Process for negative_controlCSV
process_videos_in_folder('negative_control/train', 'train', 'negative_controlCSV/train')
process_videos_in_folder('negative_control/val', 'val', 'negative_controlCSV/val')
process_videos_in_folder('negative_control/test', 'test', 'negative_controlCSV/test')

