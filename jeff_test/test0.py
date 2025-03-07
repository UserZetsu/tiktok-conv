import cv2 
import mediapipe as mp
import pandas as pd 

def estimate_landmarks(frame, model):
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = model.process(img)
    
    # Key points from the body that I think are important for a dance
    pose_landmarks = [
        model.PoseLandmark.LEFT_SHOULDER,
        model.PoseLandmark.RIGHT_SHOULDER,
        model.PoseLandmark.LEFT_ELBOW,
        model.PoseLandmark.RIGHT_ELBOW,
        model.PoseLandmark.LEFT_WRIST,
        model.PoseLandmark.RIGHT_WRIST,
        model.PoseLandmark.LEFT_HIP,  
        model.PoseLandmark.RIGHT_HIP,
        model.PoseLandmark.LEFT_KNEE,  
        model.PoseLandmark.RIGHT_KNEE
    ]
    # Key points from the hands that I think are important for a dance
    hand_landmarks = [
        model.HandLandmark.WRIST,
        model.HandLandmark.THUMB_TIP,
        model.HandLandmark.INDEX_FINGER_TIP,
        model.HandLandmark.MIDDLE_FINGER_TIP,
        model.HandLandmark.RING_FINGER_TIP,
        model.HandLandmark.PINKY_TIP
    ]
    
    landmark_data = []
    # Extracting pose landmarks 
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            model.POSE_CONNECTIONS,
            landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style()
        )
        for landmark in pose_landmarks: 
            landmark_data.extend(result.pose_landmarks.landmark[landmark].x)
        
    print(landmark_data)
        
    return
    
def video_to_dataframe(path, model):
    video = cv2.VideoCapture(path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_landmarks = []
    
    # Mediapipe's holistic model for video
    with mp_holistic.Holistic(static_image_mode = False, 
                            min_detection_confidence = 0.5,
                            model_complexity = 2,
                            enable_segmentation=True,
                            refine_face_landmarks=True) as holistic_model:
        
        for i in range(total_frames):
            print(f'======================Currently on frame: {i}===================================')
            ok, frame = video.read()
            if not ok:
                break
            
            # Finding landmarks and appending to list 
            landmarks = estimate_landmarks(frame, holistic_model)
            frames_landmarks.append(landmarks)
            
    return frames_landmarks 
    

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


video_to_dataframe('clip_1.mp4', mp_holistic)

      
