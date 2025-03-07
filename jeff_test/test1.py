import cv2 
import mediapipe as mp
import csv
import pandas as pd 


def estimate_landmarks(frame, model):
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = model.process(img)
    
    # if result.face_landmarks:
        # Face landmarks if we want to use it 
        # mp_drawing.draw_landmarks(frame,
        #                         result.face_landmarks,
        #                         mp_holistic.FACEMESH_TESSELATION,
        #                         landmark_drawing_spec=None,
        #                         connection_drawing_spec=mp_drawing_styles
        #                         .get_default_face_mesh_tesselation_style())
    
    landmark_data = []
    
    # There are 33 key landmark points for the body.
    if result.pose_landmarks:        
        # Pose AKA the body's landmark
        mp_drawing.draw_landmarks(frame,
                                result.pose_landmarks,
                                mp_holistic.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing_styles.
                                get_default_pose_landmarks_style())
        for landmark in result.pose_landmarks.landmark:
            landmark_data.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    # If the landmark cannot be found, fill 
    else:
        landmark_data.extend(0 * (33 * 4))
        
    # There are 21 key landmark points for the hand, both left and right
    if result.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            result.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
        )
        for left_hand_landmark in result.left_hand_landmarks.landmark:
            left_hand_landmark.extend([landmark.x, landmark.y, landmark.z])
            
    # If the landmark cannot be found, fill 21 points with x, y and z data with zeros
    else:
        landmark_data.extend(0 * (21 * 3))
        
    if result.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            result.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
        )
        for right_hand_landmark in result.right_hand_landmark.landmark:
            right_hand_landmark.extend([landmark.x, landmark.y, landmark.z])
    else:
        landmark_data.extend(0 * (21 * 3))
            
    # Visualize the pose 
    # mp_drawing.plot_landmarks(result.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)
        
    return  landmark_data
    

def get_video(path):
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
            print(f'Currently on frame: {i}===================================')
            ok, frame = video.read()

            if not ok:
                break
            
            landmarks = estimate_landmarks(frame, holistic_model)
            frames_landmarks.append(landmarks)
    

        
        
        
        
    return frames_landmarks 
    
# Drawing class to draw landmarks
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

#get_video('clip_1.mp4')

header = []
lhand = []
rhand = []
for i in range(33):
    header.extend([f'x_poselandmark_{i}', f'y_poselandmark_{i}', f'z_poselandmark_{i}', f'poseconfidence_{i}'])

for i in range(21):
    lhand.extend([f'x_lhandlandmark_{i}', f'y_lhandlandmark_{i}', f'z_lhandlandmark_{i}'])
    rhand.extend([f'x_rhandlandmark_{i}', f'y_rhandlandmark_{i}', f'z_rhandlandmark_{i}'])

print(header +lhand + rhand)


# [f'x_landmark_{i}', f'x_landmark_{i}', f'z_landmark_{i}', f'confidence_{i}' for i in range(33)]

print(header)
    
    