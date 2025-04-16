import cv2
import os
import numpy as np

def extract_frames(video_path, output_dir, frame_interval=1):
    """Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame (default: 1)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video properties:")
    print(f"- FPS: {fps}")
    print(f"- Total frames: {total_frames}")
    print(f"- Duration: {duration:.2f} seconds")
    
    # Extract frames
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame if it's the right interval
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    # Release video capture
    cap.release()
    
    print(f"\nExtraction complete:")
    print(f"- Processed {frame_count} frames")
    print(f"- Saved {saved_count} frames")
    print(f"- Saved to: {output_dir}")

if __name__ == "__main__":
    # Example usage
    video_path = "test_videos/example.mp4"  # Replace with your video path
    output_dir = "extracted_frames"
    frame_interval = 1  # Extract every frame
    
    extract_frames(video_path, output_dir, frame_interval) 