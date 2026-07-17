# model.py
"""
MediaPipe model loader for ExerciseIQ.
Checks for local task models, downloads them if missing, and builds landmarker options.
"""

import os
import urllib.request
import mediapipe as mp
import config

def ensure_model_exists():
    """
    Downloads the pose landmarker task file if it is not present locally.
    """
    if not os.path.exists(config.MODEL_PATH):
        print(f"Model task file '{config.MODEL_PATH}' not found. Downloading...")
        # Create parent directories if any
        parent_dir = os.path.dirname(config.MODEL_PATH)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
            
        urllib.request.urlretrieve(config.MODEL_URL, config.MODEL_PATH)
        print("Download complete!")

def get_landmarker_options():
    """
    Ensures model file exists and returns a PoseLandmarkerOptions object.
    """
    ensure_model_exists()
    
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=config.MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE
    )
    return options