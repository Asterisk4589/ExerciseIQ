# config.py
"""
Configuration parameters, thresholds, and model paths for ExerciseIQ.
These values can be tweaked by users to adjust model sensitivity.
"""

# Pose smoothing buffer size (number of frames for moving average)
SMOOTH = 5

# Knee over toe check tolerance (pixels ahead of toe)
TOLERANCE = 5

# Duration (seconds) of bad form before displaying warning overlay
BAD_FORM_THRESHOLD = 0.3

# Standing still duration (seconds) required to activate squat detection
STILL_THRESHOLD = 1.0

# Minimum duration (seconds) of a valid squat repetition to filter out jitter/noise
MIN_REP_DURATION = 1.5

# Knee depth angle threshold (degrees) that must be passed to count a rep's depth
DEPTH_THRESHOLD = 100

# MediaPipe Pose Landmarker task settings
MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"