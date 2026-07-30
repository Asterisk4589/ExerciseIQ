# config.py
"""
Configuration parameters, thresholds, and model paths for ExerciseIQ.
These values can be tweaked by users to adjust model sensitivity.
"""

# Pose smoothing buffer size (number of frames for moving average)
SMOOTH_FRAMES = 7
SMOOTH = SMOOTH_FRAMES

# Landmark visibility threshold (minimum confidence required)
VISIBILITY_MIN = 0.6

# Knee over toe check tolerance (pixels ahead of toe line)
TOLERANCE = 3

# Duration (seconds) of bad form before displaying warning overlay
BAD_FORM_SECS = 0.3
BAD_FORM_THRESHOLD = BAD_FORM_SECS

# Standing still duration (seconds) required to activate squat detection
STILL_THRESHOLD = 1.0

# Minimum duration (seconds) of a valid squat repetition to filter out jitter/noise
MIN_REP_SECS = 1.5
MIN_REP_DURATION = MIN_REP_SECS

# Knee depth angle threshold (degrees) that must be passed to count a rep's depth
DEPTH_THRESHOLD = 100

# Standing knee angle threshold (degrees)
STAND_THRESHOLD = 160

# MediaPipe Pose Landmarker task settings
MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

# Biomechanics Engine Configuration
BIOMECH_HISTORY_SIZE = 60         # Configurable history size (frames) for velocity/acceleration
BIOMECH_SMOOTHING_WINDOW = 5      # Rolling average window size for joint angles smoothing
BIOMECH_STABILITY_WINDOW = 30     # Rolling window size for computing stability/variance metrics
BIOMECH_MIN_CONFIDENCE = 0.5      # Minimum joint confidence score threshold
BIOMECH_SMOOTHNESS_NORM = 100.0   # Normalization scale for angular-acceleration → smoothness score
BIOMECH_STABILITY_NORM = 10.0     # Normalization scale for torso-lean + center-X std → stability score
BIOMECH_ACTIVE_SPEED_THRESHOLD = 15.0  # Speed threshold (px/s) for time-under-tension accumulator