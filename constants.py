# constants.py
"""
System-wide constants for ExerciseIQ.
Defines colors, MediaPipe Pose landmark indices, and other static settings.
"""

# BGR Colors for OpenCV Drawing
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_GOLD = (255, 215, 0)
COLOR_DARK_GRAY = (50, 50, 50)
COLOR_GRAY = (100, 100, 100)
COLOR_LIGHT_GRAY = (200, 200, 200)
COLOR_ORANGE = (0, 165, 255)

# MediaPipe Pose Landmark Indices (Right side)
LANDMARK_HIP = 24
LANDMARK_KNEE = 26
LANDMARK_ANKLE = 28
LANDMARK_TOE = 32
