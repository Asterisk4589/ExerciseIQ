# detector.py
"""
Pose estimation wrapper using MediaPipe Pose Landmarker.
"""

import mediapipe as mp

class PoseDetector:
    def __init__(self, landmarker):
        """
        Initializes the PoseDetector with an active MediaPipe PoseLandmarker instance.
        """
        self.landmarker = landmarker

    def detect(self, rgb_frame):
        """
        Performs pose landmark detection on the given RGB frame.
        rgb_frame: NumPy array of shape (H, W, 3) in RGB format.
        Returns: PoseLandmarkerResult object containing pose landmarks.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.landmarker.detect(mp_image)