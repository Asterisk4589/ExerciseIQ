import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "pose_landmarker_full.task"
if not os.path.exists(model_path):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        model_path
    )

def calculate_angle(a, b, c):
    """
    Calculate angle at point B formed by points A-B-C
    a, b, c are [x, y] coordinates
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b  # vector from B to A
    bc = c - b  # vector from B to C

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return round(angle, 1)

def get_coords(landmarks, index, w, h):
    lm = landmarks[index]
    return [int(lm.x * w), int(lm.y * h)]

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]

            # Draw all landmarks
            for landmark in lm:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            # ── RIGHT ELBOW ANGLE (shoulder → elbow → wrist) ──
            r_shoulder = get_coords(lm, 12, w, h)
            r_elbow    = get_coords(lm, 14, w, h)
            r_wrist    = get_coords(lm, 16, w, h)
            elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

            # ── RIGHT KNEE ANGLE (hip → knee → ankle) ──
            r_hip   = get_coords(lm, 24, w, h)
            r_knee  = get_coords(lm, 26, w, h)
            r_ankle = get_coords(lm, 28, w, h)
            knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

            # ── Display angles on frame ──
            cv2.putText(frame, f"Elbow: {elbow_angle}",
                (r_elbow[0] - 50, r_elbow[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(frame, f"Knee: {knee_angle}",
                (r_knee[0] - 50, r_knee[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # ── Simple form feedback ──
            feedback = ""
            if knee_angle < 90:
                feedback = "BEND MORE"
                color = (0, 0, 255)
            elif knee_angle > 160:
                feedback = "GOOD STANCE"
                color = (0, 255, 0)
            else:
                feedback = "ADJUST KNEE"
                color = (0, 165, 255)

            cv2.putText(frame, feedback,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        cv2.imshow("DanceIQ - Angle Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()