import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
import time
import random 

good_messages = [
    "PERFECT REP! KEEP GOING!",
    "THATS THE WAY!",
    "STRONG FORM! ONE MORE!",
    "YES! NAILED IT!",
]

bad_messages = [
    "PUSH THOSE KNEES BACK!",
    "HEELS DOWN, CHEST UP!",
    "CONTROL THE DESCENT!",
    "ALMOST! WATCH THOSE KNEES!",
]

model_path = "pose_landmarker_full.task"
if not os.path.exists(model_path):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        model_path
    )

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return round(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))), 1)

def coords(lm, idx, w, h):
    p = lm[idx]
    return [int(p.x * w), int(p.y * h)]

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

angle_buffer = []
SMOOTH = 5
rep_count = 0
bad_rep_count = 0
bad_form_start = None
BAD_FORM_THRESHOLD = 0.3  # seconds
ready = False
still_start = None
STILL_THRESHOLD = 1.0  # stand still 1 sec to activate
stage = None
current_rep_good = True  # tracks if current rep had any bad form

cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]

            hip_pt   = coords(lm, 24, w, h)
            knee_pt  = coords(lm, 26, w, h)
            ankle_pt = coords(lm, 28, w, h)
            toe_pt   = coords(lm, 32, w, h)

            # knee angle + smoothing
            knee_ang = angle(hip_pt, knee_pt, ankle_pt)
            angle_buffer.append(knee_ang)
            if len(angle_buffer) > SMOOTH:
                angle_buffer.pop(0)
            smooth_ang = round(sum(angle_buffer) / len(angle_buffer), 1)

            # knee over toe check
            knee_over_toe = knee_pt[0] > toe_pt[0]

            if knee_over_toe:
                if bad_form_start is None:
                    bad_form_start = time.time()
                elapsed = time.time() - bad_form_start

                if elapsed >= BAD_FORM_THRESHOLD:
                    form_msg = "PUSH THOSE KNEES BACK! YOU GOT THIS!"
                    form_col = (0, 0, 255)
                    current_rep_good = False
                else:
                    # within threshold — still okay
                    form_msg = "GOOD FORM"
                    form_col = (0, 255, 0)
            else:
                bad_form_start = None  # reset timer when knee back in range
                form_msg = "GOOD FORM"
                form_col = (0, 255, 0)

            # rep counter
            if smooth_ang < 90:
                stage = "DOWN"

            if smooth_ang > 160 and stage == "DOWN":
                stage = "UP"
                if current_rep_good:
                    rep_count += 1
                    # flash good message 2 seconds
                    print(random.choice(good_messages))
                else:
                    bad_rep_count += 1
                    print(random.choice(bad_messages))
                current_rep_good = True
                bad_form_start = None

            # reset flag during descent
            if stage == "DOWN" and knee_over_toe:
                current_rep_good = False

            # ── draw skeleton ──
            for p in lm:
                cv2.circle(frame, (int(p.x*w), int(p.y*h)), 4, (0, 255, 0), -1)

            cv2.line(frame, hip_pt,   knee_pt,  (255, 255, 255), 2)
            cv2.line(frame, knee_pt,  ankle_pt, (255, 255, 255), 2)
            cv2.line(frame, ankle_pt, toe_pt,   (255, 255, 255), 2)

            # toe vertical reference line
            cv2.line(frame, (toe_pt[0], 0), (toe_pt[0], h), (255, 255, 0), 2)

            # knee dot — red if over toe
            knee_dot_col = (0, 0, 255) if knee_over_toe else (0, 255, 0)
            cv2.circle(frame, tuple(knee_pt), 10, knee_dot_col, -1)

            # ── UI ──
            cv2.putText(frame, form_msg,
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, form_col, 3)

            cv2.putText(frame, f"Knee Angle: {smooth_ang}",
                (knee_pt[0]-80, knee_pt[1]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(frame, f"Stage: {stage}",
                (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            cv2.putText(frame, f"Good Reps: {rep_count}",
                (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.putText(frame, f"Bad Reps:  {bad_rep_count}",
                (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("ExerciseIQ — Squat Analyser", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Session complete — Good: {rep_count}  Bad: {bad_rep_count}")