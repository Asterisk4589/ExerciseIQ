import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
import time
import random 
import matplotlib.pyplot as plt
import config





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
            config.angle_buffer.append(knee_ang)
            if len(config.angle_buffer) > config.SMOOTH:
                config.angle_buffer.pop(0)
            smooth_ang = round(sum(config.angle_buffer) / len(config.angle_buffer), 1)
            
            # angular calculation
            config.angle_history.append(smooth_ang)
            config.frame_count += 1

            # track bad form frames
            if knee_over_toe and elapsed >= config.BAD_FORM_THRESHOLD if config.bad_form_start else False:
                config.bad_form_frames.append(config.frame_count)

            # track rep completion frames
            if smooth_ang < config.DEPTH_THRESHOLD and config.stage == "DOWN" and config.current_rep_good:
                config.rep_frames.append(config.frame_count)

            # knee over toe check
            toe_line_x = toe_pt[0] + config.TOLERANCE
            knee_over_toe = knee_pt[0] > toe_line_x
            if knee_over_toe:
                if config.bad_form_start is None:
                    config.bad_form_start = time.time()
                elapsed = time.time() - config.bad_form_start

                if elapsed >= config.BAD_FORM_THRESHOLD:
                    form_msg = "PUSH THOSE KNEES BACK! YOU GOT THIS!"
                    form_col = (0, 0, 255)
                    config.current_rep_good = False
                else:
                    # within threshold — still okay
                    form_msg = "GOOD FORM"
                    form_col = (0, 255, 0)
            else:
                config.bad_form_start = None  # reset timer when knee back in range
                form_msg = "GOOD FORM"
                form_col = (0, 255, 0)

            # rep counter
            if smooth_ang < 90:
                if config.stage != "DOWN":
                    config.stage = "DOWN"
                    config.rep_start_time = time.time()  # start timer when descent begins

            if smooth_ang > 160 and config.stage == "DOWN":
                rep_duration = time.time() - config.rep_start_time if config.rep_start_time else 0
                
                if rep_duration >= config.MIN_REP_DURATION:  # only count if took long enough
                    config.stage = "UP"
                    if config.current_rep_good:
                        config.rep_count += 1
                        print(random.choice(config.good_messages))
                    else:
                        config.bad_rep_count += 1
                        print(random.choice(config.bad_messages))
                    config.current_rep_good = True
                    config.bad_form_start = None
                else:
                    config.stage = None  # too fast = noise, reset without counting
                    config.rep_start_time = None

            # reset flag during descent
            if config.stage == "DOWN" and knee_over_toe:
                config.current_rep_good = False

            # ── draw skeleton ──
            for p in lm:
                cv2.circle(frame, (int(p.x*w), int(p.y*h)), 4, (0, 255, 0), -1)

            cv2.line(frame, hip_pt,   knee_pt,  (255, 255, 255), 2)
            cv2.line(frame, knee_pt,  ankle_pt, (255, 255, 255), 2)
            cv2.line(frame, ankle_pt, toe_pt,   (255, 255, 255), 2)

            # toe vertical reference line
            cv2.line(frame, (toe_line_x, 0), (toe_line_x, h), (255, 255, 0), 2)

            # knee dot — red if over toe
            knee_dot_col = (0, 0, 255) if knee_over_toe else (0, 255, 0)
            cv2.circle(frame, tuple(knee_pt), 10, knee_dot_col, -1)

            # ── UI ──
            
            
            # depth progress bar
            depth_progress = max(0, min(100, int((160 - smooth_ang) / (160 - config.DEPTH_THRESHOLD) * 100)))
            bar_width = int(200 * depth_progress / 100)

            # background bar
            cv2.rectangle(frame, (30, 220), (230, 245), (50, 50, 50), -1)
            # fill bar — green when deep enough
            bar_col = (0, 255, 0) if smooth_ang < config.DEPTH_THRESHOLD else (0, 165, 255)
            cv2.rectangle(frame, (30, 220), (30 + bar_width, 245), bar_col, -1)
            cv2.putText(frame, "DEPTH", (30, 215),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            
            cv2.putText(frame, form_msg,
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, form_col, 3)

            cv2.putText(frame, f"Knee Angle: {smooth_ang}",
                (knee_pt[0]-80, knee_pt[1]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(frame, f"Stage: {config.stage}",
                (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            cv2.putText(frame, f"Good Reps: {config.rep_count}",
                (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.putText(frame, f"Bad Reps:  {config.bad_rep_count}",
                (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("ExerciseIQ — Squat Analyser", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
# ── ANGLE TIMELINE GRAPH ──
if len(config.angle_history) > 0:
    plt.figure(figsize=(12, 5))
    plt.style.use('dark_background')
    
    # plot angle line
    plt.plot(config.angle_history, color='#00FF88', linewidth=1.5, label='Knee Angle')
    
    # shade bad form zones red
    for f in config.bad_form_frames:
        plt.axvspan(f-2, f+2, color='red', alpha=0.3)
    
    # mark rep completions
    for f in config.rep_frames:
        plt.axvline(x=f, color='cyan', linewidth=1, linestyle='--', alpha=0.7)
    
    # threshold lines
    plt.axhline(y=90, color='yellow', linewidth=1, linestyle=':', label='Squat Depth (90°)')
    plt.axhline(y=160, color='orange', linewidth=1, linestyle=':', label='Standing (160°)')
    
    plt.title('ExerciseIQ — Knee Angle Timeline', color='white', fontsize=14)
    plt.xlabel('Frame', color='white')
    plt.ylabel('Knee Angle (degrees)', color='white')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('session_graph.png', dpi=150)
    plt.show()
    print("Graph saved as session_graph.png")

# ── SESSION REPORT ──
total_reps = config.rep_count + config.bad_rep_count
accuracy = round((config.rep_count / total_reps) * 100) if total_reps > 0 else 0

if accuracy >= 80:
    verdict = "GREAT SESSION! CONSISTENCY IS KEY!"
elif accuracy >= 50:
    verdict = "GOOD EFFORT! WATCH THOSE KNEES!"
else:
    verdict = "KEEP PRACTICING! FORM COMES WITH REPS!"

# build report frame
report = np.zeros((500, 700, 3), dtype=np.uint8)

cv2.putText(report, "SESSION COMPLETE",
    (170, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

cv2.line(report, (50, 80), (650, 80), (50, 50, 50), 1)

cv2.putText(report, f"Total Reps   : {total_reps}",
    (80, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

cv2.putText(report, f"Good Reps    : {config.rep_count}",
    (80, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.putText(report, f"Bad Reps     : {config.bad_rep_count}",
    (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

cv2.putText(report, f"Accuracy     : {accuracy}%",
    (80, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

cv2.line(report, (50, 320), (650, 320), (50, 50, 50), 1)

cv2.putText(report, verdict,
    (80, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 215, 0), 2)

cv2.putText(report, "Press any key to exit",
    (220, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

cv2.imshow("ExerciseIQ — Session Report", report)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Session complete — Good: {config.rep_count}  Bad: {config.bad_rep_count}  Accuracy: {accuracy}%")