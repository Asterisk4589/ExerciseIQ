# # main.py
# """
# ExerciseIQ — Real-Time Squat Form Analyser
# Orchestration driver: camera → pose → biomechanics → intelligence → UI.

# main.py owns:
#   - Camera capture lifecycle
#   - MediaPipe model loading
#   - Frame-level angle buffering and history (for session graphs)
#   - SessionState updates from AnalysisResult
#   - Renderer calls (draw_skeleton, draw_ui)
#   - Post-session graph and report generation

# main.py does NOT contain:
#   - Rep counting logic
#   - Form feedback decisions
#   - Movement scoring
#   - Knee-over-toe calculation
#   - Any exercise-specific thresholds
# """

# import time

# import cv2
# import mediapipe as mp

# import config
# import constants
# from biomechanics import BiomechanicsEngine
# from detector import PoseDetector
# from geometry import get_landmark_coords, calculate_angle
# from graphs import generate_session_graph
# from intelligence.squat import SquatAnalyzer
# from model import get_landmarker_options
# from renderer import draw_skeleton, draw_ui
# from report import generate_and_show_report
# from state import SessionState
# from utils import print_welcome_message


# def main():
#     # Display startup instructions
#     print_welcome_message()

#     # Load model settings and initialize session state
#     options = get_landmarker_options()
#     state   = SessionState()

#     # Initialize Intelligence and Biomechanics engines
#     biomechanics  = BiomechanicsEngine()
#     squat_analyzer = SquatAnalyzer()

#     # Configure OpenCV camera capture
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not access the webcam. Verify connection and camera permissions.")
#         return

#     PoseLandmarker = mp.tasks.vision.PoseLandmarker

#     print("Loading MediaPipe Pose Landmarker model...")
#     with PoseLandmarker.create_from_options(options) as landmarker:
#         detector = PoseDetector(landmarker)

#         while True:
#             # ── 1. Capture frame ───────────────────────────────────────────
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             h, w, _ = frame.shape

#             # Convert OpenCV BGR → MediaPipe RGB
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             # ── 2. Detect pose landmarks ───────────────────────────────────
#             result = detector.detect(rgb)

#             if result.pose_landmarks:
#                 # Extract first detected pose
#                 lm = result.pose_landmarks[0]

#                 # ── 3. Extract pixel-space joint coordinates ───────────────
#                 hip_pt   = get_landmark_coords(lm, constants.LANDMARK_HIP,   w, h)
#                 knee_pt  = get_landmark_coords(lm, constants.LANDMARK_KNEE,  w, h)
#                 ankle_pt = get_landmark_coords(lm, constants.LANDMARK_ANKLE, w, h)
#                 toe_pt   = get_landmark_coords(lm, constants.LANDMARK_TOE,   w, h)

#                 # ── 4. Angle calculation + buffer smoothing ─────────────────
#                 knee_ang = calculate_angle(hip_pt, knee_pt, ankle_pt)
#                 state.angle_buffer.append(knee_ang)
#                 if len(state.angle_buffer) > config.SMOOTH:
#                     state.angle_buffer.pop(0)
#                 smooth_ang = round(
#                     sum(state.angle_buffer) / len(state.angle_buffer), 1
#                 )

#                 state.angle_history.append(smooth_ang)
#                 state.frame_count += 1

#                 # ── 5. Biomechanics update ─────────────────────────────────
#                 timestamp = time.monotonic()
#                 bio = biomechanics.update(lm, timestamp, w, h)

#                 # ── 6. Intelligence update ─────────────────────────────────
#                 analysis = squat_analyzer.update(
#                     bio=bio,
#                     smooth_ang=smooth_ang,
#                     knee_pt=knee_pt,
#                     toe_pt=toe_pt,
#                     frame_count=state.frame_count,
#                 )

#                 # ── 7. Sync SessionState from AnalysisResult ───────────────
#                 state.stage         = analysis.stage
#                 state.rep_count     = analysis.rep_number
#                 state.bad_rep_count = analysis.bad_rep_count
#                 state.bad_form_frames.extend(analysis.new_bad_form_frames)
#                 state.rep_frames.extend(analysis.new_rep_frames)

#                 # ── 8. Render ──────────────────────────────────────────────
#                 draw_skeleton(
#                     frame, lm,
#                     hip_pt, knee_pt, ankle_pt, toe_pt,
#                     w, h,
#                     analysis.knee_over_toe,
#                     analysis.toe_line_x,
#                 )
#                 draw_ui(
#                     frame,
#                     smooth_ang,
#                     analysis.feedback.text,
#                     analysis.form_col_bgr,
#                     knee_pt,
#                     state,
#                 )

#             # ── 9. Display frame ───────────────────────────────────────────
#             cv2.imshow("ExerciseIQ — Squat Analyser", frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     # Resource teardown
#     cap.release()
#     cv2.destroyAllWindows()

#     # ── 10. Post-session graph and report ─────────────────────────────────
#     generate_session_graph(state)
#     generate_and_show_report(state)


# if __name__ == "__main__":
#     main()

import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
import time
import random
import matplotlib.pyplot as plt

# ── constants ──────────────────────────────────────────────
MODEL_PATH       = "pose_landmarker_full.task"
SMOOTH_FRAMES    = 7        # rolling average window
BAD_FORM_SECS    = 0.3      # seconds knee must be over toe → bad form
MIN_REP_SECS     = 1.5      # minimum squat duration → filters noise
DEPTH_THRESHOLD  = 100      # degrees — must go below to count rep
STAND_THRESHOLD  = 160      # degrees — counts as standing
TOLERANCE        = 3        # pixels of forgiveness on toe line
VISIBILITY_MIN   = 0.6      # landmark confidence floor

# ── motivation ─────────────────────────────────────────────
GOOD_MSGS = [
    "PERFECT REP! KEEP GOING!",
    "THATS THE WAY!",
    "STRONG FORM! ONE MORE!",
    "YES! NAILED IT!",
]
BAD_MSGS = [
    "PUSH THOSE KNEES BACK!",
    "HEELS DOWN, CHEST UP!",
    "CONTROL THE DESCENT!",
    "ALMOST! WATCH THOSE KNEES!",
]

# ── download model ─────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print("Downloading pose model (~30MB)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        MODEL_PATH
    )
    print("Done.")

# ── mediapipe setup ────────────────────────────────────────
BaseOptions          = mp.tasks.BaseOptions
PoseLandmarker       = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions= mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode    = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE
)

# ── helpers ────────────────────────────────────────────────
def get_coords(lm, idx, w, h):
    """Return pixel coords if landmark visible, else None."""
    p = lm[idx]
    if p.visibility < VISIBILITY_MIN:
        return None
    return [int(p.x * w), int(p.y * h)]

def calc_angle(a, b, c):
    """Angle at point B formed by A-B-C vectors."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cos     = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return round(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))), 1)

def draw_text(frame, text, pos, scale=0.8, color=(255,255,255), thickness=2):
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def draw_depth_bar(frame, smooth_ang, w):
    """Progress bar — fills as squat deepens."""
    progress   = max(0.0, min(1.0, (STAND_THRESHOLD - smooth_ang) /
                                   (STAND_THRESHOLD - DEPTH_THRESHOLD)))
    bar_x, bar_y, bar_w, bar_h = w - 220, 80, 180, 22
    filled     = int(bar_w * progress)
    bar_col    = (0, 255, 0) if smooth_ang < DEPTH_THRESHOLD else (0, 165, 255)

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (50,50,50), -1)
    if filled > 0:
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + filled, bar_y + bar_h), bar_col, -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (120,120,120), 1)
    draw_text(frame, "DEPTH", (bar_x, bar_y - 8),
              scale=0.55, color=(200,200,200), thickness=1)
    pct = int(progress * 100)
    draw_text(frame, f"{pct}%", (bar_x + bar_w + 6, bar_y + 16),
              scale=0.55, color=bar_col, thickness=1)

def show_session_report(rep_count, bad_rep_count):
    total    = rep_count + bad_rep_count
    accuracy = round((rep_count / total) * 100) if total > 0 else 0

    if   accuracy >= 80: verdict = "GREAT SESSION! CONSISTENCY IS KEY!"
    elif accuracy >= 50: verdict = "GOOD EFFORT! WATCH THOSE KNEES!"
    else:                verdict = "KEEP PRACTICING! FORM COMES WITH REPS!"

    report = np.zeros((500, 700, 3), dtype=np.uint8)
    cv2.line(report, (50, 85),  (650, 85),  (50,50,50), 1)
    cv2.line(report, (50, 330), (650, 330), (50,50,50), 1)

    draw_text(report, "SESSION COMPLETE",
              (170, 60), scale=1.2, color=(0,255,255), thickness=3)
    draw_text(report, f"Total Reps   : {total}",
              (80, 145), color=(255,255,255))
    draw_text(report, f"Good Reps    : {rep_count}",
              (80, 195), color=(0,255,0))
    draw_text(report, f"Bad Reps     : {bad_rep_count}",
              (80, 245), color=(0,0,255))
    draw_text(report, f"Accuracy     : {accuracy}%",
              (80, 295), color=(0,255,255))
    draw_text(report, verdict,
              (80, 380), scale=0.75, color=(255,215,0), thickness=2)
    draw_text(report, "Press any key to exit",
              (220, 460), scale=0.6, color=(100,100,100), thickness=1)

    cv2.imshow("ExerciseIQ — Session Report", report)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_angle_graph(angle_history, bad_frames, rep_frames):
    if not angle_history:
        return
    plt.figure(figsize=(13, 5))
    plt.style.use("dark_background")

    frames = list(range(len(angle_history)))

    # smooth line
    plt.plot(frames, angle_history,
             color="#00FF88", linewidth=1.2, label="Knee Angle", zorder=3)

    # bad form shading
    for f in bad_frames:
        plt.axvspan(max(0, f-3), min(len(frames), f+3),
                    color="red", alpha=0.25)

    # rep completion markers
    for i, f in enumerate(rep_frames):
        plt.axvline(x=f, color="cyan",
                    linewidth=1.2, linestyle="--", alpha=0.6)
        plt.text(f+2, max(angle_history)-5, f"R{i+1}",
                 color="cyan", fontsize=7)

    # threshold lines
    plt.axhline(y=DEPTH_THRESHOLD, color="yellow",
                linewidth=1, linestyle=":", label=f"Depth threshold ({DEPTH_THRESHOLD}°)")
    plt.axhline(y=STAND_THRESHOLD, color="orange",
                linewidth=1, linestyle=":", label=f"Standing ({STAND_THRESHOLD}°)")

    # fill under curve
    plt.fill_between(frames, angle_history,
                     alpha=0.08, color="#00FF88")

    plt.title("ExerciseIQ — Knee Angle Timeline", color="white", fontsize=14, pad=12)
    plt.xlabel("Frame",        color="white")
    plt.ylabel("Knee Angle °", color="white")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig("session_graph.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Graph saved → session_graph.png")

# ── state ──────────────────────────────────────────────────
angle_buffer   = []
angle_history  = []
bad_frames     = []
rep_frames     = []
frame_count    = 0

rep_count      = 0
bad_rep_count  = 0
stage          = None
current_rep_good = True
bad_form_start = None
rep_start_time = None

# ── main loop ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("Stand SIDEWAYS — right knee facing camera. Press Q to quit.")

with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result   = landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )
        frame_count += 1

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]

            hip_pt   = get_coords(lm, 24, w, h)
            knee_pt  = get_coords(lm, 26, w, h)
            ankle_pt = get_coords(lm, 28, w, h)
            toe_pt   = get_coords(lm, 32, w, h)

            if None in [hip_pt, knee_pt, ankle_pt, toe_pt]:
                draw_text(frame, "MOVE INTO FRAME",
                          (w//2 - 130, h//2), color=(0,165,255))
                cv2.imshow("ExerciseIQ — Squat Analyser", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # ── knee angle + rolling average ───────────────
            raw_ang = calc_angle(hip_pt, knee_pt, ankle_pt)
            angle_buffer.append(raw_ang)
            if len(angle_buffer) > SMOOTH_FRAMES:
                angle_buffer.pop(0)
            smooth_ang = round(sum(angle_buffer) / len(angle_buffer), 1)
            angle_history.append(smooth_ang)

            # ── knee over toe ──────────────────────────────
            toe_line_x    = toe_pt[0] + TOLERANCE
            knee_over_toe = knee_pt[0] > toe_line_x

            if knee_over_toe:
                if bad_form_start is None:
                    bad_form_start = time.time()
                elapsed = time.time() - bad_form_start
                if elapsed >= BAD_FORM_SECS:
                    form_msg  = random.choice(BAD_MSGS)
                    form_col  = (0, 0, 255)
                    current_rep_good = False
                    bad_frames.append(frame_count)
                else:
                    form_msg = "GOOD FORM"
                    form_col = (0, 255, 0)
            else:
                bad_form_start = None
                form_msg  = "GOOD FORM"
                form_col  = (0, 255, 0)

            # ── rep counter ────────────────────────────────
            if smooth_ang < DEPTH_THRESHOLD and stage != "DOWN":
                stage          = "DOWN"
                rep_start_time = time.time()

            if smooth_ang > STAND_THRESHOLD and stage == "DOWN":
                duration = time.time() - rep_start_time if rep_start_time else 0
                if duration >= MIN_REP_SECS:
                    stage = "UP"
                    rep_frames.append(frame_count)
                    if current_rep_good:
                        rep_count += 1
                        print(random.choice(GOOD_MSGS))
                    else:
                        bad_rep_count += 1
                        print(random.choice(BAD_MSGS))
                    current_rep_good = True
                    bad_form_start   = None
                    rep_start_time   = None
                else:
                    stage = None  # too fast = noise

            if stage == "DOWN" and knee_over_toe:
                current_rep_good = False

            # ── draw skeleton ──────────────────────────────
            for p in lm:
                if p.visibility >= VISIBILITY_MIN:
                    cv2.circle(frame,
                               (int(p.x*w), int(p.y*h)),
                               4, (0,255,0), -1)

            cv2.line(frame, hip_pt,   knee_pt,  (255,255,255), 2)
            cv2.line(frame, knee_pt,  ankle_pt, (255,255,255), 2)
            cv2.line(frame, ankle_pt, toe_pt,   (255,255,255), 2)

            # toe reference line
            cv2.line(frame,
                     (toe_line_x, 0), (toe_line_x, h),
                     (255,255,0), 2)

            # knee dot
            knee_col = (0,0,255) if knee_over_toe else (0,255,0)
            cv2.circle(frame, tuple(knee_pt), 10, knee_col, -1)

            # angle arc label at knee
            draw_text(frame, f"{smooth_ang}°",
                      (knee_pt[0]-40, knee_pt[1]-20),
                      scale=0.75, color=(0,255,255))

            # ── HUD ───────────────────────────────────────
            draw_text(frame, form_msg,      (30, 45),  scale=1.0, color=form_col, thickness=3)
            draw_text(frame, f"Stage: {stage}", (30, 90),  scale=0.8, color=(200,200,200))
            draw_text(frame, f"Good: {rep_count}",  (30, 130), scale=0.9, color=(0,255,0))
            draw_text(frame, f"Bad:  {bad_rep_count}",  (30, 170), scale=0.9, color=(0,0,255))

            draw_depth_bar(frame, smooth_ang, w)

        cv2.imshow("ExerciseIQ — Squat Analyser", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

show_session_report(rep_count, bad_rep_count)
show_angle_graph(angle_history, bad_frames, rep_frames)
print(f"\nSession → Good: {rep_count}  Bad: {bad_rep_count}")