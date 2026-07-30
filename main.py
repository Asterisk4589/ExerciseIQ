# main.py
"""
ExerciseIQ — Real-Time Squat Form Analyser
Orchestration driver: camera → pose → biomechanics → intelligence → UI.

main.py owns:
  - Camera capture lifecycle
  - MediaPipe model loading
  - Frame-level angle buffering and history (for session graphs)
  - SessionState updates from AnalysisResult
  - Renderer calls (draw_skeleton, draw_ui)
  - Post-session graph and report generation

main.py does NOT contain:
  - Rep counting logic
  - Form feedback decisions
  - Movement scoring
  - Knee-over-toe calculation
  - Any exercise-specific thresholds
"""

import time

import cv2
import mediapipe as mp

import config
import constants
from biomechanics import BiomechanicsEngine
from detector import PoseDetector
from geometry import get_landmark_coords, calculate_angle
from graphs import generate_session_graph
from intelligence.squat import SquatAnalyzer
from model import get_landmarker_options
from renderer import draw_skeleton, draw_ui
from report import generate_and_show_report
from state import SessionState
from utils import print_welcome_message


def main():
    # Display startup instructions
    print_welcome_message()

    # Load model settings and initialize session state
    options = get_landmarker_options()
    state   = SessionState()

    # Initialize Intelligence and Biomechanics engines
    biomechanics  = BiomechanicsEngine()
    squat_analyzer = SquatAnalyzer()

    # Configure OpenCV camera capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam. Verify connection and camera permissions.")
        return

    PoseLandmarker = mp.tasks.vision.PoseLandmarker

    print("Loading MediaPipe Pose Landmarker model...")
    with PoseLandmarker.create_from_options(options) as landmarker:
        detector = PoseDetector(landmarker)

        while True:
            # ── 1. Capture frame ───────────────────────────────────────────
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape

            # Convert OpenCV BGR → MediaPipe RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── 2. Detect pose landmarks ───────────────────────────────────
            result = detector.detect(rgb)

            if result.pose_landmarks:
                # Extract first detected pose
                lm = result.pose_landmarks[0]

                # ── 3. Extract pixel-space joint coordinates ───────────────
                hip_pt   = get_landmark_coords(lm, constants.LANDMARK_HIP,   w, h)
                knee_pt  = get_landmark_coords(lm, constants.LANDMARK_KNEE,  w, h)
                ankle_pt = get_landmark_coords(lm, constants.LANDMARK_ANKLE, w, h)
                toe_pt   = get_landmark_coords(lm, constants.LANDMARK_TOE,   w, h)

                if any(pt is None for pt in (hip_pt, knee_pt, ankle_pt, toe_pt)):
                    cv2.putText(frame, "MOVE INTO FRAME", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.imshow("ExerciseIQ — Squat Analyser", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # ── 4. Angle calculation + buffer smoothing ─────────────────
                knee_ang = calculate_angle(hip_pt, knee_pt, ankle_pt)
                state.angle_buffer.append(knee_ang)
                if len(state.angle_buffer) > config.SMOOTH:
                    state.angle_buffer.pop(0)
                smooth_ang = round(
                    sum(state.angle_buffer) / len(state.angle_buffer), 1
                )

                state.angle_history.append(smooth_ang)
                state.frame_count += 1

                # ── 5. Biomechanics update ─────────────────────────────────
                timestamp = time.monotonic()
                bio = biomechanics.update(lm, timestamp, w, h)

                # ── 6. Intelligence update ─────────────────────────────────
                analysis = squat_analyzer.update(
                    bio=bio,
                    smooth_ang=smooth_ang,
                    knee_pt=knee_pt,
                    toe_pt=toe_pt,
                    frame_count=state.frame_count,
                )

                # ── 7. Sync SessionState from AnalysisResult ───────────────
                state.stage         = analysis.stage
                state.rep_count     = analysis.rep_number
                state.bad_rep_count = analysis.bad_rep_count
                state.bad_form_frames.extend(analysis.new_bad_form_frames)
                state.rep_frames.extend(analysis.new_rep_frames)

                # ── 8. Render ──────────────────────────────────────────────
                draw_skeleton(
                    frame, lm,
                    hip_pt, knee_pt, ankle_pt, toe_pt,
                    w, h,
                    analysis.knee_over_toe,
                    analysis.toe_line_x,
                )
                draw_ui(
                    frame,
                    smooth_ang,
                    analysis.feedback.text,
                    analysis.form_col_bgr,
                    knee_pt,
                    state,
                )

            # ── 9. Display frame ───────────────────────────────────────────
            cv2.imshow("ExerciseIQ — Squat Analyser", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Resource teardown
    cap.release()
    cv2.destroyAllWindows()

    # ── 10. Post-session graph and report ─────────────────────────────────
    generate_session_graph(state)
    generate_and_show_report(state)


if __name__ == "__main__":
    main()
