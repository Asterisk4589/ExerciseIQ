# main.py
"""
ExerciseIQ — Real-Time Squat Form Analyser
Orchestration driver program utilizing MediaPipe Pose landmarks and OpenCV captures.
"""

import cv2
import mediapipe as mp
import config
import constants
from state import SessionState
from model import get_landmarker_options
from detector import PoseDetector
from geometry import get_landmark_coords, calculate_angle
from form_analysis import analyze_form
from rep_counter import update_rep_counter
from renderer import draw_skeleton, draw_ui
from graphs import generate_session_graph
from report import generate_and_show_report
from utils import print_welcome_message

def main():
    # Display startup instructions and credentials
    print_welcome_message()
    
    # Load model settings and initialize mutable session state object
    options = get_landmarker_options()
    state = SessionState()
    
    # Configure OpenCV camera device capture feed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam. Verify connection and camera permissions.")
        return

    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    
    print("Loading MediaPipe Pose Landmarker model...")
    with PoseLandmarker.create_from_options(options) as landmarker:
        detector = PoseDetector(landmarker)
        
        while True:
            # 1. Capture frame
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            
            # Convert OpenCV BGR image format to MediaPipe RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 2. Detect pose landmarks
            result = detector.detect(rgb)

            if result.pose_landmarks:
                # Extract first detected pose body landmarks
                lm = result.pose_landmarks[0]

                # 3. Extract joint coordinates
                hip_pt = get_landmark_coords(lm, constants.LANDMARK_HIP, w, h)
                knee_pt = get_landmark_coords(lm, constants.LANDMARK_KNEE, w, h)
                ankle_pt = get_landmark_coords(lm, constants.LANDMARK_ANKLE, w, h)
                toe_pt = get_landmark_coords(lm, constants.LANDMARK_TOE, w, h)

                # 4. Joint Angle Calculation & Average Buffer Smoothing
                knee_ang = calculate_angle(hip_pt, knee_pt, ankle_pt)
                state.angle_buffer.append(knee_ang)
                if len(state.angle_buffer) > config.SMOOTH:
                    state.angle_buffer.pop(0)
                smooth_ang = round(sum(state.angle_buffer) / len(state.angle_buffer), 1)
                
                state.angle_history.append(smooth_ang)
                state.frame_count += 1

                # 5. Analyze Form (knee limit checks & timer trackers)
                form_msg, form_col, knee_over_toe, toe_line_x = analyze_form(
                    toe_pt, knee_pt, smooth_ang, state
                )

                # 6. Update Rep Counter scoring metrics
                update_rep_counter(smooth_ang, knee_over_toe, state)

                # 7. Render UI overlays and bones skeleton lines
                draw_skeleton(frame, lm, hip_pt, knee_pt, ankle_pt, toe_pt, w, h, knee_over_toe, toe_line_x)
                draw_ui(frame, smooth_ang, form_msg, form_col, knee_pt, state)

            # 8. Display output window frame
            cv2.imshow("ExerciseIQ — Squat Analyser", frame)
            
            # Poll for the exit key ('q')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Resource teardown
    cap.release()
    cv2.destroyAllWindows()

    # 9. Session summaries and visual progress graphs
    generate_session_graph(state)
    generate_and_show_report(state)

if __name__ == "__main__":
    main()