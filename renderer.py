# renderer.py
"""
UI and skeleton rendering overlays for ExerciseIQ.
Draws visual helpers, landmark points, reference boundaries, and rep score overlays.
"""

import cv2
import config
import constants

def draw_skeleton(frame, lm, hip_pt, knee_pt, ankle_pt, toe_pt, w, h, knee_over_toe, toe_line_x):
    """
    Draws pose skeleton bones, vertical toe lines, and highlight joint circles.
    """
    # Draw green circles on all pose landmarks
    for p in lm:
        cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, constants.COLOR_GREEN, -1)

    # Draw white bone connections
    cv2.line(frame, tuple(hip_pt), tuple(knee_pt), constants.COLOR_WHITE, 2)
    cv2.line(frame, tuple(knee_pt), tuple(ankle_pt), constants.COLOR_WHITE, 2)
    cv2.line(frame, tuple(ankle_pt), tuple(toe_pt), constants.COLOR_WHITE, 2)

    # Draw vertical reference line at the toe
    cv2.line(frame, (toe_line_x, 0), (toe_line_x, h), constants.COLOR_CYAN, 2)

    # Draw right knee joint dot (red if over toe, green if correct)
    knee_dot_col = constants.COLOR_RED if knee_over_toe else constants.COLOR_GREEN
    cv2.circle(frame, tuple(knee_pt), 10, knee_dot_col, -1)

def draw_ui(frame, smooth_ang, form_msg, form_col, knee_pt, state):
    """
    Draws progress bar HUD, status counters, and real-time posture messages.
    """
    # Calculate depth progress percent based on stand angle (160) and target depth (100)
    depth_progress = max(0, min(100, int((160 - smooth_ang) / (160 - config.DEPTH_THRESHOLD) * 100)))
    bar_width = int(200 * depth_progress / 100)

    # Draw background bar (dark gray)
    cv2.rectangle(frame, (30, 220), (230, 245), constants.COLOR_DARK_GRAY, -1)
    
    # Draw progress bar fill (green if deep enough, orange if descending/partial depth)
    bar_col = constants.COLOR_GREEN if smooth_ang < config.DEPTH_THRESHOLD else constants.COLOR_ORANGE
    cv2.rectangle(frame, (30, 220), (30 + bar_width, 245), bar_col, -1)
    
    # HUD title
    cv2.putText(frame, "DEPTH", (30, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, constants.COLOR_LIGHT_GRAY, 1)

    # Form correction message overlay
    cv2.putText(frame, form_msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, form_col, 3)

    # Dynamic angle label next to knee
    cv2.putText(frame, f"Knee Angle: {smooth_ang}", (knee_pt[0] - 80, knee_pt[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, constants.COLOR_YELLOW, 2)

    # Session stats HUD
    cv2.putText(frame, f"Stage: {state.stage}", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, constants.COLOR_LIGHT_GRAY, 2)
    cv2.putText(frame, f"Good Reps: {state.rep_count}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, constants.COLOR_GREEN, 2)
    cv2.putText(frame, f"Bad Reps:  {state.bad_rep_count}", (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 1.0, constants.COLOR_RED, 2)