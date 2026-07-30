# renderer.py
"""
UI and skeleton rendering overlays for ExerciseIQ.
Draws visual helpers, landmark points, reference boundaries, rep score overlays,
and top-right depth progress bar.
"""

import cv2
import config
import constants

def draw_depth_bar(frame, smooth_ang, w):
    """
    Draws a depth progress bar in the top-right corner of the frame.
    Fills from 0% (standing at STAND_THRESHOLD) to 100% (target depth at DEPTH_THRESHOLD).
    Fills orange during descent and turns green once DEPTH_THRESHOLD is reached.
    Displays a percentage label next to the bar.
    """
    stand_ang = getattr(config, "STAND_THRESHOLD", 160)
    depth_thresh = getattr(config, "DEPTH_THRESHOLD", 100)

    # Calculate depth progress percentage [0, 100]
    depth_progress = max(0, min(100, int((stand_ang - smooth_ang) / (stand_ang - depth_thresh) * 100)))

    # Bar dimensions and position (top-right corner)
    bar_width = 180
    bar_height = 22
    margin_right = 30
    margin_top = 30

    x_start = w - margin_right - bar_width
    y_start = margin_top

    # Fill width in pixels
    fill_width = int(bar_width * (depth_progress / 100.0))

    # Background bar (dark gray)
    cv2.rectangle(frame, (x_start, y_start), (x_start + bar_width, y_start + bar_height), constants.COLOR_DARK_GRAY, -1)

    # Progress bar fill (green if at/below depth threshold, orange if descending)
    fill_col = constants.COLOR_GREEN if smooth_ang <= depth_thresh else constants.COLOR_ORANGE
    if fill_width > 0:
        cv2.rectangle(frame, (x_start, y_start), (x_start + fill_width, y_start + bar_height), fill_col, -1)

    # Border rectangle (light gray)
    cv2.rectangle(frame, (x_start, y_start), (x_start + bar_width, y_start + bar_height), constants.COLOR_LIGHT_GRAY, 1)

    # Percentage label to the left of the progress bar
    label_text = f"DEPTH {depth_progress}%"
    cv2.putText(frame, label_text, (x_start - 120, y_start + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, constants.COLOR_WHITE, 2)

def draw_skeleton(frame, lm, hip_pt, knee_pt, ankle_pt, toe_pt, w, h, knee_over_toe, toe_line_x):
    """
    Draws pose skeleton bones, vertical toe lines, and highlight joint circles.
    """
    if lm is None:
        return

    # Draw green circles on visible pose landmarks
    vis_min = getattr(config, "VISIBILITY_MIN", 0.6)
    for p in lm:
        if hasattr(p, "visibility") and p.visibility is not None:
            if p.visibility < vis_min:
                continue
        cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, constants.COLOR_GREEN, -1)

    # Draw white bone connections
    if hip_pt and knee_pt:
        cv2.line(frame, tuple(hip_pt), tuple(knee_pt), constants.COLOR_WHITE, 2)
    if knee_pt and ankle_pt:
        cv2.line(frame, tuple(knee_pt), tuple(ankle_pt), constants.COLOR_WHITE, 2)
    if ankle_pt and toe_pt:
        cv2.line(frame, tuple(ankle_pt), tuple(toe_pt), constants.COLOR_WHITE, 2)

    # Draw vertical reference line at the toe
    if toe_line_x is not None:
        cv2.line(frame, (toe_line_x, 0), (toe_line_x, h), constants.COLOR_CYAN, 2)

    # Draw right knee joint dot (red if over toe, green if correct)
    if knee_pt:
        knee_dot_col = constants.COLOR_RED if knee_over_toe else constants.COLOR_GREEN
        cv2.circle(frame, tuple(knee_pt), 10, knee_dot_col, -1)

def draw_ui(frame, smooth_ang, form_msg, form_col, knee_pt, state):
    """
    Draws progress bar HUD, status counters, real-time posture messages,
    and top-right depth progress bar.
    """
    h, w, _ = frame.shape

    # 1. Top-right depth progress bar
    draw_depth_bar(frame, smooth_ang, w)

    # 2. Bottom-left depth progress bar HUD
    stand_ang = getattr(config, "STAND_THRESHOLD", 160)
    depth_thresh = getattr(config, "DEPTH_THRESHOLD", 100)
    depth_progress = max(0, min(100, int((stand_ang - smooth_ang) / (stand_ang - depth_thresh) * 100)))
    bar_width = int(200 * depth_progress / 100)

    # Background bar (dark gray)
    cv2.rectangle(frame, (30, 220), (230, 245), constants.COLOR_DARK_GRAY, -1)

    # Progress bar fill (green if deep enough, orange if descending/partial depth)
    bar_col = constants.COLOR_GREEN if smooth_ang <= depth_thresh else constants.COLOR_ORANGE
    cv2.rectangle(frame, (30, 220), (30 + bar_width, 245), bar_col, -1)

    # HUD title
    cv2.putText(frame, "DEPTH", (30, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, constants.COLOR_LIGHT_GRAY, 1)

    # Form correction message overlay
    cv2.putText(frame, form_msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, form_col, 3)

    # Dynamic angle label next to knee
    if knee_pt:
        cv2.putText(frame, f"Knee Angle: {smooth_ang}", (knee_pt[0] - 80, knee_pt[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, constants.COLOR_YELLOW, 2)

    # Session stats HUD
    cv2.putText(frame, f"Stage: {state.stage}", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, constants.COLOR_LIGHT_GRAY, 2)
    cv2.putText(frame, f"Good Reps: {state.rep_count}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, constants.COLOR_GREEN, 2)
    cv2.putText(frame, f"Bad Reps:  {state.bad_rep_count}", (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 1.0, constants.COLOR_RED, 2)