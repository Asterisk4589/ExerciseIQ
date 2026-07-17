# form_analysis.py
"""
Squat form analysis and alignment tracking for ExerciseIQ.
Checks joint positioning to alert the user about knee-over-toe posture.
"""

import time
import config

def analyze_form(toe_pt, knee_pt, smooth_ang, state):
    """
    Checks if the knee crosses the toe vertical line.
    Manages timers for form feedback messages and tracks bad form frames.

    toe_pt: list or tuple [x, y] of the toe landmark coordinate.
    knee_pt: list or tuple [x, y] of the knee landmark coordinate.
    smooth_ang: smoothed knee joint angle in degrees.
    state: SessionState object to record form violations.

    Returns:
        form_msg (str): User-facing alert text.
        form_col (tuple): BGR color tuple for UI overlays.
        knee_over_toe (bool): True if knee is positioned ahead of the toe limit.
        toe_line_x (int): Horizontal pixel offset of the toe boundary.
    """
    # Define vertical boundary at toe coordinate plus tolerance
    toe_line_x = toe_pt[0] + config.TOLERANCE
    knee_over_toe = knee_pt[0] > toe_line_x
    
    if knee_over_toe:
        if state.bad_form_start is None:
            state.bad_form_start = time.time()
        elapsed = time.time() - state.bad_form_start

        if elapsed >= config.BAD_FORM_THRESHOLD:
            form_msg = "PUSH THOSE KNEES BACK! YOU GOT THIS!"
            form_col = (0, 0, 255)  # Red BGR
            state.current_rep_good = False
        else:
            form_msg = "GOOD FORM"
            form_col = (0, 255, 0)  # Green BGR
    else:
        state.bad_form_start = None
        elapsed = 0.0
        form_msg = "GOOD FORM"
        form_col = (0, 255, 0)  # Green BGR

    # Track bad form frame counts
    if knee_over_toe and elapsed >= config.BAD_FORM_THRESHOLD:
        state.bad_form_frames.append(state.frame_count)

    # Track deep rep frames (within correct depth threshold and active good descent)
    if smooth_ang < config.DEPTH_THRESHOLD and state.stage == "DOWN" and state.current_rep_good:
        state.rep_frames.append(state.frame_count)

    return form_msg, form_col, knee_over_toe, toe_line_x