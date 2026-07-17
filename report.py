# report.py
"""
Session report visualization for ExerciseIQ.
Generates an OpenCV panel compiling good vs bad reps, accuracy scores, and performance feedback.
"""

import cv2
import numpy as np
import constants

def generate_and_show_report(state):
    """
    Renders the workout session summary block on a black Canvas.
    Displays the window and blocks execution until a key is pressed.
    """
    total_reps = state.rep_count + state.bad_rep_count
    accuracy = round((state.rep_count / total_reps) * 100) if total_reps > 0 else 0

    if accuracy >= 80:
        verdict = "GREAT SESSION! CONSISTENCY IS KEY!"
    elif accuracy >= 50:
        verdict = "GOOD EFFORT! WATCH THOSE KNEES!"
    else:
        verdict = "KEEP PRACTICING! FORM COMES WITH REPS!"

    # Draw summary panel frame (500x700 BGR canvas)
    report_canvas = np.zeros((500, 700, 3), dtype=np.uint8)

    # Draw header text
    cv2.putText(report_canvas, "SESSION COMPLETE",
                (170, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, constants.COLOR_YELLOW, 3)

    # Draw separator line
    cv2.line(report_canvas, (50, 80), (650, 80), constants.COLOR_DARK_GRAY, 1)

    # Print session statistics
    cv2.putText(report_canvas, f"Total Reps   : {total_reps}",
                (80, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, constants.COLOR_WHITE, 2)

    cv2.putText(report_canvas, f"Good Reps    : {state.rep_count}",
                (80, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, constants.COLOR_GREEN, 2)

    cv2.putText(report_canvas, f"Bad Reps     : {state.bad_rep_count}",
                (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, constants.COLOR_RED, 2)

    cv2.putText(report_canvas, f"Accuracy     : {accuracy}%",
                (80, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.9, constants.COLOR_YELLOW, 2)

    # Draw bottom separator line
    cv2.line(report_canvas, (50, 320), (650, 320), constants.COLOR_DARK_GRAY, 1)

    # Draw qualitative verdict overlay
    cv2.putText(report_canvas, verdict,
                (80, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.75, constants.COLOR_GOLD, 2)

    # Draw exit hint
    cv2.putText(report_canvas, "Press any key to exit",
                (220, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, constants.COLOR_GRAY, 1)

    # Show window
    cv2.imshow("ExerciseIQ — Session Report", report_canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Session complete — Good: {state.rep_count}  Bad: {state.bad_rep_count}  Accuracy: {accuracy}%")
