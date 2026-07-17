# utils.py
"""
Utility helpers and general session logger functions for ExerciseIQ.
"""

def print_welcome_message():
    """
    Prints a clean console header when ExerciseIQ starts.
    """
    print("=" * 60)
    print("              ExerciseIQ — Squat Analyser")
    print("  Real-time pose assessment, knee tracking & rep scoring")
    print("=" * 60)
    print("Instructions:")
    print("  1. Stand sideways to the camera (right side facing).")
    print("  2. Keep your full body in the frame (head to toes).")
    print("  3. Press 'q' in the camera window to finish and view reports.")
    print("-" * 60)
