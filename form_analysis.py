# form_analysis.py
"""
Form analysis compatibility shim for ExerciseIQ.
Delegates to intelligence.squat.SquatAnalyzer for backwards compatibility.
"""

from intelligence.squat import SquatAnalyzer

_global_analyzer = SquatAnalyzer()

def analyze_form(toe_pt, knee_pt, smooth_ang, state):
    """
    Backwards-compatible wrapper around SquatAnalyzer.
    """
    if state.bad_form_start is not None and _global_analyzer._bad_form_start is None:
        _global_analyzer._bad_form_start = state.bad_form_start

    res = _global_analyzer.update(
        bio={},
        smooth_ang=smooth_ang,
        knee_pt=knee_pt,
        toe_pt=toe_pt,
        frame_count=state.frame_count,
    )

    state.bad_form_start = _global_analyzer._bad_form_start
    return res.feedback.text, res.form_col_bgr, res.knee_over_toe, res.toe_line_x