# rep_counter.py
"""
Rep counter compatibility shim for ExerciseIQ.
Delegates to intelligence.rep_counter.RepCounter for backwards compatibility.
"""

from intelligence.rep_counter import RepCounter

_global_counter = RepCounter()

def update_rep_counter(smooth_ang, knee_over_toe, state):
    """
    Backwards-compatible function wrapper for RepCounter.
    """
    # Sync internal state stage if modified externally
    _global_counter._stage = state.stage
    _global_counter._current_rep_good = state.current_rep_good
    _global_counter._rep_start_time = state.rep_start_time
    _global_counter._rep_count = state.rep_count
    _global_counter._bad_rep_count = state.bad_rep_count

    res = _global_counter.update(
        smooth_ang=smooth_ang,
        knee_over_toe=knee_over_toe,
        frame_count=state.frame_count,
    )

    state.stage = res.stage
    state.rep_count = res.rep_count
    state.bad_rep_count = res.bad_rep_count
    state.bad_form_frames.extend(res.new_bad_form_frames)
    state.rep_frames.extend(res.new_rep_frames)
    state.current_rep_good = _global_counter._current_rep_good
    state.rep_start_time = _global_counter._rep_start_time
