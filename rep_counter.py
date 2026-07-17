# rep_counter.py
"""
Squat rep counting and classification for ExerciseIQ.
Determines descent (DOWN) and ascent (UP) stages and counts reps based on form.
"""

import time
import random
import config
from messages import GOOD_MESSAGES, BAD_MESSAGES

def update_rep_counter(smooth_ang, knee_over_toe, state):
    """
    Evaluates joint angles and knee position to track workout phases.
    Increments good/bad reps based on form rules.

    smooth_ang: smoothed knee joint angle in degrees.
    knee_over_toe: Boolean indicating if knee is currently over the toe.
    state: SessionState object to modify.
    """
    # Flag the entire rep as bad if knee goes over the toe at any point during descent
    if state.stage == "DOWN" and knee_over_toe:
        state.current_rep_good = False

    # Check if user begins descending (angle goes below 90 degrees)
    if smooth_ang < 90:
        if state.stage != "DOWN":
            state.stage = "DOWN"
            state.rep_start_time = time.time()  # Mark start of rep descent

    # Check if user finishes standing back up (angle goes above 160 degrees)
    if smooth_ang > 160 and state.stage == "DOWN":
        rep_duration = time.time() - state.rep_start_time if state.rep_start_time else 0
        
        if rep_duration >= config.MIN_REP_DURATION:
            state.stage = "UP"
            if state.current_rep_good:
                state.rep_count += 1
                print(random.choice(GOOD_MESSAGES))
            else:
                state.bad_rep_count += 1
                print(random.choice(BAD_MESSAGES))
                
            # Reset active rep indicators
            state.current_rep_good = True
            state.bad_form_start = None
        else:
            # Ignore rapid movements or noise
            state.stage = None
            state.rep_start_time = None
