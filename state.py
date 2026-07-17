# state.py
"""
SessionState manages the mutable runtime state of the ExerciseIQ workout session.
This encapsulates state variables and avoids global variable modification.
"""

class SessionState:
    def __init__(self):
        # Rep counters
        self.rep_count = 0
        self.bad_rep_count = 0

        # Form timers & checks
        self.bad_form_start = None
        self.ready = False
        self.still_start = None
        self.stage = None  # Tracks 'DOWN' or 'UP' or None
        self.current_rep_good = True  # Tracks if the current rep has stayed correct
        self.rep_start_time = None
        self.frame_count = 0

        # Buffers and historical data for timelines/graphs
        self.angle_buffer = []
        self.angle_history = []
        self.bad_form_frames = []
        self.rep_frames = []
