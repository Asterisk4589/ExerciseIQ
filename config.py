SMOOTH = 5
rep_count = 0
bad_rep_count = 0
bad_form_start = None
BAD_FORM_THRESHOLD = 0.3  # seconds
ready = False
still_start = None
STILL_THRESHOLD = 1.0  # stand still 1 sec to activate
stage = None
current_rep_good = True  # tracks if current rep had any bad form
TOLERANCE = 5  # pixels ahead of toe — increase if too strict
rep_start_time = None
MIN_REP_DURATION = 1.5  # seconds
frame_count = 0
DEPTH_THRESHOLD = 100  # degrees — must go below this to count


angle_buffer = []
angle_history = []
bad_form_frames = []
rep_frames = []


good_messages = [
    "PERFECT REP! KEEP GOING!",
    "THATS THE WAY!",
    "STRONG FORM! ONE MORE!",
    "YES! NAILED IT!",
]

bad_messages = [
    "PUSH THOSE KNEES BACK!",
    "HEELS DOWN, CHEST UP!",
    "CONTROL THE DESCENT!",
    "ALMOST! WATCH THOSE KNEES!",
]