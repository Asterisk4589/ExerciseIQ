# intelligence/feedback_generator.py
"""
FeedbackGenerator — coaching text and color decisions for ExerciseIQ.

Responsibility
--------------
Consume the raw form-analysis state for one frame and return a
FeedbackMessage (text + is_good flag).  Also owns the message pools
previously scattered across messages.py and form_analysis.py.

Design constraints
------------------
- NO imports of cv2, mediapipe, matplotlib, or any drawing/GUI library.
- NO printing.
- NO timing logic — bad_form_elapsed is passed in from SquatAnalyzer.
- Message text is preserved verbatim from the original code.

Message decision tree (PRESERVED from original form_analysis.py)
------------------------------------------------------------------
if knee_over_toe and elapsed >= BAD_FORM_THRESHOLD:
    → "PUSH THOSE KNEES BACK! YOU GOT THIS!"   (red)
else:
    → "GOOD FORM"                               (green)

On rep_complete:
    if good_rep → random choice from GOOD_MESSAGES pool
    else        → random choice from BAD_MESSAGES pool

The rep-complete message is printed to the console (matching the original
rep_counter.py:39,42 print calls) — this is the only side-effect allowed
in this layer.
"""

from __future__ import annotations

import random
from typing import Optional

import config
from intelligence.models import FeedbackMessage


# ---------------------------------------------------------------------------
# Message pools — moved verbatim from messages.py
# ---------------------------------------------------------------------------

GOOD_MESSAGES = [
    "PERFECT REP! KEEP GOING!",
    "THATS THE WAY!",
    "STRONG FORM! ONE MORE!",
    "YES! NAILED IT!",
]

BAD_MESSAGES = [
    "PUSH THOSE KNEES BACK!",
    "HEELS DOWN, CHEST UP!",
    "CONTROL THE DESCENT!",
    "ALMOST! WATCH THOSE KNEES!",
]


class FeedbackGenerator:
    """
    Produces one FeedbackMessage per frame.

    Stateful only because it needs to remember the last rep-complete
    message so it can persist it on screen for a short number of frames.
    (The base implementation does not yet implement persistence — that
    is a UI concern.  The generator simply produces the current-frame
    message.)
    """

    def generate(
        self,
        knee_over_toe: bool,
        bad_form_elapsed: float,
        rep_complete: bool,
        good_rep: bool,
    ) -> FeedbackMessage:
        """
        Decide and return the appropriate FeedbackMessage for this frame.

        Parameters
        ----------
        knee_over_toe : bool
            Whether the right knee is ahead of the right toe.
        bad_form_elapsed : float
            Seconds elapsed since bad form started (0 if no bad form).
        rep_complete : bool
            True on the frame a rep was confirmed finished.
        good_rep : bool
            Valid only when rep_complete is True.
            True → good rep completed; False → bad rep completed.

        Returns
        -------
        FeedbackMessage
        """
        # ── Rep-completion message (highest priority) ──────────────────────
        # Print to console — preserves original rep_counter.py:39,42 behavior
        if rep_complete:
            if good_rep:
                text = random.choice(GOOD_MESSAGES)
                print(text)
                return FeedbackMessage(text=text, is_good=True)
            else:
                text = random.choice(BAD_MESSAGES)
                print(text)
                return FeedbackMessage(text=text, is_good=False)

        # ── Per-frame form feedback ────────────────────────────────────────
        # Preserved verbatim from form_analysis.py:35-46
        if knee_over_toe and bad_form_elapsed >= config.BAD_FORM_THRESHOLD:
            return FeedbackMessage(
                text="PUSH THOSE KNEES BACK! YOU GOT THIS!",
                is_good=False,
            )

        return FeedbackMessage(text="GOOD FORM", is_good=True)
