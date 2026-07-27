# intelligence/rep_counter.py
"""
RepCounter — squat repetition detector for ExerciseIQ.

Responsibility
--------------
Maintain UP/DOWN phase state, detect completed repetitions, filter out
noisy micro-movements, and count good vs bad reps.

Design constraints
------------------
- NO imports of cv2, mediapipe, matplotlib, or any drawing/GUI library.
- NO knowledge of feedback text, scores, or UI colors.
- Only answers three questions per frame:
    1. What movement stage are we in?
    2. Did a rep complete this frame?
    3. How many good / bad reps have we counted?

Behavior (PRESERVED VERBATIM from original rep_counter.py)
----------------------------------------------------------
- DOWN stage:  smooth_ang < 90
- UP stage:    smooth_ang > 160  AND  was in DOWN
- Bad rep:     knee went over toe at any point during a DOWN phase
- Ignored rep: rep_duration < config.MIN_REP_DURATION  (noise filter)
- Stage reset to None on noise detection (preserves original behavior)
"""

from __future__ import annotations

import time
from typing import List, Optional

import config
from intelligence.models import RepCounterResult


class RepCounter:
    """
    Stateful squat repetition counter.

    Usage
    -----
    counter = RepCounter()
    while running:
        result = counter.update(smooth_ang, knee_over_toe, frame_count)
    """

    def __init__(self) -> None:
        # Movement phase: "UP" | "DOWN" | None
        self._stage: Optional[str] = None

        # Tracks whether the current in-progress rep has stayed clean
        self._current_rep_good: bool = True

        # Wall-clock time when the current DOWN phase began
        self._rep_start_time: Optional[float] = None

        # Running totals
        self._rep_count:     int = 0
        self._bad_rep_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def stage(self) -> Optional[str]:
        """Current movement stage ("UP", "DOWN", or None)."""
        return self._stage

    @property
    def rep_count(self) -> int:
        """Total good reps confirmed this session."""
        return self._rep_count

    @property
    def bad_rep_count(self) -> int:
        """Total bad reps confirmed this session."""
        return self._bad_rep_count

    def reset(self) -> None:
        """Reset all counters and state for a new set."""
        self._stage            = None
        self._current_rep_good = True
        self._rep_start_time   = None
        self._rep_count        = 0
        self._bad_rep_count    = 0

    def update(
        self,
        smooth_ang: float,
        knee_over_toe: bool,
        frame_count: int,
        smooth_ang_threshold_depth: float = config.DEPTH_THRESHOLD,
        current_bad_form_elapsed: float = 0.0,
        bad_form_threshold: float = config.BAD_FORM_THRESHOLD,
    ) -> RepCounterResult:
        """
        Process one frame and return the updated rep-counting state.

        Parameters
        ----------
        smooth_ang : float
            Smoothed right knee angle in degrees (from main.py angle buffer).
        knee_over_toe : bool
            True if the right knee is ahead of the right toe this frame.
        frame_count : int
            Current session frame index (used to stamp frame lists for graphs).
        smooth_ang_threshold_depth : float
            Angle below which a frame is counted as a deep-rep frame.
            Defaults to config.DEPTH_THRESHOLD (100°).
        current_bad_form_elapsed : float
            Seconds elapsed since bad form began (0 if no bad form).
            Used to decide whether to stamp bad_form_frames for the graph.
        bad_form_threshold : float
            Seconds threshold before bad form is recorded in the graph.

        Returns
        -------
        RepCounterResult
            Snapshot of state after processing this frame.
        """
        new_rep_frames:     List[int] = []
        new_bad_form_frames: List[int] = []
        rep_complete = False
        good_rep     = False

        # ── 1. Mark rep as bad if knee crosses toe during descent ──────────
        # Preserved verbatim from original rep_counter.py:22-23
        if self._stage == "DOWN" and knee_over_toe:
            self._current_rep_good = False

        # ── 2. Detect descent (DOWN) ───────────────────────────────────────
        # Preserved verbatim from original rep_counter.py:26-29
        if smooth_ang < 90:
            if self._stage != "DOWN":
                self._stage          = "DOWN"
                self._rep_start_time = time.time()

        # ── 3. Detect ascent / rep completion (UP) ─────────────────────────
        # Preserved verbatim from original rep_counter.py:32-50
        if smooth_ang > 160 and self._stage == "DOWN":
            rep_duration = (
                time.time() - self._rep_start_time
                if self._rep_start_time is not None
                else 0
            )

            if rep_duration >= config.MIN_REP_DURATION:
                self._stage = "UP"
                rep_complete = True

                if self._current_rep_good:
                    self._rep_count += 1
                    good_rep = True
                else:
                    self._bad_rep_count += 1
                    good_rep = False

                # Reset per-rep indicators
                self._current_rep_good = True
                # bad_form_start is managed by SquatAnalyzer / FeedbackGenerator
            else:
                # Noise / jitter — ignore movement, reset stage
                self._stage          = None
                self._rep_start_time = None

        # ── 4. Stamp frame lists for session graph ─────────────────────────
        # Track frames where bad form threshold was exceeded
        if knee_over_toe and current_bad_form_elapsed >= bad_form_threshold:
            new_bad_form_frames.append(frame_count)

        # Track deep-rep frames (below depth threshold, during a good DOWN)
        if (
            smooth_ang < smooth_ang_threshold_depth
            and self._stage == "DOWN"
            and self._current_rep_good
        ):
            new_rep_frames.append(frame_count)

        return RepCounterResult(
            stage=self._stage,
            rep_complete=rep_complete,
            good_rep=good_rep,
            rep_count=self._rep_count,
            bad_rep_count=self._bad_rep_count,
            new_rep_frames=new_rep_frames,
            new_bad_form_frames=new_bad_form_frames,
        )
