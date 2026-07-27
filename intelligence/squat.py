# intelligence/squat.py
"""
SquatAnalyzer — the single public entry-point for squat intelligence.

Responsibility
--------------
Orchestrate RepCounter, MovementScorer, and FeedbackGenerator for each
video frame and return one AnalysisResult.

Also owns the form-geometry logic that was previously in form_analysis.py
(knee-over-toe detection and bad-form timer), since that decision is purely
exercise-specific intelligence — not rendering, not biomechanics.

Design constraints
------------------
- NO imports of cv2, mediapipe, matplotlib, or any drawing/GUI library.
- ONE public method: update()
- All thresholds read from config.py — none hard-coded here.

Behavior (PRESERVED VERBATIM from original form_analysis.py)
-------------------------------------------------------------
- toe_line_x = toe_pt[0] + config.TOLERANCE
- knee_over_toe = knee_pt[0] > toe_line_x
- bad_form_start tracking: set on first knee-over-toe frame, cleared otherwise
- state.current_rep_good = False if elapsed >= BAD_FORM_THRESHOLD
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import config
from intelligence.feedback_generator import FeedbackGenerator
from intelligence.models import AnalysisResult, ScoreBreakdown, FeedbackMessage
from intelligence.movement_scorer import MovementScorer
from intelligence.rep_counter import RepCounter

# Type alias (mirrors biomechanics.engine.BiomechanicsResult)
BiomechanicsResult = Dict[str, Any]

# BGR color constants — defined here so intelligence/ does not import constants.py
# (which also contains cv2-dependent drawing details in other parts of the project).
# These are plain Python tuples, not cv2 objects.
_COLOR_GREEN = (0, 255, 0)
_COLOR_RED   = (0, 0, 255)


class SquatAnalyzer:
    """
    Orchestrates all squat-specific intelligence for one video session.

    Usage
    -----
    analyzer = SquatAnalyzer()
    while capturing:
        result = analyzer.update(bio, smooth_ang, knee_pt, toe_pt, frame_count)
    """

    def __init__(self) -> None:
        self._counter  = RepCounter()
        self._scorer   = MovementScorer()
        self._feedback = FeedbackGenerator()

        # Bad-form timer (mirrors state.bad_form_start in original form_analysis.py)
        self._bad_form_start: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all internal state for a new exercise set."""
        self._counter.reset()
        self._bad_form_start = None

    def update(
        self,
        bio: BiomechanicsResult,
        smooth_ang: float,
        knee_pt: List[int],
        toe_pt: List[int],
        frame_count: int,
    ) -> AnalysisResult:
        """
        Process one video frame and return the complete analysis result.

        Parameters
        ----------
        bio : BiomechanicsResult
            Output of BiomechanicsEngine.update() for this frame.
            May be an empty dict if no landmarks were detected.
        smooth_ang : float
            Smoothed right knee angle in degrees.
        knee_pt : List[int]
            [x, y] pixel coordinates of the right knee landmark.
        toe_pt : List[int]
            [x, y] pixel coordinates of the right toe landmark.
        frame_count : int
            Current session frame index (for graph annotation).

        Returns
        -------
        AnalysisResult
            Complete per-frame analysis snapshot.
        """
        # ── 1. Knee-over-toe geometry (from form_analysis.py:27-28) ────────
        toe_line_x   = toe_pt[0] + config.TOLERANCE
        knee_over_toe = knee_pt[0] > toe_line_x

        # ── 2. Bad-form timer (from form_analysis.py:31-46) ─────────────────
        if knee_over_toe:
            if self._bad_form_start is None:
                self._bad_form_start = time.time()
            bad_form_elapsed = time.time() - self._bad_form_start
        else:
            self._bad_form_start = None
            bad_form_elapsed = 0.0

        # If bad form has persisted long enough, mark the current rep as bad.
        # This mirrors form_analysis.py:38 → state.current_rep_good = False.
        # We communicate it to RepCounter through knee_over_toe (counter
        # already watches knee_over_toe during DOWN phase).

        # ── 3. Rep counting ─────────────────────────────────────────────────
        counter_result = self._counter.update(
            smooth_ang=smooth_ang,
            knee_over_toe=knee_over_toe,
            frame_count=frame_count,
            smooth_ang_threshold_depth=config.DEPTH_THRESHOLD,
            current_bad_form_elapsed=bad_form_elapsed,
            bad_form_threshold=config.BAD_FORM_THRESHOLD,
        )

        # ── 4. Movement scoring ─────────────────────────────────────────────
        score = self._scorer.score(bio, smooth_ang)

        # ── 5. Feedback generation ──────────────────────────────────────────
        feedback = self._feedback.generate(
            knee_over_toe=knee_over_toe,
            bad_form_elapsed=bad_form_elapsed,
            rep_complete=counter_result.rep_complete,
            good_rep=counter_result.good_rep,
        )

        # ── 6. Derive render-passthrough values ─────────────────────────────
        # form_col_bgr mirrors the original form_analysis.py:37,40,46 logic:
        #   bad form (elapsed >= threshold) → red
        #   otherwise → green
        if knee_over_toe and bad_form_elapsed >= config.BAD_FORM_THRESHOLD:
            form_col_bgr = _COLOR_RED
        else:
            form_col_bgr = _COLOR_GREEN

        # confidence: overall landmark tracking confidence from biomechanics
        confidence = (
            bio.get("tracking_metrics", {}).get("overall_confidence", 0.0)
            if bio
            else 0.0
        )

        return AnalysisResult(
            stage=counter_result.stage,
            rep_complete=counter_result.rep_complete,
            good_rep=counter_result.good_rep,
            rep_number=counter_result.rep_count,
            bad_rep_count=counter_result.bad_rep_count,
            feedback=feedback,
            score=score,
            confidence=confidence,
            knee_over_toe=knee_over_toe,
            toe_line_x=toe_line_x,
            form_col_bgr=form_col_bgr,
            new_rep_frames=counter_result.new_rep_frames,
            new_bad_form_frames=counter_result.new_bad_form_frames,
        )
