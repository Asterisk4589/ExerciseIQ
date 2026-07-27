# intelligence/models.py
"""
Shared output dataclasses for the ExerciseIQ Intelligence Layer.

All exercise analyzers (squat, push-up, plank, …) produce these types.
No imports from cv2, mediapipe, or matplotlib are permitted here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FeedbackMessage:
    """
    A single human-readable coaching cue produced by FeedbackGenerator.

    Attributes
    ----------
    text : str
        The message displayed to the user (e.g. "PUSH THOSE KNEES BACK!").
    is_good : bool
        True  → positive feedback / use green color in UI.
        False → corrective cue   / use red color in UI.
    """
    text: str
    is_good: bool


@dataclass
class ScoreBreakdown:
    """
    Per-component movement quality scores, each in the range [0, 100].

    These are computed by MovementScorer and embedded in AnalysisResult.
    Individual components are kept separate so the UI can render them
    independently (e.g. as a bar chart or overlay HUD).

    Attributes
    ----------
    depth : float
        How deeply the user descended relative to the configured threshold.
        100 = at or below target depth angle.
    tempo : float
        Pace consistency.  Reserved — always 100.0 until a timer-based
        tempo scorer is implemented.
    stability : float
        Postural stability from BiomechanicsEngine (torso lean + sway).
    symmetry : float
        Bilateral knee symmetry from BiomechanicsEngine.
    rom : float
        Range-of-motion score for the right knee joint.
    overall : float
        Weighted average of all components.
    """
    depth:     float = 0.0
    tempo:     float = 100.0   # reserved — not yet computed
    stability: float = 0.0
    symmetry:  float = 0.0
    rom:       float = 0.0
    overall:   float = 0.0


@dataclass
class RepCounterResult:
    """
    Internal result returned by RepCounter.update() each frame.

    main.py and SquatAnalyzer consume this to update SessionState
    and build the final AnalysisResult.

    Attributes
    ----------
    stage : Optional[str]
        Current movement phase: "UP", "DOWN", or None (standing still).
    rep_complete : bool
        True on the exact frame a rep was confirmed complete.
    good_rep : bool
        Valid only when rep_complete is True.
        True  → rep was counted as good.
        False → rep was counted as bad.
    rep_count : int
        Running total of confirmed good reps.
    bad_rep_count : int
        Running total of confirmed bad reps.
    new_rep_frames : List[int]
        Frame indices added this frame to the depth-tracking list
        (for session graph shading).
    new_bad_form_frames : List[int]
        Frame indices added this frame to the bad-form tracking list
        (for session graph shading).
    """
    stage:              Optional[str]
    rep_complete:       bool
    good_rep:           bool
    rep_count:          int
    bad_rep_count:      int
    new_rep_frames:     List[int] = field(default_factory=list)
    new_bad_form_frames: List[int] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """
    The single structured output of SquatAnalyzer.update() for one frame.

    main.py reads this and:
      - updates SessionState (stage, counts, frame lists)
      - passes drawing data to renderer.draw_skeleton() / draw_ui()
      - logs to analytics

    Attributes
    ----------
    stage : Optional[str]
        "UP" | "DOWN" | None
    rep_complete : bool
        True on the frame a rep was confirmed.
    good_rep : bool
        Meaningful only when rep_complete is True.
    rep_number : int
        Total confirmed good reps this session.
    bad_rep_count : int
        Total confirmed bad reps this session.
    feedback : FeedbackMessage
        Current coaching text and associated color intent.
    score : ScoreBreakdown
        Per-component and overall movement quality scores.
    confidence : float
        Overall landmark tracking confidence in [0, 1].
    knee_over_toe : bool
        Whether the right knee is ahead of the right toe this frame.
        Passed through to renderer — Intelligence Layer does not draw.
    toe_line_x : int
        Pixel x-coordinate of the vertical toe reference line.
        Passed through to renderer — Intelligence Layer does not draw.
    form_col_bgr : tuple
        BGR color tuple matching the feedback intent (green/red).
        Passed through to renderer — Intelligence Layer does not draw.
    new_rep_frames : List[int]
        Frame indices to append to SessionState.rep_frames.
    new_bad_form_frames : List[int]
        Frame indices to append to SessionState.bad_form_frames.
    """
    stage:               Optional[str]
    rep_complete:        bool
    good_rep:            bool
    rep_number:          int
    bad_rep_count:       int
    feedback:            FeedbackMessage
    score:               ScoreBreakdown
    confidence:          float
    knee_over_toe:       bool
    toe_line_x:          int
    form_col_bgr:        tuple
    new_rep_frames:      List[int] = field(default_factory=list)
    new_bad_form_frames: List[int] = field(default_factory=list)
