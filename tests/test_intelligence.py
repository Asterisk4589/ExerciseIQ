# tests/test_intelligence.py
"""
Unit test suite for ExerciseIQ Intelligence Layer.
Verifies RepCounter, MovementScorer, FeedbackGenerator, and SquatAnalyzer.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from intelligence.models import AnalysisResult, ScoreBreakdown, FeedbackMessage, RepCounterResult
from intelligence.rep_counter import RepCounter
from intelligence.movement_scorer import MovementScorer
from intelligence.feedback_generator import FeedbackGenerator
from intelligence.squat import SquatAnalyzer


def test_rep_counter():
    print("Testing RepCounter...")
    counter = RepCounter()
    
    # Stand (170 degrees)
    r1 = counter.update(smooth_ang=170.0, knee_over_toe=False, frame_count=1)
    assert r1.stage is None
    assert r1.rep_count == 0

    # Descend (80 degrees)
    r2 = counter.update(smooth_ang=80.0, knee_over_toe=False, frame_count=2)
    assert r2.stage == "DOWN"

    # Ascend (165 degrees) with delayed time to satisfy MIN_REP_DURATION
    counter._rep_start_time -= 2.0  # mock 2s elapsed
    r3 = counter.update(smooth_ang=165.0, knee_over_toe=False, frame_count=3)
    assert r3.stage == "UP"
    assert r3.rep_complete is True
    assert r3.good_rep is True
    assert r3.rep_count == 1
    print("RepCounter tests passed!")


def test_movement_scorer():
    print("Testing MovementScorer...")
    scorer = MovementScorer()
    mock_bio = {
        "movement_quality": {
            "stability": 0.85,
            "normalized_rom_pct": {"right_knee": 95.0}
        },
        "symmetry": {
            "knee_symmetry": 0.90
        }
    }
    score = scorer.score(mock_bio, smooth_ang=95.0)
    assert isinstance(score, ScoreBreakdown)
    assert score.depth > 0
    assert score.stability == 85.0
    assert score.symmetry == 90.0
    assert score.rom == 95.0
    assert score.overall > 0
    print("MovementScorer tests passed!")


def test_feedback_generator():
    print("Testing FeedbackGenerator...")
    fg = FeedbackGenerator()
    msg1 = fg.generate(knee_over_toe=False, bad_form_elapsed=0.0, rep_complete=False, good_rep=True)
    assert msg1.text == "GOOD FORM"
    assert msg1.is_good is True

    msg2 = fg.generate(knee_over_toe=True, bad_form_elapsed=0.5, rep_complete=False, good_rep=False)
    assert msg2.text == "PUSH THOSE KNEES BACK! YOU GOT THIS!"
    assert msg2.is_good is False
    print("FeedbackGenerator tests passed!")


def test_squat_analyzer():
    print("Testing SquatAnalyzer...")
    analyzer = SquatAnalyzer()
    res = analyzer.update(
        bio={},
        smooth_ang=170.0,
        knee_pt=[100, 200],
        toe_pt=[120, 200],
        frame_count=1
    )
    assert isinstance(res, AnalysisResult)
    assert res.knee_over_toe is False
    assert res.feedback.text == "GOOD FORM"
    print("SquatAnalyzer tests passed!")


if __name__ == "__main__":
    test_rep_counter()
    test_movement_scorer()
    test_feedback_generator()
    test_squat_analyzer()
    print("All Intelligence tests passed successfully!")
