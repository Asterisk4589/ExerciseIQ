# intelligence/__init__.py
"""
Intelligence Layer for ExerciseIQ.

This package contains all exercise analysis logic — rep counting,
movement scoring, and feedback generation.

It MUST NOT import: cv2, mediapipe, matplotlib, or any drawing/GUI code.
It consumes only BiomechanicsResult dicts and plain Python scalars.

Public surface
--------------
    from intelligence.models            import AnalysisResult, ScoreBreakdown, FeedbackMessage
    from intelligence.rep_counter       import RepCounter
    from intelligence.movement_scorer   import MovementScorer
    from intelligence.feedback_generator import FeedbackGenerator
    from intelligence.squat             import SquatAnalyzer
"""

from intelligence.models             import AnalysisResult, ScoreBreakdown, FeedbackMessage
from intelligence.squat              import SquatAnalyzer

__all__ = [
    "AnalysisResult",
    "ScoreBreakdown",
    "FeedbackMessage",
    "SquatAnalyzer",
]
