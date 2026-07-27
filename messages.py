# messages.py
"""
Feedback messages compatibility shim for ExerciseIQ.
Re-exports message pools from intelligence.feedback_generator.
"""

from intelligence.feedback_generator import GOOD_MESSAGES, BAD_MESSAGES

__all__ = ["GOOD_MESSAGES", "BAD_MESSAGES"]
