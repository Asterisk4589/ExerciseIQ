# biomechanics/__init__.py
"""
Biomechanics package for ExerciseIQ.

The exercise-agnostic Human Movement Engine converts MediaPipe pose landmark
streams into a rich set of joint kinematics, spatial Center of Mass metrics,
temporal derivatives, postural stability scores, and bilateral symmetry features.

Public surface
--------------
    from biomechanics.engine import BiomechanicsEngine
    from biomechanics.models import (
        BiomechanicsResult,
        JointAngles,
        TemporalFeatures,
        SpatialFeatures,
        SymmetryFeatures,
        StabilityFeatures,
        RangeOfMotionFeatures,
        TrackingMetrics,
    )
"""

from biomechanics.engine import BiomechanicsEngine
from biomechanics.models import (
    BiomechanicsResult,
    JointAngles,
    RangeOfMotionFeatures,
    SpatialFeatures,
    StabilityFeatures,
    SymmetryFeatures,
    TemporalFeatures,
    TrackingMetrics,
)

__all__ = [
    "BiomechanicsEngine",
    "BiomechanicsResult",
    "JointAngles",
    "TemporalFeatures",
    "SpatialFeatures",
    "SymmetryFeatures",
    "StabilityFeatures",
    "RangeOfMotionFeatures",
    "TrackingMetrics",
]
