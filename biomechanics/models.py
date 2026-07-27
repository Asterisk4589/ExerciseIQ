# biomechanics/models.py
"""
ExerciseIQ — Biomechanics Dataclass Models
============================================
Defines structured, strongly-typed output models for the Biomechanics Engine.
Encapsulates joint kinematics, spatial Center of Mass metrics, temporal derivatives,
postural stability scores, range-of-motion metrics, and tracking confidence.

These models are completely exercise-agnostic and serve as the standard kinematic
data contract between the Biomechanics Layer and downstream Intelligence analyzers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class JointAngles:
    """
    Smoothed 2D joint angles in degrees [0°, 180°].

    Attributes
    ----------
    left_knee : float
    right_knee : float
    left_hip : float
    right_hip : float
    left_elbow : float
    right_elbow : float
    left_shoulder : float
    right_shoulder : float
    left_ankle : float
    right_ankle : float
    torso_inclination : float
        Angle of torso axis (hip-mid to shoulder-mid) relative to vertical [0°, 180°].
    neck_inclination : float
        Angle of neck axis (shoulder-mid to nose) relative to vertical [0°, 180°].
    """
    left_knee: float = 0.0
    right_knee: float = 0.0
    left_hip: float = 0.0
    right_hip: float = 0.0
    left_elbow: float = 0.0
    right_elbow: float = 0.0
    left_shoulder: float = 0.0
    right_shoulder: float = 0.0
    left_ankle: float = 0.0
    right_ankle: float = 0.0
    torso_inclination: float = 0.0
    neck_inclination: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert joint angles to a flat dictionary."""
        return {
            "left_knee": self.left_knee,
            "right_knee": self.right_knee,
            "left_hip": self.left_hip,
            "right_hip": self.right_hip,
            "left_elbow": self.left_elbow,
            "right_elbow": self.right_elbow,
            "left_shoulder": self.left_shoulder,
            "right_shoulder": self.right_shoulder,
            "left_ankle": self.left_ankle,
            "right_ankle": self.right_ankle,
            "torso_inclination": self.torso_inclination,
            "neck_inclination": self.neck_inclination,
        }


@dataclass
class TemporalFeatures:
    """
    Time derivatives and movement speed metrics.

    Attributes
    ----------
    dt : float
        Frame delta time in seconds.
    angular_velocity : Dict[str, float]
        Per-joint angular velocity in degrees per second (°/s).
    angular_acceleration : Dict[str, float]
        Per-joint angular acceleration in degrees per second squared (°/s²).
    joint_velocity : Dict[str, float]
        Per-joint linear velocity in pixels per second (px/s).
    joint_acceleration : Dict[str, float]
        Per-joint linear acceleration in pixels per second squared (px/s²).
    movement_speed : float
        Linear speed of body Center of Mass in pixels per second (px/s).
    movement_direction : List[float]
        2D unit vector [dx, dy] indicating instantaneous CoM motion direction.
    movement_direction_deg : float
        Angle of CoM motion direction in degrees relative to horizontal [0°, 360°).
    time_under_tension : float
        Cumulative active movement duration in seconds (speed > threshold).
    rep_duration : Optional[float]
        Duration of active movement burst in seconds. Note: Exercise-specific
        repetition segmentation is owned by the Intelligence Layer.
    """
    dt: float = 0.0
    angular_velocity: Dict[str, float] = field(default_factory=dict)
    angular_acceleration: Dict[str, float] = field(default_factory=dict)
    joint_velocity: Dict[str, float] = field(default_factory=dict)
    joint_acceleration: Dict[str, float] = field(default_factory=dict)
    movement_speed: float = 0.0
    movement_direction: List[float] = field(default_factory=lambda: [0.0, 0.0])
    movement_direction_deg: float = 0.0
    time_under_tension: float = 0.0
    rep_duration: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert temporal features to a dictionary."""
        return {
            "dt": self.dt,
            "angular_velocity": self.angular_velocity,
            "angular_acceleration": self.angular_acceleration,
            "joint_velocity": self.joint_velocity,
            "joint_acceleration": self.joint_acceleration,
            "movement_speed": self.movement_speed,
            "movement_direction": self.movement_direction,
            "movement_direction_deg": self.movement_direction_deg,
            "time_under_tension": self.time_under_tension,
            "rep_duration": self.rep_duration,
        }


@dataclass
class SpatialFeatures:
    """
    Spatial positioning, postural alignments, and Center of Mass coordinates.

    Attributes
    ----------
    center_of_body : List[float]
        2D pixel coordinates [x, y] of hip midpoint.
    center_of_mass : List[float]
        2D pixel coordinates [x, y] of estimated anatomical Center of Mass.
    torso_lean : float
        Torso inclination angle from vertical in degrees (°).
    body_inclination : float
        Overall head-to-ankle inclination angle from vertical in degrees (°).
    hip_depth : float
        Vertical y-coordinate of hip midpoint in pixels.
    squat_depth : float
        Relative vertical height ratio of hip midpoint to ankle line [0.0, 1.0].
        Lower values represent deeper hip flexion / squat position.
    shoulder_alignment : float
        Horizontal alignment tilt of shoulders in degrees [0°, 90°].
    hip_alignment : float
        Horizontal alignment tilt of hips in degrees [0°, 90°].
    foot_width : float
        Pixel distance between left and right ankle landmarks.
    shoulder_width : float
        Pixel distance between left and right shoulder landmarks.
    stance_ratio : float
        Ratio of foot_width to shoulder_width.
    center_drift : float
        2D Euclidean displacement of hip center from baseline starting position (px).
    """
    center_of_body: List[float] = field(default_factory=lambda: [0.0, 0.0])
    center_of_mass: List[float] = field(default_factory=lambda: [0.0, 0.0])
    torso_lean: float = 0.0
    body_inclination: float = 0.0
    hip_depth: float = 0.0
    squat_depth: float = 0.0
    shoulder_alignment: float = 0.0
    hip_alignment: float = 0.0
    foot_width: float = 0.0
    shoulder_width: float = 0.0
    stance_ratio: float = 0.0
    center_drift: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert spatial features to a dictionary."""
        return {
            "center_of_body": self.center_of_body,
            "center_of_mass": self.center_of_mass,
            "torso_lean": self.torso_lean,
            "body_inclination": self.body_inclination,
            "hip_depth": self.hip_depth,
            "squat_depth": self.squat_depth,
            "shoulder_alignment": self.shoulder_alignment,
            "hip_alignment": self.hip_alignment,
            "foot_width": self.foot_width,
            "shoulder_width": self.shoulder_width,
            "stance_ratio": self.stance_ratio,
            "center_drift": self.center_drift,
        }


@dataclass
class SymmetryFeatures:
    """
    Bilateral left-vs-right symmetry indices and base of support balance.

    Attributes
    ----------
    knee_symmetry : float
        Bilateral knee angle symmetry score [0.0, 1.0]. (1.0 = perfect symmetry).
    hip_symmetry : float
        Bilateral hip angle symmetry score [0.0, 1.0].
    elbow_symmetry : float
        Bilateral elbow angle symmetry score [0.0, 1.0].
    shoulder_symmetry : float
        Bilateral shoulder angle symmetry score [0.0, 1.0].
    ankle_symmetry : float
        Bilateral ankle angle symmetry score [0.0, 1.0].
    balance_score : float
        Weight distribution / lateral CoM balance score relative to foot base [0.0, 1.0].
        1.0 indicates CoM is perfectly centered between feet.
    """
    knee_symmetry: float = 1.0
    hip_symmetry: float = 1.0
    elbow_symmetry: float = 1.0
    shoulder_symmetry: float = 1.0
    ankle_symmetry: float = 1.0
    balance_score: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        """Convert symmetry features to a dictionary."""
        return {
            "knee_symmetry": self.knee_symmetry,
            "hip_symmetry": self.hip_symmetry,
            "elbow_symmetry": self.elbow_symmetry,
            "shoulder_symmetry": self.shoulder_symmetry,
            "ankle_symmetry": self.ankle_symmetry,
            "balance_score": self.balance_score,
        }


@dataclass
class StabilityFeatures:
    """
    Postural stability, body sway, and direction-specific spatial drift.

    Attributes
    ----------
    stability : float
        Postural stability score [0.0, 1.0] based on torso lean and CoM sway.
    body_sway : float
        Standard deviation of lateral (horizontal) CoM position in pixels.
    horizontal_drift : float
        Signed horizontal displacement (dx) from baseline starting origin (px).
    vertical_drift : float
        Signed vertical displacement (dy) from baseline starting origin (px).
    center_drift : float
        Euclidean 2D displacement magnitude from baseline starting origin (px).
    smoothness : float
        Jerk-proxy movement smoothness score [0.0, 1.0] derived from acceleration variance.
    movement_consistency : float
        Velocity consistency score [0.0, 1.0] over history window.
    """
    stability: float = 1.0
    body_sway: float = 0.0
    horizontal_drift: float = 0.0
    vertical_drift: float = 0.0
    center_drift: float = 0.0
    smoothness: float = 1.0
    movement_consistency: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert stability features to a dictionary."""
        return {
            "stability": self.stability,
            "body_sway": self.body_sway,
            "horizontal_drift": self.horizontal_drift,
            "vertical_drift": self.vertical_drift,
            "center_drift": self.center_drift,
            "smoothness": self.smoothness,
            "movement_consistency": self.movement_consistency,
        }


@dataclass
class RangeOfMotionFeatures:
    """
    Joint Range of Motion (ROM) and maximum flexion limits.

    Attributes
    ----------
    range_of_motion : Dict[str, float]
        Current active ROM (max angle - min angle) per joint in degrees (°).
    normalized_rom_pct : Dict[str, float]
        Current ROM as a percentage of peak session ROM per joint [0%, 100%].
    max_knee_flexion : float
        Peak knee flexion achieved this session in degrees (smallest joint angle, e.g. 70°).
    max_hip_flexion : float
        Peak hip flexion achieved this session in degrees (smallest joint angle).
    """
    range_of_motion: Dict[str, float] = field(default_factory=dict)
    normalized_rom_pct: Dict[str, float] = field(default_factory=dict)
    max_knee_flexion: float = 180.0
    max_hip_flexion: float = 180.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert ROM features to a dictionary."""
        return {
            "range_of_motion": self.range_of_motion,
            "normalized_rom_pct": self.normalized_rom_pct,
            "max_knee_flexion": self.max_knee_flexion,
            "max_hip_flexion": self.max_hip_flexion,
        }


@dataclass
class TrackingMetrics:
    """
    MediaPipe landmark detection visibility and tracking confidence scores.

    Attributes
    ----------
    joint_confidences : Dict[str, float]
        Per-joint visibility score [0.0, 1.0].
    overall_confidence : float
        Mean visibility score across all 33 pose landmarks [0.0, 1.0].
    movement_confidence : float
        Confidence score weighted by spatial tracking quality and visibility [0.0, 1.0].
    low_confidence_joints : List[str]
        List of joint names falling below the configured confidence threshold.
    """
    joint_confidences: Dict[str, float] = field(default_factory=dict)
    overall_confidence: float = 1.0
    movement_confidence: float = 1.0
    low_confidence_joints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert tracking metrics to a dictionary."""
        return {
            "joint_confidences": self.joint_confidences,
            "overall_confidence": self.overall_confidence,
            "movement_confidence": self.movement_confidence,
            "low_confidence_joints": self.low_confidence_joints,
        }


@dataclass
class BiomechanicsResult:
    """
    Unified result object produced by BiomechanicsEngine.update().

    Provides strongly-typed attribute access alongside dict subscripting
    (__getitem__, get, keys, items) for 100% backward compatibility with legacy
    dict consumers.
    """
    timestamp: float = 0.0
    frame_number: int = 0
    joint_angles: Dict[str, float] = field(default_factory=dict)
    temporal_features: Dict[str, Any] = field(default_factory=dict)
    spatial_features: Dict[str, Any] = field(default_factory=dict)
    symmetry: Dict[str, float] = field(default_factory=dict)
    stability_features: Dict[str, Any] = field(default_factory=dict)
    rom_features: Dict[str, Any] = field(default_factory=dict)
    tracking_metrics: Dict[str, Any] = field(default_factory=dict)

    # Legacy dictionary views for backward compatibility
    posture_features: Dict[str, Any] = field(default_factory=dict)
    movement_quality: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access (e.g. bio['joint_angles'])."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Invalid key '{key}' in BiomechanicsResult.")

    def get(self, key: str, default: Any = None) -> Any:
        """Allow dict-style get() with default fallback."""
        if hasattr(self, key):
            val = getattr(self, key)
            return val if val is not None else default
        return default

    def __contains__(self, key: str) -> bool:
        """Allow 'key in bio' checks."""
        return hasattr(self, key)

    def __len__(self) -> int:
        """Return number of top-level keys."""
        return len(self.keys())

    def keys(self) -> List[str]:
        """Return top-level key names."""
        return [
            "timestamp",
            "frame_number",
            "joint_angles",
            "temporal_features",
            "posture_features",
            "spatial_features",
            "symmetry",
            "stability_features",
            "movement_quality",
            "rom_features",
            "tracking_metrics",
        ]

    def items(self) -> List[tuple[str, Any]]:
        """Return key-value pairs."""
        return [(k, self[k]) for k in self.keys()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire result object into a nested plain dictionary."""
        return {
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "joint_angles": self.joint_angles,
            "temporal_features": self.temporal_features,
            "posture_features": self.posture_features,
            "spatial_features": self.spatial_features,
            "symmetry": self.symmetry,
            "stability_features": self.stability_features,
            "movement_quality": self.movement_quality,
            "rom_features": self.rom_features,
            "tracking_metrics": self.tracking_metrics,
        }
