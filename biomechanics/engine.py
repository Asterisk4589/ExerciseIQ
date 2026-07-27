# biomechanics/engine.py
"""
ExerciseIQ — Biomechanics Engine
=================================
Analyzes human movement features from MediaPipe Pose landmarks in real time.
Returns reusable kinematics, postural symmetry, movement quality, center of mass,
spatial drift, joint acceleration, and tracking confidence metrics.

Design principles
-----------------
- Exercise-agnostic: zero knowledge of squats, pushups, lunges, or any specific exercise.
- No OpenCV drawing, no rep counting, no exercise-specific thresholds.
- Every threshold and window is loaded from config.py — no magic numbers.
- Modular feature extraction methods for angles, derivatives, Center of Mass,
  stability, balance, and Range of Motion.
- Downstream exercise analyzers consume the returned BiomechanicsResult object
  without modification.

Output
------
BiomechanicsEngine.update() returns a BiomechanicsResult dataclass object with:
    joint_angles         → smoothed 2D angles for 12 joints (degrees)
    temporal_features    → angular/joint velocity & acceleration, CoM speed,
                           motion direction vector, active time under tension, dt
    spatial_features     → center position, 2D Center of Mass, torso lean,
                           body inclination, stance ratio, center drift
    symmetry             → normalized bilateral symmetry & balance scores [0, 1]
    stability_features   → stability score, body sway, horizontal/vertical drift
    rom_features         → per-joint ROM, normalized ROM %, peak flexion limits
    tracking_metrics     → landmark visibility, overall confidence, movement confidence
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
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


# ---------------------------------------------------------------------------
# MediaPipe landmark index constants (kept local — engine is self-contained)
# ---------------------------------------------------------------------------
_LM = {
    "nose":           0,
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
    "left_toe":       31,
    "right_toe":      32,
}

# Joints that have bilateral (left/right) symmetry counterparts
_BILATERAL_JOINTS = ("knee", "hip", "elbow", "shoulder", "ankle")

# All joints for which angular velocity/acceleration are computed
_ANGLE_JOINTS = (
    "left_knee", "right_knee",
    "left_hip", "right_hip",
    "left_elbow", "right_elbow",
    "left_shoulder", "right_shoulder",
    "left_ankle", "right_ankle",
    "torso_inclination", "neck_inclination",
)

# Dempster's 2D Body Segment Parameters (mass weighting for Center of Mass)
# Weights sum to 1.0
_SEGMENT_WEIGHTS = {
    "head": 0.081,
    "trunk": 0.497,
    "thighs": 0.200,
    "shanks": 0.0935,
    "feet": 0.029,
    "upper_arms": 0.056,
    "forearms": 0.0435,
}


class BiomechanicsEngine:
    """
    Exercise-agnostic Kinematic Physics Engine for human body movement.

    Converts MediaPipe Pose landmark streams into high-level biomechanical
    features (kinematics, kinetics proxies, spatial drift, Center of Mass,
    bilateral symmetry, and stability).
    """

    def __init__(self) -> None:
        """Load configuration settings and initialize rolling history queues."""
        self.history_size: int = int(getattr(config, "BIOMECH_HISTORY_SIZE", 60))
        self.smoothing_window: int = int(getattr(config, "BIOMECH_SMOOTHING_WINDOW", 5))
        self.stability_window: int = int(getattr(config, "BIOMECH_STABILITY_WINDOW", 30))
        self.min_confidence: float = float(getattr(config, "BIOMECH_MIN_CONFIDENCE", 0.5))
        self.smoothness_norm: float = float(getattr(config, "BIOMECH_SMOOTHNESS_NORM", 100.0))
        self.stability_norm: float = float(getattr(config, "BIOMECH_STABILITY_NORM", 10.0))
        self.active_speed_threshold: float = float(
            getattr(config, "BIOMECH_ACTIVE_SPEED_THRESHOLD", 15.0)
        )

        # Rolling frame history — each entry is a dict of frame features
        self.history: deque = deque(maxlen=self.history_size)

        # Baseline origin references for spatial drift tracking (established on frame 0)
        self._start_center: Optional[np.ndarray] = None
        self._start_com: Optional[np.ndarray] = None

        # Lifetime session limits
        self._session_max_rom: Dict[str, float] = {}
        self._session_min_angles: Dict[str, float] = {}

        # Frame counter & time under tension accumulator
        self._frame_counter: int = 0
        self._time_under_tension: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Clear all rolling history and baseline origin references.
        Call between sets to reset measurement history without reinstantiating.
        """
        self.history.clear()
        self._start_center = None
        self._start_com = None
        self._session_max_rom.clear()
        self._session_min_angles.clear()
        self._frame_counter = 0
        self._time_under_tension = 0.0

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Return aggregate statistics over the recorded session history.
        """
        if len(self.history) < 2:
            return {}

        all_smoothed = [f["smoothed_angles"] for f in self.history]
        all_speeds = [f["movement_speed"] for f in self.history]

        joint_stats: Dict[str, Dict[str, float]] = {}
        for joint in _ANGLE_JOINTS:
            values = [f[joint] for f in all_smoothed]
            joint_stats[joint] = {
                "mean": round(float(np.mean(values)), 1),
                "min": round(float(np.min(values)), 1),
                "max": round(float(np.max(values)), 1),
                "range": round(float(np.max(values) - np.min(values)), 1),
                "std": round(float(np.std(values)), 2),
            }

        speed_arr = np.array(all_speeds, dtype=float)

        total_drift = 0.0
        if self._start_center is not None:
            all_centers = np.array([f["center_of_body"] for f in self.history])
            diffs = all_centers - self._start_center
            total_drift = float(np.max(np.linalg.norm(diffs, axis=1)))

        return {
            "frame_count": len(self.history),
            "duration_seconds": round(
                self.history[-1]["timestamp"] - self.history[0]["timestamp"], 3
            ),
            "time_under_tension": round(self._time_under_tension, 2),
            "joint_statistics": joint_stats,
            "session_max_rom": dict(self._session_max_rom),
            "session_min_angles": dict(self._session_min_angles),
            "mean_speed": round(float(np.mean(speed_arr)), 2),
            "peak_speed": round(float(np.max(speed_arr)), 2),
            "max_centre_drift": round(total_drift, 2),
        }

    def update(
        self,
        landmarks: Any,
        timestamp: float,
        w: int = 640,
        h: int = 480,
    ) -> BiomechanicsResult | Dict[str, Any]:
        """
        Process one video frame of pose landmarks.

        Parameters
        ----------
        landmarks : sequence
            List of 33 MediaPipe NormalizedLandmark objects.
        timestamp : float
            Wall-clock timestamp in seconds.
        w : int
            Frame width in pixels.
        h : int
            Frame height in pixels.

        Returns
        -------
        BiomechanicsResult
            Dataclass containing full kinematic & spatial features.
            Returns empty dict `{}` if landmarks are None or invalid.
        """
        if landmarks is None or len(landmarks) < 33:
            return {}

        self._frame_counter += 1

        # 1. De-normalize landmark coordinates
        pts = self._extract_points(landmarks, w, h)

        # Midpoints used across feature extractors
        hip_mid = (pts["left_hip"] + pts["right_hip"]) * 0.5
        shoulder_mid = (pts["left_shoulder"] + pts["right_shoulder"]) * 0.5

        # 2. Joint Angles
        raw_angles, smoothed_angles = self.calculate_joint_angles(pts, shoulder_mid, hip_mid)

        # 3. Center of Mass (CoM)
        com = self.calculate_center_of_mass(pts)

        # Establish baseline origins on first valid frame
        curr_center = hip_mid[:2]
        if self._start_center is None:
            self._start_center = curr_center.copy()
        if self._start_com is None:
            self._start_com = com.copy()

        # 4. Temporal Derivatives (Velocities, Accelerations, CoM Motion, Time Under Tension)
        (
            angular_velocity,
            angular_acceleration,
            joint_velocity,
            joint_acceleration,
            movement_speed,
            direction_vec,
            direction_deg,
            dt,
        ) = self.calculate_temporal_derivatives(smoothed_angles, pts, com, timestamp)

        # Update Time Under Tension accumulator
        if movement_speed > self.active_speed_threshold and dt > 0.0:
            self._time_under_tension += dt

        # 5. Spatial Features
        spatial = self.calculate_spatial_features(pts, hip_mid, com, smoothed_angles)

        # 6. Symmetry & Balance
        symmetry = self.calculate_symmetry_and_balance(smoothed_angles, com, pts)

        # 7. Stability & Spatial Drift
        stability = self.calculate_stability_and_drift(
            raw_angles, com, angular_acceleration, movement_speed
        )

        # 8. Range of Motion & Max Flexion Limits
        rom = self.calculate_rom_and_flexion(smoothed_angles)

        # 9. Tracking & Confidence Metrics
        tracking = self.calculate_confidence_metrics(landmarks)

        # 10. Append frame entry to rolling history
        self.history.append({
            "timestamp": timestamp,
            "pts": pts,
            "angles": raw_angles,
            "smoothed_angles": smoothed_angles,
            "center_of_body": curr_center.tolist(),
            "center_of_mass": com.tolist(),
            "angular_velocity": angular_velocity,
            "angular_acceleration": angular_acceleration,
            "joint_velocity": joint_velocity,
            "joint_acceleration": joint_acceleration,
            "movement_speed": movement_speed,
        })

        # Assemble unified BiomechanicsResult dataclass
        temporal_obj = TemporalFeatures(
            dt=round(dt, 4),
            angular_velocity=angular_velocity,
            angular_acceleration=angular_acceleration,
            joint_velocity=joint_velocity,
            joint_acceleration=joint_acceleration,
            movement_speed=movement_speed,
            movement_direction=direction_vec,
            movement_direction_deg=direction_deg,
            time_under_tension=round(self._time_under_tension, 2),
            rep_duration=None,  # Note: Rep segmentation belongs to Intelligence layer
        )

        result = BiomechanicsResult(
            timestamp=round(timestamp, 4),
            frame_number=self._frame_counter,
            joint_angles=smoothed_angles,
            temporal_features=temporal_obj.to_dict(),
            spatial_features=spatial.to_dict(),
            symmetry=symmetry.to_dict(),
            stability_features=stability.to_dict(),
            rom_features=rom.to_dict(),
            tracking_metrics=tracking.to_dict(),
            # Legacy dictionary views for backward compatibility
            posture_features=spatial.to_dict(),
            movement_quality={
                "smoothness": stability.smoothness,
                "stability": stability.stability,
                "body_sway": stability.body_sway,
                "movement_consistency": stability.movement_consistency,
                "range_of_motion": rom.range_of_motion,
                "normalized_rom_pct": rom.normalized_rom_pct,
            },
        )

        return result

    # ------------------------------------------------------------------
    # Feature Extraction Methods (Modular & Testable)
    # ------------------------------------------------------------------

    def _extract_points(self, landmarks: Any, w: int, h: int) -> Dict[str, np.ndarray]:
        """Convert normalized landmarks to scaled pixel 3D coordinates [x, y, z]."""
        scale_z = (w + h) * 0.5
        pts = {}
        for name, idx in _LM.items():
            lm = landmarks[idx]
            pts[name] = np.array([lm.x * w, lm.y * h, lm.z * scale_z], dtype=float)
        return pts

    def calculate_joint_angles(
        self,
        pts: Dict[str, np.ndarray],
        shoulder_mid: np.ndarray,
        hip_mid: np.ndarray,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute raw and rolling-smoothed 2D joint angles in degrees (°).

        Returns
        -------
        (raw_angles, smoothed_angles)
        """
        raw_angles: Dict[str, float] = {
            "left_knee": self._angle(pts["left_hip"], pts["left_knee"], pts["left_ankle"]),
            "right_knee": self._angle(pts["right_hip"], pts["right_knee"], pts["right_ankle"]),
            "left_hip": self._angle(pts["left_shoulder"], pts["left_hip"], pts["left_knee"]),
            "right_hip": self._angle(pts["right_shoulder"], pts["right_hip"], pts["right_knee"]),
            "left_elbow": self._angle(pts["left_shoulder"], pts["left_elbow"], pts["left_wrist"]),
            "right_elbow": self._angle(pts["right_shoulder"], pts["right_elbow"], pts["right_wrist"]),
            "left_shoulder": self._angle(pts["left_elbow"], pts["left_shoulder"], pts["left_hip"]),
            "right_shoulder": self._angle(pts["right_elbow"], pts["right_shoulder"], pts["right_hip"]),
            "left_ankle": self._angle(pts["left_knee"], pts["left_ankle"], pts["left_toe"]),
            "right_ankle": self._angle(pts["right_knee"], pts["right_ankle"], pts["right_toe"]),
            "torso_inclination": self._inclination(shoulder_mid - hip_mid),
            "neck_inclination": self._inclination(pts["nose"] - shoulder_mid),
        }

        # Apply causal rolling average smoothing
        smoothed_angles: Dict[str, float] = {}
        for joint, current in raw_angles.items():
            past = [f["angles"][joint] for f in self.history]
            window = min(len(past) + 1, self.smoothing_window)
            values = (past + [current])[-window:]
            smoothed_angles[joint] = round(sum(values) / len(values), 1)

        return raw_angles, smoothed_angles

    def calculate_center_of_mass(self, pts: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute estimated 2D Center of Mass (CoM) [x, y] in pixels using
        Dempster's anatomical body segment mass parameter weighting.
        """
        head_pos = pts["nose"][:2]
        trunk_pos = ((pts["left_shoulder"] + pts["right_shoulder"] + pts["left_hip"] + pts["right_hip"]) * 0.25)[:2]
        thighs_pos = ((pts["left_hip"] + pts["left_knee"] + pts["right_hip"] + pts["right_knee"]) * 0.25)[:2]
        shanks_pos = ((pts["left_knee"] + pts["left_ankle"] + pts["right_knee"] + pts["right_ankle"]) * 0.25)[:2]
        feet_pos = ((pts["left_ankle"] + pts["left_toe"] + pts["right_ankle"] + pts["right_toe"]) * 0.25)[:2]
        upper_arms_pos = ((pts["left_shoulder"] + pts["left_elbow"] + pts["right_shoulder"] + pts["right_elbow"]) * 0.25)[:2]
        forearms_pos = ((pts["left_elbow"] + pts["left_wrist"] + pts["right_elbow"] + pts["right_wrist"]) * 0.25)[:2]

        com_2d = (
            head_pos * _SEGMENT_WEIGHTS["head"]
            + trunk_pos * _SEGMENT_WEIGHTS["trunk"]
            + thighs_pos * _SEGMENT_WEIGHTS["thighs"]
            + shanks_pos * _SEGMENT_WEIGHTS["shanks"]
            + feet_pos * _SEGMENT_WEIGHTS["feet"]
            + upper_arms_pos * _SEGMENT_WEIGHTS["upper_arms"]
            + forearms_pos * _SEGMENT_WEIGHTS["forearms"]
        )

        return com_2d

    def calculate_temporal_derivatives(
        self,
        smoothed_angles: Dict[str, float],
        pts: Dict[str, np.ndarray],
        com: np.ndarray,
        timestamp: float,
    ) -> Tuple[
        Dict[str, float],
        Dict[str, float],
        Dict[str, float],
        Dict[str, float],
        float,
        List[float],
        float,
        float,
    ]:
        """
        Compute angular velocities, angular accelerations, joint velocities,
        joint accelerations, CoM movement speed, and motion direction vector.
        """
        if not self.history or (timestamp <= self.history[-1]["timestamp"]):
            av = {j: 0.0 for j in smoothed_angles}
            aa = {j: 0.0 for j in smoothed_angles}
            jv = {j: 0.0 for j in pts}
            ja = {j: 0.0 for j in pts}
            return av, aa, jv, ja, 0.0, [0.0, 0.0], 0.0, 0.0

        prev = self.history[-1]
        dt = timestamp - prev["timestamp"]
        prev_smoothed = prev["smoothed_angles"]
        prev_pts = prev["pts"]
        prev_av = prev.get("angular_velocity", {})
        prev_jv = prev.get("joint_velocity", {})
        prev_com = np.array(prev["center_of_mass"])

        # Angular Velocity & Acceleration
        angular_velocity: Dict[str, float] = {}
        angular_acceleration: Dict[str, float] = {}
        for joint, angle in smoothed_angles.items():
            v = (angle - prev_smoothed[joint]) / dt
            angular_velocity[joint] = round(v, 2)
            a = (v - prev_av.get(joint, 0.0)) / dt
            angular_acceleration[joint] = round(a, 2)

        # Linear Joint Velocity & Acceleration (px/s, px/s²)
        joint_velocity: Dict[str, float] = {}
        joint_acceleration: Dict[str, float] = {}
        for name, coord in pts.items():
            disp = np.linalg.norm(coord[:2] - prev_pts[name][:2])
            vel = disp / dt
            joint_velocity[name] = round(float(vel), 2)
            acc = (vel - prev_jv.get(name, 0.0)) / dt
            joint_acceleration[name] = round(float(acc), 2)

        # Center of Mass Motion Speed & Direction
        com_disp = com - prev_com
        com_dist = float(np.linalg.norm(com_disp))
        movement_speed = round(com_dist / dt, 2)

        if com_dist > 1e-5:
            direction_vec = [round(float(com_disp[0] / com_dist), 3), round(float(com_disp[1] / com_dist), 3)]
            direction_deg = round(float(np.degrees(np.arctan2(com_disp[1], com_disp[0]))) % 360.0, 1)
        else:
            direction_vec = [0.0, 0.0]
            direction_deg = 0.0

        return (
            angular_velocity,
            angular_acceleration,
            joint_velocity,
            joint_acceleration,
            movement_speed,
            direction_vec,
            direction_deg,
            dt,
        )

    def calculate_spatial_features(
        self,
        pts: Dict[str, np.ndarray],
        hip_mid: np.ndarray,
        com: np.ndarray,
        smoothed_angles: Dict[str, float],
    ) -> SpatialFeatures:
        """Compute spatial coordinates, body inclination, stance ratio, and squat depth."""
        curr_center = hip_mid[:2]
        torso_lean = smoothed_angles["torso_inclination"]

        # Body inclination: head (nose) to mid-ankle axis vs vertical
        shoulder_mid = (pts["left_shoulder"] + pts["right_shoulder"]) * 0.5
        ankle_mid = (pts["left_ankle"] + pts["right_ankle"]) * 0.5
        body_vector = pts["nose"] - ankle_mid
        body_inclination = self._inclination(body_vector)

        shoulder_alignment = self._alignment(pts["left_shoulder"], pts["right_shoulder"])
        hip_alignment = self._alignment(pts["left_hip"], pts["right_hip"])

        foot_width = float(np.linalg.norm(pts["left_ankle"][:2] - pts["right_ankle"][:2]))
        shoulder_width = float(np.linalg.norm(pts["left_shoulder"][:2] - pts["right_shoulder"][:2]))
        stance_ratio = round(foot_width / shoulder_width, 3) if shoulder_width > 0.0 else 0.0

        # Squat / Knee Depth Ratio: relative vertical position of hips to ankles
        shoulder_y = shoulder_mid[1]
        ankle_y = ankle_mid[1]
        torso_length = max(abs(ankle_y - shoulder_y), 1.0)
        hip_y = curr_center[1]
        squat_depth = round(float(max(0.0, min(1.0, (ankle_y - hip_y) / torso_length))), 3)

        center_drift = float(np.linalg.norm(curr_center - self._start_center)) if self._start_center is not None else 0.0

        return SpatialFeatures(
            center_of_body=[round(float(curr_center[0]), 2), round(float(curr_center[1]), 2)],
            center_of_mass=[round(float(com[0]), 2), round(float(com[1]), 2)],
            torso_lean=round(torso_lean, 2),
            body_inclination=round(body_inclination, 2),
            hip_depth=round(float(hip_y), 2),
            squat_depth=squat_depth,
            shoulder_alignment=round(shoulder_alignment, 2),
            hip_alignment=round(hip_alignment, 2),
            foot_width=round(foot_width, 2),
            shoulder_width=round(shoulder_width, 2),
            stance_ratio=stance_ratio,
            center_drift=round(center_drift, 2),
        )

    def calculate_symmetry_and_balance(
        self,
        smoothed_angles: Dict[str, float],
        com: np.ndarray,
        pts: Dict[str, np.ndarray],
    ) -> SymmetryFeatures:
        """Compute bilateral left-vs-right joint symmetry and base-of-support balance."""
        symmetry_dict = {
            f"{side}_symmetry": self._symmetry(
                smoothed_angles[f"left_{side}"], smoothed_angles[f"right_{side}"]
            )
            for side in _BILATERAL_JOINTS
        }

        # Lateral balance score: CoM x-position relative to mid-stance ankle base
        left_ankle_x = pts["left_ankle"][0]
        right_ankle_x = pts["right_ankle"][0]
        mid_stance_x = (left_ankle_x + right_ankle_x) * 0.5
        half_stance = max(abs(right_ankle_x - left_ankle_x) * 0.5, 1.0)
        com_offset = abs(com[0] - mid_stance_x)
        balance_score = round(float(max(0.0, min(1.0, 1.0 - (com_offset / half_stance)))), 3)

        return SymmetryFeatures(
            knee_symmetry=symmetry_dict["knee_symmetry"],
            hip_symmetry=symmetry_dict["hip_symmetry"],
            elbow_symmetry=symmetry_dict["elbow_symmetry"],
            shoulder_symmetry=symmetry_dict["shoulder_symmetry"],
            ankle_symmetry=symmetry_dict["ankle_symmetry"],
            balance_score=balance_score,
        )

    def calculate_stability_and_drift(
        self,
        raw_angles: Dict[str, float],
        com: np.ndarray,
        angular_acceleration: Dict[str, float],
        movement_speed: float,
    ) -> StabilityFeatures:
        """Compute stability score, body sway, signed horizontal/vertical drift, and smoothness."""
        window_frames = list(self.history)[-self.stability_window:]

        leans = [f["angles"]["torso_inclination"] for f in window_frames] + [raw_angles["torso_inclination"]]
        com_xs = [f["center_of_mass"][0] for f in window_frames] + [float(com[0])]

        std_lean = float(np.std(leans)) if len(leans) > 1 else 0.0
        std_cx = float(np.std(com_xs)) if len(com_xs) > 1 else 0.0

        stability = round(1.0 / (1.0 + (std_lean + std_cx) / self.stability_norm), 3)
        body_sway = round(std_cx, 2)

        # Direction-specific spatial drift from starting CoM origin
        if self._start_com is not None:
            horizontal_drift = round(float(com[0] - self._start_com[0]), 2)
            vertical_drift = round(float(com[1] - self._start_com[1]), 2)
            center_drift = round(float(np.linalg.norm(com - self._start_com)), 2)
        else:
            horizontal_drift = 0.0
            vertical_drift = 0.0
            center_drift = 0.0

        # Movement Smoothness (jerk proxy)
        smoothness = self._compute_smoothness(angular_acceleration)

        # Movement Speed Consistency
        consistency = self._compute_consistency(movement_speed)

        return StabilityFeatures(
            stability=stability,
            body_sway=body_sway,
            horizontal_drift=horizontal_drift,
            vertical_drift=vertical_drift,
            center_drift=center_drift,
            smoothness=smoothness,
            movement_consistency=consistency,
        )

    def calculate_rom_and_flexion(
        self,
        smoothed_angles: Dict[str, float],
    ) -> RangeOfMotionFeatures:
        """Compute Range of Motion (ROM) and session minimum joint angles (max flexion)."""
        rom: Dict[str, float] = {}
        normalized_rom_pct: Dict[str, float] = {}

        for joint, current in smoothed_angles.items():
            history_vals = [f["smoothed_angles"][joint] for f in self.history]
            all_vals = history_vals + [current]

            current_rom = round(max(all_vals) - min(all_vals), 1)
            rom[joint] = current_rom

            prev_max_rom = self._session_max_rom.get(joint, 0.0)
            new_max_rom = max(prev_max_rom, current_rom)
            self._session_max_rom[joint] = new_max_rom

            # Session peak flexion limits (minimum joint angle in degrees)
            prev_min_ang = self._session_min_angles.get(joint, 180.0)
            self._session_min_angles[joint] = min(prev_min_ang, current)

            if new_max_rom > 0.0:
                normalized_rom_pct[joint] = round((current_rom / new_max_rom) * 100.0, 1)
            else:
                normalized_rom_pct[joint] = 100.0

        max_knee_flexion = min(
            self._session_min_angles.get("left_knee", 180.0),
            self._session_min_angles.get("right_knee", 180.0),
        )
        max_hip_flexion = min(
            self._session_min_angles.get("left_hip", 180.0),
            self._session_min_angles.get("right_hip", 180.0),
        )

        return RangeOfMotionFeatures(
            range_of_motion=rom,
            normalized_rom_pct=normalized_rom_pct,
            max_knee_flexion=round(max_knee_flexion, 1),
            max_hip_flexion=round(max_hip_flexion, 1),
        )

    def calculate_confidence_metrics(self, landmarks: Any) -> TrackingMetrics:
        """Compute per-joint visibility, overall pose confidence, and movement confidence."""
        joint_confidences: Dict[str, float] = {
            "left_knee": float(landmarks[_LM["left_knee"]].visibility),
            "right_knee": float(landmarks[_LM["right_knee"]].visibility),
            "left_hip": float(landmarks[_LM["left_hip"]].visibility),
            "right_hip": float(landmarks[_LM["right_hip"]].visibility),
            "left_elbow": float(landmarks[_LM["left_elbow"]].visibility),
            "right_elbow": float(landmarks[_LM["right_elbow"]].visibility),
            "left_shoulder": float(landmarks[_LM["left_shoulder"]].visibility),
            "right_shoulder": float(landmarks[_LM["right_shoulder"]].visibility),
            "left_ankle": float(landmarks[_LM["left_ankle"]].visibility),
            "right_ankle": float(landmarks[_LM["right_ankle"]].visibility),
        }

        overall_confidence = float(np.mean([lm.visibility for lm in landmarks]))
        low_confidence_joints: List[str] = [
            j for j, conf in joint_confidences.items() if conf < self.min_confidence
        ]

        # Movement confidence: overall visibility weighted by joint tracking quality
        movement_confidence = float(np.mean(list(joint_confidences.values())))

        return TrackingMetrics(
            joint_confidences=joint_confidences,
            overall_confidence=round(overall_confidence, 3),
            movement_confidence=round(movement_confidence, 3),
            low_confidence_joints=low_confidence_joints,
        )

    # ------------------------------------------------------------------
    # Private Helper Math Functions
    # ------------------------------------------------------------------

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """2D joint angle in degrees at vertex b between ba and bc."""
        ba = a[:2] - b[:2]
        bc = c[:2] - b[:2]
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba == 0.0 or norm_bc == 0.0:
            return 0.0
        cos_a = np.dot(ba, bc) / (norm_ba * norm_bc)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    @staticmethod
    def _inclination(vec: np.ndarray) -> float:
        """Angle in degrees between a 2D vector and upward-vertical [0, -1]."""
        v = vec[:2]
        norm_v = np.linalg.norm(v)
        if norm_v == 0.0:
            return 0.0
        cos_a = -v[1] / norm_v
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    @staticmethod
    def _alignment(p1: np.ndarray, p2: np.ndarray) -> float:
        """Horizontal alignment angle of segment p1->p2 in degrees [0, 90]."""
        v = p2[:2] - p1[:2]
        raw = abs(float(np.degrees(np.arctan2(v[1], v[0]))))
        return abs(180.0 - raw) if raw > 90.0 else raw

    @staticmethod
    def _symmetry(left_val: float, right_val: float) -> float:
        """Normalized bilateral symmetry score in [0.0, 1.0]."""
        total = left_val + right_val
        if total < 0.01:
            return 1.0
        return round(1.0 - abs(left_val - right_val) / total, 3)

    def _compute_smoothness(self, angular_acceleration: Dict[str, float]) -> float:
        """Jerk-proxy smoothness score in [0.0, 1.0]."""
        if not self.history:
            return 1.0
        std_values: List[float] = []
        for joint in _ANGLE_JOINTS:
            past_acc = [f["angular_acceleration"].get(joint, 0.0) for f in self.history]
            current_acc = angular_acceleration.get(joint, 0.0)
            values = past_acc + [current_acc]
            if len(values) > 1:
                std_values.append(float(np.std(values)))
        if not std_values:
            return 1.0
        mean_std = float(np.mean(std_values))
        return round(1.0 / (1.0 + mean_std / self.smoothness_norm), 3)

    def _compute_consistency(self, movement_speed: float) -> float:
        """Movement velocity consistency score in [0.0, 1.0]."""
        speeds = [f["movement_speed"] for f in self.history] + [movement_speed]
        if len(speeds) < 2:
            return 1.0
        mean_spd = float(np.mean(speeds))
        if mean_spd <= 0.0:
            return 1.0
        cv = float(np.std(speeds)) / mean_spd
        return round(float(np.clip(1.0 - cv, 0.0, 1.0)), 3)
