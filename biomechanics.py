# biomechanics.py
"""
ExerciseIQ — Biomechanics Engine
=================================
Extracts reusable human-movement features from MediaPipe Pose landmarks.

Design principles
-----------------
- Exercise-agnostic: zero knowledge of squats, lunges, or any specific movement.
- No OpenCV drawing, no rep counting, no exercise-specific thresholds.
- Every threshold and window is loaded from config.py — no magic numbers.
- Downstream exercise analyzers consume the returned BiomechanicsResult dict
  without modification.

Output
------
BiomechanicsEngine.update() returns a BiomechanicsResult TypedDict with six
top-level sections:

    joint_angles        → smoothed 2-D angles for 12 joints (degrees)
    temporal_features   → per-joint angular velocity/acceleration, joint
                          velocity, overall movement speed, and frame Δt
    posture_features    → centre-of-body, torso lean, shoulder/hip alignment,
                          foot & shoulder widths, stance ratio, centre drift
    symmetry            → normalized left-vs-right symmetry scores [0, 1]
    movement_quality    → smoothness, stability, body sway, consistency, ROM
    tracking_metrics    → per-joint visibility, overall confidence, and a list
                          of joints below the configured confidence threshold

Session helpers
---------------
    reset()             → clear all rolling history and baseline references
    get_session_summary() → aggregate statistics over the full recorded history
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

import config


# ---------------------------------------------------------------------------
# Output type alias (TypedDict-style, but using a plain dict for broad compat)
# ---------------------------------------------------------------------------
BiomechanicsResult = Dict[str, Any]
"""
Structured feature dictionary returned by BiomechanicsEngine.update().

Top-level keys
--------------
joint_angles : Dict[str, float]
temporal_features : Dict[str, Any]
posture_features : Dict[str, Any]
symmetry : Dict[str, float]
movement_quality : Dict[str, Any]
tracking_metrics : Dict[str, Any]
"""


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


class BiomechanicsEngine:
    """
    Computes biomechanical movement features from a stream of MediaPipe
    pose landmark frames.

    Usage
    -----
    engine = BiomechanicsEngine()
    while capturing:
        result = engine.update(landmarks, timestamp, w, h)
        # result is a BiomechanicsResult dict
    summary = engine.get_session_summary()
    engine.reset()  # ready for next set
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Load configuration and initialise rolling history structures."""

        # Configuration — all loaded from config.py, with safe defaults
        self.history_size: int = int(getattr(config, "BIOMECH_HISTORY_SIZE", 60))
        self.smoothing_window: int = int(getattr(config, "BIOMECH_SMOOTHING_WINDOW", 5))
        self.stability_window: int = int(getattr(config, "BIOMECH_STABILITY_WINDOW", 30))
        self.min_confidence: float = float(getattr(config, "BIOMECH_MIN_CONFIDENCE", 0.5))
        self.smoothness_norm: float = float(getattr(config, "BIOMECH_SMOOTHNESS_NORM", 100.0))
        self.stability_norm: float = float(getattr(config, "BIOMECH_STABILITY_NORM", 10.0))

        # Rolling frame history — each entry is a _FrameData dict
        self.history: deque = deque(maxlen=self.history_size)

        # Baseline origin for centre-drift calculation (set on first frame)
        self._start_center: Optional[np.ndarray] = None

        # Session-wide maximum ROM per joint (persists across resets via
        # get_session_summary; reset() clears it)
        self._session_max_rom: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Clear all rolling history and baseline references.
        Call between exercise sets to start a fresh measurement window
        without reinstantiating the engine.
        """
        self.history.clear()
        self._start_center = None
        self._session_max_rom.clear()

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Return aggregate statistics computed over the full recorded history.

        Useful for post-session report generation by downstream consumers.

        Returns an empty dict if fewer than two frames have been recorded.
        """
        if len(self.history) < 2:
            return {}

        all_smoothed = [f["smoothed_angles"] for f in self.history]
        all_speeds   = [f["movement_speed"]   for f in self.history]

        # Per-joint aggregates
        joint_stats: Dict[str, Dict[str, float]] = {}
        for joint in _ANGLE_JOINTS:
            values = [f[joint] for f in all_smoothed]
            joint_stats[joint] = {
                "mean":  round(float(np.mean(values)),  1),
                "min":   round(float(np.min(values)),   1),
                "max":   round(float(np.max(values)),   1),
                "range": round(float(np.max(values) - np.min(values)), 1),
                "std":   round(float(np.std(values)),   2),
            }

        # Session-level speed aggregates
        speed_arr = np.array(all_speeds, dtype=float)

        # Centre-drift: how far the body has wandered from the start position
        all_centers = np.array([f["center_of_body"] for f in self.history])
        total_drift = 0.0
        if self._start_center is not None:
            diffs = all_centers - self._start_center
            total_drift = float(np.max(np.linalg.norm(diffs, axis=1)))

        return {
            "frame_count":        len(self.history),
            "duration_seconds":   round(
                self.history[-1]["timestamp"] - self.history[0]["timestamp"], 3
            ),
            "joint_statistics":   joint_stats,
            "session_max_rom":    dict(self._session_max_rom),
            "mean_speed":         round(float(np.mean(speed_arr)),  2),
            "peak_speed":         round(float(np.max(speed_arr)),   2),
            "max_centre_drift":   round(total_drift, 2),
        }

    def update(
        self,
        landmarks: Any,
        timestamp: float,
        w: int = 640,
        h: int = 480,
    ) -> BiomechanicsResult:
        """
        Process one video frame and return a full biomechanics feature dict.

        Parameters
        ----------
        landmarks : sequence
            Raw MediaPipe NormalizedLandmark list (33 elements).
            Each element must have .x, .y, .z (normalised 0-1) and
            .visibility (0-1 confidence).
        timestamp : float
            Wall-clock time in seconds for this frame.  Used to compute
            all time-derivative quantities.
        w : int
            Frame width in pixels.  Used to de-normalise coordinates.
        h : int
            Frame height in pixels.  Used to de-normalise coordinates.

        Returns
        -------
        BiomechanicsResult
            Structured dict with six top-level sections (see module docstring).
            Returns an empty dict if landmarks are missing or insufficient.
        """
        if landmarks is None or len(landmarks) < 33:
            return {}

        # ── Coordinate extraction ──────────────────────────────────────────
        # Use half the sum of w+h as a uniform z scale so depth is in the
        # same rough magnitude as pixel coordinates.
        scale_z = (w + h) * 0.5

        def _px(idx: int) -> np.ndarray:
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h, lm.z * scale_z], dtype=float)

        pts = {name: _px(idx) for name, idx in _LM.items()}

        # Derived midpoints (used across multiple sections — computed once)
        hip_mid      = (pts["left_hip"]      + pts["right_hip"])      * 0.5
        shoulder_mid = (pts["left_shoulder"] + pts["right_shoulder"]) * 0.5
        curr_center  = hip_mid[:2]   # 2-D hip centre for postural tracking

        # ── Section 1: Joint Angles ────────────────────────────────────────
        raw_angles: Dict[str, float] = {
            "left_knee":      self._angle(pts["left_hip"],      pts["left_knee"],   pts["left_ankle"]),
            "right_knee":     self._angle(pts["right_hip"],     pts["right_knee"],  pts["right_ankle"]),
            "left_hip":       self._angle(pts["left_shoulder"], pts["left_hip"],    pts["left_knee"]),
            "right_hip":      self._angle(pts["right_shoulder"],pts["right_hip"],   pts["right_knee"]),
            "left_elbow":     self._angle(pts["left_shoulder"], pts["left_elbow"],  pts["left_wrist"]),
            "right_elbow":    self._angle(pts["right_shoulder"],pts["right_elbow"], pts["right_wrist"]),
            "left_shoulder":  self._angle(pts["left_elbow"],    pts["left_shoulder"],pts["left_hip"]),
            "right_shoulder": self._angle(pts["right_elbow"],   pts["right_shoulder"],pts["right_hip"]),
            "left_ankle":     self._angle(pts["left_knee"],     pts["left_ankle"],  pts["left_toe"]),
            "right_ankle":    self._angle(pts["right_knee"],    pts["right_ankle"], pts["right_toe"]),
            # Inclination angles (torso / neck vs. vertical)
            "torso_inclination": self._inclination(shoulder_mid - hip_mid),
            "neck_inclination":  self._inclination(pts["nose"] - shoulder_mid),
        }

        # Rolling average smoothing (window size from config)
        smoothed_angles = self._smooth_angles(raw_angles)

        # ── Section 2: Temporal features ──────────────────────────────────
        dt = (timestamp - self.history[-1]["timestamp"]) if self.history else 0.0

        (
            angular_velocity,
            angular_acceleration,
            joint_velocity,
            movement_speed,
        ) = self._compute_temporal(smoothed_angles, pts, curr_center, dt)

        # ── Section 3: Posture features ────────────────────────────────────
        if self._start_center is None:
            self._start_center = curr_center.copy()

        torso_lean        = smoothed_angles["torso_inclination"]
        shoulder_alignment = self._alignment(pts["left_shoulder"], pts["right_shoulder"])
        hip_alignment      = self._alignment(pts["left_hip"],      pts["right_hip"])
        foot_width         = float(np.linalg.norm(pts["left_ankle"][:2] - pts["right_ankle"][:2]))
        shoulder_width     = float(np.linalg.norm(pts["left_shoulder"][:2] - pts["right_shoulder"][:2]))
        stance_ratio       = round(foot_width / shoulder_width, 3) if shoulder_width > 0.0 else 0.0
        center_drift       = float(np.linalg.norm(curr_center - self._start_center))

        # ── Section 4: Symmetry ────────────────────────────────────────────
        symmetry: Dict[str, float] = {
            f"{side}_symmetry": self._symmetry(
                smoothed_angles[f"left_{side}"],
                smoothed_angles[f"right_{side}"],
            )
            for side in _BILATERAL_JOINTS
        }

        # ── Section 5: Movement quality ────────────────────────────────────
        smoothness   = self._compute_smoothness(angular_acceleration)
        stability, body_sway = self._compute_stability(raw_angles, curr_center)
        consistency  = self._compute_consistency()
        rom, normalized_rom_pct = self._compute_rom(smoothed_angles)

        # ── Section 6: Tracking / confidence metrics ───────────────────────
        joint_confidences: Dict[str, float] = {
            "left_knee":      float(landmarks[_LM["left_knee"]].visibility),
            "right_knee":     float(landmarks[_LM["right_knee"]].visibility),
            "left_hip":       float(landmarks[_LM["left_hip"]].visibility),
            "right_hip":      float(landmarks[_LM["right_hip"]].visibility),
            "left_elbow":     float(landmarks[_LM["left_elbow"]].visibility),
            "right_elbow":    float(landmarks[_LM["right_elbow"]].visibility),
            "left_shoulder":  float(landmarks[_LM["left_shoulder"]].visibility),
            "right_shoulder": float(landmarks[_LM["right_shoulder"]].visibility),
            "left_ankle":     float(landmarks[_LM["left_ankle"]].visibility),
            "right_ankle":    float(landmarks[_LM["right_ankle"]].visibility),
        }
        overall_confidence = float(np.mean([lm.visibility for lm in landmarks]))
        low_confidence_joints: List[str] = [
            joint
            for joint, conf in joint_confidences.items()
            if conf < self.min_confidence
        ]

        # ── Commit frame to history ────────────────────────────────────────
        self.history.append({
            "timestamp":          timestamp,
            "pts":                pts,
            "angles":             raw_angles,
            "smoothed_angles":    smoothed_angles,
            "center_of_body":     curr_center.tolist(),
            "angular_velocity":   angular_velocity,
            "angular_acceleration": angular_acceleration,
            "movement_speed":     movement_speed,
        })

        # ── Assemble and return result ─────────────────────────────────────
        return {
            "joint_angles": smoothed_angles,
            "temporal_features": {
                "dt":                   round(dt, 4),
                "angular_velocity":     angular_velocity,
                "angular_acceleration": angular_acceleration,
                "joint_velocity":       joint_velocity,
                "movement_speed":       movement_speed,
            },
            "posture_features": {
                "center_of_body":      [round(float(curr_center[0]), 2),
                                        round(float(curr_center[1]), 2)],
                "torso_lean":          round(torso_lean, 2),
                "shoulder_alignment":  round(shoulder_alignment, 2),
                "hip_alignment":       round(hip_alignment, 2),
                "foot_width":          round(foot_width, 2),
                "shoulder_width":      round(shoulder_width, 2),
                "stance_ratio":        stance_ratio,
                "center_drift":        round(center_drift, 2),
            },
            "symmetry": symmetry,
            "movement_quality": {
                "smoothness":          smoothness,
                "stability":           stability,
                "body_sway":           body_sway,
                "movement_consistency": consistency,
                "range_of_motion":     rom,
                "normalized_rom_pct":  normalized_rom_pct,
            },
            "tracking_metrics": {
                "joint_confidences":   joint_confidences,
                "overall_confidence":  round(overall_confidence, 3),
                "low_confidence_joints": low_confidence_joints,
            },
        }

    # ------------------------------------------------------------------
    # Private geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        2-D joint angle in degrees at vertex *b* between rays b→a and b→c.
        Uses only the x-y plane; z is ignored here so depth noise doesn't
        contaminate the primary angle calculation.
        """
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
        """
        Angle in degrees between a 2-D vector and the upward-vertical axis
        [0, -1] (image coordinates: y increases downward).
        Returns 0 for a perfectly upright torso or neck.
        """
        v = vec[:2]
        norm_v = np.linalg.norm(v)
        if norm_v == 0.0:
            return 0.0
        # vertical = [0, -1]; dot(v, [0,-1]) = -v[1]
        cos_a = -v[1] / norm_v
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    @staticmethod
    def _alignment(p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Horizontal alignment angle of the segment p1→p2 in degrees.
        Returns 0.0 for a perfectly level segment and 90.0 for vertical.
        Range: [0, 90].
        """
        v = p2[:2] - p1[:2]
        raw = abs(float(np.degrees(np.arctan2(v[1], v[0]))))
        # Mirror values > 90 back into [0, 90]
        return abs(180.0 - raw) if raw > 90.0 else raw

    @staticmethod
    def _symmetry(left_val: float, right_val: float) -> float:
        """
        Normalised left-vs-right symmetry index in [0.0, 1.0].
        1.0 = perfect symmetry.  Uses a ratio of the absolute difference
        to the sum, so the result is scale-invariant.
        """
        total = left_val + right_val
        if total < 0.01:   # both near-zero → treat as symmetric
            return 1.0
        return round(1.0 - abs(left_val - right_val) / total, 3)

    # ------------------------------------------------------------------
    # Private computation helpers
    # ------------------------------------------------------------------

    def _smooth_angles(self, raw_angles: Dict[str, float]) -> Dict[str, float]:
        """
        Apply a causal rolling-average smoother of width `smoothing_window`
        to each joint angle using the stored frame history.
        """
        smoothed: Dict[str, float] = {}
        for joint, current in raw_angles.items():
            past = [f["angles"][joint] for f in self.history]
            window = min(len(past) + 1, self.smoothing_window)
            values = (past + [current])[-window:]
            smoothed[joint] = round(sum(values) / len(values), 1)
        return smoothed

    def _compute_temporal(
        self,
        smoothed_angles: Dict[str, float],
        pts: Dict[str, np.ndarray],
        curr_center: np.ndarray,
        dt: float,
    ):
        """
        Compute angular velocity, angular acceleration, per-joint linear
        velocity, and overall movement speed.

        Returns a 4-tuple:
            (angular_velocity, angular_acceleration, joint_velocity, movement_speed)
        """
        if not self.history or dt <= 0.0:
            # First frame or frozen clock — return zero derivatives
            av   = {j: 0.0 for j in smoothed_angles}
            aa   = {j: 0.0 for j in smoothed_angles}
            jv   = {j: 0.0 for j in pts}
            return av, aa, jv, 0.0

        prev           = self.history[-1]
        prev_smoothed  = prev["smoothed_angles"]
        prev_pts       = prev["pts"]
        prev_av        = prev.get("angular_velocity", {})
        prev_center    = np.array(prev["center_of_body"])

        # Angular velocity (°/s) and acceleration (°/s²) — all joints
        angular_velocity: Dict[str, float] = {}
        angular_acceleration: Dict[str, float] = {}
        for joint, angle in smoothed_angles.items():
            v = (angle - prev_smoothed[joint]) / dt
            angular_velocity[joint] = round(v, 2)
            a = (v - prev_av.get(joint, 0.0)) / dt
            angular_acceleration[joint] = round(a, 2)

        # Linear joint velocity (px/s) for every extracted landmark
        joint_velocity: Dict[str, float] = {
            name: round(float(np.linalg.norm(coord[:2] - prev_pts[name][:2])) / dt, 2)
            for name, coord in pts.items()
        }

        # Whole-body movement speed: speed of the hip mid-point (px/s)
        movement_speed = round(
            float(np.linalg.norm(curr_center - prev_center)) / dt, 2
        )

        return angular_velocity, angular_acceleration, joint_velocity, movement_speed

    def _compute_smoothness(
        self, angular_acceleration: Dict[str, float]
    ) -> float:
        """
        Jerk-proxy smoothness score in [0, 1].

        Computed as the mean std of angular acceleration across ALL tracked
        joints over the rolling history.  High acceleration variance (jerky
        motion) drives the score toward 0; near-zero variance yields 1.

        Normalisation constant loaded from config (BIOMECH_SMOOTHNESS_NORM).
        """
        if not self.history:
            return 1.0

        std_values: List[float] = []
        for joint in _ANGLE_JOINTS:
            past_acc = [
                f["angular_acceleration"].get(joint, 0.0)
                for f in self.history
            ]
            current_acc = angular_acceleration.get(joint, 0.0)
            values = past_acc + [current_acc]
            if len(values) > 1:
                std_values.append(float(np.std(values)))

        if not std_values:
            return 1.0

        mean_std = float(np.mean(std_values))
        return round(1.0 / (1.0 + mean_std / self.smoothness_norm), 3)

    def _compute_stability(
        self,
        raw_angles: Dict[str, float],
        curr_center: np.ndarray,
    ):
        """
        Postural stability score in [0, 1] and body sway (std of hip-centre
        horizontal position in px) over the stability window.

        Stability combines torso-lean variance and horizontal centre variance.
        Normalisation constant loaded from config (BIOMECH_STABILITY_NORM).

        Returns (stability: float, body_sway: float).
        """
        window_frames = list(self.history)[-self.stability_window:]

        leans = [f["angles"]["torso_inclination"] for f in window_frames] + [
            raw_angles["torso_inclination"]
        ]
        center_xs = [f["center_of_body"][0] for f in window_frames] + [
            float(curr_center[0])
        ]

        std_lean = float(np.std(leans))   if len(leans)     > 1 else 0.0
        std_cx   = float(np.std(center_xs)) if len(center_xs) > 1 else 0.0

        stability = round(
            1.0 / (1.0 + (std_lean + std_cx) / self.stability_norm), 3
        )
        body_sway = round(std_cx, 2)
        return stability, body_sway

    def _compute_consistency(self) -> float:
        """
        Movement-speed consistency score in [0, 1].

        Defined as 1 − coefficient_of_variation(movement_speed).
        Returns 1.0 when speed is perfectly constant or when fewer than two
        frames are available.
        """
        if len(self.history) < 2:
            return 1.0

        speeds    = np.array([f["movement_speed"] for f in self.history], dtype=float)
        mean_spd  = float(np.mean(speeds))
        if mean_spd <= 0.0:
            return 1.0
        cv = float(np.std(speeds)) / mean_spd
        return round(float(np.clip(1.0 - cv, 0.0, 1.0)), 3)

    def _compute_rom(
        self, smoothed_angles: Dict[str, float]
    ):
        """
        Range of Motion (ROM) for each joint over the full history window,
        and a normalised ROM percentage relative to the session maximum.

        Returns (rom: dict, normalized_rom_pct: dict).
        """
        rom: Dict[str, float] = {}
        normalized_rom_pct: Dict[str, float] = {}

        for joint, current in smoothed_angles.items():
            history_vals = [f["smoothed_angles"][joint] for f in self.history]
            all_vals = history_vals + [current]

            current_rom = round(max(all_vals) - min(all_vals), 1)
            rom[joint] = current_rom

            # Session-lifetime maximum (tracks best ever ROM per joint)
            prev_max = self._session_max_rom.get(joint, 0.0)
            new_max  = max(prev_max, current_rom)
            self._session_max_rom[joint] = new_max

            if new_max > 0.0:
                normalized_rom_pct[joint] = round((current_rom / new_max) * 100.0, 1)
            else:
                normalized_rom_pct[joint] = 100.0

        return rom, normalized_rom_pct
