# scratch/test_biomechanics.py
"""
Comprehensive verification suite for BiomechanicsEngine.

Covers:
  - Empty / short landmarks â†’ returns {}
  - All six output sections present and correctly typed
  - Joint angle values in plausible ranges
  - Angular velocity is non-zero when landmarks change
  - Symmetry scores are in [0, 1]
  - Confidence scores are in [0, 1]; low_confidence_joints populated correctly
  - Smoothness / stability / body_sway / ROM after 40+ frames
  - reset() clears history and session_max_rom
  - get_session_summary() returns correct aggregates
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from biomechanics import BiomechanicsEngine

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
_failures: list[str] = []


def _check(name: str, condition: bool) -> None:
    if condition:
        print(f"{PASS}  {name}")
    else:
        print(f"{FAIL}  {name}")
        _failures.append(name)


# ---------------------------------------------------------------------------
# Mock landmark
# ---------------------------------------------------------------------------

class MockLandmark:
    """Minimal stand-in for mediapipe.framework.formats.landmark_pb2.NormalizedLandmark."""
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(
        self,
        x: float = 0.5,
        y: float = 0.5,
        z: float = 0.0,
        visibility: float = 0.9,
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def _make_landmarks(
    *,
    right_knee_x: float = 0.62,
    visibility: float = 0.9,
) -> list[MockLandmark]:
    """
    Build 33 MediaPipe-style landmarks representing a right-side squat pose.
    All joints have sensible, physically plausible positions.
    `right_knee_x` is varied per frame to create angular motion.
    """
    lms = [MockLandmark(visibility=visibility) for _ in range(33)]

    # â”€â”€ head
    lms[0]  = MockLandmark(x=0.45, y=0.05, visibility=visibility)           # nose

    # â”€â”€ shoulders (index 11, 12)
    lms[11] = MockLandmark(x=0.40, y=0.22, visibility=visibility)
    lms[12] = MockLandmark(x=0.55, y=0.22, visibility=visibility)

    # â”€â”€ elbows (13, 14)
    lms[13] = MockLandmark(x=0.32, y=0.35, visibility=visibility)
    lms[14] = MockLandmark(x=0.63, y=0.35, visibility=visibility)

    # â”€â”€ wrists (15, 16)
    lms[15] = MockLandmark(x=0.30, y=0.48, visibility=visibility)
    lms[16] = MockLandmark(x=0.65, y=0.48, visibility=visibility)

    # â”€â”€ hips (23 left, 24 right)
    lms[23] = MockLandmark(x=0.42, y=0.50, visibility=visibility)
    lms[24] = MockLandmark(x=0.55, y=0.50, visibility=visibility)

    # â”€â”€ knees (25 left, 26 right) â€” right knee varies with frame
    lms[25] = MockLandmark(x=0.42, y=0.65, visibility=visibility)
    lms[26] = MockLandmark(x=right_knee_x, y=0.65, visibility=visibility)

    # â”€â”€ ankles (27 left, 28 right)
    lms[27] = MockLandmark(x=0.42, y=0.82, visibility=visibility)
    lms[28] = MockLandmark(x=0.55, y=0.82, visibility=visibility)

    # â”€â”€ toes (31 left, 32 right)
    lms[31] = MockLandmark(x=0.40, y=0.88, visibility=visibility)
    lms[32] = MockLandmark(x=0.57, y=0.88, visibility=visibility)

    return lms


# ---------------------------------------------------------------------------
# Helper: run N frames through the engine
# ---------------------------------------------------------------------------

def _run_frames(
    engine: BiomechanicsEngine,
    n: int = 50,
    start_t: float = 0.0,
    fps: float = 30.0,
    knee_amplitude: float = 0.08,
) -> list[dict]:
    """
    Feed `n` frames to *engine*, oscillating the right knee x-coordinate
    sinusoidally to simulate a squat motion.  Returns list of all results.
    """
    results = []
    for i in range(n):
        t  = start_t + i / fps
        # Sinusoidal knee motion so we get real velocity/ROM/stability signal
        kx = 0.55 + knee_amplitude * math.sin(2 * math.pi * i / 20)
        lms = _make_landmarks(right_knee_x=kx)
        results.append(engine.update(lms, t, w=1280, h=720))
    return results


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_empty_landmarks() -> None:
    print("\nâ”€â”€ Empty / short landmarks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    engine = BiomechanicsEngine()

    r_none  = engine.update(None, 0.0)
    r_short = engine.update([MockLandmark()] * 10, 0.0)

    _check("None landmarks â†’ {}", r_none == {})
    _check("Short landmarks (<33) â†’ {}", r_short == {})


def test_output_structure() -> None:
    print("\nâ”€â”€ Output structure â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    engine = BiomechanicsEngine()
    lms = _make_landmarks()
    result = engine.update(lms, 1.0, w=1280, h=720)

    top_keys = {
        "joint_angles", "temporal_features", "posture_features",
        "symmetry", "movement_quality", "tracking_metrics",
    }
    _check("All top-level keys present", top_keys.issubset(set(result.keys())))

    angles = result["joint_angles"]
    expected_angle_joints = {
        "left_knee", "right_knee", "left_hip", "right_hip",
        "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
        "left_ankle", "right_ankle", "torso_inclination", "neck_inclination",
    }
    _check("All 12 joint angles present", expected_angle_joints == set(angles.keys()))

    temporal = result["temporal_features"]
    _check("temporal: dt present",                   "dt" in temporal)
    _check("temporal: angular_velocity present",      "angular_velocity" in temporal)
    _check("temporal: angular_acceleration present",  "angular_acceleration" in temporal)
    _check("temporal: joint_velocity present",        "joint_velocity" in temporal)
    _check("temporal: movement_speed present",        "movement_speed" in temporal)

    posture = result["posture_features"]
    for key in ("center_of_body","torso_lean","shoulder_alignment","hip_alignment",
                "foot_width","shoulder_width","stance_ratio","center_drift"):
        _check(f"posture: {key} present", key in posture)

    symmetry = result["symmetry"]
    for side in ("knee","hip","elbow","shoulder","ankle"):
        _check(f"symmetry: {side}_symmetry present", f"{side}_symmetry" in symmetry)

    quality = result["movement_quality"]
    for key in ("smoothness","stability","body_sway","movement_consistency",
                "range_of_motion","normalized_rom_pct"):
        _check(f"quality: {key} present", key in quality)

    tracking = result["tracking_metrics"]
    _check("tracking: joint_confidences present",     "joint_confidences" in tracking)
    _check("tracking: overall_confidence present",    "overall_confidence" in tracking)
    _check("tracking: low_confidence_joints present", "low_confidence_joints" in tracking)


def test_value_ranges() -> None:
    print("\nâ”€â”€ Value ranges â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    engine = BiomechanicsEngine()
    results = _run_frames(engine, n=5)
    r = results[-1]

    angles = r["joint_angles"]
    for joint, val in angles.items():
        _check(f"angle {joint} in [0Â°, 180Â°]", 0.0 <= val <= 180.0)

    sym = r["symmetry"]
    for key, val in sym.items():
        _check(f"symmetry {key} in [0, 1]", 0.0 <= val <= 1.0)

    tracking = r["tracking_metrics"]
    _check("overall_confidence in [0,1]",
           0.0 <= tracking["overall_confidence"] <= 1.0)
    for joint, conf in tracking["joint_confidences"].items():
        _check(f"confidence {joint} in [0,1]", 0.0 <= conf <= 1.0)

    posture = r["posture_features"]
    _check("stance_ratio >= 0", posture["stance_ratio"] >= 0.0)
    _check("foot_width >= 0",   posture["foot_width"]   >= 0.0)
    _check("center_drift >= 0", posture["center_drift"] >= 0.0)

    quality = r["movement_quality"]
    _check("smoothness in [0,1]", 0.0 <= quality["smoothness"] <= 1.0)
    _check("stability in [0,1]",  0.0 <= quality["stability"]  <= 1.0)
    _check("body_sway >= 0",      quality["body_sway"]         >= 0.0)
    for joint, val in quality["range_of_motion"].items():
        _check(f"ROM {joint} >= 0", val >= 0.0)


def test_temporal_derivatives() -> None:
    print("\nâ”€â”€ Temporal derivatives (velocity, acceleration) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    engine = BiomechanicsEngine()
    results = _run_frames(engine, n=15)

    # After several sinusoidal frames, at least one joint must have non-zero velocity
    nonzero_vels = [
        j for j, v in results[-1]["temporal_features"]["angular_velocity"].items()
        if abs(v) > 0.01
    ]
    _check("At least one joint has non-zero angular velocity", len(nonzero_vels) > 0)

    # dt should be close to 1/30 â‰ˆ 0.0333 s
    dt = results[-1]["temporal_features"]["dt"]
    _check("dt â‰ˆ 1/30 s (within 1 ms)", abs(dt - 1/30) < 0.001)

    # joint_velocity keys should include all extracted landmark names
    jv_keys = set(results[-1]["temporal_features"]["joint_velocity"].keys())
    _check("joint_velocity includes right_knee", "right_knee" in jv_keys)
    _check("joint_velocity includes nose",       "nose"       in jv_keys)


def test_stability_and_sway_after_many_frames() -> None:
    print("\nâ”€â”€ Stability & body sway (40+ frames) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    engine = BiomechanicsEngine()
    results = _run_frames(engine, n=45)
    quality = results[-1]["movement_quality"]

    _check("stability in [0,1] after 45 frames",
           0.0 <= quality["stability"] <= 1.0)
    # With static hip positions body_sway should be 0 (hips don't move in mock)
    _check("body_sway is a non-negative float", quality["body_sway"] >= 0.0)
    # ROM should be > 0 after the sinusoidal motion
    rk_rom = quality["range_of_motion"]["right_knee"]
    _check("right_knee ROM > 0 after 45 sinusoidal frames", rk_rom > 0.0)


def test_low_confidence_joints() -> None:
    print("\nâ”€â”€ Low-confidence joint detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    engine = BiomechanicsEngine()

    # Feed a frame where all joints have very low visibility
    lms = _make_landmarks(visibility=0.1)
    result = engine.update(lms, 0.0, w=1280, h=720)
    low = result["tracking_metrics"]["low_confidence_joints"]

    _check("low_confidence_joints is a list",          isinstance(low, list))
    _check("All 10 tracked joints flagged at vis=0.1", len(low) == 10)

    # Feed a frame where all joints are high-confidence
    engine2 = BiomechanicsEngine()
    lms_hi = _make_landmarks(visibility=0.95)
    result2 = engine2.update(lms_hi, 0.0, w=1280, h=720)
    low2 = result2["tracking_metrics"]["low_confidence_joints"]
    _check("No joints flagged when visibility=0.95", len(low2) == 0)


def test_reset() -> None:
    print("\nâ”€â”€ reset() clears state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    engine = BiomechanicsEngine()
    _run_frames(engine, n=20)

    _check("History non-empty before reset", len(engine.history) > 0)
    _check("session_max_rom non-empty before reset", len(engine._session_max_rom) > 0)
    _check("_start_center set before reset", engine._start_center is not None)

    engine.reset()

    _check("History empty after reset",          len(engine.history) == 0)
    _check("session_max_rom empty after reset",  len(engine._session_max_rom) == 0)
    _check("_start_center is None after reset",  engine._start_center is None)

    # Engine should work correctly after reset
    results_post = _run_frames(engine, n=5, start_t=100.0)
    _check("Engine produces output after reset", len(results_post[-1]) > 0)


def test_get_session_summary() -> None:
    print("\nâ”€â”€ get_session_summary() â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    engine = BiomechanicsEngine()

    empty_summary = engine.get_session_summary()
    _check("Empty summary when no frames", empty_summary == {})

    _run_frames(engine, n=40)
    summary = engine.get_session_summary()

    _check("summary: frame_count == 40",       summary["frame_count"] == 40)
    _check("summary: duration_seconds > 0",    summary["duration_seconds"] > 0.0)
    _check("summary: joint_statistics present",
           "joint_statistics" in summary and len(summary["joint_statistics"]) == 12)
    _check("summary: session_max_rom present", "session_max_rom" in summary)
    _check("summary: mean_speed present",      "mean_speed" in summary)
    _check("summary: peak_speed >= mean_speed",
           summary["peak_speed"] >= summary["mean_speed"])
    _check("summary: max_centre_drift >= 0",   summary["max_centre_drift"] >= 0.0)

    stats = summary["joint_statistics"]["right_knee"]
    _check("right_knee stats: mean in [0,180]", 0.0 <= stats["mean"] <= 180.0)
    _check("right_knee stats: range >= 0",      stats["range"] >= 0.0)
    _check("right_knee stats: std >= 0",        stats["std"]   >= 0.0)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  BiomechanicsEngine â€” Verification Suite")
    print("=" * 60)

    test_empty_landmarks()
    test_output_structure()
    test_value_ranges()
    test_temporal_derivatives()
    test_stability_and_sway_after_many_frames()
    test_low_confidence_joints()
    test_reset()
    test_get_session_summary()

    print("\n" + "=" * 60)
    if _failures:
        print(f"\033[91m  {len(_failures)} test(s) FAILED:\033[0m")
        for f in _failures:
            print(f"    â€¢ {f}")
        sys.exit(1)
    else:
        total = sum(1 for line in open(__file__, encoding="utf-8") if "_check(" in line)
        print(f"\033[92m  All checks passed ({total} assertions).\033[0m")
    print("=" * 60)


if __name__ == "__main__":
    main()

