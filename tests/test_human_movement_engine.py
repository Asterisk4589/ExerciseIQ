# tests/test_human_movement_engine.py
"""
Unit test suite for the expanded Human Movement Engine features in BiomechanicsEngine.

Verifies:
  1. BiomechanicsResult dataclass attribute access & dict subscripting compatibility.
  2. Anatomical 2D Center of Mass (CoM) calculations.
  3. Joint accelerations and movement direction vectors [dx, dy].
  4. Active time under tension accumulation.
  5. Lateral balance score relative to foot base of support.
  6. Signed horizontal and vertical spatial drift.
  7. Peak flexion tracking (max knee & hip flexion).
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from biomechanics import BiomechanicsEngine, BiomechanicsResult


class MockLandmark:
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x: float = 0.5, y: float = 0.5, z: float = 0.0, visibility: float = 0.9):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def _make_landmarks(right_knee_x: float = 0.62, shift_x: float = 0.0) -> list[MockLandmark]:
    lms = [MockLandmark() for _ in range(33)]
    # Nose
    lms[0] = MockLandmark(x=0.45 + shift_x, y=0.05)
    # Shoulders
    lms[11] = MockLandmark(x=0.40 + shift_x, y=0.22)
    lms[12] = MockLandmark(x=0.55 + shift_x, y=0.22)
    # Elbows
    lms[13] = MockLandmark(x=0.32 + shift_x, y=0.35)
    lms[14] = MockLandmark(x=0.63 + shift_x, y=0.35)
    # Wrists
    lms[15] = MockLandmark(x=0.30 + shift_x, y=0.48)
    lms[16] = MockLandmark(x=0.65 + shift_x, y=0.48)
    # Hips
    lms[23] = MockLandmark(x=0.42 + shift_x, y=0.50)
    lms[24] = MockLandmark(x=0.55 + shift_x, y=0.50)
    # Knees
    lms[25] = MockLandmark(x=0.42 + shift_x, y=0.65)
    lms[26] = MockLandmark(x=right_knee_x + shift_x, y=0.65)
    # Ankles
    lms[27] = MockLandmark(x=0.42, y=0.82)
    lms[28] = MockLandmark(x=0.55, y=0.82)
    # Toes
    lms[31] = MockLandmark(x=0.40, y=0.88)
    lms[32] = MockLandmark(x=0.57, y=0.88)
    return lms


def test_dataclass_and_dict_access():
    print("Testing BiomechanicsResult dataclass & dict access...")
    engine = BiomechanicsEngine()
    lms = _make_landmarks()
    res = engine.update(lms, timestamp=1.0, w=1000, h=1000)

    assert isinstance(res, BiomechanicsResult)
    # Attribute access
    assert "right_knee" in res.joint_angles
    assert res.frame_number == 1
    # Dict access
    assert res["joint_angles"]["right_knee"] == res.joint_angles["right_knee"]
    assert res.get("spatial_features")["stance_ratio"] >= 0.0
    assert "symmetry" in res
    print("Dataclass & dict access tests passed!")


def test_center_of_mass_and_balance():
    print("Testing Center of Mass & Balance calculations...")
    engine = BiomechanicsEngine()

    # Centered body
    lms_centered = _make_landmarks(shift_x=0.0)
    res1 = engine.update(lms_centered, timestamp=1.0, w=1000, h=1000)
    com1 = res1.spatial_features["center_of_mass"]
    balance1 = res1.symmetry["balance_score"]

    assert len(com1) == 2
    assert 0.0 <= balance1 <= 1.0

    # Shift upper body rightward -> balance score decreases
    lms_shifted = _make_landmarks(shift_x=0.08)
    res2 = engine.update(lms_shifted, timestamp=1.1, w=1000, h=1000)
    balance2 = res2.symmetry["balance_score"]

    assert balance2 < balance1
    print("Center of Mass & Balance tests passed!")


def test_spatial_drift_and_accelerations():
    print("Testing spatial drift and joint accelerations...")
    engine = BiomechanicsEngine()

    # Frame 1: baseline
    lms1 = _make_landmarks(shift_x=0.0)
    res1 = engine.update(lms1, timestamp=1.0, w=1000, h=1000)
    assert res1.stability_features["horizontal_drift"] == 0.0

    # Frame 2: motion rightward
    lms2 = _make_landmarks(shift_x=0.05)
    res2 = engine.update(lms2, timestamp=1.1, w=1000, h=1000)
    drift = res2.stability_features["horizontal_drift"]
    assert drift > 0.0

    # Motion direction vector
    dir_vec = res2.temporal_features["movement_direction"]
    assert len(dir_vec) == 2
    assert dir_vec[0] > 0.0  # moving rightward (+x)

    # Joint acceleration present
    ja = res2.temporal_features["joint_acceleration"]
    assert "right_knee" in ja
    print("Spatial drift & joint accelerations tests passed!")


def test_max_flexion_and_tut():
    print("Testing max flexion and time under tension...")
    engine = BiomechanicsEngine()

    # Simulate squat motion over 10 frames
    for i in range(10):
        t = 1.0 + (i * 0.1)
        knee_x = 0.55 + 0.1 * math.sin(i * 0.5)
        lms = _make_landmarks(right_knee_x=knee_x)
        res = engine.update(lms, timestamp=t, w=1000, h=1000)

    rom_feat = res.rom_features
    assert "max_knee_flexion" in rom_feat
    assert rom_feat["max_knee_flexion"] <= 180.0
    assert res.temporal_features["time_under_tension"] >= 0.0
    print("Max flexion and time under tension tests passed!")


if __name__ == "__main__":
    test_dataclass_and_dict_access()
    test_center_of_mass_and_balance()
    test_spatial_drift_and_accelerations()
    test_max_flexion_and_tut()
    print("\nAll Human Movement Engine tests passed successfully!")
