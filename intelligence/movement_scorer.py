# intelligence/movement_scorer.py
"""
MovementScorer — per-frame movement quality scorer for ExerciseIQ.

Responsibility
--------------
Consume a BiomechanicsResult dict (from BiomechanicsEngine) plus the current
smoothed knee angle and produce a ScoreBreakdown with numeric scores for each
independent quality component.

Design constraints
------------------
- NO imports of cv2, mediapipe, matplotlib, or any drawing/GUI library.
- Scores are in [0, 100].
- All formulas are derived from existing code — no new thresholds introduced.

Scoring rules
-------------
depth
    Same formula as the renderer.py progress bar (line 36), which is the
    authoritative visual representation of depth:
        max(0, min(100, int((160 - smooth_ang) / (160 - DEPTH_THRESHOLD) * 100)))

stability
    BiomechanicsEngine's stability score (0–1) scaled to 0–100.

symmetry
    BiomechanicsEngine's right-knee bilateral symmetry (0–1) scaled to 0–100.

rom
    Normalized ROM percentage for the right knee from BiomechanicsEngine (0–100).

tempo
    Reserved — returns 100.0 (placeholder until timer-based scoring is added).

overall
    Weighted mean of all components.
    Weights: depth=0.35, stability=0.25, symmetry=0.20, rom=0.15, tempo=0.05
"""

from __future__ import annotations

from typing import Any, Dict

import config
from intelligence.models import ScoreBreakdown

# Type alias (mirrors biomechanics.engine.BiomechanicsResult)
BiomechanicsResult = Dict[str, Any]


class MovementScorer:
    """
    Stateless per-frame movement quality scorer.

    Can be subclassed by future exercise analyzers that need different
    depth formulas (e.g. PushupScorer overrides _score_depth).
    """

    # Component weights for overall score — must sum to 1.0
    _WEIGHTS = {
        "depth":     0.35,
        "stability": 0.25,
        "symmetry":  0.20,
        "rom":       0.15,
        "tempo":     0.05,
    }

    def score(
        self,
        bio: BiomechanicsResult,
        smooth_ang: float,
    ) -> ScoreBreakdown:
        """
        Compute ScoreBreakdown for one frame.

        Parameters
        ----------
        bio : BiomechanicsResult
            Dict returned by BiomechanicsEngine.update() for this frame.
            If empty (no landmarks detected) all scores default to 0.
        smooth_ang : float
            Smoothed right knee angle in degrees.

        Returns
        -------
        ScoreBreakdown
        """
        if not bio:
            return ScoreBreakdown()

        depth     = self._score_depth(smooth_ang)
        stability = self._score_stability(bio)
        symmetry  = self._score_symmetry(bio)
        rom       = self._score_rom(bio)
        tempo     = 100.0   # reserved

        overall = round(
            depth     * self._WEIGHTS["depth"]
            + stability * self._WEIGHTS["stability"]
            + symmetry  * self._WEIGHTS["symmetry"]
            + rom       * self._WEIGHTS["rom"]
            + tempo     * self._WEIGHTS["tempo"],
            1,
        )

        return ScoreBreakdown(
            depth=round(depth, 1),
            tempo=round(tempo, 1),
            stability=round(stability, 1),
            symmetry=round(symmetry, 1),
            rom=round(rom, 1),
            overall=overall,
        )

    # ------------------------------------------------------------------
    # Component scorers (overridable by subclasses)
    # ------------------------------------------------------------------

    def _score_depth(self, smooth_ang: float) -> float:
        """
        Depth score derived from the renderer.py progress-bar formula.

        160° = standing (0% depth)
        config.DEPTH_THRESHOLD (100°) = full depth (100%)
        """
        return float(
            max(0, min(100,
                int((160 - smooth_ang) / (160 - config.DEPTH_THRESHOLD) * 100)
            ))
        )

    @staticmethod
    def _score_stability(bio: BiomechanicsResult) -> float:
        """Stability score from BiomechanicsEngine, scaled to 0–100."""
        stability_raw = bio.get("movement_quality", {}).get("stability", 0.0)
        return float(stability_raw) * 100.0

    @staticmethod
    def _score_symmetry(bio: BiomechanicsResult) -> float:
        """Right knee bilateral symmetry from BiomechanicsEngine, scaled to 0–100."""
        sym_raw = bio.get("symmetry", {}).get("knee_symmetry", 0.0)
        return float(sym_raw) * 100.0

    @staticmethod
    def _score_rom(bio: BiomechanicsResult) -> float:
        """Normalized ROM percentage for the right knee (already 0–100)."""
        rom_pct = (
            bio.get("movement_quality", {})
               .get("normalized_rom_pct", {})
               .get("right_knee", 100.0)
        )
        return float(rom_pct)
