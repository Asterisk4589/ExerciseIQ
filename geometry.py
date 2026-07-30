# geometry.py
"""
Geometric and math calculation helpers for ExerciseIQ.
Calculates coordinate mappings and 2D knee joint angles.
"""

import numpy as np
import config

def calculate_angle(a, b, c):
    """
    Calculates the 2D joint angle at vertex b between endpoints a and c.
    Points a, b, and c are coordinate pairs (e.g., [x, y]).
    Returns the angle in degrees, rounded to 1 decimal place.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    
    # Avoid division by zero
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
        
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return round(float(np.degrees(angle_rad)), 1)

def get_landmark_coords(lm, idx, w, h):
    """
    Extracts the normalized landmark from MediaPipe and scales it to integer pixel coordinates.
    Returns None if landmark visibility is below config.VISIBILITY_MIN.

    lm: The list of landmarks for a detected pose.
    idx: The MediaPipe index of the landmark.
    w: Frame width.
    h: Frame height.
    Returns: A list [x, y] representing pixel coordinates, or None.
    """
    p = lm[idx]
    vis_min = getattr(config, "VISIBILITY_MIN", 0.6)
    if hasattr(p, "visibility") and p.visibility is not None:
        if p.visibility < vis_min:
            return None
    return [int(p.x * w), int(p.y * h)]
