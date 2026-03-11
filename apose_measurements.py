"""
A-Pose Perpendicular Bone-Axis Measurement Pipeline.

Computes anthropometric measurements on a clean A-pose mesh using
perpendicular slicing planes aligned to bone axes (NOT flat Y-axis planes).

For each limb segment (e.g. upper arm = shoulder→elbow), the slicing plane
is positioned at the midpoint and its normal is parallel to the bone vector.
This produces anatomically correct cross-sections regardless of limb angle.

Phase 2 of the A-pose measurement pipeline.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# MHR mesh vertex indices (same as body_metrics.py)
# ---------------------------------------------------------------------------
_L_CHEST, _R_CHEST = 7380, 6156
_BELLY_BUTTON = 5711
_BACK_BELLY_BUTTON = 5853
_L_THIGH_IDXS = [16438, 16442, 16443, 16444]
_R_THIGH_IDXS = [11279]
_ARM_SHOULDER = 7953
_ARM_ELBOW = 13495
_ARM_WRIST = 13947


def _make_trimesh(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    """Create a trimesh object from vertices and faces."""
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _perpendicular_slice(
    mesh: trimesh.Trimesh,
    point_a: np.ndarray,
    point_b: np.ndarray,
    fraction: float = 0.5,
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Slice a mesh with a plane perpendicular to the bone axis A→B.

    Args:
        mesh: Trimesh object.
        point_a: Start joint position [3].
        point_b: End joint position [3].
        fraction: Where along A→B to place the plane (0.5 = midpoint).

    Returns:
        cross_section_points: Ordered ring of 3D points on the slice, or None.
        plane_origin: The point where the plane is placed.
        plane_normal: Unit normal of the slicing plane (parallel to bone axis).
    """
    bone_vec = point_b - point_a
    bone_length = np.linalg.norm(bone_vec)
    if bone_length < 1e-6:
        return None, point_a, np.array([0, 1, 0], dtype=float)

    plane_normal = bone_vec / bone_length
    plane_origin = point_a + fraction * bone_vec

    try:
        lines = trimesh.intersections.mesh_plane(
            mesh,
            plane_normal=plane_normal,
            plane_origin=plane_origin,
        )
    except Exception:
        return None, plane_origin, plane_normal

    if lines is None or len(lines) == 0:
        return None, plane_origin, plane_normal

    # Build ordered ring from line segments
    ring = _stitch_segments(lines)
    if ring is None or len(ring) < 3:
        return None, plane_origin, plane_normal

    return ring, plane_origin, plane_normal


def _stitch_segments(lines: np.ndarray, tol: float = 1e-5) -> Optional[np.ndarray]:
    """Stitch line segments into the largest closed/open ring.

    Args:
        lines: (N, 2, 3) array of line segment endpoints.
        tol: Distance tolerance for joining endpoints.

    Returns:
        Ordered ring of 3D points, or None if no valid ring.
    """
    if len(lines) == 0:
        return None

    # Build adjacency from segment endpoints
    points = []
    edges = []
    point_map = {}

    def get_or_add(pt):
        key = tuple(np.round(pt, decimals=5))
        if key in point_map:
            return point_map[key]
        idx = len(points)
        points.append(pt.copy())
        point_map[key] = idx
        return idx

    for seg in lines:
        a = get_or_add(seg[0])
        b = get_or_add(seg[1])
        if a != b:
            edges.append((a, b))

    if not edges:
        return None

    # Build adjacency list
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    # Walk the longest connected chain
    visited_global = set()
    best_chain = []

    for start in adj:
        if start in visited_global:
            continue
        chain = [start]
        visited = {start}
        current = start
        while True:
            neighbors = [n for n in adj.get(current, []) if n not in visited]
            if not neighbors:
                break
            nxt = neighbors[0]
            chain.append(nxt)
            visited.add(nxt)
            current = nxt
        visited_global.update(visited)
        if len(chain) > len(best_chain):
            best_chain = chain

    if len(best_chain) < 3:
        return None

    return np.array([points[i] for i in best_chain])


def _ring_perimeter(ring: np.ndarray) -> float:
    """Compute perimeter of an ordered ring of 3D points (closed loop)."""
    diffs = np.diff(ring, axis=0)
    perimeter = float(np.sum(np.linalg.norm(diffs, axis=1)))
    # Close the loop
    perimeter += float(np.linalg.norm(ring[-1] - ring[0]))
    return perimeter


def _select_nearest_contour(
    ring: np.ndarray,
    center: np.ndarray,
    all_rings: List[np.ndarray],
) -> np.ndarray:
    """From multiple contour rings, select the one whose centroid is nearest to center."""
    if len(all_rings) <= 1:
        return ring

    best_ring = ring
    best_dist = float('inf')
    for r in all_rings:
        centroid = r.mean(axis=0)
        d = float(np.linalg.norm(centroid - center))
        if d < best_dist:
            best_dist = d
            best_ring = r
    return best_ring


def _perpendicular_girth(
    mesh: trimesh.Trimesh,
    joint_a: np.ndarray,
    joint_b: np.ndarray,
    fraction: float = 0.5,
) -> Tuple[Optional[float], Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Compute girth using perpendicular bone-axis slicing.

    Returns:
        raw_perimeter: Perimeter in mesh units (meters), or None.
        ring_points: The cross-section ring, or None.
        plane_origin: Slice position.
        plane_normal: Slice normal.
    """
    ring, origin, normal = _perpendicular_slice(mesh, joint_a, joint_b, fraction)
    if ring is None:
        return None, None, origin, normal

    perimeter = _ring_perimeter(ring)
    return perimeter, ring, origin, normal


def compute_apose_measurements(
    apose_rig: Dict,
    target_height_cm: Optional[float] = None,
) -> Dict:
    """Compute measurements on an A-pose mesh using perpendicular bone-axis slicing.

    This implements Phase 2 of the pipeline:
    - Baseline height scaling
    - Perpendicular slicing at bone midpoints
    - Both raw and normalized measurements

    Args:
        apose_rig: Rig dict with mesh.vertices, mesh.faces, skeleton.joint_positions, etc.
        target_height_cm: Real-world height for scaling. If None, uses mesh height.

    Returns:
        Dict with raw_measurements, normalized_measurements, scale_factor, etc.
    """
    mesh_data = apose_rig.get("mesh", {})
    skeleton = apose_rig.get("skeleton", {})

    vertices = np.asarray(mesh_data.get("vertices"), dtype=np.float64)
    faces_raw = mesh_data.get("faces")
    faces = np.asarray(faces_raw, dtype=np.int32) if faces_raw is not None else None
    joint_positions = np.asarray(skeleton.get("joint_positions"), dtype=np.float64)
    joint_names = skeleton.get("joint_names", [])

    if vertices.size == 0:
        raise ValueError("A-pose mesh vertices are missing")
    if faces is None or faces.size == 0:
        raise ValueError("A-pose mesh faces are missing")

    # Build trimesh
    tm = _make_trimesh(vertices, faces)

    # ---------------------------------------------------------------------------
    # Baseline scaling (Phase 2, Step 1)
    # ---------------------------------------------------------------------------
    y_min = float(vertices[:, 1].min())
    y_max = float(vertices[:, 1].max())
    mesh_height_m = y_max - y_min
    if mesh_height_m <= 0:
        raise ValueError("Invalid A-pose mesh height")

    mesh_height_cm = mesh_height_m * 100.0
    if target_height_cm is None:
        target_height_cm = mesh_height_cm

    scale_factor = float(target_height_cm) / mesh_height_cm

    # ---------------------------------------------------------------------------
    # Joint lookup
    # ---------------------------------------------------------------------------
    joint_map = {}
    for i, name in enumerate(joint_names):
        if i < len(joint_positions):
            joint_map[name] = joint_positions[i]

    def _get_joint(*names):
        for n in names:
            if n in joint_map:
                return joint_map[n]
        return None

    # Key joints
    left_shoulder = _get_joint("left-shoulder", "left_shoulder")
    right_shoulder = _get_joint("right-shoulder", "right_shoulder")
    left_elbow = _get_joint("left-elbow", "left_elbow")
    right_elbow = _get_joint("right-elbow", "right_elbow")
    left_wrist = _get_joint("left-wrist", "left_wrist")
    right_wrist = _get_joint("right-wrist", "right_wrist")
    left_hip = _get_joint("left-hip", "left_hip")
    right_hip = _get_joint("right-hip", "right_hip")
    left_knee = _get_joint("left-knee", "left_knee")
    right_knee = _get_joint("right-knee", "right_knee")
    left_ankle = _get_joint("left-ankle", "left_ankle")
    right_ankle = _get_joint("right-ankle", "right_ankle")
    neck = _get_joint("neck")

    # ---------------------------------------------------------------------------
    # Perpendicular bone-axis slicing (Phase 2, Step 2)
    # ---------------------------------------------------------------------------
    raw_measurements = {}
    slice_data = {}  # For visualization & debug

    def _measure_segment(name, joint_a, joint_b, fraction=0.5):
        """Measure a limb segment with perpendicular slicing."""
        if joint_a is None or joint_b is None:
            return
        raw_m, ring, origin, normal = _perpendicular_girth(tm, joint_a, joint_b, fraction)
        if raw_m is not None:
            raw_measurements[name] = raw_m
            slice_data[name] = {
                "ring_points": ring.round(6).tolist() if ring is not None else [],
                "plane_origin": origin.round(6).tolist(),
                "plane_normal": normal.round(6).tolist(),
                "raw_perimeter_m": round(raw_m, 6),
            }

    # Upper arm (shoulder → elbow) — THE KEY PERPENDICULAR SLICE DEMO
    _measure_segment("left_bicep_girth", left_shoulder, left_elbow, 0.5)
    _measure_segment("right_bicep_girth", right_shoulder, right_elbow, 0.5)

    # Forearm (elbow → wrist)
    _measure_segment("left_forearm_girth", left_elbow, left_wrist, 0.4)
    _measure_segment("right_forearm_girth", right_elbow, right_wrist, 0.4)

    # Thigh (hip → knee)
    _measure_segment("left_thigh_girth", left_hip, left_knee, 0.2)
    _measure_segment("right_thigh_girth", right_hip, right_knee, 0.2)

    # Calf (knee → ankle) — at widest point, ~35% down
    _measure_segment("left_calf_girth", left_knee, left_ankle, 0.35)
    _measure_segment("right_calf_girth", right_knee, right_ankle, 0.35)

    # Wrist (near wrist joint — slice perpendicular to forearm axis)
    _measure_segment("left_wrist_girth", left_elbow, left_wrist, 0.9)
    _measure_segment("right_wrist_girth", right_elbow, right_wrist, 0.9)

    # ---------------------------------------------------------------------------
    # Horizontal slicing for torso measurements (these work well on A-pose)
    # ---------------------------------------------------------------------------
    # Chest — use landmark vertices
    _chest_valid = _L_CHEST < vertices.shape[0] and _R_CHEST < vertices.shape[0]
    if _chest_valid:
        chest_y = float((vertices[_L_CHEST, 1] + vertices[_R_CHEST, 1]) / 2.0)
        chest_ring, _, _ = _perpendicular_slice(
            tm,
            np.array([0, chest_y - 0.01, 0]),
            np.array([0, chest_y + 0.01, 0]),
            0.5,
        )
        if chest_ring is not None:
            raw_measurements["bust_girth"] = _ring_perimeter(chest_ring)
            slice_data["bust_girth"] = {
                "ring_points": chest_ring.round(6).tolist(),
                "plane_origin": [0.0, round(chest_y, 6), 0.0],
                "plane_normal": [0.0, 1.0, 0.0],
                "raw_perimeter_m": round(raw_measurements["bust_girth"], 6),
            }

    # Waist — use belly button landmark
    _waist_valid = _BELLY_BUTTON < vertices.shape[0] and _BACK_BELLY_BUTTON < vertices.shape[0]
    if _waist_valid:
        waist_y = float((vertices[_BELLY_BUTTON, 1] + vertices[_BACK_BELLY_BUTTON, 1]) / 2.0)
        waist_ring, _, _ = _perpendicular_slice(
            tm,
            np.array([0, waist_y - 0.01, 0]),
            np.array([0, waist_y + 0.01, 0]),
            0.5,
        )
        if waist_ring is not None:
            raw_measurements["waist_girth"] = _ring_perimeter(waist_ring)
            slice_data["waist_girth"] = {
                "ring_points": waist_ring.round(6).tolist(),
                "plane_origin": [0.0, round(waist_y, 6), 0.0],
                "plane_normal": [0.0, 1.0, 0.0],
                "raw_perimeter_m": round(raw_measurements["waist_girth"], 6),
            }

    # Hip — at hip joint level
    hip_center = None
    if left_hip is not None and right_hip is not None:
        hip_center = (left_hip + right_hip) / 2.0
    if hip_center is not None:
        hip_y = float(hip_center[1])
        hip_ring, _, _ = _perpendicular_slice(
            tm,
            np.array([0, hip_y - 0.01, 0]),
            np.array([0, hip_y + 0.01, 0]),
            0.5,
        )
        if hip_ring is not None:
            raw_measurements["hip_girth"] = _ring_perimeter(hip_ring)
            slice_data["hip_girth"] = {
                "ring_points": hip_ring.round(6).tolist(),
                "plane_origin": [0.0, round(hip_y, 6), 0.0],
                "plane_normal": [0.0, 1.0, 0.0],
                "raw_perimeter_m": round(raw_measurements["hip_girth"], 6),
            }

    # Neck — horizontal slice at neck joint
    if neck is not None:
        neck_y = float(neck[1]) - 0.01  # slightly below neck joint
        neck_ring, _, _ = _perpendicular_slice(
            tm,
            np.array([0, neck_y - 0.005, 0]),
            np.array([0, neck_y + 0.005, 0]),
            0.5,
        )
        if neck_ring is not None:
            raw_measurements["neck_girth"] = _ring_perimeter(neck_ring)
            slice_data["neck_girth"] = {
                "ring_points": neck_ring.round(6).tolist(),
                "plane_origin": [0.0, round(neck_y, 6), 0.0],
                "plane_normal": [0.0, 1.0, 0.0],
                "raw_perimeter_m": round(raw_measurements["neck_girth"], 6),
            }

    # ---------------------------------------------------------------------------
    # Height & arm length
    # ---------------------------------------------------------------------------
    raw_measurements["body_height"] = mesh_height_m

    # Arm length — surface vertices
    if all(idx < vertices.shape[0] for idx in [_ARM_SHOULDER, _ARM_ELBOW, _ARM_WRIST]):
        sh_pt = vertices[_ARM_SHOULDER]
        el_pt = vertices[_ARM_ELBOW]
        wr_pt = vertices[_ARM_WRIST]
        arm_len = float(np.linalg.norm(el_pt - sh_pt) + np.linalg.norm(wr_pt - el_pt))
        raw_measurements["arm_length"] = arm_len

    # Shoulder width
    if left_shoulder is not None and right_shoulder is not None:
        raw_measurements["shoulder_width"] = float(np.linalg.norm(left_shoulder - right_shoulder))

    # ---------------------------------------------------------------------------
    # Average bilateral measurements
    # ---------------------------------------------------------------------------
    bilateral_pairs = [
        ("bicep_girth", "left_bicep_girth", "right_bicep_girth"),
        ("forearm_girth", "left_forearm_girth", "right_forearm_girth"),
        ("thigh_girth", "left_thigh_girth", "right_thigh_girth"),
        ("calf_girth", "left_calf_girth", "right_calf_girth"),
        ("wrist_girth", "left_wrist_girth", "right_wrist_girth"),
    ]
    for avg_name, left_key, right_key in bilateral_pairs:
        left_val = raw_measurements.get(left_key)
        right_val = raw_measurements.get(right_key)
        vals = [v for v in (left_val, right_val) if v is not None]
        if vals:
            raw_measurements[avg_name] = sum(vals) / len(vals)

    # ---------------------------------------------------------------------------
    # Build output: raw (mesh-space meters) + normalized (real-world cm)
    # ---------------------------------------------------------------------------
    raw_cm = {}
    normalized_cm = {}
    for key, val_m in raw_measurements.items():
        if val_m is None:
            continue
        raw_cm[key] = round(val_m * 100.0, 2)
        normalized_cm[key] = round(val_m * scale_factor * 100.0, 2)

    # Schema for frontend
    schema = {}
    for key in raw_measurements:
        schema[key] = {
            "unit": "cm",
            "category": _categorize(key),
            "scales_with_height": True,
        }

    return {
        "mode": "apose",
        "actual_height_cm": round(mesh_height_cm, 2),
        "target_height_cm": round(target_height_cm, 2),
        "scale_factor": round(scale_factor, 4),
        "raw_measurements": raw_cm,
        "normalized_measurements": normalized_cm,
        "slice_data": slice_data,
        "schema": schema,
    }


def _categorize(key: str) -> str:
    """Assign a display category to a measurement key."""
    if "height" in key:
        return "vertical"
    if "girth" in key:
        return "girth"
    if "width" in key or "length" in key:
        return "width"
    return "special"
