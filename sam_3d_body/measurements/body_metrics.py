import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


class MeasurementError(RuntimeError):
    """Raised when a measurement cannot be derived from the provided rig."""


@dataclass(frozen=True)
class MeasurementMeta:
    key: str
    unit: str
    scales_with_height: bool
    category: str  # e.g. "vertical", "girth", "width", "special", "angle"


MEASUREMENT_META: Dict[str, MeasurementMeta] = {
    "body_height": MeasurementMeta("body_height", "cm", True, "vertical"),
    "eye_height": MeasurementMeta("eye_height", "cm", True, "vertical"),
    "cervicale_height": MeasurementMeta("cervicale_height", "cm", True, "vertical"),
    "waist_height": MeasurementMeta("waist_height", "cm", True, "vertical"),
    "hip_height": MeasurementMeta("hip_height", "cm", True, "vertical"),
    "inside_leg_height": MeasurementMeta("inside_leg_height", "cm", True, "vertical"),
    "knee_height": MeasurementMeta("knee_height", "cm", True, "vertical"),
    "head_girth": MeasurementMeta("head_girth", "cm", True, "girth"),
    "neck_girth": MeasurementMeta("neck_girth", "cm", True, "girth"),
    "bust_girth": MeasurementMeta("bust_girth", "cm", True, "girth"),
    "underbust_girth": MeasurementMeta("underbust_girth", "cm", True, "girth"),
    "waist_girth": MeasurementMeta("waist_girth", "cm", True, "girth"),
    "hip_girth": MeasurementMeta("hip_girth", "cm", True, "girth"),
    "thigh_girth": MeasurementMeta("thigh_girth", "cm", True, "girth"),
    "knee_girth": MeasurementMeta("knee_girth", "cm", True, "girth"),
    "calf_girth": MeasurementMeta("calf_girth", "cm", True, "girth"),
    "ankle_girth": MeasurementMeta("ankle_girth", "cm", True, "girth"),
    "upper_arm_girth": MeasurementMeta("upper_arm_girth", "cm", True, "girth"),
    "wrist_girth": MeasurementMeta("wrist_girth", "cm", True, "girth"),
    "shoulder_width": MeasurementMeta("shoulder_width", "cm", True, "width"),
    "back_width": MeasurementMeta("back_width", "cm", True, "width"),
    "chest_width": MeasurementMeta("chest_width", "cm", True, "width"),
    "arm_length": MeasurementMeta("arm_length", "cm", True, "special"),
    "total_crotch_length": MeasurementMeta("total_crotch_length", "cm", True, "special"),
    "shoulder_to_crotch": MeasurementMeta("shoulder_to_crotch", "cm", True, "vertical"),
    "shoulder_slope": MeasurementMeta("shoulder_slope", "deg", False, "angle"),
}


def _vector_map(names: Iterable[str], positions: np.ndarray) -> Dict[str, np.ndarray]:
    mapping: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(names):
        if idx >= positions.shape[0]:
            break
        mapping[name] = positions[idx]
    return mapping


def _keypoint_map(keypoints: Iterable[Dict[str, Iterable[float]]]) -> Dict[str, np.ndarray]:
    mapping: Dict[str, np.ndarray] = {}
    for entry in keypoints:
        name = entry.get("name")
        pos = entry.get("position")
        if name is None or pos is None:
            continue
        arr = np.asarray(pos, dtype=np.float32)
        if arr.shape == (3,):
            mapping[name] = arr
    return mapping


def _average_points(points: Iterable[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    valid = [p for p in points if p is not None]
    if not valid:
        return None
    stacked = np.stack(valid, axis=0)
    return stacked.mean(axis=0)


def _average_values(values: Iterable[Optional[float]]) -> Optional[float]:
    valid: List[float] = []
    for value in values:
        if value is None:
            continue
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(scalar):
            continue
        valid.append(scalar)
    if not valid:
        return None
    return sum(valid) / len(valid)


def _coalesce_points(*points: Optional[np.ndarray]) -> Optional[np.ndarray]:
    for point in points:
        if point is not None:
            return point
    return None


def _convex_hull(points: np.ndarray) -> Optional[np.ndarray]:
    """2D convex hull using Andrew's monotone chain algorithm."""
    if points.shape[0] < 3:
        return None

    pts = np.unique(points, axis=0)
    if pts.shape[0] < 3:
        return None

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float32)
    if hull.shape[0] < 3:
        return None
    return hull


def _perimeter_from_points(points: np.ndarray) -> Optional[float]:
    hull = _convex_hull(points)
    if hull is None:
        return None
    shifted = np.roll(hull, -1, axis=0)
    return float(np.linalg.norm(hull - shifted, axis=1).sum())


def _slab_at_plane(vertices: np.ndarray, plane_origin: np.ndarray, plane_normal: np.ndarray, thickness: float) -> np.ndarray:
    """Select vertices within ±thickness of the given plane."""
    n = np.asarray(plane_normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-9)
    signed_dist = (vertices - np.asarray(plane_origin)) @ n
    return vertices[np.abs(signed_dist) <= thickness]


def _exclude_arms_xz(pts_3d: np.ndarray, chest_midpoint_3d: np.ndarray, arm_gap_m: float = 0.22) -> np.ndarray:
    """
    Remove arm cross-sections from a chest/torso slice.
    Keeps only points within arm_gap_m of the chest centre in the XZ plane.

    arm_gap_m = 0.22 m covers most adult chest half-widths with margin.
    Decrease to 0.18 if arms still bleed in; increase to 0.26 if the
    chest sides are being clipped.
    """
    midpoint_xz = np.array([chest_midpoint_3d[0], chest_midpoint_3d[2]])
    dist = np.linalg.norm(pts_3d[:, [0, 2]] - midpoint_xz, axis=1)
    return pts_3d[dist <= arm_gap_m]


def _trimesh_plane_slice(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
) -> Optional[np.ndarray]:
    """Exact trimesh intersection for a tilted plane. Returns (N,3) points or None."""
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        segments = trimesh.intersections.mesh_plane(mesh, plane_normal, plane_origin)
        if segments is None or len(segments) == 0:
            return None
        return segments.reshape(-1, 3)
    except Exception:
        return None


def _build_contours_from_segments(segments: np.ndarray, eps: float = 1e-4) -> List[np.ndarray]:
    """
    Stitch trimesh intersection segments (K, 2, 3) into ordered closed contours.

    trimesh.intersections.mesh_plane returns one closed loop per body-part
    cross-section (torso, left arm, right arm, …).  This function reconstructs
    those loops as ordered vertex arrays so we can measure each one separately.

    Returns a list of (M_i, 3) arrays, one per contour, largest first.
    """
    n = len(segments)
    if n == 0:
        return []

    # Hash each endpoint to a grid cell for O(1) neighbour lookup
    scale = 1.0 / max(eps, 1e-9)

    def _key(pt: np.ndarray):
        return (int(round(pt[0] * scale)),
                int(round(pt[1] * scale)),
                int(round(pt[2] * scale)))

    # endpoint_map: grid_key -> list of (segment_idx, which_end {0 or 1})
    endpoint_map: Dict[tuple, List] = {}
    for i in range(n):
        for e in (0, 1):
            k = _key(segments[i, e])
            if k not in endpoint_map:
                endpoint_map[k] = []
            endpoint_map[k].append((i, e))

    used = np.zeros(n, dtype=bool)
    contours: List[np.ndarray] = []

    for start in range(n):
        if used[start]:
            continue
        used[start] = True
        pts = [segments[start, 0].copy(), segments[start, 1].copy()]

        # Walk the chain until we can no longer extend
        for _ in range(n):
            tail_key = _key(pts[-1])
            found = False
            for seg_idx, ep_idx in endpoint_map.get(tail_key, []):
                if used[seg_idx]:
                    continue
                # Connect: other endpoint of this segment
                next_pt = segments[seg_idx, 1 - ep_idx].copy()
                pts.append(next_pt)
                used[seg_idx] = True
                found = True
                break
            if not found:
                break

        if len(pts) >= 3:
            contours.append(np.array(pts, dtype=float))

    # Sort largest first (most segments = main torso ring)
    contours.sort(key=len, reverse=True)
    return contours


def _trimesh_chest_contour(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    chest_center_3d: np.ndarray,
    exclusion_xz: Optional[List[np.ndarray]] = None,
    exclusion_radius: float = 0.08,
) -> Optional[np.ndarray]:
    """
    Slice the mesh with the plane using trimesh and return the ordered 3-D
    contour of the torso ring (the closed loop nearest to chest_center_3d).

    exclusion_xz : list of (x, z) 1-D arrays — XZ positions of wrist/arm
                   joints used to dynamically compute the torso protection zone.
                   Any segment whose XZ midpoint is farther from the body
                   centre than (82 % of the wrist-to-centre distance) is
                   dropped BEFORE contour stitching.  This keeps ALL torso
                   segments (which are always within ~75 % of the wrist
                   distance) while discarding arm cross-sections that sit
                   beyond that boundary.  Works for any body size because the
                   threshold scales with the actual wrist position.
    exclusion_radius : fallback arm-column radius (metres) used only for
                       segments that lie between the torso zone and the wrist.

    Returns (M, 3) ordered ring vertices, or None on failure.
    """
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        segments = trimesh.intersections.mesh_plane(mesh, plane_normal, plane_origin)
        if segments is None or len(segments) < 3:
            return None

        # ── Arm / hand segment exclusion ─────────────────────────────────────
        # Strategy: keep every segment whose XZ midpoint is within 82 % of the
        # minimum wrist-to-centre distance (the "torso safe zone").  Segments
        # outside that zone are excluded only if they are also within
        # exclusion_radius of a wrist joint.  This two-tier check prevents the
        # fixed exclusion radius from accidentally clipping the torso sides
        # (the failure mode that produced half-circumference rings).
        if exclusion_xz:
            body_center_xz = np.array([chest_center_3d[0], chest_center_3d[2]])
            min_wrist_dist = min(
                float(np.linalg.norm(ex - body_center_xz)) for ex in exclusion_xz
            )
            torso_safe_radius = min_wrist_dist * 0.82  # always inside the arm column

            keep = []
            for seg in segments:
                mid_xz = np.array([(seg[0, 0] + seg[1, 0]) / 2.0,
                                   (seg[0, 2] + seg[1, 2]) / 2.0])
                dist_to_center = float(np.linalg.norm(mid_xz - body_center_xz))

                # Torso zone: always keep — never clip here
                if dist_to_center <= torso_safe_radius:
                    keep.append(seg)
                    continue

                # Outside torso zone: exclude only if near a wrist/arm joint
                in_arm = any(
                    float(np.linalg.norm(mid_xz - ex)) < exclusion_radius
                    for ex in exclusion_xz
                )
                if not in_arm:
                    keep.append(seg)

            if not keep:
                return None
            segments = np.array(keep, dtype=float)

        contours = _build_contours_from_segments(segments)
        if not contours:
            return None

        # Pick the contour whose centroid is closest to the known chest centre
        best: Optional[np.ndarray] = None
        best_dist = float('inf')
        for c in contours:
            d = float(np.linalg.norm(c.mean(axis=0) - chest_center_3d))
            if d < best_dist:
                best_dist = d
                best = c

        return best if (best is not None and len(best) >= 3) else None
    except Exception:
        return None


def _chest_circumference_smplx_style(
    vertices: np.ndarray,
    faces: Optional[np.ndarray],
    left_chest_idx: int,
    right_chest_idx: int,
    pelvis_joint: Optional[np.ndarray] = None,
    spine_joint: Optional[np.ndarray] = None,
    left_wrist_joint: Optional[np.ndarray] = None,
    right_wrist_joint: Optional[np.ndarray] = None,
    thickness: float = 0.025,
    arm_gap_m: float = 0.22,
) -> Tuple[float, np.ndarray]:
    """
    Compute full chest circumference using the SMPLX-style landmark-anchored
    approach.

    Strategy
    --------
    1. Plane origin  = midpoint of the two chest landmark vertices.
    2. Plane normal  = spine axis (pelvis→spine2) or world-Y fallback.
    3. Trimesh exact slice → arm segments excluded via wrist-joint 2-tier gate
       → pick the torso contour nearest to the chest centre.
    4. Perimeter = sum of segment lengths along the ordered ring (exact).
    5. Fallback: vertex slab + _exclude_arms_xz gate + convex hull.

    Returns (girth_metres, ring_3d) where ring_3d is (M, 3) ordered ring
    vertices for Three.js visualisation (empty array on fallback path).
    """
    vertices = np.asarray(vertices, dtype=float)

    left_pt = vertices[left_chest_idx]
    right_pt = vertices[right_chest_idx]
    plane_origin = (left_pt + right_pt) / 2.0  # front-of-chest midpoint

    # Chest girth is always measured horizontally (like a tape measure),
    # regardless of body posture.
    plane_normal = np.array([0.0, 1.0, 0.0])

    # Build wrist exclusion zones (same 2-tier logic used by waist)
    _excl_xz: Optional[List[np.ndarray]] = None
    _wrist_pts = [w for w in (left_wrist_joint, right_wrist_joint) if w is not None]
    if _wrist_pts:
        _excl_xz = [np.array([float(np.asarray(w, dtype=float)[0]),
                               float(np.asarray(w, dtype=float)[2])]) for w in _wrist_pts]

    # ── TRIMESH PATH (preferred) ──────────────────────────────────────────────
    if faces is not None:
        ring_3d = _trimesh_chest_contour(
            vertices, faces, plane_origin, plane_normal, plane_origin,
            exclusion_xz=_excl_xz,
        )
        if ring_3d is not None and len(ring_3d) >= 3:
            # Perimeter = sum of consecutive distances along the ordered ring
            diffs = np.diff(ring_3d, axis=0)
            girth_m = float(np.linalg.norm(diffs, axis=1).sum())
            # Close the loop
            close_diff = np.linalg.norm(ring_3d[-1] - ring_3d[0])
            girth_m += close_diff
            return girth_m, ring_3d

    # ── FALLBACK: vertex slab + arm exclusion + convex hull ───────────────────
    dominant = int(np.argmax(np.abs(plane_normal)))
    keep = [i for i in range(3) if i != dominant]

    t = thickness
    pts: Optional[np.ndarray] = None
    for _ in range(4):
        pts = _slab_at_plane(vertices, plane_origin, plane_normal, t)
        if pts is not None and len(pts) >= 12:
            break
        t *= 1.5

    if pts is not None and len(pts) >= 4:
        # Apply arm exclusion: wrist-based dynamic gate if available,
        # else fall back to the fixed arm_gap_m XZ radius from chest centre.
        if _excl_xz:
            body_center_xz = np.array([plane_origin[0], plane_origin[2]])
            min_wrist_dist = min(np.linalg.norm(ex - body_center_xz) for ex in _excl_xz)
            max_torso_r = min_wrist_dist * 0.82
            pts = pts[np.linalg.norm(pts[:, [0, 2]] - body_center_xz, axis=1) <= max_torso_r]
        else:
            pts = _exclude_arms_xz(pts, plane_origin, arm_gap_m)

    if pts is not None and len(pts) >= 4:
        pts_2d = pts[:, keep]
        hull_2d = _convex_hull(pts_2d)
        if hull_2d is not None and len(hull_2d) >= 3:
            shifted = np.roll(hull_2d, -1, axis=0)
            girth_m = float(np.linalg.norm(hull_2d - shifted, axis=1).sum())
            avg_n = float(np.mean(pts[:, dominant]))
            hull_3d = np.zeros((len(hull_2d), 3), dtype=float)
            for i, ax in enumerate(keep):
                hull_3d[:, ax] = hull_2d[:, i]
            hull_3d[:, dominant] = avg_n
            return girth_m, hull_3d

    # Last-resort: Ramanujan ellipse from bounding box
    if pts is not None and len(pts) >= 2:
        pts_2d = pts[:, keep]
        w = float(pts_2d[:, 0].max() - pts_2d[:, 0].min())
        d = float(pts_2d[:, 1].max() - pts_2d[:, 1].min())
    else:
        w, d = 0.30, 0.20
    a, b = max(w, d) / 2, min(w, d) / 2
    h = ((a - b) / (a + b)) ** 2
    girth_m = math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))
    return girth_m, np.zeros((0, 3), dtype=float)


def _exclude_hands_at_waist(
    pts_3d: np.ndarray,
    waist_midpoint_3d: np.ndarray,
    left_wrist_joint: Optional[np.ndarray] = None,
    right_wrist_joint: Optional[np.ndarray] = None,
    hand_exclusion_radius_m: float = 0.08,
    cluster_min_gap_m: float = 0.12,
    cluster_min_size_ratio: float = 0.04,
) -> np.ndarray:
    """
    Remove hand/wrist cross-sections from the waist slice.

    WHY NEEDED: In natural standing / A-pose, hands hang at waist height.
    Each hand appears as a small disconnected blob in the XZ cross-section,
    offset left and right of the torso.  Without exclusion the convex hull
    stretches to include both hands, inflating waist by 5–15 cm.

    WHY NOT the 82% wrist-distance gate used for chest:
    At chest level wrists sit well clear of the torso; 82% of their XZ
    distance is always outside the torso edges.  At WAIST level wrists may
    be as close as 15 cm from the body centre — 82% × 15 cm = 12.3 cm,
    which clips the outer waist contour (17 cm half-width) and UNDER-
    measures.  The joint-anchored 8 cm exclusion targets only the hand blob
    regardless of how close the wrist is to the body.

    Strategy 1 (preferred — joint-anchored):
        Exclude all points within hand_exclusion_radius_m of each wrist joint
        in XZ.  Radius 8 cm covers the hand/wrist cross-section without
        touching the torso (typical wrist width 4–5 cm).

    Strategy 2 (fallback — cluster-gap, no joint data):
        Sort points by X.  Find gaps > cluster_min_gap_m between adjacent
        points.  Remove small clusters (< cluster_min_size_ratio of total)
        that are far from the waist centre — these are hand blobs.
    """
    if len(pts_3d) == 0:
        return pts_3d

    pts_xz = pts_3d[:, [0, 2]]

    # ── Strategy 1: joint-anchored ───────────────────────────────────────────
    joints_used = False
    keep_mask = np.ones(len(pts_3d), dtype=bool)
    for wrist_joint in [left_wrist_joint, right_wrist_joint]:
        if wrist_joint is not None:
            wrist_xz = np.array([float(wrist_joint[0]), float(wrist_joint[2])])
            dist = np.linalg.norm(pts_xz - wrist_xz, axis=1)
            keep_mask &= (dist > hand_exclusion_radius_m)
            joints_used = True
    if joints_used:
        return pts_3d[keep_mask]

    # ── Strategy 2: cluster-gap ───────────────────────────────────────────────
    x_coords = pts_xz[:, 0]
    sort_idx  = np.argsort(x_coords)
    x_sorted  = x_coords[sort_idx]
    gaps = np.diff(x_sorted)
    gap_positions = np.where(gaps > cluster_min_gap_m)[0]
    if len(gap_positions) == 0:
        return pts_3d

    boundaries = [0] + list(gap_positions + 1) + [len(sort_idx)]
    waist_xz = np.array([waist_midpoint_3d[0], waist_midpoint_3d[2]])
    n_total = len(pts_3d)
    keep_indices: List[int] = []
    for i in range(len(boundaries) - 1):
        cluster_idx = sort_idx[boundaries[i]:boundaries[i + 1]]
        cluster_pts = pts_xz[cluster_idx]
        centroid = cluster_pts.mean(axis=0)
        size_ratio = len(cluster_idx) / n_total
        dist_from_waist = float(np.linalg.norm(centroid - waist_xz))
        if size_ratio >= cluster_min_size_ratio or dist_from_waist < 0.15:
            keep_indices.extend(cluster_idx.tolist())

    if not keep_indices:
        return pts_3d
    return pts_3d[np.array(keep_indices)]


def _filter_waist_region(
    pts_3d: np.ndarray,
    waist_midpoint_3d: np.ndarray,
    max_radius_m: float = 0.28,
) -> np.ndarray:
    """
    Remove points beyond max_radius_m from the waist centre in XZ.

    Loose backstop (0.28 m) — hand exclusion above already removed the main
    contaminants.  This catches any remaining far-outlier artefacts (clothing
    geometry, loose mesh fragments) without clipping a wide waist.
    """
    midpoint_xz = np.array([waist_midpoint_3d[0], waist_midpoint_3d[2]])
    dist = np.linalg.norm(pts_3d[:, [0, 2]] - midpoint_xz, axis=1)
    return pts_3d[dist <= max_radius_m]


def _trimesh_waist_contour(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    belly_button_pt: np.ndarray,
    back_belly_pt: np.ndarray,
    left_wrist_joint: Optional[np.ndarray] = None,
    right_wrist_joint: Optional[np.ndarray] = None,
    wrist_exclusion_r: float = 0.08,
) -> Optional[np.ndarray]:
    """
    Slice the mesh at plane_origin/plane_normal and return the ordered 3-D
    waist ring (torso contour only).

    DESIGN: stitch-first, select-by-centroid  (mirrors _trimesh_chest_contour)
    ---------------------------------------------------------------------------
    Segment-level XZ gates were removed because the waist cross-section is an
    oval that is WIDER left-right than it is deep front-to-back.  Any gate
    derived from the front-back depth will clip the left/right side segments,
    breaking the torso ring into an open arc whose two ends are then joined by
    a straight diagonal through the mesh interior.

    Instead we rely on topology: trimesh.intersections.mesh_plane returns one
    closed loop per body-part cross-section (torso, left arm, right arm, hand
    fingers, …).  These loops are disconnected from each other — no segment in
    the torso ring shares an endpoint with any segment in the arm or hand rings.
    _build_contours_from_segments therefore produces fully separate contours for
    each body part.  We then pick the contour whose centroid is CLOSEST to the
    waist centre (midpoint of belly-button and back-belly landmarks).  The torso
    centroid is always close to the body centre; arm/hand centroids sit 20–40 cm
    away — the selection is unambiguous.

    Wrist joint filter (optional belt-and-suspenders)
    --------------------------------------------------
    If wrist joints are provided and the pose is unusually close-armed, a
    secondary per-contour filter removes any contour whose centroid is within
    wrist_exclusion_r of a known wrist position.  This never touches the torso
    contour because the torso centroid is always near (0, y, 0), far from any
    wrist joint.

    Returns (M, 3) ordered ring or None on failure.
    """
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        segments = trimesh.intersections.mesh_plane(mesh, plane_normal, plane_origin)
        if segments is None or len(segments) < 3:
            return None

        # Waist centre in XZ (midpoint of the two surface landmarks)
        center_xz = np.array([
            (belly_button_pt[0] + back_belly_pt[0]) / 2.0,
            (belly_button_pt[2] + back_belly_pt[2]) / 2.0,
        ])
        waist_center_3d = np.array([center_xz[0], plane_origin[1], center_xz[1]])

        # ── Stitch ALL segments into closed contours ──────────────────────────
        # No pre-filtering: hands/arms form separate disconnected loops and will
        # never be stitched into the torso ring by _build_contours_from_segments.
        contours = _build_contours_from_segments(segments)
        if not contours:
            return None

        # ── Optional: remove contours centred on a known wrist joint ──────────
        if left_wrist_joint is not None or right_wrist_joint is not None:
            wrists_xz = [
                np.array([float(w[0]), float(w[2])])
                for w in [left_wrist_joint, right_wrist_joint]
                if w is not None
            ]
            filtered = [
                c for c in contours
                if len(c) >= 3 and not any(
                    float(np.linalg.norm(
                        np.array([c[:, 0].mean(), c[:, 2].mean()]) - wx
                    )) < wrist_exclusion_r
                    for wx in wrists_xz
                )
            ]
            if filtered:
                contours = filtered

        # ── Select contour closest to waist centre ────────────────────────────
        best: Optional[np.ndarray] = None
        best_dist = float('inf')
        for c in contours:
            if len(c) < 3:
                continue
            d = float(np.linalg.norm(c.mean(axis=0) - waist_center_3d))
            if d < best_dist:
                best_dist = d
                best = c

        return best if (best is not None and len(best) >= 3) else None
    except Exception:
        return None


def _waist_girth_from_ring(ring_3d: np.ndarray) -> float:
    """Sum of consecutive segment lengths along ordered ring (closed loop)."""
    diffs = np.diff(ring_3d, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum()) + float(
        np.linalg.norm(ring_3d[-1] - ring_3d[0])
    )


def _waist_circumference_smplx_style(
    vertices: np.ndarray,
    faces: Optional[np.ndarray],
    belly_button_idx: int,
    pelvis_joint: Optional[np.ndarray] = None,
    spine3_joint: Optional[np.ndarray] = None,
    left_wrist_joint: Optional[np.ndarray] = None,
    right_wrist_joint: Optional[np.ndarray] = None,
    thickness: float = 0.020,
) -> Tuple[float, np.ndarray]:
    """
    SMPLX-style waist circumference — uses the same trimesh contour logic
    as chest: wrist-based 2-tier arm exclusion before stitching, then exact
    ordered-ring perimeter (no convex hull).

    Plane origin = midpoint of belly_button and back_belly vertices.
    Plane normal = spine axis (pelvis→spine3) when available, else world-Y.
    Arm exclusion = same _trimesh_chest_contour with wrist XZ gates.

    Fallback: vertex slab → adaptive XZ gate → convex hull.

    Returns (girth_metres, ring_3d).
    """
    vertices = np.asarray(vertices, dtype=float)

    front_pt = vertices[belly_button_idx]

    # Plane normal = spine axis (pelvis→spine3) when available.
    # This makes the cut perpendicular to the actual torso direction,
    # correctly handling forward-leaning or tilted bodies.
    if pelvis_joint is not None and spine3_joint is not None:
        _spine_vec = np.asarray(spine3_joint, dtype=float) - np.asarray(pelvis_joint, dtype=float)
        _spine_len = float(np.linalg.norm(_spine_vec))
        if _spine_len > 1e-6:
            plane_normal = _spine_vec / _spine_len
        else:
            plane_normal = np.array([0.0, 1.0, 0.0])
    else:
        plane_normal = np.array([0.0, 1.0, 0.0])

    # Plane origin: the front navel vertex (5711) — single anchor point.
    base_origin = front_pt.copy()

    # ── TRIMESH PATH: same logic as chest (arm exclusion before stitching) ────
    best_girth: Optional[float] = None
    best_ring:  np.ndarray      = np.zeros((0, 3), dtype=float)

    if faces is not None:
        origin = base_origin.copy()
        waist_center_3d = origin.copy()

        # Build wrist exclusion zones (same 2-tier logic as chest)
        _excl_xz: Optional[List[np.ndarray]] = None
        _wrist_pts = [w for w in (left_wrist_joint, right_wrist_joint) if w is not None]
        if _wrist_pts:
            _excl_xz = [np.array([float(np.asarray(w, dtype=float)[0]),
                                   float(np.asarray(w, dtype=float)[2])]) for w in _wrist_pts]

        ring_3d = _trimesh_chest_contour(
            vertices, faces, origin, plane_normal, waist_center_3d,
            exclusion_xz=_excl_xz,
        )
        if ring_3d is not None and len(ring_3d) >= 3:
            girth_m = _waist_girth_from_ring(ring_3d)
            if 0.45 <= girth_m <= 1.60:
                best_girth = girth_m
                best_ring  = ring_3d

    if best_girth is not None:
        return best_girth, best_ring

    # ── FALLBACK: vertex slab → XZ gate → largest cluster → hull ────────────
    # Used when trimesh is unavailable or all scan levels fail.
    fallback_gate = 0.25
    center_xz = np.array([front_pt[0], front_pt[2]])

    t = thickness
    pts: Optional[np.ndarray] = None
    for _ in range(4):
        pts = _slab_at_plane(vertices, base_origin, plane_normal, t)
        if pts is not None and len(pts) >= 12:
            break
        t *= 1.5

    if pts is None or len(pts) < 4:
        return 0.0, np.zeros((0, 3), dtype=float)

    # Apply the adaptive XZ gate (same logic as trimesh path) then wrist filter.
    dist_xz = np.linalg.norm(pts[:, [0, 2]] - center_xz, axis=1)
    pts = pts[dist_xz <= fallback_gate]

    if left_wrist_joint is not None or right_wrist_joint is not None:
        pts = _exclude_hands_at_waist(
            pts_3d=pts,
            waist_midpoint_3d=base_origin,
            left_wrist_joint=left_wrist_joint,
            right_wrist_joint=right_wrist_joint,
            hand_exclusion_radius_m=0.08,
            cluster_min_gap_m=0.10,
        )

    if len(pts) < 4:
        return 0.0, np.zeros((0, 3), dtype=float)

    pts_2d  = pts[:, [0, 2]]
    hull_2d = _convex_hull(pts_2d)
    if hull_2d is not None and len(hull_2d) >= 3:
        shifted = np.roll(hull_2d, -1, axis=0)
        girth_m = float(np.linalg.norm(hull_2d - shifted, axis=1).sum())
        if 0.45 <= girth_m <= 1.60:
            # Pin Y to exact cut-plane height (not avg_y) for a flat ring.
            hull_3d = np.zeros((len(hull_2d), 3), dtype=float)
            hull_3d[:, 0] = hull_2d[:, 0]
            hull_3d[:, 1] = float(base_origin[1])
            hull_3d[:, 2] = hull_2d[:, 1]
            return girth_m, hull_3d

    # Last-resort: Ramanujan ellipse from bounding box
    w = float(pts_2d[:, 0].max() - pts_2d[:, 0].min())
    d = float(pts_2d[:, 1].max() - pts_2d[:, 1].min())
    a, b = max(w, d) / 2, min(w, d) / 2
    h = ((a - b) / (a + b)) ** 2
    girth_m = math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))
    return girth_m, np.zeros((0, 3), dtype=float)


def _find_pubic_bone_idx(
    vertices: np.ndarray,
    hip_level_y: float,
    actual_height_m: float,
) -> int:
    """
    Locate the MHR vertex closest to the pubic symphysis at runtime.

    Search space (T/A-pose body):
    - Y: [hip_level_y - 10 % of height, hip_level_y + 3 % of height]
    - |X| < 0.07 m  (near centre-line, between the legs)
    - Z: maximum (most frontal — max Z == front in MHR coordinate space)

    Returns a valid vertex index; falls back to vertex 0 only if the mesh is
    degenerate.
    """
    h = actual_height_m
    y_lo = hip_level_y - 0.10 * h
    y_hi = hip_level_y + 0.03 * h
    mask = (
        (vertices[:, 1] >= y_lo)
        & (vertices[:, 1] <= y_hi)
        & (np.abs(vertices[:, 0]) < 0.07)
    )
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        # Widen the band
        mask2 = np.abs(vertices[:, 1] - hip_level_y) < 0.06 * h
        idxs = np.where(mask2)[0]
    if len(idxs) == 0:
        return 0
    return int(idxs[int(np.argmax(vertices[idxs, 2]))])


def _hip_circumference_smplx_style(
    vertices: np.ndarray,
    faces: Optional[np.ndarray],
    pubic_bone_idx: int,
    pelvis_joint: Optional[np.ndarray] = None,
    spine3_joint: Optional[np.ndarray] = None,
    left_wrist_joint: Optional[np.ndarray] = None,
    right_wrist_joint: Optional[np.ndarray] = None,
    max_radius_m: float = 0.30,
) -> Tuple[float, np.ndarray]:
    """
    SMPLX-style hip / seat girth measurement.

    Plane origin  = pubic bone vertex (single landmark on body centreline).
                    No midpoint averaging — the pubic bone is already at X≈0,
                    Z≈0 of the pelvis, and its Y anchors the cut at the correct
                    anatomical hip fullness level (below the hip joint).

    Plane normal  = pelvis → spine3 (same joint pair as waist, same direction).
                    The CUTTING HEIGHT differs from waist only because the plane
                    ORIGIN is lower (pubic bone < belly-button midpoint), not
                    because the normal changes.

    No extremity exclusion — at pubic-bone Y, hand cross-sections sit above
    this level (hands hang at ~waist height in A-pose) and thigh tops form
    separate contours below the ring; _trimesh_chest_contour's nearest-centroid
    selection picks the correct hip ring without any explicit arm/leg gate.

    Fallback: vertex slab + convex hull limited to max_radius_m from pubic bone.

    Returns (girth_metres, ring_3d).
    """
    vertices = np.asarray(vertices, dtype=float)
    plane_origin = vertices[pubic_bone_idx].copy()

    # Plane normal: always world-Y so the ring is a perfectly horizontal cut.
    # Using the spine axis here made the ring tilt with the body's lean angle.
    plane_normal = np.array([0.0, 1.0, 0.0])

    # Build wrist exclusion zones (same 2-tier logic as chest)
    _excl_xz: Optional[List[np.ndarray]] = None
    _wrist_pts = [w for w in (left_wrist_joint, right_wrist_joint) if w is not None]
    if _wrist_pts:
        _excl_xz = [np.array([float(np.asarray(w, dtype=float)[0]),
                               float(np.asarray(w, dtype=float)[2])]) for w in _wrist_pts]

    # ── TRIMESH PATH (preferred) — reuses chest contour function ─────────────
    if faces is not None:
        ring_3d = _trimesh_chest_contour(
            vertices, faces, plane_origin, plane_normal, plane_origin,
            exclusion_xz=_excl_xz,
        )
        if ring_3d is not None and len(ring_3d) >= 3:
            # Clamp Y to exactly the cut-plane height — eliminates visual tilt
            # from sub-mm floating-point drift in trimesh edge interpolation.
            ring_3d[:, 1] = float(plane_origin[1])
            diffs = np.diff(ring_3d, axis=0)
            girth_m = float(np.linalg.norm(diffs, axis=1).sum())
            girth_m += float(np.linalg.norm(ring_3d[-1] - ring_3d[0]))
            return girth_m, ring_3d

    # ── FALLBACK: vertex slab + convex hull ───────────────────────────────────
    dominant = int(np.argmax(np.abs(plane_normal)))
    keep = [i for i in range(3) if i != dominant]

    thickness = 0.025
    pts: Optional[np.ndarray] = None
    for _ in range(4):
        pts = _slab_at_plane(vertices, plane_origin, plane_normal, thickness)
        if pts is not None and len(pts) >= 12:
            break
        thickness *= 1.5

    if pts is not None and len(pts) >= 4:
        # XZ gate: keep only hip/pelvis region within max_radius_m of pubic bone.
        # 0.30 m is wider than waist's 0.28 m — hip cross-section is broader.
        hip_xz = np.array([plane_origin[0], plane_origin[2]])
        pts = pts[np.linalg.norm(pts[:, [0, 2]] - hip_xz, axis=1) <= max_radius_m]

    if pts is not None and len(pts) >= 4:
        pts_2d = pts[:, keep]
        hull_2d = _convex_hull(pts_2d)
        if hull_2d is not None and len(hull_2d) >= 3:
            shifted = np.roll(hull_2d, -1, axis=0)
            girth_m = float(np.linalg.norm(hull_2d - shifted, axis=1).sum())
            hull_3d = np.zeros((len(hull_2d), 3), dtype=float)
            for i, ax in enumerate(keep):
                hull_3d[:, ax] = hull_2d[:, i]
            # Pin Y to exact cut-plane height for a flat horizontal ring.
            hull_3d[:, dominant] = float(plane_origin[1])
            return girth_m, hull_3d

    # Last-resort: Ramanujan ellipse from bounding box
    if pts is not None and len(pts) >= 2:
        pts_2d = pts[:, keep]
        w = float(pts_2d[:, 0].max() - pts_2d[:, 0].min())
        d = float(pts_2d[:, 1].max() - pts_2d[:, 1].min())
    else:
        w, d = 0.42, 0.30
    a, b = max(w, d) / 2, min(w, d) / 2
    hv = ((a - b) / (a + b)) ** 2
    girth_m = math.pi * (a + b) * (1 + 3 * hv / (10 + math.sqrt(4 - 3 * hv)))
    return girth_m, np.zeros((0, 3), dtype=float)


def _ellipse_perimeter(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    # Ramanujan approximation
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))


def _section_points(vertices: np.ndarray, target_y: float, thickness: float) -> np.ndarray:
    mask = np.abs(vertices[:, 1] - target_y) <= thickness
    return vertices[mask]


def _section_circumference(vertices: np.ndarray, target_y: float, base_thickness: float = 0.015) -> Optional[float]:
    """Approximate circumference of a horizontal body slice."""
    thickness = base_thickness
    for _ in range(3):
        slice_vertices = _section_points(vertices, target_y, thickness)
        if slice_vertices.shape[0] >= 12:
            # Use XZ projection
            girth = _perimeter_from_points(slice_vertices[:, [0, 2]])
            if girth:
                return girth
        thickness *= 1.5

    # Fallback to ellipse approximation with bounding box
    slice_vertices = _section_points(vertices, target_y, thickness)
    if slice_vertices.shape[0] < 4:
        return None
    width = float(slice_vertices[:, 0].max() - slice_vertices[:, 0].min())
    depth = float(slice_vertices[:, 2].max() - slice_vertices[:, 2].min())
    a = max(width, depth) / 2
    b = min(width, depth) / 2
    if a <= 0 or b <= 0:
        return None
    return _ellipse_perimeter(a, b)


def _scan_min_circumference(vertices: np.ndarray, start_y: float, end_y: float, steps: int = 25) -> Optional[Tuple[float, float]]:
    if steps <= 0:
        return None
    y0, y1 = (start_y, end_y) if start_y <= end_y else (end_y, start_y)
    ys = np.linspace(y0, y1, steps)
    best: Optional[Tuple[float, float]] = None
    for y in ys:
        girth = _section_circumference(vertices, float(y))
        if girth is None:
            continue
        if best is None or girth < best[1]:
            best = (float(y), float(girth))
    return best


def _limb_girth(
    vertices: np.ndarray,
    center_point: Optional[np.ndarray],
    target_y: float,
    radius: float,
    thickness: float = 0.02,
) -> Optional[float]:
    if center_point is None:
        return None
    slice_vertices = _section_points(vertices, target_y, thickness)
    if slice_vertices.shape[0] < 6:
        return None
    horizontal = slice_vertices[:, [0, 2]]
    center = center_point[[0, 2]]
    radial_mask = np.linalg.norm(horizontal - center, axis=1) <= radius
    selected = slice_vertices[radial_mask][:, [0, 2]]
    if selected.shape[0] < 4:
        return None
    return _perimeter_from_points(selected)


def _build_plane_axes(plane_normal: np.ndarray):
    """Build two orthogonal unit vectors (u, v) on the cutting plane via Gram-Schmidt."""
    n = np.asarray(plane_normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-9)
    ref = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(n, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(n, ref)
    u = u / (np.linalg.norm(u) + 1e-9)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-9)
    return u, v


def _isolate_single_thigh(
    pts_3d: np.ndarray,
    thigh_vertex_3d: np.ndarray,
    hip_joint_3d: np.ndarray,
    plane_u: np.ndarray,
    plane_v: np.ndarray,
    isolation_radius_m: float = 0.16,
) -> np.ndarray:
    """Keep only points from ONE thigh using a radial gate in the plane's UV space."""
    if len(pts_3d) == 0:
        return pts_3d

    def to_uv(pt):
        return np.array([np.dot(pt, plane_u), np.dot(pt, plane_v)])

    centre_uv = (to_uv(thigh_vertex_3d) + to_uv(hip_joint_3d)) / 2.0
    pts_uv = np.column_stack([pts_3d @ plane_u, pts_3d @ plane_v])
    dist_2d = np.linalg.norm(pts_uv - centre_uv, axis=1)
    return pts_3d[dist_2d <= isolation_radius_m]


def _plane_hull_perimeter(pts_3d: np.ndarray, plane_u: np.ndarray, plane_v: np.ndarray) -> float:
    """Project 3D cross-section points onto plane (u,v) axes and compute convex hull perimeter."""
    if len(pts_3d) < 3:
        return 0.0
    pts_2d = np.column_stack([pts_3d @ plane_u, pts_3d @ plane_v])
    if len(pts_2d) >= 4:
        hull = _convex_hull(pts_2d)
        if hull is None or len(hull) < 3:
            return 0.0
        shifted = np.roll(hull, -1, axis=0)
        return float(np.linalg.norm(hull - shifted, axis=1).sum())
    # Fallback: Ramanujan ellipse from bounding box
    w = float(pts_2d[:, 0].max() - pts_2d[:, 0].min()) if len(pts_2d) > 1 else 0.20
    d = float(pts_2d[:, 1].max() - pts_2d[:, 1].min()) if len(pts_2d) > 1 else 0.16
    a, b = max(w, d) / 2, min(w, d) / 2
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))


def _trimesh_thigh_contour(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    thigh_center: np.ndarray,
    plane_u: np.ndarray,
    plane_v: np.ndarray,
    isolation_radius_m: float = 0.16,
) -> Optional[np.ndarray]:
    """
    Slice the mesh and return an ordered ring for ONE thigh only.

    Two-stage isolation
    -------------------
    Stage 1 — Generous UV filter (2× isolation_radius_m, min 0.25 m):
        Removes far body parts (torso, other leg if far away, arms).
        Deliberately wide so even when landmark vertices are biased to the
        inner thigh, the full outer arc of the thigh is still included.

    Stage 2 — Nearest-centroid selection:
        Stitch surviving segments into contours; pick the one whose 3-D
        centroid is closest to thigh_center.  If two legs survived stage 1
        (close-stance body), this correctly picks the target thigh whose
        ring centroid is always nearer to the known landmark vertex.

    Why two stages:
        A tight UV filter from an inner-vertex cluster clips the outer half
        of the thigh (the symptom: incomplete half-ring).  A loose filter
        followed by centroid selection gives the full ring while still
        rejecting the opposite leg.
    """
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        segments = trimesh.intersections.mesh_plane(mesh, plane_normal, plane_origin)
        if segments is None or len(segments) < 3:
            return None

        # Stage 1: generous UV gate — captures full thigh ring
        generous_r = max(isolation_radius_m * 2.0, 0.25)
        thigh_uv = np.array([float(np.dot(thigh_center, plane_u)),
                              float(np.dot(thigh_center, plane_v))])
        keep = []
        for seg in segments:
            mid_3d = (seg[0] + seg[1]) / 2.0
            mid_uv = np.array([float(np.dot(mid_3d, plane_u)),
                                float(np.dot(mid_3d, plane_v))])
            if float(np.linalg.norm(mid_uv - thigh_uv)) <= generous_r:
                keep.append(seg)

        if not keep:
            return None

        # Stage 2: stitch + nearest-centroid picks the correct thigh
        contours = _build_contours_from_segments(np.array(keep, dtype=float))
        if not contours:
            return None

        best: Optional[np.ndarray] = None
        best_dist = float('inf')
        for c in contours:
            if len(c) < 3:
                continue
            d = float(np.linalg.norm(c.mean(axis=0) - thigh_center))
            if d < best_dist:
                best_dist = d
                best = c

        return best
    except Exception:
        return None


def _thigh_circumference_smplx_style(
    vertices: np.ndarray,
    faces: Optional[np.ndarray],
    landmark_indices: List[int],
    hip_joint: Optional[np.ndarray],
    knee_joint: Optional[np.ndarray],
    spine3_joint: Optional[np.ndarray] = None,
    thickness: float = 0.020,
    isolation_margin_m: float = 0.03,
    min_isolation_radius_m: float = 0.12,
    plane_anchor_idx: Optional[int] = None,
):
    """
    Compute thigh circumference (raw metres) for ONE side, SMPLX-style.

    landmark_indices : list of vertex indices spread around the thigh
                       cross-section (e.g. [16438, 16442, 16443, 16444]).
                       Their centroid becomes the thigh centre; their spread
                       determines the adaptive isolation radius.
                       A single index is also valid (falls back to
                       min_isolation_radius_m).

    isolation_margin_m      : margin added beyond the landmark spread (metres).
    min_isolation_radius_m  : floor radius used when only one landmark given.

    Returns (girth_m, ring_3d) — same contract as chest/waist/hip.
    """
    vertices   = np.asarray(vertices,   dtype=float)
    hip_joint  = np.asarray(hip_joint,  dtype=float) if hip_joint  is not None else None
    knee_joint = np.asarray(knee_joint, dtype=float) if knee_joint is not None else None

    # Compute thigh centre (centroid of all landmarks) and adaptive radius
    landmark_pts = vertices[np.asarray(landmark_indices, dtype=int)]
    thigh_center = landmark_pts.mean(axis=0)
    dists = np.linalg.norm(landmark_pts - thigh_center, axis=1)
    isolation_radius_m = float(max(dists.max() + isolation_margin_m,
                                   min_isolation_radius_m))

    # A: Plane origin = thigh landmark XZ but lowered to upper-mid thigh Y.
    # Landmark vertices 16438-16444 sit at the groin/upper-thigh junction.
    # A horizontal cut at that height spans BOTH legs as one connected loop.
    # Dropping 30 % toward the knee puts the plane at the proper upper-mid
    # thigh where the two legs are clearly separated → clean single-thigh ring.
    # A: Plane origin — use hardcoded anchor vertex when available (exact, stable),
    # otherwise fall back to 4 % offset from landmark centroid toward knee.
    if plane_anchor_idx is not None and plane_anchor_idx < vertices.shape[0]:
        plane_origin = vertices[plane_anchor_idx].copy()
    else:
        plane_origin = thigh_center.copy()
        if knee_joint is not None:
            k = np.asarray(knee_joint, dtype=float)
            plane_origin[1] = float(thigh_center[1] * 0.96 + k[1] * 0.04)
        else:
            plane_origin[1] = float(thigh_center[1] - 0.10)

    # B: Plane normal = world-Y (horizontal cut, like a tape measure).
    # spine3→knee or hip→knee both create laterally-tilted planes whose UV axes
    # are not axis-aligned; when the landmark cluster sits on the inner thigh,
    # the outer-thigh arc falls outside the generous_r filter → half-ring.
    # A horizontal cut keeps plane_u=[1,0,0], plane_v=[0,0,1] so UV distances
    # equal physical XZ distances and the symmetric generous_r always captures
    # the full thigh cross-section regardless of landmark bias direction.
    plane_normal = np.array([0.0, 1.0, 0.0])

    # C: Build plane UV axes (Gram-Schmidt — needed for tilted leg-axis plane)
    plane_u, plane_v = _build_plane_axes(plane_normal)

    # D: TRIMESH PATH — UV-filtered ordered ring (one thigh only)
    #    _trimesh_thigh_contour keeps only segments within isolation_radius_m
    #    of the thigh CENTRE in UV space — adaptive to actual thigh width.
    if faces is not None:
        ring_3d = _trimesh_thigh_contour(
            vertices, faces, plane_origin, plane_normal,
            thigh_center, plane_u, plane_v, isolation_radius_m,
        )
        if ring_3d is not None and len(ring_3d) >= 3:
            # Sort points by angle around the XZ centroid so the ring renders
            # as a smooth closed loop without diagonal crossover segments.
            # (plane is world-Y so XZ is the cut plane — angle sort is exact.)
            cx = float(ring_3d[:, 0].mean())
            cz = float(ring_3d[:, 2].mean())
            angles = np.arctan2(ring_3d[:, 2] - cz, ring_3d[:, 0] - cx)
            ring_3d = ring_3d[np.argsort(angles)]
            diffs = np.diff(ring_3d, axis=0)
            girth_m = float(np.linalg.norm(diffs, axis=1).sum())
            girth_m += float(np.linalg.norm(ring_3d[-1] - ring_3d[0]))
            return girth_m, ring_3d

    # E: FALLBACK — slab → UV gate from thigh centre → convex hull → 3-D ring
    pts: Optional[np.ndarray] = None
    t = thickness
    for _ in range(3):
        pts = _slab_at_plane(vertices, plane_origin, plane_normal, t)
        if pts is not None and len(pts) >= 12:
            break
        t *= 1.5

    if pts is None or len(pts) == 0:
        return 0.0, np.zeros((0, 3), dtype=float)

    # UV-space gate centred on thigh_center — no hip_joint dependency
    center_uv = np.array([float(np.dot(thigh_center, plane_u)),
                           float(np.dot(thigh_center, plane_v))])
    pts_uv = np.column_stack([pts @ plane_u, pts @ plane_v])
    pts = pts[np.linalg.norm(pts_uv - center_uv, axis=1) <= isolation_radius_m]

    if len(pts) < 3:
        return 0.0, np.zeros((0, 3), dtype=float)

    girth_m = _plane_hull_perimeter(pts, plane_u, plane_v)

    # Reconstruct ordered 3-D ring from 2-D convex hull for visualisation
    pts_2d = np.column_stack([pts @ plane_u, pts @ plane_v])
    hull_2d = _convex_hull(pts_2d)
    if hull_2d is not None and len(hull_2d) >= 3:
        plane_n_val = float(np.dot(plane_origin, plane_normal))
        ring_3d = (
            hull_2d[:, 0:1] * plane_u[np.newaxis, :]
            + hull_2d[:, 1:2] * plane_v[np.newaxis, :]
            + plane_n_val * plane_normal[np.newaxis, :]
        )
        return girth_m, ring_3d

    return girth_m, np.zeros((0, 3), dtype=float)


def _thigh_girth_both_sides(
    vertices: np.ndarray,
    faces: Optional[np.ndarray],
    left_landmark_indices: List[int],
    right_landmark_indices: List[int],
    left_hip_joint: Optional[np.ndarray],
    right_hip_joint: Optional[np.ndarray],
    left_knee_joint: Optional[np.ndarray],
    right_knee_joint: Optional[np.ndarray],
    spine3_joint: Optional[np.ndarray] = None,
    thickness: float = 0.020,
    left_plane_anchor_idx: Optional[int] = None,
    right_plane_anchor_idx: Optional[int] = None,
):
    """Compute left thigh girth and return (girth_m, left_ring_3d).
    Right side is measured for averaging but its ring is not returned."""
    left_girth, left_ring = _thigh_circumference_smplx_style(
        vertices, faces, left_landmark_indices,
        left_hip_joint, left_knee_joint, spine3_joint, thickness,
        plane_anchor_idx=left_plane_anchor_idx,
    )
    right_girth, _ = _thigh_circumference_smplx_style(
        vertices, faces, right_landmark_indices,
        right_hip_joint, right_knee_joint, spine3_joint, thickness,
        plane_anchor_idx=right_plane_anchor_idx,
    )
    valid = [g for g in [left_girth, right_girth] if g and g > 0.0]
    avg_girth = float(np.mean(valid)) if valid else 0.0
    return avg_girth, left_ring


def _waist_front_back(vertices: np.ndarray, waist_y: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    waist_slice = _section_points(vertices, waist_y, 0.02)
    if waist_slice.shape[0] == 0:
        return None, None
    front_idx = int(np.argmax(waist_slice[:, 2]))
    back_idx = int(np.argmin(waist_slice[:, 2]))
    return waist_slice[front_idx], waist_slice[back_idx]


def _axilla_points(vertices: np.ndarray, level_y: float, center_x: float, side: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    candidates = _section_points(vertices, level_y, 0.025)
    if candidates.shape[0] == 0:
        return None, None
    if side == "left":
        side_mask = candidates[:, 0] >= center_x
    else:
        side_mask = candidates[:, 0] <= center_x
    side_points = candidates[side_mask]
    if side_points.shape[0] == 0:
        return None, None
    front = side_points[int(np.argmax(side_points[:, 2]))]
    back = side_points[int(np.argmin(side_points[:, 2]))]
    return front, back


def _crotch_point(vertices: np.ndarray, hip_center: Optional[np.ndarray], knee_level_y: Optional[float], ground_y: float) -> Optional[np.ndarray]:
    if hip_center is None:
        return None
    y_min = ground_y + 0.05
    y_max = hip_center[1]
    lower = knee_level_y if knee_level_y is not None else y_min
    mask = (
        (vertices[:, 1] < y_max)
        & (vertices[:, 1] > lower - 0.05)
        & (np.abs(vertices[:, 0] - hip_center[0]) < 0.12)
        & (np.abs(vertices[:, 2] - hip_center[2]) < 0.2)
    )
    candidates = vertices[mask]
    if candidates.shape[0] == 0:
        return None
    return candidates[int(np.argmin(candidates[:, 1]))]


def _find_back_waist_idx(
    vertices: np.ndarray,
    belly_button_idx: int,
    y_tol: float = 0.015,
    x_tol: float = 0.06,
) -> int:
    """
    Find the most-posterior (back surface) vertex at the same Y level as the
    belly_button vertex.  Used instead of a hardcoded back-belly vertex index.

    WHY RUNTIME DISCOVERY:
    Hardcoding a back-belly vertex index is fragile — the correct index
    depends on the exact mesh topology and the index we thought was correct
    (5750) is clearly at the wrong height (mid-back, not back-waist).
    Runtime search is self-correcting: it will always find the correct back
    surface at whatever Y the belly_button vertex happens to sit at.

    Algorithm
    ---------
    1. Take belly_button Y as the reference height.
    2. Collect all vertices within ±y_tol (1.5 cm) of that Y.
    3. Filter to near the body centreline: |X| ≤ x_tol (6 cm).
    4. Determine which Z direction is "back":
       - If belly_button Z > 0  → front is +Z → back is most negative Z
       - If belly_button Z ≤ 0  → front is −Z → back is most positive Z
    5. Return the vertex index at the extreme back Z.

    Falls back to the belly_button index itself only if the mesh has no
    vertices matching the search (should never happen in practice).
    """
    belly_pt = vertices[belly_button_idx].astype(float)
    belly_y  = float(belly_pt[1])
    belly_z  = float(belly_pt[2])

    y_mask = np.abs(vertices[:, 1] - belly_y) <= y_tol
    x_mask = np.abs(vertices[:, 0]) <= x_tol
    idxs   = np.where(y_mask & x_mask)[0]

    if len(idxs) == 0:
        # Widen the Y search only — keep X gate tight
        idxs = np.where(
            (np.abs(vertices[:, 1] - belly_y) <= y_tol * 3) &
            (np.abs(vertices[:, 0]) <= x_tol)
        )[0]
    if len(idxs) == 0:
        return int(belly_button_idx)

    # Back of body = opposite Z sign from the belly_button vertex
    if belly_z > 0:
        # belly is in +Z (front) → back is most negative Z
        return int(idxs[np.argmin(vertices[idxs, 2])])
    else:
        # belly is in −Z (front) → back is most positive Z
        return int(idxs[np.argmax(vertices[idxs, 2])])


def _landmark_to_list(point: Optional[np.ndarray]) -> Optional[List[float]]:
    if point is None:
        return None
    return point.astype(float).round(6).tolist()


# ---------------------------------------------------------------------------
# Inseam / Inside Leg Height helpers
# ---------------------------------------------------------------------------

def find_heel_vertex(vertices: np.ndarray, ankle_joint: np.ndarray, side: str = "left") -> int:
    """Find and print the heel vertex index (run once, then hardcode result)."""
    ankle = np.asarray(ankle_joint, dtype=float)
    xz_dist = np.linalg.norm(vertices[:, [0, 2]] - ankle[[0, 2]], axis=1)
    candidates = np.where(xz_dist <= 0.10)[0]
    if len(candidates) < 5:
        candidates = np.where(xz_dist <= 0.15)[0]
    foot_mask = vertices[candidates, 1] <= ankle[1] + 0.01
    if foot_mask.sum() > 0:
        candidates = candidates[foot_mask]
    min_y = vertices[candidates, 1].min()
    at_floor = candidates[np.abs(vertices[candidates, 1] - min_y) < 0.005]
    heel_idx = int(at_floor[np.argmax(vertices[at_floor, 2])])
    pos = vertices[heel_idx]
    print(f"[find_heel] {side} heel idx={heel_idx}  pos=x={pos[0]:.4f} y={pos[1]:.4f} z={pos[2]:.4f}")
    print(f"[find_heel] Hardcode: {side.upper()}_HEEL_IDX = {heel_idx}")
    return heel_idx


def _inseam_projected(crotch_pt: np.ndarray, heel_pt: np.ndarray,
                      knee_joint: Optional[np.ndarray] = None) -> float:
    """Project crotch→heel onto leg axis direction (Method A — best)."""
    crotch = np.asarray(crotch_pt, dtype=float)
    heel   = np.asarray(heel_pt,   dtype=float)
    vec    = crotch - heel
    if knee_joint is not None:
        knee = np.asarray(knee_joint, dtype=float)
        axis = (crotch - knee) + (knee - heel)
    else:
        axis = vec.copy()
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        return float(np.linalg.norm(vec))
    return float(max(np.dot(vec, axis / norm), 0.0))


def _inseam_euclidean(crotch_pt: np.ndarray, heel_pt: np.ndarray) -> float:
    """Straight-line chord distance crotch→heel (Method B — matches SMPLX)."""
    return float(np.linalg.norm(
        np.asarray(crotch_pt, dtype=float) - np.asarray(heel_pt, dtype=float)
    ))


def _inseam_vertical(crotch_pt: np.ndarray, heel_pt: np.ndarray) -> float:
    """Vertical Y-drop crotch→heel (Method C — weakest, upright pose only)."""
    return float(abs(
        np.asarray(crotch_pt, dtype=float)[1] - np.asarray(heel_pt, dtype=float)[1]
    ))


# ---------------------------------------------------------------------------
# Arm / Sleeve Length helpers
# ---------------------------------------------------------------------------

def _arm_length_vertex_waypoint(
    vertices: np.ndarray,
    waypoint_indices: List[int],
) -> float:
    """
    Arm length = sum of |v[i] → v[i+1]| for each consecutive pair in
    waypoint_indices (ordered shoulder → ... → wrist).

    Using multiple surface waypoints makes the path follow the arm surface
    instead of cutting through the mesh interior.
    """
    total = 0.0
    for i in range(len(waypoint_indices) - 1):
        a = vertices[waypoint_indices[i]].astype(float)
        b = vertices[waypoint_indices[i + 1]].astype(float)
        total += float(np.linalg.norm(a - b))
    return total


# ---------------------------------------------------------------------------
# Front body length — geodesic surface path (neck top → crotch)
# ---------------------------------------------------------------------------

def _build_mesh_adjacency(vertices: np.ndarray, faces: np.ndarray):
    """
    Vectorised scipy CSR graph where edge weight = Euclidean distance between
    adjacent vertices.  Used as input to Dijkstra for geodesic path finding.

    IMPORTANT — edge deduplication:
    In a manifold mesh every interior edge is shared by exactly two faces.
    Without deduplication, csr_matrix SUMS duplicate (row,col) entries, giving
    interior edges weight 2×distance and making every Dijkstra distance 2× too
    large.  We normalise each undirected edge to (min_idx, max_idx) and call
    np.unique to keep one copy per physical edge before building the graph.
    """
    from scipy.sparse import csr_matrix

    N = int(vertices.shape[0])
    # Collect all half-edges from every face triangle
    a0, a1, a2 = faces[:, 0], faces[:, 1], faces[:, 2]
    pairs = np.concatenate([
        np.stack([a0, a1], axis=1),
        np.stack([a1, a2], axis=1),
        np.stack([a2, a0], axis=1),
    ])  # shape (3F, 2)

    # Normalise to undirected edge (min, max) and deduplicate
    edges = np.sort(pairs, axis=1)          # each row is (min_idx, max_idx)
    edges = np.unique(edges, axis=0)        # one entry per physical edge

    dists = np.linalg.norm(
        vertices[edges[:, 0]].astype(float) - vertices[edges[:, 1]].astype(float),
        axis=1,
    )
    # Add both directions for the undirected graph
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([dists, dists])
    return csr_matrix((data, (rows, cols)), shape=(N, N))


def _geodesic_surface_path(
    vertices: np.ndarray,
    faces: np.ndarray,
    start: int,
    end: int,
) -> Tuple[float, List[int]]:
    """
    Dijkstra shortest path along mesh surface triangles.

    Returns
    -------
    (distance_metres, ordered_vertex_index_list)

    Raises ImportError if scipy is unavailable.
    """
    from scipy.sparse.csgraph import dijkstra

    graph = _build_mesh_adjacency(vertices, faces)
    dist, pred = dijkstra(
        csgraph=graph,
        directed=False,
        indices=start,
        return_predecessors=True,
    )
    path: List[int] = []
    cur = end
    while cur != start and cur >= 0:
        path.append(int(cur))
        cur = int(pred[cur])
    path.append(int(start))
    path.reverse()
    return float(dist[end]), path


def compute_measurements(person_rig: Dict, target_height_cm: Optional[float] = None) -> Dict[str, object]:
    """Compute anthropometric measurements for a reconstructed person."""
    mesh = person_rig.get("mesh") or {}
    skeleton = person_rig.get("skeleton") or {}

    vertices = np.asarray(mesh.get("vertices"), dtype=np.float32)
    joint_positions = np.asarray(skeleton.get("joint_positions"), dtype=np.float32)
    joint_names = skeleton.get("joint_names") or []
    keypoints = person_rig.get("keypoints") or []

    if vertices.size == 0:
        raise MeasurementError("Mesh vertices are missing")

    # Extract mesh faces for trimesh-based cross-section (preferred over slab)
    _faces_raw = mesh.get("faces")
    _faces_arr: Optional[np.ndarray] = None
    if _faces_raw is not None:
        try:
            _faces_arr = np.asarray(_faces_raw, dtype=np.int32)
        except Exception:
            pass

    # Arm length — three surface vertex anchors (shoulder → elbow → wrist).
    # SMPLX-equivalent: all endpoints are on the mesh surface (no joints).
    # shoulder=7953 (acromion tip), elbow=13495 (elbow surface), wrist=13947 (ulnar styloid).
    # Two-segment sum handles any arm pose, including slight A-pose bend.
    _ARM_WAYPOINTS: List[int] = [7953, 13495, 13947]
    _arm_valid = all(i < vertices.shape[0] for i in _ARM_WAYPOINTS)

    # Chest landmark vertex indices (known for the MHR mesh)
    _L_CHEST, _R_CHEST = 7380, 6156
    _chest_valid = int(_L_CHEST) < vertices.shape[0] and int(_R_CHEST) < vertices.shape[0]

    # Neck top / shoulder top vertex (front neck–shoulder junction, MHR mesh).
    # Used as the START point for front body length (geodesic to crotch).
    _NECK_TOP_VERTEX_IDX: int = 1640

    # Crotch vertex (hardcoded MHR index — identified via mesh explorer)
    _CROTCH_VERTEX_IDX: int = 5726
    # Heel vertices — hardcoded from find_heel_vertex() debug run.
    _L_HEEL_IDX: int = 12415
    _R_HEEL_IDX: int = 17865

    # Waist landmark vertex indices.
    # _BELLY_BUTTON      (5711) = navel front surface vertex.
    # _BACK_BELLY_BUTTON (5853) = corresponding back surface vertex — confirmed.
    # Both vertices are used to define the waist cut plane: the plane passes
    # through the Y midpoint of these two vertices with normal [0,1,0].
    _BELLY_BUTTON: int      = 5711
    _BACK_BELLY_BUTTON: int = 5853
    _waist_valid = (
        int(_BELLY_BUTTON) < vertices.shape[0]
        and int(_BACK_BELLY_BUTTON) < vertices.shape[0]
    )

    # Thigh landmark vertex indices (known for the MHR mesh).
    # Multiple indices per side → centroid = true thigh centre,
    # spread → adaptive isolation radius (self-sizing, not fixed 0.16 m).
    _L_THIGH_IDXS: List[int] = [16438, 16442, 16443, 16444]
    _R_THIGH_IDXS: List[int] = [11279]
    # Hardcoded ring-anchor vertices — the exact mesh vertex closest to the
    # desired cut height (identified via debug print, 4 % groin→knee offset).
    _L_THIGH_RING_IDX: int = 16467   # left thigh ring anchor
    _R_THIGH_RING_IDX: int = 11304   # right thigh ring anchor
    _thigh_valid = (
        all(i < vertices.shape[0] for i in _L_THIGH_IDXS)
        and all(i < vertices.shape[0] for i in _R_THIGH_IDXS)
    )

    ground_y = float(vertices[:, 1].min())
    head_idx = int(np.argmax(vertices[:, 1]))
    head_vertex = vertices[head_idx]
    actual_height_m = float(head_vertex[1] - ground_y)
    if actual_height_m <= 0:
        raise MeasurementError("Invalid reconstructed height")

    actual_height_cm = actual_height_m * 100.0
    if target_height_cm is None:
        target_height_cm = actual_height_cm

    target_height_cm = float(target_height_cm)
    if target_height_cm <= 0:
        raise MeasurementError("Target height must be positive")

    scale_factor = float(target_height_cm) / actual_height_cm

    joint_map = _vector_map(joint_names, joint_positions) if joint_positions.size else {}
    keypoint_map = _keypoint_map(keypoints)

    neck_point = _coalesce_points(keypoint_map.get("neck"), joint_map.get("neck"))
    left_shoulder = _coalesce_points(keypoint_map.get("left_shoulder"), joint_map.get("left_shoulder"))
    right_shoulder = _coalesce_points(keypoint_map.get("right_shoulder"), joint_map.get("right_shoulder"))
    left_acromion = _coalesce_points(keypoint_map.get("left_acromion"), left_shoulder)
    right_acromion = _coalesce_points(keypoint_map.get("right_acromion"), right_shoulder)
    left_hip = _coalesce_points(keypoint_map.get("left_hip"), joint_map.get("left_hip"), joint_map.get("left-hip"))
    right_hip = _coalesce_points(keypoint_map.get("right_hip"), joint_map.get("right_hip"), joint_map.get("right-hip"))
    left_knee = _coalesce_points(keypoint_map.get("left_knee"), joint_map.get("left_knee"), joint_map.get("left-knee"))
    right_knee = _coalesce_points(keypoint_map.get("right_knee"), joint_map.get("right_knee"), joint_map.get("right-knee"))
    left_ankle = _coalesce_points(keypoint_map.get("left_ankle"), joint_map.get("left_ankle"), joint_map.get("left-ankle"))
    right_ankle = _coalesce_points(keypoint_map.get("right_ankle"), joint_map.get("right_ankle"), joint_map.get("right-ankle"))
    left_elbow = _coalesce_points(keypoint_map.get("left_elbow"), joint_map.get("left_elbow"), joint_map.get("left-elbow"))
    right_elbow = _coalesce_points(keypoint_map.get("right_elbow"), joint_map.get("right_elbow"), joint_map.get("right-elbow"))
    left_wrist = _coalesce_points(keypoint_map.get("left_wrist"), joint_map.get("left_wrist"), joint_map.get("left-wrist"))
    right_wrist = _coalesce_points(keypoint_map.get("right_wrist"), joint_map.get("right_wrist"), joint_map.get("right-wrist"))
    eye_point = _average_points([
        keypoint_map.get("left_eye"),
        keypoint_map.get("right_eye"),
        keypoint_map.get("nose"),
    ])

    hip_center = _average_points([left_hip, right_hip])
    shoulder_center = _average_points([left_shoulder, right_shoulder])
    acromion_center = _average_points([left_acromion, right_acromion])
    torso_upper_y = float(acromion_center[1]) if acromion_center is not None else float(shoulder_center[1]) if shoulder_center is not None else float(head_vertex[1] * 0.85)
    hip_level_y = float(hip_center[1]) if hip_center is not None else float(ground_y + actual_height_m * 0.45)
    torso_span = max(torso_upper_y - hip_level_y, actual_height_m * 0.1)
    # Anchor bust_level_y to actual chest landmark vertices when available
    if _chest_valid:
        bust_level_y = float((vertices[_L_CHEST, 1] + vertices[_R_CHEST, 1]) / 2.0)
    else:
        bust_level_y = torso_upper_y - 0.22 * torso_span
    underbust_level_y = bust_level_y - 0.05 * torso_span

    knee_center = _average_points([left_knee, right_knee])
    if knee_center is not None:
        knee_level_y = float(knee_center[1])
    else:
        knee_level_y = ground_y + actual_height_m * 0.26

    ankle_center = _average_points([left_ankle, right_ankle])
    if ankle_center is not None:
        ankle_level_y = float(ankle_center[1])
    else:
        ankle_level_y = ground_y + actual_height_m * 0.05

    if _waist_valid:
        waist_level_y = float((vertices[_BELLY_BUTTON, 1] + vertices[_BACK_BELLY_BUTTON, 1]) / 2.0)
    else:
        _waist_scan = _scan_min_circumference(vertices, hip_level_y + 0.03, torso_upper_y - 0.05, 30)
        waist_level_y = float(_waist_scan[0]) if _waist_scan else float(hip_level_y + 0.15 * torso_span)

    calf_level_y = knee_level_y - 0.45 * (knee_level_y - ankle_level_y)

    # Crotch point — hardcoded vertex (stable, no scan needed)
    crotch_point = vertices[_CROTCH_VERTEX_IDX].copy() if _CROTCH_VERTEX_IDX < vertices.shape[0] else None

    # Heel points — Option A (hardcoded vertex), B (ankle projected), C (vertical)
    if _L_HEEL_IDX is not None and _L_HEEL_IDX < vertices.shape[0]:
        _left_heel_pt = vertices[_L_HEEL_IDX].copy()
        _heel_source  = "heel_vertex"
    elif left_ankle is not None:
        _left_heel_pt = np.array([left_ankle[0], ground_y, left_ankle[2]], dtype=float)
        _heel_source  = "ankle_projected"
    else:
        _left_heel_pt = np.array([crotch_point[0] if crotch_point is not None else 0.0,
                                   ground_y,
                                   crotch_point[2] if crotch_point is not None else 0.0], dtype=float)
        _heel_source  = "vertical_fallback"

    # Inseam — Method A (projected along leg axis): best for posed figures
    if crotch_point is not None:
        inside_leg = _inseam_projected(crotch_point, _left_heel_pt, left_knee)
    else:
        inside_leg = hip_level_y - ground_y

    torso_center_x = hip_center[0] if hip_center is not None else 0.0
    axilla_level_y = hip_level_y + 0.7 * (torso_upper_y - hip_level_y)
    left_axilla_front, left_axilla_back = _axilla_points(vertices, axilla_level_y, torso_center_x, "left")
    right_axilla_front, right_axilla_back = _axilla_points(vertices, axilla_level_y, torso_center_x, "right")

    waist_front, waist_back = _waist_front_back(vertices, waist_level_y)

    head_length = float(head_vertex[1] - neck_point[1]) if neck_point is not None else actual_height_m * 0.13
    head_girth_level = float(head_vertex[1] - 0.12 * head_length)
    neck_level_y = float(neck_point[1] - 0.01) if neck_point is not None else torso_upper_y - 0.05

    knee_girth_left = _limb_girth(vertices, left_knee, knee_level_y, radius=0.18, thickness=0.015)
    knee_girth_right = _limb_girth(vertices, right_knee, knee_level_y, radius=0.18, thickness=0.015)
    calf_center_left = _average_points([left_knee, left_ankle])
    calf_center_right = _average_points([right_knee, right_ankle])
    calf_girth_left = _limb_girth(vertices, calf_center_left, calf_level_y, radius=0.16)
    calf_girth_right = _limb_girth(vertices, calf_center_right, calf_level_y, radius=0.16)
    ankle_girth_left = _limb_girth(vertices, left_ankle, ankle_level_y, radius=0.12, thickness=0.012)
    ankle_girth_right = _limb_girth(vertices, right_ankle, ankle_level_y, radius=0.12, thickness=0.012)

    upper_arm_center_left = _average_points([left_shoulder, left_elbow])
    upper_arm_center_right = _average_points([right_shoulder, right_elbow])
    shoulder_ref = left_shoulder if left_shoulder is not None else right_shoulder if right_shoulder is not None else acromion_center
    elbow_ref = left_elbow if left_elbow is not None else right_elbow
    if shoulder_ref is not None and elbow_ref is not None:
        upper_arm_level_y = float(shoulder_ref[1] - 0.4 * (shoulder_ref[1] - elbow_ref[1]))
    else:
        upper_arm_level_y = torso_upper_y - 0.2 * torso_span
    upper_arm_girth_left = _limb_girth(vertices, upper_arm_center_left, upper_arm_level_y, radius=0.13)
    upper_arm_girth_right = _limb_girth(vertices, upper_arm_center_right, upper_arm_level_y, radius=0.13)

    wrist_center = _average_points([left_wrist, right_wrist])
    if wrist_center is not None:
        wrist_level_y = float(wrist_center[1])
    else:
        wrist_level_y = upper_arm_level_y - 0.25
    wrist_girth_left = _limb_girth(vertices, left_wrist, wrist_level_y, radius=0.08, thickness=0.01)
    wrist_girth_right = _limb_girth(vertices, right_wrist, wrist_level_y, radius=0.08, thickness=0.01)

    # Arm length — surface vertex waypoint method (shoulder→elbow→wrist vertices)
    # All 3 are mesh surface vertices: more accurate than joint-chain approach.
    if _arm_valid:
        arm_length = _arm_length_vertex_waypoint(vertices, _ARM_WAYPOINTS)
    else:
        arm_length = None

    # Front body length — front-surface-constrained geodesic, neck-top → crotch.
    #
    # WHY MULTI-SEGMENT WAS WRONG:
    #   The previous approach used unverified intermediate vertices (5696, 5717).
    #   If any waypoint is on the back or side surface, that geodesic SEGMENT routes
    #   around the body rather than straight down the front — each detour adds ~30-50 cm.
    #   4 bad segments × ~30 cm each = the ~2x inflation observed (150 cm vs ~70 cm).
    #
    # THE FIX — front-hemisphere face filter:
    #   1. Determine "front" Z direction from belly_button (5707) vs back_belly (5750).
    #      These are confirmed front/back vertices from the waist measurement code.
    #   2. Keep only mesh faces where ALL 3 vertices are on the front side of the mesh.
    #   3. Run a single Dijkstra on the filtered front-only mesh.
    #   Dijkstra on a front-only mesh physically cannot route around the back.
    #
    # THRESHOLD:
    #   30% of the front-back half-depth toward the front.  This is deliberately
    #   permissive — it excludes only the spine-side back faces, not the sides,
    #   so the path can still follow the lateral chest/hip curves correctly.
    #   If the filter makes start or end unreachable, falls back to unfiltered mesh.
    _sc_valid = (
        _NECK_TOP_VERTEX_IDX < vertices.shape[0]
        and _CROTCH_VERTEX_IDX < vertices.shape[0]
        and _BELLY_BUTTON < vertices.shape[0]
        and _BACK_BELLY_BUTTON < vertices.shape[0]
        and _faces_arr is not None
        and len(_faces_arr) > 0
    )
    _sc_path_indices: List[int] = []
    shoulder_to_crotch_raw: Optional[float] = None
    if _sc_valid:
        try:
            _verts_f = vertices.astype(float)

            # --- Determine front Z direction ---
            _belly_z  = float(_verts_f[_BELLY_BUTTON,      2])
            _backb_z  = float(_verts_f[_BACK_BELLY_BUTTON, 2])
            _front_sign = np.sign(_belly_z - _backb_z)   # +1 if front=+Z, -1 if front=−Z
            _center_z   = (_belly_z + _backb_z) / 2.0
            _half_d     = abs(_belly_z - _center_z)

            # --- Build front-face mask (permissive: 30% toward front) ---
            _z_thresh  = _center_z + _front_sign * _half_d * 0.3
            _face_zs   = _verts_f[_faces_arr, 2]   # (F, 3)
            if _front_sign >= 0:
                _front_mask = np.all(_face_zs >= _z_thresh, axis=1)
            else:
                _front_mask = np.all(_face_zs <= _z_thresh, axis=1)
            _front_faces = _faces_arr[_front_mask]

            # --- Verify start & end are reachable in filtered graph ---
            _front_vset = set(int(i) for i in _front_faces.flatten())
            _sc_faces = (
                _front_faces
                if (_NECK_TOP_VERTEX_IDX in _front_vset and _CROTCH_VERTEX_IDX in _front_vset)
                else _faces_arr   # last-resort fallback — unfiltered
            )

            # --- Single Dijkstra on front-only mesh ---
            _sc_dist, _sc_path_indices = _geodesic_surface_path(
                _verts_f, _sc_faces, _NECK_TOP_VERTEX_IDX, _CROTCH_VERTEX_IDX
            )

            # Sanity: surface distance must be >= straight-line chord
            _sc_chord = float(np.linalg.norm(
                _verts_f[_NECK_TOP_VERTEX_IDX] - _verts_f[_CROTCH_VERTEX_IDX]
            ))
            if _sc_dist >= _sc_chord * 0.98:
                shoulder_to_crotch_raw = _sc_dist
            else:
                shoulder_to_crotch_raw = None
                _sc_path_indices = []
        except Exception:
            shoulder_to_crotch_raw = None
            _sc_path_indices = []

    # shoulder_width / _shoulder_breadth_arc initialised in the combined ring block below

    back_width = None
    if left_axilla_back is not None and right_axilla_back is not None:
        back_width = float(np.linalg.norm(left_axilla_back[[0, 2]] - right_axilla_back[[0, 2]]))
    elif left_shoulder is not None and right_shoulder is not None:
        back_width = float(np.linalg.norm(left_shoulder[[0]] - right_shoulder[[0]]))

    chest_width = None
    if left_axilla_front is not None and right_axilla_front is not None:
        chest_width = float(np.linalg.norm(left_axilla_front[[0, 2]] - right_axilla_front[[0, 2]]))

    total_crotch_length = None
    if waist_front is not None and waist_back is not None and crotch_point is not None:
        total_crotch_length = float(
            np.linalg.norm(waist_front - crotch_point) + np.linalg.norm(crotch_point - waist_back)
        )

    shoulder_slope = None
    if neck_point is not None and left_acromion is not None:
        vec = left_acromion - neck_point
        horizontal = vec.copy()
        horizontal[1] = 0
        lateral = float(np.linalg.norm(horizontal))
        if lateral > 1e-6:
            shoulder_slope = math.degrees(math.atan2(abs(vec[1]), lateral))

    # Shared spine joints for both chest and waist SMPLX-style measurements.
    # Keypoints use underscores; MHR joint_map uses hyphens — try both.
    _pelvis_pt = _average_points([
        _coalesce_points(
            keypoint_map.get("left_hip"), keypoint_map.get("left-hip"), joint_map.get("left-hip")
        ),
        _coalesce_points(
            keypoint_map.get("right_hip"), keypoint_map.get("right-hip"), joint_map.get("right-hip")
        ),
    ])
    _neck_pt = _coalesce_points(keypoint_map.get("neck"), joint_map.get("neck"))

    # spine2 — at mid-chest level (~65 % up from pelvis to neck).
    # Used for the chest girth plane normal: gives a near-horizontal cut for
    # upright/slightly-tilted bodies, matching how a tape measure wraps.
    # Pelvis→spine2 is shorter and more vertical than pelvis→neck, so it
    # produces a better-conditioned cutting plane for chest measurement.
    _spine2_pt = _coalesce_points(
        joint_map.get("spine2"),
        joint_map.get("spine-2"),
        joint_map.get("spine_2"),
        joint_map.get("Spine2"),
        keypoint_map.get("spine2"),
        keypoint_map.get("spine-2"),
    )
    # If not found, approximate as 65 % of the way from pelvis up to neck —
    # spine2 sits at roughly the mid-chest / solar-plexus level.
    if _spine2_pt is None and _pelvis_pt is not None and _neck_pt is not None:
        _spine2_pt = np.asarray(_pelvis_pt, dtype=float) + 0.65 * (
            np.asarray(_neck_pt, dtype=float) - np.asarray(_pelvis_pt, dtype=float)
        )

    # spine3 — at waist level (between pelvis and neck). Try all likely MHR names.
    _spine3_pt = _coalesce_points(
        joint_map.get("spine3"),
        joint_map.get("spine-3"),
        joint_map.get("spine_3"),
        joint_map.get("Spine3"),
        keypoint_map.get("spine3"),
        keypoint_map.get("spine-3"),
    )
    # If not found, approximate as 40 % of the way from pelvis up to neck —
    # spine3 sits at roughly the navel/waist level.
    if _spine3_pt is None and _pelvis_pt is not None and _neck_pt is not None:
        _spine3_pt = np.asarray(_pelvis_pt, dtype=float) + 0.40 * (
            np.asarray(_neck_pt, dtype=float) - np.asarray(_pelvis_pt, dtype=float)
        )

    # SMPLX-style chest girth: landmark-anchored, arm-filtered, trimesh-preferred.
    # Uses spine2 (not neck) for the plane normal — gives a near-horizontal cut
    # at chest level, matching the anatomical tape-measure direction.
    if _chest_valid:
        _bust_girth_m, _chest_ring_3d = _chest_circumference_smplx_style(
            vertices=vertices,
            faces=_faces_arr,
            left_chest_idx=_L_CHEST,
            right_chest_idx=_R_CHEST,
            pelvis_joint=_pelvis_pt,
            spine_joint=_spine2_pt,
            left_wrist_joint=left_wrist,
            right_wrist_joint=right_wrist,
        )
    else:
        _bust_girth_m = _section_circumference(vertices, bust_level_y, 0.02)
        _chest_ring_3d = np.zeros((0, 3), dtype=float)

    # ── Shoulder ring + breadth ───────────────────────────────────────────────
    # ROOT CAUSE OF ALL PREVIOUS FAILURES:
    #   _trimesh_chest_contour picks the contour whose *centroid* is closest to
    #   the landmark point.  Passing vertex 7953 (left acromion, X≈−0.2 m) as
    #   the landmark caused it to return the LEFT ARM loop (small, off-centre),
    #   not the torso ring that spans the full shoulder width.
    #
    # FIX applied here:
    #   • Y-level = average of *both* acromion vertices (7952 left, 6615 right)
    #     → correct horizontal plane regardless of slight Y asymmetry
    #   • landmark = body centre at shoulder height (X = midpoint of acromion X
    #     positions ≈ 0, Z = mid of front/back anchors, shifted 3 cm posteriorly)
    #     → torso contour centroid is always closest to the body centre, so the
    #     torso ring is guaranteed to be selected
    #   • back arc (max-X → min-X via posterior Z) = shoulder breadth surface path
    _SHOULDER_RING_V: int = 7953  # kept for landmarks export
    _L_SHOULDER_V: int    = 7952  # left acromion surface vertex
    _R_SHOULDER_V: int    = 6615  # right acromion surface vertex
    _shoulder_ring_3d:     np.ndarray      = np.zeros((0, 3), dtype=float)
    _shoulder_breadth_arc: np.ndarray      = np.zeros((0, 3), dtype=float)
    shoulder_width:        Optional[float] = None

    if (_L_SHOULDER_V < vertices.shape[0] and
            _R_SHOULDER_V < vertices.shape[0] and
            _faces_arr is not None):
        try:
            _sh_vf = vertices.astype(float)

            # Correct Y: average of both acromion vertices
            _sh_y_lvl = (_sh_vf[_L_SHOULDER_V, 1] + _sh_vf[_R_SHOULDER_V, 1]) / 2.0

            # Body centre XZ:
            #   X = midpoint of both acromion vertices (≈ 0 for symmetric body)
            #   Z = midpoint of belly_button / back_belly anchors
            _sh_ctr_x     = (_sh_vf[_L_SHOULDER_V, 0] + _sh_vf[_R_SHOULDER_V, 0]) / 2.0
            _sh_belly_z_v = float(_sh_vf[_BELLY_BUTTON, 2])
            _sh_back_z_v  = float(_sh_vf[_BACK_BELLY_BUTTON, 2])
            _sh_ctr_z     = (_sh_belly_z_v + _sh_back_z_v) / 2.0

            # Plane origin at the body centre at correct shoulder height
            _sh_plane_org = np.array([_sh_ctr_x, _sh_y_lvl, _sh_ctr_z])

            # Landmark shifted 3 cm toward the back → torso ring centroid is
            # closest, so _trimesh_chest_contour reliably returns the torso ring
            _sh_back_dir_z = np.sign(_sh_back_z_v - _sh_belly_z_v)
            _sh_landmark   = np.array([_sh_ctr_x, _sh_y_lvl,
                                       _sh_ctr_z + _sh_back_dir_z * 0.03])

            _sh_new_ring = _trimesh_chest_contour(
                _sh_vf, _faces_arr,
                _sh_plane_org,
                np.array([0.0, 1.0, 0.0]),
                _sh_landmark,
                exclusion_xz=None,
            )

            if _sh_new_ring is not None and len(_sh_new_ring) >= 4:
                _shoulder_ring_3d = _sh_new_ring

                # Widest lateral extent of the torso ring at shoulder height
                _sba_ri = int(np.argmax(_sh_new_ring[:, 0]))  # max X = right side
                _sba_li = int(np.argmin(_sh_new_ring[:, 0]))  # min X = left side

                def _arc_seg(r: np.ndarray, a: int, b: int) -> np.ndarray:
                    return r[a:b+1] if a <= b else np.concatenate([r[a:], r[:b+1]])

                _arc_a = _arc_seg(_sh_new_ring, _sba_ri, _sba_li)
                _arc_b = _arc_seg(_sh_new_ring, _sba_li, _sba_ri)

                # Pick the arc whose mean Z is closer to the posterior anchor
                _za = float(_arc_a[:, 2].mean())
                _zb = float(_arc_b[:, 2].mean())
                _shoulder_breadth_arc = (
                    _arc_a if abs(_za - _sh_back_z_v) <= abs(_zb - _sh_back_z_v)
                    else _arc_b
                )

                _sba_diffs = _shoulder_breadth_arc[1:] - _shoulder_breadth_arc[:-1]
                shoulder_width = float(np.linalg.norm(_sba_diffs, axis=1).sum())
        except Exception:
            pass  # shoulder_width = None, _shoulder_breadth_arc = empty

    # SMPLX-style waist girth: sagittal anchor, spine3 plane, trimesh contour
    # for hand exclusion (same _trimesh_chest_contour mechanism as chest).
    if _waist_valid:
        _waist_girth_m, _waist_ring_3d = _waist_circumference_smplx_style(
            vertices=vertices,
            faces=_faces_arr,
            belly_button_idx=_BELLY_BUTTON,
            pelvis_joint=_pelvis_pt,
            spine3_joint=_spine3_pt,
            left_wrist_joint=left_wrist,
            right_wrist_joint=right_wrist,
        )
    else:
        _waist_girth_m = _section_circumference(vertices, waist_level_y, 0.02)
        _waist_ring_3d = np.zeros((0, 3), dtype=float)

    # Cross-section ring at the spine3 joint for viewport visualisation.
    # Slices the mesh with a horizontal plane through spine3 — gives the ring
    # that wraps around the body at that joint level (same pattern as chest /
    # waist / hip rings).
    # Build wrist exclusion zones once (reused by spine3 ring and waist slab debug)
    _wrist_excl_xz: Optional[List[np.ndarray]] = None
    _wrist_pts_list = [w for w in (left_wrist, right_wrist) if w is not None]
    if _wrist_pts_list:
        _wrist_excl_xz = [np.array([float(w[0]), float(w[2])]) for w in _wrist_pts_list]

    _spine3_ring_3d: np.ndarray = np.zeros((0, 3), dtype=float)
    if _spine3_pt is not None and _faces_arr is not None:
        _s3_origin = np.asarray(_spine3_pt, dtype=float)
        _s3_ring = _trimesh_chest_contour(
            vertices, _faces_arr,
            _s3_origin, np.array([0.0, 1.0, 0.0]), _s3_origin,
            exclusion_xz=_wrist_excl_xz,
        )
        if _s3_ring is not None and len(_s3_ring) >= 3:
            _spine3_ring_3d = _s3_ring

    # Compute waist slab vertices for the debug cut visualisation.
    # These are all mesh vertices within ±25 mm of the horizontal waist plane,
    # filtered to the torso region (XZ radius ≤ 0.28 m from the waist centre)
    # and with arm vertices excluded using wrist joint XZ positions.
    _waist_slab_vis: np.ndarray = np.zeros((0, 3), dtype=float)
    if _waist_valid:
        _wv_front = vertices[_BELLY_BUTTON]
        _wv_back = vertices[_BACK_BELLY_BUTTON]
        _waist_plane_origin_vis = (_wv_front + _wv_back) / 2.0
        _slab_raw = _slab_at_plane(vertices, _waist_plane_origin_vis,
                                   np.array([0.0, 1.0, 0.0]), 0.025)
        if len(_slab_raw) > 0:
            # Same pipeline as _waist_circumference_smplx_style for consistency.
            _slab_torso = _filter_waist_region(_slab_raw, _waist_plane_origin_vis, 0.28)
            # Joint-anchored hand exclusion (matches measurement function)
            _slab_torso = _exclude_hands_at_waist(
                _slab_torso, _waist_plane_origin_vis,
                left_wrist_joint=left_wrist,
                right_wrist_joint=right_wrist,
                hand_exclusion_radius_m=0.08,
                cluster_min_gap_m=0.12,
            )
            _waist_slab_vis = _slab_torso

    # SMPLX-style hip girth: pubic-bone-anchored, spine3-perpendicular plane,
    # trimesh exact slice.  Uses spine3 (same as waist) — not spine/neck.
    # The cutting height is lower than waist because plane origin = pubic bone
    # (below belly-button midpoint), not because the plane normal changes.
    _pubic_bone_idx = _find_pubic_bone_idx(vertices, hip_level_y, actual_height_m)
    _hip_girth_m, _hip_ring_3d = _hip_circumference_smplx_style(
        vertices=vertices,
        faces=_faces_arr,
        pubic_bone_idx=_pubic_bone_idx,
        pelvis_joint=_pelvis_pt,
        spine3_joint=_spine3_pt,
        left_wrist_joint=left_wrist,
        right_wrist_joint=right_wrist,
    )

    # SMPLX-style thigh girth: leg-axis plane through thigh landmark vertex,
    # UV-space radial gate to isolate each thigh from the other leg.
    _thigh_girth_m: float = 0.0
    _left_thigh_ring_3d: np.ndarray = np.zeros((0, 3), dtype=float)
    if _thigh_valid:
        _thigh_girth_m, _left_thigh_ring_3d = _thigh_girth_both_sides(
            vertices=vertices,
            faces=_faces_arr,
            left_landmark_indices=_L_THIGH_IDXS,
            right_landmark_indices=_R_THIGH_IDXS,
            left_hip_joint=left_hip,
            right_hip_joint=right_hip,
            left_knee_joint=left_knee,
            right_knee_joint=right_knee,
            spine3_joint=_spine3_pt,
            thickness=0.020,
            left_plane_anchor_idx=_L_THIGH_RING_IDX,
            right_plane_anchor_idx=_R_THIGH_RING_IDX,
        )

    raw_measurements: Dict[str, Optional[float]] = {
        "body_height": actual_height_m,
        "eye_height": (eye_point[1] - ground_y) if eye_point is not None else None,
        "cervicale_height": (neck_point[1] - ground_y) if neck_point is not None else None,
        "waist_height": waist_level_y - ground_y,
        "hip_height": hip_level_y - ground_y,
        "inside_leg_height": inside_leg,
        "knee_height": knee_level_y - ground_y,
        "head_girth": _section_circumference(vertices, head_girth_level, 0.01),
        "neck_girth": _section_circumference(vertices, neck_level_y, 0.01),
        "bust_girth": _bust_girth_m,
        "underbust_girth": _section_circumference(vertices, underbust_level_y, 0.02),
        "waist_girth": _waist_girth_m,
        "hip_girth": _hip_girth_m,
        "thigh_girth": _thigh_girth_m if _thigh_girth_m > 0.0 else None,
        "knee_girth": _average_values([knee_girth_left, knee_girth_right]),
        "calf_girth": _average_values([calf_girth_left, calf_girth_right]),
        "ankle_girth": _average_values([ankle_girth_left, ankle_girth_right]),
        "upper_arm_girth": _average_values([upper_arm_girth_left, upper_arm_girth_right]),
        "wrist_girth": _average_values([wrist_girth_left, wrist_girth_right]),
        "shoulder_width": shoulder_width,
        "back_width": back_width,
        "chest_width": chest_width,
        "arm_length": arm_length,
        "total_crotch_length": total_crotch_length,
        "shoulder_to_crotch": shoulder_to_crotch_raw,
        "shoulder_slope": shoulder_slope,
    }

    scaled_measurements: Dict[str, float] = {}
    for key, value in raw_measurements.items():
        if value is None:
            continue
        meta = MEASUREMENT_META.get(key)
        if meta is None:
            continue
        if meta.unit == "deg" or not meta.scales_with_height:
            scaled_measurements[key] = round(float(value), 2)
        else:
            scaled_measurements[key] = round(float(value) * scale_factor * 100.0, 2)

    landmarks = {
        "head_top": _landmark_to_list(head_vertex),
        "cervicale": _landmark_to_list(neck_point),
        "neck_side_left": _landmark_to_list(_average_points([neck_point, left_acromion])),
        "neck_side_right": _landmark_to_list(_average_points([neck_point, right_acromion])),
        "acromion_left": _landmark_to_list(left_acromion),
        "acromion_right": _landmark_to_list(right_acromion),
        "axilla_left_front": _landmark_to_list(left_axilla_front),
        "axilla_left_back": _landmark_to_list(left_axilla_back),
        "axilla_right_front": _landmark_to_list(right_axilla_front),
        "axilla_right_back": _landmark_to_list(right_axilla_back),
        "bust_point_left": _landmark_to_list(_average_points([left_axilla_front, waist_front])),
        "bust_point_right": _landmark_to_list(_average_points([right_axilla_front, waist_front])),
        "waist_level": round(float(waist_level_y), 5),
        "hip_level": round(float(hip_level_y), 5),
        "crotch": _landmark_to_list(crotch_point),
        # Inseam endpoints for 3D visualization
        "inseam_crotch_landmark": _landmark_to_list(crotch_point),
        "inseam_heel_landmark": _landmark_to_list(_left_heel_pt),
        # Arm length waypoints for 3D visualization (ordered shoulder → wrist)
        "arm_waypoints": [vertices[i].astype(float).round(6).tolist() for i in _ARM_WAYPOINTS] if _arm_valid else [],
        # Shoulder breadth — back arc of the shoulder ring (horizontal cross-section)
        "shoulder_breadth_path": _shoulder_breadth_arc.round(6).tolist() if len(_shoulder_breadth_arc) > 0 else [],
        "shoulder_left_landmark":  _shoulder_breadth_arc[-1].round(6).tolist() if len(_shoulder_breadth_arc) > 0 else None,
        "shoulder_right_landmark": _shoulder_breadth_arc[0].round(6).tolist()  if len(_shoulder_breadth_arc) > 0 else None,
        "lateral_malleolus_left": _landmark_to_list(left_ankle),
        "lateral_malleolus_right": _landmark_to_list(right_ankle),
        # Chest ring visualization data (SMPLX-style)
        "chest_ring_points": _chest_ring_3d.round(6).tolist() if len(_chest_ring_3d) > 0 else [],
        "left_chest_landmark": vertices[_L_CHEST].astype(float).round(6).tolist() if _chest_valid else None,
        "right_chest_landmark": vertices[_R_CHEST].astype(float).round(6).tolist() if _chest_valid else None,
        # Waist ring visualization data (SMPLX-style)
        "waist_ring_points": _waist_ring_3d.round(6).tolist() if len(_waist_ring_3d) > 0 else [],
        "belly_button_landmark": vertices[_BELLY_BUTTON].astype(float).round(6).tolist() if _waist_valid else None,
        "back_belly_button_landmark": vertices[_BACK_BELLY_BUTTON].astype(float).round(6).tolist() if _waist_valid else None,
        # Joints used for the waist plane normal (for viewport ring markers)
        "waist_pelvis_joint": _landmark_to_list(_pelvis_pt),
        "waist_spine3_joint": _landmark_to_list(_spine3_pt),
        "waist_left_wrist_joint": _landmark_to_list(left_wrist),
        "waist_right_wrist_joint": _landmark_to_list(right_wrist),
        # Cross-section ring AT the spine3 joint level (purple ring in viewport)
        "spine3_ring_points": _spine3_ring_3d.round(6).tolist() if len(_spine3_ring_3d) > 0 else [],
        # All mesh vertices within the horizontal waist slab (for debug cut view)
        "waist_slab_vertices": _waist_slab_vis.round(5).tolist() if len(_waist_slab_vis) > 0 else [],
        # Hip ring visualization data (SMPLX-style)
        "hip_ring_points": _hip_ring_3d.round(6).tolist() if len(_hip_ring_3d) > 0 else [],
        "pubic_bone_landmark": vertices[_pubic_bone_idx].astype(float).round(6).tolist(),
        # Left thigh ring visualization data (SMPLX-style)
        "left_thigh_ring_points": _left_thigh_ring_3d.round(6).tolist() if len(_left_thigh_ring_3d) > 0 else [],
        "left_thigh_landmark": vertices[_L_THIGH_IDXS].mean(axis=0).astype(float).round(6).tolist() if _thigh_valid else None,
        # Front body length — geodesic surface path for 3D visualization.
        # Subsampled to ≤200 points to keep JSON payload small.
        "shoulder_to_crotch_path": (
            [vertices[i].astype(float).round(6).tolist()
             for i in _sc_path_indices[::max(1, len(_sc_path_indices) // 200)]]
            if _sc_path_indices else []
        ),
        # Endpoint landmarks (neck top + crotch) — confirmed front-surface vertices
        "neck_top_landmark": vertices[_NECK_TOP_VERTEX_IDX].astype(float).round(6).tolist() if _sc_valid else None,
        # belly_button shown as mid-path anchor dot (confirmed front vertex)
        "front_waist_midline_landmark": vertices[_BELLY_BUTTON].astype(float).round(6).tolist() if _sc_valid else None,
        # Shoulder-level ring — horizontal cross-section through vertex 7953
        "shoulder_ring_points": _shoulder_ring_3d.round(6).tolist() if len(_shoulder_ring_3d) > 0 else [],
        "shoulder_ring_landmark": vertices[_SHOULDER_RING_V].astype(float).round(6).tolist() if _SHOULDER_RING_V < vertices.shape[0] else None,
    }

    schema = {
        key: {
            "unit": meta.unit,
            "category": meta.category,
            "scales_with_height": meta.scales_with_height,
        }
        for key, meta in MEASUREMENT_META.items()
    }

    return {
        "actual_height_cm": round(actual_height_cm, 2),
        "target_height_cm": round(target_height_cm, 2),
        "scale_factor": round(scale_factor, 4),
        "measurements": scaled_measurements,
        "landmarks": landmarks,
        "metadata": {
            "waist_level_y": round(float(waist_level_y), 5),
            "hip_level_y": round(float(hip_level_y), 5),
            "bust_level_y": round(float(bust_level_y), 5),
        },
        "schema": schema,
    }

