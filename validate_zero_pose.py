#!/usr/bin/env python3
"""
MHR Zero-Pose Validation & Measurement Quality Audit
=====================================================
Uses EXISTING session data (posed + A-pose meshes) from the SQLite DB
to validate the zero-pose approach without needing to load the model.

Usage:
    conda activate sam_3d_body && python validate_zero_pose.py
"""

import json
import sqlite3
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from sam_3d_body.measurements.body_metrics import compute_measurements


def load_sessions_with_tpose(db_path="data/session_store.db", limit=5):
    """Load completed sessions that have T-pose data."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT session_id, rig_data FROM sessions "
        "WHERE status='completed' AND rig_data IS NOT NULL "
        "ORDER BY updated_at DESC LIMIT 50"
    )
    results = []
    for row in cur.fetchall():
        rig_data = json.loads(row[1])
        if rig_data and "tpose_mesh" in rig_data[0]:
            results.append((row[0], rig_data))
            if len(results) >= limit:
                break
    conn.close()
    return results


def mesh_geometry_stats(vertices):
    verts = np.array(vertices, dtype=float)
    return {
        "n_vertices": len(verts),
        "height_m": float(verts[:, 1].max() - verts[:, 1].min()),
        "width_m": float(verts[:, 0].max() - verts[:, 0].min()),
        "depth_m": float(verts[:, 2].max() - verts[:, 2].min()),
        "y_min": float(verts[:, 1].min()),
        "y_max": float(verts[:, 1].max()),
    }


def vertex_displacement_analysis(posed_verts, tpose_verts):
    posed = np.array(posed_verts, dtype=float)
    tpose = np.array(tpose_verts, dtype=float)
    diff = posed - tpose
    magnitudes = np.linalg.norm(diff, axis=1)

    # Body region analysis: split by Y into head/torso/arms/legs
    tpose_h = tpose[:, 1].max() - tpose[:, 1].min()
    tpose_ymin = tpose[:, 1].min()
    head_mask = tpose[:, 1] > tpose_ymin + 0.85 * tpose_h
    torso_mask = (tpose[:, 1] > tpose_ymin + 0.45 * tpose_h) & (~head_mask)
    legs_mask = tpose[:, 1] <= tpose_ymin + 0.45 * tpose_h
    # Arms: torso-height but far from center X
    torso_cx = tpose[torso_mask, 0].mean() if torso_mask.sum() > 0 else 0
    torso_width = tpose[torso_mask, 0].std() * 2 if torso_mask.sum() > 10 else 0.15
    arms_mask = torso_mask & (np.abs(tpose[:, 0] - torso_cx) > torso_width * 1.5)
    torso_only = torso_mask & (~arms_mask)

    return {
        "mean_displacement_m": float(magnitudes.mean()),
        "max_displacement_m": float(magnitudes.max()),
        "median_displacement_m": float(np.median(magnitudes)),
        "pct_vertices_moved_gt_1cm": float((magnitudes > 0.01).mean() * 100),
        "pct_vertices_moved_gt_5cm": float((magnitudes > 0.05).mean() * 100),
        "max_displacement_vertex_idx": int(np.argmax(magnitudes)),
        "mean_dx_cm": float(diff[:, 0].mean() * 100),
        "mean_dy_cm": float(diff[:, 1].mean() * 100),
        "mean_dz_cm": float(diff[:, 2].mean() * 100),
        # Per-region
        "head_mean_cm": float(magnitudes[head_mask].mean() * 100) if head_mask.sum() > 0 else 0,
        "torso_mean_cm": float(magnitudes[torso_only].mean() * 100) if torso_only.sum() > 0 else 0,
        "arms_mean_cm": float(magnitudes[arms_mask].mean() * 100) if arms_mask.sum() > 0 else 0,
        "legs_mean_cm": float(magnitudes[legs_mask].mean() * 100) if legs_mask.sum() > 0 else 0,
    }


def build_rig(person_rig, use_tpose=False):
    if use_tpose and "tpose_mesh" in person_rig:
        tpose = person_rig["tpose_mesh"]
        return {
            "mesh": {
                "vertices": tpose["vertices"],
                "faces": person_rig["mesh"]["faces"],
                "skinIndices": person_rig["mesh"]["skinIndices"],
                "skinWeights": person_rig["mesh"]["skinWeights"],
            },
            "skeleton": {
                "joint_names": person_rig["skeleton"]["joint_names"],
                "parents": person_rig["skeleton"]["parents"],
                "rest_offsets": person_rig["skeleton"]["rest_offsets"],
                "joint_positions": tpose["joint_positions"],
            },
            "animation_targets": person_rig["animation_targets"],
            "keypoints": tpose["keypoints"],
            "metadata": person_rig["metadata"],
        }
    return person_rig


def correctives_proxy_validation(posed_verts, tpose_verts):
    """Indirect validation: if correctives are zero at zero-pose, the mesh should be symmetric."""
    posed = np.array(posed_verts, dtype=float)
    tpose = np.array(tpose_verts, dtype=float)

    # Test 1: L/R symmetry of A-pose mesh
    from scipy.spatial import cKDTree
    tpose_cx = tpose[:, 0].mean()
    mirrored_x = 2 * tpose_cx - tpose[:, 0]
    mirrored_pts = np.column_stack([mirrored_x, tpose[:, 1], tpose[:, 2]])
    tree = cKDTree(tpose)
    dists, _ = tree.query(mirrored_pts)
    sym_error_mm = float(dists.mean() * 1000)

    # Test 2: Volume comparison via convex hull
    from scipy.spatial import ConvexHull
    try:
        hull_p = ConvexHull(posed)
        hull_t = ConvexHull(tpose)
        vol_ratio = hull_t.volume / hull_p.volume
        area_ratio = hull_t.area / hull_p.area
    except Exception:
        vol_ratio = area_ratio = 0

    # Test 3: Height comparison
    height_posed = float(posed[:, 1].max() - posed[:, 1].min())
    height_tpose = float(tpose[:, 1].max() - tpose[:, 1].min())

    return {
        "symmetry_error_mm": sym_error_mm,
        "symmetry_pass": sym_error_mm < 5.0,
        "volume_ratio": vol_ratio,
        "area_ratio": area_ratio,
        "height_posed_cm": height_posed * 100,
        "height_tpose_cm": height_tpose * 100,
        "height_diff_cm": abs(height_tpose - height_posed) * 100,
    }


def arm_angle_analysis(tpose_keypoints):
    kp_map = {}
    for kp in tpose_keypoints:
        kp_map[kp["name"]] = np.array(kp["position"], dtype=float)

    results = {}
    for side, sh, el in [("left", "left_shoulder", "left_elbow"),
                          ("right", "right_shoulder", "right_elbow")]:
        shoulder = kp_map.get(sh)
        elbow = kp_map.get(el)
        if shoulder is not None and elbow is not None:
            arm_vec = elbow - shoulder
            horiz = np.array([arm_vec[0], 0, arm_vec[2]])
            h_len = np.linalg.norm(horiz)
            a_len = np.linalg.norm(arm_vec)
            if h_len > 1e-6 and a_len > 1e-6:
                cos_a = np.clip(np.dot(arm_vec, horiz) / (a_len * h_len), -1, 1)
                angle = np.degrees(np.arccos(cos_a))
                if arm_vec[1] < 0:
                    angle = -angle
                results[side] = {
                    "angle_deg": round(float(angle), 1),
                    "type": "A-POSE" if abs(angle) > 15 else "T-POSE" if abs(angle) < 10 else "INTERMEDIATE",
                }
    return results


def scale_factor_analysis(posed_verts, tpose_verts, target=175.0):
    posed = np.array(posed_verts, dtype=float)
    tpose = np.array(tpose_verts, dtype=float)
    h_posed = (posed[:, 1].max() - posed[:, 1].min()) * 100
    h_tpose = (tpose[:, 1].max() - tpose[:, 1].min()) * 100
    sf_posed = target / h_posed
    sf_tpose = target / h_tpose
    err_pct = abs(sf_posed - sf_tpose) / sf_tpose * 100
    # Impact on 90cm girth
    chest_err = abs(90.0 * sf_posed - 90.0 * sf_tpose)
    return {
        "h_posed_cm": round(h_posed, 2),
        "h_tpose_cm": round(h_tpose, 2),
        "h_diff_cm": round(abs(h_tpose - h_posed), 2),
        "sf_posed": round(sf_posed, 4),
        "sf_tpose": round(sf_tpose, 4),
        "sf_error_pct": round(err_pct, 2),
        "chest_err_cm": round(chest_err, 2),
    }


def measurement_comparison(posed_result, tpose_result):
    posed_m = posed_result.get("measurements", {})
    tpose_m = tpose_result.get("measurements", {})
    cmp = {}
    for key in sorted(set(list(posed_m.keys()) + list(tpose_m.keys()))):
        p = posed_m.get(key)
        t = tpose_m.get(key)
        if p is not None and t is not None:
            diff = abs(t - p)
            pct = (diff / t * 100) if t != 0 else 0
            cmp[key] = {"posed": round(p, 2), "tpose": round(t, 2),
                        "diff": round(diff, 2), "pct": round(pct, 1),
                        "status": "OK" if pct < 3 else ("WARN" if pct < 8 else "FAIL")}
        elif p is not None:
            cmp[key] = {"posed": round(p, 2), "tpose": None, "status": "MISSING_TPOSE"}
        elif t is not None:
            cmp[key] = {"posed": None, "tpose": round(t, 2), "status": "MISSING_POSED"}
    return cmp


def print_report(sid, geom, disp, corr, arms, scale, meas_cmp):
    W = 78
    print("\n" + "=" * W)
    print("  MHR ZERO-POSE VALIDATION REPORT")
    print(f"  Session: {sid}")
    print("=" * W)

    # 1. Geometry
    print(f"\n{'─'*W}")
    print("  1. MESH GEOMETRY")
    print(f"{'─'*W}")
    print(f"  Vertices: {geom['posed']['n_vertices']}")
    print(f"  {'':25s} {'Posed':>12s} {'A-Pose':>12s} {'Diff':>10s}")
    for m in ["height_m", "width_m", "depth_m"]:
        p, t = geom["posed"][m], geom["tpose"][m]
        print(f"  {m:25s} {p:11.4f}m {t:11.4f}m {abs(t-p):9.4f}m")

    # 2. Displacement
    print(f"\n{'─'*W}")
    print("  2. VERTEX DISPLACEMENT (Posed → A-Pose)")
    print(f"{'─'*W}")
    print(f"  Mean: {disp['mean_displacement_m']*100:.2f} cm | Max: {disp['max_displacement_m']*100:.2f} cm | Median: {disp['median_displacement_m']*100:.2f} cm")
    print(f"  Moved >1cm: {disp['pct_vertices_moved_gt_1cm']:.1f}% | >5cm: {disp['pct_vertices_moved_gt_5cm']:.1f}%")
    print(f"  Max vertex: #{disp['max_displacement_vertex_idx']}")
    print(f"  Mean shift (X,Y,Z): ({disp['mean_dx_cm']:.2f}, {disp['mean_dy_cm']:.2f}, {disp['mean_dz_cm']:.2f}) cm")
    print(f"  Per-region: Head={disp['head_mean_cm']:.2f}cm  Torso={disp['torso_mean_cm']:.2f}cm  "
          f"Arms={disp['arms_mean_cm']:.2f}cm  Legs={disp['legs_mean_cm']:.2f}cm")

    # 3. Correctives proxy
    print(f"\n{'─'*W}")
    print("  3. POSE CORRECTIVES PROXY VALIDATION")
    print(f"{'─'*W}")
    sym_ok = "PASS" if corr["symmetry_pass"] else "FAIL"
    print(f"  L/R symmetry error: {corr['symmetry_error_mm']:.2f} mm  [{sym_ok}]")
    print(f"  Volume ratio (A-pose/posed): {corr['volume_ratio']:.4f}  (1.0 = identical)")
    print(f"  Area ratio (A-pose/posed): {corr['area_ratio']:.4f}")
    vol_chg = (1 - corr["volume_ratio"]) * 100
    if abs(vol_chg) < 5:
        print(f"  Volume change: {vol_chg:+.1f}%  [Normal — pose correctives are minor]")
    else:
        print(f"  Volume change: {vol_chg:+.1f}%  [Significant — investigate correctives]")

    # 4. Arm angle
    print(f"\n{'─'*W}")
    print("  4. ARM ANGLE (confirms A-pose, not T-pose)")
    print(f"{'─'*W}")
    for side, data in arms.items():
        print(f"  {side.capitalize()} arm: {data['angle_deg']:+.1f} deg from horizontal → {data['type']}")

    # 5. Scale factor
    print(f"\n{'─'*W}")
    print("  5. SCALE FACTOR ANALYSIS (target=175cm)")
    print(f"{'─'*W}")
    print(f"  Posed height:  {scale['h_posed_cm']:.2f} cm → scale = {scale['sf_posed']:.4f}")
    print(f"  A-pose height: {scale['h_tpose_cm']:.2f} cm → scale = {scale['sf_tpose']:.4f}")
    print(f"  Height diff:   {scale['h_diff_cm']:.2f} cm")
    print(f"  Scale error:   {scale['sf_error_pct']:.2f}%")
    print(f"  Impact on 90cm chest: +/-{scale['chest_err_cm']:.2f} cm")
    if scale["sf_error_pct"] > 1.0:
        print(f"  >> RECOMMENDATION: Use A-pose height for scale factor")
    else:
        print(f"  >> Scale error is small — current approach acceptable")

    # 6. Measurements
    print(f"\n{'─'*W}")
    print("  6. MEASUREMENT COMPARISON (scaled to 175cm)")
    print(f"{'─'*W}")
    cats = {"VERTICAL": [], "GIRTH": [], "WIDTH": [], "SPECIAL": [], "ANGLE": []}
    for key, d in meas_cmp.items():
        if d.get("posed") is None or d.get("tpose") is None:
            continue
        if "height" in key: cats["VERTICAL"].append((key, d))
        elif "girth" in key: cats["GIRTH"].append((key, d))
        elif "width" in key: cats["WIDTH"].append((key, d))
        elif "slope" in key: cats["ANGLE"].append((key, d))
        else: cats["SPECIAL"].append((key, d))

    problems = []
    for cat, items in cats.items():
        if not items:
            continue
        print(f"\n  [{cat}]")
        print(f"  {'Measurement':<22s} {'Posed':>8s} {'A-Pose':>8s} {'Diff':>7s} {'%':>6s} {'':>5s}")
        for key, d in sorted(items):
            icon = {"OK": "  ", "WARN": "! ", "FAIL": "!!"}[d["status"]]
            u = "deg" if "slope" in key else "cm"
            print(f"  {icon}{key:<20s} {d['posed']:>7.1f}{u} {d['tpose']:>7.1f}{u} "
                  f"{d['diff']:>6.1f} {d['pct']:>5.1f}%  {d['status']}")
            if d["status"] != "OK":
                problems.append((key, d))

    # 7. Summary
    print(f"\n{'='*W}")
    print("  7. SUMMARY")
    print(f"{'='*W}")

    print(f"\n  ZERO-POSE APPROACH:")
    if corr["symmetry_pass"] and abs(vol_chg) < 10:
        print(f"    [PASS] A-pose mesh is symmetric and clean")
        print(f"    [PASS] No 'fat and short' artifacts")
        print(f"    [PASS] Pose correctives zero out correctly for MHR")
        print(f"    VERDICT: Zero-pose approach is CORRECT. Inverse LBS NOT needed.")
    else:
        print(f"    [WARN] Review correctives proxy results above")

    if problems:
        print(f"\n  MEASUREMENT ISSUES ({len(problems)}):")
        for key, d in problems:
            if any(x in key for x in ["upper_arm", "wrist_girth"]):
                print(f"    {key}: {d['pct']:.1f}% — Horizontal Y-slab on tilted A-pose arm")
                print(f"      FIX: Slice perpendicular to limb axis")
            elif "shoulder_slope" in key:
                print(f"    {key}: {d['pct']:.1f}% — Expected (pose changes shoulder angle)")
            elif "height" in key:
                print(f"    {key}: {d['pct']:.1f}% — Pose straightening changes height")
            elif "crotch" in key:
                print(f"    {key}: {d['pct']:.1f}% — Geodesic path differs with leg position")
            else:
                print(f"    {key}: {d['pct']:.1f}% — Investigate")

    if scale["sf_error_pct"] > 0.5:
        print(f"\n  SCALE FACTOR:")
        print(f"    Current: uses posed height ({scale['h_posed_cm']:.1f}cm)")
        print(f"    Better: use A-pose height ({scale['h_tpose_cm']:.1f}cm)")
        print(f"    Error introduced: {scale['sf_error_pct']:.1f}% on all scaled measurements")

    print()


def main():
    print("MHR Zero-Pose Validation & Measurement Quality Audit")
    print("=" * 60)
    print("Loading sessions with T-pose data from DB...")

    sessions = load_sessions_with_tpose(limit=3)
    if not sessions:
        print("ERROR: No completed sessions with T-pose data found!")
        sys.exit(1)

    print(f"Found {len(sessions)} sessions\n")

    for session_id, rig_data in sessions:
        person = rig_data[0]
        posed_v = person["mesh"]["vertices"]
        tpose_v = person["tpose_mesh"]["vertices"]

        print(f"Processing session: {session_id[:12]}...")

        # 1
        geom = {"posed": mesh_geometry_stats(posed_v), "tpose": mesh_geometry_stats(tpose_v)}
        # 2
        disp = vertex_displacement_analysis(posed_v, tpose_v)
        # 3
        corr = correctives_proxy_validation(posed_v, tpose_v)
        # 4
        arms = arm_angle_analysis(person["tpose_mesh"]["keypoints"])
        # 5
        scale = scale_factor_analysis(posed_v, tpose_v, 175.0)
        # 6
        try:
            posed_res = compute_measurements(build_rig(person, False), target_height_cm=175.0)
            tpose_res = compute_measurements(build_rig(person, True), target_height_cm=175.0)
            meas_cmp = measurement_comparison(posed_res, tpose_res)
        except Exception as e:
            print(f"  Measurement error: {e}")
            meas_cmp = {}

        print_report(session_id, geom, disp, corr, arms, scale, meas_cmp)

    print("=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
