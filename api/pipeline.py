"""
Inference pipeline: image -> 3D mesh -> body measurements.

Wraps the SAM-3D-Body model, ViTDet detector, MoGe2 FOV estimator,
and the measurement computation into a single callable pipeline.
"""

import os
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from notebook.utils import setup_sam_3d_body
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from sam_3d_body.measurements import compute_measurements, MeasurementError


MAX_LONG_EDGE = 2048

# ---------------------------------------------------------------------------
# MHR70 skeleton helpers (from app.py)
# ---------------------------------------------------------------------------

def _rotate_points_x(points: np.ndarray) -> np.ndarray:
    rotated = np.array(points, dtype=np.float32, copy=True)
    rotated[..., 1] *= -1.0
    rotated[..., 2] *= -1.0
    return rotated


def _build_mhr70_skeleton():
    kp_info = mhr70_pose_info["keypoint_info"]
    skeleton_info = mhr70_pose_info["skeleton_info"]
    joint_names = [kp_info[i]["name"] for i in range(len(kp_info))]
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    adjacency = {idx: [] for idx in range(len(joint_names))}
    for link_def in skeleton_info.values():
        joint_a, joint_b = link_def["link"]
        ia, ib = name_to_idx[joint_a], name_to_idx[joint_b]
        adjacency[ia].append(ib)
        adjacency[ib].append(ia)
    root_name = "neck" if "neck" in name_to_idx else joint_names[0]
    root_idx = name_to_idx[root_name]
    parents = [-1] * len(joint_names)
    queue = [root_idx]
    visited = {root_idx}
    while queue:
        current = queue.pop(0)
        for nb in adjacency[current]:
            if nb in visited:
                continue
            parents[nb] = current
            visited.add(nb)
            queue.append(nb)
    return joint_names, parents


MHR70_NAMES, _ = _build_mhr70_skeleton()
MHR70_NAME_TO_IDX = {name: idx for idx, name in enumerate(MHR70_NAMES)}


def _extract_mhr_template(mhr_module, max_influences=4):
    buffers = dict(mhr_module.named_buffers())
    joint_parents = buffers["character_torch.skeleton.joint_parents"].detach().cpu().numpy().astype(np.int32)
    joint_offsets = buffers["character_torch.skeleton.joint_translation_offsets"].detach().cpu().numpy() / 100.0
    joint_offsets = _rotate_points_x(joint_offsets)
    num_vertices = int(buffers["character_torch.mesh.rest_vertices"].shape[0])

    vert_indices = buffers["character_torch.linear_blend_skinning.vert_indices_flattened"].detach().cpu().numpy().astype(np.int64)
    skin_indices = buffers["character_torch.linear_blend_skinning.skin_indices_flattened"].detach().cpu().numpy().astype(np.int64)
    skin_weights = buffers["character_torch.linear_blend_skinning.skin_weights_flattened"].detach().cpu().numpy().astype(np.float32)

    per_vertex = defaultdict(list)
    for vid, jid, weight in zip(vert_indices, skin_indices, skin_weights):
        per_vertex[int(vid)].append((float(weight), int(jid)))

    skin_index_array = np.zeros((num_vertices, max_influences), dtype=np.int32)
    skin_weight_array = np.zeros((num_vertices, max_influences), dtype=np.float32)
    for vid in range(num_vertices):
        entries = per_vertex.get(vid, [])
        if not entries:
            continue
        entries.sort(key=lambda item: item[0], reverse=True)
        selected = entries[:max_influences]
        total = sum(weight for weight, _ in selected) or 1.0
        for slot, (weight, jid) in enumerate(selected):
            skin_index_array[vid, slot] = jid
            skin_weight_array[vid, slot] = weight / total

    joint_names = MHR70_NAMES[:len(joint_parents)] if len(MHR70_NAMES) >= len(joint_parents) else [f"joint_{idx}" for idx in range(len(joint_parents))]
    root_index = int(np.where(joint_parents == -1)[0][0])

    return {
        "joint_names": joint_names,
        "joint_parents": joint_parents,
        "joint_offsets": joint_offsets,
        "skin_indices": skin_index_array,
        "skin_weights": skin_weight_array,
        "root_index": root_index,
    }


def _prepare_person_rig(person_output, template, faces):
    """Build rig_payload dict for one person (same as app.py export)."""
    vertices = np.array(person_output["pred_vertices"], dtype=np.float32)
    joint_positions = np.array(person_output["pred_joint_coords"], dtype=np.float32)
    joint_positions = joint_positions[:template["joint_parents"].shape[0]]
    keypoints = np.array(person_output["pred_keypoints_3d"], dtype=np.float32)

    cam_t = person_output.get("pred_cam_t")
    if cam_t is not None:
        vertices = vertices + cam_t
        joint_positions = joint_positions + cam_t
        keypoints = keypoints + cam_t

    root_idx = template["root_index"]
    root_offset = joint_positions[root_idx].copy()
    vertices = vertices - root_offset
    joint_positions = joint_positions - root_offset
    keypoints = keypoints - root_offset

    vertices = _rotate_points_x(vertices)
    joint_positions = _rotate_points_x(joint_positions)
    keypoints = _rotate_points_x(keypoints)

    # Match keypoints to joints
    target_mapping = {}
    for name, kp_idx in MHR70_NAME_TO_IDX.items():
        if kp_idx >= len(keypoints):
            continue
        dists = np.linalg.norm(joint_positions - keypoints[kp_idx][None, :], axis=1)
        target_mapping[name] = int(np.argmin(dists))

    # Build improved joint names
    joint_idx_to_name = {}
    for kp_name, joint_idx in target_mapping.items():
        if joint_idx not in joint_idx_to_name:
            joint_idx_to_name[joint_idx] = kp_name
    improved_joint_names = []
    for ji in range(len(template["joint_names"])):
        improved_joint_names.append(joint_idx_to_name.get(ji, template["joint_names"][ji]))

    return {
        "mesh": {
            "vertices": np.round(vertices, 6).tolist(),
            "faces": faces.astype(int).tolist(),
            "skinIndices": template["skin_indices"].tolist(),
            "skinWeights": template["skin_weights"].tolist(),
        },
        "skeleton": {
            "joint_names": improved_joint_names,
            "parents": template["joint_parents"].tolist(),
            "rest_offsets": template["joint_offsets"].tolist(),
            "joint_positions": np.round(joint_positions, 6).tolist(),
        },
        "animation_targets": target_mapping,
        "keypoints": [
            {"name": name, "position": np.round(keypoints[kp_idx], 6).tolist()}
            for name, kp_idx in MHR70_NAME_TO_IDX.items()
        ],
        "metadata": {
            "focal_length": float(person_output["focal_length"]),
            "bbox": person_output["bbox"].tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Pipeline singleton
# ---------------------------------------------------------------------------

class BodyMeasurementPipeline:
    """Thread-safe singleton that loads models once and runs inference."""

    def __init__(self):
        self._estimator = None
        self._rig_template = None
        self._lock = threading.Lock()
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_models(self, lightweight: bool = False):
        """Load all models. Call once at startup."""
        with self._lock:
            if self._loaded:
                return
            print("=" * 60)
            print("Loading SAM-3D-Body pipeline...")
            print("=" * 60)

            if lightweight:
                self._estimator = setup_sam_3d_body(
                    hf_repo_id="facebook/sam-3d-body-dinov3",
                    fov_name=None,
                )
            else:
                self._estimator = setup_sam_3d_body(
                    hf_repo_id="facebook/sam-3d-body-dinov3",
                )

            self._rig_template = _extract_mhr_template(
                self._estimator.model.head_pose.mhr
            )
            self._loaded = True
            print("=" * 60)
            print("Pipeline ready.")
            print("=" * 60)

    def process_image(
        self,
        image_bytes: bytes,
        height_cm: float,
        person_index: int = 0,
    ) -> Dict[str, Any]:
        """
        Full pipeline: image bytes -> measurements dict.

        Returns dict with keys: measurements, num_persons, person_index.
        Raises RuntimeError or MeasurementError on failure.
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded")

        # Decode image
        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError("Could not decode image")

        # Resize if too large
        h, w = img_bgr.shape[:2]
        long_edge = max(h, w)
        if long_edge > MAX_LONG_EDGE:
            scale = MAX_LONG_EDGE / long_edge
            img_bgr = cv2.resize(
                img_bgr, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        # Stage 1+2: detection + 3D reconstruction
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        with self._lock:
            outputs = self._estimator.process_one_image(img_rgb)

        if not outputs:
            raise RuntimeError("No persons detected in image")

        num_persons = len(outputs)
        if person_index >= num_persons:
            raise RuntimeError(
                f"person_index {person_index} out of range (detected {num_persons})"
            )

        # Stage 3: build rig + compute measurements
        person_output = outputs[person_index]
        rig_payload = _prepare_person_rig(
            person_output, self._rig_template, self._estimator.faces
        )
        result = compute_measurements(rig_payload, target_height_cm=height_cm)

        return {
            "measurements": result.get("measurements", {}),
            "num_persons": num_persons,
            "person_index": person_index,
        }


# Module-level singleton
pipeline = BodyMeasurementPipeline()
