"""
Clean A-Pose Mesh Generator for MHR (Meta Human Rig).

Generates a structurally accurate A-pose mesh by zeroing all pose parameters
while preserving the identity (shape) and scale predicted by the monocular network.

Architecture Note:
    The MHR model uses an MLP for Stage 4 pose correctives. This MLP takes
    6D rotation features as input and has a zero-state design:
        feat[:, :, 0] -= 1
        feat[:, :, 4] -= 1
    When rotation matrices are identity (from zeroed pose), the diagonal
    elements are exactly 1. Subtracting 1 yields zero features, which produce
    zero corrective offsets from the MLP. Therefore, passing zeroed pose
    parameters cleanly bypasses the pose correctives network WITHOUT needing
    any inverse LBS or post-hoc mesh correction.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


@torch.no_grad()
def generate_clean_apose_mesh(
    shape_params: torch.Tensor,
    scale_params: torch.Tensor,
    mhr_model,  # MHR instance (from mhr.mhr.MHR)
    mhr_head,   # MHRHead instance (has scale_mean, scale_comps, hand buffers)
    expr_params: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a clean A-pose mesh by zeroing all pose parameters while
    preserving the predicted identity (shape) and anatomical scale.

    This exploits the MHR architecture's zero-state design:
    - Zero global_trans → mesh locked to origin
    - Zero global_rot   → no root rotation
    - Zero body_pose    → all joints at their rest (A-pose) angles
    - Zero hand_pose    → fingers at rest position
    - The MLP pose correctives network receives all-zero features,
      producing zero corrective offsets (no mesh distortion)

    Args:
        shape_params: Identity blendshape coefficients [1, 45] from SAM-3D.
                      These encode the person's body shape and ARE preserved.
        scale_params: Anatomical scale coefficients [1, 28] from SAM-3D.
                      These control per-joint bone lengths and ARE preserved.
        mhr_model:    The core MHR model instance (mhr.mhr.MHR).
                      Has .forward(identity_coeffs, model_parameters, face_expr_coeffs).
        mhr_head:     The MHRHead wrapper that holds scale_mean [68],
                      scale_comps [28, 68], and hand PCA buffers.
        expr_params:  Optional face expression coefficients [1, 72].
                      Zeroed if not provided (neutral expression).
        device:       Target device (inferred from shape_params if not given).

    Returns:
        vertices:        np.ndarray [N, 3] — A-pose vertex positions (meters).
        joint_positions: np.ndarray [J, 3] — Joint positions from skeleton state (meters).
    """

    # -------------------------------------------------------------------------
    # 1. Ensure shape_params and scale_params are batched [1, D] tensors on GPU
    # -------------------------------------------------------------------------
    if isinstance(shape_params, np.ndarray):
        shape_params = torch.tensor(shape_params, dtype=torch.float32)
    if isinstance(scale_params, np.ndarray):
        scale_params = torch.tensor(scale_params, dtype=torch.float32)

    if device is None:
        device = shape_params.device if shape_params.is_cuda else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    shape_params = shape_params.to(device).float()
    scale_params = scale_params.to(device).float()

    if shape_params.dim() == 1:
        shape_params = shape_params.unsqueeze(0)  # [45] → [1, 45]
    if scale_params.dim() == 1:
        scale_params = scale_params.unsqueeze(0)  # [28] → [1, 28]

    # -------------------------------------------------------------------------
    # 2. EXCLUDE all pose: zero global_trans, global_rot, body_pose
    # -------------------------------------------------------------------------
    # These zeroed tensors ensure:
    #   - No root translation (mesh at origin)
    #   - No root rotation (canonical orientation)
    #   - All joint rotations = identity → MLP correctives = 0
    global_trans = torch.zeros(1, 3, device=device, dtype=torch.float32)
    global_rot   = torch.zeros(1, 3, device=device, dtype=torch.float32)
    body_pose    = torch.zeros(1, 130, device=device, dtype=torch.float32)

    # -------------------------------------------------------------------------
    # 3. Build full_pose_params [1, 136] = [trans*10 | rot | body_pose]
    #    Then set hand joints to zero (no hand PCA transformation needed)
    # -------------------------------------------------------------------------
    # Layout: [global_trans*10 (3) | global_rot (3) | body_pose (130)] = 136
    full_pose_params = torch.cat(
        [global_trans * 10, global_rot, body_pose], dim=1
    )  # [1, 136]

    # Hands: since all pose is zero, we skip replace_hands_in_pose()
    # The hand joint indices within full_pose_params are already zeroed.

    # -------------------------------------------------------------------------
    # 4. PRESERVE scale: transform scale_params through learned PCA basis
    # -------------------------------------------------------------------------
    # scale_mean [68] and scale_comps [28, 68] are stored on the MHRHead
    scales = mhr_head.scale_mean[None, :].to(device) + scale_params @ mhr_head.scale_comps.to(device)
    # scales: [1, 68] — per-joint bone length multipliers

    # -------------------------------------------------------------------------
    # 5. Assemble model_parameters [1, 204] = [full_pose (136) | scales (68)]
    # -------------------------------------------------------------------------
    model_parameters = torch.cat([full_pose_params, scales], dim=1)  # [1, 204]

    # -------------------------------------------------------------------------
    # 6. Handle expression params (zero = neutral face)
    # -------------------------------------------------------------------------
    if expr_params is not None:
        if isinstance(expr_params, np.ndarray):
            expr_params = torch.tensor(expr_params, dtype=torch.float32)
        expr_params = expr_params.to(device).float()
        if expr_params.dim() == 1:
            expr_params = expr_params.unsqueeze(0)
    # If None, MHR.forward() will create zero padding internally

    # -------------------------------------------------------------------------
    # 7. Forward pass through MHR
    # -------------------------------------------------------------------------
    # apply_correctives=True is safe here because:
    #   - Zero pose → identity rotation matrices
    #   - _pose_features_from_joint_params subtracts 1 from diagonal
    #   - Identity diagonal = 1, so features = 0
    #   - MLP(0) produces zero corrective offsets
    #   - Result: pure rest-pose geometry + identity blendshapes
    verts, skel_state = mhr_model.forward(
        identity_coeffs=shape_params,
        model_parameters=model_parameters,
        face_expr_coeffs=expr_params,
        apply_correctives=True,  # Safe: zeroed pose → zero MLP output
    )

    # -------------------------------------------------------------------------
    # 8. Extract outputs & convert to meters (MHR internal units are cm)
    # -------------------------------------------------------------------------
    # MHR outputs are in centimeters, divide by 100 for meters
    vertices = (verts[0] / 100.0).cpu().numpy()         # [N, 3]
    joint_positions_raw, _, _ = torch.split(skel_state, [3, 4, 1], dim=2)
    joint_positions = (joint_positions_raw[0] / 100.0).cpu().numpy()  # [J, 3]

    # -------------------------------------------------------------------------
    # 9. Apply camera system difference
    # -------------------------------------------------------------------------
    # MHR natively outputs a different coordinate system than SAM-3D expects.
    # sam_3d_body's mhr_head.py negates Y and Z before returning predictions.
    # We must do the same here so our clean A-pose matches the posed mesh space.
    vertices[..., [1, 2]] *= -1.0
    joint_positions[..., [1, 2]] *= -1.0

    return vertices, joint_positions
