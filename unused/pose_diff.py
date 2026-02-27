import os
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
import argparse
from render.mesh_viz import visualize_body_obj
import smplx
from human_body_prior.body_model.body_model import BodyModel
import csv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu()

MODEL_PATH = 'models'


######################################## smplh 10 ########################################
smplh_model_male = smplx.create(MODEL_PATH, model_type='smplh',
        gender="male",
        use_pca=False,
        flat_hand_mean = True,
        ext='pkl')

smplh_model_female = smplx.create(MODEL_PATH, model_type='smplh',
        gender="female",
        use_pca=False,
        flat_hand_mean = True,
        ext='pkl')

smplh_model_neutral = smplx.create(MODEL_PATH, model_type='smplh',
        gender="neutral",
        use_pca=False,  
        flat_hand_mean = True,
        ext='pkl')

smplh10 = {'male': smplh_model_male, 'female': smplh_model_female, 'neutral': smplh_model_neutral}

######################################## smplx 10 ########################################
smplx_model_male = smplx.create(MODEL_PATH, model_type='smplx',
        gender = 'male',
        use_pca=False,
        ext='pkl')

smplx_model_female = smplx.create(MODEL_PATH, model_type='smplx',
        gender="female",
        use_pca=False,
        ext='pkl')

smplx_model_neutral = smplx.create(MODEL_PATH, model_type='smplx',
        gender="neutral",
        use_pca=False,
        ext='pkl')

smplx10 = {'male': smplx_model_male, 'female': smplx_model_female, 'neutral': smplx_model_neutral}

######################################## smplx 10 pca 12 ########################################
smplx12_model_male = smplx.create(MODEL_PATH, model_type='smplx',
        gender="male",
        num_pca_comps=12,
        use_pca=True,
        flat_hand_mean = True,
        ext='pkl')

smplx12_model_female = smplx.create(MODEL_PATH, model_type='smplx',
        gender="female",
        num_pca_comps=12,
        use_pca=True,
        flat_hand_mean = True,
        ext='pkl')
smplx12_model_neutral = smplx.create(MODEL_PATH, model_type='smplx',
        gender="neutral",
        num_pca_comps=12,
        use_pca=True,
        flat_hand_mean = True,
        ext='pkl')
smplx12 = {'male': smplx12_model_male, 'female': smplx12_model_female, 'neutral': smplx12_model_neutral}

######################################## smplh 16 ########################################
SMPLH_PATH = MODEL_PATH+'/smplh'
surface_model_male_fname = os.path.join(SMPLH_PATH,'female', "model.npz")
surface_model_female_fname = os.path.join(SMPLH_PATH, "male","model.npz")
surface_model_neutral_fname = os.path.join(SMPLH_PATH, "neutral", "model.npz")
dmpl_fname = None
num_dmpls = None 
num_expressions = None
num_betas = 16 

smplh16_model_male = BodyModel(bm_fname=surface_model_male_fname,
        num_betas=num_betas,
        num_expressions=num_expressions,
        num_dmpls=num_dmpls,
        dmpl_fname=dmpl_fname)
smplh16_model_female = BodyModel(bm_fname=surface_model_female_fname,
        num_betas=num_betas,
        num_expressions=num_expressions,
        num_dmpls=num_dmpls,
        dmpl_fname=dmpl_fname)
smplh16_model_neutral = BodyModel(bm_fname=surface_model_neutral_fname,
        num_betas=num_betas,
        num_expressions=num_expressions,
        num_dmpls=num_dmpls,
        dmpl_fname=dmpl_fname)
smplh16 = {'male': smplh16_model_male, 'female': smplh16_model_female, 'neutral': smplh16_model_neutral}

######################################## smplx 16 ########################################
SMPLX_PATH = MODEL_PATH+'/smplx'
surface_model_male_fname = os.path.join(SMPLX_PATH,"SMPLX_MALE.npz")
surface_model_female_fname = os.path.join(SMPLX_PATH,"SMPLX_FEMALE.npz")
surface_model_neutral_fname = os.path.join(SMPLX_PATH, "SMPLX_NEUTRAL.npz")

smplx16_model_male = BodyModel(bm_fname=surface_model_male_fname,
        num_betas=num_betas,
        num_expressions=num_expressions,
        num_dmpls=num_dmpls,
        dmpl_fname=dmpl_fname)
smplx16_model_female = BodyModel(bm_fname=surface_model_female_fname,
        num_betas=num_betas,
        num_expressions=num_expressions,
        num_dmpls=num_dmpls,
        dmpl_fname=dmpl_fname)
smplx16_model_neutral = BodyModel(bm_fname=surface_model_neutral_fname,
        num_betas=num_betas,
        num_expressions=num_expressions,
        num_dmpls=num_dmpls,
        dmpl_fname=dmpl_fname)
smplx16 = {'male': smplx16_model_male, 'female': smplx16_model_female, 'neutral': smplx16_model_neutral}

def visualize_smpl(name, MOTION_PATH, model_type, num_betas, use_pca=False, use_fixed=False):
    """
    Load and visualize SMPL data for a single sequence
    """
    # Choose between original and fixed pose file
    if use_fixed:
        human_file = 'joint_fixed.npz'
    else:
        human_file = 'human.npz'
    if use_fixed:
        # with np.load(os.path.join('save_pipeline_fix', 'omomo', name + '_step2.5_smoothed.npz'), allow_pickle=True) as f:
        with np.load(os.path.join('save_pipeline_fix', 'omomo', name + '_step2_palm_fixed.npz'), allow_pickle=True) as f:
        # with np.load(os.path.join('save_pipeline_fix', 'omomo', name + '_step1_joint_fixed.npz'), allow_pickle=True) as f:
            poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])
    else:
        with np.load(os.path.join(MOTION_PATH, name, human_file), allow_pickle=True) as f:
            poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])
    
    print(f"Motion loaded: {os.path.join(MOTION_PATH, name, human_file)}")
    frame_times = poses.shape[0]
    
    if num_betas == 10:
        if model_type == 'smplh':
            smpl_model = smplh10[gender]
            smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                    global_orient=torch.from_numpy(poses[:, :3]).float(),
                    left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
                    right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
                    betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
                    transl=torch.from_numpy(trans).float(),) 
        elif model_type == 'smplx':
            if use_pca:
                smpl_model = smplx12[gender]
                smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                        global_orient=torch.from_numpy(poses[:, :3]).float(),
                        left_hand_pose=torch.from_numpy(poses[:, 66:78]).float(),
                        right_hand_pose=torch.from_numpy(poses[:, 78:90]).float(),
                        jaw_pose=torch.zeros(frame_times, 3).float(),
                        leye_pose=torch.zeros(frame_times, 3).float(),
                        reye_pose=torch.zeros(frame_times, 3).float(),
                        expression=torch.zeros(frame_times, 10).float(),
                        betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
                        transl=torch.from_numpy(trans).float(),)
            else:
                smpl_model = smplx10[gender]
                smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                        global_orient=torch.from_numpy(poses[:, :3]).float(),
                        left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
                        right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
                        jaw_pose = torch.zeros([frame_times,3]).float(),
                        reye_pose = torch.zeros([frame_times,3]).float(),
                        leye_pose = torch.zeros([frame_times,3]).float(),
                        expression = torch.zeros([frame_times,10]).float(),
                        betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
                        transl=torch.from_numpy(trans).float(),)
        verts = to_cpu(smplx_output.vertices)
        joints= to_cpu(smplx_output.joints)
        faces = smpl_model.faces.astype(np.int32)
        faces = torch.from_numpy(faces).unsqueeze(0).repeat(frame_times, 1, 1)
    elif num_betas == 16: 
        if model_type == 'smplh':
            smpl_model = smplh16[gender]
        elif model_type == 'smplx':
            smpl_model = smplx16[gender]
        smplx_output = smpl_model(pose_body=torch.from_numpy(poses[:, 3:66]).float(), 
                pose_hand=torch.from_numpy(poses[:, 66:156]).float(), 
                betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(), 
                root_orient=torch.from_numpy(poses[:, :3]).float(), 
                trans=torch.from_numpy(trans).float())
        
        verts = to_cpu(smplx_output.v)
        joints= to_cpu(smplx_output.Jtr)
        faces = smpl_model.f.unsqueeze(0).repeat(frame_times, 1, 1)

    return verts, joints, faces, poses, betas, trans, gender

def regen_smpl(name, poses, betas, trans, gender, model_type, num_betas, use_pca=False):
    """Regenerate SMPL with fixed poses"""
    frame_times = poses.shape[0]
    
    if num_betas == 10:
        if model_type == 'smplh':
            smpl_model = smplh10[gender]
        elif model_type == 'smplx':
            if use_pca:
                smpl_model = smplx12[gender]
            else:
                smpl_model = smplx10[gender]
        
        smplx_output = smpl_model(
            body_pose=torch.from_numpy(poses[:, 3:66]).float(),
            global_orient=torch.from_numpy(poses[:, :3]).float(),
            left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
            right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
            betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
            transl=torch.from_numpy(trans).float()
        )
        verts = to_cpu(smplx_output.vertices)
        joints = to_cpu(smplx_output.joints)
        faces = smpl_model.faces.astype(np.int32)
        faces = torch.from_numpy(faces).unsqueeze(0).repeat(frame_times, 1, 1)
    elif num_betas == 16:
        if model_type == 'smplh':
            smpl_model = smplh16[gender]
        elif model_type == 'smplx':
            smpl_model = smplx16[gender]
        
        smplx_output = smpl_model(
            pose_body=torch.from_numpy(poses[:, 3:66]).float(),
            pose_hand=torch.from_numpy(poses[:, 66:156]).float(),
            betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
            root_orient=torch.from_numpy(poses[:, :3]).float(),
            trans=torch.from_numpy(trans).float()
        )
        verts = to_cpu(smplx_output.v)
        joints = to_cpu(smplx_output.Jtr)
        faces = smpl_model.f.unsqueeze(0).repeat(frame_times, 1, 1)

    return verts, joints, faces

# poses: (T, N, 3) axis-angle (Rodrigues) per joint per frame
# returns:
#   diff_angle: (T-1, N) radians in [0, pi]
#   diff_axis:  (T-1, N, 3) unit axes
def axis_angle_to_quat(aa):
    # aa: (..., 3) axis-angle, angle = ||aa||
    angle = torch.linalg.norm(aa, dim=-1, keepdims=True).clamp_min(1e-8)
    axis  = aa / angle
    half  = 0.5 * angle
    sin_h = torch.sin(half)
    cos_h = torch.cos(half)
    # quaternion layout: (x, y, z, w)
    return torch.cat([axis * sin_h, cos_h], dim=-1)

def quat_conjugate(q):
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)

def quat_mul(a, b):
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    x = aw*bx + ax*bw + ay*bz - az*by
    y = aw*by - ax*bz + ay*bw + az*bx
    z = aw*bz + ax*by - ay*bx + az*bw
    w = aw*bw - ax*bx - ay*by - az*bz
    return torch.stack([x,y,z,w], dim=-1)

def quat_normalize(q, eps=1e-8):
    return q / (q.norm(dim=-1, keepdim=True) + eps)

def quat_to_angle_axis(q, eps=1e-8):
    q = quat_normalize(q)
    w = q[..., 3].clamp(-1.0, 1.0)
    angle = 2.0 * torch.acos(w)                      # [0, pi]
    sin_half = torch.sqrt((1.0 - w*w).clamp_min(0))  # = ||xyz||
    axis = torch.zeros_like(q[..., :3])
    mask = sin_half > 1e-5
    axis[mask] = q[..., :3][mask] / sin_half[mask].unsqueeze(-1)
    axis[~mask] = torch.tensor([0.,0.,1.], device=q.device, dtype=q.dtype)  # default
    return angle, axis

def quat_from_axis_angle(axis, angle, eps=1e-8):
    if not isinstance(axis, torch.Tensor): axis = torch.tensor(axis, dtype=torch.float32)
    if not isinstance(angle, torch.Tensor): angle = torch.tensor(angle, dtype=torch.float32)
    # normalize axis defensively
    axis = axis / (axis.norm(dim=-1, keepdim=True) + eps)
    half = 0.5 * angle
    s = torch.sin(half)[..., None]
    c = torch.cos(half)[..., None]
    return torch.cat([axis * s, c], dim=-1)
    
def pose_delta_axis_angle(poses):
    # poses: (T, N, 3)
    T, N, _ = poses.shape
    q = axis_angle_to_quat(poses)                # (T, N, 4)
    q_prev = q[:-1]                               # (T-1, N, 4)
    q_curr = q[1:]
    q_rel  = quat_mul(quat_conjugate(q_prev), q_curr)
    q_rel  = quat_normalize(q_rel)
    diff_angle, diff_axis = quat_to_angle_axis(q_rel)  # (T-1, N), (T-1, N, 3)
    return diff_angle, diff_axis

# ---------- your two functions ----------
def reverse_rotate(pose_t1_aa, diff_angle, diff_axis):
    """
    Rotate pose at t+1 back to pose at t. Returns axis-angle vector (...,3).
    """
    q_t1   = quat_normalize(axis_angle_to_quat(pose_t1_aa))
    q_corr = quat_from_axis_angle(diff_axis, -diff_angle)   # inverse of delta
    q_t    = quat_normalize(quat_mul(q_t1, q_corr))
    ang, ax = quat_to_angle_axis(q_t)                       # tensors
    return ax * ang.unsqueeze(-1)                           # axis-angle vector

def forward_rotate(pose_t_aa, diff_angle, diff_axis):
    """
    Rotate pose at t forward to pose at t+1. Returns axis-angle vector (...,3).
    """
    q_t    = quat_normalize(axis_angle_to_quat(pose_t_aa))
    q_corr = quat_from_axis_angle(diff_axis,  diff_angle)   # apply delta
    q_t1   = quat_normalize(quat_mul(q_t, q_corr))
    ang, ax = quat_to_angle_axis(q_t1)
    return ax * ang.unsqueeze(-1)

def fix_joint_poses_simple_soft(
    poses,
    joint_idx,
    angle_thresh=0.4,
    max_passes=3,
    soft_margin=0.2,             # radians above the threshold to reach full correction
    use_smoothstep=True          # True: smoothstep; False: linear ramp
):
    """
    Soft-threshold version:
      - If diff <= angle_thresh: no change (alpha=0)
      - If diff >= angle_thresh + soft_margin: full correction (alpha=1)
      - Else: partial correction with alpha in (0,1)

    poses: (T, 156) or (T, 52, 3) axis-angle.  Assumes segment 0 is good.
    Returns: poses_fixed (same shape), fixed_boundaries (list of t indices used)
    """

    # ---- reshape to (T, 52, 3) ----
    if poses.ndim == 2 and poses.shape[1] == 156:
        if isinstance(poses, torch.Tensor):
            poses_reshaped = poses.reshape(poses.shape[0], 52, 3).clone()
        else:
            poses_reshaped = poses.reshape(poses.shape[0], 52, 3).copy()
        orig_flat = True
    elif poses.ndim == 3 and poses.shape[1:] == (52, 3):
        if isinstance(poses, torch.Tensor):
            poses_reshaped = poses.clone()
        else:
            poses_reshaped = poses.copy()
        orig_flat = False
    else:
        raise ValueError("poses must be (T,156) or (T,52,3) axis-angle")

    # work tensor
    if not isinstance(poses_reshaped, torch.Tensor):
        poses_t = torch.tensor(poses_reshaped, dtype=torch.float32)
    else:
        poses_t = poses_reshaped.float()

    T = poses_t.shape[0]
    fixed_boundaries = []

    def _soft_alpha(angle_b):
        """Compute correction fraction alpha in [0,1] from a scalar/tensor angle_b."""
        over = angle_b - angle_thresh
        # linear ramp in [0, soft_margin], clamped to [0,1]
        t = torch.clamp(over / soft_margin, min=0.0, max=1.0)
        if use_smoothstep:
            # smoothstep(t) = 3t^2 - 2t^3 : C1-continuous, gentler near 0/1
            t = t * t * (3.0 - 2.0 * t)
        return t

    for _ in range(max_passes):
        diff_angle, diff_axis = pose_delta_axis_angle(poses_t)   # (T-1, N), (T-1, N, 3)

        da = diff_angle[:, joint_idx]    # (T-1,)
        ax = diff_axis[:, joint_idx, :]  # (T-1,3)

        # boundaries above *hard* threshold (we still scale how much we fix)
        flip_idxs = torch.nonzero(da > angle_thresh, as_tuple=False).flatten().tolist()
        if not flip_idxs:
            break

        # earliest boundary first
        b = flip_idxs[0]                      # boundary t=b -> t+1=b+1
        angle_b = da[b]                       # scalar tensor
        axis_b  = ax[b]                       # (3,)

        # compute soft correction factor
        alpha_b = _soft_alpha(angle_b)        # in [0,1], tensor
        if alpha_b.item() == 0.0:
            # nothing to do at this boundary; skip to next pass
            fixed_boundaries.append(int(b))
            continue

        # segment to correct: (b+1 .. next_flip) inclusive; else (b+1 .. T-1)
        next_b = next((k for k in flip_idxs[1:] if k > b), None)
        seg_start = b + 1
        seg_end   = next_b if next_b is not None else (T - 1)
        sl = slice(seg_start, seg_end + 1)

        # apply *partial* reverse rotation: use alpha_b * angle_b
        angle_corr = angle_b * alpha_b
        poses_t[sl, joint_idx, :] = reverse_rotate(
            poses_t[sl, joint_idx, :], angle_corr, axis_b
        )

        fixed_boundaries.append(int(b))
        # recompute next pass

    poses_fixed = poses_t
    # restore original shape
    if isinstance(poses, torch.Tensor):
        if orig_flat:
            poses_fixed = poses_fixed.reshape(T, 156)
    else:
        poses_fixed = poses_fixed.cpu().numpy()
        if orig_flat:
            poses_fixed = poses_fixed.reshape(T, 156)

    return poses_fixed, fixed_boundaries

def fix_joint_poses_simple(poses, joint_idx, angle_thresh=0.4, max_passes=3):
    """
    poses: (T, 156) or (T, 52, 3) axis-angle.  Assumes segment 0 is good.
    For any boundary (t -> t+1) with diff_angle > angle_thresh for this joint,
    take the segment after that boundary and reverse-rotate all its frames by that boundary's delta.
    After fixing one segment, recompute diffs and continue, up to max_passes.
    Returns: poses_fixed (same shape), fixed_boundaries (list of t indices used)
    """
    # reshape to (T, 52, 3)
    if poses.ndim == 2 and poses.shape[1] == 156:
        if isinstance(poses, torch.Tensor):
            poses_reshaped = poses.reshape(poses.shape[0], 52, 3).clone()
        else:
            poses_reshaped = poses.reshape(poses.shape[0], 52, 3).copy()
    elif poses.ndim == 3 and poses.shape[1:] == (52, 3):
        if isinstance(poses, torch.Tensor):
            poses_reshaped = poses.clone()
        else:
            poses_reshaped = poses.copy()
    else:
        raise ValueError("poses must be (T,156) or (T,52,3) axis-angle")

    torch_device = None
    np_input = not isinstance(poses_reshaped, torch.Tensor)
    if np_input:
        poses_t = torch.tensor(poses_reshaped, dtype=torch.float32)
    else:
        poses_t = poses_reshaped.float()
        torch_device = poses_t.device

    T = poses_t.shape[0]
    fixed_boundaries = []

    for _ in range(max_passes):
        diff_angle, diff_axis = pose_delta_axis_angle(poses_t)   # (T-1, N), (T-1, N, 3)
        # flips for this joint
        da = diff_angle[:, joint_idx]       # (T-1,)
        ax = diff_axis[:, joint_idx, :]     # (T-1,3)

        # find boundaries with big jumps
        flip_idxs = torch.nonzero(da > angle_thresh, as_tuple=False).flatten().tolist()
        if not flip_idxs:
            break

        # Always take the earliest boundary first, fix the segment after it.
        b = flip_idxs[0]                     # boundary between b (good) and b+1.. (bad)
        angle_b = da[b]
        axis_b  = ax[b]

        # reverse-rotate ALL frames from b+1 to the next flip (or to end)
        next_b = next((k for k in flip_idxs[1:] if k > b), None)
        seg_start = b + 1
        seg_end = (next_b if next_b is not None else (T-1))  # inclusive end boundary for rotations
        # we rotate frames seg_start..(T-1) for the target joint; per your spec, whole segment to the end of that segment
        target_slice = slice(seg_start, seg_end + 1)

        # apply reverse rotation to that joint across the segment
        poses_t[target_slice, joint_idx, :] = reverse_rotate(
            poses_t[target_slice, joint_idx, :], angle_b, axis_b
        )
        fixed_boundaries.append(int(b))
        # loop continues: recompute diffs and handle next earliest boundary (after update)

    poses_fixed = poses_t
    if np_input:
        poses_fixed = poses_fixed.cpu().numpy()
        # reshape back to original
        if poses.ndim == 2:
            poses_fixed = poses_fixed.reshape(T, 156)
    else:
        if poses.ndim == 2:
            poses_fixed = poses_fixed.reshape(T, 156)

    return poses_fixed, fixed_boundaries

def write_joint_diffs(
    diff_angle,
    diff_axis,
    filepath,
    joints=(13, 16, 18, 20, 14, 17, 19, 21),
    overwrite=True,
):
    """
    Write per-frame rotation deltas for selected joints to CSV.

    Args:
        diff_angle: (T-1, N) torch.Tensor or np.ndarray (radians)
        diff_axis:  (T-1, N, 3) torch.Tensor or np.ndarray (unit axes)
        filepath:   output CSV path (directories will be created)
        joints:     iterable of joint indices to export
        overwrite:  if False and file exists, append (no header)

    CSV columns:
        frame_start, frame_end, joint, diff_angle_rad, axis_x, axis_y, axis_z
        (Note: frame_start=t means the delta is from frame t -> t+1)
    """
    # --- to numpy ---
    if torch.is_tensor(diff_angle):
        da = diff_angle.detach().cpu().numpy()
    else:
        da = np.asarray(diff_angle)
    if torch.is_tensor(diff_axis):
        dx = diff_axis.detach().cpu().numpy()
    else:
        dx = np.asarray(diff_axis)

    # --- basic shape checks ---
    if da.ndim != 2 or dx.ndim != 3 or da.shape[0] != dx.shape[0] or da.shape[1] != dx.shape[1] or dx.shape[2] != 3:
        raise ValueError(
            f"Shape mismatch: expected diff_angle (T-1,N) and diff_axis (T-1,N,3), "
            f"got {da.shape} and {dx.shape}"
        )

    Tm1, N = da.shape
    bad = [j for j in joints if not (0 <= j < N)]
    if bad:
        raise ValueError(f"Joint indices out of range 0..{N-1}: {bad}")

    # --- ensure dir, open file ---
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    write_header = overwrite or (not os.path.exists(filepath))
    mode = "w" if overwrite else "a"

    with open(filepath, mode, newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["frame_start", "frame_end", "joint", "diff_angle_rad", "axis_x", "axis_y", "axis_z"])
        for t in range(Tm1):
            for j in joints:
                angle = float(da[t, j])
                ax, ay, az = dx[t, j]
                # write with stable precision
                w.writerow([t, t+1, j,
                            f"{angle:.9f}",
                            f"{float(ax):.9f}",
                            f"{float(ay):.9f}",
                            f"{float(az):.9f}"])

def main():
    parser = argparse.ArgumentParser(description="Visualize a single sequence from a dataset")
    parser.add_argument("--dataset_path", required=True, help="Path to the root of the dataset")
    parser.add_argument("--sequence_name", required=True, help="Name of the sequence to visualize")
    parser.add_argument("--use_fixed", action="store_true", help="Use palm_fixed.npz instead of human.npz")
    parser.add_argument("--threshold", required=True, help="Threshold for joint fix")
    args = parser.parse_args()
    threshold = args.threshold
    # change threshold to float
    threshold = float(threshold)
    # Derived paths
    human_path = os.path.join(args.dataset_path, 'sequences_canonical')
    object_path = os.path.join(args.dataset_path, 'objects')
    dataset_name = args.dataset_path.split('/')[-1]
    
    # Check if sequence exists
    sequence_path = os.path.join(human_path, args.sequence_name)
    if not os.path.exists(sequence_path):
        print(f"Error: Sequence {args.sequence_name} not found in {human_path}")
        return
    
    
    print(f"Visualizing sequence: {args.sequence_name}")
    print(f"Dataset: {dataset_name}")
    parents = []
    # Load human data based on dataset type
    if dataset_name.upper() == 'BEHAVE' or dataset_name.upper() == 'BEHAVE_CORRECT':
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(
            args.sequence_name, human_path, 'smplh', 10, use_fixed=args.use_fixed)
        parents = smplh10[gender].parents.tolist()
    elif dataset_name.upper() == 'NEURALDOME' or dataset_name.upper() == 'IMHD':
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(
            args.sequence_name, human_path, 'smplh', 16, use_fixed=args.use_fixed)
        parents = smplh16[gender].parents.tolist()
    elif dataset_name.upper() == 'CHAIRS':
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(
            args.sequence_name, human_path, 'smplx', 10, use_fixed=args.use_fixed)
        parents = smplx10[gender].parents.tolist()
    elif dataset_name.upper() == 'INTERCAP' or dataset_name.upper() == 'INTERCAP_CORRECT':
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(
            args.sequence_name, human_path, 'smplx', 10, True, use_fixed=args.use_fixed)
        parents = smplx12[gender].parents.tolist()
    elif dataset_name.upper() == 'OMOMO' or dataset_name.upper() == 'OMOMO_CORRECT':
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(
            args.sequence_name, human_path, 'smplx', 16, use_fixed=args.use_fixed)
        parents = smplx16[gender].kintree_table[0].long().tolist()
    else:
        print(f"Error: Unsupported dataset type: {dataset_name}")
        return
    print(poses.shape)

    # Load object data
    try:
        with np.load(os.path.join(human_path, args.sequence_name, 'object.npz'), allow_pickle=True) as f:
            obj_angles, obj_trans, obj_name = f['angles'], f['trans'], str(f['name'])
    except FileNotFoundError:
        print(f"Error: object.npz not found in {sequence_path}")
        return
    
    angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
    
    # Load object mesh
    obj_mesh_path = os.path.join(object_path, obj_name, obj_name+'.obj')
    
    OBJ_MESH = trimesh.load(obj_mesh_path)
    print(f"Object name: {obj_name}")
    
    ov = np.array(OBJ_MESH.vertices).astype(np.float32)
    object_faces = OBJ_MESH.faces.astype(np.int32)
    
    # Transform object vertices
    ov = torch.from_numpy(ov).float().to(device)
    rot = torch.tensor(angle_matrix).float().to(device)
    obj_trans_tensor = torch.tensor(obj_trans).float().to(device)
    object_verts = torch.einsum('ni,tij->tnj', ov, rot.permute(0,2,1)) + obj_trans_tensor.unsqueeze(1)
    suffix = "_simple_fixed_" + str(threshold)
    output_filename = f"{args.sequence_name}{suffix}.mp4"
    output_dir = os.path.join("simple_fix", args.sequence_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    if isinstance(poses, np.ndarray):
        poses = torch.from_numpy(poses).float()  # now a torch.Tensor
    # view poses from (N, 156) to (N, 52, 3)
    poses = poses.reshape(-1, 52, 3)
    diff_angle, diff_axis = pose_delta_axis_angle(poses)
    write_joint_diffs(diff_angle, diff_axis, os.path.join(output_dir, f"{args.sequence_name}_{threshold}_before_fix.csv"))
    joints_to_fix = []
    # find the joints with any diff_angle > 0.4
    
    for j in [13, 16, 18, 20, 14, 17, 19, 21]:
        for i in range(diff_angle.shape[0]):
            if abs(diff_angle[i, j]) > threshold:
                joints_to_fix.append(j)
                break
    print(f"Joints to fix: {joints_to_fix}")

    for joint in joints_to_fix:
        poses, fixed_boundaries = fix_joint_poses_simple(poses, joint, angle_thresh=threshold, max_passes=50)
        # poses, fixed_boundaries = fix_joint_poses_simple_global(poses, joint, parents, angle_thresh=threshold, max_passes=50, strength=0.1)
        print(f"Fixed joints {joint} at {fixed_boundaries}")
    fixed_poses = poses
    # convert fixed_poses to np ndarray
    print(f"Fixed poses shape: {fixed_poses.shape}")
    diff_angle, diff_axis = pose_delta_axis_angle(fixed_poses)
    write_joint_diffs(diff_angle, diff_axis, os.path.join(output_dir, f"{args.sequence_name}_{threshold}_after_fix.csv"))
    fixed_poses = fixed_poses.reshape(-1, 156)
    fixed_poses = fixed_poses.cpu().numpy()
    if dataset_name.upper() == 'BEHAVE' or dataset_name.upper() == 'BEHAVE_CORRECT':
        verts, joints, faces = regen_smpl(args.sequence_name, fixed_poses, betas, trans, gender, 'smplh', 10)
    elif dataset_name.upper() == 'NEURALDOME' or dataset_name.upper() == 'IMHD':
        verts, joints, faces = regen_smpl(args.sequence_name, fixed_poses, betas, trans, gender, 'smplh', 16)
    elif dataset_name.upper() == 'CHAIRS':
        verts, joints, faces = regen_smpl(args.sequence_name, fixed_poses, betas, trans, gender, 'smplx', 10)
    elif dataset_name.upper() == 'INTERCAP' or dataset_name.upper() == 'INTERCAP_CORRECT':
        verts, joints, faces = regen_smpl(args.sequence_name, fixed_poses, betas, trans, gender, 'smplx', 10, True)
    elif dataset_name.upper() == 'OMOMO' or dataset_name.upper() == 'OMOMO_CORRECT':
        verts, joints, faces = regen_smpl(args.sequence_name, fixed_poses, betas, trans, gender, 'smplx', 16)

    visualize_body_obj(
        verts.float().detach().cpu().numpy(),
        faces[0].detach().cpu().numpy().astype(np.int32),
        object_verts.detach().cpu().numpy(),
        object_faces,
        save_path=output_path,
        show_frame=True,
        multi_angle=True,
    )
if __name__ == "__main__":
    main() 