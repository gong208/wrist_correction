import numpy as np
import argparse
import os
import torch
from scipy.spatial.transform import Rotation as R, Slerp
import math
import smplx
from human_body_prior.body_model.body_model import BodyModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu()
from render.mesh_viz import visualize_body_obj

MODEL_PATH = 'models'

######################################## smplh 10 ########################################
smplh_model_male = smplx.create(MODEL_PATH, model_type='smplh',
        gender="male",
        use_pca=False,
        ext='pkl')

smplh_model_female = smplx.create(MODEL_PATH, model_type='smplh',
        gender="female",
        use_pca=False,
        ext='pkl')

smplh_model_neutral = smplx.create(MODEL_PATH, model_type='smplh',
        gender="neutral",
        use_pca=False,
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


# Joint indices for SMPLX/SMPLH (based on bone_lists.py)
LEFT_COLLAR = 13
RIGHT_COLLAR = 14
LEFT_SHOULDER = 16
RIGHT_SHOULDER = 17
LEFT_ELBOW = 18
RIGHT_ELBOW = 19
LEFT_WRIST = 20
RIGHT_WRIST = 21

# Joint mapping to pose parameters (for SMPLX/SMPLH)
# Pose parameters start from index 3 (after global orientation 0:3)
# Each joint has 3 pose parameters
# Based on the actual mapping used in print_poses.py
JOINT_TO_POSE_MAPPING = {
    LEFT_COLLAR: 39,    # pose indices 39:42 (joint 13)
    RIGHT_COLLAR: 42,   # pose indices 42:45 (joint 14)
    LEFT_SHOULDER: 48,  # pose indices 48:51 (joint 16)
    RIGHT_SHOULDER: 51, # pose indices 51:54 (joint 17)
    LEFT_ELBOW: 54,     # pose indices 54:57 (joint 18)
    RIGHT_ELBOW: 57,    # pose indices 57:60 (joint 19)
    LEFT_WRIST: 60,     # pose indices 60:63 (joint 20)
    RIGHT_WRIST: 63     # pose indices 63:66 (joint 21)
}


def interpolate_pose(p1, p2, alpha):
    """Spherical interpolation between two axis-angle rotations."""
    key_times = [0, 1]
    key_rots = R.from_rotvec([p1, p2])
    slerp = Slerp(key_times, key_rots)
    return slerp(alpha).as_rotvec()

def rotate_pose_around_axis(pose_vec, axis, angle_deg):
    R_current = R.from_rotvec(pose_vec).as_matrix()
    R_correction = R.from_rotvec(np.deg2rad(angle_deg) * axis).as_matrix()
    R_fixed = R_current @ R_correction
    return R.from_matrix(R_fixed).as_rotvec()

def regen_smpl(name, poses, betas, trans, gender, model_type, num_betas, use_pca=False):
    """
    BEHAVE for SMPLH 10
    NEURALDOME or IMHD for SMPLH 16
    vertices: (N, 6890, 3)
    Chairs for SMPLX 10
    InterCap for SMPLX 10 PCA 12
    OMOMO for SMPLX 16
    vertices: (N, 10475, 3)
    """

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

    return verts, joints,faces, poses

def get_mean_pose_joints(name, gender, model_type, num_betas, use_pca=False):
    """
    Return: joints_np: (55, 3) numpy array in canonical (zero-pose) space
    """

    # with np.load(os.path.join(MOTION_PATH, name, 'human.npz'), allow_pickle=True) as f:
    #     _, _, _, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])
    # print(f"motion loaded: {os.path.join(MOTION_PATH, name, 'human.npz')}")

    frame_times = 1  # only one frame needed
    pose_zeros = torch.zeros(frame_times, 156).float()  # works for SMPLX or SMPLH

    trans_zeros = torch.zeros(frame_times, 3).float()
    betas_zeros = torch.zeros(frame_times, num_betas).float()

    if num_betas == 10:
        if model_type == 'smplh':
            smpl_model = smplh10[gender]
            output = smpl_model(body_pose=pose_zeros[:, 3:66],
                                global_orient=pose_zeros[:, :3],
                                left_hand_pose=pose_zeros[:, 66:111],
                                right_hand_pose=pose_zeros[:, 111:156],
                                transl=trans_zeros,
                                betas=betas_zeros)
        elif model_type == 'smplx':
            if use_pca:
                smpl_model = smplx12[gender]
                output = smpl_model(body_pose=pose_zeros[:, 3:66],
                                    global_orient=pose_zeros[:, :3],
                                    left_hand_pose=pose_zeros[:, 66:78],
                                    right_hand_pose=pose_zeros[:, 78:90],
                                    jaw_pose=torch.zeros(frame_times, 3),
                                    leye_pose=torch.zeros(frame_times, 3),
                                    reye_pose=torch.zeros(frame_times, 3),
                                    expression=torch.zeros(frame_times, 10),
                                    transl=trans_zeros,
                                    betas=betas_zeros)
            else:
                smpl_model = smplx10[gender]
                output = smpl_model(body_pose=pose_zeros[:, 3:66],
                                    global_orient=pose_zeros[:, :3],
                                    left_hand_pose=pose_zeros[:, 66:111],
                                    right_hand_pose=pose_zeros[:, 111:156],
                                    jaw_pose=torch.zeros(frame_times, 3),
                                    leye_pose=torch.zeros(frame_times, 3),
                                    reye_pose=torch.zeros(frame_times, 3),
                                    expression=torch.zeros(frame_times, 10),
                                    transl=trans_zeros,
                                    betas=betas_zeros)
        joints = output.joints[0].detach().cpu().numpy()  # (55, 3)
    else:
        if model_type == 'smplh':
            smpl_model = smplh16[gender]
        elif model_type == 'smplx':
            smpl_model = smplx16[gender]
        output = smpl_model(pose_body=pose_zeros[:, 3:66],
                            pose_hand=pose_zeros[:, 66:156],
                            root_orient=pose_zeros[:, :3],
                            trans=trans_zeros,
                            betas=betas_zeros)
        joints = output.Jtr[0].detach().cpu().numpy()

    return joints

def compute_palm_contact_and_orientation(
    human_joints: torch.Tensor,       # (T, J, 3)
    object_verts: torch.Tensor,       # (T, N, 3)
    hand: str = 'right',              # 'left' 或 'right'
    contact_thresh: float = 0.09,     # 接触距离阈值（单位 m）
    orient_angle_thresh: float = 75.0,# 朝向最大夹角阈值（度），90° 即半球面
    orient_dist_thresh: float = 0.09  # 朝向距离阈值（单位 m），用于筛选接触点

):
    """
    返回：
      contact_mask:  (T,) bool Tensor，是否有接触（任意顶点 distance < contact_thresh）
      orient_mask:   (T,) bool Tensor，仅当接触时，且存在顶点
                     在法线±orient_angle_thresh范围内且 distance < contact_thresh
      normals:       (T, 3)   Tensor，各帧手掌法向（归一化）
    """
    # 0) 预处理：同设备
    if human_joints.device != object_verts.device:
        human_joints = human_joints.to(object_verts.device)
    T, J, _ = human_joints.shape

    # 1) 选关节索引 & 法线翻转
    hand = hand.lower()
    if hand.startswith('r'):
        # 右手索引
        IDX_WRIST     = 21
        IDX_INDEX     = 40
        IDX_PINKY     = 46
        flip_normal   = False
    else:
        # 左手索引
        IDX_WRIST     = 20
        IDX_INDEX     = 25
        IDX_PINKY     = 31
        flip_normal   = True

    # 2) 提取关节位置
    wrist = human_joints[:, IDX_WRIST    , :]  # (T,3)
    idx   = human_joints[:, IDX_INDEX    , :]  # (T,3)
    pinky = human_joints[:, IDX_PINKY    , :]  # (T,3)

    # 3) 计算法线 & 归一化
    v1 = idx   - wrist   # (T,3)
    v2 = pinky - wrist   # (T,3)
    normals = torch.cross(v1, v2, dim=1)  # (T,3)
    if flip_normal:
        normals = -normals
    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-8)

    # 4) 计算手掌质心
    centroid = (wrist + idx + pinky) / 3.0  # (T,3)

    # 5) 计算所有顶点相对向量 & 距离
    #    object_verts: (T, N, 3)
    rel = object_verts - centroid.unsqueeze(1)   # (T, N, 3)
    dists = rel.norm(dim=2)                     # (T, N)

    # 6) contact_mask：任意顶点 distance < contact_thresh
    contact_mask = (dists < contact_thresh).any(dim=1)  # (T,)

    # 7) orient_mask：存在顶点既满足 distance < contact_thresh
    #    又满足夹角 ≤ orient_angle_thresh
    #    cos_thresh = cos(orient_angle_thresh)
    cos_thresh = torch.cos(torch.deg2rad(torch.tensor(orient_angle_thresh, device=normals.device)))

    # 7.1) 先归一化 rel 向量
    rel_dir = rel / (dists.unsqueeze(-1) + 1e-8)       # (T, N, 3)
    # 7.2) 计算余弦：normals.unsqueeze(1) 与 rel_dir 点积
    cosines = (normals.unsqueeze(1) * rel_dir).sum(dim=2)  # (T, N)
    # 7.3) 筛选：cosines >= cos_thresh 且 dists < contact_thresh
    mask = (cosines >= cos_thresh) & (dists < orient_dist_thresh)  # (T, N)
    orient_mask = mask.any(dim=1)  # (T,)

    return contact_mask, orient_mask, normals

def visualize_smpl(name, MOTION_PATH, model_type, num_betas, use_pca=False):
    """
    BEHAVE for SMPLH 10
    NEURALDOME or IMHD for SMPLH 16
    vertices: (N, 6890, 3)
    Chairs for SMPLX 10
    InterCap for SMPLX 10 PCA 12
    OMOMO for SMPLX 16
    vertices: (N, 10475, 3)
    """
    with np.load(os.path.join(MOTION_PATH, name, 'human.npz'), allow_pickle=True) as f:
        poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])
    print(f"motion loaded: {os.path.join(MOTION_PATH, name, 'human.npz')}")
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

    return verts, joints,faces, poses, betas, trans, gender

def calculate_axis_angle_difference(pose1, pose2):
    """
    Calculate the difference between two rotations using rotation matrices.
    This is often more stable than axis-angle comparisons.
    
    Args:
        pose1 (np.array): First pose vector (3D rotation vector)
        pose2 (np.array): Second pose vector (3D rotation vector)
    
    Returns:
        tuple: (angle_diff_degrees, relative_rotation_axis)
    """
    from scipy.spatial.transform import Rotation
    
    # Convert axis-angle to rotation objects
    rot1 = Rotation.from_rotvec(pose1)
    rot2 = Rotation.from_rotvec(pose2)
    # Calculate the relative rotation using matrix multiplication
    relative_rot = rot1.inv()* rot2
    
    # Get the rotation angle in degrees
    angle_diff = np.rad2deg(relative_rot.magnitude())
    
    # Get the rotation axis
    relative_axis = relative_rot.as_rotvec()
    if np.linalg.norm(relative_axis) > 1e-6:
        relative_axis = relative_axis / np.linalg.norm(relative_axis)
    else:
        relative_axis = np.array([1.0, 0.0, 0.0])  # fallback
    
    return angle_diff, relative_axis

def detect_flips(poses, joint_idx, threshold=20):
    """
    Detect flips (large axis-angle changes) for a joint.
    
    Args:
        poses: (T, 156) pose parameters
        joint_idx: joint index to analyze
        threshold: angle threshold in degrees
    
    Returns:
        tuple: (flip_indices, angle_diffs, relative_axes)
    """
    T = poses.shape[0]
    pose_start = JOINT_TO_POSE_MAPPING[joint_idx]
    flip_indices = []
    angle_diffs = []
    relative_axes = []
    
    for t in range(T-1):
        pose1 = poses[t, pose_start:pose_start+3]
        pose2 = poses[t+1, pose_start:pose_start+3]
        
        # Calculate angle difference and relative rotation axis
        angle_diff, relative_axis = calculate_axis_angle_difference(pose1, pose2)
        angle_diffs.append(angle_diff)
        relative_axes.append(relative_axis)
        # Check if this is a flip (large angle change)
        if angle_diff > threshold:
            flip_indices.append(t)
    
    return flip_indices, angle_diffs, relative_axes

def fix_joint_poses(poses, flip_indices, orient_mask, joint_idx, angle_diffs, relative_axes, canonical_joints, threshold=20, left_twist_bound_mask=None, right_twist_bound_mask=None):
    """
    Fix joint poses based on flip detection and orientation mask.
    Process flips sequentially and check if segments have already been fixed.
    
    Args:
        poses: (T, 156) pose parameters to fix
        flip_indices: list of frame indices where flips occur
        orient_mask: (T,) bool tensor, True when orientation is correct
        joint_idx: joint index to fix
        angle_diffs: list of angle differences between consecutive frames
        relative_axes: list of relative rotation axes between consecutive frames
        canonical_joints: (J, 3) canonical joint positions in rest pose
        threshold: angle threshold in degrees
        left_twist_bound_mask: list of bool indicating left wrist out-of-bounds frames (optional)
        right_twist_bound_mask: list of bool indicating right wrist out-of-bounds frames (optional)
    
    Returns:
        np.array: fixed poses
    """
    T = poses.shape[0]
    pose_start = JOINT_TO_POSE_MAPPING[joint_idx]
    fixed_poses = poses.copy()
    
    if not flip_indices:
        return fixed_poses
    
    print(f"Fixing joint {joint_idx} with {len(flip_indices)} flips detected")
    
    # Track which frames have been fixed to avoid repeated fixing
    fixed = np.zeros(T, dtype=bool)
    
    # Process flips sequentially
    for i, flip_idx in enumerate(flip_indices):
        print(f"  Processing flip {i+1}/{len(flip_indices)} at frame {flip_idx}")
        
        # For each flip, consider the segments that would be created if this flip were processed in isolation
        # This means we need to find the previous and next flips to define the segments
        
        # Find the previous flip (if any)
        prev_flip_idx = None
        if i > 0:
            prev_flip_idx = flip_indices[i-1]
        
        # Find the next flip (if any)
        next_flip_idx = None
        if i + 1 < len(flip_indices):
            next_flip_idx = flip_indices[i + 1]
        
        # Define segments for this flip
        # Left segment: from previous flip (or start) to current flip
        left_start = 0 if prev_flip_idx is None else prev_flip_idx + 1
        left_end = flip_idx
        
        # Right segment: from current flip to next flip (or end)
        right_start = flip_idx + 1
        right_end = T - 1 if next_flip_idx is None else next_flip_idx
        
        # Special handling for consecutive flips
        # If this flip is consecutive with the next flip, adjust the right segment
        if next_flip_idx is not None and next_flip_idx == flip_idx + 1:
            # For consecutive flips, the right segment should go beyond the consecutive flip
            # to include the segment that would be affected by both flips
            if i + 2 < len(flip_indices):
                # There's a flip after the consecutive one
                right_end = flip_indices[i + 2]
            else:
                # The consecutive flip is the last one
                right_end = T - 1
            print(f"    Consecutive flip detected: adjusting right segment to [{right_start},{right_end}]")
        
        # If this flip is consecutive with the previous flip, adjust the left segment
        if prev_flip_idx is not None and prev_flip_idx == flip_idx - 1:
            # For consecutive flips, the left segment should go before the consecutive flip
            if i > 1:
                # There's a flip before the consecutive one
                left_start = flip_indices[i - 2] + 1
            else:
                # The consecutive flip is the first one
                left_start = 0
            print(f"    Consecutive flip detected: adjusting left segment to [{left_start},{left_end}]")
        
        print(f"    Segments for flip {flip_idx}: left=[{left_start},{left_end}], right=[{right_start},{right_end}]")
        
        # Calculate orientation proportion for each segment
        left_orient_ratio = orient_mask[left_start:left_end+1].float().mean().item() if left_end >= left_start else 0.0
        right_orient_ratio = orient_mask[right_start:right_end+1].float().mean().item() if right_end >= right_start else 0.0
        
        print(f"    Orientation ratios: left={left_orient_ratio:.3f}, right={right_orient_ratio:.3f}")
        
        # Check if both segments have 0 orientation frames
        left_no_orient = left_orient_ratio == 0.0
        right_no_orient = right_orient_ratio == 0.0
        
        if left_no_orient and right_no_orient:
            print(f"    Both segments have 0 orientation frames - using twist angle bound analysis")
            # Use twist angle bound masks to determine which segment to fix
            # Use only the relevant bound mask for this joint
            if joint_idx in [LEFT_COLLAR, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST] and left_twist_bound_mask is not None:
                # For left wrist, use left bound mask for both segments
                left_oob_count = sum(left_twist_bound_mask[left_start:left_end+1]) if left_end >= left_start else 0
                right_oob_count = sum(left_twist_bound_mask[right_start:right_end+1]) if right_end >= right_start else 0
            elif joint_idx in [RIGHT_COLLAR, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST] and right_twist_bound_mask is not None:
                # For right wrist, use right bound mask for both segments
                left_oob_count = sum(right_twist_bound_mask[left_start:left_end+1]) if left_end >= left_start else 0
                right_oob_count = sum(right_twist_bound_mask[right_start:right_end+1]) if right_end >= right_start else 0
            else:
                # Fallback when bound masks not available
                left_oob_count = 0
                right_oob_count = 0
            
            print(f"    Left segment out-of-bounds: {left_oob_count}/{left_end-left_start+1}")
            print(f"    Right segment out-of-bounds: {right_oob_count}/{right_end-right_start+1}")
            
            # Fix the segment with more out-of-bounds frames
            if left_oob_count > right_oob_count:
                seg_start, seg_end = left_start, left_end
                jump_angle = angle_diffs[flip_idx]
                print(f"    Fixing LEFT segment [{seg_start},{seg_end}] by {jump_angle:.1f}° (more out-of-bounds)")
            elif right_oob_count > left_oob_count:
                seg_start, seg_end = right_start, right_end
                jump_angle = -angle_diffs[flip_idx]
                print(f"    Fixing RIGHT segment [{seg_start},{seg_end}] by {jump_angle:.1f}° (more out-of-bounds)")
            elif left_oob_count == right_oob_count:
                # Equal out-of-bounds, use segment length as tiebreaker
                left_length = left_end - left_start + 1
                right_length = right_end - right_start + 1
                if left_start == 0:
                    seg_start, seg_end = left_start, left_end
                    jump_angle = angle_diffs[flip_idx]
                    print(f"    Fixing LEFT segment [{seg_start},{seg_end}] by {jump_angle:.1f}° (longer segment)")
                elif left_length < right_length:
                    seg_start, seg_end = left_start, left_end
                    jump_angle = angle_diffs[flip_idx]
                    print(f"    Fixing LEFT segment [{seg_start},{seg_end}] by {jump_angle:.1f}° (longer segment)")
                else:
                    seg_start, seg_end = right_start, right_end
                    jump_angle = -angle_diffs[flip_idx]
                    print(f"    Fixing RIGHT segment [{seg_start},{seg_end}] by {jump_angle:.1f}° (longer segment)")

        else:
            # Determine which segment to fix (the one with lower orientation ratio)
            if left_orient_ratio < right_orient_ratio:
                # Fix left segment
                seg_start, seg_end = left_start, left_end
                jump_angle = angle_diffs[flip_idx]
                print(f"    Fixing LEFT segment [{seg_start},{seg_end}] by {jump_angle:.1f}°")
            else:
                # Fix right segment
                seg_start, seg_end = right_start, right_end
                jump_angle = -angle_diffs[flip_idx]
                print(f"    Fixing RIGHT segment [{seg_start},{seg_end}] by {jump_angle:.1f}°")
        
        # Check if the segment to be fixed is already mostly fixed
        fixed_ratio = fixed[np.arange(seg_start, seg_end + 1)].mean()
        if fixed_ratio > 0.8:  # If more than 80% is already fixed
            print(f"    Segment [{seg_start},{seg_end}] already {fixed_ratio:.1%} fixed - skipping")
            continue
        
        # Check segment length and determine fixing method
        segment_length = seg_end - seg_start + 1
        
        # Check if the segment to be fixed is between two flips (for interpolation)
        # If segment includes 0th frame or last frame, it's not between flips
        is_between_flips = False
        if seg_start > 0 and seg_end < T - 1:  # Segment doesn't include 0th or last frame
            is_between_flips = True
            # Find the flips before and after this segment
            prev_flip = prev_flip_idx
            next_flip = next_flip_idx
        
        # Determine fixing method based on segment length
        if segment_length <= 0 and is_between_flips:
            # Use interpolation for short segments between flips
            print(f"    Segment [{seg_start},{seg_end}] (length={segment_length}) is between flips {prev_flip} and {next_flip} - using interpolation")
            
            # Use interpolation between poses at the two flips
            pose_before = fixed_poses[seg_start - 1, pose_start:pose_start+3]
            pose_after = fixed_poses[seg_end + 1, pose_start:pose_start+3]
            
            for t in range(seg_start, seg_end + 1):
                # Calculate interpolation weight (0 at seg_start, 1 at seg_end)
                alpha = (t - seg_start) / (seg_end - seg_start) if seg_end > seg_start else 0.0
                fixed_poses[t, pose_start:pose_start+3] = interpolate_pose(pose_before, pose_after, alpha)
                fixed[t] = True
        else:
            # Use angle reversal for long segments or segments not between flips
            if segment_length > 10:
                print(f"    Segment [{seg_start},{seg_end}] (length={segment_length}) is too long - using angle reversal")
            else:
                print(f"    Segment [{seg_start},{seg_end}] (length={segment_length}) includes boundary frames - using angle reversal")
            
            # Use the relative rotation axis from the flip detection
            # This is the axis around which the flip occurred
            rotation_axis = relative_axes[flip_idx]
            print(f"    Using relative rotation axis from flip: [{rotation_axis[0]:.3f}, {rotation_axis[1]:.3f}, {rotation_axis[2]:.3f}]")
            
            # Use simple rotation correction (angle reversal)
            for t in range(seg_start, seg_end + 1):
                pose_vec = fixed_poses[t, pose_start:pose_start+3]
                fixed_poses[t, pose_start:pose_start+3] = rotate_pose_around_axis(
                    pose_vec, rotation_axis, jump_angle
                )
                fixed[t] = True
        
        print(f"    Marked frames [{seg_start},{seg_end}] as fixed")
    
    return fixed_poses




def detect_hand_twist_from_canonical_batch(poses, joints_canonical):
    """
    Detect wrist twist angles for all poses in parallel using canonical bone axis (rest pose).

    Args:
        poses: (T, 156) array of axis-angle poses for all frames
        joints_canonical: (55, 3) array of joint positions in rest pose

    Returns:
        twist_left_list, twist_right_list: lists of twist angles (in degrees) for each wrist
    """
    # Joint indices
    LEFT_ELBOW, LEFT_WRIST = 18, 20
    RIGHT_ELBOW, RIGHT_WRIST = 19, 21

    # Canonical bone axes from rest pose
    bone_axis_left = joints_canonical[LEFT_WRIST] - joints_canonical[LEFT_ELBOW]
    bone_axis_left /= np.linalg.norm(bone_axis_left)

    bone_axis_right = joints_canonical[RIGHT_WRIST] - joints_canonical[RIGHT_ELBOW]
    bone_axis_right /= np.linalg.norm(bone_axis_right)

    # Extract wrist poses for all frames
    T = poses.shape[0]
    left_wrist_poses = poses[:, LEFT_WRIST*3:(LEFT_WRIST+1)*3]  # (T, 3)
    right_wrist_poses = poses[:, RIGHT_WRIST*3:(RIGHT_WRIST+1)*3]  # (T, 3)

    # Fully vectorized computation of twist angles
    twist_left_list = compute_twist_angles_vectorized(left_wrist_poses, bone_axis_left)
    twist_right_list = compute_twist_angles_vectorized(right_wrist_poses, bone_axis_right)
    
    return twist_left_list, twist_right_list


def compute_twist_angles_vectorized(wrist_poses, bone_axis):
    """
    Compute twist angles for all frames in parallel.
    
    Args:
        wrist_poses: (T, 3) array of wrist poses for all frames
        bone_axis: (3,) array of bone axis (unit vector)
    
    Returns:
        twist_angles: (T,) array of twist angles in degrees
    """
    # Compute angles for all frames at once
    angles = np.linalg.norm(wrist_poses, axis=1)  # (T,)
    
    # Handle zero angles
    zero_mask = angles < 1e-6
    angles[zero_mask] = 1.0  # Avoid division by zero
    
    # Compute normalized axes for all frames
    axes = wrist_poses / angles[:, np.newaxis]  # (T, 3)
    
    # Compute twist cosines for all frames
    twist_cos = np.dot(axes, bone_axis)  # (T,)
    
    # Compute twist angles
    twist_angles = angles * twist_cos  # (T,)
    
    # Convert to degrees
    twist_angles_deg = np.rad2deg(twist_angles)
    
    # Set zero angles back to zero
    twist_angles_deg[zero_mask] = 0.0
    
    return twist_angles_deg.tolist()



def fix_all_joint_poses(poses, joints, object_verts, canonical_joints, threshold=20):
    """
    Fix poses for all specified joints (collar, shoulder, elbow, wrist).
    Iteratively fix all joints together until the total number of flips stops decreasing or becomes zero.
    
    Args:
        poses: (T, 156) pose parameters
        joints: (T, J, 3) joint positions for all frames
        object_verts: (T, N, 3) object vertices for all frames
        canonical_joints: (J, 3) canonical joint positions in rest pose
        threshold: angle threshold in degrees
    
    Returns:
        np.array: fixed poses
    """
    joints_to_fix = [
        LEFT_COLLAR, RIGHT_COLLAR,
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_WRIST, RIGHT_WRIST
    ]
    
    fixed_poses = poses.copy()
    
    # First, get orientation masks for both hands
    print("Computing palm contact and orientation masks...")
    contact_mask_l, orient_mask_l, _ = compute_palm_contact_and_orientation(
        joints, object_verts, hand='left'
    )
    
    contact_mask_r, orient_mask_r, _ = compute_palm_contact_and_orientation(
        joints, object_verts, hand='right'
    )
    
    print(f"Left hand: {contact_mask_l.sum().item()}/{len(contact_mask_l)} contact frames, {orient_mask_l.sum().item()}/{len(orient_mask_l)} correct orientation")
    print(f"Right hand: {contact_mask_r.sum().item()}/{len(contact_mask_r)} contact frames, {orient_mask_r.sum().item()}/{len(contact_mask_r)} correct orientation")
    
    # Compute wrist twist angles and create bound masks
    print("Computing wrist twist angles and bound masks...")
    twist_left_list, twist_right_list = detect_hand_twist_from_canonical_batch(poses, canonical_joints)
    
    # Create twist angle bound masks
    T = len(twist_left_list)
    left_twist_bound_mask = [(tw > 90 or tw < -110) for tw in twist_left_list]
    right_twist_bound_mask = [(tw > 110 or tw < -90) for tw in twist_right_list]
    
    print(f"Left twist out-of-bounds frames: {sum(left_twist_bound_mask)}/{T}")
    print(f"Right twist out-of-bounds frames: {sum(right_twist_bound_mask)}/{T}")
    
    # Iteratively fix all joints together
    iteration = 0
    prev_total_flips = float('inf')
    
    while True:
        iteration += 1
        print(f"\nIteration {iteration} - Processing all joints together...")
        
        # Track total flips across all joints
        total_flips = 0
        joint_flip_counts = {}
        
        # Process each joint in this iteration
        for joint_idx in joints_to_fix:
            # Determine which orientation mask to use based on the joint
            if joint_idx in [LEFT_COLLAR, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]:
                orient_mask = orient_mask_l
            else:  # RIGHT_COLLAR, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
                orient_mask = orient_mask_r
            
            # Detect flips for this joint
            flip_indices, angle_diffs, relative_axes = detect_flips(fixed_poses, joint_idx, threshold)
            current_flip_count = len(flip_indices)
            joint_flip_counts[joint_idx] = current_flip_count
            total_flips += current_flip_count
            
            if current_flip_count > 0:
                print(f"  Joint {joint_idx}: {current_flip_count} flips at frames: {flip_indices}")
                
                # Fix the poses for this joint
                fixed_poses = fix_joint_poses(
                    fixed_poses, flip_indices, orient_mask, 
                    joint_idx, angle_diffs, relative_axes, canonical_joints, threshold,
                    left_twist_bound_mask, right_twist_bound_mask
                )
            else:
                print(f"  Joint {joint_idx}: No flips detected")
        
        print(f"  Total flips across all joints: {total_flips}")
        
        # Check if we should stop iterating
        if total_flips == 0:
            print(f"  No flips remaining across all joints - stopping iterations")
            break
        elif total_flips >= prev_total_flips:
            print(f"  Total flip count did not decrease ({prev_total_flips} -> {total_flips}) - stopping iterations")
            break
        else:
            print(f"  Total flip count decreased from {prev_total_flips} to {total_flips} - continuing iterations")
        
        # Update for next iteration
        prev_total_flips = total_flips
        
        # Safety check to prevent infinite loops
        if iteration > 20:
            print(f"  Reached maximum iterations (20) - stopping")
            break
    
    print(f"\nCompleted {iteration} iterations across all joints")
    print("Final flip counts per joint:")
    for joint_idx in joints_to_fix:
        flip_indices, _, _ = detect_flips(fixed_poses, joint_idx, threshold)
        print(f"  Joint {joint_idx}: {len(flip_indices)} flips")
    
    return fixed_poses

def main(dataset_path, sequence_name, threshold=20):
    """
    Main function to fix joint poses for a sequence.
    
    Args:
        dataset_path: Path to the dataset
        sequence_name: Name of the sequence
        threshold: Angle threshold in degrees for flip detection
    """
    # Determine dataset type and load data
    dataset_type = dataset_path.split('/')[-1].upper()
    
    # Derived paths
    human_path = os.path.join(dataset_path, 'sequences_canonical')
    
    print(f"Loading sequence: {sequence_name}")
    print(f"Dataset type: {dataset_type}")
    
    # Load poses and canonical joints based on dataset type
    if dataset_type in ['BEHAVE', 'BEHAVE_CORRECT']:
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(
            sequence_name, human_path, 'smplh', 10
        )
        canonical_joints = get_mean_pose_joints(sequence_name, gender, 'smplh', 10)
    elif dataset_type in ['NEURALDOME', 'IMHD']:
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(
            sequence_name, human_path, 'smplh', 16
        )
        canonical_joints = get_mean_pose_joints(sequence_name, gender, 'smplh', 16)
    elif dataset_type == 'CHAIRS':
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(
            sequence_name, human_path, 'smplx', 10
        )
        canonical_joints = get_mean_pose_joints(sequence_name, gender, 'smplx', 10)
    elif dataset_type in ['INTERCAP', 'INTERCAP_CORRECT']:
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(
            sequence_name, human_path, 'smplx', 10, True
        )
        canonical_joints = get_mean_pose_joints(sequence_name, gender, 'smplx', 10, True)
    elif dataset_type in ['OMOMO', 'OMOMO_CORRECT']:
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(
            sequence_name, human_path, 'smplx', 16
        )
        canonical_joints = get_mean_pose_joints(sequence_name, gender, 'smplx', 16)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    print(f"Original poses shape: {poses.shape}")
    print(f"Canonical joints shape: {canonical_joints.shape}")
    
    # Load object data for palm contact and orientation detection
    print("Loading object data for orientation detection...")
    object_path = os.path.join(dataset_path, 'objects')
    
    with np.load(os.path.join(human_path, sequence_name, 'object.npz'), allow_pickle=True) as f:
        obj_angles, obj_trans, obj_name = f['angles'], f['trans'], str(f['name'])
    
    from scipy.spatial.transform import Rotation
    angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
    
    import trimesh
    OBJ_MESH = trimesh.load(os.path.join(object_path, obj_name, obj_name+'.obj'))
    print(f"Object name: {obj_name}")
    
    ov = np.array(OBJ_MESH.vertices).astype(np.float32)
    object_faces = OBJ_MESH.faces.astype(np.int32)
    ov = torch.from_numpy(ov).float().to(device)
    rot = torch.tensor(angle_matrix).float().to(device)
    obj_trans = torch.tensor(obj_trans).float().to(device)
    object_verts = torch.einsum('ni,tij->tnj', ov, rot.permute(0,2,1)) + obj_trans.unsqueeze(1)
    
    # Convert joints to tensor if needed
    if isinstance(joints, np.ndarray):
        joints = torch.from_numpy(joints).float().to(device)
    
    # Fix all joint poses
    print(f"\nFixing joint poses with threshold {threshold} degrees...")
    fixed_poses = fix_all_joint_poses(poses, joints, object_verts, canonical_joints, threshold)
    
    # Regenerate mesh with fixed poses
    print("\nRegenerating mesh with fixed poses...")
    if dataset_type in ['BEHAVE', 'BEHAVE_CORRECT']:
        fixed_verts, fixed_joints, fixed_faces, _ = regen_smpl(
            sequence_name, fixed_poses, betas, trans, gender, 'smplh', 10
        )
    elif dataset_type in ['NEURALDOME', 'IMHD']:
        fixed_verts, fixed_joints, fixed_faces, _ = regen_smpl(
            sequence_name, fixed_poses, betas, trans, gender, 'smplh', 16
        )
    elif dataset_type == 'CHAIRS':
        fixed_verts, fixed_joints, fixed_faces, _ = regen_smpl(
            sequence_name, fixed_poses, betas, trans, gender, 'smplx', 10
        )
    elif dataset_type in ['INTERCAP', 'INTERCAP_CORRECT']:
        fixed_verts, fixed_joints, fixed_faces   = regen_smpl(
            sequence_name, fixed_poses, betas, trans, gender, 'smplx', 10, True
        )
    elif dataset_type in ['OMOMO', 'OMOMO_CORRECT']:
        fixed_verts, fixed_joints, fixed_faces, _ = regen_smpl(
            sequence_name, fixed_poses, betas, trans, gender, 'smplx', 16
        )
    
    # Save fixed poses
    fixed_human_path = os.path.join(human_path, sequence_name, 'joint_fixed.npz')
    np.savez(fixed_human_path, 
             poses=fixed_poses, 
             betas=betas, 
             trans=trans, 
             gender=gender)
    print(f"Fixed poses saved to: {fixed_human_path}")
    
    # Print summary of changes
    print("\nSummary of changes:")
    for joint_idx in [LEFT_COLLAR, RIGHT_COLLAR, LEFT_SHOULDER, RIGHT_SHOULDER, 
                     LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]:
        joint_name = {
            LEFT_COLLAR: "Left Collar (joint 13)",
            RIGHT_COLLAR: "Right Collar (joint 14)", 
            LEFT_SHOULDER: "Left Shoulder (joint 16)",
            RIGHT_SHOULDER: "Right Shoulder (joint 17)",
            LEFT_ELBOW: "Left Elbow (joint 18)",
            RIGHT_ELBOW: "Right Elbow (joint 19)",
            LEFT_WRIST: "Left Wrist (joint 20)",
            RIGHT_WRIST: "Right Wrist (joint 21)"
        }[joint_idx]
        
        flip_indices, _, _ = detect_flips(poses, joint_idx, threshold)
        fixed_flip_indices, _, _ = detect_flips(fixed_poses, joint_idx, threshold)
        
        print(f"  {joint_name}: {len(flip_indices)} -> {len(fixed_flip_indices)} flips")
    
    # Visualization
    print("\nGenerating visualization...")
    render_path = f'./save_fix_joint/{dataset_type}'
    os.makedirs(render_path, exist_ok=True)
    
    # Convert fixed_verts to numpy if it's a tensor
    if isinstance(fixed_verts, torch.Tensor):
        fixed_verts_np = fixed_verts.float().detach().cpu().numpy()
    else:
        fixed_verts_np = fixed_verts
    
    # Convert fixed_faces to numpy if it's a tensor
    if isinstance(fixed_faces, torch.Tensor):
        fixed_faces_np = fixed_faces[0].detach().cpu().numpy().astype(np.int32)
    else:
        fixed_faces_np = fixed_faces[0].astype(np.int32)
    
    # Convert object_verts to numpy
    object_verts_np = object_verts.detach().cpu().numpy()
    
    visualize_body_obj(
        fixed_verts_np, 
        fixed_faces_np, 
        object_verts_np, 
        object_faces,
        save_path=os.path.join(render_path, f'{sequence_name}_joint_fixed.mp4'), 
        show_frame=True, 
        multi_angle=True
    )
    print(f"Visualization saved to: {os.path.join(render_path, f'{sequence_name}_joint_fixed.mp4')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix joint poses for collar, shoulder, elbow, and wrist joints.")
    parser.add_argument("--dataset_path", required=True, help="Path to the root of the dataset.")
    parser.add_argument("--sequence_name", required=True, help="Name of the sequence.")
    parser.add_argument("--threshold", type=float, default=20, help="Angle threshold in degrees for flip detection.")
    
    args = parser.parse_args()
    main(args.dataset_path, args.sequence_name, args.threshold) 