# Comprehensive Pipeline: Joint Pose Fix + Palm Fix + Optimization
import sys
from render.mesh_viz import visualize_body_obj
import math
from scipy.spatial.distance import cdist
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import smplx
import pytorch3d.loss
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from pytorch3d.ops import cot_laplacian
from pytorch3d.structures import Meshes
from human_body_prior.tools import tgm_conversion as tgm
import chamfer_distance as chd
from scipy.spatial.transform import Rotation
import trimesh
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu()
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R, Slerp
from human_body_prior.body_model.body_model import BodyModel
from bone_lists import bone_list_behave, bone_list_omomo
from loss import point2point_signed

# Joint indices
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

MODEL_PATH = 'models'

# Load SMPL models (same as in other files) - for Steps 1 and 2
smplh10 = {}
smplx10 = {}
smplx12 = {}
smplh16 = {}
smplx16 = {}

def load_models():
    """Load all SMPL models for Steps 1 and 2"""
    global smplh10, smplx10, smplx12, smplh16, smplx16
    
    # SMPLH 10
    smplh_model_male = smplx.create(MODEL_PATH, model_type='smplh', gender="male", use_pca=False, ext='pkl')
    smplh_model_female = smplx.create(MODEL_PATH, model_type='smplh', gender="female", use_pca=False, ext='pkl')
    smplh_model_neutral = smplx.create(MODEL_PATH, model_type='smplh', gender="neutral", use_pca=False, ext='pkl')
    smplh10 = {'male': smplh_model_male, 'female': smplh_model_female, 'neutral': smplh_model_neutral}
    
    # SMPLX 10
    smplx_model_male = smplx.create(MODEL_PATH, model_type='smplx', gender='male', use_pca=False, ext='pkl')
    smplx_model_female = smplx.create(MODEL_PATH, model_type='smplx', gender="female", use_pca=False, ext='pkl')
    smplx_model_neutral = smplx.create(MODEL_PATH, model_type='smplx', gender="neutral", use_pca=False, ext='pkl')
    smplx10 = {'male': smplx_model_male, 'female': smplx_model_female, 'neutral': smplx_model_neutral}
    
    # SMPLX 12
    smplx12_model_male = smplx.create(MODEL_PATH, model_type='smplx', gender="male", num_pca_comps=12, use_pca=True, flat_hand_mean=True, ext='pkl')
    smplx12_model_female = smplx.create(MODEL_PATH, model_type='smplx', gender="female", num_pca_comps=12, use_pca=True, flat_hand_mean=True, ext='pkl')
    smplx12_model_neutral = smplx.create(MODEL_PATH, model_type='smplx', gender="neutral", num_pca_comps=12, use_pca=True, flat_hand_mean=True, ext='pkl')
    smplx12 = {'male': smplx12_model_male, 'female': smplx12_model_female, 'neutral': smplx12_model_neutral}
    
    # SMPLH 16
    SMPLH_PATH = MODEL_PATH+'/smplh'
    surface_model_male_fname = os.path.join(SMPLH_PATH,'female', "model.npz")
    surface_model_female_fname = os.path.join(SMPLH_PATH, "male","model.npz")
    surface_model_neutral_fname = os.path.join(SMPLH_PATH, "neutral", "model.npz")
    dmpl_fname = None
    num_dmpls = None 
    num_expressions = None
    num_betas = 16
    
    smplh16_model_male = BodyModel(bm_fname=surface_model_male_fname, num_betas=num_betas, num_expressions=num_expressions, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname)
    smplh16_model_female = BodyModel(bm_fname=surface_model_female_fname, num_betas=num_betas, num_expressions=num_expressions, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname)
    smplh16_model_neutral = BodyModel(bm_fname=surface_model_neutral_fname, num_betas=num_betas, num_expressions=num_expressions, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname)
    smplh16 = {'male': smplh16_model_male, 'female': smplh16_model_female, 'neutral': smplh16_model_neutral}
    
    # SMPLX 16
    SMPLX_PATH = MODEL_PATH+'/smplx'
    surface_model_male_fname = os.path.join(SMPLX_PATH,"SMPLX_MALE.npz")
    surface_model_female_fname = os.path.join(SMPLX_PATH,"SMPLX_FEMALE.npz")
    surface_model_neutral_fname = os.path.join(SMPLX_PATH, "SMPLX_NEUTRAL.npz")
    
    smplx16_model_male = BodyModel(bm_fname=surface_model_male_fname, num_betas=num_betas, num_expressions=num_expressions, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname)
    smplx16_model_female = BodyModel(bm_fname=surface_model_female_fname, num_betas=num_betas, num_expressions=num_expressions, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname)
    smplx16_model_neutral = BodyModel(bm_fname=surface_model_neutral_fname, num_betas=num_betas, num_expressions=num_expressions, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname)
    smplx16 = {'male': smplx16_model_male, 'female': smplx16_model_female, 'neutral': smplx16_model_neutral}

# Load models for Steps 1 and 2
load_models()

# Load SMPLX16 models for optimization (same as in two_stage_wrist_optimize.py)
SMPLX_PATH = 'models/smplx'
surface_model_male_fname = os.path.join(SMPLX_PATH, "SMPLX_MALE.npz")
surface_model_female_fname = os.path.join(SMPLX_PATH, "SMPLX_FEMALE.npz")
surface_model_neutral_fname = os.path.join(SMPLX_PATH, "SMPLX_NEUTRAL.npz")

dmpl_fname = None
num_dmpls = None 
num_expressions = None
num_betas = 16 

# Load SMPLX16 models for optimization (same as two_stage_wrist_optimize.py)
sbj_m_female = BodyModel(bm_fname=surface_model_female_fname,
                    num_betas=num_betas,
                    num_expressions=num_expressions,
                    num_dmpls=num_dmpls,
                    dmpl_fname=dmpl_fname).to(device)
    
sbj_m_male = BodyModel(bm_fname=surface_model_male_fname,
                    num_betas=num_betas,
                    num_expressions=num_expressions,
                    num_dmpls=num_dmpls,
                    dmpl_fname=dmpl_fname).to(device)

sbj_m_all = {'male': sbj_m_male, 'female': sbj_m_female}

rhand_idx = np.load('./exp/rhand_smplx_ids.npy')
lhand_idx = np.load('./exp/lhand_smplx_ids.npy')

# Load detailed hand indices for different finger parts
RHAND_INDEXES_DETAILED = []
for i in range(5):
    for j in range(3):
        RHAND_INDEXES_DETAILED.append(np.load(f'./exp/index_778/hand_778_{i}_{j}.npy'))
RHAND_INDEXES_DETAILED.append(np.load(f'./exp/index_778/hand_778_5.npy'))

LHAND_INDEXES_DETAILED = []
for i in range(5):
    for j in range(3):
        LHAND_INDEXES_DETAILED.append(np.load(f'./exp/index_778/hand_778_{i}_{j}.npy'))
LHAND_INDEXES_DETAILED.append(np.load(f'./exp/index_778/hand_778_5.npy'))

print("Hand indices loaded successfully")


class FixTracker:
    """Track which frames and joints have been fixed by different methods"""
    def __init__(self, num_frames):
        self.num_frames = num_frames
        # Track which joints were fixed in which frames
        # Shape: (num_frames, num_joints) where num_joints = 8 (collar, shoulder, elbow, wrist)
        self.joint_fixed = np.zeros((num_frames, 8), dtype=bool)  # [left_collar, right_collar, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist]
        self.palm_fixed = np.zeros((num_frames, 8), dtype=bool)
        self.optimized = np.zeros((num_frames, 8), dtype=bool)
    
    def mark_joint_fixed(self, frame_indices, joint_indices):
        """Mark specific joints in specific frames as fixed by joint pose fixing"""
        for frame_idx in frame_indices:
            for joint_idx in joint_indices:
                if 0 <= joint_idx < 8:
                    self.joint_fixed[frame_idx, joint_idx] = True
    
    def mark_palm_fixed(self, frame_indices, joint_indices):
        """Mark specific joints in specific frames as fixed by palm orientation fixing"""
        for frame_idx in frame_indices:
            for joint_idx in joint_indices:
                if 0 <= joint_idx < 8:
                    self.palm_fixed[frame_idx, joint_idx] = True
    
    def mark_optimized(self, frame_indices, joint_indices):
        """Mark specific joints in specific frames as optimized"""
        for frame_idx in frame_indices:
            for joint_idx in joint_indices:
                if 0 <= joint_idx < 8:
                    self.optimized[frame_idx, joint_idx] = True
    
    def get_fixed_frames_and_joints(self):
        """Get all frames and joints that have been fixed by any method"""
        return self.joint_fixed | self.palm_fixed
    
    def get_joint_fixed_frames_and_joints(self):
        """Get frames and joints fixed by joint pose fixing"""
        return self.joint_fixed
    
    def get_palm_fixed_frames_and_joints(self):
        """Get frames and joints fixed by palm orientation fixing"""
        return self.palm_fixed
    
    def get_optimized_frames_and_joints(self):
        """Get frames and joints that have been optimized"""
        return self.optimized
    
    def mark_optimized_from_mask(self, mask):
        """Mark joints as optimized based on a boolean mask
        
        Args:
            mask: (T, 8) boolean array indicating which joint-frame combinations were optimized
        """
        self.optimized = mask.copy()
    
    def print_summary(self):
        """Print summary of fixes"""
        joint_names = ['left_collar', 'right_collar', 'left_shoulder', 'right_shoulder', 
                      'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
        
        print(f"Fix Summary:")
        print(f"  Joint pose fixed:")
        for i, name in enumerate(joint_names):
            fixed_count = self.joint_fixed[:, i].sum()
            print(f"    {name}: {fixed_count}/{self.num_frames} frames")
        
        print(f"  Palm orientation fixed:")
        for i, name in enumerate(joint_names):
            fixed_count = self.palm_fixed[:, i].sum()
            print(f"    {name}: {fixed_count}/{self.num_frames} frames")
        
        print(f"  Optimized:")
        for i, name in enumerate(joint_names):
            optimized_count = self.optimized[:, i].sum()
            print(f"    {name}: {optimized_count}/{self.num_frames} frames")
        
        total_fixed = (self.joint_fixed | self.palm_fixed).sum()
        print(f"  Total fixed joints: {total_fixed}")

def interpolate_pose(p1, p2, alpha):
    """Spherical interpolation between two axis-angle rotations."""
    key_times = [0, 1]
    key_rots = R.from_rotvec([p1, p2])
    slerp = Slerp(key_times, key_rots)
    return slerp(alpha).as_rotvec()
    
def compute_palm_contact_and_orientation(
    human_joints: torch.Tensor,
    object_verts: torch.Tensor,
    hand: str = 'right',
    contact_thresh: float = 0.09,
    orient_angle_thresh: float = 70.0,
    orient_dist_thresh: float = 0.09
):
    """Compute palm contact and orientation masks"""
    if human_joints.device != object_verts.device:
        human_joints = human_joints.to(object_verts.device)
    T, J, _ = human_joints.shape

    hand = hand.lower()
    if hand.startswith('r'):
        IDX_WRIST, IDX_INDEX, IDX_PINKY = 21, 40, 46
        flip_normal = False
    else:
        IDX_WRIST, IDX_INDEX, IDX_PINKY = 20, 25, 31
        flip_normal = True

    wrist = human_joints[:, IDX_WRIST, :]
    idx = human_joints[:, IDX_INDEX, :]
    pinky = human_joints[:, IDX_PINKY, :]

    v1 = idx - wrist
    v2 = pinky - wrist
    normals = torch.cross(v1, v2, dim=1)
    if flip_normal:
        normals = -normals
    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-8)

    centroid = (wrist + idx + pinky) / 3.0
    rel = object_verts - centroid.unsqueeze(1)
    dists = rel.norm(dim=2)

    contact_mask = (dists < contact_thresh).any(dim=1)

    cos_thresh = torch.cos(torch.deg2rad(torch.tensor(orient_angle_thresh, device=normals.device)))
    rel_dir = rel / (dists.unsqueeze(-1) + 1e-8)
    cosines = (normals.unsqueeze(1) * rel_dir).sum(dim=2)
    mask = (cosines >= cos_thresh) & (dists < orient_dist_thresh)
    orient_mask = mask.any(dim=1)

    return contact_mask, orient_mask, normals

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
    relative_rot = rot1.inv().as_matrix() @ rot2.as_matrix()
    relative_rot = Rotation.from_matrix(relative_rot)
    
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

def detect_hand_twist_from_canonical_batch(poses, joints_canonical):
    """Detect wrist twist angles for all frames using canonical bone axis"""
    T = poses.shape[0]
    twist_left_list = []
    twist_right_list = []
    
    for frame_idx in range(T):
        pose_i = poses[frame_idx].reshape(52, 3)
        twist_left, twist_right = detect_hand_twist_from_canonical(pose_i, joints_canonical)
        twist_left_list.append(twist_left)
        twist_right_list.append(twist_right)
    
    return twist_left_list, twist_right_list

def detect_hand_twist_from_canonical(pose_i, joints_canonical):
    """Detect wrist twist angles using canonical bone axis"""
    def compute_twist_angle(pose_wrist, bone_axis):
        rotvec = pose_wrist
        angle = np.linalg.norm(rotvec)
        if angle < 1e-6:
            return 0.0
        axis = rotvec / angle
        twist_cos = np.dot(axis, bone_axis)
        twist_angle = angle * twist_cos
        return np.rad2deg(twist_angle)

    bone_axis_left = joints_canonical[LEFT_WRIST] - joints_canonical[LEFT_ELBOW]
    bone_axis_left /= np.linalg.norm(bone_axis_left)

    bone_axis_right = joints_canonical[RIGHT_WRIST] - joints_canonical[RIGHT_ELBOW]
    bone_axis_right /= np.linalg.norm(bone_axis_right)

    twist_left = compute_twist_angle(pose_i[LEFT_WRIST], bone_axis_left)
    twist_right = compute_twist_angle(pose_i[RIGHT_WRIST], bone_axis_right)

    return twist_left, twist_right

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
        tuple: (fixed poses, list of frame indices that were actually fixed)
    """
    T = poses.shape[0]
    pose_start = JOINT_TO_POSE_MAPPING[joint_idx]
    fixed_poses = poses.copy()
    
    if not flip_indices:
        return fixed_poses, []
    
    print(f"Fixing joint {joint_idx} with {len(flip_indices)} flips detected")
    
    # Track which frames have been fixed to avoid repeated fixing
    fixed = np.zeros(T, dtype=bool)
    
    # Process flips sequentially
    for i, flip_idx in enumerate(flip_indices):
        print(f"  Processing flip {i+1}/{len(flip_indices)} at frame {flip_idx}")
        
        # Find the previous flip (if any)
        prev_flip_idx = None
        if i > 0:
            prev_flip_idx = flip_indices[i-1]
        
        # Find the next flip (if any)
        next_flip_idx = None
        if i + 1 < len(flip_indices):
            next_flip_idx = flip_indices[i + 1]
        
        # Define segments for this flip
        left_start = 0 if prev_flip_idx is None else prev_flip_idx + 1
        left_end = flip_idx
        right_start = flip_idx + 1
        right_end = T - 1 if next_flip_idx is None else next_flip_idx
        
        # Special handling for consecutive flips
        if next_flip_idx is not None and next_flip_idx == flip_idx + 1:
            if i + 2 < len(flip_indices):
                right_end = flip_indices[i + 2]
            else:
                right_end = T - 1
            print(f"    Consecutive flip detected: adjusting right segment to [{right_start},{right_end}]")
        
        if prev_flip_idx is not None and prev_flip_idx == flip_idx - 1:
            if i > 1:
                left_start = flip_indices[i - 2] + 1
            else:
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
            if joint_idx in [LEFT_COLLAR, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST] and left_twist_bound_mask is not None:
                left_oob_count = sum(left_twist_bound_mask[left_start:left_end+1]) if left_end >= left_start else 0
                right_oob_count = sum(left_twist_bound_mask[right_start:right_end+1]) if right_end >= right_start else 0
            elif joint_idx in [RIGHT_COLLAR, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST] and right_twist_bound_mask is not None:
                left_oob_count = sum(right_twist_bound_mask[left_start:left_end+1]) if left_end >= left_start else 0
                right_oob_count = sum(right_twist_bound_mask[right_start:right_end+1]) if right_end >= right_start else 0
            else:
                left_oob_count = 0
                right_oob_count = 0
            
            print(f"    Left segment out-of-bounds: {left_oob_count}/{left_end-left_start+1}")
            print(f"    Right segment out-of-bounds: {right_oob_count}/{right_end-right_start+1}")
            
            if left_oob_count > right_oob_count:
                seg_start, seg_end = left_start, left_end
                jump_angle = angle_diffs[flip_idx]
                print(f"    Fixing LEFT segment [{seg_start},{seg_end}] by {jump_angle:.1f}° (more out-of-bounds)")
            elif right_oob_count > left_oob_count:
                seg_start, seg_end = right_start, right_end
                jump_angle = -angle_diffs[flip_idx]
                print(f"    Fixing RIGHT segment [{seg_start},{seg_end}] by {jump_angle:.1f}° (more out-of-bounds)")
            elif left_oob_count == right_oob_count:
                left_length = left_end - left_start + 1
                right_length = right_end - right_start + 1
                
                if left_length < right_length:
                    seg_start, seg_end = left_start, left_end
                    jump_angle = angle_diffs[flip_idx]
                    print(f"    Fixing LEFT segment [{seg_start},{seg_end}] by {jump_angle:.1f}° (longer segment)")
                else:
                    seg_start, seg_end = right_start, right_end
                    jump_angle = -angle_diffs[flip_idx]
                    print(f"    Fixing RIGHT segment [{seg_start},{seg_end}] by {jump_angle:.1f}° (longer segment)")
        else:
            if left_orient_ratio < right_orient_ratio:
                seg_start, seg_end = left_start, left_end
                jump_angle = angle_diffs[flip_idx]
                print(f"    Fixing LEFT segment [{seg_start},{seg_end}] by {jump_angle:.1f}°")
            else:
                seg_start, seg_end = right_start, right_end
                jump_angle = -angle_diffs[flip_idx]
                print(f"    Fixing RIGHT segment [{seg_start},{seg_end}] by {jump_angle:.1f}°")
        
        # Check if the segment to be fixed is already mostly fixed
        fixed_ratio = fixed[np.arange(seg_start, seg_end + 1)].mean()
        if fixed_ratio > 0.8:
            print(f"    Segment [{seg_start},{seg_end}] already {fixed_ratio:.1%} fixed - skipping")
            continue
        
        segment_length = seg_end - seg_start + 1
        is_between_flips = False
        if seg_start > 0 and seg_end < T - 1:
            is_between_flips = True
            prev_flip = prev_flip_idx
            next_flip = next_flip_idx
        
        if segment_length <= 10 and is_between_flips and prev_flip is not None and next_flip is not None:
            print(f"    Segment [{seg_start},{seg_end}] (length={segment_length}) is between flips {prev_flip} and {next_flip} - using interpolation")
            
            pose_before = fixed_poses[seg_start - 1, pose_start:pose_start+3]
            pose_after = fixed_poses[seg_end + 1, pose_start:pose_start+3]
            
            for t in range(seg_start, seg_end + 1):
                alpha = (t - seg_start) / (seg_end - seg_start) if seg_end > seg_start else 0.0
                fixed_poses[t, pose_start:pose_start+3] = interpolate_pose(pose_before, pose_after, alpha)
                fixed[t] = True
        else:
            if segment_length > 10:
                print(f"    Segment [{seg_start},{seg_end}] (length={segment_length}) is too long - using angle reversal")
            else:
                print(f"    Segment [{seg_start},{seg_end}] (length={segment_length}) includes boundary frames - using angle reversal")
            
            rotation_axis = relative_axes[flip_idx]
            print(f"    Using relative rotation axis from flip: [{rotation_axis[0]:.3f}, {rotation_axis[1]:.3f}, {rotation_axis[2]:.3f}]")
            
            for t in range(seg_start, seg_end + 1):
                pose_vec = fixed_poses[t, pose_start:pose_start+3]
                fixed_poses[t, pose_start:pose_start+3] = rotate_pose_around_axis(
                    pose_vec, rotation_axis, jump_angle
                )
                fixed[t] = True
    
    # Return the fixed poses and the list of frames that were actually fixed
    fixed_frame_indices = np.where(fixed)[0].tolist()
    return fixed_poses, fixed_frame_indices

def rotate_pose_around_axis(pose_vec, axis, angle_deg):
    """Rotate pose around given axis"""
    R_current = R.from_rotvec(pose_vec).as_matrix()
    R_correction = R.from_rotvec(np.deg2rad(angle_deg) * axis).as_matrix()
    R_fixed = R_current @ R_correction
    return R.from_matrix(R_fixed).as_rotvec()

def compute_palm_object_angle(
    human_joints: torch.Tensor,       # (T, J, 3)
    object_verts: torch.Tensor,       # (T, N, 3)
    hand: str = 'left',               # 'left' or 'right'
    contact_thresh: float = 0.07      # contact distance threshold
):
    """
    Compute the angle between palm normal and vector from palm center to nearest object vertices.
    Returns the angle in degrees for each frame.
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
    rel = object_verts - centroid.unsqueeze(1)   # (T, N, 3)
    dists = rel.norm(dim=2)                     # (T, N)

    # 6) 找到最近的接触点
    contact_mask = (dists < contact_thresh)  # (T, N)
    
    angles = torch.zeros(T, device=human_joints.device)
    
    for t in range(T):
        if contact_mask[t].any():
            # 找到最近的接触点
            contact_dists = dists[t][contact_mask[t]]
            contact_rel = rel[t][contact_mask[t]]
            
            # 找到最近的接触点
            min_idx = torch.argmin(contact_dists)
            nearest_rel = contact_rel[min_idx]
            
            # 归一化相对向量
            nearest_rel_norm = nearest_rel / (nearest_rel.norm() + 1e-8)
            
            # 计算余弦值
            cosine = torch.dot(normals[t], nearest_rel_norm)
            cosine = torch.clamp(cosine, -1.0, 1.0)  # 确保在有效范围内
            
            # 计算角度
            angle_rad = torch.acos(cosine)
            angles[t] = torch.rad2deg(angle_rad)
        else:
            # 如果没有接触点，使用默认角度
            angles[t] = 180.0
    
    return angles.cpu().numpy()

def robust_wrist_flip_fix_left(twist_list, contact_mask, orient_mask, poses, joint_idx, axis, human_joints=None, object_verts=None, jump_thresh=20):
    """
    Fixes wrist flips:
    - Uses directional fix if twist jumps are detected
    - Otherwise uses persistent flip threshold or transient interpolation
    """
    T = len(twist_list)
    diffs = np.diff(twist_list)
    large_jump_indices = [i for i, d in enumerate(diffs) if abs(d) > jump_thresh]

    out_of_bounds = [(tw > 90 or tw < -110) for tw in twist_list]
    twist_array = np.array(twist_list, dtype=np.float32)
    
    # Calculate proportion of frames where hand is in contact but not in correct orientation
    # Move tensors to CPU for numpy operations
    contact_mask_cpu = contact_mask.bool().cpu()
    orient_mask_cpu = orient_mask.bool().cpu()
    
    contact_frames = contact_mask_cpu.sum().item()
    if contact_frames > 0:
        contact_but_wrong_orient = (contact_mask_cpu & (~orient_mask_cpu)).sum().item()
        proportion_wrong_orient_given_contact = contact_but_wrong_orient / contact_frames
    else:
        proportion_wrong_orient_given_contact = 0.0
    
    # Calculate proportion of out-of-bounds frames among non-contact frames only
    non_contact_frames = (~contact_mask_cpu).sum().item()
    if non_contact_frames > 0:
        non_contact_out_of_bounds = ((~contact_mask_cpu) & (np.array(out_of_bounds))).sum().item()
        proportion_out_of_bounds_given_no_contact = non_contact_out_of_bounds / non_contact_frames
    else:
        proportion_out_of_bounds_given_no_contact = 0.0
    
    print(f"left orient_mask: {proportion_wrong_orient_given_contact:.3f} (contact frames: {contact_frames})")
    print(f"left out_of_bounds: {proportion_out_of_bounds_given_no_contact:.3f} (non-contact frames: {non_contact_frames})")
    
    fixed_frames = []
    
    if large_jump_indices:
        print(f"left Detected {len(large_jump_indices)} flip jump(s) — using directional persistent fix: {large_jump_indices}.")
        # Apply directional fix for large jumps
        for t in range(T):
            if any(t in range(jump_idx, jump_idx + 10) for jump_idx in large_jump_indices):
                poses[t, joint_idx * 3 : joint_idx * 3 + 3] = rotate_pose_around_axis(
                    poses[t, joint_idx * 3 : joint_idx * 3 + 3], axis, 180
                )
                fixed_frames.append(t)
    elif proportion_wrong_orient_given_contact > 0.7:
        # If there is contact but wrong orientation proportion is large, calculate specific angle
        print(f"left Contact with wrong orientation detected - calculating specific angles")
        
        if human_joints is not None and object_verts is not None:
            # Calculate specific angles using the palm-object angle function
            specific_angles = compute_palm_object_angle(human_joints, object_verts, hand='left', contact_thresh=0.09)
            
            for t in range(T):
                if contact_mask_cpu[t] and not orient_mask_cpu[t]:
                    # Use the calculated specific angle
                    angle = specific_angles[t]
                    poses[t, joint_idx * 3 : joint_idx * 3 + 3] = rotate_pose_around_axis(
                        poses[t, joint_idx * 3 : joint_idx * 3 + 3], axis, angle
                    )
                    fixed_frames.append(t)
                else:
                    # Apply 180 degree rotation for other cases
                    poses[t, joint_idx * 3 : joint_idx * 3 + 3] = rotate_pose_around_axis(
                        poses[t, joint_idx * 3 : joint_idx * 3 + 3], axis, 180
                    )
                    fixed_frames.append(t)
        else:
            # Fallback to 180 degrees if data not available
            for t in range(T):
                poses[t, joint_idx * 3 : joint_idx * 3 + 3] = rotate_pose_around_axis(
                    poses[t, joint_idx * 3 : joint_idx * 3 + 3], axis, 180
                )
                fixed_frames.append(t)
    elif proportion_out_of_bounds_given_no_contact > 0.7:
        # If no contact but out of bounds proportion is large, rotate by 180 degrees
        print(f"left No contact but out of bounds - applying 180° rotation")
        for t in range(T):
            poses[t, joint_idx * 3 : joint_idx * 3 + 3] = rotate_pose_around_axis(
                poses[t, joint_idx * 3 : joint_idx * 3 + 3], axis, 180
            )
            fixed_frames.append(t)
    
    return poses, fixed_frames

def robust_wrist_flip_fix_right(twist_list, contact_mask, orient_mask, poses, joint_idx, axis, human_joints=None, object_verts=None, jump_thresh=20):
    """
    Fixes wrist flips:
    - Uses directional fix if twist jumps are detected
    - Otherwise uses persistent flip threshold or transient interpolation
    """
    T = len(twist_list)
    diffs = np.diff(twist_list)
    large_jump_indices = [i for i, d in enumerate(diffs) if abs(d) > jump_thresh]

    out_of_bounds = [(tw < -90 or tw > 110) for tw in twist_list]
    twist_array = np.array(twist_list, dtype=np.float32)
    
    # Calculate proportion of frames where hand is in contact but not in correct orientation
    # Move tensors to CPU for numpy operations
    contact_mask_cpu = contact_mask.bool().cpu()
    orient_mask_cpu = orient_mask.bool().cpu()
    
    contact_frames = contact_mask_cpu.sum().item()
    if contact_frames > 0:
        contact_but_wrong_orient = (contact_mask_cpu & (~orient_mask_cpu)).sum().item()
        proportion_wrong_orient_given_contact = contact_but_wrong_orient / contact_frames
    else:
        proportion_wrong_orient_given_contact = 0.0
    
    # Calculate proportion of out-of-bounds frames among non-contact frames only
    non_contact_frames = (~contact_mask_cpu).sum().item()
    if non_contact_frames > 0:
        non_contact_out_of_bounds = ((~contact_mask_cpu) & (np.array(out_of_bounds))).sum().item()
        proportion_out_of_bounds_given_no_contact = non_contact_out_of_bounds / non_contact_frames
    else:
        proportion_out_of_bounds_given_no_contact = 0.0
    
    print(f"right orient_mask: {proportion_wrong_orient_given_contact:.3f} (contact frames: {contact_frames})")
    print(f"right out_of_bounds: {proportion_out_of_bounds_given_no_contact:.3f} (non-contact frames: {non_contact_frames})")
    
    fixed_frames = []
    
    if large_jump_indices:
        print(f"right Detected {len(large_jump_indices)} flip jump(s) — using directional persistent fix: {large_jump_indices}.")
        # Apply directional fix for large jumps
        for t in range(T):
            if any(t in range(jump_idx, jump_idx + 10) for jump_idx in large_jump_indices):
                poses[t, joint_idx * 3 : joint_idx * 3 + 3] = rotate_pose_around_axis(
                    poses[t, joint_idx * 3 : joint_idx * 3 + 3], axis, 180
                )
                fixed_frames.append(t)
    elif proportion_wrong_orient_given_contact > 0.7:
        # If there is contact but wrong orientation proportion is large, calculate specific angle
        print(f"right Contact with wrong orientation detected - calculating specific angles")
        
        if human_joints is not None and object_verts is not None:
            # Calculate specific angles using the palm-object angle function
            specific_angles = compute_palm_object_angle(human_joints, object_verts, hand='right', contact_thresh=0.09)
            
            for t in range(T):
                if contact_mask_cpu[t] and not orient_mask_cpu[t]:
                    # Use the calculated specific angle
                    angle = specific_angles[t]
                    poses[t, joint_idx * 3 : joint_idx * 3 + 3] = rotate_pose_around_axis(
                        poses[t, joint_idx * 3 : joint_idx * 3 + 3], axis, angle
                    )
                    fixed_frames.append(t)
                else:
                    # Apply 180 degree rotation for other cases
                    poses[t, joint_idx * 3 : joint_idx * 3 + 3] = rotate_pose_around_axis(
                        poses[t, joint_idx * 3 : joint_idx * 3 + 3], axis, 180
                    )
                    fixed_frames.append(t)
        else:
            # Fallback to 180 degrees if data not available
            for t in range(T):
                poses[t, joint_idx * 3 : joint_idx * 3 + 3] = rotate_pose_around_axis(
                    poses[t, joint_idx * 3 : joint_idx * 3 + 3], axis, 180
                )
                fixed_frames.append(t)
    elif proportion_out_of_bounds_given_no_contact > 0.7:
        # If no contact but out of bounds proportion is large, rotate by 180 degrees
        print(f"right No contact but out of bounds - applying 180° rotation")
        for t in range(T):
            poses[t, joint_idx * 3 : joint_idx * 3 + 3] = rotate_pose_around_axis(
                poses[t, joint_idx * 3 : joint_idx * 3 + 3], axis, 180
            )
            fixed_frames.append(t)
    
    return poses, fixed_frames

def compute_temporal_smoothing_loss(poses, fixed_frames, weight_accel=0.5, weight_vel=0.25):
    """Compute temporal smoothing loss only for fixed frames"""
    if len(fixed_frames) < 2:
        return torch.tensor(0.0, device=poses.device)
    
    # Sort fixed frames
    fixed_frames = sorted(fixed_frames)
    
    # Compute velocities and accelerations only for fixed frames
    velocities = []
    accelerations = []
    
    for i in range(1, len(fixed_frames)):
        t_prev, t_curr = fixed_frames[i-1], fixed_frames[i]
        vel = poses[t_curr] - poses[t_prev]
        velocities.append(vel)
        
        if i > 1:
            t_prev_prev = fixed_frames[i-2]
            vel_prev = poses[t_prev] - poses[t_prev_prev]
            accel = vel - vel_prev
            accelerations.append(accel)
    
    if not velocities:
        return torch.tensor(0.0, device=poses.device)
    
    vel_loss = torch.stack(velocities).norm(dim=1).mean()
    accel_loss = torch.stack(accelerations).norm(dim=1).mean() if accelerations else torch.tensor(0.0, device=poses.device)
    
    return weight_vel * vel_loss + weight_accel * accel_loss

def compute_reference_loss(poses, reference_poses, fixed_frames, weight=1.0):
    """Compute reference loss only for fixed frames"""
    if not fixed_frames:
        return torch.tensor(0.0, device=poses.device)
    
    fixed_poses = poses[fixed_frames]
    fixed_ref_poses = reference_poses[fixed_frames]
    
    diff = fixed_poses - fixed_ref_poses
    return weight * torch.norm(diff, dim=1).mean()

def compute_temporal_smoothing_loss_selective(poses, fixed_joints_mask, weight_accel=0.5, weight_vel=0.25):
    """Compute temporal smoothing loss for frames influenced by previous steps.
    
    For each joint, we compute velocity/acceleration losses between consecutive frames,
    but only for the frames that were influenced by joint pose fixing or palm fixing.
    Non-influenced frames are detached to prevent them from affecting optimization.
    
    Example: If frames [2, 34] and [67, 89] are influenced for a joint, we compute:
    - Velocity loss: (pose[2] - pose[1]), (pose[3] - pose[2]), ..., (pose[34] - pose[33]), (pose[35] - pose[34])
    - Velocity loss: (pose[67] - pose[66]), (pose[68] - pose[67]), ..., (pose[89] - pose[88]), (pose[90] - pose[89])
    - Similar pattern for acceleration losses
    """
    # poses: (T, 24) - all joint poses
    # fixed_joints_mask: (T, 8) - which joints were fixed in which frames
    
    T, num_joints_total = poses.shape
    num_joints = fixed_joints_mask.shape[1]  # Should be 8
    
    if fixed_joints_mask.sum() < 2:
        return torch.tensor(0.0, device=poses.device)
    
    # Reshape poses to (T, num_joints, 3) for easier processing
    poses_reshaped = poses.view(T, num_joints, 3)
    
    total_vel_loss = 0.0
    total_accel_loss = 0.0
    total_vel_count = 0
    total_accel_count = 0
    
    # For each joint, compute temporal smoothing between consecutive frames
    for joint_idx in range(num_joints):
        # Get frames where this joint was fixed
        fixed_frames_for_joint = torch.where(fixed_joints_mask[:, joint_idx])[0]
        
        if len(fixed_frames_for_joint) < 2:
            continue
        
        # Create a mask for this joint: True for fixed frames, False for non-fixed frames
        joint_fixed_mask = fixed_joints_mask[:, joint_idx]  # (T,)
        
        # Compute velocities between consecutive frames (like two_stage_wrist_optimize.py)
        # But detach non-fixed frames so they don't affect optimization
        poses_joint = poses_reshaped[:, joint_idx, :]  # (T, 3)
        
        # Detach non-fixed frames (they won't be affected by gradients)
        poses_joint_detached = poses_joint.clone()
        poses_joint_detached[~joint_fixed_mask] = poses_joint[~joint_fixed_mask].detach()
        
        # Compute velocities between consecutive frames
        vel = poses_joint_detached[1:] - poses_joint_detached[:-1]  # (T-1, 3)
        
        # Only apply velocity loss to frames where either current or previous frame was fixed
        vel_mask = joint_fixed_mask[1:] | joint_fixed_mask[:-1]  # (T-1,)
        
        if vel_mask.sum() > 0:
            vel_loss_joint = torch.sum((vel[vel_mask] ** 2))
            total_vel_loss += vel_loss_joint
            total_vel_count += vel_mask.sum()
        
        # Compute accelerations between consecutive frames (if we have at least 3 frames)
        if T >= 3:
            # Second-order acceleration: (poses[t+1] - poses[t]) - (poses[t] - poses[t-1])
            accel = (poses_joint_detached[2:] - poses_joint_detached[1:-1]) - (poses_joint_detached[1:-1] - poses_joint_detached[:-2])  # (T-2, 3)
            
            # Only apply acceleration loss to frames where any of the three consecutive frames was fixed
            accel_mask = joint_fixed_mask[2:] | joint_fixed_mask[1:-1] | joint_fixed_mask[:-2]  # (T-2,)
            
            if accel_mask.sum() > 0:
                accel_loss_joint = torch.sum((accel[accel_mask] ** 2))
                total_accel_loss += accel_loss_joint
                total_accel_count += accel_mask.sum()
    
    # Normalize by counts
    vel_loss = total_vel_loss / max(total_vel_count, 1)
    accel_loss = total_accel_loss / max(total_accel_count, 1) if total_accel_count > 0 else torch.tensor(0.0, device=poses.device)
    
    return weight_vel * vel_loss + weight_accel * accel_loss

def compute_temporal_smoothing_loss_fixed_frames_only(poses, fixed_joints_mask, joint_optimization_mask, weight_accel=0.5, weight_vel=0.25):
    """Compute temporal smoothing loss ONLY between fixed frames and their neighbors for continuity.
    
    This function computes smoothness between:
    1. Fixed frames and their immediate neighbors (for smooth transitions)
    2. Between consecutive fixed frames (for internal smoothness)
    
    Example: If frames [34, 67] and [122, 150] are fixed, we compute:
    - Velocity loss: (pose[34] - pose[33]), (pose[35] - pose[34]), ..., (pose[67] - pose[66]), (pose[68] - pose[67])
    - Velocity loss: (pose[122] - pose[121]), (pose[123] - pose[122]), ..., (pose[150] - pose[149]), (pose[151] - pose[150])
    - Similar pattern for acceleration losses
    
    This ensures smooth transitions between fixed and non-fixed regions.
    """
    # poses: (T, 24) - all joint poses
    # fixed_joints_mask: (T, 8) - which joints were fixed in which frames
    
    T, num_joints_total = poses.shape
    num_joints = fixed_joints_mask.shape[1]  # Should be 8
    
    if fixed_joints_mask.sum() < 2:
        return torch.tensor(0.0, device=poses.device)
    
    # Reshape poses to (T, num_joints, 3) for easier processing
    poses_reshaped = poses.view(T, num_joints, 3)
    
    total_vel_loss = 0.0
    total_accel_loss = 0.0
    total_vel_count = 0
    total_accel_count = 0
    
    # For each joint, compute temporal smoothing between fixed frames and their neighbors
    # BUT only for joints that were fixed in previous steps
    for joint_idx in range(num_joints):
        # Skip joints that were not fixed in any frame
        if not joint_optimization_mask[joint_idx]:
            continue
            
        # Get frames where this joint was fixed
        fixed_frames_for_joint = torch.where(fixed_joints_mask[:, joint_idx])[0]
        
        if len(fixed_frames_for_joint) < 2:
            continue
        
        # Create a mask for frames that should contribute to smoothness:
        # 1. Fixed frames themselves
        # 2. Neighbors of fixed frames (for smooth transitions)
        smoothness_mask = torch.zeros(T, dtype=torch.bool, device=poses.device)
        
        for frame_idx in fixed_frames_for_joint:
            # Mark the fixed frame
            smoothness_mask[frame_idx] = True
            # Mark neighbors (for smooth transitions)
            if frame_idx > 0:
                smoothness_mask[frame_idx - 1] = True
            if frame_idx < T - 1:
                smoothness_mask[frame_idx + 1] = True
        
        # Compute velocities between consecutive frames
        vel = poses_reshaped[1:, joint_idx, :] - poses_reshaped[:-1, joint_idx, :]  # (T-1, 3)
        
        # Only apply velocity loss to frames where either current or previous frame contributes to smoothness
        vel_mask = smoothness_mask[1:] | smoothness_mask[:-1]  # (T-1,)
        
        if vel_mask.sum() > 0:
            vel_loss_joint = torch.sum((vel[vel_mask] ** 2))
            total_vel_loss += vel_loss_joint
            total_vel_count += vel_mask.sum()
        
        # Compute accelerations between consecutive frames (if we have at least 3 frames)
        if T >= 3:
            # Second-order acceleration: (poses[t+1] - poses[t]) - (poses[t] - poses[t-1])
            accel = (poses_reshaped[2:, joint_idx, :] - poses_reshaped[1:-1, joint_idx, :]) - (poses_reshaped[1:-1, joint_idx, :] - poses_reshaped[:-2, joint_idx, :])  # (T-2, 3)
            
            # Only apply acceleration loss to frames where any of the three consecutive frames contributes to smoothness
            accel_mask = smoothness_mask[2:] | smoothness_mask[1:-1] | smoothness_mask[:-2]  # (T-2,)
            
            if accel_mask.sum() > 0:
                accel_loss_joint = torch.sum((accel[accel_mask] ** 2))
                total_accel_loss += accel_loss_joint
                total_accel_count += accel_mask.sum()
    
    # Normalize by counts
    vel_loss = total_vel_loss / max(total_vel_count, 1)
    accel_loss = total_accel_loss / max(total_accel_count, 1) if total_accel_count > 0 else torch.tensor(0.0, device=poses.device)
    
    return weight_vel * vel_loss + weight_accel * accel_loss

def compute_reference_loss_selective(poses, reference_poses, fixed_joints_mask, weight=1.0):
    """Compute reference loss only for fixed joint-frame combinations.
    
    This function efficiently computes the difference between optimized poses and reference poses,
    but only for the specific joint-frame combinations that were influenced by previous steps.
    """
    # poses: (T, 24) - all joint poses
    # reference_poses: (T, 24) - reference poses
    # fixed_joints_mask: (T, 8) - which joints were fixed in which frames
    
    if fixed_joints_mask.sum() == 0:
        return torch.tensor(0.0, device=poses.device)
    
    # Reshape poses to (T, num_joints, 3) for easier processing
    poses_reshaped = poses.view(poses.shape[0], 8, 3)
    ref_poses_reshaped = reference_poses.view(reference_poses.shape[0], 8, 3)
    
    # Compute loss only for fixed joint-frame combinations
    total_loss = 0.0
    count = 0
    
    # Only iterate over frames and joints that were actually fixed
    fixed_indices = torch.where(fixed_joints_mask)
    for t, joint_idx in zip(fixed_indices[0], fixed_indices[1]):
        diff = poses_reshaped[t, joint_idx] - ref_poses_reshaped[t, joint_idx]
        total_loss += torch.norm(diff)
        count += 1
    
    if count == 0:
        return torch.tensor(0.0, device=poses.device)
    
    return weight * (total_loss / count)

def compute_palm_facing_loss_selective(joints, verts_obj_transformed, contact_mask, fixed_joints_mask, is_left_hand=True, joint_optimization_mask=None):
    """
    Compute palm facing loss: encourage palm normal to align with direction to object.
    Only applies loss when hand is in contact with object.
    
    Args:
        joints: Joint positions (T, J, 3)
        verts_obj_transformed: Object vertices (T, M, 3)
        contact_mask: Boolean mask (T,) indicating when hand is in contact
        fixed_joints_mask: (T, 8) boolean mask indicating which joint-frame combinations were fixed
        is_left_hand: Whether computing for left or right hand
    
    Returns:
        facing_loss: Loss encouraging palm to face the object (during contact)
    """
    # Define palm joint indices
    if is_left_hand:
        IDX_WRIST = 20
        IDX_INDEX = 25  # Left index MCP
        IDX_PINKY = 31  # Left pinky MCP
        flip_normal = True
    else:
        IDX_WRIST = 21
        IDX_INDEX = 40  # Right index MCP
        IDX_PINKY = 46  # Right pinky MCP
        flip_normal = False
    
    # Extract palm joints
    wrist_joints = joints[:, IDX_WRIST, :]  # (T, 3)
    index_joints = joints[:, IDX_INDEX, :]  # (T, 3)
    pinky_joints = joints[:, IDX_PINKY, :]  # (T, 3)
    
    # Compute palm normal
    v1 = index_joints - wrist_joints  # (T, 3)
    v2 = pinky_joints - wrist_joints  # (T, 3)
    palm_normals = torch.cross(v1, v2, dim=1)  # (T, 3)
    if flip_normal:
        palm_normals = -palm_normals
    palm_normals = palm_normals / (palm_normals.norm(dim=1, keepdim=True) + 1e-8)
        
    # Compute palm centroid
    palm_centroid = (wrist_joints + index_joints + pinky_joints) / 3.0  # (T, 3)
    
    # Find nearest object point to palm centroid
    rel_to_centroid = verts_obj_transformed - palm_centroid.unsqueeze(1)  # (T, N, 3)
    dists_to_centroid = rel_to_centroid.norm(dim=2)  # (T, N)
    min_dists, min_idx = dists_to_centroid.min(dim=1)  # (T,), (T,)
    
    # Use gather to maintain gradients
    batch_indices = torch.arange(verts_obj_transformed.shape[0], device=verts_obj_transformed.device)
    nearest_points = verts_obj_transformed[batch_indices, min_idx, :]  # (T, 3)
    
    # FACING LOSS: encourage palm normal to align with direction to object
    # This means palm_normal · direction_to_object ≈ 1
    direction_to_object = nearest_points - palm_centroid
    direction_to_object = direction_to_object / (direction_to_object.norm(dim=1, keepdim=True) + 1e-8)
    facing_dot_product = (palm_normals * direction_to_object).sum(dim=1)
    
    # Simple and direct: penalize deviation from perfect alignment
    # For 0° angle: cos(0°) = 1, loss = 0
    # For 90° angle: cos(90°) = 0, loss = 1  
    # For 130° angle: cos(130°) ≈ -0.64, loss = 1.64
    facing_loss_per_frame = 1.0 - facing_dot_product
    
    # Only apply loss when hand is in contact with object AND when the relevant joints were optimized
    if contact_mask.sum() == 0:
        return torch.tensor(0.0, device=joints.device)
    
    # Check if the relevant joints (wrist, index, pinky) were optimized
    if joint_optimization_mask is not None:
        if is_left_hand:
            # Left hand joints: wrist (6), index (25), pinky (31) -> check if wrist was optimized
            relevant_joint_optimized = joint_optimization_mask[6]  # Left wrist
        else:
            # Right hand joints: wrist (7), index (40), pinky (46) -> check if wrist was optimized  
            relevant_joint_optimized = joint_optimization_mask[7]  # Right wrist
        
        if not relevant_joint_optimized:
            return torch.tensor(0.0, device=joints.device)
    
    facing_loss = torch.mean(torch.where(
        contact_mask,
        facing_loss_per_frame,  # Apply loss during contact
        torch.zeros_like(facing_loss_per_frame)  # No loss otherwise
    ))
    
    return facing_loss

def compute_penetration_loss_selective(verts, verts_obj_transformed, fixed_joints_mask, contact_thresh=0.05):
    """
    Compute penetration loss: penalize when hand vertices are inside the object.
    Only applies to frames where joints were fixed.
    
    Args:
        verts: Human mesh vertices (T, V, 3)
        verts_obj_transformed: Object vertices (T, N, 3)
        fixed_joints_mask: (T, 8) boolean mask indicating which joint-frame combinations were fixed
        contact_thresh: Distance threshold for penetration detection
    
    Returns:
        penetration_loss: Loss penalizing hand-object penetration
    """
    # Load hand vertex indices
    try:
        rhand_idx = np.load('./exp/rhand_smplx_ids.npy')
        lhand_idx = np.load('./exp/lhand_smplx_ids.npy')
    except FileNotFoundError:
        # If hand indices not available, return zero loss
        return torch.tensor(0.0, device=verts.device)
    
    # Convert to tensors and move to device
    rhand_idx = torch.from_numpy(rhand_idx).to(verts.device)
    lhand_idx = torch.from_numpy(lhand_idx).to(verts.device)
    
    T = verts.shape[0]
    total_penetration_loss = 0.0
    total_fixed_frames = 0
    
    for t in range(T):
        # Check if any joints were fixed in this frame
        if fixed_joints_mask[t].any():
            total_fixed_frames += 1
            
            # Get hand vertices for this frame
            left_hand_verts = verts[t, lhand_idx, :]  # (L, 3)
            right_hand_verts = verts[t, rhand_idx, :]  # (R, 3)
            
            # Compute distances from hand vertices to object
            left_rel_to_obj = verts_obj_transformed[t] - left_hand_verts.unsqueeze(1)  # (L, N, 3)
            right_rel_to_obj = verts_obj_transformed[t] - right_hand_verts.unsqueeze(1)  # (R, N, 3)
            
            left_dists = left_rel_to_obj.norm(dim=2)  # (L, N)
            right_dists = right_rel_to_obj.norm(dim=2)  # (R, N)
            
            # Find minimum distance for each hand vertex
            left_min_dists, _ = left_dists.min(dim=1)  # (L,)
            right_min_dists, _ = right_dists.min(dim=1)  # (R,)
            
            # Penetration loss: penalize negative distances (vertices inside object)
            left_penetration = torch.clamp(-left_min_dists, min=0)  # (L,) - only negative distances become positive
            right_penetration = torch.clamp(-right_min_dists, min=0)  # (R,) - only negative distances become positive
            
            # Sum penetration losses for this frame
            frame_penetration_loss = left_penetration.sum() + right_penetration.sum()
            total_penetration_loss += frame_penetration_loss
    
    if total_fixed_frames == 0:
        return torch.tensor(0.0, device=verts.device)
    
    # Return average penetration loss per fixed frame
    return total_penetration_loss / total_fixed_frames

def compute_finger_distance_loss_selective(joints, verts_obj_transformed, contact_mask, fixed_joints_mask, is_left_hand=True, joint_optimization_mask=None):
    """
    Compute finger distance loss: penalize when pinky and index finger distances to object are different during contact.
    Only applies to frames where joints were fixed.
    
    Args:
        joints: Joint positions (T, J, 3)
        verts_obj_transformed: Object vertices (T, M, 3)
        contact_mask: Boolean mask (T,) indicating when hand is in contact
        fixed_joints_mask: (T, 8) boolean mask indicating which joint-frame combinations were fixed
        is_left_hand: Whether computing for left or right hand
    
    Returns:
        distance_loss: Loss penalizing finger distance differences
    """
    # Define finger end joint indices
    if is_left_hand:
        IDX_INDEX_END = 25  # Left index finger end joint
        IDX_PINKY_END = 31  # Left pinky finger end joint
        joint_idx_in_mask = 6  # left_wrist = index 6 in fixed_joints_mask
    else:
        IDX_INDEX_END = 40  # Right index finger end joint
        IDX_PINKY_END = 46  # Right pinky finger end joint
        joint_idx_in_mask = 7  # right_wrist = index 7 in fixed_joints_mask
    
    # Extract finger end joint positions
    index_end = joints[:, IDX_INDEX_END, :]  # (T, 3)
    pinky_end = joints[:, IDX_PINKY_END, :]  # (T, 3)
    
    # Compute distances from finger end joints to nearest object point
    index_rel_to_obj = verts_obj_transformed - index_end.unsqueeze(1)  # (T, N, 3)
    pinky_rel_to_obj = verts_obj_transformed - pinky_end.unsqueeze(1)  # (T, N, 3)
    
    index_dists = index_rel_to_obj.norm(dim=2)  # (T, N)
    pinky_dists = pinky_rel_to_obj.norm(dim=2)  # (T, N)
    
    index_min_dist, _ = index_dists.min(dim=1)  # (T,)
    pinky_min_dist, _ = pinky_dists.min(dim=1)  # (T,)
    
    # Compute distance differences
    finger_dist_diff = torch.abs(index_min_dist - pinky_min_dist)  # (T,)
    
    # Only apply loss when hand is in contact AND the wrist joint was fixed in this frame
    # AND only if this joint is being optimized
    if joint_optimization_mask is not None and not joint_optimization_mask[joint_idx_in_mask]:
        return torch.tensor(0.0, device=joints.device)
        
    wrist_fixed_mask = fixed_joints_mask[:, joint_idx_in_mask]  # (T,) - which frames had this wrist fixed
    combined_mask = contact_mask & wrist_fixed_mask  # (T,) - both contact AND fixed
    
    if combined_mask.sum() == 0:
        return torch.tensor(0.0, device=joints.device)
    
    # Loss: penalize when finger distances are different during contact
    distance_loss = torch.mean(torch.where(
        combined_mask, 
        finger_dist_diff,  # Direct penalty: larger difference = higher loss
        torch.zeros_like(finger_dist_diff)
    ))
    
    # Additional penalty: penalize total distance to object during contact
    total_distance_penalty = torch.mean(torch.where(
        combined_mask,
        index_min_dist + pinky_min_dist,  # Penalize total distance to object
        torch.zeros_like(index_min_dist)
    ))
    
    return distance_loss + total_distance_penalty

def compute_finger_distance_loss_with_mask(joints, verts_obj_transformed, contact_mask, frame_mask, is_left_hand=True):
    """
    Compute finger distance loss: penalize when pinky and index finger distances to object are different during contact.
    Only applies to frames specified by frame_mask (e.g., frames close to object).
    
    Args:
        joints: Joint positions (T, J, 3)
        verts_obj_transformed: Object vertices (T, M, 3)
        contact_mask: Boolean mask (T,) indicating when hand is in contact
        frame_mask: Boolean mask (T,) indicating which frames to compute loss for
        is_left_hand: Whether computing for left or right hand
    
    Returns:
        distance_loss: Loss penalizing finger distance differences
    """
    # Define finger end joint indices
    if is_left_hand:
        IDX_INDEX_END = 25  # Left index finger end joint
        IDX_PINKY_END = 31  # Left pinky finger end joint
    else:
        IDX_INDEX_END = 40  # Right index finger end joint
        IDX_PINKY_END = 46  # Right pinky finger end joint
    
    # Extract finger end joint positions
    index_end = joints[:, IDX_INDEX_END, :]  # (T, 3)
    pinky_end = joints[:, IDX_PINKY_END, :]  # (T, 3)
    
    # Compute distances from finger end joints to nearest object point
    index_rel_to_obj = verts_obj_transformed - index_end.unsqueeze(1)  # (T, N, 3)
    pinky_rel_to_obj = verts_obj_transformed - pinky_end.unsqueeze(1)  # (T, N, 3)
    
    index_dists = index_rel_to_obj.norm(dim=2)  # (T, N)
    pinky_dists = pinky_rel_to_obj.norm(dim=2)  # (T, N)
    
    index_min_dist, _ = index_dists.min(dim=1)  # (T,)
    pinky_min_dist, _ = pinky_dists.min(dim=1)  # (T,)
    
    # Compute distance differences
    finger_dist_diff = torch.abs(index_min_dist - pinky_min_dist)  # (T,)
    
    # Only apply loss when hand is in contact AND frame is in the specified mask
    combined_mask = contact_mask & frame_mask  # (T,) - both contact AND in frame mask
    
    if combined_mask.sum() == 0:
        return torch.tensor(0.0, device=joints.device)
    
    # Loss: penalize when finger distances are different during contact
    distance_loss = torch.mean(torch.where(
        combined_mask, 
        finger_dist_diff,  # Direct penalty: larger difference = higher loss
        torch.zeros_like(finger_dist_diff)
    ))
    
    # Additional penalty: penalize total distance to object during contact
    total_distance_penalty = torch.mean(torch.where(
        combined_mask,
        index_min_dist + pinky_min_dist,  # Penalize total distance to object
        torch.zeros_like(index_min_dist)
    ))
    
    return distance_loss + total_distance_penalty

def precompute_hand_object_distances(verts, verts_obj_transformed, rhand_idx, lhand_idx):
    """
    Pre-compute hand-object distances for the original sequence to create distance masks.
    This is computed once before optimization to identify frames that are close to the object.
    
    Args:
        verts: Human mesh vertices (T, V, 3) for the original sequence
        verts_obj_transformed: Object vertices (T, N, 3)
        rhand_idx: Right hand vertex indices
        lhand_idx: Left hand vertex indices
    
    Returns:
        dict: Contains distance masks and contact information for both hands
    """
    if rhand_idx is None or lhand_idx is None:
        print("Warning: Hand indices not available, skipping distance pre-computation")
        return None
    
    # Ensure all tensors are on the same device
    device = verts.device
    verts_obj_transformed = verts_obj_transformed.to(device)
    
    T = verts.shape[0]
    
    # Compute distances for right hand using point2point_signed (same as optimize.py)
    right_hand_verts = verts[:, rhand_idx, :]  # (T, R, 3) where R = 778
    # Use point2point_signed to get signed distances efficiently
    _, right_signed_distances, _, _, _, _ = point2point_signed(right_hand_verts, verts_obj_transformed, return_vector=True)
    right_hand_min_dist = torch.min(right_signed_distances, dim=1)[0]  # (T,) - minimum distance for each frame
    
    # Compute distances for left hand using point2point_signed (same as optimize.py)
    left_hand_verts = verts[:, lhand_idx, :]  # (T, L, 3) where L = 778
    # Use point2point_signed to get signed distances efficiently
    _, left_signed_distances, _, _, _, _ = point2point_signed(left_hand_verts, verts_obj_transformed, return_vector=True)
    left_hand_min_dist = torch.min(left_signed_distances, dim=1)[0]  # (T,) - minimum distance for each frame
    
    # Create distance masks (frames that are close to object)
    contact_thresh = 0.02  # 2cm threshold for contact
    close_thresh = 0.20    # 20cm threshold for "close" frames
    
    right_contact_mask = right_hand_min_dist <= contact_thresh
    left_contact_mask = left_hand_min_dist <= contact_thresh
    
    right_close_mask = right_hand_min_dist <= close_thresh
    left_close_mask = left_hand_min_dist <= close_thresh
    
    # Create optimization masks (frames that should be optimized for penetration)
    right_optimize_mask = right_hand_min_dist <= close_thresh
    left_optimize_mask = left_hand_min_dist <= close_thresh
    
    print(f"Distance pre-computation results:")
    print(f"  Right hand: {right_contact_mask.sum().item()}/{T} contact frames, {right_close_mask.sum().item()}/{T} close frames")
    print(f"  Left hand: {left_contact_mask.sum().item()}/{T} contact frames, {left_close_mask.sum().item()}/{T} close frames")
    
    return {
        'right_contact_mask': right_contact_mask,
        'left_contact_mask': left_contact_mask,
        'right_close_mask': right_close_mask,
        'left_close_mask': left_close_mask,
        'right_optimize_mask': right_optimize_mask,
        'left_optimize_mask': left_optimize_mask,
        'right_hand_min_dist': right_hand_min_dist,
        'left_hand_min_dist': left_hand_min_dist
    }

def compute_penetration_loss_with_masks(verts, verts_obj_transformed, fixed_joints_mask, 
                                       distance_info, rhand_idx, lhand_idx, 
                                       RHAND_INDEXES_DETAILED, LHAND_INDEXES_DETAILED):
    """
    Compute penetration loss treating hands as whole body parts.
    This function is called during optimization to compute penetration loss for the entire sequence.
    
    Args:
        verts: Current human mesh vertices (T, V, 3)
        verts_obj_transformed: Object vertices (T, N, 3)
        fixed_joints_mask: (T, 8) boolean mask indicating which joint-frame combinations were fixed
        distance_info: Pre-computed distance information from precompute_hand_object_distances
        rhand_idx: Right hand vertex indices
        lhand_idx: Left hand vertex indices
        RHAND_INDEXES_DETAILED: Right hand detailed indices
        LHAND_INDEXES_DETAILED: Left hand detailed indices
    
    Returns:
        penetration_loss: Loss penalizing hand-object penetration
    """
    if (rhand_idx is None or lhand_idx is None):
        return torch.tensor(0.0, device=verts.device)
    
    T = verts.shape[0]
    thresh = 0.00  # Penetration threshold (same as optimize.py)
    
    total_penetration_loss = torch.tensor(0.0, device=verts.device)
    
    # Process right hand - compute for all frames
    if rhand_idx is not None:
        # print("Computing right hand penetration loss")
        right_hand_verts = verts[:, rhand_idx, :]  # (T, R, 3)
        
        # Recalculate point2point distances every epoch
        # Ensure both tensors are on the same device
        device = right_hand_verts.device
        verts_obj_transformed_device = verts_obj_transformed.to(device)
        o2h_signed, sbj2obj, o2h_idx, sbj2obj_idx, o2h, sbj2obj_vector = point2point_signed(
            right_hand_verts, verts_obj_transformed_device, return_vector=True
        )
        
        # Apply penetration loss for all frames - treat entire hand as one unit
        MASK_I = sbj2obj < thresh  # Penetration mask for entire hand
        whether_pene_time = torch.sum(MASK_I, dim=-1)  # (T,) - count penetrating vertices per frame
        MASK_TIME = (whether_pene_time > 0)  # (T,) - frames with any penetration
        # print(MASK_TIME.sum())
        # Apply loss to all frames with penetration
        if MASK_TIME.sum() > 0:
            sd_pen = sbj2obj[MASK_TIME]  # Get all penetrating distances
            calc_dist = sd_pen - thresh
            mask_pen = ((sd_pen - thresh) < 0).float() * (torch.abs(sd_pen) < 0.03).float()
            num_pen = torch.sum(mask_pen, dim=-1).reshape(-1, 1)
            
            if num_pen.sum() > 0:
                penetration_loss_part = torch.sum(torch.abs(calc_dist) * mask_pen / (num_pen + 1e-8))
                total_penetration_loss += penetration_loss_part * 2.0  # Same weight as optimize.py
    
    # Process left hand - compute for all frames
    if lhand_idx is not None:
        # print("Computing left hand penetration loss")
        left_hand_verts = verts[:, lhand_idx, :]  # (T, L, 3)
        
        # Recalculate point2point distances every epoch
        # Ensure both tensors are on the same device
        device = left_hand_verts.device
        verts_obj_transformed_device = verts_obj_transformed.to(device)
        o2h_signed, sbj2obj, o2h_idx, sbj2obj_idx, o2h, sbj2obj_vector = point2point_signed(
            left_hand_verts, verts_obj_transformed_device, return_vector=True
        )
        
        # Apply penetration loss for all frames - treat entire hand as one unit
        MASK_I = sbj2obj < thresh  # Penetration mask for entire hand
        whether_pene_time = torch.sum(MASK_I, dim=-1)  # (T,) - count penetrating vertices per frame
        MASK_TIME = (whether_pene_time > 0)  # (T,) - frames with any penetration
        # print(MASK_TIME.sum())
        # Apply loss to all frames with penetration
        if MASK_TIME.sum() > 0:
            sd_pen = sbj2obj[MASK_TIME]  # Get all penetrating distances
            calc_dist = sd_pen - thresh
            mask_pen = ((sd_pen - thresh) < 0).float() * (torch.abs(sd_pen) < 0.03).float()
            num_pen = torch.sum(mask_pen, dim=-1).reshape(-1, 1)
            
            if num_pen.sum() > 0:
                penetration_loss_part = torch.sum(torch.abs(calc_dist) * mask_pen / (num_pen + 1e-8))
                total_penetration_loss += penetration_loss_part * 2.0  # Same weight as optimize.py
    
    return total_penetration_loss

def optimize_poses_with_fixed_tracking(poses, betas, trans, gender, verts_obj_transformed, 
                                     fix_tracker, distance_info, rhand_idx, lhand_idx,
                                     RHAND_INDEXES_DETAILED, LHAND_INDEXES_DETAILED,
                                     num_epochs=500, lr=0.01):
    """Optimize only specific joints in specific frames that were fixed.
    
    This function optimizes ALL joint poses (all 8 joints × 3 parameters = 24 parameters) 
    for ALL frames, but the loss functions are designed to only affect the fixed 
    joint-frame combinations. The temporal smoothing and reference losses only compute
    gradients for the specific frames and joints that were marked as fixed in previous steps.
    
    Key features:
    1. Non-fixed frames are detached to prevent gradient flow
    2. Temporal smoothing computes velocity/acceleration between consecutive frames
    3. Only fixed joint-frame combinations contribute to the loss gradients
    """
    
    # Get which joints were fixed in which frames
    fixed_joints_mask = fix_tracker.get_fixed_frames_and_joints()  # Shape: (T, 8)
    # Convert to tensor and move to device
    fixed_joints_mask = torch.from_numpy(fixed_joints_mask).bool().to(device)
    
    # Extract all joint poses for optimization (collar, shoulder, elbow, wrist joints only)
    # Joint to pose mapping:
    # Joint index 13 (left collar) → pose index 39:42
    # Joint index 14 (right collar) → pose index 42:45
    # Joint index 16 (left shoulder) → pose index 48:51
    # Joint index 17 (right shoulder) → pose index 51:54
    # Joint index 18 (left elbow) → pose index 54:57
    # Joint index 19 (right elbow) → pose index 57:60
    # Joint index 20 (left wrist) → pose index 60:63
    # Joint index 21 (right wrist) → pose index 63:66
    
    # Extract all joint poses into one tensor
    all_joint_poses = np.concatenate([
        poses[:, 39:42],   # Left collar (joint 13)
        poses[:, 42:45],   # Right collar (joint 14) 
        poses[:, 48:51],   # Left shoulder (joint 16)
        poses[:, 51:54],   # Right shoulder (joint 17)
        poses[:, 54:57],   # Left elbow (joint 18)
        poses[:, 57:60],   # Right elbow (joint 19)
        poses[:, 60:63],   # Left wrist (joint 20)
        poses[:, 63:66],   # Right wrist (joint 21)
    ], axis=1)  # Shape: (T, 24) - 8 joints × 3 parameters each
    
    # Convert to tensor for optimization
    all_joint_poses_tensor = torch.from_numpy(all_joint_poses).float().to(device).requires_grad_(True)
    
    # Create reference poses (original poses)
    reference_all_joint_poses = torch.from_numpy(all_joint_poses).float().to(device)
    
    # Create a mask tensor that indicates which joints should be optimized
    # We want to optimize ALL frames for joints that were fixed in previous steps
    joint_optimization_mask = torch.zeros(8, dtype=torch.bool)  # Which joints to optimize
    for joint_idx in range(8):  # 8 joints
        # Mark joints that were fixed in any frame
        if fixed_joints_mask[:, joint_idx].any():  # If any frame was fixed for this joint
            joint_optimization_mask[joint_idx] = True  # Mark this joint for optimization
    
    # Create optimization mask for all poses: ALL frames for fixed joints
    optimization_mask = torch.zeros_like(all_joint_poses_tensor, dtype=torch.bool)
    for joint_idx in range(8):  # 8 joints
        pose_start = joint_idx * 3
        pose_end = pose_start + 3
        # Mark ALL frames for joints that were fixed
        if joint_optimization_mask[joint_idx]:
            optimization_mask[:, pose_start:pose_end] = True  # Mark all frames for this joint
    
    print(f"Optimizing joints: {[i for i, x in enumerate(joint_optimization_mask) if x]}")
    print(f"Joint names: ['left_collar', 'right_collar', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']")
    
    # Optimizer for all joint poses
    optimizer = torch.optim.Adam([all_joint_poses_tensor], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=500, verbose=True)
    
    # Count how many joint-frame combinations need optimization
    total_optimizations = fixed_joints_mask.sum()
    print(f"Optimizing {total_optimizations} joint-frame combinations over {num_epochs} epochs")
    
    # Early stopping setup
    best_loss = float('inf')
    best_poses = None
    patience_counter = 0
    patience = 200
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Reconstruct full poses for SMPL computation
        # Only update the fixed joint-frame combinations, preserve original poses for non-fixed ones
        poses_reconstructed = poses.copy()
        
        # Create a tensor that combines optimized poses (for fixed frames) and original poses (for non-fixed frames)
        combined_poses = all_joint_poses_tensor.clone()
        # For non-fixed frames, use the original reference poses
        combined_poses[~optimization_mask] = reference_all_joint_poses[~optimization_mask]
        
        # Create poses tensor directly from combined_poses to maintain gradients
        poses_tensor = torch.zeros((poses_reconstructed.shape[0], poses_reconstructed.shape[1]), 
                                  dtype=torch.float32, device=device)
        
        # Copy original poses first
        poses_tensor = torch.from_numpy(poses_reconstructed).float().to(device)
        
        # Then update with optimized poses (maintaining gradients for optimized parts)
        poses_tensor[:, 39:42] = combined_poses[:, :3]      # Left collar
        poses_tensor[:, 42:45] = combined_poses[:, 3:6]     # Right collar
        poses_tensor[:, 48:51] = combined_poses[:, 6:9]     # Left shoulder
        poses_tensor[:, 51:54] = combined_poses[:, 9:12]    # Right shoulder
        poses_tensor[:, 54:57] = combined_poses[:, 12:15]  # Left elbow
        poses_tensor[:, 57:60] = combined_poses[:, 15:18]  # Right elbow
        poses_tensor[:, 60:63] = combined_poses[:, 18:21]  # Left wrist
        poses_tensor[:, 63:66] = combined_poses[:, 21:24]  # Right wrist
        frame_times = poses_reconstructed.shape[0]
        
        # Use SMPLX16 model (same as two_stage_wrist_optimize.py)
        model = sbj_m_all[gender]
        
        # SMPLX16 format
        output = model(
            pose_body=poses_tensor[:, 3:66],
            pose_hand=poses_tensor[:, 66:156],
            betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float().to(device),
            root_orient=poses_tensor[:, :3],
            trans=torch.from_numpy(trans).float().to(device)
        )
        joints = output.Jtr
        
        # Compute smooth loss only between fixed frames and their neighbors (for continuity)
        # Only for joints that were fixed in previous steps
        smooth_loss = compute_temporal_smoothing_loss_fixed_frames_only(all_joint_poses_tensor, fixed_joints_mask, joint_optimization_mask)
        ref_loss = compute_reference_loss_selective(all_joint_poses_tensor, reference_all_joint_poses, fixed_joints_mask)
        # ref_loss = torch.tensor(0.0, device=all_joint_poses_tensor.device)
        
        # Compute contact and orientation masks for palm-related losses
        contact_mask_l, orient_mask_l, _ = compute_palm_contact_and_orientation(
            joints, verts_obj_transformed, hand='left'
        )
        contact_mask_r, orient_mask_r, _ = compute_palm_contact_and_orientation(
            joints, verts_obj_transformed, hand='right'
        )
        
        # Compute palm orientation losses (only for frames close to object AND with fixed joints)
        # Only for joints that were fixed in previous steps
        left_palm_loss = compute_palm_facing_loss_selective(
            joints, verts_obj_transformed, contact_mask_l, fixed_joints_mask, is_left_hand=True, joint_optimization_mask=joint_optimization_mask
        )
        right_palm_loss = compute_palm_facing_loss_selective(
            joints, verts_obj_transformed, contact_mask_r, fixed_joints_mask, is_left_hand=False, joint_optimization_mask=joint_optimization_mask
        )
        palm_loss = left_palm_loss + right_palm_loss
        
        # Compute penetration loss using pre-computed distance masks (only for frames close to object)
        penetration_loss = compute_penetration_loss_with_masks(
            output.v, verts_obj_transformed, fixed_joints_mask, 
            distance_info, rhand_idx, lhand_idx, 
            RHAND_INDEXES_DETAILED, LHAND_INDEXES_DETAILED
        )
        
        # Compute finger distance losses (only for frames close to object AND with fixed joints)
        # Use distance masks to only compute for frames that are close to the object
        if distance_info is not None:
            # Create combined mask: frames that are both close to object AND have fixed joints
            left_finger_mask = distance_info['left_close_mask'] & fixed_joints_mask.any(dim=1)
            right_finger_mask = distance_info['right_close_mask'] & fixed_joints_mask.any(dim=1)
            
            left_finger_loss = compute_finger_distance_loss_with_mask(
                joints, verts_obj_transformed, contact_mask_l, left_finger_mask, is_left_hand=True
            )
            right_finger_loss = compute_finger_distance_loss_with_mask(
                joints, verts_obj_transformed, contact_mask_r, right_finger_mask, is_left_hand=False
            )
        else:
            # Fallback to original method if distance info not available
            # Only for joints that were fixed in previous steps
            left_finger_loss = compute_finger_distance_loss_selective(
                joints, verts_obj_transformed, contact_mask_l, fixed_joints_mask, is_left_hand=True, joint_optimization_mask=joint_optimization_mask
            )
            right_finger_loss = compute_finger_distance_loss_selective(
                joints, verts_obj_transformed, contact_mask_r, fixed_joints_mask, is_left_hand=False, joint_optimization_mask=joint_optimization_mask
            )
        
        finger_loss = left_finger_loss + right_finger_loss
        
        # Ensure all losses are tensors on the correct device
        smooth_loss = smooth_loss.to(all_joint_poses_tensor.device)
        palm_loss = palm_loss.to(all_joint_poses_tensor.device)
        penetration_loss = penetration_loss.to(all_joint_poses_tensor.device)
        finger_loss = finger_loss.to(all_joint_poses_tensor.device)
        
        # Combine all losses with appropriate weights
        total_loss = 10.0 * smooth_loss + 2.0 * ref_loss +  palm_loss + 2.0 * penetration_loss + finger_loss
        # total_loss =  finger_loss
        # Check for improvement
        if total_loss < best_loss - 1e-6:
            best_loss = total_loss
            best_poses = all_joint_poses_tensor.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 0
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: smooth={smooth_loss.item():.6f}, ref={ref_loss.item():.6f}, palm={palm_loss.item():.6f}, penetration={penetration_loss.item():.6f}, finger={finger_loss.item():.6f},  total={total_loss.item():.6f}")
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([all_joint_poses_tensor], max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss)
    
    # Reconstruct final poses using best poses if available
    final_poses = poses.copy()
    if best_poses is not None:
        poses_to_use = best_poses
        print(f"Using best poses with loss: {best_loss:.6f}")
    else:
        poses_to_use = all_joint_poses_tensor
        print("Using final poses (no improvement detected)")
    
    # Create final combined poses: optimized for fixed frames, original for non-fixed frames
    final_combined_poses = poses_to_use.clone()
    final_combined_poses[~optimization_mask] = reference_all_joint_poses[~optimization_mask]
    
    final_poses[:, 39:42] = final_combined_poses[:, :3].detach().cpu().numpy()      # Left collar
    final_poses[:, 42:45] = final_combined_poses[:, 3:6].detach().cpu().numpy()     # Right collar
    final_poses[:, 48:51] = final_combined_poses[:, 6:9].detach().cpu().numpy()     # Left shoulder
    final_poses[:, 51:54] = final_combined_poses[:, 9:12].detach().cpu().numpy()    # Right shoulder
    final_poses[:, 54:57] = final_combined_poses[:, 12:15].detach().cpu().numpy()  # Left elbow
    final_poses[:, 57:60] = final_combined_poses[:, 15:18].detach().cpu().numpy()  # Right elbow
    final_poses[:, 60:63] = final_combined_poses[:, 18:21].detach().cpu().numpy()  # Left wrist
    final_poses[:, 63:66] = final_combined_poses[:, 21:24].detach().cpu().numpy()  # Right wrist
    
    # Analyze pose changes before and after optimization
    analyze_pose_changes(poses, final_poses, fixed_joints_mask, joint_optimization_mask)
    
    return final_poses

def analyze_pose_changes(original_poses, optimized_poses, fixed_joints_mask, joint_optimization_mask):
    """
    Analyze and report changes in poses before and after optimization.
    
    Args:
        original_poses: (T, 156) - Original poses before optimization
        optimized_poses: (T, 156) - Poses after optimization
        fixed_joints_mask: (T, 8) - Which joint-frame combinations were fixed
        joint_optimization_mask: (8,) - Which joints were optimized
    """
    print("\n" + "="*80)
    print("POSE CHANGE ANALYSIS")
    print("="*80)
    
    # Joint names and their pose indices
    joint_info = [
        ("left_collar", 39, 42),
        ("right_collar", 42, 45),
        ("left_shoulder", 48, 51),
        ("right_shoulder", 51, 54),
        ("left_elbow", 54, 57),
        ("right_elbow", 57, 60),
        ("left_wrist", 60, 63),
        ("right_wrist", 63, 66)
    ]
    
    # Calculate changes for each joint
    total_changes = 0
    significant_changes = 0
    threshold = 0.01  # 0.01 radians ≈ 0.57 degrees
    
    for joint_idx, (joint_name, start_idx, end_idx) in enumerate(joint_info):
        if not joint_optimization_mask[joint_idx]:
            print(f"{joint_name:15s}: NOT OPTIMIZED (was never fixed)")
            continue
            
        # Extract pose parameters for this joint
        original_joint_poses = original_poses[:, start_idx:end_idx]  # (T, 3)
        optimized_joint_poses = optimized_poses[:, start_idx:end_idx]  # (T, 3)
        
        # Calculate absolute differences
        pose_diff = np.abs(optimized_joint_poses - original_joint_poses)  # (T, 3)
        max_diff = np.max(pose_diff)
        mean_diff = np.mean(pose_diff)
        
        # Count frames with significant changes
        significant_frames = np.sum(pose_diff > threshold, axis=1)  # (T,) - count per frame
        frames_with_changes = np.sum(significant_frames > 0)  # Total frames with any significant change
        
        # Check which frames were actually fixed
        fixed_frames_for_joint = np.where(fixed_joints_mask[:, joint_idx].cpu().numpy())[0]
        num_fixed_frames = len(fixed_frames_for_joint)
        
        print(f"{joint_name:15s}:")
        print(f"  {'':15s}  Fixed frames: {num_fixed_frames}")
        print(f"  {'':15s}  Max change: {max_diff:.6f} rad ({np.degrees(max_diff):.2f}°)")
        print(f"  {'':15s}  Mean change: {mean_diff:.6f} rad ({np.degrees(mean_diff):.2f}°)")
        print(f"  {'':15s}  Frames with changes: {frames_with_changes}/{original_poses.shape[0]}")
        
        if frames_with_changes > 0:
            total_changes += frames_with_changes
            if max_diff > threshold:
                significant_changes += 1
    
    print(f"\nSUMMARY:")
    print(f"  Total frames with changes: {total_changes}")
    print(f"  Joints with significant changes: {significant_changes}/{joint_optimization_mask.sum()}")
    print(f"  Optimization threshold: {threshold:.3f} rad ({np.degrees(threshold):.1f}°)")
    
    # Check if any poses were actually changed
    if total_changes == 0:
        print("  ⚠️  WARNING: No pose changes detected! Check if optimization is working.")
    else:
        print(f"  ✅  Optimization successful: {total_changes} pose changes detected.")
    
    print("="*80 + "\n")

def quick_pose_comparison(original_poses, optimized_poses, joint_names=None):
    """
    Quick comparison of poses before and after optimization.
    
    Args:
        original_poses: (T, 156) - Original poses
        optimized_poses: (T, 156) - Optimized poses
        joint_names: List of joint names to check (default: all 8 joints)
    
    Returns:
        dict: Summary of changes
    """
    if joint_names is None:
        joint_names = ['left_collar', 'right_collar', 'left_shoulder', 'right_shoulder', 
                      'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
    
    # Joint pose indices
    joint_indices = {
        'left_collar': (39, 42),
        'right_collar': (42, 45),
        'left_shoulder': (48, 51),
        'right_shoulder': (51, 54),
        'left_elbow': (54, 57),
        'right_elbow': (57, 60),
        'left_wrist': (60, 63),
        'right_wrist': (63, 66)
    }
    
    changes = {}
    total_max_change = 0.0
    
    for joint_name in joint_names:
        if joint_name in joint_indices:
            start_idx, end_idx = joint_indices[joint_name]
            
            # Extract poses for this joint
            orig_joint = original_poses[:, start_idx:end_idx]
            opt_joint = optimized_poses[:, start_idx:end_idx]
            
            # Calculate differences
            diff = np.abs(opt_joint - orig_joint)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            changes[joint_name] = {
                'max_change': max_diff,
                'mean_change': mean_diff,
                'max_change_deg': np.degrees(max_diff),
                'mean_change_deg': np.degrees(mean_diff)
            }
            
            total_max_change = max(total_max_change, max_diff)
    
    # Print summary
    print(f"\nQuick Pose Comparison:")
    print(f"{'Joint':<15} {'Max Change':<12} {'Mean Change':<12}")
    print("-" * 45)
    
    for joint_name in joint_names:
        if joint_name in changes:
            change = changes[joint_name]
            print(f"{joint_name:<15} {change['max_change_deg']:<12.2f}° {change['mean_change_deg']:<12.2f}°")
    
    print(f"\nOverall max change: {np.degrees(total_max_change):.2f}°")
    
    return changes

def visualize_smpl(name, MOTION_PATH, model_type, num_betas, use_pca=False):
    """Load SMPL data"""
    with np.load(os.path.join(MOTION_PATH, name, 'human.npz'), allow_pickle=True) as f:
        poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])
    
    frame_times = poses.shape[0]
    
    if num_betas == 10:
        if model_type == 'smplh':
            smpl_model = smplh10[gender]
            smplx_output = smpl_model(
                body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                global_orient=torch.from_numpy(poses[:, :3]).float(),
                left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
                right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
                betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
                transl=torch.from_numpy(trans).float()
            )
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
            # Use SMPLX16 model (same as two_stage_wrist_optimize.py)
            smpl_model = sbj_m_all[gender]
        smplx_output = smpl_model(
            pose_body=torch.from_numpy(poses[:, 3:66]).float().to(device),
            pose_hand=torch.from_numpy(poses[:, 66:156]).float().to(device),
            betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float().to(device),
            root_orient=torch.from_numpy(poses[:, :3]).float().to(device),
            trans=torch.from_numpy(trans).float().to(device)
        )
        verts = to_cpu(smplx_output.v)
        joints = to_cpu(smplx_output.Jtr)
        faces = smpl_model.f.unsqueeze(0).repeat(frame_times, 1, 1)

    return verts, joints, faces, poses, betas, trans, gender

def get_mean_pose_joints(name, gender, model_type, num_betas, use_pca=False):
    """Get canonical joint positions - Only supports SMPLX16 for OMOMO"""
    frame_times = 1
    pose_zeros = torch.zeros(frame_times, 156).float()
    trans_zeros = torch.zeros(frame_times, 3).float()
    betas_zeros = torch.zeros(frame_times, num_betas).float()

    if num_betas == 10:
        if model_type == 'smplh':
            smpl_model = smplh10[gender]
        elif model_type == 'smplx':
            if use_pca:
                smpl_model = smplx12[gender]
            else:
                smpl_model = smplx10[gender]
        output = smpl_model(
            body_pose=pose_zeros[:, 3:66],
            global_orient=pose_zeros[:, :3],
            left_hand_pose=pose_zeros[:, 66:111],
            right_hand_pose=pose_zeros[:, 111:156],
            transl=trans_zeros,
            betas=betas_zeros
        )
        joints = output.joints[0].detach().cpu().numpy()
    else:
        if model_type == 'smplh':
            smpl_model = smplh16[gender]
        elif model_type == 'smplx':
            smpl_model = smplx16[gender]
        output = smpl_model(
            pose_body=pose_zeros[:, 3:66],
            pose_hand=pose_zeros[:, 66:156],
            root_orient=pose_zeros[:, :3],
            trans=trans_zeros,
            betas=betas_zeros
        )
        joints = output.Jtr[0].detach().cpu().numpy()

    return joints

def main(dataset_path, sequence_name):
    """Main pipeline function - Supports all datasets for Steps 1&2, SMPLX16 for optimization"""
    print(f"Starting comprehensive pipeline for {sequence_name}")
    
    # Derived paths
    human_path = os.path.join(dataset_path, 'sequences_canonical')
    object_path = os.path.join(dataset_path, 'objects')
    dataset_path_name = dataset_path.split('/')[-1]

    # Load data based on dataset
    if dataset_path_name.upper() == 'BEHAVE' or dataset_path_name.upper() == 'BEHAVE_CORRECT':
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(sequence_name, human_path, 'smplh', 10)
        canonical_joints = get_mean_pose_joints(sequence_name, gender, 'smplh', 10)
    elif dataset_path_name.upper() == 'NEURALDOME' or dataset_path_name.upper() == 'IMHD':
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(sequence_name, human_path, 'smplh', 16)
        canonical_joints = get_mean_pose_joints(sequence_name, gender, 'smplh', 16)
    elif dataset_path_name.upper() == 'CHAIRS':
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(sequence_name, human_path, 'smplx', 10)
        canonical_joints = get_mean_pose_joints(sequence_name, gender, 'smplx', 10)
    elif dataset_path_name.upper() == 'INTERCAP' or dataset_path_name.upper() == 'INTERCAP_CORRECT':
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(sequence_name, human_path, 'smplx', 10, True)
        canonical_joints = get_mean_pose_joints(sequence_name, gender, 'smplx', 10, True)
    elif dataset_path_name.upper() == 'OMOMO' or dataset_path_name.upper() == 'OMOMO_CORRECT':
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(sequence_name, human_path, 'smplx', 16)
        canonical_joints = get_mean_pose_joints(sequence_name, gender, 'smplx', 16)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_path_name}")

    # Load object data
    with np.load(os.path.join(human_path, sequence_name, 'object.npz'), allow_pickle=True) as f:
        obj_angles, obj_trans, obj_name = f['angles'], f['trans'], str(f['name'])
    angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()

    OBJ_MESH = trimesh.load(os.path.join(object_path, obj_name, obj_name+'.obj'))
    # For now, create a mock object mesh
    # OBJ_MESH = type('MockMesh', (), {'vertices': np.zeros((100, 3)), 'faces': np.zeros((50, 3), dtype=np.int32)})()
    print("object name:", obj_name)
    ov = np.array(OBJ_MESH.vertices).astype(np.float32)
    object_faces = OBJ_MESH.faces.astype(np.int32)
    device = torch.device('cuda:0')
    ov = torch.from_numpy(ov).float().to(device)
    rot = torch.tensor(angle_matrix).float().to(device)
    obj_trans = torch.tensor(obj_trans).float().to(device)
    object_verts = torch.einsum('ni,tij->tnj', ov, rot.permute(0,2,1)) + obj_trans.unsqueeze(1)

    T = poses.shape[0]
    print(f"Processing {T} frames")
            
    # Pre-compute hand-object distances for the original sequence to create distance masks
    print("Pre-computing hand-object distances for distance-based optimization...")
    # Ensure verts is on the same device as object_verts
    if isinstance(verts, torch.Tensor):
        verts = verts.to(device)
    else:
        verts = torch.from_numpy(verts).float().to(device)
    distance_info = precompute_hand_object_distances(verts, object_verts, rhand_idx, lhand_idx)
    print("left_hand_min_dist:", distance_info['left_hand_min_dist'])
    print("right_hand_min_dist:", distance_info['right_hand_min_dist'])
    if distance_info is not None:
        print("Distance masks computed successfully - using distance-limited penetration loss")
    else:
        print("Distance masks not available - using standard penetration loss")
        

    # Initialize fix tracker
    fix_tracker = FixTracker(T)

    # Step 1: Joint pose fixing (exact implementation from joint_pose_fix.py)
    print("\n=== Step 1: Joint Pose Fixing ===")
    
    # Use the exact joint pose fixing logic from joint_pose_fix.py
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
    
    # Track which frames were fixed for each joint using exact information from fix_joint_poses
    joint_fixed_frames = {joint_idx: [] for joint_idx in [LEFT_COLLAR, RIGHT_COLLAR, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]}
    
    # Iteratively fix all joints together (exact logic from joint_pose_fix.py)
    iteration = 0
    prev_total_flips = float('inf')
    
    while True:
        iteration += 1
        print(f"\nIteration {iteration} - Processing all joints together...")
        
        # Track total flips across all joints
        total_flips = 0
        joint_flip_counts = {}
        
        # Process each joint in this iteration
        for joint_idx in [LEFT_COLLAR, RIGHT_COLLAR, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]:
            # Determine which orientation mask to use based on the joint
            if joint_idx in [LEFT_COLLAR, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]:
                orient_mask = orient_mask_l
            else:  # RIGHT_COLLAR, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
                orient_mask = orient_mask_r
            
            # Detect flips for this joint
            flip_indices, angle_diffs, relative_axes = detect_flips(poses, joint_idx, threshold=20)
            current_flip_count = len(flip_indices)
            joint_flip_counts[joint_idx] = current_flip_count
            total_flips += current_flip_count
            
            if current_flip_count > 0:
                print(f"  Joint {joint_idx}: {current_flip_count} flips at frames: {flip_indices}")
                
                # Fix the poses for this joint using exact logic from joint_pose_fix.py
                poses, fixed_frames = fix_joint_poses(
                    poses, flip_indices, orient_mask, 
                    joint_idx, angle_diffs, relative_axes, canonical_joints, 20,
                    left_twist_bound_mask, right_twist_bound_mask
                )
                
                # Track which frames were actually fixed for this joint using the exact information returned
                joint_fixed_frames[joint_idx].extend(fixed_frames)
                # Ensure unique frame indices (in case the same joint is processed multiple times across iterations)
                joint_fixed_frames[joint_idx] = list(set(joint_fixed_frames[joint_idx]))
                print(f"    Joint {joint_idx}: {len(fixed_frames)} frames fixed in this iteration")
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
    for joint_idx in [LEFT_COLLAR, RIGHT_COLLAR, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]:
        flip_indices, _, _ = detect_flips(poses, joint_idx, threshold=20)
        print(f"  Joint {joint_idx}: {len(flip_indices)} flips")
    
    # Track which frames and joints were fixed by joint pose fixing
    joints_to_fix = [LEFT_COLLAR, RIGHT_COLLAR, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]
    joint_to_tracker_idx = {
        LEFT_COLLAR: 0, RIGHT_COLLAR: 1,
        LEFT_SHOULDER: 2, RIGHT_SHOULDER: 3,
        LEFT_ELBOW: 4, RIGHT_ELBOW: 5,
        LEFT_WRIST: 6, RIGHT_WRIST: 7
    }
    
    # Mark the frames that were actually fixed during the joint pose fixing process
    for joint_idx, fixed_frames in joint_fixed_frames.items():
        if fixed_frames:  # Only process joints that had frames fixed
            tracker_joint_idx = joint_to_tracker_idx[joint_idx]
            print(f"Joint {joint_idx}: {len(fixed_frames)} frames fixed")
            fix_tracker.mark_joint_fixed(fixed_frames, [tracker_joint_idx])
    
    total_joint_fixes = fix_tracker.get_joint_fixed_frames_and_joints().sum()
    print(f"Joint pose fixing applied to {total_joint_fixes} joint-frame combinations")
    
    # Save temporary result after Step 1
    render_path = f'./save_pipeline_fix/{dataset_path_name}'
    os.makedirs(render_path, exist_ok=True)
    temp_step1_path = os.path.join(render_path, f'{sequence_name}_step1_joint_fixed.npz')
    np.savez(temp_step1_path, 
             poses=poses, 
             betas=betas, 
             trans=trans, 
             gender=gender)
    print(f"Step 1 temporary result saved to: {temp_step1_path}")

    # Step 2: Palm orientation fixing
    print("\n=== Step 2: Palm Orientation Fixing ===")
    
    # Compute twist angles for palm fixing
    twist_left_list, twist_right_list = detect_hand_twist_from_canonical_batch(poses, canonical_joints)
    
    contact_mask_l, orient_mask_l, _ = compute_palm_contact_and_orientation(
        joints, object_verts, hand='left'
    )
    contact_mask_r, orient_mask_r, _ = compute_palm_contact_and_orientation(
        joints, object_verts, hand='right'
    )

    # Fix left hand
    axis_left = canonical_joints[LEFT_WRIST] - canonical_joints[LEFT_ELBOW]
    axis_left /= np.linalg.norm(axis_left)
    poses, left_fixed_frames = robust_wrist_flip_fix_left(
        twist_left_list, contact_mask_l, orient_mask_l, poses, LEFT_WRIST, axis_left, joints, object_verts
    )
    if left_fixed_frames:
        fix_tracker.mark_palm_fixed(left_fixed_frames, [6])  # left_wrist = index 6

    # Fix right hand
    axis_right = canonical_joints[RIGHT_WRIST] - canonical_joints[RIGHT_ELBOW]
    axis_right /= np.linalg.norm(axis_right)
    poses, right_fixed_frames = robust_wrist_flip_fix_right(
        twist_right_list, contact_mask_r, orient_mask_r, poses, RIGHT_WRIST, axis_right, joints, object_verts
    )
    
    if right_fixed_frames:
        fix_tracker.mark_palm_fixed(right_fixed_frames, [7])  # right_wrist = index 7
    
    # Save temporary result after Step 2
    temp_step2_path = os.path.join(render_path, f'{sequence_name}_step2_palm_fixed.npz')
    np.savez(temp_step2_path, 
             poses=poses, 
             betas=betas, 
             trans=trans, 
             gender=gender)
    print(f"Step 2 temporary result saved to: {temp_step2_path}")

    # Step 3: Optimization with selective loss (uses SMPLX16 model)
    print("\n=== Step 3: Optimization with Selective Loss (SMPLX16) ===")
    all_fixed_joints = fix_tracker.get_fixed_frames_and_joints()
    total_fixed_combinations = all_fixed_joints.sum()
    
    if total_fixed_combinations > 0:
        print(f"Optimizing {total_fixed_combinations} joint-frame combinations using SMPLX16 model")
        
        # Store original poses for comparison
        original_poses = poses.copy()
        
        poses = optimize_poses_with_fixed_tracking(
            poses, betas, trans, gender, object_verts, fix_tracker, 
            distance_info, rhand_idx, lhand_idx, RHAND_INDEXES_DETAILED, LHAND_INDEXES_DETAILED,
            num_epochs=500, lr = 0.001
        )
        
        # Quick comparison of poses before and after optimization
        print("\n" + "="*60)
        print("QUICK POSE COMPARISON (Before vs After Optimization)")
        print("="*60)
        quick_pose_comparison(original_poses, poses)
        
        # Mark only the specific joint-frame combinations that were actually fixed as optimized
        fix_tracker.mark_optimized_from_mask(all_fixed_joints)
    else:
        print("No joint-frame combinations to optimize")
    
    # Save temporary result after Step 3
    temp_step3_path = os.path.join(render_path, f'{sequence_name}_step3_optimized.npz')
    np.savez(temp_step3_path, 
             poses=poses, 
             betas=betas, 
             trans=trans, 
             gender=gender)
    print(f"Step 3 temporary result saved to: {temp_step3_path}")

    # Print summary
    fix_tracker.print_summary()

    # Save final results
    fixed_human_path = os.path.join(human_path, sequence_name, 'pipeline_fixed.npz')
    np.savez(fixed_human_path, 
             poses=poses, 
             betas=betas, 
             trans=trans, 
             gender=gender)
    print(f"Final pipeline results saved to: {fixed_human_path}")

    # Generate visualization
    
    # Regenerate SMPL with fixed poses
    if dataset_path_name.upper() == 'BEHAVE' or dataset_path_name.upper() == 'BEHAVE_CORRECT':
        verts, joints, faces = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplh', 10)
    elif dataset_path_name.upper() == 'NEURALDOME' or dataset_path_name.upper() == 'IMHD':
        verts, joints, faces = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplh', 16)
    elif dataset_path_name.upper() == 'CHAIRS':
        verts, joints, faces = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplx', 10)
    elif dataset_path_name.upper() == 'INTERCAP' or dataset_path_name.upper() == 'INTERCAP_CORRECT':
        verts, joints, faces = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplx', 10, True)
    elif dataset_path_name.upper() == 'OMOMO' or dataset_path_name.upper() == 'OMOMO_CORRECT':
        verts, joints, faces = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplx', 16)

    visualize_body_obj(
        verts.float().detach().cpu().numpy(),
        faces[0].detach().cpu().numpy().astype(np.int32),
        object_verts.detach().cpu().numpy(),
        object_faces,
        save_path=os.path.join(render_path, f'{sequence_name}_pipeline_fixed.mp4'),
        show_frame=True,
        multi_angle=True
    )

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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive pipeline: Joint fix + Palm fix + Optimization")
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset root.")
    parser.add_argument("--sequence_name", required=True, help="Name of the sequence.")
    args = parser.parse_args()

    main(args.dataset_path, args.sequence_name)
