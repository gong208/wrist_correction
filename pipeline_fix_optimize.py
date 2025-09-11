# Comprehensive Pipeline: Joint Pose Fix + Palm Fix + Optimization
import sys
from render.mesh_viz import visualize_body_obj
# from validate_hand_indices import visualize_body_obj
import math
from scipy.spatial.distance import cdist
from utils import vertex_normals
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
from pytorch3d.transforms import matrix_to_euler_angles,axis_angle_to_matrix, matrix_to_axis_angle,rotation_6d_to_matrix,matrix_to_rotation_6d
from torch.autograd import Variable
import torch.optim as optim
import copy
import argparse
from prior import *
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
import pickle
import time
# Joint indices
LEFT_COLLAR = 13
RIGHT_COLLAR = 14
LEFT_SHOULDER = 16
RIGHT_SHOULDER = 17
LEFT_ELBOW = 18
RIGHT_ELBOW = 19
LEFT_WRIST = 20
RIGHT_WRIST = 21
global flag
flag = True
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

hand_prior=HandPrior(prior_path='./assets',device=device)
import open3d as o3d
hand_distance_init=0
gt_rhand=0
whether_all_minus=np.zeros(15)
BIGGEST_VALUE=torch.zeros((15,3)).float().to(device)-99999
SMALLEST_VALUE=torch.zeros((15,3)).float().to(device)+99999
biggest_name=np.array(['B']*45).reshape(15,3)
smallest_name=np.array(['B']*45).reshape(15,3)

rhand_idx = np.load('./exp/rhand_smplx_ids.npy')
lhand_idx = np.load('./exp/lhand_smplx_ids.npy')

HAND_MEAN_TITLE='behave'

## MANO MEAN OF HANDS
rhand_mean=np.load(f'./exp/{HAND_MEAN_TITLE}_rhand_mean.npy')
rhand_mean_torch_single=torch.from_numpy(rhand_mean.reshape(1,-1,3)).float().to(device)
lhand_mean=np.load(F'./exp/{HAND_MEAN_TITLE}_lhand_mean.npy')
lhand_mean_torch_single=torch.from_numpy(lhand_mean.reshape(1,-1,3)).float().to(device)


## HAND INDEX: 778 INDEXES
RHAND_INDEXES_DETAILED=[]
for i in range(5):
    for j in range(3):
        RHAND_INDEXES_DETAILED.append(np.load(f'./exp/index_778/hand_778_{i}_{j}.npy'))
RHAND_INDEXES_DETAILED.append(np.load(f'./exp/index_778/hand_778_5.npy'))

## SMALL: HALF of the HAND, the palms's side
## i means specific finger, j means the specific finger joint
# comment: 为什么会分成small和正常的indexes，区别是什么分别在什么时候用
RHAND_SMALL_INDEXES_DETAILED=[]
for i in range(5):
    for j in range(3):
        RHAND_SMALL_INDEXES_DETAILED.append(np.load(f'./exp/index_778/hand_778_small_{i}_{j}.npy'))
RHAND_SMALL_INDEXES_DETAILED.append(np.load(f'./exp/index_778/hand_778_small_5.npy'))

RHAND_INDEXES=[]
for i in range(6):
    RHAND_INDEXES.append(np.load(f'./exp/index_778/hand_778_{i}.npy'))

RHAND_SMALL_INDEXES=[]
for i in range(6):
    RHAND_SMALL_INDEXES.append(np.load(f'./exp/index_778/hand_778_small_{i}.npy'))

LHAND_INDEXES=[]
for i in range(6):
    LHAND_INDEXES.append(np.load(f'./exp/index_778/lhand_778_{i}.npy'))

LHAND_SMALL_INDEXES=[]
for i in range(6):
    LHAND_SMALL_INDEXES.append(np.load(f'./exp/index_778/lhand_778_small_{i}.npy'))

LHAND_INDEXES_DETAILED=[]
for i in range(5):
    for j in range(3):
        LHAND_INDEXES_DETAILED.append(np.load(f'./exp/index_778/lhand_778_{i}_{j}.npy'))
LHAND_INDEXES_DETAILED.append(np.load(f'./exp/index_778/lhand_778_5.npy'))

LHAND_SMALL_INDEXES_DETAILED=[]
for i in range(5):
    for j in range(3):
        LHAND_SMALL_INDEXES_DETAILED.append(np.load(f'./exp/index_778/lhand_778_small_{i}_{j}.npy'))
LHAND_SMALL_INDEXES_DETAILED.append(np.load(f'./exp/index_778/lhand_778_small_5.npy'))

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

def obj_forward(raw_points, obj_rot_6d, obj_transl):
        # N_points, 3
        # B, 6
        # B, 3
    B = obj_rot_6d.shape[0]
    obj_rot = rotation_6d_to_matrix(obj_rot_6d[:, :]).permute(0, 2, 1)  # B,3,3 don't forget to transpose
    obj_points_pred = torch.matmul(raw_points.unsqueeze(0)[:, :, :3], obj_rot) + obj_transl.unsqueeze(1)
    
    return obj_points_pred

def compute_palm_contact_and_orientation(
    human_joints: torch.Tensor,       # (T, J, 3)
    object_verts: torch.Tensor,       # (T, N, 3)
    hand: str = 'right',              # 'left' 或 'right'
    contact_thresh: float = 0.09,     # 接触距离阈值（单位 m）
    orient_angle_thresh: float = 70.0,# 朝向最大夹角阈值（度），90° 即半球面
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
                    print(f"    Fixing LEFT segment [{seg_start},{seg_end}] by {jump_angle:.1f}° (shorter segment)")
                else:
                    seg_start, seg_end = right_start, right_end
                    jump_angle = -angle_diffs[flip_idx]
                    print(f"    Fixing RIGHT segment [{seg_start},{seg_end}] by {jump_angle:.1f}° (shorter segment)")
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
        #  and is_between_flips and prev_flip is not None and next_flip is not None:
        if segment_length <= 10:
            print(f"    Segment [{seg_start},{seg_end}] (length={segment_length})")
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
    obj_normals: torch.Tensor,        # (T, N, 3)
    hand: str = 'left',               # 'left' or 'right'
    K: int = 500                      # number of closest verts to use
):
    """
    Compute anti-parallel angle between palm normal and average of K closest object normals.
    """
    # pick a common device (use human_joints as reference)
    device = human_joints.device
    human_joints = human_joints.to(device)
    object_verts = object_verts.to(device)
    obj_normals  = obj_normals.to(device)

    T, J, _ = human_joints.shape
    N = object_verts.shape[1]
    K = min(K, N)

    # --- select palm joints ---
    if hand.lower().startswith('r'):
        IDX_WRIST, IDX_INDEX, IDX_PINKY = 21, 40, 46
        flip_normal = False
    else:
        IDX_WRIST, IDX_INDEX, IDX_PINKY = 20, 25, 31
        flip_normal = True

    wrist = human_joints[:, IDX_WRIST, :]
    index = human_joints[:, IDX_INDEX, :]
    pinky = human_joints[:, IDX_PINKY, :]

    # --- palm normal ---
    v1 = index - wrist
    v2 = pinky - wrist
    palm_normals = torch.cross(v1, v2, dim=1)
    if flip_normal:
        palm_normals = -palm_normals
    palm_normals = F.normalize(palm_normals, dim=1, eps=1e-8)  # (T,3)

    # --- centroid ---
    centroid = (wrist + index + pinky) / 3.0  # (T,3)

    # --- find K nearest object vertices per frame ---
    rel = object_verts - centroid.unsqueeze(1)   # (T,N,3)
    dists = rel.norm(dim=2)                      # (T,N)
    _, topi = torch.topk(dists, k=K, dim=1, largest=False)  # (T,K)

    # gather normals
    obj_normals_u = F.normalize(obj_normals, dim=2, eps=1e-8)           # (T,N,3)
    sel_normals = torch.gather(
        obj_normals_u, dim=1, index=topi.unsqueeze(-1).expand(T, K, 3)
    )                                                                   # (T,K,3)

    # --- average normals ---
    avg_norm = sel_normals.mean(dim=1)                                  # (T,3)
    avg_norm = F.normalize(avg_norm, dim=1, eps=1e-8)

    # --- angle (anti-parallel) ---
    cosine = -(palm_normals * avg_norm).sum(dim=1)                      # (T,)
    cosine = torch.clamp(cosine, -1.0, 1.0)
    angles = torch.rad2deg(torch.acos(cosine))

    return angles.detach().cpu().numpy()

def compute_palm_object_angle_old(
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
    wrist = human_joints[:, IDX_WRIST, :]  # (T,3)
    idx   = human_joints[:, IDX_INDEX, :]  # (T,3)
    pinky = human_joints[:, IDX_PINKY, :]  # (T,3)

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


def fix_left_palm(
    twist_list,
    contact_mask,
    orient_mask,
    poses,
    joint_idx,
    axis,
    human_joints=None,
    object_verts=None,
    object_normals=None,
    contact_thresh=0.09,
    twist_bounds=( -110.0, 90.0 ),   # (min, max) acceptable wrist twist in degrees
):
    """
    Simplified wrist correction for LEFT hand.

    Logic per frame t:
      - If contact_mask[t] and not orient_mask[t]:
            angle = compute_palm_object_angle(...)[t]
            rotate_pose_around_axis(poses[t, joint_idx*3:joint_idx*3+3], axis, angle)
      - Else if not contact_mask[t] and twist is out-of-bounds:
            angle = -twist_list[t]   # neutralize twist toward 0 (change sign if desired)
            rotate_pose_around_axis(..., angle)

    Returns:
      poses (modified in-place), fixed_frames (list of frame indices that were adjusted)
    """

    T = len(twist_list)
    # build out-of-bounds mask from twist bounds
    lo, hi = twist_bounds
    out_of_bounds = [(tw > hi or tw < lo) for tw in twist_list]
    # convert masks to CPU numpy/bool for indexing consistency
    contact_mask_cpu = contact_mask.bool().cpu().numpy()
    orient_mask_cpu  = orient_mask.bool().cpu().numpy()
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
    
    # expects degrees; one angle per frame
    specific_angles = compute_palm_object_angle(
        human_joints, object_verts, object_normals, hand='left'
    )
       

    fixed_frames = []
    if proportion_wrong_orient_given_contact > 0.7:
        # If there is contact but wrong orientation proportion is large, calculate specific angle
        print(f"left Contact with wrong orientation detected - calculating specific angles")
        
        if human_joints is not None and object_verts is not None:
            # Calculate specific angles using the palm-object angle function

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
    else:
        for t in range(T):
            if out_of_bounds[t]:
                angle = 0.0
                if contact_mask_cpu[t]:
                    print(f"Frame {t} left palm out of bounds while in contact")
                    angle = float(specific_angles[t])
                else:
                    print(f"Frame {t} left palm out of bounds")
                    angle = float(twist_list[t])
                poses[t, joint_idx*3 : joint_idx*3 + 3] = rotate_pose_around_axis(
                    poses[t, joint_idx*3 : joint_idx*3 + 3], axis, angle
                )
                fixed_frames.append(t)

    return poses, fixed_frames


def fix_right_palm(
    twist_list,
    contact_mask,
    orient_mask,
    poses,
    joint_idx,
    axis,
    human_joints=None,
    object_verts=None,
    object_normals=None,
    contact_thresh=0.09,
    twist_bounds=(-90.0, 110.0),   # acceptable right-wrist twist range (deg)
):
    """
    Simplified wrist correction for RIGHT hand.

    Per frame t:
      - If contact_mask[t] and not orient_mask[t]:
            angle = compute_palm_object_angle(..., hand='right')[t]
            rotate_pose_around_axis(poses[t, joint_idx*3:joint_idx*3+3], axis, angle)
      - Else if not contact_mask[t] and twist is out-of-bounds:
            rotate by the twist angle itself (as requested)

    Returns:
      poses (modified in-place), fixed_frames (list of corrected frame indices)
    """
    T = len(twist_list)

    # out-of-bounds mask from twist bounds
    lo, hi = twist_bounds
    out_of_bounds = [(tw < lo or tw > hi) for tw in twist_list]

    # masks to CPU/numpy for indexing
    contact_mask_cpu = contact_mask.bool().cpu().numpy()
    orient_mask_cpu  = orient_mask.bool().cpu().numpy()
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
    
    specific_angles = compute_palm_object_angle(
        human_joints, object_verts,object_normals, hand='right'
    )
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
    if proportion_wrong_orient_given_contact > 0.7:
        # If there is contact but wrong orientation proportion is large, calculate specific angle
        print(f"right Contact with wrong orientation detected - calculating specific angles")
        
        if human_joints is not None and object_verts is not None:
            # Calculate specific angles using the palm-object angle function

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
    # precompute palm-object angles if geometry available

    else: 
        for t in range(T):
            # Case 1: in contact but wrong orientation -> use palm-object angle
            if out_of_bounds[t]:
                angle = 0.0
                if contact_mask_cpu[t]:
                    print(f"Frame {t} right palm out of bounds while in contact")
                    angle = float(specific_angles[t])
                else:
                    print(f"Frame {t} right palm out of bounds")
                    angle = float(twist_list[t])
                poses[t, joint_idx*3 : joint_idx*3 + 3] = rotate_pose_around_axis(
                    poses[t, joint_idx*3 : joint_idx*3 + 3], axis, angle
                )
                fixed_frames.append(t)
            
    return poses, fixed_frames

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

    # Use global debug flag for printing smoothness information
    global flag

    T, num_joints_total = poses.shape
    num_joints = fixed_joints_mask.shape[1]  # Should be 8
    
    if fixed_joints_mask.sum() < 2:
        return torch.tensor(0.0, device=poses.device)
    
    # For each joint, compute temporal smoothing between fixed frames and their neighbors
    # BUT only for joints that were fixed in previous steps
    loss = torch.tensor(0.0, device=poses.device)
    for joint_idx in range(num_joints):
        # Skip joints that were not fixed in any frame
        if not joint_optimization_mask[joint_idx]:
            continue
            
        # Get frames where this joint was fixed
        fixed_frames_for_joint = torch.where(fixed_joints_mask[:, joint_idx])[0]
        
        # Create a mask for frames that should contribute to smoothness:
        # 1. Fixed frames themselves
        # 2. Neighbors of fixed frames (for smooth transitions)
        smoothness_mask = torch.zeros(T, dtype=torch.bool, device=poses.device)
        
        for frame_idx in fixed_frames_for_joint:
            # Mark the fixed frame
            smoothness_mask[frame_idx] = True
            if frame_idx > 0:
                smoothness_mask[frame_idx - 1] = True
            if frame_idx < T - 1:
                smoothness_mask[frame_idx + 1] = True
        
        if flag:
            print("smoothness mask at frames", torch.where(smoothness_mask)[0])
            # print("velocity calculated at frames", torch.where(vel_calc_mask)[0])
            flag = False
        joint_poses = poses[:, joint_idx * 3:joint_idx * 3 + 3]
        selective_poses = joint_poses * smoothness_mask.float().unsqueeze(1) + joint_poses.detach() * (~smoothness_mask).float().unsqueeze(1)
        
        # Compute velocity loss (first differences)
        vel_loss = torch.sum((selective_poses[1:] - selective_poses[:-1]) ** 2)
        # Compute acceleration loss (second differences) - only if we have at least 3 frames
        if T >= 3:
            accel_loss = torch.sum(((selective_poses[2:] - selective_poses[1:-1]) - (selective_poses[1:-1] - selective_poses[:-2])) ** 2)
        else:
            accel_loss = torch.tensor(0.0, device=poses.device)
        
        # Add to total loss
        loss += weight_vel * vel_loss + weight_accel * accel_loss        
 
    return loss

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


def compute_palm_loss_selective(
    joints,                      # (T, J, 3)
    verts_obj_transformed,       # (T, N, 3)
    obj_normals,                 # (T, N, 3)
    contact_mask,                # (T,) bool
    fixed_joints_mask,           # (T, 8) bool
    is_left_hand=True,
    joint_optimization_mask=None,
    K: int = 700,
    align_mode: str = "antiparallel",      # "abs" | "parallel" | "antiparallel",
    epoch: int = 0
):
    """
    Palm-facing loss using OBJECT NORMALS around the palm centroid.
    Prints per-frame facing loss for debugging.
    """

    device = joints.device
    T = joints.shape[0]
    N = verts_obj_transformed.shape[1]
    K = min(K, N)

    # --- pick hand joint indices ---
    if is_left_hand:
        IDX_WRIST, IDX_INDEX, IDX_PINKY = 20, 25, 31
        wrist_col = 6
        flip_normal = True
    else:
        IDX_WRIST, IDX_INDEX, IDX_PINKY = 21, 40, 46
        wrist_col = 7
        flip_normal = False

    wrist = joints[:, IDX_WRIST, :]
    index = joints[:, IDX_INDEX, :]
    pinky = joints[:, IDX_PINKY, :]

    # Palm normal
    v1 = index - wrist
    v2 = pinky - wrist
    palm_normals = torch.cross(v1, v2, dim=1)
    if flip_normal:
        palm_normals = -palm_normals
    palm_normals = F.normalize(palm_normals, dim=1, eps=1e-8)

    # Palm centroid
    palm_centroid = (wrist + index + pinky) / 3.0  # (T,3)

    # Distances to object vertices
    dists = torch.norm(verts_obj_transformed - palm_centroid.unsqueeze(1), dim=2)  # (T,N)
    _, topi = torch.topk(dists, k=K, dim=1, largest=False)  # (T,K)

    # Gather K object normals
    obj_normals = F.normalize(obj_normals, dim=2, eps=1e-8)
    obj_n_K = torch.gather(
        obj_normals, dim=1, index=topi.unsqueeze(-1).expand(T, K, 3)
    )  # (T,K,3)

    # Cosine scores
    dots = (palm_normals.unsqueeze(1) * obj_n_K).sum(dim=2)  # (T,K)

    if align_mode == "abs":
        score = dots.abs()
    elif align_mode == "parallel":
        score = dots
    elif align_mode == "antiparallel":
        score = -dots
    else:
        raise ValueError(f"Unknown align_mode: {align_mode}")

    facing_loss_per_frame = 1.0 - score.mean(dim=1)  # (T,)

    # Mask frames
    if contact_mask.sum() == 0:
        return torch.zeros((), device=device, dtype=joints.dtype)

    combined_mask = contact_mask & fixed_joints_mask[:, wrist_col]

    if combined_mask.any():
        # Print debug info
        if epoch == 0 or epoch == 499:
            frame_ids = combined_mask.nonzero(as_tuple=True)[0]
            # for f in frame_ids:
            #     print(f"[PalmFacing DEBUG] Frame {f.item():04d} | "
            #         f"loss={facing_loss_per_frame[f].item():.6f}")
        return facing_loss_per_frame[combined_mask].mean()
    else:
        return torch.zeros((), device=device, dtype=joints.dtype)

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
    combined_mask = contact_mask

    # Check if the relevant joints (wrist, index, pinky) were optimized
    if is_left_hand:
        # Left hand joints: wrist (6), index (25), pinky (31) -> check if wrist was optimized
        combined_mask = contact_mask & fixed_joints_mask[:, 6]
        # combined_mask = contact_mask
        # print(f"left hand facing loss largest {torch.max(facing_loss_per_frame[combined_mask])} at frame {torch.argmax(facing_loss_per_frame[combined_mask])}")
    else:
        # Right hand joints: wrist (7), index (40), pinky (46) -> check if wrist was optimized  
        combined_mask = contact_mask & fixed_joints_mask[:, 7]
        # combined_mask = contact_mask
    
    
    facing_loss = torch.mean(torch.where(
        combined_mask,
        facing_loss_per_frame,  # Apply loss during contact
        torch.zeros_like(facing_loss_per_frame)  # No loss otherwise
    ))
    
    return facing_loss

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
        IDX_MIDDLE_END = 28  # Left middle finger end joint
        IDX_PINKY_END = 31  # Left pinky finger end joint
    else:
        IDX_INDEX_END = 40  # Right index finger end joint
        IDX_MIDDLE_END = 43  # Right middle finger end joint
        IDX_PINKY_END = 46  # Right pinky finger end joint
    
    # Extract finger end joint positions
    index_end = joints[:, IDX_INDEX_END, :]  # (T, 3)
    middle_end = joints[:, IDX_MIDDLE_END, :]
    pinky_end = joints[:, IDX_PINKY_END, :]  # (T, 3)
    
    # Compute distances from finger end joints to nearest object point
    index_rel_to_obj = verts_obj_transformed - index_end.unsqueeze(1)  # (T, N, 3)
    middle_rel_to_obj = verts_obj_transformed - middle_end.unsqueeze(1)  # (T, N, 3)
    pinky_rel_to_obj = verts_obj_transformed - pinky_end.unsqueeze(1)  # (T, N, 3)
    
    index_dists = index_rel_to_obj.norm(dim=2)  # (T, N)
    middle_dists = middle_rel_to_obj.norm(dim=2)  # (T, N)
    pinky_dists = pinky_rel_to_obj.norm(dim=2)  # (T, N)
    
    index_min_dist, _ = index_dists.min(dim=1)  # (T,)
    middle_min_dist, _ = middle_dists.min(dim=1)  # (T,)
    pinky_min_dist, _ = pinky_dists.min(dim=1)  # (T,)
    
    # Compute distance differences
    finger_dist_diff = torch.abs(index_min_dist - pinky_min_dist) + torch.abs(middle_min_dist - pinky_min_dist) + torch.abs(index_min_dist - middle_min_dist) # (T,)
    
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


def hand_joint_regulation_epoch(
    object_verts: torch.Tensor,        # (T, Nobj, 3), static over epochs
    prev_anchor_verts: torch.Tensor,   # (T, 5, 3), anchors from prev epoch (use as EMA seed)
    prev_anchor_dists: torch.Tensor,   # (T, 5),    dists from prev epoch (use as EMA seed)
    joints_curr: torch.Tensor,         # (T, 5, 3), current-epoch fingertip joints (one hand)
    close_mask: torch.Tensor,          # (T,) bool
    fixed_joints_mask: torch.Tensor,   # (T, 8) bool
    epoch: int,
    ema_alpha: float = 1.0,            # higher = smoother EMA target
):
    """
    EMA regulation:
      1) Find current anchors (nearest object verts to current joints).
      2) Update EMA anchors/distances:
            ema_next = alpha * prev + (1-alpha) * current
      3) Loss (masked frames): mean_{t,j} ( ||j_curr - ema_anchor|| - ema_dist )^2
         (EMA reference is detached; gradients flow only to joints_curr.)
      4) Return loss and (ema_anchor_verts_next, ema_anchor_dists_next) for next epoch.

    Returns:
      loss: scalar tensor
      ema_anchor_verts_next: (T, 5, 3)
      ema_anchor_dists_next: (T, 5)
    """
    device = joints_curr.device
    dtype  = joints_curr.dtype
    T, Nobj, _ = object_verts.shape

    # --- 1) Current anchors (argmin) for this epoch (non-differentiable bookkeeping) ---
    with torch.no_grad():
        # (T, 5, Nobj)
        dmat = torch.cdist(joints_curr.detach(), object_verts)
        idx  = dmat.argmin(dim=2)  # (T, 5)
        curr_anchor_verts = torch.gather(
            object_verts, dim=1, index=idx.unsqueeze(-1).expand(T, 5, 3)
        )  # (T, 5, 3)
        curr_anchor_dists = torch.norm(
            joints_curr.detach() - curr_anchor_verts, dim=2
        )  # (T, 5)

    # --- 2) EMA update (initialize at epoch 0 with current anchors if needed) ---
    if epoch == 0 or prev_anchor_verts is None or prev_anchor_dists is None:
        ema_anchor_verts_next = curr_anchor_verts
        ema_anchor_dists_next = curr_anchor_dists
        # No regulation on epoch 0, but return initialized EMA for next epoch
        return torch.zeros((), device=device, dtype=dtype), ema_anchor_verts_next, ema_anchor_dists_next

    with torch.no_grad():
        ema_anchor_verts_next = ema_alpha * prev_anchor_verts + (1.0 - ema_alpha) * curr_anchor_verts
        ema_anchor_dists_next = ema_alpha * prev_anchor_dists + (1.0 - ema_alpha) * curr_anchor_dists

    # --- 3) Loss against EMA reference (masking by close & fixed for this hand) ---
    frame_mask = close_mask & fixed_joints_mask
    if not torch.any(frame_mask):
        return torch.zeros((), device=device, dtype=dtype), ema_anchor_verts_next, ema_anchor_dists_next

    idx_t = torch.nonzero(frame_mask, as_tuple=False).squeeze(-1)  # (B,)
    joints_B   = joints_curr.index_select(0, idx_t)                       # (B, 5, 3)
    ema_vertsB = ema_anchor_verts_next.index_select(0, idx_t)             # (B, 5, 3)
    ema_distsB = ema_anchor_dists_next.index_select(0, idx_t)             # (B, 5)

    d_curr = torch.norm(joints_B - ema_vertsB.detach(), dim=2)            # (B, 5)
    d_ref  = ema_distsB.detach()                                          # (B, 5)

    loss = ((d_curr - d_ref) ** 2).mean()  # mean over joints and frames

    # --- 4) Return loss and updated EMA for next epoch ---
    return loss, ema_anchor_verts_next, ema_anchor_dists_next

def precompute_hand_object_distances(verts, verts_obj_transformed, obj_normals, rhand_idx, lhand_idx):
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
    
    T = verts.shape[0]# Compute distances for right hand using point2point_signed (same as optimize.py)
    right_hand_verts = verts[:, rhand_idx, :]  # (T, R, 3) where R = 778
    # Use point2point_signed to get signed distances efficiently
    _, right_signed_distances, _, _, _, _ = point2point_signed(right_hand_verts, verts_obj_transformed, y_normals = obj_normals, return_vector=True)
    right_hand_min_dist = torch.min(right_signed_distances, dim=1)[0]  # (T,) - minimum distance for each frame
    # right_o2h_min_dist = torch.min(right_o2h_signed, dim=1)[0]  # (T,) - minimum distance for each frame
    # Compute distances for left hand using point2point_signed (same as optimize.py)
    left_hand_verts = verts[:, lhand_idx, :]  # (T, L, 3) where L = 778
    # Use point2point_signed to get signed distances efficiently
    _, left_signed_distances, _, _, _, _ = point2point_signed(left_hand_verts, verts_obj_transformed, y_normals = obj_normals, return_vector=True)
    left_hand_min_dist = torch.min(left_signed_distances, dim=1)[0]  # (T,) - minimum distance for each frame
 # Create distance masks (frames that are close to object)
    correction_thresh = -0.1
    penetration_thresh = 0
    contact_thresh = 0.02  # 2cm threshold for contact
    close_thresh = 0.06    # 20cm threshold for "close" frames
    
    right_pen_mask = (right_hand_min_dist <= penetration_thresh) & (right_hand_min_dist >= correction_thresh)
    left_pen_mask = (left_hand_min_dist <= penetration_thresh) & (left_hand_min_dist >= correction_thresh)
    
    right_contact_mask = (right_hand_min_dist <= contact_thresh) & (right_hand_min_dist >= correction_thresh)
    left_contact_mask = (left_hand_min_dist <= contact_thresh) & (left_hand_min_dist >= correction_thresh)
    
    right_close_mask = (right_hand_min_dist <= close_thresh) & (right_hand_min_dist >= correction_thresh)
    left_close_mask = (left_hand_min_dist <= close_thresh) & (left_hand_min_dist >= correction_thresh)
    
    # Create optimization masks (frames that should be optimized for penetration)
    right_optimize_mask = (right_hand_min_dist <= close_thresh) & (right_hand_min_dist >= correction_thresh)
    left_optimize_mask = (left_hand_min_dist <= close_thresh) & (left_hand_min_dist >= correction_thresh)
    
    print(f"Distance pre-computation results:")
    print(f"  Right hand: {right_contact_mask.sum().item()}/{T} contact frames, {right_close_mask.sum().item()}/{T} close frames")
    print(f"  Left hand: {left_contact_mask.sum().item()}/{T} contact frames, {left_close_mask.sum().item()}/{T} close frames")
    
    return {
        'right_pen_mask': right_pen_mask,
        'left_pen_mask': left_pen_mask,
        'right_contact_mask': right_contact_mask,
        'left_contact_mask': left_contact_mask,
        'right_close_mask': right_close_mask,
        'left_close_mask': left_close_mask,
        'right_optimize_mask': right_optimize_mask,
        'left_optimize_mask': left_optimize_mask,
        'right_hand_min_dist': right_hand_min_dist,
        'left_hand_min_dist': left_hand_min_dist,
        # 'right_o2h_min_dist': right_o2h_min_dist,
        # 'left_o2h_min_dist': left_o2h_min_dist,
    }

def compute_hand_penetration_loss(
    verts_full: torch.Tensor,         # (T, V, 3) all human vertices
    verts_obj_transformed: torch.Tensor,
    obj_normals: torch.Tensor,
    fixed_joints_mask: torch.Tensor,  # (T, 8)
    lhand_idx: torch.Tensor,
    rhand_idx: torch.Tensor,
    thresh: float = 0.0,
    detach_opposite: bool = True,
    epoch: int = 0,
):
    """
    Compute penetration loss for both hands, ensuring no gradient flow from the opposite hand.

    Returns:
      pen_left, pen_right
    """

    device = verts_full.device

    # Optionally detach the *other* hand vertices
    if detach_opposite:
        verts_left  = verts_full.clone()
        verts_right = verts_full.clone()

        # Detach right hand when computing left
        verts_left[:, rhand_idx, :] = verts_left[:, rhand_idx, :].detach()
        # Detach left hand when computing right
        verts_right[:, lhand_idx, :] = verts_right[:, lhand_idx, :].detach()
    else:
        verts_left = verts_full
        verts_right = verts_full

    # --- Left hand ---
    pen_left = _compute_single_hand_penetration(
        verts_left[:, lhand_idx, :],
        verts_obj_transformed, obj_normals, fixed_joints_mask[:, 6],
        thresh=thresh, epoch=epoch
    )

    # --- Right hand ---
    pen_right = _compute_single_hand_penetration(
        verts_right[:, rhand_idx, :],
        verts_obj_transformed, obj_normals, fixed_joints_mask[:, 7],
        thresh=thresh, epoch=epoch
    )

    return pen_left, pen_right


def _compute_single_hand_penetration(
    hand_verts, verts_obj_transformed, obj_normals,
    hand_fixed_mask, thresh=0.0, eps=1e-8, epoch=0
):
    """
    Per-hand penetration loss with per-frame debug prints:
      - number of penetrating vertices
      - deepest penetration depth
      - per-frame penetration loss
    Ignores frames where penetration goes deeper than -0.3 units.
    """
    device = hand_verts.device
    dtype  = hand_verts.dtype
    T = hand_verts.shape[0]

    total_penetration_loss = torch.zeros((), device=device, dtype=dtype)

    # signed distances: sbj2obj[t,v] is distance for vertex v in frame t
    o2h_signed, sbj2obj, *_ = point2point_signed(
        hand_verts, verts_obj_transformed.to(device),
        y_normals=obj_normals, return_vector=True
    )

    # penetration mask and depths
    pen_mask = (sbj2obj < thresh)                       # (T, Vh)
    depths   = (thresh - sbj2obj).clamp_min(0.0)        # positive depth = penetration amount

    counts = pen_mask.sum(dim=-1)                       # (T,) number of penetrating verts per frame
    max_depths = depths.max(dim=-1).values              # (T,) deepest penetration per frame

    # Frame mask: only keep frames with penetration, fixed joint, and not too deep
    frame_mask = (counts > 0) & hand_fixed_mask & (max_depths <= 0.1)

    if frame_mask.any():
        depth_sum = (depths * pen_mask).sum(dim=-1)     # (T,) sum of depths for each frame
        per_frame = depth_sum                           # loss contribution per frame

        # ---- Debug prints ----
        if epoch == 0 or epoch == 299:
            frame_ids = frame_mask.nonzero(as_tuple=True)[0]
            for f in frame_ids:
                num_pene = counts[f].item()
                if num_pene > 0:
                    max_depth = depths[f].max().item()
                    frame_loss = per_frame[f].item()
                    print(f"[Penetration DEBUG] Frame {f.item():04d}: "
                          f"minimum sbj2obj={sbj2obj[f].min().item():.6f}, "
                          f"{num_pene} verts penetrating, "
                          f"deepest={max_depth:.6f}, "
                          f"loss={frame_loss:.6f}")

        total_penetration_loss = per_frame[frame_mask].sum()
        return total_penetration_loss / T
    else:
        return torch.tensor(0.0)
    # Return average over T frames (so sequence length doesn’t bias)
    



def optimize_poses_with_fixed_tracking(poses, betas, trans, gender, verts_obj_transformed, obj_normals,
                                     fix_tracker, distance_info, rhand_idx, lhand_idx, 
                                     num_epochs=500, lr=0.01, canonical_joints=None):
    """Two-stage optimization: Stage 1 optimizes only wrist poses, Stage 2 optimizes all joints.
    
    Stage 1: Only optimize wrist poses (left and right) to fix palm orientation
    Stage 2: Optimize all joints (collar, shoulder, elbow, wrist) with full loss functions
    
    This approach ensures that palm orientation loss only affects wrist poses in Stage 1,
    preventing unwanted gradients to other joints.
    """
    
    # Get which joints were fixed in which frames
    fixed_joints_mask = fix_tracker.get_fixed_frames_and_joints()  # Shape: (T, 8)
    # Convert to tensor and move to device
    fixed_joints_mask = torch.from_numpy(fixed_joints_mask).bool().to(device)
    
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
    # Print out for each joint which frames are fixed according to fixed_joints_mask
    joint_names = [
        "left_collar", "right_collar", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist"
    ]
    for joint_idx in range(8):
        fixed_frames = torch.where(fixed_joints_mask[:, joint_idx])[0].cpu().numpy().tolist()
        print(f"Joint {joint_idx} ({joint_names[joint_idx]}): fixed at frames {fixed_frames}")
    # Create optimization mask for all poses: ALL frames for fixed joints
    optimization_mask = torch.zeros_like(all_joint_poses_tensor, dtype=torch.bool)
    for joint_idx in range(8):  # 8 joints
        pose_start = joint_idx * 3
        pose_end = pose_start + 3
        # Mark ALL frames for joints that were fixed
        if joint_optimization_mask[joint_idx]:
            optimization_mask[:, pose_start:pose_end] = True  # Mark all frames for this joint
    
    # Early stopping setup
    best_loss = float('inf')
    best_poses = None
    patience_counter = 0
    patience = 400
    left_finger_mask = distance_info['left_close_mask'] & fixed_joints_mask[:, 6]
    right_finger_mask = distance_info['right_close_mask'] & fixed_joints_mask[:, 7]
    # left_finger_mask = distance_info['left_close_mask']
    # right_finger_mask = distance_info['right_close_mask']
    left_wrist_fixed_mask = torch.where(left_finger_mask)[0]
    right_wrist_fixed_mask = torch.where(right_finger_mask)[0]

    # Convert to unique frame indices (since each frame has 778 hand vertices)
    left_fixed_unique_frames = sorted(list(set(left_wrist_fixed_mask)))
    right_fixed_unique_frames = sorted(list(set(right_wrist_fixed_mask)))

    
    # ==================== STAGE 1: Optimize only wrist poses ====================
    print("\n=== STAGE 1: Optimizing only wrist poses ===")
    
    # Create wrist-only optimizer (only wrist poses)
    # We need to create a separate tensor for wrist poses to make it a leaf tensor
    wrist_poses_tensor = torch.cat([
        all_joint_poses_tensor[:, 18:21].detach(),  # Left wrist (indices 18:21)
        all_joint_poses_tensor[:, 21:24].detach(),  # Right wrist (indices 21:24)
    ], dim=1).clone().requires_grad_(True)  # Shape: (T, 6) - left wrist (3) + right wrist (3)
    
    wrist_optimizer = torch.optim.Adam([wrist_poses_tensor], lr=0.001)
    wrist_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(wrist_optimizer, mode='min', factor=0.7, patience=300, verbose=True)
    
    # Stage 1 optimization loop
    stage1_epochs = 300  # Use half of total epochs for stage 1
    best_wrist_loss = float('inf')
    best_wrist_poses = None
    T = poses.shape[0]
    prev_anchor_verts_r = torch.zeros(T, 5, 3).to(device)
    prev_anchor_dists_r = torch.zeros(T, 5).to(device)
    prev_anchor_verts_l = torch.zeros(T, 5, 3).to(device)
    prev_anchor_dists_l = torch.zeros(T, 5).to(device)
    reference_wrist_poses = reference_all_joint_poses[:, 18:24]  # Shape: (T, 6)
    # print(distance_info['left_close_mask'])
    for epoch in range(stage1_epochs):
        wrist_optimizer.zero_grad()
        
        # Create poses tensor with only wrist poses updated
        poses_tensor = torch.from_numpy(poses).float().to(device)
        
        # Update only wrist poses in the full poses tensor
        poses_tensor[:, 60:63] = wrist_poses_tensor[:, :3]      # Left wrist
        poses_tensor[:, 63:66] = wrist_poses_tensor[:, 3:6]     # Right wrist
        
        # Use SMPLX16 model
        model = sbj_m_all[gender]
        
        # ---- Forward pass for other losses (posedirs ON) ----
        output = model(
            pose_body=poses_tensor[:, 3:66],
            pose_hand=poses_tensor[:, 66:156],
            betas=torch.from_numpy(betas[None, :]).repeat(poses_tensor.shape[0], 1).float().to(device),
            root_orient=poses_tensor[:, :3],
            trans=torch.from_numpy(trans).float().to(device)
        )
            
        joints = output.Jtr
        if epoch == 0:
            joints_epoch1 = joints.detach().clone()
            joints_epoch2 = joints.clone()
        else:
            joints_epoch2 = joints.clone()
        
        # Compute bone axes for left and right forearm (elbow to wrist direction)
        left_forearm_axis = canonical_joints[LEFT_WRIST] - canonical_joints[LEFT_ELBOW]  # (3,)
        right_forearm_axis = canonical_joints[RIGHT_WRIST] - canonical_joints[RIGHT_ELBOW]  # (3,)
        
        # Convert to PyTorch tensors and move to device
        left_forearm_axis = torch.from_numpy(left_forearm_axis).float().to(device)
        right_forearm_axis = torch.from_numpy(right_forearm_axis).float().to(device)
        
        # Normalize bone axes
        left_forearm_axis = left_forearm_axis / torch.norm(left_forearm_axis)
        right_forearm_axis = right_forearm_axis / torch.norm(right_forearm_axis)
        
        # Extract initial twist angles from Stage 2 wrist poses
        # Calculate proper twist angles: normalize pose, get cosine with bone axis, multiply by pose magnitude
        left_wrist_poses = wrist_poses_tensor[:, :3]  # (T, 3)
        right_wrist_poses = wrist_poses_tensor[:, 3:6]  # (T, 3)
        
        # Left hand twist calculation
        left_pose_magnitudes = torch.norm(left_wrist_poses, dim=1, keepdim=True)  # (T, 1)
        left_pose_axes = left_wrist_poses / (left_pose_magnitudes + 1e-8)  # (T, 3) - normalized rotation axes
        left_cosines = torch.sum(left_pose_axes * left_forearm_axis.unsqueeze(0), dim=1)  # (T,) - cosine between axes
        left_twist_rad = left_pose_magnitudes.squeeze(1) * left_cosines  # (T,) - twist angle = magnitude * cosine
        
        # Right hand twist calculation  
        right_pose_magnitudes = torch.norm(right_wrist_poses, dim=1, keepdim=True)  # (T, 1)
        right_pose_axes = right_wrist_poses / (right_pose_magnitudes + 1e-8)  # (T, 3) - normalized rotation axes
        right_cosines = torch.sum(right_pose_axes * right_forearm_axis.unsqueeze(0), dim=1)  # (T,) - cosine between axes
        right_twist_rad = right_pose_magnitudes.squeeze(1) * right_cosines  # (T,) - twist angle = magnitude * cosine

        # Create masks for frames where hands are NOT close to object
        left_not_close_mask = ~distance_info['left_close_mask']  # (T,)
        right_not_close_mask = ~distance_info['right_close_mask']  # (T,)
        
        # Compute loss: encourage twist to be close to 0 radians when not close to object
        # Use MSE loss: (twist_angle - 0)^2 = twist_angle^2
        left_spare_twist_loss = torch.mean(left_twist_rad[left_not_close_mask] ** 2) if left_not_close_mask.any() else torch.tensor(0.0, device=device)
        right_spare_twist_loss = torch.mean(right_twist_rad[right_not_close_mask] ** 2) if right_not_close_mask.any() else torch.tensor(0.0, device=device)
        
        spare_hand_twist_loss = left_spare_twist_loss + right_spare_twist_loss 
        # Only compute palm orientation loss in Stage 1

        left_palm_loss = compute_palm_loss_selective(
            joints, verts_obj_transformed, obj_normals, distance_info['left_close_mask'], fixed_joints_mask, is_left_hand=True, joint_optimization_mask=joint_optimization_mask, epoch=epoch
        )
        right_palm_loss = compute_palm_loss_selective(
            joints, verts_obj_transformed, obj_normals, distance_info['right_close_mask'], fixed_joints_mask, is_left_hand=False, joint_optimization_mask=joint_optimization_mask, epoch=epoch
        )
        palm_loss = left_palm_loss + right_palm_loss
        left_finger_loss = compute_finger_distance_loss_with_mask(
            joints, verts_obj_transformed, distance_info['left_close_mask'], left_finger_mask, is_left_hand=True
        )
        right_finger_loss = compute_finger_distance_loss_with_mask(
            joints, verts_obj_transformed, distance_info['right_close_mask'], right_finger_mask, is_left_hand=False
        )
        finger_loss = left_finger_loss + right_finger_loss
        # Simple reference loss for wrist poses only
        # Extract wrist poses from reference (indices 18:24 correspond to left and right wrist)
        wrist_ref_loss = torch.mean((wrist_poses_tensor - reference_wrist_poses) ** 2)

        left_regulation = torch.tensor(0.0)
        right_regulation = torch.tensor(0.0)
        left_regulation, curr_anchor_verts_l, curr_anchor_dists_l = hand_joint_regulation_epoch(
            verts_obj_transformed,           # Constant object vertices
            prev_anchor_verts_l,
            prev_anchor_dists_l,
            joints[:, [25, 28, 31, 34, 37]],
            distance_info['left_close_mask'],
            fixed_joints_mask[:, 6],
            epoch
        )

            # For right hand regulation
        right_regulation, curr_anchor_verts_r, curr_anchor_dists_r = hand_joint_regulation_epoch(
            verts_obj_transformed,
            prev_anchor_verts_r,
            prev_anchor_dists_r,
            joints[:, [40, 43, 46, 49, 52]],
            distance_info['right_close_mask'],
            fixed_joints_mask[:, 7],
            epoch
        )
        pen_left, pen_right = compute_hand_penetration_loss(
            output.v, verts_obj_transformed, obj_normals, fixed_joints_mask,
            lhand_idx, rhand_idx, detach_opposite=True, epoch=epoch
        )
        # pen_left, pen_right = torch.tensor(0.0), torch.tensor(0.0)
        # Total loss for Stage 1
        total_wrist_loss = 0.2 * pen_left + 0.2 * pen_right + 10 * wrist_ref_loss + palm_loss + 10 * left_regulation + 10 * right_regulation
        # + 2 * palm_loss + 2 * finger_loss + left_regulation + right_regulation
        # 
        # + 0.2 * spare_hand_twist_loss + 10 * wrist_ref_loss

        # Check for improvement
        if total_wrist_loss < best_wrist_loss - 1e-6:
            best_wrist_loss = total_wrist_loss
            best_wrist_poses = wrist_poses_tensor.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience // 2:
            print(f"Stage 1 early stopping at epoch {epoch}")
            break
        
        if epoch % 20 == 0:
            print(f"Stage 1 Epoch {epoch}: palm={palm_loss.item():.6f}, finger={finger_loss.item():.6f}, pen_left={pen_left.item():.6f}, pen_right={pen_right.item():.6f}, left_regulation={left_regulation.item():.6f}, right_regulation={right_regulation.item():.6f}, total={total_wrist_loss.item():.6f}")
        
        total_wrist_loss.backward()
        joints_epoch1 = joints.detach().clone()
        torch.nn.utils.clip_grad_norm_([wrist_poses_tensor], max_norm=1.0)
        wrist_optimizer.step()
        wrist_scheduler.step(total_wrist_loss)
        prev_anchor_verts_r = curr_anchor_verts_r.detach().clone()
        prev_anchor_dists_r = curr_anchor_dists_r.detach().clone()
        prev_anchor_verts_l = curr_anchor_verts_l.detach().clone()
        prev_anchor_dists_l = curr_anchor_dists_l.detach().clone()
        
    
    # Update all_joint_poses_tensor with optimized wrist poses
    if best_wrist_poses is not None:
        # Detach and update the wrist poses in all_joint_poses_tensor
        with torch.no_grad():
            all_joint_poses_tensor[:, 18:21] = best_wrist_poses[:, :3].detach()      # Left wrist
            all_joint_poses_tensor[:, 21:24] = best_wrist_poses[:, 3:6].detach()     # Right wrist
        print(f"Stage 1 completed with best wrist loss: {best_wrist_loss:.6f}")
    else:
        # If no best poses, update with final wrist poses
        with torch.no_grad():
            all_joint_poses_tensor[:, 18:21] = wrist_poses_tensor[:, :3].detach()      # Left wrist
            all_joint_poses_tensor[:, 21:24] = wrist_poses_tensor[:, 3:6].detach()     # Right wrist
        print("Stage 1 completed using final wrist poses")
    reference_all_joint_poses = all_joint_poses_tensor.clone()

    # ==================== STAGE 2: Optimize all joints ====================
    print("\n=== STAGE 2: Optimizing all joints ===")
    
    # Optimizer for all joint poses
    optimizer = torch.optim.Adam([all_joint_poses_tensor], lr=0.001)  # Lower learning rate for stage 2
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=200, verbose=True)
    
    # Stage 2 optimization loop
    stage2_epochs = 400
    
    for epoch in range(stage2_epochs):
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
        
        penetration_loss = torch.tensor(0.0, device=all_joint_poses_tensor.device)
        # Ensure all losses are tensors on the correct device
        smooth_loss = smooth_loss.to(all_joint_poses_tensor.device)
        # palm_loss = palm_loss.to(all_joint_poses_tensor.device)
        penetration_loss = penetration_loss.to(all_joint_poses_tensor.device)
        # finger_loss = finger_loss.to(all_joint_poses_tensor.device)
        
        # Combine all losses with appropriate weights for Stage 2
        total_loss = 0.5 * smooth_loss + 2.0 * ref_loss
        
        # Check for improvement
        if total_loss < best_loss - 1e-6:
            best_loss = total_loss
            best_poses = all_joint_poses_tensor.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Stage 2 early stopping at epoch {epoch}")
            break
        
        if epoch % 20 == 0:
            print(f"Stage 2 Epoch {epoch}: smooth={smooth_loss.item():.6f}, ref={ref_loss.item():.6f},  penetration={penetration_loss.item():.6f}, total={total_loss.item():.6f}")
        
        total_loss.backward()
        # if epoch % 100 == 0:
        #     check_gradients_fn()
        # Comprehensive analysis of frames 45-90
        # analyze_gradients_fn(45, 90)
        torch.nn.utils.clip_grad_norm_([all_joint_poses_tensor], max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss)
    
    # Reconstruct final poses using best poses if available
    final_poses = poses.copy()
    poses_to_use = best_poses if best_poses is not None else all_joint_poses_tensor
    print(f"Using {'best' if best_poses is not None else 'final'} poses with loss: {best_loss:.6f}")
    
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
    
    print("Two-stage optimization completed successfully!")
    return final_poses

def restrict_angles(theta,theta_max,theta_min,mode,flag,alpha=0.01):
    MASK_MAX=(theta-theta_max)>0
    MASK_MIN=(theta-theta_min)<0
    T=theta.shape[0]
    loss_max=torch.sum(MASK_MAX.detach().float()*(theta-theta_max)**2)
    loss_min=torch.sum(MASK_MIN.detach().float()*(theta_min-theta)**2)
    if mode==0:
        return 10*loss_max**2+10*loss_min**2+torch.sum(theta**2)*alpha
    else:
        return 1*loss_max+loss_min*1

def smooth_mask(hand_pose_rec_o,hand_verts_o,mask):
    hand_pose_rec=torch.zeros_like(hand_pose_rec_o).to(hand_pose_rec_o.device)
    hand_pose_rec=hand_pose_rec+hand_pose_rec_o*mask
    hand_pose_rec=hand_pose_rec+hand_pose_rec_o.detach()*(1-mask)

    hand_verts=torch.zeros_like(hand_verts_o).to(hand_pose_rec_o.device)
    hand_verts = hand_verts+hand_verts_o*mask
    hand_verts = hand_verts+hand_verts_o.detach()*(1-mask)
    loss1 = 0.25 * torch.sum((((hand_pose_rec[1:-1] - hand_pose_rec[:-2]) - (hand_pose_rec[2:] - hand_pose_rec[1:-1])) ** 2)) + \
                                        0.5 * torch.sum(((hand_pose_rec[1:] - hand_pose_rec[:-1]) ** 2))
    
    loss2= 0.25 * torch.sum((((hand_verts[1:-1] - hand_verts[:-2]) - (hand_verts[2:] - hand_verts[1:-1])) ** 2)) + \
                                        0.5 * torch.sum(((hand_verts[1:] - hand_verts[:-1]) ** 2))
    return loss1+loss2

def optimize_finger(index, name, visualize=False, fname='', render_path=''):
    print(f"Starting optimize1 for {fname}")
    
    human_npz_path=os.path.join(render_path,f"{fname}_step3_optimized.npz")
    object_npz_path=os.path.join(name,"object.npz")
    with np.load(human_npz_path, allow_pickle=True) as f:
            poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])

    #print(poses.shape)
    with np.load(object_npz_path, allow_pickle=True) as f:
        #print(f.files)
        obj_angles,obj_trans,obj_name=f['angles'],f['trans'],str(f['name'])
    
    
    T=2
    # comment: 这个T的作用是什么
    bs = T
    MASK2=list(range(T))    
    if T<1:
        return

    SMPLX_PATH='models/smplx'
    sbj_m=sbj_m_all[gender]
    OBJ_PATH='data/omomo/objects'
    obj_dir_name=os.path.join(OBJ_PATH,obj_name)

    MMESH=trimesh.load(os.path.join(obj_dir_name,obj_name+'.obj'))
    verts_obj=np.array(MMESH.vertices)
    faces_obj=np.array(MMESH.faces)
    MEAN=np.mean(verts_obj,0)
    #verts_obj=verts_obj-MEAN
    verts_sampled=np.load(os.path.join(obj_dir_name,'sample_points.npy'))
    obj_info={'verts': verts_obj, 'faces': faces_obj, 'verts_sample': verts_sampled,}

    frame_times=poses.shape[0]
    body_pose=torch.from_numpy(poses[:, 3:66]).float().to(device)
    betas_tensor=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float().to(device)
    trans_tensor=torch.from_numpy(trans).float().to(device)
    # comment: 这个root对应的哪个部位的pose？应该是躯干？
    root_tensor=torch.from_numpy(poses[:, :3]).float().to(device)
    hand_pose_tensor=torch.from_numpy(poses[:, 66:156]).float().to(device)
    # comment: HAND_MEAN_TITLE是behave没问题嘛？这些.npy文件是怎么来的？有对应手腕的吗
    rhand_mean=np.load(f'./exp/{HAND_MEAN_TITLE}_rhand_mean.npy')
    rhand_mean_torch=torch.from_numpy(rhand_mean.reshape(1,-1,3)).float().to(device).repeat(frame_times,1,1)
    lhand_mean=np.load(f'./exp/{HAND_MEAN_TITLE}_lhand_mean.npy')
    lhand_mean_torch=torch.from_numpy(lhand_mean.reshape(1,-1,3)).float().to(device).repeat(frame_times,1,1)
    #hand_pose_rec=Variable(torch.tensor([0.0,0.0,0.0]).float().reshape(1,1,3).repeat(frame_times,15,1).to(device),requires_grad=True)
    hand_pose_tensor=torch.cat([lhand_mean_torch,rhand_mean_torch],dim=1).reshape(-1,90)
    
    smplx_output = sbj_m(pose_body=body_pose, 
                            pose_hand=hand_pose_tensor, 
                            betas=betas_tensor, 
                            root_orient=root_tensor, 
                            trans=trans_tensor)
    
    verts_sbj_mean=smplx_output.v

    obj_trans_tensor=torch.from_numpy(obj_trans).float().to(device)
    obj_rot_mat_tensor=axis_angle_to_matrix(torch.from_numpy(obj_angles).float().to(device))
    obj_rot_inv_tensor=torch.linalg.inv(obj_rot_mat_tensor)
    obj_6d_tensor=matrix_to_rotation_6d(obj_rot_mat_tensor).float()
    verts_obj=obj_forward(torch.from_numpy(verts_obj).float().to(device),obj_6d_tensor,obj_trans_tensor)
    obj_normals=vertex_normals(verts_obj,torch.tensor(obj_info['faces'].astype(np.float32)).unsqueeze(0).repeat(verts_obj.shape[0],1,1).to(device))

    rhand_idx=np.load('./exp/rhand_smplx_ids.npy')
    lhand_idx_path='./exp/lhand_smplx_ids.npy'
    lhand_idx=np.load(lhand_idx_path)

    h2o_signed_mean=point2point_signed(verts_sbj_mean[:,rhand_idx],verts_obj)[1]
    #print(h2o_signed_mean.shape,'KJKJKJKJ')
    
    ## DEFINE THE CONTACT STATE: Pre Contact -> Contact -> Post Contact
    
    min_distance_mean_hand=torch.min(h2o_signed_mean,dim=1)[0]
    MASK1=min_distance_mean_hand <= 0.02
    # comment: 这一步在算什么
    thresh2=0.20
    k_l=1/(0.02-thresh2)
    b_l=thresh2/(thresh2-0.02)

    MASK2 = min_distance_mean_hand > thresh2
    WHETHER_TOUCH_RIGHT_L=torch.zeros_like(min_distance_mean_hand).float().to(device)
    WHETHER_TOUCH_RIGHT_L[MASK1] = 1
    WHETHER_TOUCH_RIGHT_L[MASK2] = 0

    MASK_MID=~MASK1 & ~MASK2
    # comment: 这个参数代表什么min_distance_mean_hand[MASK_MID]*k_l+b_l
    WHETHER_TOUCH_RIGHT_L[MASK_MID]=min_distance_mean_hand[MASK_MID]*k_l+b_l

    WHETHER_TOUCH_RIGHT_L = WHETHER_TOUCH_RIGHT_L.float().detach().reshape(-1,1)
    WHETHER_TOUCH_RIGHT = MASK1.float().detach().reshape(-1,1)
    # comment: 只有距离小于0.2时才optimize
    WHETHER_OPTIMIZE_RIGHT = (~MASK2).float().detach().reshape(-1,1)
    WHETHER_MIDTOUCH_RIGHT = (~MASK1).float().detach().reshape(-1,1)

    
    frame_times_right=torch.sum(WHETHER_TOUCH_RIGHT)
    # comment: h2o signed mean shape (N, T)
    h2o_signed_mean = point2point_signed(verts_sbj_mean[:,lhand_idx],verts_obj)[1]
    # comment: min_distance_mean_hand shape (N, )
    min_distance_mean_hand=torch.min(h2o_signed_mean,dim=1)[0]
    MASK1=min_distance_mean_hand<=0.02
    thresh2=0.20
    k_l=1/(0.02-thresh2)
    b_l=thresh2/(thresh2-0.02)
    MASK2=min_distance_mean_hand>thresh2
    WHETHER_TOUCH_LEFT_L=torch.zeros_like(min_distance_mean_hand).float().to(device)
    WHETHER_TOUCH_LEFT_L[MASK1] = 1
    WHETHER_TOUCH_LEFT_L[MASK2] = 0
    MASK_MID=~MASK1 & ~MASK2
    WHETHER_TOUCH_LEFT_L[MASK_MID] = min_distance_mean_hand[MASK_MID]*k_l+b_l
    WHETHER_TOUCH_LEFT_L = WHETHER_TOUCH_LEFT_L.float().detach().reshape(-1,1)
    WHETHER_TOUCH_LEFT = MASK1.float().detach().reshape(-1,1)
    WHETHER_OPTIMIZE_LEFT= (~MASK2).float().detach().reshape(-1,1)
    WHETHER_MIDTOUCH_LEFT=(~MASK1).float().detach().reshape(-1,1)

    frame_times_left=torch.sum(WHETHER_TOUCH_LEFT)
      
    def calc_loss(verts, jtr, hand_pose_rec, epoch, left_or_right):
        
        if left_or_right: ## LEFT HAND OR RIGHT HAND
            WHETHER_TOUCH = WHETHER_TOUCH_RIGHT
            WHETHER_TOUCH_L = WHETHER_TOUCH_RIGHT_L
            frame_times_touch = frame_times_right
            hand_idx=np.load('./exp/rhand_smplx_ids.npy')
            
            HAND_INDEXES = RHAND_INDEXES_DETAILED
            HAND_SMALL_INDEXES = RHAND_SMALL_INDEXES_DETAILED
           
            hand_mean_single = rhand_mean_torch_single
            WHETHER_OPTIMIZE = WHETHER_OPTIMIZE_RIGHT
            WHETHER_MIDTOUCH = WHETHER_MIDTOUCH_RIGHT
        else:
            WHETHER_TOUCH=WHETHER_TOUCH_LEFT
            WHETHER_TOUCH_L=WHETHER_TOUCH_LEFT_L
            frame_times_touch=frame_times_left
            lhand_idx_path='./exp/lhand_smplx_ids.npy'

            hand_idx=np.load(lhand_idx_path)
            # HAND_INDEXES=LHAND_INDEXES
            # HAND_SMALL_INDEXES=LHAND_SMALL_INDEXES
            HAND_INDEXES=LHAND_INDEXES_DETAILED
            HAND_SMALL_INDEXES=LHAND_SMALL_INDEXES_DETAILED
           
                #WHETHER_MID_TOUCH=WHETHER_MIDTOUCH_LEFT
            hand_mean_single=lhand_mean_torch_single
            WHETHER_OPTIMIZE=WHETHER_OPTIMIZE_LEFT
            WHETHER_MIDTOUCH=WHETHER_MIDTOUCH_LEFT
        #print(torch.sum(WHETHER_TOUCH_L),'JKJKJKJ')
        
        with torch.enable_grad():       
            
            if epoch==0:
                global hand_distance_init
                hand_distance_init=(torch.norm((jtr[:,[45,48,51]]-jtr[:,[42,45,48]]),dim=-1)).detach()
            
            point2point_start = time.time()
            o2h_signed, sbj2obj, o2h_idx, sbj2obj_idx, o2h, sbj2obj_vector = point2point_signed(verts[:,hand_idx], verts_obj,y_normals=obj_normals,return_vector=True) #y_normals=obj_normals
            point2point_time = time.time() - point2point_start
            
            # J_rhand=jtr[:,40:55]
            
            # list_30=[4775, 4644, 4957, 5001, 5012, 5069, 5218, 5246, 5302, 5194,
            #  5086, 5181, 4601, 4730, 5354, 7511, 7380, 7693, 
            #  7737, 7710, 7805, 7954, 7982, 8038, 7930, 7822, 7917, 7338, 7466, 8093]
            # rhand_indexes=np.array(list_30)[15:30]
            
            # EXTRA_IDS=[7669,7794,8022,7905,8070]
            # comments: 这里jtr是什么，为什么要计算手部关节之间的距离？
            if left_or_right==1:
                loss_touch = 0.05 * torch.sum(((jtr[:,[41,42,44,45,47,48,50,51,53,54]] - jtr[:,[40,41,43,44,46,47,49,50,52,53]])**2) * WHETHER_TOUCH.view(-1,1,1))#+ 1*torch.sum(h2o_signed**2)
            elif left_or_right==0:
                loss_touch = 0.05 * torch.sum(((jtr[:,[26, 27, 29, 30, 32, 33, 35, 36, 38, 39]] - jtr[:,[25, 26, 28, 29, 31, 32, 34, 35, 37, 38]])**2) * WHETHER_TOUCH.view(-1,1,1))#+ 1*torch.sum(h2o_signed**2)

                     
            # loss_dist_o = collision_loss*0
            collision_loss = torch.tensor(0.0).to(device)
            loss_dist_o = torch.tensor(0.0).to(device)
            loss_verts_reg = torch.tensor(0)

            ## intersection
            # sbj2obj = intersection( verts[:,hand_idx], verts_obj, sbj_m.f, torch.tensor(obj_info['faces'].astype(np.int64)).to(device), do_sbj2obj=True,do_obj2sbj=False)#,full_body=True, adjacency_matrix=None):
            # print(sbj2obj.shape,'SBJ2OBJ',verts.shape,'JKJKJKJII')
            thresh = 0.00
            #print(torch.min(sbj2obj[:,hand_idx]),'VALUE-kaolin')
            
            ratio = min(epoch / 350, 1)
            loss_touch = torch.tensor(0.0).to(device)
            ## CONTACT CALCULATION FOR PALM HALF
            contact_loop_start = time.time()
            for i in range(16):
                num_verts = HAND_INDEXES[i].shape[0]
                # comment: sbj2obj shape (N, T)
                sd_i = sbj2obj[:,HAND_INDEXES[i]]
                MASK_I = sd_i < thresh
                # print(sd_i.shape)
                # print(MASK_I.shape)
                # print(sd_i[MASK_I][0])
                whether_pene_time = torch.sum(MASK_I,dim=-1) ## T
                MASK_TIME = (whether_pene_time>0).detach()
                # print(h2o_signed.shape,MASK_TIME.shape)
                # print(MASK_TIME)
                SUMM=torch.sum(MASK_TIME)
                # print(SUMM,epoch,i)
                if torch.sum(MASK_TIME)>0:
                    
                    sd_pen=(sd_i*WHETHER_TOUCH_L.view(-1,1))[MASK_TIME]
                    
                    
                    zeros_s2o, ones_s2o = torch.zeros_like(sd_pen).float().to(device), torch.ones_like(sd_pen).float().to(device)
                
                    calc_dist = sd_pen - thresh
                    #mask_pen=((sd_pen-thresh)<0).float()
                    mask_pen = ((sd_pen-thresh)<0).float()*(torch.abs(sd_pen)<0.03).float()

                    num_pen=torch.sum(mask_pen,dim=-1).reshape(-1,1)
                    #print(calc_dist.shape,mask_pen.shape,'QQQ')
                    
                    loss_dist_o += torch.sum(torch.abs(calc_dist)*mask_pen/(num_pen+1e-8))*2 #num_pen+1e-8)

                OP_MASK_TIME = ~MASK_TIME
                #print(OP_MASK_TIME,HAND_SMALL_INDEXES[i].shape)
                
                # if torch.sum(OP_MASK_TIME)>0 or 1 : 
                if 1:
                    num_verts_small=HAND_SMALL_INDEXES[i].shape[0]
                    
                    sbj2obj_mask_first = sbj2obj * WHETHER_TOUCH_L
                    sd_closer = sbj2obj_mask_first[OP_MASK_TIME][:,HAND_SMALL_INDEXES[i]]#/num_verts_small
                    
                    ratio = min(epoch/350,1)
                    #loss_touch+=torch.mean(torch.abs(sbj2obj[OP_MASK_TIME][:,EXTRA_IN_778_RH])*WHETHER_TOUCH)*(1-ratio)*2

                    loss_touch += 5 * torch.sum(torch.abs(sd_closer)**2)

              
            for i in range(6):
                break
                
                num_verts=HAND_INDEXES[i].shape[0]
                sd_i=sbj2obj[:,HAND_INDEXES[i]]
                MASK_I=sd_i<thresh
                # print(sd_i.shape)
                # print(MASK_I.shape)
                #print(sd_i[MASK_I][0])
                whether_pene_time=torch.sum(MASK_I,dim=-1) ## T
                MASK_TIME=(whether_pene_time>0).detach()
                #print(h2o_signed.shape,MASK_TIME.shape)
                #print(MASK_TIME)
                SUMM=torch.sum(MASK_TIME)
                #print(SUMM,epoch,i)
                if torch.sum(MASK_TIME)>0:
                    
                    sd_pen=sd_i[MASK_TIME]
                    
                    zeros_s2o, ones_s2o = torch.zeros_like(sd_pen).float().to(device), torch.ones_like(sd_pen).float().to(device)
                # 4.1. Special case for sbj2obj negative values - check whether to do connected components or not.
                    calc_dist=sd_pen-thresh
                    #mask_pen=((sd_pen-thresh)<0).float()
                    mask_pen=((sd_pen-thresh)<0).float()*(torch.abs(sd_pen)<0.1).float()
                    num_pen=torch.sum(mask_pen,dim=-1).reshape(-1,1)
                    
                    loss_dist_o+=torch.sum(torch.abs(calc_dist**2)*mask_pen)*2

                OP_MASK_TIME=~MASK_TIME
                #print(OP_MASK_TIME,HAND_SMALL_INDEXES[i].shape)
                
                if torch.sum(OP_MASK_TIME)>0 or 1 :
                    num_verts_small=HAND_SMALL_INDEXES[i].shape[0]
                    sbj2obj_mask_first=sbj2obj*WHETHER_TOUCH
                    sd_closer=sbj2obj_mask_first[OP_MASK_TIME][:,HAND_SMALL_INDEXES[i]]#/num_verts_small
                    
                    #sd_closer=sbj2obj_mask_first[:,HAND_SMALL_INDEXES[i]]/num_verts_small
                    ratio=min(epoch/350,1)
                    #loss_touch+=torch.mean(torch.abs(sbj2obj[OP_MASK_TIME][:,EXTRA_IN_778_RH])*WHETHER_TOUCH)*(1-ratio)*2

                    loss_touch+=2*torch.sum(torch.abs(sd_closer)**2)
                    ## try
            contact_loop_time = time.time() - contact_loop_start
            
            #print(hand_pose_rec.shape,'QQQ')
            euler_angles = hand_pose_rec[:,:,[2,1,0]]
            #euler_angles=matrix_to_euler_angles(axis_angle_to_matrix(hand_pose_rec).float(),convention='ZYX').float()
            if not left_or_right:
                #torch.tensor([-1.0,-1.0,1.0]).to(device).reshape(1,1,3)
                euler_angles=euler_angles*(torch.tensor([-1.0,-1.0,1.0]).to(device).reshape(1,1,3))


            ## ROM Restirction
            thumb_pinky_out_notmid=euler_angles[:,[0,3,9]]
            thumb_pinky_out_mid=euler_angles[:,[1,2,4,5]] #10,11]
            thumb_pinky_out3=euler_angles[:,[10,11]]

            pinky_in=euler_angles[:,[6,7,8]]
            thumb_in=euler_angles[:,12:15]

            ## Y 0.4 Z 0.5,-0.4
            # original 1.3
            theta_max1_1=torch.tensor([1.10,0.09,0.13]).reshape(1,1,3).float().to(device)
            theta_min1_1=torch.tensor([-0.8,-0.08,-0.2]).reshape(1,1,3).float().to(device)
            loss_pinky_thumb_out_notmid=restrict_angles(thumb_pinky_out_notmid,theta_max1_1,theta_min1_1,mode=1,flag='1')
            
            theta_max1_2=torch.tensor([1.10,0.15,0.12]).reshape(1,1,3).float().to(device)
            theta_min1_2=torch.tensor([-0.1,-0.10,-0.15]).reshape(1,1,3).float().to(device)
            loss_pinky_thumb_out_mid=restrict_angles(thumb_pinky_out_mid,theta_max1_2,theta_min1_2,mode=1,flag='2')

            theta_max1_3=torch.tensor([1.10,0.15,0.10]).reshape(1,1,3).float().to(device)
            theta_min1_3=torch.tensor([-0.1,-0.10,-0.35]).reshape(1,1,3).float().to(device)
            loss_pinky_thumb_out3=restrict_angles(thumb_pinky_out3,theta_max1_3,theta_min1_3,mode=1,flag='2-2')
            
            loss_pinky_thumb_out=loss_pinky_thumb_out_notmid+loss_pinky_thumb_out_mid+loss_pinky_thumb_out3
            
            theta_max2=torch.tensor([1.10,0.5,1.10]).reshape(1,1,3).float().to(device)
            theta_min2=torch.tensor([[-0.8,-0.5,-0.8],[-0.4,-0.5,-0.8],[-0.5,-0.5,-0.8]]).reshape(1,3,3).float().to(device)

            #theta_min2=torch.tensor([]).reshape(1,1,3).float().to(device)

            loss_pinky=restrict_angles(pinky_in,theta_max2,theta_min2,mode=1,flag='3')
            #                                                  -0.1
            theta_max3=torch.tensor([[0.45,0.45,1.5],[0.45,0.45,-0.1],[0.45,0.45,1.5]]).reshape(1,3,3).float().to(device)
            
            theta_min3=torch.tensor([[-0.5,-0.5,-0.2],[-0.5,-0.5,-0.8],[-0.5,-0.5,-0.8]]).reshape(1,3,3).float().to(device)

            loss_thumb=restrict_angles(thumb_in,theta_max3,theta_min3,mode=1,flag='4')

            loss_rot_reg =loss_pinky_thumb_out+loss_pinky+loss_thumb
            #print(loss_rot_reg,'RESTRICT')
            #print(loss_rot_reg,'LLLL')
            ## restrict angles
            if left_or_right:
                d1=0.2*torch.sum(((jtr[:,[41,42,44,45,50,51,41,42]]-jtr[:,[44,45,50,51,47,48,53,54]])**2)*WHETHER_TOUCH_L.view(-1,1,1))
                #DD={2:7669,5:7794,8:8022,11:7905,14:8070}
                d2=0.2*torch.sum(((verts[:,[7669,7794,7905]]-verts[:,[7794,7905,8022]])**2)*WHETHER_TOUCH_L.view(-1,1,1))
                #print(torch.sum(WHETHER_TOUCH),'KJKJKJKJKJK')
            else:
                d1=0.2*torch.sum(((jtr[:,[26, 27, 29, 30, 35, 36, 26, 27]]-jtr[:,[29, 30, 35, 36, 32, 33, 38, 39]])**2)*WHETHER_TOUCH_L.view(-1,1,1))
                #DD={2:7669,5:7794,8:8022,11:7905,14:8070}
                d2=0.2*torch.sum(((verts[:,[4933,5058,5169]]-verts[:,[5058,5169,5286]])**2)*WHETHER_TOUCH_L.view(-1,1,1))

            loss_rot_reg+=2-d1-d2
            
            # [7669,7794,8022,7905]
            #[4933,5058,5286,5169,5361]



            if epoch>100:
                hand_verts=verts[:,hand_idx]
                # comment: 这三部分在计算什么
                smooth_start = time.time()
                loss_hand_pose_v_reg=smooth_mask(hand_pose_rec,hand_verts,(WHETHER_TOUCH).view(-1,1,1))*0.5
                loss_hand_pose_v_reg+=smooth_mask(hand_pose_rec,hand_verts,(WHETHER_OPTIMIZE-WHETHER_TOUCH).view(-1,1,1))
                loss_hand_pose_v_reg+=smooth_mask(hand_pose_rec,hand_verts, 1 - WHETHER_OPTIMIZE.view(-1,1,1))*2
                smooth_time = time.time() - smooth_start

                
                ## Initialization reg
                # comment: 非接触帧手部姿态尽量靠近平均姿态
                loss_hand_pose_v_reg+=0.05*torch.sum((hand_pose_rec-hand_mean_single.view(1,-1,3))**2*(1-WHETHER_TOUCH_L).view(-1,1,1))
                
                
            elif epoch <= 100:
                loss_hand_pose_v_reg=torch.tensor(0).to(device)
           
            ## Relieve Contact sliding
            if epoch > 200:
                CONTACT_MASK=(torch.abs(sbj2obj)<0.01).unsqueeze(2).float().detach()#*((sbj2obj)>=0).unsqueeze(2).float().detach()
                delta_inv_minus=torch.einsum('tij,tjk->tik',(verts[:,hand_idx]-sbj2obj_vector.detach()-obj_trans_tensor.reshape(-1,1,3)),obj_rot_inv_tensor.permute(0,2,1)) # T,3,3

                if epoch%2==0:
                    delta_temporal_inv_minus=0.0*((delta_inv_minus[1:]-delta_inv_minus[:-1])*WHETHER_TOUCH[1:].view(-1,1,1))**2+((delta_inv_minus[1:]-delta_inv_minus[:-1])*CONTACT_MASK[1:]*WHETHER_TOUCH[1:].view(-1,1,1))**2 ##T,N,3
                else:
                    delta_temporal_inv_minus=0.0*((delta_inv_minus[1:]-delta_inv_minus[:-1])*WHETHER_TOUCH[:-1].view(-1,1,1))**2+((delta_inv_minus[1:]-delta_inv_minus[:-1])*CONTACT_MASK[1:]*WHETHER_TOUCH[1:].view(-1,1,1))**2 ##T,N,3
                loss_hand_pose_v_reg += 0.01*torch.sum(delta_temporal_inv_minus)
                prior_start = time.time()
                SKT=hand_prior(hand_pose_rec.view(-1,45),left_or_right=left_or_right).squeeze(0)
                prior_time = time.time() - prior_start
                loss_hand_pose_v_reg += 0.1*torch.sum(SKT**2*WHETHER_OPTIMIZE)
                
            loss_body_v_reg = torch.tensor(0.0).to(device)#100 * torch.mean((((body_rec[1:-1] - body_rec[:-2]) - (body_rec[2:] - body_rec[1:-1])) ** 2).sum(dim=2).sum(dim=1)) + 100 * torch.mean(((body_rec[1:] - body_rec[:-1]) ** 2).sum(dim=2).sum(dim=1)) + 1000 * (loss_left + loss_right)
            #global hand_distance_init
            loss_hand_distance_reg=0*torch.norm((jtr[:,[45,48,51]]-jtr[:,[42,45,48]]),dim=-1)-hand_distance_init
            
            loss_v_reg = 1 * (loss_hand_pose_v_reg + loss_body_v_reg) + loss_rot_reg

            if 1:
                loss = (
                        loss_dist_o+loss_touch+
                        loss_v_reg
                        )
            else:
                loss=loss_dist_o*1+1*loss_v_reg+ 1*(loss_obj_transl_reg + loss_obj_rot_reg)

            loss_dict = {}
            # Only convert to CPU when needed for printing (every epoch)
            if epoch % 50 == 0:
                loss_dict['total'] = loss.detach().cpu().numpy()
                loss_dict['collision'] = loss_dist_o.detach().cpu().numpy()
                loss_dict['reg'] = loss_touch.detach().cpu().numpy()
                loss_dict['reg_v'] = loss_v_reg.detach().cpu().numpy()
            else:
                # Keep on GPU to avoid expensive CPU-GPU transfers
                loss_dict['total'] = loss.detach()
                loss_dict['collision'] = loss_dist_o.detach()
                loss_dict['reg'] = loss_touch.detach()
                loss_dict['reg_v'] = loss_v_reg.detach()
            
            # Print timing information every epoch
            if epoch % 50 == 0:
                print(f"  [Timing] Point2Point: {point2point_time:.3f}s, Contact Loop: {contact_loop_time:.3f}s")
                if epoch > 100:
                    print(f"  [Timing] Smooth Mask: {smooth_time:.3f}s")
                if epoch > 200:
                    print(f"  [Timing] Hand Prior: {prior_time:.3f}s")
                    
        return loss, collision_loss, loss_dict

    def calc_loss_common(hand_pose_rec,epoch):
        smplx_start_time = time.time()
        SBJ_OUTPUT=sbj_m(pose_body=body_pose, 
                            pose_hand=hand_pose_rec.view(-1,90), 
                            betas=betas_tensor, 
                            root_orient=root_tensor, 
                            trans=trans_tensor)
        smplx_time = time.time() - smplx_start_time
        
        verts=SBJ_OUTPUT.v
        jtr=SBJ_OUTPUT.Jtr
        save_path=render_path
        os.makedirs(save_path,exist_ok=True)
        save_path=os.path.join(save_path,fname)

        left_loss_start = time.time()
        loss_left, collision_loss_left, loss_dict_left = calc_loss(verts,jtr,hand_pose_rec[:,:15,:],epoch,0)
        left_loss_time = time.time() - left_loss_start
        
        right_loss_start = time.time()
        loss_right, collision_loss_right, loss_dict_right = calc_loss(verts,jtr,hand_pose_rec[:,15:,:],epoch,1)
        right_loss_time = time.time() - right_loss_start
        loss_dict_all={}

        LOSS_RATIO=1
        for key in loss_dict_right.keys():
            loss_dict_all[key]=(loss_dict_right[key]+loss_dict_left[key])*LOSS_RATIO
            #print(loss_dict_left[key],key)
        loss_all=loss_left+loss_right
        collision_loss_all=collision_loss_left+collision_loss_right
        
        # Print timing summary every epoch
        if epoch % 1 == 0:
            print(f"  [Timing] SMPLX: {smplx_time:.3f}s, Left Loss: {left_loss_time:.3f}s, Right Loss: {right_loss_time:.3f}s")
            
        return loss_all*LOSS_RATIO, collision_loss_all, loss_dict_all, None
        
    best_eval_grasp = 1e7
    tmp_smplhparams = {}
    tmp_objparams = {}
    #lhand_mean=axis_angle_to_matrix(torch.tensor(np.load('./exp/smplx_lhand_mean.npy')).reshape(1,-1,3).float()).float().repeat(bs,1,1,1).to(device)
    rhand_mean=np.load(f'./exp/{HAND_MEAN_TITLE}_rhand_mean.npy')
    rhand_mean_torch=torch.from_numpy(rhand_mean.reshape(1,-1,3)).float().to(device).repeat(frame_times,1,1)
    lhand_mean=np.load(f'./exp/{HAND_MEAN_TITLE}_lhand_mean.npy')
    lhand_mean_torch=torch.from_numpy(lhand_mean.reshape(1,-1,3)).float().to(device).repeat(frame_times,1,1)
    #hand_pose_rec=Variable(torch.tensor([0.0,0.0,0.0]).float().reshape(1,1,3).repeat(frame_times,30,1).to(device),requires_grad=True)
    hand_pose_rec=Variable(torch.cat([lhand_mean_torch,rhand_mean_torch],dim=1),requires_grad=True)

    optimizer=optim.Adam([hand_pose_rec],lr=0.001)
    print(f"Starting optimization loop for {fname} with {frame_times} frames")

    for ii in (tqdm(range(1000))):
        epoch_start_time = time.time()
        
        optimizer.zero_grad()
        
        loss_start_time = time.time()
        loss, coll, loss_dict, endflag = calc_loss_common(hand_pose_rec,ii)
        loss_time = time.time() - loss_start_time
        
        if endflag==1:
            print(f"Breaking optimization loop at epoch {ii} due to endflag=1")
            break
            
        # Handle both numpy arrays and tensors in loss_dict
        losses_str = ' '.join(['{}: {:.4f} | '.format(x, loss_dict[x].item() if hasattr(loss_dict[x], 'item') else loss_dict[x]) for x in loss_dict.keys()])
        
        backward_start_time = time.time()
        loss.backward(retain_graph=False)
        backward_time = time.time() - backward_start_time
        
        step_start_time = time.time()
        optimizer.step()
        step_time = time.time() - step_start_time
        
        epoch_time = time.time() - epoch_start_time
        
        if ii % 1 == 0:  # Print every epoch
            print(f"Epoch {ii}: Total={epoch_time:.2f}s, Loss={loss_time:.2f}s, Backward={backward_time:.2f}s, Step={step_time:.2f}s")
            print(f"  Losses: {losses_str}")
        
        #print(hand_pose_rec.grad)
        eval_grasp = loss

        if ii > 1: #and eval_grasp < best_eval_grasp:  # and contact_num>=5:
            best_eval_grasp = eval_grasp
            tmp_smplhparams = {}
            tmp_objparams = {}
            tmp_smplhparams['hand_pose'] = copy.deepcopy(hand_pose_rec.detach())

    SBJ_OUTPUT=sbj_m(pose_body=body_pose, 
                            pose_hand=tmp_smplhparams['hand_pose'].view(-1,90), 
                            betas=betas_tensor, 
                            root_orient=root_tensor, 
                            trans=trans_tensor)
    
    hand_pose=tmp_smplhparams['hand_pose'].view(-1,90).detach().cpu().numpy()
    
    # Use different export directories based on whether fixed poses were used

    os.makedirs(render_path, exist_ok=True)
    save_path = os.path.join(render_path, fname+'.npy')
    np.save(save_path,hand_pose)
    export_file_losses = f"{save_path}_optimization_0.22_f_losses/"
    os.makedirs(export_file_losses, exist_ok=True)
    save_path = os.path.join(export_file_losses, fname+'.pkl')
    with open(save_path,'wb') as f:
        pickle.dump(loss_dict,f)
        
    visualize_body_obj(SBJ_OUTPUT.v.detach().cpu().numpy(), sbj_m.f.detach().cpu().numpy(), verts_obj.detach().cpu().numpy(), obj_info['faces'], save_path=os.path.join(render_path,f'{fname}_finger.mp4'),show_frame=True, multi_angle=True)

    return tmp_smplhparams, tmp_objparams


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
        if num_fixed_frames > 0:
            print(f"  {'':15s}  Specific frames with changes: {fixed_frames_for_joint.tolist()}")
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
    obj_normals=vertex_normals(object_verts,torch.tensor(object_faces.astype(np.float32)).unsqueeze(0).repeat(object_verts.shape[0],1,1).to(device))
    distance_info = precompute_hand_object_distances(verts, object_verts, obj_normals, rhand_idx, lhand_idx)
    
    # Print contact information for both hands
    if distance_info is not None:
        # Get contact frames (where distance <= 0.02m = 2cm)
        # left_contact_frames = torch.where(distance_info['left_contact_mask'])[0].tolist()
        # right_contact_frames = torch.where(distance_info['right_contact_mask'])[0].tolist()
        left_pen_frames = torch.where(distance_info['left_pen_mask'])[0].tolist()
        right_pen_frames = torch.where(distance_info['right_pen_mask'])[0].tolist()
        
        # Convert to unique frame indices (since each frame has 778 hand vertices)
        left_pen_unique_frames = sorted(list(set(left_pen_frames)))
        right_pen_unique_frames = sorted(list(set(right_pen_frames)))
        
        print("Left hand penetration at frames:", left_pen_unique_frames)
        print("Right hand penetration at frames:", right_pen_unique_frames)
        
        # Print summary statistics
        print(f"Left hand: {len(left_pen_unique_frames)} frames with penetration out of {T} total frames")
        print(f"Right hand: {len(right_pen_unique_frames)} frames with penetration out of {T} total frames")
        
        # Also show contact frames for reference
        left_contact_frames = torch.where(distance_info['left_contact_mask'])[0].tolist()
        right_contact_frames = torch.where(distance_info['right_contact_mask'])[0].tolist()
        left_contact_unique_frames = sorted(list(set(left_contact_frames)))
        right_contact_unique_frames = sorted(list(set(right_contact_frames)))
        
        print("Left hand contact at frames:", left_contact_unique_frames)
        print("Right hand contact at frames:", right_contact_unique_frames)
        print(f"Left hand: {len(left_contact_unique_frames)} contact frames, {len(left_pen_unique_frames)} penetration frames")
        print(f"Right hand: {len(right_contact_unique_frames)} contact frames, {len(right_pen_unique_frames)} penetration frames")
        # print(f"Left hand: {len(left_contact_frames)} contact frames out of {len(distance_info['left_contact_mask'])} total frames")
        # print(f"Right hand: {len(right_contact_frames)} contact frames out of {len(distance_info['right_contact_mask'])} total frames")
    else:
        print("Distance info not available")
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
        prev_poses = poses
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
        if dataset_path_name.upper() == 'BEHAVE' or dataset_path_name.upper() == 'BEHAVE_CORRECT':
            _, updated_joints, _ = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplh', 10)
        elif dataset_path_name.upper() == 'NEURALDOME' or dataset_path_name.upper() == 'IMHD':
            _, updated_joints, _ = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplh', 16)
        elif dataset_path_name.upper() == 'CHAIRS':
            _, updated_joints, _ = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplx', 10)
        elif dataset_path_name.upper() == 'INTERCAP' or dataset_path_name.upper() == 'INTERCAP_CORRECT':
            _, updated_joints, _ = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplx', 10, True)
        elif dataset_path_name.upper() == 'OMOMO' or dataset_path_name.upper() == 'OMOMO_CORRECT':
            _, updated_joints, _ = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplx', 16)

        contact_mask_l, orient_mask_l, _ = compute_palm_contact_and_orientation(
            updated_joints, object_verts, hand='left'
        )
        
        contact_mask_r, orient_mask_r, _ = compute_palm_contact_and_orientation(
            updated_joints, object_verts, hand='right'
        )
        # Check if we should stop iterating
        
        if total_flips == 0:
            print(f"  No flips remaining across all joints - stopping iterations")
            break
        # elif total_flips > prev_total_flips:
        #     print(f"  Total flip count did not decrease ({prev_total_flips} -> {total_flips}) - stopping iterations")
        #     poses = prev_poses
        #     break
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
    
    # Regenerate joints from updated poses after joint fixing
    print("Regenerating joints from updated poses...")
    if dataset_path_name.upper() == 'BEHAVE' or dataset_path_name.upper() == 'BEHAVE_CORRECT':
        _, updated_joints, _ = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplh', 10)
    elif dataset_path_name.upper() == 'NEURALDOME' or dataset_path_name.upper() == 'IMHD':
        _, updated_joints, _ = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplh', 16)
    elif dataset_path_name.upper() == 'CHAIRS':
        _, updated_joints, _ = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplx', 10)
    elif dataset_path_name.upper() == 'INTERCAP' or dataset_path_name.upper() == 'INTERCAP_CORRECT':
        _, updated_joints, _ = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplx', 10, True)
    elif dataset_path_name.upper() == 'OMOMO' or dataset_path_name.upper() == 'OMOMO_CORRECT':
        _, updated_joints, _ = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplx', 16)
    
    # Compute twist angles for palm fixing
    twist_left_list, twist_right_list = detect_hand_twist_from_canonical_batch(poses, canonical_joints)
    contact_mask_l, orient_mask_l, _ = compute_palm_contact_and_orientation(
        updated_joints, object_verts, hand='left'
    )
    contact_mask_r, orient_mask_r, _ = compute_palm_contact_and_orientation(
        updated_joints, object_verts, hand='right'
    )

    # Fix left hand
    axis_left = canonical_joints[LEFT_WRIST] - canonical_joints[LEFT_ELBOW]
    axis_left /= np.linalg.norm(axis_left)
    poses, left_fixed_frames = fix_left_palm(
        twist_left_list, distance_info['left_close_mask'], orient_mask_l, poses, LEFT_WRIST, axis_left, updated_joints, object_verts, obj_normals
    )
    # poses, left_fixed_frames = robust_wrist_flip_fix_left(twist_left_list, orient_mask_l, poses, LEFT_WRIST, axis_left, updated_joints, object_verts)
    if left_fixed_frames:
        fix_tracker.mark_palm_fixed(left_fixed_frames, [6])  # left_wrist = index 6
    # for t in range(poses.shape[0]):
    #     print(f"Frame {t} left twist: {twist_left_list[t]} contact mask: {distance_info['left_close_mask'][t]} orient mask: {orient_mask_l[t]}")
    # # Fix right hand
    axis_right = canonical_joints[RIGHT_WRIST] - canonical_joints[RIGHT_ELBOW]
    axis_right /= np.linalg.norm(axis_right)
    poses, right_fixed_frames = fix_right_palm(
        twist_right_list, distance_info['right_close_mask'], orient_mask_r, poses, RIGHT_WRIST, axis_right, updated_joints, object_verts, obj_normals
    )
    # poses, right_fixed_frames = robust_wrist_flip_fix_right(twist_right_list, orient_mask_r, poses, RIGHT_WRIST, axis_right, updated_joints, object_verts)
    
    if right_fixed_frames:
        fix_tracker.mark_palm_fixed(right_fixed_frames, [7])  # right_wrist = index 7
    twist_left_list, twist_right_list = detect_hand_twist_from_canonical_batch(poses, canonical_joints)
    # for t in range(len(twist_left_list)):
    #     print(f"Frame {t}: Left wrist twist angle: {twist_left_list[t]:.1f} deg")
    #     print(f"Frame {t}: Right wrist twist angle: {twist_right_list[t]:.1f} deg")
    temp_step2_path = os.path.join(render_path, f'{sequence_name}_step2_palm_fixed.npz')
    np.savez(temp_step2_path, 
             poses=poses, 
             betas=betas, 
             trans=trans, 
             gender=gender)
    print(f"Step 2 temporary result saved to: {temp_step2_path}")
    
    # Step 2.5: Smooth remaining large flips using adjacent frames
    print("\n=== Step 2.5: Smoothing Remaining Large Flips ===")
    
    # Get information about which joints were fixed from the fix tracker
    all_fixed_joints = fix_tracker.get_fixed_frames_and_joints()
    
    # Create mask for the smoothing function
    # joint_optimization_mask: (8,) - which joints were optimized
    joint_optimization_mask = np.any(all_fixed_joints, axis=0)  # (8,) - True if joint was fixed in any frame
    poses = smooth_remaining_flips(poses, joint_optimization_mask, window_size=20)
    
    # Save temporary result after Step 2
    temp_step2_5_path = os.path.join(render_path, f'{sequence_name}_step2.5_smoothed.npz')
    np.savez(temp_step2_5_path, 
             poses=poses, 
             betas=betas, 
             trans=trans, 
             gender=gender)
            
    print(f"Step 2.5 temporary result saved to: {temp_step2_5_path}")

    # Step 3: Optimization with selective loss (uses SMPLX16 model)
    
    print("\n=== Step 3: Optimization with Selective Loss (SMPLX16) ===")
    all_fixed_joints = fix_tracker.get_fixed_frames_and_joints()
    total_fixed_combinations = all_fixed_joints.sum()
    original_poses = poses.copy()
    if total_fixed_combinations < 0:
        print(f"Optimizing {total_fixed_combinations} joint-frame combinations using SMPLX16 model")
        
        # Store original poses for comparison
        poses = optimize_poses_with_fixed_tracking(
            poses, betas, trans, gender, object_verts, obj_normals, fix_tracker, 
            distance_info, rhand_idx = rhand_idx, lhand_idx = lhand_idx,
            num_epochs=500, lr = 0.001, canonical_joints=canonical_joints
        )
        # poses = smooth_remaining_flips(poses, joint_o÷ptimization_mask, window_size=20, threshold=10)
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
    # optimize_finger(0, os.path.join(human_path,sequence_name), visualize=False, fname=sequence_name, render_path=render_path)

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
        save_path=os.path.join(render_path, f'{sequence_name}_optimized.mp4'),
        show_frame=True,
        multi_angle=True,
        # lhand_idx=lhand_idx,
        # rhand_idx=rhand_idx
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

def smooth_remaining_flips(poses, joint_optimization_mask, window_size=10, threshold=5):
    """
    Smooth remaining large flips for ALL joints that were fixed in previous steps.
    
    Args:
        poses: (T, 156) pose parameters
        joint_optimization_mask: (8,) mask indicating which joints were optimized
        window_size: number of adjacent frames to use for smoothing (default: 10)
    
    Returns:
        poses: smoothed pose parameters
    """
    T = poses.shape[0]
    poses_smoothed = poses.copy()
    
    # Joint mapping: joint index -> pose parameter indices
    joint_to_pose_mapping = {
        0: (39, 42),   # left_collar: 39:42
        1: (42, 45),   # right_collar: 42:45
        2: (48, 51),   # left_shoulder: 48:51
        3: (51, 54),   # right_shoulder: 51:54
        4: (54, 57),   # left_elbow: 54:57
        5: (57, 60),   # right_elbow: 57:60
        6: (60, 63),   # left_wrist: 60:63
        7: (63, 66)    # right_wrist: 63:66
    }
    
    total_flips_smoothed = 0
    
    # Process each joint that was optimized
    for joint_idx in range(8):
        if not joint_optimization_mask[joint_idx]:
            continue
            
        # Get pose parameter indices for this joint
        pose_start, pose_end = joint_to_pose_mapping[joint_idx]
        # Use detect_flips function with 10-degree threshold for this joint
        # The joint index in detect_flips corresponds to the actual joint index in the pose parameters
        joint_flip_indices, angle_diffs, _ = detect_flips(poses, joint_idx + 13 + (joint_idx > 1), threshold)
        # print(f"joint {joint_idx + 13 + (joint_idx > 1)} has {len(joint_flip_indices)} flips at frames {joint_flip_indices}")
        if len(joint_flip_indices) == 0:
            continue
        # sort the joint_flip_indices by angle_diffs
        joint_flip_indices = [x for _, x in sorted(zip(angle_diffs, joint_flip_indices))]
        # Smooth flips for this joint
        for flip_frame in joint_flip_indices:
            if flip_frame < T:
                # Calculate window boundaries
                start_frame = max(0, flip_frame - window_size // 2 + 1)
                end_frame = min(T, flip_frame + window_size // 2 + 1)
                # print(f"fixing joint {joint_idx + 13 + (joint_idx > 1)} of flip at frame {flip_frame} from {start_frame} to {end_frame}")
                # Extract joint poses for the window
                window_poses = poses[start_frame:end_frame, pose_start:pose_end].copy()
                num_frames = end_frame - start_frame
                flip_idx_in_window = flip_frame - start_frame
                
                # Get the specific poses right before and after the flip as reference poses
                reference_before_flip = window_poses[flip_idx_in_window]  # Frame right before the flip
                reference_after_flip = window_poses[flip_idx_in_window + 1]       # Frame right after the flip

                for i in range(num_frames // 2):
                    if i <= flip_idx_in_window:
                        # Before the flip: use reference_after_flip as reference (poses after flip)
                        distance_from_flip = flip_idx_in_window - i
                        # Weight decreases as we move away from the flip (max 0.4 at flip boundary)
                        reference_weight = max(0.1, 0.46 * (1.0 - i / flip_idx_in_window))
                        # reference_weight = 1.0
                        if i == 0:
                            poses_smoothed[flip_frame - i, pose_start:pose_end] = (
                                (1 - reference_weight) * window_poses[flip_idx_in_window - i] + reference_weight * window_poses[flip_idx_in_window + 1 + i]
                            )
                            poses_smoothed[flip_frame + i + 1, pose_start:pose_end] = (
                                (1 - reference_weight) * window_poses[flip_idx_in_window + 1 + i] + reference_weight * window_poses[flip_idx_in_window - i]
                            )
                        else:
                            if flip_frame - i >= 0:
                                poses_smoothed[flip_frame - i, pose_start:pose_end] = (
                                    (1 - reference_weight) * window_poses[flip_idx_in_window - i] + reference_weight * poses_smoothed[flip_frame - i + 1, pose_start:pose_end]
                                )
                            if flip_frame + i + 1 < T:
                                poses_smoothed[flip_frame + i + 1, pose_start:pose_end] = (
                                    (1 - reference_weight) * window_poses[flip_idx_in_window + 1 + i] + reference_weight * poses_smoothed[flip_frame + i + 1, pose_start:pose_end]
                                )
                total_flips_smoothed += 1
    
    return poses_smoothed

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive pipeline: Joint fix + Palm fix + Optimization")
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset root.")
    parser.add_argument("--sequence_name", required=True, help="Name of the sequence.")
    args = parser.parse_args()

    main(args.dataset_path, args.sequence_name)
