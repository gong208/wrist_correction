"""
Two-Stage Wrist and Elbow Pose Optimization with Palm Orientation Guidance

This module implements a two-stage optimization approach:
Stage 1: Optimize shoulder and elbow joints with smooth loss and reference loss
Stage 2: Lock shoulder and elbow joints, optimize only wrist joints with smooth loss, 
         reference loss, and palm orientation loss

The palm orientation loss encourages the wrist to adjust its pose so that
the palm normal direction points toward the object, improving realistic
hand-object interactions.
"""

import os
import numpy as np
import torch
from utils import markerset_ssm67_smplh, markerset_wfinger, vertex_normals
from loss import point2point_signed
import smplx
from prior import *
import trimesh
from smplx import SMPLXLayer
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import matrix_to_euler_angles, axis_angle_to_matrix, matrix_to_axis_angle, rotation_6d_to_matrix, matrix_to_rotation_6d
import torch.optim as optim
import copy
import argparse
import json
from tqdm import tqdm
from render.mesh_viz import visualize_body_obj
import pickle
from torch.utils import data
import trimesh
import time
from human_body_prior.body_model.body_model import BodyModel

DEVICE_NUMBER = 0
device = torch.device(f"cuda:{DEVICE_NUMBER}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# SMPLX model setup
SMPLX_PATH = 'models/smplx'
surface_model_male_fname = os.path.join(SMPLX_PATH, "SMPLX_MALE.npz")
surface_model_female_fname = os.path.join(SMPLX_PATH, "SMPLX_FEMALE.npz")
surface_model_neutral_fname = os.path.join(SMPLX_PATH, "SMPLX_NEUTRAL.npz")

dmpl_fname = None
num_dmpls = None 
num_expressions = None
num_betas = 16 

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

# Joint indices for shoulders, elbows, and wrists
LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST = 16, 18, 20
RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST = 17, 19, 21

# Joint to pose mapping (joint_index * 3 = pose_index in full pose):
# Joint 13 (left collar) → poses[:, 39:42]
# Joint 14 (right collar) → poses[:, 42:45] 
# Joint 15 (spine) → poses[:, 45:48] (NOT OPTIMIZED)
# Joint 16 (left shoulder) → poses[:, 48:51]
# Joint 17 (right shoulder) → poses[:, 51:54]
# Joint 18 (left elbow) → poses[:, 54:57]
# Joint 19 (right elbow) → poses[:, 57:60]
# Joint 20 (left wrist) → poses[:, 60:63]
# Joint 21 (right wrist) → poses[:, 63:66]

# Hand indices for penetration loss computation
rhand_idx = np.load('./exp/rhand_smplx_ids.npy')
lhand_idx = np.load('./exp/lhand_smplx_ids.npy')

# Hand detailed indexes for penetration loss
RHAND_INDEXES_DETAILED = []
for i in range(5):
    RHAND_INDEXES_DETAILED.append(np.load(f'./exp/index_778/hand_778_{i}_0.npy'))

LHAND_INDEXES_DETAILED = []
for i in range(5):
    LHAND_INDEXES_DETAILED.append(np.load(f'./exp/index_778/lhand_778_{i}_0.npy'))

def compute_palm_contact_and_orientation(
    human_joints: torch.Tensor,       # (T, J, 3)
    object_verts: torch.Tensor,       # (T, N, 3)
    hand: str = 'right',              # 'left' 或 'right'
    contact_thresh: float = 0.09,     # 接触距离阈值（单位 m）
    orient_angle_thresh: float = 90.0,# 朝向最大夹角阈值（度），90° 即半球面
    orient_dist_thresh: float = 0.05  # 朝向距离阈值（单位 m），用于筛选接触点
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
    contact_thresh_tensor = torch.tensor(contact_thresh, device=dists.device, dtype=dists.dtype)
    contact_mask = (dists < contact_thresh_tensor).any(dim=1)  # (T,)

    # 7) orient_mask：存在顶点既满足 distance < contact_thresh
    #    又满足夹角 ≤ orient_angle_thresh
    #    cos_thresh = cos(orient_angle_thresh)
    cos_thresh = torch.cos(torch.deg2rad(torch.tensor(orient_angle_thresh, device=normals.device)))
    orient_dist_thresh_tensor = torch.tensor(orient_dist_thresh, device=dists.device, dtype=dists.dtype)

    # 7.1) 先归一化 rel 向量
    rel_dir = rel / (dists.unsqueeze(-1) + 1e-8)       # (T, num_sampled, 3)
    # 7.2) 计算余弦：normals.unsqueeze(1) 与 rel_dir 点积
    cosines = (normals.unsqueeze(1) * rel_dir).sum(dim=2)  # (T, num_sampled)
    # 7.3) 筛选：cosines >= cos_thresh 且 dists < contact_thresh
    mask = (cosines >= cos_thresh) & (dists < orient_dist_thresh_tensor)  # (T, num_sampled)
    orient_mask = mask.any(dim=1)  # (T,)

    return contact_mask, orient_mask, normals

def compute_temporal_smoothing_loss(poses, weight_accel=0.5, weight_vel=0.25):
    """
    Compute temporal smoothing loss for pose parameters.
    
    Args:
        poses: Pose parameters (T, P)
        weight_accel: Weight for acceleration term
        weight_vel: Weight for velocity term
    
    Returns:
        smoothing_loss: Temporal smoothing loss
    """
    if poses.shape[0] < 3:
        return torch.tensor(0.0, device=poses.device, dtype=poses.dtype)
    
    # Second-order acceleration loss
    accel_loss = torch.sum((((poses[1:-1] - poses[:-2]) - 
                           (poses[2:] - poses[1:-1])) ** 2))
    
    # First-order velocity loss
    vel_loss = torch.sum(((poses[1:] - poses[:-1]) ** 2))
    
    smoothing_loss = weight_accel * accel_loss + weight_vel * vel_loss
    return smoothing_loss

def compute_reference_loss(poses, reference_poses, weight=1.0):
    """
    Compute reference loss to keep poses close to original poses.
    
    Args:
        poses: Current pose parameters (T, P)
        reference_poses: Reference pose parameters (T, P)
        weight: Weight for reference loss
    
    Returns:
        reference_loss: Reference loss
    """
    # Use smooth L1 loss for better robustness
    def smooth_l1_loss(diff, beta=0.1):
        abs_diff = torch.abs(diff)
        quadratic = torch.clamp(abs_diff, max=beta)
        linear = abs_diff - quadratic
        return 0.5 * quadratic**2 / beta + linear
    
    pose_diff = poses - reference_poses
    reference_loss = weight * torch.mean(smooth_l1_loss(pose_diff))
    return reference_loss

def compute_palm_facing_loss(joints, verts_obj_transformed, contact_mask, is_left_hand=True):
    """
    Compute palm facing loss: encourage palm normal to align with direction to object.
    Only applies loss when hand is in contact with object.
    
    Args:
        joints: Joint positions (T, J, 3)
        verts_obj_transformed: Object vertices (T, M, 3)
        contact_mask: Boolean mask (T,) indicating when hand is in contact
        is_left_hand: Whether computing for left or right hand
    
    Returns:
        facing_loss: Loss encouraging palm to face the object (only during contact)
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
    nearest_points = verts_obj_transformed[torch.arange(verts_obj_transformed.shape[0]), min_idx, :]  # (T, 3)
    
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
    
    # Only apply loss when hand is in contact with object
    facing_loss = torch.mean(torch.where(
        contact_mask,
        facing_loss_per_frame,  # Apply loss during contact
        torch.zeros_like(facing_loss_per_frame)  # No loss when not in contact
    ))
    
    return facing_loss

def stage1_optimize_shoulder_elbow(poses, betas, trans, gender, verts_obj_transformed, 
                                  initial_collar_shoulder_elbow_poses, num_epochs=100):
    """
    Stage 1: Optimize collar, shoulder and elbow joints with smooth loss, reference loss, and palm orientation loss.
    
    Args:
        poses: Full pose parameters (T, 156)
        betas: Body shape parameters
        trans: Translation parameters
        gender: Gender for SMPLX model
        verts_obj_transformed: Object vertices
        initial_collar_shoulder_elbow_poses: Initial collar, shoulder and elbow poses
        num_epochs: Number of optimization epochs
    
    Returns:
        optimized_collar_shoulder_elbow_poses: Optimized collar, shoulder and elbow poses
    """
    print("[STAGE 1] Starting collar, shoulder and elbow optimization...")
    
    # Setup SMPLX model
    sbj_m = sbj_m_all[gender]
    
    # Convert to tensors
    frame_times = poses.shape[0]
    body_pose = torch.from_numpy(poses[:, 3:66]).float().to(device)
    betas_tensor = torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float().to(device)
    trans_tensor = torch.from_numpy(trans).float().to(device)
    root_tensor = torch.from_numpy(poses[:, :3]).float().to(device)
    hand_pose_tensor = torch.from_numpy(poses[:, 66:156]).float().to(device)
    
    # Initialize collar, shoulder and elbow pose variables for optimization
    # Collar, shoulder and elbow poses (joints 13,14,16,17,18,19) - 18 parameters total
    # Excludes joint 15 (spine) which should not be optimized
    collar_shoulder_elbow_poses = torch.from_numpy(initial_collar_shoulder_elbow_poses).float().to(device).requires_grad_(True)
    
    # Create reference poses (original collar, shoulder and elbow poses)
    reference_collar_shoulder_elbow_poses = torch.from_numpy(initial_collar_shoulder_elbow_poses).float().to(device)
    
    # Optimizer
    optimizer = optim.Adam([collar_shoulder_elbow_poses], lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=50, verbose=True)
    
    best_loss = float('inf')
    best_poses = None
    patience_counter = 0
    patience = 100
    
    for epoch in tqdm(range(num_epochs), desc="Stage 1: Optimizing collar, shoulder and elbow"):
        optimizer.zero_grad()

        # Compute losses
        # 1. Temporal smoothing loss
        smooth_loss = compute_temporal_smoothing_loss(collar_shoulder_elbow_poses, weight_accel=0.35, weight_vel=0.25)
        
        # 2. Reference loss
        reference_loss = compute_reference_loss(collar_shoulder_elbow_poses, reference_collar_shoulder_elbow_poses, weight=3.0)
        
        # Combine losses
        total_loss = smooth_loss + reference_loss
        
        # Check for improvement
        if total_loss < best_loss - 1e-6:
            best_loss = total_loss
            best_poses = collar_shoulder_elbow_poses.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"[STAGE 1] Early stopping at epoch {epoch}")
            break
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([collar_shoulder_elbow_poses], max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss)
        
        if epoch % 50 == 0:
            print(f"[STAGE 1] Epoch {epoch}: Smooth: {smooth_loss.item():.6f}, Reference: {reference_loss.item():.6f}, Total: {total_loss.item():.6f}")
    
    print(f"[STAGE 1] Optimization completed. Best loss: {best_loss:.6f}")
    return best_poses

def stage2_optimize_wrist(poses, betas, trans, gender, verts_obj_transformed, 
                         optimized_collar_shoulder_elbow_poses, initial_wrist_poses, num_epochs=500):
    """
    Stage 2: Lock collar, shoulder and elbow joints, optimize only wrist joints.
    
    Args:
        poses: Full pose parameters (T, 156)
        betas: Body shape parameters
        trans: Translation parameters
        gender: Gender for SMPLX model
        verts_obj_transformed: Object vertices
        optimized_collar_shoulder_elbow_poses: Optimized collar, shoulder and elbow poses from Stage 1
        initial_wrist_poses: Initial wrist poses
        num_epochs: Number of optimization epochs
    
    Returns:
        optimized_wrist_poses: Optimized wrist poses
    """
    print("[STAGE 2] Starting wrist optimization...")
    
    # Setup SMPLX model
    sbj_m = sbj_m_all[gender]
    
    # Convert to tensors
    frame_times = poses.shape[0]
    body_pose = torch.from_numpy(poses[:, 3:66]).float().to(device)
    betas_tensor = torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float().to(device)
    trans_tensor = torch.from_numpy(trans).float().to(device)
    root_tensor = torch.from_numpy(poses[:, :3]).float().to(device)
    hand_pose_tensor = torch.from_numpy(poses[:, 66:156]).float().to(device)
    
    # Lock collar, shoulder and elbow poses (use optimized poses from Stage 1)
    locked_collar_shoulder_elbow_poses = optimized_collar_shoulder_elbow_poses.detach()
    
    # Initialize wrist pose variables for optimization
    # Wrist poses in body pose (3:66): indices 57:63 (6 parameters)
    # Left wrist: 57:60, Right wrist: 60:63
    # These correspond to full pose indices 60:66
    wrist_poses = torch.from_numpy(initial_wrist_poses).float().to(device).requires_grad_(True)
    
    # Create reference poses (original wrist poses)
    reference_wrist_poses = torch.from_numpy(initial_wrist_poses).float().to(device)
    
    # Optimizer
    optimizer = optim.Adam([wrist_poses], lr=0.005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=50, verbose=True)
    
    best_loss = float('inf')
    best_poses = None
    patience_counter = 0
    patience = 250
    for epoch in tqdm(range(num_epochs), desc="Stage 2: Optimizing wrist"):
        optimizer.zero_grad()
        
        # Create full body pose with locked collar/shoulder/elbow and current wrist poses
        # Body pose indices: 3:66 (63 parameters)
        # Need to reconstruct body pose with correct joint indices:
        # 0-35: First 12 joints (0-11)
        # 36-38: Left collar (joint 13)
        # 39-41: Right collar (joint 14) 
        # 42-44: Spine (joint 15) - NOT optimized, keep original
        # 45-47: Left shoulder (joint 16)
        # 48-50: Right shoulder (joint 17)
        # 51-53: Left elbow (joint 18)
        # 54-56: Right elbow (joint 19)
        # 57-59: Left wrist (joint 20)
        # 60-62: Right wrist (joint 21)
        
        # Reconstruct body pose with optimized poses
        body_pose_current = torch.cat([
            body_pose[:, :36],         # First 36 body pose parameters (joints 0-11)
            locked_collar_shoulder_elbow_poses[:, :3],   # Left collar (joint 13)
            locked_collar_shoulder_elbow_poses[:, 3:6],  # Right collar (joint 14)
            body_pose[:, 42:45],       # Spine (joint 15) - keep original, not optimized
            locked_collar_shoulder_elbow_poses[:, 6:9],  # Left shoulder (joint 16)
            locked_collar_shoulder_elbow_poses[:, 9:12], # Right shoulder (joint 17)
            locked_collar_shoulder_elbow_poses[:, 12:15], # Left elbow (joint 18)
            locked_collar_shoulder_elbow_poses[:, 15:18], # Right elbow (joint 19)
            wrist_poses[:, :3],        # Left wrist (joint 20)
            wrist_poses[:, 3:6],       # Right wrist (joint 21)
        ], dim=1)
        
        # Forward pass through SMPLX
        smplx_output = sbj_m(pose_body=body_pose_current, 
                           pose_hand=hand_pose_tensor, 
                           betas=betas_tensor, 
                           root_orient=root_tensor, 
                           trans=trans_tensor)
        
        joints = smplx_output.Jtr
        
        # Compute losses
        # 1. Temporal smoothing loss
        if epoch < 200:
            smooth_loss = compute_temporal_smoothing_loss(wrist_poses, weight_accel=0.5, weight_vel=0.3)
        else:
            smooth_loss = compute_temporal_smoothing_loss(wrist_poses, weight_accel=0.25, weight_vel=0.1)
        
        # 2. Reference loss

        if epoch < 200:
            reference_loss = compute_reference_loss(wrist_poses, reference_wrist_poses, weight=0.3)
        else:
            reference_loss = compute_reference_loss(wrist_poses, reference_wrist_poses, weight=0.5)
            
        # Combine losses (palm orientation loss moved to Stage 3)
        total_loss = smooth_loss + reference_loss
        
        # Check for improvement
        if total_loss < best_loss - 1e-6:
            best_loss = total_loss
            best_poses = wrist_poses.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"[STAGE 2] Early stopping at epoch {epoch}")
            break
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([wrist_poses], max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss)
        
        if epoch % 50 == 0:
            print(f"[STAGE 2] Epoch {epoch}: Smooth: {smooth_loss.item():.6f}, Reference: {reference_loss.item():.6f}, Total: {total_loss.item():.6f}")
    
    print(f"[STAGE 2] Optimization completed. Best loss: {best_loss:.6f}")
    return best_poses


def stage3_optimize_wrist_poses(poses, betas, trans, gender, verts_obj_transformed, 
                               optimized_collar_shoulder_elbow_poses, optimized_wrist_poses, 
                               num_epochs=500):
    """
    Stage 3: Optimize wrist poses directly to improve palm orientation toward objects.
    
    Args:
        poses: Full pose parameters (T, 156)
        betas: Body shape parameters
        trans: Translation parameters
        gender: Gender for SMPLX model
        verts_obj_transformed: Object vertices
        optimized_collar_shoulder_elbow_poses: Optimized collar, shoulder and elbow poses from Stage 1
        optimized_wrist_poses: Optimized wrist poses from Stage 2
        num_epochs: Number of optimization epochs
    
    Returns:
        optimized_wrist_poses: Optimized wrist poses with improved palm orientation
    """
    print("[STAGE 3] Starting direct wrist pose optimization for palm orientation...")
    
    # Setup SMPLX model
    sbj_m = sbj_m_all[gender]
    
    # Convert to tensors
    frame_times = poses.shape[0]
    body_pose = torch.from_numpy(poses[:, 3:66]).float().to(device)
    betas_tensor = torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float().to(device)
    trans_tensor = torch.from_numpy(trans).float().to(device)
    root_tensor = torch.from_numpy(poses[:, :3]).float().to(device)
    hand_pose_tensor = torch.from_numpy(poses[:, 66:156]).float().to(device)
    
    # Lock collar, shoulder and elbow poses (use optimized poses from Stage 1)
    locked_collar_shoulder_elbow_poses = optimized_collar_shoulder_elbow_poses.detach()
    
    # Get canonical joints to compute bone axes for twist rotation
    sbj_m = sbj_m_all[gender]
    # Get canonical joint positions by running forward pass with zero poses
    canonical_output = sbj_m(pose_body=torch.zeros(1, 63, device=device), 
                           pose_hand=torch.zeros(1, 90, device=device), 
                           betas=torch.zeros(1, num_betas, device=device), 
                           root_orient=torch.zeros(1, 3, device=device), 
                           trans=torch.zeros(1, 3, device=device))
    canonical_joints = canonical_output.Jtr.squeeze(0)  # Get canonical joint positions
    
    # Compute bone axes for left and right forearm (elbow to wrist direction)
    left_forearm_axis = canonical_joints[LEFT_WRIST] - canonical_joints[LEFT_ELBOW]  # (3,)
    right_forearm_axis = canonical_joints[RIGHT_WRIST] - canonical_joints[RIGHT_ELBOW]  # (3,)
    
    # Normalize bone axes
    left_forearm_axis = left_forearm_axis / torch.norm(left_forearm_axis)
    right_forearm_axis = right_forearm_axis / torch.norm(right_forearm_axis)
    
    # Extract initial twist angles from Stage 2 wrist poses
    # Calculate proper twist angles: normalize pose, get cosine with bone axis, multiply by pose magnitude
    left_wrist_poses = optimized_wrist_poses[:, :3]  # (T, 3)
    right_wrist_poses = optimized_wrist_poses[:, 3:6]  # (T, 3)
    
    # Left hand twist calculation
    left_pose_magnitudes = torch.norm(left_wrist_poses, dim=1, keepdim=True)  # (T, 1)
    left_pose_axes = left_wrist_poses / (left_pose_magnitudes + 1e-8)  # (T, 3) - normalized rotation axes
    left_cosines = torch.sum(left_pose_axes * left_forearm_axis.unsqueeze(0), dim=1)  # (T,) - cosine between axes
    left_initial_twist = left_pose_magnitudes.squeeze(1) * left_cosines  # (T,) - twist angle = magnitude * cosine
    
    # Right hand twist calculation  
    right_pose_magnitudes = torch.norm(right_wrist_poses, dim=1, keepdim=True)  # (T, 1)
    right_pose_axes = right_wrist_poses / (right_pose_magnitudes + 1e-8)  # (T, 3) - normalized rotation axes
    right_cosines = torch.sum(right_pose_axes * right_forearm_axis.unsqueeze(0), dim=1)  # (T,) - cosine between axes
    right_initial_twist = right_pose_magnitudes.squeeze(1) * right_cosines  # (T,) - twist angle = magnitude * cosine
    
    # # Convert twist angles to degrees for printing
    # left_initial_twist_deg = torch.rad2deg(left_initial_twist)
    # right_initial_twist_deg = torch.rad2deg(right_initial_twist)
    
    # # Print initial twist angles for debugging
    # print(f"[STAGE 3] Left initial twist - min: {left_initial_twist_deg.min().item():.2f}°, max: {left_initial_twist_deg.max().item():.2f}°, mean: {left_initial_twist_deg.mean().item():.2f}°")
    # print(f"[STAGE 3] Right initial twist - min: {right_initial_twist_deg.min().item():.2f}°, max: {right_initial_twist_deg.max().item():.2f}°, mean: {right_initial_twist_deg.mean().item():.2f}°")
    # print(f"[STAGE 3] Left forearm axis: {left_forearm_axis.cpu().numpy()}")
    # print(f"[STAGE 3] Right forearm axis: {right_forearm_axis.cpu().numpy()}")
    
    # # Print frame-by-frame twist angles in degrees
    # print(f"[STAGE 3] Frame-by-frame initial twist angles (degrees):")
    # for frame in range(len(left_initial_twist)):
    #     print(f"  Frame {frame:3d}: Left={left_initial_twist_deg[frame].item():8.2f}°, Right={right_initial_twist_deg[frame].item():8.2f}°")
    
    # Initialize twist angles for optimization (only rotation around forearm axis)
    left_twist_angles = left_initial_twist.clone().requires_grad_(True)  # (T,)
    right_twist_angles = right_initial_twist.clone().requires_grad_(True)  # (T,)
    
    # Optimizer
    optimizer = optim.Adam([left_twist_angles, right_twist_angles], lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=50, verbose=True)
    
    best_loss = float('inf')
    best_left_twist = None
    best_right_twist = None
    patience_counter = 0
    patience = 300
    
    for epoch in tqdm(range(num_epochs), desc="Stage 3: Optimizing wrist twist angles"):
        optimizer.zero_grad()
        
        # Convert twist angles to wrist poses by scaling the bone axes
        left_wrist_poses_current = left_twist_angles.unsqueeze(1) * left_forearm_axis.unsqueeze(0)  # (T, 3)
        right_wrist_poses_current = right_twist_angles.unsqueeze(1) * right_forearm_axis.unsqueeze(0)  # (T, 3)
        
        # Combine wrist poses
        wrist_poses_current = torch.cat([left_wrist_poses_current, right_wrist_poses_current], dim=1)
        
        # Create full body pose with locked collar/shoulder/elbow and current wrist poses
        body_pose_current = torch.cat([
            body_pose[:, :36],         # First 36 body pose parameters (joints 0-11)
            locked_collar_shoulder_elbow_poses[:, :3],   # Left collar (joint 13)
            locked_collar_shoulder_elbow_poses[:, 3:6],  # Right collar (joint 14)
            body_pose[:, 42:45],       # Spine (joint 15) - keep original, not optimized
            locked_collar_shoulder_elbow_poses[:, 6:9],  # Left shoulder (joint 16)
            locked_collar_shoulder_elbow_poses[:, 9:12], # Right shoulder (joint 17)
            locked_collar_shoulder_elbow_poses[:, 12:15], # Left elbow (joint 18)
            locked_collar_shoulder_elbow_poses[:, 15:18], # Right elbow (joint 19)
            wrist_poses_current[:, :3],        # Left wrist (joint 20)
            wrist_poses_current[:, 3:6],       # Right wrist (joint 21)
        ], dim=1)
        
        # Forward pass through SMPLX
        smplx_output = sbj_m(pose_body=body_pose_current, 
                           pose_hand=hand_pose_tensor, 
                           betas=betas_tensor, 
                           root_orient=root_tensor, 
                           trans=trans_tensor)
        
        joints = smplx_output.Jtr
        verts = smplx_output.v  # Get mesh vertices for penetration loss
        
        # Compute contact and orientation masks
        contact_l, orient_l, normals_l = compute_palm_contact_and_orientation(
            joints, verts_obj_transformed, hand='left', contact_thresh=0.09, orient_angle_thresh=80.0
        )
        contact_r, orient_r, normals_r = compute_palm_contact_and_orientation(
            joints, verts_obj_transformed, hand='right', contact_thresh=0.09, orient_angle_thresh=80.0
        )
        
        # Compute losses
        # 1. Temporal smoothing loss on twist angles
        smooth_loss_left = compute_temporal_smoothing_loss(left_twist_angles, weight_accel=0.3, weight_vel=0.2)
        smooth_loss_right = compute_temporal_smoothing_loss(right_twist_angles, weight_accel=0.3, weight_vel=0.2)
        smooth_loss = smooth_loss_left + smooth_loss_right
        
        # 2. Reference loss to keep close to Stage 2 optimized poses
        reference_loss_left = compute_reference_loss(left_twist_angles, left_initial_twist, weight=0.8)
        reference_loss_right = compute_reference_loss(right_twist_angles, right_initial_twist, weight=0.8)
        reference_loss = reference_loss_left + reference_loss_right
        
        # 3. New orientation loss: penalize when pinky and index finger distances to object are different during contact
        # This encourages the palm to be oriented so both fingers are at similar distances from the object surface
        
        # Get finger end joint positions (SMPLX joint indices)
        # Left hand: index end = 25, pinky end = 31
        # Right hand: index end = 40, pinky end = 46
        left_index_end = joints[:, 25, :]  # Left index finger end joint
        left_pinky_end = joints[:, 31, :]  # Left pinky finger end joint
        right_index_end = joints[:, 40, :]  # Right index finger end joint
        right_pinky_end = joints[:, 46, :]  # Right pinky finger end joint
        
        # Compute distances from finger end joints to nearest object point
        # Left hand
        left_index_rel_to_obj = verts_obj_transformed - left_index_end.unsqueeze(1)  # (T, N, 3)
        left_pinky_rel_to_obj = verts_obj_transformed - left_pinky_end.unsqueeze(1)  # (T, N, 3)
        
        left_index_dists = left_index_rel_to_obj.norm(dim=2)  # (T, N)
        left_pinky_dists = left_pinky_rel_to_obj.norm(dim=2)  # (T, N)
        
        left_index_min_dist, _ = left_index_dists.min(dim=1)  # (T,)
        left_pinky_min_dist, _ = left_pinky_dists.min(dim=1)  # (T,)
        
        # Right hand
        right_index_rel_to_obj = verts_obj_transformed - right_index_end.unsqueeze(1)  # (T, N, 3)
        right_pinky_rel_to_obj = verts_obj_transformed - right_pinky_end.unsqueeze(1)  # (T, N, 3)
        
        right_index_dists = right_index_rel_to_obj.norm(dim=2)  # (T, N)
        right_pinky_dists = right_pinky_rel_to_obj.norm(dim=2)  # (T, N)
        
        right_index_min_dist, _ = right_index_dists.min(dim=1)  # (T,)
        right_pinky_min_dist, _ = right_pinky_dists.min(dim=1)  # (T,)
        
        # Compute distance differences
        left_finger_dist_diff = torch.abs(left_index_min_dist - left_pinky_min_dist)  # (T,)
        right_finger_dist_diff = torch.abs(right_index_min_dist - right_pinky_min_dist)  # (T,)
        
        # Loss: penalize when finger distances are different during contact
        left_orient_loss = torch.mean(torch.where(
            contact_l, 
            left_finger_dist_diff,  # Direct penalty: larger difference = higher loss
            torch.zeros_like(left_finger_dist_diff)
        ))
        
        right_orient_loss = torch.mean(torch.where(
            contact_r, 
            right_finger_dist_diff,  # Direct penalty: larger difference = higher loss
            torch.zeros_like(right_finger_dist_diff)
        ))
        
        # Additional penalty: penalize distance between finger joints and object during contact
        left_distance_penalty = torch.mean(torch.where(
            contact_l,
            left_index_min_dist + left_pinky_min_dist,  # Penalize total distance to object
            torch.zeros_like(left_index_min_dist)
        ))
        
        right_distance_penalty = torch.mean(torch.where(
            contact_r,
            right_index_min_dist + right_pinky_min_dist,  # Penalize total distance to object
            torch.zeros_like(right_index_min_dist)
        ))
        
        # Penetration loss: penalize when hand vertices are inside the object (negative distances)
        # Use hand vertices instead of joints for more accurate penetration detection
        
        # Get hand indices
        left_hand_idx = np.load('./exp/lhand_smplx_ids.npy')
        right_hand_idx = np.load('./exp/rhand_smplx_ids.npy')
        

        
        # Palm orientation loss: encourage palm to face the object
        # Left hand palm orientation loss
        left_palm_normal_loss = compute_palm_facing_loss(joints, verts_obj_transformed, contact_l, is_left_hand=True)
        
        # Right hand palm orientation loss  
        right_palm_normal_loss = compute_palm_facing_loss(joints, verts_obj_transformed, contact_r, is_left_hand=False)
        
        # # Print palm normal angles at 400th epoch
        if epoch == 400:
            print(f"\n[STAGE 3] Epoch 400 - Palm normal angles to object:")
            
            # Compute palm normals and angles for left hand
            left_wrist_joints = joints[:, 20, :]  # Left wrist
            left_index_joints = joints[:, 25, :]  # Left index MCP
            left_pinky_joints = joints[:, 31, :]  # Left pinky MCP
            
            left_v1 = left_index_joints - left_wrist_joints
            left_v2 = left_pinky_joints - left_wrist_joints
            left_palm_normals = torch.cross(left_v1, left_v2, dim=1)
            left_palm_normals = -left_palm_normals  # Flip for left hand
            left_palm_normals = left_palm_normals / (left_palm_normals.norm(dim=1, keepdim=True) + 1e-8)
            
            left_palm_centroid = (left_wrist_joints + left_index_joints + left_pinky_joints) / 3.0
            left_rel_to_centroid = verts_obj_transformed - left_palm_centroid.unsqueeze(1)
            left_dists_to_centroid = left_rel_to_centroid.norm(dim=2)
            left_min_dists, left_min_idx = left_dists_to_centroid.min(dim=1)
            left_nearest_vec = left_rel_to_centroid[torch.arange(verts_obj_transformed.shape[0]), left_min_idx, :]
            left_nearest_dir = left_nearest_vec / (left_nearest_vec.norm(dim=1, keepdim=True) + 1e-8)
            left_palm_angles = torch.acos(torch.clamp((left_palm_normals * left_nearest_dir).sum(dim=1), -1.0, 1.0))
            left_palm_angles_deg = torch.rad2deg(left_palm_angles)
            
            # Compute palm normals and angles for right hand
            right_wrist_joints = joints[:, 21, :]  # Right wrist
            right_index_joints = joints[:, 40, :]  # Right index MCP
            right_pinky_joints = joints[:, 46, :]  # Right pinky MCP
            
            right_v1 = right_index_joints - right_wrist_joints
            right_v2 = right_pinky_joints - right_wrist_joints
            right_palm_normals = torch.cross(right_v1, right_v2, dim=1)
            right_palm_normals = right_palm_normals / (right_palm_normals.norm(dim=1, keepdim=True) + 1e-8)
            
            right_palm_centroid = (right_wrist_joints + right_index_joints + right_pinky_joints) / 3.0
            right_rel_to_centroid = verts_obj_transformed - right_palm_centroid.unsqueeze(1)
            right_dists_to_centroid = right_rel_to_centroid.norm(dim=2)
            right_min_dists, right_min_idx = right_dists_to_centroid.min(dim=1)
            right_nearest_vec = right_rel_to_centroid[torch.arange(verts_obj_transformed.shape[0]), right_min_idx, :]
            right_nearest_dir = right_nearest_vec / (right_nearest_vec.norm(dim=1, keepdim=True) + 1e-8)
            right_palm_angles = torch.acos(torch.clamp((right_palm_normals * right_nearest_dir).sum(dim=1), -1.0, 1.0))
            right_palm_angles_deg = torch.rad2deg(right_palm_angles)
            
            # Print statistics
            print(f"  Left palm angles - min: {left_palm_angles_deg.min().item():.2f}°, max: {left_palm_angles_deg.max().item():.2f}°, mean: {left_palm_angles_deg.mean().item():.2f}°")
            print(f"  Right palm angles - min: {right_palm_angles_deg.min().item():.2f}°, max: {right_palm_angles_deg.max().item():.2f}°, mean: {right_palm_angles_deg.mean().item():.2f}°")
            
            # Print frame-by-frame for all frames
            # print(f"  Frame-by-frame palm angles (degrees) - all frames:")
            # for frame in range(len(left_palm_angles_deg)):
            #     print(f"    Frame {frame:3d}: Left={left_palm_angles_deg[frame].item():8.2f}°, Right={right_palm_angles_deg[frame].item():8.2f}°")
            
            # print()  # Empty line for readability
        
        # Focus primarily on palm normal loss - this should be sufficient
        orient_loss = 10.0 * (left_palm_normal_loss + right_palm_normal_loss)
        
        # 4. Small regularization to keep twist angles reasonable
        twist_reg_loss = 0.8 * (torch.mean(left_twist_angles**2) + torch.mean(right_twist_angles**2))
        
        # Combine losses
        total_loss = smooth_loss + reference_loss + orient_loss + twist_reg_loss
        
        # Check for improvement
        if total_loss < best_loss - 1e-6:
            best_loss = total_loss
            best_left_twist = left_twist_angles.detach().clone()
            best_right_twist = right_twist_angles.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"[STAGE 3] Early stopping at epoch {epoch}")
            break
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([left_twist_angles, right_twist_angles], max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss)
        
        if epoch % 50 == 0:
            print(f"[STAGE 3] Epoch {epoch}: Smooth: {smooth_loss.item():.6f}, Reference: {reference_loss.item():.6f}, Palm: {orient_loss.item():.6f}, Reg: {twist_reg_loss.item():.6f}, Total: {total_loss.item():.6f}")
            print(f"  [STAGE 3] Palm loss breakdown - Left: {left_palm_normal_loss.item():.6f}, Right: {right_palm_normal_loss.item():.6f}")
            print(f"  [STAGE 3] Contact frames - Left: {contact_l.sum().item()}/{len(contact_l)}, Right: {contact_r.sum().item()}/{len(contact_r)}")
    
    print(f"[STAGE 3] Optimization completed. Best loss: {best_loss:.6f}")
    
    # Convert best twist angles back to wrist poses
    best_left_wrist_poses = best_left_twist.unsqueeze(1) * left_forearm_axis.unsqueeze(0)
    best_right_wrist_poses = best_right_twist.unsqueeze(1) * right_forearm_axis.unsqueeze(0)
    
    # Return the best optimized wrist poses
    final_wrist_poses = torch.cat([best_left_wrist_poses, best_right_wrist_poses], dim=1)
    
    return final_wrist_poses

def stage3_optimize_wrist_poses_full(poses, betas, trans, gender, verts_obj_transformed, 
                                    optimized_collar_shoulder_elbow_poses, optimized_wrist_poses, 
                                    num_epochs=500):
    """
    Stage 3 (Full): Optimize complete wrist poses (6 scalars: 3 for left wrist + 3 for right wrist) 
    to improve palm orientation toward objects. This version allows full wrist rotation instead of 
    constraining to forearm axis only.
    
    Args:
        poses: Full pose parameters (T, 156)
        betas: Body shape parameters
        trans: Translation parameters
        gender: Gender for SMPLX model
        verts_obj_transformed: Object vertices
        optimized_collar_shoulder_elbow_poses: Optimized collar, shoulder and elbow poses from Stage 1
        optimized_wrist_poses: Optimized wrist poses from Stage 2 (T, 6) - 3 scalars per wrist joint
        num_epochs: Number of optimization epochs
    
    Returns:
        optimized_wrist_poses: Optimized wrist poses with improved palm orientation (T, 6)
    """
    print("[STAGE 3 FULL] Starting full wrist pose optimization for palm orientation...")
    
    # Setup SMPLX model
    sbj_m = sbj_m_all[gender]
    
    # Convert to tensors
    frame_times = poses.shape[0]
    body_pose = torch.from_numpy(poses[:, 3:66]).float().to(device)
    betas_tensor = torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float().to(device)
    trans_tensor = torch.from_numpy(trans).float().to(device)
    root_tensor = torch.from_numpy(poses[:, :3]).float().to(device)
    hand_pose_tensor = torch.from_numpy(poses[:, 66:156]).float().to(device)
    
    # Lock collar, shoulder and elbow poses (use optimized poses from Stage 1)
    locked_collar_shoulder_elbow_poses = optimized_collar_shoulder_elbow_poses.detach()
    
    # Initialize full wrist pose variables for optimization (6 scalars: 3 for left wrist + 3 for right wrist)
    # Handle both numpy arrays and tensors
    if isinstance(optimized_wrist_poses, np.ndarray):
        wrist_poses = torch.from_numpy(optimized_wrist_poses).float().to(device).requires_grad_(True)
        reference_wrist_poses = torch.from_numpy(optimized_wrist_poses).float().to(device)
    else:
        # Already a tensor
        wrist_poses = optimized_wrist_poses.clone().detach().requires_grad_(True)
        reference_wrist_poses = optimized_wrist_poses.clone().detach()
    
    # Optimizer
    optimizer = optim.Adam([wrist_poses], lr=0.005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=50, verbose=True)
    
    best_loss = float('inf')
    best_poses = None
    patience_counter = 0
    patience = 300
    
    for epoch in tqdm(range(num_epochs), desc="Stage 3 Full: Optimizing full wrist poses"):
        optimizer.zero_grad()
        
        # Create full body pose with locked collar/shoulder/elbow and current wrist poses
        body_pose_current = torch.cat([
            body_pose[:, :36],         # First 36 body pose parameters (joints 0-11)
            locked_collar_shoulder_elbow_poses[:, :3],   # Left collar (joint 13)
            locked_collar_shoulder_elbow_poses[:, 3:6],  # Right collar (joint 14)
            body_pose[:, 42:45],       # Spine (joint 15) - keep original, not optimized
            locked_collar_shoulder_elbow_poses[:, 6:9],  # Left shoulder (joint 16)
            locked_collar_shoulder_elbow_poses[:, 9:12], # Right shoulder (joint 17)
            locked_collar_shoulder_elbow_poses[:, 12:15], # Left elbow (joint 18)
            locked_collar_shoulder_elbow_poses[:, 15:18], # Right elbow (joint 19)
            wrist_poses[:, :3],        # Left wrist (joint 20)
            wrist_poses[:, 3:6],       # Right wrist (joint 21)
        ], dim=1)
        
        # Forward pass through SMPLX
        smplx_output = sbj_m(pose_body=body_pose_current, 
                           pose_hand=hand_pose_tensor, 
                           betas=betas_tensor, 
                           root_orient=root_tensor, 
                           trans=trans_tensor)
        
        joints = smplx_output.Jtr
        verts = smplx_output.v  # Get mesh vertices for penetration loss
        
        # Compute contact and orientation masks
        contact_l, orient_l, normals_l = compute_palm_contact_and_orientation(
            joints, verts_obj_transformed, hand='left', contact_thresh=0.09, orient_angle_thresh=80.0
        )
        contact_r, orient_r, normals_r = compute_palm_contact_and_orientation(
            joints, verts_obj_transformed, hand='right', contact_thresh=0.09, orient_angle_thresh=80.0
        )
        
        # Compute losses
        # 1. Temporal smoothing loss on full wrist poses
        smooth_loss = compute_temporal_smoothing_loss(wrist_poses, weight_accel=0.3, weight_vel=0.2)
        
        # 2. Reference loss to keep close to Stage 2 optimized poses
        reference_loss = compute_reference_loss(wrist_poses, reference_wrist_poses, weight=1.0)
        
        # 3. Palm orientation loss: encourage palm to face the object
        # Left hand palm orientation loss
        left_palm_normal_loss = compute_palm_facing_loss(joints, verts_obj_transformed, contact_l, is_left_hand=True)
        
        # Right hand palm orientation loss  
        right_palm_normal_loss = compute_palm_facing_loss(joints, verts_obj_transformed, contact_r, is_left_hand=False)
        
        # 4. Additional orientation loss: penalize when pinky and index finger distances to object are different during contact
        # Get finger end joint positions (SMPLX joint indices)
        left_index_end = joints[:, 25, :]  # Left index finger end joint
        left_pinky_end = joints[:, 31, :]  # Left pinky finger end joint
        right_index_end = joints[:, 40, :]  # Right index finger end joint
        right_pinky_end = joints[:, 46, :]  # Right pinky finger end joint
        
        # Compute distances from finger end joints to nearest object point
        # Left hand
        left_index_rel_to_obj = verts_obj_transformed - left_index_end.unsqueeze(1)  # (T, N, 3)
        left_pinky_rel_to_obj = verts_obj_transformed - left_pinky_end.unsqueeze(1)  # (T, N, 3)
        
        left_index_dists = left_index_rel_to_obj.norm(dim=2)  # (T, N)
        left_pinky_dists = left_pinky_rel_to_obj.norm(dim=2)  # (T, N)
        
        left_index_min_dist, _ = left_index_dists.min(dim=1)  # (T,)
        left_pinky_min_dist, _ = left_pinky_dists.min(dim=1)  # (T,)
        
        # Right hand
        right_index_rel_to_obj = verts_obj_transformed - right_index_end.unsqueeze(1)  # (T, N, 3)
        right_pinky_rel_to_obj = verts_obj_transformed - right_pinky_end.unsqueeze(1)  # (T, N, 3)
        
        right_index_dists = right_index_rel_to_obj.norm(dim=2)  # (T, N)
        right_pinky_dists = right_pinky_rel_to_obj.norm(dim=2)  # (T, N)
        
        right_index_min_dist, _ = right_index_dists.min(dim=1)  # (T,)
        right_pinky_min_dist, _ = right_pinky_dists.min(dim=1)  # (T,)
        
        # Compute distance differences
        left_finger_dist_diff = torch.abs(left_index_min_dist - left_pinky_min_dist)  # (T,)
        right_finger_dist_diff = torch.abs(right_index_min_dist - right_pinky_min_dist)  # (T,)
        
        # Loss: penalize when finger distances are different during contact
        left_orient_loss = torch.mean(torch.where(
            contact_l, 
            left_finger_dist_diff,  # Direct penalty: larger difference = higher loss
            torch.zeros_like(left_finger_dist_diff)
        ))
        
        right_orient_loss = torch.mean(torch.where(
            contact_r, 
            right_finger_dist_diff,  # Direct penalty: larger difference = higher loss
            torch.zeros_like(right_finger_dist_diff)
        ))
        
        # 5. Distance penalty: penalize distance between finger joints and object during contact
        left_distance_penalty = torch.mean(torch.where(
            contact_l,
            left_index_min_dist + left_pinky_min_dist,  # Penalize total distance to object
            torch.zeros_like(left_index_min_dist)
        ))
        
        right_distance_penalty = torch.mean(torch.where(
            contact_r,
            right_index_min_dist + right_pinky_min_dist,  # Penalize total distance to object
            torch.zeros_like(right_index_min_dist)
        ))
        
        # Print palm normal angles at 400th epoch for debugging
        if epoch == 400:
            print(f"\n[STAGE 3 FULL] Epoch 400 - Palm normal angles to object:")
            
            # Compute palm normals and angles for left hand
            left_wrist_joints = joints[:, 20, :]  # Left wrist
            left_index_joints = joints[:, 25, :]  # Left index MCP
            left_pinky_joints = joints[:, 31, :]  # Left pinky MCP
            
            left_v1 = left_index_joints - left_wrist_joints
            left_v2 = left_pinky_joints - left_wrist_joints
            left_palm_normals = torch.cross(left_v1, left_v2, dim=1)
            left_palm_normals = -left_palm_normals  # Flip for left hand
            left_palm_normals = left_palm_normals / (left_palm_normals.norm(dim=1, keepdim=True) + 1e-8)
            
            left_palm_centroid = (left_wrist_joints + left_index_joints + left_pinky_joints) / 3.0
            left_rel_to_centroid = verts_obj_transformed - left_palm_centroid.unsqueeze(1)
            left_dists_to_centroid = left_rel_to_centroid.norm(dim=2)
            left_min_dists, left_min_idx = left_dists_to_centroid.min(dim=1)
            left_nearest_vec = left_rel_to_centroid[torch.arange(verts_obj_transformed.shape[0]), left_min_idx, :]
            left_nearest_dir = left_nearest_vec / (left_nearest_vec.norm(dim=1, keepdim=True) + 1e-8)
            left_palm_angles = torch.acos(torch.clamp((left_palm_normals * left_nearest_dir).sum(dim=1), -1.0, 1.0))
            left_palm_angles_deg = torch.rad2deg(left_palm_angles)
            
            # Compute palm normals and angles for right hand
            right_wrist_joints = joints[:, 21, :]  # Right wrist
            right_index_joints = joints[:, 40, :]  # Right index MCP
            right_pinky_joints = joints[:, 46, :]  # Right pinky MCP
            
            right_v1 = right_index_joints - right_wrist_joints
            right_v2 = right_pinky_joints - right_wrist_joints
            right_palm_normals = torch.cross(right_v1, right_v2, dim=1)
            right_palm_normals = right_palm_normals / (right_palm_normals.norm(dim=1, keepdim=True) + 1e-8)
            
            right_palm_centroid = (right_wrist_joints + right_index_joints + right_pinky_joints) / 3.0
            right_rel_to_centroid = verts_obj_transformed - right_palm_centroid.unsqueeze(1)
            right_dists_to_centroid = right_rel_to_centroid.norm(dim=2)
            right_min_dists, right_min_idx = right_dists_to_centroid.min(dim=1)
            right_nearest_vec = right_rel_to_centroid[torch.arange(verts_obj_transformed.shape[0]), right_min_idx, :]
            right_nearest_dir = right_nearest_vec / (right_nearest_vec.norm(dim=1, keepdim=True) + 1e-8)
            right_palm_angles = torch.acos(torch.clamp((right_palm_normals * right_nearest_dir).sum(dim=1), -1.0, 1.0))
            right_palm_angles_deg = torch.rad2deg(right_palm_angles)
            
            # Print statistics
            print(f"  Left palm angles - min: {left_palm_angles_deg.min().item():.2f}°, max: {left_palm_angles_deg.max().item():.2f}°, mean: {left_palm_angles_deg.mean().item():.2f}°")
            print(f"  Right palm angles - min: {right_palm_angles_deg.min().item():.2f}°, max: {right_palm_angles_deg.max().item():.2f}°, mean: {right_palm_angles_deg.mean().item():.2f}°")
        
        # Combine orientation losses
        orient_loss = 5.0 * (left_palm_normal_loss + right_palm_normal_loss) + \
                     2.0 * (left_orient_loss + right_orient_loss) + \
                     2.0 * (left_distance_penalty + right_distance_penalty)
        
        # 6. Small regularization to keep wrist poses reasonable
        pose_reg_loss = 1.0 * torch.mean(wrist_poses**2)
        
        # Combine all losses
        total_loss = smooth_loss + reference_loss + orient_loss + pose_reg_loss
        
        # Check for improvement
        if total_loss < best_loss - 1e-6:
            best_loss = total_loss
            best_poses = wrist_poses.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"[STAGE 3 FULL] Early stopping at epoch {epoch}")
            break
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([wrist_poses], max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss)
        
        if epoch % 50 == 0:
            print(f"[STAGE 3 FULL] Epoch {epoch}: Smooth: {smooth_loss.item():.6f}, Reference: {reference_loss.item():.6f}, Palm: {orient_loss.item():.6f}, Reg: {pose_reg_loss.item():.6f}, Total: {total_loss.item():.6f}")
            print(f"  [STAGE 3 FULL] Palm loss breakdown - Left: {left_palm_normal_loss.item():.6f}, Right: {right_palm_normal_loss.item():.6f}")
            print(f"  [STAGE 3 FULL] Contact frames - Left: {contact_l.sum().item()}/{len(contact_l)}, Right: {contact_r.sum().item()}/{len(contact_r)}")
    
    print(f"[STAGE 3 FULL] Optimization completed. Best loss: {best_loss:.6f}")
    
    return best_poses

def compute_wrist_twist_statistics(wrist_poses, gender, is_left_hand=True):
    """
    Compute twist statistics for wrist poses around the elbow-wrist axis.
    
    Args:
        wrist_poses: (T, 6) tensor with left and right wrist poses
        gender: Gender for SMPLX model
        is_left_hand: Whether computing for left or right hand
    
    Returns:
        twist_angles: (T,) tensor with twist angles in degrees
        twist_stats: dict with min, max, mean, std of twist angles
    """
    T = wrist_poses.shape[0]
    
    # Get canonical joints to compute bone axes
    sbj_m = sbj_m_all[gender]
    canonical_output = sbj_m(pose_body=torch.zeros(1, 63, device=device), 
                           pose_hand=torch.zeros(1, 90, device=device), 
                           betas=torch.zeros(1, num_betas, device=device), 
                           root_orient=torch.zeros(1, 3, device=device), 
                           trans=torch.zeros(1, 3, device=device))
    canonical_joints = canonical_output.Jtr.squeeze(0)
    
    # Get bone axis for twist computation
    if is_left_hand:
        bone_axis = canonical_joints[LEFT_WRIST] - canonical_joints[LEFT_ELBOW]
        wrist_pose = wrist_poses[:, :3]  # Left wrist
    else:
        bone_axis = canonical_joints[RIGHT_WRIST] - canonical_joints[RIGHT_ELBOW]
        wrist_pose = wrist_poses[:, 3:6]  # Right wrist
    
    # Ensure bone_axis is a tensor and normalize
    if isinstance(bone_axis, np.ndarray):
        bone_axis = torch.from_numpy(bone_axis).float().to(device)
    bone_axis = bone_axis / torch.norm(bone_axis)
    
    # Compute twist angles for each frame
    twist_angles = []
    for t in range(T):
        pose_wrist = wrist_pose[t]
        angle = torch.norm(pose_wrist)
        if angle < 1e-6:
            twist_angles.append(torch.tensor(0.0, device=pose_wrist.device))
        else:
            axis = pose_wrist / angle
            twist_cos = torch.dot(axis, bone_axis)
            twist_angle = angle * twist_cos
            twist_angles.append(torch.rad2deg(twist_angle))
    
    twist_angles = torch.stack(twist_angles)
    
    # Compute statistics
    twist_stats = {
        'min': twist_angles.min().item(),
        'max': twist_angles.max().item(),
        'mean': twist_angles.mean().item(),
        'std': twist_angles.std().item(),
        'range': twist_angles.max().item() - twist_angles.min().item()
    }
    
    return twist_angles, twist_stats

def compute_wrist_rom_loss(wrist_poses, gender, is_left_hand=True):
    """
    Compute range of motion loss for wrist poses with locked axes.
    This is a simplified version that works with the locked axis approach.
    """
    T = wrist_poses.shape[0]
    
    # Get canonical joints to compute bone axes
    sbj_m = sbj_m_all[gender]
    canonical_output = sbj_m(pose_body=torch.zeros(1, 63, device=device), 
                           pose_hand=torch.zeros(1, 90, device=device), 
                           betas=torch.zeros(1, num_betas, device=device), 
                           root_orient=torch.zeros(1, 3, device=device), 
                           trans=torch.zeros(1, 3, device=device))
    canonical_joints = canonical_output.Jtr.squeeze(0)
    
    # Get bone axis for twist computation
    if is_left_hand:
        bone_axis = canonical_joints[LEFT_WRIST] - canonical_joints[LEFT_ELBOW]
        wrist_pose = wrist_poses[:, :3]  # Left wrist
    else:
        bone_axis = canonical_joints[RIGHT_WRIST] - canonical_joints[RIGHT_ELBOW]
        wrist_pose = wrist_poses[:, 3:6]  # Right wrist
    
    # Ensure bone_axis is a tensor and normalize
    if isinstance(bone_axis, np.ndarray):
        bone_axis = torch.from_numpy(bone_axis).float().to(device)
    bone_axis = bone_axis / torch.norm(bone_axis)
    
    # Compute twist angles for each frame
    twist_angles = []
    for t in range(T):
        pose_wrist = wrist_pose[t]
        angle = torch.norm(pose_wrist)
        if angle < 1e-6:
            twist_angles.append(torch.tensor(0.0, device=pose_wrist.device))
        else:
            axis = pose_wrist / angle
            twist_cos = torch.dot(axis, bone_axis)
            twist_angle = angle * twist_cos
            twist_angles.append(torch.rad2deg(twist_angle))
    
    twist_angles = torch.stack(twist_angles)
    
    # Apply range of motion constraints
    rom_loss = restrict_wrist_angles(twist_angles, is_left_hand)
    
    return rom_loss

def optimize_wrist_pose_two_stage(index, name, visualize=False, fname='', use_full_wrist_optimization=False):
    """
    Two-stage optimization: first shoulder/elbow, then wrist only.
    
    Args:
        index: Index parameter (unused)
        name: Path to the data directory
        visualize: Whether to generate visualization
        fname: Filename for saving results
        use_full_wrist_optimization: If True, use full wrist pose optimization in Stage 3
                                    If False, use constrained wrist optimization (forearm axis only)
    """
    human_npz_path = os.path.join(name, "joint_fixed.npz")
    object_npz_path = os.path.join(name, "object.npz")
    
    with np.load(human_npz_path, allow_pickle=True) as f:
        poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])
    
    with np.load(object_npz_path, allow_pickle=True) as f:
        obj_angles, obj_trans, obj_name = f['angles'], f['trans'], str(f['name'])
    
    T = poses.shape[0]
    if T < 1:
        return
    
    # Load object mesh
    OBJ_PATH = '../gt/omomo/objects'
    obj_dir_name = os.path.join(OBJ_PATH, obj_name)
    MMESH = trimesh.load(os.path.join(obj_dir_name, obj_name + '.obj'))
    verts_obj = np.array(MMESH.vertices)
    faces_obj = np.array(MMESH.faces)
    
    # Object setup
    obj_trans_tensor = torch.from_numpy(obj_trans).float().to(device)
    obj_rot_mat_tensor = axis_angle_to_matrix(torch.from_numpy(obj_angles).float().to(device))
    verts_obj_tensor = torch.from_numpy(verts_obj).float().to(device)
    verts_obj_transformed = torch.einsum('ni,tij->tnj', verts_obj_tensor, obj_rot_mat_tensor.permute(0, 2, 1)) + obj_trans_tensor.unsqueeze(1)
    
    # Extract initial collar, shoulder/elbow and wrist poses
    # Full pose structure (0:156):
    # Root orientation: 0:3
    # Body pose: 3:66 (63 parameters)
    # Hand pose: 66:156 (90 parameters)
    
    # Joint to pose mapping:
    # Joint index 13 (left collar) → pose index 13*3=39 → full pose 39:42, body pose 36:39
    # Joint index 14 (right collar) → pose index 14*3=42 → full pose 42:45, body pose 39:42
    # Joint index 15 (spine) → pose index 15*3=45 → full pose 45:48, body pose 42:45 (NOT OPTIMIZED)
    # Joint index 16 (left shoulder) → pose index 16*3=48 → full pose 48:51, body pose 45:48
    # Joint index 17 (right shoulder) → pose index 17*3=51 → full pose 51:54, body pose 48:51
    # Joint index 18 (left elbow) → pose index 18*3=54 → full pose 54:57, body pose 51:54
    # Joint index 19 (right elbow) → pose index 19*3=57 → full pose 57:60, body pose 54:57
    # Joint index 20 (left wrist) → pose index 20*3=60 → full pose 60:63, body pose 57:60
    # Joint index 21 (right wrist) → pose index 21*3=63 → full pose 63:66, body pose 60:63
    
    # So in full pose (0:156):
    # Stage 1: Collar/shoulder/elbow (joints 13,14,16,17,18,19) → poses[:, 39:42, 42:45, 48:51, 51:54, 54:57, 57:60]
    # Stage 2: Wrist (joints 20,21) → poses[:, 60:63, 63:66]
    # Note: Joint 15 (spine) at poses[:, 45:48] is NOT optimized
    
    # Extract poses for optimization (excluding joint 15)
    initial_collar_shoulder_elbow_poses = np.concatenate([
        poses[:, 39:42],   # Left collar (joint 13)
        poses[:, 42:45],   # Right collar (joint 14) 
        poses[:, 48:51],   # Left shoulder (joint 16)
        poses[:, 51:54],   # Right shoulder (joint 17)
        poses[:, 54:57],   # Left elbow (joint 18)
        poses[:, 57:60],   # Right elbow (joint 19)
    ], axis=1)  # Shape: (T, 18)
    
    initial_wrist_poses = np.concatenate([
        poses[:, 60:63],   # Left wrist (joint 20)
        poses[:, 63:66],   # Right wrist (joint 21)
    ], axis=1)  # Shape: (T, 6)
    
    print(f"[INFO] Starting two-stage optimization for {fname}")
    print(f"[INFO] Sequence length: {T}")
    print(f"[INFO] Gender: {gender}")
    
    # Stage 1: Optimize collar, shoulder and elbow joints
    optimized_collar_shoulder_elbow_poses = stage1_optimize_shoulder_elbow(
        poses, betas, trans, gender, verts_obj_transformed, 
        initial_collar_shoulder_elbow_poses, num_epochs=200
    )


    poses_stage1 = poses.copy()
    # Reconstruct full pose with optimized poses (excluding joint 15)
    poses_stage1[:, 39:42] = optimized_collar_shoulder_elbow_poses[:, :3].cpu().numpy()      # Left collar (joint 13)
    poses_stage1[:, 42:45] = optimized_collar_shoulder_elbow_poses[:, 3:6].cpu().numpy()     # Right collar (joint 14)
    # poses_stage2[:, 45:48] = original spine (joint 15) - keep unchanged
    poses_stage1[:, 48:51] = optimized_collar_shoulder_elbow_poses[:, 6:9].cpu().numpy()     # Left shoulder (joint 16)
    poses_stage1[:, 51:54] = optimized_collar_shoulder_elbow_poses[:, 9:12].cpu().numpy()    # Right shoulder (joint 17)
    poses_stage1[:, 54:57] = optimized_collar_shoulder_elbow_poses[:, 12:15].cpu().numpy()   # Left elbow (joint 18)
    poses_stage1[:, 57:60] = optimized_collar_shoulder_elbow_poses[:, 15:18].cpu().numpy()   # Right elbow (joint 19)

    if visualize:
        vis_file = './three_stage_wrist_opt_vis/'
        os.makedirs(vis_file, exist_ok=True)
        
        verts_stage1, joints_stage1, faces_stage1, _ = regen_smpl(
            fname, poses_stage1, betas, trans, gender, 'smplx', 16
        )
        
        visualize_body_obj(
            verts_stage1, faces_stage1[0].cpu().numpy().astype(np.int32),
            verts_obj_transformed.cpu().numpy(), faces_obj,
            save_path=os.path.join(vis_file, f'{fname}_stage1.mp4'),
            show_frame=True,
            multi_angle=True
        )
        print(f"[INFO] Stage 1 visualization saved to {os.path.join(vis_file, f'{fname}_stage1.mp4')}")
    
    # Stage 2: Lock collar/shoulder/elbow and optimize wrist joints only
    optimized_wrist_poses = stage2_optimize_wrist(
        poses, betas, trans, gender, verts_obj_transformed,
        optimized_collar_shoulder_elbow_poses, initial_wrist_poses, num_epochs=600
    )
        # Combine optimized poses
    poses_stage2 = poses.copy()
    # Reconstruct full pose with optimized poses (excluding joint 15)
    poses_stage2[:, 39:42] = optimized_collar_shoulder_elbow_poses[:, :3].cpu().numpy()      # Left collar (joint 13)
    poses_stage2[:, 42:45] = optimized_collar_shoulder_elbow_poses[:, 3:6].cpu().numpy()     # Right collar (joint 14)
    # poses_stage2[:, 45:48] = original spine (joint 15) - keep unchanged
    poses_stage2[:, 48:51] = optimized_collar_shoulder_elbow_poses[:, 6:9].cpu().numpy()     # Left shoulder (joint 16)
    poses_stage2[:, 51:54] = optimized_collar_shoulder_elbow_poses[:, 9:12].cpu().numpy()    # Right shoulder (joint 17)
    poses_stage2[:, 54:57] = optimized_collar_shoulder_elbow_poses[:, 12:15].cpu().numpy()   # Left elbow (joint 18)
    poses_stage2[:, 57:60] = optimized_collar_shoulder_elbow_poses[:, 15:18].cpu().numpy()   # Right elbow (joint 19)
    poses_stage2[:, 60:63] = optimized_wrist_poses[:, :3].cpu().numpy()                      # Left wrist (joint 20)
    poses_stage2[:, 63:66] = optimized_wrist_poses[:, 3:6].cpu().numpy()                     # Right wrist (joint 21)

    # Save optimized poses
    export_file = f"./three_stage_wrist_optimization_results/"
    os.makedirs(export_file, exist_ok=True)
    save_path = os.path.join(export_file, fname + '_stage2.npy')
    np.save(save_path, poses_stage2)


    # Visualization after Stage 2 (before palm orientation optimization)
    if visualize:
        vis_file = './three_stage_wrist_opt_vis/'
        os.makedirs(vis_file, exist_ok=True)
        
        # Create poses with Stage 1 and Stage 2 results
                 # Wrist
        
        # Regenerate mesh with Stage 2 optimized poses
        verts_stage2, joints_stage2, faces_stage2, _ = regen_smpl(
            fname, poses_stage2, betas, trans, gender, 'smplx', 16
        )
        
        visualize_body_obj(
            verts_stage2, faces_stage2[0].cpu().numpy().astype(np.int32),
            verts_obj_transformed.cpu().numpy(), faces_obj,
            save_path=os.path.join(vis_file, f'{fname}_stage2_before_palm_orientation.mp4'),
            show_frame=True,
            multi_angle=True
        )
        print(f"[INFO] Stage 2 visualization saved to {os.path.join(vis_file, f'{fname}_stage2_before_palm_orientation.mp4')}")
    
    # Stage 3: Optimize wrist poses directly to improve palm orientation toward objects
    if use_full_wrist_optimization:
        print(f"[INFO] Using full wrist pose optimization for {fname}")
        optimized_wrist_poses_stage3 = stage3_optimize_wrist_poses_full(
            poses, betas, trans, gender, verts_obj_transformed,
            optimized_collar_shoulder_elbow_poses, optimized_wrist_poses, num_epochs=1000
        )
    else:
        print(f"[INFO] Using constrained wrist pose optimization (forearm axis only) for {fname}")
        optimized_wrist_poses_stage3 = stage3_optimize_wrist_poses(
            poses, betas, trans, gender, verts_obj_transformed,
            optimized_collar_shoulder_elbow_poses, optimized_wrist_poses, num_epochs=1000
        )
    
    # Log Stage 3 optimization completion
    print(f"[INFO] Stage 3 wrist pose optimization completed for {fname}")
    print(f"[INFO] Final wrist poses shape: {optimized_wrist_poses_stage3.shape}")
    
    # Combine optimized poses
    poses_optimized = poses.copy()
    # Reconstruct full pose with optimized poses (excluding joint 15)
    poses_optimized[:, 39:42] = optimized_collar_shoulder_elbow_poses[:, :3].cpu().numpy()      # Left collar (joint 13)
    poses_optimized[:, 42:45] = optimized_collar_shoulder_elbow_poses[:, 3:6].cpu().numpy()     # Right collar (joint 14)
    # poses_optimized[:, 45:48] = original spine (joint 15) - keep unchanged
    poses_optimized[:, 48:51] = optimized_collar_shoulder_elbow_poses[:, 6:9].cpu().numpy()     # Left shoulder (joint 16)
    poses_optimized[:, 51:54] = optimized_collar_shoulder_elbow_poses[:, 9:12].cpu().numpy()    # Right shoulder (joint 17)
    poses_optimized[:, 54:57] = optimized_collar_shoulder_elbow_poses[:, 12:15].cpu().numpy()   # Left elbow (joint 18)
    poses_optimized[:, 57:60] = optimized_collar_shoulder_elbow_poses[:, 15:18].cpu().numpy()   # Right elbow (joint 19)
    poses_optimized[:, 60:63] = optimized_wrist_poses_stage3[:, :3].cpu().numpy()              # Left wrist (joint 20)
    poses_optimized[:, 63:66] = optimized_wrist_poses_stage3[:, 3:6].cpu().numpy()             # Right wrist (joint 21)
    
    # Save optimized poses
    export_file = f"./three_stage_wrist_optimization_results/"
    os.makedirs(export_file, exist_ok=True)
    
    # Add suffix to distinguish between optimization approaches
    if use_full_wrist_optimization:
        save_path = os.path.join(export_file, fname + '_three_stage_full_wrist_optimized.npy')
    else:
        save_path = os.path.join(export_file, fname + '_three_stage_constrained_wrist_optimized.npy')
    
    np.save(save_path, poses_optimized)
    
    # # Save intermediate results
    # stage1_save_path = os.path.join(export_file, fname + '_stage1_collar_shoulder_elbow.npy')
    # np.save(stage1_save_path, optimized_collar_shoulder_elbow_poses.cpu().numpy())
    
    
    # stage3_save_path = os.path.join(export_file, fname + '_stage3_wrist_rotation.npy')
    # np.save(stage3_save_path, optimized_wrist_poses_stage3.cpu().numpy())
    
    print(f"[INFO] Optimization completed. Results saved to {save_path}")
    
    # Visualization
    if visualize:
        vis_file = './three_stage_wrist_opt_vis/'
        os.makedirs(vis_file, exist_ok=True)
        
        # Regenerate mesh with optimized poses
        verts_optimized, joints_optimized, faces_optimized, _ = regen_smpl(
            fname, poses_optimized, betas, trans, gender, 'smplx', 16
        )
        
        # Add suffix to distinguish between optimization approaches
        if use_full_wrist_optimization:
            vis_filename = f'{fname}_three_stage_full_wrist_optimized.mp4'
        else:
            vis_filename = f'{fname}_three_stage_constrained_wrist_optimized.mp4'
            
        visualize_body_obj(
            verts_optimized, faces_optimized[0].cpu().numpy().astype(np.int32),
            verts_obj_transformed.cpu().numpy(), faces_obj,
            save_path=os.path.join(vis_file, vis_filename),
            show_frame=True,
            multi_angle=True
        )
    
    return poses_optimized

def regen_smpl(name, poses, betas, trans, gender, model_type, num_betas, use_pca=False):
    """Regenerate SMPL mesh from poses."""
    frame_times = poses.shape[0]
    smpl_model = sbj_m_all[gender]
    
    smplx_output = smpl_model(pose_body=torch.from_numpy(poses[:, 3:66]).float().to(device), 
                             pose_hand=torch.from_numpy(poses[:, 66:156]).float().to(device), 
                             betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float().to(device), 
                             root_orient=torch.from_numpy(poses[:, :3]).float().to(device), 
                             trans=torch.from_numpy(trans).float().to(device))
    
    verts = smplx_output.v.detach().cpu()
    joints = smplx_output.Jtr.detach().cpu()
    faces = smpl_model.f.unsqueeze(0).repeat(frame_times, 1, 1)
    
    return verts, joints, faces, poses

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--full_wrist', action='store_true', 
                       help='Use full wrist pose optimization instead of constrained forearm axis optimization')
    args = parser.parse_args()
    return args


def extract_rotation_axes_from_poses(wrist_poses):
    """
    Extract rotation axes from wrist poses.
    
    Args:
        wrist_poses: (T, 6) tensor with left and right wrist poses
        
    Returns:
        dict: {'left_wrist': (T, 3), 'right_wrist': (T, 3)} rotation axes
    """
    left_wrist_poses = wrist_poses[:, :3]
    right_wrist_poses = wrist_poses[:, 3:6]
    
    # Compute rotation axes (normalized direction vectors)
    left_axes = left_wrist_poses / (torch.norm(left_wrist_poses, dim=1, keepdim=True) + 1e-8)
    right_axes = right_wrist_poses / (torch.norm(right_wrist_poses, dim=1, keepdim=True) + 1e-8)
    
    return {
        'left_wrist': left_axes.cpu().numpy(),
        'right_wrist': right_axes.cpu().numpy()
    }

def create_fixed_rotation_axes(frame_times, axis_type='canonical'):
    """
    Create fixed rotation axes for wrist optimization.
    
    Args:
        frame_times: Number of frames
        axis_type: Type of fixed axis ('canonical', 'x_axis', 'y_axis', 'z_axis', 'custom')
        
    Returns:
        dict: {'left_wrist': (T, 3), 'right_wrist': (T, 3)} rotation axes
    """
    if axis_type == 'canonical':
        # Use canonical bone axes (elbow to wrist direction)
        # These would typically come from the SMPLX rest pose
        left_axis = np.array([0.0, 0.0, 1.0])  # Example: pointing along forearm
        right_axis = np.array([0.0, 0.0, 1.0])
    elif axis_type == 'x_axis':
        left_axis = np.array([1.0, 0.0, 0.0])
        right_axis = np.array([1.0, 0.0, 0.0])
    elif axis_type == 'y_axis':
        left_axis = np.array([0.0, 1.0, 0.0])
        right_axis = np.array([0.0, 1.0, 0.0])
    elif axis_type == 'z_axis':
        left_axis = np.array([0.0, 0.0, 1.0])
        right_axis = np.array([0.0, 0.0, 1.0])
    elif axis_type == 'custom':
        # Custom axes - you can define specific axes for your use case
        left_axis = np.array([0.707, 0.0, 0.707])  # 45 degrees between x and z
        right_axis = np.array([0.707, 0.0, 0.707])
    else:
        raise ValueError(f"Unknown axis_type: {axis_type}")
    
    # Normalize axes
    left_axis = left_axis / np.linalg.norm(left_axis)
    right_axis = right_axis / np.linalg.norm(right_axis)
    
    # Repeat for all frames
    left_axes = np.tile(left_axis, (frame_times, 1))
    right_axes = np.tile(right_axis, (frame_times, 1))
    
    return {
        'left_wrist': left_axes,
        'right_wrist': right_axes
    }



if __name__ == '__main__':
    args = parse_args()
    SECTION_NUMBER = args.number
    USE_FULL_WRIST_OPTIMIZATION = args.full_wrist
    
    OMOMO_DATA_ROOT = f'../gt/omomo/sequences_canonical'
    
    LISTDIR = np.load('./omomo_listdir.npy', allow_pickle=True)
    LEN_LISTDIR = len(LISTDIR)
    
    # If we have fewer than 30 files, process all files regardless of section number
    if LEN_LISTDIR <= 30:
        LIST = list(LISTDIR)
    else:
        # For larger datasets, use proper sectioning
        num_sections = 30
        SEQ_LEN = LEN_LISTDIR // num_sections
        start_idx = SECTION_NUMBER * SEQ_LEN
        end_idx = start_idx + SEQ_LEN if SECTION_NUMBER < num_sections - 1 else LEN_LISTDIR
        LIST = list(LISTDIR[start_idx:end_idx])
    
    if USE_FULL_WRIST_OPTIMIZATION:
        export_file = f"./three_stage_wrist_optimization_results_full/"
    else:
        export_file = f"./three_stage_wrist_optimization_results_constrained/"
    
    print(f"Total files: {LEN_LISTDIR}")
    print(f"Files to process: {len(LIST)}")
    print(f"Section number: {SECTION_NUMBER}")
    for i, fn in tqdm(enumerate(LIST)):
        try:
            # Check for existing files based on optimization type
            if USE_FULL_WRIST_OPTIMIZATION:
                existing_file = os.path.join(export_file, fn + '_three_stage_full_wrist_optimized.npy')
            else:
                existing_file = os.path.join(export_file, fn + '_three_stage_constrained_wrist_optimized.npy')
                
            if os.path.isfile(existing_file):
                continue
            
            name = os.path.join(OMOMO_DATA_ROOT, fn)
            optimize_wrist_pose_two_stage(0, name, True, fn, USE_FULL_WRIST_OPTIMIZATION)
        except Exception as e:
            print(f"[ERROR] Error processing {fn}: {e}")
            print(f"[ERROR] Error type: {type(e).__name__}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            continue 