# from pose_to_verts_render import visualize_smpl, matrix_to_rotation_6d_np
import sys
from render.mesh_viz import visualize_body_obj
# sys.path.append('/media/volume/Sirui-2/interactcodes/thirdparty/OakInk-Grasp-Generation')
import math
# from lib.metrics.penetration import penetration
from scipy.spatial.distance import cdist
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"  # must come before importing trimesh
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

# Joint indices
LEFT_WRIST = 20
RIGHT_WRIST = 21
LEFT_ELBOW = 18
RIGHT_ELBOW = 19

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
    with np.load(os.path.join(MOTION_PATH, name, 'joint_fixed.npz'), allow_pickle=True) as f:
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

def draw_skeleton(joints_np, frame_idx = 0):
    # Swap Y-up to Z-up for matplotlib
    x = joints_np[:, 0]
    y = joints_np[:, 1]
    z = joints_np[:, 2]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"SMPLH Skeleton - Frame {frame_idx}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Plot joints
    ax.scatter(x, z, -y, c='red', label='All joints')

    # Plot bones
    for i, j in bone_list_omomo:
        if i < 55 and j < 55:
            ax.plot([x[i], x[j]], [z[i], z[j]], [-y[i], -y[j]], c='black')

    # Highlight left hand joints
    left_hand_indices = [20] + list(range(25, 40))  # 20 is left_wrist

    ax.scatter(x[left_hand_indices], z[left_hand_indices], -y[left_hand_indices],
            c='green', label='Left Hand')

    # Highlight right hand joints
    right_hand_indices = [21] + list(range(40, 55))  # 21 is right_wrist

    ax.scatter(x[right_hand_indices], z[right_hand_indices], -y[right_hand_indices],
            c='blue', label='Right Hand')

    # Annotate joint indices
    for i in range(55):
        ax.text(x[i], z[i], -y[i], str(i), fontsize=8)

    ax.view_init(elev=20, azim=-90)
    ax.legend()
    plt.tight_layout()
    plt.show()

def detect_hand_twist_from_canonical(pose_i, joints_canonical):
    """
    Detect wrist twist angles using canonical bone axis (rest pose).

    Args:
        pose_i: (52, 3) array of axis-angle poses per joint
        joints_canonical: (55, 3) array of joint positions in rest pose

    Returns:
        twist_left_deg, twist_right_deg: twist angles (in degrees) for each wrist
    """
    def compute_twist_angle(pose_wrist, bone_axis):
        """
        Compute rotation around the given bone axis (twist).
        """
        # Convert wrist local rotation to rotation matrix
        R_wrist = R.from_rotvec(pose_wrist).as_matrix()
        
        # Convert to axis-angle
        rotvec = pose_wrist
        angle = np.linalg.norm(rotvec)
        if angle < 1e-6:
            return 0.0

        axis = rotvec / angle
        twist_cos = np.dot(axis, bone_axis)
        twist_angle = angle * twist_cos  # project rotation onto bone axis
        return np.rad2deg(twist_angle)

    # Joint indices
    LEFT_ELBOW, LEFT_WRIST = 18, 20
    RIGHT_ELBOW, RIGHT_WRIST = 19, 21

    # Canonical bone axes from rest pose
    bone_axis_left = joints_canonical[LEFT_WRIST] - joints_canonical[LEFT_ELBOW]
    bone_axis_left /= np.linalg.norm(bone_axis_left)

    bone_axis_right = joints_canonical[RIGHT_WRIST] - joints_canonical[RIGHT_ELBOW]
    bone_axis_right /= np.linalg.norm(bone_axis_right)

    # Compute twist angles
    twist_left = compute_twist_angle(pose_i[LEFT_WRIST], bone_axis_left)
    twist_right = compute_twist_angle(pose_i[RIGHT_WRIST], bone_axis_right)

    return twist_left, twist_right


LEFT_ELBOW, LEFT_WRIST = 18, 20
RIGHT_ELBOW, RIGHT_WRIST = 19, 21


def rotate_pose_around_axis(pose_vec, axis, angle_deg):
    R_current = R.from_rotvec(pose_vec).as_matrix()
    R_correction = R.from_rotvec(np.deg2rad(angle_deg) * axis).as_matrix()
    R_fixed = R_current @ R_correction
    return R.from_matrix(R_fixed).as_rotvec()


def interpolate_pose(p1, p2, alpha):
    """Spherical interpolation between two axis-angle rotations."""
    key_times = [0, 1]
    key_rots = R.from_rotvec([p1, p2])
    slerp = Slerp(key_times, key_rots)
    return slerp(alpha).as_rotvec()

def fix_transient_flips(poses, flip_flags, joint_idx):
    T = len(poses)
    joint_slice = slice(joint_idx * 3, joint_idx * 3 + 3)

    # 1. Fix leading flip span
    first_valid = next((t for t in range(T) if not flip_flags[t]), None)
    if first_valid is not None and first_valid > 0:
        for t in range(0, first_valid):
            poses[t, joint_slice] = poses[first_valid, joint_slice]

    # 2. Fix trailing flip span
    last_valid = next((t for t in reversed(range(T)) if not flip_flags[t]), None)
    if last_valid is not None and last_valid < T - 1:
        for t in range(last_valid + 1, T):
            poses[t, joint_slice] = poses[last_valid, joint_slice]

    # 3. Fix mid-sequence flip segments via interpolation
    t = 1
    while t < T - 1:
        if flip_flags[t] and not flip_flags[t - 1]:
            # Start of a mid-sequence flipped segment
            start = t - 1
            end = t
            while end + 1 < T and flip_flags[end + 1]:
                end += 1
            if end + 1 < T:
                for i in range(start + 1, end + 1):
                    alpha = (i - start) / (end + 1 - start)
                    poses[i, joint_slice] = interpolate_pose(
                        poses[start, joint_slice],
                        poses[end + 1, joint_slice],
                        alpha
                    )
            t = end + 1
        else:
            t += 1


def fix_flips_left(
    twist_angles: np.ndarray,         # (T,) 手腕绕骨骼轴的扭转角度（度）
    orient_mask: np.ndarray,          # (T,) bool，True 表示该帧手掌面向物体
    flip_indices: np.ndarray,         # 所有 |twist[t+1]−twist[t]|>threshold 的 t 索引
    joint_idx: int,                   # 要修正的手腕 joint 在 poses 中的索引
    poses: np.ndarray,                # (T, D) 每帧的 axis‐angle pose 向量打平后的数组
    axis: np.ndarray,                  # (3,) 骨骼轴（例如肘到腕的单位向量）
    specific_angles: np.ndarray         # (T,) 每帧的手掌与物体的角度
):
    """
    根据突然的大跳变（flip_indices）和 orientation mask
    自动判定哪个段是不正常的 pose，并用 rotate_pose_around_axis 修正它。

    参数：
      twist_angles   (T,) — 每帧扭转角度
      orient_mask    (T,) — 每帧面向物体的布尔掩码
      flip_indices   (K,) — 所有相邻帧扭转跳变 > 40° 的索引 t
      joint_idx      — axis-angle 向量在 poses 中的起始维度索引
      poses         (T, D) — 每帧的 pose 向量，joint_idx*3:(joint_idx+1)*3 是腕部 rotvec
      axis           — 骨骼轴（单位向量）

    逻辑：
      1. 将整个序列按 flip_indices 分成若干段。
      2. 对每个 flip，将它左右两段分别评估：
         - 优先看 orient_mask：哪段 “不朝向” 的比例更高，就判为不正常。
         - 若比值相同，则看 twist_angles 超出 [−110,90] 的点更多者判为不正常。
      3. 用两帧差值 jump = twist[t+1] − twist[t]：
         - 若左段不正常：对左段所有帧的 wrist pose +jump°；
         - 若右段不正常：对右段所有帧的 wrist pose −jump°；
      4. 同步更新 twist_angles，避免连锁误判。
    """
    T = twist_angles.shape[0]
    fixed = np.zeros(T, dtype=bool)
    # 1) 构造段边界
    flips = np.sort(flip_indices)
    # 每个 flip t 拆分成 [seg_start, t+1) 和 [t+1, seg_end)
    boundaries = [0] + (flips + 1).tolist() + [T]
    starts = boundaries[:-1]
    ends   = boundaries[1:]

    for k, t in enumerate(flips):
        left_start, left_end   = starts[k], ends[k]     # 左段 = [left_start, left_end)
        right_start, right_end = starts[k+1], ends[k+1] # 右段 = [right_start, right_end)

        if fixed[np.arange(left_start, left_end)].mean() > 0.8 or fixed[np.arange(right_start, right_end)].mean() > 0.8:
            print(f"跳过 flip {t} 的左或右段修复（大部分已修复）")
            continue

        # 2.1) orientation 不朝向的比例
        left_len  = left_end - left_start
        right_len = right_end - right_start

        left_bad_ratio  = 1.0 - float(orient_mask[left_start:left_end].float().mean().cpu().item())
        right_bad_ratio = 1.0 - float(orient_mask[right_start:right_end].float().mean().cpu().item())
        
        print(f"left_bad_ratio: {left_bad_ratio}, right_bad_ratio: {right_bad_ratio}")

        # 2.2) 若相等则用 twist 越界点数判断，如果还是相等则用 segment length
        if math.isclose(left_bad_ratio, right_bad_ratio, rel_tol=1e-6):
            print(f"⚠️ Flip {t} 的左右段 orientation 比例相同：{left_bad_ratio:.2f} vs {right_bad_ratio:.2f}")
            # Use twist out-of-bounds as first tiebreaker
            left_oob  = ((twist_angles[left_start:left_end] >  90) |
                         (twist_angles[left_start:left_end] < -110)).sum()
            right_oob = ((twist_angles[right_start:right_end] >  90) |
                         (twist_angles[right_start:right_end] < -110)).sum()
            print(f"    Twist OOB tiebreaker: left={left_oob}, right={right_oob}")
            
            if left_oob > right_oob:
                left_bad = True
            elif right_oob > left_oob:
                left_bad = False
            else:
                # If OOB counts are also equal, use segment length as final tiebreaker
                left_bad = left_len < right_len
                print(f"    OOB counts equal, using segment length as final tiebreaker: left={left_len}, right={right_len}")
        else:
            left_bad = left_bad_ratio > right_bad_ratio


        if left_bad:
            seg = np.arange(left_start, left_end)
            ref_left_idx = left_start - 1
            ref_right_idx = right_start
        else:
            seg = np.arange(right_start, right_end)
            ref_left_idx = left_end - 1
            ref_right_idx = right_end if right_end < T else right_end - 1

        # 找两侧的参考角度
        if 0 <= ref_left_idx <= T - 1 and 0 <= ref_right_idx <= T - 1 and ref_left_idx - ref_right_idx <= 10:
            left_val = twist_angles[ref_left_idx]
            right_val = twist_angles[ref_right_idx]
            interp_vals = np.linspace(left_val, right_val, len(seg))

            print(f"🛠 Smooth-fixing {'LEFT' if left_bad else 'RIGHT'} segment [{seg[0]},{seg[-1]}] using linear interp: {left_val:.1f}° → {right_val:.1f}°")

            for idx, i in enumerate(seg):
                target_twist = interp_vals[idx]
                delta = target_twist - twist_angles[i]
                # 修改 pose
                rotvec = poses[i, joint_idx * 3 : joint_idx * 3 + 3]
                poses[i, joint_idx * 3 : joint_idx * 3 + 3] = rotate_pose_around_axis(rotvec, axis, delta)
                # 更新 twist_angle
                twist_angles[i] += delta
                fixed[i] = True
        else:
            print(f"⚠️ 无法进行平滑插值，fallback 到常规 jump 修复")
            # angle = (twist_angles[t+1] - twist_angles[t]) if left_bad else -(twist_angles[t+1] - twist_angles[t])
            angle = specific_angles[t]
            if left_bad:
                print(f"🛠 Fixing LEFT segment [{left_start},{left_end}) by {angle:.1f}° at flip {t}")
            else:
                print(f"🛠 Fixing RIGHT segment [{right_start},{right_end}) by {angle:.1f}° at flip {t}")
            for i in seg:
                rotvec = poses[i, joint_idx*3 : joint_idx*3+3]
                poses[i, joint_idx*3 : joint_idx*3+3] = rotate_pose_around_axis(rotvec, axis, angle)
                twist_angles[i] += angle
                fixed[i] = True
    


        print(f"🛠 已修复左手 flip {t} 的 {'左' if left_bad else '右'}段：{seg[0]}–{seg[-1]}")
    return poses


def fix_flips_right(
    twist_angles: np.ndarray,         # (T,) 手腕绕骨骼轴的扭转角度（度）
    orient_mask: np.ndarray,          # (T,) bool，True 表示该帧手掌面向物体
    flip_indices: np.ndarray,         # 所有 |twist[t+1]−twist[t]|>threshold 的 t 索引
    joint_idx: int,                   # 要修正的手腕 joint 在 poses 中的索引
    poses: np.ndarray,                # (T, D) 每帧的 axis‐angle pose 向量打平后的数组
    axis: np.ndarray,                  # (3,) 骨骼轴（例如肘到腕的单位向量）
    specific_angles: np.ndarray         # (T,) 每帧的手掌与物体的角度
):
    """
    根据突然的大跳变（flip_indices）和 orientation mask
    自动判定哪个段是不正常的 pose，并用 rotate_pose_around_axis 修正它。

    参数：
      twist_angles   (T,) — 每帧扭转角度
      orient_mask    (T,) — 每帧面向物体的布尔掩码
      flip_indices   (K,) — 所有相邻帧扭转跳变 > 40° 的索引 t
      joint_idx      — axis-angle 向量在 poses 中的起始维度索引
      poses         (T, D) — 每帧的 pose 向量，joint_idx*3:(joint_idx+1)*3 是腕部 rotvec
      axis           — 骨骼轴（单位向量）

    逻辑：
      1. 将整个序列按 flip_indices 分成若干段。
      2. 对每个 flip，将它左右两段分别评估：
         - 优先看 orient_mask：哪段 “不朝向” 的比例更高，就判为不正常。
         - 若比值相同，则看 twist_angles 超出 [−110,90] 的点更多者判为不正常。
      3. 用两帧差值 jump = twist[t+1] − twist[t]：
         - 若左段不正常：对左段所有帧的 wrist pose +jump°；
         - 若右段不正常：对右段所有帧的 wrist pose −jump°；
      4. 同步更新 twist_angles，避免连锁误判。
    """
    T = twist_angles.shape[0]
    fixed = np.zeros(T, dtype=bool)
    # 1) 构造段边界
    flips = np.sort(flip_indices)
    # 每个 flip t 拆分成 [seg_start, t+1) 和 [t+1, seg_end)
    boundaries = [0] + (flips + 1).tolist() + [T]
    starts = boundaries[:-1]
    ends   = boundaries[1:]

    for k, t in enumerate(flips):
        left_start, left_end   = starts[k], ends[k]     # 左段 = [left_start, left_end)
        right_start, right_end = starts[k+1], ends[k+1] # 右段 = [right_start, right_end)

        if fixed[np.arange(left_start, left_end)].mean() > 0.8 or fixed[np.arange(right_start, right_end)].mean() > 0.8:
            print(f"跳过 flip {t} 的左或右段修复（大部分已修复）")
            continue

        # 2.1) orientation 不朝向的比例
        left_len  = left_end - left_start
        right_len = right_end - right_start

        left_bad_ratio  = 1.0 - float(orient_mask[left_start:left_end].float().mean().cpu().item())
        right_bad_ratio = 1.0 - float(orient_mask[right_start:right_end].float().mean().cpu().item())

        # 2.2) 若相等则用 twist 越界点数判断，如果还是相等则用 segment length
        if math.isclose(left_bad_ratio, right_bad_ratio, rel_tol=1e-6):
            print(f"⚠️ Flip {t} 的左右段 orientation 比例相同：{left_bad_ratio:.2f} vs {right_bad_ratio:.2f}")
            # Use twist out-of-bounds as first tiebreaker
            left_oob  = ((twist_angles[left_start:left_end] > 90) |
                         (twist_angles[left_start:left_end] < -110)).sum()
            right_oob = ((twist_angles[right_start:right_end] > 110) |
                         (twist_angles[right_start:right_end] < -90)).sum()
            print(f"    Twist OOB tiebreaker: left={left_oob}, right={right_oob}")
            
            if left_oob > right_oob:
                left_bad = True
            elif right_oob > left_oob:
                left_bad = False
            else:
                # If OOB counts are also equal, use segment length as final tiebreaker
                left_bad = left_len < right_len
                print(f"    OOB counts equal, using segment length as final tiebreaker: left={left_len}, right={right_len}")
        else:
            left_bad = left_bad_ratio > right_bad_ratio


        if left_bad:
            seg = np.arange(left_start, left_end)
            ref_left_idx = left_start - 1
            ref_right_idx = right_start
        else:
            seg = np.arange(right_start, right_end)
            ref_left_idx = left_end - 1
            ref_right_idx = right_end if right_end < T else right_end - 1

        # 找两侧的参考角度
        if 0 < ref_left_idx < T - 1 and 0 < ref_right_idx < T - 1:
            left_val = twist_angles[ref_left_idx]
            right_val = twist_angles[ref_right_idx]
            interp_vals = np.linspace(left_val, right_val, len(seg))

            print(f"🛠 Smooth-fixing {'LEFT' if left_bad else 'RIGHT'} segment [{seg[0]},{seg[-1]}] using linear interp: {left_val:.1f}° → {right_val:.1f}°")

            for idx, i in enumerate(seg):
                target_twist = interp_vals[idx]
                delta = target_twist - twist_angles[i]
                # 修改 pose
                rotvec = poses[i, joint_idx * 3 : joint_idx * 3 + 3]
                poses[i, joint_idx * 3 : joint_idx * 3 + 3] = rotate_pose_around_axis(rotvec, axis, delta)
                # 更新 twist_angle
                twist_angles[i] += delta
                fixed[i] = True
        else:
            print(f"⚠️ 无法进行平滑插值，fallback 到常规 jump 修复")
            # angle = (twist_angles[t+1] - twist_angles[t]) if left_bad else -(twist_angles[t+1] - twist_angles[t])
            angle = specific_angles[t]
            if left_bad:
                print(f"🛠 Fixing LEFT segment [{left_start},{left_end}) by {angle:.1f}° at flip {t}")
            else:
                print(f"🛠 Fixing RIGHT segment [{right_start},{right_end}) by {angle:.1f}° at flip {t}")
            for i in seg:
                rotvec = poses[i, joint_idx*3 : joint_idx*3+3]
                poses[i, joint_idx*3 : joint_idx*3+3] = rotate_pose_around_axis(rotvec, axis, angle)
                twist_angles[i] += angle
                fixed[i] = True


        print(f"🛠 已修复右手 flip {t} 的 {'左' if left_bad else '右'}段：{seg[0]}–{seg[-1]}")
    return poses


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
    specific_angles = compute_palm_object_angle(human_joints, object_verts, hand='left', contact_thresh=0.09)
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
    if large_jump_indices:
        print(f"left Detected {len(large_jump_indices)} flip jump(s) — using directional persistent fix: {large_jump_indices}.")
        poses = fix_flips_left(
            twist_array,         # (T,) 手腕绕骨骼轴的扭转角度（度）
            orient_mask,          # (T,) bool，True 表示该帧手掌面向物体
            large_jump_indices,         # 所有 |twist[t+1]−twist[t]|>threshold 的 t 索引
            joint_idx,                   # 要修正的手腕 joint 在 poses 中的索引
            poses,                # (T, D) 每帧的 axis‐angle pose 向量打平后的数组
            axis,                 # (3,) 骨骼轴（例如肘到腕的单位向量）
            specific_angles      # (T,) 每帧的手掌与物体的角度
        )        # Detect and fix all persistent flips
        # detect_and_fix_all_persistent_flips_left(twist_list, poses, joint_idx, axis, threshold=jump_thresh)

    elif proportion_wrong_orient_given_contact > 0.7:
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
        else:
            # Fallback to 180 degrees if data not available
            for t in range(T):
                poses[t, joint_idx * 3 : joint_idx * 3 + 3] = rotate_pose_around_axis(
                    poses[t, joint_idx * 3 : joint_idx * 3 + 3], axis, 180
                )
    elif proportion_out_of_bounds_given_no_contact > 0.7:
        # If no contact but out of bounds proportion is large, rotate by 180 degrees
        print(f"left No contact but out of bounds - applying 180° rotation")
        for t in range(T):
            poses[t, joint_idx * 3 : joint_idx * 3 + 3] = rotate_pose_around_axis(
                poses[t, joint_idx * 3 : joint_idx * 3 + 3], axis, 180
            )

    # else:
    #     print(f"left No persistent flip — applying transient interpolation for joint {joint_idx}")
    #     # Build transient flags
    #     twist_diff = [abs(twist_list[i+1] - twist_list[i]) for i in range(T - 1)]
    #     sudden_flags = [False] + [d > jump_thresh for d in twist_diff]
    #     flip_flags = [tw > 90 or tw < -110 for tw in twist_list]
    #     combined_flags = [a or b for a, b in zip(flip_flags, sudden_flags)]
    #     fix_transient_flips(poses, combined_flags, joint_idx)
    return poses

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
    specific_angles = compute_palm_object_angle(human_joints, object_verts, hand='right', contact_thresh=0.09)
                
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
    
    if large_jump_indices:
        print(f"right Detected {len(large_jump_indices)} flip jump(s) — using directional persistent fix: {large_jump_indices}.")
        poses = fix_flips_right(
            twist_array,         # (T,) 手腕绕骨骼轴的扭转角度（度）
            orient_mask,          # (T,) bool，True 表示该帧手掌面向物体
            large_jump_indices,         # 所有 |twist[t+1]−twist[t]|>threshold 的 t 索引
            joint_idx,                   # 要修正的手腕 joint 在 poses 中的索引
            poses,                # (T, D) 每帧的 axis‐angle pose 向量打平后的数组
            axis,                 # (3,) 骨骼轴（例如肘到腕的单位向量）
            specific_angles      # (T,) 每帧的手掌与物体的角度
        )        # Detect and fix all persistent flips
    # if orient mask is false when contact mask is true at the same frame for 0.7 of the frames, apply global correction as well
    
    elif proportion_wrong_orient_given_contact > 0.7 or proportion_out_of_bounds_given_no_contact > 0.7:
        print(f"right Persistent flip (>{int(0.7 * 100)}%) — applying specific angle correction to joint {joint_idx}")
        
        # Calculate specific angles based on contact and orientation conditions
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
                    else:
                        # Apply 180 degree rotation for other cases
                        poses[t, joint_idx * 3 : joint_idx * 3 + 3] = rotate_pose_around_axis(
                            poses[t, joint_idx * 3 : joint_idx * 3 + 3], axis, 180
                        )
            else:
                # Fallback to 180 degrees if data not available
                for t in range(T):
                    poses[t, joint_idx * 3 : joint_idx * 3 + 3] = rotate_pose_around_axis(
                        poses[t, joint_idx * 3 : joint_idx * 3 + 3], axis, 180
                    )
        else:
            # If no contact but out of bounds proportion is large, rotate by 180 degrees
            print(f"right No contact but out of bounds - applying 180° rotation")
            for t in range(T):
                poses[t, joint_idx * 3 : joint_idx * 3 + 3] = rotate_pose_around_axis(
                    poses[t, joint_idx * 3 : joint_idx * 3 + 3], axis, 180
                )

    # else:
        # print(f"right No persistent flip — applying transient interpolation for joint {joint_idx}")
        # # Build transient flags
        # twist_diff = [abs(twist_list[i+1] - twist_list[i]) for i in range(T - 1)]
        # sudden_flags = [False] + [d > jump_thresh for d in twist_diff]
        # flip_flags = [tw < -90 or tw > 110 for tw in twist_list]
        # combined_flags = [a or b for a, b in zip(flip_flags, sudden_flags)]
        # fix_transient_flips(poses, combined_flags, joint_idx)
    return poses

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

def main(dataset_path, sequence_name):
    # Derived paths
    human_path = os.path.join(dataset_path, 'sequences_canonical')
    object_path = os.path.join(dataset_path, 'objects')
    dataset_path = dataset_path.split('/')[-1]

    # data_name = os.listdir(human_path)
    if dataset_path.upper() == 'GRAB':
        verts, joints, faces, poses = visualize_grab(sequence_name, human_path)
    elif dataset_path.upper() == 'BEHAVE' or dataset_path.upper() =='BEHAVE_CORRECT':
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(sequence_name, human_path, 'smplh', 10)
        canonical_joints = get_mean_pose_joints(sequence_name, gender, 'smplh', 10)
    elif dataset_path.upper() == 'NEURALDOME' or dataset_path.upper() == 'IMHD':
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(sequence_name, human_path, 'smplh', 16)
        canonical_joints = get_mean_pose_joints(sequence_name, gender, 'smplh', 16)
    elif dataset_path.upper() == 'CHAIRS':
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(sequence_name, human_path, 'smplx', 10)
        canonical_joints = get_mean_pose_joints(sequence_name, gender, 'smplx', 10)
    elif dataset_path.upper() == 'INTERCAP' or dataset_path.upper() =='INTERCAP_CORRECT':
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(sequence_name, human_path, 'smplx', 10, True)
        canonical_joints = get_mean_pose_joints(sequence_name, gender, 'smplx', 10, True)
    elif dataset_path.upper() == 'OMOMO' or dataset_path.upper() == 'OMOMO_CORRECT':
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(sequence_name, human_path, 'smplx', 16)
        canonical_joints = get_mean_pose_joints(sequence_name, gender, 'smplx', 16)
    
    with np.load(os.path.join(human_path, sequence_name, 'object.npz'), allow_pickle=True) as f:
        obj_angles, obj_trans, obj_name = f['angles'], f['trans'], str(f['name'])
    angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()

    OBJ_MESH=trimesh.load(os.path.join(object_path, obj_name, obj_name+'.obj'))
    print("object name:", obj_name)
    ov = np.array(OBJ_MESH.vertices).astype(np.float32)
    object_faces = OBJ_MESH.faces.astype(np.int32)
    device = torch.device('cuda:0')
    ov = torch.from_numpy(ov).float().to(device)
    rot = torch.tensor(angle_matrix).float().to(device)
    obj_trans = torch.tensor(obj_trans).float().to(device) # t
    object_verts=torch.einsum('ni,tij->tnj',ov,rot.permute(0,2,1))+obj_trans.unsqueeze(1)

    # Load SMPL human mesh
    human_verts = torch.from_numpy(verts.float().cpu().numpy()).to(device).float()
    human_face = torch.from_numpy(faces[0].cpu().numpy().astype(np.int32)).to(device)
    T, N_v, _ = human_verts.shape

    print("poses shape: ", poses.shape)
    print("joints shape: ", joints.shape)
    # print(joints[0])

    twist_left_list = []
    twist_right_list = []

    for frame_idx in range(T):
        pose_i = poses[frame_idx].reshape(52, 3)
        twist_left, twist_right = detect_hand_twist_from_canonical(pose_i, canonical_joints)
        twist_left_list.append(twist_left)
        twist_right_list.append(twist_right)

    contact_mask_l, orient_mask_l, _ = compute_palm_contact_and_orientation(
        joints, object_verts, hand='left'
    )
    print("Frames where orient_mask_l == 1:", torch.where(orient_mask_l == 1)[0].cpu().numpy())
    contact_mask_r, orient_mask_r, _ = compute_palm_contact_and_orientation(
        joints, object_verts, hand='right'
    )
    # LEFT
    axis_left = canonical_joints[LEFT_WRIST] - canonical_joints[LEFT_ELBOW]
    axis_left /= np.linalg.norm(axis_left)
    poses = robust_wrist_flip_fix_left(twist_left_list, contact_mask_l, orient_mask_l, poses, LEFT_WRIST, axis_left, joints, object_verts)

    # RIGHT
    axis_right = canonical_joints[RIGHT_WRIST] - canonical_joints[RIGHT_ELBOW]
    axis_right /= np.linalg.norm(axis_right)
    poses = robust_wrist_flip_fix_right(twist_right_list, contact_mask_r, orient_mask_r, poses, RIGHT_WRIST, axis_right, joints, object_verts)


    if dataset_path.upper() == 'GRAB':
        verts, joints, faces = visualize_grab(sequence_name, poses, betas, trans, gender)
    elif dataset_path.upper() == 'BEHAVE' or dataset_path.upper() =='BEHAVE_CORRECT':
        verts, joints, faces = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplh', 10)
    elif dataset_path.upper() == 'NEURALDOME' or dataset_path.upper() == 'IMHD':
        verts, joints, faces = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplh', 16)
    elif dataset_path.upper() == 'CHAIRS':
        verts, joints, faces = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplx', 10)
    elif dataset_path.upper() == 'INTERCAP' or dataset_path.upper() =='INTERCAP_CORRECT':
        verts, joints, faces = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplx', 10, True)
    elif dataset_path.upper() == 'OMOMO' or dataset_path.upper() == 'OMOMO_CORRECT':
        verts, joints, faces, poses = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplx', 16)

    twist_left_prev = 0
    twist_right_prev = 0
    # for frame_idx in range(T):
    #     joints_np = joints[frame_idx].cpu().numpy()  # (55, 3)
    #     pose_i = poses[frame_idx].reshape(52, 3)  # axis-angle per joint

    #     twist_left, twist_right = detect_hand_twist_from_canonical(pose_i, canonical_joints)
    #     if frame_idx == 0:
    #         twist_left_prev = twist_left
    #         twist_right_prev = twist_right
    #     print(f"Frame {frame_idx}: Left wrist twist angle: {twist_left:.1f} deg")
    #     print(f"Frame {frame_idx}: Right wrist twist angle: {twist_right:.1f} deg")

    #     if twist_left > 90 or twist_left < -110 :
    #         print("⚠️ Left hand might be flipped!")

    #     if twist_right > 110 or twist_right < -90 :
    #         print("⚠️ Right hand might be flipped!")


    #     if abs(twist_left - twist_left_prev) > 40 :
    #         print("⚠️ Left hand might be flipped between frames!")

    #     if abs(twist_right - twist_right_prev) > 40:
    #         print("⚠️ Right hand might be flipped between frames!")
    #     twist_left_prev = twist_left
    #     twist_right_prev = twist_right

    

    # Save fixed poses to new .npz file
    fixed_human_path = os.path.join(human_path, sequence_name, 'palm_fixed.npz')
    np.savez(fixed_human_path, 
             poses=poses, 
             betas=betas, 
             trans=trans, 
             gender=gender)
    print(f"Fixed poses saved to: {fixed_human_path}")

    render_path =f'./save_fix_palm/{dataset_path}'
    os.makedirs(render_path,exist_ok=True)

    visualize_body_obj(verts.float().detach().cpu().numpy(),faces[0].detach().cpu().numpy().astype(np.int32),object_verts.detach().cpu().numpy(),object_faces,save_path=os.path.join(render_path,f'{sequence_name}_palm_fixed.mp4'), show_frame=True, multi_angle=True)
            
    



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute penetration score for a SMPL + object sequence.")
    parser.add_argument("--dataset_path", required=True, help="Path to the root of the BEHAVE dataset.")
    parser.add_argument("--sequence_name", required=True, help="Name of the sequence (e.g., Date03_Sub03_chairwood_hand_300).")
    # parser.add_argument("--output_txt", default=False, help="list flipping sequences in a file")
    args = parser.parse_args()

    main(args.dataset_path, args.sequence_name)
    # main(args.dataset_path, args.output_txt)