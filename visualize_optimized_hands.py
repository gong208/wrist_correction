import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
import trimesh
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d
from human_body_prior.body_model.body_model import BodyModel
from render.mesh_viz import visualize_body_obj

# Set device
DEVICE_NUMBER = 0
device = torch.device(f"cuda:{DEVICE_NUMBER}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# SMPLX model paths
SMPLX_PATH = '../models/smplx'
surface_model_male_fname = os.path.join(SMPLX_PATH, "SMPLX_MALE.npz")
surface_model_female_fname = os.path.join(SMPLX_PATH, "SMPLX_FEMALE.npz")

# Initialize SMPLX models
num_betas = 16
num_expressions = None
num_dmpls = None
dmpl_fname = None

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

def obj_forward(raw_points, obj_rot_6d, obj_transl):
    """Transform object points using rotation and translation"""
    B = obj_rot_6d.shape[0]
    obj_rot = rotation_6d_to_matrix(obj_rot_6d[:, :]).permute(0, 2, 1)  # B,3,3 don't forget to transpose
    obj_points_pred = torch.matmul(raw_points.unsqueeze(0)[:, :, :3], obj_rot) + obj_transl.unsqueeze(1)
    return obj_points_pred

def visualize_optimized_sequence(sequence_name, optimized_hand_pose_path, output_dir):
    """
    Visualize a sequence using optimized hand poses
    
    Args:
        sequence_name: Name of the sequence
        optimized_hand_pose_path: Path to the optimized hand pose .npy file
        output_dir: Directory to save the visualization
    """
    
    # Load original sequence data
    OMOMO_DATA_ROOT = '../gt/omomo/sequences_canonical'
    sequence_path = os.path.join(OMOMO_DATA_ROOT, sequence_name)
    
    # Load human data
    human_npz_path = os.path.join(sequence_path, "palm_fixed.npz")
    with np.load(human_npz_path, allow_pickle=True) as f:
        poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])
    
    # Load object data
    object_npz_path = os.path.join(sequence_path, "object.npz")
    with np.load(object_npz_path, allow_pickle=True) as f:
        obj_angles, obj_trans, obj_name = f['angles'], f['trans'], str(f['name'])
    
    # Load optimized hand poses
    optimized_hand_pose = np.load(optimized_hand_pose_path)
    print(f"Loaded optimized hand pose with shape: {optimized_hand_pose.shape}")
    
    # Get SMPLX model
    sbj_m = sbj_m_all[gender]
    
    # Load object mesh
    OBJ_PATH = '../gt/omomo/objects'
    obj_dir_name = os.path.join(OBJ_PATH, obj_name)
    MMESH = trimesh.load(os.path.join(obj_dir_name, obj_name + '.obj'))
    verts_obj = np.array(MMESH.vertices)
    faces_obj = np.array(MMESH.faces)
    
    # Prepare tensors
    frame_times = optimized_hand_pose.shape[0]
    body_pose = torch.from_numpy(optimized_hand_pose[:, 3:66]).float().to(device)
    betas_tensor = torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float().to(device)
    trans_tensor = torch.from_numpy(trans).float().to(device)
    root_tensor = torch.from_numpy(optimized_hand_pose[:, :3]).float().to(device)
    hand_pose_tensor = torch.from_numpy(poses[:, 66:156]).float().to(device)
    # # Check if optimized hand pose is full pose or just hand pose
    # if optimized_hand_pose.shape[1] == 156:
    #     # This is the full pose, extract only hand pose (indices 66:156)
    #     hand_pose_tensor = torch.from_numpy(optimized_hand_pose[:, 66:156]).float().to(device)
    #     print(f"Extracted hand pose with shape: {hand_pose_tensor.shape}")
    # elif optimized_hand_pose.shape[1] == 90:
    #     # This is already just hand pose
    #     hand_pose_tensor = torch.from_numpy(optimized_hand_pose).float().to(device)
    #     print(f"Using hand pose directly with shape: {hand_pose_tensor.shape}")
    # else:
    #     raise ValueError(f"Unexpected optimized hand pose shape: {optimized_hand_pose.shape}. Expected 156 (full pose) or 90 (hand pose only)")
    
    # Generate SMPLX output with optimized hand poses
    smplx_output = sbj_m(pose_body=body_pose,
                        pose_hand=hand_pose_tensor,
                        betas=betas_tensor,
                        root_orient=root_tensor,
                        trans=trans_tensor)
    
    verts_sbj = smplx_output.v
    
    # Transform object vertices
    obj_trans_tensor = torch.from_numpy(obj_trans).float().to(device)
    obj_rot_mat_tensor = torch.from_numpy(Rotation.from_rotvec(obj_angles).as_matrix()).float().to(device)
    obj_6d_tensor = matrix_to_rotation_6d(obj_rot_mat_tensor).float()
    verts_obj_tensor = obj_forward(torch.from_numpy(verts_obj).float().to(device), 
                                  obj_6d_tensor, obj_trans_tensor)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize
    save_path = os.path.join(output_dir, f'{sequence_name}_optimized.mp4')
    print(f"Generating visualization: {save_path}")
    
    visualize_body_obj(
        verts_sbj.detach().cpu().numpy(),
        sbj_m.f.detach().cpu().numpy(),
        verts_obj_tensor.detach().cpu().numpy(),
        faces_obj,
        save_path=save_path,
        multi_angle=False,
        show_frame=True
    )
    
    print(f"Visualization saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize optimized hand poses')
    parser.add_argument('--sequence_name', type=str, required=True,
                       help='Name of the sequence to visualize')
    parser.add_argument('--optimized_hand_pose_path', type=str, required=True,
                       help='Path to the optimized hand pose .npy file')
    parser.add_argument('--output_dir', type=str, default='./optimized_visualizations',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.optimized_hand_pose_path):
        print(f"Error: Optimized hand pose file not found: {args.optimized_hand_pose_path}")
        return
    
    OMOMO_DATA_ROOT = '../gt/omomo/sequences_canonical'
    sequence_path = os.path.join(OMOMO_DATA_ROOT, args.sequence_name)
    if not os.path.exists(sequence_path):
        print(f"Error: Sequence not found: {sequence_path}")
        return
    
    # Visualize the sequence
    visualize_optimized_sequence(
        args.sequence_name,
        args.optimized_hand_pose_path,
        args.output_dir
    )

if __name__ == '__main__':
    main() 