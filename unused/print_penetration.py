import os
import argparse
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

from visualize_single_sequence import visualize_smpl
from loss import point2point_signed
from utils import vertex_normals


def load_human_mesh(dataset_path: str, sequence_name: str, use_fixed: bool):
    human_path = os.path.join(dataset_path, 'sequences_canonical')
    dataset_name = dataset_path.split('/')[-1]

    dataset_upper = dataset_name.upper()
    if dataset_upper in ('BEHAVE', 'BEHAVE_CORRECT'):
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(
            sequence_name, human_path, 'smplh', 10, use_fixed=use_fixed
        )
    elif dataset_upper in ('NEURALDOME', 'IMHD'):
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(
            sequence_name, human_path, 'smplh', 16, use_fixed=use_fixed
        )
    elif dataset_upper == 'CHAIRS':
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(
            sequence_name, human_path, 'smplx', 10, use_fixed=use_fixed
        )
    elif dataset_upper in ('INTERCAP', 'INTERCAP_CORRECT'):
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(
            sequence_name, human_path, 'smplx', 10, True, use_fixed=use_fixed
        )
    elif dataset_upper in ('OMOMO', 'OMOMO_CORRECT'):
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(
            sequence_name, human_path, 'smplx', 16, use_fixed=use_fixed
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_name}")

    return verts, faces, poses, trans, gender


def load_object_mesh(dataset_path: str, sequence_name: str):
    human_path = os.path.join(dataset_path, 'sequences_canonical')
    object_path = os.path.join(dataset_path, 'objects')

    with np.load(os.path.join(human_path, sequence_name, 'object.npz'), allow_pickle=True) as f:
        obj_angles, obj_trans, obj_name = f['angles'], f['trans'], str(f['name'])

    angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()

    obj_mesh_path = os.path.join(object_path, obj_name, obj_name + '.obj')
    
    mesh = trimesh.load(obj_mesh_path)
    verts_np = np.array(mesh.vertices).astype(np.float32)
    faces_np = mesh.faces.astype(np.int32)
    return verts_np, faces_np, angle_matrix, obj_trans


def compute_object_vertices_per_frame(ov: np.ndarray, angle_mats: np.ndarray, obj_trans: np.ndarray, device: torch.device):
    # ov: (N, 3), angle_mats: (T, 3, 3), obj_trans: (T, 3)
    ov_t = torch.from_numpy(ov).float().to(device)
    rot = torch.tensor(angle_mats).float().to(device)
    trans = torch.tensor(obj_trans).float().to(device)
    object_verts = torch.einsum('ni,tij->tnj', ov_t, rot.permute(0, 2, 1)) + trans.unsqueeze(1)
    return object_verts


def main():
    parser = argparse.ArgumentParser(description='Print per-frame hand-object penetration stats')
    parser.add_argument('--dataset_path', required=True, help='Path to the root of the dataset')
    parser.add_argument('--sequence_name', required=True, help='Sequence name (folder under sequences_canonical)')
    parser.add_argument('--use_fixed', action='store_true', help='Use palm_fixed/joint_fixed poses if available')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load human and object data
    verts, faces, poses, trans, gender = load_human_mesh(args.dataset_path, args.sequence_name, args.use_fixed)
    ov_np, faces_np, angle_mats, obj_trans = load_object_mesh(args.dataset_path, args.sequence_name)

    # Prepare tensors
    human_verts = torch.from_numpy(verts.float().cpu().numpy()).to(device).float()  # (T, V, 3)
    object_verts = compute_object_vertices_per_frame(ov_np, angle_mats, obj_trans, device)  # (T, N, 3)
    object_faces = torch.from_numpy(faces_np.astype(np.int32)).unsqueeze(0).repeat(object_verts.shape[0], 1, 1).to(device)
    obj_normals = vertex_normals(object_verts, object_faces)

    # Load hand indices
    left_hand_idx = np.load('./exp/lhand_smplx_ids.npy')
    right_hand_idx = np.load('./exp/rhand_smplx_ids.npy')

    left_hand_verts = human_verts[:, left_hand_idx, :]  # (T, 778, 3)
    right_hand_verts = human_verts[:, right_hand_idx, :]  # (T, 778, 3)

    # Compute signed distances hand->object
    _, sbj2obj_left, _, _, _, _ = point2point_signed(left_hand_verts, object_verts, y_normals=obj_normals, return_vector=True)
    _, sbj2obj_right, _, _, _, _ = point2point_signed(right_hand_verts, object_verts, y_normals=obj_normals, return_vector=True)

    # Per-frame stats
    min_per_frame_left = torch.min(sbj2obj_left, dim=1)[0]
    min_per_frame_right = torch.min(sbj2obj_right, dim=1)[0]
    count_per_frame_left = (sbj2obj_left < 0).sum(dim=1)
    count_per_frame_right = (sbj2obj_right < 0).sum(dim=1)

    # Print results
    print(f"Sequence: {args.sequence_name}")
    T = sbj2obj_left.shape[0]
    for t in range(T):
        depth_left = min_per_frame_left[t].item()
        depth_right = min_per_frame_right[t].item()
        count_left = int(count_per_frame_left[t].item())
        count_right = int(count_per_frame_right[t].item())
        print(f"frame{t} left hand depth {depth_left} with {count_left} penetrating vertices, right hand depth {depth_right} with {count_right} penetrating vertices")



if __name__ == '__main__':
    main()


