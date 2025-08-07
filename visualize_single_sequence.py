import os
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
import argparse
from render.mesh_viz import visualize_body_obj
import smplx
from human_body_prior.body_model.body_model import BodyModel

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

def main():
    parser = argparse.ArgumentParser(description="Visualize a single sequence from a dataset")
    parser.add_argument("--dataset_path", required=True, help="Path to the root of the dataset")
    parser.add_argument("--sequence_name", required=True, help="Name of the sequence to visualize")
    parser.add_argument("--use_fixed", action="store_true", help="Use palm_fixed.npz instead of human.npz")
    parser.add_argument("--output_path", default="./visualization_output", help="Output directory for rendered video")
    parser.add_argument("--highlight_frame", type=int, help="Highlight a specific frame")
    parser.add_argument("--highlight_vertex", type=int, help="Highlight a specific vertex")
    parser.add_argument("--multi_angle", action="store_true", help="Render from multiple angles")
    
    args = parser.parse_args()
    
    # Derived paths
    human_path = os.path.join(args.dataset_path, 'sequences_canonical')
    object_path = os.path.join(args.dataset_path, 'objects')
    dataset_name = args.dataset_path.split('/')[-1]
    
    # Check if sequence exists
    sequence_path = os.path.join(human_path, args.sequence_name)
    if not os.path.exists(sequence_path):
        print(f"Error: Sequence {args.sequence_name} not found in {human_path}")
        return
    
    # Check if human file exists
    human_file = 'palm_fixed.npz' if args.use_fixed else 'human.npz'
    human_file_path = os.path.join(sequence_path, human_file)
    if not os.path.exists(human_file_path):
        print(f"Error: {human_file} not found in {sequence_path}")
        if args.use_fixed:
            print("Try running without --use_fixed to use the original human.npz file")
        return
    
    print(f"Visualizing sequence: {args.sequence_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Using file: {human_file}")
    
    # Load human data based on dataset type
    if dataset_name.upper() == 'BEHAVE' or dataset_name.upper() == 'BEHAVE_CORRECT':
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(
            args.sequence_name, human_path, 'smplh', 10, use_fixed=args.use_fixed)
    elif dataset_name.upper() == 'NEURALDOME' or dataset_name.upper() == 'IMHD':
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(
            args.sequence_name, human_path, 'smplh', 16, use_fixed=args.use_fixed)
    elif dataset_name.upper() == 'CHAIRS':
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(
            args.sequence_name, human_path, 'smplx', 10, use_fixed=args.use_fixed)
    elif dataset_name.upper() == 'INTERCAP' or dataset_name.upper() == 'INTERCAP_CORRECT':
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(
            args.sequence_name, human_path, 'smplx', 10, True, use_fixed=args.use_fixed)
    elif dataset_name.upper() == 'OMOMO' or dataset_name.upper() == 'OMOMO_CORRECT':
        verts, joints, faces, poses, betas, trans, gender = visualize_smpl(
            args.sequence_name, human_path, 'smplx', 16, use_fixed=args.use_fixed)
    else:
        print(f"Error: Unsupported dataset type: {dataset_name}")
        return
    
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
    if not os.path.exists(obj_mesh_path):
        # Try with _pca_canonical suffix
        obj_mesh_path = os.path.join(object_path, obj_name, obj_name+'_pca_canonical.obj')
        if not os.path.exists(obj_mesh_path):
            print(f"Error: Object mesh not found for {obj_name}")
            return
    
    OBJ_MESH = trimesh.load(obj_mesh_path)
    print(f"Object name: {obj_name}")
    
    ov = np.array(OBJ_MESH.vertices).astype(np.float32)
    object_faces = OBJ_MESH.faces.astype(np.int32)
    
    # Transform object vertices
    ov = torch.from_numpy(ov).float().to(device)
    rot = torch.tensor(angle_matrix).float().to(device)
    obj_trans_tensor = torch.tensor(obj_trans).float().to(device)
    object_verts = torch.einsum('ni,tij->tnj', ov, rot.permute(0,2,1)) + obj_trans_tensor.unsqueeze(1)
    
    # Prepare human mesh data
    human_verts = torch.from_numpy(verts.float().cpu().numpy()).to(device).float()
    human_face = torch.from_numpy(faces[0].cpu().numpy().astype(np.int32)).to(device)
    
    print(f"Sequence loaded: {poses.shape[0]} frames")
    print(f"Human vertices: {human_verts.shape}")
    print(f"Object vertices: {object_verts.shape}")
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Generate output filename
    suffix = "_fixed" if args.use_fixed else ""
    output_filename = f"{args.sequence_name}{suffix}.mp4"
    output_path = os.path.join(args.output_path, output_filename)
    
    # Render visualization
    print(f"Rendering to: {output_path}")
    visualize_body_obj(
        human_verts.float().detach().cpu().numpy(),
        faces[0].detach().cpu().numpy().astype(np.int32),
        object_verts.detach().cpu().numpy(),
        object_faces,
        save_path=output_path,
        multi_angle=args.multi_angle,
        show_frame=True
    )
    
    print(f"Visualization complete: {output_path}")

if __name__ == "__main__":
    main() 