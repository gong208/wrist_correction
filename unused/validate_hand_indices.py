import os
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
import argparse
import smplx
from human_body_prior.body_model.body_model import BodyModel
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import math
from render.mesh_utils import MeshViewer
from render.utils import colors
import imageio
import pyrender
from PIL import Image
from PIL import ImageDraw 
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
smplx16_model_female = BodyModel(bm_fname=surface_model_male_fname,
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


def c2rgba(c):
    if len(c) == 3:
        c.append(1)
    c = [c_i/255 for c_i in c[:3]]

    return c


def visualize_body_obj(body_verts, body_face, obj_verts, obj_face, save_path,
                       multi_angle=False, h=1024, w=1024, bg_color='white', show_frame=True,
                       rhand_idx=None, lhand_idx=None):


    im_height = h
    im_width = w
    seqlen = len(body_verts)

    # Convert tensors to numpy if needed
    mesh_rec = body_verts.detach().cpu().numpy() if torch.is_tensor(body_verts) else np.asarray(body_verts)
    obj_mesh_rec = obj_verts.detach().cpu().numpy() if torch.is_tensor(obj_verts) else np.asarray(obj_verts)

    # Normalize hand indices to numpy int arrays (and ensure unique & in-range later)
    def _to_np_idx(idx, V):
        if idx is None:
            return None
        if torch.is_tensor(idx):
            idx = idx.detach().cpu().numpy()
        idx = np.asarray(idx).astype(np.int64).ravel()
        # clip to valid range & dedup to be safe
        idx = np.unique(idx[(idx >= 0) & (idx < V)])
        return idx

    V = mesh_rec.shape[1]
    r_idx = _to_np_idx(rhand_idx, V)
    l_idx = _to_np_idx(lhand_idx, V)

    # Compute bbox to scale marker size (kept from your original code)
    minx, _, miny = mesh_rec.min(axis=(0, 1))
    maxx, _, maxy = mesh_rec.max(axis=(0, 1))
    bbox_size = max(maxx - minx, maxy - miny)
    marker_radius = bbox_size * 0.01  # not used below, but preserved

    minsxy = (minx, maxx, miny, maxy)

    mesh_rec = mesh_rec.copy()
    obj_mesh_rec = obj_mesh_rec.copy()
    # center X and Z (your original alignment)
    mesh_rec[:, :, 0] -= (minx + maxx) / 2
    obj_mesh_rec[:, :, 0] -= (minx + maxx) / 2
    mesh_rec[:, :, 2] -= (miny + maxy) / 2
    obj_mesh_rec[:, :, 2] -= (miny + maxy) / 2

    mv = MeshViewer(width=im_width, height=im_height,
                    add_ground_plane=True, plane_mins=minsxy,
                    use_offscreen=True,
                    bg_color=bg_color)
    mv.render_wireframe = False

    # Allocate video buffer
    if multi_angle:
        video = np.zeros([seqlen, 1 * im_width, 2 * im_height, 3], dtype=np.uint8)
    else:
        video = np.zeros([seqlen, im_width, im_height, 3], dtype=np.uint8)

    for i in range(seqlen):
        # Object mesh color: pink @ 0.5 alpha
        rgba_obj = np.concatenate([c2rgba(colors['pink'])[:3], [0.5]], axis=0)
        obj_mesh_color = np.tile(rgba_obj, (obj_mesh_rec.shape[1], 1))
        obj_m_rec = trimesh.Trimesh(vertices=obj_mesh_rec[i],
                                    faces=obj_face,
                                    vertex_colors=obj_mesh_color)

        # Body mesh base color: yellow_pale @ 1.0 alpha
        rgba_body = np.concatenate([c2rgba(colors['yellow_pale'])[:3], [1.0]], axis=0)

        # Hand highlight colors
        rgba_left = np.concatenate([c2rgba(colors['green'])[:3], [1.0]], axis=0)     # left hand: blue
        rgba_right = np.concatenate([c2rgba(colors['red'])[:3], [1.0]], axis=0)  # right hand: yellow

        # Per-vertex colors (start from base)
        mesh_color = np.tile(rgba_body, (mesh_rec.shape[1], 1))

        # Paint hands if indices provided
        if l_idx is not None and l_idx.size > 0:
            mesh_color[l_idx] = rgba_left
        if r_idx is not None and r_idx.size > 0:
            mesh_color[r_idx] = rgba_right

        m_rec = trimesh.Trimesh(vertices=mesh_rec[i],
                                faces=body_face,
                                vertex_colors=mesh_color)

        # Render
        all_meshes = [obj_m_rec, m_rec]
        mv.set_meshes(all_meshes, group_name='static')
        frame0 = mv.render()

        if multi_angle:
            # rotate 270° around Y for the second view
            Ry = trimesh.transformations.rotation_matrix(np.radians(270), [0, 1, 0])
            obj_m_rot = obj_m_rec.copy()
            body_m_rot = m_rec.copy()
            obj_m_rot.apply_transform(Ry)
            body_m_rot.apply_transform(Ry)
            mv.set_meshes([obj_m_rot, body_m_rot], group_name='static')
            frame1 = mv.render()
            video_i = np.concatenate((frame0, frame1), axis=1)
        else:
            video_i = frame0

        video[i] = video_i.astype(np.uint8)

    # Write video
    video_writer = imageio.get_writer(save_path, fps=30)
    for i in range(seqlen):
        frame = video[i]
        pil_image = Image.fromarray(frame)
        if show_frame:
            draw = ImageDraw.Draw(pil_image)
            text = f"{i}".zfill(4)
            draw.text((5, 5), text, fill='red')
        video_writer.append_data(np.array(pil_image).astype(np.uint8))
    video_writer.close()
    del mv



def load_hand_indices():
    """Load all hand indices for validation"""
    print("Loading hand indices...")
    
    # Basic hand indices
    rhand_idx = np.load('./exp/rhand_smplx_ids.npy')
    lhand_idx = np.load('./exp/lhand_smplx_ids.npy')
    
    print(f"Right hand vertices: {len(rhand_idx)}")
    print(f"Left hand vertices: {len(lhand_idx)}")
    
    # Check for overlaps
    overlap = np.intersect1d(rhand_idx, lhand_idx)
    if len(overlap) > 0:
        print(f"WARNING: {len(overlap)} vertices overlap between left and right hands!")
        print(f"Overlapping indices: {overlap[:10]}...")  # Show first 10
    
    # Load detailed finger indices
    RHAND_INDEXES_DETAILED = []
    for i in range(5):
        for j in range(3):
            idx = np.load(f'./exp/index_778/hand_778_{i}_{j}.npy')
            RHAND_INDEXES_DETAILED.append(idx)
            print(f"Right hand finger {i} joint {j}: {len(idx)} vertices")
    RHAND_INDEXES_DETAILED.append(np.load(f'./exp/index_778/hand_778_5.npy'))
    print(f"Right hand palm: {len(RHAND_INDEXES_DETAILED[-1])} vertices")
    
    LHAND_INDEXES_DETAILED = []
    for i in range(5):
        for j in range(3):
            idx = np.load(f'./exp/index_778/lhand_778_{i}_{j}.npy')
            LHAND_INDEXES_DETAILED.append(idx)
            print(f"Left hand finger {i} joint {j}: {len(idx)} vertices")
    LHAND_INDEXES_DETAILED.append(np.load(f'./exp/index_778/lhand_778_5.npy'))
    print(f"Left hand palm: {len(LHAND_INDEXES_DETAILED[-1])} vertices")
    
    return rhand_idx, lhand_idx, RHAND_INDEXES_DETAILED, LHAND_INDEXES_DETAILED

def create_highlighted_mesh(verts, faces, hand_indices, highlight_color=[1.0, 0.0, 0.0]):
    """Create a mesh with highlighted hand vertices"""
    # Create a copy of vertices for coloring
    colored_verts = verts.copy()
    
    # Create vertex colors (default white)
    vertex_colors = np.ones((verts.shape[0], 3)) * 0.8  # Light gray default
    
    # Highlight hand vertices
    vertex_colors[hand_indices] = highlight_color
    
    # Create trimesh with colors
    mesh = trimesh.Trimesh(vertices=colored_verts, faces=faces, vertex_colors=vertex_colors)
    return mesh

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
    parser = argparse.ArgumentParser(description="Validate hand indices by visualizing them on human mesh")
    parser.add_argument("--dataset_path", required=True, help="Path to the root of the dataset")
    parser.add_argument("--sequence_name", required=True, help="Name of the sequence to visualize")
    parser.add_argument("--use_fixed", action="store_true", help="Use joint_fixed.npz instead of human.npz")
    parser.add_argument("--output_path", default="./hand_validation_output", help="Output directory for rendered video")
    parser.add_argument("--highlight_fingers", action="store_true", help="Highlight individual finger parts")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame for sequence (default: 0)")
    parser.add_argument("--end_frame", type=int, default=-1, help="End frame for sequence (default: -1 for all frames)")
    
    args = parser.parse_args()
    
    # Load hand indices first
    rhand_idx, lhand_idx, RHAND_INDEXES_DETAILED, LHAND_INDEXES_DETAILED = load_hand_indices()
    
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
    human_file = 'joint_fixed.npz' if args.use_fixed else 'human.npz'
    human_file_path = os.path.join(sequence_path, human_file)
    if not os.path.exists(human_file_path):
        print(f"Error: {human_file} not found in {sequence_path}")
        if args.use_fixed:
            print("Try running without --use_fixed to use the original human.npz file")
        return
    
    print(f"Validating hand indices for sequence: {args.sequence_name}")
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
        # Determine frame range
    start_frame = args.start_frame
    end_frame = args.end_frame if args.end_frame != -1 else verts.shape[0]
    print(f"Visualizing frames {start_frame} to {end_frame} (total: {end_frame - start_frame} frames)")
    # Load object data
    try:
        with np.load(os.path.join(human_path, args.sequence_name, 'object.npz'), allow_pickle=True) as f:
            obj_angles, obj_trans, obj_name = f['angles'], f['trans'], str(f['name'])
    except FileNotFoundError:
        print(f"Warning: object.npz not found in {sequence_path}, skipping object visualization")
        obj_angles, obj_trans, obj_name = None, None, None
    
    # Validate hand indices
    print("\n=== HAND INDEX VALIDATION ===")
    print(f"Total vertices per frame: {verts.shape[1]}")
    print(f"Right hand indices range: {rhand_idx.min()} - {rhand_idx.max()}")
    print(f"Left hand indices range: {lhand_idx.min()} - {lhand_idx.max()}")
    
    # Check if indices are within bounds
    if rhand_idx.max() >= verts.shape[1]:
        print(f"ERROR: Right hand indices exceed vertex count! Max index: {rhand_idx.max()}")
    else:
        print(f"✓ Right hand indices are valid")
    
    if lhand_idx.max() >= verts.shape[1]:
        print(f"ERROR: Left hand indices exceed vertex count! Max index: {lhand_idx.max()}")
    else:
        print(f"✓ Left hand indices are valid")
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Create visualization video
    if obj_angles is not None:
        angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
        
        # Load object mesh
        obj_mesh_path = os.path.join(object_path, obj_name, obj_name+'.obj')
        if not os.path.exists(obj_mesh_path):
            # Try with _pca_canonical suffix
            obj_mesh_path = os.path.join(object_path, obj_name, obj_name+'_pca_canonical.obj')
            if not os.path.exists(obj_mesh_path):
                print(f"Warning: Object mesh not found for {obj_name}, skipping object visualization")
                obj_angles, obj_trans, obj_name = None, None, None
        
        if obj_angles is not None:
            OBJ_MESH = trimesh.load(obj_mesh_path)
            print(f"Object name: {obj_name}")
            
            ov = np.array(OBJ_MESH.vertices).astype(np.float32)
            object_faces = OBJ_MESH.faces.astype(np.int32)
            
            # Transform object vertices
            ov = torch.from_numpy(ov).float().to(device)
            rot = torch.tensor(angle_matrix).float().to(device)
            obj_trans_tensor = torch.tensor(obj_trans).float().to(device)
            object_verts = torch.einsum('ni,tij->tnj', ov, rot.permute(0,2,1)) + obj_trans_tensor.unsqueeze(1)
            human_verts = torch.from_numpy(verts.float().cpu().numpy()).to(device).float()
            human_face = torch.from_numpy(faces[0].cpu().numpy().astype(np.int32)).to(device)
            
            # Generate output filename for video
            suffix = "_fixed" if args.use_fixed else ""
            video_filename = f"{args.sequence_name}{suffix}_hand_validation.mp4"
            video_path = os.path.join(args.output_path, video_filename)
            
            # Render visualization
            print(f"Rendering validation video to: {video_path}")
            visualize_body_obj(
                human_verts.float().detach().cpu().numpy(),
                human_face.detach().cpu().numpy().astype(np.int32),
                object_verts.detach().cpu().numpy(),
                object_faces,
                save_path=video_path,
                rhand_idx=rhand_idx,
                lhand_idx=lhand_idx
            )
            print(f"Validation video complete: {video_path}")
    
    print(f"\n=== VALIDATION COMPLETE ===")
    if obj_angles is not None:
        print(f"Check the validation video: {video_path}")
    print(f"Right hand: {len(rhand_idx)} vertices (red)")
    print(f"Left hand: {len(lhand_idx)} vertices (green)")
    if args.highlight_fingers:
        print("Individual finger parts are highlighted with different colors:")
        finger_names = ["thumb", "index", "middle", "ring", "pinky", "palm"]
        for i, name in enumerate(finger_names):
            print(f"  {name}: {len(RHAND_INDEXES_DETAILED[i])} vertices (right), {len(LHAND_INDEXES_DETAILED[i])} vertices (left)")

if __name__ == "__main__":
    main()
