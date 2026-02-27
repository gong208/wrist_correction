import os
import numpy as np
import torch
#from data.dataset_smpl import Dataset, MODEL_PATH, OBJECT_PATH
from utils import markerset_ssm67_smplh, markerset_wfinger, vertex_normals

#from data.utils import SIMPLIFIED_MESH
#from tools import point2point_signed
from loss import point2point_signed
import smplx
from prior import *
#from libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer
#from psbody.mesh import Mesh
import trimesh
from smplx import SMPLXLayer
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import matrix_to_euler_angles,axis_angle_to_matrix, matrix_to_axis_angle,rotation_6d_to_matrix,matrix_to_rotation_6d
from torch.autograd import Variable
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

DEVICE_NUMBER=0

device = torch.device(f"cuda:{DEVICE_NUMBER}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
#device=torch.device('cpu')
SMPLX_PATH='models/smplx'
surface_model_male_fname = os.path.join(SMPLX_PATH,"SMPLX_MALE.npz")
surface_model_female_fname = os.path.join(SMPLX_PATH,"SMPLX_FEMALE.npz")
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
sbj_m_all={'male':sbj_m_male,'female':sbj_m_female}

hand_prior=HandPrior(prior_path='./assets',device=device)
import open3d as o3d
hand_distance_init=0
gt_rhand=0
whether_all_minus=np.zeros(15)
BIGGEST_VALUE=torch.zeros((15,3)).float().to(device)-99999
SMALLEST_VALUE=torch.zeros((15,3)).float().to(device)+99999
biggest_name=np.array(['B']*45).reshape(15,3)
smallest_name=np.array(['B']*45).reshape(15,3)

## Rhand id of smplx
rhand_idx=np.load('./exp/rhand_smplx_ids.npy')


WHETHER_TOUCH_LEFT=0
WHETHER_TOUCH_RIGHT=0
WHETHER_MIDTOUCH_LEFT=0
WHETHER_MIDTOUCH_RIGHT=0

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


SIMPLIFIED_MESH = {
    "backpack":"backpack/backpack_f1000.ply",
    'basketball':"basketball/basketball_f1000.ply",
    'boxlarge':"boxlarge/boxlarge_f1000.ply",
    'boxtiny':"boxtiny/boxtiny_f1000.ply",
    'boxlong':"boxlong/boxlong_f1000.ply",
    'boxsmall':"boxsmall/boxsmall_f1000.ply",
    'boxmedium':"boxmedium/boxmedium_f1000.ply",
    'chairblack': "chairblack/chairblack_f2500.ply",
    'chairwood': "chairwood/chairwood_f2500.ply",
    'monitor': "monitor/monitor_closed_f1000.ply",
    'keyboard':"keyboard/keyboard_f1000.ply",
    'plasticcontainer':"plasticcontainer/plasticcontainer_f1000.ply",
    'stool':"stool/stool_f1000.ply",
    'tablesquare':"tablesquare/tablesquare_f2000.ply",
    'toolbox':"toolbox/toolbox_f1000.ply",
    "suitcase":"suitcase/suitcase_f1000.ply",
    'tablesmall':"tablesmall/tablesmall_f1000.ply",
    'yogamat': "yogamat/yogamat_f1000.ply",
    'yogaball':"yogaball/yogaball_f1000.ply",
    'trashbin':"trashbin/trashbin_f1000.ply",
}

## SMOOTHING LOSS
# comment: 这个loss是用来做什么的？为什么要mask和1-mask
# answer: 时间平滑约束，不optimize的部分用detach版本不参与反向传播
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

def obj_forward(raw_points, obj_rot_6d, obj_transl):
        # N_points, 3
        # B, 6
        # B, 3
    B = obj_rot_6d.shape[0]
    obj_rot = rotation_6d_to_matrix(obj_rot_6d[:, :]).permute(0, 2, 1)  # B,3,3 don't forget to transpose
    obj_points_pred = torch.matmul(raw_points.unsqueeze(0)[:, :, :3], obj_rot) + obj_transl.unsqueeze(1)
    
    return obj_points_pred

def numpy_to_obj_trimesh(np_array, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_array)
    o3d.io.write_point_cloud(filename, pcd)
full_mesh = {
    "backpack":"backpack/backpack.obj",
    'basketball':"basketball/basketball.obj",
    'boxlarge':"boxlarge/boxlarge.obj",
    'boxtiny':"boxtiny/boxtiny.obj",
    'boxlong':"boxlong/boxlong.obj",
    'boxsmall':"boxsmall/boxsmall.obj",
    'boxmedium':"boxmedium/boxmedium.obj",
    'chairblack': "chairblack/chairblack.obj",
    'chairwood': "chairwood/chairwood.obj",
    'monitor': "monitor/monitor.obj",
    'keyboard':"keyboard/keyboard.obj",
    'plasticcontainer':"plasticcontainer/plasticcontainer.obj",
    'stool':"stool/stool.obj",
    'tablesquare':"tablesquare/tablesquare.obj",
    'toolbox':"toolbox/toolbox.obj",
    "suitcase":"suitcase/suitcase.obj",
    'tablesmall':"tablesmall/tablesmall.obj",
    'yogamat': "yogamat/yogamat.obj",
    'yogaball':"yogaball/yogaball.obj",
    'trashbin':"trashbin/trashbin.obj",
}
JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
    "right_eye_brow1",
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",
    "left_eye_brow5",
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1",
    "nose1",
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1",
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6",
    "left_eye4",
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",
    "left_eye5",
    "right_mouth_1",
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",  # 59 in OpenPose output
    "left_mouth_4",  # 58 in OpenPose output
    "mouth_bottom",
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom",
    "right_lip_3",
    # Face contour
    "right_contour_1",
    "right_contour_2",
    "right_contour_3",
    "right_contour_4",
    "right_contour_5",
    "right_contour_6",
    "right_contour_7",
    "right_contour_8",
    "contour_middle",
    "left_contour_8",
    "left_contour_7",
    "left_contour_6",
    "left_contour_5",
    "left_contour_4",
    "left_contour_3",
    "left_contour_2",
    "left_contour_1",
]

## HARDCODED, restrict the local rotation to be within a rang of motion(ROM)
def restrict_angles(theta,theta_max,theta_min,mode,flag,alpha=0.01):
    MASK_MAX=(theta-theta_max)>0
    MASK_MIN=(theta-theta_min)<0
    T=theta.shape[0]
    loss_max=torch.sum(MASK_MAX.detach().float()*(theta-theta_max)**2)
    loss_min=torch.sum(MASK_MIN.detach().float()*(theta_min-theta)**2)
    # loss_max=torch.sum(MASK_MAX.detach()*torch.exp((-theta_max+theta)))
    # loss_min=torch.sum(MASK_MIN.detach()*torch.exp((-theta+theta_min)))

    # if loss_max>0:
    #     print('MAX',flag)
    # elif loss_min>0:
    #     print('MIN',flag)


    #print(loss_max+loss_min,'LOSS')
    # loss_max=1* torch.sum(torch.where((theta-theta_max)>0,torch.abs(theta-theta_max),torch.zeros_like(theta).float().to(theta.device)))#to(theta.device)
    # loss_min=1* torch.sum(torch.where((theta-theta_min)<0,torch.abs(theta-theta_min),torch.zeros_like(theta).float().to(theta.device)))#.to(theta.device))
    if mode==0:
        return 10*loss_max**2+10*loss_min**2+torch.sum(theta**2)*alpha
    else:
        return 1*loss_max+loss_min*1


def optimize1(index, name, visualize=False, fname='', use_fixed_poses=False):
    print(f"Starting optimize1 for {fname} with use_fixed_poses={use_fixed_poses}")
    
    human_npz_path=os.path.join(name,"palm_fixed.npz")
    object_npz_path=os.path.join(name,"object.npz")
    with np.load(human_npz_path, allow_pickle=True) as f:
            poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])
    # Check if we should use fixed poses from wrist optimization
    if use_fixed_poses:
        fixed_pose_path = f"./wrist_optimization_results/{fname}_wrist_optimized.npy"
        if os.path.exists(fixed_pose_path):
            print(f"Loading fixed poses from: {fixed_pose_path}")
            poses = np.load(fixed_pose_path)
            print(f"Fixed poses shape: {poses.shape}")

        
    
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

    #visualize_body_obj(-verts_sbj.detach().cpu().numpy(), sbj_m.f.detach().cpu().numpy()[...,::-1], -verts_obj.detach().cpu().numpy(), obj_info['faces'][...,::-1], save_path='./aex',save_gif=True,save_both=False )

    rhand_idx=np.load('./exp/rhand_smplx_ids.npy')
    lhand_idx_path='./exp/lhand_smplx_ids.npy'
    lhand_idx=np.load(lhand_idx_path)
    # lhand_faces_path='./exp/lhand_faces.npy'
    # lhand_faces=np.load(lhand_faces_path)
    # rhand_faces_path='./exp/rhand_faces.npy'
    # rhand_faces=np.load(rhand_faces_path)
    # export_file = F"./zoptimization_examples2/render/"
    # os.makedirs(export_file, exist_ok=True)
    # # mask_video_paths = [join(seq_save_path, f'mask_k{x}.mp4') for x in reader.seq_info.kids]
    # #rend_video_path = os.path.join(export_file, 'smplx_'+name.split('/')[-1].split('.')[0]+'_gt')
    # rend_video_path_optim = os.path.join(export_file, 'smplx_'+name.split('/')[-1]+'_stat_mean')
    # verts_sbj_mean=verts_sbj_mean.detach().cpu().numpy()
    # m1 = visualize_body_obj_hand(-verts_sbj_mean[:,rhand_idx],rhand_faces[...,:] ,-verts_sbj_mean[:,lhand_idx],lhand_faces[...,:],-verts_obj[:].detach().cpu().numpy(), obj_info['faces'][...,::-1], save_path=rend_video_path_optim,save_gif=True,save_both=False)
    #print()
    #print(verts_sbj.shape,'KKK')

    # visualize_body_obj(-verts_sbj[:20,rhand_idx].detach().cpu().numpy(), rhand_faces[...,::-1], -verts_obj[:20].detach().cpu().numpy(), obj_info['faces'][...,::-1], save_path='./aex_rhand',save_gif=True,save_both=False )
    # visualize_body_obj(-verts_sbj[:20,lhand_idx].detach().cpu().numpy(), lhand_faces[...,::-1], -verts_obj[:20].detach().cpu().numpy(), obj_info['faces'][...,::-1], save_path='./aex_lhand',save_gif=True,save_both=False )
    # print()


    h2o_signed_mean=point2point_signed(verts_sbj_mean[:,rhand_idx],verts_obj)[1]
    #print(h2o_signed_mean.shape,'KJKJKJKJ')
    #global WHETHER_TOUCH_LEFT,WHETHER_TOUCH_RIGHT
    #S=torch.sum((h2o_signed_mean[:,:]<0.02),dim=-1)

    #WHETHER_TOUCH_RIGHT=(S>0).float().detach().reshape(-1,1)
    
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


            #print(loss_touch,'JKJK')
            #loss_touch=0*torch.sum((verts[:,rhand_indexes]-target_rhand_locations)**2)+0.05*torch.sum((jtr[:,[41,42,44,45,47,48,50,51,53,54]]-jtr[:,[40,41,43,44,46,47,49,50,52,53]])**2)#+ 1*torch.sum(h2o_signed**2)
            
            
            # search_tree = BVH(max_collisions=8)

            # pen_distance = collisions_loss.DistanceFieldPenetrationLoss(
            #                sigma=0.5, point2plane=False, vectorized=True, penalize_outside=True)

            # part_segm_fn='/var/lib/docker/data/users/wangziyin/render/Human_Object_Interaction_Prediction-main/deps/smpl_models/smplx_parts_segm.pkl'
            # with open(part_segm_fn, 'rb') as faces_parents_file:
            #     face_segm_data = pickle.load(faces_parents_file,  encoding='latin1')
            # faces_segm = face_segm_data['segm']
            # faces_parents = face_segm_data['parents']
            # # Create the module used to filter invalid collision pairs
            # filter_faces = FilterFaces(
            #     faces_segm=faces_segm, faces_parents=faces_parents,
            #     ign_part_pairs=None).to(hand_pose_rec.device)
            # # print(type(verts))
            # # print(type(sbj_m.faces_tensor.view(-1)))
            # # print(verts.device)
            # # print(sbj_m.faces_tensor.device)
            # triangles = torch.index_select(
            # verts, 1,
            # sbj_m.faces_tensor.view(-1)).view(bs, -1, 3, 3).float().to(hand_pose_rec.device)
            # # print(triangles.shape)
            # # print(sbj_m.faces_tensor.shape)

            # with torch.no_grad():
            #     collision_idxs = search_tree(triangles)

        # Remove unwanted collisions
            # if filter_faces is not None:
            #     #print('NOT NONE')
            #     collision_idxs = filter_faces(collision_idxs)

            # if collision_idxs.ge(0).sum().item() > 0:
            #     collision_loss = torch.sum(0.1 * pen_distance(triangles, collision_idxs))
            # else:
            #     collision_loss=torch.tensor(0).to(hand_pose_rec.device)
            #print(collision_loss,'CL')
    
    #pose_preserve_loss = (pose_preserve_weight ** 2) * ((body_pose - preserve_pose) ** 2).sum(dim=-1)
            
                     
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
                    
                    #print(SUMM,epoch)
                    # if SUMM<2:
                    #     return None,None,None,1
                        
                    
                    sd_pen=(sd_i*WHETHER_TOUCH_L.view(-1,1))[MASK_TIME]
                    
                    
                    zeros_s2o, ones_s2o = torch.zeros_like(sd_pen).float().to(device), torch.ones_like(sd_pen).float().to(device)
                
                
                    calc_dist = sd_pen - thresh
                    #mask_pen=((sd_pen-thresh)<0).float()
                    mask_pen = ((sd_pen-thresh)<0).float()*(torch.abs(sd_pen)<0.03).float()

                    num_pen=torch.sum(mask_pen,dim=-1).reshape(-1,1)
                    #print(calc_dist.shape,mask_pen.shape,'QQQ')
                    
                    loss_dist_o += torch.sum(torch.abs(calc_dist)*mask_pen/(num_pen+1e-8))*2 #num_pen+1e-8)

                    #loss_dist_o += 2*torch.mean(abs(torch.where(sd_pen<thresh, sd_pen-thresh, zeros_s2o))) #/ num_verts
                          # (bs,)  -- averaged across (bs, N_sbj)
                OP_MASK_TIME = ~MASK_TIME
                #print(OP_MASK_TIME,HAND_SMALL_INDEXES[i].shape)
                
                # if torch.sum(OP_MASK_TIME)>0 or 1 : 
                if 1:
                    num_verts_small=HAND_SMALL_INDEXES[i].shape[0]
                    
                    #sbj2obj_mask_first=sbj2obj*WHETHER_TOUCH
                    
                    sbj2obj_mask_first = sbj2obj * WHETHER_TOUCH_L
                    sd_closer = sbj2obj_mask_first[OP_MASK_TIME][:,HAND_SMALL_INDEXES[i]]#/num_verts_small
                    


                    #sd_closer=sbj2obj_mask_first[:,HAND_SMALL_INDEXES[i]]/num_verts_small
                    ratio = min(epoch/350,1)
                    #loss_touch+=torch.mean(torch.abs(sbj2obj[OP_MASK_TIME][:,EXTRA_IN_778_RH])*WHETHER_TOUCH)*(1-ratio)*2

                    loss_touch += 5 * torch.sum(torch.abs(sd_closer)**2)
                    ## try
                    # sd_closer2=sbj2obj_mask_first[MASK_TIME][:,HAND_SMALL_INDEXES[i]]
                    # #sd_closer=sbj2obj_mask_first[:,HAND_SMALL_INDEXES[i]]
                    # ratio=min(epoch/350,1)
                    #loss_touch+=20*torch.sum(torch.abs(sd_closer))*ratio
                    #loss_touch+=5*torch.sum(torch.abs(sd_closer2))*ratio

                    
                    ##
                #     sd_closer=sd_i[OP_MASK_TIME]
                    
                #     zeros_s2o, ones_s2o = torch.zeros_like(sd_closer).to(device), torch.ones_like(sd_).to(device)
                
                
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
                    
                    #print(SUMM,epoch)
                    # if SUMM<2:
                    #     return None,None,None,1
                        
                    
                    sd_pen=sd_i[MASK_TIME]
                    
                    zeros_s2o, ones_s2o = torch.zeros_like(sd_pen).float().to(device), torch.ones_like(sd_pen).float().to(device)
                
                #loss_s2o_out = torch.sum(torch.log(torch.where(sbj2obj>thresh, sbj2obj+ones_s2o, ones_s2o)), 1) / num_verts  # (bs,)  -- averaged across (bs, N_sbj)
                # 4.1. Special case for sbj2obj negative values - check whether to do connected components or not.
                    calc_dist=sd_pen-thresh
                    #mask_pen=((sd_pen-thresh)<0).float()
                    mask_pen=((sd_pen-thresh)<0).float()*(torch.abs(sd_pen)<0.1).float()
                    num_pen=torch.sum(mask_pen,dim=-1).reshape(-1,1)
                    
                    loss_dist_o+=torch.sum(torch.abs(calc_dist**2)*mask_pen)*2

                    #loss_dist_o += 2*torch.mean(abs(torch.where(sd_pen<thresh, sd_pen-thresh, zeros_s2o))) #/ num_verts
                          # (bs,)  -- averaged across (bs, N_sbj)
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
                    # sd_closer2=sbj2obj_mask_first[MASK_TIME][:,HAND_SMALL_INDEXES[i]]
                    # #sd_closer=sbj2obj_mask_first[:,HAND_SMALL_INDEXES[i]]
                    # ratio=min(epoch/350,1)
                    #loss_touch+=20*torch.sum(torch.abs(sd_closer))*ratio
                    #loss_touch+=5*torch.sum(torch.abs(sd_closer2))*ratio

                    
                    ##
                #     sd_closer=sd_i[OP_MASK_TIME]
                    
                #     zeros_s2o, ones_s2o = torch.zeros_like(sd_closer).to(device), torch.ones_like(sd_).to(device)
                
            contact_loop_time = time.time() - contact_loop_start
            
            #print(hand_pose_rec.shape,'QQQ')
            euler_angles = hand_pose_rec[:,:,[2,1,0]]
            #euler_angles=matrix_to_euler_angles(axis_angle_to_matrix(hand_pose_rec).float(),convention='ZYX').float()
            if not left_or_right:
                #torch.tensor([-1.0,-1.0,1.0]).to(device).reshape(1,1,3)
                euler_angles=euler_angles*(torch.tensor([-1.0,-1.0,1.0]).to(device).reshape(1,1,3))

            #euler_angles=hand_pose_rec[:,:,[2,1,0]]
            #aa=matrix_to_axis_angle(rotation_6d_to_matrix(hand_pose_rec).float()).float()
            # print(aa[-1])
            # print(euler_angles[-1])

            # thumb_pinky_out_notmid=euler_angles[:,[0,2,3,5,9,11]]
            # thumb_pinky_out_mid=euler_angles[:,[1,4,10]]


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

                # loss_hand_pose_v_reg = 0.25 * torch.sum((((hand_pose_rec[1:-1] - hand_pose_rec[:-2]) - (hand_pose_rec[2:] - hand_pose_rec[1:-1])) ** 2)*WHETHER_TOUCH[2:].view(-1,1,1)) + \
                #                         0.5 * torch.sum(((hand_pose_rec[1:] - hand_pose_rec[:-1]) ** 2)*WHETHER_TOUCH[1:].view(-1,1,1))
                
                # loss_hand_pose_v_reg+= 0.25 * torch.sum((((hand_verts[1:-1] - hand_verts[:-2]) - (hand_verts[2:] - hand_verts[1:-1])) ** 2)*WHETHER_TOUCH[2:].view(-1,1,1)) + \
                #                         0.5 * torch.sum(((hand_verts[1:] - hand_verts[:-1]) ** 2)*WHETHER_TOUCH[1:].view(-1,1,1))
                # loss_hand_pose_v_reg = 2.5 * torch.sum((((hand_pose_rec[1:-1] - hand_pose_rec[:-2]) - (hand_pose_rec[2:] - hand_pose_rec[1:-1])) ** 2)) + \
                #                          5 * torch.sum(((hand_pose_rec[1:] - hand_pose_rec[:-1]) ** 2))
                # hand_verts=verts[:,hand_idx]
                # loss_hand_pose_v_reg+= 2.5 * torch.sum((((hand_verts[1:-1] - hand_verts[:-2]) - (hand_verts[2:] - hand_verts[1:-1])) ** 2)) + \
                #                         5 * torch.torch.sum(((hand_verts[1:] - hand_verts[:-1]) ** 2).view(-1,1,1))
                # #print(hand_mean_single.shape,WHETHER_MIDTOUCH.shape,hand_pose_rec.shape)
                
                ## Initialization reg
                # comment: 非接触帧手部姿态尽量靠近平均姿态
                loss_hand_pose_v_reg+=0.05*torch.sum((hand_pose_rec-hand_mean_single.view(1,-1,3))**2*(1-WHETHER_TOUCH_L).view(-1,1,1))
                
                
                #loss_hand_pose_v_reg+=0.05*torch.sum((hand_pose_rec)**2*WHETHER_MIDTOUCH.view(-1,1,1))

                #loss_hand_pose_v_reg+=1*torch.sum(((hand_verts[:-1]-hand_verts[1:])-(verts_obj[:-1]-verts_obj[1:]).gather(1,sbj2obj_idx[1:]))**2*WHETHER_TOUCH[1:].view(-1,1,1))

                # if epoch>EPOCH_SMOOTH:
                #     hand_pose_rec_c=torch.zeros_like(hand_pose_rec).float().to(device)
                #     hand_pose_rec_c[~(WHETHER_MID_TOUCH.bool()).view(-1)]=hand_pose_rec[~(WHETHER_MID_TOUCH.bool()).view(-1)].detach()
                #     hand_pose_rec_c[WHETHER_MID_TOUCH.bool().view(-1)]=hand_pose_rec[WHETHER_MID_TOUCH.bool().view(-1)]
                #     hand_pose_rec_c[WHETHER_TOUCH.bool().view(-1)]=hand_pose_rec[WHETHER_TOUCH.bool().view(-1)].detach()
                #     loss_hand_pose_v_reg+=0.25 * torch.sum((((hand_pose_rec_c[1:-1] - hand_pose_rec_c[:-2]) - (hand_pose_rec_c[2:] - hand_pose_rec_c[1:-1])) ** 2)*WHETHER_MID_TOUCH[2:].view(-1,1,1)) + \
                #                         0.5 * torch.sum(((hand_pose_rec_c[1:] - hand_pose_rec_c[:-1]) ** 2)*WHETHER_MID_TOUCH[1:].view(-1,1,1))
                    
                #     hand_verts_c=torch.zeros_like(hand_verts).float().to(device)
                #     hand_verts_c[~(WHETHER_MID_TOUCH.bool()).view(-1)]=hand_verts[~(WHETHER_MID_TOUCH.bool()).view(-1)].detach()
                #     hand_verts_c[WHETHER_MID_TOUCH.bool().view(-1)]=hand_verts[WHETHER_MID_TOUCH.bool().view(-1)]
                #     hand_verts_c[WHETHER_TOUCH.bool().view(-1)]=hand_verts[WHETHER_TOUCH.bool().view(-1)].detach()
                #     loss_hand_pose_v_reg+=0.25 * torch.sum((((hand_verts_c[1:-1] - hand_verts_c[:-2]) - (hand_verts_c[2:] - hand_verts_c[1:-1])) ** 2)*WHETHER_MID_TOUCH[2:].view(-1,1,1)) + \
                #     0.5 * torch.sum(((hand_verts_c[1:] - hand_verts_c[:-1]) ** 2)*WHETHER_MID_TOUCH[1:].view(-1,1,1))
                #     loss_hand_pose_v_reg+=0.2*torch.sum((hand_pose_rec_c-hand_mean_single)**2)
            elif epoch <= 100:
                loss_hand_pose_v_reg=torch.tensor(0).to(device)
           
            ## Relieve Contact sliding
            if epoch > 200:
                CONTACT_MASK=(torch.abs(sbj2obj)<0.01).unsqueeze(2).float().detach()#*((sbj2obj)>=0).unsqueeze(2).float().detach()
                delta_inv_minus=torch.einsum('tij,tjk->tik',(verts[:,hand_idx]-sbj2obj_vector.detach()-obj_trans_tensor.reshape(-1,1,3)),obj_rot_inv_tensor.permute(0,2,1)) # T,3,3
                ## sbj2obj_vector.detach()
                ##
               
                # print(delta_inv_minus.shape,'jkjkjjk')
                # print(CONTACT_MASK.shape)
                # if epoch%2==0:
                #     delta_temporal_inv_minus=(delta_inv_minus[1:]-delta_inv_minus[:-1])*CONTACT_MASK[1:]*WHETHER_TOUCH[1:].view(-1,1,1) ##T,N,3
                # else:
                #     delta_temporal_inv_minus=(delta_inv_minus[1:]-delta_inv_minus[:-1])*CONTACT_MASK[:-1]*WHETHER_TOUCH[:-1].view(-1,1,1) ##T,N,3
                if epoch%2==0:
                    delta_temporal_inv_minus=0.0*((delta_inv_minus[1:]-delta_inv_minus[:-1])*WHETHER_TOUCH[1:].view(-1,1,1))**2+((delta_inv_minus[1:]-delta_inv_minus[:-1])*CONTACT_MASK[1:]*WHETHER_TOUCH[1:].view(-1,1,1))**2 ##T,N,3
                else:
                    delta_temporal_inv_minus=0.0*((delta_inv_minus[1:]-delta_inv_minus[:-1])*WHETHER_TOUCH[:-1].view(-1,1,1))**2+((delta_inv_minus[1:]-delta_inv_minus[:-1])*CONTACT_MASK[1:]*WHETHER_TOUCH[1:].view(-1,1,1))**2 ##T,N,3
                loss_hand_pose_v_reg += 0.01*torch.sum(delta_temporal_inv_minus)
                prior_start = time.time()
                SKT=hand_prior(hand_pose_rec.view(-1,45),left_or_right=left_or_right).squeeze(0)
                prior_time = time.time() - prior_start
                loss_hand_pose_v_reg += 0.1*torch.sum(SKT**2*WHETHER_OPTIMIZE)
                
                #loss_hand_pose_v_reg+=0.05*hand_prior(hand_pose_rec.view(-1,45),left_or_right=left_or_right)

                #print(H.shape,'KJKJKJ')
                
                # xidx_near_expanded = xidx_near.view(frame_times, 778, 1).contiguous().expand(frame_times, 778, D).to(torch.long)
                # x_near = y.gather(1, xidx_near_expanded)
                #sbj2obj_idx
                # delta_smooth_mid_and_touch=
                # hand_idx
            #loss_hand_pose_v_reg=torch.tensor(0.0).to(device)
            #loss_hand_pose_v_reg=torch.tensor(0.0).to(device)
            # loss_obj_v_reg = 1000 * torch.mean(((obj_transl_rec[1:-1] - obj_transl_rec[:-2]) - (obj_transl_rec[2:] - obj_transl_rec[1:-1])) ** 2) + \
            #                  100 * torch.mean(((obj_transl_rec[1:] - obj_transl_rec[:-1])) ** 2) +\
            #                  1000 * torch.mean(((obj_rot_rec[1:-1] - obj_rot_rec[:-2]) - (obj_rot_rec[2:] - obj_rot_rec[1:-1])) ** 2) + \
            #                  100 * torch.mean(((obj_rot_rec[1:] - obj_rot_rec[:-1])) ** 2)
            
            loss_body_v_reg = torch.tensor(0.0).to(device)#100 * torch.mean((((body_rec[1:-1] - body_rec[:-2]) - (body_rec[2:] - body_rec[1:-1])) ** 2).sum(dim=2).sum(dim=1)) + 100 * torch.mean(((body_rec[1:] - body_rec[:-1]) ** 2).sum(dim=2).sum(dim=1)) + 1000 * (loss_left + loss_right)
            #global hand_distance_init
            loss_hand_distance_reg=0*torch.norm((jtr[:,[45,48,51]]-jtr[:,[42,45,48]]),dim=-1)-hand_distance_init
            
            loss_v_reg = 1 * (loss_hand_pose_v_reg + loss_body_v_reg) + loss_rot_reg
            # if epoch<100 or 1:
            #     # if epoch>100:
            #     #     print(loss_hand_pose_v_reg)
            #     loss_v_reg = 1 * (loss_hand_pose_v_reg + loss_body_v_reg ) +loss_rot_reg
            # else:
            #     loss_v_reg = 1 * (loss_obj_v_reg)
            
            #print(loss_v_reg,loss_hand_pose_v_reg,loss_body_v_reg,loss_rot_reg)

            if 1:
                loss = (
                        loss_dist_o+loss_touch+
                        loss_v_reg
                        )
            else:
                loss=loss_dist_o*1+1*loss_v_reg+ 1*(loss_obj_transl_reg + loss_obj_rot_reg)

            loss_dict = {}
            # Only convert to CPU when needed for printing (every epoch)
            if epoch % 1 == 0:
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
            if epoch % 1 == 0:
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
        save_path='./save/omomo2_1500_mano_square_bigparameter_0.01_up'
        os.makedirs(save_path,exist_ok=True)
        save_path=os.path.join(save_path,fn)
        # if epoch==999:
        #     visualize_body_obj(-verts.detach().cpu().numpy(),sbj_m.f.detach().cpu().numpy()[...,::-1],-verts_obj.detach().cpu().numpy(), obj_info['faces'][...,::-1], save_path=save_path,save_gif=True,save_both=False )

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
            
        # loss_all=loss_right
        # collision_loss_all=collision_loss_right
        # loss_dict_all=loss_dict_right
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
    if use_fixed_poses:
        export_file = f"./omomo_optimization_0.22_f_palm/"
        export_file_losses = f"./omomo_optimization_0.22_f_palm_losses/"
    else:
        export_file = f"./omomo_optimization_0.22_f/"
        export_file_losses = f"./omomo_optimization_0.22_f_losses/"
    
    os.makedirs(export_file, exist_ok=True)
    save_path = os.path.join(export_file, fname+'.npy')
    np.save(save_path,hand_pose)

    os.makedirs(export_file_losses, exist_ok=True)
    save_path = os.path.join(export_file_losses, fname+'.pkl')
    with open(save_path,'wb') as f:
        pickle.dump(loss_dict,f)
        
    ###### Visualization

    # Use different visualization directory based on whether fixed poses were used
    if use_fixed_poses:
        vis_file='./omomo_opt_vis_behavemeanstart_fixed/'
    else:
        vis_file='./omomo_opt_vis_behavemeanstart/'
    
    os.makedirs(vis_file, exist_ok=True)
    visualize_body_obj(SBJ_OUTPUT.v.detach().cpu().numpy(), sbj_m.f.detach().cpu().numpy(), verts_obj.detach().cpu().numpy(), obj_info['faces'], save_path=os.path.join(vis_file,f'{fn}.mp4'),show_frame=True)


  
    # if visualize or 0 :
    #     verts = verts.detach().cpu().numpy()
    #     #verts_gt = verts_gt.detach().cpu().numpy()
    #     obj_verts = []
    #     obj_verts_gt = []
    #     # visualize
    #     export_file = F"./zoptimization_examples2/render/"
    #     os.makedirs(export_file, exist_ok=True)
    #     # mask_video_paths = [join(seq_save_path, f'mask_k{x}.mp4') for x in reader.seq_info.kids]
    #     #rend_video_path = os.path.join(export_file, 'smplx_'+name.split('/')[-1].split('.')[0]+'_gt')
    #     rend_video_path_optim = os.path.join(export_file, 'smplx_'+name.split('/')[-1])
    #     rhand_faces=np.load('./exp/rhand_faces.npy')
    #     rhand_idx=np.load('./exp/rhand_smplx_ids.npy')
        
    #     lhand_idx_path='./exp/lhand_smplx_ids.npy'
    #     lhand_faces_path="./exp/lhand_faces.npy"
    #     lhand_faces=np.load(lhand_faces_path)
    #     lhand_idx=np.load(lhand_idx_path)
    #     hand_pose=tmp_smplhparams['hand_pose'].view(-1,90).detach().cpu().numpy()
    #     #m1 = visualize_body_obj_hand(-verts[:,rhand_idx],rhand_faces[...,::-1] ,-verts[:,lhand_idx],lhand_faces[...,::-1],-verts_obj[:].detach().cpu().numpy(), obj_info['faces'][...,::-1], save_path=rend_video_path_optim,save_gif=True,save_both=False)
    #     INDEXX=-1
        

    return tmp_smplhparams, tmp_objparams

def parse_args():
                
    parser = argparse.ArgumentParser()        
   
    parser.add_argument('--number',type=int)
    parser.add_argument('--dataset',type=str)
    parser.add_argument('--use_fixed_poses', action='store_true', 
                       help='Use wrist-optimized poses from .npy files instead of original poses')
    
    args = parser.parse_args()                                      
    return args

if __name__ == '__main__':
    args=parse_args()
    SECTION_NUMBER=args.number
    USE_FIXED_POSES = args.use_fixed_poses
    
    if USE_FIXED_POSES:
        print("Using wrist-optimized poses from .npy files")
        export_file = f"./omomo_optimization_0.22_f_palm/"
    else:
        print("Using original poses from .npz files")
        export_file = f"./omomo_optimization_0.22_f/"

    OMOMO_DATA_ROOT=f'data/omomo/sequences_canonical' 
    
    LISTDIR=np.load('./omomo_listdir.npy', allow_pickle=True)
    # LIST=LISTDIR
    LEN_LISTDIR=len(LISTDIR)#.shape[0]

    SEQ_LEN=LEN_LISTDIR//30+1
    LIST=list(LISTDIR[SECTION_NUMBER*SEQ_LEN:min(LEN_LISTDIR,SECTION_NUMBER*SEQ_LEN+SEQ_LEN)])

    print(f"Processing {len(LIST)} sequences...")
    print(f"Export directory: {export_file}")
    
    for i,fn in tqdm(enumerate(LIST)):
        try:
            output_file = os.path.join(export_file,fn+'.npy')
            if os.path.isfile(output_file):
                print(f"Skipping {fn} - output file already exists: {output_file}")
                continue
            
            print(f"Processing sequence {i+1}/{len(LIST)}: {fn}")
            name=os.path.join(OMOMO_DATA_ROOT,fn)
            
            optimize1(0,name,True,fn,USE_FIXED_POSES)
            print(f"Completed processing: {fn}")
        except Exception as e:
            print(f"Error processing {fn}: {e}")
            import traceback
            traceback.print_exc()

        
        
    
    
    
    