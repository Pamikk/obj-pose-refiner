# Generate coarse pose to simulate predicted pose from network
import numpy as np
from .utils import *
import torch
import cv2
def project_pose(pose,model_points):
    rot = pose[:3,:3]
    trans = pose[:3,3]
    rot = rot.transpose(1, 0)
    new_points = torch.add(torch.matmul(model_points, rot), trans.view(1,3))
    return new_points
def calculate_distance(target,points):
    return torch.mean(torch.norm(points - target,p=2, dim=-1)).item()
def CalRotMatDist(rotA,rotB,if_degree=True):
    error_vec, _ = cv2.Rodrigues(rotA.dot(rotB.T))
    diff_rad = np.linalg.norm(error_vec)
    if if_degree:
        return np.rad2deg(diff_rad)
    else:
        return diff_rad
def CalPoseDist(poseA,poseB,if_degree=True):
    r_dist = CalRotMatDist(np.array(poseA[:3,:3]),np.array(poseB[:3,:3]),if_degree)
    t_dist = np.linalg.norm(poseA[:,3]-poseB[:,3])
    return r_dist,t_dist
gen_num_default = 1 #generation number for per source pose
# Parameters referring DeepIM
K_default = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])# Intrinsic Matrix, different for different dataset
# generate a set of my val
angle_std, angle_max, x_std, y_std, z_std = default_params = [15.0, 45.0, 0.01, 0.01, 0.05]
def sample_points_rand_batch(points,num_points):
    """
        randomly sample the points by batch

    """
    if len(points.shape)==2:
        points = points.unsqueeze(dim=0)
    bs,pt_num,dims = points.shape
    if pt_num<=num_points:
        return points
    idx = torch.stack([torch.randperm(pt_num) for _ in range(bs)],dim=0)
    idx,_ = torch.sort(idx[:,:num_points],dim=-1) # keep order of points
    sampled = torch.stack([torch.gather(points[...,i],1,idx) for i in range(dims)],dim=-1)
    return sampled
def gen_coarse_pose_rand(src_pose,intrinsic=K_default,params=default_params,gen_num = gen_num_default):
    # input 
    # src_pose: gt pose mat[R|t](3x4) 
    # params:[angle_std,angle_max,x_std,y_std,z_std],degree for angles and cm for translations
    # K: intrinsic mat
    # output
    # des_pose: generated pose
    # errs = [rot_err,x_err,y_err,z_err]
    # r_err: deviation of rotation
    # t_err: deviation of translation
    #currently set for Linemod
    
    if isinstance(src_pose,torch.Tensor):
       dtype = src_pose.dtype
       device = src_pose.device
       to_tensor = True
       src_pose = src_pose.cpu().numpy()
    else:
        to_tensor = False
    if not isrotMat(src_pose[:3,:3]):
        print("why?")
    angle_std, angle_max, x_std, y_std, z_std = params
    src_angles = np.squeeze(np.array(Mat2Euler(src_pose[:3,:3])))
    src_trans = np.squeeze(np.array(src_pose[:,3]))
    gen_poses = []
    for i in range(gen_num):
        r_dist = angle_max+1
        x_c = 0
        y_c = 0
        count = 0 #genration counter
        errors = []
        while r_dist> angle_max or not((16 < x_c < (640 - 16) )and (16 < y_c < (480 - 16))):
            #not valid generation standard, change dependiing on dataset
            if angle_std>0:
                dst_angles = src_angles+np.rad2deg(np.random.normal(0,np.radians(angle_std),3))
                dst_rot = Euler2Mat(*dst_angles)
            else:
                dst_rot = src_pose[:3,:3]
            if x_std>0:
                x_err = np.random.normal(0,x_std)
            else:
                x_err = 0.0
            if y_std>0:
                y_err = np.random.normal(0,y_std)
            else:
                y_err = 0.0
            if z_std>0:
                z_err = np.random.normal(0,z_std)
            else:
                z_err = 0.0
            dst_trans = src_trans + np.array([x_err,y_err,z_err])
            new_pose = np.hstack((dst_rot,dst_trans.reshape(3,1)))
            r_dist = CalRotMatDist(src_pose[:3,:3],dst_rot)
            transform = np.matmul(intrinsic,dst_trans)
            x_c,y_c = transform[:2]/transform[2]#model center
            count+=1
            if (count>100):
                print("Can't get valid generation after 100 times generation")
                break
        gen_poses.append(new_pose)
        errors.append([r_dist,x_err,y_err,z_err])
    if to_tensor:
            gen_poses = torch.tensor(gen_poses,dtype=dtype,device=device)
    return gen_poses,errors
def visualize_pt_cloud(cloud):
    pass
def depthmap2ptcloud(dmap,intrinsic=K_default,depth_scale =1.0):
    cx,cy = intrinsic[:2,3]
    fx,fy = intrinsic[0,0],intrinsic[1,1]
    w,h = dmap.shape[:2]
    xmap,ymap = np.meshgrid(range(w),range(h),indexing='ij')
    zmap = dmap/depth_scale
    point_cloud = np.stack(((xmap-cx)*zmap/fx,(ymap-cy)*zmap/fy,zmap.squeeze()),axis=-1).flatten()
    return point_cloud
def project_pose_batch(rot,trans,points):
    bs = rot.shape[0]
    points = points.view(1,-1,3).repeat(bs,1,1)
    points.to(rot.device)
    target = torch.add(torch.bmm(points, rot.transpose(2, 1)), trans.view(-1,1,3))
    return target
def gen_coarse_pose_rand_batch(src_pose,parameters):
    "Efficently genreate coarse pose without checking pose validity"
    bs = src_pose.shape[0]
    angle_std, angle_max, x_std, y_std, z_std = parameters
    
    if bs==1:
        new_pose = src_pose.clone()
        new_pose[0,...],_ = gen_coarse_pose_rand(src_pose[0,:,:],params=parameters)
        return new_pose
    src_trans =  src_pose[:,:,3]
    #not valid generation standard, change dependiing on dataset
    if angle_std>0:
        src_angles = Mat2Euler_torch(src_pose[:,:3,:3])
        deg_err = np.degrees(torch.normal(0,np.deg2rad(angle_std),size=(bs,3)))
        mask = deg_err.sum(dim=1)>=angle_max
        while torch.any(mask):
            deg_err[mask] = np.degrees(torch.normal(0,np.deg2rad(angle_std),size=(int(mask.sum().item()),3)))
            mask = deg_err.sum(dim=1)>angle_max
        dst_angles = torch.add(src_angles,deg_err)
        dst_rot = Euler2Mat_torch(dst_angles[:,0],dst_angles[:,1],dst_angles[:,2])
    else:
        dst_rot = src_pose[:,:3,:3].clone()
    if x_std>0:
        x_err = torch.normal(0,x_std,size=(bs,1))
    else:
        x_err = torch.zeros(bs,1,dtype=torch.float)
    if y_std>0:
        y_err = torch.normal(0,y_std,size=(bs,1))
    else:
        y_err = torch.zeros(bs,1,dtype=torch.float)
    if z_std>0:
        z_err = torch.normal(0,z_std,size=(bs,1))
    else:
        z_err = torch.zeros(bs,1,dtype=torch.float)
    dst_trans = src_trans + torch.cat([x_err,y_err,z_err],dim=1)
    new_pose = torch.cat((dst_rot,dst_trans.view(bs,3,1)),dim=-1)
    return new_pose