import math
from re import L
import numpy as np
from scipy.linalg import logm
import os
import sys
from abc import abstractmethod
from typing import TypeVar

import argparse
from open3d import *
from ref_utils import gen_coarse_pose_rand, gen_coarse_pose_rand_batch,CalPoseDist
from com_utils import *
from algs import *
from containers import Dataset
import time

import torch
from containers import *
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from ref_config import Config as Params
from ref_utils import *
def train():
    print("rewrite train function")
def write_3d_open3d(path,name,pcd):
    io.write_point_cloud(os.path.join(path,name), pcd, write_ascii=False, compressed=False, print_progress=False)
cfg_file = "configs/linemod-test.yaml"
cfg = config.create_config(cfg_file)
cfg.is_training = False
save_path ="vis_out"
def evaluate(cfg,params):
    cfg.wandb.log_imgs = False
    writer = None
    cfg.wandb.enable = False
    if cfg.wandb.enable:
        writer = events.WandBWriter(
            name="{}-{}".format(cfg.cls_type,"test"),
            group_name=cfg.wandbgroup,
            project_name="ope-lm-refiner-test",
            output_dir="../../exp/icp_lm"
        )
    train_ds = Dataset(cfg,cfg.train_data)
    print(train_ds.cfg.dataset.data_dir)
    model_path = os.path.join(train_ds.cfg.dataset.data_dir,"models",f'obj_{train_ds.cls_id:02d}.ply')
    model_xyz = torch.tensor(mesh_utils.get_p3ds_from_ply(model_path))/1000
    train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=1, shuffle=False,
                drop_last=True, num_workers=4,  pin_memory=True)
    icp_refiner = ICP(params)
    errors = {'rot':[],'trans':[]}
    accuracy = {'rot':[],'trans':[]}
    total_time = 0.0
    count = 0
    params.print_properties()
    np.random.seed(2333)
    dis_init = 0.0
    dis=0.0
    for data in tqdm(train_loader):
        gt_pose = data['RTs'][:,0,...]
        #print(gt_pose)
        '''for k in data:
            if 'cld' in k:
                print(k,data[k].shape)'''
        pt_cloud = data['cld_rgb_nrm'][:,:3,:].transpose(1,2)
        init_pose = gen_coarse_pose_rand_batch(gt_pose,parameters = params.gen_params).squeeze()
        start = time.time()
        ref_pose,_ = icp_refiner(model_xyz,pt_cloud,init_pose)
        inf_time = time.time()-start
        gt_pose = gt_pose.squeeze()
        init_rot,init_trans = CalPoseDist(gt_pose,init_pose.squeeze())
        rot_err,trans_err = CalPoseDist(gt_pose,ref_pose)
        print(rot_err,trans_err)
        print(init_rot,init_trans)
        exit()
        dis_init += calculate_distance(project_pose(gt_pose,model_xyz),project_pose(init_pose,model_xyz))
        dis += calculate_distance(project_pose(gt_pose,model_xyz),project_pose(ref_pose,model_xyz))
        errors['rot'].append(init_rot)
        errors['trans'].append(init_trans)
        accuracy['rot'].append(rot_err)
        accuracy['trans'].append(trans_err)
        total_time += inf_time
        count+=1
    obj_id = cfg.dataset.obj_id_dict[cfg.cls_type]
    threshold = 0.25
    dm_add = (cfg.dataset.r_lst[obj_id]['diameter'] / 1000.0 * threshold)
    loss_dict = {
            'dis_init':dis_init/count,
            'dis_ref' :dis/count,
            'rot_ref': np.mean(accuracy['rot']), 'trans_ref': np.mean(accuracy['trans']),
            'rot_gen': np.mean(errors['rot']), 'trans_gen': np.mean(errors['trans']),
            'rot_diff': np.mean(errors['rot'])-np.mean(accuracy['rot']), 'trans_diff': np.mean(errors['trans'])- np.mean(accuracy['trans']),
            'rot_diff_std': np.std(np.array(errors['rot'])-np.array(accuracy['rot'])), 'trans_diff_std':np.std(np.array(errors['trans'])- np.array(accuracy['trans'])),
            'dis_diff':(dis_init-dis)/count,'runtime':total_time/count
    }
    if not(writer==None):
        writer.write({"{}/{}".format(
                "test-m", k): v for k, v in loss_dict.items()})
    for k,v in loss_dict.items():
        if 'rot' not in k:
            loss_dict[k] = v/dm_add
        print(k,v)
    if not(writer==None):
        writer.write({"{}/{}".format(
                "test-diameter", k): v for k, v in loss_dict.items()})
    
    log_file = open(f'vis_out/icp_res.txt','a')
    log_file.write(f"class:{cfg.cls_type},params:{params.gen_params},dataset size:{len(train_ds)}\n")
    log_file.write(f"{round(np.mean(errors['rot']),3)}\t{round(np.mean(errors['trans'])/dm_add,3)}\t{round(np.mean(accuracy['rot']),3)}\t{round(np.mean(accuracy['trans'])/dm_add,3)}\t{round(np.std(accuracy['rot']),3)}\t{round(np.std(accuracy['trans']),3)}\t{total_time/count}\n")
    log_file.close()
    writer.finish()
def parse_arguments():
    parser = argparse.ArgumentParser("Train and Evaluate Object Pose Estimation")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gpu", type=str, default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()

    return args




if __name__ =="__main__":
    params = Params("icp")
    obj_id = cfg.dataset.obj_id_dict[cfg.cls_type]
    threshold = 0.5
    dm_add = (cfg.dataset.r_lst[obj_id]['diameter'] / 1000.0 * threshold)
    params.gen_params = [25,30,dm_add/5,dm_add/5,dm_add]
    cfg.wandbgroup = 'icp'
    cfg.name = 'icp_top500'
    evaluate(cfg,params)