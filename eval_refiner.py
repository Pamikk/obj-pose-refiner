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
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
import torch.backends.cudnn as cudnn
from containers import *
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from ref_config import Config as Params
from ref_utils import *
from train_ref import compute_pose_error

cfg_file = "configs/linemod-test.yaml"
cfg = config.create_config(cfg_file)
cfg.is_training = False
save_path ="vis_out"
symlist = []
genp = '_10_50'
def model_fn_decorator(
    criterion, cfg,cfg_ref,model_pts):
    teval = TorchEval(cfg)

    def model_fn(model,data):
        model.eval()
        criterion.eval= True

        with torch.set_grad_enabled(False):
            gt_pose = data['RTs'][:,0,...]
            init_pose = gen_coarse_pose_rand_batch(gt_pose,cfg_ref.gen_params)
            model_points = sample_points_rand_batch(model_pts,cfg_ref.num_points_mesh).cuda()
            
            pred_r = init_pose[:,:3,:3].cuda()
            pred_t = init_pose[:,:3,3].view(-1,3,1).cuda()
            gt_r = gt_pose[:,:3,:3].cuda()
            gt_t = gt_pose[:,:3,3].view(-1,3,1).cuda()
            target = project_pose_batch(gt_r,gt_t,criterion.mesh)
            loss_src = criterion.mesh
            cur_dis = criterion(pred_r, pred_t, target,loss_src,cfg.cls_id in symlist)
            cld = data['cld_rgb_nrm'][:,:3,:].transpose(1,2)
            dmap = sample_points_rand_batch(cld,cfg_ref.num_points).cuda()
            dis_init = cur_dis
            start = time.time()
            pose = init_pose.cuda()
            prev = 10000
            src,tgt = model_points,dmap
            scale = cfg_ref.scale
            if  cfg_ref.model in ['df','dfv2']:
                for _ in range(0, cfg_ref.iters):
                    pred_r, pred_t = model(src*scale,tgt*scale,pose)
                    pose = pose.detach()
            elif 'cor' == cfg_ref.model:
                grid = np.array(cfg_ref.grids)
                for i in range(cfg_ref.iters):
                    src,tgt,pred_r,pred_t = model(src*scale,tgt*scale,pose,tuple(grid))
                    src = src.detach()
                    tgt = tgt.detach()
                    grid = grid/2
                    pose = torch.cat((pred_r,pred_t),dim=-1).detach()
            elif cfg_ref.model in['dcor','csel','transf']:
                for i in range(cfg_ref.iters):
                    src,tgt,pred_r,pred_t = model(src*scale,tgt*scale,pose)
                    src = model_points
                    tgt = tgt.detach()
                    pose = torch.cat((pred_r,pred_t),dim=-1).detach()
            elif cfg_ref.model == 'icp':
                src,tgt,pred_r,pred_t= model(src*scale,tgt*scale,pose)
                pose = torch.cat((pred_r,pred_t),dim=-1).detach()
                    #print(pose,compute_pose_error(pose[0,...],gt_pose[0,...].cuda()))
                #exit()
                print(gt_t)
            dis = criterion(pred_r, pred_t, target, loss_src,cfg.cls_id in symlist)
            dis_r = criterion(pred_r, gt_t, target, loss_src,cfg.cls_id in symlist)
            dis_t = criterion(gt_r, pred_t, target, loss_src,cfg.cls_id in symlist)
            dis_pt = criterion(gt_r, gt_t, target, loss_src,cfg.cls_id in symlist)
            pose = torch.cat((pred_r,pred_t),dim=-1)
            if cfg_ref.model =='df':
                for i in range(gt_pose.shape[0]):
                    rot_err,trans_err = compute_pose_error(pose[i,...],gt_pose[i,...].cuda())
                    dis_pt += rot_err+trans_err
            loss = dis_pt + dis_r + 2*dis_t
            cur_dis = dis.item()
            pred_pose = pose
            inf_time = time.time()-start
            accuracy = {'rot':[],'trans':[]}
            err = {'rot':[],'trans':[]}
            pred_pose = pred_pose.detach().cpu().numpy()
            for i in range(gt_pose.shape[0]):
                rot_err,trans_err = CalPoseDist(gt_pose[i,...],pred_pose[i,...])
                accuracy['rot'].append(rot_err)
                accuracy['trans'].append(trans_err)
                rot_err,trans_err = CalPoseDist(gt_pose[i,...],init_pose[i,...])
                err['rot'].append(rot_err)
                err['trans'].append(trans_err)
            div_factor = cfg_ref.num_points_mesh if cfg_ref.if_sum else 1.0
            loss_dict = {
                'loss_target': loss.item(),
                'dis_init':dis_init.item()/div_factor,
                'dis_ref' :dis.item()/div_factor,
                'rot_ref': np.mean(accuracy['rot']), 'trans_ref': np.mean(accuracy['trans']),
                'rot_gen': np.mean(err['rot']), 'trans_gen': np.mean(err['trans']),
                'rot_diff': np.mean(err['rot'])-np.mean(accuracy['rot']), 'trans_diff': np.mean(err['trans'])-np.mean(accuracy['trans']),
                'dis_diff':dis_init.item()/div_factor-dis.item()/div_factor,'runtime':inf_time}
            loss_dict['gain'] = loss_dict['dis_diff']/loss_dict['dis_init']
            info_dict = loss_dict.copy()

        return info_dict

    return model_fn
def evaluate(cfg,cfg_ref):
    cfg.wandb.log_imgs = False
    writer = None
    if cfg.wandb.enable:
        writer = events.WandBWriter(
            name="{}-{}".format(cfg.cls_type,cfg_ref.model),
            group_name='runtime',
            project_name="ope-lm-refiner-test",
            output_dir=cfg.exp_log_dir
        )
    train_ds = Dataset(cfg,cfg.train_data)
    print(train_ds.cfg.dataset.data_dir)
    model_path = os.path.join(train_ds.cfg.dataset.data_dir,"models",f'obj_{train_ds.cls_id:02d}.ply')
    cfg.cls_id = train_ds.cls_id
    model_xyz = torch.tensor(mesh_utils.get_p3ds_from_ply(model_path))/1000.0
    train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=1, shuffle=False,
                drop_last=True, num_workers=4,  pin_memory=True)
    model = Refiners[cfg_ref.model](cfg_ref)
    model_fn = model_fn_decorator(
            ref_loss(mesh=model_xyz,bs=cfg_ref.bs,if_sum=cfg_ref.if_sum),
            cfg,
            cfg_ref,
            model_xyz
        )
    model = model.cuda()
    checkpoint_filename = os.path.join(cfg.log_model_dir, f"df_{cfg.cls_type}_best")
    if (cfg_ref.model=='icp') and (cfg_ref.decode =='svd'):
        checkpoint_filename = ""
    if checkpoint_filename:
        print(f"load from:{checkpoint_filename}")
        checkpoint_status = load_checkpoint(
            model, None, filename=checkpoint_filename
        )
    cfg_ref.print_properties()
    np.random.seed(2333)
    res = {}
    for data in tqdm(train_loader):
        
        info_dict = model_fn(model,data)
        for k,v in info_dict.items():
            if k in res.keys():
                res[k].append(v)
            else:
                res[k] = [v]
    loss_dict = {}
    for k,v in res.items():
        loss_dict[k] = np.mean(v)
        res[k] = np.array(v)
    if not(writer==None):
        writer.write({"{}/{}".format(
                "test", k): v for k, v in loss_dict.items()})
    print(loss_dict)
    log_file = open(f'vis_out/{cfg.group+cfg.name}_res.txt','a')
    log_file.write(f"class:{cfg.cls_type},params:{cfg_ref.gen_params},dataset size:{len(train_ds)}\n")
    log_file.write(f"{loss_dict['rot_gen']}\t{loss_dict['rot_ref']}\t{loss_dict['trans_gen']}\t{loss_dict['trans_ref']}\t{loss_dict['dis_init']}\t{loss_dict['dis_ref']}\t{loss_dict['runtime']}\n")
    log_file.close()
    if writer !=None:
        writer.finish()
def parse_arguments():
    parser = argparse.ArgumentParser("Train and Evaluate Object Pose Estimation")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gpu", type=str, default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()

    return args
if __name__ =="__main__":
    cfg_ref = Params("dfv2")
    obj_id = cfg.dataset.obj_id_dict[cfg.cls_type]
    threshold = 0.5
    dm_add = (cfg.dataset.r_lst[obj_id]['diameter'] / 1000.0 * threshold)
    cfg_ref.gen_params = [15,30,dm_add,dm_add,dm_add]
    decode = ['orhto']
    cfg_ref.scale = 1.0
    cfg_ref.dtrans = 'none'
    for mode in decode:
        cfg_ref.decode = mode
        cfg.group = "dfv2lm"
        cfg.name = ''
        cfg.__post_init__()
        evaluate(cfg,cfg_ref)