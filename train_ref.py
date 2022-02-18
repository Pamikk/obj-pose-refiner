from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import os
import time
import tqdm
import argparse
import resource
import numpy as np
import yaml

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR, MultiStepLR
import torch.backends.cudnn as cudnn
import sys
from open3d import *

#utils and funs from ffb6d
from com_utils import *

from ref_config import Config as Config_ref
from algs import Refiners,ref_loss
from ref_utils import *
from containers import *
def parse_arguments():
    parser = argparse.ArgumentParser("Train and Evaluate Object Pose Refinement")
    parser.add_argument(
        "cfg_file", type=str,
        help="Config File for training"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("-g", "--gpus", default=8, type=int,
                        help="number of gpus per node")
    parser.add_argument("--gpu", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("-n", "--nodes", default=1, type=int, metavar="N")
    parser.add_argument("--eval_net", action="store_true", help="Evaluate network only")
    parser.add_argument("--resume", action="store_true", help="resume frome previous exp")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint to start from"
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("overrides", nargs="*", default=[])
    args = parser.parse_args()

    return args


lr_clip = 1e-5
bnm_clip = 1e-2
symlist = [7,8] #linemod
def compute_pose_error(pose,gt):
    R = torch.matmul(pose[:3,:3],gt[:3,:3].transpose(0,1).contiguous())
    rot_err = torch.acos((torch.trace(R)-1)/2)
    trans_err = torch.norm(pose[:3,3]-gt[:3,3],p=2)
    return rot_err,trans_err

def model_fn_decorator(
    criterion, cfg,cfg_ref,model_pts, args, writer):
    teval = TorchEval(cfg)

    def model_fn(
        model,
        data,
        it=0,
        epoch=0,
        is_eval=False,
        is_test=False,
        finish_test=False,
        test_pose=False,
        log_wandb=False):
        if finish_test:
            return teval.cal_lm_add(cfg.cls_id)
        if is_eval:
            model.eval()
            criterion.eval= True
        else:
            criterion.eval = False

        with torch.set_grad_enabled(not is_eval):
            gt_pose = data['RTs'][:,0,...]
            init_pose = gen_coarse_pose_rand_batch(gt_pose,cfg_ref.gen_params)
            model_points = sample_points_rand_batch(model_pts,cfg_ref.num_points_mesh).cuda()
            
            pred_r = init_pose[:,:3,:3].cuda()
            pred_t = init_pose[:,:3,3].view(-1,3,1).cuda()
            gt_r = gt_pose[:,:3,:3].cuda()
            gt_t = gt_pose[:,:3,3].view(-1,3,1).cuda()
            if is_eval:
                target = project_pose_batch(gt_r,gt_t,criterion.mesh)
                loss_src = criterion.mesh
            else:
                target = project_pose_batch(gt_r,gt_t,model_points)
                loss_src = model_points
            cur_dis = criterion(pred_r, pred_t, target,loss_src,cfg.cls_id in symlist)
            new_points = project_pose_batch(pred_r,pred_t,model_points)
            cld = data['cld_rgb_nrm'][:,:3,:].transpose(1,2)
            dmap = sample_points_rand_batch(cld,cfg_ref.num_points).cuda()
            dis_init = cur_dis
            start = time.time()
            pose = init_pose
            prev = 1.0
            src,tgt = model_points,dmap
            scale = cfg_ref.scale
            if  cfg_ref.model in ['df','dfv2']:
                for _ in range(0, cfg_ref.iters):
                    pred_r, pred_t = model(src*scale,tgt*scale,pose)
                    dis = criterion(pred_r, pred_t, target, loss_src,cfg.cls_id in symlist)
                    dis_r = criterion(pred_r, gt_t, target, loss_src,cfg.cls_id in symlist)
                    dis_t = criterion(gt_r, pred_t, target, loss_src,cfg.cls_id in symlist)
                    dis_pt = criterion(gt_r, gt_t, target, loss_src,cfg.cls_id in symlist)
                    pose = torch.cat((pred_r,pred_t),dim=-1)
                    for i in range(gt_pose.shape[0]):
                        rot_err,trans_err = compute_pose_error(pose[i,...],gt_pose[i,...].cuda())
                        dis_pt += rot_err+trans_err
                    loss = dis_pt + dis_r + 2*dis_t
                    if not is_eval:
                        loss.backward()
                    if abs(cur_dis-dis)<0:
                        break
                    cur_dis = dis.item()
                    pose = pose.detach()
            elif 'cor' == cfg_ref.model:
                grid = np.array(cfg_ref.grids)
                for i in range(cfg_ref.iters):
                    src,tgt,pred_r,pred_t = model(src*scale,tgt*scale,pose,tuple(grid))
                    dis = criterion(pred_r, pred_t, target, loss_src,cfg.cls_id in symlist)
                    dis_r = criterion(pred_r, gt_t, target, loss_src,cfg.cls_id in symlist)
                    dis_t = criterion(gt_r, pred_t,target, loss_src,cfg.cls_id in symlist)
                    dis_pt= criterion(gt_r,gt_t,tgt,src,cfg.cls_id in symlist)
                    if cfg_ref.disentangle:
                        loss = 0.5*(dis_r + dis_t)+2.5*dis_pt
                    else:
                        loss = 0.5*dis+0.5*dis_pt
                    if not is_eval:
                        loss.backward()
                    src = src.detach()
                    tgt = tgt.detach()
                    grid = grid/2
                    pose = torch.cat((pred_r,pred_t),dim=-1).detach()
            elif cfg_ref.model in['dcor','csel','transf']:
                for i in range(cfg_ref.iters):
                    src,tgt,pred_r,pred_t = model(src*scale,tgt*scale,pose)
                    dis = criterion(pred_r, pred_t, target, loss_src,cfg.cls_id in symlist)
                    dis_r = criterion(pred_r, gt_t, target, loss_src,cfg.cls_id in symlist)
                    dis_t = criterion(gt_r, pred_t,target, loss_src,cfg.cls_id in symlist)
                    dis_pt = criterion(gt_r,gt_t,tgt,src,cfg.cls_id in symlist)
                    if cfg_ref.disentangle:
                        loss = 0.5*(dis_r + dis_t)+2.5*dis_pt
                    else:
                        loss = 0.5*dis+0.5*dis_pt
                    if not is_eval:
                        loss.backward()
                    src = model_points
                    tgt = tgt.detach()
                    pose = torch.cat((pred_r,pred_t),dim=-1).detach()
            elif cfg_ref.model == 'icp':
                for i in range(cfg_ref.iters):
                    src,tgt,pred_r,pred_t = model(src*scale,tgt*scale,pose)
                    dis = criterion(pred_r, pred_t, target, loss_src,cfg.cls_id in symlist)
                    dis_r = criterion(pred_r, gt_t, target, loss_src,cfg.cls_id in symlist)
                    dis_t = criterion(gt_r, pred_t,target, loss_src,cfg.cls_id in symlist)
                    dis_pt = criterion(gt_r,gt_t,tgt,src,cfg.cls_id in symlist)
                    cur = criterion(pred_r,pred_t,tgt,src,cfg.cls_id in symlist).item()
                    if cfg_ref.disentangle:
                        loss = 0.5*(dis_r + dis_t)+2.5*dis_pt
                    else:
                        loss = 0.5*dis+0.5*dis_pt
                    if not is_eval:
                        loss.backward()
                    src = model_points
                    tgt = tgt.detach()
                    pose = torch.cat((pred_r,pred_t),dim=-1).detach()
                    if abs(cur-prev)<cfg_ref.threshold:
                        break
                    prev = cur

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
            if is_test:
                loss_dict = {
                'loss_target': loss.item(),
                'dis_init':dis_init.item()/div_factor,
                'dis_ref' :dis.item()/div_factor,
                'rot_ref': np.mean(accuracy['rot']), 'trans_ref': np.mean(accuracy['trans']),
                'rot_gen': np.mean(err['rot']), 'trans_gen': np.mean(err['trans']),
                'rot_diff': np.mean(err['rot'])-np.mean(accuracy['rot']), 'trans_diff': np.mean(err['trans'])-np.mean(accuracy['trans']),
                'dis_diff':dis_init.item()/div_factor-dis.item()/div_factor,'runtime':inf_time,'gain':(dis_init.item()/div_factor-dis.item()/div_factor)/(dis_init.item()/div_factor)
            }
            else:
                loss_dict = {
                'loss_target': loss.item(),
                'loss_rot':dis_r.item(),
                'loss_t':dis_t.item(),
                'loss_pt':dis_pt.item() if dis_pt!=None else 0,
                'dis_init':dis_init.item()/div_factor,
                'dis_ref' :dis.item()/div_factor,
                'rot_ref': np.mean(accuracy['rot']), 'trans_ref': np.mean(accuracy['trans']),
                'rot_gen': np.mean(err['rot']), 'trans_gen': np.mean(err['trans'])
            }
            if test_pose:

                # eval pose from point cloud prediction.
                teval.eval_pose_ref(
                    gt_pose,pred_pose,model_points,
                    ds='linemod', obj_id=cfg.dataset.cls_id,
                    min_cnt=1, use_ctr_clus_flter=True, use_ctr=True,
                )
            info_dict = loss_dict.copy()

            if not is_eval:
                if args.local_rank == 0 and writer is not None:
                    writer.write({"train/{}".format(k): v for k, v in info_dict.items()}, it)

        return (
            None, loss, info_dict
        )

    return model_fn

def train(args, cfg, cfg_ref,writer):
    print("local_rank:", args.local_rank)
    if not args.eval_net:
        train_ds = Dataset(cfg, cfg.train_data)
        #train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=cfg_ref.bs, shuffle=True,
            drop_last=True, num_workers=4, pin_memory=True
        )

        val_ds = Dataset(cfg, cfg.train_data)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=1, shuffle=False,
            drop_last=False, num_workers=4
        )
    else:
        test_ds = Dataset(cfg, cfg.train_data)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=cfg.test_mini_batch_size, shuffle=False,
            num_workers=10
        )
    
    model = Refiners[cfg_ref.model](cfg_ref)

    #model = convert_syncbn_model(model)
    model = model
    device = torch.device('cuda:{}'.format(args.local_rank))
    print('local_rank:', args.local_rank)
    model.to(device)
    #print(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=cfg_ref.lr,weight_decay=1e-5)
    opt_level = cfg.train.opt_level
    #model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    #model.initialization()

    # default value
    it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
    best_loss = 1e10
    start_epoch = 1
    # load status from checkpoint
    if args.checkpoint is not None:
        checkpoint_filename = args.checkpoint[:-8]
        checkpoint_filename = os.path.join(cfg.log_model_dir, checkpoint_filename)
    elif args.checkpoint is None and args.eval_net:
        checkpoint_filename = os.path.join(cfg.log_model_dir, f"{cfg_ref.model}_{cfg.cls_type}_best")
    else:
        checkpoint_filename = ""
    if (cfg_ref.model=='icp') and (cfg_ref.decode =='svd'):
        checkpoint_filename = ""
    if checkpoint_filename:
        print(f"load from:{checkpoint_filename}")
        checkpoint_status = load_checkpoint(
            model, None if args.eval_net else optimizer, filename=checkpoint_filename
        )
        if checkpoint_status is not None:
            it, start_epoch, best_loss = checkpoint_status
        if args.eval_net:
            assert checkpoint_status is not None, "Failed loading checkpoint for evaluation."

    if not args.eval_net:
        model = torch.nn.DataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )
        clr_div = cfg.num_of_lr_cycles * 2
        #step_size_up = cfg.n_total_epoch * train_ds.minibatch_per_epoch // clr_div // args.gpus
        #step_size_down = cfg.n_total_epoch * train_ds.minibatch_per_epoch // clr_div // args.gpus
        #print(step_size_down,step_size_up)
        '''lr_scheduler = CyclicLR(
            optimizer,
            base_lr=cfg_ref.lr,
            max_lr=1e-4,
            cycle_momentum=False,
            step_size_up=step_size_up,
            step_size_down=step_size_down,
            mode='triangular'
        )'''
        lr_scheduler = MultiStepLR(optimizer,milestones=cfg_ref.schedule,gamma=0.1)
    else:
        lr_scheduler = None


    it = max(it, 0)  # for the initialize value of `trainer.train`
    cfg.cls_id = cfg.dataset.obj_id_dict[cfg.cls_type]
    model_path = os.path.join(cfg.dataset.data_dir,"models",f'obj_{cfg.cls_id:02d}.ply')
    model_xyz = torch.tensor(get_p3ds_from_ply(model_path))/1000.0
    cfg_ref.print_properties()
    #print(model_xyz.shape)
    model_fn = model_fn_decorator(
            ref_loss(mesh=model_xyz,bs=cfg_ref.bs,if_sum=cfg_ref.if_sum),
            cfg,
            cfg_ref,
            model_xyz,
            args,
            writer
        )

    trainer = Trainer_ref(
        args,
        cfg,
        model,
        model_fn,
        optimizer,
        checkpoint_name=os.path.join(cfg.log_model_dir, f"{cfg_ref.model}_{cfg.cls_type}"),
        best_name=os.path.join(cfg.log_model_dir, f"{cfg_ref.model}_{cfg.cls_type}_best"),
        lr_scheduler=lr_scheduler,
        bnm_scheduler=None,
        writer=writer
    )

    if args.eval_net:
        start = time.time()
        np.random.seed(2333)
        _ = trainer.eval_epoch(
            test_loader, is_test=True, test_pose=cfg.test.test_pose
        )
        end = time.time()
        print("\nUse time: ", end - start, 's')
    else:
        trainer.train(
            it, start_epoch, cfg.n_total_epoch, train_loader, None,
            val_loader, best_loss=best_loss,
            tot_iter=cfg.n_total_epoch * len(train_ds)/cfg_ref.bs // args.gpus,
            clr_div=clr_div
        )
        val_loss, res = trainer.eval_epoch(val_loader, it=it, is_test=True)
        print("final result")
        print("val_loss", val_loss)
        for k in res:
            print(k,np.mean(res[k]))

def main(cfg,cfg_ref,args):
    writer = None
    cfg.wandb.log_imgs = False

    if args.local_rank == 0 and cfg.wandb.enable:
        writer = events.WandBWriter(
            name="{}-{}".format(cfg.cls_type, "{}-{}".format(
                cfg.name,
                "test" if args.eval_net else "train")),
            group_name=cfg.group,
            project_name=cfg.wandb.project_name,
            output_dir=cfg.exp_log_dir
        )

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (30000, rlimit[1]))

    args.world_size = args.gpus * args.nodes
    train(args, cfg,cfg_ref, writer)

    if not args.eval_net:
        config.save_config(cfg)
        yaml.dump(Config_ref(),open(os.path.join(cfg.log_traininfo_dir,"config_ref.yaml"),'w'))

    if args.local_rank == 0:
        writer.finish()
import itertools
def find_underscore(s,num):
    start = 0
    while (num>0):
        start = s.find('_',start)+1
        if (start==-1):
            return -1
        num-=1
    return start-1
if __name__ == "__main__":
    args = parse_arguments()
    cudnn.benchmark = True
    cfg = config.create_config(args.cfg_file)
    if cfg.train.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)
    torch.cuda.set_device(args.local_rank)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    net_params = list(itertools.product(['ortho','quat'],['none']))
    #net_params = [(False,True)]
    trained = []#['df_15_15_quat_dis_sum','df_15_15_quat_dis','df_15_15_quat_sum','df_15_15_quat']
    genp = '_05_50'
    if 'df' in cfg.name:
        cfg_ref = Config_ref(cfg.name)
    elif 'icp' in cfg.name:
        cfg_ref = Config_ref('icp')
    if (cfg_ref.model=='icp') and (cfg_ref.decode =='svd'):
        args.eval_net = True
    
    obj_id = cfg.dataset.obj_id_dict[cfg.cls_type]
    threshold = 0.5
    test_genp = 'df_15_25'
    dm_add = (cfg.dataset.r_lst[obj_id]['diameter'] / 1000.0 * threshold)
    cfg_ref.gen_params = [10,12.5,dm_add,dm_add,dm_add]
    
    cfg.group = cfg_ref.model+'_lm'
    cfg_ref.scale = 1.0
    net_params =net_params
    for net_op in net_params:
        cfg_ref.decode,cfg_ref.dtrans = net_op
        if tuple(net_op) in trained:
            continue
        cfg.name = ''
        cfg.__post_init__()
        main(cfg,cfg_ref,args)