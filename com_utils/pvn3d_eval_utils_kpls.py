#!/usr/bin/env python3
import os
import torch
import numpy as np
import concurrent.futures

# from common import Config
# import config
from com_utils.meanshift_pytorch import MeanShiftTorch
from com_utils import visualization_utils, pcl_utils, metrics_utils, file_utils
from itertools import repeat

from configs import config

from cv2 import imshow, waitKey


# ###############################YCB Evaluation###############################
def cal_frame_poses(
    cfg, pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter,
    gt_kps, gt_ctrs, debug=False, kp_type='farthest'
):
    """
    Calculates pose parameters by 3D keypoints & center points voting to build
    the 3D-3D corresponding then use least-squares fitting to get the pose parameters.
    """
    n_kps, n_pts, _ = pred_kp_of.size()
    pred_ctr = pcld - ctr_of[0]
    pred_kp = pcld.view(1, n_pts, 3).repeat(n_kps, 1, 1) - pred_kp_of

    radius = cfg.ms_radius
    if use_ctr:
        cls_kps = torch.zeros(n_cls, n_kps + 1, 3).cuda()
    else:
        cls_kps = torch.zeros(n_cls, n_kps, 3).cuda()

    # Use center clustering filter to improve the predicted mask.
    pred_cls_ids = np.unique(mask[mask > 0].contiguous().cpu().numpy())
    if use_ctr_clus_flter:
        ctrs = []
        for icls, cls_id in enumerate(pred_cls_ids):
            cls_msk = (mask == cls_id)
            ms = MeanShiftTorch(bandwidth=radius)
            ctr, ctr_labels = ms.fit(pred_ctr[cls_msk, :])
            ctrs.append(ctr.detach().contiguous().cpu().numpy())
        try:
            ctrs = torch.from_numpy(np.array(ctrs).astype(np.float32)).cuda()
            n_ctrs, _ = ctrs.size()
            pred_ctr_rp = pred_ctr.view(n_pts, 1, 3).repeat(1, n_ctrs, 1)
            ctrs_rp = ctrs.view(1, n_ctrs, 3).repeat(n_pts, 1, 1)
            ctr_dis = torch.norm((pred_ctr_rp - ctrs_rp), dim=2)
            min_dis, min_idx = torch.min(ctr_dis, dim=1)
            msk_closest_ctr = torch.LongTensor(pred_cls_ids).cuda()[min_idx]
            new_msk = mask.clone()
            for cls_id in pred_cls_ids:
                if cls_id == 0:
                    break
                min_msk = min_dis < cfg.r_lst[cls_id - 1] * 0.8
                update_msk = (mask > 0) & (msk_closest_ctr == cls_id) & min_msk
                new_msk[update_msk] = msk_closest_ctr[update_msk]
            mask = new_msk
        except Exception:
            pass

    # 3D keypoints voting and least squares fitting for pose parameters estimation.
    pred_pose_lst = []
    pred_kps_lst = []
    for icls, cls_id in enumerate(pred_cls_ids):
        if cls_id == 0:
            break
        cls_msk = mask == cls_id
        if cls_msk.sum() < 1:
            pred_pose_lst.append(np.identity(4)[:3, :])
            pred_kps_lst.append(np.zeros((n_kps + 1, 3)))
            continue

        cls_voted_kps = pred_kp[:, cls_msk, :]
        ms = MeanShiftTorch(bandwidth=radius)
        ctr, ctr_labels = ms.fit(pred_ctr[cls_msk, :])
        if ctr_labels.sum() < 1:
            ctr_labels[0] = 1
        if use_ctr:
            cls_kps[cls_id, n_kps, :] = ctr

        if use_ctr_clus_flter:
            in_pred_kp = cls_voted_kps[:, ctr_labels, :]
        else:
            in_pred_kp = cls_voted_kps

        for ikp, kps3d in enumerate(in_pred_kp):
            cls_kps[cls_id, ikp, :], _ = ms.fit(kps3d)

        # visualize
        if debug:
            show_kp_img = np.zeros((480, 640, 3), np.uint8)
            kp_2ds = pcl_utils.project_p3d(
                cls_kps[cls_id].cpu().numpy(), K="linemod", scale_m=1000.0
            )
            color = visualization_utils.get_label_color(cls_id.item())
            show_kp_img = visualization_utils.draw_p2ds(show_kp_img, kp_2ds, r=3, color=color)
            imshow("kp: cls_id=%d" % cls_id, show_kp_img)
            waitKey(0)

        # Get mesh keypoint & center point in the object coordinate system.
        # If you use your own objects, check that you load them correctly.
        mesh_kps = cfg.dataset.kps(cfg.dataset.cls_lst[cls_id - 1], kp_type=kp_type)
        if use_ctr:
            mesh_ctr = cfg.dataset.ctr(cfg.dataset.cls_lst[cls_id - 1]).reshape(1, 3)
            mesh_kps = np.concatenate((mesh_kps, mesh_ctr), axis=0)
        pred_kpc = cls_kps[cls_id].squeeze().contiguous().cpu().numpy()
        pred_RT = metrics_utils.best_fit_transform(mesh_kps, pred_kpc)
        pred_kps_lst.append(pred_kpc)
        pred_pose_lst.append(pred_RT)

    return (pred_cls_ids, pred_pose_lst, pred_kps_lst)


def eval_metric(
    cfg, n_cls, cls_ids, pred_pose_lst, pred_cls_ids, RTs, mask, label,
    gt_kps, gt_ctrs, pred_kpc_lst
):
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]
    cls_kp_err = [list() for i in range(n_cls)]
    for icls, cls_id in enumerate(cls_ids):
        if cls_id == 0:
            break

        gt_kp = gt_kps[icls].contiguous().cpu().numpy()

        cls_idx = np.where(pred_cls_ids == cls_id[0].item())[0]
        if len(cls_idx) == 0:
            pred_RT = torch.zeros(3, 4).cuda()
            pred_kp = np.zeros(gt_kp.shape)
        else:
            pred_RT = pred_pose_lst[cls_idx[0]]
            pred_kp = pred_kpc_lst[cls_idx[0]][:-1, :]
            pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
        kp_err = np.linalg.norm(gt_kp - pred_kp, axis=1).mean()
        cls_kp_err[cls_id].append(kp_err)
        gt_RT = RTs[icls]
        mesh_pts = cfg.dataset.get_pointxyz_cuda().clone()
        add = metrics_utils.cal_add_cuda(pred_RT, gt_RT, mesh_pts)
        adds = metrics_utils.cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
        cls_add_dis[cls_id].append(add.item())
        cls_adds_dis[cls_id].append(adds.item())
        cls_add_dis[0].append(add.item())
        cls_adds_dis[0].append(adds.item())

    return (cls_add_dis, cls_adds_dis, cls_kp_err)


def eval_one_frame_pose(item):
    pcld, mask, ctr_of, pred_kp_of, RTs, cls_ids, use_ctr, n_cls, \
        min_cnt, use_ctr_clus_flter, label, epoch, ibs, gt_kps, gt_ctrs, kp_type = item

    pred_cls_ids, pred_pose_lst, pred_kpc_lst = cal_frame_poses(
        pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter,
        gt_kps, gt_ctrs, kp_type=kp_type
    )

    cls_add_dis, cls_adds_dis, cls_kp_err = eval_metric(
        cls_ids, pred_pose_lst, pred_cls_ids, RTs, mask, label, gt_kps, gt_ctrs,
        pred_kpc_lst
    )
    return (cls_add_dis, cls_adds_dis, pred_cls_ids, pred_pose_lst, cls_kp_err)

# ###############################End YCB Evaluation###############################


# ###############################LineMOD Evaluation###############################


def cal_frame_poses_lm(
    pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter, obj_id,
    mesh_pts, mesh_kps, mesh_ctr, ms_radius,
    debug=False
):
    """
    Calculates pose parameters by 3D keypoints & center points voting to build
    the 3D-3D corresponding then use least-squares fitting to get the pose parameters.
    """
    n_kps, n_pts, _ = pred_kp_of.size()
    pred_ctr = pcld - ctr_of[0]
    pred_kp = pcld.view(1, n_pts, 3).repeat(n_kps, 1, 1) - pred_kp_of

    radius = ms_radius
    if use_ctr:
        cls_kps = torch.zeros(n_cls, n_kps + 1, 3).cuda()
    else:
        cls_kps = torch.zeros(n_cls, n_kps, 3).cuda()

    pred_pose_lst = []
    cls_id = 1
    cls_msk = mask == cls_id
    if cls_msk.sum() < 1:
        pred_pose_lst.append(np.identity(4)[:3, :])
        cls_kps_np = np.zeros((cls_kps.shape[1], 3))
    else:
        cls_voted_kps = pred_kp[:, cls_msk, :]
        ms = MeanShiftTorch(bandwidth=radius)
        ctr, ctr_labels = ms.fit(pred_ctr[cls_msk, :])
        if ctr_labels.sum() < 1:
            ctr_labels[0] = 1
        if use_ctr:
            cls_kps[cls_id, n_kps, :] = ctr

        if use_ctr_clus_flter:
            in_pred_kp = cls_voted_kps[:, ctr_labels, :]
        else:
            in_pred_kp = cls_voted_kps

        for ikp, kps3d in enumerate(in_pred_kp):
            cls_kps[cls_id, ikp, :], _ = ms.fit(kps3d)

        # visualize
        if debug:
            show_kp_img = np.zeros((480, 640, 3), np.uint8)
            kp_2ds = pcl_utils.project_p3d(
                cls_kps[cls_id].cpu().numpy(), K='linemod', scale_m=1000.0
            )
            print("cls_id = ", cls_id)
            print("kp3d:", cls_kps[cls_id])
            print("kp2d:", kp_2ds, "\n")
            color = (0, 0, 255)
            show_kp_img = visualization_utils.draw_p2ds(show_kp_img, kp_2ds, r=3, color=color)
            imshow("kp: cls_id=%d" % cls_id, show_kp_img)
            waitKey(0)

        if use_ctr:
            mesh_ctr = mesh_ctr.copy().reshape(1, 3)
            mesh_kps = np.concatenate((mesh_kps, mesh_ctr), axis=0)

        cls_kps_np = cls_kps[cls_id].squeeze().contiguous().cpu().numpy()

        pred_RT = metrics_utils.best_fit_transform(
            mesh_kps,
            cls_kps_np
        )
        pred_pose_lst.append(pred_RT)
    return pred_pose_lst, cls_kps_np


def eval_metric_lm(n_cls, cls_ids, pred_pose_lst, RTs, mask, label,
        obj_id, mesh_pts, mesh_kps, mesh_ctr):
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]

    pred_RT = pred_pose_lst[0]
    pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
    gt_RT = RTs[0]
    mesh_pts_cu = torch.from_numpy(mesh_pts.astype(np.float32)).cuda()
    add = metrics_utils.cal_add_cuda(pred_RT, gt_RT, mesh_pts_cu)
    adds = metrics_utils.cal_adds_cuda(pred_RT, gt_RT, mesh_pts_cu)
    # print("obj_id:", obj_id, add, adds)
    cls_add_dis[obj_id].append(add.item())
    cls_adds_dis[obj_id].append(adds.item())
    cls_add_dis[0].append(add.item())
    cls_adds_dis[0].append(adds.item())

    return (cls_add_dis, cls_adds_dis)
#evaluation scripts for evaluations
def eval_metric_lm_ref(n_cls, pred_pose_lst, RTs,obj_id, mesh_pts):
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]

    pred_RT = pred_pose_lst[0]
    pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
    gt_RT = RTs[0]
    mesh_pts_cu = torch.from_numpy(mesh_pts.astype(np.float32)).cuda()
    add = metrics_utils.cal_add_cuda(pred_RT, gt_RT, mesh_pts_cu)
    adds = metrics_utils.cal_adds_cuda(pred_RT, gt_RT, mesh_pts_cu)
    # print("obj_id:", obj_id, add, adds)
    cls_add_dis[obj_id].append(add.item())
    cls_adds_dis[obj_id].append(adds.item())
    cls_add_dis[0].append(add.item())
    cls_adds_dis[0].append(adds.item())

    return (cls_add_dis, cls_adds_dis)
def eval_one_frame_pose_lm_ref(item):
    RTs,pred_pose,n_cls, obj_id,mesh_pts = item
    cls_add_dis, cls_adds_dis = eval_metric_lm_ref(
        n_cls,pred_pose, RTs, obj_id, mesh_pts)
    return (cls_add_dis, cls_adds_dis)
def eval_one_frame_pose_lm(item):
    pcld, mask, ctr_of, pred_kp_of, RTs, cls_ids, use_ctr, n_cls, \
        min_cnt, use_ctr_clus_flter, label, epoch, ibs, obj_id, \
        mesh_pts, mesh_kps, mesh_ctr, ms_radius = item
    pred_pose_lst, _ = cal_frame_poses_lm(
        pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter,
        obj_id, mesh_pts, mesh_kps, mesh_ctr, ms_radius
    )

    cls_add_dis, cls_adds_dis = eval_metric_lm(
        n_cls, cls_ids, pred_pose_lst, RTs, mask, label,
        obj_id, mesh_pts, mesh_kps, mesh_ctr
    )
    return (cls_add_dis, cls_adds_dis)


def get_poses_from_data(
        cfg: config.Config, cuda_data: dict, end_points: dict, cls_id: int
) -> np.array:
    """
    Get pose estimate from data and predicted keypoints

    Args:
        cuda_data (dict): data dictionary
        end_points (dict): network output
        cls_id (int): class ID

    Returns:
        [np.array]: RT pose matrix 3x4
    """
    _, cls_rgbd = torch.max(end_points['pred_rgbd_segs'], 1)
    pred_poses, pred_kps = cal_frame_poses_lm(
        cuda_data['cld_rgb_nrm'][:, :3, :].permute(0, 2, 1).contiguous()[0],
        cls_rgbd[0],
        end_points['pred_ctr_ofs'][0],
        end_points['pred_kp_ofs'][0],
        True,
        cfg.n_objects,
        True,
        cls_id,
        cfg.dataset.get_pointxyz(),
        cfg.dataset.kps,
        cfg.dataset.ctr,
        cfg.ms_radius
    )

    return pred_poses[0], pred_kps

# ###############################End LineMOD Evaluation###############################

# ###############################Shared Evaluation Entry###############################


class TorchEval():
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_cls = 580
        self.cls_add_dis = [list() for i in range(self.n_cls)]
        self.cls_adds_dis = [list() for i in range(self.n_cls)]
        self.cls_add_s_dis = [list() for i in range(self.n_cls)]
        self.pred_kp_errs = [list() for i in range(self.n_cls)]
        self.pred_id2pose_lst = []
        self.sym_cls_ids = []

    def cal_auc(self):
        add_auc_lst = []
        adds_auc_lst = []
        add_s_auc_lst = []
        for cls_id in range(1, self.n_cls):
            if (cls_id) in self.cfg.sym_cls_ids:
                self.cls_add_s_dis[cls_id] = self.cls_adds_dis[cls_id]
            else:
                self.cls_add_s_dis[cls_id] = self.cls_add_dis[cls_id]
            self.cls_add_s_dis[0] += self.cls_add_s_dis[cls_id]
        for i in range(self.n_cls):
            add_auc = metrics_utils.cal_auc(self.cls_add_dis[i])
            adds_auc = metrics_utils.cal_auc(self.cls_adds_dis[i])
            add_s_auc = metrics_utils.cal_auc(self.cls_add_s_dis[i])
            add_auc_lst.append(add_auc)
            adds_auc_lst.append(adds_auc)
            add_s_auc_lst.append(add_s_auc)
            if i == 0:
                continue
            print(self.cfg.dataset.cls_lst[i - 1])
            print("***************add:\t", add_auc)
            print("***************adds:\t", adds_auc)
            print("***************add(-s):\t", add_s_auc)
        # kp errs:
        n_objs = sum([len(l) for l in self.pred_kp_errs])
        all_errs = 0.0
        for cls_id in range(1, self.n_cls):
            all_errs += sum(self.pred_kp_errs[cls_id])
        print("mean kps errs:", all_errs / n_objs)

        print("Average of all object:")
        print("***************add:\t", np.mean(add_auc_lst[1:]))
        print("***************adds:\t", np.mean(adds_auc_lst[1:]))
        print("***************add(-s):\t", np.mean(add_s_auc_lst[1:]))

        print("All object (following PoseCNN):")
        print("***************add:\t", add_auc_lst[0])
        print("***************adds:\t", adds_auc_lst[0])
        print("***************add(-s):\t", add_s_auc_lst[0])

        sv_info = dict(
            add_dis_lst=self.cls_add_dis,
            adds_dis_lst=self.cls_adds_dis,
            add_auc_lst=add_auc_lst,
            adds_auc_lst=adds_auc_lst,
            add_s_auc_lst=add_s_auc_lst,
            pred_kp_errs=self.pred_kp_errs,
        )
        sv_pth = os.path.join(
            self.cfg.log_eval_dir,
            'pvn3d_eval_cuda_{}_{}_{}.pkl'.format(
                adds_auc_lst[0], add_auc_lst[0], add_s_auc_lst[0]
            )
        )
        file_utils.save_pickle(sv_pth, sv_info)
        sv_pth = os.path.join(
            self.cfg.log_eval_dir,
            'pvn3d_eval_cuda_{}_{}_{}_id2pose.pkl'.format(
                adds_auc_lst[0], add_auc_lst[0], add_s_auc_lst[0]
            )
        )
        file_utils.save_pickle(sv_pth, self.pred_id2pose_lst)

    def cal_lm_add(self, obj_id, test_occ=False):
        add_auc_lst = []
        adds_auc_lst = []
        add_s_auc_lst = []
        cls_id = obj_id
        if (obj_id) in self.cfg.dataset.sym_cls_ids:
            self.cls_add_s_dis[cls_id] = self.cls_adds_dis[cls_id]
        else:
            self.cls_add_s_dis[cls_id] = self.cls_add_dis[cls_id]
        self.cls_add_s_dis[0] += self.cls_add_s_dis[cls_id]
        add_auc = metrics_utils.cal_auc(self.cls_add_dis[cls_id])
        adds_auc = metrics_utils.cal_auc(self.cls_adds_dis[cls_id])
        add_s_auc = metrics_utils.cal_auc(self.cls_add_s_dis[cls_id])
        add_auc_lst.append(add_auc)
        adds_auc_lst.append(adds_auc)
        add_s_auc_lst.append(add_s_auc)
        # d = self.cfg.dataset.r_lst[obj_id]['diameter'] / 1000.0 * 0.1
        d = self.cfg.dataset.radius * 2.0 * 0.1
        print("obj_id: ", obj_id, "0.1 diameter: ", d)
        add = np.mean(np.array(self.cls_add_dis[cls_id]) < d) * 100
        adds = np.mean(np.array(self.cls_adds_dis[cls_id]) < d) * 100

        cls_type = self.cfg.dataset.id_obj_dict[obj_id]
        print(obj_id, cls_type)
        print("***************add auc:\t", add_auc)
        print("***************adds auc:\t", adds_auc)
        print("***************add(-s) auc:\t", add_s_auc)
        print("***************add < 0.1 diameter:\t", add)
        print("***************adds < 0.1 diameter:\t", adds)

        sv_info = dict(
            add_dis_lst=self.cls_add_dis,
            adds_dis_lst=self.cls_adds_dis,
            add_auc_lst=add_auc_lst,
            adds_auc_lst=adds_auc_lst,
            add_s_auc_lst=add_s_auc_lst,
            add=add,
            adds=adds,
        )
        occ = "occlusion" if test_occ else ""
        sv_pth = os.path.join(
            self.cfg.log_eval_dir,
            'pvn3d_eval_cuda_{}_{}_{}_{}.pkl'.format(
                cls_type, occ, add, adds
            )
        )
        # pkl.dump(sv_info, open(sv_pth, 'wb'))
        file_utils.save_pickle(sv_pth, sv_info)

        return dict(
            add_auc=add_auc,
            adds_auc=adds_auc,
            add_s_auc=add_s_auc,
            add=add,
            adds=adds
        )

    def eval_pose_parallel(
        self, pclds, masks, pred_ctr_ofs, gt_ctr_ofs, labels, cnt,
        cls_ids, RTs, pred_kp_ofs, gt_kps, gt_ctrs, min_cnt=20, merge_clus=False,
        use_ctr_clus_flter=True, use_ctr=True, obj_id=0, kp_type='farthest',
        ds='ycb'
    ):
        bs, n_kps, n_pts, c = pred_kp_ofs.size()
        masks = masks.long()
        cls_ids = cls_ids.long()
        use_ctr_lst = [use_ctr for i in range(bs)]
        n_cls_lst = [self.n_cls for i in range(bs)]
        min_cnt_lst = [min_cnt for i in range(bs)]
        epoch_lst = [cnt * bs for i in range(bs)]
        bs_lst = [i for i in range(bs)]
        use_ctr_clus_flter_lst = [use_ctr_clus_flter for i in range(bs)]
        obj_id_lst = [obj_id for i in range(bs)]
        kp_type = [kp_type for i in range(bs)]
        mesh_pts = self.cfg.dataset.get_pointxyz()

        if ds == "ycb":
            data_gen = zip(
                pclds, masks, pred_ctr_ofs, pred_kp_ofs, RTs,
                cls_ids, use_ctr_lst, n_cls_lst, min_cnt_lst, use_ctr_clus_flter_lst,
                labels, epoch_lst, bs_lst, gt_kps, gt_ctrs, kp_type
            )
        else:
            data_gen = zip(
                pclds, masks, pred_ctr_ofs, pred_kp_ofs, RTs,
                cls_ids, use_ctr_lst, n_cls_lst, min_cnt_lst,
                use_ctr_clus_flter_lst, labels, epoch_lst,
                bs_lst,
                repeat(obj_id),
                repeat(mesh_pts),
                repeat(self.cfg.dataset.kps),
                repeat(self.cfg.dataset.ctr),
                repeat(self.cfg.ms_radius)
            )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=bs
        ) as executor:
            if ds == "ycb":
                eval_func = eval_one_frame_pose
            else:
                eval_func = eval_one_frame_pose_lm
            for res in executor.map(eval_func, data_gen):
                if ds == 'ycb':
                    cls_add_dis_lst, cls_adds_dis_lst, pred_cls_ids, pred_poses, pred_kp_errs = res
                    self.pred_id2pose_lst.append(
                        {cid: pose for cid, pose in zip(pred_cls_ids, pred_poses)}
                    )
                    self.pred_kp_errs = self.merge_lst(
                        self.pred_kp_errs, pred_kp_errs
                    )
                else:
                    cls_add_dis_lst, cls_adds_dis_lst = res
                self.cls_add_dis = self.merge_lst(
                    self.cls_add_dis, cls_add_dis_lst
                )
                self.cls_adds_dis = self.merge_lst(
                    self.cls_adds_dis, cls_adds_dis_lst
                )
    def eval_pose_ref(
        self,gt_pose,pred_pose,mesh_pts,min_cnt=20, merge_clus=False,
        use_ctr_clus_flter=True, use_ctr=True, obj_id=0, kp_type='farthest',
        ds='ycb'
    ):
        bs = gt_pose.shape[0]
        n_cls_lst = [self.n_cls for i in range(bs)]
        gt_pose_lst = [pose for pose in gt_pose]
        pred_pose_lst = [pose for pose in pred_pose]
        data_gen = zip(
            gt_pose_lst,
            pred_pose_lst,
            n_cls_lst,
            repeat(obj_id),
            repeat(mesh_pts)
        )
        cls_add_dis_lst, cls_adds_dis_lst = eval_one_frame_pose_lm_ref(data_gen) #RTs,pred_pose,n_cls, obj_id,mesh_pts
        self.cls_add_dis = self.merge_lst(
            self.cls_add_dis, cls_add_dis_lst
        )
        self.cls_adds_dis = self.merge_lst(
            self.cls_adds_dis, cls_adds_dis_lst
        )
    def merge_lst(self, targ, src):
        for i in range(len(targ)):
            targ[i] += src[i]
        return targ

# vim: ts=4 sw=4 sts=4 expandtab
