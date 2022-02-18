#!/usr/bin/env python3
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2021
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import os
import glob
from configs import (
    config,
    dataset_config
)
import numpy as np
import cv2
import tqdm
import pickle as pkl
import argparse
from matplotlib import pyplot as plt
from torchvision import transforms
from com_utils import (
    file_utils,
    img_utils,
    pcl_utils,
    visualization_utils,
    events
)
from termcolor import colored
import normalSpeed
from models.RandLA.helper_tool import DataProcessing

__all__ = ["Dataset"]


class Dataset():
    def __init__(self, cfg: config.Config, data_config: config.DataConfig, DEBUG: bool = False):

        self.DEBUG = DEBUG
        self.cfg = cfg
        self.data_config = data_config
        self.dsize = (
            int(self.cfg.dataset.w * self.cfg.dataset.scale_w),
            int(self.cfg.dataset.h * self.cfg.dataset.scale_h)
        )

        self.xymap = (
            np.array([[j for i in range(self.dsize[0])] for j in range(self.dsize[1])]),
            np.array([[i for i in range(self.dsize[0])] for j in range(self.dsize[1])])
        )

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224]
        )
        self.obj_dict = self.cfg.dataset.obj_id_dict

        self.cls_type = cfg.cls_type
        self.cls_id = self.obj_dict[self.cls_type]
        print("Init {} dataset.py with class {} ({})".format(
            "train" if self.data_config.is_training else "test",
            self.cls_type,
            self.cls_id)
        )
        self.cls_root = os.path.join(self.cfg.dataset.data_dir, "data/{:02d}/".format(self.cls_id))
        self.rng = np.random
        self.rng.seed(data_config.seed)

        # loading gt poses with keys in string format
        self.meta_lst = file_utils.read_yaml(self.cfg.dataset.gt_file)
        if self.cfg.dataset_name == "linemod":
            self.meta_lst = {"{:04d}".format(k): v for k, v in self.meta_lst.items()}
        real_img_pth = os.path.join(self.cls_root, data_config.index_filename)
        self.data_idx_lst = file_utils.read_lines(real_img_pth)
        if self.data_config.is_training:
            self.add_noise = True

            self.rnd_lst = []
            if self.cfg.use_render_data:
                rnd_img_ptn = os.path.join(
                    self.cfg.dataset.data_dir, "renders/{}/*.pkl".format(self.cls_type)
                )
                self.rnd_lst = glob.glob(rnd_img_ptn)
                print("render data length: ", len(self.rnd_lst))
                if len(self.rnd_lst) == 0:
                    warning = "Warning: "
                    warning += (
                        "Trainnig without rendered data can hurt model performance \n"
                    )
                    warning += "Please generate rendered data from "
                    warning += "ope_utils/dataset_tools/raster_triangle'.\n"
                    print(colored(warning, "red", attrs=["bold"]))

            self.fuse_lst = []
            if self.cfg.use_fuse_data:
                fuse_img_ptn = os.path.join(
                    self.cfg.dataset.data_dir, "fuse/{}/*.pkl".format(self.cls_type)
                )
                self.fuse_lst = glob.glob(fuse_img_ptn)
                print("fused data length: ", len(self.fuse_lst))
                if len(self.fuse_lst) == 0:
                    warning = "Warning: "
                    warning += "Trainnig without fused data will hurt model performance \n"
                    warning += "Please generate fused data from "
                    warning += "ope_utils/dataset_tools/raster_triangle'.\n"
                    print(colored(warning, "red", attrs=["bold"]))

            self.all_lst = self.data_idx_lst + self.rnd_lst + self.fuse_lst
            if self.data_config.subsampling > 1:
                self.all_lst = sorted(
                    self.rng.choice(self.all_lst, len(self.all_lst) // self.data_config.subsampling))

            self.minibatch_per_epoch = len(self.all_lst) // self.cfg.mini_batch_size
        else:
            self.add_noise = False
            self.all_lst = self.data_idx_lst
            if self.data_config.subsampling > 1:
                self.all_lst = self.all_lst[::self.data_config.subsampling]

        print("{} - {}: dataset_size: {}".format(
            "Train Data" if self.data_config.is_training else "Test Data",
            data_config.index_filename,
            len(self.all_lst)))

    def real_syn_gen(self, real_ratio=0.3):
        if len(self.rnd_lst + self.fuse_lst) == 0:
            real_ratio = 1.0
        if self.rng.rand() < real_ratio:  # real
            n_imgs = len(self.data_idx_lst)
            idx = self.rng.randint(0, n_imgs)
            pth = self.data_idx_lst[idx]
            return pth
        else:
            if len(self.fuse_lst) > 0 and len(self.rnd_lst) > 0:
                fuse_ratio = 0.4
            elif len(self.fuse_lst) == 0:
                fuse_ratio = 0.0
            else:
                fuse_ratio = 1.0
            if self.rng.rand() < fuse_ratio:
                idx = self.rng.randint(0, len(self.fuse_lst))
                pth = self.fuse_lst[idx]
            else:
                idx = self.rng.randint(0, len(self.rnd_lst))
                pth = self.rnd_lst[idx]
            return pth

    def real_gen(self):
        n = len(self.data_idx_lst)
        idx = self.rng.randint(0, n)
        item = self.data_idx_lst[idx]
        return item

    def add_real_back(self, rgb, labels, dpt, dpt_msk):
        real_item = self.real_gen()

        real_dpt = file_utils.read_image(
            self.cfg.dataset.depth_pattern.format(real_item),
            self.dsize,
            cv2.INTER_NEAREST
        )
        bk_label = file_utils.read_image(
            self.cfg.dataset.mask_pattern.format(real_item),
            self.dsize,
            cv2.INTER_NEAREST
        )
        bk_label = (bk_label < 255).astype(rgb.dtype)
        if len(bk_label.shape) > 2:
            bk_label = bk_label[:, :, 0]
        back = file_utils.read_image(
            self.cfg.dataset.img_pattern.format(real_item),
            self.dsize,
            cv2.INTER_NEAREST,
            self.trancolor if self.add_noise else None
        )
        dpt_back = real_dpt.astype(np.float32) * bk_label.astype(np.float32)

        if self.rng.rand() < 0.6:
            msk_back = (labels <= 0).astype(rgb.dtype)
            msk_back = msk_back[:, :, None]
            rgb = rgb * (msk_back == 0).astype(rgb.dtype) + back * msk_back

        dpt = dpt * (dpt_msk > 0).astype(dpt.dtype) + dpt_back * (dpt_msk <= 0).astype(
            dpt.dtype
        )
        return rgb, dpt

    def get_item(self, item_name):
        if "pkl" in item_name:
            data = pkl.load(open(item_name, "rb"))
            dpt_mm = data["depth"] * 1000.0
            rgb = data["rgb"]
            labels = data["mask"]
            K = data["K"]
            RT = data["RT"]
            rnd_typ = data["rnd_typ"]
            if rnd_typ == "fuse":
                labels = (labels == self.cls_id).astype("uint8")
            else:
                labels = (labels > 0).astype("uint8")

            rgb = cv2.resize(rgb, dsize=self.dsize, interpolation=cv2.INTER_CUBIC)
            labels = cv2.resize(labels, dsize=self.dsize, interpolation=cv2.INTER_NEAREST)
            dpt_mm = cv2.resize(dpt_mm, dsize=self.dsize, interpolation=cv2.INTER_NEAREST)
        else:
            dpt_mm = file_utils.read_image(
                os.path.join(self.cfg.dataset.depth_pattern.format(item_name)),
                self.dsize,
                cv2.INTER_NEAREST
            )
            rgb = file_utils.read_image(
                os.path.join(self.cfg.dataset.img_pattern.format(item_name)),
                self.dsize,
                cv2.INTER_NEAREST,
                self.trancolor if self.add_noise else None,
                "RGB"
            )[:, :, :3]
            labels = file_utils.read_image(
                os.path.join(self.cfg.dataset.mask_pattern.format(item_name)),
                self.dsize,
                cv2.INTER_NEAREST
            )
            labels = (labels > 0).astype("uint8")

            meta = self.meta_lst[item_name]
            if self.cls_id == 2:
                for i in range(0, len(meta)):
                    if meta[i]["obj_id"] == 2:
                        meta = meta[i]
                        break
            elif self.cfg.dataset_name != "ml2_sensim":
                meta = meta[0]
            if "cam_R_m2c" in meta:
                R = np.resize(np.array(meta["cam_R_m2c"]), (3, 3))
                T = np.array(meta["cam_t_m2c"]) / self.cfg.dataset.scale_m
                RT = np.concatenate((R, T[:, None]), axis=1)
            else:
                RT = np.array(meta["obj_pose"])[:3, :]
            rnd_typ = "real"
            K = self.cfg.dataset.K
        if len(labels.shape) > 2:
            labels = labels[:, :, 0]
        rgb_labels = labels.copy()
        if self.add_noise and rnd_typ != "real":
            if rnd_typ == "render" or self.rng.rand() < 0.8:
                rgb = img_utils.rgb_add_noise(rgb, self.rng)
                rgb_labels = labels.copy()
                msk_dp = dpt_mm > 1e-6
                rgb, dpt_mm = self.add_real_back(rgb, rgb_labels, dpt_mm, msk_dp)
                if self.rng.rand() > 0.8:
                    rgb = img_utils.rgb_add_noise(rgb, self.rng)

        dpt_mm = dpt_mm.copy().astype(np.uint16)
        nrm_map = normalSpeed.depth_normal(dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False)

        dpt_m = dpt_mm.astype(np.float32) / self.cfg.dataset.scale_depth_m
        dpt_xyz = pcl_utils.dpt2cld(dpt_m, K, 1.0, self.xymap)
        dpt_xyz[np.isnan(dpt_xyz)] = 0.0
        dpt_xyz[np.isinf(dpt_xyz)] = 0.0

        msk_dp = (dpt_mm > 1e-6)&(rgb_labels==1)
        choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
        if len(choose) < 400:
            return None
        choose_2 = np.array([i for i in range(len(choose))])
        if len(choose_2) < 400:
            return None
        if len(choose_2) > self.cfg.n_sample_points:
            c_mask = np.zeros(len(choose_2), dtype=int)
            c_mask[: self.cfg.n_sample_points] = 1
            np.random.shuffle(c_mask)
            choose_2 = choose_2[c_mask.nonzero()]
        else:
            choose_2 = np.pad(
                choose_2, (0, self.cfg.n_sample_points - len(choose_2)), "wrap"
            )
        choose = np.array(choose)[choose_2]

        sf_idx = np.arange(choose.shape[0])
        np.random.shuffle(sf_idx)
        choose = choose[sf_idx]

        cld = dpt_xyz.reshape(-1, 3)[choose, :]
        rgb_pt = rgb.reshape(-1, 3)[choose, :].astype(np.float32)
        nrm_pt = nrm_map[:, :, :3].reshape(-1, 3)[choose, :]
        labels_pt = labels.flatten()[choose]
        choose = np.array([choose])
        cld_rgb_nrm = np.concatenate((cld, rgb_pt, nrm_pt), axis=1).transpose(1, 0)

        (
            RTs,
            kp3ds,
            ctr3ds,
            cls_ids,
            kp_targ_ofst,
            ctr_targ_ofst,
        ) = self.get_pose_gt_info(cld, labels_pt, RT)

        h, w = rgb_labels.shape
        dpt_6c = np.concatenate((dpt_xyz, nrm_map[:, :, :3]), axis=2).transpose(2, 0, 1)
        rgb = np.transpose(rgb, (2, 0, 1))  # hwc2chw

        xyz_lst = [dpt_xyz.transpose(2, 0, 1)]  # c, h, w
        msk_lst = [dpt_xyz[2, :, :] > 1e-8]

        ctr2ds = pcl_utils.project_p3d(ctr3ds, K=K)[0]
        if ctr2ds[0] < w and ctr2ds[1] < h and ctr2ds[0] >= 0 and ctr2ds[1] >= 0:
            ctr_in_img = True
        else:
            ctr_in_img = False

        for i in range(3):
            scale = pow(2, i + 1)
            nh, nw = h // pow(2, i + 1), w // pow(2, i + 1)
            ys, xs = np.mgrid[:nh, :nw]
            xyz_lst.append(xyz_lst[0][:, ys * scale, xs * scale])
            msk_lst.append(xyz_lst[-1][2, :, :] > 1e-8)
        sr2dptxyz = {
            pow(2, ii): item.reshape(3, -1).transpose(1, 0)
            for ii, item in enumerate(xyz_lst)
        }

        rgb_ds_sr = [4, 8, 8, 8]
        n_ds_layers = 4
        pcld_sub_s_r = [4, 4, 4, 4]
        inputs = {}
        # DownSample stage
        for i in range(n_ds_layers):
            nei_idx = (
                DataProcessing.knn_search(cld[None, ...], cld[None, ...], 16)
                .astype(np.int32)
                .squeeze(0)
            )
            sub_pts = cld[: cld.shape[0] // pcld_sub_s_r[i], :]
            pool_i = nei_idx[: cld.shape[0] // pcld_sub_s_r[i], :]
            up_i = (
                DataProcessing.knn_search(sub_pts[None, ...], cld[None, ...], 1)
                .astype(np.int32)
                .squeeze(0)
            )
            inputs["cld_xyz%d" % i] = cld.astype(np.float32).copy()
            inputs["cld_nei_idx%d" % i] = nei_idx.astype(np.int32).copy()
            inputs["cld_sub_idx%d" % i] = pool_i.astype(np.int32).copy()
            inputs["cld_interp_idx%d" % i] = up_i.astype(np.int32).copy()
            nei_r2p = (
                DataProcessing.knn_search(
                    sr2dptxyz[rgb_ds_sr[i]][None, ...], sub_pts[None, ...], 16
                )
                .astype(np.int32)
                .squeeze(0)
            )
            inputs["r2p_ds_nei_idx%d" % i] = nei_r2p.copy()
            nei_p2r = (
                DataProcessing.knn_search(sub_pts[None, ...], sr2dptxyz[rgb_ds_sr[i]][None, ...], 1)
                .astype(np.int32)
                .squeeze(0)
            )
            inputs["p2r_ds_nei_idx%d" % i] = nei_p2r.copy()
            cld = sub_pts

        n_up_layers = 3
        rgb_up_sr = [4, 2, 2]
        for i in range(n_up_layers):
            r2p_nei = (
                DataProcessing.knn_search(
                    sr2dptxyz[rgb_up_sr[i]][None, ...],
                    inputs["cld_xyz%d" % (n_ds_layers - i - 1)][None, ...],
                    16,
                )
                .astype(np.int32)
                .squeeze(0)
            )
            inputs["r2p_up_nei_idx%d" % i] = r2p_nei.copy()
            p2r_nei = (
                DataProcessing.knn_search(
                    inputs["cld_xyz%d" % (n_ds_layers - i - 1)][None, ...],
                    sr2dptxyz[rgb_up_sr[i]][None, ...],
                    1,
                )
                .astype(np.int32)
                .squeeze(0)
            )
            inputs["p2r_up_nei_idx%d" % i] = p2r_nei.copy()

        item_dict = dict(
            rgb=rgb.astype(np.uint8),  # [c, h, w]
            cld_rgb_nrm=cld_rgb_nrm.astype(np.float32),  # [9, npts]
            choose=choose.astype(np.int32),  # [1, npts]
            labels=labels_pt.astype(np.int32),  # [npts]
            rgb_labels=rgb_labels.astype(np.int32),  # [h, w]
            dpt_map_m=dpt_m.astype(np.float32),  # [h, w]
            RTs=RTs.astype(np.float32),
            kp_targ_ofst=kp_targ_ofst.astype(np.float32),
            ctr_targ_ofst=ctr_targ_ofst.astype(np.float32),
            cls_ids=cls_ids.astype(np.int32),
            ctr_3ds=ctr3ds.astype(np.float32),
            kp_3ds=kp3ds.astype(np.float32),
            ctr_in_img=ctr_in_img
        )
        item_dict.update(inputs)
        if self.DEBUG:
            extra_d = dict(
                dpt_xyz_nrm=dpt_6c.astype(np.float32),  # [6, h, w]
                cam_scale=np.array([self.cfg.dataset.scale_m]).astype(np.float32),
                K=K.astype(np.float32),
            )
            item_dict.update(extra_d)
            item_dict["normal_map"] = nrm_map[:, :, :3].astype(np.float32)
        return item_dict

    def get_pose_gt_info(self, cld, labels, RT):
        RTs = np.zeros((self.cfg.n_objects, 3, 4))
        kp3ds = np.zeros((self.cfg.n_objects, self.cfg.n_keypoints, 3))
        ctr3ds = np.zeros((self.cfg.n_objects, 3))
        cls_ids = np.zeros((self.cfg.n_objects, 1))
        kp_targ_ofst = np.zeros(
            (self.cfg.n_sample_points, self.cfg.n_keypoints, 3)
        )
        ctr_targ_ofst = np.zeros((self.cfg.n_sample_points, 3))
        for i, cls_id in enumerate([1]):
            RTs[i] = RT
            r = RT[:, :3]
            t = RT[:, 3]

            ctr = self.cfg.dataset.ctr[:, None]
            ctr = np.dot(ctr.T, r.T) + t
            ctr3ds[i, :] = ctr[0]
            msk_idx = np.where(labels == cls_id)[0]

            target_offset = np.array(np.add(cld, -1.0 * ctr3ds[i, :]))
            ctr_targ_ofst[msk_idx, :] = target_offset[msk_idx, :]
            cls_ids[i, :] = np.array([1])

            self.minibatch_per_epoch = len(self.all_lst) // self.cfg.mini_batch_size
            kps = self.cfg.dataset.kps.copy()
            kps = np.dot(kps, r.T) + t
            kp3ds[i] = kps

            target = []
            for kp in kps:
                target.append(np.add(cld, -1.0 * kp))
            target_offset = np.array(target).transpose(1, 0, 2)  # [npts, nkps, c]
            kp_targ_ofst[msk_idx, :, :] = target_offset[msk_idx, :, :]

        return RTs, kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst

    def __len__(self):
        return len(self.all_lst)

    def __getitem__(self, idx):
        if self.data_config.is_training:
            item_name = self.real_syn_gen()
            data = self.get_item(item_name)
            while data is None:
                item_name = self.real_syn_gen()
                data = self.get_item(item_name)
            return data
        else:
            item_name = self.all_lst[idx]
            return self.get_item(item_name)


def test_dataset(ds):
    idx = 0
    while True:
        datum = ds.__getitem__(idx)
        idx += 1
        K = datum["K"]
        cam_scale = datum["cam_scale"]
        rgb = datum["rgb"].transpose(1, 2, 0)[..., ::-1].copy()
        for i in range(22):
            pcld = datum["cld_rgb_nrm"][:3, :].transpose(1, 0).copy()
            kp3d = datum["kp_3ds"][i]
            if kp3d.sum() < 1e-6:
                break
            kp_2ds = pcl_utils.project_p3d(kp3d, K, cam_scale)
            rgb = visualization_utils.draw_p2ds(
                rgb,
                kp_2ds,
                3,
                visualization_utils.get_label_color(datum["cls_ids"][i][0], mode=1),
            )
            ctr3d = datum["ctr_3ds"][i]
            ctr_2ds = pcl_utils.project_p3d(ctr3d[None, :], K, cam_scale)
            rgb = visualization_utils.draw_p2ds(rgb, ctr_2ds, 4, (0, 0, 255))
        plt.imshow(rgb)
        plt.show()


def visualize_dataset(dataset: Dataset, cfg: config.Config, sv_pth: str = None):
    """Visualize images with rendered groundtruth of dataset.

    Args:
        dataset (Dataset)
        cfg (config.Config)
        sv_pth (str, optional): If set, full filepath to store images and video.
            Defaults to None.
    """

    print("Start dataset visualization ...")
    if sv_pth:
        file_utils.ensure_dir(os.path.dirname(sv_pth))
        video_writer = events.VideoWriter(
            sv_pth,
            (cfg.width, cfg.height)
        )

    model_3ds = cfg.dataset.get_pointxyz()

    for idx, data_item in enumerate(tqdm.tqdm(dataset, total=len(dataset), desc="vis")):
        if idx >= len(dataset):
            break
        K = cfg.dataset.K
        RT = data_item["RTs"][0]
        cam_scale = cfg.dataset.scale_m
        rgb = data_item["rgb"].transpose(1, 2, 0)[..., ::-1].copy()
        kp3d = data_item["kp_3ds"][0]
        if kp3d.sum() < 1e-6:
            break

        rgb = visualization_utils.draw_p3ds(
            rgb,
            model_3ds,
            K=K,
            RT=RT,
            thickness=1,
            color=(0, 0, 255)
        )
        ctr3d = data_item["ctr_3ds"][0]
        rgb = visualization_utils.draw_p3ds(
            rgb,
            ctr3d,
            K=K,
            RT=None,
            thickness=10,
            color=(255, 0, 255)
        )

        rgb = visualization_utils.draw_coordinate_system(
            img=rgb,
            ctr3d=None,
            K=K,
            RT=RT,
            len_m=cfg.dataset.radius,
            color="rgb"
        )
        rgb = visualization_utils.draw_bbox3d(rgb, cfg.dataset.corners, K, RT, (128, 128, 128))

        kp_2ds = pcl_utils.project_p3d(kp3d, K, cam_scale)
        rgb = visualization_utils.draw_p2ds(
            rgb,
            kp_2ds,
            6,
            (0, 255, 0),
        )
        if sv_pth:
            video_writer.write(rgb)
        else:
            plt.imshow("{:04d}-vis".format(idx), rgb)
            plt.show()

    if sv_pth:
        video_writer.finish()


def parse_arguments(dataset_name):
    parser = argparse.ArgumentParser("Test and analyse dataset")
    parser.add_argument(
        "--cls", "-cls",
        type=str,
        choices=list(dataset_config.obj_id_dict[dataset_name].keys()),
        help="Class name"
    )
    parser.add_argument(
        "--type", "-t",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset Type"
    )
    parser.add_argument(
        "--sv_pth", type=str, default=None,
        help="Path to save the visualization resutls"
    )
    parser.add_argument(
        "--vis", action="store_true", help="Visualize Dataset")
    parser.add_argument(
        "--h", type=int, default=600, help="Height of rendered images."
    )
    parser.add_argument(
        "--w", type=int, default=800, help="Width of rendered images."
    )
    parser.add_argument(
        "--subsampling", "-sub", type=int, default=10, help="Subsampling of items in dataset."
    )
    parser.add_argument(
        "--depth_src", type=str, default="user_rgb", help="Depth sensor source"
    )
    parser.add_argument(
        "--img_src", type=str, default="user_rgb", help="Image sensor source"
    )
    args = parser.parse_args()

    if args.sv_pth:
        args.sv_pth = os.path.join(
            args.sv_pth, "{}-{}-{}-vis.avi".format(dataset_name, args.cls, args.type)
        )

    return args


def eval_datasets(dataset_name):

    args = parse_arguments(dataset_name)

    cfg = config.create_config({
        "wandb": {"enable": False},
        "name": "{}-{}".format(dataset_name, args.type),
        "dataset_name": dataset_name,
        "cls_type": args.cls,
        "description": "{}-{}".format(dataset_name, args.type),
        "width": args.w,
        "height": args.h,
        "depth_src": args.depth_src,
        "img_src": args.img_src
    })

    cfg_ds = cfg.test_data if args.type == "test" else cfg.train_data
    cfg_ds.subsampling = args.subsampling

    dataset = Dataset(cfg, cfg_ds, DEBUG=True)

    if args.vis:
        visualize_dataset(dataset, cfg, sv_pth=args.sv_pth)
    else:
        test_dataset(dataset)
