import os
import numpy as np
import random
import torch
import sys
from com_utils import file_utils, mesh_utils

datasets = ["linemod", "ml2_sensim"]

obj_id_dict = {
    "linemod": {
        "ape": 1,
        "benchvise": 2,
        "cam": 4,
        "can": 5,
        "cat": 6,
        "driller": 8,
        "duck": 9,
        "eggbox": 10,
        "glue": 11,
        "holepuncher": 12,
        "iron": 13,
        "lamp": 14,
        "phone": 15,
    },
    "ml2_sensim": {
        "robot1": 579,
        "robot2": 19
    }
}

scales_m = {"linemod": 1000.0, "ml2_sensim": 1.0}
scales_depth_m = {"linemod": 1000.0, "ml2_sensim": 1000.0}

intrinsic_matrix = {
    "linemod": np.array([[572.4114, 0.,         325.2611],
                        [0.,        573.57043,  242.04899],
                        [0.,        0.,         1.]]),
    "blender": np.array([[700.,     0.,     320.],
                         [0.,       700.,   240.],
                         [0.,       0.,     1.]]),
    "pascal": np.asarray([[-3000.0, 0.0,    0.0],
                         [0.0,      3000.0, 0.0],
                         [0.0,      0.0,    1.0]]),
    "ycb_K1": np.array([[1066.778, 0.        , 312.9869],
                        [0.      , 1067.487  , 241.3109],
                        [0.      , 0.        , 1.0]], np.float32),
    "ycb_K2": np.array([[1077.836, 0.        , 323.7872],
                        [0.      , 1078.189  , 279.6921],
                        [0.      , 0.        , 1.0]], np.float32),
    "ml2_sensim_rgb": np.array([[1346.03728074, 0., 1017.5],
                                [0., 1346.03728074,  765.5],
                                [0.,            0.,    1.0]], np.float32)
}

image_sizes = {"linemod": (480, 640), "ml2_sensim_rgb": (1532, 2036)}


class DatasetConfig():
    """
    Info class for dataset specific settings for rendering
    """
    def __init__(self, cfg) -> None:
        if cfg.dataset_name not in datasets:
            raise ValueError(
                "Dataset '{}' not in list of available datasets [{}]".format(
                    cfg.dataset_name, ", ".join(datasets)
                )
            )

        self.obj_id_dict = obj_id_dict[cfg.dataset_name]
        self.id_obj_dict = dict(zip(self.obj_id_dict.values(), self.obj_id_dict.keys()))
        self.scale_m = scales_m[cfg.dataset_name]
        self.scale_depth_m = scales_depth_m[cfg.dataset_name]
        self.pointxyz = None
        self.ptsxyz_cuda = None
        self.cls_id = self.obj_id_dict[cfg.cls_type]
        if cfg.dataset_name == "linemod":
            self.data_dir = cfg.dataset_base_dir
            self.K = self.K_depth = intrinsic_matrix[cfg.dataset_name]
            self.h = self.h_depth = image_sizes[cfg.dataset_name][0]
            self.w = self.w_depth = image_sizes[cfg.dataset_name][1]

            self.kp_orbfps_dir = os.path.join(cfg.dataset_base_dir, "kps_orb9_fps")
            self.kp_orbfps_ptn = os.path.join(self.kp_orbfps_dir, '%s_%d_kps.txt')
            self.fps_kps_dir = os.path.abspath(
                os.path.join(cfg.dataset_base_dir, "lm_obj_kps")
            )
            self.sym_cls_ids = [10, 11]

            self.r_lst = file_utils.read_yaml(
                os.path.join(cfg.dataset_base_dir, "dataset_config/models_info.yml")
            )
            self.cls_lst = list(self.obj_id_dict.values())

            self.corners = file_utils.read_np(os.path.join(
                self.kp_orbfps_dir, '{}_corners.txt'.format(cfg.cls_type),
            ))

            self.ctr = self.corners.mean(0)
            self.radius = mesh_utils.get_r(self.corners)

            if cfg.use_orbfps:
                kps_pth = os.path.join(
                    cfg.dataset_base_dir, "kps_orb9_fps", "{}_{}_kps.txt".format(
                        cfg.cls_type, cfg.n_keypoints
                    )
                )
            else:
                kps_pth = os.path.join(
                    cfg.dataset_base_dir, "lm_obj_kps", "{}_{}_kps.txt".format(
                        cfg.cls_type, cfg.n_keypoints
                    )
                )

            instance_folder = "{:02d}".format(self.cls_id)

            self.ptxyz_pth = os.path.join(
                self.data_dir, "models", "obj_{:02d}.ply".format(self.cls_id)
            )

            self.img_pattern = os.path.join(
                self.data_dir, "data", instance_folder, "rgb", "{}.png"
            )
            self.depth_pattern = os.path.join(
                self.data_dir, "data", instance_folder, "depth", "{}.png"
            )
            self.mask_pattern = os.path.join(
                self.data_dir, "data", instance_folder, "mask", "{}.png"
            )
            self.gt_file = os.path.join(
                self.data_dir, "data", instance_folder, "gt.yaml"
            )
            self.depth2img = np.eye(4)

        elif cfg.dataset_name == "ml2_sensim":
            self.data_dir = os.path.join(cfg.dataset_base_dir, "ml2_sensim")

            self.sym_cls_ids = []

            self.ctr = file_utils.read_np(
                os.path.join(
                    self.data_dir, "models", str(self.cls_id),
                    "{}_center.txt".format(self.cls_id)
                )
            )
            self.corners = file_utils.read_np(
                os.path.join(
                    self.data_dir, "models", str(self.cls_id),
                    "{}_corners.txt".format(self.cls_id)
                )
            )
            self.radius = mesh_utils.get_r(self.corners)

            if cfg.use_orbfps:
                kps_pth = os.path.join(
                    self.data_dir, "models", str(self.cls_id),
                    "{}_ORB_{}_fps.txt".format(self.cls_id, cfg.n_keypoints)
                )
            else:
                kps_pth = os.path.join(
                    self.data_dir, "models", str(self.cls_id),
                    "{}_{}_fps.txt".format(self.cls_id, cfg.n_keypoints)
                )

            self.ptxyz_pth = os.path.join(
                self.data_dir, "models", str(self.cls_id),
                "{}.ply".format(self.cls_id)
            )

            # get sensor-related paths and calibration data
            self.img_pattern = os.path.join(
                self.data_dir, "data", str(self.cls_id), cfg.img_src,
                "img_{}.png"
            )
            self.depth_pattern = os.path.join(
                self.data_dir, "data", str(self.cls_id), cfg.depth_src,
                "depth_{}.png"
            )
            self.mask_pattern = os.path.join(
                self.data_dir, "data", str(self.cls_id), cfg.img_src,
                "mask_{}.png"
            )
            self.gt_file = os.path.join(
                self.data_dir, "data", str(self.cls_id), "gt_{}.yaml".format(cfg.img_src)
            )
            calib_data = file_utils.read_yaml(
                os.path.join(self.data_dir, "data", str(self.cls_id), "cam_params.yaml")
            )
            self.K = np.array(calib_data[cfg.img_src]["intrinsics"])
            self.h = calib_data[cfg.img_src]["height"]
            self.w = calib_data[cfg.img_src]["width"]
            self.K_depth = np.array(calib_data[cfg.depth_src]["intrinsics"])
            self.h_depth = calib_data[cfg.depth_src]["height"]
            self.w_depth = calib_data[cfg.depth_src]["width"]
            self.depth2img = np.matmul(
                np.linalg.inv(calib_data[cfg.img_src]["extrinsics"]),
                calib_data[cfg.depth_src]["extrinsics"]
            )

        # change image input resolution to a base of 8 (needed in FFB6D)
        cfg.width = (cfg.width if cfg.width else self.w) // 8 * 8
        cfg.height  = (cfg.height if cfg.height else self.h) // 8 * 8

        self.scale_w = cfg.width / self.w
        self.scale_h = cfg.height / self.h

        if self.scale_w != 1.0 or self.scale_h != 1.0:
            print("=> Resizing image input resolution to {}x{}".format(cfg.width, cfg.height))

        self.K = np.matmul(np.diag([self.scale_w, self.scale_h, 1.0]), self.K)
        self.w = self.w * self.scale_w
        self.h = self.h * self.scale_h

        if cfg.dataset_name == "ml2_sensim" and cfg.depth_src == cfg.img_src:
            self.K_depth = self.K
            self.w_depth = self.w
            self.h_depth = self.h

        self.kps = file_utils.read_np(kps_pth)

    def get_pointxyz(self, all: bool = False):
        if self.pointxyz is not None:
            return self.pointxyz
        pointxyz = mesh_utils.get_p3ds_from_ply(self.ptxyz_pth, self.scale_m)
        if all:
            return pointxyz
        dellist = [j for j in range(0, len(pointxyz))]
        dellist = random.sample(dellist, len(pointxyz) - 2000)
        self.pointxyz = np.delete(pointxyz, dellist, axis=0)
        return self.pointxyz

    def get_pointxyz_cuda(self):
        if self.ptsxyz_cuda is not None:
            return self.ptsxyz_cuda

        self.ptsxyz_cuda = torch.from_numpy(self.get_pointxyz().astype(np.float32)).cuda()
        return self.ptsxyz_cuda.clone()
