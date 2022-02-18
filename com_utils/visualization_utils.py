#!/usr/bin/env python

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

# viszualization package to create rendering images form meshes and
# inpaing mesh rendeings with mesh info (e.g. keypoints, bbox, coordinate system)
#
# Usage:
#   python visualization.py --obj-pth example_mesh/ape.ply \
#       --mesh-info-prefix ape_info/ape --vis --scale2m 1000.
#
#   python visualization.py \
#       --obj-pth /mnt/nfs/or2d-data-usc1f/object_pose_estimation_datasets/SenSim_OPE_v1/models/SensimRobot/SensimRobot-centered.ply \
#       --mesh-info-prefix sensim_robot --vis
#

import os
import numpy as np
import cv2
import ctypes as ct
from typing import Union, List, Tuple
import normalSpeed

from com_utils.ip_basic.ip_basic import depth_map_utils_ycb
from com_utils import (
    pcl_utils,
    pose_utils,
    pvn3d_eval_utils_kpls
)

'''SO_P = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dataset_tools", "raster_triangle", "rastertriangle_so.so"
)
#RENDERER = np.ctypeslib.load_library(SO_P, '.')'''


color_dict = {
    "r": (255, 0, 0),
    "g": (0, 255, 0),
    "b": (0, 0, 255),
    "m": (255, 0, 255),
    "y": (255, 255, 0),
    "c": (0, 255, 255),
    "w": (255, 255, 255)
}


def get_label_color(cls_id, n_obj=22, mode=0):
    if mode == 0:
        cls_color = [
            255, 255, 255,  # 0
            180, 105, 255,   # 194, 194, 0,    # 1 # 194, 194, 0
            0, 255, 0,      # 2
            0, 0, 255,      # 3
            0, 255, 255,    # 4
            255, 0, 255,    # 5
            180, 105, 255,  # 128, 128, 0,    # 6
            128, 0, 0,      # 7
            0, 128, 0,      # 8
            0, 165, 255,    # 0, 0, 128,      # 9
            128, 128, 0,    # 10
            0, 0, 255,      # 11
            255, 0, 0,      # 12
            0, 194, 0,      # 13
            0, 194, 0,      # 14
            255, 255, 0,    # 15 # 0, 194, 194
            64, 64, 0,      # 16
            64, 0, 64,      # 17
            185, 218, 255,  # 0, 0, 64,       # 18
            0, 0, 255,      # 19
            0, 64, 0,       # 20
            0, 0, 192       # 21
        ]
        cls_color = np.array(cls_color).reshape(-1, 3)
        color = cls_color[cls_id]
        bgr = (int(color[0]), int(color[1]), int(color[2]))
    elif mode == 1:
        cls_color = [
            255, 255, 255,  # 0
            0, 127, 255,    # 180, 105, 255,   # 194, 194, 0,    # 1 # 194, 194, 0
            0, 255, 0,      # 2
            255, 0, 0,      # 3
            180, 105, 255, # 0, 255, 255,    # 4
            255, 0, 255,    # 5
            180, 105, 255,  # 128, 128, 0,    # 6
            128, 0, 0,      # 7
            0, 128, 0,      # 8
            185, 218, 255,# 0, 0, 255, # 0, 165, 255,    # 0, 0, 128,      # 9
            128, 128, 0,    # 10
            0, 0, 255,      # 11
            255, 0, 0,      # 12
            0, 194, 0,      # 13
            0, 194, 0,      # 14
            255, 255, 0,    # 15 # 0, 194, 194
            0, 0, 255, # 64, 64, 0,      # 16
            64, 0, 64,      # 17
            185, 218, 255,  # 0, 0, 64,       # 18
            0, 0, 255,      # 19
            0, 0, 255, # 0, 64, 0,       # 20
            0, 255, 255,# 0, 0, 192       # 21
        ]
        cls_color = np.array(cls_color).reshape(-1, 3)
        color = cls_color[cls_id]
        bgr = (int(color[0]), int(color[1]), int(color[2]))
    else:
        mul_col = 255 * 255 * 255 // n_obj * cls_id
        r, g, b = mul_col // 255 // 255, (mul_col // 255) % 255, mul_col % 255
        bgr = (int(r), int(g) , int(b))
    return bgr


def get_color_tuples(color_str):
    color_list = []
    for c in color_str:
        color_list.append(color_dict[c])
    return color_list


def draw_p2ds_lb(img, p2ds, label, r=1, color=(255, 0, 0)):
    h, w = img.shape[0], img.shape[1]
    for pt_2d, lb in zip(p2ds, label):
        pt_2d[0] = np.clip(pt_2d[0], 0, w)
        pt_2d[1] = np.clip(pt_2d[1], 0, h)
        color = get_label_color(lb)
        img = cv2.circle(
            img, (pt_2d[0], pt_2d[1]), r, color, -1
        )
    return img


def draw_coordinate_system(
    img: np.array,
    ctr3d: np.array,
    K: np.array,
    RT: np.array,
    len_m: int,
    color: str = "rgb",
    thickness: int = 2
) -> np.array:
    """Draws the object coodinate system onto an image

    Color is fixed with R/G/B for x/y/z axis

    Args:
        img (np.array): image array WxHx3
        ctr3d (np.array): object center in 3D 1x3
        K (np.array): camera projection matrix 3x3
        RT (np.array): object pose matrix 3x4
        len_m (int): length of coordinate axis in meter
        color (str): color coding (Options: "rgb", "gray")
        thickness (int, optional): line thickness in pixel. Defaults to 2.

    Returns:
        np.array: image array WxHx3 with coordinate system overlay
    """

    R, T = RT[:3, :3], RT[:3, 3]

    if ctr3d is None:
        cs_3ds = np.vstack((
            np.array([0.0, 0.0, 0.0]),
            np.array([len_m, 0.0, 0.0]),
            np.array([0.0, len_m, 0.0]),
            np.array([0.0, 0.0, len_m])
        ))
    else:
        cs_3ds = np.vstack((
            ctr3d,
            ctr3d + np.array([len_m, 0.0, 0.0]),
            ctr3d + np.array([0.0, len_m, 0.0]),
            ctr3d + np.array([0.0, 0.0, len_m])
        ))
    cs_3ds = np.dot(cs_3ds.copy(), R.T) + T

    cs_2ds = pcl_utils.project_p3d(cs_3ds, K=K, scale_m=1.0).astype("int")

    colors = get_color_tuples(color)

    img = cv2.arrowedLine(
        img, tuple(cs_2ds[0]), tuple(cs_2ds[1]), color=colors[0], thickness=thickness
    )
    img = cv2.arrowedLine(
        img, tuple(cs_2ds[0]), tuple(cs_2ds[2]), color=colors[1], thickness=thickness
    )
    img = cv2.arrowedLine(
        img, tuple(cs_2ds[0]), tuple(cs_2ds[3]), color=colors[2], thickness=thickness
    )
    img = cv2.circle(
        img, tuple(cs_2ds[0]), radius=thickness * 3, color=colors[0], thickness=-1
    )

    return img


def draw_bbox3d(
        img: np.array,
        bbox3ds: np.array,
        K: np.array,
        RT: np.array = None,
        color: Union[tuple, List] = (0, 0, 255),
        thickness=2
) -> np.array:
    """Draws a bounding box on the given image .

    Args:
        img (np.array): image array WxHx3
        bbox3ds (np.array): box corners 8x3 in object coordinate system
        K (np.array): camera projection matrix 3x3
        RT (np.array): object pose matrix 3x4 (if None, no transformation will be performed)
        color (Union[tuple, List], optional): color vector. Defaults to (0, 0, 255).
        thickness (int, optional): line thickness in pixel. Defaults to 2.

    Returns:
        np.array: image array WxHx3 with coordinate system overlay
    """

    if RT is not None:
        R, T = RT[:3, :3], RT[:3, 3]
        bbox3d = np.dot(bbox3ds.copy(), R.T) + T
    else:
        bbox3d = bbox3ds

    box2ds = pcl_utils.project_p3d(bbox3d, K=K, scale_m=1.0).astype("int").astype("int")

    img = cv2.line(img, tuple(box2ds[0]), tuple(box2ds[1]), color=color, thickness=thickness)
    img = cv2.line(img, tuple(box2ds[0]), tuple(box2ds[2]), color=color, thickness=thickness)
    img = cv2.line(img, tuple(box2ds[0]), tuple(box2ds[4]), color=color, thickness=thickness)
    img = cv2.line(img, tuple(box2ds[1]), tuple(box2ds[3]), color=color, thickness=thickness)
    img = cv2.line(img, tuple(box2ds[1]), tuple(box2ds[5]), color=color, thickness=thickness)
    img = cv2.line(img, tuple(box2ds[2]), tuple(box2ds[3]), color=color, thickness=thickness)
    img = cv2.line(img, tuple(box2ds[2]), tuple(box2ds[6]), color=color, thickness=thickness)
    img = cv2.line(img, tuple(box2ds[3]), tuple(box2ds[7]), color=color, thickness=thickness)
    img = cv2.line(img, tuple(box2ds[4]), tuple(box2ds[5]), color=color, thickness=thickness)
    img = cv2.line(img, tuple(box2ds[4]), tuple(box2ds[6]), color=color, thickness=thickness)
    img = cv2.line(img, tuple(box2ds[5]), tuple(box2ds[7]), color=color, thickness=thickness)
    img = cv2.line(img, tuple(box2ds[6]), tuple(box2ds[7]), color=color, thickness=thickness)

    img = draw_p2ds(img, box2ds, thickness=thickness * 3, color=color)

    return img


def draw_bbox2d(
        img: np.array,
        bbox2ds: Union[tuple, List],
        color: Union[tuple, List] = (0, 0, 255),
        thickness=2
) -> np.array:
    """Draws a 2D bounding box on the given image.

    Args:
        img (np.array): image array WxHx3
        bbox2ds (Union[np.array, List]): box corners in 2D image space
            (xmin, ymin, xmax, ymax)
        color (Union[tuple, List], optional): color vector. Defaults to (0, 0, 255).
        thickness (int, optional): line thickness in pixel. Defaults to 2.

    Returns:
        np.array: image array WxHx3 with coordinate system overlay
    """

    bbox_pts = [
        (int(bbox2ds[0]), int(bbox2ds[1])),  # (xmin, ymin)
        (int(bbox2ds[2]), int(bbox2ds[1])),  # (xmax, ymin)
        (int(bbox2ds[2]), int(bbox2ds[3])),  # (xmax, ymax)
        (int(bbox2ds[0]), int(bbox2ds[3]))   # (xmin, ymax)
    ]

    img = cv2.line(img, bbox_pts[0], bbox_pts[1], color=color, thickness=thickness)
    img = cv2.line(img, bbox_pts[1], bbox_pts[2], color=color, thickness=thickness)
    img = cv2.line(img, bbox_pts[2], bbox_pts[3], color=color, thickness=thickness)
    img = cv2.line(img, bbox_pts[3], bbox_pts[0], color=color, thickness=thickness)

    return img


def draw_p2ds(img, p2ds, thickness=1, color=[(255, 0, 0)]):
    if type(color) == tuple:
        color = [color]
    if len(color) != p2ds.shape[0]:
        color = [color[0] for i in range(p2ds.shape[0])]
    h, w = img.shape[0], img.shape[1]
    for pt_2d, c in zip(p2ds, color):
        if pt_2d[0] < 0 or pt_2d[0] >= w or pt_2d[1] < 0 or pt_2d[1] >= h:
            continue
        img = cv2.circle(
            img, (pt_2d[0], pt_2d[1]), radius=thickness, color=c, thickness=-1
        )
    return img


def draw_p3ds(
        img: np.array,
        pts3d: np.array,
        K: np.array,
        RT: np.array = None,
        color: Union[tuple, List] = (255, 0, 0),
        thickness: int = 2
) -> np.array:
    """Draws 3D object points onto the given image

    Args:
        img (np.array): image array WxHx3
        pts3d (np.array): 3D points Nx3 in object coordinate system
        K (np.array): camera projection matrix 3x3
        RT (np.array): object pose matrix 3x4. Defaults to None
        color (Union[tuple, List], optional): color vector. Defaults to (0, 0, 255).
        thickness (int, optional): line thickness in pixel. Defaults to 2.

    Returns:
        np.array: image array WxHx3 with coordinate system overlay

    Returns:
        np.array: [description]
    """

    if len(pts3d.shape) == 1:
        pts3d = pts3d[np.newaxis, :]

    assert pts3d.shape[1] == 3, "3D points must have size Nx3"

    if RT is not None:
        R, T = RT[:3, :3], RT[:3, 3]
        pts3d_new = np.dot(pts3d.copy(), R.T) + T
    else:
        pts3d_new = pts3d

    pts2ds = pcl_utils.project_p3d(pts3d_new, K=K, scale_m=1.0).astype("int")

    img = draw_p2ds(img, pts2ds, thickness, color)

    return img


def draw_model(
        meshc: dict,
        K: np.array,
        RT: np.array,
        img_size: Union[tuple, List] = (480, 640)
) -> Tuple[np.array, np.array, np.array]:
    """ Render image from object mesh given camera projection and object pose

    Args:
        meshc (dict): meshc dictionary from rgbd_rnder_sift_kp3ds.load_mesh_c
        K (np.array): camera projection matrix 3x3
        RT (np.array): object pose matrix 3x4
        img_size (Union[tuple, List]): output image size

    Returns:
        Tuple[np.array, np.array, np.array]: rendered rgb image, rendered z-buffer, object mask
    """

    h, w = img_size

    R, T = RT[:3, :3], RT[:3, 3]

    new_xyz = meshc['xyz'].copy()
    new_xyz = np.dot(new_xyz, R.T) + T
    p2ds = np.dot(new_xyz.copy(), K.T)
    p2ds = p2ds[:, :2] / p2ds[:, 2:]
    p2ds = np.require(p2ds.flatten(), 'float32', 'C')

    zs = np.require(new_xyz[:, 2].copy(), 'float32', 'C')
    zbuf = np.require(np.zeros(h * w), 'float32', 'C')
    rbuf = np.require(np.zeros(h * w), 'int32', 'C')
    gbuf = np.require(np.zeros(h * w), 'int32', 'C')
    bbuf = np.require(np.zeros(h * w), 'int32', 'C')

    RENDERER.rgbzbuffer(
        ct.c_int(h),
        ct.c_int(w),
        p2ds.ctypes.data_as(ct.c_void_p),
        new_xyz.ctypes.data_as(ct.c_void_p),
        zs.ctypes.data_as(ct.c_void_p),
        meshc['r'].ctypes.data_as(ct.c_void_p),
        meshc['g'].ctypes.data_as(ct.c_void_p),
        meshc['b'].ctypes.data_as(ct.c_void_p),
        ct.c_int(meshc['n_face']),
        meshc['face'].ctypes.data_as(ct.c_void_p),
        zbuf.ctypes.data_as(ct.c_void_p),
        rbuf.ctypes.data_as(ct.c_void_p),
        gbuf.ctypes.data_as(ct.c_void_p),
        bbuf.ctypes.data_as(ct.c_void_p),
    )

    zbuf.resize((h, w))
    mask = (zbuf > 1e-8).astype('uint8')
    if len(np.where(mask.flatten() > 0)[0]) < 500:
        raise RuntimeError(
            "ERROR: Object could not be rendered. "
            + "Please make sure that it is centered and big enough!"
        )

    zbuf *= mask.astype(zbuf.dtype)  # * 1000.0

    bbuf.resize((h, w)), rbuf.resize((h, w)), gbuf.resize((h, w))
    rgb = np.concatenate((rbuf[:, :, None], gbuf[:, :, None], bbuf[:, :, None]), axis=2)
    rgb = rgb.astype('uint8')

    return rgb, zbuf, mask


def depth2jetmap(depth_img):
    """
    Visualize depthmap as Jetmap rgb image
    """
    min_d, max_d = depth_img[depth_img > 0].min(), depth_img.max()
    depth_img[depth_img > 0] = (depth_img[depth_img > 0] - min_d) / (max_d - min_d) * 255
    depth_img = depth_img.astype(np.uint8)
    im_color = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_img, alpha=1), cv2.COLORMAP_JET
    )
    return im_color


def depth2normals(
    dpt, scale_to_mm, K, with_show=False
):
    dpt_mm = (dpt.copy() * scale_to_mm).astype(np.uint16)
    nrm_map = normalSpeed.depth_normal(dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False)
    if with_show:
        nrm_map[np.isnan(nrm_map)] = 0.0
        nrm_map[np.isinf(nrm_map)] = 0.0
        show_nrm = ((nrm_map[:, :, :3] + 1.0) * 127).astype(np.uint8)
        return nrm_map, show_nrm
    return nrm_map


def get_normal_map(nrm, choose):
    nrm_map = np.zeros((480, 640, 3), dtype=np.uint8)
    nrm = nrm[:, :3]
    nrm[np.isnan(nrm)] = 0.0
    nrm[np.isinf(nrm)] = 0.0
    nrm_color = ((nrm + 1.0) * 127).astype(np.uint8)
    nrm_map = nrm_map.reshape(-1, 3)
    nrm_map[choose, :] = nrm_color
    nrm_map = nrm_map.reshape((480, 640, 3))
    return nrm_map


def get_show_label_img(labels, mode=1):
    cls_ids = np.unique(labels)
    n_obj = np.max(cls_ids)
    if len(labels.shape) > 2:
        labels = labels[:, :, 0]
    h, w = labels.shape
    show_labels = np.zeros(
        (h, w, 3), dtype='uint8'
    )
    labels = labels.reshape(-1)
    show_labels = show_labels.reshape(-1, 3)
    for cls_id in cls_ids:
        if cls_id == 0:
            continue
        cls_color = np.array(
            get_label_color(cls_id, n_obj=n_obj, mode=mode)
        )
        show_labels[labels == cls_id, :] = cls_color
    show_labels = show_labels.reshape(h, w, 3)
    return show_labels


def get_rgb_pts_map(pts, choose):
    pts_map = np.zeros((480, 640, 3), dtype=np.uint8)
    pts = pts[:, :3]
    pts[np.isnan(pts)] = 0.0
    pts[np.isinf(pts)] = 0.0
    pts_color = pts.astype(np.uint8)
    pts_map = pts_map.reshape(-1, 3)
    pts_map[choose, :] = pts_color
    pts_map = pts_map.reshape((480, 640, 3))
    return pts_map


def fill_missing(
        dpt, cam_scale, scale_2_80m, fill_type='multiscale',
        extrapolate=False, show_process=False, blur_type='bilateral'
):
    dpt = dpt / cam_scale * scale_2_80m
    projected_depth = dpt.copy()
    if fill_type == 'fast':
        final_dpt = depth_map_utils_ycb.fill_in_fast(
            projected_depth, extrapolate=extrapolate, blur_type=blur_type,
            # max_depth=2.0
        )
    elif fill_type == 'multiscale':
        final_dpt, process_dict = depth_map_utils_ycb.fill_in_multiscale(
            projected_depth, extrapolate=extrapolate, blur_type=blur_type,
            show_process=show_process,
            max_depth=3.0
        )
    else:
        raise ValueError('Invalid fill_type {}'.format(fill_type))
    dpt = final_dpt / scale_2_80m * cam_scale
    return dpt


def visualize_prediction(cfg, data, cu_dt, end_points):
    rgb_hwc = np.transpose(data["rgb"][0].numpy(), (1, 2, 0)).astype("uint8").copy()
    pred_pose, pred_kps = pvn3d_eval_utils_kpls.get_poses_from_data(
        cfg, cu_dt, end_points, cfg.dataset.cls_id
    )
    img = draw_coordinate_system(
        img=rgb_hwc,
        ctr3d=None,
        K=cfg.dataset.K,
        RT=pred_pose,
        len_m=cfg.dataset.radius,
        color="rgb"
    )
    img = draw_p3ds(
        img,
        data["kp_3ds"][0, 0].numpy(),
        K=cfg.dataset.K,
        RT=None,
        thickness=4,
        color=(255, 0, 255)
    )
    img = draw_p3ds(
        img,
        pred_kps[:9, :],
        K=cfg.dataset.K,
        RT=None,
        thickness=6,
        color=(0, 255, 0)
    )
    img = draw_coordinate_system(
        img=img,
        ctr3d=None,
        K=cfg.dataset.K,
        RT=data["RTs"][0, 0].numpy(),
        len_m=cfg.dataset.radius,
        color="mmm",
        thickness=1
    )
    return img


class GenEnum():
    all = "all"
    rgb = "rgb"
    zbuf = "zbuf"
    mask = "mask"


def main():
    import argparse
    import os
    from matplotlib import pyplot as plt
    from rgbd_rnder_sift_kp3ds import load_mesh_c

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obj-pth",
        type=str,
        default="example_mesh/ape.ply",
        help="Path to object ply."
    )
    parser.add_argument(
        "--mesh-info-prefix",
        type=str,
        help="Path or name to/of generated mesh info files with prefix."
    )
    parser.add_argument(
        "--sv-img-pth",
        type=str,
        help="Path where images should be saved",
        default=None
    )
    parser.add_argument(
        "--gen", "-g",
        type=str,
        help="What images should be generated [{}]".format(
            ", ".join([GenEnum.all, GenEnum.rgb, GenEnum.zbuf, GenEnum.mask])
        ),
        default=GenEnum.all,
        choices=[GenEnum.all, GenEnum.rgb, GenEnum.zbuf, GenEnum.mask]
    )
    parser.add_argument(
        '--scale2m',
        type=float,
        default=1.0,
        help="Scale to transform unit of object mesh to be in meter."
    )
    parser.add_argument(
        '--vis', action="store_true", help="Visulaize rendered images."
    )
    parser.add_argument(
        '--sv', action="store_true", help="Save rendered images."
    )
    parser.add_argument(
        '--n_longitude',
        type=int,
        default=3,
        help="Number of longitude on sphere to sample."
    )
    parser.add_argument(
        '--n_latitude',
        type=int,
        default=3,
        help="Number of latitude on sphere to sample."
    )
    parser.add_argument(
        '--h', type=int, default=480, help="Height of rendered RGBD images."
    )
    parser.add_argument(
        '--w', type=int, default=640, help="Width of rendered RGBD images."
    )

    args = parser.parse_args()

    if not args.vis and not args.sv:
        raise IOError("Please specify either saving (--sv) or visualizing (--vis)")

    path = os.path.dirname(os.path.abspath(args.obj_pth))

    if "/" not in args.mesh_info_prefix:
        args.mesh_info_prefix = os.path.join(path, args.mesh_info_prefix)

    if not args.sv_img_pth:
        args.sv_img_pth = path

    print("load mesh info ...")
    fn_kpts = "{}_ORB_fps.txt".format(args.mesh_info_prefix)
    fn_ctr = "{}_center.txt".format(args.mesh_info_prefix)
    fn_corners = "{}_corners.txt".format(args.mesh_info_prefix)
    fn_r = "{}_radius.txt".format(args.mesh_info_prefix)

    kps = np.loadtxt(os.path.join(fn_kpts), dtype=np.float32)
    ctr = np.loadtxt(os.path.join(fn_ctr), dtype=np.float32)
    c3ds = np.loadtxt(os.path.join(fn_corners), dtype=np.float32)
    r = np.loadtxt(os.path.join(fn_r), dtype=np.float32)

    img_size = (args.h, args.w)

    sph_r = r / 0.035 * 0.18

    positions = pose_utils.camera_positions(
        args.n_longitude, args.n_latitude, sph_r
    )

    cam_poses = [pose_utils.get_camera_pose(pos) for pos in positions]

    print("load mesh ...")
    meshc = load_mesh_c(args.obj_pth, args.scale2m)

    K = [1.6 * args.w, 0, args.w // 2, 0, 1.6 * args.w, args.h // 2, 0, 0, 1]
    # K = [700, 0, 320, 0, 700, 240, 0, 0, 1]
    if type(K) == list:
        K = np.array(K).reshape(3, 3)

    print("generate visualizations ...")
    for pose_id in range(len(cam_poses)):

        cam_pose = cam_poses[pose_id]
        o2c_pose = pose_utils.get_o2c_pose_cv(cam_pose, np.eye(4))

        rgb, zbuf, mask = draw_model(meshc, K, o2c_pose, img_size)

        rgb = rgb.astype("uint8").copy()

        rgb = draw_coordinate_system(rgb, ctr, K, o2c_pose, r, 1)
        rgb = draw_bbox3d(rgb, c3ds, K, o2c_pose, (0, 0, 255), 1)
        rgb = draw_p3ds(rgb, kps, K, o2c_pose, (0, 255, 0), 2)

        img = np.zeros((args.h, 0, 3), dtype=np.uint8)
        if args.gen == GenEnum.all or args.gen == GenEnum.rgb:
            img = np.hstack((img, rgb))

        if args.gen == GenEnum.all or args.gen == GenEnum.zbuf:
            show_zbuf = zbuf.copy()
            min_d, max_d = show_zbuf[show_zbuf > 0].min(), show_zbuf.max()
            show_zbuf[show_zbuf > 0] = (show_zbuf[show_zbuf > 0] - min_d) / (max_d - min_d) * 255
            show_zbuf = show_zbuf.astype(np.uint8)
            img = np.hstack((img, np.dstack((show_zbuf, show_zbuf, show_zbuf))))

        if args.gen == GenEnum.all or args.gen == GenEnum.mask:
            show_mask = (mask / mask.max() * 255).astype(np.uint8)
            img = np.hstack((img, np.dstack((show_mask, show_mask, show_mask))))

        if args.sv:
            cv2.imwrite(
                os.path.join(args.sv_img_pth, "rendering_{}.png".format(pose_id)),
                img[:, :, ::-1]
            )

        if args.vis:
            plt.figure()
            plt.imshow(img)
            plt.show()


if __name__ == "__main__":
    main()
