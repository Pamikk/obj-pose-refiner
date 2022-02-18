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

# Script to extract Object Informations from a textured object, either in 'ply'
#   or in 'obj' format. The extracted information is
#    * Farthest Point Keypoints
#    * Textured Farthest Point Keypoints
#    * Object Center
#    * Object Boundingbox Corners
#    * Object Radius
#
# Usage:
# - Generate information of objects, eg. radius, 3D keypoints, etc. by:
#   ```
#   python3 gen_obj_info.py --help
#   ```
#   If you use ply model and the vertex color is contained in the ply model, you
#   can use the default raster triangle for rendering. For example, you can
#   generate the information of the example ape object by running:
#
#   ```shell
#   python gen_obj_info.py \
#       --obj_name='ape' \
#       --obj_pth='example_mesh/ape.ply' \
#       --scale2m=1000. \
#       --sv_fd='ape_info'
#   ```
#   You need to set the parameter ```scale2m``` according to the original unit
#   of you object so that the generated info are all in unit meter.

#   If you use obj model, you can convert each vertex in meter and use
#   pyrender. For example, you can generate the information of the example
#   cracker box by running:
#   ```shell
#   python gen_obj_info.py \
#       --obj_name='cracker_box' \
#       --obj_pth='example_mesh/003_cracker_box/textured.obj' \
#       --scale2m=1. --sv_fd='cracker_box_info'
#       --use_pyrender
#   ```

import os
import numpy as np
from argparse import ArgumentParser
from ope_utils.dataset_tools.fps.fps_utils import farthest_point_sampling
from ope_utils import file_utils
from ope_utils import mesh_utils


# fixed intrinsic matrices and image sizes to perfectly fit object in image
intrinsic_matrix = {
    "linemod": np.array([[700, 0, 320], [0, 700, 240], [0, 0, 1]]),
    "ml2_sensim_rgb": np.array([[3500., 0., 1017.],
                                [0., 3500.,  765.],
                                [0.,    0.,    1.0]], np.float32)
}

image_sizes = {
    "linemod": (480, 640),
    "ml2_sensim_rgb": (1532, 2036)
}


# Select keypoint with Farthest Point Sampling (FPS) algorithm
def get_farthest_3d(p3ds, num=8, init_center=False):
    fps = farthest_point_sampling(p3ds, num, init_center=init_center)
    return fps


# Compute and save all mesh info
def gen_one_mesh_info(args, obj_pth, sv_fd):

    if args.use_pyrender:
        print("Use PyRender SIFT Keypoints")
        from ope_utils.dataset_tools.pyrender_sift_kp3ds import extract_textured_kp3ds
    else:
        print("Use RGBD Render SIFT Keypoints")
        from ope_utils.dataset_tools.rgbd_rnder_sift_kp3ds import extract_textured_kp3ds

    if sv_fd is None:
        sv_fd = os.path.dirname(obj_pth)

    file_utils.ensure_dir(sv_fd)

    p3ds = mesh_utils.get_p3ds_from_mesh(obj_pth, scale2m=args.scale2m)

    c3ds = mesh_utils.get_3d_bbox(p3ds)
    c3ds_pth = os.path.join(sv_fd, "%s_corners.txt" % args.obj_name)
    with open(c3ds_pth, 'w') as of:
        for p3d in c3ds:
            print(p3d[0], p3d[1], p3d[2], file=of)

    radius = mesh_utils.get_r(c3ds)
    r_pth = os.path.join(sv_fd, "{}_radius.txt".format(args.obj_name))
    with open(r_pth, 'w') as of:
        print(radius, file=of)

    ctr = mesh_utils.get_centers_3d(c3ds)
    ctr_pth = os.path.join(sv_fd, "{}_center.txt".format(args.obj_name))
    with open(ctr_pth, 'w') as of:
        print(ctr[0], ctr[1], ctr[2], file=of)

    fps = get_farthest_3d(p3ds, num=args.n_keypoint)
    fps_pth = os.path.join(sv_fd, "{}_{}_fps.txt".format(args.obj_name, args.n_keypoint))
    with open(fps_pth, 'w') as of:
        for p3d in fps:
            print(p3d[0], p3d[1], p3d[2], file=of)

    textured_kp3ds = np.array(extract_textured_kp3ds(args, args.obj_pth))
    print(p3ds.shape, textured_kp3ds.shape)
    textured_fps = get_farthest_3d(textured_kp3ds, num=args.n_keypoint)
    textured_fps_pth = os.path.join(
        sv_fd, "{}_{}_{}_fps.txt".format(args.obj_name, args.extractor, args.n_keypoint)
    )
    with open(textured_fps_pth, 'w') as of:
        for p3d in textured_fps:
            print(p3d[0], p3d[1], p3d[2], file=of)

    textured_fps_pth_obj = os.path.join(
        sv_fd, "{}_{}_{}_fps.obj".format(args.obj_name, args.extractor, args.n_keypoint)
    )
    with open(textured_fps_pth_obj, 'w') as of:
        for p3d in textured_fps:
            print("v ", p3d[0], p3d[1], p3d[2], file=of)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--obj_name", type=str, default="ape", help="Object name."
    )
    parser.add_argument(
        "--obj_pth", type=str, default="example_mesh/ape.ply",
        help="Path to object ply."
    )
    parser.add_argument(
        "--sv_fd", type=str, default=None,
        help="Path to save the generated mesh info."
    )
    parser.add_argument(
        '--scale2m', type=float, default=1.0,
        help="Scale to transform unit of object to be in meter."
    )
    parser.add_argument(
        '--vis', action="store_true", help="Visulaize rendered images."
    )
    parser.add_argument(
        '--h', type=int, default=480, help="Height of rendered RGBD images."
    )
    parser.add_argument(
        '--w', type=int, default=640, help="Width of rendered RGBD images."
    )
    parser.add_argument(
        '--K', type=int, default=[700, 0, 320, 0, 700, 240, 0, 0, 1],
        help="camera intrinsics."
    )
    parser.add_argument(
        '--dataset', type=str, default=None,
        help="dataset name to get K and image size"
    )
    parser.add_argument(
        '--n_longitude', type=int, default=3,
        help="Number of longitude on sphere to sample."
    )
    parser.add_argument(
        '--n_latitude', type=int, default=3,
        help="Number of latitude on sphere to sample."
    )
    parser.add_argument(
        '--extractor', type=str, default="ORB",
        help="2D keypoint extractor, SIFT or ORB"
    )
    parser.add_argument(
        '--n_keypoint', type=int, default=8,
        help="Number of keypoints to extract."
    )
    parser.add_argument(
        '--textured_3dkps_fd', type=str, default="textured_3D_keypoints",
        help="Folder to store textured 3D keypoints."
    )
    parser.add_argument(
        '--use_pyrender', action='store_true',
        help="Use pyrender or raster_triangle."
    )
    args = parser.parse_args()

    if args.dataset:
        args.K = intrinsic_matrix[args.dataset]
        args.h = image_sizes[args.dataset][0]
        args.w = image_sizes[args.dataset][1]

    gen_one_mesh_info(args, args.obj_pth, args.sv_fd)


if __name__ == "__main__":
    main()


# vim: ts=4 sw=4 sts=4 expandtab
