#!/usr/bin/env python3
import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
from ope_utils import visualization_utils, pose_utils, mesh_utils, pcl_utils

SO_P = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "raster_triangle",
    "rastertriangle_so.so"
)
RENDERER = np.ctypeslib.load_library(SO_P, '.')


def load_mesh_c(mdl_p, scale2m):
    if 'ply' in mdl_p:
        meshc = mesh_utils.load_ply_model(mdl_p, scale2m=scale2m)
    meshc['face'] = np.require(meshc['face'], 'int32', 'C')
    meshc['r'] = np.require(np.array(meshc['r']), 'float32', 'C')
    meshc['g'] = np.require(np.array(meshc['g']), 'float32', 'C')
    meshc['b'] = np.require(np.array(meshc['b']), 'float32', 'C')
    return meshc


def gen_one_zbuf_render(args, meshc, RT):
    if args.extractor == 'SIFT':
        extractor = cv2.xfeatures2d.SIFT_create()
    else:  # use orb
        extractor = cv2.ORB_create()

    h, w = args.h, args.w
    if type(args.K) == list:
        K = np.array(args.K).reshape(3, 3)
    else:
        K = args.K

    rgb, zbuf, msk = visualization_utils.draw_model(meshc, K, RT, (h, w))

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if args.vis:
        show_zbuf = zbuf.copy()
        min_d, max_d = show_zbuf[show_zbuf > 0].min(), show_zbuf.max()
        show_zbuf[show_zbuf > 0] = (show_zbuf[show_zbuf > 0] - min_d) / (max_d - min_d) * 255
        show_zbuf = show_zbuf.astype(np.uint8)
        show_msk = (msk / msk.max() * 255).astype("uint8")

        plt.figure()
        plt.subplot(131)
        plt.imshow(rgb.astype("uint8"))
        plt.title('RGB')
        plt.subplot(132)
        plt.imshow(show_zbuf.astype("uint8"))
        plt.title('zbuf')
        plt.subplot(133)
        plt.imshow(show_msk.astype("uint8"))
        plt.title('mask')
        plt.show()

    data = {}
    data['depth'] = zbuf
    data['rgb'] = rgb
    data['mask'] = msk
    data['K'] = K
    data['RT'] = RT
    data['cls_typ'] = args.obj_name
    data['rnd_typ'] = 'render'

    kps, des = extractor.detectAndCompute(bgr, None)

    kp_xys = np.array([kp.pt for kp in kps]).astype(np.int32)
    kp_idxs = (kp_xys[:, 1], kp_xys[:, 0])

    dpt_xyz = pcl_utils.dpt2cld(zbuf, K, 1.0)
    kp_x = dpt_xyz[:, :, 0][kp_idxs][..., None]
    kp_y = dpt_xyz[:, :, 1][kp_idxs][..., None]
    kp_z = dpt_xyz[:, :, 2][kp_idxs][..., None]
    kp_xyz = np.concatenate((kp_x, kp_y, kp_z), axis=1)

    # filter by dpt (pcld)
    kp_xyz, msk = pcl_utils.filter_pcld(kp_xyz)
    kps = [kp for kp, valid in zip(kps, msk) if valid]  # kps[msk]
    des = des[msk, :]

    # 6D pose of object in cv camer coordinate system
    # transform to object coordinate system
    kp_xyz = (kp_xyz - RT[:3, 3]).dot(RT[:3, :3])
    dpt_xyz = dpt_xyz[dpt_xyz[:, :, 2] > 0, :]
    dpt_pcld = (dpt_xyz.reshape(-1, 3) - RT[:3, 3]).dot(RT[:3, :3])

    data['kp_xyz'] = kp_xyz
    data['dpt_pcld'] = dpt_pcld

    return data


def extract_textured_kp3ds(args, mesh_pth, sv_kp=False):
    meshc = load_mesh_c(mesh_pth, args.scale2m)
    xyzs = meshc['xyz']
    # mean = np.mean(xyzs, axis=0)
    obj_pose = np.eye(4)
    # obj_pose[:3, 3] = -1.0 * mean
    bbox = mesh_utils.get_3d_bbox(xyzs)
    r = mesh_utils.get_r(bbox)

    print("r:", r)

    sph_r = r / 0.035 * 0.18
    positions = pose_utils.camera_positions(
        args.n_longitude, args.n_latitude, sph_r
    )
    cam_poses = [pose_utils.get_camera_pose(pos) for pos in positions]
    kp3ds = []
    # pclds = []
    for cam_pose in cam_poses:
        o2c_pose = pose_utils.get_o2c_pose_cv(cam_pose, obj_pose)
        # transform to object coordinate system
        data = gen_one_zbuf_render(args, meshc, o2c_pose)
        kp3ds += list(data['kp_xyz'])
        # pclds += list(data['dpt_pcld'])

    if sv_kp:
        with open("%s_%s_textured_kp3ds.obj" % (args.obj_name, args.extractor), 'w') as of:
            for p3d in kp3ds:
                print('v ', p3d[0], p3d[1], p3d[2], file=of)
    return kp3ds


def test():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "--obj_name", type=str, default="ape",
        help="Object name."
    )
    parser.add_argument(
        "--ply_pth", type=str, default="example_mesh/ape.ply",
        help="path to object ply."
    )
    parser.add_argument(
        '--debug', action="store_true",
        help="To show the generated images or not."
    )
    parser.add_argument(
        '--vis', action="store_true",
        help="visulaize generated images."
    )
    parser.add_argument(
        '--h', type=int, default=480,
        help="height of rendered RGBD images."
    )
    parser.add_argument(
        '--w', type=int, default=640,
        help="width of rendered RGBD images."
    )
    parser.add_argument(
        '--K', type=int, default=[700, 0, 320, 0, 700, 240, 0, 0, 1],
        help="camera intrinsix."
    )
    parser.add_argument(
        '--scale2m', type=float, default=1.0,
        help="scale to transform unit of object to be in meter."
    )
    parser.add_argument(
        '--n_longitude', type=int, default=3,
        help="number of longitude on sphere to sample."
    )
    parser.add_argument(
        '--n_latitude', type=int, default=3,
        help="number of latitude on sphere to sample."
    )
    parser.add_argument(
        '--extractor', type=str, default="ORB",
        help="2D keypoint extractor, SIFT or ORB"
    )
    parser.add_argument(
        '--textured_3dkps_fd', type=str, default="textured_3D_keypoints",
        help="folder to store textured 3D keypoints."
    )
    args = parser.parse_args()
    args.K = np.array(args.K).reshape(3, 3)

    _ = extract_textured_kp3ds(args, args.ply_pth)


if __name__ == "__main__":
    test()

# vim: ts=4 sw=4 sts=4 expandtab
