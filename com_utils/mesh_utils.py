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

import numpy as np
from plyfile import PlyData, PlyElement


def get_p3ds_from_obj(path, scale2m=1.):
    """
    Read pointcloud from obj mesh (scaled to meters)
    """
    xyz_lst = []
    with open(path, 'r') as f:
        for line in f.readlines():
            if 'v ' not in line or line[0] != 'v':
                continue
            xyz_str = [
                item.strip() for item in line.split(' ')
                if len(item.strip()) > 0 and 'v' not in item
            ]
            xyz = np.array(xyz_str[0:3]).astype(np.float)
            xyz_lst.append(xyz)
    return np.array(xyz_lst) / scale2m


def load_ply_model(path: str, scale2m: float = 1., ret_dict: bool = True):
    """
    Read colorized mesh from ply model (scaled to meters)

    Args:
        path (str): path to file
        scale2m (float, optional): scale of point coordinates to meters.
            Defaults to 1..
        ret_dict (bool, optional): if True returns dict of format mesh

    Returns:
        if ret_dict == True:
            dict(
                n_pts ... number of points
                xyz ... points (x,y,z),
                r ... red channel
                g ... green channel
                b ... blue channel
                n_face ... number of faces
                face ... face vertices flattened (3*n_face)
            )
        else:
            list of same content

    """
    with open(path, "rb") as fp:
        ply = PlyData.read(fp)
    data = ply.elements[0].data
    x = data["x"]
    y = data["y"]
    z = data["z"]
    r = data["red"]
    g = data["green"]
    b = data["blue"]
    n_pts = len(x)
    xyz = np.stack([x, y, z], axis=-1) / scale2m

    face = []
    n_face = 0
    if "face" in ply:
        face_raw = ply["face"]
        for item in face_raw.data:
            face.append(item[0])

        n_face = len(face)
        face = np.array(face).flatten()

    if not ret_dict:
        return n_pts, xyz, r, g, b, n_face, face
    else:
        ret_dict = dict(
            n_pts=n_pts, xyz=xyz, r=r, g=g, b=b, n_face=n_face, face=face
        )
        return ret_dict


def save_ply_model(path: str, mesh_dict: dict, use_ascii=False):
    """
    Write colorized mesh to ply model (scaled to meters)

    Args:
        path (str): path to file
        mesh_dict (dict): mesh dictionary of format
            dict(
                n_pts ... number of points
                xyz ... points (x,y,z),
                r ... red channel
                g ... green channel
                b ... blue channel
                n_face ... number of faces
                face ... face vertices flattened (3*n_face)
            )
        use_ascii (bool, optional): Save ply model as ascii text of binary.
            Defaults to False.
    """
    xyz = mesh_dict["xyz"]
    r = mesh_dict["r"]
    g = mesh_dict["g"]
    b = mesh_dict["b"]

    assert xyz.shape[0] == r.shape[0]

    xyzrgb_points = [
        (xyz[i, 0], xyz[i, 1], xyz[i, 2], r[i], g[i], b[i])
        for i in range(mesh_dict["n_pts"])
    ]

    vertex = np.array(xyzrgb_points, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                            ("red", "u1"), ("green", "u1"), ("blue", "u1")])

    ply_data = [PlyElement.describe(vertex, "vertex", comments=["vertices"])]
    if "n_face" in mesh_dict and mesh_dict["n_face"] > 0:
        vertex_idxs_raw = np.reshape(mesh_dict["face"], (mesh_dict["n_face"], 3))
        vertex_idxs = np.array(
            [(vertex_idx, ) for vertex_idx in vertex_idxs_raw],
            dtype=[('vertex_indices', 'O')])

        ply_data.append(PlyElement.describe(vertex_idxs, 'face'))

    PlyData(ply_data, text=use_ascii, byte_order='<').write(path)


# Read object vertexes from ply file
def get_p3ds_from_ply(path: str, scale2m: float = 1.):
    """
    Read pointcloud from ply file (scaled to meters)
    """
    print("loading p3ds from ply:", path)
    with open(path, "rb") as fp:
        ply = PlyData.read(fp)
    data = ply.elements[0].data
    x = data['x']
    y = data['y']
    z = data['z']
    p3ds = np.stack([x, y, z], axis=-1)
    p3ds = p3ds / float(scale2m)
    print("finish loading ply.")
    return p3ds


def get_p3ds_from_mesh(mesh_pth: str, scale2m: bool = 1.0):
    """
    Get pointcloud from mesh file (scaled to meters)
    """
    if '.ply' in mesh_pth:
        return get_p3ds_from_ply(mesh_pth, scale2m=scale2m)
    else:
        return get_p3ds_from_obj(mesh_pth, scale2m=scale2m)


def get_3d_bbox(pcld: np.ndarray, small: bool = False):
    """
    Compute the 3D bounding box from object vertexes
    """
    min_x, max_x = pcld[:, 0].min(), pcld[:, 0].max()
    min_y, max_y = pcld[:, 1].min(), pcld[:, 1].max()
    min_z, max_z = pcld[:, 2].min(), pcld[:, 2].max()
    bbox = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    if small:
        center = np.mean(bbox, 0)
        bbox = (bbox - center[None, :]) * 2.0 / 3.0 + center[None, :]
    return bbox


def get_r(bbox: np.ndarray):
    """
    Compute the radius of object
    """
    return np.linalg.norm(bbox[7, :] - bbox[0, :]) / 2.0


def get_centers_3d(corners_3d: np.ndarray):
    """
    Compute the center of object
    """
    centers_3d = (np.max(corners_3d, 0) + np.min(corners_3d, 0)) / 2
    return centers_3d


def get_dimensions(corners_3d: np.ndarray):
    """
    Get x, y, z dimensions of 3D boundingbox
    """
    x_dim = np.abs(np.max(corners_3d[:, 0]) - np.min(corners_3d[:, 0]))
    y_dim = np.abs(np.max(corners_3d[:, 1]) - np.min(corners_3d[:, 1]))
    z_dim = np.abs(np.max(corners_3d[:, 2]) - np.min(corners_3d[:, 2]))

    return x_dim, y_dim, z_dim
