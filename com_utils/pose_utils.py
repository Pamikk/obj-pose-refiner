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
import math


# Returns camera rotation and translation matrices from OpenGL.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(camera_pose):
    # bcam stands for blender camera
    R_bcam2cv = np.array([(1, 0, 0), (0, -1, 0), (0, 0, -1)])

    # Use matrix_world instead to account for all constraints
    location, rotation = camera_pose[:3, 3], camera_pose[:3, :3]
    R_world2bcam = rotation.T

    # Convert camera location to translation vector used in coordinate changes
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    # put into 4x4 matrix
    RT = np.eye(4)
    RT[:3, :3] = R_world2cv
    RT[:3, 3] = T_world2cv
    return RT


def get_o2c_pose_cv(cam2world_pose, obj2world_pose):
    """
    Get object 6D pose in cv camera coordinate system
    cam_pose: camera rotation and translation matrices from get_3x4_RT_matrix_from_blender().
    obj2world_pose: obj_pose in world coordinate system
    """
    world2cam_pose = get_3x4_RT_matrix_from_blender(cam2world_pose)
    obj2cam_pose = np.matmul(world2cam_pose, obj2world_pose)
    return obj2cam_pose


def is_rotation_matrix(R):
    """
    Check if given matrix is rotation matrix
    """
    Rt = np.transpose(R)
    check_identity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - check_identity)
    return n < 1e-6


def rotation_matrix_to_euler_angles(R):
    """
    Calculates rotation matrix to euler angles
    The result is the same as MATLAB except the order
    of the euler angles ( x and z are swapped ).
    """
    assert(is_rotation_matrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular :
        x = math.atan2(R[2, 1] , R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else :
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def euler_angles_to_rotation_matrix(theta):
    """
    Calculates Rotation Matrix given euler angle vector
    """
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def sample_sphere(num_samples, cls='ape'):
    """
    sample angles from the sphere
    reference: https://zhuanlan.zhihu.com/p/25988652?group_id=828963677192491008
    """
    flat_objects = ['037_scissors', '051_large_clamp', '052_extra_large_clamp']
    if cls in flat_objects:
        begin_elevation = 30
    else:
        begin_elevation = 0
    ratio = (begin_elevation + 90) / 180
    num_points = int(num_samples // (1 - ratio))
    phi = (np.sqrt(5) - 1.0) / 2.
    azimuths = []
    elevations = []
    for n in range(num_points - num_samples, num_points):
        z = 2. * n / num_points - 1.
        azimuths.append(np.rad2deg(2 * np.pi * n * phi % (2 * np.pi)))
        elevations.append(np.rad2deg(np.arcsin(z)))
    return np.array(azimuths), np.array(elevations)


def sample_poses(num_samples):
    s = np.sqrt(2) / 2
    cam_pose = np.array([
        [0.0, -s, s, 0.50],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, s, s, 0.60],
        [0.0, 0.0, 0.0, 1.0],
    ])
    eulers = rotation_matrix_to_euler_angles(cam_pose[:3, :3]).reshape(1, -1)
    eulers = np.repeat(eulers, num_samples, axis=0)
    translations = cam_pose[:3, 3].reshape(1, 3)
    translations = np.repeat(translations, num_samples, axis=0)
    # print(eulers.shape, translations.shape)

    azimuths, elevations = sample_sphere(num_samples)
    # euler_sampler = stats.gaussian_kde(eulers.T)
    # eulers = euler_sampler.resample(num_samples).T
    eulers[:, 0] = azimuths
    eulers[:, 1] = elevations
    # translation_sampler = stats.gaussian_kde(translations.T)
    # translations = translation_sampler.resample(num_samples).T
    RTs = []
    for euler in eulers:
        RTs.append(euler_angles_to_rotation_matrix(euler))
    RTs = np.array(RTs)
    # print(eulers.shape, translations.shape, RTs.shape)
    return RTs, translations
    # np.save(self.blender_poses_path, np.concatenate([eulers, translations], axis=-1))


def camera_positions(n_longitude, n_latitude, radius):
    """
    Sample 'n1' x lateral and 'n2' x horizontal camera positions in
    a 'radius' around the center
    """
    theta_list = np.linspace(0.1, np.pi, n_longitude + 1)
    theta_list = theta_list[:-1]
    phi_list = np.linspace(0, 2 * np.pi, n_latitude + 1)
    phi_list = phi_list[:-1]

    def product(a_lst, b_lst):
        res_lst = []
        for a in a_lst:
            for b in b_lst:
                res_lst.append((a, b))
        return res_lst

    cpList = product(theta_list, phi_list)
    PositionList = []
    for theta, phi in cpList:
        x = radius * math.sin(theta) * math.cos(phi)
        y = radius * math.sin(theta) * math.sin(phi)
        z = radius * math.cos(theta)
        PositionList.append((x, y, z))
    return PositionList


def get_camera_pose(T):
    """
    OpenGL camera coordinates, the camera z-axis points away from the scene,
    the x-axis points right in image space, and the y-axis points up in
    image space.
    see https://pyrender.readthedocs.io/en/latest/examples/cameras.html
    """
    z_direct = np.array(T)
    z_direct = z_direct / np.linalg.norm(z_direct)
    g_direct = np.array([0, 0, 1])
    x_direct = -np.cross(z_direct, g_direct)
    x_direct = x_direct / np.linalg.norm(x_direct)
    y_direct = np.cross(z_direct, x_direct)
    y_direct = y_direct / np.linalg.norm(y_direct)

    pose = np.array([x_direct, y_direct, z_direct])
    pose = np.transpose(pose)

    camera_pose = np.eye(4)
    camera_pose[:3, :3] = pose
    camera_pose[:3, 3] = T
    return camera_pose


def cal_degree_from_vec(v1, v2):
    """
    Calculate degree between two vectors
    """
    cos = np.dot(v1, v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    if abs(cos) > 1.0:
        cos = 1.0 * (-1.0 if cos < 0 else 1.0)
        print(cos, v1, v2)
    dg = np.arccos(cos) / np.pi * 180
    return dg


def cal_directional_degree_from_vec(v1, v2):
    """
    Calculate directional degree between two vectors
    """
    dg12 = cal_degree_from_vec(v1, v2)
    cross = v1[0] * v2[1] - v2[0] * v1[1]
    if cross < 0:
        dg12 = 360 - dg12
    return dg12
