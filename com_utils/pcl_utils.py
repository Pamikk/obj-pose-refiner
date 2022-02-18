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

from typing import List, Tuple, Union
import numpy as np
from configs import dataset_config


def project_p3d(p3d: np.ndarray, K: np.ndarray, scale_m: float = 1.0):
    """
    Project 3D pointcloud to 2D image plane
    p3d: Nx3
    """
    if type(K) == str:
        K = dataset_config.intrinsic_matrix[K]
    p3d = p3d * scale_m
    p2d = np.dot(p3d, K.T)
    p2d_3 = p2d[:, 2]
    p2d_3[np.where(p2d_3 < 1e-8)] = 1.0
    p2d[:, 2] = p2d_3
    p2d = np.around((p2d[:, :2] / p2d[:, 2:])).astype(np.int32)
    return p2d


def dpt2cld(
    dpt: np.ndarray,
    K: np.ndarray,
    scale_m: float = 1.0,
    xymap: Union[Tuple, List, np.ndarray] = None,
    projective: bool = False
):
    """
    Backproject depth map to pointcloud
    """
    h, w = dpt.shape[0], dpt.shape[1]

    if xymap:
        xmap = xymap[0]
        ymap = xymap[1]
    else:
        xmap, ymap = np.meshgrid(np.arange(w), np.arange(h))

    if len(dpt.shape) > 2:
        dpt = dpt[:, :, 0]
    msk_dp = dpt > 1e-6
    choose = msk_dp.flatten()
    choose[:] = True
    if len(choose) < 1:
        return None, None

    dpt_mskd = dpt.flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_mskd = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_mskd = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

    pts2d = np.concatenate((xmap_mskd, ymap_mskd, np.ones(xmap_mskd.shape)), axis=1)
    pts2d_proj = np.matmul(np.linalg.inv(K), pts2d.T).T
    if projective:
        pts2d_proj = pts2d_proj / np.linalg.norm(pts2d_proj, axis=1)[:, np.newaxis]
    d = dpt_mskd / scale_m
    pts3d = pts2d_proj * d
    cld = pts3d.reshape(h, w, 3)

    return cld


def filter_pcld(pcld: np.ndarray):
    """
    filter out infinite and very small points from poincloud
    """
    if len(pcld.shape) > 2:
        pcld = pcld.reshape(-1, 3)
    msk1 = np.isfinite(pcld[:, 0])
    msk2 = pcld[:, 2] > 1e-8
    msk = msk1 & msk2
    pcld = pcld[msk, :]
    return pcld, msk


def transform_pcld(pts3d: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Transform pointcloud by transformation matrix P

    Args:
        pts3d (np.ndarray): pointcloud of size Nx3
        RT (np.ndarray): transformation matrix of size 3x4 or 4x4

    Returns:
        np.ndarray: transformed pointcloud
    """

    assert pts3d.shape[1] == 3, "Pointcloud must be of size Nx3"
    assert (P.shape[0] == 3 or P.shape[0] == 4) and P.shape[1] == 4, \
        "Transformation matrix must be of size 3x4 or 4x4"

    if P.shape[0] == 3:
        P = np.vstack((P, np.array([.0, .0, .0, 1.0])))

    pts3d_hom = np.hstack((pts3d.copy(), np.ones((len(pts3d), 1))))
    pts3d_trans = np.dot(P, pts3d_hom.T)
    pts3d_trans = pts3d_trans[:3, :] / pts3d_trans[3:, :]
    return pts3d_trans.T
