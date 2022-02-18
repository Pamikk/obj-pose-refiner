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
import torch


def VOCap(rec, prec):
    """
    Get VOC average precision
    """
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0] + list(rec) + [0.1])
    mpre = np.array([0.0] + list(prec) + [prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i - 1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i]) * 10
    return ap


def cal_auc(add_dis, max_dis=0.1):
    """
    Calculate area-under-curve from ADD values
    """
    D = np.array(add_dis)
    D[np.where(D > max_dis)] = np.inf
    D = np.sort(D)
    n = len(add_dis)
    acc = np.cumsum(np.ones((1, n)), dtype=np.float32) / n
    aps = VOCap(D, acc)
    return aps * 100


def cal_add_cuda(pred_RT, gt_RT, p3ds):
    """
    Calculate ADD metrics from torch tensors
    """
    pred_p3ds = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
    gt_p3ds = torch.mm(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
    dis = torch.norm(pred_p3ds - gt_p3ds, dim=1)
    return torch.mean(dis)


def cal_adds_cuda(pred_RT, gt_RT, p3ds):
    """
    Calculate ADDs metrics from torch tensors
    """
    N, _ = p3ds.size()
    pd = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
    pd = pd.view(1, N, 3).repeat(N, 1, 1)
    gt = torch.mm(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
    gt = gt.view(N, 1, 3).repeat(1, N, 1)
    dis = torch.norm(pd - gt, dim=2)
    mdis = torch.min(dis, dim=1)[0]
    return torch.mean(mdis)


def mean_shift(data, radius=5.0):
    """
    Calculate mean shift from 'data' with a given 'radius'
    """
    clusters = []
    for i in range(len(data)):
        cluster_centroid = data[i]
        cluster_frequency = np.zeros(len(data))
        # Search points in circle
        while True:
            temp_data = []
            for j in range(len(data)):
                v = data[j]
                # Handle points in the circles
                if np.linalg.norm(v - cluster_centroid) <= radius:
                    temp_data.append(v)
                    cluster_frequency[i] += 1
            # Update centroid
            old_centroid = cluster_centroid
            new_centroid = np.average(temp_data, axis=0)
            cluster_centroid = new_centroid
            # Find the mode
            if np.array_equal(new_centroid, old_centroid):
                break
        # Combined 'same' clusters
        has_same_cluster = False
        for cluster in clusters:
            if np.linalg.norm(cluster['centroid'] - cluster_centroid) <= radius:
                has_same_cluster = True
                cluster['frequency'] = cluster['frequency'] + cluster_frequency
                break
        if not has_same_cluster:
            clusters.append({
                'centroid': cluster_centroid,
                'frequency': cluster_frequency
            })

    print('clusters (', len(clusters), '): ', clusters)
    clustering(data, clusters)
    return clusters


def clustering(data, clusters):
    """
    Clustering data using cluster frequency
    """
    t = []
    for cluster in clusters:
        cluster['data'] = []
        t.append(cluster['frequency'])
    t = np.array(t)
    # Clustering
    for i in range(len(data)):
        column_frequency = t[:, i]
        cluster_index = np.where(column_frequency == np.max(column_frequency))[0][0]
        clusters[cluster_index]['data'].append(data[i])


def best_fit_transform_torch(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding
    points A to B in m spatial dimensions
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl
        B: Nxm numpy array of corresponding points, usually points on camera axis
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    '''
    assert A.size() == B.size()
    # get number of dimensions
    m = A.size()[1]
    # translate points to their centroids
    centroid_A = torch.mean(A, dim=0)
    centroid_B = torch.mean(B, dim=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matirx
    H = torch.mm(AA.transpose(1, 0), BB)
    U, S, Vt = torch.svd(H)
    R = torch.mm(Vt.transpose(1, 0), U.transpose(1, 0))
    # special reflection case
    if torch.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = torch.mm(Vt.transpose(1, 0), U.transpose(1, 0))
    # translation
    t = centroid_B - torch.mm(R, centroid_A.view(3, 1))[:, 0]
    T = torch.zeros(3, 4).cuda()
    T[:, :3] = R
    T[:, 3] = t
    return T


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding
    points A to B in m spatial dimensions
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl
        B: Nxm numpy array of corresponding points, usually points on camera axis
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    '''

    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matirx
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.zeros((3, 4))
    T[:, :3] = R
    T[:, 3] = t
    return T
