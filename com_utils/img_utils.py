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
import cv2


def translate(img: np.ndarray, x, y):
    """
    Translate image in 'x', 'y' direction
    """
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return shifted


def rotate(img: np.ndarray, angle: float, ctr=None, scale=1.0):
    """
    Rotate image by 'angle' around center 'ctr'
    """
    (h, w) = img.shape[:2]
    if ctr is None:
        ctr = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(ctr, -1.0 * angle, scale)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated


def gaussian_noise(rng, img, sigma):
    """add gaussian noise of given sigma to image"""
    img = img + rng.randn(*img.shape) * sigma
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def linear_motion_blur(img, angle, length):
    """:param angle: in degree"""
    rad = np.deg2rad(angle)
    dx = np.cos(rad)
    dy = np.sin(rad)
    a = int(max(list(map(abs, (dx, dy)))) * length * 2)
    if a <= 0:
        return img
    kern = np.zeros((a, a))
    cx, cy = a // 2, a // 2
    dx, dy = list(map(int, (dx * length + cx, dy * length + cy)))
    cv2.line(kern, (cx, cy), (dx, dy), 1.0)
    s = kern.sum()
    if s == 0:
        kern[cx, cy] = 1.0
    else:
        kern /= s
    return cv2.filter2D(img, -1, kern)


def rgb_add_noise(img: np.ndarray, rng=None) -> np.ndarray:
    """
    Add noise to RGB image

    Args:
        img (np.ndarray): input image
        rng (optional): Random number generator. If None, np.random is
            usedDefaults to None.

    Returns:
        np.ndarray: input image with added noise
    """

    if rng is None:
        rng = np.random

    def rand_range(rng, lo, hi):
        return rng.rand() * (hi - lo) + lo

    # apply HSV augmentor
    if rng.rand() > 0:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.uint16)
        hsv_img[:, :, 1] = hsv_img[:, :, 1] * rand_range(
            rng, 1 - 0.25, 1 + 0.25
        )
        hsv_img[:, :, 2] = hsv_img[:, :, 2] * rand_range(
            rng, 1 - 0.15, 1 + 0.15
        )
        hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
        hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2], 0, 255)
        img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if rng.rand() > 0.8:  # motion blur
        r_angle = int(rng.rand() * 360)
        r_len = int(rng.rand() * 15) + 1
        img = linear_motion_blur(img, r_angle, r_len)

    if rng.rand() > 0.8:
        if rng.rand() > 0.2:
            img = cv2.GaussianBlur(img, (3, 3), rng.rand())
        else:
            img = cv2.GaussianBlur(img, (5, 5), rng.rand())

    return np.clip(img, 0, 255).astype(np.uint8)
