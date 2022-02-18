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
import pickle
from typing import List
import yaml
import json
from plyfile import PlyData
import numpy as np
import cv2
from PIL import Image


def read_lines(path: str):
    """
    Read file line by line
    """
    with open(path, 'r') as f:
        lines = [
            line.strip() for line in f.readlines()
        ]
    return lines


def save_lines(path: str, line_lst: List):
    """
    Save list to path line by line
    """
    with open(path, 'w') as f:
        for line in line_lst:
            print(line, file=f)


def read_pickle(pkl_path: str):
    """
    Read pickle file
    """
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: str, data):
    """
    Save data to pickle file
    """
    with open(path, "wb") as f:
        pickle.dump(data, f)


def ensure_dir(pth):
    """
    Create directory if not existing
    """
    if not os.path.exists(pth):
        os.system("mkdir -p %s" % pth)


def read_np(path: str, dtype=np.float32):
    """
    Read numpy array from txt file
    """
    return np.loadtxt(path, dtype=dtype)


def save_np(path: str, data: np.ndarray):
    """
    Save numpy array to txt file
    """
    np.savetxt(path, data)


def read_yaml(path):
    """
    Read yaml file (avoid file-ending missmatch 'yaml' vs 'yml')
    """
    if not os.path.isfile(path):
        if path.endswith(".yml"):
            path = os.path.splitext(path)[0] + ".yaml"
        elif path.endswith(".yaml"):
            path = os.path.splitext(path)[0] + ".yml"

    with open(path, "r") as rf:
        data = yaml.safe_load(rf)
    return data


def save_yaml(
    path: str,
    data
):
    with open(path, "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def read_image(
        path: str,
        img_size: tuple = None,
        interpolation: int = cv2.INTER_CUBIC,
        noise=None,
        force_format=None
) -> np.ndarray:
    """
    Read image from 'path', resize it to 'img_size' with 'interpolation'
    if noise is set, it will be applied to the original image
    if force_format is set, the image is converted to the desired format
        options: "RGB", "L", ...
        (https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes)
    """
    with Image.open(path) as pil_img:
        if force_format:
            pil_img = pil_img.convert(force_format)

        if noise:
            pil_img = noise(pil_img)
        img = np.array(pil_img)

    if img_size is not None:
        img = cv2.resize(
            img,
            dsize=img_size,
            interpolation=interpolation
        )
    return img
