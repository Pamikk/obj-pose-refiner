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
import cv2
import logging
import wandb
import numpy as np
from collections import defaultdict
from typing import (
    Any,
    List,
    Dict,
    Tuple,
    Union
)
from com_utils import file_utils


class EventWriter:
    """
    Base class for writers that obtain events from :class:`EventStorage` and process them.
    """

    def write(self):
        raise NotImplementedError

    def finish(self):
        pass

    def __exit__(self):
        self.finish()


class WandBWriter(EventWriter):
    """
    Write all scalar metrics to  W&B dashboard
    """
    def __init__(
            self,
            name: str,
            group_name: str,
            project_name: str,
            output_dir: str,
            cfg: dict = None,
            resume: bool = False
    ) -> None:
        """
        Args:
            window_size (int): the scalars will be median-smoothed by this window size
            cfg: Config file
        """

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        self._logger.addHandler(ch)

        if not name or not project_name:
            raise ValueError("WANDB Info settings not set.\n"
                             "  Please give your experiment a NAME and PROJECT_NAME")

        if group_name is None:
            group_name = name

        # resume wandb writer from previous runs
        if resume and os.path.isfile(os.path.join(output_dir, "wandb_trainingid.txt")):
            with open(os.path.join(output_dir, "wandb_trainingid.txt"), 'r') as fp:
                wandb_id = fp.read().replace("\n", "")
            self._logger.info("Resume WandB logging from run {}".format(wandb_id))
        else:
            wandb_id = wandb.util.generate_id()
            with open(os.path.join(output_dir, "wandb_trainingid.txt"), 'w') as fp:
                fp.write(wandb_id)

        if wandb.run and wandb.run.name == name \
                and wandb.run.group == group_name \
                and wandb.run.project_name == project_name:
            self._writer = wandb.run
        else:
            self._writer = wandb.init(
                id=wandb_id,
                resume="allow" if resume else "never",
                group=group_name,
                project=project_name,
                name=name,
                force=True,
                dir=output_dir,
                config=cfg
            )
        if cfg is not None:
            self._writer.summary["Description"] = cfg.description

        self.curr_log_msg = defaultdict(list)

    def aggregate(self, log_dict: Dict[str, Any]):
        """
        Aggregate log message of a certain itearation with new log_dict data

        Args:
            log_dict (Dict[str, Any]): logging dictionary
        """
        for k, v in log_dict.items():
            if isinstance(v, np.ndarray):
                v = wandb.Image(v.copy(), caption=k)
            if isinstance(v, list):
                self.curr_log_msg[k].extend(v)
            else:
                self.curr_log_msg[k].append(v)

    def push(self, iter: int = None):
        """
        Push aggregated log message at iteration iter

        Args:
            iter (int, optional): Iteration/Step. Defaults to None.
        """
        if self.curr_log_msg:
            self._writer.log(self.curr_log_msg, step=iter)
            self.curr_log_msg = defaultdict(list)
        else:
            self._logger.warning("Push: log message is empty at. Dropping.")

    def write(self, log_dict: Dict[str, Any], iter: int = None):
        """
        Log wandb dictionary info

        Args:
            log_dict (Dict[str, Any]): logging dictionary
            iter (int, optional): iteration/step. Defaults to None.
        """
        self._writer.log(log_dict, step=iter)

    def write_image(self, caption: str, img: np.ndarray, iter: int = None):
        """
        Logging numpy array as wandb image

        Args:
            caption (str): image caption / logging field
            img (np.ndarray): image matrix
            iter (int, optional): iteration/step. Defaults to None.
        """
        self._writer.log(
            {caption: wandb.Image(img, caption=caption)}, step=iter
        )

    def finish(self):
        if self._writer:
            self._writer.finish()
            self._writer = None


class VideoWriter(EventWriter):
    """
    Write videostream to file
    """
    def __init__(
            self,
            filename: str,
            output_size: Union[List[int], Tuple[int]],
            frame_rate: int = 15,
            save_imgs: bool = False
    ) -> None:
        """Initialize a video writer.

        Args:
            filename (str): filename of output video file
            output_size (Union[List[int], Tuple[int]]): output size of format (width, height)
            frame_rate (int, optional): Framerate of output video. Defaults to 15.
        """

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        self._logger.addHandler(ch)

        self._logger.info("Start VideWriter stream to {}".format(filename))

        self._save_imgs = save_imgs
        self._base_dir = os.path.dirname(filename)
        file_utils.ensure_dir(self._base_dir)
        self.idx = 0

        self._writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*'DIVX'),
            frame_rate,
            tuple(output_size)
        )

    def write(self, img: np.ndarray):
        """
        Write image to stream
        """

        if self._save_imgs:
            cv2.imwrite(os.path.join(self._base_dir, "{}.jpg".format(self.idx)), img)
            self.idx += 1

        self._writer.write(img)

    def finish(self):
        self._writer.release()
        self._logger.info("VideoWriter finished.")
