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
import subprocess
from dataclasses import dataclass, field
from omegaconf import MISSING, OmegaConf
from typing import List, Any, Optional, Tuple, Union
from enum import Enum
from configs import dataset_config
from com_utils import file_utils


dataset_folder = {
    "linemod": "linemod",
    "ml2_sensim": "magicleap"
}


class ModelEnum(Enum):
    FFB6D = "ffb6d"
    DenseFusion = "densefusion"
    PVN3D = "pvn3d"
    SimpleReg = "simplereg"


def create_list_field(lst: List[Any] = []) -> field:
    """Create a field for a list.

    Args:
        lst (List[Any], optional): input list. Defaults to [].

    Returns:
        field: output filed
    """
    return field(default_factory=lambda: lst)


def set_if_empty(field: field, value):
    if not field:
        return value
    return field


@dataclass
class RandLA:
    """
    Configurations fro RandLA network
    """
    k_n: int = 16  # KNN
    num_layers: int = 4  # Number of layers
    num_points: int = 480 * 640 // 24  # Number of input points
    num_classes: int = 22  # Number of valid classes
    sub_grid_size: float = 0.06  # preprocess_parameter

    batch_size: int = 3  # batch_size during training
    val_batch_size: int = 3  # batch_size during validation and test
    train_steps: int = 500  # Number of steps per epochs
    val_steps: int = 100  # Number of validation steps per epoch
    in_c: int = 9  # input channels [x, y, z, r, g, b]
    # Typed list can hold Any, int, float, bool, str and Enums as well
    # as arbitrary Structured configs
    ints: List[int] = field(default_factory=lambda: [10, 20, 30])
    bools: Tuple[bool, bool] = field(default_factory=lambda: (True, False))
    sub_sampling_ratio: List[int] = field(default_factory=lambda: [4, 4, 4, 4])
    d_out: List[int] = field(default_factory=lambda: [32, 64, 128, 256])  # feature dimension

    num_sub_points: List[int] = create_list_field()

    def __post_init__(self):
        self.num_sub_points = set_if_empty(self.num_sub_points, [
            self.num_points // 4,
            self.num_points // 16,
            self.num_points // 64,
            self.num_points // 256
        ])


@dataclass
class TestConfig:
    test_pose: bool = True
    test_gt: bool = False


@dataclass
class TrainConfig:
    deterministic: bool = False
    opt_level: str = "O0"
    lr: float = 0.5
    lr_decay: float = 0.5
    weight_decay: float = 0
    decay_step: int = 200000
    bn_momentum: float = 0.9
    bn_decay: float = 0.5


@dataclass
class WandB:
    enable: bool = True
    log_imgs: bool = False
    log_img_per_eval: int = 5
    project_name: str = "ope_ffb6d"


@dataclass
class DataConfig:
    # common data config fields
    index_filename: str = ""
    index_filenames: List[str] = create_list_field()
    is_training: bool = True
    subsampling: int = 1
    seed: Optional[int] = None

    def __post_init__(self):
        # ensure backwards capability
        if self.index_filename:
            self.index_filenames = [self.index_filename]
        elif len(self.index_filenames)==1:
            self.index_filename = self.index_filenames[0]


@dataclass
class TrainDataConfig(DataConfig):
    # train data specific configurations
    index_filenames: List[str] = create_list_field(["train.txt"])
    is_training: bool = True


@dataclass
class TestDataConfig(DataConfig):
    # test data specific configurations
    index_filenames: List[str] = create_list_field(["test.txt"])
    is_training: bool = False


@dataclass
class Config:
    name: str = MISSING
    group: str = ""
    description: str = MISSING
    git_hash: str = ""

    randla: RandLA = RandLA()
    test: TestConfig = TestConfig()
    train: TrainConfig = TrainConfig()
    wandb: WandB = WandB()

    train_data: DataConfig = TrainDataConfig()
    test_data: DataConfig = TestDataConfig()

    depth_src: str = "user_rgb"
    img_src: str = "user_rgb"
    projective_depth: bool = False

    width: int = 0
    height: int = 0

    dataset_name: str = MISSING
    cls_type: str = MISSING

    model: ModelEnum = ModelEnum.FFB6D
    exp_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    log_dir: str = ""
    dataset_base_dir: str = "/home/pxu/pose_dataset/linemod"

    n_total_epoch: int = 25
    mini_batch_size: int = 3
    val_mini_batch_size: int = 3
    test_mini_batch_size: int = 1
    evals_per_clr_up: int = 6
    evals_per_clr_down: int = 15
    num_of_lr_cycles: int = 1

    ms_radius: float = 0.08

    n_sample_points: int = 480 * 640 // 24  # Number of input points
    n_keypoints: int = 8

    noise_trans: float = 0.05  # range of the random noise of translation added to the training data

    intrinsic_matrix: List[int] = create_list_field()

    n_objects: int = 1 + 1
    n_classes: int = 1 + 1

    use_orbfps: bool = True

    use_normalmap: bool = True
    nrm_max_depth_mm: int = 20000
    nrm_max_step_mm: int = 500

    # Use sythetic renderings or fused examples for training (if available)
    use_render_data: bool = False
    use_fuse_data: bool = False

    def __post_init__(self):
        if not self.description:
            raise ValueError("Please give your experiment a description!")

        self.resnet_ptr_mdl_p = os.path.abspath(
            os.path.join(
                self.exp_dir,
                "models/cnn/ResNet_pretrained_mdl"
            )
        )
        file_utils.ensure_dir(self.resnet_ptr_mdl_p)

        # store git commit hash if available
        try:
            self.git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
        except subprocess.CalledProcessError:
            pass

        self.log_dir = set_if_empty(
            self.log_dir,
            os.path.abspath(os.path.join(self.exp_dir, 'train_log'))
        )
        self.exp_log_dir = os.path.join(self.log_dir,self.group,self.name)

        self.dataset_base_dir = set_if_empty(
            self.dataset_base_dir,
            os.path.abspath(
                os.path.join(self.exp_dir, "datasets", dataset_folder[self.dataset_name])
            )
        )

        file_utils.ensure_dir(self.exp_log_dir)
        self.log_model_dir = os.path.join(self.exp_log_dir, 'checkpoints', self.cls_type)
        file_utils.ensure_dir(self.log_model_dir)
        self.log_eval_dir = os.path.join(self.exp_log_dir, 'eval_results', self.cls_type)
        file_utils.ensure_dir(self.log_eval_dir)
        self.log_traininfo_dir = os.path.join(self.exp_log_dir, 'train_info', self.cls_type)
        file_utils.ensure_dir(self.log_traininfo_dir)
        self.log_debug_imgs_dir = os.path.join(self.exp_log_dir, 'imgs', self.cls_type)
        file_utils.ensure_dir(self.log_debug_imgs_dir)

        self.dataset = dataset_config.DatasetConfig(self)

        if self.use_normalmap:
            self.randla.in_c = 9
        else:
            print("Warning: Using point features without normal map")
            self.randla.in_c = 6


def create_config(config_overwrite: Union[str, dict]) -> Config:
    """
    Create configuration from Config class with file or dict overwrites
    """
    cfg = OmegaConf.structured(Config)
    if isinstance(config_overwrite, str):
        from_file = OmegaConf.load(config_overwrite)
        cfg = OmegaConf.merge(cfg, from_file)
    elif isinstance(config_overwrite, dict):
        from_dict = OmegaConf.create(config_overwrite)
        cfg = OmegaConf.merge(cfg, from_dict)

    return OmegaConf.to_object(cfg)


def save_config(cfg: Config, save_dir: str = None):
    """
    Save full config object to config.yaml
    # """
    if not save_dir:
        save_dir = cfg.log_traininfo_dir

    with open(os.path.join(cfg.log_traininfo_dir, "config.yaml"), "w") as fp:
        OmegaConf.save(config=cfg, f=fp.name)


if __name__ == "__main__":
    cfg = create_config("configs/example.yaml")
