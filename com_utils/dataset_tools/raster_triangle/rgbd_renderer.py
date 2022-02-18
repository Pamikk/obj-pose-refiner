#!/usr/bin/env python3
import os
import pickle as pkl
import numpy as np
from plyfile import PlyData
import cv2
import random
from random import randint
from tqdm import tqdm
from glob import glob
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from ope_utils.dataset_tools import dataset_config
from ope_utils import visualization_utils, mesh_utils
import config

FPATH = os.path.dirname(os.path.abspath(__file__))


class RenderDB:
    def __init__(self, cls_type, render_num=10, dataset="linemod", vis=False):

        cfg = config.create_config({
            "name": "RenderDB",
            "description": "RenderDB",
            "dataset_name": dataset,
            "cls_type": cls_type,
        })

        self.dataset_config = (
            dataset_config.DatasetConfig(cfg)
        )
        self.h, self.w = self.dataset_config.h, self.dataset_config.w

        self.vis = vis
        self.cls_type = cls_type
        self.dataset = dataset
        self.cls_id = self.dataset_config.obj_id_dict[cls_type]

        self.render_dir = os.path.join(self.dataset_config.data_dir, "renders", cls_type)
        if not os.path.exists(self.render_dir):
            os.makedirs(self.render_dir)
        self.render_num = render_num

        self.dll = np.ctypeslib.load_library(
            os.path.join(FPATH, "rastertriangle_so.so"), "."
        )

        self.bg_img_pth_lst = glob(
            os.path.join(FPATH, "SUN2012pascalformat", "JPEGImages", "*.jpg")
        )

        print("begin loading '{}' render set:".format(cls_type))

        if dataset == "linemod":
            RT_pth = os.path.join(
                FPATH, "sampled_poses", "{}_sampled_RTs.pkl".format(cls_type)
            )
            self.RT_lst = pkl.load(open(RT_pth, "rb"))
        elif dataset == "ml2_sensim":
            # Using existing poses from LineMod 'ape' class. Will be generated
            # for ml2_sensim classes if necessary
            RT_pth = os.path.join(
                FPATH, "sampled_poses", "{}_sampled_RTs.pkl".format("ape")
            )
            self.RT_lst = pkl.load(open(RT_pth, "rb"))
            # adjust translation to account for the different object size
            for RT in self.RT_lst:
                RT[:, 3] = RT[:, 3] * [2, 2, 6.0]

        self.render_dir = os.path.join(self.dataset_config.data_dir, "renders", cls_type)
        if not os.path.exists(self.render_dir):
            os.makedirs(self.render_dir)

        random.seed(19763)
        if render_num < len(self.RT_lst):
            random.shuffle(self.RT_lst)
            self.RT_lst = self.RT_lst[:render_num]

        (
            self.npts,
            self.xyz,
            self.r,
            self.g,
            self.b,
            self.n_face,
            self.face,
        ) = mesh_utils.load_ply_model(
            self.dataset_config.ptxyz_pth,
            scale2m=self.dataset_config.scale_m,
            ret_dict=False
        )
        self.face = np.require(self.face, "int32", "C")
        self.r = np.require(np.array(self.r), "float32", "C")
        self.g = np.require(np.array(self.g), "float32", "C")
        self.b = np.require(np.array(self.b), "float32", "C")

    def gen_pack_zbuf_render(self):
        pth_lst = []

        for idx, RT in tqdm(enumerate(self.RT_lst)):
            meshc = {
                "xyz": self.xyz.copy(),
                "r": self.r,
                "g": self.g,
                "b": self.b,
                "n_face": self.n_face,
                "face": self.face,
            }
            rgb, zbuf, mask = visualization_utils.draw_model(
                meshc, self.dataset_config.K, RT, (self.h, self.w)
            )

            bg = None
            len_bg_lst = len(self.bg_img_pth_lst)
            while bg is None or len(bg.shape) < 3:
                bg_pth = self.bg_img_pth_lst[randint(0, len_bg_lst - 1)]
                bg = cv2.cvtColor(cv2.imread(bg_pth), cv2.COLOR_BGR2RGB)
                if len(bg.shape) < 3:
                    bg = None
                    continue
                bg_h, bg_w, _ = bg.shape
                if bg_h < self.h:
                    new_w = int(float(self.h) / bg_h * bg_w)
                    bg = cv2.resize(bg, (new_w, self.h))
                bg_h, bg_w, _ = bg.shape
                if bg_w < self.w:
                    new_h = int(float(self.w) / bg_w * bg_h)
                    bg = cv2.resize(bg, (self.w, new_h))
                bg_h, bg_w, _ = bg.shape
                if bg_h > self.h:
                    sh = randint(0, bg_h - self.h)
                    bg = bg[sh : sh + self.h, :, :]
                bg_h, bg_w, _ = bg.shape
                if bg_w > self.w:
                    sw = randint(0, bg_w - self.w)
                    bg = bg[:, sw : sw + self.w, :]

            msk_3c = np.repeat(mask[:, :, None], 3, axis=2)
            rgb = bg * (msk_3c <= 0).astype(bg.dtype) + rgb * (msk_3c).astype(bg.dtype)

            if self.vis:
                show_zbuf = zbuf.copy()
                min_d, max_d = show_zbuf[show_zbuf > 0].min(), show_zbuf.max()
                show_zbuf[show_zbuf > 0] = (
                    (show_zbuf[show_zbuf > 0] - min_d) / (max_d - min_d) * 255
                )
                show_zbuf = show_zbuf.astype(np.uint8)
                show_mask = (mask / mask.max() * 255).astype("uint8")
                vis = np.hstack(
                    (
                        rgb.astype("uint8"),
                        np.stack((show_zbuf,) * 3, axis=-1),
                        np.stack((show_mask,) * 3, axis=-1),
                    )
                )
                plt.imshow(vis)
                plt.show()

            data = {}
            data["depth"] = zbuf
            data["rgb"] = rgb
            data["mask"] = mask
            data["K"] = self.dataset_config.K
            data["RT"] = RT
            data["cls_typ"] = self.cls_type
            data["rnd_typ"] = "render"
            sv_pth = os.path.join(self.render_dir, "{}.pkl".format(idx))
            pkl.dump(data, open(sv_pth, "wb"))
            pth_lst.append(os.path.abspath(sv_pth))

        plst_pth = os.path.join(self.render_dir, "file_list.txt")
        with open(plst_pth, "w") as of:
            for pth in pth_lst:
                print(pth, file=of)


def main():
    parser = ArgumentParser("Render dataset with object on random background")
    parser.add_argument(
        "--cls",
        type=str,
        default="ape",
        help="Target object from {ape, benchvise, cam, can, cat, driller, duck, \
        eggbox, glue, holepuncher, iron, lamp, phone} (default ape)",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="linemod",
        help="Dataset name (linemod, sensim)",
    )
    parser.add_argument(
        "--render_num",
        type=int,
        default=7000,
        help="Number of images you want to generate.",
    )
    parser.add_argument(
        "--vis", action="store_true", help="Visualize generated images."
    )
    args = parser.parse_args()

    print("create {} samples of cls '{}'".format(args.render_num, args.cls))
    gen = RenderDB(
        cls_type=args.cls,
        render_num=args.render_num,
        dataset=args.dataset,
        vis=args.vis,
    )
    gen.gen_pack_zbuf_render()


if __name__ == "__main__":
    main()
