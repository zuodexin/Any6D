import copy
import hashlib
import os
import pickle
import cv2
from einops import rearrange
import ipdb
import numpy as np
import pandas as pd
import skimage
import os.path as osp
import torch
from torch.utils.data import Dataset
from tqdm import tqdm, trange
from mmcv.image.io import imread
from bop_toolkit_lib import inout
from src.data.megapose.obj_ds.ov9d_dataset import OV9DObjectDataset
from src.utils.inout import convert_list_to_dataframe


from src.utils.logging import get_logger
from src.utils.mask_utils import binary_mask_to_rle
from src.utils.misc import prepare_dir

logger = get_logger(__name__)


class OV9DSceneDataset(Dataset):

    def __init__(
        self,
        root_dir: str,
        split="oo3d9dsingle",
        detection_threshold=0.0,
        min_visib=0.0,
        choose_obj=[],
        top_k_per_obj=100,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.split = split
        self.detection_threshold = detection_threshold
        self.min_visib = min_visib
        self.choose_obj = choose_obj
        self.top_k_per_obj = top_k_per_obj
        self.obj_ds = OV9DObjectDataset(self.root_dir)

        self.scene_ids = self.get_scene_ids()
        self.obj_ids = self.obj_ds.get_obj_ids()

        self.build_index()

    def get_scene_ids(self):
        with open(osp.join(self.root_dir, f"{self.split}.txt"), "r") as f:
            scene_ids = [line.strip().replace(".tgz", "") for line in f.readlines()]
        return scene_ids

    def get_signature(self):
        hashed_file_name = hashlib.md5(
            (
                "{}_{}_{}_{}_{}_scene".format(
                    self.split,
                    self.detection_threshold,
                    self.min_visib,
                    self.top_k_per_obj,
                    "-".join(
                        str(self.choose_obj) if len(self.choose_obj) > 0 else ["all"]
                    ),
                )
            ).encode("utf-8")
        ).hexdigest()
        return hashed_file_name

    def load_detections_(self):
        # use gt detection
        instance_id = -1
        num_instances_without_valid_segmentation = 0
        detections = {}
        print("Loading detections from ground truth annotations...")
        for scene_id, scene_name in enumerate(tqdm(self.scene_ids)):
            gt_info = inout.load_scene_gt_info(
                f"{self.root_dir}/{self.split}/{scene_name}/scene_gt_info.json"
            )
            scene_gt = inout.load_scene_gt(
                f"{self.root_dir}/{self.split}/{scene_name}/scene_gt.json"
            )
            for im_id in sorted(scene_gt.keys()):
                info = gt_info[im_id]
                for gt_id, inst_gt in enumerate(scene_gt[im_id]):
                    obj_id = inst_gt["obj_id"]
                    visibility = info[gt_id]["visib_fract"]
                    instance_id += 1
                    if visibility < self.min_visib:
                        continue
                    if len(self.choose_obj) > 0 and obj_id not in self.choose_obj:
                        continue
                    cam_R_m2c = np.array(inst_gt["cam_R_m2c"]).reshape(3, 3)
                    cam_t_m2c = np.array(inst_gt["cam_t_m2c"]).reshape(3)
                    TCO = np.eye(4, dtype=np.float32)
                    TCO[:3, :3] = cam_R_m2c
                    TCO[:3, 3] = cam_t_m2c / 1000  # mm to m
                    bbox = gt_info[im_id][gt_id]["bbox_obj"]

                    instance = dict(
                        bbox=bbox,  # in xywh format
                        obj_id=obj_id,
                        label=f"ov9d_{obj_id:06d}",
                        score=1.0,
                        scene_id=scene_id,
                        im_id=int(im_id),
                        scene_name=scene_name,
                        time=-1,
                        gt_id=gt_id,
                        det_id=-1,
                        TCO=TCO,
                        visibility=visibility,
                    )
                    # cache mask
                    mask_visib_file = f"{self.root_dir}/{self.split}/{scene_name}/mask_visib/{im_id:06d}_{gt_id:06d}.png"
                    mask_file = f"{self.root_dir}/{self.split}/{scene_name}/mask/{im_id:06d}_{gt_id:06d}.png"
                    assert osp.exists(mask_file), mask_file
                    assert osp.exists(mask_visib_file), mask_visib_file
                    mask_single = imread(mask_visib_file, "unchanged")
                    area = mask_single.sum()
                    if area <= 64:  # filter out too small or nearly invisible instances
                        num_instances_without_valid_segmentation += 1
                        continue
                    mask_rle = binary_mask_to_rle(mask_single, compressed=True)
                    mask_full = imread(mask_file, "unchanged")
                    mask_full = mask_full.astype("bool")
                    mask_full_rle = binary_mask_to_rle(mask_full, compressed=True)
                    instance.update(
                        dict(
                            mask_visib_rle=mask_rle,
                            mask_amodal_rle=mask_full_rle,
                        )
                    )
                    detections.setdefault(scene_id, {}).setdefault(
                        int(im_id), []
                    ).append(instance)
        print(
            "num_instances_without_valid_segmentation:",
            num_instances_without_valid_segmentation,
        )

        return detections

    def build_index(self):
        cache_path = f".cache/ov9d_scene_{self.get_signature()}.pkl"
        if osp.exists(cache_path):
            print("get scene from cache file: {}".format(cache_path))
            metaDatas = pickle.load(open(cache_path, "rb"))
            print("num scene: {}".format(len(metaDatas)))
            assert len(metaDatas) > 0
        else:
            detections = self.load_detections_()
            pbar = tqdm(self.scene_ids)
            metaDatas = []
            for scene_id, scene_name in enumerate(pbar):
                scene_camera = inout.load_scene_camera(
                    f"{self.root_dir}/{self.split}/{scene_name}/scene_camera.json"
                )
                scene_gt = inout.load_scene_gt(
                    f"{self.root_dir}/{self.split}/{scene_name}/scene_gt.json"
                )
                for im_id in sorted(scene_gt.keys()):
                    pbar.set_description(f"loading scene_{scene_name}_{im_id}")
                    K = scene_camera[im_id]["cam_K"]
                    depth_scale = scene_camera[im_id]["depth_scale"]
                    if scene_id in detections and im_id in detections[scene_id]:
                        metaDatas.append(
                            dict(
                                scene_id=scene_id,
                                scene_name=scene_name,
                                im_id=int(im_id),
                                K=K,
                                depth_scale=depth_scale,
                                instances=detections[scene_id][im_id],
                            )
                        )
            prepare_dir(osp.dirname(cache_path))
            pickle.dump(metaDatas, open(cache_path, "wb"))
        self.metaDatas = convert_list_to_dataframe(metaDatas)
        logger.info(f"Loaded {len(self.metaDatas)} images!")

    def __len__(self):
        return len(self.metaDatas)

    def __getitem__(self, idx: int):

        meta_data = self.metaDatas.iloc[idx]

        scene_id = meta_data["scene_id"]
        scene_name = meta_data["scene_name"]
        im_id = meta_data["im_id"]
        depth_scale = meta_data["depth_scale"]

        rgb = skimage.io.imread(
            f"{self.root_dir}/{self.split}/{scene_name}/rgb/{im_id:06d}.png"
        )
        depth = (
            cv2.imread(
                f"{self.root_dir}/{self.split}/{scene_name}/depth/{im_id:06d}.png",
                cv2.IMREAD_UNCHANGED,
            ).astype(np.float32)
            * depth_scale
            * 0.001
        )

        return dict(
            rgb=rearrange(rgb, "h w c -> c h w"),
            depth=depth.astype(np.float32),
        )


# python -m src.data.ov9d.scene_dataset
if __name__ == "__main__":
    scene_dataset = OV9DSceneDataset("data/OV9D", choose_obj=[])
    print(len(scene_dataset))

    for i in trange(len(scene_dataset)):
        scene_dataset[i]
