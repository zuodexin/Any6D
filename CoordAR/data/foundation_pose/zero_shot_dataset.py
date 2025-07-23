from functools import partial
import hashlib
import os
import pickle
import random
import time
from einops import einsum, rearrange
import ipdb
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm, trange
from torch.utils.data import Dataset
import os.path as osp
from torchvision import transforms
from multiprocessing import Pool
from scipy.spatial.transform import Rotation
from torchvision.utils import make_grid

from src.data.collate_importer import default_collate_fn
from src.data.foundation_pose.instance_dataset import FoundationPoseInstance
from src.data.foundation_pose.scene_dataset import FoundationPoseScene
from src.data.megapose.obj_ds.gso_dataset import GoogleScannedObjectDataset
from src.data.megapose.obj_ds.objaverse_dataset import ObjaverseDataset
from src.utils.inout import convert_list_to_dataframe
from src.utils.logging import get_logger
from src.utils.misc import prepare_dir
from src.utils.pysixd.RT_transform import (
    allocentric_to_egocentric,
    egocentric_to_allocentric,
)

logger = get_logger(__name__)


class FoundationPoseZeroShot(Dataset):

    def __init__(
        self,
        instance_ds: FoundationPoseInstance,
        num_samples=1000,
        num_ref=5,
        split="train",
    ):
        super().__init__()

        self.instance_ds = instance_ds
        self.num_samples = num_samples
        self.num_ref = num_ref
        self.split = split
        self.normalize_by = instance_ds.normalize_by

        self.build_index()

    def get_signature(self):
        hash_code = hashlib.md5(
            f"{self.instance_ds.get_signature()}_{self.num_samples}_{self.num_ref}_{self.split}".encode(
                "utf-8"
            )
        ).hexdigest()
        return hash_code

    def save_label_list(self, labels, out_dir):
        with open(f"{out_dir}/few_shot_{self.split}.txt", "w") as fp:
            for label in labels:
                fp.write(f"{label}\n")

    def build_index(self):
        cache_path = f".cache/fewshot_{self.get_signature()}.pkl"
        if osp.exists(cache_path):
            print("get instances from cache file: {}".format(cache_path))
            metaDatas = pickle.load(open(cache_path, "rb"))
            print("num instances: {}".format(len(metaDatas)))
            assert len(metaDatas) > 0
        else:
            print("building few shot indices ...")
            # group by obj label
            instances = self.instance_ds.metaDatas
            labels = sorted(instances["label"].unique())
            # filter labels with less than num_ref + 1 instances
            grouped = self.instance_ds.metaDatas.groupby("label")
            labels = [
                label
                for label in labels
                if len(grouped.get_group(label)) >= self.num_ref + 1
            ]
            num_train = int(len(labels) * 0.9)
            if self.split == "train":
                labels = labels[:num_train]
            else:
                labels = labels[num_train:]
            self.save_label_list(labels, os.path.dirname(cache_path))
            # sample examples
            label_to_indices = instances.groupby("label").indices
            selected_labels = np.random.choice(labels, size=self.num_samples)
            metaDatas = [
                {
                    "label": label,
                    "indices": np.random.choice(
                        label_to_indices[label], size=self.num_ref + 1, replace=False
                    ).tolist(),
                }
                for label in tqdm(selected_labels)
            ]
            prepare_dir(osp.dirname(cache_path))
            pickle.dump(metaDatas, open(cache_path, "wb"))
        self.metaDatas = convert_list_to_dataframe(metaDatas)
        logger.info(f"Loaded {len(self.metaDatas)} samples!")

    def __len__(self):
        return len(self.metaDatas)

    def __getitem__(self, idx):
        indices = self.metaDatas.iloc[idx]["indices"]
        label = self.metaDatas.iloc[idx]["label"]

        instance_list = []
        for i in indices:
            instance = self.instance_ds[i]
            rgb = instance["rgb_patch"]
            depth = instance["depth_patch"]
            mask = instance["vis_mask_patch"]
            roc = instance["roc"]
            TCO = instance["TCO"]
            rot = instance["TCO"][:3, :3]
            rot_allo = egocentric_to_allocentric(TCO)[:3, :3]
            extents = instance["extents"]
            model_center = instance["model_center"]
            diameter = instance["diameter"]
            if self.normalize_by == "diameter":
                obj_size = diameter
            elif self.normalize_by == "max_axis":
                obj_size = np.max(extents) * 1.1
            else:
                raise ValueError(f"Unknown normalization type: {self.normalize_by}")
            instance_list.append(
                dict(
                    rgb=rgb,
                    depth=depth,
                    mask=mask,
                    roc=roc,
                    rot=rot,
                    rot_allo=rot_allo,
                    extents=extents,
                    model_center=model_center,
                    diameter=diameter,
                    obj_size=obj_size,
                    TCO=TCO,
                    K_crop=instance["K_crop"],
                    center_in_crop=instance["center_in_crop"],
                    scale_in_crop=instance["scale_in_crop"],
                    points=instance["points"],
                    symmetries=instance["symmetries"],
                    resize_ratio=instance["resize_ratio"],
                )
            )

        # permute
        instance_list = np.random.permutation(instance_list)

        stack = lambda key: np.stack(
            [instance_list[i][key] for i in range(len(instance_list))]
        ).astype(np.float32)
        center_in_crop = stack("center_in_crop")
        rel_center_in_crop = (
            center_in_crop - center_in_crop[0:1, :]
        )  # first as reference

        scale_in_crop = stack("scale_in_crop").clip(min=1e-3)  # avoid log(0)
        rel_scale_in_crop = np.log(
            scale_in_crop / scale_in_crop[0:1]
        )  # first as reference

        rot_allo = stack("rot_allo")
        rel_rot_allo = rot_allo @ rot_allo[0:1].transpose(0, 2, 1)  # first as reference

        # relative roc
        roc = stack("roc")
        TCO = stack("TCO")
        mask = stack("mask")
        trans = TCO[:, :3, 3]
        rel_pose = TCO[0:1] @ np.linalg.inv(TCO)  # first as reference
        obj_size = stack("obj_size").reshape(-1, 1, 1, 1)
        rel_roc = (
            (
                einsum(
                    rel_pose[:, :3, :3],
                    obj_size * (roc - 0.5) + trans[:, :, None, None],
                    "b i j, b j h w -> b i h w",
                )
                + rel_pose[:, :3, 3, None, None]
            )
            - trans[:1, :, None, None]
        ) * mask[:, None] / obj_size + 0.5

        data = dict(
            rgb=stack("rgb"),
            depth=stack("depth"),
            mask=stack("mask"),
            roc=roc,
            extents=stack("extents"),
            model_center=stack("model_center"),
            TCO=stack("TCO"),
            K_crop=stack("K_crop"),
            center_in_crop=center_in_crop,
            rel_center_in_crop=rel_center_in_crop,
            scale_in_crop=scale_in_crop,
            rel_scale_in_crop=rel_scale_in_crop,
            rot_allo=rot_allo,
            rel_rot_allo=rel_rot_allo,
            rel_roc=rel_roc.clip(0, 1),
            points=instance_list[0]["points"],
            diameter=instance_list[0]["diameter"],
            obj_size=instance_list[0]["obj_size"],
            symmetries=instance_list[0]["symmetries"],
            resize_ratio=stack("resize_ratio"),
        )
        return data


def get_dataset_gso():
    scene_dataset = FoundationPoseScene(root_dir="data/foundation_pose/gso")
    print(len(scene_dataset))
    obj_ds = GoogleScannedObjectDataset("data/MegaPose-GSO/google_scanned_objects")
    dzi_config = dict(
        dzi_type="uniform", dzi_pad_scale=1.5, dzi_scale_ratio=0.0, dzi_shift_ratio=0.0
    )
    instance_dataset = FoundationPoseInstance(
        scene_dataset, obj_ds, dzi_config=dzi_config, load_cad=False
    )
    print(len(instance_dataset))

    zero_shot_ds = FoundationPoseZeroShot(instance_dataset)
    return zero_shot_ds


def get_dataset_objaverse():
    scene_dataset = FoundationPoseScene(
        root_dir="data/foundation_pose/objaverse", subset="objaverse"
    )
    print(len(scene_dataset))
    obj_ds = ObjaverseDataset("data/Objaverse_cache")
    dzi_config = dict(
        dzi_type="uniform", dzi_pad_scale=1.5, dzi_scale_ratio=0.0, dzi_shift_ratio=0.0
    )
    instance_dataset = FoundationPoseInstance(
        scene_dataset, obj_ds, dzi_config=dzi_config, load_cad=False
    )
    print(len(instance_dataset))

    zero_shot_ds = FoundationPoseZeroShot(instance_dataset)
    return zero_shot_ds


def show_batch(batch, log_dir="logs/debug"):
    rgb = batch["rgb"]
    depth = batch["depth"]
    mask = batch["mask"]
    center_in_crop = batch["center_in_crop"]
    rel_center_in_crop = batch["rel_center_in_crop"]
    rel_rot_allo = batch["rel_rot_allo"]
    roc = batch["roc"]
    rel_roc = batch["rel_roc"]

    bs = len(rgb)
    fig, axs = plt.subplots(bs, 5, figsize=(40, 5 * bs))
    if bs == 1:
        axs = [axs]
    for i in range(bs):
        axs[i][0].imshow(make_grid(rgb[i]).permute(1, 2, 0).cpu().numpy())
        axs[i][0].set_title("rgb")

        axs[i][1].imshow(make_grid(roc[i]).permute(1, 2, 0).clip(0, 1).cpu().numpy())
        axs[i][1].set_title("roc")

        axs[i][2].imshow(
            make_grid(rel_roc[i], normalize=True).permute(1, 2, 0).cpu().numpy()
        )
        axs[i][2].set_title("rel roc")

        for j in range(5):
            axs[i][j].axis("off")

    plt.tight_layout()
    plt.savefig(f"{log_dir}/zero_shot_batch.png")
    plt.close()


# python -m src.data.foundation_pose.zero_shot_dataset
if __name__ == "__main__":

    zero_shot_ds = get_dataset_gso()

    for item in tqdm(zero_shot_ds):
        break

    data_loader = torch.utils.data.DataLoader(
        zero_shot_ds,
        num_workers=8,
        collate_fn=default_collate_fn,
        batch_size=8,
    )

    for batch in tqdm(data_loader):
        pass
        show_batch(batch)
        ipdb.set_trace()
