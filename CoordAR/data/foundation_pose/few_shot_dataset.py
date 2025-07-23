from functools import partial
import hashlib
import os
import pickle
import random
import time
from einops import einsum, rearrange
import ipdb
import numpy as np
import torch
from tqdm import tqdm, trange
from torch.utils.data import Dataset
import os.path as osp
from torchvision import transforms
from multiprocessing import Pool
from scipy.spatial.transform import Rotation

from src.data.augmentation.depth_bp_augmentor import DepthBPAugmentor
from src.data.augmentation.roc_augmentor import ROCAugmentor
from src.data.collate_importer import default_collate_fn
from src.data.foundation_pose.instance_dataset import FoundationPoseInstance
from src.data.foundation_pose.scene_dataset import FoundationPoseScene
from src.data.megapose.obj_ds.gso_dataset import GoogleScannedObjectDataset
from src.data.megapose.obj_ds.objaverse_dataset import ObjaverseDataset
from src.models.memchip.visualization import show_batch
from src.utils.inout import convert_list_to_dataframe
from src.utils.logging import get_logger
from src.utils.misc import prepare_dir
from src.utils.pysixd.RT_transform import (
    allocentric_to_egocentric,
    egocentric_to_allocentric,
)

logger = get_logger(__name__)


class FoundationPoseFewShot(Dataset):

    def __init__(
        self,
        instance_ds: FoundationPoseInstance,
        num_samples=1000,
        num_ref=1,
        split="train",
        random_base_rot=False,
        roc_augmentor=None,
        depth_bp_augmentor=None,
        no_cache=False,
        shuffle_indices=True,
        unique_queries=False,
    ):
        super().__init__()

        self.instance_ds = instance_ds
        self.num_samples = num_samples
        self.num_ref = num_ref
        self.split = split
        self.roc_augmentor = roc_augmentor
        self.depth_bp_augmentor = depth_bp_augmentor
        self.random_base_rot = random_base_rot
        self.normalize_by = instance_ds.normalize_by
        self.no_cache = no_cache
        self.shuffle_indices = shuffle_indices
        self.unique_queries = unique_queries

        self.build_index()

    def get_signature(self):
        hash_code = hashlib.md5(
            f"{self.instance_ds.get_signature()}_{self.num_samples}_{self.num_ref}_{self.split}_{self.unique_queries}".encode(
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
        if not self.no_cache and osp.exists(cache_path):
            print("get instances from cache file: {}".format(cache_path))
            metaDatas = pickle.load(open(cache_path, "rb"))
            print("num instances: {}".format(len(metaDatas)))
            assert len(metaDatas) > 0
        else:
            print("building few shot indices ...")
            # group by obj label
            instances = self.instance_ds.metaDatas
            labels = sorted(instances["label"].unique())
            num_train = int(len(labels) * 0.9)
            if self.split == "train":
                labels = labels[:num_train]
            elif self.split == "test":
                labels = labels[num_train:]
            elif self.split == "all":
                labels = labels
            self.save_label_list(labels, os.path.dirname(cache_path))
            # sample examples
            label_to_indices = instances.groupby("label").indices
            if (
                self.unique_queries
            ):  # when evaluating on real275 or tyol, we need unique queries
                metaDatas = []
                query_indices = set()
                while len(metaDatas) < self.num_samples and len(query_indices) < len(
                    instances
                ):
                    query_id = np.random.choice(
                        instances.index,
                        size=1,
                        replace=False,
                    ).tolist()[0]
                    label = instances.iloc[query_id]["label"]
                    ref_indices = np.random.choice(
                        label_to_indices[label],
                        size=self.num_ref,
                        replace=False,
                    ).tolist()
                    indices = ref_indices + [query_id]  # last one is query
                    if query_id in query_indices:
                        continue
                    query_indices.add(query_id)
                    metaDatas.append(
                        {
                            "label": label,
                            "indices": indices,
                        }
                    )
            else:
                selected_labels = np.random.choice(labels, size=self.num_samples)
                metaDatas = [
                    {
                        "label": label,
                        "indices": np.random.choice(
                            label_to_indices[label],
                            size=self.num_ref + 1,
                            replace=False,
                        ).tolist(),
                    }
                    for label in tqdm(selected_labels)
                ]
            prepare_dir(osp.dirname(cache_path))
            pickle.dump(metaDatas, open(cache_path, "wb"))
        self.metaDatas = convert_list_to_dataframe(metaDatas)
        logger.info(f"Loaded {len(self.metaDatas)} samples!")

    def get_test_targets(self):
        # generate test_targets_bop19 for tyol and real275 using sampled image pairs
        instances = []
        logger.info("Generating test targets...")
        for i in trange(len(self)):
            assert (
                not self.shuffle_indices
            ), "Indices should not be shuffled for test targets generation."
            query_index = self.metaDatas.iloc[i]["indices"][-1]  # last one is query
            scene_ds = self.instance_ds.scene_ds
            instance_meta = self.instance_ds.metaDatas.iloc[query_index]
            scene_idx = instance_meta["scene_idx"]
            instance_idx = instance_meta["instance_idx"]
            scene_meta = scene_ds.metaDatas.iloc[scene_idx]
            scene_instances = scene_meta["instances"]

            instance = scene_instances[instance_idx]
            instances.append(
                dict(
                    scene_id=instance["scene_id"],
                    im_id=instance["im_id"],
                    obj_id=instance["obj_id"],
                    gt_id=instance["gt_id"],
                )
            )
        instance_cnt = {}
        for instance in instances:
            scene_id = instance["scene_id"]
            im_id = instance["im_id"]
            obj_id = instance["obj_id"]
            gt_id = instance["gt_id"]

            instance_cnt.setdefault(scene_id, {}).setdefault(im_id, {}).setdefault(
                obj_id, set()
            ).add(gt_id)

        test_target = []
        for scene_id, im_dict in sorted(instance_cnt.items()):
            for im_id, obj_dict in sorted(im_dict.items()):
                for obj_id, gt_ids in sorted(obj_dict.items()):
                    test_target.append(
                        dict(
                            scene_id=scene_id,
                            im_id=im_id,
                            obj_id=obj_id,
                            inst_count=len(gt_ids),
                        )
                    )
        return test_target

    def __len__(self):
        return len(self.metaDatas)

    def __getitem__(self, idx):
        indices = self.metaDatas.iloc[idx]["indices"]
        label = self.metaDatas.iloc[idx]["label"]

        base_rot = np.eye(3, dtype=np.float32)
        if self.random_base_rot:
            base_rot = Rotation.random().as_matrix().astype(np.float32)

        def apply_base_rot(nocs, rot, base_rot):
            h, w = nocs.shape[1:]
            nocs = nocs - 0.5
            nocs = rearrange(nocs, "c h w -> c (h w)")
            nocs = base_rot @ nocs + 0.5
            nocs = nocs.reshape(3, h, w)
            rot = rot @ base_rot.T
            return nocs, rot

        instance_list = []
        for i in indices:
            instance = self.instance_ds[i]
            rgb = instance["rgb_patch"]
            depth = instance["depth_patch"]
            nocs = instance["nocs"]
            mask = instance["vis_mask_patch"]
            roc = instance["roc"]
            TCO = instance["TCO"]
            rot = instance["TCO"][:3, :3]
            rot_allo = egocentric_to_allocentric(TCO)[:3, :3]
            extents = instance["extents"]
            diameter = instance["diameter"]
            if self.normalize_by == "diameter":
                obj_size = diameter
            elif self.normalize_by == "max_axis":
                obj_size = np.max(extents) * 1.1
            else:
                raise ValueError(f"Unknown normalization type: {self.normalize_by}")

            model_center = instance["model_center"]
            nocs, rot = apply_base_rot(nocs, rot, base_rot)
            instance_list.append(
                dict(
                    rgb=rgb,
                    depth=depth,
                    nocs=nocs,
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
                    scene_id=instance["scene_id"],
                    im_id=instance["im_id"],
                    obj_id=instance["obj_id"],
                    gt_id=instance["gt_id"],
                    det_id=instance["det_id"],
                    score=instance["score"],
                    bbox=instance["bbox"],
                    depth_bp=instance["depth_bp"],
                )
            )

        # permute
        if self.shuffle_indices:
            instance_list = np.random.permutation(instance_list)
        # last as query
        query = instance_list[-1]
        templates = instance_list[:-1]

        """
            augment template roc and depth_bp
        """
        for template in templates:
            if self.roc_augmentor is not None:
                roc = template["roc"]
                roc = rearrange(roc, "c h w -> h w c")  # to HWC
                roc = self.roc_augmentor(roc)
                roc = rearrange(roc, "h w c -> c h w")  # back to CHW
                template["roc"] = roc

            if self.depth_bp_augmentor is not None:
                depth_bp = template["depth_bp"]
                depth_bp = rearrange(depth_bp, "c h w -> h w c")
                depth_bp = self.depth_bp_augmentor(depth_bp)
                depth_bp = rearrange(depth_bp, "h w c -> c h w")
                template["depth_bp"] = depth_bp

        """
            augment query depth_bp
        """
        if self.depth_bp_augmentor is not None:
            depth_bp = query["depth_bp"]
            depth_bp = rearrange(depth_bp, "c h w -> h w c")
            depth_bp = self.depth_bp_augmentor(depth_bp)
            depth_bp = rearrange(depth_bp, "h w c -> c h w")
            query["depth_bp"] = depth_bp

        template_data = {}

        stack = lambda key: np.stack([templates[i][key] for i in range(len(templates))])
        template_data["template_imgs"] = stack("rgb")
        template_data["template_depths"] = stack("depth")
        template_data["template_nocs"] = stack("nocs")
        template_data["template_rocs"] = stack("roc")
        template_data["template_masks_visib"] = stack("mask")
        template_data["template_rots"] = stack("rot")
        template_data["template_tco"] = stack("TCO")
        template_data["template_K_crop"] = stack("K_crop")
        template_data["template_depth_bp"] = stack("depth_bp")

        # relative data
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

        roc = stack("roc")
        TCO = stack("TCO")
        mask = stack("mask")
        trans = TCO[:, :3, 3]
        rel_pose = TCO[0:1] @ np.linalg.inv(TCO)  # first as reference
        obj_size = stack("obj_size").reshape(-1, 1, 1, 1)
        rel_rocs = (
            (
                einsum(
                    rel_pose[:, :3, :3],
                    obj_size * (roc - 0.5) + trans[:, :, None, None],
                    "b i j, b j h w -> b i h w",
                )
                + rel_pose[:, :3, 3, None, None]
            )
            - trans[:1, :, None, None]
        ) * mask[:, None] / (obj_size + 1e-6) + 0.5
        template_data["template_rel_rocs"] = rel_rocs[:-1]
        template_data["template_rel_scale"] = rel_scale_in_crop[:-1]
        template_data["template_rot_allo"] = rot_allo[:-1]
        template_data["template_rel_rot_allo"] = rel_rot_allo[:-1]
        template_data["template_center_in_crop"] = rel_center_in_crop[:-1]

        data = dict(
            scene_id=query["scene_id"],
            im_id=query["im_id"],
            obj_id=query["obj_id"],
            gt_id=query["gt_id"],
            det_id=query["det_id"],
            bbox=query["bbox"],
            score=query["score"],
            query=query["rgb"],
            query_rot=query["rot"],
            query_depth=query["depth"],
            query_nocs=query["nocs"],
            query_mask=query["mask"],
            query_roc=query["roc"],
            query_rel_roc=rel_rocs[-1],
            query_center_in_crop=rel_center_in_crop[-1],
            query_rel_z=rel_scale_in_crop[-1],
            query_rot_allo=rot_allo[-1],
            query_rel_rot_allo=rel_rot_allo[-1],
            query_depth_bp=query["depth_bp"],
            extents=query["extents"],
            model_center=query["model_center"],
            diameter=query["diameter"],
            obj_size=query["obj_size"],
            query_TCO=query["TCO"],
            query_K_crop=query["K_crop"],
            points=query["points"],
            symmetries=query["symmetries"],
        )
        data.update(dict(**template_data))
        return data


def get_dataset_gso():
    scene_dataset = FoundationPoseScene(root_dir="data/foundation_pose/gso")
    print(len(scene_dataset))
    obj_ds = GoogleScannedObjectDataset("data/MegaPose-GSO/google_scanned_objects")
    dzi_config = dict(
        dzi_type="uniform", dzi_pad_scale=1.5, dzi_scale_ratio=0.0, dzi_shift_ratio=0.0
    )
    instance_dataset = FoundationPoseInstance(
        scene_dataset,
        obj_ds,
        dzi_config=dzi_config,
        load_cad=False,
        diameter="estimated",
        min_visib=0.5,
    )
    print(len(instance_dataset))
    roc_augmentor = ROCAugmentor()
    depth_bp_augmentor = DepthBPAugmentor()

    few_shot_ds = FoundationPoseFewShot(
        instance_dataset,
        random_base_rot=True,
        roc_augmentor=roc_augmentor,
        depth_bp_augmentor=depth_bp_augmentor,
    )
    return few_shot_ds


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

    roc_augmentor = ROCAugmentor()

    few_shot_ds = FoundationPoseFewShot(
        instance_dataset, random_base_rot=True, roc_augmentor=roc_augmentor
    )
    return few_shot_ds


# python -m src.data.foundation_pose.few_shot_dataset
if __name__ == "__main__":

    few_shot_ds = get_dataset_gso()

    for item in tqdm(few_shot_ds):
        break

    data_loader = torch.utils.data.DataLoader(
        few_shot_ds,
        num_workers=8,
        collate_fn=default_collate_fn,
        batch_size=8,
    )

    for batch in tqdm(data_loader):
        pass
        show_batch(batch)
        ipdb.set_trace()
