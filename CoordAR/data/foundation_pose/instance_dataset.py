import hashlib
import json
import os
import pickle
import warnings
import cv2
from einops import rearrange
import ipdb
from joblib import Memory
from matplotlib import pyplot as plt
import numpy as np
import skimage
import torch
from torch.utils.data import Dataset
import glob
import os.path as osp
from tqdm import tqdm, trange
from torchvision import transforms

from src import data
from src.data.augmentation.bg_augmentor import BGAugmentor
from src.data.augmentation.color_augmentor import ColorAugmentor
from src.data.bop.instance_dataset import estimate_obj_size
from src.data.collate_importer import default_collate_fn
from src.data.foundation_pose.scene_dataset import FoundationPoseScene
from src.data.megapose.obj_ds.objaverse_dataset import ObjaverseDataset
from src.data.megapose.shapenet import (
    depth_backproject,
    depth_to_roc_map,
    depth_to_xyz,
    normalize_depth_bp_zscore,
)
from src.data.megapose.obj_ds.gso_dataset import GoogleScannedObjectDataset
from src.utils.augment import aug_bbox_DZI
from src.utils.cropping import (
    crop_resize_by_warp_affine,
    get_K_crop_resize,
    get_affine_transform,
)
from src.utils.inout import convert_list_to_dataframe
from src.utils.logging import get_logger
from src.utils.misc import prepare_dir, to_device
from src.utils.pytorch3d.diff_render import render_posed_object


from src.utils.logging import get_logger

logger = get_logger(__name__)


class FoundationPoseInstance(Dataset):
    def __init__(
        self,
        scene_dataset: FoundationPoseScene,
        obj_ds,
        zoom_size=(224, 224),
        dzi_config={},
        load_cad=False,
        min_visib=0.8,
        color_augmentor=None,
        bg_augmentor=None,
        max_instance_per_obj=-1,
        invalid_mesh_list="",
        image_size=(640, 480),
        clean_bg=False,
        normalize_by="diameter",
        diameter="oracle",
    ):
        self.scene_dataset = scene_dataset
        self.zoom_size = np.array(zoom_size)
        self.dzi_config = dzi_config
        self.obj_ds = obj_ds
        self.load_cad = load_cad
        self.min_visib = min_visib
        self.max_instance_per_obj = max_instance_per_obj
        self.invalid_mesh_list = invalid_mesh_list
        self.invalid_meshes = self.load_invalid_meshes()
        self.color_augmentor = color_augmentor
        self.bg_augmentor = bg_augmentor
        self.image_size = image_size
        self.clean_bg = clean_bg
        self.normalize_by = normalize_by
        self.diameter = diameter

        self.build_index()

    def get_signature(self):
        hash_code = hashlib.md5(
            f"{self.scene_dataset.get_signature()}_{self.min_visib}_{self.max_instance_per_obj}_{self.invalid_mesh_list}".encode(
                "utf-8"
            )
        ).hexdigest()
        return hash_code

    def load_invalid_meshes(self):
        if not self.invalid_mesh_list:
            return set()
        with open(self.invalid_mesh_list, "r") as f:
            invalid_meshes = set([line.strip() for line in f.readlines()])
        return invalid_meshes

    def build_index(self):
        cache_path = f".cache/instances_{self.get_signature()}.pkl"
        print("loading instances from cache file: {}".format(cache_path))
        if osp.exists(cache_path):
            metaDatas = pickle.load(open(cache_path, "rb"))
            print("num instances: {}".format(len(metaDatas)))
            assert len(metaDatas) > 0
        else:
            metaDatas = []
            print("building instance indices ...")
            instance_cnt = {}
            for i in trange(len(self.scene_dataset.data)):
                state = self.scene_dataset.data[i]["state"]
                base_dir = self.scene_dataset.data[i]["base_dir"]
                obj_poses, _ = self.scene_dataset.load_poses(state)
                prims = sorted(list(obj_poses.keys()))
                # filter instance without bbox or segid
                with open(
                    f"{base_dir}instance_segmentation/instance_segmentation_mapping_000000.json",
                    "r",
                ) as ff:
                    segmentation = json.load(ff)
                reverse_map = lambda d: {
                    v.replace("/model/mesh", "").replace("/mesh", ""): int(k)
                    for k, v in d.items()
                }
                prim2segid = reverse_map(segmentation)
                try:
                    bbox_list = json.load(
                        open(
                            f"{base_dir}/bounding_box_2d_loose/bounding_box_2d_loose_prim_paths_000000.json",
                            "r",
                        )
                    )
                except:
                    bbox_list = []
                prim2bbox_id = {
                    prim.replace("/model/mesh", "").replace("/mesh", ""): i
                    for i, prim in enumerate(bbox_list)
                }
                bboxes = np.load(
                    f"{base_dir}/bounding_box_2d_loose/bounding_box_2d_loose_000000.npy"
                )
                for j, prim in enumerate(prims):
                    if prim not in prim2segid or prim not in prim2bbox_id:
                        continue
                    label = prim.split("/")[-1]
                    bbox_id = prim2bbox_id[prim]
                    bbox = bboxes[bbox_id]
                    occlusionRatio = bbox["occlusionRatio"]
                    if 1 - occlusionRatio < self.min_visib:
                        continue

                    def is_out_of_bound(bbox):
                        margin = 20
                        return (
                            bbox["x_min"] < margin
                            or bbox["y_min"] < margin
                            or bbox["x_max"] > self.image_size[0] - margin
                            or bbox["y_max"] > self.image_size[1] - margin
                        )

                    if is_out_of_bound(bbox):
                        continue
                    if label not in self.obj_ds.meta_data:
                        warnings.warn(f"no metadata for {label} ")
                        continue
                    if not self.obj_ds.contains(label):
                        warnings.warn(f"{label} not in object dataset")
                        continue
                    if (
                        self.max_instance_per_obj > 0
                        and instance_cnt.get(label, 0) >= self.max_instance_per_obj
                    ):
                        continue
                    if label in self.invalid_meshes:
                        continue
                    instance_cnt[label] = instance_cnt.get(label, 0) + 1
                    metaDatas.append(
                        dict(
                            **self.scene_dataset.data[i],
                            prim=prim,
                            label=label,
                            scene_idx=i,
                            pose_id=j,
                        )
                    )
            prepare_dir(osp.dirname(cache_path))
            pickle.dump(metaDatas, open(cache_path, "wb"))
        self.metaDatas = convert_list_to_dataframe(metaDatas)
        logger.info(f"Loaded {len(self.metaDatas)} images!")

    def __len__(self):
        return len(self.metaDatas)

    def __getitem__(self, idx):
        scene_idx = self.metaDatas.iloc[idx]["scene_idx"]
        pose_id = self.metaDatas.iloc[idx]["pose_id"]
        scene_data = self.scene_dataset[scene_idx]
        bboxes = scene_data["bboxes"]
        gt_poses = scene_data["gt_poses"]
        TWC = scene_data["TWC"]
        K = scene_data["K"]
        focal_length = K[0, 0].item()
        seg_map = scene_data["seg_map"]
        rgb = scene_data["rgb"]
        depth = scene_data["depth"]
        scale = scene_data["scales"][pose_id]
        h, w = rgb.shape[:2]

        prim2segid = scene_data["prim2segid"]
        prim2bbox_id = scene_data["prim2bbox_id"]

        prim = self.metaDatas.iloc[idx]["prim"]
        obj_name = prim.split("/")[-1]
        obj = self.obj_ds.get_object_by_label(obj_name)
        obj_id = self.obj_ds.get_id_by_label(obj_name)
        model_center = obj.model_center
        points = obj.points
        symmetries = obj.make_symmetry_poses()
        if self.load_cad:
            mesh = obj.model_p3d
            mesh.scale_verts_(scale[0])

        bbox_id = prim2bbox_id[prim]
        seg_id = prim2segid[prim]
        bbox = bboxes[bbox_id]
        bbox_xyxy = np.array(
            [bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]]
        )
        occlusionRatio = float(bbox["occlusionRatio"])
        gt_pose = gt_poses[pose_id]

        def normalizeRotation(pose):
            new_pose = pose.copy()
            scales = np.linalg.norm(pose[:3, :3], axis=0)
            new_pose[:3, :3] /= scales.reshape(1, 3)
            return new_pose

        ob_in_world = normalizeRotation(gt_pose)
        tco = np.linalg.inv(TWC) @ ob_in_world
        gt_t = tco[:3, 3]
        gt_2d = gt_t @ K.T
        gt_2d[0] /= gt_2d[2]
        gt_2d[1] /= gt_2d[2]
        center_in_img = gt_2d[:2]

        mask_visib = (seg_map == seg_id).astype(np.float32)

        if self.diameter == "estimated":
            try:
                extents, diameter = estimate_obj_size(depth, mask_visib, K, tco)
            except Exception as e:
                logger.warning(
                    f"Failed to estimate extents for {scene_idx} {pose_id}: {e}"
                )
                extents = obj.extents * scale
                diameter = np.linalg.norm(extents)
        elif self.diameter == "oracle":
            extents = obj.extents * scale
            diameter = np.linalg.norm(extents)
        else:
            raise ValueError(
                f"Invalid diameter type: {self.diameter}. Choose from ['estimated', 'oracle']."
            )

        detection_center_2d_gt = (bbox_xyxy[:2] + bbox_xyxy[2:]) / 2
        if self.dzi_config:
            detection_center_2d, crop_size = aug_bbox_DZI(
                bbox_xyxy, h, w, **self.dzi_config, return_box=False
            )
            detection_center_2d = detection_center_2d
        else:
            detection_center_2d = detection_center_2d_gt
            crop_size = (bbox_xyxy[2:] - bbox_xyxy[:2]).max()
        crop_size = float(crop_size)

        crop_image = lambda x, interpolation: crop_resize_by_warp_affine(
            x,
            detection_center_2d,
            crop_size,
            self.zoom_size,
            interpolation,
        )
        vis_mask_patch = crop_image(mask_visib, cv2.INTER_LINEAR)
        vis_mask_patch = vis_mask_patch >= 1.0
        rgb_patch = crop_image(rgb, cv2.INTER_LINEAR)
        depth_patch = crop_image(depth, cv2.INTER_NEAREST)

        if self.bg_augmentor is not None:
            rgb_patch = self.bg_augmentor(rgb_patch, vis_mask_patch)

        if self.color_augmentor is not None:
            rgb_patch = self.color_augmentor(rgb_patch)

        # calc bbox of the affine transform
        crop_trans = get_affine_transform(
            detection_center_2d, crop_size, 0, self.zoom_size
        )
        x1 = (0 - crop_trans[0, 2]) / crop_trans[0, 0]
        y1 = (0 - crop_trans[1, 2]) / crop_trans[1, 1]
        x2 = (self.zoom_size[0].item() - crop_trans[0, 2]) / crop_trans[0, 0]
        y2 = (self.zoom_size[1].item() - crop_trans[1, 2]) / crop_trans[1, 1]
        bbox_affine = np.array([x1, y1, x2, y2]).astype(np.float32)
        bw = max(bbox_affine[2] - bbox_affine[0], 1)
        bh = max(bbox_affine[3] - bbox_affine[1], 1)

        delta_c = center_in_img - detection_center_2d
        center_in_crop = (
            np.array([delta_c[0] / bw, delta_c[1] / bh]) + 0.5
        ) * self.zoom_size
        resize_ratio = scale[0] * focal_length * (self.zoom_size[0] / crop_size)
        scale_in_crop = 1 / (gt_t[2]) * resize_ratio

        K_crop = get_K_crop_resize(
            K,
            bbox_affine,
            orig_size=(h, w),
            crop_resize=self.zoom_size,
        )[0].numpy()

        nocs = depth_to_xyz(
            depth_patch * vis_mask_patch,
            K_crop,
            tco[:3, :3],
            tco[:3, 3],
            extents=extents,
            normalize_by=self.normalize_by,
        )
        roc = depth_to_roc_map(
            depth_patch * vis_mask_patch,
            K_crop,
            tco[:3, 3],
            extents=extents,
            normalize_by=self.normalize_by,
        )

        depth_bp = (
            normalize_depth_bp_zscore(
                depth_backproject(
                    depth_patch,
                    K_crop,
                ),
                vis_mask_patch,
            )
            * vis_mask_patch[..., None]
        )

        if self.clean_bg:
            rgb_patch = rgb_patch * vis_mask_patch[..., None]

        data = dict(
            scene_id=scene_idx,
            im_id=0,
            gt_id=pose_id,
            det_id=pose_id,
            obj_id=obj_id,
            score=1.0,
            bbox=bbox_affine,
            vis_mask_patch=vis_mask_patch,
            rgb_patch=rearrange(rgb_patch, "h w c -> c h w"),
            depth_patch=depth_patch,
            depth_bp=rearrange(depth_bp, "h w c -> c h w"),
            occlusionRatio=occlusionRatio,
            TCO=tco.astype(np.float32),
            nocs=rearrange(nocs, "h w c -> c h w"),
            roc=rearrange(roc, "h w c -> c h w"),
            K_crop=K_crop,
            extents=extents,
            diameter=diameter,
            model_center=model_center.astype(np.float32),
            center_in_crop=center_in_crop.astype(np.float32),
            scale_in_crop=scale_in_crop.astype(np.float32),
            label=obj_name,
            points=points,
            symmetries=torch.from_numpy(symmetries).float(),
            resize_ratio=resize_ratio.astype(np.float32),
        )
        if self.load_cad:
            data.update(dict(mesh=mesh))
        return data


def show_batch(batch, log_dir="logs/debug"):

    rgb_patch = batch["rgb_patch"]
    vis_mask_patch = batch["vis_mask_patch"]
    depth_patch = batch["depth_patch"]
    nocs = batch["nocs"]
    roc = batch["roc"]
    mesh = batch.get("mesh", None)
    K = batch["K_crop"]
    TCO = batch["TCO"]
    res = batch["rgb_patch"].shape[2:]
    label = batch.get("label", None)
    occlussion_ratio = batch.get("occlusionRatio", None)
    center_in_crop = batch.get("center_in_crop", None)

    if mesh is not None:
        rendered_rgb, depth_gt = render_posed_object(
            mesh,
            TCO,
            K,
            res,
            light_loc=(1.0, -1.0, -1.0),
            dim_order="torch",
            znear=0.001,
            zfar=10,
            pixel2face=False,
            shader_type="soft_phong",
            with_alpha_channel=False,
            re_render_iters=10,
        )
        depth_gt = depth_gt.squeeze(1)
    else:
        rendered_rgb = torch.zeros_like(rgb_patch)
        depth_gt = torch.zeros_like(vis_mask_patch).float()

    depth_error_thresh = 0.01
    bs = len(rgb_patch)
    # statistics to check if mesh is valid
    stats = []
    for i in range(bs):
        stats.append(
            {
                "label": label[i],
                "vis_mask_sum": max(vis_mask_patch[i].sum().item(), 1),
                "depth_correct_sum": (
                    (depth_gt[i] - depth_patch[i]).abs() < depth_error_thresh
                )
                .sum()
                .item(),
            }
        )

    fig, axs = plt.subplots(bs, 8, figsize=(20, 8 * bs))
    if bs == 1:
        axs = [axs]
    for i in range(bs):
        axs[i][0].imshow(rgb_patch[i].permute(1, 2, 0).cpu().numpy())
        # show center in image
        if center_in_crop is not None:
            center = center_in_crop[i].cpu().numpy()
            axs[i][0].scatter(center[0], center[1], color="red", s=1000, marker="*")
        axs[i][0].set_title(f"{label[i]}")
        axs[i][1].imshow(vis_mask_patch[i].cpu().numpy())
        axs[i][1].set_title(f"occ_{occlussion_ratio[i]:.2f}")
        axs[i][2].imshow(depth_patch[i].cpu().numpy())
        axs[i][2].set_title("depth")
        axs[i][3].imshow(nocs[i].permute(1, 2, 0).cpu().numpy())
        axs[i][3].set_title("nocs")
        axs[i][4].imshow(roc[i].permute(1, 2, 0).cpu().numpy())
        axs[i][4].set_title("roc")
        axs[i][5].imshow(rendered_rgb[i].permute(1, 2, 0).cpu().numpy())
        axs[i][5].set_title("rendered pose")
        axs[i][6].imshow(depth_gt[i].cpu().numpy())
        axs[i][6].set_title("rendered depth")
        axs[i][7].imshow(
            (depth_gt[i] - depth_patch[i])
            .abs()
            .clip(0, depth_error_thresh)
            .cpu()
            .numpy()
        )
        axs[i][7].set_title("depth_diff")
        for j in range(8):
            axs[i][j].axis("off")
    # plt.show()
    plt.tight_layout()
    plt.savefig(f"{log_dir}/batch.png")
    plt.close()

    return stats


def get_dataset_gso():
    scene_dataset = FoundationPoseScene(root_dir="data/foundation_pose/gso")
    print(len(scene_dataset))
    obj_ds = GoogleScannedObjectDataset("data/MegaPose-GSO/google_scanned_objects")

    bg_augmentor = BGAugmentor("VOC", "./data/VOC2012/VOCdevkit/VOC2012", 10000)
    color_augmentor = ColorAugmentor()

    dzi_config = dict(
        dzi_type="uniform", dzi_pad_scale=1.5, dzi_scale_ratio=0.0, dzi_shift_ratio=0.0
    )

    instance_dataset = FoundationPoseInstance(
        scene_dataset,
        obj_ds,
        dzi_config=dzi_config,
        load_cad=True,
        bg_augmentor=bg_augmentor,
        color_augmentor=color_augmentor,
        clean_bg=True,
    )
    print(len(instance_dataset))
    return instance_dataset


def get_dataset_objaverse():
    scene_dataset = FoundationPoseScene(
        root_dir="data/foundation_pose/objaverse", subset="objaverse"
    )
    print(len(scene_dataset))
    obj_ds = ObjaverseDataset("data/Objaverse_cache")

    bg_augmentor = BGAugmentor("VOC", "./data/VOC2012/VOCdevkit/VOC2012", 10000)
    color_augmentor = ColorAugmentor()

    dzi_config = dict(
        dzi_type="uniform", dzi_pad_scale=1.5, dzi_scale_ratio=0.0, dzi_shift_ratio=0.0
    )

    instance_dataset = FoundationPoseInstance(
        scene_dataset,
        obj_ds,
        dzi_config=dzi_config,
        load_cad=True,
        bg_augmentor=bg_augmentor,
        color_augmentor=color_augmentor,
        max_instance_per_obj=-1,
        min_visib=0.3,
        invalid_mesh_list="configs/data/foundation_pose/objaverse_invalid_meshes.txt",
        clean_bg=True,
    )
    print(len(instance_dataset))
    return instance_dataset


# python -m src.data.foundation_pose.instance_dataset
if __name__ == "__main__":

    instance_dataset = get_dataset_gso()

    for item in tqdm(instance_dataset):
        break

    data_loader = torch.utils.data.DataLoader(
        instance_dataset,
        num_workers=1,
        collate_fn=default_collate_fn,
        batch_size=4,
    )

    stats = []

    for batch in tqdm(data_loader):
        batch = to_device(batch, "cuda:0")
        s = show_batch(batch)
        stats.extend(s)

    df = convert_list_to_dataframe(stats)
    df.to_csv("logs/instance_stats.csv", index=False)
    # filter
    df = df[df["depth_correct_sum"] / df["vis_mask_sum"] < 0.8]
    labels = df["label"].unique()
    print("labels with <80% depth correct:", len(labels))
    # write labels
    with open("logs/invalid_meshes.txt", "w") as f:
        for label in labels:
            f.write(f"{label}\n")
    print("labels written to logs/invalid_meshes.txt")
