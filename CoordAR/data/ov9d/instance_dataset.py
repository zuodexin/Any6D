import copy
import hashlib
import os
import pickle
import cv2
from einops import rearrange
import ipdb
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import skimage
from mmcv.image.io import imread
import os.path as osp
import torch
from torch.utils.data import Dataset
from tqdm import tqdm, trange
from src.data.bop.instance_dataset import (
    estimate_obj_size,
)
from src.data.collate_importer import default_collate_fn
from src.data.foundation_pose.instance_dataset import show_batch
from src.data.megapose.obj_ds.ov9d_dataset import OV9DObjectDataset
from src.data.ov9d.scene_dataset import OV9DSceneDataset
from src.third_party.custom_bop_toolkit.bop_toolkit_lib import inout
from src.data.megapose.shapenet import (
    depth_backproject,
    depth_to_roc_map,
    depth_to_xyz,
    normalize_depth_bp_zscore,
)
from src.third_party.custom_bop_toolkit.bop_toolkit_lib.dataset_params import (
    get_model_params,
    get_present_scene_ids,
    get_split_params,
)
from src.utils.augment import aug_bbox_DZI
from src.utils.cropping import (
    crop_resize_by_warp_affine,
    get_K_crop_resize,
    get_affine_transform,
    xywh2xyxy,
)
from src.utils.inout import convert_list_to_dataframe


from src.utils.logging import get_logger
from src.utils.mask_utils import binary_mask_to_rle, cocosegm2mask
from src.utils.misc import prepare_dir, to_device
from src.utils.pysixd.RT_transform import allocentric_to_egocentric
from src.utils.system import Timer

logger = get_logger(__name__)


class OV9DInstanceDataset(Dataset):

    def __init__(
        self,
        scene_ds: OV9DSceneDataset,
        obj_ds: OV9DObjectDataset,
        dzi_config={},
        zoom_size=(224, 224),
        load_cad=False,
        clean_bg=False,
        normalize_by="diameter",
        color_augmentor=None,
        bg_augmentor=None,
        diameter="oracle",
    ):
        super().__init__()

        self.scene_ds = scene_ds
        self.obj_ds = obj_ds
        self.dzi_config = dzi_config
        self.zoom_size = np.array(zoom_size)
        self.load_cad = load_cad
        self.clean_bg = clean_bg
        self.normalize_by = normalize_by
        self.color_augmentor = color_augmentor
        self.bg_augmentor = bg_augmentor
        self.diameter = diameter

        self.build_index()

    def get_signature(self):
        hashed_file_name = hashlib.md5(
            (
                "{}_instances".format(
                    self.scene_ds.get_signature(),
                )
            ).encode("utf-8")
        ).hexdigest()
        return hashed_file_name

    def build_index(self):
        cache_path = f".cache/ov9d_instances_{self.get_signature()}.pkl"
        if osp.exists(cache_path):
            logger.info("get instances from cache file: {}".format(cache_path))
            metaDatas = pickle.load(open(cache_path, "rb"))
            logger.info("num instances: {}".format(len(metaDatas)))
            assert len(metaDatas) > 0
        else:
            metaDatas = []
            scene_meta = self.scene_ds.metaDatas
            num_scenes = len(scene_meta)

            for i in trange(num_scenes):
                scene_data = scene_meta.iloc[i]
                scene_id = scene_data["scene_id"]
                instances = scene_data["instances"]
                for j, instance in enumerate(instances):
                    label = instance["label"]
                    TCO = instance["TCO"]
                    rot = allocentric_to_egocentric(TCO)[:3, :3]
                    metaDatas.append(
                        dict(scene_idx=i, instance_idx=j, label=label, rot=rot)
                    )
            prepare_dir(osp.dirname(cache_path))
            pickle.dump(metaDatas, open(cache_path, "wb"))
        self.metaDatas = convert_list_to_dataframe(metaDatas)
        logger.info(f"num instances:{len(self.metaDatas)}")

    def __len__(self):
        return len(self.metaDatas)

    def __getitem__(self, idx: int):

        timer = Timer()
        timer.start("get_item")

        metaData = self.metaDatas.iloc[idx]
        scene_idx = metaData["scene_idx"]
        instance_idx = metaData["instance_idx"]
        label = metaData["label"]

        scene_data = self.scene_ds[scene_idx]
        scene_meta = self.scene_ds.metaDatas.iloc[scene_idx]
        instances = scene_meta["instances"]
        instance = instances[instance_idx]

        K = scene_meta["K"]
        scene_id = scene_meta["scene_id"]
        scene_name = scene_meta["scene_name"]
        im_id = scene_meta["im_id"]

        rgb = rearrange(scene_data["rgb"], "c h w -> h w c")
        depth = scene_data["depth"]
        TCO = instance["TCO"]
        gt_id = instance["gt_id"]
        score = instance["score"]
        det_id = instance["det_id"]
        visibility = instance["visibility"]

        gt_t = TCO[:3, 3]
        gt_2d = gt_t @ K.T
        gt_2d[0] /= gt_2d[2]
        gt_2d[1] /= gt_2d[2]
        center_in_img = gt_2d[:2]

        timer.start("get_object")

        obj = self.obj_ds.get_object_by_label(label)
        model_center = obj.model_center
        # points = obj.points
        points = np.zeros((1000, 3), dtype=np.float32)  # dummy points, not used in OV9D
        symmetries = obj.make_symmetry_poses()
        if self.load_cad:
            mesh = obj.model_p3d
        obj_id = int(label.split("_")[-1])

        timer.end("get_object")

        im_H, im_W = rgb.shape[:2]  # h, w
        if gt_id >= 0:
            if "mask_visib_rle" in instance:
                mask_visib = cocosegm2mask(instance["mask_visib_rle"], h=im_H, w=im_W)
                mask_amodal = cocosegm2mask(instance["mask_amodal_rle"], h=im_H, w=im_W)
            else:
                mask_visib = (
                    cv2.imread(
                        f"{self.scene_ds.root_dir}/{self.scene_ds.split}/{scene_name}/mask_visib/{im_id:06d}_{gt_id:06d}.png"
                    )[..., 0]
                    // 255
                )
        else:
            mask_visib = np.zeros_like(rgb[..., 0])
        h, w = rgb.shape[:2]

        bbox = instance["bbox"]
        bbox_xyxy = xywh2xyxy(bbox)

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

        if self.diameter == "estimated":
            try:
                extents, diameter = estimate_obj_size(
                    depth_patch, vis_mask_patch, K, TCO
                )
            except Exception as e:
                logger.warning(
                    f"Failed to estimate extents for {scene_id} {im_id} {gt_id}: {e}"
                )
                diameter = np.float32(obj.diameter_meters)
                extents = obj.extents
        elif self.diameter == "oracle":
            diameter = np.float32(obj.diameter_meters)
            extents = obj.extents
        else:
            raise ValueError(
                f"Invalid diameter type: {self.diameter}. Choose from ['estimated', 'oracle']."
            )

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
        resize_ratio = self.zoom_size[0] / crop_size
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
            TCO[:3, :3],
            TCO[:3, 3],
            extents=extents,
            normalize_by=self.normalize_by,
        )
        roc = depth_to_roc_map(
            depth_patch * vis_mask_patch,
            K_crop,
            TCO[:3, 3],
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
            label=label,
            scene_id=scene_id,
            im_id=im_id,
            gt_id=gt_id,
            det_id=det_id,
            obj_id=obj_id,
            score=score,
            bbox=bbox.astype(np.float32),
            vis_mask_patch=vis_mask_patch,
            rgb_patch=rearrange(rgb_patch, "h w c -> c h w"),
            depth_patch=depth_patch,
            depth_bp=rearrange(depth_bp, "h w c -> c h w"),
            TCO=TCO.astype(np.float32),
            K_crop=K_crop,
            nocs=rearrange(nocs, "h w c -> c h w"),
            roc=rearrange(roc, "h w c -> c h w"),
            extents=extents,
            diameter=diameter,
            model_center=model_center.astype(np.float32),
            center_in_crop=center_in_crop.astype(np.float32),
            scale_in_crop=scale_in_crop.astype(np.float32),
            points=points,
            symmetries=torch.from_numpy(symmetries).float(),
            resize_ratio=resize_ratio.astype(np.float32),
            occlusionRatio=1 - visibility,
        )
        if self.load_cad:
            data.update(dict(mesh=mesh))
        timer.end("get_item")
        # timer.summary()
        return data


# python -m src.data.ov9d.instance_dataset
if __name__ == "__main__":
    scene_dataset = OV9DSceneDataset(
        "data/OV9D",
        choose_obj=[],
    )
    obj_ds = OV9DObjectDataset("data/OV9D")
    instance_dataset = OV9DInstanceDataset(scene_dataset, obj_ds, load_cad=False)
    print(len(instance_dataset))

    for i in trange(len(instance_dataset)):
        data = instance_dataset[i]
        break

    data_loader = torch.utils.data.DataLoader(
        instance_dataset,
        num_workers=1,
        collate_fn=default_collate_fn,
        batch_size=4,
    )

    for batch in tqdm(data_loader):
        batch = to_device(batch, "cuda:0")
        ipdb.set_trace()
    # s = show_batch(batch)
