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

from src.utils.logging import get_logger

CACHE_DIR = "./.cache/foundation_pose"
MEMORY = Memory(CACHE_DIR, verbose=0)

logger = get_logger(__name__)


def load_camera(base_dir):
    glcam_in_cvcam = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    ).astype(np.float32)
    W, H = 640, 480
    with open(f"{base_dir}/camera_params/camera_params_000000.json", "r") as ff:
        camera_params = json.load(ff)
    world_in_glcam = (
        np.array(camera_params["cameraViewTransform"], dtype=np.float32).reshape(4, 4).T
    )
    cam_in_world = np.linalg.inv(world_in_glcam) @ glcam_in_cvcam
    world_in_cam = np.linalg.inv(cam_in_world)
    focal_length = camera_params["cameraFocalLength"]
    horiz_aperture = camera_params["cameraAperture"][0]
    vert_aperture = H / W * horiz_aperture
    focal_y = H * focal_length / vert_aperture
    focal_x = W * focal_length / horiz_aperture
    center_y = H * 0.5
    center_x = W * 0.5

    fx, fy, cx, cy = focal_x, focal_y, center_x, center_y
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy

    return K, cam_in_world


@MEMORY.cache()
def load_data(root_dir):
    datas = []
    sub_dirs = glob.glob(f"{root_dir}/*/")
    for sub_dir in tqdm(sub_dirs):
        scene_dirs = glob.glob(f"{sub_dir}*/")
        for scene_dir in scene_dirs:
            state = f"{scene_dir}states.json"
            inner_dir = glob.glob(f"{scene_dir}*/")
            if len(inner_dir) == 0:
                continue
            else:
                inner_dir = inner_dir[0]
            for name in ["RenderProduct_Replicator", "RenderProduct_Replicator_01"]:
                base_dir = f"{inner_dir}{name}/"
                if not os.path.exists(base_dir):
                    continue
                # filter incomplete data
                sub_dirs = os.listdir(base_dir)
                check_dirs = [
                    "rgb",
                    "instance_segmentation",
                    "distance_to_image_plane",
                    "camera_params",
                    "bounding_box_2d_loose",
                ]
                is_complete = True
                for d in check_dirs:
                    if d not in sub_dirs:
                        is_complete = False
                if not is_complete:
                    print(f"skip incomplete dir {base_dir}")
                    continue
                # filter invalid cameras
                try:
                    json.load(
                        open(f"{base_dir}/camera_params/camera_params_000000.json", "r")
                    )
                except Exception as e:
                    print(f"Error loading camera for {base_dir}: {e}")
                    continue
                datas.append(
                    dict(
                        state=state,
                        base_dir=base_dir,
                    )
                )
    return datas


class FoundationPoseScene(Dataset):

    def __init__(self, root_dir, subset="gso"):
        self.root_dir = root_dir
        self.subset = subset
        self.data = load_data(root_dir)

    def __len__(self):
        return len(self.data)

    def get_signature(self):
        hash_code = hashlib.md5(
            f"{self.root_dir}_{self.subset}".encode("utf-8")
        ).hexdigest()
        return hash_code

    def load_poses(self, state):
        state_data = json.load(open(state))
        obj_poses = {}
        scales = {}
        for name, value in state_data["objects"].items():
            prim_path = value["prim_path"]
            pose = np.eye(4)
            pose[:3, :3] = np.array(value["rotation_matrix"]).reshape(3, 3)
            pose[:3, 3] = np.array(value["translation"]).reshape(3)
            scales[prim_path] = np.array(value["scale"], dtype=np.float32).reshape(3)
            obj_poses[prim_path] = pose.astype(np.float32)
        return obj_poses, scales

    def __getitem__(self, idx):
        state = self.data[idx]["state"]
        obj_poses, obj_scales = self.load_poses(state)
        prims = sorted(list(obj_poses.keys()))

        base_dir = self.data[idx]["base_dir"]
        K, cam_in_world = load_camera(base_dir)
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

        gt_poses = [obj_poses[prim_path] for prim_path in prims]
        scales = [obj_scales[prim_path] for prim_path in prims]
        rgb = skimage.io.imread(f"{base_dir}/rgb/rgb_000000.png")[:, :, :3]
        seg_map = skimage.io.imread(
            f"{base_dir}/instance_segmentation/instance_segmentation_000000.png"
        )
        depth = np.load(
            f"{base_dir}/distance_to_image_plane/distance_to_image_plane_000000.npy"
        )
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
        depth = np.where(np.isinf(depth), 0, depth)

        return dict(
            rgb=rgb,
            depth=depth,
            seg_map=seg_map,
            bboxes=bboxes,
            prims=prims,
            prim2bbox_id=prim2bbox_id,
            gt_poses=gt_poses,
            prim2segid=prim2segid,
            K=K,
            TWC=cam_in_world,
            scales=scales,
        )


# python -m src.data.foundation_pose.scene_dataset
if __name__ == "__main__":
    scene_dataset = FoundationPoseScene(root_dir="data/foundation_pose/gso")
    print(len(scene_dataset))

    scene_dataset = FoundationPoseScene(
        root_dir="data/foundation_pose/objaverse", subset="objaverse"
    )
    print(len(scene_dataset))
