from ast import parse
import os
import pandas as pd
from tqdm import tqdm
import trimesh
import yaml
import numpy as np
import cv2
import torch

from PIL import Image
from CoordAR.data.bop.few_shot_dataset import BOPFewShot
from CoordAR.data.bop.instance_dataset import BOPInstanceDataset
from CoordAR.data.bop.scene_dataset import BOPSceneDataset
from CoordAR.data.megapose.obj_ds.bop_object_dataset import BOPObjectDataset
from CoordAR.utils.logging import get_logger
from CoordAR.utils.misc import prepare_dir
from estimater import Any6D

from foundationpose.Utils import (
    get_bounding_box,
    visualize_frame_results,
    calculate_chamfer_distance_gt_mesh,
    align_mesh_to_coordinate,
)
import nvdiffrast.torch as dr
import argparse
from pytorch_lightning import seed_everything

from sam2_instantmesh import *

glctx = dr.RasterizeCudaContext()

logger = get_logger(__name__)

if __name__ == "__main__":

    seed_everything(0)

    parser = argparse.ArgumentParser(description="Predict on BOP")
    parser.add_argument(
        "--datasets_path",
        type=str,
        help="Path to the datasets directory",
        default="data/BOP",
    )
    parser.add_argument(
        "--dataset", type=str, help="Name of the dataset to partition", default="lm"
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Dataset split to partition",
        default="test",
    )
    parser.add_argument("--split_type", default="none", help="Type of dataset split")
    parser.add_argument(
        "--img_to_3d", action="store_true", help="Running with InstantMesh+SAM2"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the output results",
        default="logs/predict",
    )
    args = parser.parse_args()

    """
        init dataset
    """
    scene_dataset = BOPSceneDataset(
        args.datasets_path,
        args.dataset,
        args.split,
        args.split_type if args.split_type != "none" else None,
        only_bop19_test=True,
    )
    obj_ds = BOPObjectDataset(f"{args.datasets_path}/{args.dataset}/models")
    instance_dataset = BOPInstanceDataset(scene_dataset, obj_ds)
    few_shot_ds = BOPFewShot(
        instance_dataset,
        num_ref=1,
        ref="first_frame",
    )

    img_to_3d = args.img_to_3d

    # build model for each object
    model_out = f"{args.output_dir}/{args.dataset}_{args.split}/model"
    prepare_dir(model_out)
    if img_to_3d:
        for sample in tqdm(few_shot_ds):
            obj_id = sample["obj_id"]
            obj_name = f"{obj_id:06d}"
            if os.path.exists(os.path.join(model_out, f"center_mesh_{obj_name}.obj")):
                logger.info(f"center_mesh_{obj_name}.obj found, skip generation")
                break
            else:
                mesh_path = os.path.join(model_out, f"{obj_id:06d}.obj")
                mask = sample["template_masks_visib"][0]
                color = rearrange(sample["template_imgs"][0], "c h w -> h w c")
                Image.fromarray(color).save(os.path.join(model_out, "color.png"))

                """ call 123 model"""
                cmin, rmin, cmax, rmax = get_bounding_box(mask).astype(np.int32)
                input_box = np.array([cmin, rmin, cmax, rmax])[None, :]
                mask_refine = running_sam_box(color, input_box)

                input_image = preprocess_image(color, mask_refine, model_out, obj_name)
                images = diffusion_image_generation(
                    model_out, model_out, obj_name, input_image=input_image
                )
                instant_mesh_process(images, model_out, obj_name)

                mesh = trimesh.load(os.path.join(model_out, f"mesh_{obj_name}.obj"))
                mesh = align_mesh_to_coordinate(mesh)
                mesh.export(os.path.join(model_out, f"center_mesh_{obj_name}.obj"))

    pose_out = f"{args.output_dir}/{args.dataset}_{args.split}/pose"
    prepare_dir(pose_out)
    predictions = []
    for sample in tqdm(few_shot_ds):
        obj_id = sample["obj_id"]
        intrinsic = sample["query_K_crop"]
        obj_name = f"{obj_id:06d}"
        mask = sample["template_masks_visib"][0]
        color = rearrange(sample["template_imgs"][0], "c h w -> h w c")
        depth = sample["query_depth"]

        mesh = trimesh.load(os.path.join(model_out, f"center_mesh_{obj_name}.obj"))

        est = Any6D(symmetry_tfs=None, mesh=mesh, debug_dir=pose_out, debug=2)

        pred_pose = est.register_any6d(
            K=intrinsic,
            rgb=color,
            depth=depth,
            ob_mask=mask,
            iteration=5,
            name=obj_name,
        )

        print(pred_pose)
        predictions.append(
            dict(
                scene_id=sample["scene_id"],
                im_id=sample["im_id"],
                obj_id=obj_id,
                gt_id=sample["gt_id"],
                R=pred_pose[:3, :3],
                t=pred_pose[:3, 3],
                time=-1,
            )
        )
    df = pd.DataFrame(predictions)
    df.to_csv(os.path.join(pose_out, "predictions.csv"), index=False)
