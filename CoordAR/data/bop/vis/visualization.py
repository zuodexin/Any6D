import os
from einops import rearrange
import imageio
import ipdb
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
import torch.nn.functional as F

from CoordAR.utils.vis.pose import (
    draw_projected_box3d,
    draw_projected_box3d_xyz,
    get_bbox3d_from_center_ext,
    points_to_2D,
)

resize_as = lambda x, y: F.interpolate(
    x.float(),
    size=y.shape[2:],
    mode="nearest",
)


def normalize_depth_bp_torch(depth_bp):
    max_value = torch.max(depth_bp.flatten(2, 3), dim=2)[0].view(-1, 3, 1, 1)
    min_value = torch.min(depth_bp.flatten(2, 3), dim=2)[0].view(-1, 3, 1, 1)
    depth_bp = (depth_bp - min_value) / (max_value - min_value + 1e-6)
    return depth_bp


def show_batch(batch, log_dir="logs/debug"):

    template_imgs = batch["template_imgs"]
    template_rel_rocs = batch["template_rel_rocs"]
    template_rel_rocs = batch["template_rel_rocs"]
    template_rocs = batch["template_rocs"]
    template_depths = batch["template_depths"]
    template_masks = batch["template_masks_visib"]
    bs = len(template_imgs)
    template_depth_bp = batch["template_depth_bp"]
    query = batch["query"]
    query_rel_roc = batch["query_rel_roc"]
    query_roc = batch["query_roc"]
    query_depth = batch["query_depth"]
    query_depth_bp = batch["query_depth_bp"]
    query_rel_roc = batch["query_rel_roc"]
    query_mask = batch["query_mask"]
    diameter = batch["diameter"].tolist()
    template_obj_size = batch["template_obj_size"].tolist()

    bs = len(template_imgs)

    fig, axs = plt.subplots(bs, 14, figsize=(14 * 2, bs * 2))
    if bs == 1:
        axs = [axs]
    for i in range(bs):
        scene_id = batch["scene_id"][i].item()
        im_id = batch["im_id"][i].item()
        gt_id = batch["gt_id"][i].item()
        obj_id = batch["obj_id"][i].item()
        axs[i][0].imshow(make_grid(template_imgs[i]).permute(1, 2, 0).cpu().numpy())
        axs[i][0].set_title(
            f"template_img, diameter: {template_obj_size[i][0]*1000:.1f}mm"
        )
        axs[i][1].imshow(template_depths[i][0].cpu().numpy(), cmap="gray")
        axs[i][1].set_title("template_depth")
        axs[i][2].imshow(query[i].permute(1, 2, 0).cpu().numpy())
        axs[i][2].set_title(f"query: {scene_id}_{im_id}_{obj_id}_{gt_id}")
        axs[i][3].imshow(query_depth[i].cpu().numpy(), cmap="gray")
        axs[i][3].set_title(f"query_depth, diameter:{diameter[i]*1000:.1f}")
        axs[i][4].imshow(
            make_grid(template_rel_rocs[i]).permute(1, 2, 0).clip(0, 1).cpu().numpy()
        )
        axs[i][4].set_title("template_rel_rocs")
        axs[i][5].imshow(
            make_grid(template_rocs[i]).permute(1, 2, 0).clip(0, 1).cpu().numpy()
        )
        axs[i][5].set_title("template_rocs")

        axs[i][6].imshow(
            make_grid(template_rel_rocs[i]).permute(1, 2, 0).clip(0, 1).cpu().numpy()
        )
        axs[i][6].set_title("template_rel_rocs")

        axs[i][7].imshow(query_rel_roc[i].permute(1, 2, 0).clip(0, 1).cpu().numpy())
        axs[i][7].set_title("query_rel_roc")
        axs[i][8].imshow(query_roc[i].permute(1, 2, 0).clip(0, 1).cpu().numpy())
        axs[i][8].set_title("query_roc")
        axs[i][9].imshow(query_rel_roc[i].permute(1, 2, 0).clip(0, 1).cpu().numpy())
        axs[i][9].set_title("query_rel_roc")

        axs[i][10].imshow(template_masks[i][0].cpu().numpy(), cmap="grey")
        axs[i][10].set_title("template_mask")
        axs[i][11].imshow(query_mask[i].cpu().numpy(), cmap="grey")
        axs[i][11].set_title("query_mask")

        axs[i][12].imshow(query_depth_bp[i].permute(1, 2, 0).cpu().numpy())
        axs[i][12].set_title("query_depth_bp")

        axs[i][13].imshow(
            make_grid(template_depth_bp[i]).permute(1, 2, 0).clip(0, 1).cpu().numpy()
        )
        axs[i][13].set_title("template_depth_bp")

        for j in range(14):
            axs[i][j].axis("off")
    plt.tight_layout()
    plt.savefig(f"{log_dir}/batch.png")
    plt.close()
