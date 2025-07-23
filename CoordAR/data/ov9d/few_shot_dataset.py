import torch
from tqdm import tqdm
from src.data.collate_importer import default_collate_fn
from src.data.foundation_pose.few_shot_dataset import FoundationPoseFewShot
from src.data.megapose.obj_ds.ov9d_dataset import OV9DObjectDataset
from src.data.ov9d.instance_dataset import OV9DInstanceDataset
from src.data.ov9d.scene_dataset import OV9DSceneDataset
from src.models.memchip.visualization import show_batch

# python -m src.data.ov9d.few_shot_dataset
if __name__ == "__main__":
    scene_dataset = OV9DSceneDataset("data/OV9D")
    print(len(scene_dataset))
    dzi_config = dict(
        dzi_type="uniform", dzi_pad_scale=1.5, dzi_scale_ratio=0.0, dzi_shift_ratio=0.0
    )
    obj_ds = OV9DObjectDataset("data/OV9D")
    instance_dataset = OV9DInstanceDataset(scene_dataset, obj_ds, dzi_config=dzi_config)
    print(len(instance_dataset))

    few_shot_ds = FoundationPoseFewShot(instance_dataset, num_ref=1)

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
        # ipdb.set_trace()
        show_batch(batch)
