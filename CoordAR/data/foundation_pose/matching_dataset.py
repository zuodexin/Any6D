import torch
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

from src.data.collate_importer import default_collate_fn
from src.data.foundation_pose.instance_dataset import FoundationPoseInstance
from src.data.foundation_pose.scene_dataset import FoundationPoseScene
from src.data.megapose.obj_ds.gso_dataset import GoogleScannedObjectDataset
from src.data.megapose.obj_ds.objaverse_dataset import ObjaverseDataset
from src.utils.misc import to_device


class MatchingDataset(IterableDataset):

    def __init__(self, instance_ds: FoundationPoseInstance):
        super().__init__()
        self.instance_ds = instance_ds

    def __iter__(self):
        while True:
            yield self.generate_item()

    def generate_item(self):
        return {}


def show_batch(batch):
    pass


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

    few_shot_ds = MatchingDataset(instance_dataset)
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

    few_shot_ds = MatchingDataset(instance_dataset)
    return few_shot_ds


# python -m src.data.foundation_pose.matching_dataset
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
