import torch
from torch.utils.data import Dataset
import numpy as np


class RandomDigitDataset(Dataset):
    def __init__(self, num_samples, sequence_length, seed=None):
        """
        生成随机数字序列的数据集
        Args:
            num_samples (int): 数据集样本数量
            sequence_length (int): 每个样本的序列长度 (n)
            seed (int, optional): 随机种子（保证可复现性）
        """
        self.num_samples = num_samples
        self.sequence_length = sequence_length

        if seed is not None:
            np.random.seed(seed)

        # 生成所有样本 (num_samples, sequence_length)
        self.data = np.random.randint(0, 10, size=(num_samples, sequence_length))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 返回一个样本 (Tensor类型)
        return torch.tensor(self.data[idx], dtype=torch.long)


# python -m src.data.random_digit
if __name__ == "__main__":
    # 参数设置
    num_samples = 1000  # 数据集大小
    sequence_length = 100  # 每个样本的长度 (n)
    seed = 42  # 随机种子

    # 创建数据集
    dataset = RandomDigitDataset(num_samples, sequence_length, seed)

    # 打印第一个样本
    print("Sample 0:", dataset[0])

    # 创建 DataLoader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 迭代一个 batch
    for batch in dataloader:
        print("Batch shape:", batch.shape)  # [32, 16]
        break
