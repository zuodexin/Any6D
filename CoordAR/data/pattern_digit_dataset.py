import torch
from torch.utils.data import Dataset
import numpy as np
from enum import Enum


class PatternType(Enum):
    ASCENDING = 1  # 递增序列
    DESCENDING = 2  # 递减序列
    REPEAT = 3  # 重复模式
    ODDEVEN = 4  # 奇偶交替
    FIBONACCI = 5  # 类斐波那契
    RANDOM = 6  # 完全随机


class PatternDigitDataset(Dataset):
    def __init__(
        self,
        num_samples,
        sequence_length,
        pattern_type=PatternType.ASCENDING,
        pattern_length=3,
        seed=None,
    ):
        """
        生成带有规律模式的数字序列数据集

        Args:
            num_samples: 样本数量
            sequence_length: 序列长度
            pattern_type: 规律类型 (PatternType枚举)
            pattern_length: 基础模式长度(用于REPEAT等模式)
            seed: 随机种子
        """
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.pattern_type = pattern_type
        self.pattern_length = pattern_length

        if seed is not None:
            np.random.seed(seed)

        # 生成所有样本
        self.data = np.zeros((num_samples, sequence_length), dtype=np.int64)
        self.patterns = []  # 存储每个样本的base_pattern

        for i in range(num_samples):
            # 为每个样本生成独特的base_pattern
            if pattern_type == PatternType.ASCENDING:
                start = np.random.randint(0, 10)
                base_pattern = np.arange(start, start + pattern_length) % 10
            elif pattern_type == PatternType.DESCENDING:
                start = np.random.randint(0, 10)
                base_pattern = np.arange(start, start - pattern_length, -1) % 10
            elif pattern_type == PatternType.REPEAT:
                base_pattern = np.random.randint(0, 10, size=pattern_length)
            elif pattern_type == PatternType.ODDEVEN:
                start = np.random.choice([1, 2])
                base_pattern = (
                    np.array([(start + i) % 2 * 2 + 1 for i in range(pattern_length)])
                    % 10
                )
            elif pattern_type == PatternType.FIBONACCI:
                a, b = np.random.randint(1, 5, size=2)
                fib = [a, b]
                for _ in range(pattern_length - 2):
                    fib.append(fib[-1] + fib[-2])
                base_pattern = np.array(fib) % 10
            else:
                base_pattern = None

            self.patterns.append(base_pattern)

            if base_pattern is not None:
                # 应用模式并添加噪声
                repeats = sequence_length // pattern_length + 1
                patterned = np.tile(base_pattern, repeats)[:sequence_length]
                noise = np.random.randint(0, 3, size=sequence_length) * 0
                self.data[i] = (patterned + noise) % 10
            else:
                # 完全随机
                self.data[i] = np.random.randint(0, 10, size=sequence_length)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)

    def get_pattern(self, idx):
        """获取指定样本的base_pattern"""
        return self.patterns[idx]

    def describe_pattern(self, idx=0):
        """返回样本模式的文字描述"""
        if self.pattern_type == PatternType.RANDOM:
            return "Completely random sequence"

        base_pattern = self.patterns[idx]
        if self.pattern_type == PatternType.ASCENDING:
            return f"Ascending pattern (mod 10) starting with {base_pattern[0]}, length {self.pattern_length}"
        elif self.pattern_type == PatternType.DESCENDING:
            return f"Descending pattern (mod 10) starting with {base_pattern[0]}, length {self.pattern_length}"
        elif self.pattern_type == PatternType.REPEAT:
            return f"Repeating pattern {base_pattern}"
        elif self.pattern_type == PatternType.ODDEVEN:
            return f"Odd-even alternating pattern starting with {base_pattern[0]}"
        elif self.pattern_type == PatternType.FIBONACCI:
            return f"Fibonacci-like pattern (mod 10) starting with {base_pattern[:2]}"


# python -m src.data.pattern_digit_dataset
if __name__ == "__main__":
    # 创建不同模式的数据集
    ascending_data = PatternDigitDataset(
        5, 10, PatternType.ASCENDING, pattern_length=10
    )
    repeat_data = PatternDigitDataset(5, 10, PatternType.REPEAT, pattern_length=2)
    fib_data = PatternDigitDataset(5, 10, PatternType.FIBONACCI)

    print("=== Ascending Samples ===")
    for i in range(5):
        print(ascending_data.describe_pattern(i))
        print("Sample:", ascending_data[i])
        print("Base pattern:", ascending_data.get_pattern(i))
        print()

    print("\n=== Repeat Samples ===")
    for i in range(5):
        print(repeat_data.describe_pattern(i))
        print("Sample:", repeat_data[i])
        print("Base pattern:", repeat_data.get_pattern(i))
        print()

    print("\n=== Fibonacci Samples ===")
    for i in range(5):
        print(fib_data.describe_pattern(i))
        print("Sample:", fib_data[i])
        print("Base pattern:", fib_data.get_pattern(i))
        print()
