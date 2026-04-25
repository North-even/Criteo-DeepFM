import torch
from torch.utils.data import IterableDataset

from src.data.preprocessor import process_line


class CTRDataset(IterableDataset):
    def __init__(self, file_path, hash_size=1_000_000):
        self.file_path = file_path
        self.hash_size = hash_size

    def __iter__(self):
        with open(self.file_path, "r") as f:
            for line in f:
                y, dense, sparse = process_line(line, self.hash_size)

                yield(
                    torch.tensor(y, dtype=torch.float32),
                    torch.tensor(dense, dtype=torch.float32),
                    torch.tensor(sparse, dtype=torch.long)
                )