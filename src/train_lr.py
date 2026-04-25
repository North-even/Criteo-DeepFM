import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.dataset import CTRDataset
from src.models.lr import LRModel

def train():
    data_path = project_root / "data" / "train.txt"

    hash_size = 1_000_000
    batch_size = 1024
    lr = 1e-3
    max_steps = 1000

    dataset = CTRDataset(str(data_path), hash_size=hash_size)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    model = LRModel(hash_size=hash_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()

    for step, (y, dense, sparse) in enumerate(tqdm(loader)):
        logits = model(dense, sparse)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"step={step}, loss={loss.item():.6f}")

        if step >= max_steps:
            break

    save_path = project_root / "outputs" / "lr_baseline.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"model saved to {save_path}")


if __name__ == "__main__":
    train()