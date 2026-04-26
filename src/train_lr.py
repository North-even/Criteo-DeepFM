import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.dataset import CTRDataset
from src.models.lr import LRModel
from src.utils.metrics import compute_auc, compute_logloss


def evaluate(model, data_path, hash_size, batch_size, device):
    dataset = CTRDataset(str(data_path), hash_size=hash_size)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    model.eval()

    all_y = []
    all_pred = []

    with torch.no_grad():
        for y, dense, sparse in tqdm(loader, desc="Evaluating"):
            y = y.to(device)
            dense = dense.to(device)
            sparse = sparse.to(device)

            logits = model(dense, sparse)
            pred = torch.sigmoid(logits)

            all_y.extend(y.detach().cpu().numpy().tolist())
            all_pred.extend(pred.detach().cpu().numpy().tolist())

    auc = compute_auc(all_y, all_pred)
    logloss = compute_logloss(all_y, all_pred)

    return auc, logloss


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = project_root / "data" / "processed" / "criteo_split"
    train_path = data_dir / "train.txt"
    valid_path = data_dir / "valid.txt"

    output_dir = project_root / "outputs" / "lr_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    hash_size = 1_000_000
    batch_size = 1024
    lr = 1e-3
    epochs = 3

    train_dataset = CTRDataset(str(train_path), hash_size=hash_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    model = LRModel(hash_size=hash_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0

    log_path = output_dir / "train_log.txt"

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(
            f"model=LR\n"
            f"device={device}\n"
            f"hash_size={hash_size}\n"
            f"batch_size={batch_size}\n"
            f"lr={lr}\n"
            f"epochs={epochs}\n\n"
        )

        for epoch in range(1, epochs + 1):
            model.train()

            total_loss = 0.0
            total_steps = 0

            progress = tqdm(train_loader, desc=f"Epoch {epoch}")

            for step, (y, dense, sparse) in enumerate(progress):
                y = y.to(device, non_blocking=True)
                dense = dense.to(device, non_blocking=True)
                sparse = sparse.to(device, non_blocking=True)

                logits = model(dense, sparse)
                loss = criterion(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_steps += 1

                if step % 100 == 0:
                    progress.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = total_loss / total_steps

            valid_auc, valid_logloss = evaluate(
                model=model,
                data_path=valid_path,
                hash_size=hash_size,
                batch_size=batch_size,
                device=device
            )

            msg = (
                f"epoch={epoch}, "
                f"train_loss={avg_train_loss:.6f}, "
                f"valid_auc={valid_auc:.6f}, "
                f"valid_logloss={valid_logloss:.6f}"
            )

            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

            if valid_auc > best_auc:
                best_auc = valid_auc
                save_path = output_dir / "best.pt"
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved to {save_path}")

    print("Training finished.")
    print(f"Best valid AUC: {best_auc:.6f}")


if __name__ == "__main__":
    train()