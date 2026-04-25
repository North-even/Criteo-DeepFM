import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.dataset import CTRDataset
from src.models.fm import FMModel
from src.utils.metrics import compute_auc, compute_logloss


def evaluate(model, data_path, hash_size, batch_size):
    dataset = CTRDataset(str(data_path), hash_size=hash_size)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    model.eval()

    all_y = []
    all_pred = []

    with torch.no_grad():
        for y, dense, sparse in tqdm(loader, desc="Evaluating"):
            logits = model(dense, sparse)
            pred = torch.sigmoid(logits)

            all_y.extend(y.numpy().tolist())
            all_pred.extend(pred.numpy().tolist())

    auc = compute_auc(all_y, all_pred)
    logloss = compute_logloss(all_y, all_pred)

    return auc, logloss


def train():
    data_dir = project_root / "data" / "processed" / "criteo_split"
    train_path = data_dir / "train.txt"
    valid_path = data_dir / "valid.txt"

    hash_size = 1_000_000
    embed_dim = 16
    batch_size = 1024
    lr = 1e-4
    epochs = 3

    output_dir = project_root / "outputs" / f"fm_baseline_dim{embed_dim}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = CTRDataset(str(train_path), hash_size=hash_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0
    )

    model = FMModel(
        hash_size=hash_size,
        num_dense=13,
        embed_dim=embed_dim
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_epoch = 0

    log_path = output_dir / "train_log.txt"

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(
            f"model=FM\n"
            f"hash_size={hash_size}\n"
            f"embed_dim={embed_dim}\n"
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
                batch_size=batch_size
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
                best_epoch = epoch

                save_path = output_dir / "best.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model": "FM",
                        "hash_size": hash_size,
                        "embed_dim": embed_dim,
                        "batch_size": batch_size,
                        "lr": lr,
                        "epoch": epoch,
                        "valid_auc": valid_auc,
                        "valid_logloss": valid_logloss,
                    },
                    save_path
                )

                print(f"Best model saved to {save_path}")

        summary = (
            f"\nTraining finished.\n"
            f"Best epoch: {best_epoch}\n"
            f"Best valid AUC: {best_auc:.6f}\n"
        )

        print(summary)
        log_file.write(summary)


if __name__ == "__main__":
    train()