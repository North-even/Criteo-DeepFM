import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.dataset import CTRDataset
from src.models.deepfm import DeepFMModel
from src.analysis.bin_analysis import bin_analysis
from src.utils.metrics import compute_auc, compute_logloss


def infer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = project_root / "outputs" / "deepfm_dim16" / "best.pt"
    data_path = project_root / "data" / "processed" / "criteo_split" / "valid.txt"

    checkpoint = torch.load(model_path, map_location=device)

    hash_size = checkpoint["hash_size"]
    embed_dim = checkpoint["embed_dim"]
    mlp_dims = checkpoint["mlp_dims"]
    dropout = checkpoint["dropout"]

    model = DeepFMModel(
        hash_size=hash_size,
        num_dense=13,
        num_sparse=26,
        embed_dim=embed_dim,
        mlp_dims=mlp_dims,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = CTRDataset(str(data_path), hash_size=hash_size)

    loader = DataLoader(
        dataset,
        batch_size=1024,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    all_y = []
    all_pred = []

    with torch.no_grad():
        for y, dense, sparse in tqdm(loader, desc="Inferencing"):
            y = y.to(device, non_blocking=True)
            dense = dense.to(device, non_blocking=True)
            sparse = sparse.to(device, non_blocking=True)

            logits = model(dense, sparse)
            pred = torch.sigmoid(logits)

            all_y.extend(y.detach().cpu().numpy().tolist())
            all_pred.extend(pred.detach().cpu().numpy().tolist())

    auc = compute_auc(all_y, all_pred)
    logloss = compute_logloss(all_y, all_pred)

    print(f"\nAUC: {auc:.6f}")
    print(f"LogLoss: {logloss:.6f}")

    bin_analysis(all_y, all_pred)


if __name__ == "__main__":
    infer()