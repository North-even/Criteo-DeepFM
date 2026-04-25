from pathlib import Path
from tqdm import tqdm


def split_criteo(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    if not 0 <= valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1")

    if train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be less than 1")

    test_ratio = 1.0 - train_ratio - valid_ratio

    train_path = output_dir / "train.txt"
    valid_path = output_dir / "valid.txt"
    test_path = output_dir / "test.txt"

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(train_path, "w", encoding="utf-8") as ftrain, \
         open(valid_path, "w", encoding="utf-8") as fvalid, \
         open(test_path, "w", encoding="utf-8") as ftest:

        for idx, line in enumerate(tqdm(fin)):
            r = (idx % 10_000) / 10_000

            if r < train_ratio:
                ftrain.write(line)
            elif r < train_ratio + valid_ratio:
                fvalid.write(line)
            else:
                ftest.write(line)

    print("Split finished.")
    print(f"train_ratio: {train_ratio}")
    print(f"valid_ratio: {valid_ratio}")
    print(f"test_ratio:  {test_ratio}")
    print(f"train: {train_path}")
    print(f"valid: {valid_path}")
    print(f"test:  {test_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    split_criteo(
        input_path=project_root / "data" / "train.txt",
        output_dir=project_root / "data" / "processed" / "criteo_split",
        train_ratio=0.8,
        valid_ratio=0.1,
    )