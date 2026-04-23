import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt

chunk_size = 1_000_000
data_dir = pathlib.Path(__file__).parent.parent / "data"
file_path = data_dir / "train.txt"

num_dense = 13
num_sparse = 26

label_col = 0
dense_cols = list(range(1, 14))      # 1~13
sparse_cols = list(range(14, 40))    # 14~39

# ---------- 全局统计 ----------
total_rows = 0
total_clicks = 0

# ---------- 数值特征统计 ----------
dense_missing = np.zeros(num_dense, dtype=np.int64)
dense_sum = np.zeros(num_dense, dtype=np.float64)
dense_sq_sum = np.zeros(num_dense, dtype=np.float64)
dense_non_missing = np.zeros(num_dense, dtype=np.int64)

# ---------- 类别特征统计 ----------
sparse_missing = np.zeros(num_sparse, dtype=np.int64)
sparse_unique_sets = [set() for _ in range(num_sparse)]

reader = pd.read_csv(
    file_path,
    sep="\t",
    header=None,
    chunksize=chunk_size,
    low_memory=False
)

for chunk_idx, chunk in enumerate(reader):
    print(f"Processing chunk {chunk_idx} ...")

    # 样本数 & CTR
    total_rows += len(chunk)
    total_clicks += chunk[label_col].sum()

    # ========= 数值特征 =========
    dense_chunk = chunk[dense_cols]

    for i, col in enumerate(dense_cols):
        col_data = dense_chunk[col]

        missing_count = col_data.isna().sum()
        dense_missing[i] += missing_count

        valid_data = col_data.dropna()
        dense_non_missing[i] += len(valid_data)
        dense_sum[i] += valid_data.sum()
        dense_sq_sum[i] += (valid_data ** 2).sum()

    # ========= 类别特征 =========
    sparse_chunk = chunk[sparse_cols]

    for i, col in enumerate(sparse_cols):
        col_data = sparse_chunk[col]

        sparse_missing[i] += col_data.isna().sum()

        # 把当前 chunk 的非缺失唯一值加入集合
        sparse_unique_sets[i].update(col_data.dropna().unique())

# =========================
# 计算最终统计量
# =========================

ctr = total_clicks / total_rows

dense_mean = dense_sum / dense_non_missing
dense_var = dense_sq_sum / dense_non_missing - dense_mean ** 2
dense_missing_rate = dense_missing / total_rows

sparse_missing_rate = sparse_missing / total_rows
sparse_cardinality = np.array([len(s) for s in sparse_unique_sets])

# =========================
# 打印结果
# =========================

print("\n===== Global Stats =====")
print(f"Total rows: {total_rows}")
print(f"Total clicks: {total_clicks}")
print(f"CTR: {ctr:.6f}")

print("\n===== Dense Feature Stats =====")
for i in range(num_dense):
    print(
        f"I{i+1}: "
        f"missing_rate={dense_missing_rate[i]:.6f}, "
        f"mean={dense_mean[i]:.6f}, "
        f"var={dense_var[i]:.6f}"
    )

print("\n===== Sparse Feature Stats =====")
for i in range(num_sparse):
    print(
        f"C{i+1}: "
        f"missing_rate={sparse_missing_rate[i]:.6f}, "
        f"cardinality={sparse_cardinality[i]}"
    )

# =========================
# 保存为 DataFrame
# =========================

dense_df = pd.DataFrame({
    "feature": [f"I{i+1}" for i in range(num_dense)],
    "missing_rate": dense_missing_rate,
    "mean": dense_mean,
    "var": dense_var
})

sparse_df = pd.DataFrame({
    "feature": [f"C{i+1}" for i in range(num_sparse)],
    "missing_rate": sparse_missing_rate,
    "cardinality": sparse_cardinality
})

dense_df.to_csv(data_dir / "dense_feature_stats.csv", index=False)
sparse_df.to_csv(data_dir / "sparse_feature_stats.csv", index=False)

print("\nDense stats saved to dense_feature_stats.csv")
print("Sparse stats saved to sparse_feature_stats.csv")