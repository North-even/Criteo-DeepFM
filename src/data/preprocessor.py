import math
import hashlib


def stable_hash(text: str, hash_size: int) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % hash_size


def process_line(line: str, hash_size: int = 1_000_000):
    fields = line.rstrip("\r\n").split("\t")

    if len(fields) != 40:
        raise ValueError(f"Expected 40 fields, got {len(fields)}. Raw line tail may contain empty fields.")

    # label
    y = int(fields[0])

    # dense features: I1 ~ I13
    dense = []
    for j, val in enumerate(fields[1:14], start=1):
        if val == "":
            v = 0.0
        else:
            v = float(val)

        # 防御性处理，避免 log1p 域错误
        if v < 0:
            v = 0.0

        v = math.log1p(v)
        dense.append(v)

    # sparse features: C1 ~ C26
    sparse = []
    for i, val in enumerate(fields[14:], start=14):
        if val == "":
            val = "missing"
        sparse.append(stable_hash(f"C{i}_{val}", hash_size))

    return y, dense, sparse