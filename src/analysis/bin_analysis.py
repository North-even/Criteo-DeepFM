import numpy as np


def bin_analysis(y_true, y_pred, num_bins=10):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    bins = np.linspace(0, 1, num_bins + 1)

    print("\n=== Bin Analysis ===")

    for i in range(num_bins):
        left = bins[i]
        right = bins[i + 1]

        mask = (y_pred >= left) & (y_pred < right)

        if mask.sum() == 0:
            continue

        ctr = y_true[mask].mean()
        print(f"{left:.1f}-{right:.1f}: count={mask.sum()}, ctr={ctr:.4f}")