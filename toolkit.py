import numpy as np

def my_mean(arr: np.ndarray) -> np.ndarray:
    """Compute mean of each column in a 2D array."""
    m, n = arr.shape
    means = []
    for j in range(n):
        s = 0.0
        for i in range(m):
            s += arr[i, j]
        means.append(s / m if m > 0 else 0.0)
    return np.array(means)


def my_std(arr: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Compute standard deviation of each column in a 2D array."""
    m, n = arr.shape
    stds = []
    for j in range(n):
        s = 0.0
        for i in range(m):
            diff = arr[i, j] - means[j]
            s += diff * diff
        variance = s / m if m > 0 else 0.0
        stds.append(variance ** 0.5)
    return np.array(stds)


def my_nan_to_num(arr: np.ndarray, replace_val=0.0) -> np.ndarray:
    """Replace NaN and +/- inf values with a chosen replacement value."""
    m, n = arr.shape
    out = arr.copy()
    for i in range(m):
        for j in range(n):
            if out[i, j] != out[i, j]:  # NaN check
                out[i, j] = replace_val
            elif out[i, j] == float("inf") or out[i, j] == float("-inf"):
                out[i, j] = replace_val
    return out
