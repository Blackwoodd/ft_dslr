import numpy as np
import pandas as pd
import sys
import json
from toolkit import my_mean, my_std, my_nan_to_num

IGNORE_FEATURES = [
    "Index", "Hogwarts House", "Transfiguration", "Arithmancy", "Potions",
    "Defense Against the Dark Arts", "Care of Magical Creatures", "Ancient Runes",
    "History of Magic", "Divination"
]

def sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

def gradient_descent(X, y, lr=0.1, epochs=2000, optimizer="batch", batch_size=32, tol=1e-3):
    m, n = X.shape
    theta = np.zeros(n)

    for epoch in range(epochs):
        prev_theta = theta.copy()

        if optimizer == "batch":
            z = X @ theta                       # -> z = θ^T x
            h = sigmoid(z)                      # -> hθ(x) = g(z)
            gradient = (1/m) * (X.T @ (h - y))  # (hθ​(xi​)−yi​)xi
            theta -= lr * gradient

        elif optimizer == "stochastic":
            indices = np.random.permutation(m)
            for i in indices:
                hi = sigmoid(np.dot(X[i], theta))
                theta -= lr * (hi - y[i]) * X[i]

        elif optimizer == "mini-batch":
            indices = np.random.permutation(m)
            for start in range(0, m, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                z = X_batch @ theta
                h = sigmoid(z)
                gradient = (1/len(batch_idx)) * (X_batch.T @ (h - y_batch))
                theta -= lr * gradient

        else:
            raise ValueError("Unknown optimizer: choose 'batch', 'stochastic', or 'mini-batch'")

        # early stopping check
        if np.linalg.norm(theta - prev_theta, ord=1) < tol:
            print(f"Converged at epoch {epoch+1}")
            break

    return theta

def one_vs_all(X, y, classes, lr=0.1, epochs=2000, optimizer="batch", batch_size=32):
    weights = {}
    for c in classes:
        y_c = (y == c).astype(int)
        theta = gradient_descent(X, y_c, lr, epochs, optimizer, batch_size)
        weights[c] = theta.tolist()
    return weights

def predict_one_vs_all(X: np.ndarray, weights: dict, classes: np.ndarray) -> list:
    all_theta = np.array([np.array(weights[c]) for c in classes])
    probs = sigmoid(X @ all_theta.T)
    preds = classes[np.argmax(probs, axis=1)]
    return preds

def main():
    if len(sys.argv) < 2:
        print("Usage: python logreg_train.py dataset_train.csv [batch(default)|stochastic|mini-batch]")
        return

    data = pd.read_csv(sys.argv[1])
    optimizer = sys.argv[2] if len(sys.argv) > 2 else "batch"

    y = data["Hogwarts House"].values

    # keep only numeric columns
    X = data.drop(IGNORE_FEATURES, axis=1, errors="ignore")
    X = X.select_dtypes(include=[np.number])
    print("using features:", list(X.columns))
    X = my_nan_to_num(X.to_numpy())

    means = my_mean(X)
    stds = my_std(X, means) + 1e-8

    # normalize and add bias
    X = (X - means) / stds
    X = np.c_[np.ones(X.shape[0]), X]

    classes = np.unique(y)
    weights = one_vs_all(X, y, classes, optimizer=optimizer, batch_size=32)

    # compute accuracy
    y_pred = predict_one_vs_all(X, weights, classes)
    accuracy = np.mean(y_pred == y) * 100
    print(f"🎯 Training accuracy: {accuracy:.2f}%")

    # save model
    model = {
        "classes": classes.tolist(),
        "weights": weights,
        "means": means.tolist(),
        "stds": stds.tolist()
    }
    with open("weights.json", "w") as f:
        json.dump(model, f)

    print(f"✅ Training finished using {optimizer} gradient descent, weights saved to weights.json")


if __name__ == "__main__":
    main()
