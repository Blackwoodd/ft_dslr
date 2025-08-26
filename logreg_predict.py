import numpy as np
import pandas as pd
import sys
import json

IGNORE_FEATURES = [
    "Index", "Hogwarts House", "Transfiguration", "Arithmancy", "Potions",
    "Defense Against the Dark Arts", "Care of Magical Creatures", "Ancient Runes",
    "History of Magic", "Divination"
]

def sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

def predict_one_vs_all(X: np.ndarray, weights: dict, classes: np.ndarray) -> list:
    all_theta = np.array([np.array(weights[c]) for c in classes])  # shape (num_classes, num_features)
    probs = sigmoid(X @ all_theta.T)  # shape (num_samples, num_classes)
    preds = classes[np.argmax(probs, axis=1)]
    return preds.tolist()

def main():
    if len(sys.argv) != 3:
        print("Usage: python logreg_predict.py <dataset.csv> weights.json")
        return

    data = pd.read_csv(sys.argv[1])
    index = data["Index"] if "Index" in data.columns else range(len(data))

    # keep only numeric columns
    X = data.drop(IGNORE_FEATURES, axis=1, errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X = np.nan_to_num(X)

    # load model
    with open(sys.argv[2], "r") as f:
        model = json.load(f)
    classes = np.array(model["classes"])
    weights = model["weights"]
    means = np.array(model["means"])
    stds = np.array(model["stds"])

    # normalize using training stats
    X = (X - means) / stds
    X = np.c_[np.ones(X.shape[0]), X]

    preds = predict_one_vs_all(X, weights, classes)

    out = pd.DataFrame({"Index": index, "Hogwarts House": preds})
    out.to_csv("houses.csv", index=False)
    print("✅ Predictions saved to houses.csv")

if __name__ == "__main__":
    main()
