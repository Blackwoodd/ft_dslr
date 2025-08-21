#!/usr/bin/env python3
import numpy as np
import sys
import csv
import math

def sigmoid(z):
    if z < -700:  # math.exp(-z) overflow
        return 0.0
    elif z > 700:
        return 1.0
    return 1 / (1 + math.exp(-z))


def predict_prob(X, w):
    """Retourne la probabilité pour chaque ligne de X."""
    probs = []
    for row in X:
        z = sum(row[j] * w[j] for j in range(len(w)))  # produit scalaire
        probs.append(sigmoid(z))
    return probs

# def gradient_descent(X, y, learning_rate=0.1, iterations=1000):
#     """Effectue la descente de gradient pour un modèle binaire."""
#     n_features = len(X[0])
#     m = len(X)
#     w = [0.0] * n_features  # initialisation des poids

#     for _ in range(iterations):
#         y_pred = predict_prob(X, w)

#         # calcul du gradient
#         gradient = [0.0] * n_features
#         for j in range(n_features):
#             for i in range(m):
#                 gradient[j] += (y_pred[i] - y[i]) * X[i][j]
#             gradient[j] /= m  # moyenne

#         # mise à jour des poids
#         for j in range(n_features):
#             w[j] -= learning_rate * gradient[j]

#     return w

def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    X = np.array(X)
    y = np.array(y)
    m, n = X.shape
    w = np.zeros(n)

    for _ in range(iterations):
        z = X.dot(w)
        y_pred = 1 / (1 + np.exp(-np.clip(z, -700, 700)))  # évite overflow
        gradient = (1/m) * X.T.dot(y_pred - y)
        w -= learning_rate * gradient

        if np.any(np.isnan(w)):
            print("⚠️ NaN détecté dans w")
            break

    return w.tolist()


def train_one_vs_all(X, one_vs_all_labels, learning_rate=0.1, iterations=1000):
    """Entraîne un modèle binaire pour chaque classe."""
    weights_all = {}
    for cls, y_bin in one_vs_all_labels.items():
        print(f"Entraînement modèle pour {cls}...")
        w = gradient_descent(X, y_bin, learning_rate, iterations)
        weights_all[cls] = w
    return weights_all


def load_data(path):
    """Charge uniquement les colonnes numériques et le label House."""
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        house_index = header.index("Hogwarts House")

        X = []
        y = []

        for row in reader:
            label = row[house_index]
            features = []

            for i, val in enumerate(row):
                if i == house_index:
                    continue
                try:
                    features.append(float(val))  # garder seulement les numériques
                except ValueError:
                    continue  # ignorer les non numériques

            # ne plus vérifier expected_len
            if len(features) == 0:
                continue  # ignorer les lignes sans features numériques

            X.append(features)
            y.append(label)

    if len(X) == 0:
        raise ValueError("Aucune ligne n’a été chargée : vérifie ton CSV et les colonnes numériques.")

    print("Exemple de X après filtrage:", X[0])
    print("Exemple de y:", y[0])

    return X, y





def one_vs_all_labels(y):
    """Crée un dictionnaire de labels binaires pour One-vs-All."""
    classes = list(set(y))
    one_vs_all = {}

    for c in classes:
        binary_labels = [1 if label == c else 0 for label in y]
        one_vs_all[c] = binary_labels

    return one_vs_all

def normalize(X):
    import math

    n_features = max(len(row) for row in X)
    X_filled = [row + [0.0]*(n_features - len(row)) for row in X]

    means = []
    stds = []

    for j in range(n_features):
        col = [row[j] for row in X_filled]

        # ⚠️ filtrer les NaN
        col_clean = [v for v in col if not math.isnan(v)]
        if len(col_clean) == 0:
            means.append(0.0)
            stds.append(1.0)
            continue

        mean = sum(col_clean) / len(col_clean)
        variance = sum((v - mean)**2 for v in col_clean) / len(col_clean)
        std = math.sqrt(variance) if variance > 0 else 1.0

        means.append(mean)
        stds.append(std)

    # appliquer normalisation
    X_norm = []
    for row in X_filled:
        new_row = []
        for j in range(n_features):
            val = row[j]
            if math.isnan(val):  # remplacer NaN par 0
                new_row.append(0.0)
            else:
                new_row.append((val - means[j]) / stds[j])
        X_norm.append(new_row)

    return X_norm

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py dataset_train.csv")
        sys.exit(1)

    X, y = load_data(sys.argv[1])
    one_vs_all = one_vs_all_labels(y)

    X = normalize(X)
    weights = train_one_vs_all(X, one_vs_all, learning_rate=0.01, iterations=5000)
    with open("weights.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["House"] + [f"w{i}" for i in range(len(next(iter(weights.values()))))])
        for house, w in weights.items():
            writer.writerow([house] + w)

    print("✅ Poids sauvegardés dans weights.csv")