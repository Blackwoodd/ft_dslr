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

def gradient_descent(X, y, learning_rate=0.1, iterations=1000):
    """Effectue la descente de gradient pour un modèle binaire."""
    n_features = len(X[0])
    m = len(X)
    w = [0.0] * n_features  # initialisation des poids

    for _ in range(iterations):
        y_pred = predict_prob(X, w)

        # calcul du gradient
        gradient = [0.0] * n_features
        for j in range(n_features):
            for i in range(m):
                gradient[j] += (y_pred[i] - y[i]) * X[i][j]
            gradient[j] /= m  # moyenne

        # mise à jour des poids
        for j in range(n_features):
            w[j] -= learning_rate * gradient[j]

    return w

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
    """Normalise les colonnes numériques de X (mean=0, std=1)."""
    # trouver le nombre maximum de colonnes
    n_features = max(len(row) for row in X)

    # compléter les lignes plus courtes avec 0
    X_filled = [row + [0.0]*(n_features - len(row)) for row in X]

    # calcul des moyennes
    means = [sum(row[j] for row in X_filled) / len(X_filled) for j in range(n_features)]
    stds = []
    for j in range(n_features):
        variance = sum((row[j] - means[j])**2 for row in X_filled) / len(X_filled)
        stds.append(variance**0.5 if variance > 0 else 1.0)

    # normalisation
    X_norm = []
    for row in X_filled:
        X_norm.append([(row[j] - means[j]) / stds[j] for j in range(n_features)])

    return X_norm



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py dataset_train.csv")
        sys.exit(1)

    X, y = load_data(sys.argv[1])
    one_vs_all = one_vs_all_labels(y)

    X = normalize(X)

    weights = train_one_vs_all(X, one_vs_all, learning_rate=0.000001, iterations=100)
    print(weights["Gryffindor"])  # poids pour le modèle Gryffindor
