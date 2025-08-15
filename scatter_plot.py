#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import math

# === Load dataset ===
df = pd.read_csv("dataset_train.csv")
df.columns = df.columns.str.strip()
df = df.dropna(subset=["Hogwarts House"])

houses = df["Hogwarts House"].unique()

house_colors = {
    "Gryffindor": "#AE0001",
    "Hufflepuff": "#FFDB00",
    "Ravenclaw": "#222F5B",
    "Slytherin": "#2A623D",
}

# === Find numeric columns manually (no select_dtypes) ===
course_columns = []
for col in df.columns:
    if col != "Index" and col != "Hogwarts House":
        try:
            df[col].astype(float)
            course_columns.append(col)
        except ValueError:
            pass

# === Manual Pearson correlation function ===
def pearson_corr(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    den_x = (sum((x[i] - mean_x) ** 2 for i in range(n))) ** 0.5
    den_y = (sum((y[i] - mean_y) ** 2 for i in range(n))) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0
    return num / (den_x * den_y)

# === Compute correlations manually ===
all_pairs = []
for i in range(len(course_columns)):
    for j in range(i + 1, len(course_columns)):
        c1 = course_columns[i]
        c2 = course_columns[j]

        # Drop NaN rows for this pair
        valid_rows = df[[c1, c2]].dropna()
        x_vals = valid_rows[c1].to_list()
        y_vals = valid_rows[c2].to_list()

        corr_value = pearson_corr(x_vals, y_vals)
        all_pairs.append((c1, c2, corr_value))

# === Print all correlations ===
print("All pairs correlations:\n")
for c1, c2, corr_val in all_pairs:
    print(f"{c1} & {c2} : correlation = {corr_val:.3f}")

# === Filter pairs by threshold ===
threshold = 0.7
filtered_pairs = [(c1, c2, corr) for c1, c2, corr in all_pairs if abs(corr) >= threshold]

print(f"\nPairs with correlation >= {threshold}:\n")
for c1, c2, corr in filtered_pairs:
    print(f"{c1} & {c2} : {corr:.3f}")

pairs_to_plot = [(c1, c2) for c1, c2, _ in filtered_pairs]

# === Plotting function ===
def plot_all_pairs(pairs):
    cols = 3
    rows = math.ceil(len(pairs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2 * rows), constrained_layout=True)
    axes = axes.flatten()

    for i, (x_course, y_course) in enumerate(pairs):
        ax = axes[i]
        for house in houses:
            subset = df[df["Hogwarts House"] == house]
            subset = subset.dropna(subset=[x_course, y_course])
            ax.scatter(
                subset[x_course], subset[y_course],
                alpha=0.6,
                s=40,
                color=house_colors.get(house)
            )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{x_course} vs {y_course}")

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"|corr| >= {threshold}", fontsize=16)
    plt.show()

# === Plot filtered correlations ===
plot_all_pairs(pairs_to_plot)
