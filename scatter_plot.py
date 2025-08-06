#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

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

course_columns = [col for col in df.select_dtypes(include="number").columns if col != "Index"]

corr_matrix = df[course_columns].corr()

all_pairs = []
for i in range(len(course_columns)):
    for j in range(i + 1, len(course_columns)):
        c1 = course_columns[i]
        c2 = course_columns[j]
        corr_value = corr_matrix.loc[c1, c2]
        all_pairs.append((c1, c2, corr_value))

print("All pairs correlations:\n")
for c1, c2, corr_val in all_pairs:
    print(f"{c1} & {c2} : correlation = {corr_val:.3f}")

threshold = 0.7
filtered_pairs = [(c1, c2, corr) for c1, c2, corr in all_pairs if abs(corr) >= threshold]

print(f"\nPairs with correlation >= {threshold}:\n")
for c1, c2, corr in filtered_pairs:
    print(f"{c1} & {c2} : {corr:.3f}")

pairs_to_plot = [(c1, c2) for c1, c2, _ in filtered_pairs]

def plot_all_pairs(pairs):
    cols = 3
    rows = math.ceil(len(pairs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2 * rows), constrained_layout=True)
    axes = axes.flatten()

    for i, (x_course, y_course) in enumerate(pairs):
        ax = axes[i]
        for house in houses:
            subset = df[df["Hogwarts House"] == house]
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

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"|corr| >= {threshold}", fontsize=16)
    plt.show()

plot_all_pairs(pairs_to_plot)
