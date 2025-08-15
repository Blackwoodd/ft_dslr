#!/usr/bin/env python3

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import math

# df = pd.read_csv("dataset_train.csv")
# df.columns = df.columns.str.strip()
# df = df.dropna(subset=["Hogwarts House"])
# houses = df["Hogwarts House"].unique()

# house_colors = {
#     "Gryffindor": "#AE0001",
#     "Hufflepuff": "#FFDB00",
#     "Ravenclaw": "#222F5B",
#     "Slytherin": "#2A623D",
# }

# course_columns = [col for col in df.select_dtypes(include="number").columns if col != "Index"]

# rows = math.ceil(len(course_columns) / 3)
# cols = 3

# fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), constrained_layout=True)
# axes = axes.flatten()

# for i, course in enumerate(course_columns):
#     ax = axes[i]
#     for house in houses:
#         sns.histplot(
#             df[df["Hogwarts House"] == house][course],
#             ax=ax,
#             bins=15,
#             element="step",
#             alpha=0.6,
#             color=house_colors.get(house)
#         )
#     ax.set_title(course)
#     ax.set_xlabel("")
#     ax.set_ylabel("")
#     ax.set_xticks([])
#     ax.set_yticks([])

# for j in range(i + 1, len(axes)):
#     fig.delaxes(axes[j])

# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

# === Tries to convert column to float|If fails → it's non-numeric → skip ===
course_columns = []
for col in df.columns:
    if col != "Index":
        try:
            df[col].astype(float)
            course_columns.append(col)
        except ValueError:
            pass

# === Histogram parameters ===
num_bins = 30
rows = math.ceil(len(course_columns) / 3)
cols = 3

fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), constrained_layout=True)
axes = axes.flatten()

# === Plot histograms manually ===
for i, course in enumerate(course_columns):
    ax = axes[i]
    for house in houses:
        values = df[df["Hogwarts House"] == house][course].dropna().to_numpy()

        if len(values) == 0:
            continue

        # Calculate bin edges manually
        min_val = values.min()
        max_val = values.max()
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)

        # Count values per bin manually
        counts = np.zeros(num_bins, dtype=int)
        for val in values:
            # Find bin index
            placed = False
            for b in range(num_bins):
                if bin_edges[b] <= val < bin_edges[b + 1]:
                    counts[b] += 1
                    placed = True
                    break
            # Edge case: max value
            if not placed and val == max_val:
                counts[-1] += 1

        # Plot bars manually
        bin_width = bin_edges[1] - bin_edges[0]
        ax.bar(bin_edges[:-1], counts, width=bin_width, alpha=0.5,
               color=house_colors[house], align="edge", label=house)

    ax.set_title(course)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

# Remove unused subplots if any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.show()
