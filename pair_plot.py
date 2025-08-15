#!/usr/bin/env python3

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# df = pd.read_csv("dataset_train.csv")
# df.columns = df.columns.str.strip()
# df = df.dropna(subset=["Hogwarts House"])

# house_colors = {
#     "Gryffindor": "#AE0001",
#     "Hufflepuff": "#FFDB00",
#     "Ravenclaw": "#222F5B",
#     "Slytherin": "#2A623D",
# }

# rename_map = {
#     "Arithmancy": "Arithm",
#     "Astronomy": "Astro",
#     "Herbology": "Herbs",
#     "Defense Against the Dark Arts": "DarkArts",
#     "Divination": "Divinat",
#     "Muggle Studies": "Muggles",
#     "Ancient Runes": "Runes",
#     "History of Magic": "History",
#     "Transfiguration": "Transfig",
#     "Potions": "Potions",
#     "Care of Magical Creatures": "Creatures",
#     "Charms": "Charms",
#     "Flying": "Flying"
# }

# course_columns = [col for col in df.select_dtypes(include="number").columns if col != "Index"]
# df = df.rename(columns=rename_map)
# renamed_columns = [rename_map[col] for col in course_columns if col in rename_map]

# pair_plot = sns.pairplot(
#     df,
#     vars=renamed_columns,
#     hue="Hogwarts House",
#     palette=house_colors,
#     plot_kws={"alpha": 0.5, "s": 30},
# )

# for ax in pair_plot.axes.flatten():
#     ax.set_xticks([])
#     ax.set_yticks([])

# pair_plot._legend.remove()
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.02, left=0.01)
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# === Rename map for shorter labels ===
rename_map = {
    "Arithmancy": "Arithm",
    "Astronomy": "Astro",
    "Herbology": "Herbs",
    "Defense Against the Dark Arts": "DarkArts",
    "Divination": "Divinat",
    "Muggle Studies": "Muggles",
    "Ancient Runes": "Runes",
    "History of Magic": "History",
    "Transfiguration": "Transfig",
    "Potions": "Potions",
    "Care of Magical Creatures": "Creatures",
    "Charms": "Charms",
    "Flying": "Flying"
}

# === Find numeric columns manually ===
course_columns = []
for col in df.columns:
    if col != "Index" and col != "Hogwarts House":
        try:
            df[col].astype(float)
            course_columns.append(col)
        except ValueError:
            pass

num_vars = len(course_columns)

# === Create figure ===
fig, axes = plt.subplots(num_vars, num_vars, figsize=(4 * num_vars, 4 * num_vars))
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# === Loop through grid ===
for i in range(num_vars):
    for j in range(num_vars):
        ax = axes[i, j]
        ax.set_xticks([])
        ax.set_yticks([])
        if i == j:
            # === Diagonal: histogram ===
            course = course_columns[i]
            num_bins = 15
            for house in houses:
                values = df[df["Hogwarts House"] == house][course].dropna().to_numpy()
                if len(values) == 0:
                    continue

                min_val = values.min()
                max_val = values.max()
                bin_edges = np.linspace(min_val, max_val, num_bins + 1)
                counts = np.zeros(num_bins, dtype=int)

                for val in values:
                    placed = False
                    for b in range(num_bins):
                        if bin_edges[b] <= val < bin_edges[b + 1]:
                            counts[b] += 1
                            placed = True
                            break
                    if not placed and val == max_val:
                        counts[-1] += 1

                bin_width = bin_edges[1] - bin_edges[0]
                ax.bar(bin_edges[:-1], counts, width=bin_width,
                       alpha=0.5, color=house_colors[house], align="edge")

        else:
            # === Off-diagonal: scatter plot ===
            x_course = course_columns[j]
            y_course = course_columns[i]
            for house in houses:
                subset = df[df["Hogwarts House"] == house]
                ax.scatter(
                    subset[x_course],
                    subset[y_course],
                    alpha=0.5,
                    s=10,
                    color=house_colors[house]
                )

        # === Remove axis ticks for cleaner look ===
        if i < num_vars - 1:
            ax.set_xticks([])
        if j > 0:
            ax.set_yticks([])

        # === Only label the leftmost and bottom axes, using rename_map ===
        if j == 0:
            ax.set_ylabel(rename_map.get(course_columns[i], course_columns[i]), fontsize=8)
        else:
            ax.set_ylabel("")
        if i == num_vars - 1:
            ax.set_xlabel(rename_map.get(course_columns[j], course_columns[j]), fontsize=8)
        else:
            ax.set_xlabel("")

plt.tight_layout()
plt.subplots_adjust(bottom=0.02, left=0.01)
plt.show()
