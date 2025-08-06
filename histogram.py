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

rows = math.ceil(len(course_columns) / 3)
cols = 3

fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), constrained_layout=True)
axes = axes.flatten()

for i, course in enumerate(course_columns):
    ax = axes[i]
    for house in houses:
        sns.histplot(
            df[df["Hogwarts House"] == house][course],
            ax=ax,
            bins=15,
            element="step",
            alpha=0.6,
            color=house_colors.get(house)
        )
    ax.set_title(course)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.show()
