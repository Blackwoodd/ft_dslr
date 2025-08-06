#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_train.csv")
df.columns = df.columns.str.strip()
df = df.dropna(subset=["Hogwarts House"])

house_colors = {
    "Gryffindor": "#AE0001",
    "Hufflepuff": "#FFDB00",
    "Ravenclaw": "#222F5B",
    "Slytherin": "#2A623D",
}

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

course_columns = [col for col in df.select_dtypes(include="number").columns if col != "Index"]
df = df.rename(columns=rename_map)
renamed_columns = [rename_map[col] for col in course_columns if col in rename_map]

pair_plot = sns.pairplot(
    df,
    vars=renamed_columns,
    hue="Hogwarts House",
    palette=house_colors,
    plot_kws={"alpha": 0.5, "s": 30},
)

for ax in pair_plot.axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

pair_plot._legend.remove()
plt.tight_layout()
plt.subplots_adjust(bottom=0.02, left=0.01)
plt.show()
