# histogram.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Load and clean data
df = pd.read_csv("dataset_train.csv")
df.columns = df.columns.str.strip()
df = df.dropna(subset=["Hogwarts House"])
houses = df["Hogwarts House"].unique()

house_colors = {
    "Gryffindor": "#D4170D",
    "Hufflepuff": "#D9A516",
    "Ravenclaw": "#279CF5",
    "Slytherin":  "#008000",
}

# Select course columns (assume numerical features from column 6 onward)
course_columns = df.select_dtypes(include="number").columns
half = len(course_columns) // 2
left_courses = course_columns[:half]
right_courses = course_columns[half:]

sns.set(style="whitegrid")

def plot_course_set(course_list, title):
    rows = math.ceil(len(course_list) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows), constrained_layout=True)
    axes = axes.flatten()

    for i, course in enumerate(course_list):
        ax = axes[i]
        for house in houses:
            sns.histplot(
                df[df["Hogwarts House"] == house][course],
                label=house,
                ax=ax,
                kde=False,
                stat="density",
                bins=15,
                element="step",
                fill=False,
                color=house_colors.get(house, "gray")  # fallback color
            )
        ax.set_title(course)
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.legend()

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=16)

    return fig

fig1 = plot_course_set(left_courses, "Course Score Distributions (Part 1)")
fig2 = plot_course_set(right_courses, "Course Score Distributions (Part 2)")

plt.show()
