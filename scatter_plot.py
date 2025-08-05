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
    "Gryffindor": "#AE0001",
    "Hufflepuff": "#FFDB00",
    "Ravenclaw": "#222F5B",
    "Slytherin": "#2A623D",
}

course_columns = df.select_dtypes(include="number").columns

corr_matrix = df[course_columns].corr()

all_pairs = []
for i in range(len(course_columns)):
    for j in range(i + 1, len(course_columns)):
        c1 = course_columns[i]
        c2 = course_columns[j]
        corr_value = corr_matrix.loc[c1, c2]
        all_pairs.append((c1, c2, corr_value))

print("All pairs and their correlations:\n")
for c1, c2, corr_val in all_pairs:
    print(f"{c1} & {c2} : correlation = {corr_val:.3f}")

pairs_to_plot = [(c1, c2) for c1, c2, _ in all_pairs]

sns.set(style="whitegrid")

def plot_two_pairs(pairs):
    batch_size = 2
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        rows = 1  # just 1 row, 2 columns for 2 plots
        cols = 2
        fig, axes = plt.subplots(rows, cols, figsize=(14, 5), tight_layout=True)
        if batch_size == 1:
            axes = [axes]  # make axes iterable if only one plot
        axes = axes.flatten()

        for i, (x_course, y_course) in enumerate(batch):
            ax = axes[i]
            for house in houses:
                subset = df[df["Hogwarts House"] == house]
                ax.scatter(
                    subset[x_course], subset[y_course],
                    label=house,
                    alpha=0.6,
                    s=40,
                    color=house_colors.get(house, "gray")
                )
            ax.set_xlabel(x_course)
            ax.set_ylabel(y_course)
            ax.set_title(f"{x_course} vs {y_course}")
            ax.legend()

        # Hide unused axes if batch smaller than 2
        for j in range(len(batch), cols):
            fig.delaxes(axes[j])

        plt.suptitle(f"Scatter Plots of Hogwarts Courses by House (Pairs {start+1} to {start+len(batch)})", fontsize=16)
        plt.show()

threshold = 0.7

# Filter pairs with |correlation| >= threshold
filtered_pairs = [(c1, c2, corr) for c1, c2, corr in all_pairs if abs(corr) >= threshold]

print(f"\nPairs with correlation magnitude >= {threshold}:\n")
for c1, c2, corr in filtered_pairs:
    print(f"{c1} & {c2} : correlation = {corr:.3f}")

# Extract pairs without correlation for plotting
pairs_to_plot = [(c1, c2) for c1, c2, _ in filtered_pairs]

# Then pass pairs_to_plot to your plotting function
plot_two_pairs(pairs_to_plot)
