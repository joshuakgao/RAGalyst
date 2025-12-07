"""Plot evaluation results for embedder retrieval."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

# Your input data
recall_data = {
    "Army": [0.86, 0.86, 0.846, 0.912, 0.904, 0.936, 0.93, 0.888],
    "Cybersecurity": [0.866, 0.874, 0.852, 0.904, 0.892, 0.918, 0.928, 0.87],
    "Engineering": [0.872, 0.878, 0.88, 0.924, 0.926, 0.946, 0.95, 0.9],
}

mrr_data = {
    "Army": [0.601, 0.606, 0.582, 0.694, 0.701, 0.761, 0.759, 0.673],
    "Cybersecurity": [0.662, 0.660, 0.648, 0.726, 0.723, 0.756, 0.783, 0.649],
    "Engineering": [0.642, 0.653, 0.668, 0.754, 0.745, 0.801, 0.800, 0.739],
}

models = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "gemini-embedding-001",
    "bge-m3",
    "Qwen3-Embedding-0.6B",
    "Qwen3-Embedding-4B",
    "Qwen3-Embedding-8B",
    "nomic-embed-text-v1",
]

markers = ["o", "s", "v", "^", "<", ">", "D", "p", "*"]

# Convert to DataFrames
df_recall = pd.DataFrame(recall_data, index=models)
df_mrr = pd.DataFrame(mrr_data, index=models)

# Melt to long format
df_recall_long = df_recall.reset_index().melt(id_vars="index", var_name="Task", value_name="Score")
df_recall_long.rename(columns={"index": "Model"}, inplace=True)

df_mrr_long = df_mrr.reset_index().melt(id_vars="index", var_name="Task", value_name="Score")
df_mrr_long.rename(columns={"index": "Model"}, inplace=True)

# Plotting side-by-side
fig, axs = plt.subplots(1, 2, figsize=(7, 6), sharey=False)

# Recall Plot
for i, model in enumerate(df_recall_long["Model"].unique()):
    subset = df_recall_long[df_recall_long["Model"] == model]
    marker = markers[i % len(markers)]
    axs[0].plot(subset["Task"], subset["Score"], marker=marker, label=model)
axs[0].set_title("Recall@10")
axs[0].set_ylabel("Score")
axs[0].set_ylim(0.84, 0.955)
axs[0].yaxis.set_major_locator(MultipleLocator(0.02))
axs[0].grid(True, linestyle="--", alpha=0.5)
axs[0].tick_params(axis="x", rotation=45)

# MRR Plot
for i, model in enumerate(df_mrr_long["Model"].unique()):
    subset = df_mrr_long[df_mrr_long["Model"] == model]
    marker = markers[i % len(markers)]
    axs[1].plot(subset["Task"], subset["Score"], marker=marker, label=model)
axs[1].set_title("MRR@10")
axs[1].set_ylim(0.575, 0.805)
axs[1].yaxis.set_major_locator(MultipleLocator(0.05))
axs[1].grid(True, linestyle="--", alpha=0.5)
axs[1].tick_params(axis="x", rotation=45)

# Shared legend outside the plots
fig.legend(
    df_recall_long["Model"].unique(),
    loc="center right",
    # bbox_to_anchor=(0.97, 0.5),
    fontsize="small",
)

plt.tight_layout(rect=(0.0, 0.0, 0.75, 1.0))
plt.savefig("outputs/embedder_retrieval_eval.png")
plt.show()
