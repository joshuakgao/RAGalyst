"""Plot the impact of k in LabeledFewShot on MIPROv2 correctness and answerability with a broken y-axis."""

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

# Data
k_values = [1, 2, 4, 8, 16, 32, 64, 128]

miprov2_correctness = [0.871, 0.882, 0.891, 0.894, 0.893, 0.881, 0.881, 0.877]
answerability = [0.626, 0.632, 0.579, 0.622, 0.538, 0.528, 0.441, 0.452]

# Create subplots with shared x-axis for broken y-axis effect
fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1, sharex=True, figsize=(10, 6), gridspec_kw={"height_ratios": [1, 1]}
)

# Plot on both axes
ax_top.plot(k_values, miprov2_correctness, marker="o", label="MIPROv2 Optimized Answer Correctness")
ax_top.plot(k_values, answerability, marker="o", label="Answerability")

ax_bottom.plot(
    k_values, miprov2_correctness, marker="o", label="MIPROv2 Optimized Answer Correctness"
)
ax_bottom.plot(k_values, answerability, marker="o", label="Answerability")

# Add value labels (slightly bigger)
for x, y in zip(k_values, miprov2_correctness):
    ax_top.text(x, y - 0.006, f"{y:.3f}", ha="center", fontsize=14)

for x, y in zip(k_values, answerability):
    ax_bottom.text(x, y + 0.02, f"{y:.3f}", ha="center", fontsize=14)

# Set log scale for x-axis
ax_top.set_xscale("log")
ax_bottom.set_xscale("log")

# Force ticks to be the exact k_values and remove minor ticks
for ax in (ax_top, ax_bottom):
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values], fontsize=12)  # bigger tick labels
    ax.yaxis.set_tick_params(labelsize=12)  # bigger y-axis tick labels
    ax.xaxis.set_minor_locator(NullLocator())  # remove minor ticks

# Remove ALL x-axis ticks/labels on the top axis
ax_top.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

# Y-axis limits for broken axis
ax_top.set_ylim(0.85, 0.90)  # Zoom in on correctness
ax_bottom.set_ylim(0.40, 0.65)  # Zoom in on answerability

# Hide spines between axes
ax_top.spines["bottom"].set_visible(False)
ax_bottom.spines["top"].set_visible(False)

# Add slanted lines to indicate the break
d = 0.005
kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False)
ax_top.plot((-d, +d), (-d, +d), **kwargs)
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

kwargs.update(transform=ax_bottom.transAxes)
ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# Labels and title (slightly bigger)
fig.suptitle("Impact of LabeledFewShot k on Metric Performance", fontsize=16)
ax_bottom.set_xlabel("k Examples Retrieved", fontsize=16)

# Single y-axis label across both subplots
fig.text(0.04, 0.5, "Score", va="center", rotation="vertical", fontsize=16)

# Legend with bigger font
ax_top.legend(fontsize=14)

plt.tight_layout()
plt.subplots_adjust(hspace=0.05, left=0.12)  # leave space for shared ylabel
plt.savefig("k_vs_metrics.png", dpi=300)
