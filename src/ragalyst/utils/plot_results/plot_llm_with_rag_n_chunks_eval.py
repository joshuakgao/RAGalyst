"""Plot evaluation results for LLM with RAG varying number of chunks."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ==============================
# Embed your data directly
# ==============================
data = [
    (
        "army_20250826_092932.json",
        "army",
        1,
        False,
        0.8800786266092534,
        0.8128901023167915,
        0.7923999999999979,
    ),
    (
        "engineering_20250827_082558.json",
        "engineering",
        1,
        False,
        0.905348191110575,
        0.8459465803523595,
        0.7871999999999969,
    ),
    (
        "cybersecurity_20250827_082604.json",
        "cybersecurity",
        1,
        False,
        0.8570614512938048,
        0.8597018312794023,
        0.7759999999999972,
    ),
    (
        "army_20250826_095658.json",
        "army",
        2,
        False,
        0.8999131830845455,
        0.8345361077856339,
        0.8079999999999972,
    ),
    (
        "engineering_20250827_084903.json",
        "engineering",
        2,
        False,
        0.9006247862388733,
        0.8313912785441534,
        0.8117999999999963,
    ),
    (
        "cybersecurity_20250827_090027.json",
        "cybersecurity",
        2,
        False,
        0.892525324185619,
        0.8681539520618554,
        0.800199999999997,
    ),
    (
        "army_20250826_102657.json",
        "army",
        3,
        False,
        0.8744987703282492,
        0.8530506789342233,
        0.8103999999999967,
    ),
    (
        "cybersecurity_20250827_114944.json",
        "cybersecurity",
        3,
        False,
        0.8887942482783819,
        0.8708048856056508,
        0.7967999999999962,
    ),
    (
        "engineering_20250827_174900.json",
        "engineering",
        3,
        False,
        0.8962295418633659,
        0.8472329160897378,
        0.8081999999999963,
    ),
    (
        "army_20250826_110239.json",
        "army",
        4,
        False,
        0.8839885397965281,
        0.8410555081306537,
        0.8199999999999962,
    ),
    (
        "cybersecurity_20250827_115849.json",
        "cybersecurity",
        4,
        False,
        0.8925213861373071,
        0.876837281079574,
        0.7923999999999962,
    ),
    (
        "engineering_20250827_175505.json",
        "engineering",
        4,
        False,
        0.8938996431960529,
        0.8659714612368036,
        0.8115999999999963,
    ),
    (
        "army_20250826_114044.json",
        "army",
        5,
        False,
        0.863090188469493,
        0.8710526066286024,
        0.8203999999999966,
    ),
    (
        "cybersecurity_20250827_124543.json",
        "cybersecurity",
        5,
        False,
        0.867211416524652,
        0.8835571345278143,
        0.7839999999999961,
    ),
    (
        "engineering_20250827_183141.json",
        "engineering",
        5,
        False,
        0.8871850406129819,
        0.8681389869879019,
        0.8167999999999962,
    ),
    (
        "army_20250826_121743.json",
        "army",
        6,
        False,
        0.8761472650242779,
        0.855754843383564,
        0.8121999999999967,
    ),
    (
        "cybersecurity_20250827_133215.json",
        "cybersecurity",
        6,
        False,
        0.8831226817059794,
        0.8986719150678125,
        0.7819999999999961,
    ),
    (
        "engineering_20250827_191048.json",
        "engineering",
        6,
        False,
        0.8912234601114035,
        0.8730917317507714,
        0.7931999999999967,
    ),
    (
        "army_20250826_141340.json",
        "army",
        7,
        False,
        0.8636270495484691,
        0.8522923621449703,
        0.7941999999999971,
    ),
    (
        "cybersecurity_20250827_142016.json",
        "cybersecurity",
        7,
        False,
        0.8788653617310436,
        0.8933831825873787,
        0.789199999999996,
    ),
    (
        "engineering_20250827_195331.json",
        "engineering",
        7,
        False,
        0.8870275829261903,
        0.8592364436135225,
        0.7915999999999966,
    ),
    (
        "army_20250826_142148.json",
        "army",
        8,
        False,
        0.8749475144443416,
        0.8544036528733967,
        0.8081999999999963,
    ),
    (
        "cybersecurity_20250827_150842.json",
        "cybersecurity",
        8,
        False,
        0.8759068741543984,
        0.8920213835661441,
        0.7861999999999962,
    ),
    (
        "engineering_20250827_203638.json",
        "engineering",
        8,
        False,
        0.8909942223754923,
        0.8779553754727734,
        0.8013999999999962,
    ),
    (
        "army_20250826_151041.json",
        "army",
        9,
        False,
        0.8649858233184087,
        0.8542359659423469,
        0.7909999999999974,
    ),
    (
        "cybersecurity_20250827_155857.json",
        "cybersecurity",
        9,
        False,
        0.8858428156037292,
        0.9006083593756651,
        0.7789999999999967,
    ),
    (
        "engineering_20250827_212059.json",
        "engineering",
        9,
        False,
        0.8632707121377303,
        0.8721448544799612,
        0.7903999999999963,
    ),
    (
        "army_20250826_155620.json",
        "army",
        10,
        False,
        0.8784688294333809,
        0.8643691624053992,
        0.7919999999999964,
    ),
    (
        "cybersecurity_20250827_165115.json",
        "cybersecurity",
        10,
        False,
        0.8753174366998678,
        0.8773878867286045,
        0.7685999999999964,
    ),
    (
        "engineering_20250827_220730.json",
        "engineering",
        10,
        False,
        0.8959642937300059,
        0.8729581202268785,
        0.7877999999999963,
    ),
]

df = pd.DataFrame(
    data,
    columns=[
        "file",
        "domain",
        "top_k",
        "order_preserve",
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
    ],
)

# Sort for plotting
df = df.sort_values(["domain", "top_k"])

# Metrics and their colors for subplots
metrics = ["faithfulness", "answer_relevancy", "answer_correctness"]
metric_titles = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "answer_correctness": "Answer Correctness",
}

# Colors and markers for domains
domains = df["domain"].unique()
domain_colors = {
    domain: color for domain, color in zip(domains, ["tab:blue", "tab:orange", "tab:green"])
}
domain_markers = {domain: marker for domain, marker in zip(domains, ["o", "s", "^"])}

# Create output directory
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True, parents=True)

# Create subplots: one per metric
fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 4 * len(metrics)), sharex=True)

if len(metrics) == 1:
    axes = [axes]  # make iterable

for ax, metric in zip(axes, metrics):
    for domain in domains:
        group = df[df["domain"] == domain]
        ax.plot(
            group["top_k"],
            group[metric],
            marker=domain_markers[domain],
            label=domain,
            color=domain_colors[domain],
            linestyle="-",
        )

        # Highlight the best top_k for this domain & metric
        best_row = group.loc[group[metric].idxmax()]
        best_top_k = best_row["top_k"]
        y_val = best_row[metric]

        # ax.plot(
        #     best_top_k,
        #     y_val,
        #     marker=domain_markers[domain],
        #     markersize=12,
        #     color=domain_colors[domain],
        # )

        # ax.text(
        #     best_top_k,
        #     y_val + 0.005,
        #     f"{y_val:.3f}",
        #     color=domain_colors[domain],
        #     fontsize=10,
        #     ha="center",
        # )

    ax.set_ylabel("Score", fontsize=16)
    ax.set_title(f"{metric_titles[metric]}", fontsize=16)
    ax.grid(True)

axes[-1].set_xlabel("Number of Chunks", fontsize=16)

# Shared legend for domains
handles, labels = axes[0].get_legend_handles_labels()
labels = [label.capitalize() for label in labels]  # capitalize each domain
fig.legend(handles, labels, loc="lower center", ncol=len(domains), fontsize=14)

fig.suptitle("Metrics vs Number of Chunks", fontsize=16, x=0.53)
plt.tight_layout(rect=(0, 0.05, 1, 0.97))

plt.savefig(output_dir / "llm_with_rag_n_chunks_eval.png", dpi=300)
plt.close()
