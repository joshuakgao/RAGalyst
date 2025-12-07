"""Plot evaluation results for LLM with RAG."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

# Your input data
faithfulness = {
    "Army": [
        0.9320208255796493,
        0.8747398300018133,
        0.9264618088381421,
        0.9445547484373797,
        0.9326572833911378,
        0.8725284640863062,
        0.892190901104848,
        0.9119393196678294,
        0.8777484321486125,
    ],
    "Cybersecurity": [
        0.937706878707808,
        0.9032102884903196,
        0.9318488307283596,
        0.9514310455826711,
        0.9359948863448312,
        0.9054212459323338,
        0.9039516060824178,
        0.941098643816535,
        0.9135962955892574,
    ],
    "Engineering": [
        0.9408653650786779,
        0.887753997407746,
        0.9337934479675811,
        0.940408107850023,
        0.934619769528073,
        0.9113390734878922,
        0.9057913221773417,
        0.9204599253510435,
        0.8968013201397824,
    ],
}
answer_relevancy = {
    "Army": [
        0.9061084645326846,
        0.9425789864716778,
        0.9373845068708304,
        0.9237456488594886,
        0.9148096436604345,
        0.9507191791627235,
        0.9537013376493119,
        0.9472392328367206,
        0.9476208750603404,
    ],
    "Cybersecurity": [
        0.9153999414955503,
        0.9501027186449111,
        0.9502998558026223,
        0.9436457617007147,
        0.9309119546953726,
        0.947453985313526,
        0.9602689290127869,
        0.945711814055199,
        0.9404480015639916,
    ],
    "Engineering": [
        0.9132954290494117,
        0.9425057249934374,
        0.942264670278978,
        0.9276942348560474,
        0.9213189357953568,
        0.9528049072946695,
        0.9554610247332496,
        0.954415127180121,
        0.9437276557012667,
    ],
}
answer_correctness = {
    "Army": [
        0.8453999999999968,
        0.8371999999999958,
        0.8531999999999964,
        0.8551999999999963,
        0.8615071283095685,
        0.8197999999999962,
        0.8309999999999961,
        0.8407999999999961,
        0.8417999999999954,
    ],
    "Cybersecurity": [
        0.8543999999999954,
        0.8255999999999957,
        0.844199999999996,
        0.8653999999999955,
        0.8639175257731923,
        0.7939999999999955,
        0.8177999999999951,
        0.8271999999999952,
        0.8103999999999957,
    ],
    "Engineering": [
        0.8615999999999948,
        0.8407999999999951,
        0.8631999999999953,
        0.8741999999999944,
        0.8735470941883723,
        0.8293999999999946,
        0.8409999999999944,
        0.8511999999999946,
        0.8457999999999946,
    ],
}

models = [
    "gemma-3-27b-it",
    "Qwen3-30B-A3B-Instruct-2507",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gpt-4o-mini",
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-4.1",
]

markers = ["o", "s", "v", "^", "<", ">", "D", "p", "*"]

# Convert to DataFrames
df_faithfulness = pd.DataFrame(faithfulness, index=models)
df_answer_relevancy = pd.DataFrame(answer_relevancy, index=models)
df_answer_correctness = pd.DataFrame(answer_correctness, index=models)


# Clean up (drop any non-numeric entries like "pro" or "")
df_faithfulness = df_faithfulness.apply(pd.to_numeric, errors="coerce")
df_answer_relevancy = df_answer_relevancy.apply(pd.to_numeric, errors="coerce")
df_answer_correctness = df_answer_correctness.apply(pd.to_numeric, errors="coerce")

# Melt to long format
df_faithfulness_long = (
    df_faithfulness.reset_index()
    .melt(id_vars="index", var_name="Task", value_name="Score")
    .rename(columns={"index": "Model"})
)

df_answer_relevancy_long = (
    df_answer_relevancy.reset_index()
    .melt(id_vars="index", var_name="Task", value_name="Score")
    .rename(columns={"index": "Model"})
)

df_answer_correctness_long = (
    df_answer_correctness.reset_index()
    .melt(id_vars="index", var_name="Task", value_name="Score")
    .rename(columns={"index": "Model"})
)

# Plotting side-by-side (3 panels instead of 2)
fig, axs = plt.subplots(1, 3, figsize=(10, 8), sharey=False)

# Faithfulness Plot
for i, model in enumerate(df_faithfulness_long["Model"].unique()):
    subset = df_faithfulness_long[df_faithfulness_long["Model"] == model]
    marker = markers[i % len(markers)]
    axs[1].plot(subset["Task"], subset["Score"], marker=marker, label=model)
axs[1].set_title("Faithfulness")
axs[1].set_ylim(0.865, 0.955)
axs[1].yaxis.set_major_locator(MultipleLocator(0.02))
axs[1].grid(True, linestyle="--", alpha=0.5)
axs[1].tick_params(axis="x", rotation=45)

# Answer Relevancy Plot
for i, model in enumerate(df_answer_relevancy_long["Model"].unique()):
    subset = df_answer_relevancy_long[df_answer_relevancy_long["Model"] == model]
    marker = markers[i % len(markers)]
    axs[2].plot(subset["Task"], subset["Score"], marker=marker, label=model)
axs[2].set_title("Answer Relevancy")
axs[2].set_ylim(0.90, 0.965)
axs[2].yaxis.set_major_locator(MultipleLocator(0.01))
axs[2].grid(True, linestyle="--", alpha=0.5)
axs[2].tick_params(axis="x", rotation=45)

# Answer Correctness Plot
for i, model in enumerate(df_answer_correctness_long["Model"].unique()):
    subset = df_answer_correctness_long[df_answer_correctness_long["Model"] == model]
    marker = markers[i % len(markers)]
    axs[0].plot(subset["Task"], subset["Score"], marker=marker, label=model)
axs[0].set_title("Answer Correctness")
axs[0].set_ylim(0.79, 0.88)
axs[0].set_ylabel("Mean Score")
axs[0].yaxis.set_major_locator(MultipleLocator(0.02))
axs[0].grid(True, linestyle="--", alpha=0.5)
axs[0].tick_params(axis="x", rotation=45)

# Shared legend outside the plots
fig.legend(
    df_faithfulness_long["Model"].unique(),
    loc="center right",
    fontsize="small",
)

plt.tight_layout(rect=(0.0, 0.0, 0.8, 1.0))
plt.savefig("outputs/model_eval_metrics.png")
plt.show()
