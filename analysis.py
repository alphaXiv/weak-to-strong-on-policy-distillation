# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "altair>=5.4,<6",
#   "marimo>=0.14,<1",
#   "pandas>=2.2,<3",
# ]
# ///
"""Self-contained analysis notebook for the controlled W2S-OPD reproduction."""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    from io import StringIO
    from textwrap import dedent

    import altair as alt
    import marimo as mo
    import pandas as pd

    return StringIO, alt, dedent, math, mo, pd


@app.cell
def _(mo):
    mo.md(r"""
    # Weak-to-Strong OPD: controlled reproduction

    This notebook is **self-contained**: the run-level terminal metrics used below are
    embedded in the source. It summarizes a Qwen3-8B GSM8K scale-contrast reproduction
    of [Yu et al. (2026)](https://arxiv.org/abs/2607.26246).

    **Headline:** after learning-rate tuning, W2S-OPD scored 434/512—matching zero-shot
    and beating its matched direct-OPD control by 11 examples. It did not produce an
    absolute gain over the untrained student in this implementation.
    """)
    return


@app.cell
def _(StringIO, dedent, pd):
    embedded_csv = """phase,label,method,seed,eval_examples,correct,train_steps,rollouts_per_step,learning_rate,alpha,top_k
    main,zero-shot,Zero-shot,20260728,512,434,0,0,,,
    main,direct-best,Direct OPD,20260728,512,430,25,8,0.000002,,32
    main,w2s-best,W2S-OPD,20260728,512,434,25,8,0.000004,1.0,32
    horizon,direct-12x64,Direct OPD,20260728,512,427,12,8,0.000002,,32
    horizon,w2s-12x64,W2S-OPD,20260728,512,428,12,8,0.000002,1.0,32
    horizon,direct-18x64,Direct OPD,20260728,512,423,18,8,0.000002,,32
    horizon,w2s-18x64,W2S-OPD,20260728,512,426,18,8,0.000002,1.0,32
    horizon,direct-25x64,Direct OPD,20260728,512,430,25,8,0.000002,,32
    horizon,w2s-25x64,W2S-OPD,20260728,512,429,25,8,0.000002,1.0,32
    horizon,direct-32x64,Direct OPD,20260728,512,426,32,8,0.000002,,32
    horizon,w2s-32x64,W2S-OPD,20260728,512,422,32,8,0.000002,1.0,32
    horizon,direct-100x64,Direct OPD,20260728,512,418,100,8,0.000002,,32
    horizon,w2s-100x64,W2S-OPD,20260728,512,424,100,8,0.000002,1.0,32
    lr,direct-lr-1e-6,Direct OPD,20260728,512,423,25,8,0.000001,,32
    lr,w2s-lr-1e-6,W2S-OPD,20260728,512,425,25,8,0.000001,1.0,32
    lr,direct-lr-2e-6,Direct OPD,20260728,512,430,25,8,0.000002,,32
    lr,w2s-lr-2e-6,W2S-OPD,20260728,512,429,25,8,0.000002,1.0,32
    lr,direct-lr-4e-6,Direct OPD,20260728,512,423,25,8,0.000004,,32
    lr,w2s-lr-4e-6,W2S-OPD,20260728,512,434,25,8,0.000004,1.0,32
    seed,baseline-20260728,Zero-shot,20260728,128,108,0,0,,,
    seed,baseline-20260729,Zero-shot,20260729,128,108,0,0,,,
    seed,baseline-20260730,Zero-shot,20260730,128,104,0,0,,,
    seed,direct-20260728,Direct OPD,20260728,128,108,6,1,0.000002,,32
    seed,direct-20260729,Direct OPD,20260729,128,109,6,1,0.000002,,32
    seed,direct-20260730,Direct OPD,20260730,128,108,6,1,0.000002,,32
    seed,w2s-20260728,W2S-OPD,20260728,128,109,6,1,0.000002,1.0,32
    seed,w2s-20260729,W2S-OPD,20260729,128,107,6,1,0.000002,1.0,32
    seed,w2s-20260730,W2S-OPD,20260730,128,105,6,1,0.000002,1.0,32
    """
    results = pd.read_csv(StringIO(dedent(embedded_csv)))
    results["accuracy"] = results["correct"] / results["eval_examples"]
    return (results,)


@app.cell
def _(alt, results):
    primary = results[results["phase"] == "main"].copy()
    primary_chart = (
        alt.Chart(primary)
        .mark_bar(size=48)
        .encode(
            x=alt.X("method:N", title=None, sort=["Zero-shot", "Direct OPD", "W2S-OPD"]),
            y=alt.Y("accuracy:Q", title="GSM8K accuracy", scale=alt.Scale(domain=[0.80, 0.87])),
            color=alt.Color("method:N", legend=None),
            tooltip=["method", "correct", "eval_examples", alt.Tooltip("accuracy:Q", format=".2%")],
        )
        .properties(title="Best observed 512-example result by method", height=320)
    )
    primary_chart
    return


@app.cell
def _(alt, pd, results):
    horizon = results[results["phase"] == "horizon"].copy()
    baseline = pd.DataFrame({"accuracy": [434 / 512]})
    horizon_chart = (
        alt.Chart(horizon)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("train_steps:Q", title="Optimizer steps (effective batch 64)"),
            y=alt.Y("accuracy:Q", title="GSM8K accuracy", scale=alt.Scale(domain=[0.80, 0.86])),
            color=alt.Color("method:N", title=None),
            tooltip=["method", "train_steps", "correct", alt.Tooltip("accuracy:Q", format=".2%")],
        )
        + alt.Chart(baseline).mark_rule(strokeDash=[6, 4], color="black").encode(y="accuracy:Q")
    ).properties(title="Training longer did not recover the zero-shot baseline", height=330)
    horizon_chart
    return


@app.cell
def _(alt, results):
    lr = results[results["phase"] == "lr"].copy()
    lr["lr_label"] = lr["learning_rate"].map(lambda value: f"{value:.0e}")
    lr_chart = (
        alt.Chart(lr)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("lr_label:O", title="Learning rate", sort=["1e-06", "2e-06", "4e-06"]),
            y=alt.Y("accuracy:Q", title="GSM8K accuracy", scale=alt.Scale(domain=[0.81, 0.86])),
            color=alt.Color("method:N", title=None),
            tooltip=["method", "learning_rate", "correct", alt.Tooltip("accuracy:Q", format=".2%")],
        )
        .properties(title="25×64 learning-rate sweep", height=320)
    )
    lr_chart
    return


@app.cell
def _(pd, results):
    seed = results[results["phase"] == "seed"].copy()
    paired = seed.pivot(index="seed", columns="method", values="correct")
    paired["Direct − zero-shot"] = paired["Direct OPD"] - paired["Zero-shot"]
    paired["W2S − zero-shot"] = paired["W2S-OPD"] - paired["Zero-shot"]
    paired_summary = pd.DataFrame(
        {
            "mean correct / 128": paired[["Zero-shot", "Direct OPD", "W2S-OPD"]].mean(),
            "mean paired delta": [0.0, paired["Direct − zero-shot"].mean(), paired["W2S − zero-shot"].mean()],
        }
    )
    paired_summary
    return


@app.cell
def _(math, mo, results):
    def wilson(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
        p = correct / total
        denominator = 1 + z * z / total
        center = (p + z * z / (2 * total)) / denominator
        margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
        return center - margin, center + margin

    rows = []
    for row in results[results["phase"] == "main"].itertuples():
        lo, hi = wilson(row.correct, row.eval_examples)
        rows.append(
            {
                "method": row.method,
                "score": f"{row.correct}/{row.eval_examples}",
                "accuracy": f"{row.accuracy:.2%}",
                "95% Wilson interval": f"{lo:.2%}–{hi:.2%}",
            }
        )
    mo.ui.table(rows, selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation

    1. The 512-example differences are small relative to binomial uncertainty, and the
       three-seed 128-example check also changes sign across seeds.
    2. At LR 4e-6, W2S-OPD beats matched direct OPD by 11/512 and exactly matches
       zero-shot. This interaction is stronger than the default-LR horizon effect.
    3. No trained setting exceeds the same zero-shot checkpoint. The best W2S run
       matches it; the best direct run remains 4/512 below it.
    4. This implementation averages eight independently optimized LoRA deltas. It is
       not equivalent to the paper's full distributed optimizer and is a plausible source
       of the reproduction gap.

    These observations support a **partial, negative reproduction outcome**, not a
    falsification of W2S-OPD. See `REPORT.md` for the full protocol and threat model.
    """)
    return


if __name__ == "__main__":
    app.run()
