#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    REPO_ROOT
    / "results/revision_2026/lodo_fewshot/analysis_v1/"
      "lodo_fewshot_descriptive_summary.csv"
)

OUTDIR = REPO_ROOT / "figures/revision_2026"
OUTDIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["HUST", "CALCE", "MATR"]
CONDITIONS = ["ZERO_SHOT", "K1", "K3", "K5"]
MODELS = ["MLP", "GCN"]

# Rounded manuscript values used only as an audit barrier.
EXPECTED = {
    ("HUST", "MLP"): [0.37833, 0.20618, 0.13067, 0.12376],
    ("HUST", "GCN"): [0.42140, 0.23047, 0.13648, 0.13872],
    ("CALCE", "MLP"): [2.31001, 0.18525, 0.17121, 0.16723],
    ("CALCE", "GCN"): [2.40041, 0.20373, 0.18391, 0.18435],
    ("MATR", "MLP"): [2.17447, 0.50618, 0.37846, 0.36759],
    ("MATR", "GCN"): [2.63302, 0.63884, 0.51190, 0.50027],
}

df = pd.read_csv(SOURCE)

required = {
    "held_out",
    "condition",
    "model",
    "scope",
    "metric",
    "mean_over_15_runs",
    "sd_over_5_split_means",
    "n_runs",
    "n_split_seeds",
    "initializations_per_split",
}
missing = required - set(df.columns)
if missing:
    raise RuntimeError(f"Missing required columns: {sorted(missing)}")

plot_df = df.loc[
    (df["held_out"].isin(DATASETS))
    & (df["condition"].isin(CONDITIONS))
    & (df["model"].isin(MODELS))
    & (df["scope"] == "CELL_BALANCED")
    & (df["metric"] == "RMSE")
].copy()

# -----------------------------
# Strict audit barriers
# -----------------------------
if len(plot_df) != 24:
    raise RuntimeError(
        f"Expected 24 rows (3 datasets x 4 conditions x 2 models), "
        f"found {len(plot_df)}"
    )

if plot_df.duplicated(
    ["held_out", "condition", "model", "scope", "metric"]
).any():
    raise RuntimeError("Duplicate summary rows detected.")

if not np.all(plot_df["n_runs"].to_numpy() == 15):
    raise RuntimeError("Expected n_runs=15 for every plotted condition.")

if not np.all(plot_df["n_split_seeds"].to_numpy() == 5):
    raise RuntimeError("Expected n_split_seeds=5 for every plotted condition.")

if not np.all(
    plot_df["initializations_per_split"].to_numpy() == 3
):
    raise RuntimeError(
        "Expected initializations_per_split=3 for every plotted condition."
    )

if plot_df[
    ["mean_over_15_runs", "sd_over_5_split_means"]
].isna().any().any():
    raise RuntimeError("Missing RMSE mean or SD value.")

for dataset in DATASETS:
    for model in MODELS:
        sub = (
            plot_df.loc[
                (plot_df["held_out"] == dataset)
                & (plot_df["model"] == model)
            ]
            .set_index("condition")
            .loc[CONDITIONS]
        )

        got = sub["mean_over_15_runs"].to_numpy(dtype=float)
        exp = np.array(EXPECTED[(dataset, model)], dtype=float)

        if not np.allclose(got, exp, rtol=0, atol=5e-5):
            raise RuntimeError(
                f"{dataset}/{model}: manuscript-value audit failed.\n"
                f"Expected: {exp}\n"
                f"Observed: {got}"
            )

# -----------------------------
# Figure
# -----------------------------
plt.rcParams.update(
    {
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(10.8, 3.35),
    constrained_layout=True,
)

x = np.arange(len(CONDITIONS))
xlabels = ["Zero-shot", "K1", "K3", "K5"]

styles = {
    "MLP": dict(
        marker="o",
        linestyle="--",
        linewidth=1.8,
        markersize=5.5,
        capsize=3,
    ),
    "GCN": dict(
        marker="s",
        linestyle="-",
        linewidth=1.9,
        markersize=5.5,
        capsize=3,
    ),
}

labels = {
    "MLP": "MLP",
    "GCN": "Causal GCN",
}

panel_letters = ["(a)", "(b)", "(c)"]

for ax, dataset, letter in zip(axes, DATASETS, panel_letters):

    for model in MODELS:
        sub = (
            plot_df.loc[
                (plot_df["held_out"] == dataset)
                & (plot_df["model"] == model)
            ]
            .set_index("condition")
            .loc[CONDITIONS]
        )

        y = sub["mean_over_15_runs"].to_numpy(dtype=float)
        yerr = sub["sd_over_5_split_means"].to_numpy(dtype=float)

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=labels[model],
            **styles[model],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Target-domain calibration")
    ax.set_ylabel("Cell-balanced RMSE")
    ax.set_title(f"{letter} {dataset} held out")
    ax.grid(alpha=0.25)

    # Dataset-specific y scale is intentional because absolute
    # zero-shot errors differ substantially across datasets.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(bottom=0, top=ymax)

    ax.legend(frameon=False)

base = OUTDIR / "fig3_lodo_fewshot_transfer"

fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
plt.close(fig)

# -----------------------------
# Audit export
# -----------------------------
audit = (
    plot_df[
        [
            "held_out",
            "condition",
            "model",
            "scope",
            "metric",
            "mean_over_15_runs",
            "sd_over_5_split_means",
            "n_runs",
            "n_split_seeds",
            "initializations_per_split",
        ]
    ]
    .assign(
        condition_order=lambda x:
            x["condition"].map(
                {c: i for i, c in enumerate(CONDITIONS)}
            ),
        model_order=lambda x:
            x["model"].map(
                {m: i for i, m in enumerate(MODELS)}
            ),
        dataset_order=lambda x:
            x["held_out"].map(
                {d: i for i, d in enumerate(DATASETS)}
            ),
    )
    .sort_values(
        ["dataset_order", "model_order", "condition_order"]
    )
    .drop(
        columns=[
            "condition_order",
            "model_order",
            "dataset_order",
        ]
    )
    .reset_index(drop=True)
)

audit_csv = OUTDIR / "fig3_lodo_fewshot_transfer_source_values.csv"
audit.to_csv(audit_csv, index=False)

manifest = {
    "figure": "fig3_lodo_fewshot_transfer",
    "source_file": str(SOURCE.relative_to(REPO_ROOT)),
    "scope": "CELL_BALANCED",
    "metric": "RMSE",
    "held_out_datasets": DATASETS,
    "conditions": CONDITIONS,
    "models": MODELS,
    "summary_definition": (
        "mean over 15 runs; three initialization runs per each "
        "of five complete-cell split seeds"
    ),
    "error_bar_definition": (
        "sample standard deviation across five split-level means"
    ),
    "important_interpretation": (
        "Few-shot calibration reduces RMSE relative to zero-shot "
        "across all 90 split-model-fold adaptation comparisons, "
        "but substantial residual domain mismatch remains. "
        "Do not describe adaptation as restoring generalization."
    ),
    "outputs": [
        str(base.with_suffix(".pdf").relative_to(REPO_ROOT)),
        str(base.with_suffix(".png").relative_to(REPO_ROOT)),
        str(base.with_suffix(".svg").relative_to(REPO_ROOT)),
        str(audit_csv.relative_to(REPO_ROOT)),
    ],
}

manifest_path = OUTDIR / "fig3_lodo_fewshot_transfer_manifest.json"
manifest_path.write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("OK")
print(f"Source:   {SOURCE}")
print(f"PDF:      {base.with_suffix('.pdf')}")
print(f"PNG:      {base.with_suffix('.png')}")
print(f"SVG:      {base.with_suffix('.svg')}")
print(f"Audit:    {audit_csv}")
print(f"Manifest: {manifest_path}")

print("\nPlotted CELL_BALANCED RMSE values:")
print(
    audit[
        [
            "held_out",
            "condition",
            "model",
            "mean_over_15_runs",
            "sd_over_5_split_means",
        ]
    ].to_string(index=False)
)
