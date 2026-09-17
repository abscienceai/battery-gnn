#!/usr/bin/env python3

from pathlib import Path
import json
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = REPO_ROOT / "figures/revision_2026"
OUTDIR.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, title, body,
        face="#F7F7F7", edge="#4A4A4A",
        title_size=10.0, body_size=8.2,
        dashed=False, title_y=0.72, body_y=0.36):

    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.010,rounding_size=0.014",
        linewidth=1.1,
        edgecolor=edge,
        facecolor=face,
        linestyle="--" if dashed else "-",
        zorder=2,
    )
    ax.add_patch(patch)

    ax.text(
        x + w / 2,
        y + h * title_y,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        zorder=3,
    )

    ax.text(
        x + w / 2,
        y + h * body_y,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        linespacing=1.18,
        zorder=3,
    )

    return patch


def arrow(ax, x1, y1, x2, y2, lw=1.0):
    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=lw,
        color="#666666",
        shrinkA=2,
        shrinkB=2,
        zorder=1,
    )
    ax.add_patch(a)
    return a


plt.rcParams.update({
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(12.6, 7.9))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


# ================================================================
# Top pipeline
# ================================================================
box(
    ax, 0.035, 0.815, 0.19, 0.13,
    "Open battery datasets",
    "HUST: 77 cells\nCALCE: 13 cells\nMATR: 169 cells\nRWTH: 48 cells",
    face="#EEF4FA",
)

box(
    ax, 0.285, 0.815, 0.19, 0.13,
    "Primary SOH cohort",
    "HUST + CALCE + MATR\n300,553 observations\n259 complete cells",
    face="#EEF4FA",
)

box(
    ax, 0.535, 0.815, 0.19, 0.13,
    "Cycle representation",
    "Raw voltage/current traces\n20 target-independent descriptors\ntrain-only standardization",
    face="#F2F7EF",
)

box(
    ax, 0.785, 0.815, 0.18, 0.13,
    "Matched predictors",
    "MLP vs causal GCN\n20→64→64→1\n5,569 parameters each",
    face="#F2F7EF",
)

arrow(ax, 0.225, 0.880, 0.285, 0.880)
arrow(ax, 0.475, 0.880, 0.535, 0.880)
arrow(ax, 0.725, 0.880, 0.785, 0.880)


# ================================================================
# RWTH side note
# ================================================================
box(
    ax, 0.050, 0.710, 0.160, 0.065,
    "RWTH",
    "provenance / chemistry sensitivity\nnot pooled into primary SOH",
    face="#FAFAFA",
    title_size=8.6,
    body_size=7.0,
    dashed=True,
)

arrow(ax, 0.130, 0.815, 0.130, 0.775, lw=0.8)


# ================================================================
# Causal receptive field note — separated from contract box
# ================================================================
ax.text(
    0.875,
    0.785,
    r"Causal receptive field: $t-2 \rightarrow t-1 \rightarrow t$",
    ha="center",
    va="center",
    fontsize=7.6,
    fontweight="bold",
)

ax.text(
    0.875,
    0.765,
    "no future observations",
    ha="center",
    va="center",
    fontsize=7.1,
    style="italic",
)


# ================================================================
# Common protocol
# ================================================================
box(
    ax, 0.255, 0.665, 0.520, 0.090,
    "Common evaluation contract",
    "Complete-cell partitions • 5 split seeds • 3 initialization seeds per split\n"
    "validation-only model selection • no future / reverse / cross-cell edges",
    face="#FFF8E8",
    title_size=9.8,
    body_size=7.8,
    title_y=0.70,
    body_y=0.30,
)


# ================================================================
# Branch boxes
# ================================================================
w = 0.285
h = 0.145

top_y = 0.445
bottom_y = 0.245

centers = [0.1775, 0.5000, 0.8225]

box(
    ax, 0.035, top_y, w, h,
    "1. Clean SOH + temporal controls",
    "Raw MLP / causal GCN\n"
    "MA3 • LP3 • SG5 • Kalman • causal TCN",
    face="#F8F3FF",
)

box(
    ax, 0.3575, top_y, w, h,
    "2. Physical-unit sensor robustness",
    "Held-out raw V/I corruption\n"
    "feature recomputation → frozen models\n"
    "Gaussian • offset • drift • dropout",
    face="#F8F3FF",
)

box(
    ax, 0.680, top_y, w, h,
    "3. Frequency-mechanism diagnostics",
    "Sinusoidal • AR(1) • bandwidth • drift\n"
    "causal transfer response\n"
    "path-Laplacian Fourier basis = diagnostic only",
    face="#F8F3FF",
    body_size=7.8,
)

box(
    ax, 0.035, bottom_y, w, h,
    "4. dQ/dV leakage-control sensitivity",
    "PRIMARY20 vs FULL\n"
    "PREFIX25 / PREFIX50\n"
    "fixed training-derived causal cutoffs",
    face="#EEF8F6",
)

box(
    ax, 0.3575, bottom_y, w, h,
    "5. Prospective RUL",
    "Observed 80%-SOH endpoint\n"
    "K25 • K50 • K100 • FULL scope\n"
    "HUST threshold-censored",
    face="#EEF8F6",
)

box(
    ax, 0.680, bottom_y, w, h,
    "6. Cross-dataset transfer",
    "Source-only LODO\n"
    "ZERO → K1 → K3 → K5\n"
    "complete-cell target calibration",
    face="#EEF8F6",
)


# ================================================================
# Clean connector topology
# ================================================================

# Vertical stem from common contract to a horizontal distribution rail.
rail_y = 0.625
arrow(ax, 0.515, 0.665, 0.515, rail_y, lw=1.0)

# Top horizontal rail.
ax.plot(
    [centers[0], centers[2]],
    [rail_y, rail_y],
    color="#777777",
    linewidth=0.9,
    zorder=1,
)

# Three short vertical arrows into top-row boxes.
for cx in centers:
    arrow(ax, cx, rail_y, cx, top_y + h, lw=0.9)

# Direct vertical connectors from each top branch to its
# corresponding lower branch; no diagonal lines cross text.
for cx in centers:
    arrow(ax, cx, top_y, cx, bottom_y + h, lw=0.9)


# ================================================================
# Bottom interpretation
# ================================================================
interpretation = (
    "No universal architectural winner. Model preference depends on "
    "sensor-error spectrum, preprocessing, available causal history, "
    "prediction horizon, and target-domain calibration."
)

wrapped_interpretation = "\n".join(
    textwrap.wrap(interpretation, width=105)
)

box(
    ax, 0.105, 0.055, 0.790, 0.115,
    "Deployment-oriented interpretation",
    wrapped_interpretation,
    face="#FFF2F2",
    edge="#666666",
    title_size=10.0,
    body_size=8.2,
    title_y=0.72,
    body_y=0.32,
)

# Bottom-row boxes join a rail, then one arrow enters the
# interpretation box. This avoids diagonal crossings.
bottom_rail_y = 0.205

for cx in centers:
    ax.plot(
        [cx, cx],
        [bottom_y, bottom_rail_y],
        color="#777777",
        linewidth=0.8,
        zorder=1,
    )

ax.plot(
    [centers[0], centers[2]],
    [bottom_rail_y, bottom_rail_y],
    color="#777777",
    linewidth=0.8,
    zorder=1,
)

arrow(ax, 0.500, bottom_rail_y, 0.500, 0.170, lw=0.9)


# ================================================================
# Save
# ================================================================
base = OUTDIR / "fig1_benchmark_workflow"

fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
plt.close(fig)


manifest = {
    "figure": "fig1_benchmark_workflow",
    "figure_type": "protocol/workflow schematic",
    "primary_soh_cohort": {
        "datasets": ["HUST", "CALCE", "MATR"],
        "observations": 300553,
        "cells": 259,
        "dataset_counts": {
            "HUST": {"cells": 77, "observations": 146122},
            "CALCE": {"cells": 13, "observations": 14298},
            "MATR": {"cells": 169, "observations": 140133},
        },
    },
    "rwth": {
        "cells": 48,
        "primary_soh_role": False,
        "role": "provenance and chemistry-oriented sensitivity",
    },
    "primary_features": 20,
    "model_parameters_each": 5569,
    "architecture": "20->64->64->1",
    "causal_receptive_field": ["t", "t-1", "t-2"],
    "split_seeds": 5,
    "initializations_per_split": 3,
    "evaluation_blocks": [
        "clean SOH and temporal controls",
        "physical-unit raw-sensor robustness",
        "frequency-mechanism diagnostics",
        "dQ/dV leakage-control sensitivity",
        "prospective RUL",
        "LODO and complete-cell few-shot transfer",
    ],
    "important_boundary": (
        "The undirected path-Laplacian basis is diagnostic only "
        "and is not the deployed causal aggregation operator."
    ),
}

manifest_path = OUTDIR / "fig1_benchmark_workflow_manifest.json"
manifest_path.write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("OK")
print(f"PDF:      {base.with_suffix('.pdf')}")
print(f"PNG:      {base.with_suffix('.png')}")
print(f"SVG:      {base.with_suffix('.svg')}")
print(f"Manifest: {manifest_path}")

print("\nScientific audit facts unchanged:")
print("Primary cohort: HUST + CALCE + MATR = 300,553 observations / 259 cells")
print("RWTH: 48 cells, outside primary dense SOH cohort")
print("Primary features: 20")
print("MLP / causal GCN parameters: 5,569 each")
print("Causal receptive field: {t, t-1, t-2}")
print("Replication: 5 split seeds x 3 initialization seeds")
