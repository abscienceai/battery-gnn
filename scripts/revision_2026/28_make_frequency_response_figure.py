#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]

RESULTS = (
    REPO_ROOT
    / "results/revision_2026/spectral_frequency/final_spectral_v1/"
      "sinusoidal_frequency_v1/analysis_v1/frequency_response_summary.csv"
)

OUTDIR = REPO_ROOT / "figures/revision_2026"

OUTDIR.mkdir(parents=True, exist_ok=True)

EXPECTED_FREQ = np.array([0.0625, 0.125, 0.25, 0.5, 0.75, 1.0])
EXPECTED_MODELS = {"MLP", "GCN"}
EXPECTED_METRICS = {"rmse", "mae"}

df = pd.read_csv(RESULTS)

# -----------------------------
# Strict audit / input barriers
# -----------------------------
required_columns = {
    "model",
    "domain",
    "metric",
    "omega_over_pi",
    "mean_relative_degradation",
    "sd_across_splits",
    "mean_perturbed_metric",
    "mean_clean_metric",
    "split_count",
}

missing = required_columns - set(df.columns)
if missing:
    raise RuntimeError(f"Missing required columns: {sorted(missing)}")

plot_df = df.loc[
    (df["domain"] == "BALANCED_ALL")
    & (df["model"].isin(EXPECTED_MODELS))
    & (df["metric"].isin(EXPECTED_METRICS))
].copy()

if plot_df.empty:
    raise RuntimeError("No BALANCED_ALL MLP/GCN RMSE/MAE rows found.")

for metric in sorted(EXPECTED_METRICS):
    for model in sorted(EXPECTED_MODELS):
        sub = (
            plot_df.loc[
                (plot_df["metric"] == metric)
                & (plot_df["model"] == model)
            ]
            .sort_values("omega_over_pi")
            .reset_index(drop=True)
        )

        got_freq = sub["omega_over_pi"].to_numpy(dtype=float)

        if len(sub) != 6:
            raise RuntimeError(
                f"{model}/{metric}: expected 6 frequencies, found {len(sub)}"
            )

        if not np.allclose(got_freq, EXPECTED_FREQ, rtol=0, atol=1e-12):
            raise RuntimeError(
                f"{model}/{metric}: unexpected frequency grid: {got_freq}"
            )

        if not np.all(sub["split_count"].to_numpy() == 5):
            raise RuntimeError(
                f"{model}/{metric}: split_count is not 5 for every point."
            )

        if sub[
            [
                "mean_relative_degradation",
                "sd_across_splits",
                "mean_perturbed_metric",
                "mean_clean_metric",
            ]
        ].isna().any().any():
            raise RuntimeError(f"{model}/{metric}: missing numeric values.")

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

# Panel (a): analytic response
ax = axes[0]

omega_over_pi_dense = np.linspace(0.0, 1.0, 501)
omega = np.pi * omega_over_pi_dense

# d = 2 causal aggregation stages:
# |H_d(w)| = |cos(w/2)|^d
H2 = np.abs(np.cos(omega / 2.0)) ** 2

ax.plot(
    omega_over_pi_dense,
    H2,
    linewidth=2.0,
)

ax.scatter(
    EXPECTED_FREQ,
    np.abs(np.cos(np.pi * EXPECTED_FREQ / 2.0)) ** 2,
    s=24,
    zorder=3,
)

ax.set_xlabel(r"Normalized frequency, $\omega/\pi$")
ax.set_ylabel(r"Operator magnitude, $|H_2(\omega)|$")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(-0.03, 1.03)
ax.set_title("(a) Analytic causal response")
ax.grid(alpha=0.25)

# Helper for empirical panels
def plot_metric(ax, metric, panel_title):
    styles = {
        "MLP": dict(marker="o", linestyle="--", linewidth=1.7, markersize=5),
        "GCN": dict(marker="s", linestyle="-", linewidth=1.9, markersize=5),
    }

    labels = {
        "MLP": "MLP",
        "GCN": "Causal GCN",
    }

    for model in ["MLP", "GCN"]:
        sub = (
            plot_df.loc[
                (plot_df["metric"] == metric)
                & (plot_df["model"] == model)
            ]
            .sort_values("omega_over_pi")
            .reset_index(drop=True)
        )

        x = sub["omega_over_pi"].to_numpy(dtype=float)
        y = 100.0 * sub["mean_relative_degradation"].to_numpy(dtype=float)
        yerr = 100.0 * sub["sd_across_splits"].to_numpy(dtype=float)

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            capsize=2.5,
            label=labels[model],
            **styles[model],
        )

    ax.axhline(0.0, linewidth=0.8, linestyle=":")
    ax.set_xlabel(r"Normalized frequency, $\omega/\pi$")
    ax.set_ylabel("Relative degradation (%)")
    # Use a readable common frequency axis while preserving the
    # actual experimental x-coordinates of all six data points.
    ax.set_xlim(0.0, 1.02)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    ax.set_title(panel_title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)


plot_metric(
    axes[1],
    "rmse",
    "(b) Empirical RMSE response",
)

plot_metric(
    axes[2],
    "mae",
    "(c) Empirical MAE response",
)

base = OUTDIR / "fig2_frequency_selective_response"

fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
plt.close(fig)

# -----------------------------
# Audit table + manifest
# -----------------------------
audit_cols = [
    "model",
    "domain",
    "metric",
    "omega_over_pi",
    "mean_relative_degradation",
    "sd_across_splits",
    "mean_perturbed_metric",
    "mean_clean_metric",
    "split_count",
]

audit_df = (
    plot_df[audit_cols]
    .sort_values(["metric", "model", "omega_over_pi"])
    .reset_index(drop=True)
)

audit_csv = OUTDIR / "fig2_frequency_selective_response_source_values.csv"
audit_df.to_csv(audit_csv, index=False)

manifest = {
    "figure": "fig2_frequency_selective_response",
    "source_file": str(RESULTS.relative_to(REPO_ROOT)),
    "source_domain": "BALANCED_ALL",
    "models": ["MLP", "GCN"],
    "metrics": ["rmse", "mae"],
    "frequency_grid_omega_over_pi": EXPECTED_FREQ.tolist(),
    "split_count_per_point": 5,
    "error_bar_definition": "sample SD across five split-level means",
    "perturbation_rms_standardized_units": 0.1,
    "analytic_operator": "|H_d(omega)| = |cos(omega/2)|^d",
    "deployed_aggregation_stages_d": 2,
    "outputs": [
        str(base.with_suffix(".pdf").relative_to(REPO_ROOT)),
        str(base.with_suffix(".png").relative_to(REPO_ROOT)),
        str(base.with_suffix(".svg").relative_to(REPO_ROOT)),
        str(audit_csv.relative_to(REPO_ROOT)),
    ],
    "important_interpretation": (
        "Mechanistic/descriptive frequency-selective response; "
        "do not present as multiplicity-corrected statistical superiority."
    ),
}

manifest_path = OUTDIR / "fig2_frequency_selective_response_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("OK")
print(f"Source:   {RESULTS}")
print(f"PDF:      {base.with_suffix('.pdf')}")
print(f"PNG:      {base.with_suffix('.png')}")
print(f"SVG:      {base.with_suffix('.svg')}")
print(f"Audit:    {audit_csv}")
print(f"Manifest: {manifest_path}")

print("\nBALANCED_ALL plotted values (% relative degradation):")
show = audit_df.copy()
show["mean_relative_degradation_pct"] = (
    100.0 * show["mean_relative_degradation"]
)
show["sd_across_splits_pct"] = (
    100.0 * show["sd_across_splits"]
)
print(
    show[
        [
            "metric",
            "model",
            "omega_over_pi",
            "mean_relative_degradation_pct",
            "sd_across_splits_pct",
        ]
    ].to_string(index=False)
)
