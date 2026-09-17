from pathlib import Path
import hashlib
import json
import os
import re
import subprocess

import numpy as np
import pandas as pd


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

THIS = (
    REPO
    / "scripts/revision_2026/75_analyze_correlation_bandwidth_drift.py"
)

ROOT = (
    BASE
    / "results/revision_2026/spectral_frequency/final_spectral_v1"
)

PROTOCOL = ROOT / "protocol.json"

EVAL = ROOT / "correlation_bandwidth_drift_v1"
RUNS = EVAL / "runs"

EXEC = (
    EVAL
    / "correlation_bandwidth_drift_execution_manifest.json"
)

PERTURBATION_DIAGNOSTICS = (
    EVAL
    / "perturbation_spectral_diagnostics.csv"
)

FINAL_V3 = (
    BASE
    / "results/revision_2026/feature_matched_soh/final_v3"
)

OUT = EVAL / "analysis_v1"

CLEAN_CHECK = OUT / "clean_final_v3_reference_check.csv"

AR1_SPLIT = OUT / "ar1_response_by_split.csv"
AR1_SUMMARY = OUT / "ar1_response_summary.csv"

BANDWIDTH_SPLIT = OUT / "bandwidth_response_by_split.csv"
BANDWIDTH_SUMMARY = OUT / "bandwidth_response_summary.csv"

DRIFT_SPLIT = OUT / "controlled_drift_response_by_split.csv"
DRIFT_SUMMARY = OUT / "controlled_drift_response_summary.csv"

CONTRAST = OUT / "model_contrast_summary.csv"

SPECTRAL = OUT / "perturbation_spectral_summary.csv"

MANIFEST = OUT / "correlation_bandwidth_drift_analysis_manifest.json"


EXPECTED_PROTOCOL_SHA = (
    "a8d3b65b87980470f326717f0cb50e3d4fb4d2e0429c93171eb0d21d6c713e45"
)

EXPECTED_EXEC_SHA = (
    "f43e69afae5e6d6a9a59edab0ab530990b8dab1dc790c3de3d775ba522835317"
)

EXPECTED_FINAL_V3_EXEC_SHA = (
    "5a8bfbe0f11a6158f2749a9da4f18f1e62adf8c839976c1bf0af9c9f6c07440d"
)

CLEAN_TOL = 1e-7

SPLITS = [0, 1, 2, 3, 4]

MODELS = ["MLP", "GCN"]

DOMAINS = [
    "BALANCED_ALL",
    "HUST",
    "CALCE",
    "MATR",
]

METRICS = [
    "rmse",
    "mae",
]

AR1_RHO = [
    -0.90,
    -0.50,
    0.00,
    0.50,
    0.90,
]

FILTER_ORDER = [
    "white",
    "lowpass_ma3",
    "lowpass_ma9",
    "bandpass_ma3_minus_ma9",
    "highpass_first_difference",
]

DRIFT_LEVELS = [
    0.10,
    0.25,
    0.50,
]


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def atomic_csv(df, path):
    tmp = Path(str(path) + ".tmp")

    df.to_csv(
        tmp,
        index=False,
        float_format="%.17g",
    )

    os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = Path(str(path) + ".tmp")

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(tmp, path)


def norm_model_name(x):
    s = str(x).strip().upper()

    if "GCN" in s:
        return "GCN"

    if "MLP" in s:
        return "MLP"

    raise AssertionError(
        ("unknown model", x)
    )


def make_setting_columns(df):
    x = df.copy()

    x["setting"] = ""

    x["setting_order"] = -1

    ar1 = (
        x["scenario"]
        == "ar1_correlation"
    )

    bandwidth = (
        x["scenario"]
        == "bandwidth"
    )

    drift = (
        x["scenario"]
        == "controlled_drift"
    )


    x.loc[
        ar1,
        "setting",
    ] = x.loc[
        ar1,
        "rho",
    ].map(
        lambda v:
            f"rho={float(v):+.2f}"
    )

    rho_order = {
        float(v): i
        for i, v in enumerate(
            AR1_RHO
        )
    }

    x.loc[
        ar1,
        "setting_order",
    ] = x.loc[
        ar1,
        "rho",
    ].astype(float).map(
        rho_order
    )


    filter_order = {
        name: i
        for i, name in enumerate(
            FILTER_ORDER
        )
    }

    x.loc[
        bandwidth,
        "setting",
    ] = x.loc[
        bandwidth,
        "filter_name",
    ].astype(str)

    x.loc[
        bandwidth,
        "setting_order",
    ] = x.loc[
        bandwidth,
        "filter_name",
    ].astype(str).map(
        filter_order
    )


    x.loc[
        drift,
        "setting",
    ] = x.loc[
        drift,
        "terminal_shift_std_units",
    ].map(
        lambda v:
            f"terminal={float(v):.2f}"
    )

    drift_order = {
        float(v): i
        for i, v in enumerate(
            DRIFT_LEVELS
        )
    }

    x.loc[
        drift,
        "setting_order",
    ] = x.loc[
        drift,
        "terminal_shift_std_units",
    ].astype(float).map(
        drift_order
    )


    assert (
        x[
            "setting"
        ] != ""
    ).all()

    assert (
        x[
            "setting_order"
        ] >= 0
    ).all()

    return x


def build_split_response(
    paired,
    scenario,
    parameter_column,
):
    g = paired[
        paired[
            "scenario"
        ] == scenario
    ].copy()

    result = (
        g
        .groupby(
            [
                "split_seed",
                "model",
                "domain",
                "metric",
                parameter_column,
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            relative_degradation=(
                "relative_degradation",
                "mean",
            ),

            nested_sd=(
                "relative_degradation",
                "std",
            ),

            perturbed_metric_mean=(
                "perturbed_value",
                "mean",
            ),

            clean_metric_mean=(
                "clean_value",
                "mean",
            ),

            nested_count=(
                "relative_degradation",
                "size",
            ),
        )
    )

    assert set(
        result[
            "nested_count"
        ]
    ) == {15}

    return result


def summarize_split_response(
    split_table,
    parameter_column,
):
    result = (
        split_table
        .groupby(
            [
                "model",
                "domain",
                "metric",
                parameter_column,
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            mean_relative_degradation=(
                "relative_degradation",
                "mean",
            ),

            sd_across_splits=(
                "relative_degradation",
                "std",
            ),

            mean_perturbed_metric=(
                "perturbed_metric_mean",
                "mean",
            ),

            mean_clean_metric=(
                "clean_metric_mean",
                "mean",
            ),

            split_count=(
                "split_seed",
                "nunique",
            ),
        )
    )

    assert set(
        result[
            "split_count"
        ]
    ) == {5}

    return result


# =============================================================================
# PRE-VALUE-READ PROVENANCE GUARDS
# =============================================================================

assert sha(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

assert sha(
    EXEC
) == EXPECTED_EXEC_SHA


protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)

execution = json.loads(
    EXEC.read_text(
        encoding="utf-8"
    )
)


assert protocol[
    "status"
] == "FROZEN_BEFORE_SPECTRAL_RUNS"

assert protocol[
    "ar1_correlation_sweep"
][
    "rho"
] == AR1_RHO

assert set(
    protocol[
        "bandwidth_sweep"
    ][
        "causal_FIR_filters"
    ]
) == set(
    FILTER_ORDER
)

assert protocol[
    "controlled_drift_sweep"
][
    "terminal_shift_standardized_units"
] == DRIFT_LEVELS


assert execution[
    "contract"
] == "final_correlation_bandwidth_drift_execution_v1"

assert execution[
    "status"
] == "COMPLETE_BEFORE_VALUE_LEVEL_ANALYSIS"

assert execution[
    "completed_runs"
] == 30

assert execution[
    "expected_total_metric_rows"
] == 7920

assert execution[
    "analysis_performed"
] is False

assert execution[
    "metric_values_surfaced_during_execution"
] is False

assert execution[
    "perturbation_diagnostic_values_surfaced_during_execution"
] is False

assert execution[
    "final_v3_execution_manifest_sha256"
] == EXPECTED_FINAL_V3_EXEC_SHA


metric_files = sorted(
    RUNS.glob(
        "*_cbd_metrics.csv"
    )
)

assert len(
    metric_files
) == 30


final_v3_files = sorted(
    FINAL_V3.glob(
        "split*_train*/metrics.csv"
    )
)

assert len(
    final_v3_files
) == 15


# Analysis code itself must already be committed and unchanged.
rel = str(
    THIS.relative_to(
        REPO
    )
)

assert subprocess.run(
    [
        "git",
        "ls-files",
        "--error-unmatch",
        rel,
    ],
    cwd=REPO,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
).returncode == 0

assert subprocess.run(
    [
        "git",
        "diff",
        "--quiet",
        "HEAD",
        "--",
        rel,
    ],
    cwd=REPO,
).returncode == 0


script_commit = subprocess.check_output(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
    text=True,
).strip()

script_sha = sha(
    THIS
)


# =============================================================================
# FIRST CBD MODEL-PERFORMANCE VALUE READ OCCURS BELOW THIS LINE
# =============================================================================

raw_frames = []


for p in metric_files:

    df = pd.read_csv(
        p,
        low_memory=False,
    )

    assert len(
        df
    ) == 264

    raw_frames.append(
        df
    )


raw = pd.concat(
    raw_frames,
    ignore_index=True,
)

assert len(
    raw
) == 7920


for metric in METRICS:

    raw[
        metric
    ] = pd.to_numeric(
        raw[
            metric
        ],
        errors="raise",
    )

    assert np.isfinite(
        raw[
            metric
        ]
    ).all()


# =============================================================================
# Independent clean final-v3 barrier
# =============================================================================

clean_new = raw[
    raw[
        "scenario"
    ] == "clean"
][
    [
        "split_seed",
        "train_seed",
        "model",
        "domain",
        "n_rows",
        "rmse",
        "mae",
    ]
].copy()

assert len(
    clean_new
) == 120


old_frames = []


for p in final_v3_files:

    m = re.search(
        r"split(\d+)_train(\d+)",
        str(p),
    )

    assert m

    split_seed = int(
        m.group(1)
    )

    train_seed = int(
        m.group(2)
    )

    df = pd.read_csv(
        p,
        low_memory=False,
    )

    df = df[
        df[
            "split"
        ].astype(str).str.lower()
        == "test"
    ].copy()

    assert len(
        df
    ) == 8


    old_frames.append(
        pd.DataFrame({
            "split_seed":
                split_seed,

            "train_seed":
                train_seed,

            "model":
                df[
                    "model"
                ].map(
                    norm_model_name
                ),

            "domain":
                df[
                    "dataset"
                ].astype(str),

            "n_old":
                pd.to_numeric(
                    df[
                        "n"
                    ],
                    errors="raise",
                ).astype(int),

            "rmse_old":
                pd.to_numeric(
                    df[
                        "RMSE"
                    ],
                    errors="raise",
                ),

            "mae_old":
                pd.to_numeric(
                    df[
                        "MAE"
                    ],
                    errors="raise",
                ),
        })
    )


clean_old = pd.concat(
    old_frames,
    ignore_index=True,
)

assert len(
    clean_old
) == 120


keys = [
    "split_seed",
    "train_seed",
    "model",
    "domain",
]


clean_check = clean_new.merge(
    clean_old,
    on=keys,
    how="inner",
    validate="one_to_one",
)

assert len(
    clean_check
) == 120


clean_check[
    "n_match"
] = (
    clean_check[
        "n_rows"
    ].astype(int)
    == clean_check[
        "n_old"
    ].astype(int)
)

clean_check[
    "rmse_abs_difference"
] = np.abs(
    clean_check[
        "rmse"
    ]
    - clean_check[
        "rmse_old"
    ]
)

clean_check[
    "mae_abs_difference"
] = np.abs(
    clean_check[
        "mae"
    ]
    - clean_check[
        "mae_old"
    ]
)

clean_check[
    "pass"
] = (
    clean_check[
        "n_match"
    ]
    & (
        clean_check[
            "rmse_abs_difference"
        ] <= CLEAN_TOL
    )
    & (
        clean_check[
            "mae_abs_difference"
        ] <= CLEAN_TOL
    )
)


assert clean_check[
    "pass"
].all()


max_clean_rmse = float(
    clean_check[
        "rmse_abs_difference"
    ].max()
)

max_clean_mae = float(
    clean_check[
        "mae_abs_difference"
    ].max()
)


# =============================================================================
# Relative degradation paired to same frozen clean run
# =============================================================================

clean_long = clean_new.melt(
    id_vars=[
        "split_seed",
        "train_seed",
        "model",
        "domain",
    ],
    value_vars=METRICS,
    var_name="metric",
    value_name="clean_value",
)


perturbed = raw[
    raw[
        "scenario"
    ].isin(
        [
            "ar1_correlation",
            "bandwidth",
            "controlled_drift",
        ]
    )
].copy()

assert len(
    perturbed
) == 7800


perturbed_long = perturbed.melt(
    id_vars=[
        "split_seed",
        "train_seed",
        "model",
        "domain",
        "scenario",
        "rho",
        "filter_name",
        "terminal_shift_std_units",
        "perturbation_seed",
    ],
    value_vars=METRICS,
    var_name="metric",
    value_name="perturbed_value",
)


paired = perturbed_long.merge(
    clean_long,
    on=[
        "split_seed",
        "train_seed",
        "model",
        "domain",
        "metric",
    ],
    how="left",
    validate="many_to_one",
)


assert len(
    paired
) == 15600

assert (
    paired[
        "clean_value"
    ] > 0
).all()


paired[
    "relative_degradation"
] = (
    paired[
        "perturbed_value"
    ]
    - paired[
        "clean_value"
    ]
) / paired[
    "clean_value"
]


assert np.isfinite(
    paired[
        "relative_degradation"
    ]
).all()


# =============================================================================
# Scenario-specific split-level summaries
# =============================================================================

ar1_split = build_split_response(
    paired,
    "ar1_correlation",
    "rho",
)

bandwidth_split = build_split_response(
    paired,
    "bandwidth",
    "filter_name",
)

drift_split = build_split_response(
    paired,
    "controlled_drift",
    "terminal_shift_std_units",
)


assert len(
    ar1_split
) == 400

assert len(
    bandwidth_split
) == 400

assert len(
    drift_split
) == 240


ar1_summary = summarize_split_response(
    ar1_split,
    "rho",
)

bandwidth_summary = summarize_split_response(
    bandwidth_split,
    "filter_name",
)

drift_summary = summarize_split_response(
    drift_split,
    "terminal_shift_std_units",
)


assert len(
    ar1_summary
) == 80

assert len(
    bandwidth_summary
) == 80

assert len(
    drift_summary
) == 48


# =============================================================================
# Model contrast across the five split units — descriptive only
# =============================================================================

contrast_parts = []


def contrast_table(
    split_table,
    scenario,
    parameter_column,
):

    pivot = split_table.pivot(
        index=[
            "split_seed",
            "domain",
            "metric",
            parameter_column,
        ],
        columns="model",
        values="relative_degradation",
    ).reset_index()


    assert set(
        [
            "MLP",
            "GCN",
        ]
    ).issubset(
        pivot.columns
    )


    pivot[
        "delta_mlp_minus_gcn"
    ] = (
        pivot[
            "MLP"
        ]
        - pivot[
            "GCN"
        ]
    )


    result = (
        pivot
        .groupby(
            [
                "domain",
                "metric",
                parameter_column,
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            mlp_relative_degradation_mean=(
                "MLP",
                "mean",
            ),

            gcn_relative_degradation_mean=(
                "GCN",
                "mean",
            ),

            mean_delta_mlp_minus_gcn=(
                "delta_mlp_minus_gcn",
                "mean",
            ),

            median_delta_mlp_minus_gcn=(
                "delta_mlp_minus_gcn",
                "median",
            ),

            sd_delta_across_splits=(
                "delta_mlp_minus_gcn",
                "std",
            ),

            gcn_favored_splits=(
                "delta_mlp_minus_gcn",
                lambda x:
                    int(
                        np.sum(
                            np.asarray(x) > 0
                        )
                    ),
            ),

            mlp_favored_splits=(
                "delta_mlp_minus_gcn",
                lambda x:
                    int(
                        np.sum(
                            np.asarray(x) < 0
                        )
                    ),
            ),

            ties=(
                "delta_mlp_minus_gcn",
                lambda x:
                    int(
                        np.sum(
                            np.asarray(x) == 0
                        )
                    ),
            ),
        )
    )


    result[
        "scenario"
    ] = scenario


    if scenario == "ar1_correlation":

        result[
            "setting"
        ] = result[
            parameter_column
        ].map(
            lambda v:
                f"rho={float(v):+.2f}"
        )

        order = {
            float(v): i
            for i, v in enumerate(
                AR1_RHO
            )
        }

        result[
            "setting_order"
        ] = result[
            parameter_column
        ].astype(float).map(
            order
        )


    elif scenario == "bandwidth":

        result[
            "setting"
        ] = result[
            parameter_column
        ].astype(str)

        order = {
            v: i
            for i, v in enumerate(
                FILTER_ORDER
            )
        }

        result[
            "setting_order"
        ] = result[
            parameter_column
        ].astype(str).map(
            order
        )


    else:

        result[
            "setting"
        ] = result[
            parameter_column
        ].map(
            lambda v:
                f"terminal={float(v):.2f}"
        )

        order = {
            float(v): i
            for i, v in enumerate(
                DRIFT_LEVELS
            )
        }

        result[
            "setting_order"
        ] = result[
            parameter_column
        ].astype(float).map(
            order
        )


    return result[
        [
            "scenario",
            "setting",
            "setting_order",
            "domain",
            "metric",
            "mlp_relative_degradation_mean",
            "gcn_relative_degradation_mean",
            "mean_delta_mlp_minus_gcn",
            "median_delta_mlp_minus_gcn",
            "sd_delta_across_splits",
            "gcn_favored_splits",
            "mlp_favored_splits",
            "ties",
        ]
    ]


contrast_parts.append(
    contrast_table(
        ar1_split,
        "ar1_correlation",
        "rho",
    )
)

contrast_parts.append(
    contrast_table(
        bandwidth_split,
        "bandwidth",
        "filter_name",
    )
)

contrast_parts.append(
    contrast_table(
        drift_split,
        "controlled_drift",
        "terminal_shift_std_units",
    )
)


contrast = pd.concat(
    contrast_parts,
    ignore_index=True,
)

assert len(
    contrast
) == 104


# =============================================================================
# FIRST PERTURBATION-SPECTRUM VALUE READ
# =============================================================================

diag = pd.read_csv(
    PERTURBATION_DIAGNOSTICS,
    low_memory=False,
)

assert len(
    diag
) == 325


diag = make_setting_columns(
    diag
)


spectral = (
    diag
    .groupby(
        [
            "scenario",
            "setting",
            "setting_order",
        ],
        as_index=False,
        observed=True,
    )
    .agg(
        diagnostic_count=(
            "condition_id",
            "size",
        ),

        global_rms_mean=(
            "global_rms",
            "mean",
        ),

        global_rms_sd=(
            "global_rms",
            "std",
        ),

        spectral_centroid_mean=(
            "spectral_centroid_omega_over_pi",
            "mean",
        ),

        spectral_centroid_sd=(
            "spectral_centroid_omega_over_pi",
            "std",
        ),

        low_energy_fraction_mean=(
            "low_energy_fraction",
            "mean",
        ),

        mid_low_energy_fraction_mean=(
            "mid_low_energy_fraction",
            "mean",
        ),

        mid_high_energy_fraction_mean=(
            "mid_high_energy_fraction",
            "mean",
        ),

        high_energy_fraction_mean=(
            "high_energy_fraction",
            "mean",
        ),

        parseval_max_relative_error=(
            "parseval_max_relative_error",
            "max",
        ),
    )
)


assert len(
    spectral
) == 13

assert set(
    spectral[
        "diagnostic_count"
    ]
) == {25}


fraction_sum = (
    spectral[
        "low_energy_fraction_mean"
    ]
    + spectral[
        "mid_low_energy_fraction_mean"
    ]
    + spectral[
        "mid_high_energy_fraction_mean"
    ]
    + spectral[
        "high_energy_fraction_mean"
    ]
)

assert np.allclose(
    fraction_sum,
    1.0,
    rtol=1e-10,
    atol=1e-12,
)


# =============================================================================
# Stable ordering + write
# =============================================================================

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


clean_check = clean_check.sort_values(
    keys,
    kind="mergesort",
).reset_index(
    drop=True
)


ar1_split = ar1_split.sort_values(
    [
        "metric",
        "domain",
        "model",
        "split_seed",
        "rho",
    ],
    kind="mergesort",
).reset_index(
    drop=True
)


bandwidth_split[
    "_order"
] = bandwidth_split[
    "filter_name"
].map(
    {
        v: i
        for i, v in enumerate(
            FILTER_ORDER
        )
    }
)

bandwidth_split = bandwidth_split.sort_values(
    [
        "metric",
        "domain",
        "model",
        "split_seed",
        "_order",
    ],
    kind="mergesort",
).drop(
    columns="_order"
).reset_index(
    drop=True
)


drift_split = drift_split.sort_values(
    [
        "metric",
        "domain",
        "model",
        "split_seed",
        "terminal_shift_std_units",
    ],
    kind="mergesort",
).reset_index(
    drop=True
)


ar1_summary = ar1_summary.sort_values(
    [
        "metric",
        "domain",
        "model",
        "rho",
    ],
    kind="mergesort",
).reset_index(
    drop=True
)


bandwidth_summary[
    "_order"
] = bandwidth_summary[
    "filter_name"
].map(
    {
        v: i
        for i, v in enumerate(
            FILTER_ORDER
        )
    }
)

bandwidth_summary = bandwidth_summary.sort_values(
    [
        "metric",
        "domain",
        "model",
        "_order",
    ],
    kind="mergesort",
).drop(
    columns="_order"
).reset_index(
    drop=True
)


drift_summary = drift_summary.sort_values(
    [
        "metric",
        "domain",
        "model",
        "terminal_shift_std_units",
    ],
    kind="mergesort",
).reset_index(
    drop=True
)


contrast = contrast.sort_values(
    [
        "scenario",
        "setting_order",
        "metric",
        "domain",
    ],
    kind="mergesort",
).reset_index(
    drop=True
)


spectral = spectral.sort_values(
    [
        "scenario",
        "setting_order",
    ],
    kind="mergesort",
).reset_index(
    drop=True
)


atomic_csv(
    clean_check,
    CLEAN_CHECK,
)

atomic_csv(
    ar1_split,
    AR1_SPLIT,
)

atomic_csv(
    ar1_summary,
    AR1_SUMMARY,
)

atomic_csv(
    bandwidth_split,
    BANDWIDTH_SPLIT,
)

atomic_csv(
    bandwidth_summary,
    BANDWIDTH_SUMMARY,
)

atomic_csv(
    drift_split,
    DRIFT_SPLIT,
)

atomic_csv(
    drift_summary,
    DRIFT_SUMMARY,
)

atomic_csv(
    contrast,
    CONTRAST,
)

atomic_csv(
    spectral,
    SPECTRAL,
)


manifest = {
    "contract":
        "correlation_bandwidth_drift_analysis_v1",

    "status":
        "COMPLETE",

    "spectral_protocol_sha256":
        EXPECTED_PROTOCOL_SHA,

    "execution_manifest_sha256":
        EXPECTED_EXEC_SHA,

    "analysis_script_sha256":
        script_sha,

    "analysis_script_commit":
        script_commit,

    "value_level_read_started_only_after_analysis_script_commit":
        True,

    "raw_metric_files_read":
        30,

    "raw_metric_rows_read":
        7920,

    "perturbation_spectral_rows_read":
        325,

    "clean_final_v3_reference_barrier":
        "PASS",

    "clean_reference_rows":
        120,

    "clean_reference_tolerance":
        CLEAN_TOL,

    "clean_max_abs_rmse_difference":
        max_clean_rmse,

    "clean_max_abs_mae_difference":
        max_clean_mae,

    "relative_degradation_definition":
        "(perturbed_metric-clean_metric)/clean_metric",

    "clean_pairing":
        (
            "same split_seed + same train_seed + "
            "same model + same domain + same metric"
        ),

    "nested_aggregation":
        (
            "3 train initializations x 5 perturbation seeds "
            "averaged within each split and perturbation setting"
        ),

    "nested_count_per_split_setting":
        15,

    "split_units":
        5,

    "ar1_settings":
        AR1_RHO,

    "bandwidth_settings":
        FILTER_ORDER,

    "drift_settings":
        DRIFT_LEVELS,

    "analysis_role":
        "descriptive mechanism analysis",

    "new_hypothesis_tests_performed":
        False,

    "p_values_computed":
        False,

    "model_contrast_definition":
        (
            "MLP relative degradation minus "
            "GCN relative degradation"
        ),

    "positive_contrast_interpretation":
        "GCN lower relative degradation",

    "models_retrained":
        False,

    "scaler_refit":
        False,

    "claim_guard": {
        "universal_gcn_robustness":
            False,

        "spectral_gcn":
            False,

        "physical_sensor_interpretation":
            False,

        "feature_space_mechanism_stress_only":
            True,
    },

    "outputs": {
        "clean_reference_check": {
            "path":
                str(
                    CLEAN_CHECK
                ),

            "sha256":
                sha(
                    CLEAN_CHECK
                ),

            "rows":
                len(
                    clean_check
                ),
        },

        "ar1_response_by_split": {
            "path":
                str(
                    AR1_SPLIT
                ),

            "sha256":
                sha(
                    AR1_SPLIT
                ),

            "rows":
                len(
                    ar1_split
                ),
        },

        "ar1_response_summary": {
            "path":
                str(
                    AR1_SUMMARY
                ),

            "sha256":
                sha(
                    AR1_SUMMARY
                ),

            "rows":
                len(
                    ar1_summary
                ),
        },

        "bandwidth_response_by_split": {
            "path":
                str(
                    BANDWIDTH_SPLIT
                ),

            "sha256":
                sha(
                    BANDWIDTH_SPLIT
                ),

            "rows":
                len(
                    bandwidth_split
                ),
        },

        "bandwidth_response_summary": {
            "path":
                str(
                    BANDWIDTH_SUMMARY
                ),

            "sha256":
                sha(
                    BANDWIDTH_SUMMARY
                ),

            "rows":
                len(
                    bandwidth_summary
                ),
        },

        "controlled_drift_response_by_split": {
            "path":
                str(
                    DRIFT_SPLIT
                ),

            "sha256":
                sha(
                    DRIFT_SPLIT
                ),

            "rows":
                len(
                    drift_split
                ),
        },

        "controlled_drift_response_summary": {
            "path":
                str(
                    DRIFT_SUMMARY
                ),

            "sha256":
                sha(
                    DRIFT_SUMMARY
                ),

            "rows":
                len(
                    drift_summary
                ),
        },

        "model_contrast_summary": {
            "path":
                str(
                    CONTRAST
                ),

            "sha256":
                sha(
                    CONTRAST
                ),

            "rows":
                len(
                    contrast
                ),
        },

        "perturbation_spectral_summary": {
            "path":
                str(
                    SPECTRAL
                ),

            "sha256":
                sha(
                    SPECTRAL
                ),

            "rows":
                len(
                    spectral
                ),
        },
    },
}


atomic_json(
    manifest,
    MANIFEST,
)


# =============================================================================
# Compact first surface: Balanced-All only
# =============================================================================

print("=" * 96)
print("CORRELATION / BANDWIDTH / CONTROLLED-DRIFT ANALYSIS")
print("=" * 96)

print(
    "clean final-v3 reference barrier: PASS"
)

print(
    "clean reference rows:",
    120
)

print(
    "max abs RMSE difference:",
    f"{max_clean_rmse:.3e}"
)

print(
    "max abs MAE difference:",
    f"{max_clean_mae:.3e}"
)

print(
    "analysis role: descriptive mechanism analysis"
)

print(
    "new hypothesis tests performed:",
    False
)

print()


print(
    "[BALANCED_ALL — CLEAN-RELATIVE DEGRADATION %, "
    "POSITIVE DELTA FAVORS GCN]"
)

print(
    "scenario          setting                     "
    "RMSE_MLP% RMSE_GCN% delta_pp wins   "
    "MAE_MLP% MAE_GCN% delta_pp wins   centroid"
)


spec_lookup = spectral.set_index(
    [
        "scenario",
        "setting",
    ]
)


for scenario in [
    "ar1_correlation",
    "bandwidth",
    "controlled_drift",
]:

    subset = contrast[
        (
            contrast[
                "scenario"
            ] == scenario
        )
        & (
            contrast[
                "domain"
            ] == "BALANCED_ALL"
        )
    ]


    settings = (
        subset[
            [
                "setting",
                "setting_order",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "setting_order",
            kind="mergesort",
        )
    )


    for srow in settings.itertuples(
        index=False
    ):

        setting = str(
            srow.setting
        )

        r = subset[
            (
                subset[
                    "setting"
                ] == setting
            )
            & (
                subset[
                    "metric"
                ] == "rmse"
            )
        ].iloc[
            0
        ]

        a = subset[
            (
                subset[
                    "setting"
                ] == setting
            )
            & (
                subset[
                    "metric"
                ] == "mae"
            )
        ].iloc[
            0
        ]


        centroid = float(
            spec_lookup.loc[
                (
                    scenario,
                    setting,
                ),
                "spectral_centroid_mean",
            ]
        )


        print(
            f"{scenario:<17} "
            f"{setting:<27} "
            f"{100*float(r['mlp_relative_degradation_mean']):>9.3f} "
            f"{100*float(r['gcn_relative_degradation_mean']):>9.3f} "
            f"{100*float(r['mean_delta_mlp_minus_gcn']):>8.3f} "
            f"{int(r['gcn_favored_splits'])}/5   "
            f"{100*float(a['mlp_relative_degradation_mean']):>8.3f} "
            f"{100*float(a['gcn_relative_degradation_mean']):>8.3f} "
            f"{100*float(a['mean_delta_mlp_minus_gcn']):>8.3f} "
            f"{int(a['gcn_favored_splits'])}/5   "
            f"{centroid:.4f}"
        )


print()
print("[PERTURBATION SPECTRAL ENERGY — MEAN FRACTIONS]")

print(
    "scenario          setting                     "
    "RMS      centroid low      midL     midH     high"
)


for row in spectral.itertuples(
    index=False
):

    print(
        f"{row.scenario:<17} "
        f"{row.setting:<27} "
        f"{float(row.global_rms_mean):>7.4f} "
        f"{float(row.spectral_centroid_mean):>8.4f} "
        f"{float(row.low_energy_fraction_mean):>8.4f} "
        f"{float(row.mid_low_energy_fraction_mean):>8.4f} "
        f"{float(row.mid_high_energy_fraction_mean):>8.4f} "
        f"{float(row.high_energy_fraction_mean):>8.4f}"
    )


print()
print(
    "model contrast rows:",
    len(
        contrast
    )
)

print(
    "perturbation spectral settings:",
    len(
        spectral
    )
)

print(
    "analysis manifest SHA256:",
    sha(
        MANIFEST
    )
)

print()
print(
    "CORRELATION_BANDWIDTH_DRIFT_ANALYSIS_PASS=True"
)
