from pathlib import Path
from itertools import product
import hashlib
import json
import math
import os
import re
import subprocess

import numpy as np
import pandas as pd


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

THIS = (
    REPO
    / "scripts/revision_2026/71_analyze_sinusoidal_frequency_response.py"
)

ROOT = (
    BASE
    / "results/revision_2026/spectral_frequency/final_spectral_v1"
)

PROTOCOL = ROOT / "protocol.json"

EVAL = ROOT / "sinusoidal_frequency_v1"
RUNS = EVAL / "runs"

EXEC_MANIFEST = (
    EVAL
    / "sinusoidal_frequency_execution_manifest.json"
)

FINAL_V3 = (
    BASE
    / "results/revision_2026/feature_matched_soh/final_v3"
)

OUT = EVAL / "analysis_v1"

CLEAN_CHECK = OUT / "clean_final_v3_reference_check.csv"
CURVE_SPLIT = OUT / "frequency_response_by_split.csv"
CURVE_SUMMARY = OUT / "frequency_response_summary.csv"
AUC_SPLIT = OUT / "frequency_auc_by_split.csv"
INFERENCE = OUT / "frequency_primary_inference.csv"
TRANSFER = OUT / "analytic_causal_aggregation_transfer.csv"
MANIFEST = OUT / "frequency_analysis_manifest.json"


EXPECTED_PROTOCOL_SHA = (
    "a8d3b65b87980470f326717f0cb50e3d4fb4d2e0429c93171eb0d21d6c713e45"
)

EXPECTED_EXEC_SHA = (
    "ac56254c584990651ee7430786f3a39b3f7ed96d278091e65e30b115651d19fd"
)

EXPECTED_FINAL_V3_EXEC_SHA = (
    "5a8bfbe0f11a6158f2749a9da4f18f1e62adf8c839976c1bf0af9c9f6c07440d"
)

CLEAN_TOL = 1e-7

FREQUENCIES = [
    0.0625,
    0.125,
    0.25,
    0.50,
    0.75,
    1.00,
]

DEPTHS = [0, 1, 2, 3, 4]

SPLITS = [0, 1, 2, 3, 4]
TRAIN_SEEDS = [0, 1, 2]
PERTURBATION_SEEDS = [0, 1, 2, 3, 4]

MODELS = ["MLP", "GCN"]

DOMAINS = [
    "BALANCED_ALL",
    "HUST",
    "CALCE",
    "MATR",
]

METRICS = ["rmse", "mae"]


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


def normalized_trapezoid_auc(x, y):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    assert x.ndim == 1
    assert y.ndim == 1
    assert len(x) == len(y) == 6

    assert np.all(
        np.diff(x) > 0
    )

    area = float(
        np.sum(
            (
                x[1:]
                - x[:-1]
            )
            * (
                y[1:]
                + y[:-1]
            )
            / 2.0
        )
    )

    width = float(
        x[-1]
        - x[0]
    )

    assert width > 0.0

    return area / width


def exact_signflip_p(deltas):
    d = np.asarray(
        deltas,
        dtype=np.float64,
    )

    assert d.shape == (5,)
    assert np.isfinite(d).all()

    observed = abs(
        float(
            np.mean(d)
        )
    )

    null = []

    for signs in product(
        [-1.0, 1.0],
        repeat=5,
    ):

        s = np.asarray(
            signs,
            dtype=np.float64,
        )

        null.append(
            abs(
                float(
                    np.mean(
                        s * d
                    )
                )
            )
        )

    null = np.asarray(
        null,
        dtype=np.float64,
    )

    return float(
        np.mean(
            null
            >= observed - 1e-15
        )
    )


def holm_adjust(pvalues):
    p = np.asarray(
        pvalues,
        dtype=np.float64,
    )

    assert np.isfinite(p).all()

    order = np.argsort(
        p,
        kind="mergesort",
    )

    out = np.empty(
        len(p),
        dtype=np.float64,
    )

    running = 0.0

    for rank, idx in enumerate(order):

        candidate = min(
            1.0,
            float(
                (
                    len(p)
                    - rank
                )
                * p[idx]
            ),
        )

        running = max(
            running,
            candidate,
        )

        out[idx] = running

    return out


# ===========================================================================
# PRE-VALUE-READ GUARDS
# ===========================================================================

assert sha(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

assert sha(
    EXEC_MANIFEST
) == EXPECTED_EXEC_SHA


protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)

execution = json.loads(
    EXEC_MANIFEST.read_text(
        encoding="utf-8"
    )
)


assert protocol[
    "status"
] == "FROZEN_BEFORE_SPECTRAL_RUNS"

assert protocol[
    "sinusoidal_frequency_sweep"
][
    "omega_over_pi"
] == FREQUENCIES

assert protocol[
    "graph_operator"
][
    "analytic_depth_grid"
] == DEPTHS

assert protocol[
    "primary_frequency_inference"
][
    "paired_unit"
] == "split_seed"

assert protocol[
    "primary_frequency_inference"
][
    "n"
] == 5

assert protocol[
    "primary_frequency_inference"
][
    "minimum_attainable_two_sided_p"
] == 0.0625


assert execution[
    "contract"
] == "final_sinusoidal_frequency_execution_v1"

assert execution[
    "status"
] == "COMPLETE_BEFORE_VALUE_LEVEL_ANALYSIS"

assert execution[
    "completed_runs"
] == 30

assert execution[
    "expected_total_metric_rows"
] == 3720

assert execution[
    "paired_realization_across_models"
] is True

assert execution[
    "paired_realization_across_train_initializations"
] is True

assert execution[
    "training_reexecuted"
] is False

assert execution[
    "scaler_refit"
] is False

assert execution[
    "metric_values_surfaced_during_execution"
] is False

assert execution[
    "analysis_performed"
] is False

assert execution[
    "final_v3_execution_manifest_sha256"
] == EXPECTED_FINAL_V3_EXEC_SHA


metric_files = sorted(
    RUNS.glob(
        "*_frequency_metrics.csv"
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


# Analysis script itself must already be committed and unchanged.
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


# ===========================================================================
# FIRST FREQUENCY METRIC VALUE READ OCCURS BELOW THIS LINE
# ===========================================================================

frames = []

for p in metric_files:

    df = pd.read_csv(
        p,
        low_memory=False,
    )

    assert len(
        df
    ) == 124

    frames.append(
        df
    )


raw = pd.concat(
    frames,
    ignore_index=True,
)

assert len(
    raw
) == 3720


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
        ].to_numpy(
            dtype=np.float64
        )
    ).all()


assert set(
    raw[
        "split_seed"
    ].astype(int)
) == set(
    SPLITS
)

assert set(
    raw[
        "train_seed"
    ].astype(int)
) == set(
    TRAIN_SEEDS
)

assert set(
    raw[
        "model"
    ].astype(str)
) == set(
    MODELS
)

assert set(
    raw[
        "domain"
    ].astype(str)
) == set(
    DOMAINS
)


# ===========================================================================
# INDEPENDENT CLEAN FINAL-V3 VALUE BARRIER
# ===========================================================================

clean_new = raw[
    raw[
        "scenario"
    ].astype(str)
    == "clean"
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

    assert m is not None

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


assert not clean_new.duplicated(
    keys
).any()

assert not clean_old.duplicated(
    keys
).any()


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
        ]
        <= CLEAN_TOL
    )
    & (
        clean_check[
            "mae_abs_difference"
        ]
        <= CLEAN_TOL
    )
)


assert clean_check[
    "pass"
].all()


max_clean_rmse_diff = float(
    clean_check[
        "rmse_abs_difference"
    ].max()
)

max_clean_mae_diff = float(
    clean_check[
        "mae_abs_difference"
    ].max()
)


# ===========================================================================
# PAIRED CLEAN-RELATIVE DEGRADATION
#
# IMPORTANT:
# Each perturbed metric is first normalized to the CLEAN metric from the
# SAME split_seed + train_seed + model + domain.
#
# perturbation seeds and train initializations remain nested within split.
# ===========================================================================

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


frequency = raw[
    raw[
        "scenario"
    ].astype(str)
    == "sinusoidal_frequency"
].copy()

assert len(
    frequency
) == 3600


assert sorted(
    frequency[
        "omega_over_pi"
    ].drop_duplicates()
    .astype(float)
    .tolist()
) == FREQUENCIES


assert sorted(
    frequency[
        "perturbation_seed"
    ].drop_duplicates()
    .astype(int)
    .tolist()
) == PERTURBATION_SEEDS


freq_long = frequency.melt(
    id_vars=[
        "split_seed",
        "train_seed",
        "model",
        "domain",
        "omega_over_pi",
        "perturbation_seed",
    ],
    value_vars=METRICS,
    var_name="metric",
    value_name="perturbed_value",
)


assert len(
    freq_long
) == 7200


paired = freq_long.merge(
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
) == 7200

assert np.isfinite(
    paired[
        "clean_value"
    ].to_numpy(
        dtype=np.float64
    )
).all()

assert (
    paired[
        "clean_value"
    ] > 0
).all()


paired[
    "relative_degradation"
] = (
    (
        paired[
            "perturbed_value"
        ]
        - paired[
            "clean_value"
        ]
    )
    / paired[
        "clean_value"
    ]
)


assert np.isfinite(
    paired[
        "relative_degradation"
    ].to_numpy(
        dtype=np.float64
    )
).all()


nested_counts = (
    paired
    .groupby(
        [
            "split_seed",
            "model",
            "domain",
            "metric",
            "omega_over_pi",
        ],
        observed=True,
    )
    .size()
)

# 3 train initializations x 5 perturbation seeds.
assert set(
    nested_counts.unique()
) == {15}


curve_split = (
    paired
    .groupby(
        [
            "split_seed",
            "model",
            "domain",
            "metric",
            "omega_over_pi",
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


assert len(
    curve_split
) == 480

assert set(
    curve_split[
        "nested_count"
    ]
) == {15}


curve_summary = (
    curve_split
    .groupby(
        [
            "model",
            "domain",
            "metric",
            "omega_over_pi",
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


assert len(
    curve_summary
) == 96

assert set(
    curve_summary[
        "split_count"
    ]
) == {5}


# ===========================================================================
# SPLIT-LEVEL NORMALIZED FREQUENCY AUC
# ===========================================================================

auc_rows = []


for split_seed in SPLITS:

    for model in MODELS:

        for domain in DOMAINS:

            for metric in METRICS:

                g = curve_split[
                    (
                        curve_split[
                            "split_seed"
                        ] == split_seed
                    )
                    & (
                        curve_split[
                            "model"
                        ] == model
                    )
                    & (
                        curve_split[
                            "domain"
                        ] == domain
                    )
                    & (
                        curve_split[
                            "metric"
                        ] == metric
                    )
                ].sort_values(
                    "omega_over_pi",
                    kind="mergesort",
                )


                assert len(
                    g
                ) == 6

                x = g[
                    "omega_over_pi"
                ].to_numpy(
                    dtype=np.float64
                )

                y = g[
                    "relative_degradation"
                ].to_numpy(
                    dtype=np.float64
                )

                assert np.allclose(
                    x,
                    np.asarray(
                        FREQUENCIES,
                        dtype=np.float64,
                    ),
                    rtol=0.0,
                    atol=1e-15,
                )


                auc_rows.append({
                    "split_seed":
                        split_seed,

                    "model":
                        model,

                    "domain":
                        domain,

                    "metric":
                        metric,

                    "normalized_frequency_auc":
                        normalized_trapezoid_auc(
                            x,
                            y,
                        ),

                    "frequency_min_over_pi":
                        float(
                            x[0]
                        ),

                    "frequency_max_over_pi":
                        float(
                            x[-1]
                        ),

                    "frequency_range_width":
                        float(
                            x[-1]
                            - x[0]
                        ),
                })


auc_split = pd.DataFrame(
    auc_rows
)

assert len(
    auc_split
) == 80


# ===========================================================================
# PRIMARY PAIRED INFERENCE — n=5 split units
# ===========================================================================

infer_rows = []


for metric in METRICS:

    for domain in DOMAINS:

        g = auc_split[
            (
                auc_split[
                    "metric"
                ] == metric
            )
            & (
                auc_split[
                    "domain"
                ] == domain
            )
        ]


        pivot = g.pivot(
            index="split_seed",
            columns="model",
            values="normalized_frequency_auc",
        ).loc[
            SPLITS
        ]


        mlp = pivot[
            "MLP"
        ].to_numpy(
            dtype=np.float64
        )

        gcn = pivot[
            "GCN"
        ].to_numpy(
            dtype=np.float64
        )

        delta = (
            mlp
            - gcn
        )


        infer_rows.append({
            "metric":
                metric,

            "domain":
                domain,

            "split_count":
                5,

            "mlp_degradation_auc_mean":
                float(
                    np.mean(
                        mlp
                    )
                ),

            "gcn_degradation_auc_mean":
                float(
                    np.mean(
                        gcn
                    )
                ),

            "mean_delta_mlp_minus_gcn":
                float(
                    np.mean(
                        delta
                    )
                ),

            "median_delta_mlp_minus_gcn":
                float(
                    np.median(
                        delta
                    )
                ),

            "gcn_favored_splits":
                int(
                    np.sum(
                        delta > 0
                    )
                ),

            "mlp_favored_splits":
                int(
                    np.sum(
                        delta < 0
                    )
                ),

            "ties":
                int(
                    np.sum(
                        delta == 0
                    )
                ),

            "p_exact_signflip":
                exact_signflip_p(
                    delta
                ),
        })


inference = pd.DataFrame(
    infer_rows
)

assert len(
    inference
) == 8


inference[
    "p_holm"
] = np.nan


# Frozen multiplicity contract:
# four domains separately for RMSE; four separately for MAE.
for metric in METRICS:

    idx = inference.index[
        inference[
            "metric"
        ] == metric
    ]

    assert len(
        idx
    ) == 4

    inference.loc[
        idx,
        "p_holm",
    ] = holm_adjust(
        inference.loc[
            idx,
            "p_exact_signflip",
        ].to_numpy(
            dtype=np.float64
        )
    )


inference[
    "direction"
] = np.where(
    inference[
        "mean_delta_mlp_minus_gcn"
    ] > 0,
    "GCN_LOWER_DEGRADATION_AUC",
    np.where(
        inference[
            "mean_delta_mlp_minus_gcn"
        ] < 0,
        "MLP_LOWER_DEGRADATION_AUC",
        "TIE",
    ),
)


inference[
    "significant_holm_0p05"
] = (
    inference[
        "p_holm"
    ] < 0.05
)


# ===========================================================================
# ANALYTIC CAUSAL AGGREGATION TRANSFER
#
# Frozen operator:
# |H_d(omega)| = |cos(omega/2)|^d
# omega = pi * (omega/pi coordinate)
# ===========================================================================

transfer_rows = []


for depth in DEPTHS:

    for f in FREQUENCIES:

        omega = (
            math.pi
            * float(f)
        )

        amplitude = (
            abs(
                math.cos(
                    omega / 2.0
                )
            )
            ** int(
                depth
            )
        )

        transfer_rows.append({
            "depth":
                int(
                    depth
                ),

            "omega_over_pi":
                float(
                    f
                ),

            "amplitude_gain":
                float(
                    amplitude
                ),

            "power_gain":
                float(
                    amplitude ** 2
                ),
        })


transfer = pd.DataFrame(
    transfer_rows
)

assert len(
    transfer
) == 30


# Basic analytic sanity.
assert np.allclose(
    transfer[
        transfer[
            "depth"
        ] == 0
    ][
        "amplitude_gain"
    ],
    1.0,
    rtol=0.0,
    atol=1e-15,
)

assert (
    transfer[
        transfer[
            "depth"
        ] > 0
    ][
        "amplitude_gain"
    ]
    <= 1.0
).all()


# ===========================================================================
# WRITE OUTPUTS
# ===========================================================================

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


curve_split = curve_split.sort_values(
    [
        "metric",
        "domain",
        "model",
        "split_seed",
        "omega_over_pi",
    ],
    kind="mergesort",
).reset_index(
    drop=True
)


curve_summary = curve_summary.sort_values(
    [
        "metric",
        "domain",
        "model",
        "omega_over_pi",
    ],
    kind="mergesort",
).reset_index(
    drop=True
)


auc_split = auc_split.sort_values(
    [
        "metric",
        "domain",
        "model",
        "split_seed",
    ],
    kind="mergesort",
).reset_index(
    drop=True
)


inference = inference.sort_values(
    [
        "metric",
        "domain",
    ],
    kind="mergesort",
).reset_index(
    drop=True
)


transfer = transfer.sort_values(
    [
        "depth",
        "omega_over_pi",
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
    curve_split,
    CURVE_SPLIT,
)

atomic_csv(
    curve_summary,
    CURVE_SUMMARY,
)

atomic_csv(
    auc_split,
    AUC_SPLIT,
)

atomic_csv(
    inference,
    INFERENCE,
)

atomic_csv(
    transfer,
    TRANSFER,
)


manifest = {
    "contract":
        "sinusoidal_frequency_analysis_v1",

    "status":
        "COMPLETE",

    "spectral_protocol_sha256":
        EXPECTED_PROTOCOL_SHA,

    "frequency_execution_manifest_sha256":
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
        3720,

    "clean_final_v3_reference_barrier":
        "PASS",

    "clean_reference_rows":
        120,

    "clean_reference_tolerance":
        CLEAN_TOL,

    "clean_max_abs_rmse_difference":
        max_clean_rmse_diff,

    "clean_max_abs_mae_difference":
        max_clean_mae_diff,

    "frequency_grid_omega_over_pi":
        FREQUENCIES,

    "frequency_count":
        6,

    "perturbation_seed_count":
        5,

    "train_initialization_count_per_split":
        3,

    "relative_degradation_definition":
        "(perturbed_metric-clean_metric)/clean_metric",

    "clean_pairing":
        (
            "same split_seed + same train_seed + "
            "same model + same domain + same metric"
        ),

    "nested_aggregation":
        (
            "relative degradation computed before averaging; "
            "3 train initializations x 5 perturbation seeds "
            "averaged within each split-frequency-model-domain-metric"
        ),

    "nested_count_per_split_frequency":
        15,

    "inferential_unit":
        "split_seed",

    "inferential_n":
        5,

    "frequency_auc":
        (
            "trapezoidal AUC of split-level mean relative degradation "
            "over frozen six-point omega/pi grid divided by "
            "frequency range width"
        ),

    "frequency_auc_range_width":
        float(
            FREQUENCIES[-1]
            - FREQUENCIES[0]
        ),

    "primary_delta_definition":
        "MLP_degradation_AUC_minus_GCN_degradation_AUC",

    "positive_delta_interpretation":
        "GCN lower integrated relative degradation",

    "test":
        "exact two-sided paired sign-flip over five split units",

    "multiplicity":
        (
            "Holm across four domains separately for RMSE "
            "and four domains separately for MAE"
        ),

    "primary_comparisons":
        8,

    "minimum_attainable_two_sided_p":
        0.0625,

    "minimum_possible_first_holm_adjusted_p":
        0.25,

    "holm_significant_0p05_count":
        int(
            inference[
                "significant_holm_0p05"
            ].sum()
        ),

    "analytic_transfer_definition":
        "|H_d(omega)|=|cos(omega/2)|^d",

    "analytic_depths":
        DEPTHS,

    "models_retrained":
        False,

    "scaler_refit":
        False,

    "claim_guard": {
        "universal_gcn_robustness":
            False,

        "spectral_gcn":
            False,

        "frequency_selective_temporal_aggregation":
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

        "frequency_response_by_split": {
            "path":
                str(
                    CURVE_SPLIT
                ),

            "sha256":
                sha(
                    CURVE_SPLIT
                ),

            "rows":
                len(
                    curve_split
                ),
        },

        "frequency_response_summary": {
            "path":
                str(
                    CURVE_SUMMARY
                ),

            "sha256":
                sha(
                    CURVE_SUMMARY
                ),

            "rows":
                len(
                    curve_summary
                ),
        },

        "frequency_auc_by_split": {
            "path":
                str(
                    AUC_SPLIT
                ),

            "sha256":
                sha(
                    AUC_SPLIT
                ),

            "rows":
                len(
                    auc_split
                ),
        },

        "frequency_primary_inference": {
            "path":
                str(
                    INFERENCE
                ),

            "sha256":
                sha(
                    INFERENCE
                ),

            "rows":
                len(
                    inference
                ),
        },

        "analytic_causal_aggregation_transfer": {
            "path":
                str(
                    TRANSFER
                ),

            "sha256":
                sha(
                    TRANSFER
                ),

            "rows":
                len(
                    transfer
                ),
        },
    },
}


atomic_json(
    manifest,
    MANIFEST,
)


# ===========================================================================
# COMPACT FIRST RESULT SURFACE
# ===========================================================================

print("=" * 94)
print("SINUSOIDAL FREQUENCY-RESPONSE ANALYSIS")
print("=" * 94)

print(
    "clean final-v3 reference barrier: PASS"
)

print(
    "clean reference rows:",
    120
)

print(
    "max abs RMSE difference:",
    f"{max_clean_rmse_diff:.3e}"
)

print(
    "max abs MAE difference:",
    f"{max_clean_mae_diff:.3e}"
)

print(
    "inferential unit: split_seed (n=5)"
)

print(
    "nested observations per split/frequency/model/domain/metric:",
    15
)

print(
    "frequency AUC range width:",
    f"{FREQUENCIES[-1] - FREQUENCIES[0]:.4f}"
)

print(
    "minimum attainable two-sided p:",
    "0.0625"
)

print()


print("[BALANCED_ALL RELATIVE DEGRADATION BY FREQUENCY — %]")

print(
    "metric omega/pi    MLP_mean%    GCN_mean%    MLP-GCN pp"
)


balanced = curve_summary[
    curve_summary[
        "domain"
    ] == "BALANCED_ALL"
]


for metric in METRICS:

    for f in FREQUENCIES:

        g = balanced[
            (
                balanced[
                    "metric"
                ] == metric
            )
            & (
                np.isclose(
                    balanced[
                        "omega_over_pi"
                    ].astype(float),
                    f,
                )
            )
        ].set_index(
            "model"
        )

        mlp = float(
            g.loc[
                "MLP",
                "mean_relative_degradation",
            ]
        )

        gcn = float(
            g.loc[
                "GCN",
                "mean_relative_degradation",
            ]
        )

        print(
            f"{metric:<6} "
            f"{f:>7.4f} "
            f"{100.0 * mlp:>11.3f} "
            f"{100.0 * gcn:>11.3f} "
            f"{100.0 * (mlp-gcn):>11.3f}"
        )


print()
print("[PRIMARY NORMALIZED FREQUENCY AUC]")

print(
    "metric domain        MLP_AUC     GCN_AUC     delta(M-G) wins p_exact p_holm direction"
)


for _, r in inference.iterrows():

    print(
        f"{r['metric']:<6} "
        f"{r['domain']:<12} "
        f"{float(r['mlp_degradation_auc_mean']):>11.6f} "
        f"{float(r['gcn_degradation_auc_mean']):>11.6f} "
        f"{float(r['mean_delta_mlp_minus_gcn']):>11.6f} "
        f"{int(r['gcn_favored_splits'])}/5 "
        f"{float(r['p_exact_signflip']):>7.4f} "
        f"{float(r['p_holm']):>6.4f} "
        f"{r['direction']}"
    )


print()
print("[ANALYTIC DEPTH-2 CAUSAL AGGREGATION AMPLITUDE]")

depth2 = transfer[
    transfer[
        "depth"
    ] == 2
]

for _, r in depth2.iterrows():

    print(
        f"omega/pi={float(r['omega_over_pi']):.4f} "
        f"|H2|={float(r['amplitude_gain']):.6f} "
        f"power={float(r['power_gain']):.6f}"
    )


print()
print(
    "Holm-significant primary comparisons:",
    int(
        inference[
            "significant_holm_0p05"
        ].sum()
    ),
    "/ 8"
)

print(
    "analysis manifest SHA256:",
    sha(
        MANIFEST
    )
)

print()
print(
    "SINUSOIDAL_FREQUENCY_ANALYSIS_PASS=True"
)
