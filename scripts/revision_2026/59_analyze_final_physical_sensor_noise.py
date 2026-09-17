from pathlib import Path
from itertools import product
import hashlib
import json
import math
import os
import subprocess

import numpy as np
import pandas as pd


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo" / "battery-gnn"

THIS = (
    REPO
    / "scripts"
    / "revision_2026"
    / "59_analyze_final_physical_sensor_noise.py"
)

ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "physical_sensor_noise"
    / "final_physical_v1"
)

EVAL = ROOT / "evaluation"
RUNS = EVAL / "runs"

EXECUTION_MANIFEST = (
    EVAL
    / "physical_noise_execution_manifest.json"
)

CONDITIONS = (
    ROOT
    / "corrupted_features"
    / "conditions.json"
)

OUT = ROOT / "analysis"

LEVEL_SPLIT_CSV = (
    OUT
    / "physical_noise_level_by_split.csv"
)

LEVEL_SUMMARY_CSV = (
    OUT
    / "physical_noise_level_summary.csv"
)

AUC_SPLIT_CSV = (
    OUT
    / "physical_noise_auc_by_split.csv"
)

INFERENCE_CSV = (
    OUT
    / "physical_noise_primary_inference.csv"
)

COMPOUND_SPLIT_CSV = (
    OUT
    / "physical_noise_compound_medium_by_split.csv"
)

COMPOUND_SUMMARY_CSV = (
    OUT
    / "physical_noise_compound_medium_summary.csv"
)

CLEAN_REFERENCE_CSV = (
    OUT
    / "physical_noise_clean_reference_check.csv"
)

MANIFEST = (
    OUT
    / "physical_noise_analysis_manifest.json"
)


EXPECTED_EXECUTION_MANIFEST_SHA = (
    "7880a61bca31ed2f3329b5bdf9aad0481a4306ffd9f05402e60764b3c0e124ad"
)

EXPECTED_FEATURE_CLOSURE_SHA = (
    "382ea97c8f1d8f219436dd5071bb8fcbd63811f24e76b8c887e8efba4e54675f"
)

FAMILIES = [
    "gaussian",
    "calibration_offset",
    "sensor_drift",
    "sample_dropout",
]

DOMAINS = [
    "BALANCED_ALL",
    "HUST",
    "CALCE",
    "MATR",
]

MODELS = [
    "MLP",
    "GCN",
]

PRIMARY_METRICS = [
    "rmse",
    "mae",
]

ALL_METRICS = [
    "rmse",
    "mae",
    "r2",
]

LEVELS = [
    "clean",
    "low",
    "medium",
    "high",
]

SEVERITY = {
    "clean": 0.0,
    "low": 1.0 / 3.0,
    "medium": 2.0 / 3.0,
    "high": 1.0,
}

SPLITS = [
    0,
    1,
    2,
    3,
    4,
]

TRAIN_SEEDS = [
    0,
    1,
    2,
]


# These are rounded PRE-EXISTING clean final-v3 results from the
# already completed clean benchmark analysis. They are used only as
# a coarse regression barrier, not as physical-noise outcomes.
CLEAN_REFERENCE = {
    ("BALANCED_ALL", "rmse", "MLP"): 0.037983,
    ("BALANCED_ALL", "rmse", "GCN"): 0.037586,
    ("BALANCED_ALL", "mae",  "MLP"): 0.026423,
    ("BALANCED_ALL", "mae",  "GCN"): 0.026207,

    ("HUST", "rmse", "MLP"): 0.049742,
    ("HUST", "rmse", "GCN"): 0.049180,
    ("HUST", "mae",  "MLP"): 0.039128,
    ("HUST", "mae",  "GCN"): 0.038552,

    ("CALCE", "rmse", "MLP"): 0.028785,
    ("CALCE", "rmse", "GCN"): 0.028457,
    ("CALCE", "mae",  "MLP"): 0.021641,
    ("CALCE", "mae",  "GCN"): 0.021784,

    ("MATR", "rmse", "MLP"): 0.027232,
    ("MATR", "rmse", "GCN"): 0.026712,
    ("MATR", "mae",  "MLP"): 0.017153,
    ("MATR", "mae",  "GCN"): 0.016717,
}

CLEAN_REFERENCE_TOL = 1.0e-5


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


def exact_two_sided_signflip(deltas):
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

    stats = []

    for signs in product(
        [-1.0, 1.0],
        repeat=len(d),
    ):
        s = np.asarray(
            signs,
            dtype=np.float64,
        )

        stats.append(
            abs(
                float(
                    np.mean(
                        s * d
                    )
                )
            )
        )

    stats = np.asarray(
        stats,
        dtype=np.float64,
    )

    p = float(
        np.mean(
            stats
            >= (
                observed
                - 1e-15
            )
        )
    )

    assert (
        0.0
        <= p
        <= 1.0
    )

    return p


def holm_adjust(pvalues):
    p = np.asarray(
        pvalues,
        dtype=np.float64,
    )

    assert np.isfinite(p).all()
    assert (
        (p >= 0.0)
        & (p <= 1.0)
    ).all()

    m = len(p)

    order = np.argsort(
        p,
        kind="mergesort",
    )

    adjusted = np.empty(
        m,
        dtype=np.float64,
    )

    running = 0.0

    for rank, idx in enumerate(
        order
    ):
        candidate = min(
            1.0,
            float(
                (m - rank)
                * p[idx]
            ),
        )

        running = max(
            running,
            candidate,
        )

        adjusted[
            idx
        ] = running

    return adjusted


def trapezoid_auc(x, y):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    assert x.shape == (4,)
    assert y.shape == (4,)

    assert np.array_equal(
        x,
        np.asarray(
            [
                0.0,
                1.0 / 3.0,
                2.0 / 3.0,
                1.0,
            ],
            dtype=np.float64,
        ),
    )

    return float(
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


# ============================================================================
# Pre-value-read immutable guards
# ============================================================================

assert sha(
    EXECUTION_MANIFEST
) == EXPECTED_EXECUTION_MANIFEST_SHA


execution = json.loads(
    EXECUTION_MANIFEST.read_text(
        encoding="utf-8"
    )
)


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
    "metric_values_surfaced_during_execution"
] is False

assert execution[
    "analysis_performed"
] is False

assert execution[
    "feature_closure_sha256"
] == EXPECTED_FEATURE_CLOSURE_SHA


conditions_doc = json.loads(
    CONDITIONS.read_text(
        encoding="utf-8"
    )
)

assert conditions_doc[
    "condition_count"
] == 66


# Analysis script itself MUST already be committed and unchanged before
# any metric CSV is parsed.
tracked = subprocess.run(
    [
        "git",
        "ls-files",
        "--error-unmatch",
        str(
            THIS.relative_to(
                REPO
            )
        ),
    ],
    cwd=REPO,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

assert tracked.returncode == 0


unchanged = subprocess.run(
    [
        "git",
        "diff",
        "--quiet",
        "HEAD",
        "--",
        str(
            THIS.relative_to(
                REPO
            )
        ),
    ],
    cwd=REPO,
)

assert unchanged.returncode == 0


SCRIPT_SHA = sha(
    THIS
)

SCRIPT_COMMIT = subprocess.check_output(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
    text=True,
).strip()


metric_files = sorted(
    RUNS.glob(
        "*_physical_test_metrics.csv"
    )
)

assert len(
    metric_files
) == 30


# ============================================================================
# FIRST VALUE-LEVEL READ OCCURS BELOW THIS LINE
# ============================================================================

frames = []

for p in metric_files:

    df = pd.read_csv(
        p,
        low_memory=False,
    )

    assert len(
        df
    ) == 264

    assert list(
        df.columns
    ) == [
        "split_seed",
        "train_seed",
        "model",
        "condition_index",
        "condition_id",
        "scenario",
        "level",
        "corruption_seed",
        "severity_axis",
        "domain",
        "n_rows",
        "rmse",
        "mae",
        "r2",
    ]

    frames.append(
        df
    )


raw = pd.concat(
    frames,
    ignore_index=True,
)


assert len(
    raw
) == 7920


for c in [
    "rmse",
    "mae",
    "r2",
]:
    raw[c] = pd.to_numeric(
        raw[c],
        errors="raise",
    )

    assert np.isfinite(
        raw[c].to_numpy(
            dtype=np.float64
        )
    ).all()


assert set(
    raw[
        "split_seed"
    ].astype(int)
) == set(SPLITS)

assert set(
    raw[
        "train_seed"
    ].astype(int)
) == set(TRAIN_SEEDS)

assert set(
    raw[
        "model"
    ].astype(str)
) == set(MODELS)

assert set(
    raw[
        "domain"
    ].astype(str)
) == set(DOMAINS)


key_cols = [
    "split_seed",
    "train_seed",
    "model",
    "condition_index",
    "domain",
]

assert not raw.duplicated(
    key_cols
).any()


expected_scenarios = {
    "clean",
    "gaussian",
    "calibration_offset",
    "sensor_drift",
    "sample_dropout",
    "compound_medium",
}

assert set(
    raw[
        "scenario"
    ].astype(str)
) == expected_scenarios


# ============================================================================
# Convert to long metric representation
# ============================================================================

id_cols = [
    "split_seed",
    "train_seed",
    "model",
    "condition_index",
    "condition_id",
    "scenario",
    "level",
    "corruption_seed",
    "severity_axis",
    "domain",
    "n_rows",
]

long = raw.melt(
    id_vars=id_cols,
    value_vars=ALL_METRICS,
    var_name="metric",
    value_name="value",
)


# ============================================================================
# Clean reference regression barrier
# ============================================================================

clean_long = long[
    long[
        "scenario"
    ] == "clean"
].copy()


clean_counts = (
    clean_long
    .groupby(
        [
            "split_seed",
            "model",
            "domain",
            "metric",
        ],
        observed=True,
    )
    .size()
)

assert set(
    clean_counts.unique()
) == {3}


clean_split = (
    clean_long
    .groupby(
        [
            "split_seed",
            "model",
            "domain",
            "metric",
        ],
        as_index=False,
        observed=True,
    )[
        "value"
    ]
    .mean()
    .rename(
        columns={
            "value":
                "clean_value"
        }
    )
)


clean_observed = (
    clean_split[
        clean_split[
            "metric"
        ].isin(
            PRIMARY_METRICS
        )
    ]
    .groupby(
        [
            "model",
            "domain",
            "metric",
        ],
        as_index=False,
        observed=True,
    )[
        "clean_value"
    ]
    .mean()
)


clean_reference_rows = []


for _, r in clean_observed.iterrows():

    key = (
        str(
            r[
                "domain"
            ]
        ),
        str(
            r[
                "metric"
            ]
        ),
        str(
            r[
                "model"
            ]
        ),
    )

    assert key in CLEAN_REFERENCE

    ref = float(
        CLEAN_REFERENCE[
            key
        ]
    )

    observed = float(
        r[
            "clean_value"
        ]
    )

    diff = (
        observed
        - ref
    )

    passed = (
        abs(
            diff
        )
        <= CLEAN_REFERENCE_TOL
    )

    clean_reference_rows.append({
        "domain":
            key[0],

        "metric":
            key[1],

        "model":
            key[2],

        "observed_clean_mean":
            observed,

        "preexisting_rounded_reference":
            ref,

        "difference":
            diff,

        "tolerance":
            CLEAN_REFERENCE_TOL,

        "pass":
            bool(
                passed
            ),
    })


clean_reference_df = pd.DataFrame(
    clean_reference_rows
).sort_values(
    [
        "domain",
        "metric",
        "model",
    ],
    kind="mergesort",
).reset_index(
    drop=True
)


assert len(
    clean_reference_df
) == 16

assert clean_reference_df[
    "pass"
].all()


# ============================================================================
# Primary four-family level aggregation
#
# inferential hierarchy:
#   train init seeds + corruption seeds = nested within split
#   split = only inferential unit
# ============================================================================

primary = long[
    long[
        "scenario"
    ].isin(
        FAMILIES
    )
].copy()


primary_counts = (
    primary
    .groupby(
        [
            "split_seed",
            "model",
            "scenario",
            "level",
            "domain",
            "metric",
        ],
        observed=True,
    )
    .size()
)


# 3 model initializations x 5 corruption seeds = 15 observations
# within each split/family/level/model/domain/metric.
assert set(
    primary_counts.unique()
) == {15}


primary_split = (
    primary
    .groupby(
        [
            "split_seed",
            "model",
            "scenario",
            "level",
            "domain",
            "metric",
        ],
        as_index=False,
        observed=True,
    )[
        "value"
    ]
    .mean()
)


# Attach a clean baseline to each family.
level_rows = []


for family in FAMILIES:

    c = clean_split.copy()

    c[
        "scenario"
    ] = family

    c[
        "level"
    ] = "clean"

    c[
        "severity_axis"
    ] = 0.0

    c[
        "value"
    ] = c[
        "clean_value"
    ]

    c[
        "change_from_clean"
    ] = 0.0

    c[
        "pct_change_from_clean"
    ] = 0.0

    level_rows.append(
        c[
            [
                "split_seed",
                "model",
                "scenario",
                "level",
                "severity_axis",
                "domain",
                "metric",
                "value",
                "clean_value",
                "change_from_clean",
                "pct_change_from_clean",
            ]
        ]
    )


p = primary_split.merge(
    clean_split,
    on=[
        "split_seed",
        "model",
        "domain",
        "metric",
    ],
    how="left",
    validate="many_to_one",
)


p[
    "severity_axis"
] = p[
    "level"
].map(
    SEVERITY
).astype(
    float
)


p[
    "change_from_clean"
] = (
    p[
        "value"
    ]
    - p[
        "clean_value"
    ]
)


p[
    "pct_change_from_clean"
] = np.where(
    np.abs(
        p[
            "clean_value"
        ]
    )
    > 1e-15,

    100.0
    * p[
        "change_from_clean"
    ]
    / p[
        "clean_value"
    ],

    np.nan,
)


level_rows.append(
    p[
        [
            "split_seed",
            "model",
            "scenario",
            "level",
            "severity_axis",
            "domain",
            "metric",
            "value",
            "clean_value",
            "change_from_clean",
            "pct_change_from_clean",
        ]
    ]
)


level_split = pd.concat(
    level_rows,
    ignore_index=True,
)


assert len(
    level_split
) == (
    5
    * 2
    * 4
    * 4
    * 4
    * 3
)


assert len(
    level_split
) == 1920


level_summary = (
    level_split
    .groupby(
        [
            "model",
            "scenario",
            "level",
            "severity_axis",
            "domain",
            "metric",
        ],
        as_index=False,
        observed=True,
    )
    .agg(
        mean_value=(
            "value",
            "mean",
        ),

        sd_across_splits=(
            "value",
            "std",
        ),

        mean_change_from_clean=(
            "change_from_clean",
            "mean",
        ),

        mean_pct_change_from_clean=(
            "pct_change_from_clean",
            "mean",
        ),

        split_count=(
            "split_seed",
            "nunique",
        ),
    )
)


assert len(
    level_summary
) == (
    2
    * 4
    * 4
    * 4
    * 3
)

assert len(
    level_summary
) == 384

assert set(
    level_summary[
        "split_count"
    ]
) == {5}


# ============================================================================
# Normalized severity AUC
# ============================================================================

auc_rows = []


for split_seed in SPLITS:

    for model in MODELS:

        for family in FAMILIES:

            for domain in DOMAINS:

                for metric in PRIMARY_METRICS:

                    g = level_split[
                        (
                            level_split[
                                "split_seed"
                            ] == split_seed
                        )
                        & (
                            level_split[
                                "model"
                            ] == model
                        )
                        & (
                            level_split[
                                "scenario"
                            ] == family
                        )
                        & (
                            level_split[
                                "domain"
                            ] == domain
                        )
                        & (
                            level_split[
                                "metric"
                            ] == metric
                        )
                    ].copy()


                    assert len(
                        g
                    ) == 4

                    assert set(
                        g[
                            "level"
                        ]
                    ) == set(LEVELS)


                    g[
                        "_order"
                    ] = g[
                        "level"
                    ].map({
                        "clean": 0,
                        "low": 1,
                        "medium": 2,
                        "high": 3,
                    })


                    g = g.sort_values(
                        "_order",
                        kind="mergesort",
                    )


                    x = g[
                        "severity_axis"
                    ].to_numpy(
                        dtype=np.float64
                    )

                    y = g[
                        "value"
                    ].to_numpy(
                        dtype=np.float64
                    )


                    auc = trapezoid_auc(
                        x,
                        y,
                    )


                    auc_rows.append({
                        "split_seed":
                            split_seed,

                        "model":
                            model,

                        "scenario":
                            family,

                        "domain":
                            domain,

                        "metric":
                            metric,

                        "auc":
                            auc,

                        "clean":
                            float(
                                y[0]
                            ),

                        "low":
                            float(
                                y[1]
                            ),

                        "medium":
                            float(
                                y[2]
                            ),

                        "high":
                            float(
                                y[3]
                            ),
                    })


auc_split = pd.DataFrame(
    auc_rows
)


assert len(
    auc_split
) == (
    5
    * 2
    * 4
    * 4
    * 2
)

assert len(
    auc_split
) == 320

assert np.isfinite(
    auc_split[
        "auc"
    ]
).all()


# ============================================================================
# Primary paired split-level inference
# ============================================================================

inference_rows = []


for metric in PRIMARY_METRICS:

    for family in FAMILIES:

        for domain in DOMAINS:

            g = auc_split[
                (
                    auc_split[
                        "metric"
                    ] == metric
                )
                & (
                    auc_split[
                        "scenario"
                    ] == family
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
                values="auc",
            ).loc[
                SPLITS
            ]


            assert list(
                pivot.columns
            ) == [
                "GCN",
                "MLP",
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


            p_exact = (
                exact_two_sided_signflip(
                    delta
                )
            )


            mlp_mean = float(
                np.mean(
                    mlp
                )
            )

            gcn_mean = float(
                np.mean(
                    gcn
                )
            )

            mean_delta = float(
                np.mean(
                    delta
                )
            )


            relative_gcn_advantage_pct = (
                100.0
                * mean_delta
                / mlp_mean
            )


            inference_rows.append({
                "metric":
                    metric,

                "scenario":
                    family,

                "domain":
                    domain,

                "split_count":
                    5,

                "mlp_auc_mean":
                    mlp_mean,

                "gcn_auc_mean":
                    gcn_mean,

                "mean_delta_mlp_minus_gcn":
                    mean_delta,

                "relative_gcn_advantage_pct":
                    relative_gcn_advantage_pct,

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
                    p_exact,
            })


inference = pd.DataFrame(
    inference_rows
)


assert len(
    inference
) == 32


inference[
    "p_holm"
] = np.nan


for metric in PRIMARY_METRICS:

    idx = inference.index[
        inference[
            "metric"
        ] == metric
    ]

    assert len(
        idx
    ) == 16

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
    "significant_holm_0p05"
] = (
    inference[
        "p_holm"
    ]
    < 0.05
)


inference[
    "direction"
] = np.where(
    inference[
        "mean_delta_mlp_minus_gcn"
    ] > 0,

    "GCN_LOWER_ERROR_AUC",

    np.where(
        inference[
            "mean_delta_mlp_minus_gcn"
        ] < 0,

        "MLP_LOWER_ERROR_AUC",

        "TIE",
    ),
)


# ============================================================================
# Compound-medium — descriptive only
# ============================================================================

compound = long[
    long[
        "scenario"
    ] == "compound_medium"
].copy()


compound_counts = (
    compound
    .groupby(
        [
            "split_seed",
            "model",
            "domain",
            "metric",
        ],
        observed=True,
    )
    .size()
)


# 3 train seeds x 5 corruption seeds
assert set(
    compound_counts.unique()
) == {15}


compound_split = (
    compound
    .groupby(
        [
            "split_seed",
            "model",
            "domain",
            "metric",
        ],
        as_index=False,
        observed=True,
    )[
        "value"
    ]
    .mean()
    .rename(
        columns={
            "value":
                "compound_medium_value"
        }
    )
)


compound_split = compound_split.merge(
    clean_split,
    on=[
        "split_seed",
        "model",
        "domain",
        "metric",
    ],
    how="left",
    validate="one_to_one",
)


compound_split[
    "change_from_clean"
] = (
    compound_split[
        "compound_medium_value"
    ]
    - compound_split[
        "clean_value"
    ]
)


compound_split[
    "pct_change_from_clean"
] = np.where(
    np.abs(
        compound_split[
            "clean_value"
        ]
    )
    > 1e-15,

    100.0
    * compound_split[
        "change_from_clean"
    ]
    / compound_split[
        "clean_value"
    ],

    np.nan,
)


assert len(
    compound_split
) == (
    5
    * 2
    * 4
    * 3
)

assert len(
    compound_split
) == 120


compound_summary = (
    compound_split
    .groupby(
        [
            "model",
            "domain",
            "metric",
        ],
        as_index=False,
        observed=True,
    )
    .agg(
        mean_compound_medium_value=(
            "compound_medium_value",
            "mean",
        ),

        sd_across_splits=(
            "compound_medium_value",
            "std",
        ),

        mean_clean_value=(
            "clean_value",
            "mean",
        ),

        mean_change_from_clean=(
            "change_from_clean",
            "mean",
        ),

        mean_pct_change_from_clean=(
            "pct_change_from_clean",
            "mean",
        ),

        split_count=(
            "split_seed",
            "nunique",
        ),
    )
)


assert len(
    compound_summary
) == (
    2
    * 4
    * 3
)

assert len(
    compound_summary
) == 24

assert set(
    compound_summary[
        "split_count"
    ]
) == {5}


# ============================================================================
# Persist analysis
# ============================================================================

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


level_split = level_split.sort_values(
    [
        "metric",
        "scenario",
        "domain",
        "model",
        "split_seed",
        "severity_axis",
    ],
    kind="mergesort",
).reset_index(
    drop=True
)


level_summary = level_summary.sort_values(
    [
        "metric",
        "scenario",
        "domain",
        "model",
        "severity_axis",
    ],
    kind="mergesort",
).reset_index(
    drop=True
)


auc_split = auc_split.sort_values(
    [
        "metric",
        "scenario",
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
        "scenario",
        "domain",
    ],
    kind="mergesort",
).reset_index(
    drop=True
)


compound_split = compound_split.sort_values(
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


compound_summary = compound_summary.sort_values(
    [
        "metric",
        "domain",
        "model",
    ],
    kind="mergesort",
).reset_index(
    drop=True
)


atomic_csv(
    level_split,
    LEVEL_SPLIT_CSV,
)

atomic_csv(
    level_summary,
    LEVEL_SUMMARY_CSV,
)

atomic_csv(
    auc_split,
    AUC_SPLIT_CSV,
)

atomic_csv(
    inference,
    INFERENCE_CSV,
)

atomic_csv(
    compound_split,
    COMPOUND_SPLIT_CSV,
)

atomic_csv(
    compound_summary,
    COMPOUND_SUMMARY_CSV,
)

atomic_csv(
    clean_reference_df,
    CLEAN_REFERENCE_CSV,
)


analysis_manifest = {
    "contract":
        "final_physical_sensor_noise_analysis_v1",

    "status":
        "COMPLETE",

    "execution_manifest_sha256":
        EXPECTED_EXECUTION_MANIFEST_SHA,

    "feature_closure_sha256":
        EXPECTED_FEATURE_CLOSURE_SHA,

    "analysis_script_sha256":
        SCRIPT_SHA,

    "analysis_script_commit":
        SCRIPT_COMMIT,

    "value_level_read_started_only_after_analysis_script_commit":
        True,

    "raw_metric_files_read":
        30,

    "raw_metric_rows_read":
        7920,

    "split_count":
        5,

    "train_initializations_nested_per_split":
        3,

    "corruption_seeds_nested_per_split":
        5,

    "primary_families":
        FAMILIES,

    "primary_metrics":
        PRIMARY_METRICS,

    "domains":
        DOMAINS,

    "severity_axis":
        SEVERITY,

    "auc_direction":
        "lower_error_auc_is_better",

    "primary_delta_definition":
        "MLP_AUC_minus_GCN_AUC",

    "positive_primary_delta_interpretation":
        "GCN_lower_error_AUC",

    "inferential_unit":
        "split_seed",

    "inferential_n":
        5,

    "test":
        "exact_two_sided_paired_signflip_of_split_level_auc_differences",

    "multiplicity":
        "Holm_adjustment_across_16_family_by_domain_comparisons_separately_for_RMSE_and_MAE",

    "minimum_attainable_two_sided_signflip_p_at_n5":
        0.0625,

    "compound_medium_inferential":
        False,

    "clean_reference_check":
        "PASS",

    "clean_reference_tolerance":
        CLEAN_REFERENCE_TOL,

    "significant_holm_0p05_count":
        int(
            inference[
                "significant_holm_0p05"
            ].sum()
        ),

    "outputs": {
        "level_by_split": {
            "path":
                str(
                    LEVEL_SPLIT_CSV
                ),

            "sha256":
                sha(
                    LEVEL_SPLIT_CSV
                ),

            "rows":
                int(
                    len(
                        level_split
                    )
                ),
        },

        "level_summary": {
            "path":
                str(
                    LEVEL_SUMMARY_CSV
                ),

            "sha256":
                sha(
                    LEVEL_SUMMARY_CSV
                ),

            "rows":
                int(
                    len(
                        level_summary
                    )
                ),
        },

        "auc_by_split": {
            "path":
                str(
                    AUC_SPLIT_CSV
                ),

            "sha256":
                sha(
                    AUC_SPLIT_CSV
                ),

            "rows":
                int(
                    len(
                        auc_split
                    )
                ),
        },

        "primary_inference": {
            "path":
                str(
                    INFERENCE_CSV
                ),

            "sha256":
                sha(
                    INFERENCE_CSV
                ),

            "rows":
                int(
                    len(
                        inference
                    )
                ),
        },

        "compound_by_split": {
            "path":
                str(
                    COMPOUND_SPLIT_CSV
                ),

            "sha256":
                sha(
                    COMPOUND_SPLIT_CSV
                ),

            "rows":
                int(
                    len(
                        compound_split
                    )
                ),
        },

        "compound_summary": {
            "path":
                str(
                    COMPOUND_SUMMARY_CSV
                ),

            "sha256":
                sha(
                    COMPOUND_SUMMARY_CSV
                ),

            "rows":
                int(
                    len(
                        compound_summary
                    )
                ),
        },

        "clean_reference_check": {
            "path":
                str(
                    CLEAN_REFERENCE_CSV
                ),

            "sha256":
                sha(
                    CLEAN_REFERENCE_CSV
                ),

            "rows":
                int(
                    len(
                        clean_reference_df
                    )
                ),
        },
    },
}


atomic_json(
    analysis_manifest,
    MANIFEST,
)


# ============================================================================
# Compact first result surface
# ============================================================================

def f6(x):
    return f"{float(x):.6f}"


def f2(x):
    return f"{float(x):.2f}"


def fp(x):
    return f"{float(x):.4f}"


print("=" * 88)
print("FINAL PHYSICAL SENSOR NOISE ANALYSIS")
print("=" * 88)

print(
    "clean reference regression barrier:",
    "PASS"
)

print(
    "inferential unit:",
    "split_seed (n=5)"
)

print(
    "nested train initializations per split:",
    3
)

print(
    "nested corruption seeds per split:",
    5
)

print(
    "primary comparisons per metric:",
    16
)

print(
    "minimum attainable two-sided p:",
    "0.0625"
)

print()


# --------------------------------------------------------------------------
# Balanced-all AUC
# --------------------------------------------------------------------------

print("[BALANCED_ALL PRIMARY AUC]")
print(
    "metric family               "
    "MLP_AUC   GCN_AUC   "
    "delta(M-G)  GCN_adv% wins  p_exact p_holm"
)


balanced = inference[
    inference[
        "domain"
    ] == "BALANCED_ALL"
].copy()


for _, r in balanced.iterrows():

    print(
        f"{str(r['metric']):<6} "
        f"{str(r['scenario']):<20} "
        f"{f6(r['mlp_auc_mean']):>9} "
        f"{f6(r['gcn_auc_mean']):>9} "
        f"{f6(r['mean_delta_mlp_minus_gcn']):>10} "
        f"{f2(r['relative_gcn_advantage_pct']):>8} "
        f"{int(r['gcn_favored_splits'])}/5  "
        f"{fp(r['p_exact_signflip']):>7} "
        f"{fp(r['p_holm']):>6}"
    )


# --------------------------------------------------------------------------
# All domains compact paired AUC deltas
# --------------------------------------------------------------------------

print()
print("[DOMAIN PRIMARY AUC DELTAS: MLP AUC - GCN AUC]")
print(
    "family               domain        "
    "RMSE_delta RMSE_adv% RMSE_wins RMSE_Holm "
    "MAE_delta  MAE_adv% MAE_wins MAE_Holm"
)


rm = inference[
    inference[
        "metric"
    ] == "rmse"
][
    [
        "scenario",
        "domain",
        "mean_delta_mlp_minus_gcn",
        "relative_gcn_advantage_pct",
        "gcn_favored_splits",
        "p_holm",
    ]
].rename(
    columns={
        "mean_delta_mlp_minus_gcn":
            "rmse_delta",

        "relative_gcn_advantage_pct":
            "rmse_adv",

        "gcn_favored_splits":
            "rmse_wins",

        "p_holm":
            "rmse_holm",
    }
)


ma = inference[
    inference[
        "metric"
    ] == "mae"
][
    [
        "scenario",
        "domain",
        "mean_delta_mlp_minus_gcn",
        "relative_gcn_advantage_pct",
        "gcn_favored_splits",
        "p_holm",
    ]
].rename(
    columns={
        "mean_delta_mlp_minus_gcn":
            "mae_delta",

        "relative_gcn_advantage_pct":
            "mae_adv",

        "gcn_favored_splits":
            "mae_wins",

        "p_holm":
            "mae_holm",
    }
)


wide = rm.merge(
    ma,
    on=[
        "scenario",
        "domain",
    ],
    validate="one_to_one",
)


family_order = {
    x: i
    for i, x in enumerate(
        FAMILIES
    )
}

domain_order = {
    x: i
    for i, x in enumerate(
        DOMAINS
    )
}


wide[
    "_fo"
] = wide[
    "scenario"
].map(
    family_order
)

wide[
    "_do"
] = wide[
    "domain"
].map(
    domain_order
)


wide = wide.sort_values(
    [
        "_fo",
        "_do",
    ],
    kind="mergesort",
)


for _, r in wide.iterrows():

    print(
        f"{str(r['scenario']):<20} "
        f"{str(r['domain']):<12} "
        f"{f6(r['rmse_delta']):>10} "
        f"{f2(r['rmse_adv']):>9} "
        f"{int(r['rmse_wins'])}/5      "
        f"{fp(r['rmse_holm']):>8} "
        f"{f6(r['mae_delta']):>10} "
        f"{f2(r['mae_adv']):>8} "
        f"{int(r['mae_wins'])}/5     "
        f"{fp(r['mae_holm']):>8}"
    )


# --------------------------------------------------------------------------
# Balanced high severity
# --------------------------------------------------------------------------

print()
print("[BALANCED_ALL HIGH-SEVERITY ERROR]")
print(
    "metric family               "
    "MLP_high   GCN_high   "
    "MLP_%vs_clean GCN_%vs_clean"
)


high = level_summary[
    (
        level_summary[
            "domain"
        ] == "BALANCED_ALL"
    )
    & (
        level_summary[
            "level"
        ] == "high"
    )
    & (
        level_summary[
            "metric"
        ].isin(
            PRIMARY_METRICS
        )
    )
]


for metric in PRIMARY_METRICS:

    for family in FAMILIES:

        g = high[
            (
                high[
                    "metric"
                ] == metric
            )
            & (
                high[
                    "scenario"
                ] == family
            )
        ].set_index(
            "model"
        )


        assert set(
            g.index
        ) == set(MODELS)


        print(
            f"{metric:<6} "
            f"{family:<20} "
            f"{f6(g.loc['MLP','mean_value']):>9} "
            f"{f6(g.loc['GCN','mean_value']):>9} "
            f"{f2(g.loc['MLP','mean_pct_change_from_clean']):>12} "
            f"{f2(g.loc['GCN','mean_pct_change_from_clean']):>13}"
        )


# --------------------------------------------------------------------------
# Compound medium descriptive
# --------------------------------------------------------------------------

print()
print("[BALANCED_ALL COMPOUND-MEDIUM — DESCRIPTIVE ONLY]")
print(
    "metric MLP_value  GCN_value  "
    "MLP_%vs_clean GCN_%vs_clean"
)


for metric in PRIMARY_METRICS:

    g = compound_summary[
        (
            compound_summary[
                "domain"
            ] == "BALANCED_ALL"
        )
        & (
            compound_summary[
                "metric"
            ] == metric
        )
    ].set_index(
        "model"
    )


    print(
        f"{metric:<6} "
        f"{f6(g.loc['MLP','mean_compound_medium_value']):>9} "
        f"{f6(g.loc['GCN','mean_compound_medium_value']):>9} "
        f"{f2(g.loc['MLP','mean_pct_change_from_clean']):>12} "
        f"{f2(g.loc['GCN','mean_pct_change_from_clean']):>13}"
    )


print()
print(
    "Holm-significant primary comparisons:",
    int(
        inference[
            "significant_holm_0p05"
        ].sum()
    ),
    "/ 32"
)

print(
    "analysis manifest SHA256:",
    sha(
        MANIFEST
    )
)

print()
print(
    "FINAL_PHYSICAL_SENSOR_NOISE_ANALYSIS_PASS=True"
)
