from pathlib import Path
from itertools import product
import hashlib
import json
import os
import re
import subprocess

import numpy as np
import pandas as pd


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo" / "battery-gnn"

THIS = (
    REPO
    / "scripts/revision_2026/63_analyze_final_physical_sensor_noise_v2.py"
)

ROOT = (
    BASE
    / "results/revision_2026/physical_sensor_noise/final_physical_v1"
)

V2 = ROOT / "evaluation_v2_domain_contract"
RUNS = V2 / "runs"

EXECUTION_MANIFEST = (
    V2
    / "physical_noise_execution_manifest_v2.json"
)

FINAL_V3 = (
    BASE
    / "results/revision_2026/feature_matched_soh/final_v3"
)

OUT = ROOT / "analysis_v2_domain_contract"

CLEAN_CHECK = OUT / "clean_final_v3_runlevel_check.csv"
LEVEL_SPLIT = OUT / "physical_noise_level_by_split.csv"
LEVEL_SUMMARY = OUT / "physical_noise_level_summary.csv"
AUC_SPLIT = OUT / "physical_noise_auc_by_split.csv"
INFERENCE = OUT / "physical_noise_primary_inference.csv"
COMPOUND_SPLIT = OUT / "physical_noise_compound_medium_by_split.csv"
COMPOUND_SUMMARY = OUT / "physical_noise_compound_medium_summary.csv"
MANIFEST = OUT / "physical_noise_analysis_v2_manifest.json"


EXPECTED_EXECUTION_SHA = (
    "cdc02692291041e3fc3e2ef192af857fa7e95c146b7d91dcb7a43bb5e268a127"
)

CLEAN_RUN_TOL = 1e-7

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

MODELS = ["MLP", "GCN"]
METRICS = ["rmse", "mae"]
SPLITS = [0, 1, 2, 3, 4]
TRAIN_SEEDS = [0, 1, 2]

LEVELS = ["clean", "low", "medium", "high"]

SEVERITY = {
    "clean": 0.0,
    "low": 1.0 / 3.0,
    "medium": 2.0 / 3.0,
    "high": 1.0,
}


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


def signflip_p(deltas):
    d = np.asarray(
        deltas,
        dtype=np.float64,
    )
    assert d.shape == (5,)
    assert np.isfinite(d).all()

    observed = abs(float(np.mean(d)))

    stats = []
    for signs in product(
        [-1.0, 1.0],
        repeat=5,
    ):
        s = np.asarray(
            signs,
            dtype=np.float64,
        )
        stats.append(
            abs(float(np.mean(s * d)))
        )

    stats = np.asarray(
        stats,
        dtype=np.float64,
    )

    return float(
        np.mean(
            stats >= observed - 1e-15
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
                (len(p) - rank)
                * p[idx]
            ),
        )

        running = max(
            running,
            candidate,
        )

        out[idx] = running

    return out


def auc4(y):
    y = np.asarray(
        y,
        dtype=np.float64,
    )

    assert y.shape == (4,)

    x = np.asarray(
        [
            0.0,
            1.0 / 3.0,
            2.0 / 3.0,
            1.0,
        ],
        dtype=np.float64,
    )

    return float(
        np.sum(
            (x[1:] - x[:-1])
            * (y[1:] + y[:-1])
            / 2.0
        )
    )


# ===========================================================================
# PRE-VALUE-READ GUARDS
# ===========================================================================

assert sha(
    EXECUTION_MANIFEST
) == EXPECTED_EXECUTION_SHA


execution = json.loads(
    EXECUTION_MANIFEST.read_text(
        encoding="utf-8"
    )
)


assert execution[
    "contract"
] == "final_physical_sensor_noise_execution_v2"

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
    "dataset_metric_contract"
] == "exact_final_v3_unweighted_row_level_dataset_metrics"


# The analysis script itself must already be committed and unchanged.
tracked = subprocess.run(
    [
        "git",
        "ls-files",
        "--error-unmatch",
        str(
            THIS.relative_to(REPO)
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
            THIS.relative_to(REPO)
        ),
    ],
    cwd=REPO,
)

assert unchanged.returncode == 0


SCRIPT_SHA = sha(THIS)

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

assert len(metric_files) == 30


final_v3_files = sorted(
    FINAL_V3.glob(
        "split*_train*/metrics.csv"
    )
)

assert len(final_v3_files) == 15


# ===========================================================================
# FIRST V2 VALUE-LEVEL READ OCCURS BELOW THIS LINE
# ===========================================================================

frames = []

for p in metric_files:
    df = pd.read_csv(
        p,
        low_memory=False,
    )

    assert len(df) == 264

    frames.append(df)


raw = pd.concat(
    frames,
    ignore_index=True,
)

assert len(raw) == 7920


for c in METRICS:
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
    raw["split_seed"].astype(int)
) == set(SPLITS)

assert set(
    raw["train_seed"].astype(int)
) == set(TRAIN_SEEDS)

assert set(
    raw["model"].astype(str)
) == set(MODELS)

assert set(
    raw["domain"].astype(str)
) == set(DOMAINS)


# ===========================================================================
# CLEAN CONDITION: EXACT RUN-LEVEL FINAL-V3 REGRESSION BARRIER
# ===========================================================================

clean_new = raw[
    raw[
        "scenario"
    ].astype(str) == "clean"
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


assert len(clean_new) == 120


old_frames = []


for p in final_v3_files:

    m = re.search(
        r"split(\d+)_train(\d+)",
        str(p),
    )

    assert m is not None

    split_seed = int(m.group(1))
    train_seed = int(m.group(2))

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

    assert len(df) == 8


    def norm_model(x):
        s = str(x).strip().upper()

        if "GCN" in s:
            return "GCN"

        if "MLP" in s:
            return "MLP"

        raise AssertionError(
            ("unknown model", x)
        )


    def norm_domain(x):
        s = str(x).strip().upper()

        mapping = {
            "BALANCED_ALL": "BALANCED_ALL",
            "BALANCED": "BALANCED_ALL",
            "ALL": "BALANCED_ALL",
            "HUST": "HUST",
            "CALCE": "CALCE",
            "MATR": "MATR",
        }

        assert s in mapping, s

        return mapping[s]


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
                    norm_model
                ),

            "domain":
                df[
                    "dataset"
                ].map(
                    norm_domain
                ),

            "n_old":
                pd.to_numeric(
                    df["n"],
                    errors="raise",
                ).astype(int),

            "rmse_old":
                pd.to_numeric(
                    df["RMSE"],
                    errors="raise",
                ),

            "mae_old":
                pd.to_numeric(
                    df["MAE"],
                    errors="raise",
                ),
        })
    )


clean_old = pd.concat(
    old_frames,
    ignore_index=True,
)

assert len(clean_old) == 120


key = [
    "split_seed",
    "train_seed",
    "model",
    "domain",
]


assert not clean_new.duplicated(
    key
).any()

assert not clean_old.duplicated(
    key
).any()


clean_check = clean_new.merge(
    clean_old,
    on=key,
    how="inner",
    validate="one_to_one",
)


assert len(clean_check) == 120


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
    "rmse_difference"
] = (
    clean_check["rmse"]
    - clean_check["rmse_old"]
)


clean_check[
    "mae_difference"
] = (
    clean_check["mae"]
    - clean_check["mae_old"]
)


clean_check[
    "rmse_abs_difference"
] = np.abs(
    clean_check[
        "rmse_difference"
    ]
)


clean_check[
    "mae_abs_difference"
] = np.abs(
    clean_check[
        "mae_difference"
    ]
)


clean_check[
    "rmse_pass"
] = (
    clean_check[
        "rmse_abs_difference"
    ]
    <= CLEAN_RUN_TOL
)


clean_check[
    "mae_pass"
] = (
    clean_check[
        "mae_abs_difference"
    ]
    <= CLEAN_RUN_TOL
)


clean_check[
    "pass"
] = (
    clean_check[
        "n_match"
    ]
    & clean_check[
        "rmse_pass"
    ]
    & clean_check[
        "mae_pass"
    ]
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
# LONG PRIMARY-METRIC TABLE
# ===========================================================================

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
    value_vars=METRICS,
    var_name="metric",
    value_name="value",
)


# ===========================================================================
# CLEAN BASELINE WITHIN SPLIT
# ===========================================================================

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


# ===========================================================================
# FOUR PRIMARY CORRUPTION FAMILIES
# ===========================================================================

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


# 3 train initializations x 5 corruption seeds.
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


primary_split = primary_split.merge(
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


primary_split[
    "severity_axis"
] = primary_split[
    "level"
].map(
    SEVERITY
).astype(float)


primary_split[
    "change_from_clean"
] = (
    primary_split[
        "value"
    ]
    - primary_split[
        "clean_value"
    ]
)


primary_split[
    "pct_change_from_clean"
] = (
    100.0
    * primary_split[
        "change_from_clean"
    ]
    / primary_split[
        "clean_value"
    ]
)


level_parts = []


for family in FAMILIES:

    c = clean_split.copy()

    c["scenario"] = family
    c["level"] = "clean"
    c["severity_axis"] = 0.0
    c["value"] = c["clean_value"]
    c["change_from_clean"] = 0.0
    c["pct_change_from_clean"] = 0.0

    level_parts.append(
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


level_parts.append(
    primary_split[
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
    level_parts,
    ignore_index=True,
)


assert len(level_split) == 1280


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
        mean_value=("value", "mean"),
        sd_across_splits=("value", "std"),
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


assert len(level_summary) == 256
assert set(
    level_summary[
        "split_count"
    ]
) == {5}


# ===========================================================================
# NORMALIZED SEVERITY AUC
# ===========================================================================

auc_rows = []


for split_seed in SPLITS:
    for model in MODELS:
        for family in FAMILIES:
            for domain in DOMAINS:
                for metric in METRICS:

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


                    assert len(g) == 4


                    order = {
                        "clean": 0,
                        "low": 1,
                        "medium": 2,
                        "high": 3,
                    }


                    g[
                        "_order"
                    ] = g[
                        "level"
                    ].map(order)


                    g = g.sort_values(
                        "_order",
                        kind="mergesort",
                    )


                    y = g[
                        "value"
                    ].to_numpy(
                        dtype=np.float64
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
                            auc4(y),

                        "clean":
                            float(y[0]),

                        "low":
                            float(y[1]),

                        "medium":
                            float(y[2]),

                        "high":
                            float(y[3]),
                    })


auc_split = pd.DataFrame(
    auc_rows
)


assert len(auc_split) == 320


# ===========================================================================
# SPLIT-LEVEL PAIRED INFERENCE
# ===========================================================================

infer_rows = []


for metric in METRICS:
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
            ).loc[SPLITS]


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


            delta = mlp - gcn

            p_exact = signflip_p(
                delta
            )


            mlp_mean = float(
                np.mean(mlp)
            )

            gcn_mean = float(
                np.mean(gcn)
            )

            mean_delta = float(
                np.mean(delta)
            )


            infer_rows.append({
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
                    100.0
                    * mean_delta
                    / mlp_mean,

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
    infer_rows
)


assert len(inference) == 32


inference["p_holm"] = np.nan


for metric in METRICS:

    idx = inference.index[
        inference[
            "metric"
        ] == metric
    ]

    assert len(idx) == 16

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


# ===========================================================================
# COMPOUND MEDIUM — DESCRIPTIVE ONLY
# ===========================================================================

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
] = (
    100.0
    * compound_split[
        "change_from_clean"
    ]
    / compound_split[
        "clean_value"
    ]
)


assert len(compound_split) == 80


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


assert len(compound_summary) == 16
assert set(
    compound_summary[
        "split_count"
    ]
) == {5}


# ===========================================================================
# WRITE IMMUTABLE ANALYSIS OUTPUTS
# ===========================================================================

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


clean_check = clean_check.sort_values(
    key,
    kind="mergesort",
).reset_index(drop=True)


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
).reset_index(drop=True)


level_summary = level_summary.sort_values(
    [
        "metric",
        "scenario",
        "domain",
        "model",
        "severity_axis",
    ],
    kind="mergesort",
).reset_index(drop=True)


auc_split = auc_split.sort_values(
    [
        "metric",
        "scenario",
        "domain",
        "model",
        "split_seed",
    ],
    kind="mergesort",
).reset_index(drop=True)


inference = inference.sort_values(
    [
        "metric",
        "scenario",
        "domain",
    ],
    kind="mergesort",
).reset_index(drop=True)


compound_split = compound_split.sort_values(
    [
        "metric",
        "domain",
        "model",
        "split_seed",
    ],
    kind="mergesort",
).reset_index(drop=True)


compound_summary = compound_summary.sort_values(
    [
        "metric",
        "domain",
        "model",
    ],
    kind="mergesort",
).reset_index(drop=True)


atomic_csv(
    clean_check,
    CLEAN_CHECK,
)

atomic_csv(
    level_split,
    LEVEL_SPLIT,
)

atomic_csv(
    level_summary,
    LEVEL_SUMMARY,
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
    compound_split,
    COMPOUND_SPLIT,
)

atomic_csv(
    compound_summary,
    COMPOUND_SUMMARY,
)


manifest = {
    "contract":
        "final_physical_sensor_noise_analysis_v2",

    "status":
        "COMPLETE",

    "execution_manifest_sha256":
        EXPECTED_EXECUTION_SHA,

    "analysis_script_sha256":
        SCRIPT_SHA,

    "analysis_script_commit":
        SCRIPT_COMMIT,

    "value_level_read_started_only_after_analysis_script_commit":
        True,

    "source_execution_version":
        "v2_domain_metric_contract",

    "v1_metric_analysis_used":
        False,

    "raw_metric_files_read":
        30,

    "raw_metric_rows_read":
        7920,

    "primary_metrics":
        METRICS,

    "primary_families":
        FAMILIES,

    "domains":
        DOMAINS,

    "split_count":
        5,

    "train_initializations_nested_per_split":
        3,

    "corruption_seeds_nested_per_split":
        5,

    "inferential_unit":
        "split_seed",

    "inferential_n":
        5,

    "severity_axis":
        SEVERITY,

    "auc_direction":
        "lower_error_auc_is_better",

    "primary_delta_definition":
        "MLP_AUC_minus_GCN_AUC",

    "positive_primary_delta_interpretation":
        "GCN_lower_error_AUC",

    "test":
        "exact_two_sided_paired_signflip_of_split_level_auc_differences",

    "multiplicity":
        "Holm_across_16_family_by_domain_comparisons_separately_for_RMSE_and_MAE",

    "minimum_attainable_two_sided_signflip_p_at_n5":
        0.0625,

    "compound_medium_inferential":
        False,

    "clean_final_v3_runlevel_barrier":
        "PASS",

    "clean_run_tolerance":
        CLEAN_RUN_TOL,

    "clean_run_count":
        120,

    "clean_max_abs_rmse_difference":
        max_clean_rmse_diff,

    "clean_max_abs_mae_difference":
        max_clean_mae_diff,

    "significant_holm_0p05_count":
        int(
            inference[
                "significant_holm_0p05"
            ].sum()
        ),

    "outputs": {
        "clean_check": {
            "path": str(CLEAN_CHECK),
            "sha256": sha(CLEAN_CHECK),
            "rows": len(clean_check),
        },

        "level_by_split": {
            "path": str(LEVEL_SPLIT),
            "sha256": sha(LEVEL_SPLIT),
            "rows": len(level_split),
        },

        "level_summary": {
            "path": str(LEVEL_SUMMARY),
            "sha256": sha(LEVEL_SUMMARY),
            "rows": len(level_summary),
        },

        "auc_by_split": {
            "path": str(AUC_SPLIT),
            "sha256": sha(AUC_SPLIT),
            "rows": len(auc_split),
        },

        "primary_inference": {
            "path": str(INFERENCE),
            "sha256": sha(INFERENCE),
            "rows": len(inference),
        },

        "compound_by_split": {
            "path": str(COMPOUND_SPLIT),
            "sha256": sha(COMPOUND_SPLIT),
            "rows": len(compound_split),
        },

        "compound_summary": {
            "path": str(COMPOUND_SUMMARY),
            "sha256": sha(COMPOUND_SUMMARY),
            "rows": len(compound_summary),
        },
    },
}


atomic_json(
    manifest,
    MANIFEST,
)


# ===========================================================================
# COMPACT FIRST VALUE SURFACE
# ===========================================================================

def f6(x):
    return f"{float(x):.6f}"


def f2(x):
    return f"{float(x):.2f}"


def fp(x):
    return f"{float(x):.4f}"


print("=" * 94)
print("FINAL PHYSICAL SENSOR NOISE ANALYSIS V2")
print("=" * 94)

print(
    "clean final-v3 run-level barrier: PASS"
)

print(
    "clean run pairs: 120"
)

print(
    "clean max abs RMSE difference:",
    f"{max_clean_rmse_diff:.3e}"
)

print(
    "clean max abs MAE difference:",
    f"{max_clean_mae_diff:.3e}"
)

print(
    "inferential unit: split_seed (n=5)"
)

print(
    "nested train initializations: 3"
)

print(
    "nested corruption seeds: 5"
)

print(
    "minimum attainable two-sided p: 0.0625"
)

print()


print("[BALANCED_ALL PRIMARY AUC]")

print(
    "metric family               "
    "MLP_AUC   GCN_AUC   delta(M-G) GCN_adv% wins p_exact p_holm"
)


b = inference[
    inference[
        "domain"
    ] == "BALANCED_ALL"
]


for _, r in b.iterrows():

    print(
        f"{r['metric']:<6} "
        f"{r['scenario']:<20} "
        f"{f6(r['mlp_auc_mean']):>9} "
        f"{f6(r['gcn_auc_mean']):>9} "
        f"{f6(r['mean_delta_mlp_minus_gcn']):>10} "
        f"{f2(r['relative_gcn_advantage_pct']):>8} "
        f"{int(r['gcn_favored_splits'])}/5 "
        f"{fp(r['p_exact_signflip']):>7} "
        f"{fp(r['p_holm']):>6}"
    )


print()
print("[DOMAIN PRIMARY AUC DELTAS: MLP AUC - GCN AUC]")

print(
    "family               domain        "
    "RMSE_delta RMSE_adv% wins Holm   "
    "MAE_delta MAE_adv% wins Holm"
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
            "rd",

        "relative_gcn_advantage_pct":
            "ra",

        "gcn_favored_splits":
            "rw",

        "p_holm":
            "rh",
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
            "md",

        "relative_gcn_advantage_pct":
            "ma",

        "gcn_favored_splits":
            "mw",

        "p_holm":
            "mh",
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


fo = {
    x: i
    for i, x in enumerate(FAMILIES)
}

do = {
    x: i
    for i, x in enumerate(DOMAINS)
}


wide["_f"] = wide["scenario"].map(fo)
wide["_d"] = wide["domain"].map(do)


wide = wide.sort_values(
    ["_f", "_d"],
    kind="mergesort",
)


for _, r in wide.iterrows():

    print(
        f"{r['scenario']:<20} "
        f"{r['domain']:<12} "
        f"{f6(r['rd']):>10} "
        f"{f2(r['ra']):>9} "
        f"{int(r['rw'])}/5 "
        f"{fp(r['rh']):>6}   "
        f"{f6(r['md']):>10} "
        f"{f2(r['ma']):>8} "
        f"{int(r['mw'])}/5 "
        f"{fp(r['mh']):>6}"
    )


print()
print("[BALANCED_ALL HIGH-SEVERITY ERROR]")

print(
    "metric family               "
    "MLP_high  GCN_high  MLP_%clean GCN_%clean"
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
]


for metric in METRICS:
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
        ].set_index("model")

        print(
            f"{metric:<6} "
            f"{family:<20} "
            f"{f6(g.loc['MLP','mean_value']):>9} "
            f"{f6(g.loc['GCN','mean_value']):>9} "
            f"{f2(g.loc['MLP','mean_pct_change_from_clean']):>10} "
            f"{f2(g.loc['GCN','mean_pct_change_from_clean']):>10}"
        )


print()
print("[BALANCED_ALL COMPOUND-MEDIUM — DESCRIPTIVE ONLY]")

for metric in METRICS:

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
    ].set_index("model")

    print(
        f"{metric}: "
        f"MLP={f6(g.loc['MLP','mean_compound_medium_value'])} "
        f"GCN={f6(g.loc['GCN','mean_compound_medium_value'])} "
        f"MLP_change={f2(g.loc['MLP','mean_pct_change_from_clean'])}% "
        f"GCN_change={f2(g.loc['GCN','mean_pct_change_from_clean'])}%"
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
    sha(MANIFEST)
)

print()
print(
    "FINAL_PHYSICAL_SENSOR_NOISE_ANALYSIS_V2_PASS=True"
)
