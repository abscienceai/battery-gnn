from pathlib import Path
import hashlib
import itertools
import json
import math
import os

import numpy as np
import pandas as pd


BASE = Path(
    os.environ.get(
        "BATTERY_REVISION_BASE",
        Path.home()
        / "alper"
        / "battery_revision_2026",
    )
)

ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "feature_matched_soh"
)

V3 = (
    ROOT
    / "final_v3"
)

PROTOCOL = (
    ROOT
    / "full_clean_benchmark_protocol_v3.json"
)

MANIFEST = (
    V3
    / "final_v3_execution_manifest.json"
)

OUT = (
    V3
    / "analysis"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


EXPECTED_PROTOCOL_SHA = (
    "2dec14c210ae3f02e5f65e4acc24f9c4adc747bc071509cb144e69dba707f4dd"
)

EXPECTED_MANIFEST_SHA = (
    "5a8bfbe0f11a6158f2749a9da4f18f1e62adf8c839976c1bf0af9c9f6c07440d"
)


DATASETS = [
    "BALANCED_ALL",
    "HUST",
    "CALCE",
    "MATR",
]

MODELS = [
    "MLP",
    "CAUSAL_GCN",
]

PLANNED = [
    (s, t)
    for s in range(5)
    for t in range(3)
]


EXPECTED_TEST_N = {
    0: {
        "BALANCED_ALL": 43974,
        "HUST": 20766,
        "CALCE": 1685,
        "MATR": 21523,
    },
    1: {
        "BALANCED_ALL": 46273,
        "HUST": 22297,
        "CALCE": 3147,
        "MATR": 20829,
    },
    2: {
        "BALANCED_ALL": 43818,
        "HUST": 21373,
        "CALCE": 1535,
        "MATR": 20910,
    },
    3: {
        "BALANCED_ALL": 46156,
        "HUST": 21835,
        "CALCE": 2286,
        "MATR": 22035,
    },
    4: {
        "BALANCED_ALL": 43749,
        "HUST": 19164,
        "CALCE": 3147,
        "MATR": 21438,
    },
}


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def exact_signflip_p(diff):
    """
    Exact two-sided paired sign-flip randomization test.

    Higher-level statistical unit = split_seed.
    n = 5 split-level paired differences.

    Statistic = abs(mean paired difference).
    """
    d = np.asarray(
        diff,
        dtype=float,
    )

    assert d.shape == (5,)
    assert np.isfinite(d).all()

    observed = abs(
        float(np.mean(d))
    )

    permuted = []

    for signs in itertools.product(
        (-1.0, 1.0),
        repeat=len(d),
    ):
        s = np.asarray(
            signs,
            dtype=float,
        )

        permuted.append(
            abs(
                float(
                    np.mean(
                        s * d
                    )
                )
            )
        )

    permuted = np.asarray(
        permuted,
        dtype=float,
    )

    p = float(
        np.mean(
            permuted
            >= (
                observed
                - 1e-15
            )
        )
    )

    return p


def holm_adjust(p_values):
    """
    Holm family-wise error correction.
    """
    p = np.asarray(
        p_values,
        dtype=float,
    )

    m = len(p)

    order = np.argsort(p)

    adjusted = np.empty(
        m,
        dtype=float,
    )

    running = 0.0

    for rank, idx in enumerate(
        order
    ):
        candidate = (
            (m - rank)
            * p[idx]
        )

        running = max(
            running,
            candidate,
        )

        adjusted[idx] = min(
            1.0,
            running,
        )

    return adjusted


# =============================================================================
# Frozen-final guards
# =============================================================================

assert sha256_file(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

assert sha256_file(
    MANIFEST
) == EXPECTED_MANIFEST_SHA


protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)

manifest = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)


assert protocol[
    "status"
] == "FINAL_FROZEN_BEFORE_V3_RUNS"

assert protocol[
    "test_used_for_ceiling_selection"
] is False

assert protocol[
    "v2_exclusion_from_final_aggregation"
] is True

assert manifest[
    "completed_pair_count"
] == 15

assert manifest[
    "completed_model_fit_count"
] == 30

assert manifest[
    "all_fits_patience_valid"
] is True

assert manifest[
    "test_metric_values_surfaced_during_series"
] is False


# =============================================================================
# FIRST READ of final-v3 test metric values
# =============================================================================

frames = []

metrics_hashes = {}


for split_seed, train_seed in PLANNED:

    run = (
        V3
        / f"split{split_seed}_train{train_seed}"
    )

    metrics_path = (
        run
        / "metrics.csv"
    )

    assert metrics_path.is_file()

    metrics_hashes[
        f"split{split_seed}_train{train_seed}"
    ] = sha256_file(
        metrics_path
    )


    m = pd.read_csv(
        metrics_path,
        low_memory=False,
    )


    assert len(m) == 24

    assert set(
        m["model"]
    ) == set(MODELS)

    assert set(
        m["split"]
    ) == {
        "train",
        "validation",
        "test",
    }

    assert set(
        m["dataset"]
    ) == set(DATASETS)

    assert set(
        m["split_seed"]
    ) == {
        split_seed
    }

    assert set(
        m["train_seed"]
    ) == {
        train_seed
    }


    test = (
        m[
            m["split"]
            == "test"
        ]
        .copy()
    )


    assert len(test) == 8

    assert not test.duplicated(
        [
            "model",
            "dataset",
        ]
    ).any()


    for dataset in DATASETS:

        d = test[
            test[
                "dataset"
            ] == dataset
        ]

        assert len(d) == 2

        assert set(
            d["n"].astype(int)
        ) == {
            EXPECTED_TEST_N[
                split_seed
            ][
                dataset
            ]
        }


    assert np.isfinite(
        test["RMSE"]
    ).all()

    assert np.isfinite(
        test["MAE"]
    ).all()


    non_balanced = test[
        test[
            "dataset"
        ] != "BALANCED_ALL"
    ]

    assert np.isfinite(
        non_balanced["R2"]
    ).all()


    test[
        "source_metrics_sha256"
    ] = metrics_hashes[
        f"split{split_seed}_train{train_seed}"
    ]


    frames.append(
        test
    )


all_test = pd.concat(
    frames,
    ignore_index=True,
)


assert len(
    all_test
) == 120


all_test.to_csv(
    OUT
    / "test_metrics_all_15_pairs.csv",
    index=False,
    float_format="%.12g",
)


# =============================================================================
# Level 1 aggregation:
# average over THREE initialization seeds WITHIN each split.
#
# Three train seeds are NOT treated as 15 independent splits.
# =============================================================================

agg_spec = {
    "RMSE": [
        "mean",
        "std",
    ],
    "MAE": [
        "mean",
        "std",
    ],
    "R2": [
        "mean",
        "std",
    ],
}


split_level = (
    all_test
    .groupby(
        [
            "split_seed",
            "model",
            "dataset",
        ],
        as_index=False,
    )
    .agg(
        RMSE_mean=(
            "RMSE",
            "mean",
        ),
        RMSE_seed_sd=(
            "RMSE",
            "std",
        ),
        MAE_mean=(
            "MAE",
            "mean",
        ),
        MAE_seed_sd=(
            "MAE",
            "std",
        ),
        R2_mean=(
            "R2",
            "mean",
        ),
        R2_seed_sd=(
            "R2",
            "std",
        ),
        n=(
            "n",
            "first",
        ),
        initialization_count=(
            "train_seed",
            "nunique",
        ),
    )
)


assert len(
    split_level
) == 40

assert set(
    split_level[
        "initialization_count"
    ]
) == {
    3
}


split_level.to_csv(
    OUT
    / "split_level_mean_over_initializations.csv",
    index=False,
    float_format="%.12g",
)


# =============================================================================
# Paired MLP vs GCN differences at split level
#
# ERROR metrics:
#   diff = GCN - MLP
#   negative = GCN lower/better
#
# R2:
#   diff = GCN - MLP
#   positive = GCN higher/better
# =============================================================================

paired_rows = []


for split_seed in range(5):

    for dataset in DATASETS:

        d = split_level[
            (
                split_level[
                    "split_seed"
                ] == split_seed
            )
            &
            (
                split_level[
                    "dataset"
                ] == dataset
            )
        ]


        assert len(d) == 2


        mlp = d[
            d[
                "model"
            ] == "MLP"
        ].iloc[0]

        gcn = d[
            d[
                "model"
            ] == "CAUSAL_GCN"
        ].iloc[0]


        paired_rows.append({
            "split_seed":
                split_seed,

            "dataset":
                dataset,

            "RMSE_MLP":
                float(
                    mlp[
                        "RMSE_mean"
                    ]
                ),

            "RMSE_GCN":
                float(
                    gcn[
                        "RMSE_mean"
                    ]
                ),

            "RMSE_diff_GCN_minus_MLP":
                float(
                    gcn[
                        "RMSE_mean"
                    ]
                    - mlp[
                        "RMSE_mean"
                    ]
                ),

            "MAE_MLP":
                float(
                    mlp[
                        "MAE_mean"
                    ]
                ),

            "MAE_GCN":
                float(
                    gcn[
                        "MAE_mean"
                    ]
                ),

            "MAE_diff_GCN_minus_MLP":
                float(
                    gcn[
                        "MAE_mean"
                    ]
                    - mlp[
                        "MAE_mean"
                    ]
                ),

            "R2_MLP":
                float(
                    mlp[
                        "R2_mean"
                    ]
                )
                if np.isfinite(
                    mlp[
                        "R2_mean"
                    ]
                )
                else np.nan,

            "R2_GCN":
                float(
                    gcn[
                        "R2_mean"
                    ]
                )
                if np.isfinite(
                    gcn[
                        "R2_mean"
                    ]
                )
                else np.nan,

            "R2_diff_GCN_minus_MLP":
                (
                    float(
                        gcn[
                            "R2_mean"
                        ]
                        - mlp[
                            "R2_mean"
                        ]
                    )
                    if (
                        np.isfinite(
                            gcn[
                                "R2_mean"
                            ]
                        )
                        and np.isfinite(
                            mlp[
                                "R2_mean"
                            ]
                        )
                    )
                    else np.nan
                ),
        })


paired = pd.DataFrame(
    paired_rows
)


assert len(
    paired
) == 20


paired.to_csv(
    OUT
    / "split_level_paired_differences.csv",
    index=False,
    float_format="%.12g",
)


# =============================================================================
# Cross-split summary: primary statistical unit n=5 splits
# =============================================================================

summary_rows = []


for dataset in DATASETS:

    d = paired[
        paired[
            "dataset"
        ] == dataset
    ].copy()

    assert len(d) == 5


    rmse_diff = d[
        "RMSE_diff_GCN_minus_MLP"
    ].to_numpy(
        dtype=float
    )

    mae_diff = d[
        "MAE_diff_GCN_minus_MLP"
    ].to_numpy(
        dtype=float
    )


    rmse_mlp = float(
        d[
            "RMSE_MLP"
        ].mean()
    )

    rmse_gcn = float(
        d[
            "RMSE_GCN"
        ].mean()
    )

    mae_mlp = float(
        d[
            "MAE_MLP"
        ].mean()
    )

    mae_gcn = float(
        d[
            "MAE_GCN"
        ].mean()
    )


    summary_rows.append({
        "dataset":
            dataset,

        "n_split_units":
            5,

        "initializations_per_split":
            3,

        "RMSE_MLP_mean":
            rmse_mlp,

        "RMSE_MLP_sd_across_splits":
            float(
                d[
                    "RMSE_MLP"
                ].std(
                    ddof=1
                )
            ),

        "RMSE_GCN_mean":
            rmse_gcn,

        "RMSE_GCN_sd_across_splits":
            float(
                d[
                    "RMSE_GCN"
                ].std(
                    ddof=1
                )
            ),

        "RMSE_diff_mean_GCN_minus_MLP":
            float(
                np.mean(
                    rmse_diff
                )
            ),

        "RMSE_diff_median_GCN_minus_MLP":
            float(
                np.median(
                    rmse_diff
                )
            ),

        "RMSE_diff_sd":
            float(
                np.std(
                    rmse_diff,
                    ddof=1,
                )
            ),

        "RMSE_relative_change_percent":
            float(
                100.0
                * (
                    rmse_gcn
                    - rmse_mlp
                )
                / rmse_mlp
            ),

        "RMSE_GCN_wins":
            int(
                np.sum(
                    rmse_diff < 0
                )
            ),

        "RMSE_MLP_wins":
            int(
                np.sum(
                    rmse_diff > 0
                )
            ),

        "RMSE_ties":
            int(
                np.sum(
                    rmse_diff == 0
                )
            ),

        "RMSE_exact_signflip_p":
            exact_signflip_p(
                rmse_diff
            ),

        "MAE_MLP_mean":
            mae_mlp,

        "MAE_MLP_sd_across_splits":
            float(
                d[
                    "MAE_MLP"
                ].std(
                    ddof=1
                )
            ),

        "MAE_GCN_mean":
            mae_gcn,

        "MAE_GCN_sd_across_splits":
            float(
                d[
                    "MAE_GCN"
                ].std(
                    ddof=1
                )
            ),

        "MAE_diff_mean_GCN_minus_MLP":
            float(
                np.mean(
                    mae_diff
                )
            ),

        "MAE_diff_median_GCN_minus_MLP":
            float(
                np.median(
                    mae_diff
                )
            ),

        "MAE_diff_sd":
            float(
                np.std(
                    mae_diff,
                    ddof=1,
                )
            ),

        "MAE_relative_change_percent":
            float(
                100.0
                * (
                    mae_gcn
                    - mae_mlp
                )
                / mae_mlp
            ),

        "MAE_GCN_wins":
            int(
                np.sum(
                    mae_diff < 0
                )
            ),

        "MAE_MLP_wins":
            int(
                np.sum(
                    mae_diff > 0
                )
            ),

        "MAE_ties":
            int(
                np.sum(
                    mae_diff == 0
                )
            ),

        "MAE_exact_signflip_p":
            exact_signflip_p(
                mae_diff
            ),
    })


summary = pd.DataFrame(
    summary_rows
)


# Holm correction separately within RMSE and MAE families.
summary[
    "RMSE_Holm_p_4_domains"
] = holm_adjust(
    summary[
        "RMSE_exact_signflip_p"
    ].to_numpy(
        dtype=float
    )
)

summary[
    "MAE_Holm_p_4_domains"
] = holm_adjust(
    summary[
        "MAE_exact_signflip_p"
    ].to_numpy(
        dtype=float
    )
)


# R2 is descriptive only for physical datasets.
for dataset in [
    "HUST",
    "CALCE",
    "MATR",
]:

    d = paired[
        paired[
            "dataset"
        ] == dataset
    ]

    idx = summary[
        "dataset"
    ] == dataset

    summary.loc[
        idx,
        "R2_MLP_mean"
    ] = float(
        d[
            "R2_MLP"
        ].mean()
    )

    summary.loc[
        idx,
        "R2_GCN_mean"
    ] = float(
        d[
            "R2_GCN"
        ].mean()
    )

    summary.loc[
        idx,
        "R2_diff_mean_GCN_minus_MLP"
    ] = float(
        d[
            "R2_diff_GCN_minus_MLP"
        ].mean()
    )


summary.to_csv(
    OUT
    / "cross_split_primary_summary.csv",
    index=False,
    float_format="%.12g",
)


# =============================================================================
# Initialization sensitivity descriptive table
# =============================================================================

seed_stability = (
    split_level
    .groupby(
        [
            "model",
            "dataset",
        ],
        as_index=False,
    )
    .agg(
        mean_within_split_RMSE_seed_sd=(
            "RMSE_seed_sd",
            "mean",
        ),
        max_within_split_RMSE_seed_sd=(
            "RMSE_seed_sd",
            "max",
        ),
        mean_within_split_MAE_seed_sd=(
            "MAE_seed_sd",
            "mean",
        ),
        max_within_split_MAE_seed_sd=(
            "MAE_seed_sd",
            "max",
        ),
    )
)


seed_stability.to_csv(
    OUT
    / "initialization_seed_stability.csv",
    index=False,
    float_format="%.12g",
)


# =============================================================================
# Analysis manifest
# =============================================================================

analysis_manifest = {
    "analysis_contract":
        "final_v3_first_test_unlock_v1",

    "protocol_sha256":
        EXPECTED_PROTOCOL_SHA,

    "execution_manifest_sha256":
        EXPECTED_MANIFEST_SHA,

    "test_values_first_read_after_execution_audit":
        True,

    "source_result_root":
        str(V3),

    "v1_v2_metrics_included":
        False,

    "paired_run_count":
        15,

    "higher_level_split_units":
        5,

    "initialization_seeds_per_split":
        3,

    "primary_replication_policy":
        (
            "Average three initialization seeds within each "
            "split_seed, then perform paired comparison over "
            "the five split-level units."
        ),

    "error_difference_definition":
        "CAUSAL_GCN minus MLP; negative favors CAUSAL_GCN",

    "primary_test":
        (
            "exact two-sided paired sign-flip test over five "
            "split-level differences"
        ),

    "multiple_testing":
        (
            "Holm adjustment separately across four RMSE domains "
            "and four MAE domains"
        ),

    "metrics_sha256":
        metrics_hashes,

    "outputs": {
        "all_test":
            sha256_file(
                OUT
                / "test_metrics_all_15_pairs.csv"
            ),

        "split_level":
            sha256_file(
                OUT
                / "split_level_mean_over_initializations.csv"
            ),

        "paired":
            sha256_file(
                OUT
                / "split_level_paired_differences.csv"
            ),

        "summary":
            sha256_file(
                OUT
                / "cross_split_primary_summary.csv"
            ),

        "seed_stability":
            sha256_file(
                OUT
                / "initialization_seed_stability.csv"
            ),
    },
}


manifest_path = (
    OUT
    / "analysis_manifest.json"
)

manifest_path.write_text(
    json.dumps(
        analysis_manifest,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


# =============================================================================
# Compact terminal output
# =============================================================================

print("=" * 80)
print("FINAL V3 — FIRST TEST RESULTS")
print("=" * 80)

print(
    "source:",
    "final_v3 only"
)

print(
    "paired runs read:",
    15
)

print(
    "primary split units:",
    5
)

print(
    "initialization seeds averaged within split:",
    3
)

print(
    "v1/v2 results included:",
    False
)

print(
    "test values used to change protocol:",
    False
)

print()


for dataset in DATASETS:

    r = summary[
        summary[
            "dataset"
        ] == dataset
    ].iloc[0]


    print(dataset + ":")

    print(
        "  RMSE MLP / GCN:",
        f"{r['RMSE_MLP_mean']:.6f} / "
        f"{r['RMSE_GCN_mean']:.6f}"
    )

    print(
        "  RMSE Δ GCN-MLP:",
        f"{r['RMSE_diff_mean_GCN_minus_MLP']:+.6f} "
        f"({r['RMSE_relative_change_percent']:+.2f}%)"
    )

    print(
        "  RMSE split wins GCN/MLP/tie:",
        f"{int(r['RMSE_GCN_wins'])}/"
        f"{int(r['RMSE_MLP_wins'])}/"
        f"{int(r['RMSE_ties'])}"
    )

    print(
        "  RMSE exact p / Holm p:",
        f"{r['RMSE_exact_signflip_p']:.4f} / "
        f"{r['RMSE_Holm_p_4_domains']:.4f}"
    )

    print(
        "  MAE MLP / GCN:",
        f"{r['MAE_MLP_mean']:.6f} / "
        f"{r['MAE_GCN_mean']:.6f}"
    )

    print(
        "  MAE Δ GCN-MLP:",
        f"{r['MAE_diff_mean_GCN_minus_MLP']:+.6f} "
        f"({r['MAE_relative_change_percent']:+.2f}%)"
    )

    print(
        "  MAE split wins GCN/MLP/tie:",
        f"{int(r['MAE_GCN_wins'])}/"
        f"{int(r['MAE_MLP_wins'])}/"
        f"{int(r['MAE_ties'])}"
    )

    print(
        "  MAE exact p / Holm p:",
        f"{r['MAE_exact_signflip_p']:.4f} / "
        f"{r['MAE_Holm_p_4_domains']:.4f}"
    )


    if dataset != "BALANCED_ALL":

        print(
            "  R2 MLP / GCN / Δ:",
            f"{r['R2_MLP_mean']:.6f} / "
            f"{r['R2_GCN_mean']:.6f} / "
            f"{r['R2_diff_mean_GCN_minus_MLP']:+.6f}"
        )


    print()


print(
    "minimum attainable two-sided sign-flip p at n=5:",
    2.0 / 32.0
)

print(
    "analysis summary SHA256:",
    sha256_file(
        OUT
        / "cross_split_primary_summary.csv"
    )
)

print(
    "analysis manifest SHA256:",
    sha256_file(
        manifest_path
    )
)

print()
print(
    "FINAL_V3_FIRST_TEST_ANALYSIS_PASS=True"
)
