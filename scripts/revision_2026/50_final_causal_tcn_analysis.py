from pathlib import Path
import hashlib
import itertools
import json
import numpy as np
import pandas as pd


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo" / "battery-gnn"

TCN_ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "sequence_controls"
    / "final_tcn_v1"
)

TCN_RUNS = TCN_ROOT / "runs"
TCN_PROTOCOL = TCN_ROOT / "protocol.json"
TCN_CLOSURE = TCN_ROOT / "training_closure.json"
TCN_MANIFEST = TCN_ROOT / "execution_manifest.json"

TEMP_ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "temporal_controls"
    / "final_temporal_benchmark_v1"
)

TEMP_ANALYSIS = TEMP_ROOT / "analysis"
TEMP_MANIFEST = TEMP_ANALYSIS / "analysis_manifest.json"

SIX_SPLIT = (
    TEMP_ANALYSIS
    / "six_methods_split_level_mean_over_initializations.csv"
)

OUT = TCN_ROOT / "analysis"
OUT.mkdir(parents=True, exist_ok=True)


EXPECTED = {
    "tcn_protocol":
        "19fe5012478bd1d70d3f580d08ee8391c905858dbbb3b3e1015fb8627a54f2c3",

    "tcn_closure":
        "5b78737644195a0d17298ad6660d6a32911a573bfcadbb45f3d6281f639ef2ed",

    "tcn_manifest":
        "35b246580e7c6153d72d93e15e78bfd5b0e1beb2c18aa2f9ea41bd0a60bd9d35",

    "temporal_analysis_manifest":
        "c67f1cae7264a6495f9454707fca4b0797ce1ca12234a83dc3d6de40481d1f20",
}


DATASETS = [
    "BALANCED_ALL",
    "HUST",
    "CALCE",
    "MATR",
]

SPLIT_SEEDS = [0, 1, 2, 3, 4]
TRAIN_SEEDS = [0, 1, 2]

PRIMARY_COMPARATORS = [
    "RAW_MLP",
    "CAUSAL_GCN",
]

SEVEN_METHODS = [
    "RAW_MLP",
    "MOVING_AVERAGE3_MLP",
    "LOWPASS_BINOMIAL3_MLP",
    "SAVGOL5_POLY2_MLP",
    "KALMAN_Q0P01_MLP",
    "CAUSAL_GCN",
    "CAUSAL_TCN",
]

SHORT = {
    "RAW_MLP": "RAW",
    "MOVING_AVERAGE3_MLP": "MA3",
    "LOWPASS_BINOMIAL3_MLP": "LP3",
    "SAVGOL5_POLY2_MLP": "SG5",
    "KALMAN_Q0P01_MLP": "KAL",
    "CAUSAL_GCN": "GCN",
    "CAUSAL_TCN": "TCN",
}


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
    d = np.asarray(diff, dtype=float)

    assert d.shape == (5,)
    assert np.isfinite(d).all()

    observed = abs(float(np.mean(d)))

    permuted = []

    for signs in itertools.product(
        (-1.0, 1.0),
        repeat=5,
    ):
        s = np.asarray(signs, dtype=float)

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

    return float(
        np.mean(
            permuted
            >= (
                observed
                - 1e-15
            )
        )
    )


def holm_adjust(p_values):
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

    for rank, idx in enumerate(order):
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
# Provenance guards BEFORE value-level read
# =============================================================================

assert sha256_file(
    TCN_PROTOCOL
) == EXPECTED["tcn_protocol"]

assert sha256_file(
    TCN_CLOSURE
) == EXPECTED["tcn_closure"]

assert sha256_file(
    TCN_MANIFEST
) == EXPECTED["tcn_manifest"]

assert sha256_file(
    TEMP_MANIFEST
) == EXPECTED["temporal_analysis_manifest"]


protocol = json.loads(
    TCN_PROTOCOL.read_text(
        encoding="utf-8"
    )
)

closure = json.loads(
    TCN_CLOSURE.read_text(
        encoding="utf-8"
    )
)

execution = json.loads(
    TCN_MANIFEST.read_text(
        encoding="utf-8"
    )
)

temporal_manifest = json.loads(
    TEMP_MANIFEST.read_text(
        encoding="utf-8"
    )
)


assert protocol[
    "status"
] == "FINAL_FROZEN_BEFORE_ANY_TCN_TRAINING_OR_TEST"

assert protocol[
    "primary_tcn_comparators"
] == [
    "RAW_MLP",
    "CAUSAL_GCN",
]

assert protocol[
    "multiple_testing"
][
    "family_per_error_metric"
] == 8

assert closure[
    "training_fit_count"
] == 15

assert closure[
    "test_metrics_read_before_closure"
] is False

assert execution[
    "completed_training_fit_count"
] == 15

assert execution[
    "completed_test_evaluation_count"
] == 15

assert execution[
    "all_training_complete_before_first_test_evaluation"
] is True

assert execution[
    "test_metrics_read_before_training_closure"
] is False

assert execution[
    "reference_result_files_accessed_by_runner"
] is False

assert execution[
    "test_metric_values_surfaced_during_runner"
] is False

assert execution[
    "test_metrics_used_to_change_protocol"
] is False


assert sha256_file(
    SIX_SPLIT
) == temporal_manifest[
    "outputs"
][
    "split_level"
]


# =============================================================================
# FIRST VALUE-LEVEL READ OF TCN TEST METRICS
# =============================================================================

frames = []


for split_seed in SPLIT_SEEDS:

    for train_seed in TRAIN_SEEDS:

        d = (
            TCN_RUNS
            / (
                f"split{split_seed}"
                f"_train{train_seed}"
            )
        )

        evaluation = json.loads(
            (
                d
                / "evaluation.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        metrics_path = (
            d
            / "test_metrics.csv"
        )

        assert evaluation[
            "test_evaluated_only_after_training_closure"
        ] is True

        assert evaluation[
            "test_metrics_used_to_change_protocol"
        ] is False

        assert evaluation[
            "test_metrics_surfaced_by_runner"
        ] is False

        assert sha256_file(
            metrics_path
        ) == evaluation[
            "test_metrics_sha256"
        ]


        m = pd.read_csv(
            metrics_path,
            low_memory=False,
        )

        assert len(m) == 4

        assert set(
            m["dataset"]
        ) == set(
            DATASETS
        )

        assert set(
            m["split"]
        ) == {
            "test"
        }

        assert set(
            m["method"]
        ) == {
            "CAUSAL_TCN"
        }

        assert set(
            m[
                "split_seed"
            ].astype(int)
        ) == {
            split_seed
        }

        assert set(
            m[
                "train_seed"
            ].astype(int)
        ) == {
            train_seed
        }


        for dataset in DATASETS:

            row = m[
                m["dataset"]
                == dataset
            ]

            assert len(row) == 1

            assert int(
                row["n"].iloc[0]
            ) == EXPECTED_TEST_N[
                split_seed
            ][
                dataset
            ]


        assert np.isfinite(
            m["RMSE"]
        ).all()

        assert np.isfinite(
            m["MAE"]
        ).all()

        assert np.isfinite(
            m.loc[
                m["dataset"]
                != "BALANCED_ALL",
                "R2",
            ]
        ).all()


        frames.append(
            m[
                [
                    "split_seed",
                    "train_seed",
                    "dataset",
                    "n",
                    "RMSE",
                    "MAE",
                    "R2",
                ]
            ].copy()
        )


tcn_test = pd.concat(
    frames,
    ignore_index=True,
)


assert len(
    tcn_test
) == 60


tcn_test.to_csv(
    OUT
    / "tcn_test_metrics_15_runs.csv",
    index=False,
    float_format="%.12g",
)


# =============================================================================
# Average 3 initialization seeds within split
# =============================================================================

tcn_split = (
    tcn_test
    .groupby(
        [
            "split_seed",
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

        initialization_count=(
            "train_seed",
            "nunique",
        ),

        n=(
            "n",
            "first",
        ),
    )
)


assert len(
    tcn_split
) == 20

assert set(
    tcn_split[
        "initialization_count"
    ]
) == {
    3
}


tcn_split[
    "analysis_method"
] = "CAUSAL_TCN"


tcn_split.to_csv(
    OUT
    / "tcn_split_level_mean_over_initializations.csv",
    index=False,
    float_format="%.12g",
)


# =============================================================================
# Load frozen six-method split-level reference
# =============================================================================

six = pd.read_csv(
    SIX_SPLIT,
    low_memory=False,
)


assert len(
    six
) == 120

assert set(
    six[
        "analysis_method"
    ]
) == set(
    SEVEN_METHODS[:-1]
)


tcn_for_merge = tcn_split[
    [
        "split_seed",
        "analysis_method",
        "dataset",
        "RMSE_mean",
        "RMSE_seed_sd",
        "MAE_mean",
        "MAE_seed_sd",
        "R2_mean",
        "R2_seed_sd",
        "initialization_count",
        "n",
    ]
].copy()


seven_split = pd.concat(
    [
        six,
        tcn_for_merge,
    ],
    ignore_index=True,
)


assert len(
    seven_split
) == 140

assert set(
    seven_split[
        "analysis_method"
    ]
) == set(
    SEVEN_METHODS
)


seven_split.to_csv(
    OUT
    / "seven_methods_split_level.csv",
    index=False,
    float_format="%.12g",
)


# =============================================================================
# Seven-method descriptive cross-split summary
# =============================================================================

seven_summary = (
    seven_split
    .groupby(
        [
            "analysis_method",
            "dataset",
        ],
        as_index=False,
    )
    .agg(
        RMSE_mean=(
            "RMSE_mean",
            "mean",
        ),

        RMSE_sd_across_splits=(
            "RMSE_mean",
            "std",
        ),

        MAE_mean=(
            "MAE_mean",
            "mean",
        ),

        MAE_sd_across_splits=(
            "MAE_mean",
            "std",
        ),

        R2_mean=(
            "R2_mean",
            "mean",
        ),

        R2_sd_across_splits=(
            "R2_mean",
            "std",
        ),

        split_count=(
            "split_seed",
            "nunique",
        ),
    )
)


assert len(
    seven_summary
) == 28

assert set(
    seven_summary[
        "split_count"
    ]
) == {
    5
}


seven_summary.to_csv(
    OUT
    / "seven_methods_cross_split_summary.csv",
    index=False,
    float_format="%.12g",
)


# =============================================================================
# PRIMARY inferential sensitivity family:
# TCN vs RAW and TCN vs GCN across 4 domains = 8 / metric.
#
# Δ = comparator - TCN.
# Positive error delta => TCN lower error => favors TCN.
# =============================================================================

paired_rows = []


for comparator in PRIMARY_COMPARATORS:

    for dataset in DATASETS:

        for split_seed in SPLIT_SEEDS:

            d = seven_split[
                (
                    seven_split[
                        "split_seed"
                    ] == split_seed
                )
                &
                (
                    seven_split[
                        "dataset"
                    ] == dataset
                )
            ]


            comp = d[
                d[
                    "analysis_method"
                ] == comparator
            ]

            tcn = d[
                d[
                    "analysis_method"
                ] == "CAUSAL_TCN"
            ]


            assert len(comp) == 1
            assert len(tcn) == 1


            comp = comp.iloc[0]
            tcn = tcn.iloc[0]


            paired_rows.append({
                "comparator":
                    comparator,

                "dataset":
                    dataset,

                "split_seed":
                    split_seed,

                "RMSE_comparator":
                    float(
                        comp[
                            "RMSE_mean"
                        ]
                    ),

                "RMSE_TCN":
                    float(
                        tcn[
                            "RMSE_mean"
                        ]
                    ),

                "RMSE_delta_comparator_minus_TCN":
                    float(
                        comp[
                            "RMSE_mean"
                        ]
                        - tcn[
                            "RMSE_mean"
                        ]
                    ),

                "MAE_comparator":
                    float(
                        comp[
                            "MAE_mean"
                        ]
                    ),

                "MAE_TCN":
                    float(
                        tcn[
                            "MAE_mean"
                        ]
                    ),

                "MAE_delta_comparator_minus_TCN":
                    float(
                        comp[
                            "MAE_mean"
                        ]
                        - tcn[
                            "MAE_mean"
                        ]
                    ),
            })


paired = pd.DataFrame(
    paired_rows
)


assert len(
    paired
) == 40


paired.to_csv(
    OUT
    / "tcn_vs_primary_comparators_split_level.csv",
    index=False,
    float_format="%.12g",
)


rows = []


for comparator in PRIMARY_COMPARATORS:

    for dataset in DATASETS:

        d = paired[
            (
                paired[
                    "comparator"
                ] == comparator
            )
            &
            (
                paired[
                    "dataset"
                ] == dataset
            )
        ]


        assert len(d) == 5


        rmse_diff = d[
            "RMSE_delta_comparator_minus_TCN"
        ].to_numpy(
            dtype=float
        )

        mae_diff = d[
            "MAE_delta_comparator_minus_TCN"
        ].to_numpy(
            dtype=float
        )


        rmse_comp = float(
            d[
                "RMSE_comparator"
            ].mean()
        )

        rmse_tcn = float(
            d[
                "RMSE_TCN"
            ].mean()
        )

        mae_comp = float(
            d[
                "MAE_comparator"
            ].mean()
        )

        mae_tcn = float(
            d[
                "MAE_TCN"
            ].mean()
        )


        rows.append({
            "comparator":
                comparator,

            "dataset":
                dataset,

            "n_split_units":
                5,

            "RMSE_comparator_mean":
                rmse_comp,

            "RMSE_TCN_mean":
                rmse_tcn,

            "RMSE_delta_mean_comparator_minus_TCN":
                float(
                    rmse_diff.mean()
                ),

            "RMSE_relative_TCN_change_vs_comparator_percent":
                float(
                    100.0
                    * (
                        rmse_tcn
                        - rmse_comp
                    )
                    / rmse_comp
                ),

            "RMSE_TCN_wins":
                int(
                    np.sum(
                        rmse_diff > 0
                    )
                ),

            "RMSE_comparator_wins":
                int(
                    np.sum(
                        rmse_diff < 0
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

            "MAE_comparator_mean":
                mae_comp,

            "MAE_TCN_mean":
                mae_tcn,

            "MAE_delta_mean_comparator_minus_TCN":
                float(
                    mae_diff.mean()
                ),

            "MAE_relative_TCN_change_vs_comparator_percent":
                float(
                    100.0
                    * (
                        mae_tcn
                        - mae_comp
                    )
                    / mae_comp
                ),

            "MAE_TCN_wins":
                int(
                    np.sum(
                        mae_diff > 0
                    )
                ),

            "MAE_comparator_wins":
                int(
                    np.sum(
                        mae_diff < 0
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


comparison = pd.DataFrame(
    rows
)


assert len(
    comparison
) == 8


comparison[
    "RMSE_Holm_p_8"
] = holm_adjust(
    comparison[
        "RMSE_exact_signflip_p"
    ].to_numpy(
        dtype=float
    )
)

comparison[
    "MAE_Holm_p_8"
] = holm_adjust(
    comparison[
        "MAE_exact_signflip_p"
    ].to_numpy(
        dtype=float
    )
)


comparison.to_csv(
    OUT
    / "tcn_vs_primary_comparators_summary.csv",
    index=False,
    float_format="%.12g",
)


# =============================================================================
# Seven-method rankings — descriptive only
# =============================================================================

ranking_rows = []


for dataset in DATASETS:

    d = seven_summary[
        seven_summary[
            "dataset"
        ] == dataset
    ]


    for metric in [
        "RMSE",
        "MAE",
    ]:

        ordered = d.sort_values(
            metric
            + "_mean",
            ascending=True,
        )


        for rank, (_, row) in enumerate(
            ordered.iterrows(),
            start=1,
        ):

            ranking_rows.append({
                "dataset":
                    dataset,

                "metric":
                    metric,

                "rank":
                    rank,

                "method":
                    row[
                        "analysis_method"
                    ],

                "value":
                    float(
                        row[
                            metric
                            + "_mean"
                        ]
                    ),
            })


rankings = pd.DataFrame(
    ranking_rows
)


assert len(
    rankings
) == 56


rankings.to_csv(
    OUT
    / "seven_methods_rankings.csv",
    index=False,
    float_format="%.12g",
)


# =============================================================================
# Analysis manifest
# =============================================================================

manifest = {
    "contract":
        "final_causal_tcn_analysis_v1",

    "tcn_protocol_sha256":
        EXPECTED[
            "tcn_protocol"
        ],

    "tcn_training_closure_sha256":
        EXPECTED[
            "tcn_closure"
        ],

    "tcn_execution_manifest_sha256":
        EXPECTED[
            "tcn_manifest"
        ],

    "temporal_analysis_manifest_sha256":
        EXPECTED[
            "temporal_analysis_manifest"
        ],

    "tcn_test_values_first_parsed_after_execution_audit":
        True,

    "tcn_training_reexecuted_for_analysis":
        False,

    "higher_level_units":
        5,

    "initialization_seeds_per_split":
        3,

    "primary_comparators":
        PRIMARY_COMPARATORS,

    "primary_domains":
        DATASETS,

    "holm_family_per_error_metric":
        8,

    "difference_definition":
        (
            "comparator minus TCN for RMSE/MAE; "
            "positive favors TCN"
        ),

    "signflip_role":
        (
            "paired repeated-split sensitivity analysis; "
            "five splits are not claimed as independent datasets"
        ),

    "seven_method_ranking_role":
        "descriptive only",

    "outputs": {
        "tcn_test_metrics":
            sha256_file(
                OUT
                / "tcn_test_metrics_15_runs.csv"
            ),

        "tcn_split_level":
            sha256_file(
                OUT
                / "tcn_split_level_mean_over_initializations.csv"
            ),

        "seven_split":
            sha256_file(
                OUT
                / "seven_methods_split_level.csv"
            ),

        "seven_summary":
            sha256_file(
                OUT
                / "seven_methods_cross_split_summary.csv"
            ),

        "paired":
            sha256_file(
                OUT
                / "tcn_vs_primary_comparators_split_level.csv"
            ),

        "comparison":
            sha256_file(
                OUT
                / "tcn_vs_primary_comparators_summary.csv"
            ),

        "rankings":
            sha256_file(
                OUT
                / "seven_methods_rankings.csv"
            ),
    },
}


manifest_path = (
    OUT
    / "analysis_manifest.json"
)


manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


# =============================================================================
# Compact terminal report
# =============================================================================

print("=" * 82)
print("FINAL CAUSAL TCN — FIRST VALUE-LEVEL TEST ANALYSIS")
print("=" * 82)

print(
    "TCN test values first parsed after execution audit:",
    True
)

print(
    "training reexecuted for analysis:",
    False
)

print(
    "primary comparator/domain family:",
    "2 x 4 = 8 per metric"
)

print(
    "split units:",
    5
)

print(
    "initializations averaged within split:",
    3
)

print()


for dataset in DATASETS:

    d = seven_summary[
        seven_summary[
            "dataset"
        ] == dataset
    ].set_index(
        "analysis_method"
    )


    print(dataset + ":")

    print(
        "  RMSE "
        + " | ".join(
            (
                SHORT[m]
                + "="
                + f"{float(d.loc[m, 'RMSE_mean']):.6f}"
            )
            for m in SEVEN_METHODS
        )
    )

    print(
        "  MAE  "
        + " | ".join(
            (
                SHORT[m]
                + "="
                + f"{float(d.loc[m, 'MAE_mean']):.6f}"
            )
            for m in SEVEN_METHODS
        )
    )


    rmse_rank = (
        d[
            "RMSE_mean"
        ].sort_values()
    )

    mae_rank = (
        d[
            "MAE_mean"
        ].sort_values()
    )


    print(
        "  best RMSE / MAE:",
        SHORT[
            rmse_rank.index[0]
        ],
        "/",
        SHORT[
            mae_rank.index[0]
        ],
        "| TCN RMSE rank:",
        int(
            list(
                rmse_rank.index
            ).index(
                "CAUSAL_TCN"
            )
            + 1
        ),
        "/7",
        "| TCN MAE rank:",
        int(
            list(
                mae_rank.index
            ).index(
                "CAUSAL_TCN"
            )
            + 1
        ),
        "/7",
    )

    print()


print("PRIMARY TCN COMPARISONS:")


for comparator in PRIMARY_COMPARATORS:

    print(
        " ",
        SHORT[
            comparator
        ],
        "vs TCN"
    )


    for dataset in DATASETS:

        r = comparison[
            (
                comparison[
                    "comparator"
                ] == comparator
            )
            &
            (
                comparison[
                    "dataset"
                ] == dataset
            )
        ].iloc[0]


        print(
            "   ",
            dataset,
            "RMSE Δ(comp-TCN)=",
            f"{r['RMSE_delta_mean_comparator_minus_TCN']:+.6f}",
            "relTCN=",
            f"{r['RMSE_relative_TCN_change_vs_comparator_percent']:+.2f}%",
            "wins TCN/comp=",
            f"{int(r['RMSE_TCN_wins'])}/"
            f"{int(r['RMSE_comparator_wins'])}",
            "p/Holm=",
            f"{r['RMSE_exact_signflip_p']:.4f}/"
            f"{r['RMSE_Holm_p_8']:.4f}",
        )


print()

print(
    "minimum RMSE Holm p across 8 primary comparisons:",
    f"{comparison['RMSE_Holm_p_8'].min():.4f}"
)

print(
    "minimum MAE Holm p across 8 primary comparisons:",
    f"{comparison['MAE_Holm_p_8'].min():.4f}"
)

print(
    "minimum attainable unadjusted two-sided p at n=5:",
    "0.0625"
)

print(
    "analysis manifest SHA256:",
    sha256_file(
        manifest_path
    )
)

print()
print(
    "FINAL_CAUSAL_TCN_ANALYSIS_PASS=True"
)
