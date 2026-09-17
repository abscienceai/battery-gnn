from pathlib import Path
import hashlib
import itertools
import json
import os

import numpy as np
import pandas as pd


BASE = (
    Path.home()
    / "alper"
    / "battery_revision_2026"
)

REPO = (
    BASE
    / "repo"
    / "battery-gnn"
)

TEMP_ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "temporal_controls"
    / "final_temporal_benchmark_v1"
)

TEMP_RUNS = (
    TEMP_ROOT
    / "runs"
)

TEMP_PROTOCOL = (
    TEMP_ROOT
    / "protocol.json"
)

TRAINING_CLOSURE = (
    TEMP_ROOT
    / "training_closure.json"
)

CONT_MANIFEST = (
    TEMP_ROOT
    / "test_evaluation_continuation_manifest.json"
)

CLEAN_ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "feature_matched_soh"
    / "final_v3"
)

CLEAN_PROTOCOL = (
    BASE
    / "results"
    / "revision_2026"
    / "feature_matched_soh"
    / "full_clean_benchmark_protocol_v3.json"
)

CLEAN_MANIFEST = (
    CLEAN_ROOT
    / "final_v3_execution_manifest.json"
)

OUT = (
    TEMP_ROOT
    / "analysis"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


EXPECTED = {
    "temporal_protocol":
        "b6116666e861c79b7332dbd599594846b86511e420d8305eadf82d30b87f0106",

    "training_closure":
        "07a75bfc5ab81d467afda76a8b4cd1097ca0609356bf4bdbd6eb2254a2cc1b55",

    "continuation_manifest":
        "42927309284c6e84f9aee0f80d10747f4a65242f80c361484d30e1d1b0f6c433",

    "clean_protocol":
        "2dec14c210ae3f02e5f65e4acc24f9c4adc747bc071509cb144e69dba707f4dd",

    "clean_manifest":
        "5a8bfbe0f11a6158f2749a9da4f18f1e62adf8c839976c1bf0af9c9f6c07440d",
}


METHOD_ORDER = [
    "RAW_MLP",
    "MOVING_AVERAGE3_MLP",
    "LOWPASS_BINOMIAL3_MLP",
    "SAVGOL5_POLY2_MLP",
    "KALMAN_Q0P01_MLP",
    "CAUSAL_GCN",
]


TEMPORAL_METHOD_MAP = {
    "moving_average3_mlp":
        "MOVING_AVERAGE3_MLP",

    "lowpass_binomial3_mlp":
        "LOWPASS_BINOMIAL3_MLP",

    "savgol5_poly2_mlp":
        "SAVGOL5_POLY2_MLP",

    "kalman_q0p01_mlp":
        "KALMAN_Q0P01_MLP",
}


COMPARATORS = [
    "RAW_MLP",
    "MOVING_AVERAGE3_MLP",
    "LOWPASS_BINOMIAL3_MLP",
    "SAVGOL5_POLY2_MLP",
    "KALMAN_Q0P01_MLP",
]


DATASETS = [
    "BALANCED_ALL",
    "HUST",
    "CALCE",
    "MATR",
]


SPLIT_SEEDS = [
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
            b = f.read(
                1024 * 1024
            )

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def exact_signflip_p(diff):
    """
    Exact two-sided paired sign-flip sensitivity test.
    Higher-level unit = split_seed, n=5.
    """
    d = np.asarray(
        diff,
        dtype=float,
    )

    assert d.shape == (5,)
    assert np.isfinite(d).all()

    observed = abs(
        float(
            np.mean(d)
        )
    )

    stats = []

    for signs in itertools.product(
        (-1.0, 1.0),
        repeat=5,
    ):

        s = np.asarray(
            signs,
            dtype=float,
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
        dtype=float,
    )

    return float(
        np.mean(
            stats
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
# Frozen provenance guards
# =============================================================================

assert sha256_file(
    TEMP_PROTOCOL
) == EXPECTED[
    "temporal_protocol"
]

assert sha256_file(
    TRAINING_CLOSURE
) == EXPECTED[
    "training_closure"
]

assert sha256_file(
    CONT_MANIFEST
) == EXPECTED[
    "continuation_manifest"
]

assert sha256_file(
    CLEAN_PROTOCOL
) == EXPECTED[
    "clean_protocol"
]

assert sha256_file(
    CLEAN_MANIFEST
) == EXPECTED[
    "clean_manifest"
]


tp = json.loads(
    TEMP_PROTOCOL.read_text(
        encoding="utf-8"
    )
)

closure = json.loads(
    TRAINING_CLOSURE.read_text(
        encoding="utf-8"
    )
)

continuation = json.loads(
    CONT_MANIFEST.read_text(
        encoding="utf-8"
    )
)


assert tp[
    "status"
] == "FINAL_FROZEN_BEFORE_NEW_TEMPORAL_TEST_RUNS"

assert tp[
    "kalman_selected_q"
] == 0.01

assert tp[
    "kalman_selected_using_test"
] is False

assert closure[
    "training_fit_count"
] == 60

assert closure[
    "test_metrics_read_before_closure"
] is False

assert continuation[
    "completed_test_evaluations"
] == 60

assert continuation[
    "training_reexecuted"
] is False

assert continuation[
    "test_metrics_surfaced"
] is False

assert continuation[
    "test_metrics_used_to_change_protocol"
] is False


# =============================================================================
# FIRST VALUE-LEVEL READ OF NEW TEMPORAL TEST ARTIFACTS
# =============================================================================

frames = []


for source_method, analysis_method in TEMPORAL_METHOD_MAP.items():

    for split_seed in SPLIT_SEEDS:

        for train_seed in TRAIN_SEEDS:

            d = (
                TEMP_RUNS
                / source_method
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

            assert sha256_file(
                metrics_path
            ) == evaluation[
                "test_metrics_sha256"
            ]

            assert evaluation[
                "training_reexecuted"
            ] is False

            assert evaluation[
                "test_evaluated_only_after_training_closure"
            ] is True

            assert evaluation[
                "test_metrics_used_to_change_protocol"
            ] is False

            m = pd.read_csv(
                metrics_path,
                low_memory=False,
            )

            assert len(m) == 4

            assert set(
                m[
                    "dataset"
                ]
            ) == set(
                DATASETS
            )

            assert set(
                m[
                    "split"
                ]
            ) == {
                "test"
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
                    m[
                        "dataset"
                    ] == dataset
                ]

                assert len(row) == 1

                assert int(
                    row[
                        "n"
                    ].iloc[0]
                ) == EXPECTED_TEST_N[
                    split_seed
                ][
                    dataset
                ]


            assert np.isfinite(
                m[
                    "RMSE"
                ]
            ).all()

            assert np.isfinite(
                m[
                    "MAE"
                ]
            ).all()

            assert np.isfinite(
                m.loc[
                    m[
                        "dataset"
                    ] != "BALANCED_ALL",
                    "R2",
                ]
            ).all()


            m[
                "analysis_method"
            ] = analysis_method

            frames.append(
                m[
                    [
                        "split_seed",
                        "train_seed",
                        "dataset",
                        "analysis_method",
                        "n",
                        "RMSE",
                        "MAE",
                        "R2",
                    ]
                ]
            )


new_test = pd.concat(
    frames,
    ignore_index=True,
)


assert len(
    new_test
) == 240


# =============================================================================
# Frozen clean references: RAW MLP + CAUSAL GCN
# =============================================================================

reference_frames = []


for split_seed in SPLIT_SEEDS:

    for train_seed in TRAIN_SEEDS:

        metrics_path = (
            CLEAN_ROOT
            / (
                f"split{split_seed}"
                f"_train{train_seed}"
            )
            / "metrics.csv"
        )

        m = pd.read_csv(
            metrics_path,
            low_memory=False,
        )

        m = m[
            m[
                "split"
            ] == "test"
        ].copy()


        assert len(
            m
        ) == 8


        m = m[
            m[
                "model"
            ].isin(
                [
                    "MLP",
                    "CAUSAL_GCN",
                ]
            )
        ].copy()


        m[
            "analysis_method"
        ] = m[
            "model"
        ].map(
            {
                "MLP":
                    "RAW_MLP",

                "CAUSAL_GCN":
                    "CAUSAL_GCN",
            }
        )


        for dataset in DATASETS:

            d = m[
                m[
                    "dataset"
                ] == dataset
            ]

            assert len(d) == 2

            assert set(
                d[
                    "n"
                ].astype(int)
            ) == {
                EXPECTED_TEST_N[
                    split_seed
                ][
                    dataset
                ]
            }


        reference_frames.append(
            m[
                [
                    "split_seed",
                    "train_seed",
                    "dataset",
                    "analysis_method",
                    "n",
                    "RMSE",
                    "MAE",
                    "R2",
                ]
            ]
        )


reference = pd.concat(
    reference_frames,
    ignore_index=True,
)


assert len(
    reference
) == 120


all_test = pd.concat(
    [
        reference,
        new_test,
    ],
    ignore_index=True,
)


assert len(
    all_test
) == 360

assert set(
    all_test[
        "analysis_method"
    ]
) == set(
    METHOD_ORDER
)


all_test.to_csv(
    OUT
    / "all_six_methods_test_metrics_15_pairs.csv",
    index=False,
    float_format="%.12g",
)


# =============================================================================
# Average 3 initialization seeds WITHIN split
# =============================================================================

split_level = (
    all_test
    .groupby(
        [
            "split_seed",
            "analysis_method",
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
    split_level
) == (
    5
    * 6
    * 4
)

assert set(
    split_level[
        "initialization_count"
    ]
) == {
    3
}


split_level.to_csv(
    OUT
    / "six_methods_split_level_mean_over_initializations.csv",
    index=False,
    float_format="%.12g",
)


# =============================================================================
# Cross-split descriptive summary for all six methods
# =============================================================================

method_summary = (
    split_level
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
    method_summary
) == 24

assert set(
    method_summary[
        "split_count"
    ]
) == {
    5
}


method_summary.to_csv(
    OUT
    / "six_methods_cross_split_summary.csv",
    index=False,
    float_format="%.12g",
)


# =============================================================================
# GCN versus each comparator.
#
# Δ = comparator - GCN for error metrics.
# Positive => comparator has larger error => GCN better.
# =============================================================================

paired_rows = []


for comparator in COMPARATORS:

    for dataset in DATASETS:

        for split_seed in SPLIT_SEEDS:

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


            comp = d[
                d[
                    "analysis_method"
                ] == comparator
            ]

            gcn = d[
                d[
                    "analysis_method"
                ] == "CAUSAL_GCN"
            ]


            assert len(comp) == 1
            assert len(gcn) == 1


            comp = comp.iloc[0]
            gcn = gcn.iloc[0]


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

                "RMSE_GCN":
                    float(
                        gcn[
                            "RMSE_mean"
                        ]
                    ),

                "RMSE_delta_comparator_minus_GCN":
                    float(
                        comp[
                            "RMSE_mean"
                        ]
                        - gcn[
                            "RMSE_mean"
                        ]
                    ),

                "MAE_comparator":
                    float(
                        comp[
                            "MAE_mean"
                        ]
                    ),

                "MAE_GCN":
                    float(
                        gcn[
                            "MAE_mean"
                        ]
                    ),

                "MAE_delta_comparator_minus_GCN":
                    float(
                        comp[
                            "MAE_mean"
                        ]
                        - gcn[
                            "MAE_mean"
                        ]
                    ),
            })


paired = pd.DataFrame(
    paired_rows
)


assert len(
    paired
) == (
    5
    * 4
    * 5
)


paired.to_csv(
    OUT
    / "gcn_vs_controls_split_level_paired.csv",
    index=False,
    float_format="%.12g",
)


comparison_rows = []


for comparator in COMPARATORS:

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
            "RMSE_delta_comparator_minus_GCN"
        ].to_numpy(
            dtype=float
        )

        mae_diff = d[
            "MAE_delta_comparator_minus_GCN"
        ].to_numpy(
            dtype=float
        )


        rmse_comp = float(
            d[
                "RMSE_comparator"
            ].mean()
        )

        rmse_gcn = float(
            d[
                "RMSE_GCN"
            ].mean()
        )

        mae_comp = float(
            d[
                "MAE_comparator"
            ].mean()
        )

        mae_gcn = float(
            d[
                "MAE_GCN"
            ].mean()
        )


        comparison_rows.append({
            "comparator":
                comparator,

            "dataset":
                dataset,

            "n_split_units":
                5,

            "RMSE_comparator_mean":
                rmse_comp,

            "RMSE_GCN_mean":
                rmse_gcn,

            "RMSE_delta_mean_comparator_minus_GCN":
                float(
                    rmse_diff.mean()
                ),

            "RMSE_relative_GCN_change_vs_comparator_percent":
                float(
                    100.0
                    * (
                        rmse_gcn
                        - rmse_comp
                    )
                    / rmse_comp
                ),

            "RMSE_GCN_wins":
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

            "MAE_GCN_mean":
                mae_gcn,

            "MAE_delta_mean_comparator_minus_GCN":
                float(
                    mae_diff.mean()
                ),

            "MAE_relative_GCN_change_vs_comparator_percent":
                float(
                    100.0
                    * (
                        mae_gcn
                        - mae_comp
                    )
                    / mae_comp
                ),

            "MAE_GCN_wins":
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
    comparison_rows
)


assert len(
    comparison
) == 20


# Frozen family = all five controls x four domains.
comparison[
    "RMSE_Holm_p_20"
] = holm_adjust(
    comparison[
        "RMSE_exact_signflip_p"
    ].to_numpy(
        dtype=float
    )
)

comparison[
    "MAE_Holm_p_20"
] = holm_adjust(
    comparison[
        "MAE_exact_signflip_p"
    ].to_numpy(
        dtype=float
    )
)


comparison.to_csv(
    OUT
    / "gcn_vs_controls_cross_split_summary.csv",
    index=False,
    float_format="%.12g",
)


# =============================================================================
# New controls vs raw MLP — descriptive mechanism comparison
#
# Δ = control - RAW.
# Negative error delta means temporal control improves over raw MLP.
# =============================================================================

raw_rows = []


for control in [
    "MOVING_AVERAGE3_MLP",
    "LOWPASS_BINOMIAL3_MLP",
    "SAVGOL5_POLY2_MLP",
    "KALMAN_Q0P01_MLP",
]:

    for dataset in DATASETS:

        for split_seed in SPLIT_SEEDS:

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


            raw = d[
                d[
                    "analysis_method"
                ] == "RAW_MLP"
            ].iloc[0]

            c = d[
                d[
                    "analysis_method"
                ] == control
            ].iloc[0]


            raw_rows.append({
                "control":
                    control,

                "dataset":
                    dataset,

                "split_seed":
                    split_seed,

                "RMSE_delta_control_minus_raw":
                    float(
                        c[
                            "RMSE_mean"
                        ]
                        - raw[
                            "RMSE_mean"
                        ]
                    ),

                "MAE_delta_control_minus_raw":
                    float(
                        c[
                            "MAE_mean"
                        ]
                        - raw[
                            "MAE_mean"
                        ]
                    ),
            })


raw_compare = pd.DataFrame(
    raw_rows
)


raw_summary = (
    raw_compare
    .groupby(
        [
            "control",
            "dataset",
        ],
        as_index=False,
    )
    .agg(
        RMSE_delta_mean_control_minus_raw=(
            "RMSE_delta_control_minus_raw",
            "mean",
        ),
        RMSE_control_wins=(
            "RMSE_delta_control_minus_raw",
            lambda x: int(
                np.sum(
                    np.asarray(x) < 0
                )
            ),
        ),
        MAE_delta_mean_control_minus_raw=(
            "MAE_delta_control_minus_raw",
            "mean",
        ),
        MAE_control_wins=(
            "MAE_delta_control_minus_raw",
            lambda x: int(
                np.sum(
                    np.asarray(x) < 0
                )
            ),
        ),
    )
)


raw_summary.to_csv(
    OUT
    / "temporal_controls_vs_raw_descriptive.csv",
    index=False,
    float_format="%.12g",
)


# =============================================================================
# Rankings
# =============================================================================

ranking_rows = []


for dataset in DATASETS:

    d = method_summary[
        method_summary[
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


rankings.to_csv(
    OUT
    / "six_methods_rankings.csv",
    index=False,
    float_format="%.12g",
)


# =============================================================================
# Analysis manifest
# =============================================================================

analysis_manifest = {
    "contract":
        "final_temporal_control_analysis_v1",

    "temporal_protocol_sha256":
        EXPECTED[
            "temporal_protocol"
        ],

    "training_closure_sha256":
        EXPECTED[
            "training_closure"
        ],

    "continuation_manifest_sha256":
        EXPECTED[
            "continuation_manifest"
        ],

    "clean_protocol_sha256":
        EXPECTED[
            "clean_protocol"
        ],

    "clean_execution_manifest_sha256":
        EXPECTED[
            "clean_manifest"
        ],

    "new_temporal_test_values_first_parsed_after_artifact_audit":
        True,

    "methods":
        METHOD_ORDER,

    "new_temporal_methods":
        4,

    "reference_methods":
        [
            "RAW_MLP",
            "CAUSAL_GCN",
        ],

    "higher_level_units":
        5,

    "initialization_seeds_per_split":
        3,

    "gcn_difference_definition":
        (
            "comparator minus GCN for RMSE/MAE; "
            "positive favors GCN"
        ),

    "signflip_role":
        (
            "paired repeated-split sensitivity analysis; "
            "five splits are not claimed as five independent datasets"
        ),

    "multiple_testing":
        (
            "Holm separately over 20 GCN-vs-comparator RMSE "
            "comparisons and 20 MAE comparisons"
        ),

    "outputs": {
        "all_test":
            sha256_file(
                OUT
                / "all_six_methods_test_metrics_15_pairs.csv"
            ),

        "split_level":
            sha256_file(
                OUT
                / "six_methods_split_level_mean_over_initializations.csv"
            ),

        "method_summary":
            sha256_file(
                OUT
                / "six_methods_cross_split_summary.csv"
            ),

        "paired":
            sha256_file(
                OUT
                / "gcn_vs_controls_split_level_paired.csv"
            ),

        "comparison":
            sha256_file(
                OUT
                / "gcn_vs_controls_cross_split_summary.csv"
            ),

        "raw_comparison":
            sha256_file(
                OUT
                / "temporal_controls_vs_raw_descriptive.csv"
            ),

        "rankings":
            sha256_file(
                OUT
                / "six_methods_rankings.csv"
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
# Compact terminal report
# =============================================================================

short = {
    "RAW_MLP":
        "RAW",

    "MOVING_AVERAGE3_MLP":
        "MA3",

    "LOWPASS_BINOMIAL3_MLP":
        "LP3",

    "SAVGOL5_POLY2_MLP":
        "SG5",

    "KALMAN_Q0P01_MLP":
        "KAL",

    "CAUSAL_GCN":
        "GCN",
}


print("=" * 82)
print("FINAL TEMPORAL CONTROL — FIRST VALUE-LEVEL TEST ANALYSIS")
print("=" * 82)

print(
    "new temporal test values first parsed after artifact audit:",
    True
)

print(
    "methods:",
    6
)

print(
    "split units:",
    5
)

print(
    "initializations averaged within split:",
    3
)

print(
    "Holm family per error metric:",
    20
)

print()


for dataset in DATASETS:

    d = method_summary[
        method_summary[
            "dataset"
        ] == dataset
    ].set_index(
        "analysis_method"
    )


    rmse_text = " | ".join(
        (
            short[m]
            + "="
            + f"{float(d.loc[m, 'RMSE_mean']):.6f}"
        )
        for m in METHOD_ORDER
    )


    mae_text = " | ".join(
        (
            short[m]
            + "="
            + f"{float(d.loc[m, 'MAE_mean']):.6f}"
        )
        for m in METHOD_ORDER
    )


    best_rmse = (
        d[
            "RMSE_mean"
        ].idxmin()
    )

    best_mae = (
        d[
            "MAE_mean"
        ].idxmin()
    )


    classical = [
        "MOVING_AVERAGE3_MLP",
        "LOWPASS_BINOMIAL3_MLP",
        "SAVGOL5_POLY2_MLP",
        "KALMAN_Q0P01_MLP",
    ]


    best_classical_rmse = (
        d.loc[
            classical,
            "RMSE_mean",
        ].idxmin()
    )


    c = comparison[
        (
            comparison[
                "dataset"
            ] == dataset
        )
        &
        (
            comparison[
                "comparator"
            ] == best_classical_rmse
        )
    ].iloc[0]


    print(dataset + ":")

    print(
        "  RMSE:",
        rmse_text
    )

    print(
        "  MAE :",
        mae_text
    )

    print(
        "  best RMSE / MAE:",
        short[
            best_rmse
        ],
        "/",
        short[
            best_mae
        ]
    )

    print(
        "  best classical RMSE vs GCN Δ(comp-GCN):",
        short[
            best_classical_rmse
        ],
        f"{c['RMSE_delta_mean_comparator_minus_GCN']:+.6f}",
        "wins GCN/control=",
        f"{int(c['RMSE_GCN_wins'])}/"
        f"{int(c['RMSE_comparator_wins'])}",
        "p/Holm=",
        f"{c['RMSE_exact_signflip_p']:.4f}/"
        f"{c['RMSE_Holm_p_20']:.4f}",
    )

    print()


balanced = comparison[
    comparison[
        "dataset"
    ] == "BALANCED_ALL"
].copy()


print("BALANCED_ALL — GCN vs each comparator:")


for comparator in COMPARATORS:

    r = balanced[
        balanced[
            "comparator"
        ] == comparator
    ].iloc[0]


    print(
        " ",
        short[
            comparator
        ],
        "RMSE Δ=",
        f"{r['RMSE_delta_mean_comparator_minus_GCN']:+.6f}",
        "GCN/control wins=",
        f"{int(r['RMSE_GCN_wins'])}/"
        f"{int(r['RMSE_comparator_wins'])}",
        "p/Holm=",
        f"{r['RMSE_exact_signflip_p']:.4f}/"
        f"{r['RMSE_Holm_p_20']:.4f}",
    )


print()

print(
    "minimum RMSE Holm p across 20 comparisons:",
    f"{comparison['RMSE_Holm_p_20'].min():.4f}"
)

print(
    "minimum MAE Holm p across 20 comparisons:",
    f"{comparison['MAE_Holm_p_20'].min():.4f}"
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
    "FINAL_TEMPORAL_CONTROL_ANALYSIS_PASS=True"
)
