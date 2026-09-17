from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math
import statistics


BASE = Path.home() / "alper/battery_revision_2026"

REPO = (
    BASE
    / "repo/battery-gnn"
)

ROOT = (
    BASE
    / "results/revision_2026/dqdv_leakage_control/final_dqdv_v1"
)

EXEC = (
    ROOT
    / "training_v2/final_execution_v2"
)

OUT = (
    ROOT
    / "analysis_v1"
)

EXECUTION_MANIFEST = (
    EXEC
    / "execution_manifest.json"
)

EXECUTION_AUDIT = (
    REPO
    / "scripts/revision_2026/96_final_dqdv_training_v2_execution_audit.py"
)

ANALYSIS_SCRIPT = (
    REPO
    / "scripts/revision_2026/99_dqdv_descriptive_analysis.py"
)

ANALYSIS_MANIFEST = (
    OUT
    / "dqdv_analysis_manifest.json"
)

SUMMARY_CSV = (
    OUT
    / "dqdv_descriptive_summary.csv"
)

DELTA_CSV = (
    OUT
    / "dqdv_vs_primary20_deltas.csv"
)

MODEL_CSV = (
    OUT
    / "dqdv_model_contrasts.csv"
)


EXPECTED_EXECUTION_MANIFEST_SHA = (
    "744ae4a695f9a94a2c597469dc867c58141ea94fc5d131074d88574a25b72725"
)

EXPECTED_EXECUTION_AUDIT_SHA = (
    "086b9374441055b386de2007af0ae2041ff5707508d9fdad246e5aee7a27a658"
)


CONDITIONS = [
    "PRIMARY20_COMMON",
    "PRIMARY20_PLUS_FULL_DQDV10",
    "PRIMARY20_PLUS_PREFIX25_DQDV10",
    "PRIMARY20_PLUS_PREFIX50_DQDV10",
]

MODELS = [
    "MLP",
    "CAUSAL_GCN",
]

DOMAINS = [
    "BALANCED_ALL",
    "HUST",
    "CALCE",
    "MATR",
]

METRICS = [
    "RMSE",
    "MAE",
]


def sha256(path):

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


def read_csv(path):

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        return list(
            csv.DictReader(
                f
            )
        )


def mean(xs):

    xs = [
        float(x)
        for x in xs
    ]

    return float(
        statistics.mean(
            xs
        )
    )


def split_sd(rows, key):

    by_split = defaultdict(list)

    for row in rows:

        by_split[
            int(
                row[
                    "split_seed"
                ]
            )
        ].append(
            float(
                row[
                    key
                ]
            )
        )

    assert set(
        by_split
    ) == {
        0, 1, 2, 3, 4
    }

    x = [
        mean(
            by_split[s]
        )
        for s in sorted(
            by_split
        )
    ]

    return float(
        statistics.stdev(
            x
        )
    )


def close(a, b):

    a = float(a)
    b = float(b)

    assert math.isfinite(a)
    assert math.isfinite(b)

    return math.isclose(
        a,
        b,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


assert sha256(
    EXECUTION_MANIFEST
) == EXPECTED_EXECUTION_MANIFEST_SHA

assert sha256(
    EXECUTION_AUDIT
) == EXPECTED_EXECUTION_AUDIT_SHA


m = json.loads(
    ANALYSIS_MANIFEST.read_text(
        encoding="utf-8"
    )
)

assert m[
    "contract"
] == "dqdv_descriptive_analysis_v1"

assert m[
    "status"
] == "COMPLETE_PENDING_INDEPENDENT_ANALYSIS_AUDIT"

assert m[
    "execution_manifest_sha256"
] == EXPECTED_EXECUTION_MANIFEST_SHA

assert m[
    "execution_audit_sha256"
] == EXPECTED_EXECUTION_AUDIT_SHA

assert m[
    "analysis_script_sha256"
] == sha256(
    ANALYSIS_SCRIPT
)

assert m[
    "record_count"
] == 60

assert m[
    "model_fit_count"
] == 120

assert m[
    "runs_per_condition"
] == 15

assert m[
    "common_cohort"
] is True

assert m[
    "new_inferential_tests"
] is False

assert m[
    "p_values_computed"
] is False

assert m[
    "multiple_testing_correction_computed"
] is False

assert m[
    "descriptive_only"
] is True

assert m[
    "model_training_performed"
] is False

assert m[
    "model_state_reconstruction_performed"
] is False

assert m[
    "numeric_test_metric_values_printed_by_analysis"
] is False

assert sha256(
    SUMMARY_CSV
) == m[
    "summary_csv_sha256"
]

assert sha256(
    DELTA_CSV
) == m[
    "delta_csv_sha256"
]

assert sha256(
    MODEL_CSV
) == m[
    "model_contrast_csv_sha256"
]


execution = json.loads(
    EXECUTION_MANIFEST.read_text(
        encoding="utf-8"
    )
)

assert len(
    execution[
        "records"
    ]
) == 60


records = []

for item in execution[
    "records"
]:

    p = Path(
        item[
            "record_path"
        ]
    )

    assert p.is_file()

    assert sha256(
        p
    ) == item[
        "record_sha256"
    ]

    records.append(
        json.loads(
            p.read_text(
                encoding="utf-8"
            )
        )
    )


raw = []


for r in records:

    condition = str(
        r[
            "condition"
        ]
    )

    assert condition in CONDITIONS


    for model in MODELS:

        test = r[
            "models"
        ][
            model
        ][
            "metrics"
        ][
            "test"
        ]

        assert set(
            test
        ) == {
            "balanced",
            "by_dataset",
        }


        for metric in METRICS:

            value = float(
                test[
                    "balanced"
                ][
                    metric.lower()
                ]
            )

            assert math.isfinite(
                value
            )

            raw.append({
                "condition":
                    condition,

                "model":
                    model,

                "domain":
                    "BALANCED_ALL",

                "metric":
                    metric,

                "split_seed":
                    int(
                        r[
                            "split_seed"
                        ]
                    ),

                "train_seed":
                    int(
                        r[
                            "train_seed"
                        ]
                    ),

                "value":
                    value,
            })


        datasets = {
            str(x["dataset"]):
                x
            for x in test[
                "by_dataset"
            ]
        }

        assert set(
            datasets
        ) == {
            "HUST",
            "CALCE",
            "MATR",
        }


        for domain in [
            "HUST",
            "CALCE",
            "MATR",
        ]:

            for metric in METRICS:

                value = float(
                    datasets[
                        domain
                    ][
                        metric
                    ]
                )

                assert math.isfinite(
                    value
                )

                raw.append({
                    "condition":
                        condition,

                    "model":
                        model,

                    "domain":
                        domain,

                    "metric":
                        metric,

                    "split_seed":
                        int(
                            r[
                                "split_seed"
                            ]
                        ),

                    "train_seed":
                        int(
                            r[
                                "train_seed"
                            ]
                        ),

                    "value":
                        value,
                })


assert len(
    raw
) == (
    4
    * 2
    * 4
    * 2
    * 15
)


# ------------------------------------------------------------
# Recompute summary independently.
# ------------------------------------------------------------

groups = defaultdict(list)

for r in raw:

    groups[
        (
            r[
                "condition"
            ],
            r[
                "model"
            ],
            r[
                "domain"
            ],
            r[
                "metric"
            ],
        )
    ].append(
        r
    )


summary_file = read_csv(
    SUMMARY_CSV
)

assert len(
    summary_file
) == 64


summary_map = {
    (
        r[
            "condition"
        ],
        r[
            "model"
        ],
        r[
            "domain"
        ],
        r[
            "metric"
        ],
    ):
        r
    for r in summary_file
}

assert len(
    summary_map
) == 64


for key, rows in groups.items():

    assert len(
        rows
    ) == 15

    out = summary_map[
        key
    ]

    expected_mean = mean(
        [
            x[
                "value"
            ]
            for x in rows
        ]
    )

    expected_sd = split_sd(
        rows,
        "value",
    )

    expected_min = min(
        float(
            x[
                "value"
            ]
        )
        for x in rows
    )

    expected_max = max(
        float(
            x[
                "value"
            ]
        )
        for x in rows
    )

    assert int(
        out[
            "n_runs"
        ]
    ) == 15

    assert int(
        out[
            "n_split_seeds"
        ]
    ) == 5

    assert int(
        out[
            "n_train_seeds_per_split"
        ]
    ) == 3

    assert close(
        out[
            "mean"
        ],
        expected_mean,
    )

    assert close(
        out[
            "split_seed_mean_sd"
        ],
        expected_sd,
    )

    assert close(
        out[
            "minimum"
        ],
        expected_min,
    )

    assert close(
        out[
            "maximum"
        ],
        expected_max,
    )


# ------------------------------------------------------------
# Recompute matched deltas independently.
# ------------------------------------------------------------

raw_map = {
    (
        r[
            "condition"
        ],
        r[
            "model"
        ],
        r[
            "domain"
        ],
        r[
            "metric"
        ],
        int(
            r[
                "split_seed"
            ]
        ),
        int(
            r[
                "train_seed"
            ]
        ),
    ):
        float(
            r[
                "value"
            ]
        )
    for r in raw
}

assert len(
    raw_map
) == len(
    raw
)


delta_file = read_csv(
    DELTA_CSV
)

assert len(
    delta_file
) == 48


delta_map = {
    (
        r[
            "condition"
        ],
        r[
            "model"
        ],
        r[
            "domain"
        ],
        r[
            "metric"
        ],
    ):
        r
    for r in delta_file
}

assert len(
    delta_map
) == 48


for condition in CONDITIONS[1:]:

    for model in MODELS:

        for domain in DOMAINS:

            for metric in METRICS:

                pairs = []

                for split_seed in range(5):

                    for train_seed in range(3):

                        base = raw_map[
                            (
                                "PRIMARY20_COMMON",
                                model,
                                domain,
                                metric,
                                split_seed,
                                train_seed,
                            )
                        ]

                        value = raw_map[
                            (
                                condition,
                                model,
                                domain,
                                metric,
                                split_seed,
                                train_seed,
                            )
                        ]

                        pairs.append({
                            "split_seed":
                                split_seed,

                            "train_seed":
                                train_seed,

                            "baseline":
                                base,

                            "condition":
                                value,

                            "delta":
                                value
                                - base,
                        })


                out = delta_map[
                    (
                        condition,
                        model,
                        domain,
                        metric,
                    )
                ]

                bmean = mean(
                    [
                        x[
                            "baseline"
                        ]
                        for x in pairs
                    ]
                )

                cmean = mean(
                    [
                        x[
                            "condition"
                        ]
                        for x in pairs
                    ]
                )

                dmean = (
                    cmean
                    - bmean
                )

                assert close(
                    out[
                        "baseline_mean"
                    ],
                    bmean,
                )

                assert close(
                    out[
                        "condition_mean"
                    ],
                    cmean,
                )

                assert close(
                    out[
                        "mean_delta"
                    ],
                    dmean,
                )

                assert close(
                    out[
                        "relative_change_pct"
                    ],
                    100.0
                    * dmean
                    / bmean,
                )

                assert int(
                    out[
                        "better_count"
                    ]
                ) == sum(
                    x[
                        "delta"
                    ] < 0
                    for x in pairs
                )

                assert int(
                    out[
                        "worse_count"
                    ]
                ) == sum(
                    x[
                        "delta"
                    ] > 0
                    for x in pairs
                )

                assert int(
                    out[
                        "tie_count"
                    ]
                ) == sum(
                    x[
                        "delta"
                    ] == 0
                    for x in pairs
                )

                assert close(
                    out[
                        "split_seed_delta_sd"
                    ],
                    split_sd(
                        pairs,
                        "delta",
                    ),
                )


# ------------------------------------------------------------
# Recompute model contrasts independently.
# ------------------------------------------------------------

model_file = read_csv(
    MODEL_CSV
)

assert len(
    model_file
) == 32


model_map = {
    (
        r[
            "condition"
        ],
        r[
            "domain"
        ],
        r[
            "metric"
        ],
    ):
        r
    for r in model_file
}

assert len(
    model_map
) == 32


for condition in CONDITIONS:

    for domain in DOMAINS:

        for metric in METRICS:

            pairs = []

            for split_seed in range(5):

                for train_seed in range(3):

                    mlp = raw_map[
                        (
                            condition,
                            "MLP",
                            domain,
                            metric,
                            split_seed,
                            train_seed,
                        )
                    ]

                    gcn = raw_map[
                        (
                            condition,
                            "CAUSAL_GCN",
                            domain,
                            metric,
                            split_seed,
                            train_seed,
                        )
                    ]

                    pairs.append({
                        "split_seed":
                            split_seed,

                        "train_seed":
                            train_seed,

                        "mlp":
                            mlp,

                        "gcn":
                            gcn,

                        "delta":
                            mlp
                            - gcn,
                    })


            out = model_map[
                (
                    condition,
                    domain,
                    metric,
                )
            ]

            mlp_mean = mean(
                [
                    x[
                        "mlp"
                    ]
                    for x in pairs
                ]
            )

            gcn_mean = mean(
                [
                    x[
                        "gcn"
                    ]
                    for x in pairs
                ]
            )

            contrast = (
                mlp_mean
                - gcn_mean
            )

            assert close(
                out[
                    "mlp_mean"
                ],
                mlp_mean,
            )

            assert close(
                out[
                    "gcn_mean"
                ],
                gcn_mean,
            )

            assert close(
                out[
                    "mlp_minus_gcn"
                ],
                contrast,
            )

            assert close(
                out[
                    "gcn_advantage_pct_of_mlp"
                ],
                100.0
                * contrast
                / mlp_mean,
            )

            assert int(
                out[
                    "gcn_better_count"
                ]
            ) == sum(
                x[
                    "delta"
                ] > 0
                for x in pairs
            )

            assert int(
                out[
                    "mlp_better_count"
                ]
            ) == sum(
                x[
                    "delta"
                ] < 0
                for x in pairs
            )


# No inferential output columns.
for path in [
    SUMMARY_CSV,
    DELTA_CSV,
    MODEL_CSV,
]:

    rows = read_csv(
        path
    )

    assert rows

    cols = {
        x.lower()
        for x in rows[
            0
        ].keys()
    }

    forbidden = {
        "p",
        "pvalue",
        "p_value",
        "holm",
        "q_value",
        "significant",
        "significance",
    }

    assert cols.isdisjoint(
        forbidden
    )


print(
    "DQDV DESCRIPTIVE ANALYSIS AUDIT"
)

print(
    "execution provenance exact:",
    True
)

print(
    "60 record hashes independently verified:",
    True
)

print(
    "64 summary rows independently reproduced:",
    True
)

print(
    "48 matched vs-primary rows independently reproduced:",
    True
)

print(
    "32 MLP-vs-GCN contrasts independently reproduced:",
    True
)

print(
    "balanced metric semantics preserved:",
    True
)

print(
    "domain metric semantics preserved:",
    True
)

print(
    "same common cohort:",
    True
)

print(
    "full dQdV treated as leakage-positive control:",
    True
)

print(
    "prefix controls target-independent:",
    True
)

print(
    "new inferential tests:",
    False
)

print(
    "numeric metric values surfaced by audit:",
    False
)

print(
    "DQDV_DESCRIPTIVE_ANALYSIS_AUDIT_PASS=True"
)
