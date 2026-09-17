from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess


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

PROTOCOL = (
    ROOT
    / "training_v2/dqdv_training_protocol_v2.json"
)

EXECUTION_MANIFEST = (
    EXEC
    / "execution_manifest.json"
)

METADATA_RECEIPT = (
    EXEC
    / "execution_manifest_metadata_correction.json"
)

EXECUTION_AUDIT = (
    REPO
    / "scripts/revision_2026/96_final_dqdv_training_v2_execution_audit.py"
)

SELF = (
    REPO
    / "scripts/revision_2026/99_dqdv_descriptive_analysis.py"
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

ANALYSIS_MANIFEST = (
    OUT
    / "dqdv_analysis_manifest.json"
)


EXPECTED_PROTOCOL_SHA = (
    "4612a450eeaf909a04c31be16957bdc0567aac67635241e14d7bbceea507fe9d"
)

EXPECTED_EXECUTION_MANIFEST_SHA = (
    "744ae4a695f9a94a2c597469dc867c58141ea94fc5d131074d88574a25b72725"
)

EXPECTED_EXECUTION_AUDIT_SHA = (
    "086b9374441055b386de2007af0ae2041ff5707508d9fdad246e5aee7a27a658"
)

EXPECTED_METADATA_RECEIPT_SHA = (
    "ebc46e8d1eb265e8850f42c9af30f91246f3ccb56f84cb86a4f5a5a9cd1ad343"
)


CONDITIONS = [
    "PRIMARY20_COMMON",
    "PRIMARY20_PLUS_FULL_DQDV10",
    "PRIMARY20_PLUS_PREFIX25_DQDV10",
    "PRIMARY20_PLUS_PREFIX50_DQDV10",
]

CONDITION_ROLES = {
    "PRIMARY20_COMMON":
        "primary20_common_cohort_baseline",

    "PRIMARY20_PLUS_FULL_DQDV10":
        "full_curve_leakage_positive_control",

    "PRIMARY20_PLUS_PREFIX25_DQDV10":
        "target_independent_training_derived_prefix25",

    "PRIMARY20_PLUS_PREFIX50_DQDV10":
        "target_independent_training_derived_prefix50",
}

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

COND_ORDER = {
    x: i
    for i, x in enumerate(CONDITIONS)
}

MODEL_ORDER = {
    x: i
    for i, x in enumerate(MODELS)
}

DOMAIN_ORDER = {
    x: i
    for i, x in enumerate(DOMAINS)
}

METRIC_ORDER = {
    x: i
    for i, x in enumerate(METRICS)
}


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


def atomic_json(obj, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(path) + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def write_csv(path, fieldnames, rows):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(path) + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        w.writeheader()

        for row in rows:
            w.writerow(row)

    os.replace(
        tmp,
        path,
    )


def mean(xs):

    xs = [
        float(x)
        for x in xs
    ]

    assert xs

    assert all(
        math.isfinite(x)
        for x in xs
    )

    return float(
        statistics.mean(xs)
    )


def split_sd(rows, value_key):

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
                    value_key
                ]
            )
        )

    assert set(
        by_split
    ) == {
        0, 1, 2, 3, 4
    }

    split_means = [
        mean(
            by_split[s]
        )
        for s in sorted(by_split)
    ]

    assert len(
        split_means
    ) == 5

    return float(
        statistics.stdev(
            split_means
        )
    )


# ------------------------------------------------------------
# Immutable input provenance
# ------------------------------------------------------------

assert sha256(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

assert sha256(
    EXECUTION_MANIFEST
) == EXPECTED_EXECUTION_MANIFEST_SHA

assert sha256(
    EXECUTION_AUDIT
) == EXPECTED_EXECUTION_AUDIT_SHA

assert sha256(
    METADATA_RECEIPT
) == EXPECTED_METADATA_RECEIPT_SHA


rel = str(
    SELF.relative_to(
        REPO
    )
)

subprocess.check_call(
    [
        "git",
        "ls-files",
        "--error-unmatch",
        rel,
    ],
    cwd=REPO,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

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


execution_manifest = json.loads(
    EXECUTION_MANIFEST.read_text(
        encoding="utf-8"
    )
)

assert execution_manifest[
    "status"
] == "COMPLETE_PENDING_INDEPENDENT_AUDIT"

assert execution_manifest[
    "completed_pair_count"
] == 60

assert execution_manifest[
    "completed_model_fit_count"
] == 120

assert execution_manifest[
    "all_fits_patience_valid"
] is True

assert execution_manifest[
    "all_fits_below_epoch_ceiling"
] is True

assert execution_manifest[
    "epoch_ceiling"
] == 10000


# ------------------------------------------------------------
# Read only independently audited record metrics.
# ------------------------------------------------------------

summaries = execution_manifest[
    "records"
]

assert len(
    summaries
) == 60


records = []

record_provenance = []

for item in summaries:

    path = Path(
        item[
            "record_path"
        ]
    )

    assert path.is_file()

    assert sha256(
        path
    ) == item[
        "record_sha256"
    ]

    r = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    records.append(
        r
    )

    record_provenance.append({
        "path":
            str(
                path
            ),

        "sha256":
            item[
                "record_sha256"
            ],
    })


assert len(
    records
) == 60

assert {
    int(r["split_seed"])
    for r in records
} == {
    0, 1, 2, 3, 4
}

assert {
    int(r["train_seed"])
    for r in records
} == {
    0, 1, 2
}

for condition in CONDITIONS:

    assert sum(
        r[
            "condition"
        ] == condition
        for r in records
    ) == 15


# ------------------------------------------------------------
# Long-form audited test metrics.
#
# BALANCED_ALL:
#   audited hierarchical weighted metric.
#
# HUST/CALCE/MATR:
#   plain row-level unweighted dataset metric.
# ------------------------------------------------------------

raw = []


for r in records:

    condition = str(
        r[
            "condition"
        ]
    )

    assert condition in CONDITIONS

    expected_dim = (
        20
        if condition == "PRIMARY20_COMMON"
        else 30
    )

    assert int(
        r[
            "input_dimension"
        ]
    ) == expected_dim


    for model in MODELS:

        q = r[
            "models"
        ][
            model
        ]

        test = q[
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


        balanced = test[
            "balanced"
        ]

        for metric in METRICS:

            value = float(
                balanced[
                    metric.lower()
                ]
            )

            assert math.isfinite(
                value
            )

            assert value >= 0

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


        by_dataset = {
            str(row["dataset"]):
                row
            for row in test[
                "by_dataset"
            ]
        }

        assert set(
            by_dataset
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

            row = by_dataset[
                domain
            ]

            for metric in METRICS:

                value = float(
                    row[
                        metric
                    ]
                )

                assert math.isfinite(
                    value
                )

                assert value >= 0

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


expected_raw_rows = (
    4
    * 2
    * 4
    * 2
    * 15
)

assert len(
    raw
) == expected_raw_rows


# ------------------------------------------------------------
# Condition/model/domain/metric summary.
# Mean is over 15 matched runs.
# Uncertainty descriptor = SD across five split-seed means,
# avoiding treating the three initialization seeds as independent
# held-out partitions.
# ------------------------------------------------------------

groups = defaultdict(list)

for row in raw:

    key = (
        row[
            "condition"
        ],
        row[
            "model"
        ],
        row[
            "domain"
        ],
        row[
            "metric"
        ],
    )

    groups[
        key
    ].append(
        row
    )


summary_rows = []


for key, rows in groups.items():

    condition, model, domain, metric = key

    assert len(
        rows
    ) == 15

    values = [
        float(
            r[
                "value"
            ]
        )
        for r in rows
    ]

    summary_rows.append({
        "condition":
            condition,

        "condition_role":
            CONDITION_ROLES[
                condition
            ],

        "input_dimension":
            (
                20
                if condition
                == "PRIMARY20_COMMON"
                else 30
            ),

        "model":
            model,

        "domain":
            domain,

        "metric":
            metric,

        "n_runs":
            15,

        "n_split_seeds":
            5,

        "n_train_seeds_per_split":
            3,

        "mean":
            mean(
                values
            ),

        "split_seed_mean_sd":
            split_sd(
                rows,
                "value",
            ),

        "minimum":
            min(
                values
            ),

        "maximum":
            max(
                values
            ),
    })


summary_rows.sort(
    key=lambda r: (
        COND_ORDER[
            r[
                "condition"
            ]
        ],
        MODEL_ORDER[
            r[
                "model"
            ]
        ],
        DOMAIN_ORDER[
            r[
                "domain"
            ]
        ],
        METRIC_ORDER[
            r[
                "metric"
            ]
        ],
    )
)

assert len(
    summary_rows
) == 64


# ------------------------------------------------------------
# Descriptive augmented-condition deltas vs PRIMARY20_COMMON.
# Same split seed + train seed + model + domain + metric.
# Negative error delta = augmentation improved that metric.
# No inferential tests.
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


delta_rows = []


for condition in CONDITIONS[1:]:

    for model in MODELS:

        for domain in DOMAINS:

            for metric in METRICS:

                pair_rows = []

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

                        assert base > 0

                        delta = (
                            value
                            - base
                        )

                        pair_rows.append({
                            "split_seed":
                                split_seed,

                            "train_seed":
                                train_seed,

                            "baseline":
                                base,

                            "condition_value":
                                value,

                            "delta":
                                delta,
                        })


                assert len(
                    pair_rows
                ) == 15

                baseline_mean = mean(
                    [
                        x[
                            "baseline"
                        ]
                        for x in pair_rows
                    ]
                )

                condition_mean = mean(
                    [
                        x[
                            "condition_value"
                        ]
                        for x in pair_rows
                    ]
                )

                mean_delta = (
                    condition_mean
                    - baseline_mean
                )

                delta_rows.append({
                    "condition":
                        condition,

                    "condition_role":
                        CONDITION_ROLES[
                            condition
                        ],

                    "model":
                        model,

                    "domain":
                        domain,

                    "metric":
                        metric,

                    "n_matched_pairs":
                        15,

                    "baseline_mean":
                        baseline_mean,

                    "condition_mean":
                        condition_mean,

                    "mean_delta":
                        mean_delta,

                    "relative_change_pct":
                        (
                            100.0
                            * mean_delta
                            / baseline_mean
                        ),

                    "better_count":
                        sum(
                            x[
                                "delta"
                            ] < 0
                            for x in pair_rows
                        ),

                    "worse_count":
                        sum(
                            x[
                                "delta"
                            ] > 0
                            for x in pair_rows
                        ),

                    "tie_count":
                        sum(
                            x[
                                "delta"
                            ] == 0
                            for x in pair_rows
                        ),

                    "split_seed_delta_sd":
                        split_sd(
                            pair_rows,
                            "delta",
                        ),
                })


delta_rows.sort(
    key=lambda r: (
        COND_ORDER[
            r[
                "condition"
            ]
        ],
        MODEL_ORDER[
            r[
                "model"
            ]
        ],
        DOMAIN_ORDER[
            r[
                "domain"
            ]
        ],
        METRIC_ORDER[
            r[
                "metric"
            ]
        ],
    )
)

assert len(
    delta_rows
) == 48


# ------------------------------------------------------------
# Descriptive GCN-vs-MLP contrast.
# Positive MLP-GCN error difference favors GCN.
# ------------------------------------------------------------

model_rows = []


for condition in CONDITIONS:

    for domain in DOMAINS:

        for metric in METRICS:

            pair_rows = []

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

                    assert mlp > 0

                    pair_rows.append({
                        "split_seed":
                            split_seed,

                        "train_seed":
                            train_seed,

                        "mlp":
                            mlp,

                        "gcn":
                            gcn,

                        "delta":
                            mlp - gcn,
                    })


            assert len(
                pair_rows
            ) == 15

            mlp_mean = mean(
                [
                    x[
                        "mlp"
                    ]
                    for x in pair_rows
                ]
            )

            gcn_mean = mean(
                [
                    x[
                        "gcn"
                    ]
                    for x in pair_rows
                ]
            )

            delta = (
                mlp_mean
                - gcn_mean
            )

            model_rows.append({
                "condition":
                    condition,

                "condition_role":
                    CONDITION_ROLES[
                        condition
                    ],

                "domain":
                    domain,

                "metric":
                    metric,

                "n_matched_pairs":
                    15,

                "mlp_mean":
                    mlp_mean,

                "gcn_mean":
                    gcn_mean,

                "mlp_minus_gcn":
                    delta,

                "gcn_advantage_pct_of_mlp":
                    (
                        100.0
                        * delta
                        / mlp_mean
                    ),

                "gcn_better_count":
                    sum(
                        x[
                            "delta"
                        ] > 0
                        for x in pair_rows
                    ),

                "mlp_better_count":
                    sum(
                        x[
                            "delta"
                        ] < 0
                        for x in pair_rows
                    ),

                "tie_count":
                    sum(
                        x[
                            "delta"
                        ] == 0
                        for x in pair_rows
                    ),

                "split_seed_contrast_sd":
                    split_sd(
                        pair_rows,
                        "delta",
                    ),
            })


model_rows.sort(
    key=lambda r: (
        COND_ORDER[
            r[
                "condition"
            ]
        ],
        DOMAIN_ORDER[
            r[
                "domain"
            ]
        ],
        METRIC_ORDER[
            r[
                "metric"
            ]
        ],
    )
)

assert len(
    model_rows
) == 32


# ------------------------------------------------------------
# Write results.
# ------------------------------------------------------------

write_csv(
    SUMMARY_CSV,
    [
        "condition",
        "condition_role",
        "input_dimension",
        "model",
        "domain",
        "metric",
        "n_runs",
        "n_split_seeds",
        "n_train_seeds_per_split",
        "mean",
        "split_seed_mean_sd",
        "minimum",
        "maximum",
    ],
    summary_rows,
)

write_csv(
    DELTA_CSV,
    [
        "condition",
        "condition_role",
        "model",
        "domain",
        "metric",
        "n_matched_pairs",
        "baseline_mean",
        "condition_mean",
        "mean_delta",
        "relative_change_pct",
        "better_count",
        "worse_count",
        "tie_count",
        "split_seed_delta_sd",
    ],
    delta_rows,
)

write_csv(
    MODEL_CSV,
    [
        "condition",
        "condition_role",
        "domain",
        "metric",
        "n_matched_pairs",
        "mlp_mean",
        "gcn_mean",
        "mlp_minus_gcn",
        "gcn_advantage_pct_of_mlp",
        "gcn_better_count",
        "mlp_better_count",
        "tie_count",
        "split_seed_contrast_sd",
    ],
    model_rows,
)


manifest = {
    "contract":
        "dqdv_descriptive_analysis_v1",

    "status":
        "COMPLETE_PENDING_INDEPENDENT_ANALYSIS_AUDIT",

    "execution_manifest_sha256":
        EXPECTED_EXECUTION_MANIFEST_SHA,

    "execution_audit_sha256":
        EXPECTED_EXECUTION_AUDIT_SHA,

    "protocol_sha256":
        EXPECTED_PROTOCOL_SHA,

    "metadata_correction_receipt_sha256":
        EXPECTED_METADATA_RECEIPT_SHA,

    "analysis_script_path":
        str(
            SELF
        ),

    "analysis_script_sha256":
        sha256(
            SELF
        ),

    "analysis_script_commit":
        subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=REPO,
            text=True,
        ).strip(),

    "record_count":
        60,

    "model_fit_count":
        120,

    "conditions":
        CONDITIONS,

    "condition_roles":
        CONDITION_ROLES,

    "models":
        MODELS,

    "domains":
        DOMAINS,

    "metrics":
        METRICS,

    "runs_per_condition":
        15,

    "split_seeds":
        [
            0, 1, 2, 3, 4
        ],

    "training_initialization_seeds":
        [
            0, 1, 2
        ],

    "balanced_metric_semantics":
        (
            "dataset-balanced + cell-balanced hierarchical weighted "
            "test RMSE/MAE inherited from audited execution records"
        ),

    "domain_metric_semantics":
        (
            "plain row-level unweighted dataset-specific test RMSE/MAE "
            "for HUST, CALCE, and MATR inherited from audited execution records"
        ),

    "common_cohort":
        True,

    "feature_dimension_disclosure": {
        "PRIMARY20_COMMON":
            20,

        "all_dqdv_augmented_conditions":
            30,
    },

    "full_dqdv_interpretation":
        (
            "retrospective leakage-positive control; not deployment-valid"
        ),

    "prefix_interpretation":
        (
            "target-independent partial-curve controls using fixed "
            "training-derived time cutoffs"
        ),

    "new_inferential_tests":
        False,

    "p_values_computed":
        False,

    "multiple_testing_correction_computed":
        False,

    "descriptive_only":
        True,

    "model_training_performed":
        False,

    "model_state_reconstruction_performed":
        False,

    "numeric_test_metric_values_written_to_outputs":
        True,

    "numeric_test_metric_values_printed_by_analysis":
        False,

    "summary_row_count":
        len(
            summary_rows
        ),

    "delta_row_count":
        len(
            delta_rows
        ),

    "model_contrast_row_count":
        len(
            model_rows
        ),

    "summary_csv_path":
        str(
            SUMMARY_CSV
        ),

    "summary_csv_sha256":
        sha256(
            SUMMARY_CSV
        ),

    "delta_csv_path":
        str(
            DELTA_CSV
        ),

    "delta_csv_sha256":
        sha256(
            DELTA_CSV
        ),

    "model_contrast_csv_path":
        str(
            MODEL_CSV
        ),

    "model_contrast_csv_sha256":
        sha256(
            MODEL_CSV
        ),

    "input_records":
        record_provenance,
}


atomic_json(
    manifest,
    ANALYSIS_MANIFEST,
)


print(
    "DQDV DESCRIPTIVE ANALYSIS"
)

print(
    "audited input records:",
    60
)

print(
    "summary rows:",
    len(
        summary_rows
    )
)

print(
    "vs-primary rows:",
    len(
        delta_rows
    )
)

print(
    "model-contrast rows:",
    len(
        model_rows
    )
)

print(
    "balanced metric semantics hierarchical weighted:",
    True
)

print(
    "domain metric semantics plain row-level unweighted:",
    True
)

print(
    "same common cohort:",
    True
)

print(
    "new inferential tests:",
    False
)

print(
    "full dQdV leakage-positive control:",
    True
)

print(
    "prefix controls target-independent:",
    True
)

print(
    "feature dimension difference disclosed:",
    True
)

print(
    "numeric test metric values printed:",
    False
)

print(
    "analysis_manifest_sha256:",
    sha256(
        ANALYSIS_MANIFEST
    )
)

print(
    "DQDV_DESCRIPTIVE_ANALYSIS_COMPLETE_READY_FOR_AUDIT=True"
)
