from pathlib import Path
import hashlib
import json
import math
import os
import subprocess

import numpy as np
import pandas as pd


BASE = Path.home() / "alper/battery_revision_2026"

REPO = (
    BASE
    / "repo/battery-gnn"
)

ROOT = (
    BASE
    / "results/revision_2026/prospective_rul"
)

OUT = (
    ROOT
    / "data_v1"
)

PROTOCOL = (
    ROOT
    / "prospective_rul_protocol_v1.json"
)

LABELS = (
    BASE
    / "data/processed/revision_2026"
    / "unified_label_backbone.csv"
)

TASKS = (
    BASE
    / "data/processed/revision_2026/canonical_tasks"
    / "canonical_cycle_supervised_table.csv"
)

SOURCE_ASSIGN = (
    BASE
    / "data/processed/revision_2026/splits_and_scalers"
    / "cell_split_assignments.csv"
)

SOURCE_SPLIT_MANIFEST = (
    BASE
    / "data/processed/revision_2026/splits_and_scalers"
    / "grouped_split_scaler_manifest.json"
)

SELF = (
    REPO
    / "scripts/revision_2026"
    / "103_prospective_rul_target_split_scaler_builder.py"
)


TARGET_OUT = (
    OUT
    / "prospective_rul_target_table.csv"
)

ASSIGN_OUT = (
    OUT
    / "inherited_cell_split_assignments.csv"
)

FEATURE_SCALER_OUT = (
    OUT
    / "train_only_feature_scaler_parameters.csv"
)

TARGET_SCALER_OUT = (
    OUT
    / "train_only_target_scaler_parameters.csv"
)

SUMMARY_OUT = (
    OUT
    / "train_only_scaler_fit_summary.csv"
)

MANIFEST_OUT = (
    OUT
    / "prospective_rul_data_manifest.json"
)


EXPECTED_PROTOCOL_SHA = (
    "2f61fd329008286bca986411a88945c3ac0e5df09aeffab48939bdb92a9f4592"
)

EXPECTED_ASSIGN_SHA = (
    "914c8b90131f496ca396f63e0502147ca99848fdfd4b2c6b2a5887225b47d055"
)

EXPECTED_SPLIT_MANIFEST_SHA = (
    "91aee27fa332e053454c4f2300c367dc363b938e5ab3d12513372370ec4f2238"
)


FEATURES = [
    "v_min",
    "v_q10",
    "v_q25",
    "v_median",
    "v_q75",
    "v_q90",
    "v_max",
    "v_mean",
    "v_std",
    "v_phase_slope",
    "i_min",
    "i_q10",
    "i_q25",
    "i_median",
    "i_q75",
    "i_q90",
    "i_max",
    "i_mean",
    "i_std",
    "i_phase_slope",
]

PRIMARY_DATASETS = [
    "CALCE",
    "MATR",
]

SEEDS = [
    0, 1, 2, 3, 4
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


def atomic_csv(
    frame,
    path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(path) + ".tmp"
    )

    frame.to_csv(
        tmp,
        index=False,
    )

    os.replace(
        tmp,
        path,
    )


def atomic_json(
    obj,
    path,
):

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


def boolish(series):

    if series.dtype == bool:

        return series.astype(bool)

    out = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
    )

    assert out.notna().all()

    return out.astype(bool)


def scaler_stats(x):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    assert len(x) > 0
    assert np.isfinite(x).all()

    mean = float(
        np.mean(
            x,
            dtype=np.float64,
        )
    )

    variance = float(
        np.var(
            x,
            ddof=0,
            dtype=np.float64,
        )
    )

    assert math.isfinite(mean)
    assert math.isfinite(variance)
    assert variance >= 0

    zero = bool(
        variance == 0.0
    )

    scale = (
        1.0
        if zero
        else float(
            math.sqrt(
                variance
            )
        )
    )

    return (
        mean,
        variance,
        scale,
        zero,
    )


# ------------------------------------------------------------
# Immutable provenance
# ------------------------------------------------------------

assert sha256(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

assert sha256(
    SOURCE_ASSIGN
) == EXPECTED_ASSIGN_SHA

assert sha256(
    SOURCE_SPLIT_MANIFEST
) == EXPECTED_SPLIT_MANIFEST_SHA


p = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)

assert p[
    "status"
] == "FROZEN_BEFORE_TARGET_BUILDING_OR_MODEL_TRAINING"

assert p[
    "scientific_scope"
][
    "primary_datasets"
] == PRIMARY_DATASETS

assert p[
    "endpoint_contract"
][
    "required_consecutive_observations"
] == 2

assert p[
    "endpoint_contract"
][
    "threshold"
] == 0.80

assert p[
    "input_contract"
][
    "input_eligibility"
] == "feature_valid"

assert p[
    "input_contract"
][
    "existing_rul_task_eligible_used"
] is False


for name, artifact in p[
    "source_artifacts"
].items():

    path = Path(
        artifact[
            "path"
        ]
    )

    assert path.is_file()

    assert sha256(
        path
    ) == artifact[
        "sha256"
    ]


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


# ------------------------------------------------------------
# Load canonical source artifacts.
# ------------------------------------------------------------

labels = pd.read_csv(
    LABELS,
    low_memory=False,
)

tasks = pd.read_csv(
    TASKS,
    low_memory=False,
)

assign = pd.read_csv(
    SOURCE_ASSIGN,
    low_memory=False,
)


required_task_columns = {
    "dataset",
    "cell_id",
    "observation_index",
    "SOH",
    "feature_valid",
    *FEATURES,
}

assert required_task_columns.issubset(
    tasks.columns
)

required_label_columns = {
    "dataset",
    "cell_id",
    "observation_index",
    "SOH",
}

assert required_label_columns.issubset(
    labels.columns
)

required_assign_columns = {
    "split_seed",
    "split_group",
    "dataset",
    "cell_id",
    "censoring_status",
    "split",
    "split_contract_version",
}

assert required_assign_columns.issubset(
    assign.columns
)


labels = labels[
    labels[
        "dataset"
    ].isin(
        PRIMARY_DATASETS
    )
].copy()

tasks = tasks[
    tasks[
        "dataset"
    ].isin(
        PRIMARY_DATASETS
    )
].copy()


labels[
    "observation_index"
] = pd.to_numeric(
    labels[
        "observation_index"
    ],
    errors="raise",
).astype(np.int64)

labels[
    "SOH"
] = pd.to_numeric(
    labels[
        "SOH"
    ],
    errors="coerce",
)

tasks[
    "observation_index"
] = pd.to_numeric(
    tasks[
        "observation_index"
    ],
    errors="raise",
).astype(np.int64)

tasks[
    "feature_valid"
] = boolish(
    tasks[
        "feature_valid"
    ]
)


assert not labels.duplicated(
    [
        "dataset",
        "cell_id",
        "observation_index",
    ]
).any()

assert not tasks.duplicated(
    [
        "dataset",
        "cell_id",
        "observation_index",
    ]
).any()


# ------------------------------------------------------------
# Harmonized k2 EOL.
# ------------------------------------------------------------

eol_map = {}


for (
    dataset,
    cell_id,
), g in labels.groupby(
    [
        "dataset",
        "cell_id",
    ],
    sort=False,
):

    g = (
        g.sort_values(
            "observation_index",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    obs = g[
        "observation_index"
    ].to_numpy(
        dtype=np.int64
    )

    soh = g[
        "SOH"
    ].to_numpy(
        dtype=np.float64
    )

    low = (
        np.isfinite(
            soh
        )
        & (
            soh <= 0.80
        )
    )

    first = None

    for i in range(
        len(low) - 1
    ):

        if not (
            bool(
                low[i]
            )
            and bool(
                low[i + 1]
            )
        ):
            continue

        if int(
            obs[
                i + 1
            ]
        ) != (
            int(
                obs[
                    i
                ]
            )
            + 1
        ):
            continue

        first = int(
            obs[
                i
            ]
        )

        break


    if first is not None:

        eol_map[
            (
                str(
                    dataset
                ),
                str(
                    cell_id
                ),
            )
        ] = first


calce_cells = {
    key[
        1
    ]
    for key in eol_map
    if key[
        0
    ] == "CALCE"
}

matr_cells = {
    key[
        1
    ]
    for key in eol_map
    if key[
        0
    ] == "MATR"
}

assert len(
    calce_cells
) == 13

assert len(
    matr_cells
) == 88


# ------------------------------------------------------------
# Exact-RUL target table.
# ------------------------------------------------------------

tasks[
    "_key"
] = list(
    zip(
        tasks[
            "dataset"
        ].astype(str),
        tasks[
            "cell_id"
        ].astype(str),
    )
)

tasks[
    "_harmonized_eol"
] = tasks[
    "_key"
].map(
    eol_map
)

mask = (
    tasks[
        "feature_valid"
    ]
    & tasks[
        "_harmonized_eol"
    ].notna()
    & (
        tasks[
            "observation_index"
        ]
        <= tasks[
            "_harmonized_eol"
        ]
    )
)

target = tasks[
    mask
].copy()


target[
    "harmonized_eol_observation_index"
] = target[
    "_harmonized_eol"
].astype(
    np.int64
)

target[
    "RUL_observations_80pct_k2"
] = (
    target[
        "harmonized_eol_observation_index"
    ]
    - target[
        "observation_index"
    ]
).astype(
    np.int64
)


assert (
    target[
        "RUL_observations_80pct_k2"
    ]
    >= 0
).all()


target = target.sort_values(
    [
        "dataset",
        "cell_id",
        "observation_index",
    ],
    kind="stable",
).reset_index(
    drop=True
)


target[
    "feature_valid_rank_one_based"
] = (
    target.groupby(
        [
            "dataset",
            "cell_id",
        ],
        sort=False,
    )
    .cumcount()
    + 1
)


assert int(
    (
        target[
            "dataset"
        ]
        == "CALCE"
    ).sum()
) == 7355

assert int(
    (
        target[
            "dataset"
        ]
        == "MATR"
    ).sum()
) == 56814

assert len(
    target
) == 64169


eol_rows = target[
    target[
        "RUL_observations_80pct_k2"
    ]
    == 0
]

assert len(
    eol_rows
) == 101

assert eol_rows[
    [
        "dataset",
        "cell_id",
    ]
].drop_duplicates().shape[
    0
] == 101


target_columns = [
    "dataset",
    "cell_id",
    "observation_index",
    "harmonized_eol_observation_index",
    "RUL_observations_80pct_k2",
    "feature_valid_rank_one_based",
] + FEATURES


target_out = target[
    target_columns
].copy()


for feature in FEATURES:

    target_out[
        feature
    ] = pd.to_numeric(
        target_out[
            feature
        ],
        errors="raise",
    )

    assert np.isfinite(
        target_out[
            feature
        ].to_numpy(
            dtype=np.float64
        )
    ).all()


# ------------------------------------------------------------
# Inherit frozen complete-cell assignments.
# ------------------------------------------------------------

assign[
    "split_seed"
] = pd.to_numeric(
    assign[
        "split_seed"
    ],
    errors="raise",
).astype(int)

assert set(
    assign[
        "split_seed"
    ].unique()
) == set(
    SEEDS
)

assert set(
    assign[
        "split"
    ].astype(str)
    .unique()
) == {
    "train",
    "validation",
    "test",
}


event_keys = set(
    zip(
        target_out[
            "dataset"
        ].astype(str),
        target_out[
            "cell_id"
        ].astype(str),
    )
)


assignment_subset = assign[
    [
        (
            str(dataset),
            str(cell),
        )
        in event_keys
        for dataset, cell in zip(
            assign[
                "dataset"
            ],
            assign[
                "cell_id"
            ],
        )
    ]
].copy()


assignment_subset = assignment_subset.sort_values(
    [
        "split_seed",
        "dataset",
        "cell_id",
    ],
    kind="stable",
).reset_index(
    drop=True
)


assert len(
    assignment_subset
) == 505

assert assignment_subset[
    [
        "dataset",
        "cell_id",
    ]
].drop_duplicates().shape[
    0
] == 101


for seed in SEEDS:

    g = assignment_subset[
        assignment_subset[
            "split_seed"
        ]
        == seed
    ]

    assert len(
        g
    ) == 101

    assert not g.duplicated(
        [
            "dataset",
            "cell_id",
        ]
    ).any()

    expected = {
        (
            "CALCE",
            "train",
        ):
            9,

        (
            "CALCE",
            "validation",
        ):
            2,

        (
            "CALCE",
            "test",
        ):
            2,

        (
            "MATR",
            "train",
        ):
            62,

        (
            "MATR",
            "validation",
        ):
            13,

        (
            "MATR",
            "test",
        ):
            13,
    }

    actual = {
        (
            str(k[0]),
            str(k[1]),
        ):
            int(v)
        for k, v in (
            g.groupby(
                [
                    "dataset",
                    "split",
                ]
            )
            .size()
            .to_dict()
            .items()
        )
    }

    assert actual == expected


assignment_out_columns = [
    "split_seed",
    "split_group",
    "dataset",
    "cell_id",
    "censoring_status",
    "split",
    "split_contract_version",
]

assignment_out = assignment_subset[
    assignment_out_columns
].copy()


# ------------------------------------------------------------
# Fit prospective-RUL train-only feature and target scalers.
# ------------------------------------------------------------

feature_scaler_rows = []
target_scaler_rows = []
summary_rows = []
seed_row_counts = {}


for seed in SEEDS:

    a = assignment_out[
        assignment_out[
            "split_seed"
        ]
        == seed
    ][
        [
            "dataset",
            "cell_id",
            "split",
        ]
    ].copy()


    joined = target_out.merge(
        a,
        on=[
            "dataset",
            "cell_id",
        ],
        how="left",
        validate="many_to_one",
    )

    assert joined[
        "split"
    ].notna().all()


    split_rows = {
        split:
            int(
                (
                    joined[
                        "split"
                    ]
                    == split
                ).sum()
            )
        for split in [
            "train",
            "validation",
            "test",
        ]
    }

    split_cells = {
        split:
            int(
                joined.loc[
                    joined[
                        "split"
                    ]
                    == split,
                    [
                        "dataset",
                        "cell_id",
                    ],
                ]
                .drop_duplicates()
                .shape[
                    0
                ]
            )
        for split in [
            "train",
            "validation",
            "test",
        ]
    }


    assert split_cells == {
        "train":
            71,

        "validation":
            15,

        "test":
            15,
    }

    assert sum(
        split_rows.values()
    ) == len(
        target_out
    )


    train = joined[
        joined[
            "split"
        ]
        == "train"
    ].copy()

    assert len(
        train
    ) > 0

    assert train[
        [
            "dataset",
            "cell_id",
        ]
    ].drop_duplicates().shape[
        0
    ] == 71


    zero_features = 0


    for feature in FEATURES:

        (
            mean,
            variance,
            scale,
            zero,
        ) = scaler_stats(
            train[
                feature
            ].to_numpy(
                dtype=np.float64
            )
        )

        zero_features += int(
            zero
        )

        feature_scaler_rows.append({
            "split_seed":
                seed,

            "scaler_scope":
                "pooled_prospective_rul",

            "feature":
                feature,

            "mean":
                mean,

            "variance":
                variance,

            "scale":
                scale,

            "zero_variance":
                zero,

            "n_train_rows":
                len(
                    train
                ),

            "n_train_cells":
                71,

            "fit_split":
                "train_only",
        })


    (
        target_mean,
        target_variance,
        target_scale,
        target_zero,
    ) = scaler_stats(
        train[
            "RUL_observations_80pct_k2"
        ].to_numpy(
            dtype=np.float64
        )
    )


    target_scaler_rows.append({
        "split_seed":
            seed,

        "scaler_scope":
            "prospective_rul_target",

        "target":
            "RUL_observations_80pct_k2",

        "mean":
            target_mean,

        "variance":
            target_variance,

        "scale":
            target_scale,

        "zero_variance":
            target_zero,

        "n_train_rows":
            len(
                train
            ),

        "n_train_cells":
            71,

        "fit_split":
            "train_only",
    })


    summary_rows.append({
        "split_seed":
            seed,

        "scaler_scope":
            "pooled_prospective_rul_features",

        "n_train_rows":
            len(
                train
            ),

        "n_train_cells":
            71,

        "zero_variance_features":
            zero_features,

        "fit_split":
            "train_only",
    })


    summary_rows.append({
        "split_seed":
            seed,

        "scaler_scope":
            "prospective_rul_target",

        "n_train_rows":
            len(
                train
            ),

        "n_train_cells":
            71,

        "zero_variance_features":
            int(
                target_zero
            ),

        "fit_split":
            "train_only",
    })


    seed_row_counts[
        str(
            seed
        )
    ] = {
        "train_rows":
            split_rows[
                "train"
            ],

        "validation_rows":
            split_rows[
                "validation"
            ],

        "test_rows":
            split_rows[
                "test"
            ],

        "train_cells":
            split_cells[
                "train"
            ],

        "validation_cells":
            split_cells[
                "validation"
            ],

        "test_cells":
            split_cells[
                "test"
            ],
    }


feature_scaler = pd.DataFrame(
    feature_scaler_rows
)

target_scaler = pd.DataFrame(
    target_scaler_rows
)

summary = pd.DataFrame(
    summary_rows
)


assert len(
    feature_scaler
) == 100

assert len(
    target_scaler
) == 5

assert len(
    summary
) == 10

assert set(
    feature_scaler[
        "fit_split"
    ]
) == {
    "train_only"
}

assert set(
    target_scaler[
        "fit_split"
    ]
) == {
    "train_only"
}


# ------------------------------------------------------------
# Write immutable outputs.
# ------------------------------------------------------------

atomic_csv(
    target_out,
    TARGET_OUT,
)

atomic_csv(
    assignment_out,
    ASSIGN_OUT,
)

atomic_csv(
    feature_scaler,
    FEATURE_SCALER_OUT,
)

atomic_csv(
    target_scaler,
    TARGET_SCALER_OUT,
)

atomic_csv(
    summary,
    SUMMARY_OUT,
)


manifest = {
    "contract":
        "prospective_rul_data_v1",

    "status":
        "COMPLETE_PENDING_INDEPENDENT_AUDIT",

    "protocol_sha256":
        EXPECTED_PROTOCOL_SHA,

    "source_split_assignment_sha256":
        EXPECTED_ASSIGN_SHA,

    "source_split_manifest_sha256":
        EXPECTED_SPLIT_MANIFEST_SHA,

    "primary_datasets":
        PRIMARY_DATASETS,

    "endpoint": {
        "SOH_threshold":
            0.80,

        "required_consecutive_observations":
            2,

        "event_cells": {
            "CALCE":
                13,

            "MATR":
                88,
        },
    },

    "target_contract": {
        "name":
            "RUL_observations_80pct_k2",

        "units":
            "observation cycles",

        "zero_at_eol":
            True,

        "post_eol_rows_present":
            False,

        "censored_cells_present":
            False,

        "HUST_rows_present":
            False,

        "total_rows":
            len(
                target_out
            ),

        "rows_by_dataset": {
            str(k):
                int(v)
            for k, v in (
                target_out[
                    "dataset"
                ]
                .value_counts()
                .to_dict()
                .items()
            )
        },

        "event_cell_count":
            101,
    },

    "split_contract": {
        "source_assignments_reused_exactly":
            True,

        "reassignment_performed":
            False,

        "assignment_rows":
            len(
                assignment_out
            ),

        "seeds":
            SEEDS,

        "cells_per_seed":
            101,

        "cells_by_split_per_seed": {
            "train":
                71,

            "validation":
                15,

            "test":
                15,
        },

        "row_counts_by_seed":
            seed_row_counts,
    },

    "scaler_contract": {
        "feature_scaler_scope":
            "pooled_prospective_rul",

        "target_scaler_scope":
            "prospective_rul_target",

        "feature_count":
            len(
                FEATURES
            ),

        "feature_scalers_fit_on":
            "prospective-RUL train rows only",

        "target_scalers_fit_on":
            "prospective-RUL train targets only",

        "validation_statistics_used":
            False,

        "test_statistics_used":
            False,

        "SOH_task_scaler_reused":
            False,
    },

    "early_checkpoint_support": {
        "feature_valid_rank_one_based_materialized":
            True,

        "future_lifetime_fraction_required":
            False,

        "checkpoints":
            [
                25,
                50,
                100,
            ],
    },

    "outputs": {
        "prospective_rul_target_table": {
            "path":
                str(
                    TARGET_OUT
                ),

            "sha256":
                sha256(
                    TARGET_OUT
                ),
        },

        "inherited_cell_split_assignments": {
            "path":
                str(
                    ASSIGN_OUT
                ),

            "sha256":
                sha256(
                    ASSIGN_OUT
                ),
        },

        "train_only_feature_scaler_parameters": {
            "path":
                str(
                    FEATURE_SCALER_OUT
                ),

            "sha256":
                sha256(
                    FEATURE_SCALER_OUT
                ),
        },

        "train_only_target_scaler_parameters": {
            "path":
                str(
                    TARGET_SCALER_OUT
                ),

            "sha256":
                sha256(
                    TARGET_SCALER_OUT
                ),
        },

        "train_only_scaler_fit_summary": {
            "path":
                str(
                    SUMMARY_OUT
                ),

            "sha256":
                sha256(
                    SUMMARY_OUT
                ),
        },
    },

    "builder": {
        "path":
            str(
                SELF
            ),

        "sha256":
            sha256(
                SELF
            ),

        "commit":
            subprocess.check_output(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ],
                cwd=REPO,
                text=True,
            ).strip(),
    },

    "execution_state": {
        "model_training_performed":
            False,

        "test_metrics_read":
            False,

        "test_metrics_surfaced":
            False,
    },
}


atomic_json(
    manifest,
    MANIFEST_OUT,
)


print(
    "PROSPECTIVE RUL DATA BUILD"
)

print(
    "target rows:",
    len(
        target_out
    )
)

print(
    "CALCE rows:",
    int(
        (
            target_out[
                "dataset"
            ]
            == "CALCE"
        ).sum()
    )
)

print(
    "MATR rows:",
    int(
        (
            target_out[
                "dataset"
            ]
            == "MATR"
        ).sum()
    )
)

print(
    "event cells:",
    101
)

print(
    "inherited assignment rows:",
    len(
        assignment_out
    )
)

for seed in SEEDS:

    x = seed_row_counts[
        str(
            seed
        )
    ]

    print(
        f"seed{seed}: "
        f"train_rows={x['train_rows']} "
        f"validation_rows={x['validation_rows']} "
        f"test_rows={x['test_rows']} "
        f"cells=71/15/15"
    )

print(
    "feature scaler rows:",
    len(
        feature_scaler
    )
)

print(
    "target scaler rows:",
    len(
        target_scaler
    )
)

print(
    "SOH scaler reused:",
    False
)

print(
    "model training performed:",
    False
)

print(
    "test metrics read:",
    False
)

print(
    "data_manifest_sha256:",
    sha256(
        MANIFEST_OUT
    )
)

print(
    "PROSPECTIVE_RUL_DATA_BUILD_COMPLETE_READY_FOR_AUDIT=True"
)
