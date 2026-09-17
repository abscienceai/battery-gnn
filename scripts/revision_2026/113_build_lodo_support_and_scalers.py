from pathlib import Path
import hashlib
import json
import math
import os
import subprocess

import numpy as np
import pandas as pd


BASE = Path.home() / "alper/battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

ROOT = BASE / "results/revision_2026/lodo_fewshot"
OUT = ROOT / "data_v1"

PROTOCOL = ROOT / "lodo_fewshot_protocol_v1.json"

TASKS = (
    BASE
    / "data/processed/revision_2026/canonical_tasks"
    / "canonical_cycle_supervised_table.csv"
)

ASSIGN = (
    BASE
    / "data/processed/revision_2026/splits_and_scalers"
    / "cell_split_assignments.csv"
)

SPLIT_MANIFEST = (
    BASE
    / "data/processed/revision_2026/splits_and_scalers"
    / "grouped_split_scaler_manifest.json"
)

SELF = (
    REPO
    / "scripts/revision_2026"
    / "113_build_lodo_support_and_scalers.py"
)

INVENTORY_OUT = OUT / "lodo_fold_inventory.csv"
SCALER_OUT = OUT / "lodo_source_only_feature_scalers.csv"
SCALER_SUMMARY_OUT = OUT / "lodo_source_only_scaler_summary.csv"
SUPPORT_CELLS_OUT = OUT / "fewshot_support_cells.csv"
SUPPORT_SETS_OUT = OUT / "fewshot_support_sets.csv"
MANIFEST_OUT = OUT / "lodo_fewshot_data_manifest.json"


EXPECTED = {
    PROTOCOL:
        "966d8c6b95f29e671ce4be9a785c5088e6d8e7eda9d25b8389f204859641897f",

    TASKS:
        "49a7b7629cf9ee3f8ce0a7e3c52f2807878b752b4c9a30a876285fb891b00c14",

    ASSIGN:
        "914c8b90131f496ca396f63e0502147ca99848fdfd4b2c6b2a5887225b47d055",

    SPLIT_MANIFEST:
        "91aee27fa332e053454c4f2300c367dc363b938e5ab3d12513372370ec4f2238",
}


DATASETS = [
    "HUST",
    "CALCE",
    "MATR",
]

SEEDS = [
    0, 1, 2, 3, 4,
]

SUPPORT_K = [
    1, 3, 5,
]

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


def sha256(path):

    h = hashlib.sha256()

    with Path(path).open("rb") as f:

        while True:

            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def text_sha256(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def atomic_csv(frame, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(str(path) + ".tmp")

    frame.to_csv(
        tmp,
        index=False,
    )

    os.replace(
        tmp,
        path,
    )


def atomic_json(obj, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(str(path) + ".tmp")

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def compact_json(obj):

    return json.dumps(
        obj,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def boolish(series):

    if series.dtype == bool:
        return series.astype(bool)

    x = (
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

    assert x.notna().all()

    return x.astype(bool)


def scaler_stats(values):

    x = np.asarray(
        values,
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
    assert variance >= 0.0

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


def subset_by_cells(frame, cells):

    keys = cells[
        [
            "dataset",
            "cell_id",
        ]
    ].drop_duplicates()

    assert len(keys) == len(cells)

    out = frame.merge(
        keys,
        on=[
            "dataset",
            "cell_id",
        ],
        how="inner",
        validate="many_to_one",
    )

    return out


# ------------------------------------------------------------
# Immutable provenance
# ------------------------------------------------------------

for path, expected in EXPECTED.items():

    assert path.is_file()
    assert sha256(path) == expected


protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)

assert protocol[
    "status"
] == "FROZEN_BEFORE_SUPPORT_SELECTION_SCALER_FIT_OR_MODEL_TRAINING"

assert protocol[
    "few_shot_support_contract"
][
    "support_cell_counts"
] == SUPPORT_K

assert protocol[
    "few_shot_support_contract"
][
    "selection_uses_target_labels"
] is False

assert protocol[
    "few_shot_support_contract"
][
    "selection_uses_target_features"
] is False


rel = str(
    SELF.relative_to(REPO)
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
# Load canonical data with textual cell identity preserved.
# ------------------------------------------------------------

tasks = pd.read_csv(
    TASKS,
    dtype={
        "dataset": "string",
        "cell_id": "string",
    },
    low_memory=False,
)

assign = pd.read_csv(
    ASSIGN,
    dtype={
        "dataset": "string",
        "cell_id": "string",
        "split": "string",
    },
    low_memory=False,
)


tasks[
    "feature_valid"
] = boolish(
    tasks[
        "feature_valid"
    ]
)

tasks[
    "soh_task_eligible"
] = boolish(
    tasks[
        "soh_task_eligible"
    ]
)

tasks[
    "SOH"
] = pd.to_numeric(
    tasks[
        "SOH"
    ],
    errors="raise",
)

assign[
    "split_seed"
] = pd.to_numeric(
    assign[
        "split_seed"
    ],
    errors="raise",
).astype(int)


primary = tasks[
    tasks[
        "dataset"
    ].isin(
        DATASETS
    )
    & tasks[
        "feature_valid"
    ]
    & tasks[
        "soh_task_eligible"
    ]
].copy()


assert len(
    primary
) == 300553

assert primary[
    [
        "dataset",
        "cell_id",
    ]
].drop_duplicates().shape[
    0
] == 259


for feature in FEATURES:

    primary[
        feature
    ] = pd.to_numeric(
        primary[
            feature
        ],
        errors="raise",
    )

    assert np.isfinite(
        primary[
            feature
        ].to_numpy(
            dtype=np.float64
        )
    ).all()


assert np.isfinite(
    primary[
        "SOH"
    ].to_numpy(
        dtype=np.float64
    )
).all()


# ------------------------------------------------------------
# Build all fold/seed artifacts.
# ------------------------------------------------------------

inventory_rows = []
scaler_rows = []
scaler_summary_rows = []
support_cell_rows = []
support_set_rows = []

support_print_rows = []


for held_out in DATASETS:

    source_datasets = [
        d
        for d in DATASETS
        if d != held_out
    ]


    assert source_datasets == protocol[
        "scientific_scope"
    ][
        "folds"
    ][
        held_out
    ]


    for seed in SEEDS:

        a = assign[
            assign[
                "split_seed"
            ]
            == seed
        ].copy()


        source_train_cells = a[
            a[
                "dataset"
            ].isin(
                source_datasets
            )
            & (
                a[
                    "split"
                ]
                == "train"
            )
        ][
            [
                "dataset",
                "cell_id",
            ]
        ].drop_duplicates()


        source_val_cells = a[
            a[
                "dataset"
            ].isin(
                source_datasets
            )
            & (
                a[
                    "split"
                ]
                == "validation"
            )
        ][
            [
                "dataset",
                "cell_id",
            ]
        ].drop_duplicates()


        target_train_cells = a[
            (
                a[
                    "dataset"
                ]
                == held_out
            )
            & (
                a[
                    "split"
                ]
                == "train"
            )
        ][
            [
                "dataset",
                "cell_id",
            ]
        ].drop_duplicates()


        target_val_cells = a[
            (
                a[
                    "dataset"
                ]
                == held_out
            )
            & (
                a[
                    "split"
                ]
                == "validation"
            )
        ][
            [
                "dataset",
                "cell_id",
            ]
        ].drop_duplicates()


        target_test_cells = a[
            (
                a[
                    "dataset"
                ]
                == held_out
            )
            & (
                a[
                    "split"
                ]
                == "test"
            )
        ][
            [
                "dataset",
                "cell_id",
            ]
        ].drop_duplicates()


        target_pool_expected = {
            "HUST":
                54,

            "CALCE":
                9,

            "MATR":
                119,
        }[
            held_out
        ]

        assert len(
            target_train_cells
        ) == target_pool_expected

        assert len(
            target_train_cells
        ) >= 5


        # Cell identity barriers.
        sets = [
            set(
                zip(
                    x[
                        "dataset"
                    ].astype(str),
                    x[
                        "cell_id"
                    ].astype(str),
                )
            )
            for x in [
                source_train_cells,
                source_val_cells,
                target_train_cells,
                target_val_cells,
                target_test_cells,
            ]
        ]

        for i in range(
            len(sets)
        ):

            for j in range(
                i + 1,
                len(sets),
            ):

                assert sets[i].isdisjoint(
                    sets[j]
                )


        source_train = subset_by_cells(
            primary,
            source_train_cells,
        )

        source_val = subset_by_cells(
            primary,
            source_val_cells,
        )

        target_train = subset_by_cells(
            primary,
            target_train_cells,
        )

        target_val = subset_by_cells(
            primary,
            target_val_cells,
        )

        target_test = subset_by_cells(
            primary,
            target_test_cells,
        )


        assert source_train[
            [
                "dataset",
                "cell_id",
            ]
        ].drop_duplicates().shape[
            0
        ] == len(
            source_train_cells
        )

        assert source_val[
            [
                "dataset",
                "cell_id",
            ]
        ].drop_duplicates().shape[
            0
        ] == len(
            source_val_cells
        )

        assert target_train[
            "cell_id"
        ].nunique() == len(
            target_train_cells
        )

        assert target_val[
            "cell_id"
        ].nunique() == len(
            target_val_cells
        )

        assert target_test[
            "cell_id"
        ].nunique() == len(
            target_test_cells
        )


        source_train_rows_by_dataset = {
            d:
                int(
                    (
                        source_train[
                            "dataset"
                        ]
                        == d
                    ).sum()
                )
            for d in source_datasets
        }

        source_train_cells_by_dataset = {
            d:
                int(
                    source_train.loc[
                        source_train[
                            "dataset"
                        ]
                        == d,
                        "cell_id",
                    ].nunique()
                )
            for d in source_datasets
        }


        inventory_rows.append({
            "held_out":
                held_out,

            "split_seed":
                seed,

            "source_datasets_json":
                compact_json(
                    source_datasets
                ),

            "source_train_rows":
                len(
                    source_train
                ),

            "source_train_cells":
                len(
                    source_train_cells
                ),

            "source_validation_rows":
                len(
                    source_val
                ),

            "source_validation_cells":
                len(
                    source_val_cells
                ),

            "target_train_rows":
                len(
                    target_train
                ),

            "target_train_cells":
                len(
                    target_train_cells
                ),

            "target_validation_rows":
                len(
                    target_val
                ),

            "target_validation_cells":
                len(
                    target_val_cells
                ),

            "target_test_rows":
                len(
                    target_test
                ),

            "target_test_cells":
                len(
                    target_test_cells
                ),

            "source_train_rows_by_dataset_json":
                compact_json(
                    source_train_rows_by_dataset
                ),

            "source_train_cells_by_dataset_json":
                compact_json(
                    source_train_cells_by_dataset
                ),
        })


        # ----------------------------------------------------
        # Source-train-only scaler.
        # ----------------------------------------------------

        zero_count = 0


        for feature in FEATURES:

            (
                mean,
                variance,
                scale,
                zero,
            ) = scaler_stats(
                source_train[
                    feature
                ].to_numpy(
                    dtype=np.float64
                )
            )

            zero_count += int(
                zero
            )


            scaler_rows.append({
                "held_out":
                    held_out,

                "split_seed":
                    seed,

                "scaler_scope":
                    "LODO_SOURCE_TRAIN_ONLY",

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

                "n_source_train_rows":
                    len(
                        source_train
                    ),

                "n_source_train_cells":
                    len(
                        source_train_cells
                    ),

                "fit_split":
                    "source_train_only",

                "target_statistics_used":
                    False,
            })


        scaler_summary_rows.append({
            "held_out":
                held_out,

            "split_seed":
                seed,

            "source_datasets_json":
                compact_json(
                    source_datasets
                ),

            "n_source_train_rows":
                len(
                    source_train
                ),

            "n_source_train_cells":
                len(
                    source_train_cells
                ),

            "zero_variance_features":
                zero_count,

            "fit_split":
                "source_train_only",

            "target_train_statistics_used":
                False,

            "target_validation_statistics_used":
                False,

            "target_test_statistics_used":
                False,
        })


        # ----------------------------------------------------
        # Label- and feature-blind support selection.
        #
        # Ranking is built strictly from target TRAIN cell IDs
        # from the pre-existing frozen assignment artifact.
        # ----------------------------------------------------

        candidate_ids = sorted(
            target_train_cells[
                "cell_id"
            ].astype(str)
            .tolist()
        )

        ranked = []


        for cell_id in candidate_ids:

            token = (
                "lodo_fewshot_v1"
                f"|held_out={held_out}"
                f"|split_seed={seed}"
                f"|cell_id={cell_id}"
            )

            digest = text_sha256(
                token
            )

            ranked.append(
                (
                    digest,
                    cell_id,
                )
            )


        ranked = sorted(
            ranked,
            key=lambda x: (
                x[0],
                x[1],
            ),
        )


        assert len(
            {
                digest
                for digest, _
                in ranked
            }
        ) == len(
            ranked
        )


        top5 = ranked[
            :5
        ]


        selected_ids = [
            cell_id
            for _, cell_id
            in top5
        ]


        # Only AFTER ranking is frozen do we look up the number
        # of eligible rows for provenance/adaptation workload.
        per_cell_rows = {}


        for rank, (
            digest,
            cell_id,
        ) in enumerate(
            top5,
            start=1,
        ):

            n_rows = int(
                (
                    (
                        target_train[
                            "cell_id"
                        ].astype(str)
                        == cell_id
                    )
                ).sum()
            )

            assert n_rows > 0

            per_cell_rows[
                cell_id
            ] = n_rows


            support_cell_rows.append({
                "held_out":
                    held_out,

                "split_seed":
                    seed,

                "support_rank":
                    rank,

                "cell_id":
                    cell_id,

                "selection_sha256":
                    digest,

                "n_support_rows_for_cell":
                    n_rows,

                "in_K1":
                    rank <= 1,

                "in_K3":
                    rank <= 3,

                "in_K5":
                    rank <= 5,
            })


        for k in SUPPORT_K:

            ids = selected_ids[
                :k
            ]

            digests = [
                digest
                for digest, _
                in top5[
                    :k
                ]
            ]

            n_rows = int(
                sum(
                    per_cell_rows[
                        cell_id
                    ]
                    for cell_id
                    in ids
                )
            )


            set_token = (
                f"held_out={held_out}"
                f"|split_seed={seed}"
                f"|condition=K{k}"
                "|cells="
                + "|".join(
                    ids
                )
            )


            support_set_rows.append({
                "held_out":
                    held_out,

                "split_seed":
                    seed,

                "condition":
                    f"K{k}",

                "k_cells":
                    k,

                "support_cell_ids_json":
                    compact_json(
                        ids
                    ),

                "support_selection_sha256s_json":
                    compact_json(
                        digests
                    ),

                "n_support_rows":
                    n_rows,

                "target_train_pool_cells":
                    len(
                        target_train_cells
                    ),

                "support_set_sha256":
                    text_sha256(
                        set_token
                    ),
            })


        support_print_rows.append(
            (
                held_out,
                seed,
                [
                    (
                        cell_id,
                        per_cell_rows[
                            cell_id
                        ],
                    )
                    for cell_id
                    in selected_ids
                ],
            )
        )


inventory = pd.DataFrame(
    inventory_rows
)

scalers = pd.DataFrame(
    scaler_rows
)

scaler_summary = pd.DataFrame(
    scaler_summary_rows
)

support_cells = pd.DataFrame(
    support_cell_rows
)

support_sets = pd.DataFrame(
    support_set_rows
)


assert len(
    inventory
) == 15

assert len(
    scalers
) == 300

assert len(
    scaler_summary
) == 15

assert len(
    support_cells
) == 75

assert len(
    support_sets
) == 45


assert set(
    scalers[
        "fit_split"
    ]
) == {
    "source_train_only"
}

assert not scalers[
    "target_statistics_used"
].astype(bool).any()


assert not scaler_summary[
    "target_train_statistics_used"
].astype(bool).any()

assert not scaler_summary[
    "target_validation_statistics_used"
].astype(bool).any()

assert not scaler_summary[
    "target_test_statistics_used"
].astype(bool).any()


# Nested support barriers.
for (
    held_out,
    seed,
), g in support_sets.groupby(
    [
        "held_out",
        "split_seed",
    ],
    sort=False,
):

    by_condition = {
        row[
            "condition"
        ]:
            set(
                json.loads(
                    row[
                        "support_cell_ids_json"
                    ]
                )
            )
        for _, row
        in g.iterrows()
    }

    assert by_condition[
        "K1"
    ].issubset(
        by_condition[
            "K3"
        ]
    )

    assert by_condition[
        "K3"
    ].issubset(
        by_condition[
            "K5"
        ]
    )


inventory = inventory.sort_values(
    [
        "held_out",
        "split_seed",
    ],
    kind="stable",
).reset_index(
    drop=True
)

scalers = scalers.sort_values(
    [
        "held_out",
        "split_seed",
        "feature",
    ],
    kind="stable",
).reset_index(
    drop=True
)

scaler_summary = scaler_summary.sort_values(
    [
        "held_out",
        "split_seed",
    ],
    kind="stable",
).reset_index(
    drop=True
)

support_cells = support_cells.sort_values(
    [
        "held_out",
        "split_seed",
        "support_rank",
    ],
    kind="stable",
).reset_index(
    drop=True
)

support_sets = support_sets.sort_values(
    [
        "held_out",
        "split_seed",
        "k_cells",
    ],
    kind="stable",
).reset_index(
    drop=True
)


atomic_csv(
    inventory,
    INVENTORY_OUT,
)

atomic_csv(
    scalers,
    SCALER_OUT,
)

atomic_csv(
    scaler_summary,
    SCALER_SUMMARY_OUT,
)

atomic_csv(
    support_cells,
    SUPPORT_CELLS_OUT,
)

atomic_csv(
    support_sets,
    SUPPORT_SETS_OUT,
)


manifest = {
    "contract":
        "lodo_fewshot_data_v1",

    "status":
        "COMPLETE_PENDING_INDEPENDENT_AUDIT",

    "protocol_sha256":
        EXPECTED[
            PROTOCOL
        ],

    "selection_contract": {
        "candidate_source":
            "frozen target-train cell IDs only",

        "target_labels_used_for_ranking":
            False,

        "target_features_used_for_ranking":
            False,

        "target_validation_used":
            False,

        "target_test_used":
            False,

        "support_K":
            SUPPORT_K,

        "nested":
            True,

        "selected_cell_rows":
            len(
                support_cells
            ),

        "support_set_rows":
            len(
                support_sets
            ),
    },

    "scaler_contract": {
        "scaler_rows":
            len(
                scalers
            ),

        "fold_seed_scalers":
            15,

        "features_per_scaler":
            20,

        "fit_source":
            "source train rows only",

        "target_train_statistics_used":
            False,

        "target_validation_statistics_used":
            False,

        "target_test_statistics_used":
            False,

        "source_validation_statistics_used":
            False,

        "target_scaler_created":
            False,
    },

    "model_training_state": {
        "zero_shot_training_started":
            False,

        "few_shot_adaptation_started":
            False,

        "test_metrics_read":
            False,

        "test_metrics_surfaced":
            False,
    },

    "inputs": {
        str(path):
            sha256(path)
        for path
        in EXPECTED
    },

    "outputs": {
        "fold_inventory": {
            "path":
                str(
                    INVENTORY_OUT
                ),

            "sha256":
                sha256(
                    INVENTORY_OUT
                ),
        },

        "source_only_feature_scalers": {
            "path":
                str(
                    SCALER_OUT
                ),

            "sha256":
                sha256(
                    SCALER_OUT
                ),
        },

        "source_only_scaler_summary": {
            "path":
                str(
                    SCALER_SUMMARY_OUT
                ),

            "sha256":
                sha256(
                    SCALER_SUMMARY_OUT
                ),
        },

        "support_cells": {
            "path":
                str(
                    SUPPORT_CELLS_OUT
                ),

            "sha256":
                sha256(
                    SUPPORT_CELLS_OUT
                ),
        },

        "support_sets": {
            "path":
                str(
                    SUPPORT_SETS_OUT
                ),

            "sha256":
                sha256(
                    SUPPORT_SETS_OUT
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
}


atomic_json(
    manifest,
    MANIFEST_OUT,
)


print("LODO / FEW-SHOT DATA BUILD")
print("fold-seed inventory rows: 15")
print("source-only scaler rows: 300")
print("support-cell rows: 75")
print("support-set rows: 45")
print("support ranking uses labels: False")
print("support ranking uses features: False")
print("target statistics used in scaler: False")
print("target validation used: False")
print("target test used: False")
print("model training started: False")
print("test metrics read: False")

for held_out, seed, cells in support_print_rows:

    encoded = ",".join(
        f"{cell_id}({n_rows})"
        for cell_id, n_rows
        in cells
    )

    print(
        f"SUPPORT held={held_out} "
        f"seed={seed} "
        f"ranked_top5={encoded}"
    )

print(
    "data_manifest_sha256:",
    sha256(
        MANIFEST_OUT
    )
)

print(
    "LODO_FEWSHOT_DATA_BUILD_COMPLETE_READY_FOR_AUDIT=True"
)
