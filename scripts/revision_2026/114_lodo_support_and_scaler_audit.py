from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd


BASE = Path.home() / "alper/battery_revision_2026"

ROOT = BASE / "results/revision_2026/lodo_fewshot"
DATA = ROOT / "data_v1"

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

INVENTORY = DATA / "lodo_fold_inventory.csv"
SCALER = DATA / "lodo_source_only_feature_scalers.csv"
SCALER_SUMMARY = DATA / "lodo_source_only_scaler_summary.csv"
SUPPORT_CELLS = DATA / "fewshot_support_cells.csv"
SUPPORT_SETS = DATA / "fewshot_support_sets.csv"
MANIFEST = DATA / "lodo_fewshot_data_manifest.json"


EXPECTED_PROTOCOL_SHA = (
    "966d8c6b95f29e671ce4be9a785c5088e6d8e7eda9d25b8389f204859641897f"
)

EXPECTED_TASK_SHA = (
    "49a7b7629cf9ee3f8ce0a7e3c52f2807878b752b4c9a30a876285fb891b00c14"
)

EXPECTED_ASSIGN_SHA = (
    "914c8b90131f496ca396f63e0502147ca99848fdfd4b2c6b2a5887225b47d055"
)


DATASETS = [
    "HUST",
    "CALCE",
    "MATR",
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


def subset_by_cells(frame, cells):

    keys = cells[
        [
            "dataset",
            "cell_id",
        ]
    ].drop_duplicates()

    return frame.merge(
        keys,
        on=[
            "dataset",
            "cell_id",
        ],
        how="inner",
        validate="many_to_one",
    )


def close(a, b):

    return math.isclose(
        float(a),
        float(b),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


assert sha256(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

assert sha256(
    TASKS
) == EXPECTED_TASK_SHA

assert sha256(
    ASSIGN
) == EXPECTED_ASSIGN_SHA


manifest = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)

assert manifest[
    "contract"
] == "lodo_fewshot_data_v1"

assert manifest[
    "status"
] == "COMPLETE_PENDING_INDEPENDENT_AUDIT"

assert manifest[
    "protocol_sha256"
] == EXPECTED_PROTOCOL_SHA


for item in manifest[
    "outputs"
].values():

    path = Path(
        item[
            "path"
        ]
    )

    assert path.is_file()

    assert sha256(
        path
    ) == item[
        "sha256"
    ]


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

assign[
    "split_seed"
] = pd.to_numeric(
    assign[
        "split_seed"
    ],
    errors="raise",
).astype(int)

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


actual_inventory = pd.read_csv(
    INVENTORY,
    dtype={
        "held_out": "string",
    },
    low_memory=False,
)

actual_scaler = pd.read_csv(
    SCALER,
    dtype={
        "held_out": "string",
        "feature": "string",
    },
    low_memory=False,
)

actual_summary = pd.read_csv(
    SCALER_SUMMARY,
    dtype={
        "held_out": "string",
    },
    low_memory=False,
)

actual_cells = pd.read_csv(
    SUPPORT_CELLS,
    dtype={
        "held_out": "string",
        "cell_id": "string",
    },
    low_memory=False,
)

actual_sets = pd.read_csv(
    SUPPORT_SETS,
    dtype={
        "held_out": "string",
        "condition": "string",
    },
    low_memory=False,
)


expected_inventory = []
expected_scaler = []
expected_summary = []
expected_cells = []
expected_sets = []


for held_out in DATASETS:

    source_datasets = [
        d
        for d in DATASETS
        if d != held_out
    ]


    for seed in range(5):

        a = assign[
            assign[
                "split_seed"
            ]
            == seed
        ]


        def cells(
            datasets,
            split,
        ):

            if isinstance(
                datasets,
                str,
            ):
                datasets = [
                    datasets
                ]

            return a[
                a[
                    "dataset"
                ].isin(
                    datasets
                )
                & (
                    a[
                        "split"
                    ]
                    == split
                )
            ][
                [
                    "dataset",
                    "cell_id",
                ]
            ].drop_duplicates()


        src_train_cells = cells(
            source_datasets,
            "train",
        )

        src_val_cells = cells(
            source_datasets,
            "validation",
        )

        tgt_train_cells = cells(
            held_out,
            "train",
        )

        tgt_val_cells = cells(
            held_out,
            "validation",
        )

        tgt_test_cells = cells(
            held_out,
            "test",
        )


        role_sets = [
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
                src_train_cells,
                src_val_cells,
                tgt_train_cells,
                tgt_val_cells,
                tgt_test_cells,
            ]
        ]

        for i in range(
            len(role_sets)
        ):

            for j in range(
                i + 1,
                len(role_sets),
            ):

                assert role_sets[i].isdisjoint(
                    role_sets[j]
                )


        src_train = subset_by_cells(
            primary,
            src_train_cells,
        )

        src_val = subset_by_cells(
            primary,
            src_val_cells,
        )

        tgt_train = subset_by_cells(
            primary,
            tgt_train_cells,
        )

        tgt_val = subset_by_cells(
            primary,
            tgt_val_cells,
        )

        tgt_test = subset_by_cells(
            primary,
            tgt_test_cells,
        )


        rows_by_dataset = {
            d:
                int(
                    (
                        src_train[
                            "dataset"
                        ]
                        == d
                    ).sum()
                )
            for d in source_datasets
        }

        cells_by_dataset = {
            d:
                int(
                    src_train.loc[
                        src_train[
                            "dataset"
                        ]
                        == d,
                        "cell_id",
                    ].nunique()
                )
            for d in source_datasets
        }


        expected_inventory.append({
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
                    src_train
                ),

            "source_train_cells":
                len(
                    src_train_cells
                ),

            "source_validation_rows":
                len(
                    src_val
                ),

            "source_validation_cells":
                len(
                    src_val_cells
                ),

            "target_train_rows":
                len(
                    tgt_train
                ),

            "target_train_cells":
                len(
                    tgt_train_cells
                ),

            "target_validation_rows":
                len(
                    tgt_val
                ),

            "target_validation_cells":
                len(
                    tgt_val_cells
                ),

            "target_test_rows":
                len(
                    tgt_test
                ),

            "target_test_cells":
                len(
                    tgt_test_cells
                ),

            "source_train_rows_by_dataset_json":
                compact_json(
                    rows_by_dataset
                ),

            "source_train_cells_by_dataset_json":
                compact_json(
                    cells_by_dataset
                ),
        })


        zero_count = 0


        for feature in FEATURES:

            x = src_train[
                feature
            ].to_numpy(
                dtype=np.float64
            )

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

            zero = bool(
                variance == 0.0
            )

            scale = (
                1.0
                if zero
                else math.sqrt(
                    variance
                )
            )

            zero_count += int(
                zero
            )


            expected_scaler.append({
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
                        src_train
                    ),

                "n_source_train_cells":
                    len(
                        src_train_cells
                    ),

                "fit_split":
                    "source_train_only",

                "target_statistics_used":
                    False,
            })


        expected_summary.append({
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
                    src_train
                ),

            "n_source_train_cells":
                len(
                    src_train_cells
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


        # Reconstruct support ordering from assignment IDs only.
        candidates = sorted(
            tgt_train_cells[
                "cell_id"
            ].astype(str)
            .tolist()
        )

        ranked = []


        for cell_id in candidates:

            token = (
                "lodo_fewshot_v1"
                f"|held_out={held_out}"
                f"|split_seed={seed}"
                f"|cell_id={cell_id}"
            )

            ranked.append(
                (
                    text_sha256(
                        token
                    ),
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


        top5 = ranked[
            :5
        ]

        ids = [
            cell_id
            for _, cell_id
            in top5
        ]

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
                    tgt_train[
                        "cell_id"
                    ].astype(str)
                    == cell_id
                ).sum()
            )

            assert n_rows > 0

            per_cell_rows[
                cell_id
            ] = n_rows


            expected_cells.append({
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


        for k in [
            1, 3, 5,
        ]:

            set_ids = ids[
                :k
            ]

            digests = [
                digest
                for digest, _
                in top5[
                    :k
                ]
            ]

            set_token = (
                f"held_out={held_out}"
                f"|split_seed={seed}"
                f"|condition=K{k}"
                "|cells="
                + "|".join(
                    set_ids
                )
            )


            expected_sets.append({
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
                        set_ids
                    ),

                "support_selection_sha256s_json":
                    compact_json(
                        digests
                    ),

                "n_support_rows":
                    int(
                        sum(
                            per_cell_rows[
                                c
                            ]
                            for c in set_ids
                        )
                    ),

                "target_train_pool_cells":
                    len(
                        tgt_train_cells
                    ),

                "support_set_sha256":
                    text_sha256(
                        set_token
                    ),
            })


expected_inventory = pd.DataFrame(
    expected_inventory
).sort_values(
    [
        "held_out",
        "split_seed",
    ],
    kind="stable",
).reset_index(
    drop=True
)

expected_scaler = pd.DataFrame(
    expected_scaler
).sort_values(
    [
        "held_out",
        "split_seed",
        "feature",
    ],
    kind="stable",
).reset_index(
    drop=True
)

expected_summary = pd.DataFrame(
    expected_summary
).sort_values(
    [
        "held_out",
        "split_seed",
    ],
    kind="stable",
).reset_index(
    drop=True
)

expected_cells = pd.DataFrame(
    expected_cells
).sort_values(
    [
        "held_out",
        "split_seed",
        "support_rank",
    ],
    kind="stable",
).reset_index(
    drop=True
)

expected_sets = pd.DataFrame(
    expected_sets
).sort_values(
    [
        "held_out",
        "split_seed",
        "k_cells",
    ],
    kind="stable",
).reset_index(
    drop=True
)


actual_inventory = actual_inventory.sort_values(
    [
        "held_out",
        "split_seed",
    ],
    kind="stable",
).reset_index(
    drop=True
)

actual_scaler = actual_scaler.sort_values(
    [
        "held_out",
        "split_seed",
        "feature",
    ],
    kind="stable",
).reset_index(
    drop=True
)

actual_summary = actual_summary.sort_values(
    [
        "held_out",
        "split_seed",
    ],
    kind="stable",
).reset_index(
    drop=True
)

actual_cells = actual_cells.sort_values(
    [
        "held_out",
        "split_seed",
        "support_rank",
    ],
    kind="stable",
).reset_index(
    drop=True
)

actual_sets = actual_sets.sort_values(
    [
        "held_out",
        "split_seed",
        "k_cells",
    ],
    kind="stable",
).reset_index(
    drop=True
)


pd.testing.assert_frame_equal(
    actual_inventory,
    expected_inventory,
    check_dtype=False,
    check_exact=True,
)

pd.testing.assert_frame_equal(
    actual_scaler,
    expected_scaler,
    check_dtype=False,
    check_exact=False,
    rtol=1e-12,
    atol=1e-12,
)

pd.testing.assert_frame_equal(
    actual_summary,
    expected_summary,
    check_dtype=False,
    check_exact=True,
)

pd.testing.assert_frame_equal(
    actual_cells,
    expected_cells,
    check_dtype=False,
    check_exact=True,
)

pd.testing.assert_frame_equal(
    actual_sets,
    expected_sets,
    check_dtype=False,
    check_exact=True,
)


# Explicit nested-set and isolation audit.
for (
    held_out,
    seed,
), g in actual_sets.groupby(
    [
        "held_out",
        "split_seed",
    ],
):

    sets = {
        row["condition"]:
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

    assert sets[
        "K1"
    ].issubset(
        sets[
            "K3"
        ]
    )

    assert sets[
        "K3"
    ].issubset(
        sets[
            "K5"
        ]
    )


    a = assign[
        (
            assign[
                "split_seed"
            ]
            == int(
                seed
            )
        )
        & (
            assign[
                "dataset"
            ]
            == held_out
        )
    ]

    train_ids = set(
        a.loc[
            a[
                "split"
            ]
            == "train",
            "cell_id",
        ].astype(str)
    )

    val_ids = set(
        a.loc[
            a[
                "split"
            ]
            == "validation",
            "cell_id",
        ].astype(str)
    )

    test_ids = set(
        a.loc[
            a[
                "split"
            ]
            == "test",
            "cell_id",
        ].astype(str)
    )

    assert sets[
        "K5"
    ].issubset(
        train_ids
    )

    assert sets[
        "K5"
    ].isdisjoint(
        val_ids
    )

    assert sets[
        "K5"
    ].isdisjoint(
        test_ids
    )


state = manifest[
    "model_training_state"
]

assert state[
    "zero_shot_training_started"
] is False

assert state[
    "few_shot_adaptation_started"
] is False

assert state[
    "test_metrics_read"
] is False

assert state[
    "test_metrics_surfaced"
] is False


print("LODO / FEW-SHOT DATA AUDIT")
print("15 fold-seed inventories independently reproduced:", True)
print("300 source-only scaler rows independently reproduced:", True)
print("source validation statistics used in scaler:", False)
print("target train statistics used in scaler:", False)
print("target validation statistics used in scaler:", False)
print("target test statistics used in scaler:", False)
print("75 selected support-cell rows independently reproduced:", True)
print("45 K1/K3/K5 support sets independently reproduced:", True)
print("support ranking reconstructed from target-train IDs only:", True)
print("support sets nested K1<K3<K5:", True)
print("support cells overlap target validation:", False)
print("support cells overlap target test:", False)
print("exact support row counts reproduced:", True)
print("model training started:", False)
print("test metrics read:", False)
print("LODO_FEWSHOT_DATA_AUDIT_PASS=True")
