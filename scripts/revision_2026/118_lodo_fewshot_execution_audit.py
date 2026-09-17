from pathlib import Path
import hashlib
import json
import math
import os

import numpy as np
import pandas as pd
import torch
from torch import nn


BASE = Path.home() / "alper/battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

ROOT = BASE / "results/revision_2026/lodo_fewshot"
DATA = ROOT / "data_v1"
EXEC = ROOT / "few_shot_v1/final_execution_v1"

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

SCALERS = DATA / "lodo_source_only_feature_scalers.csv"
SUPPORT_SETS = DATA / "fewshot_support_sets.csv"
DATA_MANIFEST = DATA / "lodo_fewshot_data_manifest.json"

ZERO = ROOT / "zero_shot_v1/final_execution_v1"
ZERO_MANIFEST = ZERO / "lodo_zero_shot_execution_manifest.json"
ZERO_AUDIT = ZERO / "lodo_zero_shot_execution_audit_receipt.json"

MANIFEST = EXEC / "lodo_fewshot_execution_manifest.json"

RUNNER = (
    REPO
    / "scripts/revision_2026"
    / "117_final_lodo_fewshot_runner.py"
)

RECEIPT = (
    EXEC
    / "lodo_fewshot_execution_audit_receipt.json"
)


EXPECTED = {
    PROTOCOL:
        "966d8c6b95f29e671ce4be9a785c5088e6d8e7eda9d25b8389f204859641897f",

    TASKS:
        "49a7b7629cf9ee3f8ce0a7e3c52f2807878b752b4c9a30a876285fb891b00c14",

    ASSIGN:
        "914c8b90131f496ca396f63e0502147ca99848fdfd4b2c6b2a5887225b47d055",

    SCALERS:
        "4ebb18c5292eb8e8b47437a22aa37ab4b68f17fdd90491039faed17b0f8ea155",

    SUPPORT_SETS:
        "4924654ce2fa70ad4680f9fc821e53e590124f5d87bdcf196e998edd235e48e9",

    DATA_MANIFEST:
        "8d3fa7c8b614305608e8db9f9f8d235388e6b304776cc64ffc8aa2fda6312a67",

    ZERO_MANIFEST:
        "b3e82cf395ce543c6452c62df02e34662b768f2cf24124d7eb530c88e11afbc1",

    ZERO_AUDIT:
        "21f2f4236a8fbb0f8d1293f72e69bccd9ce679be3c1002dbe57c6137cb68bd21",
}


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

CONDITIONS = [
    "K1",
    "K3",
    "K5",
]


torch.set_num_threads(
    max(
        1,
        min(
            8,
            os.cpu_count() or 1,
        ),
    )
)

torch.use_deterministic_algorithms(
    True
)


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


def atomic_json(
    obj,
    path,
):

    tmp = Path(
        str(path) + ".tmp"
    )

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


def state_tensor_hash(
    state_dict,
):

    h = hashlib.sha256()

    for key in sorted(
        state_dict
    ):

        tensor = (
            state_dict[
                key
            ]
            .detach()
            .cpu()
            .contiguous()
        )

        h.update(
            key.encode(
                "utf-8"
            )
        )

        h.update(
            str(
                tensor.dtype
            ).encode(
                "utf-8"
            )
        )

        h.update(
            np.asarray(
                tensor.shape,
                dtype=np.int64,
            ).tobytes()
        )

        h.update(
            tensor.numpy().tobytes()
        )

    return h.hexdigest()


def causal_mean(
    x,
    prev,
):

    safe = torch.clamp(
        prev,
        min=0,
    )

    px = x[
        safe
    ]

    mask = (
        prev >= 0
    ).unsqueeze(
        1
    )

    return torch.where(
        mask,
        0.5 * (
            x + px
        ),
        x,
    )


class MLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(
            20,
            64,
        )

        self.fc2 = nn.Linear(
            64,
            64,
        )

        self.out = nn.Linear(
            64,
            1,
        )


    def forward(
        self,
        x,
        prev=None,
    ):

        x = torch.relu(
            self.fc1(
                x
            )
        )

        x = torch.relu(
            self.fc2(
                x
            )
        )

        return self.out(
            x
        )


class GCN(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(
            20,
            64,
        )

        self.fc2 = nn.Linear(
            64,
            64,
        )

        self.out = nn.Linear(
            64,
            1,
        )


    def forward(
        self,
        x,
        prev,
    ):

        x = causal_mean(
            x,
            prev,
        )

        x = torch.relu(
            self.fc1(
                x
            )
        )

        x = causal_mean(
            x,
            prev,
        )

        x = torch.relu(
            self.fc2(
                x
            )
        )

        return self.out(
            x
        )


MODELS = {
    "MLP":
        MLP,

    "GCN":
        GCN,
}


for path, expected in EXPECTED.items():

    assert sha256(
        path
    ) == expected


zero_audit = json.loads(
    ZERO_AUDIT.read_text(
        encoding="utf-8"
    )
)

assert zero_audit[
    "audit_pass"
] is True


zero_manifest = json.loads(
    ZERO_MANIFEST.read_text(
        encoding="utf-8"
    )
)


manifest = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)


assert manifest[
    "contract"
] == "lodo_fewshot_execution_v1"

assert manifest[
    "status"
] == "COMPLETE_PENDING_INDEPENDENT_EXECUTION_AUDIT"

assert manifest[
    "runner"
][
    "sha256"
] == sha256(
    RUNNER
)

assert manifest[
    "execution"
][
    "adaptation_fits"
] == 270

assert manifest[
    "execution"
][
    "paired_test_records"
] == 45

assert manifest[
    "execution"
][
    "adaptation_epochs_each"
] == 100

assert manifest[
    "execution"
][
    "all_conditions_start_independently_from_zero_shot"
] is True

assert manifest[
    "execution"
][
    "source_only_scaler_retained"
] is True

assert manifest[
    "execution"
][
    "target_validation_used"
] is False

assert manifest[
    "execution"
][
    "target_test_used_during_adaptation"
] is False

assert manifest[
    "execution"
][
    "all_adaptations_complete_before_target_test_evaluation"
] is True

assert manifest[
    "execution"
][
    "numeric_test_values_printed_by_runner"
] is False


assert len(
    manifest[
        "adapted_states"
    ]
) == 270

assert len(
    manifest[
        "fit_metadata"
    ]
) == 270

assert len(
    manifest[
        "run_records"
    ]
) == 45


for collection in [
    "adapted_states",
    "fit_metadata",
    "run_records",
]:

    for item in manifest[
        collection
    ]:

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


# ============================================================
# Inputs
# ============================================================

tasks = pd.read_csv(
    TASKS,
    dtype={
        "dataset":
            "string",

        "cell_id":
            "string",
    },
    low_memory=False,
)

assign = pd.read_csv(
    ASSIGN,
    dtype={
        "dataset":
            "string",

        "cell_id":
            "string",

        "split":
            "string",
    },
    low_memory=False,
)

scalers = pd.read_csv(
    SCALERS,
    dtype={
        "held_out":
            "string",

        "feature":
            "string",
    },
    low_memory=False,
)

support_sets = pd.read_csv(
    SUPPORT_SETS,
    dtype={
        "held_out":
            "string",

        "condition":
            "string",
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

tasks[
    "observation_index"
] = pd.to_numeric(
    tasks[
        "observation_index"
    ],
    errors="raise",
).astype(
    np.int64
)

assign[
    "split_seed"
] = pd.to_numeric(
    assign[
        "split_seed"
    ],
    errors="raise",
).astype(
    int
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


for feature in FEATURES:

    primary[
        feature
    ] = pd.to_numeric(
        primary[
            feature
        ],
        errors="raise",
    )


# ============================================================
# Helpers
# ============================================================

def scaler_for(
    held,
    seed,
):

    g = scalers[
        (
            scalers[
                "held_out"
            ]
            == held
        )
        & (
            scalers[
                "split_seed"
            ]
            == seed
        )
    ].set_index(
        "feature"
    )


    assert len(
        g
    ) == 20

    assert set(
        g[
            "fit_split"
        ].astype(str)
    ) == {
        "source_train_only"
    }


    mean = np.asarray(
        [
            float(
                g.loc[
                    f,
                    "mean",
                ]
            )
            for f in FEATURES
        ],
        dtype=np.float64,
    )

    scale = np.asarray(
        [
            float(
                g.loc[
                    f,
                    "scale",
                ]
            )
            for f in FEATURES
        ],
        dtype=np.float64,
    )


    return (
        mean,
        scale,
    )


def build_prev(
    frame,
):

    p = np.full(
        len(
            frame
        ),
        -1,
        dtype=np.int64,
    )


    for _, g in frame.groupby(
        [
            "dataset",
            "cell_id",
        ],
        sort=False,
    ):

        idx = g.index.to_numpy(
            dtype=np.int64
        )


        if len(
            idx
        ) > 1:

            p[
                idx[
                    1:
                ]
            ] = idx[
                :-1
            ]


    valid = np.flatnonzero(
        p >= 0
    )


    if len(
        valid
    ):

        earlier = p[
            valid
        ]

        assert np.all(
            earlier < valid
        )

        assert np.array_equal(
            frame.loc[
                valid,
                "cell_id",
            ].astype(str).to_numpy(),
            frame.loc[
                earlier,
                "cell_id",
            ].astype(str).to_numpy(),
        )

        assert np.all(
            frame.loc[
                earlier,
                "observation_index",
            ].to_numpy(
                dtype=np.int64
            )
            <
            frame.loc[
                valid,
                "observation_index",
            ].to_numpy(
                dtype=np.int64
            )
        )


    return p


def cell_weights(
    frame,
):

    w = np.zeros(
        len(
            frame
        ),
        dtype=np.float64,
    )


    cells = sorted(
        frame[
            "cell_id"
        ].astype(str)
        .unique()
    )


    for cell in cells:

        idx = np.flatnonzero(
            frame[
                "cell_id"
            ].astype(str)
            .to_numpy()
            == cell
        )


        w[
            idx
        ] = (
            1.0
            / len(
                cells
            )
            / len(
                idx
            )
        )


    assert math.isclose(
        float(
            w.sum()
        ),
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    return w


def frozen_support(
    held,
    seed,
    condition,
):

    row = support_sets[
        (
            support_sets[
                "held_out"
            ]
            == held
        )
        & (
            support_sets[
                "split_seed"
            ]
            == seed
        )
        & (
            support_sets[
                "condition"
            ]
            == condition
        )
    ]


    assert len(
        row
    ) == 1


    ids = json.loads(
        row.iloc[
            0
        ][
            "support_cell_ids_json"
        ]
    )


    return (
        ids,
        str(
            row.iloc[
                0
            ][
                "support_set_sha256"
            ]
        ),
        int(
            row.iloc[
                0
            ][
                "n_support_rows"
            ]
        ),
    )


def prepare_support(
    held,
    seed,
    condition,
):

    ids, set_sha, expected_rows = frozen_support(
        held,
        seed,
        condition,
    )


    a = assign[
        (
            assign[
                "split_seed"
            ]
            == seed
        )
        & (
            assign[
                "dataset"
            ]
            == held
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


    assert set(
        ids
    ).issubset(
        train_ids
    )

    assert set(
        ids
    ).isdisjoint(
        val_ids
    )

    assert set(
        ids
    ).isdisjoint(
        test_ids
    )


    frame = primary[
        (
            primary[
                "dataset"
            ]
            == held
        )
        & primary[
            "cell_id"
        ].astype(str).isin(
            ids
        )
    ].copy()


    frame = frame.sort_values(
        [
            "dataset",
            "cell_id",
            "observation_index",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


    assert len(
        frame
    ) == expected_rows


    mean, scale = scaler_for(
        held,
        seed,
    )


    X = (
        frame[
            FEATURES
        ].to_numpy(
            dtype=np.float64
        )
        - mean
    ) / scale


    return {
        "frame":
            frame,

        "x":
            torch.tensor(
                X,
                dtype=torch.float32,
            ),

        "y":
            frame[
                "SOH"
            ].to_numpy(
                dtype=np.float64
            ),

        "prev":
            torch.tensor(
                build_prev(
                    frame
                ),
                dtype=torch.long,
            ),

        "weights":
            cell_weights(
                frame
            ),

        "ids":
            ids,

        "set_sha":
            set_sha,
    }


def prepare_test(
    held,
    seed,
):

    a = assign[
        (
            assign[
                "split_seed"
            ]
            == seed
        )
        & (
            assign[
                "dataset"
            ]
            == held
        )
        & (
            assign[
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


    frame = primary.merge(
        a,
        on=[
            "dataset",
            "cell_id",
        ],
        how="inner",
        validate="many_to_one",
    )


    frame = frame.sort_values(
        [
            "dataset",
            "cell_id",
            "observation_index",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


    mean, scale = scaler_for(
        held,
        seed,
    )


    X = (
        frame[
            FEATURES
        ].to_numpy(
            dtype=np.float64
        )
        - mean
    ) / scale


    return {
        "frame":
            frame,

        "x":
            torch.tensor(
                X,
                dtype=torch.float32,
            ),

        "y":
            frame[
                "SOH"
            ].to_numpy(
                dtype=np.float64
            ),

        "prev":
            torch.tensor(
                build_prev(
                    frame
                ),
                dtype=torch.long,
            ),

        "weights":
            cell_weights(
                frame
            ),
    }


def metrics(
    y,
    pred,
    w,
):

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    pred = np.asarray(
        pred,
        dtype=np.float64,
    )

    w = np.asarray(
        w,
        dtype=np.float64,
    )

    w = w / w.sum()

    err = pred - y

    mse = float(
        np.sum(
            w
            * err
            * err
        )
    )

    rmse = float(
        math.sqrt(
            mse
        )
    )

    mae = float(
        np.sum(
            w
            * np.abs(
                err
            )
        )
    )

    ym = float(
        np.sum(
            w * y
        )
    )

    sst = float(
        np.sum(
            w
            * (
                y - ym
            ) ** 2
        )
    )

    r2 = (
        float("nan")
        if sst == 0.0
        else float(
            1.0
            - mse / sst
        )
    )


    return {
        "RMSE":
            rmse,

        "MAE":
            mae,

        "R2":
            r2,
    }


def bundle(
    data,
    pred,
):

    frame = data[
        "frame"
    ]


    cell = metrics(
        data[
            "y"
        ],
        pred,
        data[
            "weights"
        ],
    )

    row = metrics(
        data[
            "y"
        ],
        pred,
        np.ones(
            len(
                frame
            ),
            dtype=np.float64,
        ),
    )


    return {
        "CELL_BALANCED": {
            **cell,

            "n_rows":
                len(
                    frame
                ),

            "n_cells":
                frame[
                    "cell_id"
                ].astype(str)
                .nunique(),
        },

        "ROW": {
            **row,

            "n_rows":
                len(
                    frame
                ),

            "n_cells":
                frame[
                    "cell_id"
                ].astype(str)
                .nunique(),
        },
    }


def predict(
    model,
    tag,
    data,
):

    model.eval()


    with torch.no_grad():

        if tag == "MLP":

            p = model(
                data[
                    "x"
                ]
            ).squeeze(
                1
            )

        else:

            p = model(
                data[
                    "x"
                ],
                data[
                    "prev"
                ],
            ).squeeze(
                1
            )


    return (
        p.detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )


def assert_metric_tree(
    actual,
    expected,
):

    for scope in [
        "CELL_BALANCED",
        "ROW",
    ]:

        for metric in [
            "RMSE",
            "MAE",
            "R2",
        ]:

            a = float(
                actual[
                    scope
                ][
                    metric
                ]
            )

            e = float(
                expected[
                    scope
                ][
                    metric
                ]
            )


            if (
                math.isnan(
                    a
                )
                or math.isnan(
                    e
                )
            ):

                assert (
                    math.isnan(
                        a
                    )
                    and math.isnan(
                        e
                    )
                )

            else:

                assert math.isclose(
                    a,
                    e,
                    rel_tol=1e-8,
                    abs_tol=1e-8,
                )


        assert int(
            actual[
                scope
            ][
                "n_rows"
            ]
        ) == int(
            expected[
                scope
            ][
                "n_rows"
            ]
        )

        assert int(
            actual[
                scope
            ][
                "n_cells"
            ]
        ) == int(
            expected[
                scope
            ][
                "n_cells"
            ]
        )


# ============================================================
# Zero-shot lookup
# ============================================================

zero_states = {}


for item in zero_manifest[
    "model_states"
]:

    key = (
        str(
            item[
                "held_out"
            ]
        ),
        int(
            item[
                "split_seed"
            ]
        ),
        int(
            item[
                "training_initialization_seed"
            ]
        ),
        str(
            item[
                "model_tag"
            ]
        ),
    )

    zero_states[
        key
    ] = item


assert len(
    zero_states
) == 90


# ============================================================
# Audit all 270 fits and 45 test records
# ============================================================

validated_states = 0
validated_records = 0
support_isolation_checks = 0
causal_test_edge_checks = 0


for record_item in manifest[
    "run_records"
]:

    record_path = Path(
        record_item[
            "path"
        ]
    )

    record = json.loads(
        record_path.read_text(
            encoding="utf-8"
        )
    )


    held = str(
        record[
            "held_out"
        ]
    )

    seed = int(
        record[
            "split_seed"
        ]
    )

    train_seed = int(
        record[
            "training_initialization_seed"
        ]
    )


    test = prepare_test(
        held,
        seed,
    )


    for model_tag in [
        "MLP",
        "GCN",
    ]:

        for condition in CONDITIONS:

            rec = record[
                "models"
            ][
                model_tag
            ][
                condition
            ]


            support = prepare_support(
                held,
                seed,
                condition,
            )


            assert rec[
                "support_cell_ids"
            ] == support[
                "ids"
            ]

            assert rec[
                "support_set_sha256"
            ] == support[
                "set_sha"
            ]

            assert int(
                rec[
                    "n_support_rows"
                ]
            ) == len(
                support[
                    "frame"
                ]
            )


            zero_key = (
                held,
                seed,
                train_seed,
                model_tag,
            )

            zero_item = zero_states[
                zero_key
            ]

            zero_path = Path(
                zero_item[
                    "path"
                ]
            )

            assert sha256(
                zero_path
            ) == zero_item[
                "sha256"
            ]


            zero_payload = torch.load(
                zero_path,
                map_location="cpu",
                weights_only=False,
            )

            zero_tensor_sha = state_tensor_hash(
                zero_payload[
                    "state_dict"
                ]
            )


            assert rec[
                "zero_shot_state_file_sha256"
            ] == zero_item[
                "sha256"
            ]

            assert rec[
                "zero_shot_state_tensor_sha256"
            ] == zero_tensor_sha


            state_path = Path(
                rec[
                    "state_path"
                ]
            )

            assert sha256(
                state_path
            ) == rec[
                "state_file_sha256"
            ]


            payload = torch.load(
                state_path,
                map_location="cpu",
                weights_only=False,
            )


            assert state_tensor_hash(
                payload[
                    "state_dict"
                ]
            ) == rec[
                "state_tensor_sha256"
            ]


            assert int(
                rec[
                    "adaptation_epochs"
                ]
            ) == 100


            model = MODELS[
                model_tag
            ]()

            model.load_state_dict(
                payload[
                    "state_dict"
                ]
            )


            pred = predict(
                model,
                model_tag,
                test,
            )


            expected_metrics = bundle(
                test,
                pred,
            )


            assert_metric_tree(
                rec[
                    "test"
                ],
                expected_metrics,
            )


            # Support cells must remain target-train only.
            a = assign[
                (
                    assign[
                        "split_seed"
                    ]
                    == seed
                )
                & (
                    assign[
                        "dataset"
                    ]
                    == held
                )
            ]


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


            assert set(
                support[
                    "ids"
                ]
            ).isdisjoint(
                val_ids
            )

            assert set(
                support[
                    "ids"
                ]
            ).isdisjoint(
                test_ids
            )


            support_isolation_checks += 1


            if model_tag == "GCN":

                prev = test[
                    "prev"
                ].numpy()

                valid = np.flatnonzero(
                    prev >= 0
                )

                earlier = prev[
                    valid
                ]


                assert np.all(
                    earlier < valid
                )

                assert np.array_equal(
                    test[
                        "frame"
                    ].loc[
                        valid,
                        "cell_id",
                    ].astype(str).to_numpy(),
                    test[
                        "frame"
                    ].loc[
                        earlier,
                        "cell_id",
                    ].astype(str).to_numpy(),
                )


                assert np.all(
                    test[
                        "frame"
                    ].loc[
                        earlier,
                        "observation_index",
                    ].to_numpy(
                        dtype=np.int64
                    )
                    <
                    test[
                        "frame"
                    ].loc[
                        valid,
                        "observation_index",
                    ].to_numpy(
                        dtype=np.int64
                    )
                )


                causal_test_edge_checks += int(
                    len(
                        valid
                    )
                )


            validated_states += 1


    validated_records += 1


assert validated_states == 270
assert validated_records == 45
assert support_isolation_checks == 270
assert causal_test_edge_checks > 0


# Independent fit-metadata checks.
for item in manifest[
    "fit_metadata"
]:

    meta = json.loads(
        Path(
            item[
                "path"
            ]
        ).read_text(
            encoding="utf-8"
        )
    )


    assert meta[
        "adaptation_epochs"
    ] == 100

    assert meta[
        "adaptation_learning_rate"
    ] == 0.0001

    assert meta[
        "weight_decay"
    ] == 0.00001

    assert meta[
        "early_stopping"
    ] is False

    assert meta[
        "source_only_scaler_retained"
    ] is True

    assert meta[
        "target_validation_used"
    ] is False

    assert meta[
        "target_test_used_during_adaptation"
    ] is False

    assert meta[
        "starting_state"
    ] == "corresponding_zero_shot_best_state"


receipt = {
    "contract":
        "lodo_fewshot_execution_audit_receipt_v1",

    "execution_manifest_sha256":
        sha256(
            MANIFEST
        ),

    "runner_sha256":
        sha256(
            RUNNER
        ),

    "validated_adapted_states":
        270,

    "validated_paired_test_records":
        45,

    "all_starting_zero_shot_state_hashes_exact":
        True,

    "all_adaptation_conditions_start_independently_from_zero_shot":
        True,

    "all_adaptations_fixed_100_epochs":
        True,

    "all_adaptations_source_only_scaler":
        True,

    "target_validation_used":
        False,

    "target_test_used_during_adaptation":
        False,

    "all_support_cells_target_train_only":
        True,

    "all_support_sets_match_frozen_manifest":
        True,

    "support_isolation_checks":
        270,

    "all_target_test_metrics_independently_reproduced":
        True,

    "future_edges_detected":
        False,

    "cross_cell_edges_detected":
        False,

    "causal_test_edge_checks":
        causal_test_edge_checks,

    "numeric_test_metrics_surfaced_by_audit":
        False,

    "audit_pass":
        True,
}


atomic_json(
    receipt,
    RECEIPT,
)


print("LODO FEW-SHOT EXECUTION AUDIT")
print("270 adapted states hash-exact:", True)
print("45 paired test records independently verified:", True)
print("all starting zero-shot state hashes exact:", True)
print("K1/K3/K5 each start independently from zero-shot:", True)
print("all adaptations fixed at 100 epochs:", True)
print("source-only scaler retained:", True)
print("target validation used:", False)
print("target test used during adaptation:", False)
print("all support cells target-train only:", True)
print("all support sets match frozen manifest:", True)
print("all target-test metrics independently reproduced:", True)
print("future edges detected:", False)
print("cross-cell edges detected:", False)
print("numeric test metrics surfaced by audit:", False)
print("METRIC_EMBARGO_OUTPUT_GUARD=PASS")
print("audit_receipt_sha256:", sha256(RECEIPT))
print("LODO_FEWSHOT_EXECUTION_AUDIT_PASS=True")
