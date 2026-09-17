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
EXEC = ROOT / "zero_shot_v1/final_execution_v1"

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
DATA_MANIFEST = DATA / "lodo_fewshot_data_manifest.json"

MANIFEST = EXEC / "lodo_zero_shot_execution_manifest.json"

RUNNER = (
    REPO
    / "scripts/revision_2026"
    / "115_final_lodo_zero_shot_runner.py"
)

RECEIPT = (
    EXEC
    / "lodo_zero_shot_execution_audit_receipt.json"
)


EXPECTED_PROTOCOL_SHA = (
    "966d8c6b95f29e671ce4be9a785c5088e6d8e7eda9d25b8389f204859641897f"
)

EXPECTED_DATA_MANIFEST_SHA = (
    "8d3fa7c8b614305608e8db9f9f8d235388e6b304776cc64ffc8aa2fda6312a67"
)

EXPECTED_TASK_SHA = (
    "49a7b7629cf9ee3f8ce0a7e3c52f2807878b752b4c9a30a876285fb891b00c14"
)

EXPECTED_ASSIGN_SHA = (
    "914c8b90131f496ca396f63e0502147ca99848fdfd4b2c6b2a5887225b47d055"
)

EXPECTED_SCALER_SHA = (
    "4ebb18c5292eb8e8b47437a22aa37ab4b68f17fdd90491039faed17b0f8ea155"
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


torch.set_num_threads(
    max(
        1,
        min(
            8,
            os.cpu_count() or 1,
        ),
    )
)

torch.use_deterministic_algorithms(True)


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


def state_tensor_hash(state_dict):

    h = hashlib.sha256()

    for key in sorted(state_dict):

        tensor = (
            state_dict[key]
            .detach()
            .cpu()
            .contiguous()
        )

        h.update(
            key.encode("utf-8")
        )

        h.update(
            str(tensor.dtype).encode("utf-8")
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


def causal_mean(x, prev):

    safe = torch.clamp(
        prev,
        min=0,
    )

    px = x[
        safe
    ]

    mask = (
        prev >= 0
    ).unsqueeze(1)

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
            self.fc1(x)
        )

        x = torch.relu(
            self.fc2(x)
        )

        return self.out(x)


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
            self.fc1(x)
        )

        x = causal_mean(
            x,
            prev,
        )

        x = torch.relu(
            self.fc2(x)
        )

        return self.out(x)


MODELS = {
    "MLP":
        MLP,

    "GCN":
        GCN,
}


def parameter_count(model):

    return sum(
        int(p.numel())
        for p in model.parameters()
        if p.requires_grad
    )


assert sha256(PROTOCOL) == EXPECTED_PROTOCOL_SHA
assert sha256(DATA_MANIFEST) == EXPECTED_DATA_MANIFEST_SHA
assert sha256(TASKS) == EXPECTED_TASK_SHA
assert sha256(ASSIGN) == EXPECTED_ASSIGN_SHA
assert sha256(SCALERS) == EXPECTED_SCALER_SHA


manifest = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)


assert manifest[
    "contract"
] == "lodo_zero_shot_execution_v1"

assert manifest[
    "status"
] == "COMPLETE_PENDING_INDEPENDENT_EXECUTION_AUDIT"

assert manifest[
    "protocol_sha256"
] == EXPECTED_PROTOCOL_SHA

assert manifest[
    "data_manifest_sha256"
] == EXPECTED_DATA_MANIFEST_SHA

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
    "paired_records"
] == 45

assert manifest[
    "execution"
][
    "model_fits"
] == 90

assert manifest[
    "execution"
][
    "safety_ceiling_hits"
] == 0

assert manifest[
    "execution"
][
    "all_training_complete_before_target_test_evaluation"
] is True

assert manifest[
    "execution"
][
    "target_train_used_for_training"
] is False

assert manifest[
    "execution"
][
    "target_validation_used_for_training_or_selection"
] is False

assert manifest[
    "execution"
][
    "target_test_used_for_model_selection"
] is False

assert manifest[
    "execution"
][
    "numeric_test_values_printed_by_runner"
] is False


assert len(
    manifest[
        "model_states"
    ]
) == 90

assert len(
    manifest[
        "fit_metadata"
    ]
) == 90

assert len(
    manifest[
        "run_records"
    ]
) == 45


for collection in [
    "model_states",
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

scalers = pd.read_csv(
    SCALERS,
    dtype={
        "held_out": "string",
        "feature": "string",
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
).astype(np.int64)

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


for feature in FEATURES:

    primary[
        feature
    ] = pd.to_numeric(
        primary[
            feature
        ],
        errors="raise",
    )


def source_datasets(
    held,
):

    return [
        d
        for d in DATASETS
        if d != held
    ]


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


    assert len(g) == 20


    mean = np.asarray(
        [
            float(
                g.loc[
                    feature,
                    "mean",
                ]
            )
            for feature in FEATURES
        ],
        dtype=np.float64,
    )

    scale = np.asarray(
        [
            float(
                g.loc[
                    feature,
                    "scale",
                ]
            )
            for feature in FEATURES
        ],
        dtype=np.float64,
    )


    return (
        mean,
        scale,
    )


def cells_for(
    held,
    seed,
    role,
):

    a = assign[
        assign[
            "split_seed"
        ]
        == seed
    ]

    source = source_datasets(
        held
    )


    if role == "source_train":

        q = a[
            a[
                "dataset"
            ].isin(
                source
            )
            & (
                a[
                    "split"
                ]
                == "train"
            )
        ]

    elif role == "source_validation":

        q = a[
            a[
                "dataset"
            ].isin(
                source
            )
            & (
                a[
                    "split"
                ]
                == "validation"
            )
        ]

    elif role == "target_test":

        q = a[
            (
                a[
                    "dataset"
                ]
                == held
            )
            & (
                a[
                    "split"
                ]
                == "test"
            )
        ]

    else:

        raise ValueError(
            role
        )


    return q[
        [
            "dataset",
            "cell_id",
        ]
    ].drop_duplicates()


def prev_index(
    frame,
):

    p = np.full(
        len(frame),
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

        if len(idx) > 1:

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


    if len(valid):

        earlier = p[
            valid
        ]


        assert np.all(
            earlier < valid
        )

        assert np.array_equal(
            frame.loc[
                valid,
                "dataset",
            ].astype(str).to_numpy(),
            frame.loc[
                earlier,
                "dataset",
            ].astype(str).to_numpy(),
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


def weights(
    frame,
):

    w = np.zeros(
        len(frame),
        dtype=np.float64,
    )


    datasets = sorted(
        frame[
            "dataset"
        ].astype(str)
        .unique()
    )


    for dataset in datasets:

        idx_d = np.flatnonzero(
            frame[
                "dataset"
            ].astype(str)
            .to_numpy()
            == dataset
        )


        cells = sorted(
            frame.iloc[
                idx_d
            ][
                "cell_id"
            ].astype(str)
            .unique()
        )


        for cell in cells:

            idx = idx_d[
                frame.iloc[
                    idx_d
                ][
                    "cell_id"
                ].astype(str)
                .to_numpy()
                == cell
            ]


            w[
                idx
            ] = (
                1.0
                / len(
                    datasets
                )
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


CACHE = {}


def prepare(
    held,
    seed,
    role,
):

    key = (
        held,
        seed,
        role,
    )


    if key in CACHE:

        return CACHE[
            key
        ]


    cells = cells_for(
        held,
        seed,
        role,
    )


    frame = primary.merge(
        cells,
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


    observed = set(
        frame[
            "dataset"
        ].astype(str)
        .unique()
    )


    if role in {
        "source_train",
        "source_validation",
    }:

        assert observed == set(
            source_datasets(
                held
            )
        )

        assert held not in observed

    else:

        assert observed == {
            held
        }


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


    y = frame[
        "SOH"
    ].to_numpy(
        dtype=np.float64
    )


    out = {
        "frame":
            frame,

        "x":
            torch.tensor(
                X,
                dtype=torch.float32,
            ),

        "y":
            y,

        "prev":
            torch.tensor(
                prev_index(
                    frame
                ),
                dtype=torch.long,
            ),

        "weights":
            weights(
                frame
            ),
    }


    CACHE[
        key
    ] = out


    return out


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

    e = pred - y


    mse = float(
        np.sum(
            w
            * e
            * e
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
                e
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


    if sst == 0.0:

        r2 = float("nan")

    else:

        r2 = float(
            1.0
            - mse / sst
        )


    return {
        "RMSE":
            rmse,

        "MAE":
            mae,

        "R2":
            r2,
    }


def test_bundle(
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
        weights(
            frame
        ),
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
            ).squeeze(1)

        else:

            p = model(
                data[
                    "x"
                ],
                data[
                    "prev"
                ],
            ).squeeze(1)


    return (
        p.detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )


def assert_metric_tree_close(
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
                math.isnan(a)
                or math.isnan(e)
            ):

                assert (
                    math.isnan(a)
                    and math.isnan(e)
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


validated_states = 0
validated_records = 0
causal_edge_checks = 0


for item in manifest[
    "run_records"
]:

    record = json.loads(
        Path(
            item[
                "path"
            ]
        ).read_text(
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


    assert held in DATASETS


    train = prepare(
        held,
        seed,
        "source_train",
    )

    val = prepare(
        held,
        seed,
        "source_validation",
    )

    test = prepare(
        held,
        seed,
        "target_test",
    )


    assert held not in set(
        train[
            "frame"
        ][
            "dataset"
        ].astype(str)
    )

    assert held not in set(
        val[
            "frame"
        ][
            "dataset"
        ].astype(str)
    )

    assert set(
        test[
            "frame"
        ][
            "dataset"
        ].astype(str)
    ) == {
        held
    }


    pair_initial_hashes = []


    for tag in [
        "MLP",
        "GCN",
    ]:

        rec = record[
            "models"
        ][
            tag
        ]


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


        torch.manual_seed(
            train_seed
        )


        initial_model = MODELS[
            tag
        ]()


        assert parameter_count(
            initial_model
        ) == 5569


        initial_hash = state_tensor_hash(
            initial_model.state_dict()
        )


        assert initial_hash == rec[
            "initial_state_tensor_sha256"
        ]


        pair_initial_hashes.append(
            initial_hash
        )


        model = MODELS[
            tag
        ]()


        model.load_state_dict(
            payload[
                "state_dict"
            ]
        )


        assert state_tensor_hash(
            model.state_dict()
        ) == rec[
            "state_tensor_sha256"
        ]


        training = rec[
            "training"
        ]


        assert training[
            "hit_safety_ceiling"
        ] is False


        assert int(
            training[
                "stop_epoch"
            ]
        ) < 10000


        assert (
            int(
                training[
                    "stop_epoch"
                ]
            )
            -
            int(
                training[
                    "best_epoch"
                ]
            )
        ) == 30


        val_pred = predict(
            model,
            tag,
            val,
        )


        val_rmse = metrics(
            val[
                "y"
            ],
            val_pred,
            val[
                "weights"
            ],
        )[
            "RMSE"
        ]


        assert math.isclose(
            float(
                training[
                    "best_source_validation_balanced_rmse"
                ]
            ),
            float(
                val_rmse
            ),
            rel_tol=1e-8,
            abs_tol=1e-8,
        )


        pred = predict(
            model,
            tag,
            test,
        )


        expected = test_bundle(
            test,
            pred,
        )


        assert_metric_tree_close(
            rec[
                "test"
            ],
            expected,
        )


        if tag == "GCN":

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


            causal_edge_checks += int(
                len(
                    valid
                )
            )


        validated_states += 1


    assert pair_initial_hashes[
        0
    ] == pair_initial_hashes[
        1
    ]


    validated_records += 1


assert validated_states == 90
assert validated_records == 45
assert causal_edge_checks > 0


receipt = {
    "contract":
        "lodo_zero_shot_execution_audit_receipt_v1",

    "execution_manifest_sha256":
        sha256(
            MANIFEST
        ),

    "runner_sha256":
        sha256(
            RUNNER
        ),

    "validated_model_states":
        90,

    "validated_paired_records":
        45,

    "paired_initializations_exact":
        True,

    "held_out_target_absent_from_source_training":
        True,

    "held_out_target_absent_from_source_validation":
        True,

    "target_test_contains_only_held_out_dataset":
        True,

    "all_source_validation_best_states_reproduced":
        True,

    "all_target_test_metrics_independently_reproduced":
        True,

    "all_fits_patience_valid_below_ceiling":
        True,

    "causal_test_edge_checks":
        causal_edge_checks,

    "future_edges_detected":
        False,

    "cross_cell_edges_detected":
        False,

    "numeric_test_metrics_surfaced_by_audit":
        False,

    "audit_pass":
        True,
}


atomic_json(
    receipt,
    RECEIPT,
)


print("LODO ZERO-SHOT EXECUTION AUDIT")
print("90 model states hash-exact:", True)
print("45 paired records independently verified:", True)
print("paired initialization identical:", True)
print("held-out target present in source train:", False)
print("held-out target present in source validation:", False)
print("target test contains held-out dataset only:", True)
print("source-validation best states reproduced:", True)
print("all target-test metrics independently reproduced:", True)
print("all fits patience-valid below ceiling:", True)
print("future edges detected:", False)
print("cross-cell edges detected:", False)
print("numeric test metrics surfaced by audit:", False)
print("METRIC_EMBARGO_OUTPUT_GUARD=PASS")
print("audit_receipt_sha256:", sha256(RECEIPT))
print("LODO_ZERO_SHOT_EXECUTION_AUDIT_PASS=True")
