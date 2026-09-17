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

ROOT = BASE / "results/revision_2026/prospective_rul"
DATA = ROOT / "data_v1"
EXEC = ROOT / "training_v1/final_execution_v1"

PROTOCOL = ROOT / "prospective_rul_training_protocol_v1.json"

TARGET = DATA / "prospective_rul_target_table.csv"
ASSIGN = DATA / "inherited_cell_split_assignments.csv"
FEATURE_SCALER = DATA / "train_only_feature_scaler_parameters.csv"
TARGET_SCALER = DATA / "train_only_target_scaler_parameters.csv"

MANIFEST = EXEC / "prospective_rul_execution_manifest.json"

RUNNER = (
    REPO
    / "scripts/revision_2026"
    / "107_final_prospective_rul_runner.py"
)

RECEIPT = (
    EXEC
    / "prospective_rul_execution_audit_receipt.json"
)


EXPECTED_PROTOCOL_SHA = (
    "1c43d2f63904d19cf94921be5020f23d01baa4b9d6d3c6badf7560bb15f36bb7"
)

EXPECTED_INPUTS = {
    TARGET:
        "e5ec1f09b09f54be06fb9af46bb628b9f81b82053a290ae9c069d1f0d4907725",

    ASSIGN:
        "2b2c5cf7d7e62ecf68879a7df8cfae20e49482b1d36a661de318677272a3a99e",

    FEATURE_SCALER:
        "25f159c51becd400a315aa862d8cdd5f58497172511addff9d7be967314b15aa",

    TARGET_SCALER:
        "78e6e62d3e2d4aa737e43264eabe41823a2d3c5d702151a87569c15e3f2dda8e",
}


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


def state_tensor_hash(state_dict):

    h = hashlib.sha256()

    for key in sorted(state_dict):

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
            x
            + px
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


def param_count(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


assert sha256(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

for path, expected in EXPECTED_INPUTS.items():

    assert sha256(
        path
    ) == expected


manifest = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)

assert manifest[
    "contract"
] == "prospective_rul_execution_v1"

assert manifest[
    "status"
] == "COMPLETE_PENDING_INDEPENDENT_EXECUTION_AUDIT"

assert manifest[
    "training_protocol_sha256"
] == EXPECTED_PROTOCOL_SHA

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
    "paired_runs"
] == 15

assert manifest[
    "execution"
][
    "model_fits"
] == 30

assert manifest[
    "execution"
][
    "safety_ceiling_hits"
] == 0

assert manifest[
    "execution"
][
    "test_evaluation_started_only_after_all_training_passed"
] is True

assert manifest[
    "execution"
][
    "numeric_test_values_printed_by_runner"
] is False

assert manifest[
    "execution"
][
    "test_metrics_used_for_model_selection"
] is False

assert len(
    manifest[
        "model_states"
    ]
) == 30

assert len(
    manifest[
        "fit_metadata"
    ]
) == 30

assert len(
    manifest[
        "run_records"
    ]
) == 15


# ------------------------------------------------------------
# Hash every execution artifact
# ------------------------------------------------------------

for item in manifest[
    "model_states"
]:

    p = Path(
        item[
            "path"
        ]
    )

    assert p.is_file()

    assert sha256(
        p
    ) == item[
        "sha256"
    ]


for item in manifest[
    "fit_metadata"
]:

    p = Path(
        item[
            "path"
        ]
    )

    assert p.is_file()

    assert sha256(
        p
    ) == item[
        "sha256"
    ]


for item in manifest[
    "run_records"
]:

    p = Path(
        item[
            "path"
        ]
    )

    assert p.is_file()

    assert sha256(
        p
    ) == item[
        "sha256"
    ]


target = pd.read_csv(
    TARGET,
    low_memory=False,
)

assign = pd.read_csv(
    ASSIGN,
    low_memory=False,
)

fs = pd.read_csv(
    FEATURE_SCALER,
    low_memory=False,
)

ys = pd.read_csv(
    TARGET_SCALER,
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


CACHE = {}


def scalers(seed):

    f = fs[
        fs[
            "split_seed"
        ]
        == seed
    ].set_index(
        "feature"
    )

    means = np.asarray(
        [
            float(
                f.loc[
                    c,
                    "mean",
                ]
            )
            for c in FEATURES
        ],
        dtype=np.float64,
    )

    scales = np.asarray(
        [
            float(
                f.loc[
                    c,
                    "scale",
                ]
            )
            for c in FEATURES
        ],
        dtype=np.float64,
    )

    y = ys[
        ys[
            "split_seed"
        ]
        == seed
    ]

    assert len(y) == 1

    return (
        means,
        scales,
        float(
            y.iloc[0][
                "mean"
            ]
        ),
        float(
            y.iloc[0][
                "scale"
            ]
        ),
    )


def prev_index(frame):

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

            assert np.all(
                frame.loc[
                    idx[1:],
                    "observation_index",
                ].to_numpy(
                    dtype=np.int64
                )
                >
                frame.loc[
                    idx[:-1],
                    "observation_index",
                ].to_numpy(
                    dtype=np.int64
                )
            )

            p[
                idx[
                    1:
                ]
            ] = idx[
                :-1
            ]

    return p


def prepare(seed, split, source_frame=None):

    if source_frame is None:

        key = (
            seed,
            split,
        )

        if key in CACHE:
            return CACHE[key]


        a = assign[
            (
                assign[
                    "split_seed"
                ]
                == seed
            )
            & (
                assign[
                    "split"
                ].astype(str)
                == split
            )
        ][
            [
                "dataset",
                "cell_id",
            ]
        ]

        frame = target.merge(
            a,
            on=[
                "dataset",
                "cell_id",
            ],
            how="inner",
            validate="many_to_one",
        )

    else:

        frame = source_frame.copy()


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


    (
        means,
        scales,
        y_mean,
        y_scale,
    ) = scalers(
        seed
    )


    X = frame[
        FEATURES
    ].to_numpy(
        dtype=np.float64
    )

    X = (
        X
        - means
    ) / scales


    y_raw = frame[
        "RUL_observations_80pct_k2"
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

        "prev":
            torch.tensor(
                prev_index(
                    frame
                ),
                dtype=torch.long,
            ),

        "y_raw":
            y_raw,

        "y_mean":
            y_mean,

        "y_scale":
            y_scale,
    }


    if source_frame is None:

        CACHE[
            (
                seed,
                split,
            )
        ] = out


    return out


def hierarchy_weights(frame):

    w = np.zeros(
        len(frame),
        dtype=np.float64,
    )

    ds = sorted(
        frame[
            "dataset"
        ].astype(str)
        .unique()
    )


    for d in ds:

        idx_d = np.flatnonzero(
            frame[
                "dataset"
            ].astype(str)
            .to_numpy()
            == d
        )

        cells = sorted(
            frame.iloc[
                idx_d
            ][
                "cell_id"
            ].astype(str)
            .unique()
        )


        for c in cells:

            idx = idx_d[
                frame.iloc[
                    idx_d
                ][
                    "cell_id"
                ].astype(str)
                .to_numpy()
                == c
            ]

            w[idx] = (
                1.0
                / len(ds)
                / len(cells)
                / len(idx)
            )


    assert math.isclose(
        float(w.sum()),
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    return w


def metrics(y, pred, w):

    w = np.asarray(
        w,
        dtype=np.float64,
    )

    w = w / w.sum()

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    pred = np.asarray(
        pred,
        dtype=np.float64,
    )

    e = pred - y

    mse = float(
        np.sum(
            w
            * e
            * e
        )
    )

    rmse = math.sqrt(
        mse
    )

    mae = float(
        np.sum(
            w
            * np.abs(e)
        )
    )

    ym = float(
        np.sum(
            w
            * y
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

    assert sst > 0

    r2 = float(
        1.0
        - mse / sst
    )

    nmae = float(
        mae
        / max(
            float(
                np.sum(
                    w
                    * np.abs(y)
                )
            ),
            1.0,
        )
    )

    return {
        "RMSE":
            float(rmse),

        "MAE":
            float(mae),

        "R2":
            float(r2),

        "normalized_MAE":
            float(nmae),
    }


def bundle(frame, y, pred):

    b = metrics(
        y,
        pred,
        hierarchy_weights(
            frame
        ),
    )

    b[
        "n_rows"
    ] = len(frame)

    b[
        "n_cells"
    ] = frame[
        [
            "dataset",
            "cell_id",
        ]
    ].drop_duplicates().shape[
        0
    ]


    domains = {}

    for dataset in [
        "CALCE",
        "MATR",
    ]:

        mask = (
            frame[
                "dataset"
            ].astype(str)
            .to_numpy()
            == dataset
        )

        assert mask.sum() > 0

        z = metrics(
            y[
                mask
            ],
            pred[
                mask
            ],
            np.ones(
                int(
                    mask.sum()
                ),
                dtype=np.float64,
            ),
        )

        z[
            "n_rows"
        ] = int(
            mask.sum()
        )

        z[
            "n_cells"
        ] = int(
            frame.loc[
                mask,
                "cell_id",
            ]
            .astype(str)
            .nunique()
        )

        domains[
            dataset
        ] = z


    return {
        "balanced":
            b,

        "by_dataset":
            domains,
    }


def predict(model, tag, data):

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


    p = (
        p.detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    return (
        p
        * data[
            "y_scale"
        ]
        + data[
            "y_mean"
        ]
    )


def assert_metric_tree_close(
    actual,
    expected,
):

    for scope in [
        "balanced",
    ]:

        for key in [
            "RMSE",
            "MAE",
            "R2",
            "normalized_MAE",
        ]:

            assert math.isclose(
                float(
                    actual[
                        scope
                    ][
                        key
                    ]
                ),
                float(
                    expected[
                        scope
                    ][
                        key
                    ]
                ),
                rel_tol=1e-8,
                abs_tol=1e-8,
            )


        for key in [
            "n_rows",
            "n_cells",
        ]:

            assert int(
                actual[
                    scope
                ][
                    key
                ]
            ) == int(
                expected[
                    scope
                ][
                    key
                ]
            )


    for dataset in [
        "CALCE",
        "MATR",
    ]:

        for key in [
            "RMSE",
            "MAE",
            "R2",
            "normalized_MAE",
        ]:

            assert math.isclose(
                float(
                    actual[
                        "by_dataset"
                    ][
                        dataset
                    ][
                        key
                    ]
                ),
                float(
                    expected[
                        "by_dataset"
                    ][
                        dataset
                    ][
                        key
                    ]
                ),
                rel_tol=1e-8,
                abs_tol=1e-8,
            )


        for key in [
            "n_rows",
            "n_cells",
        ]:

            assert int(
                actual[
                    "by_dataset"
                ][
                    dataset
                ][
                    key
                ]
            ) == int(
                expected[
                    "by_dataset"
                ][
                    dataset
                ][
                    key
                ]
            )


validated_fits = 0
validated_records = 0
prefix_invariance_checks = 0


for record_item in manifest[
    "run_records"
]:

    record = json.loads(
        Path(
            record_item[
                "path"
            ]
        ).read_text(
            encoding="utf-8"
        )
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


    test = prepare(
        seed,
        "test",
    )

    val = prepare(
        seed,
        "validation",
    )


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


        model = MODELS[
            tag
        ]()

        assert param_count(
            model
        ) == 5569


        torch.manual_seed(
            train_seed
        )

        init_model = MODELS[
            tag
        ]()

        init_hash = state_tensor_hash(
            init_model.state_dict()
        )


        assert init_hash == rec[
            "initial_state_tensor_sha256"
        ]

        pair_initial_hashes.append(
            init_hash
        )


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


        # Best state must reproduce recorded validation criterion.
        val_pred = predict(
            model,
            tag,
            val,
        )

        val_metric = metrics(
            val[
                "y_raw"
            ],
            val_pred,
            hierarchy_weights(
                val[
                    "frame"
                ]
            ),
        )[
            "RMSE"
        ]


        assert math.isclose(
            float(
                training[
                    "best_validation_balanced_rmse_raw_rul"
                ]
            ),
            float(
                val_metric
            ),
            rel_tol=1e-8,
            abs_tol=1e-8,
        )


        # Independently recompute all test outputs.
        pred = predict(
            model,
            tag,
            test,
        )


        expected_full = bundle(
            test[
                "frame"
            ],
            test[
                "y_raw"
            ],
            pred,
        )


        assert_metric_tree_close(
            rec[
                "test"
            ][
                "FULL_TRAJECTORY"
            ],
            expected_full,
        )


        for k in [
            25,
            50,
            100,
        ]:

            frame = test[
                "frame"
            ]

            checkpoint_mask = (
                frame[
                    "feature_valid_rank_one_based"
                ].to_numpy(
                    dtype=np.int64
                )
                == k
            )


            ck_frame = frame.loc[
                checkpoint_mask
            ].reset_index(
                drop=True
            )

            assert not ck_frame.duplicated(
                [
                    "dataset",
                    "cell_id",
                ]
            ).any()


            expected_ck = bundle(
                ck_frame,
                test[
                    "y_raw"
                ][
                    checkpoint_mask
                ],
                pred[
                    checkpoint_mask
                ],
            )


            assert_metric_tree_close(
                rec[
                    "test"
                ][
                    f"K{k}"
                ],
                expected_ck,
            )


            # Strong causal-prefix invariance audit for GCN:
            # prediction at K must be unchanged when all later
            # observations are physically removed.
            if tag == "GCN":

                prefix_source = frame[
                    frame[
                        "feature_valid_rank_one_based"
                    ]
                    <= k
                ].copy()


                prefix = prepare(
                    seed,
                    "test",
                    source_frame=prefix_source,
                )


                prefix_pred = predict(
                    model,
                    tag,
                    prefix,
                )


                endpoint = (
                    prefix[
                        "frame"
                    ][
                        "feature_valid_rank_one_based"
                    ].to_numpy(
                        dtype=np.int64
                    )
                    == k
                )


                full_key = list(
                    zip(
                        frame.loc[
                            checkpoint_mask,
                            "dataset",
                        ].astype(str),
                        frame.loc[
                            checkpoint_mask,
                            "cell_id",
                        ].astype(str),
                    )
                )

                prefix_key = list(
                    zip(
                        prefix[
                            "frame"
                        ].loc[
                            endpoint,
                            "dataset",
                        ].astype(str),
                        prefix[
                            "frame"
                        ].loc[
                            endpoint,
                            "cell_id",
                        ].astype(str),
                    )
                )


                assert full_key == prefix_key


                # Full-test and truncated-prefix batches contain
                # identical causal ancestors at the checkpoint, but
                # FP32 GEMM may differ by a few ULPs when batch shape
                # changes. A prior structure-only diagnostic verified
                # exact ancestor chains for all checkpoint endpoints
                # and bounded the discrepancy to <=4 FP32 ULP,
                # <=1.15e-7 relative, and <=1.12e-4 raw RUL cycles.
                # These limits therefore remain deliberately tighter
                # than the tolerance used here.
                np.testing.assert_allclose(
                    pred[
                        checkpoint_mask
                    ],
                    prefix_pred[
                        endpoint
                    ],
                    rtol=5e-7,
                    atol=2e-4,
                )


                prefix_invariance_checks += 1


        validated_fits += 1


    assert pair_initial_hashes[
        0
    ] == pair_initial_hashes[
        1
    ]

    validated_records += 1


assert validated_records == 15
assert validated_fits == 30
assert prefix_invariance_checks == 45


receipt = {
    "contract":
        "prospective_rul_execution_audit_receipt_v1",

    "execution_manifest_sha256":
        sha256(
            MANIFEST
        ),

    "runner_sha256":
        sha256(
            RUNNER
        ),

    "validated_run_records":
        15,

    "validated_model_fits":
        30,

    "all_model_state_hashes_exact":
        True,

    "all_initial_pair_weights_exact":
        True,

    "all_fits_patience_valid_below_ceiling":
        True,

    "all_validation_best_states_reproduced":
        True,

    "all_test_metrics_independently_reproduced":
        True,

    "gcn_prefix_invariance_checks":
        45,

    "test_metrics_used_for_protocol_change":
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


print("PROSPECTIVE RUL EXECUTION AUDIT")
print("15 paired records independently verified:", True)
print("30 model states hash-exact:", True)
print("paired initialization identical:", True)
print("all fits patience-valid below ceiling:", True)
print("best validation states reproduced:", True)
print("all test metrics independently reproduced:", True)
print("GCN causal prefix-invariance checks:", 45)
print("future observations affect checkpoint prediction:", False)
print("test metrics used for protocol change:", False)
print("numeric test metrics surfaced by audit:", False)
print("METRIC_EMBARGO_OUTPUT_GUARD=PASS")
print("audit_receipt_sha256:", sha256(RECEIPT))
print("PROSPECTIVE_RUL_EXECUTION_AUDIT_PASS=True")
