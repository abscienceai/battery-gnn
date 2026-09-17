from pathlib import Path
import copy
import hashlib
import json
import math
import os
import subprocess

import numpy as np
import pandas as pd
import torch
from torch import nn


# ============================================================
# Paths / frozen contract
# ============================================================

BASE = Path.home() / "alper/battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

ROOT = BASE / "results/revision_2026/prospective_rul"
DATA = ROOT / "data_v1"

PROTOCOL = ROOT / "prospective_rul_training_protocol_v1.json"

TARGET = DATA / "prospective_rul_target_table.csv"
ASSIGN = DATA / "inherited_cell_split_assignments.csv"
FEATURE_SCALER = DATA / "train_only_feature_scaler_parameters.csv"
TARGET_SCALER = DATA / "train_only_target_scaler_parameters.csv"

OUT = ROOT / "training_v1/final_execution_v1"
STATES = OUT / "model_states"
FITMETA = OUT / "fit_metadata"
RECORDS = OUT / "run_records"
MANIFEST = OUT / "prospective_rul_execution_manifest.json"

SELF = (
    REPO
    / "scripts/revision_2026"
    / "107_final_prospective_rul_runner.py"
)


EXPECTED = {
    PROTOCOL:
        "1c43d2f63904d19cf94921be5020f23d01baa4b9d6d3c6badf7560bb15f36bb7",

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

SPLIT_SEEDS = [0, 1, 2, 3, 4]
TRAIN_SEEDS = [0, 1, 2]

LR = 0.001
WEIGHT_DECAY = 0.00001
PATIENCE = 30
MAX_EPOCHS = 10000


# ============================================================
# Deterministic CPU execution
# ============================================================

torch.set_num_threads(
    max(
        1,
        min(
            8,
            os.cpu_count() or 1,
        ),
    )
)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

torch.use_deterministic_algorithms(True)


# ============================================================
# Utilities
# ============================================================

def sha256(path):

    h = hashlib.sha256()

    with Path(path).open("rb") as f:

        while True:

            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


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

    os.replace(tmp, path)


def atomic_torch(obj, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(str(path) + ".tmp")

    torch.save(
        obj,
        tmp,
    )

    os.replace(
        tmp,
        path,
    )


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


def parameter_count(model):

    return sum(
        int(p.numel())
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# Models
# ============================================================

def causal_mean(x, prev_index):

    prev_safe = torch.clamp(
        prev_index,
        min=0,
    )

    prev_x = x[
        prev_safe
    ]

    mask = (
        prev_index >= 0
    ).unsqueeze(1)

    return torch.where(
        mask,
        0.5 * (
            x
            + prev_x
        ),
        x,
    )


class FeatureMatchedMLP_RUL(nn.Module):

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
        prev_index=None,
    ):

        h = torch.relu(
            self.fc1(x)
        )

        h = torch.relu(
            self.fc2(h)
        )

        return self.out(h)


class FeatureMatchedCausalGCN_RUL(nn.Module):

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
        prev_index,
    ):

        h = causal_mean(
            x,
            prev_index,
        )

        h = torch.relu(
            self.fc1(h)
        )

        h = causal_mean(
            h,
            prev_index,
        )

        h = torch.relu(
            self.fc2(h)
        )

        return self.out(h)


MODELS = {
    "MLP":
        FeatureMatchedMLP_RUL,

    "GCN":
        FeatureMatchedCausalGCN_RUL,
}


# ============================================================
# Frozen-input checks
# ============================================================

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
] == "FROZEN_BEFORE_MODEL_TRAINING"

assert protocol[
    "split_and_seed_contract"
][
    "total_model_fits"
] == 30

assert protocol[
    "early_stopping"
][
    "maximum_epochs_safety_ceiling"
] == MAX_EPOCHS


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


RUNNER_SHA = sha256(SELF)
RUNNER_COMMIT = subprocess.check_output(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
    text=True,
).strip()


# ============================================================
# Load frozen data
# ============================================================

target = pd.read_csv(
    TARGET,
    low_memory=False,
)

assign = pd.read_csv(
    ASSIGN,
    low_memory=False,
)

feature_scaler = pd.read_csv(
    FEATURE_SCALER,
    low_memory=False,
)

target_scaler = pd.read_csv(
    TARGET_SCALER,
    low_memory=False,
)


target[
    "observation_index"
] = pd.to_numeric(
    target[
        "observation_index"
    ],
    errors="raise",
).astype(np.int64)

target[
    "feature_valid_rank_one_based"
] = pd.to_numeric(
    target[
        "feature_valid_rank_one_based"
    ],
    errors="raise",
).astype(np.int64)

target[
    "RUL_observations_80pct_k2"
] = pd.to_numeric(
    target[
        "RUL_observations_80pct_k2"
    ],
    errors="raise",
).astype(np.float64)

assign[
    "split_seed"
] = pd.to_numeric(
    assign[
        "split_seed"
    ],
    errors="raise",
).astype(int)


assert len(target) == 64169
assert len(assign) == 505


# ============================================================
# Split preparation
# ============================================================

CACHE = {}


def hierarchical_weights(frame):

    n = len(frame)

    assert n > 0

    w = np.zeros(
        n,
        dtype=np.float64,
    )

    datasets = sorted(
        frame[
            "dataset"
        ].astype(str)
        .unique()
    )

    assert len(datasets) > 0

    for dataset in datasets:

        dm = (
            frame[
                "dataset"
            ].astype(str)
            .to_numpy()
            == dataset
        )

        idx_d = np.flatnonzero(dm)

        cells = sorted(
            frame.iloc[
                idx_d
            ][
                "cell_id"
            ]
            .astype(str)
            .unique()
        )

        for cell in cells:

            local_mask = (
                frame.iloc[
                    idx_d
                ][
                    "cell_id"
                ]
                .astype(str)
                .to_numpy()
                == cell
            )

            idx = idx_d[
                local_mask
            ]

            w[idx] = (
                1.0
                / len(datasets)
                / len(cells)
                / len(idx)
            )

    assert np.isfinite(w).all()

    assert math.isclose(
        float(w.sum()),
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    return w


def build_prev_index(frame):

    prev = np.full(
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

        assert np.all(
            np.diff(
                frame.loc[
                    idx,
                    "observation_index",
                ].to_numpy(
                    dtype=np.int64
                )
            )
            > 0
        )

        if len(idx) > 1:

            prev[
                idx[
                    1:
                ]
            ] = idx[
                :-1
            ]

    return prev


def scaler_for_seed(seed):

    fs = feature_scaler[
        feature_scaler[
            "split_seed"
        ]
        == seed
    ]

    assert len(fs) == 20

    fs = fs.set_index(
        "feature"
    )

    means = np.asarray(
        [
            float(
                fs.loc[
                    f,
                    "mean",
                ]
            )
            for f in FEATURES
        ],
        dtype=np.float64,
    )

    scales = np.asarray(
        [
            float(
                fs.loc[
                    f,
                    "scale",
                ]
            )
            for f in FEATURES
        ],
        dtype=np.float64,
    )

    assert np.isfinite(means).all()
    assert np.isfinite(scales).all()
    assert (scales > 0).all()


    ys = target_scaler[
        target_scaler[
            "split_seed"
        ]
        == seed
    ]

    assert len(ys) == 1

    y_mean = float(
        ys.iloc[0][
            "mean"
        ]
    )

    y_scale = float(
        ys.iloc[0][
            "scale"
        ]
    )

    assert math.isfinite(y_mean)
    assert math.isfinite(y_scale)
    assert y_scale > 0

    return (
        means,
        scales,
        y_mean,
        y_scale,
    )


def prepare_split(seed, split):

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
    ].copy()


    frame = target.merge(
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


    assert len(frame) > 0

    (
        means,
        scales,
        y_mean,
        y_scale,
    ) = scaler_for_seed(
        seed
    )


    X = frame[
        FEATURES
    ].to_numpy(
        dtype=np.float64
    )

    assert np.isfinite(X).all()

    X = (
        X
        - means
    ) / scales


    y_raw = frame[
        "RUL_observations_80pct_k2"
    ].to_numpy(
        dtype=np.float64
    )

    y_scaled = (
        y_raw
        - y_mean
    ) / y_scale


    prev = build_prev_index(
        frame
    )


    data = {
        "frame":
            frame,

        "x":
            torch.tensor(
                X,
                dtype=torch.float32,
            ),

        "y_scaled":
            torch.tensor(
                y_scaled,
                dtype=torch.float32,
            ),

        "y_raw":
            y_raw,

        "prev":
            torch.tensor(
                prev,
                dtype=torch.long,
            ),

        "weights_np":
            hierarchical_weights(
                frame
            ),

        "y_mean":
            y_mean,

        "y_scale":
            y_scale,
    }


    data[
        "weights_t"
    ] = torch.tensor(
        data[
            "weights_np"
        ],
        dtype=torch.float32,
    )


    CACHE[
        key
    ] = data

    return data


# ============================================================
# Metric functions
# ============================================================

def weighted_metrics(
    y,
    pred,
    weights,
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
        weights,
        dtype=np.float64,
    )

    assert len(y) == len(pred) == len(w)
    assert len(y) > 0
    assert np.isfinite(y).all()
    assert np.isfinite(pred).all()
    assert np.isfinite(w).all()
    assert (w >= 0).all()

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

    y_mean = float(
        np.sum(
            w
            * y
        )
    )

    sst = float(
        np.sum(
            w
            * (
                y
                - y_mean
            ) ** 2
        )
    )

    assert sst > 0

    r2 = float(
        1.0
        - mse / sst
    )

    mean_abs_y = float(
        np.sum(
            w
            * np.abs(
                y
            )
        )
    )

    nmae = float(
        mae
        / max(
            mean_abs_y,
            1.0,
        )
    )

    for value in [
        rmse,
        mae,
        r2,
        nmae,
    ]:
        assert math.isfinite(value)

    return {
        "RMSE":
            rmse,

        "MAE":
            mae,

        "R2":
            r2,

        "normalized_MAE":
            nmae,
    }


def evaluate_metrics(
    frame,
    y,
    pred,
):

    balanced_w = hierarchical_weights(
        frame
    )

    balanced = weighted_metrics(
        y,
        pred,
        balanced_w,
    )

    balanced[
        "n_rows"
    ] = int(
        len(frame)
    )

    balanced[
        "n_cells"
    ] = int(
        frame[
            [
                "dataset",
                "cell_id",
            ]
        ].drop_duplicates().shape[
            0
        ]
    )


    by_dataset = {}

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

        assert int(
            mask.sum()
        ) > 0

        zf = frame.loc[
            mask
        ]

        w = np.ones(
            int(
                mask.sum()
            ),
            dtype=np.float64,
        )

        metrics = weighted_metrics(
            y[
                mask
            ],
            pred[
                mask
            ],
            w,
        )

        metrics[
            "n_rows"
        ] = int(
            mask.sum()
        )

        metrics[
            "n_cells"
        ] = int(
            zf[
                "cell_id"
            ]
            .astype(str)
            .nunique()
        )

        by_dataset[
            dataset
        ] = metrics


    return {
        "balanced":
            balanced,

        "by_dataset":
            by_dataset,
    }


def validation_rmse(
    data,
    pred_raw,
):

    metrics = weighted_metrics(
        data[
            "y_raw"
        ],
        pred_raw,
        data[
            "weights_np"
        ],
    )

    return float(
        metrics[
            "RMSE"
        ]
    )


# ============================================================
# Fit one model, resume-safe
# ============================================================

def paths_for_fit(
    split_seed,
    train_seed,
    model_tag,
):

    stem = (
        f"split{split_seed}"
        f"__train{train_seed}"
        f"__{model_tag}"
    )

    return (
        STATES / (
            stem + ".pt"
        ),
        FITMETA / (
            stem + ".json"
        ),
    )


def fit_one(
    split_seed,
    train_seed,
    model_tag,
):

    state_path, meta_path = paths_for_fit(
        split_seed,
        train_seed,
        model_tag,
    )


    if state_path.exists() or meta_path.exists():

        assert (
            state_path.exists()
            and meta_path.exists()
        )

        meta = json.loads(
            meta_path.read_text(
                encoding="utf-8"
            )
        )

        assert meta[
            "runner_sha256"
        ] == RUNNER_SHA

        assert meta[
            "training_protocol_sha256"
        ] == EXPECTED[
            PROTOCOL
        ]

        assert meta[
            "split_seed"
        ] == split_seed

        assert meta[
            "training_initialization_seed"
        ] == train_seed

        assert meta[
            "model_tag"
        ] == model_tag

        assert sha256(
            state_path
        ) == meta[
            "state_file_sha256"
        ]

        return meta


    train = prepare_split(
        split_seed,
        "train",
    )

    val = prepare_split(
        split_seed,
        "validation",
    )


    torch.manual_seed(
        train_seed
    )

    model = MODELS[
        model_tag
    ]()


    assert parameter_count(
        model
    ) == 5569


    initial_hash = state_tensor_hash(
        model.state_dict()
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )


    best_val = math.inf
    best_epoch = None
    best_state = None
    bad_epochs = 0
    stop_epoch = None
    hit_ceiling = False


    x_train = train[
        "x"
    ]

    y_train = train[
        "y_scaled"
    ]

    w_train = train[
        "weights_t"
    ]

    p_train = train[
        "prev"
    ]


    x_val = val[
        "x"
    ]

    p_val = val[
        "prev"
    ]


    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        if model_tag == "MLP":

            pred = model(
                x_train
            ).squeeze(1)

        else:

            pred = model(
                x_train,
                p_train,
            ).squeeze(1)


        loss = torch.sum(
            w_train
            * (
                pred
                - y_train
            ) ** 2
        )

        assert torch.isfinite(
            loss
        ).item()

        loss.backward()

        optimizer.step()


        model.eval()

        with torch.no_grad():

            if model_tag == "MLP":

                vp = model(
                    x_val
                ).squeeze(1)

            else:

                vp = model(
                    x_val,
                    p_val,
                ).squeeze(1)


            vp_raw = (
                vp.detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64
                )
                * val[
                    "y_scale"
                ]
                + val[
                    "y_mean"
                ]
            )


        score = validation_rmse(
            val,
            vp_raw,
        )


        if score < best_val:

            best_val = score
            best_epoch = epoch

            best_state = {
                k:
                    v.detach()
                    .cpu()
                    .clone()
                for k, v in model.state_dict().items()
            }

            bad_epochs = 0

        else:

            bad_epochs += 1


        if bad_epochs >= PATIENCE:

            stop_epoch = epoch
            break


    if stop_epoch is None:

        stop_epoch = MAX_EPOCHS
        hit_ceiling = True


    if stop_epoch >= MAX_EPOCHS:

        hit_ceiling = True


    assert best_state is not None
    assert best_epoch is not None


    payload = {
        "state_dict":
            best_state,

        "split_seed":
            split_seed,

        "training_initialization_seed":
            train_seed,

        "model_tag":
            model_tag,

        "best_epoch":
            int(
                best_epoch
            ),

        "stop_epoch":
            int(
                stop_epoch
            ),

        "best_validation_balanced_rmse_raw_rul":
            float(
                best_val
            ),
    }


    atomic_torch(
        payload,
        state_path,
    )


    state_file_sha = sha256(
        state_path
    )


    meta = {
        "contract":
            "prospective_rul_fit_metadata_v1",

        "training_protocol_sha256":
            EXPECTED[
                PROTOCOL
            ],

        "runner_sha256":
            RUNNER_SHA,

        "runner_commit":
            RUNNER_COMMIT,

        "split_seed":
            split_seed,

        "training_initialization_seed":
            train_seed,

        "model_tag":
            model_tag,

        "model_name":
            (
                "FeatureMatchedMLP_RUL"
                if model_tag == "MLP"
                else "FeatureMatchedCausalGCN_RUL"
            ),

        "learnable_parameters":
            5569,

        "initial_state_tensor_sha256":
            initial_hash,

        "best_epoch":
            int(
                best_epoch
            ),

        "stop_epoch":
            int(
                stop_epoch
            ),

        "patience":
            PATIENCE,

        "maximum_epochs_safety_ceiling":
            MAX_EPOCHS,

        "hit_safety_ceiling":
            bool(
                hit_ceiling
            ),

        "best_validation_balanced_rmse_raw_rul":
            float(
                best_val
            ),

        "n_train_rows":
            len(
                train[
                    "frame"
                ]
            ),

        "n_validation_rows":
            len(
                val[
                    "frame"
                ]
            ),

        "state_path":
            str(
                state_path
            ),

        "state_file_sha256":
            state_file_sha,

        "state_tensor_sha256":
            state_tensor_hash(
                best_state
            ),

        "test_evaluated_at_fit_time":
            False,
    }


    atomic_json(
        meta,
        meta_path,
    )

    return meta


# ============================================================
# State loading / test evaluation
# ============================================================

def load_best_model(
    split_seed,
    train_seed,
    model_tag,
):

    state_path, meta_path = paths_for_fit(
        split_seed,
        train_seed,
        model_tag,
    )

    meta = json.loads(
        meta_path.read_text(
            encoding="utf-8"
        )
    )

    assert sha256(
        state_path
    ) == meta[
        "state_file_sha256"
    ]


    payload = torch.load(
        state_path,
        map_location="cpu",
        weights_only=False,
    )


    model = MODELS[
        model_tag
    ]()

    model.load_state_dict(
        payload[
            "state_dict"
        ]
    )

    model.eval()

    return (
        model,
        meta,
    )


def predict_raw(
    model,
    model_tag,
    data,
):

    with torch.no_grad():

        if model_tag == "MLP":

            pred = model(
                data[
                    "x"
                ]
            ).squeeze(1)

        else:

            pred = model(
                data[
                    "x"
                ],
                data[
                    "prev"
                ],
            ).squeeze(1)


    pred = (
        pred.detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    return (
        pred
        * data[
            "y_scale"
        ]
        + data[
            "y_mean"
        ]
    )


def test_bundle(
    model,
    model_tag,
    test,
):

    pred_full = predict_raw(
        model,
        model_tag,
        test,
    )

    frame = test[
        "frame"
    ]

    y = test[
        "y_raw"
    ]


    out = {
        "FULL_TRAJECTORY":
            evaluate_metrics(
                frame,
                y,
                pred_full,
            )
    }


    for k in [
        25,
        50,
        100,
    ]:

        mask = (
            frame[
                "feature_valid_rank_one_based"
            ].to_numpy(
                dtype=np.int64
            )
            == k
        )

        assert int(
            mask.sum()
        ) > 0


        ck = frame.loc[
            mask
        ].reset_index(
            drop=True
        )

        assert not ck.duplicated(
            [
                "dataset",
                "cell_id",
            ]
        ).any()


        out[
            f"K{k}"
        ] = evaluate_metrics(
            ck,
            y[
                mask
            ],
            pred_full[
                mask
            ],
        )


    return out


# ============================================================
# Stage 1: ALL training first
# ============================================================

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

STATES.mkdir(
    parents=True,
    exist_ok=True,
)

FITMETA.mkdir(
    parents=True,
    exist_ok=True,
)

RECORDS.mkdir(
    parents=True,
    exist_ok=True,
)


all_fit_meta = []


for split_seed in SPLIT_SEEDS:

    # Force split preparation and all invariant checks once.
    prepare_split(
        split_seed,
        "train",
    )

    prepare_split(
        split_seed,
        "validation",
    )

    prepare_split(
        split_seed,
        "test",
    )


    for train_seed in TRAIN_SEEDS:

        pair = {}

        for model_tag in [
            "MLP",
            "GCN",
        ]:

            pair[
                model_tag
            ] = fit_one(
                split_seed,
                train_seed,
                model_tag,
            )

            all_fit_meta.append(
                pair[
                    model_tag
                ]
            )


        assert (
            pair[
                "MLP"
            ][
                "initial_state_tensor_sha256"
            ]
            ==
            pair[
                "GCN"
            ][
                "initial_state_tensor_sha256"
            ]
        )


        print(
            "TRAIN_PAIR_COMPLETE "
            f"split={split_seed} "
            f"init={train_seed} "
            f"mlp_best={pair['MLP']['best_epoch']} "
            f"mlp_stop={pair['MLP']['stop_epoch']} "
            f"gcn_best={pair['GCN']['best_epoch']} "
            f"gcn_stop={pair['GCN']['stop_epoch']} "
            f"ceiling="
            f"{pair['MLP']['hit_safety_ceiling'] or pair['GCN']['hit_safety_ceiling']}",
            flush=True,
        )


assert len(
    all_fit_meta
) == 30


ceiling_hits = [
    m
    for m in all_fit_meta
    if m[
        "hit_safety_ceiling"
    ]
]


training_stage = {
    "contract":
        "prospective_rul_training_stage_v1",

    "training_protocol_sha256":
        EXPECTED[
            PROTOCOL
        ],

    "runner_sha256":
        RUNNER_SHA,

    "total_expected_fits":
        30,

    "completed_fit_metadata":
        30,

    "safety_ceiling_hit_count":
        len(
            ceiling_hits
        ),

    "test_evaluation_started":
        False,

    "fit_metadata": [
        {
            "path":
                str(
                    paths_for_fit(
                        m[
                            "split_seed"
                        ],
                        m[
                            "training_initialization_seed"
                        ],
                        m[
                            "model_tag"
                        ],
                    )[1]
                ),

            "sha256":
                sha256(
                    paths_for_fit(
                        m[
                            "split_seed"
                        ],
                        m[
                            "training_initialization_seed"
                        ],
                        m[
                            "model_tag"
                        ],
                    )[1]
                ),
        }
        for m in all_fit_meta
    ],
}


atomic_json(
    training_stage,
    OUT / "training_stage_manifest.json",
)


if ceiling_hits:

    print(
        "PROSPECTIVE_RUL_SAFETY_CEILING_HIT=True",
        flush=True,
    )

    print(
        "TEST_EVALUATION_PERFORMED=False",
        flush=True,
    )

    raise SystemExit(2)


print(
    "ALL_30_FITS_PATIENCE_VALID_BELOW_CEILING=True",
    flush=True,
)


# ============================================================
# Stage 2: test evaluation only after all 30 fits pass ceiling
# ============================================================

record_entries = []


for split_seed in SPLIT_SEEDS:

    test = prepare_split(
        split_seed,
        "test",
    )


    for train_seed in TRAIN_SEEDS:

        record = {
            "contract":
                "prospective_rul_paired_run_record_v1",

            "training_protocol_sha256":
                EXPECTED[
                    PROTOCOL
                ],

            "runner_sha256":
                RUNNER_SHA,

            "split_seed":
                split_seed,

            "training_initialization_seed":
                train_seed,

            "models":
                {},
        }


        for model_tag in [
            "MLP",
            "GCN",
        ]:

            model, meta = load_best_model(
                split_seed,
                train_seed,
                model_tag,
            )

            bundle = test_bundle(
                model,
                model_tag,
                test,
            )


            record[
                "models"
            ][
                model_tag
            ] = {
                "model_name":
                    meta[
                        "model_name"
                    ],

                "state_path":
                    meta[
                        "state_path"
                    ],

                "state_file_sha256":
                    meta[
                        "state_file_sha256"
                    ],

                "state_tensor_sha256":
                    meta[
                        "state_tensor_sha256"
                    ],

                "initial_state_tensor_sha256":
                    meta[
                        "initial_state_tensor_sha256"
                    ],

                "training": {
                    "best_epoch":
                        meta[
                            "best_epoch"
                        ],

                    "stop_epoch":
                        meta[
                            "stop_epoch"
                        ],

                    "patience":
                        meta[
                            "patience"
                        ],

                    "hit_safety_ceiling":
                        meta[
                            "hit_safety_ceiling"
                        ],

                    "best_validation_balanced_rmse_raw_rul":
                        meta[
                            "best_validation_balanced_rmse_raw_rul"
                        ],
                },

                "test":
                    bundle,
            }


        assert (
            record[
                "models"
            ][
                "MLP"
            ][
                "initial_state_tensor_sha256"
            ]
            ==
            record[
                "models"
            ][
                "GCN"
            ][
                "initial_state_tensor_sha256"
            ]
        )


        path = (
            RECORDS
            / (
                f"split{split_seed}"
                f"__train{train_seed}.json"
            )
        )

        atomic_json(
            record,
            path,
        )


        record_entries.append({
            "split_seed":
                split_seed,

            "training_initialization_seed":
                train_seed,

            "path":
                str(
                    path
                ),

            "sha256":
                sha256(
                    path
                ),
        })


        print(
            "TEST_RECORD_WRITTEN "
            f"split={split_seed} "
            f"init={train_seed} "
            "numeric_metrics_printed=False",
            flush=True,
        )


assert len(
    record_entries
) == 15


state_entries = []
fit_meta_entries = []


for split_seed in SPLIT_SEEDS:

    for train_seed in TRAIN_SEEDS:

        for model_tag in [
            "MLP",
            "GCN",
        ]:

            sp, mp = paths_for_fit(
                split_seed,
                train_seed,
                model_tag,
            )

            state_entries.append({
                "split_seed":
                    split_seed,

                "training_initialization_seed":
                    train_seed,

                "model_tag":
                    model_tag,

                "path":
                    str(
                        sp
                    ),

                "sha256":
                    sha256(
                        sp
                    ),
            })

            fit_meta_entries.append({
                "split_seed":
                    split_seed,

                "training_initialization_seed":
                    train_seed,

                "model_tag":
                    model_tag,

                "path":
                    str(
                        mp
                    ),

                "sha256":
                    sha256(
                        mp
                    ),
            })


manifest = {
    "contract":
        "prospective_rul_execution_v1",

    "status":
        "COMPLETE_PENDING_INDEPENDENT_EXECUTION_AUDIT",

    "training_protocol_sha256":
        EXPECTED[
            PROTOCOL
        ],

    "runner": {
        "path":
            str(
                SELF
            ),

        "sha256":
            RUNNER_SHA,

        "commit":
            RUNNER_COMMIT,
    },

    "execution": {
        "device":
            "CPU",

        "paired_runs":
            15,

        "model_fits":
            30,

        "safety_ceiling_hits":
            0,

        "all_fits_patience_stopped_below_ceiling":
            True,

        "test_evaluation_started_only_after_all_training_passed":
            True,

        "test_evaluation_complete":
            True,

        "test_metrics_used_for_model_selection":
            False,

        "test_metrics_used_for_protocol_change":
            False,

        "numeric_test_values_printed_by_runner":
            False,

        "metric_embargo_active_pending_audit":
            True,
    },

    "frozen_inputs": {
        str(path):
            expected
        for path, expected in EXPECTED.items()
    },

    "model_states":
        state_entries,

    "fit_metadata":
        fit_meta_entries,

    "run_records":
        record_entries,
}


atomic_json(
    manifest,
    MANIFEST,
)


print(
    "PROSPECTIVE RUL FINAL EXECUTION",
    flush=True,
)

print(
    "paired runs: 15",
    flush=True,
)

print(
    "model fits: 30",
    flush=True,
)

print(
    "safety ceiling hits: 0",
    flush=True,
)

print(
    "test records: 15",
    flush=True,
)

print(
    "numeric test metrics printed: False",
    flush=True,
)

print(
    "test used for model selection: False",
    flush=True,
)

print(
    "execution_manifest_sha256:",
    sha256(
        MANIFEST
    ),
    flush=True,
)

print(
    "PROSPECTIVE_RUL_EXECUTION_COMPLETE_READY_FOR_AUDIT=True",
    flush=True,
)
