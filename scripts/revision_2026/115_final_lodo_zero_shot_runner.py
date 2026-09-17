from pathlib import Path
import hashlib
import json
import math
import os
import subprocess

import numpy as np
import pandas as pd
import torch
from torch import nn


BASE = Path.home() / "alper/battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

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

SCALERS = DATA / "lodo_source_only_feature_scalers.csv"
INVENTORY = DATA / "lodo_fold_inventory.csv"
DATA_MANIFEST = DATA / "lodo_fewshot_data_manifest.json"

OUT = ROOT / "zero_shot_v1/final_execution_v1"
STATES = OUT / "model_states"
FITMETA = OUT / "fit_metadata"
RECORDS = OUT / "run_records"

TRAIN_STAGE = OUT / "zero_shot_training_stage_manifest.json"
MANIFEST = OUT / "lodo_zero_shot_execution_manifest.json"

SELF = (
    REPO
    / "scripts/revision_2026"
    / "115_final_lodo_zero_shot_runner.py"
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

    INVENTORY:
        "c5aed9d2ce48a07c725a23e8ebf1fe19fdb102bf71725ec6ada0704a4fc8f97e",

    DATA_MANIFEST:
        "8d3fa7c8b614305608e8db9f9f8d235388e6b304776cc64ffc8aa2fda6312a67",
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

SPLIT_SEEDS = [0, 1, 2, 3, 4]
TRAIN_SEEDS = [0, 1, 2]

LR = 0.001
WEIGHT_DECAY = 0.00001
PATIENCE = 30
MAX_EPOCHS = 10000


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
            str(
                tensor.dtype
            ).encode("utf-8")
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


def causal_mean(
    x,
    prev_index,
):

    safe = torch.clamp(
        prev_index,
        min=0,
    )

    previous = x[
        safe
    ]

    mask = (
        prev_index >= 0
    ).unsqueeze(1)

    return torch.where(
        mask,
        0.5 * (
            x
            + previous
        ),
        x,
    )


class FeatureMatchedMLP(nn.Module):

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


class FeatureMatchedCausalGCN(nn.Module):

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
        FeatureMatchedMLP,

    "GCN":
        FeatureMatchedCausalGCN,
}


# ============================================================
# Frozen-input validation
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
    "contract"
] == "lodo_fewshot_protocol_v1"

assert protocol[
    "zero_shot_model_contract"
][
    "learnable_parameters_each"
] == 5569

assert protocol[
    "zero_shot_model_contract"
][
    "early_stopping_data"
] == "source validation cells only"

assert protocol[
    "lodo_split_contract"
][
    "target_train_cells_used_by_zero_shot_model"
] is False

assert protocol[
    "lodo_split_contract"
][
    "target_validation_cells_used_by_zero_shot_model"
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


RUNNER_SHA = sha256(
    SELF
)

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
# Load canonical data
# ============================================================

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

inventory = pd.read_csv(
    INVENTORY,
    dtype={
        "held_out": "string",
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


assert len(
    primary
) == 300553


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


# ============================================================
# Data preparation
# ============================================================

CACHE = {}


def fold_sources(
    held_out,
):

    return [
        d
        for d in DATASETS
        if d != held_out
    ]


def scaler_for(
    held_out,
    split_seed,
):

    g = scalers[
        (
            scalers[
                "held_out"
            ]
            == held_out
        )
        & (
            scalers[
                "split_seed"
            ]
            == split_seed
        )
    ].copy()


    assert len(
        g
    ) == 20

    assert set(
        g[
            "feature"
        ].astype(str)
    ) == set(
        FEATURES
    )

    assert set(
        g[
            "fit_split"
        ].astype(str)
    ) == {
        "source_train_only"
    }

    assert not g[
        "target_statistics_used"
    ].astype(bool).any()


    g = g.set_index(
        "feature"
    )


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


    assert np.isfinite(
        mean
    ).all()

    assert np.isfinite(
        scale
    ).all()

    assert (
        scale > 0
    ).all()


    return (
        mean,
        scale,
    )


def selected_cells(
    held_out,
    split_seed,
    role,
):

    source = fold_sources(
        held_out
    )

    a = assign[
        assign[
            "split_seed"
        ]
        == split_seed
    ]


    if role == "source_train":

        g = a[
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

        g = a[
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

        g = a[
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
        ]

    else:
        raise ValueError(
            role
        )


    return g[
        [
            "dataset",
            "cell_id",
        ]
    ].drop_duplicates()


def build_prev_index(
    frame,
):

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

        obs = frame.loc[
            idx,
            "observation_index",
        ].to_numpy(
            dtype=np.int64
        )

        assert np.all(
            np.diff(obs) > 0
        )


        if len(idx) > 1:

            prev[
                idx[
                    1:
                ]
            ] = idx[
                :-1
            ]


    # Structural causality barrier.
    valid = np.flatnonzero(
        prev >= 0
    )

    if len(valid):

        earlier = prev[
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


    return prev


def hierarchical_weights(
    frame,
):

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


    for dataset in datasets:

        idx_dataset = np.flatnonzero(
            frame[
                "dataset"
            ].astype(str)
            .to_numpy()
            == dataset
        )


        cells = sorted(
            frame.iloc[
                idx_dataset
            ][
                "cell_id"
            ].astype(str)
            .unique()
        )


        for cell in cells:

            local = (
                frame.iloc[
                    idx_dataset
                ][
                    "cell_id"
                ].astype(str)
                .to_numpy()
                == cell
            )

            idx = idx_dataset[
                local
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


    assert np.isfinite(
        w
    ).all()

    assert math.isclose(
        float(
            w.sum()
        ),
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    return w


def prepare(
    held_out,
    split_seed,
    role,
):

    key = (
        held_out,
        split_seed,
        role,
    )


    if key in CACHE:

        return CACHE[
            key
        ]


    cells = selected_cells(
        held_out,
        split_seed,
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


    assert len(
        frame
    ) > 0


    source = set(
        fold_sources(
            held_out
        )
    )

    observed_datasets = set(
        frame[
            "dataset"
        ].astype(str)
        .unique()
    )


    if role in {
        "source_train",
        "source_validation",
    }:

        assert observed_datasets == source
        assert held_out not in observed_datasets

    else:

        assert role == "target_test"
        assert observed_datasets == {
            held_out
        }


    mean, scale = scaler_for(
        held_out,
        split_seed,
    )


    X = frame[
        FEATURES
    ].to_numpy(
        dtype=np.float64
    )


    X = (
        X
        - mean
    ) / scale


    y = frame[
        "SOH"
    ].to_numpy(
        dtype=np.float64
    )


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

        "y_np":
            y,

        "y_t":
            torch.tensor(
                y,
                dtype=torch.float32,
            ),

        "prev":
            torch.tensor(
                prev,
                dtype=torch.long,
            ),

        "weights_np":
            hierarchical_weights(
                frame
            ),
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

    assert np.isfinite(
        y
    ).all()

    assert np.isfinite(
        pred
    ).all()

    assert np.isfinite(
        w
    ).all()

    assert (
        w >= 0
    ).all()


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


def test_metrics(
    data,
    pred,
):

    frame = data[
        "frame"
    ]

    y = data[
        "y_np"
    ]


    cell_balanced = weighted_metrics(
        y,
        pred,
        hierarchical_weights(
            frame
        ),
    )


    row = weighted_metrics(
        y,
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
            **cell_balanced,

            "n_rows":
                int(
                    len(
                        frame
                    )
                ),

            "n_cells":
                int(
                    frame[
                        "cell_id"
                    ].astype(str)
                    .nunique()
                ),
        },

        "ROW": {
            **row,

            "n_rows":
                int(
                    len(
                        frame
                    )
                ),

            "n_cells":
                int(
                    frame[
                        "cell_id"
                    ].astype(str)
                    .nunique()
                ),
        },
    }


# ============================================================
# Fit one zero-shot state
# ============================================================

def fit_paths(
    held_out,
    split_seed,
    train_seed,
    model_tag,
):

    stem = (
        f"held_{held_out}"
        f"__split{split_seed}"
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
    held_out,
    split_seed,
    train_seed,
    model_tag,
):

    state_path, meta_path = fit_paths(
        held_out,
        split_seed,
        train_seed,
        model_tag,
    )


    if (
        state_path.exists()
        or meta_path.exists()
    ):

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
            "protocol_sha256"
        ] == EXPECTED[
            PROTOCOL
        ]

        assert meta[
            "held_out"
        ] == held_out

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


    train = prepare(
        held_out,
        split_seed,
        "source_train",
    )

    val = prepare(
        held_out,
        split_seed,
        "source_validation",
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
                train[
                    "x"
                ]
            ).squeeze(1)

        else:

            pred = model(
                train[
                    "x"
                ],
                train[
                    "prev"
                ],
            ).squeeze(1)


        loss = torch.sum(
            train[
                "weights_t"
            ]
            * (
                pred
                - train[
                    "y_t"
                ]
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

                val_pred = model(
                    val[
                        "x"
                    ]
                ).squeeze(1)

            else:

                val_pred = model(
                    val[
                        "x"
                    ],
                    val[
                        "prev"
                    ],
                ).squeeze(1)


        val_pred_np = (
            val_pred.detach()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )


        val_score = weighted_metrics(
            val[
                "y_np"
            ],
            val_pred_np,
            val[
                "weights_np"
            ],
        )[
            "RMSE"
        ]


        assert math.isfinite(
            val_score
        )


        if val_score < best_val:

            best_val = val_score
            best_epoch = epoch

            best_state = {
                key:
                    value.detach()
                    .cpu()
                    .clone()
                for key, value
                in model.state_dict().items()
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

        "held_out":
            held_out,

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

        "best_source_validation_balanced_rmse":
            float(
                best_val
            ),
    }


    atomic_torch(
        payload,
        state_path,
    )


    meta = {
        "contract":
            "lodo_zero_shot_fit_metadata_v1",

        "protocol_sha256":
            EXPECTED[
                PROTOCOL
            ],

        "data_manifest_sha256":
            EXPECTED[
                DATA_MANIFEST
            ],

        "runner_sha256":
            RUNNER_SHA,

        "runner_commit":
            RUNNER_COMMIT,

        "held_out":
            held_out,

        "source_datasets":
            fold_sources(
                held_out
            ),

        "split_seed":
            split_seed,

        "training_initialization_seed":
            train_seed,

        "model_tag":
            model_tag,

        "model_name":
            (
                "FeatureMatchedMLP"
                if model_tag == "MLP"
                else "FeatureMatchedCausalGCN"
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

        "best_source_validation_balanced_rmse":
            float(
                best_val
            ),

        "n_source_train_rows":
            int(
                len(
                    train[
                        "frame"
                    ]
                )
            ),

        "n_source_train_cells":
            int(
                train[
                    "frame"
                ][
                    [
                        "dataset",
                        "cell_id",
                    ]
                ].drop_duplicates().shape[
                    0
                ]
            ),

        "n_source_validation_rows":
            int(
                len(
                    val[
                        "frame"
                    ]
                )
            ),

        "n_source_validation_cells":
            int(
                val[
                    "frame"
                ][
                    [
                        "dataset",
                        "cell_id",
                    ]
                ].drop_duplicates().shape[
                    0
                ]
            ),

        "state_path":
            str(
                state_path
            ),

        "state_file_sha256":
            sha256(
                state_path
            ),

        "state_tensor_sha256":
            state_tensor_hash(
                best_state
            ),

        "held_out_target_seen_during_training":
            False,

        "target_test_evaluated_at_fit_time":
            False,
    }


    atomic_json(
        meta,
        meta_path,
    )


    return meta


# ============================================================
# Train ALL 90 fits first
# ============================================================

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


all_meta = []


for held_out in DATASETS:

    for split_seed in SPLIT_SEEDS:

        prepare(
            held_out,
            split_seed,
            "source_train",
        )

        prepare(
            held_out,
            split_seed,
            "source_validation",
        )

        prepare(
            held_out,
            split_seed,
            "target_test",
        )


        block_meta = []


        for train_seed in TRAIN_SEEDS:

            pair = {}


            for model_tag in [
                "MLP",
                "GCN",
            ]:

                pair[
                    model_tag
                ] = fit_one(
                    held_out,
                    split_seed,
                    train_seed,
                    model_tag,
                )

                block_meta.append(
                    pair[
                        model_tag
                    ]
                )

                all_meta.append(
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


        assert len(
            block_meta
        ) == 6


        print(
            "ZERO_SHOT_BLOCK_COMPLETE "
            f"held={held_out} "
            f"split={split_seed} "
            "fits=6 "
            f"max_stop={max(int(x['stop_epoch']) for x in block_meta)} "
            f"ceiling={any(bool(x['hit_safety_ceiling']) for x in block_meta)}",
            flush=True,
        )


assert len(
    all_meta
) == 90


ceiling_hits = [
    x
    for x in all_meta
    if x[
        "hit_safety_ceiling"
    ]
]


fit_entries = []


for meta in all_meta:

    _, meta_path = fit_paths(
        meta[
            "held_out"
        ],
        meta[
            "split_seed"
        ],
        meta[
            "training_initialization_seed"
        ],
        meta[
            "model_tag"
        ],
    )


    fit_entries.append({
        "held_out":
            meta[
                "held_out"
            ],

        "split_seed":
            meta[
                "split_seed"
            ],

        "training_initialization_seed":
            meta[
                "training_initialization_seed"
            ],

        "model_tag":
            meta[
                "model_tag"
            ],

        "path":
            str(
                meta_path
            ),

        "sha256":
            sha256(
                meta_path
            ),
    })


training_stage = {
    "contract":
        "lodo_zero_shot_training_stage_v1",

    "protocol_sha256":
        EXPECTED[
            PROTOCOL
        ],

    "data_manifest_sha256":
        EXPECTED[
            DATA_MANIFEST
        ],

    "runner_sha256":
        RUNNER_SHA,

    "expected_model_fits":
        90,

    "completed_model_fits":
        90,

    "safety_ceiling_hit_count":
        len(
            ceiling_hits
        ),

    "target_test_evaluation_started":
        False,

    "fit_metadata":
        fit_entries,
}


atomic_json(
    training_stage,
    TRAIN_STAGE,
)


if ceiling_hits:

    print(
        "LODO_ZERO_SHOT_SAFETY_CEILING_HIT=True",
        flush=True,
    )

    print(
        "TARGET_TEST_EVALUATION_PERFORMED=False",
        flush=True,
    )

    raise SystemExit(2)


print(
    "ALL_90_ZERO_SHOT_FITS_PATIENCE_VALID_BELOW_CEILING=True",
    flush=True,
)


# ============================================================
# Test only AFTER all 90 fits are frozen.
# ============================================================

def load_model(
    held_out,
    split_seed,
    train_seed,
    model_tag,
):

    state_path, meta_path = fit_paths(
        held_out,
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


def predict(
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


    assert np.isfinite(
        pred
    ).all()


    return pred


record_entries = []


for held_out in DATASETS:

    for split_seed in SPLIT_SEEDS:

        test = prepare(
            held_out,
            split_seed,
            "target_test",
        )


        for train_seed in TRAIN_SEEDS:

            record = {
                "contract":
                    "lodo_zero_shot_paired_run_record_v1",

                "protocol_sha256":
                    EXPECTED[
                        PROTOCOL
                    ],

                "data_manifest_sha256":
                    EXPECTED[
                        DATA_MANIFEST
                    ],

                "runner_sha256":
                    RUNNER_SHA,

                "held_out":
                    held_out,

                "source_datasets":
                    fold_sources(
                        held_out
                    ),

                "split_seed":
                    split_seed,

                "training_initialization_seed":
                    train_seed,

                "target_test_rows":
                    int(
                        len(
                            test[
                                "frame"
                            ]
                        )
                    ),

                "target_test_cells":
                    int(
                        test[
                            "frame"
                        ][
                            "cell_id"
                        ].astype(str)
                        .nunique()
                    ),

                "models":
                    {},
            }


            for model_tag in [
                "MLP",
                "GCN",
            ]:

                model, meta = load_model(
                    held_out,
                    split_seed,
                    train_seed,
                    model_tag,
                )


                pred = predict(
                    model,
                    model_tag,
                    test,
                )


                metrics = test_metrics(
                    test,
                    pred,
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

                        "best_source_validation_balanced_rmse":
                            meta[
                                "best_source_validation_balanced_rmse"
                            ],

                        "hit_safety_ceiling":
                            meta[
                                "hit_safety_ceiling"
                            ],
                    },

                    "test":
                        metrics,
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


            record_path = (
                RECORDS
                / (
                    f"held_{held_out}"
                    f"__split{split_seed}"
                    f"__train{train_seed}.json"
                )
            )


            atomic_json(
                record,
                record_path,
            )


            record_entries.append({
                "held_out":
                    held_out,

                "split_seed":
                    split_seed,

                "training_initialization_seed":
                    train_seed,

                "path":
                    str(
                        record_path
                    ),

                "sha256":
                    sha256(
                        record_path
                    ),
            })


assert len(
    record_entries
) == 45


state_entries = []
meta_entries = []


for held_out in DATASETS:

    for split_seed in SPLIT_SEEDS:

        for train_seed in TRAIN_SEEDS:

            for model_tag in [
                "MLP",
                "GCN",
            ]:

                state_path, meta_path = fit_paths(
                    held_out,
                    split_seed,
                    train_seed,
                    model_tag,
                )


                state_entries.append({
                    "held_out":
                        held_out,

                    "split_seed":
                        split_seed,

                    "training_initialization_seed":
                        train_seed,

                    "model_tag":
                        model_tag,

                    "path":
                        str(
                            state_path
                        ),

                    "sha256":
                        sha256(
                            state_path
                        ),
                })


                meta_entries.append({
                    "held_out":
                        held_out,

                    "split_seed":
                        split_seed,

                    "training_initialization_seed":
                        train_seed,

                    "model_tag":
                        model_tag,

                    "path":
                        str(
                            meta_path
                        ),

                    "sha256":
                        sha256(
                            meta_path
                        ),
                })


manifest = {
    "contract":
        "lodo_zero_shot_execution_v1",

    "status":
        "COMPLETE_PENDING_INDEPENDENT_EXECUTION_AUDIT",

    "protocol_sha256":
        EXPECTED[
            PROTOCOL
        ],

    "data_manifest_sha256":
        EXPECTED[
            DATA_MANIFEST
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
        "held_out_folds":
            3,

        "split_seeds":
            5,

        "initialization_seeds":
            3,

        "models":
            2,

        "paired_records":
            45,

        "model_fits":
            90,

        "safety_ceiling_hits":
            0,

        "all_training_complete_before_target_test_evaluation":
            True,

        "target_train_used_for_training":
            False,

        "target_validation_used_for_training_or_selection":
            False,

        "target_test_used_for_model_selection":
            False,

        "numeric_test_values_printed_by_runner":
            False,

        "metric_embargo_active_pending_audit":
            True,
    },

    "model_states":
        state_entries,

    "fit_metadata":
        meta_entries,

    "run_records":
        record_entries,
}


atomic_json(
    manifest,
    MANIFEST,
)


print(
    "LODO ZERO-SHOT FINAL EXECUTION",
    flush=True,
)

print(
    "held-out folds: 3",
    flush=True,
)

print(
    "paired records: 45",
    flush=True,
)

print(
    "model fits: 90",
    flush=True,
)

print(
    "safety ceiling hits: 0",
    flush=True,
)

print(
    "target train used: False",
    flush=True,
)

print(
    "target validation used: False",
    flush=True,
)

print(
    "test used for selection: False",
    flush=True,
)

print(
    "numeric test metrics printed: False",
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
    "LODO_ZERO_SHOT_EXECUTION_COMPLETE_READY_FOR_AUDIT=True",
    flush=True,
)
