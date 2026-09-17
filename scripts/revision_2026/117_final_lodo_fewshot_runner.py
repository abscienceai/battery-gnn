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
SUPPORT_SETS = DATA / "fewshot_support_sets.csv"
DATA_MANIFEST = DATA / "lodo_fewshot_data_manifest.json"

ZERO = ROOT / "zero_shot_v1/final_execution_v1"
ZERO_MANIFEST = ZERO / "lodo_zero_shot_execution_manifest.json"
ZERO_AUDIT = ZERO / "lodo_zero_shot_execution_audit_receipt.json"

OUT = ROOT / "few_shot_v1/final_execution_v1"

STATES = OUT / "model_states"
FITMETA = OUT / "fit_metadata"
RECORDS = OUT / "run_records"

STAGE = OUT / "fewshot_adaptation_stage_manifest.json"
MANIFEST = OUT / "lodo_fewshot_execution_manifest.json"

SELF = (
    REPO
    / "scripts/revision_2026"
    / "117_final_lodo_fewshot_runner.py"
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

SPLIT_SEEDS = [
    0, 1, 2, 3, 4,
]

TRAIN_SEEDS = [
    0, 1, 2,
]

CONDITIONS = [
    "K1",
    "K3",
    "K5",
]

ADAPT_LR = 0.0001
WEIGHT_DECAY = 0.00001
ADAPT_EPOCHS = 100


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
        ) + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def atomic_torch(
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


def parameter_count(model):

    return sum(
        int(
            p.numel()
        )
        for p in model.parameters()
        if p.requires_grad
    )


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


# ============================================================
# Frozen input checks
# ============================================================

for path, expected in EXPECTED.items():

    assert path.is_file()

    assert sha256(
        path
    ) == expected


protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)

assert protocol[
    "few_shot_adaptation_contract"
][
    "epochs"
] == 100

assert protocol[
    "few_shot_adaptation_contract"
][
    "learning_rate"
] == 0.0001

assert protocol[
    "few_shot_adaptation_contract"
][
    "early_stopping"
] is False

assert protocol[
    "few_shot_adaptation_contract"
][
    "K3_starts_from_K1_state"
] is False

assert protocol[
    "few_shot_adaptation_contract"
][
    "K5_starts_from_K3_state"
] is False


zero_audit = json.loads(
    ZERO_AUDIT.read_text(
        encoding="utf-8"
    )
)

assert zero_audit[
    "audit_pass"
] is True

assert zero_audit[
    "all_target_test_metrics_independently_reproduced"
] is True


zero_manifest = json.loads(
    ZERO_MANIFEST.read_text(
        encoding="utf-8"
    )
)

assert zero_manifest[
    "execution"
][
    "model_fits"
] == 90

assert zero_manifest[
    "execution"
][
    "target_validation_used_for_training_or_selection"
] is False


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
# Load data
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
# Frozen zero-shot state lookup
# ============================================================

ZERO_STATE_LOOKUP = {}


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

    assert key not in ZERO_STATE_LOOKUP

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

    ZERO_STATE_LOOKUP[
        key
    ] = {
        "path":
            path,

        "sha256":
            item[
                "sha256"
            ],
    }


assert len(
    ZERO_STATE_LOOKUP
) == 90


# ============================================================
# Dataset preparation
# ============================================================

CACHE = {}


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
    ].copy()


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


def build_prev(
    frame,
):

    prev = np.full(
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

            prev[
                idx[
                    1:
                ]
            ] = idx[
                :-1
            ]


    valid = np.flatnonzero(
        prev >= 0
    )


    if len(
        valid
    ):

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


def cell_balanced_weights(
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


    assert len(
        cells
    ) > 0


    for cell in cells:

        idx = np.flatnonzero(
            frame[
                "cell_id"
            ].astype(str)
            .to_numpy()
            == cell
        )

        assert len(
            idx
        ) > 0


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


def support_ids(
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


    expected_k = int(
        condition[
            1:
        ]
    )


    assert len(
        ids
    ) == expected_k

    assert len(
        set(
            ids
        )
    ) == expected_k


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


def support_data(
    held,
    seed,
    condition,
):

    key = (
        "support",
        held,
        seed,
        condition,
    )


    if key in CACHE:

        return CACHE[
            key
        ]


    ids, set_sha, expected_rows = support_ids(
        held,
        seed,
        condition,
    )


    target_assignment = assign[
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
        target_assignment.loc[
            target_assignment[
                "split"
            ]
            == "train",
            "cell_id",
        ].astype(str)
    )

    val_ids = set(
        target_assignment.loc[
            target_assignment[
                "split"
            ]
            == "validation",
            "cell_id",
        ].astype(str)
    )

    test_ids = set(
        target_assignment.loc[
            target_assignment[
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

    assert set(
        frame[
            "cell_id"
        ].astype(str)
        .unique()
    ) == set(
        ids
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

        "y_np":
            y,

        "y_t":
            torch.tensor(
                y,
                dtype=torch.float32,
            ),

        "prev":
            torch.tensor(
                build_prev(
                    frame
                ),
                dtype=torch.long,
            ),

        "weights_np":
            cell_balanced_weights(
                frame
            ),

        "support_cell_ids":
            ids,

        "support_set_sha256":
            set_sha,
    }


    out[
        "weights_t"
    ] = torch.tensor(
        out[
            "weights_np"
        ],
        dtype=torch.float32,
    )


    CACHE[
        key
    ] = out


    return out


def test_data(
    held,
    seed,
):

    key = (
        "test",
        held,
        seed,
    )


    if key in CACHE:

        return CACHE[
            key
        ]


    cells = assign[
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


    assert set(
        frame[
            "dataset"
        ].astype(str)
        .unique()
    ) == {
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

        "y_np":
            y,

        "prev":
            torch.tensor(
                build_prev(
                    frame
                ),
                dtype=torch.long,
            ),

        "weights_np":
            cell_balanced_weights(
                frame
            ),
    }


    CACHE[
        key
    ] = out


    return out


# ============================================================
# Metrics
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


    assert len(
        y
    ) == len(
        pred
    ) == len(
        w
    )

    assert len(
        y
    ) > 0

    assert np.isfinite(
        y
    ).all()

    assert np.isfinite(
        pred
    ).all()

    assert np.isfinite(
        w
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

    mean_y = float(
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
                - mean_y
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


def test_bundle(
    data,
    pred,
):

    frame = data[
        "frame"
    ]


    cell = weighted_metrics(
        data[
            "y_np"
        ],
        pred,
        data[
            "weights_np"
        ],
    )


    row = weighted_metrics(
        data[
            "y_np"
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
# Adaptation
# ============================================================

def fit_paths(
    held,
    split_seed,
    train_seed,
    model_tag,
    condition,
):

    stem = (
        f"held_{held}"
        f"__split{split_seed}"
        f"__train{train_seed}"
        f"__{model_tag}"
        f"__{condition}"
    )


    return (
        STATES / (
            stem + ".pt"
        ),
        FITMETA / (
            stem + ".json"
        ),
    )


def zero_state_info(
    held,
    split_seed,
    train_seed,
    model_tag,
):

    key = (
        held,
        split_seed,
        train_seed,
        model_tag,
    )


    assert key in ZERO_STATE_LOOKUP


    return ZERO_STATE_LOOKUP[
        key
    ]


def adapt_one(
    held,
    split_seed,
    train_seed,
    model_tag,
    condition,
):

    state_path, meta_path = fit_paths(
        held,
        split_seed,
        train_seed,
        model_tag,
        condition,
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
            "held_out"
        ] == held

        assert meta[
            "split_seed"
        ] == split_seed

        assert meta[
            "training_initialization_seed"
        ] == train_seed

        assert meta[
            "model_tag"
        ] == model_tag

        assert meta[
            "condition"
        ] == condition

        assert meta[
            "adaptation_epochs"
        ] == 100

        assert sha256(
            state_path
        ) == meta[
            "adapted_state_file_sha256"
        ]


        return meta


    support = support_data(
        held,
        split_seed,
        condition,
    )


    zero = zero_state_info(
        held,
        split_seed,
        train_seed,
        model_tag,
    )


    zero_payload = torch.load(
        zero[
            "path"
        ],
        map_location="cpu",
        weights_only=False,
    )


    zero_state = zero_payload[
        "state_dict"
    ]


    zero_tensor_hash = state_tensor_hash(
        zero_state
    )


    model = MODELS[
        model_tag
    ]()


    assert parameter_count(
        model
    ) == 5569


    model.load_state_dict(
        zero_state
    )


    assert state_tensor_hash(
        model.state_dict()
    ) == zero_tensor_hash


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ADAPT_LR,
        weight_decay=WEIGHT_DECAY,
    )


    for epoch in range(
        1,
        ADAPT_EPOCHS + 1,
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )


        if model_tag == "MLP":

            pred = model(
                support[
                    "x"
                ]
            ).squeeze(
                1
            )

        else:

            pred = model(
                support[
                    "x"
                ],
                support[
                    "prev"
                ],
            ).squeeze(
                1
            )


        loss = torch.sum(
            support[
                "weights_t"
            ]
            * (
                pred
                - support[
                    "y_t"
                ]
            ) ** 2
        )


        assert torch.isfinite(
            loss
        ).item()


        loss.backward()

        optimizer.step()


    final_state = {
        key:
            value.detach()
            .cpu()
            .clone()
        for key, value
        in model.state_dict().items()
    }


    assert all(
        torch.isfinite(
            value
        ).all().item()
        for value
        in final_state.values()
    )


    payload = {
        "state_dict":
            final_state,

        "held_out":
            held,

        "split_seed":
            split_seed,

        "training_initialization_seed":
            train_seed,

        "model_tag":
            model_tag,

        "condition":
            condition,

        "adaptation_epochs":
            ADAPT_EPOCHS,
    }


    atomic_torch(
        payload,
        state_path,
    )


    meta = {
        "contract":
            "lodo_fewshot_fit_metadata_v1",

        "protocol_sha256":
            EXPECTED[
                PROTOCOL
            ],

        "data_manifest_sha256":
            EXPECTED[
                DATA_MANIFEST
            ],

        "zero_shot_execution_manifest_sha256":
            EXPECTED[
                ZERO_MANIFEST
            ],

        "zero_shot_execution_audit_sha256":
            EXPECTED[
                ZERO_AUDIT
            ],

        "runner_sha256":
            RUNNER_SHA,

        "runner_commit":
            RUNNER_COMMIT,

        "held_out":
            held,

        "split_seed":
            split_seed,

        "training_initialization_seed":
            train_seed,

        "model_tag":
            model_tag,

        "condition":
            condition,

        "support_cell_ids":
            support[
                "support_cell_ids"
            ],

        "support_set_sha256":
            support[
                "support_set_sha256"
            ],

        "n_support_cells":
            len(
                support[
                    "support_cell_ids"
                ]
            ),

        "n_support_rows":
            int(
                len(
                    support[
                        "frame"
                    ]
                )
            ),

        "source_only_scaler_retained":
            True,

        "target_validation_used":
            False,

        "target_test_used_during_adaptation":
            False,

        "starting_state":
            "corresponding_zero_shot_best_state",

        "zero_shot_state_path":
            str(
                zero[
                    "path"
                ]
            ),

        "zero_shot_state_file_sha256":
            zero[
                "sha256"
            ],

        "zero_shot_state_tensor_sha256":
            zero_tensor_hash,

        "adaptation_epochs":
            ADAPT_EPOCHS,

        "adaptation_learning_rate":
            ADAPT_LR,

        "weight_decay":
            WEIGHT_DECAY,

        "early_stopping":
            False,

        "adapted_state_path":
            str(
                state_path
            ),

        "adapted_state_file_sha256":
            sha256(
                state_path
            ),

        "adapted_state_tensor_sha256":
            state_tensor_hash(
                final_state
            ),
    }


    atomic_json(
        meta,
        meta_path,
    )


    return meta


# ============================================================
# Stage 1 — ALL 270 adaptations first
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


for held in DATASETS:

    for split_seed in SPLIT_SEEDS:

        block = []


        for condition in CONDITIONS:

            support_data(
                held,
                split_seed,
                condition,
            )


        for train_seed in TRAIN_SEEDS:

            for model_tag in [
                "MLP",
                "GCN",
            ]:

                for condition in CONDITIONS:

                    meta = adapt_one(
                        held,
                        split_seed,
                        train_seed,
                        model_tag,
                        condition,
                    )

                    block.append(
                        meta
                    )

                    all_meta.append(
                        meta
                    )


        assert len(
            block
        ) == 18


        print(
            "FEWSHOT_BLOCK_COMPLETE "
            f"held={held} "
            f"split={split_seed} "
            "adaptations=18 "
            "epochs_each=100 "
            "target_validation_used=False "
            "target_test_used=False",
            flush=True,
        )


assert len(
    all_meta
) == 270


meta_entries = []


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
        meta[
            "condition"
        ],
    )


    meta_entries.append({
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

        "condition":
            meta[
                "condition"
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


stage = {
    "contract":
        "lodo_fewshot_adaptation_stage_v1",

    "protocol_sha256":
        EXPECTED[
            PROTOCOL
        ],

    "data_manifest_sha256":
        EXPECTED[
            DATA_MANIFEST
        ],

    "zero_shot_execution_manifest_sha256":
        EXPECTED[
            ZERO_MANIFEST
        ],

    "zero_shot_execution_audit_sha256":
        EXPECTED[
            ZERO_AUDIT
        ],

    "expected_adaptations":
        270,

    "completed_adaptations":
        270,

    "adaptation_epochs_each":
        100,

    "target_validation_used":
        False,

    "target_test_evaluation_started":
        False,

    "fit_metadata":
        meta_entries,
}


atomic_json(
    stage,
    STAGE,
)


print(
    "ALL_270_FEWSHOT_ADAPTATIONS_COMPLETE=True",
    flush=True,
)

print(
    "TARGET_TEST_EVALUATION_NOT_STARTED_DURING_ADAPTATION=True",
    flush=True,
)


# ============================================================
# Stage 2 — target test evaluation
# ============================================================

def load_adapted_model(
    held,
    split_seed,
    train_seed,
    model_tag,
    condition,
):

    state_path, meta_path = fit_paths(
        held,
        split_seed,
        train_seed,
        model_tag,
        condition,
    )


    meta = json.loads(
        meta_path.read_text(
            encoding="utf-8"
        )
    )


    assert sha256(
        state_path
    ) == meta[
        "adapted_state_file_sha256"
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
            ).squeeze(
                1
            )

        else:

            pred = model(
                data[
                    "x"
                ],
                data[
                    "prev"
                ],
            ).squeeze(
                1
            )


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


for held in DATASETS:

    for split_seed in SPLIT_SEEDS:

        test = test_data(
            held,
            split_seed,
        )


        for train_seed in TRAIN_SEEDS:

            record = {
                "contract":
                    "lodo_fewshot_paired_run_record_v1",

                "protocol_sha256":
                    EXPECTED[
                        PROTOCOL
                    ],

                "data_manifest_sha256":
                    EXPECTED[
                        DATA_MANIFEST
                    ],

                "zero_shot_execution_manifest_sha256":
                    EXPECTED[
                        ZERO_MANIFEST
                    ],

                "runner_sha256":
                    RUNNER_SHA,

                "held_out":
                    held,

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

                model_block = {}


                for condition in CONDITIONS:

                    model, meta = load_adapted_model(
                        held,
                        split_seed,
                        train_seed,
                        model_tag,
                        condition,
                    )


                    pred = predict(
                        model,
                        model_tag,
                        test,
                    )


                    model_block[
                        condition
                    ] = {
                        "state_path":
                            meta[
                                "adapted_state_path"
                            ],

                        "state_file_sha256":
                            meta[
                                "adapted_state_file_sha256"
                            ],

                        "state_tensor_sha256":
                            meta[
                                "adapted_state_tensor_sha256"
                            ],

                        "zero_shot_state_file_sha256":
                            meta[
                                "zero_shot_state_file_sha256"
                            ],

                        "zero_shot_state_tensor_sha256":
                            meta[
                                "zero_shot_state_tensor_sha256"
                            ],

                        "support_cell_ids":
                            meta[
                                "support_cell_ids"
                            ],

                        "support_set_sha256":
                            meta[
                                "support_set_sha256"
                            ],

                        "n_support_rows":
                            meta[
                                "n_support_rows"
                            ],

                        "adaptation_epochs":
                            meta[
                                "adaptation_epochs"
                            ],

                        "test":
                            test_bundle(
                                test,
                                pred,
                            ),
                    }


                record[
                    "models"
                ][
                    model_tag
                ] = model_block


            record_path = (
                RECORDS
                / (
                    f"held_{held}"
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
                    held,

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


            print(
                "FEWSHOT_TEST_RECORD_WRITTEN "
                f"held={held} "
                f"split={split_seed} "
                f"init={train_seed} "
                "numeric_metrics_printed=False",
                flush=True,
            )


assert len(
    record_entries
) == 45


state_entries = []


for meta in all_meta:

    state_path, _ = fit_paths(
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
        meta[
            "condition"
        ],
    )


    state_entries.append({
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

        "condition":
            meta[
                "condition"
            ],

        "path":
            str(
                state_path
            ),

        "sha256":
            sha256(
                state_path
            ),
    })


manifest = {
    "contract":
        "lodo_fewshot_execution_v1",

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

    "zero_shot_execution_manifest_sha256":
        EXPECTED[
            ZERO_MANIFEST
        ],

    "zero_shot_execution_audit_sha256":
        EXPECTED[
            ZERO_AUDIT
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

        "fewshot_conditions":
            3,

        "adaptation_fits":
            270,

        "paired_test_records":
            45,

        "adaptation_epochs_each":
            100,

        "all_conditions_start_independently_from_zero_shot":
            True,

        "source_only_scaler_retained":
            True,

        "target_validation_used":
            False,

        "target_test_used_during_adaptation":
            False,

        "all_adaptations_complete_before_target_test_evaluation":
            True,

        "numeric_test_values_printed_by_runner":
            False,

        "metric_embargo_active_pending_audit":
            True,
    },

    "adapted_states":
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
    "LODO FEW-SHOT FINAL EXECUTION",
    flush=True,
)

print(
    "adaptation fits: 270",
    flush=True,
)

print(
    "paired test records: 45",
    flush=True,
)

print(
    "conditions: K1,K3,K5",
    flush=True,
)

print(
    "adaptation epochs each: 100",
    flush=True,
)

print(
    "target validation used: False",
    flush=True,
)

print(
    "target test used during adaptation: False",
    flush=True,
)

print(
    "all conditions independently start from zero-shot: True",
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
    "LODO_FEWSHOT_EXECUTION_COMPLETE_READY_FOR_AUDIT=True",
    flush=True,
)
