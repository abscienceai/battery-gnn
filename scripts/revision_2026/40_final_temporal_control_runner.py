from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch


# =============================================================================
# Paths
# =============================================================================

BASE = Path(
    os.environ.get(
        "BATTERY_REVISION_BASE",
        Path.home()
        / "alper"
        / "battery_revision_2026",
    )
)

REPO = (
    BASE
    / "repo"
    / "battery-gnn"
)

RUNNER = (
    REPO
    / "scripts"
    / "revision_2026"
    / "40_final_temporal_control_runner.py"
)

BENCH = (
    REPO
    / "scripts"
    / "revision_2026"
    / "24_feature_matched_soh_benchmark.py"
)

KERNEL = (
    REPO
    / "scripts"
    / "revision_2026"
    / "32_causal_temporal_filter_kernel.py"
)

CANON = (
    BASE
    / "data"
    / "processed"
    / "revision_2026"
    / "canonical_tasks"
    / "canonical_cycle_supervised_table.csv"
)

ASSIGNMENTS = (
    BASE
    / "data"
    / "processed"
    / "revision_2026"
    / "splits_and_scalers"
    / "cell_split_assignments.csv"
)

BASE_SCALERS = (
    BASE
    / "data"
    / "processed"
    / "revision_2026"
    / "splits_and_scalers"
    / "train_only_scaler_parameters.csv"
)

SECONDARY = (
    BASE
    / "data"
    / "processed"
    / "revision_2026"
    / "temporal_controls"
    / "temporal_filter_secondary_scalers.csv"
)

PROTOCOL = (
    BASE
    / "results"
    / "revision_2026"
    / "temporal_controls"
    / "final_temporal_benchmark_v1"
    / "protocol.json"
)

ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "temporal_controls"
    / "final_temporal_benchmark_v1"
)

RUNS = (
    ROOT
    / "runs"
)

TRAINING_CLOSURE = (
    ROOT
    / "training_closure.json"
)

PROGRESS = (
    ROOT
    / "runner_progress.json"
)

EXECUTION_MANIFEST = (
    ROOT
    / "execution_manifest.json"
)


# =============================================================================
# Frozen hashes
# =============================================================================

EXPECTED = {
    "benchmark":
        "5d621f8d5764e417e41bc518ec72b9608566a087c552b8365d438c2344c06c5e",

    "kernel":
        "23f733540a23954c278722c43a36e994c5431b603bfeb9e7ebd4474451bd3ff7",

    "canonical":
        "49a7b7629cf9ee3f8ce0a7e3c52f2807878b752b4c9a30a876285fb891b00c14",

    "assignments":
        "914c8b90131f496ca396f63e0502147ca99848fdfd4b2c6b2a5887225b47d055",

    "base_scalers":
        "b9708b9321b521b87bdd05f969558134d7e0148285ef7c50f648c2028cd86de1",

    "secondary":
        "98607908e0cb8ee2bb9c102c061990d7282bef20f1535c16e9825a2eb3d1a82c",

    "protocol":
        "b6116666e861c79b7332dbd599594846b86511e420d8305eadf82d30b87f0106",
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

DATASETS = [
    "HUST",
    "CALCE",
    "MATR",
]

METHODS = [
    "moving_average3_mlp",
    "lowpass_binomial3_mlp",
    "savgol5_poly2_mlp",
    "kalman_q0p01_mlp",
]

FILTER_NAME = {
    "moving_average3_mlp":
        "moving_average3",

    "lowpass_binomial3_mlp":
        "lowpass_binomial3",

    "savgol5_poly2_mlp":
        "savgol5_poly2",

    "kalman_q0p01_mlp":
        "kalman_q0p01",
}

SPLIT_SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

TRAIN_SEEDS = [
    0,
    1,
    2,
]

MAX_EPOCHS = 5000
PATIENCE = 30
THREADS = 4

HIDDEN_DIM = 64
LR = 0.001
WEIGHT_DECAY = 0.00001


EXPECTED_ROWS = {
    0: {
        "train": 211243,
        "validation": 45336,
        "test": 43974,
    },
    1: {
        "train": 210640,
        "validation": 43640,
        "test": 46273,
    },
    2: {
        "train": 211505,
        "validation": 45230,
        "test": 43818,
    },
    3: {
        "train": 213879,
        "validation": 40518,
        "test": 46156,
    },
    4: {
        "train": 212447,
        "validation": 44357,
        "test": 43749,
    },
}

EXPECTED_GROUPS = {
    "train": 182,
    "validation": 39,
    "test": 38,
}


# =============================================================================
# Helpers
# =============================================================================

def sha256_file(path):
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
    path = Path(path)

    tmp = path.with_suffix(
        path.suffix + ".tmp"
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


def state_tensor_hash(state):
    h = hashlib.sha256()

    for key in sorted(
        state.keys()
    ):

        tensor = (
            state[
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
                tuple(
                    tensor.shape
                )
            ).encode(
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
            tensor.numpy().tobytes()
        )

    return h.hexdigest()


def as_bool(s):
    return (
        s
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "1.0",
                "yes",
            ]
        )
    )


def current_commit():
    return subprocess.check_output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=REPO,
        text=True,
    ).strip()


def fit_dir(
    method,
    split_seed,
    train_seed,
):
    return (
        RUNS
        / method
        / (
            f"split{split_seed}"
            f"_train{train_seed}"
        )
    )


def fit_paths(
    method,
    split_seed,
    train_seed,
):
    d = fit_dir(
        method,
        split_seed,
        train_seed,
    )

    return {
        "dir":
            d,

        "fit":
            d / "fit.json",

        "history":
            d / "history.csv",

        "state":
            d / "best_state.pt",

        "evaluation":
            d / "evaluation.json",

        "test_metrics":
            d / "test_metrics.csv",
    }


# =============================================================================
# Immutable source guards
# =============================================================================

paths = {
    "benchmark": BENCH,
    "kernel": KERNEL,
    "canonical": CANON,
    "assignments": ASSIGNMENTS,
    "base_scalers": BASE_SCALERS,
    "secondary": SECONDARY,
    "protocol": PROTOCOL,
}


for name, expected in EXPECTED.items():

    observed = sha256_file(
        paths[
            name
        ]
    )

    assert observed == expected, (
        name,
        observed,
        expected,
    )


protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)


assert protocol[
    "contract"
] == "temporal_control_benchmark_v1"

assert protocol[
    "status"
] == "FINAL_FROZEN_BEFORE_NEW_TEMPORAL_TEST_RUNS"

assert protocol[
    "new_temporal_control_methods"
] == METHODS

assert protocol[
    "new_fit_count"
] == 60

assert protocol[
    "max_epochs"
] == MAX_EPOCHS

assert protocol[
    "early_stopping_patience"
] == PATIENCE

assert protocol[
    "kalman_selected_q"
] == 0.01

assert protocol[
    "kalman_selected_r"
] == 1.0


# Runner must already be tracked.
tracked = subprocess.run(
    [
        "git",
        "ls-files",
        "--error-unmatch",
        str(
            RUNNER.relative_to(
                REPO
            )
        ),
    ],
    cwd=REPO,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

assert tracked.returncode == 0


RUNNER_SHA = sha256_file(
    RUNNER
)

RUNNER_COMMIT = current_commit()


# CPU-only execution contract.
assert torch.version.cuda is None
assert torch.cuda.is_available() is False

torch.set_num_threads(
    THREADS
)

try:
    torch.set_num_interop_threads(
        1
    )
except RuntimeError:
    pass


# =============================================================================
# Import frozen model and filters
# =============================================================================

spec = importlib.util.spec_from_file_location(
    "benchmark24",
    BENCH,
)

bench = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    bench
)


spec = importlib.util.spec_from_file_location(
    "causal_filters",
    KERNEL,
)

filters = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    filters
)


# =============================================================================
# Load canonical task data
# =============================================================================

cols = [
    "dataset",
    "split_group",
    "observation_index",
    "SOH",
    "soh_task_eligible",
] + FEATURES


df = pd.read_csv(
    CANON,
    usecols=cols,
    low_memory=False,
)


df = df[
    as_bool(
        df[
            "soh_task_eligible"
        ]
    )
].copy()


assert len(
    df
) == 300553


assignments = pd.read_csv(
    ASSIGNMENTS,
    low_memory=False,
)

base_scalers = pd.read_csv(
    BASE_SCALERS,
    low_memory=False,
)

secondary = pd.read_csv(
    SECONDARY,
    low_memory=False,
)


# =============================================================================
# Balanced cell/dataset weighting
# =============================================================================

def balanced_weights(frame):

    frame = frame.reset_index(
        drop=True
    )

    w = np.zeros(
        len(frame),
        dtype=np.float64,
    )


    assert set(
        frame[
            "dataset"
        ].unique()
    ) == set(
        DATASETS
    )


    for dataset in DATASETS:

        dm = (
            frame[
                "dataset"
            ] == dataset
        )

        groups = (
            frame.loc[
                dm,
                "split_group",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        assert len(
            groups
        ) > 0


        for group in groups:

            gm = (
                dm
                &
                (
                    frame[
                        "split_group"
                    ].astype(str)
                    == group
                )
            )

            n_obs = int(
                gm.sum()
            )

            assert n_obs > 0

            w[
                gm.to_numpy()
            ] = (
                1.0
                / 3.0
                / len(
                    groups
                )
                / n_obs
            )


    assert np.isclose(
        w.sum(),
        1.0,
        rtol=0.0,
        atol=1e-12,
    )


    return torch.tensor(
        w.astype(
            np.float32
        ),
        dtype=torch.float32,
    )


# =============================================================================
# Temporal transforms
# =============================================================================

def apply_method(
    x,
    method,
):

    if method == "moving_average3_mlp":

        return filters.trailing_mean(
            x,
            window=3,
        )


    if method == "lowpass_binomial3_mlp":

        return filters.causal_binomial_lowpass3(
            x
        )


    if method == "savgol5_poly2_mlp":

        return filters.causal_savgol_endpoint(
            x,
            window=5,
            polyorder=2,
        )


    if method == "kalman_q0p01_mlp":

        return filters.causal_kalman_local_level(
            x,
            process_variance=0.01,
            measurement_variance=1.0,
            initial_variance=1.0,
        )


    raise ValueError(
        method
    )


# =============================================================================
# Data preparation
#
# Requested split set is filtered BEFORE trajectories are transformed.
# Because split assignment is cell-level, this never truncates a cell that
# belongs to one of the requested sets.
#
# During training phase only train+validation cells are constructed.
# Test cells are first constructed after training_closure.json exists.
# =============================================================================

def prepare_method_split(
    method,
    split_seed,
    requested_splits,
):

    requested_splits = tuple(
        requested_splits
    )


    a = assignments[
        assignments[
            "split_seed"
        ] == split_seed
    ][
        [
            "split_group",
            "split",
        ]
    ].copy()


    assert len(
        a
    ) == 307

    assert not a[
        "split_group"
    ].duplicated().any()


    work = df.merge(
        a,
        on="split_group",
        how="left",
        validate="many_to_one",
    )


    assert work[
        "split"
    ].notna().all()


    work = work[
        work[
            "split"
        ].isin(
            requested_splits
        )
    ].copy()


    assert set(
        work[
            "split"
        ].unique()
    ) == set(
        requested_splits
    )


    work = work.reset_index(
        drop=True
    )


    # -------------------------------------------------------------------------
    # Stage 1 train-only pooled standardization
    # -------------------------------------------------------------------------

    p0 = base_scalers[
        (
            base_scalers[
                "split_seed"
            ] == split_seed
        )
        &
        (
            base_scalers[
                "scaler_scope"
            ] == "pooled_supervised"
        )
    ].set_index(
        "feature"
    ).loc[
        FEATURES
    ]


    assert len(
        p0
    ) == 20

    assert set(
        p0[
            "fit_split"
        ].astype(str)
    ) == {
        "train_only"
    }


    raw = (
        work[
            FEATURES
        ]
        .apply(
            pd.to_numeric,
            errors="raise",
        )
        .to_numpy(
            dtype=np.float64
        )
    )


    z = (
        raw
        - p0[
            "mean"
        ].to_numpy(
            dtype=np.float64
        )
    ) / p0[
        "scale"
    ].to_numpy(
        dtype=np.float64
    )


    assert np.isfinite(
        z
    ).all()


    # -------------------------------------------------------------------------
    # Strict causal filtering independently within each cell
    # -------------------------------------------------------------------------

    filtered = np.empty_like(
        z
    )


    for _, g in work.groupby(
        "split_group",
        sort=True,
    ):

        g = g.sort_values(
            "observation_index"
        )


        idx = g.index.to_numpy(
            dtype=np.int64
        )


        obs = pd.to_numeric(
            g[
                "observation_index"
            ],
            errors="raise",
        ).to_numpy(
            dtype=np.int64
        )


        if len(obs) > 1:

            assert np.all(
                np.diff(
                    obs
                ) == 1
            )


        out = apply_method(
            z[
                idx
            ],
            method,
        )


        assert out.shape == z[
            idx
        ].shape


        assert np.isfinite(
            out
        ).all()


        filtered[
            idx
        ] = out


    # -------------------------------------------------------------------------
    # Stage 2 method-specific train-only secondary standardization
    # -------------------------------------------------------------------------

    filter_name = FILTER_NAME[
        method
    ]


    p1 = secondary[
        (
            secondary[
                "split_seed"
            ] == split_seed
        )
        &
        (
            secondary[
                "filter_name"
            ] == filter_name
        )
    ].set_index(
        "feature"
    ).loc[
        FEATURES
    ]


    assert len(
        p1
    ) == 20

    assert set(
        p1[
            "fit_split"
        ].astype(str)
    ) == {
        "train_only"
    }


    x = (
        filtered
        - p1[
            "secondary_mean"
        ].to_numpy(
            dtype=np.float64
        )
    ) / p1[
        "secondary_scale"
    ].to_numpy(
        dtype=np.float64
    )


    assert np.isfinite(
        x
    ).all()


    y = pd.to_numeric(
        work[
            "SOH"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )


    assert np.isfinite(
        y
    ).all()


    result = {}


    for split_name in requested_splits:

        mask = (
            work[
                "split"
            ].to_numpy()
            == split_name
        )


        meta = (
            work.loc[
                mask,
                [
                    "dataset",
                    "split_group",
                ],
            ]
            .reset_index(
                drop=True
            )
        )


        rows = int(
            mask.sum()
        )

        groups = int(
            meta[
                "split_group"
            ].nunique()
        )


        assert rows == EXPECTED_ROWS[
            split_seed
        ][
            split_name
        ]


        assert groups == EXPECTED_GROUPS[
            split_name
        ]


        result[
            split_name
        ] = {
            "x":
                torch.tensor(
                    x[
                        mask
                    ].astype(
                        np.float32
                    ),
                    dtype=torch.float32,
                ),

            "y":
                torch.tensor(
                    y[
                        mask
                    ].astype(
                        np.float32
                    ),
                    dtype=torch.float32,
                ),

            "weight":
                balanced_weights(
                    meta
                ),

            "dataset":
                meta[
                    "dataset"
                ].astype(
                    str
                ).to_numpy(),

            "split_group":
                meta[
                    "split_group"
                ].astype(
                    str
                ).to_numpy(),

            "rows":
                rows,

            "groups":
                groups,
        }


    return result


# =============================================================================
# Training artifact inspection
# =============================================================================

def inspect_fit(
    method,
    split_seed,
    train_seed,
):

    p = fit_paths(
        method,
        split_seed,
        train_seed,
    )


    if not (
        p[
            "fit"
        ].is_file()
        and p[
            "history"
        ].is_file()
        and p[
            "state"
        ].is_file()
    ):

        return None


    try:

        fit = json.loads(
            p[
                "fit"
            ].read_text(
                encoding="utf-8"
            )
        )


        assert fit[
            "contract"
        ] == "temporal_control_fit_v1"

        assert fit[
            "protocol_sha256"
        ] == EXPECTED[
            "protocol"
        ]

        assert fit[
            "runner_sha256"
        ] == RUNNER_SHA

        assert fit[
            "method"
        ] == method

        assert fit[
            "split_seed"
        ] == split_seed

        assert fit[
            "train_seed"
        ] == train_seed

        assert fit[
            "parameter_count"
        ] == 5569

        assert fit[
            "max_epochs"
        ] == MAX_EPOCHS

        assert fit[
            "patience"
        ] == PATIENCE

        assert fit[
            "test_rows_constructed_during_training"
        ] == 0

        assert fit[
            "test_metrics_read_during_training"
        ] is False

        assert fit[
            "stopped_by_patience"
        ] is True

        assert fit[
            "hit_epoch_ceiling"
        ] is False


        h = pd.read_csv(
            p[
                "history"
            ],
            low_memory=False,
        )


        assert list(
            h.columns
        ) == [
            "epoch",
            "train_weighted_mse",
            "validation_balanced_rmse",
        ]


        assert len(
            h
        ) == fit[
            "stop_epoch"
        ]


        assert int(
            h[
                "epoch"
            ].iloc[
                -1
            ]
        ) == fit[
            "stop_epoch"
        ]


        best = int(
            h.loc[
                h[
                    "validation_balanced_rmse"
                ].idxmin(),
                "epoch",
            ]
        )


        assert best == fit[
            "best_epoch"
        ]


        assert (
            fit[
                "stop_epoch"
            ]
            - fit[
                "best_epoch"
            ]
            == PATIENCE
        )


        assert fit[
            "stop_epoch"
        ] < MAX_EPOCHS


        assert sha256_file(
            p[
                "history"
            ]
        ) == fit[
            "history_sha256"
        ]


        assert sha256_file(
            p[
                "state"
            ]
        ) == fit[
            "state_file_sha256"
        ]


        state = torch.load(
            p[
                "state"
            ],
            map_location="cpu",
            weights_only=True,
        )


        assert state_tensor_hash(
            state
        ) == fit[
            "best_state_tensor_sha256"
        ]


        return fit


    except Exception:
        return None


# =============================================================================
# Train one fit
# =============================================================================

def train_one(
    method,
    split_seed,
    train_seed,
    data,
):

    bench.set_seed(
        train_seed
    )


    model = bench.FeatureMatchedMLP(
        input_dim=20,
        hidden_dim=HIDDEN_DIM,
    )


    assert bench.parameter_count(
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


    train = data[
        "train"
    ]

    val = data[
        "validation"
    ]


    best_val = np.inf
    best_epoch = 0
    best_state = None

    wait = 0
    stopped_by_patience = False

    history = []

    start = time.time()


    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )


        pred = model(
            train[
                "x"
            ],
            None,
        )


        err = (
            pred
            - train[
                "y"
            ]
        )


        loss = torch.sum(
            train[
                "weight"
            ]
            * err.square()
        )


        assert torch.isfinite(
            loss
        )


        loss.backward()
        optimizer.step()


        model.eval()


        with torch.no_grad():

            vp = model(
                val[
                    "x"
                ],
                None,
            )


            vm = bench.weighted_metrics(
                vp,
                val[
                    "y"
                ],
                val[
                    "weight"
                ],
            )


        val_rmse = float(
            vm[
                "rmse"
            ]
        )


        history.append({
            "epoch":
                epoch,

            "train_weighted_mse":
                float(
                    loss.item()
                ),

            "validation_balanced_rmse":
                val_rmse,
        })


        if val_rmse < (
            best_val
            - 1e-12
        ):

            best_val = val_rmse
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            wait = 0

        else:

            wait += 1


        if wait >= PATIENCE:

            stopped_by_patience = True
            break


    assert best_state is not None


    stop_epoch = int(
        history[
            -1
        ][
            "epoch"
        ]
    )


    hit_ceiling = bool(
        stop_epoch >= MAX_EPOCHS
        and not stopped_by_patience
    )


    if hit_ceiling:

        raise RuntimeError(
            (
                "EPOCH_CEILING_HIT "
                f"method={method} "
                f"split={split_seed} "
                f"train={train_seed}"
            )
        )


    assert stopped_by_patience is True

    assert (
        stop_epoch
        - best_epoch
        == PATIENCE
    )


    p = fit_paths(
        method,
        split_seed,
        train_seed,
    )


    p[
        "dir"
    ].mkdir(
        parents=True,
        exist_ok=True,
    )


    pd.DataFrame(
        history
    ).to_csv(
        p[
            "history"
        ],
        index=False,
        float_format="%.12g",
    )


    tmp_state = p[
        "state"
    ].with_suffix(
        ".pt.tmp"
    )


    torch.save(
        best_state,
        tmp_state,
    )


    os.replace(
        tmp_state,
        p[
            "state"
        ],
    )


    fit = {
        "contract":
            "temporal_control_fit_v1",

        "protocol_sha256":
            EXPECTED[
                "protocol"
            ],

        "runner_sha256":
            RUNNER_SHA,

        "runner_commit_at_execution":
            RUNNER_COMMIT,

        "method":
            method,

        "filter_name":
            FILTER_NAME[
                method
            ],

        "split_seed":
            split_seed,

        "train_seed":
            train_seed,

        "architecture":
            "20 -> 64 -> 64 -> 1",

        "parameter_count":
            5569,

        "initial_state_tensor_sha256":
            initial_hash,

        "best_state_tensor_sha256":
            state_tensor_hash(
                best_state
            ),

        "best_validation_rmse":
            float(
                best_val
            ),

        "best_epoch":
            int(
                best_epoch
            ),

        "stop_epoch":
            stop_epoch,

        "epochs_after_best":
            int(
                stop_epoch
                - best_epoch
            ),

        "stopped_by_patience":
            True,

        "hit_epoch_ceiling":
            False,

        "max_epochs":
            MAX_EPOCHS,

        "patience":
            PATIENCE,

        "learning_rate":
            LR,

        "weight_decay":
            WEIGHT_DECAY,

        "train_rows":
            train[
                "rows"
            ],

        "validation_rows":
            val[
                "rows"
            ],

        "train_groups":
            train[
                "groups"
            ],

        "validation_groups":
            val[
                "groups"
            ],

        "test_rows_constructed_during_training":
            0,

        "test_metrics_read_during_training":
            False,

        "elapsed_seconds":
            float(
                time.time()
                - start
            ),

        "history_sha256":
            sha256_file(
                p[
                    "history"
                ]
            ),

        "state_file_sha256":
            sha256_file(
                p[
                    "state"
                ]
            ),
    }


    atomic_json(
        fit,
        p[
            "fit"
        ],
    )


    return fit


# =============================================================================
# Test evaluation helpers
# =============================================================================

def evaluate_test(
    method,
    split_seed,
    train_seed,
    test,
):

    p = fit_paths(
        method,
        split_seed,
        train_seed,
    )


    state = torch.load(
        p[
            "state"
        ],
        map_location="cpu",
        weights_only=True,
    )


    model = bench.FeatureMatchedMLP(
        input_dim=20,
        hidden_dim=HIDDEN_DIM,
    )


    model.load_state_dict(
        state
    )

    model.eval()


    with torch.no_grad():

        pred = model(
            test[
                "x"
            ],
            None,
        )


    assert torch.isfinite(
        pred
    ).all()


    rows = []


    # -------------------------------------------------------------------------
    # Dataset-balanced + cell-balanced aggregate
    # -------------------------------------------------------------------------

    overall = bench.weighted_metrics(
        pred,
        test[
            "y"
        ],
        test[
            "weight"
        ],
    )


    rows.append({
        "split_seed":
            split_seed,

        "train_seed":
            train_seed,

        "method":
            method,

        "split":
            "test",

        "dataset":
            "BALANCED_ALL",

        "n":
            test[
                "rows"
            ],

        "RMSE":
            float(
                overall[
                    "rmse"
                ]
            ),

        "MAE":
            float(
                overall[
                    "mae"
                ]
            ),

        "R2":
            np.nan,
    })


    # -------------------------------------------------------------------------
    # Individual physical datasets.
    #
    # Renormalizing the frozen balanced weights inside one dataset yields
    # cell-balanced weighting within that dataset.
    # -------------------------------------------------------------------------

    dataset_array = test[
        "dataset"
    ]


    for dataset in DATASETS:

        mask_np = (
            dataset_array
            == dataset
        )


        mask = torch.tensor(
            mask_np,
            dtype=torch.bool,
        )


        sub_w = test[
            "weight"
        ][
            mask
        ].clone()


        sub_w = (
            sub_w
            / sub_w.sum()
        )


        metric = bench.weighted_metrics(
            pred[
                mask
            ],
            test[
                "y"
            ][
                mask
            ],
            sub_w,
        )


        rows.append({
            "split_seed":
                split_seed,

            "train_seed":
                train_seed,

            "method":
                method,

            "split":
                "test",

            "dataset":
                dataset,

            "n":
                int(
                    mask_np.sum()
                ),

            "RMSE":
                float(
                    metric[
                        "rmse"
                    ]
                ),

            "MAE":
                float(
                    metric[
                        "mae"
                    ]
                ),

            "R2":
                float(
                    metric[
                        "r2"
                    ]
                ),
        })


    metrics = pd.DataFrame(
        rows
    )


    assert len(
        metrics
    ) == 4


    assert set(
        metrics[
            "dataset"
        ]
    ) == {
        "BALANCED_ALL",
        "HUST",
        "CALCE",
        "MATR",
    }


    assert np.isfinite(
        metrics[
            "RMSE"
        ]
    ).all()

    assert np.isfinite(
        metrics[
            "MAE"
        ]
    ).all()

    assert np.isfinite(
        metrics.loc[
            metrics[
                "dataset"
            ] != "BALANCED_ALL",
            "R2",
        ]
    ).all()


    metrics.to_csv(
        p[
            "test_metrics"
        ],
        index=False,
        float_format="%.12g",
    )


    evaluation = {
        "contract":
            "temporal_control_test_evaluation_v1",

        "protocol_sha256":
            EXPECTED[
                "protocol"
            ],

        "runner_sha256":
            RUNNER_SHA,

        "training_closure_sha256":
            sha256_file(
                TRAINING_CLOSURE
            ),

        "method":
            method,

        "split_seed":
            split_seed,

        "train_seed":
            train_seed,

        "test_rows":
            test[
                "rows"
            ],

        "test_groups":
            test[
                "groups"
            ],

        "test_evaluated_only_after_training_closure":
            True,

        "test_metrics_used_to_change_protocol":
            False,

        "test_metrics_surfaced_by_runner":
            False,

        "test_metrics_sha256":
            sha256_file(
                p[
                    "test_metrics"
                ]
            ),
    }


    atomic_json(
        evaluation,
        p[
            "evaluation"
        ],
    )


    return evaluation


def inspect_evaluation(
    method,
    split_seed,
    train_seed,
):

    p = fit_paths(
        method,
        split_seed,
        train_seed,
    )


    if not (
        p[
            "evaluation"
        ].is_file()
        and p[
            "test_metrics"
        ].is_file()
    ):

        return None


    try:

        e = json.loads(
            p[
                "evaluation"
            ].read_text(
                encoding="utf-8"
            )
        )


        assert e[
            "contract"
        ] == "temporal_control_test_evaluation_v1"

        assert e[
            "protocol_sha256"
        ] == EXPECTED[
            "protocol"
        ]

        assert e[
            "runner_sha256"
        ] == RUNNER_SHA

        assert e[
            "training_closure_sha256"
        ] == sha256_file(
            TRAINING_CLOSURE
        )

        assert e[
            "method"
        ] == method

        assert e[
            "split_seed"
        ] == split_seed

        assert e[
            "train_seed"
        ] == train_seed

        assert e[
            "test_evaluated_only_after_training_closure"
        ] is True

        assert e[
            "test_metrics_used_to_change_protocol"
        ] is False

        assert e[
            "test_metrics_surfaced_by_runner"
        ] is False

        assert sha256_file(
            p[
                "test_metrics"
            ]
        ) == e[
            "test_metrics_sha256"
        ]


        # Values are read only after the immutable training closure exists.
        m = pd.read_csv(
            p[
                "test_metrics"
            ],
            low_memory=False,
        )


        assert len(
            m
        ) == 4

        assert np.isfinite(
            m[
                "RMSE"
            ]
        ).all()

        assert np.isfinite(
            m[
                "MAE"
            ]
        ).all()


        return e


    except Exception:
        return None


# =============================================================================
# Start report
# =============================================================================

RUNS.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 80)
print("FINAL TEMPORAL CONTROL BENCHMARK RUNNER")
print("=" * 80)

print(
    "protocol SHA locked:",
    True
)

print(
    "runner tracked before execution:",
    True
)

print(
    "runner SHA256:",
    RUNNER_SHA
)

print(
    "new methods:",
    len(
        METHODS
    )
)

print(
    "planned training fits:",
    60
)

print(
    "max_epochs safety ceiling:",
    MAX_EPOCHS
)

print(
    "patience:",
    PATIENCE
)

print(
    "device:",
    "CPU"
)

print(
    "test rows constructed during training phase:",
    0
)

print(
    "test values surfaced during runner:",
    False
)

print()


# =============================================================================
# PHASE 1 — train/validation only
# =============================================================================

training_records = []

counter = 0


for method in METHODS:

    for split_seed in SPLIT_SEEDS:

        # Test cells are not included here.
        data = prepare_method_split(
            method,
            split_seed,
            (
                "train",
                "validation",
            ),
        )


        for train_seed in TRAIN_SEEDS:

            counter += 1


            existing = inspect_fit(
                method,
                split_seed,
                train_seed,
            )


            if existing is not None:

                fit = existing

                print(
                    f"[TRAIN {counter:02d}/60] "
                    f"{method} "
                    f"split={split_seed} "
                    f"seed={train_seed} "
                    f"SKIP best/stop="
                    f"{fit['best_epoch']}/"
                    f"{fit['stop_epoch']}"
                )


            else:

                start = time.time()


                fit = train_one(
                    method,
                    split_seed,
                    train_seed,
                    data,
                )


                print(
                    f"[TRAIN {counter:02d}/60] "
                    f"{method} "
                    f"split={split_seed} "
                    f"seed={train_seed} "
                    f"DONE {time.time()-start:.1f}s "
                    f"best/stop="
                    f"{fit['best_epoch']}/"
                    f"{fit['stop_epoch']}"
                )


            training_records.append(
                fit
            )


            atomic_json(
                {
                    "contract":
                        "temporal_control_runner_progress_v1",

                    "phase":
                        "training",

                    "protocol_sha256":
                        EXPECTED[
                            "protocol"
                        ],

                    "runner_sha256":
                        RUNNER_SHA,

                    "training_fits_complete":
                        len(
                            training_records
                        ),

                    "training_fits_planned":
                        60,

                    "test_evaluations_complete":
                        0,

                    "test_values_surfaced":
                        False,
                },
                PROGRESS,
            )


assert len(
    training_records
) == 60


# =============================================================================
# Training closure barrier
# =============================================================================

# Reinspect ALL 60 from disk before any test tensor is built.
closed_records = []


for method in METHODS:

    for split_seed in SPLIT_SEEDS:

        for train_seed in TRAIN_SEEDS:

            fit = inspect_fit(
                method,
                split_seed,
                train_seed,
            )

            assert fit is not None

            assert fit[
                "stopped_by_patience"
            ] is True

            assert fit[
                "hit_epoch_ceiling"
            ] is False

            assert fit[
                "stop_epoch"
            ] < MAX_EPOCHS

            assert fit[
                "test_rows_constructed_during_training"
            ] == 0

            assert fit[
                "test_metrics_read_during_training"
            ] is False

            closed_records.append(
                fit
            )


assert len(
    closed_records
) == 60


# Same train_seed must start from exactly the same learnable initialization
# regardless of temporal control or split.
for train_seed in TRAIN_SEEDS:

    hashes = {
        r[
            "initial_state_tensor_sha256"
        ]
        for r in closed_records
        if r[
            "train_seed"
        ] == train_seed
    }

    assert len(
        hashes
    ) == 1


closure = {
    "contract":
        "temporal_control_training_closure_v1",

    "protocol_sha256":
        EXPECTED[
            "protocol"
        ],

    "runner_sha256":
        RUNNER_SHA,

    "training_fit_count":
        60,

    "all_training_fits_patience_valid":
        True,

    "all_training_fits_below_epoch_ceiling":
        True,

    "all_training_fits_test_rows_constructed":
        0,

    "test_metrics_read_before_closure":
        False,

    "initialization_hash_equal_across_methods_for_same_train_seed":
        True,

    "training_fit_records":
        closed_records,
}


if TRAINING_CLOSURE.exists():

    existing_closure = json.loads(
        TRAINING_CLOSURE.read_text(
            encoding="utf-8"
        )
    )

    assert existing_closure == closure

else:

    atomic_json(
        closure,
        TRAINING_CLOSURE,
    )


CLOSURE_SHA = sha256_file(
    TRAINING_CLOSURE
)


print()
print("=" * 80)
print("TRAINING CLOSURE")
print("=" * 80)

print(
    "training fits complete:",
    60,
    "/ 60"
)

print(
    "all fits patience-valid:",
    True
)

print(
    "all fits below 5000 ceiling:",
    True
)

print(
    "test rows constructed before closure:",
    0
)

print(
    "test metrics read before closure:",
    False
)

print(
    "same-seed initialization equal across methods:",
    True
)

print(
    "training closure SHA256:",
    CLOSURE_SHA
)

print()


# =============================================================================
# PHASE 2 — test evaluation
#
# Only now are test cells constructed.
# No test values are printed to stdout.
# =============================================================================

evaluation_records = []

eval_counter = 0


for method in METHODS:

    for split_seed in SPLIT_SEEDS:

        test_data = prepare_method_split(
            method,
            split_seed,
            (
                "test",
            ),
        )[
            "test"
        ]


        for train_seed in TRAIN_SEEDS:

            eval_counter += 1


            existing = inspect_evaluation(
                method,
                split_seed,
                train_seed,
            )


            if existing is not None:

                evaluation = existing

                status_word = "SKIP"


            else:

                evaluation = evaluate_test(
                    method,
                    split_seed,
                    train_seed,
                    test_data,
                )

                status_word = "DONE"


            evaluation_records.append(
                evaluation
            )


            print(
                f"[TEST {eval_counter:02d}/60] "
                f"{method} "
                f"split={split_seed} "
                f"seed={train_seed} "
                f"{status_word} "
                f"(values hidden)"
            )


            atomic_json(
                {
                    "contract":
                        "temporal_control_runner_progress_v1",

                    "phase":
                        "test_evaluation",

                    "protocol_sha256":
                        EXPECTED[
                            "protocol"
                        ],

                    "runner_sha256":
                        RUNNER_SHA,

                    "training_fits_complete":
                        60,

                    "training_fits_planned":
                        60,

                    "training_closure_sha256":
                        CLOSURE_SHA,

                    "test_evaluations_complete":
                        len(
                            evaluation_records
                        ),

                    "test_evaluations_planned":
                        60,

                    "test_values_surfaced":
                        False,

                    "test_values_used_to_change_protocol":
                        False,
                },
                PROGRESS,
            )


assert len(
    evaluation_records
) == 60


# =============================================================================
# Final disk-level verification
# =============================================================================

final_fits = []

final_evals = []


for method in METHODS:

    for split_seed in SPLIT_SEEDS:

        for train_seed in TRAIN_SEEDS:

            fit = inspect_fit(
                method,
                split_seed,
                train_seed,
            )

            evaluation = inspect_evaluation(
                method,
                split_seed,
                train_seed,
            )

            assert fit is not None
            assert evaluation is not None

            final_fits.append(
                fit
            )

            final_evals.append(
                evaluation
            )


assert len(
    final_fits
) == 60

assert len(
    final_evals
) == 60


manifest = {
    "contract":
        "temporal_control_execution_manifest_v1",

    "protocol_sha256":
        EXPECTED[
            "protocol"
        ],

    "runner_sha256":
        RUNNER_SHA,

    "runner_commit_at_execution":
        RUNNER_COMMIT,

    "new_method_count":
        4,

    "completed_training_fit_count":
        60,

    "completed_test_evaluation_count":
        60,

    "training_closure_sha256":
        CLOSURE_SHA,

    "all_training_complete_before_first_test_evaluation":
        True,

    "all_fits_patience_valid":
        True,

    "all_fits_below_epoch_ceiling":
        True,

    "test_rows_constructed_before_training_closure":
        0,

    "test_metrics_read_before_training_closure":
        False,

    "test_metrics_surfaced_during_runner":
        False,

    "test_metrics_used_to_change_protocol":
        False,

    "device":
        "cpu",

    "training_records":
        final_fits,

    "evaluation_records":
        final_evals,
}


atomic_json(
    manifest,
    EXECUTION_MANIFEST,
)


atomic_json(
    {
        "contract":
            "temporal_control_runner_progress_v1",

        "phase":
            "complete",

        "protocol_sha256":
            EXPECTED[
                "protocol"
            ],

        "runner_sha256":
            RUNNER_SHA,

        "training_fits_complete":
            60,

        "training_fits_planned":
            60,

        "test_evaluations_complete":
            60,

        "test_evaluations_planned":
            60,

        "training_closure_sha256":
            CLOSURE_SHA,

        "execution_manifest_sha256":
            sha256_file(
                EXECUTION_MANIFEST
            ),

        "test_values_surfaced":
            False,

        "test_values_used_to_change_protocol":
            False,
    },
    PROGRESS,
)


print()
print("=" * 80)
print("FINAL TEMPORAL CONTROL EXECUTION SUMMARY")
print("=" * 80)

print(
    "training fits:",
    60,
    "/ 60"
)

print(
    "test evaluations:",
    60,
    "/ 60"
)

print(
    "all training complete before first test evaluation:",
    True
)

print(
    "all fits patience-valid:",
    True
)

print(
    "all fits below 5000 ceiling:",
    True
)

print(
    "test rows constructed before training closure:",
    0
)

print(
    "test metrics read before training closure:",
    False
)

print(
    "test metric values surfaced:",
    False
)

print(
    "test metrics used to change protocol:",
    False
)

print(
    "training closure SHA256:",
    CLOSURE_SHA
)

print(
    "execution manifest SHA256:",
    sha256_file(
        EXECUTION_MANIFEST
    )
)

print()
print(
    "FINAL_TEMPORAL_CONTROL_RUNNER_PASS=True"
)
