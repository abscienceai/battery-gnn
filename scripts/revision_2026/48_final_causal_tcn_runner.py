from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
import torch


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo" / "battery-gnn"

THIS_SCRIPT = (
    REPO
    / "scripts"
    / "revision_2026"
    / "48_final_causal_tcn_runner.py"
)

BENCH = (
    REPO
    / "scripts"
    / "revision_2026"
    / "24_feature_matched_soh_benchmark.py"
)

TCN_MODULE = (
    REPO
    / "scripts"
    / "revision_2026"
    / "44_causal_tcn_sequence_model.py"
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

ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "sequence_controls"
    / "final_tcn_v1"
)

PROTOCOL = ROOT / "protocol.json"
RUNS = ROOT / "runs"

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


EXPECTED = {
    "benchmark":
        "5d621f8d5764e417e41bc518ec72b9608566a087c552b8365d438c2344c06c5e",

    "tcn_module":
        "2fcab55c9e96d4a39e0ba34ec3daad065a75e3d32c98143a5c6fbf6830d6c487",

    "canonical":
        "49a7b7629cf9ee3f8ce0a7e3c52f2807878b752b4c9a30a876285fb891b00c14",

    "assignments":
        "914c8b90131f496ca396f63e0502147ca99848fdfd4b2a5887225b47d055",

    "base_scalers":
        "b9708b9321b521b87bdd05f969558134d7e0148285ef7c50f648c2028cd86de1",

    "protocol":
        "19fe5012478bd1d70d3f580d08ee8391c905858dbbb3b3e1015fb8627a54f2c3",
}

# Correct exact assignment hash guard.
EXPECTED["assignments"] = (
    "914c8b90131f496ca396f63e0502147ca99848fdfd4b2c6b2a5887225b47d055"
)


FEATURES = [
    "v_min","v_q10","v_q25","v_median","v_q75","v_q90",
    "v_max","v_mean","v_std","v_phase_slope",
    "i_min","i_q10","i_q25","i_median","i_q75","i_q90",
    "i_max","i_mean","i_std","i_phase_slope",
]

DATASETS = [
    "HUST",
    "CALCE",
    "MATR",
]

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

MAX_EPOCHS = 5000
PATIENCE = 30
LR = 0.001
WEIGHT_DECAY = 0.00001
THREADS = 4


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
    tmp = Path(
        str(path)
        + ".tmp"
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


def as_bool(s):
    return (
        s.fillna(False)
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


def state_tensor_hash(state):
    h = hashlib.sha256()

    for key in sorted(
        state.keys()
    ):

        t = (
            state[key]
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
                    t.shape
                )
            ).encode(
                "utf-8"
            )
        )

        h.update(
            str(
                t.dtype
            ).encode(
                "utf-8"
            )
        )

        h.update(
            t.numpy().tobytes()
        )

    return h.hexdigest()


def fit_paths(
    split_seed,
    train_seed,
):

    d = (
        RUNS
        / (
            f"split{split_seed}"
            f"_train{train_seed}"
        )
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

        "test_metrics":
            d / "test_metrics.csv",

        "evaluation":
            d / "evaluation.json",
    }


def weighted_r2(
    pred,
    y,
    weight,
):

    pred = (
        pred.detach()
        .cpu()
        .double()
    )

    y = (
        y.detach()
        .cpu()
        .double()
    )

    weight = (
        weight.detach()
        .cpu()
        .double()
    )

    w = weight / weight.sum()

    ybar = torch.sum(
        w * y
    )

    sse = torch.sum(
        w
        * (
            y - pred
        ).square()
    )

    sst = torch.sum(
        w
        * (
            y - ybar
        ).square()
    )

    assert float(
        sst
    ) > 0.0

    return float(
        1.0
        - sse / sst
    )


# =============================================================================
# Immutable guards
# =============================================================================

source_paths = {
    "benchmark":
        BENCH,

    "tcn_module":
        TCN_MODULE,

    "canonical":
        CANON,

    "assignments":
        ASSIGNMENTS,

    "base_scalers":
        BASE_SCALERS,

    "protocol":
        PROTOCOL,
}


for name, expected in EXPECTED.items():

    observed = sha256_file(
        source_paths[
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
    "status"
] == "FINAL_FROZEN_BEFORE_ANY_TCN_TRAINING_OR_TEST"

assert protocol[
    "planned_training_fit_count"
] == 15

assert protocol[
    "learnable_parameters"
] == 5489

assert protocol[
    "sequence_length"
] == 3

assert protocol[
    "training"
][
    "max_epochs"
] == MAX_EPOCHS

assert protocol[
    "training"
][
    "early_stopping_patience"
] == PATIENCE


tracked = subprocess.run(
    [
        "git",
        "ls-files",
        "--error-unmatch",
        str(
            THIS_SCRIPT.relative_to(
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
    THIS_SCRIPT
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
# Imports
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
    "causal_tcn44",
    TCN_MODULE,
)

tcn_mod = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    tcn_mod
)


# =============================================================================
# Canonical data
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


def prepare_split(
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


    scaler = base_scalers[
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
        scaler
    ) == 20

    assert set(
        scaler[
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


    standardized = (
        raw
        - scaler[
            "mean"
        ].to_numpy(
            dtype=np.float64
        )
    ) / scaler[
        "scale"
    ].to_numpy(
        dtype=np.float64
    )


    assert np.isfinite(
        standardized
    ).all()


    sequence = np.empty(
        (
            len(work),
            3,
            20,
        ),
        dtype=np.float64,
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


        s = tcn_mod.build_causal_three_cycle_sequences(
            standardized[
                idx
            ]
        )


        assert s.shape == (
            len(idx),
            3,
            20,
        )


        sequence[
            idx
        ] = s


    assert np.isfinite(
        sequence
    ).all()


    y = pd.to_numeric(
        work[
            "SOH"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )


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
                    sequence[
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

            "rows":
                rows,

            "groups":
                groups,
        }


    return result


# =============================================================================
# Resume-safe training artifacts
# =============================================================================

def inspect_fit(
    split_seed,
    train_seed,
):

    p = fit_paths(
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
        ] == "causal_tcn_fit_v1"

        assert fit[
            "protocol_sha256"
        ] == EXPECTED[
            "protocol"
        ]

        assert fit[
            "runner_sha256"
        ] == RUNNER_SHA

        assert fit[
            "split_seed"
        ] == split_seed

        assert fit[
            "train_seed"
        ] == train_seed

        assert fit[
            "parameter_count"
        ] == 5489

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


        h = pd.read_csv(
            p[
                "history"
            ],
            low_memory=False,
        )


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


        best_epoch = int(
            h.loc[
                h[
                    "validation_balanced_rmse"
                ].idxmin(),
                "epoch",
            ]
        )


        assert best_epoch == fit[
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


def train_one(
    split_seed,
    train_seed,
    data,
):

    bench.set_seed(
        train_seed
    )


    model = tcn_mod.CausalThreeCycleTCN()


    assert tcn_mod.parameter_count(
        model
    ) == 5489


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
            ]
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
                ]
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


        # Strict minimum: exact history argmin == recorded best epoch.
        if val_rmse < best_val:

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
                "TCN_EPOCH_CEILING_HIT "
                f"split={split_seed} "
                f"seed={train_seed}"
            )
        )


    assert stopped_by_patience is True

    assert (
        stop_epoch
        - best_epoch
        == PATIENCE
    )


    p = fit_paths(
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


    tmp_state = Path(
        str(
            p[
                "state"
            ]
        )
        + ".tmp"
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
            "causal_tcn_fit_v1",

        "protocol_sha256":
            EXPECTED[
                "protocol"
            ],

        "runner_sha256":
            RUNNER_SHA,

        "runner_commit_at_execution":
            RUNNER_COMMIT,

        "split_seed":
            split_seed,

        "train_seed":
            train_seed,

        "parameter_count":
            5489,

        "sequence_length":
            3,

        "receptive_field":
            [
                "t-2",
                "t-1",
                "t",
            ],

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
# Resume-safe test evaluation
# =============================================================================

def inspect_evaluation(
    split_seed,
    train_seed,
):

    p = fit_paths(
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
        ] == "causal_tcn_test_evaluation_v1"

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


        return e


    except Exception:
        return None


def evaluate_test(
    split_seed,
    train_seed,
    test,
):

    p = fit_paths(
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


    model = tcn_mod.CausalThreeCycleTCN()

    model.load_state_dict(
        state
    )

    model.eval()


    with torch.no_grad():

        pred = model(
            test[
                "x"
            ]
        )


    assert torch.isfinite(
        pred
    ).all()


    rows = []


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
            "CAUSAL_TCN",

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


    for dataset in DATASETS:

        mask_np = (
            test[
                "dataset"
            ] == dataset
        )

        mask = torch.tensor(
            mask_np,
            dtype=torch.bool,
        )


        w = test[
            "weight"
        ][
            mask
        ].clone()


        w = w / w.sum()


        metric = bench.weighted_metrics(
            pred[
                mask
            ],
            test[
                "y"
            ][
                mask
            ],
            w,
        )


        r2 = weighted_r2(
            pred[
                mask
            ],
            test[
                "y"
            ][
                mask
            ],
            w,
        )


        assert np.isfinite(
            r2
        )


        rows.append({
            "split_seed":
                split_seed,

            "train_seed":
                train_seed,

            "method":
                "CAUSAL_TCN",

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
                r2,
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


    metrics.to_csv(
        p[
            "test_metrics"
        ],
        index=False,
        float_format="%.12g",
    )


    evaluation = {
        "contract":
            "causal_tcn_test_evaluation_v1",

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


# =============================================================================
# Startup safety checks
# =============================================================================

RUNS.mkdir(
    parents=True,
    exist_ok=True,
)


# Test artifacts are forbidden before a training closure exists.
if not TRAINING_CLOSURE.exists():

    assert len(
        list(
            RUNS.rglob(
                "test_metrics.csv"
            )
        )
    ) == 0

    assert len(
        list(
            RUNS.rglob(
                "evaluation.json"
            )
        )
    ) == 0


print("=" * 80)
print("FINAL CAUSAL TCN RUNNER")
print("=" * 80)

print(
    "protocol SHA locked:",
    True
)

print(
    "TCN parameters:",
    5489
)

print(
    "sequence length:",
    3
)

print(
    "planned training fits:",
    15
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
    "reference result files accessed by runner:",
    False
)

print(
    "test rows constructed during training phase:",
    0
)

print(
    "test metric values surfaced during runner:",
    False
)

print()


# =============================================================================
# PHASE 1 — ALL train/validation fits
# =============================================================================

training_records = []

counter = 0


for split_seed in SPLIT_SEEDS:

    data = prepare_split(
        split_seed,
        (
            "train",
            "validation",
        ),
    )


    for train_seed in TRAIN_SEEDS:

        counter += 1


        existing = inspect_fit(
            split_seed,
            train_seed,
        )


        if existing is not None:

            fit = existing

            status = "SKIP"


        else:

            fit = train_one(
                split_seed,
                train_seed,
                data,
            )

            status = "DONE"


        training_records.append(
            fit
        )


        print(
            f"[TRAIN {counter:02d}/15] "
            f"split={split_seed} "
            f"seed={train_seed} "
            f"{status} "
            f"best/stop="
            f"{fit['best_epoch']}/"
            f"{fit['stop_epoch']}"
        )


        atomic_json(
            {
                "contract":
                    "causal_tcn_runner_progress_v1",

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
                    15,

                "test_evaluations_complete":
                    0,

                "test_metric_values_surfaced":
                    False,
            },
            PROGRESS,
        )


assert len(
    training_records
) == 15


# =============================================================================
# Immutable training closure barrier
# =============================================================================

closed_records = []


for split_seed in SPLIT_SEEDS:

    for train_seed in TRAIN_SEEDS:

        fit = inspect_fit(
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
) == 15


# Same train seed starts from same TCN initialization across split seeds.
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
        "causal_tcn_training_closure_v1",

    "protocol_sha256":
        EXPECTED[
            "protocol"
        ],

    "runner_sha256":
        RUNNER_SHA,

    "training_fit_count":
        15,

    "all_training_fits_patience_valid":
        True,

    "all_training_fits_below_epoch_ceiling":
        True,

    "test_rows_constructed_before_closure":
        0,

    "test_metrics_read_before_closure":
        False,

    "same_seed_initialization_equal_across_splits":
        True,

    "fit_records":
        closed_records,
}


if TRAINING_CLOSURE.exists():

    existing = json.loads(
        TRAINING_CLOSURE.read_text(
            encoding="utf-8"
        )
    )

    assert existing == closure

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
print("TCN TRAINING CLOSURE")
print("=" * 80)

print(
    "training fits complete:",
    15,
    "/ 15"
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
    "training closure SHA256:",
    CLOSURE_SHA
)

print()


# =============================================================================
# PHASE 2 — TEST only after closure exists
# =============================================================================

evaluation_records = []

counter = 0


for split_seed in SPLIT_SEEDS:

    test = prepare_split(
        split_seed,
        (
            "test",
        ),
    )[
        "test"
    ]


    for train_seed in TRAIN_SEEDS:

        counter += 1


        existing = inspect_evaluation(
            split_seed,
            train_seed,
        )


        if existing is not None:

            evaluation = existing

            status = "SKIP"


        else:

            evaluation = evaluate_test(
                split_seed,
                train_seed,
                test,
            )

            status = "DONE"


        evaluation_records.append(
            evaluation
        )


        print(
            f"[TEST {counter:02d}/15] "
            f"split={split_seed} "
            f"seed={train_seed} "
            f"{status} "
            f"(values hidden)"
        )


        atomic_json(
            {
                "contract":
                    "causal_tcn_runner_progress_v1",

                "phase":
                    "test_evaluation",

                "protocol_sha256":
                    EXPECTED[
                        "protocol"
                    ],

                "runner_sha256":
                    RUNNER_SHA,

                "training_fits_complete":
                    15,

                "training_fits_planned":
                    15,

                "training_closure_sha256":
                    CLOSURE_SHA,

                "test_evaluations_complete":
                    len(
                        evaluation_records
                    ),

                "test_evaluations_planned":
                    15,

                "test_metric_values_surfaced":
                    False,

                "test_metrics_used_to_change_protocol":
                    False,
            },
            PROGRESS,
        )


assert len(
    evaluation_records
) == 15


# =============================================================================
# Final independent-on-disk reinspection
# =============================================================================

final_fits = []

final_evals = []


for split_seed in SPLIT_SEEDS:

    for train_seed in TRAIN_SEEDS:

        fit = inspect_fit(
            split_seed,
            train_seed,
        )

        evaluation = inspect_evaluation(
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
) == 15

assert len(
    final_evals
) == 15


manifest = {
    "contract":
        "causal_tcn_execution_manifest_v1",

    "protocol_sha256":
        EXPECTED[
            "protocol"
        ],

    "runner_sha256":
        RUNNER_SHA,

    "runner_commit_at_execution":
        RUNNER_COMMIT,

    "completed_training_fit_count":
        15,

    "completed_test_evaluation_count":
        15,

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

    "reference_result_files_accessed_by_runner":
        False,

    "test_metric_values_surfaced_during_runner":
        False,

    "test_metrics_used_to_change_protocol":
        False,

    "device":
        "cpu",

    "fit_records":
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
            "causal_tcn_runner_progress_v1",

        "phase":
            "complete",

        "protocol_sha256":
            EXPECTED[
                "protocol"
            ],

        "runner_sha256":
            RUNNER_SHA,

        "training_fits_complete":
            15,

        "training_fits_planned":
            15,

        "test_evaluations_complete":
            15,

        "test_evaluations_planned":
            15,

        "training_closure_sha256":
            CLOSURE_SHA,

        "execution_manifest_sha256":
            sha256_file(
                EXECUTION_MANIFEST
            ),

        "test_metric_values_surfaced":
            False,

        "test_metrics_used_to_change_protocol":
            False,
    },
    PROGRESS,
)


print()
print("=" * 80)
print("FINAL CAUSAL TCN EXECUTION SUMMARY")
print("=" * 80)

print(
    "training fits:",
    15,
    "/ 15"
)

print(
    "test evaluations:",
    15,
    "/ 15"
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
    "reference result files accessed:",
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
    "FINAL_CAUSAL_TCN_RUNNER_PASS=True"
)
