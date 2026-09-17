from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
import torch


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

PREPROCESS_MANIFEST = (
    BASE
    / "data"
    / "processed"
    / "revision_2026"
    / "temporal_controls"
    / "temporal_filter_preprocessing_manifest.json"
)

ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "temporal_controls"
    / "kalman_validation_selection_v1"
)

ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

PROTOCOL_PATH = (
    ROOT
    / "selection_protocol.json"
)

FIT_SUMMARY_PATH = (
    ROOT
    / "fit_summary.csv"
)

SPLIT_SUMMARY_PATH = (
    ROOT
    / "split_level_validation_summary.csv"
)

Q_SUMMARY_PATH = (
    ROOT
    / "q_validation_summary.csv"
)

SELECTED_PATH = (
    ROOT
    / "selected_kalman.json"
)

MANIFEST_PATH = (
    ROOT
    / "selection_manifest.json"
)


EXPECTED_HASHES = {
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

    "secondary_scalers":
        "98607908e0cb8ee2bb9c102c061990d7282bef20f1535c16e9825a2eb3d1a82c",

    "preprocess_manifest":
        "1397ac3cf505e309fa9d4adf37f95bf58436e4354903a8b7f38345143086e975",
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

Q_GRID = [
    0.001,
    0.01,
    0.1,
    1.0,
]

Q_NAME = {
    0.001: "kalman_q0p001",
    0.01: "kalman_q0p01",
    0.1: "kalman_q0p1",
    1.0: "kalman_q1p0",
}

R = 1.0
INITIAL_VARIANCE = 1.0

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

HIDDEN_DIM = 64
LR = 0.001
WEIGHT_DECAY = 0.00001
MAX_EPOCHS = 5000
PATIENCE = 30
THREADS = 4
TIE_TOLERANCE = 1e-12


EXPECTED_ROWS = {
    0: {
        "train": 211243,
        "validation": 45336,
    },
    1: {
        "train": 210640,
        "validation": 43640,
    },
    2: {
        "train": 211505,
        "validation": 45230,
    },
    3: {
        "train": 213879,
        "validation": 40518,
    },
    4: {
        "train": 212447,
        "validation": 44357,
    },
}


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


# =============================================================================
# Immutable input guards
# =============================================================================

paths = {
    "benchmark": BENCH,
    "kernel": KERNEL,
    "canonical": CANON,
    "assignments": ASSIGNMENTS,
    "base_scalers": BASE_SCALERS,
    "secondary_scalers": SECONDARY,
    "preprocess_manifest": PREPROCESS_MANIFEST,
}


for name, expected in EXPECTED_HASHES.items():

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


# =============================================================================
# Freeze selection rule BEFORE any fitting
# =============================================================================

protocol = {
    "contract":
        "kalman_validation_selection_v1",

    "purpose":
        (
            "Select one global Kalman process-variance q "
            "using train/validation information only."
        ),

    "q_grid":
        Q_GRID,

    "measurement_variance_r":
        R,

    "initial_variance":
        INITIAL_VARIANCE,

    "split_seeds":
        SPLIT_SEEDS,

    "training_initialization_seeds":
        TRAIN_SEEDS,

    "fit_count":
        60,

    "model":
        "FeatureMatchedMLP",

    "architecture":
        "20 -> 64 -> 64 -> 1",

    "hidden_dim":
        HIDDEN_DIM,

    "parameter_count":
        5569,

    "optimizer":
        "AdamW",

    "learning_rate":
        LR,

    "weight_decay":
        WEIGHT_DECAY,

    "max_epochs":
        MAX_EPOCHS,

    "patience":
        PATIENCE,

    "objective":
        "dataset-balanced + cell-balanced MSE",

    "early_stopping_metric":
        "validation dataset-balanced cell-balanced RMSE",

    "selection_rule":
        (
            "For each q, average best validation RMSE across the "
            "three initialization seeds within each split_seed; "
            "then take the unweighted mean across the five "
            "split-level means. Select the lowest global mean."
        ),

    "tie_rule":
        (
            "Candidates within absolute 1e-12 of the minimum "
            "global split-mean validation RMSE are treated as tied; "
            "select the numerically smallest q."
        ),

    "tie_tolerance":
        TIE_TOLERANCE,

    "test_rows_in_model_tensors":
        0,

    "test_metrics_read":
        False,

    "final_test_result_root_accessed":
        False,

    "input_sha256":
        EXPECTED_HASHES,
}


if PROTOCOL_PATH.exists():

    existing = json.loads(
        PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert existing == protocol

else:

    atomic_json(
        protocol,
        PROTOCOL_PATH,
    )


PROTOCOL_SHA = sha256_file(
    PROTOCOL_PATH
)


# =============================================================================
# Import frozen model + filter kernel
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
# Canonical table
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
# Data construction
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

        n_groups = len(
            groups
        )

        assert n_groups > 0


        for group in groups:

            gm = (
                frame[
                    "split_group"
                ].astype(str)
                == group
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
                / n_groups
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


def prepare_q_split(
    q,
    split_seed,
):

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


    # IMPORTANT:
    # Test cells are removed before any model tensor or
    # filter trajectory is constructed.
    work = work[
        work[
            "split"
        ].isin(
            [
                "train",
                "validation",
            ]
        )
    ].copy()


    assert set(
        work[
            "split"
        ].unique()
    ) == {
        "train",
        "validation",
    }


    assert len(
        work[
            work[
                "split"
            ] == "train"
        ]
    ) == EXPECTED_ROWS[
        split_seed
    ][
        "train"
    ]

    assert len(
        work[
            work[
                "split"
            ] == "validation"
        ]
    ) == EXPECTED_ROWS[
        split_seed
    ][
        "validation"
    ]


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


    base_mean = p0[
        "mean"
    ].to_numpy(
        dtype=np.float64
    )

    base_scale = p0[
        "scale"
    ].to_numpy(
        dtype=np.float64
    )


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
        - base_mean
    ) / base_scale


    assert np.isfinite(
        z
    ).all()


    filtered = np.empty_like(
        z
    )


    # Work indices are reset so group positions map exactly
    # to the feature matrix.
    work = work.reset_index(
        drop=True
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
                np.diff(obs) == 1
            )


        filtered[
            idx
        ] = filters.causal_kalman_local_level(
            z[
                idx
            ],
            process_variance=float(
                q
            ),
            measurement_variance=R,
            initial_variance=INITIAL_VARIANCE,
        )


    assert np.isfinite(
        filtered
    ).all()


    filter_name = Q_NAME[
        q
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


    out = {}


    for split_name in (
        "train",
        "validation",
    ):

        mask = (
            work[
                "split"
            ].to_numpy()
            == split_name
        )

        frame = (
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


        out[
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
                    frame
                ),

            "rows":
                int(
                    mask.sum()
                ),

            "groups":
                int(
                    frame[
                        "split_group"
                    ].nunique()
                ),
        }


    assert out[
        "train"
    ][
        "rows"
    ] == EXPECTED_ROWS[
        split_seed
    ][
        "train"
    ]

    assert out[
        "validation"
    ][
        "rows"
    ] == EXPECTED_ROWS[
        split_seed
    ][
        "validation"
    ]


    return out


# =============================================================================
# Fit / resume
# =============================================================================

def fit_dir(
    q,
    split_seed,
    train_seed,
):

    return (
        ROOT
        / Q_NAME[q]
        / (
            f"split{split_seed}"
            f"_train{train_seed}"
        )
    )


def inspect_fit(
    q,
    split_seed,
    train_seed,
):

    d = fit_dir(
        q,
        split_seed,
        train_seed,
    )

    fit_path = (
        d
        / "fit.json"
    )

    history_path = (
        d
        / "history.csv"
    )


    if not (
        fit_path.is_file()
        and history_path.is_file()
    ):
        return None


    try:

        fit = json.loads(
            fit_path.read_text(
                encoding="utf-8"
            )
        )


        assert fit[
            "protocol_sha256"
        ] == PROTOCOL_SHA

        assert np.isclose(
            float(
                fit[
                    "q"
                ]
            ),
            float(
                q
            ),
            rtol=0.0,
            atol=0.0,
        )

        assert fit[
            "split_seed"
        ] == split_seed

        assert fit[
            "train_seed"
        ] == train_seed

        assert fit[
            "max_epochs"
        ] == MAX_EPOCHS

        assert fit[
            "patience"
        ] == PATIENCE

        assert fit[
            "test_rows_used"
        ] == 0

        assert fit[
            "test_metrics_read"
        ] is False

        assert fit[
            "final_test_result_root_accessed"
        ] is False


        h = pd.read_csv(
            history_path,
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
            ].iloc[-1]
        ) == fit[
            "stop_epoch"
        ]

        assert int(
            h.loc[
                h[
                    "validation_balanced_rmse"
                ].idxmin(),
                "epoch",
            ]
        ) == fit[
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

        assert fit[
            "stopped_by_patience"
        ] is True

        assert fit[
            "hit_epoch_ceiling"
        ] is False


        return fit

    except Exception:
        return None


def train_one(
    q,
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

            val_pred = model(
                val[
                    "x"
                ],
                None,
            )


            vm = bench.weighted_metrics(
                val_pred,
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
                f"Epoch ceiling hit: q={q}, "
                f"split={split_seed}, seed={train_seed}"
            )
        )


    assert (
        stop_epoch
        - best_epoch
        == PATIENCE
    )


    d = fit_dir(
        q,
        split_seed,
        train_seed,
    )

    d.mkdir(
        parents=True,
        exist_ok=True,
    )


    history_path = (
        d
        / "history.csv"
    )


    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False,
        float_format="%.12g",
    )


    fit = {
        "contract":
            "kalman_validation_fit_v1",

        "protocol_sha256":
            PROTOCOL_SHA,

        "q":
            float(
                q
            ),

        "r":
            R,

        "initial_variance":
            INITIAL_VARIANCE,

        "split_seed":
            split_seed,

        "train_seed":
            train_seed,

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
            bool(
                stopped_by_patience
            ),

        "hit_epoch_ceiling":
            hit_ceiling,

        "max_epochs":
            MAX_EPOCHS,

        "patience":
            PATIENCE,

        "hidden_dim":
            HIDDEN_DIM,

        "learning_rate":
            LR,

        "weight_decay":
            WEIGHT_DECAY,

        "parameter_count":
            5569,

        "train_rows":
            data[
                "train"
            ][
                "rows"
            ],

        "validation_rows":
            data[
                "validation"
            ][
                "rows"
            ],

        "train_groups":
            data[
                "train"
            ][
                "groups"
            ],

        "validation_groups":
            data[
                "validation"
            ][
                "groups"
            ],

        "test_rows_used":
            0,

        "test_metrics_read":
            False,

        "final_test_result_root_accessed":
            False,

        "elapsed_seconds":
            float(
                time.time()
                - start
            ),

        "history_sha256":
            sha256_file(
                history_path
            ),
    }


    atomic_json(
        fit,
        d
        / "fit.json",
    )


    return fit


# =============================================================================
# Execute all 60 fits
# =============================================================================

print("=" * 80)
print("KALMAN VALIDATION-ONLY SELECTION")
print("=" * 80)

print(
    "protocol SHA256:",
    PROTOCOL_SHA
)

print(
    "q grid:",
    Q_GRID
)

print(
    "r:",
    R
)

print(
    "planned fits:",
    60
)

print(
    "test rows in model tensors:",
    0
)

print(
    "test metrics read:",
    False
)

print(
    "final test result root accessed:",
    False
)

print()


all_fits = []


for q in Q_GRID:

    q_fits = []


    for split_seed in SPLIT_SEEDS:

        data = prepare_q_split(
            q,
            split_seed,
        )


        for train_seed in TRAIN_SEEDS:

            existing = inspect_fit(
                q,
                split_seed,
                train_seed,
            )


            if existing is not None:

                fit = existing

            else:

                fit = train_one(
                    q,
                    split_seed,
                    train_seed,
                    data,
                )


            q_fits.append(
                fit
            )

            all_fits.append(
                fit
            )


    q_frame = pd.DataFrame(
        q_fits
    )


    split_means = (
        q_frame
        .groupby(
            "split_seed",
            as_index=False,
        )[
            "best_validation_rmse"
        ]
        .mean()
    )


    assert len(
        split_means
    ) == 5


    q_mean = float(
        split_means[
            "best_validation_rmse"
        ].mean()
    )


    print(
        f"q={q:g}: "
        f"completed=15/15 "
        f"split-mean validation RMSE="
        f"{q_mean:.8f} "
        f"max_stop_epoch="
        f"{int(q_frame['stop_epoch'].max())}"
    )


assert len(
    all_fits
) == 60


# =============================================================================
# Aggregate selection
# =============================================================================

fit_summary = pd.DataFrame(
    all_fits
).sort_values(
    [
        "q",
        "split_seed",
        "train_seed",
    ]
).reset_index(
    drop=True
)


fit_summary.to_csv(
    FIT_SUMMARY_PATH,
    index=False,
    float_format="%.12g",
)


split_summary = (
    fit_summary
    .groupby(
        [
            "q",
            "split_seed",
        ],
        as_index=False,
    )
    .agg(
        validation_rmse_mean_over_3_seeds=(
            "best_validation_rmse",
            "mean",
        ),
        validation_rmse_sd_over_3_seeds=(
            "best_validation_rmse",
            "std",
        ),
        max_stop_epoch=(
            "stop_epoch",
            "max",
        ),
        fit_count=(
            "train_seed",
            "count",
        ),
    )
)


assert len(
    split_summary
) == 20

assert set(
    split_summary[
        "fit_count"
    ]
) == {
    3
}


split_summary.to_csv(
    SPLIT_SUMMARY_PATH,
    index=False,
    float_format="%.12g",
)


q_summary = (
    split_summary
    .groupby(
        "q",
        as_index=False,
    )
    .agg(
        validation_rmse_mean_over_5_splits=(
            "validation_rmse_mean_over_3_seeds",
            "mean",
        ),
        validation_rmse_sd_over_5_splits=(
            "validation_rmse_mean_over_3_seeds",
            "std",
        ),
        max_stop_epoch=(
            "max_stop_epoch",
            "max",
        ),
        split_count=(
            "split_seed",
            "count",
        ),
    )
    .sort_values(
        "q"
    )
    .reset_index(
        drop=True
    )
)


assert len(
    q_summary
) == 4

assert set(
    q_summary[
        "split_count"
    ]
) == {
    5
}


minimum = float(
    q_summary[
        "validation_rmse_mean_over_5_splits"
    ].min()
)


tied = q_summary[
    q_summary[
        "validation_rmse_mean_over_5_splits"
    ]
    <= (
        minimum
        + TIE_TOLERANCE
    )
].copy()


selected_q = float(
    tied[
        "q"
    ].min()
)


selected_score = float(
    q_summary.loc[
        q_summary[
            "q"
        ] == selected_q,
        "validation_rmse_mean_over_5_splits",
    ].iloc[0]
)


q_summary[
    "selected"
] = (
    q_summary[
        "q"
    ] == selected_q
)


q_summary.to_csv(
    Q_SUMMARY_PATH,
    index=False,
    float_format="%.12g",
)


selected = {
    "contract":
        "selected_kalman_v1",

    "protocol_sha256":
        PROTOCOL_SHA,

    "selected_q":
        selected_q,

    "selected_r":
        R,

    "selected_initial_variance":
        INITIAL_VARIANCE,

    "selection_score":
        selected_score,

    "selection_score_definition":
        (
            "mean of five split-level means, with each split-level "
            "mean computed over three initialization seeds"
        ),

    "tie_tolerance":
        TIE_TOLERANCE,

    "tie_rule":
        "smallest numeric q among tied candidates",

    "candidate_count":
        4,

    "fit_count":
        60,

    "split_count":
        5,

    "initializations_per_split":
        3,

    "test_rows_used":
        0,

    "test_metrics_read":
        False,

    "final_test_result_root_accessed":
        False,
}


atomic_json(
    selected,
    SELECTED_PATH,
)


manifest = {
    "contract":
        "kalman_validation_selection_manifest_v1",

    "protocol_sha256":
        PROTOCOL_SHA,

    "completed_fit_count":
        60,

    "all_fits_patience_valid":
        bool(
            (
                fit_summary[
                    "epochs_after_best"
                ]
                == PATIENCE
            ).all()
        ),

    "all_fits_below_epoch_ceiling":
        bool(
            (
                fit_summary[
                    "stop_epoch"
                ]
                < MAX_EPOCHS
            ).all()
        ),

    "test_rows_used":
        0,

    "test_metrics_read":
        False,

    "final_test_result_root_accessed":
        False,

    "selected_q":
        selected_q,

    "selected_r":
        R,

    "output_sha256": {
        "fit_summary":
            sha256_file(
                FIT_SUMMARY_PATH
            ),

        "split_summary":
            sha256_file(
                SPLIT_SUMMARY_PATH
            ),

        "q_summary":
            sha256_file(
                Q_SUMMARY_PATH
            ),

        "selected":
            sha256_file(
                SELECTED_PATH
            ),
    },
}


atomic_json(
    manifest,
    MANIFEST_PATH,
)


print()
print("=" * 80)
print("VALIDATION-ONLY SELECTION RESULT")
print("=" * 80)


for _, row in q_summary.iterrows():

    print(
        f"q={row['q']:g}: "
        f"mean={row['validation_rmse_mean_over_5_splits']:.8f} "
        f"sd={row['validation_rmse_sd_over_5_splits']:.8f} "
        f"selected={bool(row['selected'])}"
    )


print()
print(
    "selected q:",
    selected_q
)

print(
    "selected r:",
    R
)

print(
    "completed fits:",
    len(
        fit_summary
    )
)

print(
    "all fits patience-valid:",
    manifest[
        "all_fits_patience_valid"
    ]
)

print(
    "all fits below 5000 ceiling:",
    manifest[
        "all_fits_below_epoch_ceiling"
    ]
)

print(
    "test rows used:",
    0
)

print(
    "test metrics read:",
    False
)

print(
    "final test result root accessed:",
    False
)

print(
    "selected artifact SHA256:",
    sha256_file(
        SELECTED_PATH
    )
)

print(
    "selection manifest SHA256:",
    sha256_file(
        MANIFEST_PATH
    )
)

print()
print(
    "KALMAN_VALIDATION_SELECTION_PASS=True"
)
