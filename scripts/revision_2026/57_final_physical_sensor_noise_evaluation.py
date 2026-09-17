from pathlib import Path
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import subprocess
import time

import numpy as np
import pandas as pd
import torch


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo" / "battery-gnn"

THIS = (
    REPO
    / "scripts"
    / "revision_2026"
    / "57_final_physical_sensor_noise_evaluation.py"
)

BENCH = (
    REPO
    / "scripts"
    / "revision_2026"
    / "24_feature_matched_soh_benchmark.py"
)

CANON = (
    BASE
    / "data"
    / "processed"
    / "revision_2026"
    / "canonical_tasks"
    / "canonical_cycle_supervised_table.csv"
)

ASSIGN = (
    BASE
    / "data"
    / "processed"
    / "revision_2026"
    / "splits_and_scalers"
    / "cell_split_assignments.csv"
)

SCALERS = (
    BASE
    / "data"
    / "processed"
    / "revision_2026"
    / "splits_and_scalers"
    / "train_only_scaler_parameters.csv"
)

FINAL_V3 = (
    BASE
    / "results"
    / "revision_2026"
    / "feature_matched_soh"
    / "final_v3"
)

FINAL_V3_MANIFEST = (
    FINAL_V3
    / "final_v3_execution_manifest.json"
)

ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "physical_sensor_noise"
    / "final_physical_v1"
)

PROTOCOL = ROOT / "protocol.json"

CORRUPTED = ROOT / "corrupted_features"

CONDITIONS = (
    CORRUPTED
    / "conditions.json"
)

FEATURE_CLOSURE = (
    CORRUPTED
    / "corruption_feature_closure.json"
)

CELL_DIR = (
    CORRUPTED
    / "cells"
)

EVAL_ROOT = (
    ROOT
    / "evaluation"
)

RUN_DIR = (
    EVAL_ROOT
    / "runs"
)

EXEC_MANIFEST = (
    EVAL_ROOT
    / "physical_noise_execution_manifest.json"
)


EXPECTED = {
    "protocol":
        "7ae0836fe69c4b14a1f93a094b2d42bb96209e302c8262f7b8bd8625baf78bfc",

    "feature_closure":
        "382ea97c8f1d8f219436dd5071bb8fcbd63811f24e76b8c887e8efba4e54675f",

    "canonical":
        "49a7b7629cf9ee3f8ce0a7e3c52f2807878b752b4c9a30a876285fb891b00c14",

    "assignments":
        "914c8b90131f496ca396f63e0502147ca99848fdfd4b2c6b2a5887225b47d055",

    "scalers":
        "b9708b9321b521b87bdd05f969558134d7e0148285ef7c50f648c2028cd86de1",

    "final_v3_manifest":
        "5a8bfbe0f11a6158f2749a9da4f18f1e62adf8c839976c1bf0af9c9f6c07440d",
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

DOMAINS = [
    "BALANCED_ALL",
    "HUST",
    "CALCE",
    "MATR",
]

SPLITS = [0, 1, 2, 3, 4]
TRAIN_SEEDS = [0, 1, 2]

EXPECTED_TEST_ROWS = {
    0: 43974,
    1: 46273,
    2: 43818,
    3: 46156,
    4: 43749,
}


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
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
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(tmp, path)


def atomic_csv(df, path):
    tmp = Path(str(path) + ".tmp")

    df.to_csv(
        tmp,
        index=False,
        float_format="%.17g",
    )

    os.replace(tmp, path)


def as_bool(s):
    return (
        s.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            ["true", "1", "1.0", "yes"]
        )
    )


def safe_name(dataset, cell):
    return (
        str(dataset)
        + "__"
        + re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            str(cell),
        )
    )


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    mod = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(mod)

    return mod


def metric_triplet(
    pred,
    target,
    weight,
):
    pred = np.asarray(
        pred,
        dtype=np.float64,
    ).reshape(-1)

    target = np.asarray(
        target,
        dtype=np.float64,
    ).reshape(-1)

    weight = np.asarray(
        weight,
        dtype=np.float64,
    ).reshape(-1)

    assert (
        len(pred)
        == len(target)
        == len(weight)
    )

    assert len(pred) > 0

    assert np.isfinite(pred).all()
    assert np.isfinite(target).all()
    assert np.isfinite(weight).all()

    assert (weight > 0).all()

    weight = weight / weight.sum()

    err = pred - target

    mse = float(
        np.sum(
            weight
            * err
            * err
        )
    )

    rmse = math.sqrt(mse)

    mae = float(
        np.sum(
            weight
            * np.abs(err)
        )
    )

    mu = float(
        np.sum(
            weight
            * target
        )
    )

    sst = float(
        np.sum(
            weight
            * (
                target
                - mu
            ) ** 2
        )
    )

    assert sst > 0.0

    r2 = (
        1.0
        - mse
        / sst
    )

    assert math.isfinite(rmse)
    assert math.isfinite(mae)
    assert math.isfinite(r2)

    return rmse, mae, r2


def make_weights(df):
    n = len(df)

    w = np.zeros(
        n,
        dtype=np.float64,
    )

    datasets = sorted(
        df[
            "dataset"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    assert datasets == sorted(DATASETS)

    for dataset in DATASETS:

        dmask = (
            df[
                "dataset"
            ].astype(str).to_numpy()
            == dataset
        )

        groups = (
            df.loc[
                dmask,
                "split_group",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        assert len(groups) > 0

        dataset_mass = 1.0 / len(DATASETS)

        cell_mass = (
            dataset_mass
            / len(groups)
        )

        for group in groups:

            gmask = (
                dmask
                & (
                    df[
                        "split_group"
                    ].astype(str).to_numpy()
                    == group
                )
            )

            ng = int(
                gmask.sum()
            )

            assert ng > 0

            w[
                gmask
            ] = (
                cell_mass
                / ng
            )

    assert np.all(
        w > 0
    )

    assert abs(
        float(
            w.sum()
        )
        - 1.0
    ) < 1e-12

    for dataset in DATASETS:

        mask = (
            df[
                "dataset"
            ].astype(str).to_numpy()
            == dataset
        )

        assert abs(
            float(
                w[
                    mask
                ].sum()
            )
            - 1.0 / 3.0
        ) < 1e-12

    return w


def make_prev_index(df):
    n = len(df)

    prev = np.arange(
        n,
        dtype=np.int64,
    )

    for _, idx in df.groupby(
        "split_group",
        sort=False,
    ).groups.items():

        idx = np.asarray(
            list(idx),
            dtype=np.int64,
        )

        assert len(idx) > 0

        if len(idx) > 1:

            prev[
                idx[
                    1:
                ]
            ] = idx[
                :-1
            ]

    assert np.all(
        prev
        <= np.arange(
            n,
            dtype=np.int64,
        )
    )

    return prev


def scaler_for_seed(
    scaler_df,
    seed,
):
    s = scaler_df[
        (
            scaler_df[
                "split_seed"
            ] == seed
        )
        & (
            scaler_df[
                "scaler_scope"
            ].astype(str)
            == "pooled_supervised"
        )
    ].copy()

    assert len(s) == 20

    if "fit_split" in s.columns:
        assert set(
            s[
                "fit_split"
            ].astype(str)
        ) == {"train_only"}

    assert set(
        s[
            "feature"
        ].astype(str)
    ) == set(FEATURES)

    s = (
        s.set_index(
            "feature"
        )
        .loc[
            FEATURES
        ]
    )

    mean = s[
        "mean"
    ].to_numpy(
        dtype=np.float64
    )

    scale = s[
        "scale"
    ].to_numpy(
        dtype=np.float64
    )

    assert np.isfinite(mean).all()
    assert np.isfinite(scale).all()
    assert (scale > 0).all()

    return mean, scale


def state_path(
    split_seed,
    train_seed,
    model_name,
):
    d = (
        FINAL_V3
        / (
            f"split{split_seed}"
            f"_train{train_seed}"
        )
    )

    if model_name == "MLP":
        p = d / "mlp_best_state.pt"

    elif model_name == "GCN":
        p = d / "causal_gcn_best_state.pt"

    else:
        raise AssertionError(
            model_name
        )

    assert p.is_file()

    return p


def output_paths(
    split_seed,
    train_seed,
    model_name,
):
    stem = (
        f"split{split_seed}"
        f"_train{train_seed}"
        f"_{model_name.lower()}"
    )

    return (
        RUN_DIR
        / f"{stem}_physical_test_metrics.csv",

        RUN_DIR
        / f"{stem}_evaluation.json",
    )


def valid_existing_run(
    metric_path,
    meta_path,
    *,
    runner_sha,
    state_sha,
    conditions_sha,
):
    if not (
        metric_path.is_file()
        and meta_path.is_file()
    ):
        return False

    try:
        m = json.loads(
            meta_path.read_text(
                encoding="utf-8"
            )
        )

        return (
            m.get("contract")
            == "physical_noise_test_evaluation_run_v1"

            and m.get(
                "protocol_sha256"
            )
            == EXPECTED["protocol"]

            and m.get(
                "feature_closure_sha256"
            )
            == EXPECTED["feature_closure"]

            and m.get(
                "runner_sha256"
            )
            == runner_sha

            and m.get(
                "state_sha256"
            )
            == state_sha

            and m.get(
                "conditions_file_sha256"
            )
            == conditions_sha

            and m.get(
                "metric_row_count"
            )
            == 264

            and m.get(
                "all_metrics_finite"
            )
            is True

            and m.get(
                "metric_values_surfaced"
            )
            is False

            and sha(
                metric_path
            )
            == m.get(
                "metric_file_sha256"
            )
        )

    except Exception:
        return False


# ============================================================================
# Guards
# ============================================================================

for path, expected in [
    (PROTOCOL, EXPECTED["protocol"]),
    (
        FEATURE_CLOSURE,
        EXPECTED["feature_closure"],
    ),
    (CANON, EXPECTED["canonical"]),
    (ASSIGN, EXPECTED["assignments"]),
    (SCALERS, EXPECTED["scalers"]),
    (
        FINAL_V3_MANIFEST,
        EXPECTED["final_v3_manifest"],
    ),
]:
    assert sha(path) == expected, (
        path,
        sha(path),
        expected,
    )


protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)

closure = json.loads(
    FEATURE_CLOSURE.read_text(
        encoding="utf-8"
    )
)

conditions_doc = json.loads(
    CONDITIONS.read_text(
        encoding="utf-8"
    )
)

conditions = conditions_doc[
    "conditions"
]


assert closure[
    "status"
] == "COMPLETE_BEFORE_ANY_MODEL_EVALUATION"

assert closure[
    "cell_count"
] == 143

assert closure[
    "row_count"
] == 166304

assert closure[
    "condition_count"
] == 66

assert closure[
    "model_loaded"
] is False

assert closure[
    "model_evaluated"
] is False

assert closure[
    "metric_computed"
] is False


assert len(
    conditions
) == 66

assert [
    int(
        x[
            "condition_index"
        ]
    )
    for x in conditions
] == list(
    range(66)
)


assert conditions[
    0
][
    "condition_id"
] == "clean"


assert protocol[
    "training_policy"
][
    "no_retraining"
] is True

assert protocol[
    "training_policy"
][
    "train_only_scaler_reused_without_refit"
] is True


bench = load_module(
    BENCH,
    "feature_matched_benchmark24_physical_eval",
)


assert sha(
    BENCH
) == protocol[
    "source_sha256"
][
    "benchmark_model"
]


tracked = subprocess.run(
    [
        "git",
        "ls-files",
        "--error-unmatch",
        str(
            THIS.relative_to(
                REPO
            )
        ),
    ],
    cwd=REPO,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

assert tracked.returncode == 0


RUNNER_SHA = sha(
    THIS
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


# ============================================================================
# Data
# ============================================================================

canon = pd.read_csv(
    CANON,
    usecols=[
        "dataset",
        "split_group",
        "observation_index",
        "SOH",
        "soh_task_eligible",
        "feature_source_cell_id",
        "feature_raw_cycle_position_zero_based",
    ]
    + FEATURES,
    low_memory=False,
)


canon = canon[
    as_bool(
        canon[
            "soh_task_eligible"
        ]
    )
].copy()


assert len(
    canon
) == 300553


canon[
    "dataset"
] = canon[
    "dataset"
].astype(str)

canon[
    "split_group"
] = canon[
    "split_group"
].astype(str)

canon[
    "feature_source_cell_id"
] = canon[
    "feature_source_cell_id"
].astype(str)

canon[
    "observation_index"
] = pd.to_numeric(
    canon[
        "observation_index"
    ],
    errors="raise",
).astype(int)

canon[
    "feature_raw_cycle_position_zero_based"
] = pd.to_numeric(
    canon[
        "feature_raw_cycle_position_zero_based"
    ],
    errors="raise",
).astype(int)


assign = pd.read_csv(
    ASSIGN,
    low_memory=False,
)

assign[
    "split_group"
] = assign[
    "split_group"
].astype(str)


scaler_df = pd.read_csv(
    SCALERS,
    low_memory=False,
)


EVAL_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

RUN_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


torch.set_num_threads(4)
torch.set_num_interop_threads(1)


# ============================================================================
# Evaluation
# ============================================================================

start = time.time()

completed_runs = 0
reused_runs = 0
generated_runs = 0

records = []


print("=" * 80)
print("FINAL PHYSICAL SENSOR NOISE TEST EVALUATION")
print("=" * 80)

print(
    "protocol locked:",
    True
)

print(
    "feature closure locked:",
    True
)

print(
    "test conditions:",
    66
)

print(
    "split seeds:",
    5
)

print(
    "train init seeds per split:",
    3
)

print(
    "models:",
    2
)

print(
    "planned evaluation runs:",
    30
)

print(
    "training code executed:",
    False
)

print(
    "GPU used:",
    False
)

print(
    "metric values surfaced:",
    False
)

print()


for split_seed in SPLITS:

    test_groups = set(
        assign.loc[
            (
                assign[
                    "split_seed"
                ] == split_seed
            )
            & (
                assign[
                    "split"
                ].astype(str)
                == "test"
            ),
            "split_group",
        ].astype(str)
    )


    test = canon[
        canon[
            "split_group"
        ].isin(
            test_groups
        )
    ].copy()


    test = (
        test.sort_values(
            [
                "dataset",
                "split_group",
                "observation_index",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )


    assert len(
        test
    ) == EXPECTED_TEST_ROWS[
        split_seed
    ]


    y = test[
        "SOH"
    ].to_numpy(
        dtype=np.float64
    )

    assert np.isfinite(y).all()


    weight = make_weights(
        test
    )


    prev_idx = make_prev_index(
        test
    )


    mean, scale = scaler_for_seed(
        scaler_df,
        split_seed,
    )


    # --------------------------------------------------------------
    # Check which of six model-runs need evaluation.
    # --------------------------------------------------------------

    model_entries = []


    for train_seed in TRAIN_SEEDS:

        for model_name in [
            "MLP",
            "GCN",
        ]:

            sp = state_path(
                split_seed,
                train_seed,
                model_name,
            )

            state_sha = sha(
                sp
            )

            metric_path, meta_path = (
                output_paths(
                    split_seed,
                    train_seed,
                    model_name,
                )
            )


            reusable = valid_existing_run(
                metric_path,
                meta_path,
                runner_sha=RUNNER_SHA,
                state_sha=state_sha,
                conditions_sha=sha(
                    CONDITIONS
                ),
            )


            if reusable:

                reused_runs += 1
                completed_runs += 1

                meta = json.loads(
                    meta_path.read_text(
                        encoding="utf-8"
                    )
                )

                records.append(
                    {
                        "split_seed":
                            split_seed,

                        "train_seed":
                            train_seed,

                        "model":
                            model_name,

                        "state_sha256":
                            state_sha,

                        "metric_file":
                            str(
                                metric_path
                            ),

                        "metric_file_sha256":
                            meta[
                                "metric_file_sha256"
                            ],

                        "evaluation_meta":
                            str(
                                meta_path
                            ),

                        "evaluation_meta_sha256":
                            sha(
                                meta_path
                            ),

                        "reused":
                            True,
                    }
                )

                continue


            model_entries.append(
                (
                    train_seed,
                    model_name,
                    sp,
                    state_sha,
                    metric_path,
                    meta_path,
                )
            )


    if not model_entries:
        print(
            f"split {split_seed}: "
            "all 6 runs reused",
            flush=True,
        )
        continue


    # --------------------------------------------------------------
    # Load corrupted-feature shards for this split once.
    # --------------------------------------------------------------

    grouped = list(
        test.groupby(
            [
                "dataset",
                "feature_source_cell_id",
            ],
            sort=False,
        )
    )


    cell_cache = []


    for (
        dataset,
        raw_cell
    ), g in grouped:

        npz_path = (
            CELL_DIR
            / (
                safe_name(
                    dataset,
                    raw_cell,
                )
                + ".npz"
            )
        )

        assert npz_path.is_file()


        with np.load(
            npz_path,
            allow_pickle=False,
        ) as z:

            features = z[
                "features"
            ].copy()

            positions = z[
                "positions"
            ].astype(
                np.int64
            )

            condition_ids = (
                z[
                    "condition_ids"
                ]
                .astype(str)
                .tolist()
            )


        assert condition_ids == [
            str(
                x[
                    "condition_id"
                ]
            )
            for x in conditions
        ]


        expected_positions = g[
            "feature_raw_cycle_position_zero_based"
        ].to_numpy(
            dtype=np.int64
        )


        loc = np.searchsorted(
            positions,
            expected_positions,
        )


        assert np.all(
            loc
            < len(
                positions
            )
        )

        assert np.array_equal(
            positions[
                loc
            ],
            expected_positions,
        )


        rows = g.index.to_numpy(
            dtype=np.int64
        )


        assert len(
            rows
        ) == len(
            loc
        )


        cell_cache.append(
            (
                rows,
                loc,
                features,
            )
        )


    # --------------------------------------------------------------
    # Before first prediction, clean condition must reproduce canonical
    # features at the frozen roundtrip tolerance.
    # --------------------------------------------------------------

    clean_x = np.empty(
        (
            len(test),
            20,
        ),
        dtype=np.float64,
    )


    for rows, loc, features in cell_cache:

        clean_x[
            rows,
            :,
        ] = features[
            0,
            loc,
            :,
        ]


    canonical_x = test[
        FEATURES
    ].to_numpy(
        dtype=np.float64
    )


    assert np.allclose(
        clean_x,
        canonical_x,
        atol=1e-8,
        rtol=1e-8,
    )


    # --------------------------------------------------------------
    # Load frozen states only after clean reconstruction barrier.
    # --------------------------------------------------------------

    models = {}


    for (
        train_seed,
        model_name,
        sp,
        state_sha,
        metric_path,
        meta_path,
    ) in model_entries:

        if model_name == "MLP":

            model = (
                bench.FeatureMatchedMLP(
                    input_dim=20,
                    hidden_dim=64,
                )
            )

        else:

            model = (
                bench.FeatureMatchedCausalGCN(
                    input_dim=20,
                    hidden_dim=64,
                )
            )


        state = torch.load(
            sp,
            map_location="cpu",
            weights_only=True,
        )


        model.load_state_dict(
            state,
            strict=True,
        )


        assert bench.parameter_count(
            model
        ) == 5569


        sig_n = len(
            inspect.signature(
                model.forward
            ).parameters
        )


        if model_name == "MLP":
            sig = inspect.signature(
                model.forward
            )

            assert sig_n == 2

            params = list(
                sig.parameters.values()
            )

            assert params[0].name == "x"
            assert params[1].name == "prev_index"
            assert params[1].default is None

        else:
            sig = inspect.signature(
                model.forward
            )

            assert sig_n == 2

            params = list(
                sig.parameters.values()
            )

            assert params[0].name == "x"
            assert params[1].name == "prev_index"
            assert (
                params[1].default
                is inspect.Parameter.empty
            )


        model.eval()


        models[
            (
                train_seed,
                model_name,
            )
        ] = (
            model,
            sp,
            state_sha,
            metric_path,
            meta_path,
        )


    result_rows = {
        key: []
        for key in models
    }


    # --------------------------------------------------------------
    # Evaluate 66 frozen conditions.
    # --------------------------------------------------------------

    for condition in conditions:

        ci = int(
            condition[
                "condition_index"
            ]
        )


        x_raw = np.empty(
            (
                len(test),
                20,
            ),
            dtype=np.float64,
        )


        for rows, loc, features in cell_cache:

            x_raw[
                rows,
                :,
            ] = features[
                ci,
                loc,
                :,
            ]


        assert np.isfinite(
            x_raw
        ).all()


        x_std = (
            x_raw
            - mean[
                None,
                :
            ]
        ) / scale[
            None,
            :
        ]


        assert np.isfinite(
            x_std
        ).all()


        x_t = torch.from_numpy(
            x_std.astype(
                np.float32,
                copy=False,
            )
        )

        prev_t = torch.from_numpy(
            prev_idx
        )


        with torch.inference_mode():

            predictions = {}


            for key, (
                model,
                _sp,
                _state_sha,
                _metric_path,
                _meta_path,
            ) in models.items():

                if key[
                    1
                ] == "MLP":

                    pred = model(
                        x_t
                    )

                else:

                    pred = model(
                        x_t,
                        prev_t,
                    )


                pred = (
                    pred
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .astype(
                        np.float64,
                        copy=False,
                    )
                )


                assert len(
                    pred
                ) == len(
                    test
                )

                assert np.isfinite(
                    pred
                ).all()


                predictions[
                    key
                ] = pred


        for key, pred in predictions.items():

            for domain in DOMAINS:

                if domain == "BALANCED_ALL":

                    mask = np.ones(
                        len(
                            test
                        ),
                        dtype=bool,
                    )

                else:

                    mask = (
                        test[
                            "dataset"
                        ].to_numpy()
                        == domain
                    )


                assert mask.any()


                rmse, mae, r2 = (
                    metric_triplet(
                        pred[
                            mask
                        ],
                        y[
                            mask
                        ],
                        weight[
                            mask
                        ],
                    )
                )


                result_rows[
                    key
                ].append(
                    {
                        "split_seed":
                            split_seed,

                        "train_seed":
                            key[
                                0
                            ],

                        "model":
                            key[
                                1
                            ],

                        "condition_index":
                            ci,

                        "condition_id":
                            condition[
                                "condition_id"
                            ],

                        "scenario":
                            condition[
                                "scenario"
                            ],

                        "level":
                            condition[
                                "level"
                            ],

                        "corruption_seed":
                            int(
                                condition[
                                    "corruption_seed"
                                ]
                            ),

                        "severity_axis":
                            float(
                                condition[
                                    "severity_axis"
                                ]
                            ),

                        "domain":
                            domain,

                        "n_rows":
                            int(
                                mask.sum()
                            ),

                        "rmse":
                            rmse,

                        "mae":
                            mae,

                        "r2":
                            r2,
                    }
                )


    # --------------------------------------------------------------
    # Persist each of six paired runs independently.
    # --------------------------------------------------------------

    for key, rows in result_rows.items():

        train_seed, model_name = key

        model, sp, state_sha, metric_path, meta_path = (
            models[
                key
            ]
        )


        assert len(
            rows
        ) == (
            66
            * 4
        )


        out = pd.DataFrame(
            rows
        )


        assert out[
            [
                "rmse",
                "mae",
                "r2",
            ]
        ].apply(
            pd.to_numeric,
            errors="raise",
        ).apply(
            np.isfinite
        ).all().all()


        atomic_csv(
            out,
            metric_path,
        )


        metric_hash = sha(
            metric_path
        )


        meta = {
            "contract":
                "physical_noise_test_evaluation_run_v1",

            "protocol_sha256":
                EXPECTED[
                    "protocol"
                ],

            "feature_closure_sha256":
                EXPECTED[
                    "feature_closure"
                ],

            "runner_sha256":
                RUNNER_SHA,

            "runner_commit":
                RUNNER_COMMIT,

            "conditions_file_sha256":
                sha(
                    CONDITIONS
                ),

            "split_seed":
                int(
                    split_seed
                ),

            "train_seed":
                int(
                    train_seed
                ),

            "model":
                model_name,

            "state_file":
                str(
                    sp
                ),

            "state_sha256":
                state_sha,

            "parameter_count":
                5569,

            "test_row_count":
                int(
                    len(
                        test
                    )
                ),

            "condition_count":
                66,

            "domain_count":
                4,

            "metric_row_count":
                264,

            "all_metrics_finite":
                True,

            "training_reexecuted":
                False,

            "test_only_evaluation":
                True,

            "train_only_scaler_reused":
                True,

            "scaler_refit":
                False,

            "clean_raw_feature_barrier_passed":
                True,

            "metric_values_surfaced":
                False,

            "metric_file":
                str(
                    metric_path
                ),

            "metric_file_sha256":
                metric_hash,
        }


        atomic_json(
            meta,
            meta_path,
        )


        records.append(
            {
                "split_seed":
                    int(
                        split_seed
                    ),

                "train_seed":
                    int(
                        train_seed
                    ),

                "model":
                    model_name,

                "state_sha256":
                    state_sha,

                "metric_file":
                    str(
                        metric_path
                    ),

                "metric_file_sha256":
                    metric_hash,

                "evaluation_meta":
                    str(
                        meta_path
                    ),

                "evaluation_meta_sha256":
                    sha(
                        meta_path
                    ),

                "reused":
                    False,
            }
        )


        generated_runs += 1
        completed_runs += 1


    print(
        f"split {split_seed}: "
        f"complete "
        f"total_runs={completed_runs}/30 "
        f"generated={generated_runs} "
        f"reused={reused_runs}",
        flush=True,
    )


assert completed_runs == 30


records = sorted(
    records,
    key=lambda x: (
        x[
            "split_seed"
        ],
        x[
            "train_seed"
        ],
        x[
            "model"
        ],
    ),
)


assert len(
    records
) == 30


assert len({
    (
        x[
            "split_seed"
        ],
        x[
            "train_seed"
        ],
        x[
            "model"
        ],
    )
    for x in records
}) == 30


manifest = {
    "contract":
        "final_physical_sensor_noise_execution_v1",

    "status":
        "COMPLETE_BEFORE_VALUE_LEVEL_ANALYSIS",

    "protocol_sha256":
        EXPECTED[
            "protocol"
        ],

    "feature_closure_sha256":
        EXPECTED[
            "feature_closure"
        ],

    "canonical_sha256":
        EXPECTED[
            "canonical"
        ],

    "assignments_sha256":
        EXPECTED[
            "assignments"
        ],

    "scaler_sha256":
        EXPECTED[
            "scalers"
        ],

    "final_v3_execution_manifest_sha256":
        EXPECTED[
            "final_v3_manifest"
        ],

    "runner_sha256":
        RUNNER_SHA,

    "runner_commit":
        RUNNER_COMMIT,

    "device":
        "cpu",

    "split_count":
        5,

    "train_initializations_per_split":
        3,

    "model_count":
        2,

    "planned_runs":
        30,

    "completed_runs":
        30,

    "conditions_per_run":
        66,

    "domains_per_condition":
        4,

    "metric_rows_per_run":
        264,

    "expected_total_metric_rows":
        7920,

    "training_reexecuted":
        False,

    "state_files_reused":
        True,

    "train_only_scalers_reused":
        True,

    "scaler_refit":
        False,

    "feature_generation_reexecuted":
        False,

    "clean_raw_feature_barrier_checked":
        True,

    "metric_values_surfaced_during_execution":
        False,

    "metric_values_used_to_change_protocol":
        False,

    "analysis_performed":
        False,

    "generated_runs_this_invocation":
        int(
            generated_runs
        ),

    "reused_runs_this_invocation":
        int(
            reused_runs
        ),

    "records":
        records,

    "elapsed_seconds":
        float(
            time.time()
            - start
        ),
}


atomic_json(
    manifest,
    EXEC_MANIFEST,
)


print()
print("=" * 80)
print("PHYSICAL SENSOR NOISE EXECUTION CLOSURE")
print("=" * 80)

print(
    "completed runs:",
    manifest[
        "completed_runs"
    ]
)

print(
    "conditions per run:",
    manifest[
        "conditions_per_run"
    ]
)

print(
    "domains per condition:",
    manifest[
        "domains_per_condition"
    ]
)

print(
    "expected metric rows:",
    manifest[
        "expected_total_metric_rows"
    ]
)

print(
    "training reexecuted:",
    manifest[
        "training_reexecuted"
    ]
)

print(
    "scaler refit:",
    manifest[
        "scaler_refit"
    ]
)

print(
    "feature generation reexecuted:",
    manifest[
        "feature_generation_reexecuted"
    ]
)

print(
    "metric values surfaced:",
    manifest[
        "metric_values_surfaced_during_execution"
    ]
)

print(
    "analysis performed:",
    manifest[
        "analysis_performed"
    ]
)

print(
    "execution manifest SHA256:",
    sha(
        EXEC_MANIFEST
    )
)

print()
print(
    "FINAL_PHYSICAL_SENSOR_NOISE_EXECUTION_PASS=True"
)
