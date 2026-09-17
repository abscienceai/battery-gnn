from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os
import subprocess

import numpy as np
import pandas as pd
import torch


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

BENCH = (
    REPO
    / "scripts/revision_2026/24_feature_matched_soh_benchmark.py"
)

THIS = (
    REPO
    / "scripts/revision_2026/69_sinusoidal_frequency_sweep.py"
)

ROOT = (
    BASE
    / "results/revision_2026/spectral_frequency/final_spectral_v1"
)

PROTOCOL = ROOT / "protocol.json"

CLEAN_MANIFEST = (
    ROOT
    / "clean_graph_fourier_v1"
    / "clean_graph_fourier_manifest.json"
)

FINAL_V3 = (
    BASE
    / "results/revision_2026/feature_matched_soh/final_v3"
)

EVAL = ROOT / "sinusoidal_frequency_v1"
RUNS = EVAL / "runs"

EXEC_MANIFEST = (
    EVAL
    / "sinusoidal_frequency_execution_manifest.json"
)


EXPECTED_PROTOCOL_SHA = (
    "a8d3b65b87980470f326717f0cb50e3d4fb4d2e0429c93171eb0d21d6c713e45"
)

EXPECTED_CLEAN_MANIFEST_SHA = (
    "da9653f823f764ab85d69c2726f2de752a0cb7091b7c13d79a7db0e915751d12"
)

EXPECTED_FINAL_V3_EXEC_SHA = (
    "5a8bfbe0f11a6158f2749a9da4f18f1e62adf8c839976c1bf0af9c9f6c07440d"
)

CLEAN_TOL = 1e-7
RMS_TOL = 2e-6

MODELS = ["MLP", "GCN"]
DOMAINS = [
    "BALANCED_ALL",
    "HUST",
    "CALCE",
    "MATR",
]


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def sha_bytes(data):
    return hashlib.sha256(
        data
    ).hexdigest()


def atomic_csv(df, path):
    tmp = Path(str(path) + ".tmp")

    df.to_csv(
        tmp,
        index=False,
        float_format="%.17g",
    )

    os.replace(
        tmp,
        path,
    )


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

    os.replace(
        tmp,
        path,
    )


def stable_phase(
    dataset,
    group,
    feature,
    perturbation_seed,
):
    token = (
        f"{dataset}|{group}|{feature}|"
        f"{perturbation_seed}"
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        token
    ).digest()

    integer = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    u = integer / float(
        2 ** 64
    )

    return (
        2.0
        * math.pi
        * u
    )


def norm_model_name(x):
    s = str(x).strip().upper()

    if "GCN" in s:
        return "GCN"

    if "MLP" in s:
        return "MLP"

    raise AssertionError(
        ("unknown model", x)
    )


def load_state(path):
    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if isinstance(state, dict):

        if "model_state_dict" in state:
            state = state[
                "model_state_dict"
            ]

        elif "state_dict" in state:
            state = state[
                "state_dict"
            ]

    assert isinstance(
        state,
        dict,
    )

    return state


def csv_data_rows(path):
    with Path(path).open(
        "rb"
    ) as f:

        lines = sum(
            1
            for _ in f
        )

    assert lines >= 1

    return lines - 1


def make_sinusoidal_input(
    base_x,
    frame,
    features,
    omega_over_pi,
    perturbation_seed,
    target_rms,
):
    base = np.asarray(
        base_x,
        dtype=np.float64,
    )

    assert base.ndim == 2
    assert base.shape[1] == len(
        features
    )

    out = base.copy()

    dataset_arr = (
        frame[
            "dataset"
        ]
        .astype(str)
        .to_numpy()
    )

    group_arr = (
        frame[
            "split_group"
        ]
        .astype(str)
        .to_numpy()
    )

    groups = (
        frame[
            [
                "dataset",
                "split_group",
            ]
        ]
        .drop_duplicates()
    )

    for row in groups.itertuples(
        index=False
    ):

        dataset = str(
            row.dataset
        )

        group = str(
            row.split_group
        )

        idx = np.flatnonzero(
            group_arr == group
        )

        assert len(idx) > 0

        assert np.all(
            dataset_arr[
                idx
            ] == dataset
        )

        t = np.arange(
            len(idx),
            dtype=np.float64,
        )

        omega = (
            math.pi
            * float(
                omega_over_pi
            )
        )

        for j, feature in enumerate(
            features
        ):

            phase = stable_phase(
                dataset,
                group,
                feature,
                perturbation_seed,
            )

            wave = np.cos(
                omega * t
                + phase
            )

            rms = float(
                np.sqrt(
                    np.mean(
                        wave ** 2
                    )
                )
            )

            assert rms > 1e-10

            noise = (
                float(
                    target_rms
                )
                * wave
                / rms
            )

            out[
                idx,
                j,
            ] += noise


    # The actual tensor entering the model is float32.
    out32 = out.astype(
        np.float32
    )

    actual_delta = (
        out32.astype(
            np.float64
        )
        - base.astype(
            np.float32
        ).astype(
            np.float64
        )
    )


    max_abs_rms_error = 0.0

    for row in groups.itertuples(
        index=False
    ):

        group = str(
            row.split_group
        )

        idx = np.flatnonzero(
            group_arr == group
        )

        d = actual_delta[
            idx,
            :
        ]

        rms = np.sqrt(
            np.mean(
                d ** 2,
                axis=0,
            )
        )

        err = np.max(
            np.abs(
                rms
                - float(
                    target_rms
                )
            )
        )

        max_abs_rms_error = max(
            max_abs_rms_error,
            float(err),
        )


    assert max_abs_rms_error <= RMS_TOL

    return (
        out32,
        max_abs_rms_error,
    )


def evaluate_model(
    bench,
    model_name,
    model,
    x_t,
    y_t,
    prev_t,
    weight_t,
    frame,
):
    with torch.no_grad():

        if model_name == "MLP":

            pred = model(
                x_t
            )

        else:

            pred = model(
                x_t,
                prev_t,
            )


    assert pred.shape == y_t.shape

    assert torch.isfinite(
        pred
    ).all()


    balanced = bench.weighted_metrics(
        pred,
        y_t,
        weight_t,
    )

    per_dataset = bench.dataset_metrics(
        pred,
        y_t,
        frame,
    )


    result = {
        "BALANCED_ALL": {
            "n":
                int(
                    len(frame)
                ),

            "rmse":
                float(
                    balanced[
                        "rmse"
                    ]
                ),

            "mae":
                float(
                    balanced[
                        "mae"
                    ]
                ),
        }
    }


    for row in per_dataset:

        result[
            str(
                row[
                    "dataset"
                ]
            )
        ] = {
            "n":
                int(
                    row[
                        "n"
                    ]
                ),

            "rmse":
                float(
                    row[
                        "RMSE"
                    ]
                ),

            "mae":
                float(
                    row[
                        "MAE"
                    ]
                ),
        }


    assert set(
        result
    ) == set(
        DOMAINS
    )

    return result


# ===========================================================================
# PRE-EVALUATION PROVENANCE GUARDS
# ===========================================================================

assert sha(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

assert sha(
    CLEAN_MANIFEST
) == EXPECTED_CLEAN_MANIFEST_SHA


protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)


assert protocol[
    "status"
] == "FROZEN_BEFORE_SPECTRAL_RUNS"

assert protocol[
    "final_v3_execution_manifest"
][
    "sha256"
] == EXPECTED_FINAL_V3_EXEC_SHA


frequency_grid = protocol[
    "sinusoidal_frequency_sweep"
][
    "omega_over_pi"
]

perturbation_seeds = protocol[
    "replication"
][
    "perturbation_seeds"
]

target_rms = float(
    protocol[
        "sinusoidal_frequency_sweep"
    ][
        "rms_per_feature_standardized_units"
    ]
)


assert frequency_grid == [
    0.0625,
    0.125,
    0.25,
    0.50,
    0.75,
    1.00,
]

assert perturbation_seeds == [
    0,
    1,
    2,
    3,
    4,
]

assert math.isclose(
    target_rms,
    0.10,
    rel_tol=0.0,
    abs_tol=1e-15,
)


# Runner itself must have been committed unchanged.
rel = str(
    THIS.relative_to(
        REPO
    )
)

assert subprocess.run(
    [
        "git",
        "ls-files",
        "--error-unmatch",
        rel,
    ],
    cwd=REPO,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
).returncode == 0

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


script_commit = subprocess.check_output(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
    text=True,
).strip()

script_sha = sha(
    THIS
)


# ===========================================================================
# IMPORT EXACT FROZEN BENCHMARK IMPLEMENTATION
# ===========================================================================

spec = importlib.util.spec_from_file_location(
    "benchmark24_frequency",
    BENCH,
)

bench = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    bench
)


assert len(
    bench.FEATURES
) == 20

assert list(
    bench.DATASETS
) == [
    "HUST",
    "CALCE",
    "MATR",
]


RUNS.mkdir(
    parents=True,
    exist_ok=True,
)


completed_run_meta = []

global_max_rms_error = 0.0


# ===========================================================================
# FIVE FROZEN SPLITS
# ===========================================================================

for split_seed in range(5):

    full_df = bench.load_canonical_soh(
        split_seed
    )

    test = bench.make_split_tensor(
        full_df,
        "test",
    )

    frame = (
        test[
            "frame"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    base_x = (
        test[
            "x"
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    y_t = test[
        "y"
    ]

    prev_t = test[
        "prev"
    ]

    weight_t = test[
        "weight"
    ]


    # -----------------------------------------------------------------------
    # Generate each perturbation ONCE per split.
    # Every model and every training initialization reuses these tensors.
    # -----------------------------------------------------------------------

    condition_cache = []

    clean_tensor = torch.tensor(
        base_x,
        dtype=torch.float32,
    )

    clean_hash = sha_bytes(
        np.ascontiguousarray(
            base_x
        ).tobytes()
    )


    condition_cache.append({
        "condition_index":
            0,

        "condition_id":
            "clean",

        "scenario":
            "clean",

        "omega_over_pi":
            None,

        "perturbation_seed":
            None,

        "target_rms":
            0.0,

        "max_abs_rms_error":
            0.0,

        "input_tensor_sha256":
            clean_hash,

        "x":
            clean_tensor,
    })


    condition_index = 1


    for omega_over_pi in frequency_grid:

        for perturbation_seed in perturbation_seeds:

            x32, rms_error = (
                make_sinusoidal_input(
                    base_x,
                    frame,
                    bench.FEATURES,
                    omega_over_pi,
                    perturbation_seed,
                    target_rms,
                )
            )

            global_max_rms_error = max(
                global_max_rms_error,
                rms_error,
            )

            cid = (
                "sinusoid_"
                f"f{float(omega_over_pi):.4f}_"
                f"seed{int(perturbation_seed)}"
            )

            condition_cache.append({
                "condition_index":
                    condition_index,

                "condition_id":
                    cid,

                "scenario":
                    "sinusoidal_frequency",

                "omega_over_pi":
                    float(
                        omega_over_pi
                    ),

                "perturbation_seed":
                    int(
                        perturbation_seed
                    ),

                "target_rms":
                    target_rms,

                "max_abs_rms_error":
                    rms_error,

                "input_tensor_sha256":
                    sha_bytes(
                        np.ascontiguousarray(
                            x32
                        ).tobytes()
                    ),

                "x":
                    torch.tensor(
                        x32,
                        dtype=torch.float32,
                    ),
            })

            condition_index += 1


    assert len(
        condition_cache
    ) == 31


    pairing_hashes = {
        c[
            "condition_id"
        ]:
            c[
                "input_tensor_sha256"
            ]
        for c in condition_cache
    }


    # -----------------------------------------------------------------------
    # Three initializations x two frozen models.
    # -----------------------------------------------------------------------

    split_generated = 0


    for train_seed in range(3):

        run_dir = (
            FINAL_V3
            / f"split{split_seed}_train{train_seed}"
        )

        old_metrics = pd.read_csv(
            run_dir
            / "metrics.csv",
            low_memory=False,
        )

        old_metrics = old_metrics[
            old_metrics[
                "split"
            ].astype(str).str.lower()
            == "test"
        ].copy()

        assert len(
            old_metrics
        ) == 8

        old_metrics[
            "_model"
        ] = old_metrics[
            "model"
        ].map(
            norm_model_name
        )


        for model_name in MODELS:

            stem = (
                f"split{split_seed}_"
                f"train{train_seed}_"
                f"{model_name.lower()}"
            )

            metric_path = (
                RUNS
                / f"{stem}_frequency_metrics.csv"
            )

            meta_path = (
                RUNS
                / f"{stem}_evaluation.json"
            )


            # ---------------------------------------------------------------
            # Resume only an already atomically closed run.
            # No metric values are parsed during reuse.
            # ---------------------------------------------------------------

            if (
                metric_path.exists()
                or meta_path.exists()
            ):

                assert (
                    metric_path.exists()
                    and meta_path.exists()
                ), (
                    "orphaned run artifact",
                    stem,
                )

                meta = json.loads(
                    meta_path.read_text(
                        encoding="utf-8"
                    )
                )

                assert meta[
                    "contract"
                ] == "sinusoidal_frequency_evaluation_run_v1"

                assert meta[
                    "status"
                ] == "COMPLETE"

                assert meta[
                    "metric_file_sha256"
                ] == sha(
                    metric_path
                )

                assert meta[
                    "metric_rows"
                ] == 124

                assert csv_data_rows(
                    metric_path
                ) == 124

                assert meta[
                    "input_tensor_sha256_by_condition"
                ] == pairing_hashes

                completed_run_meta.append(
                    meta
                )

                continue


            if model_name == "MLP":

                model = bench.FeatureMatchedMLP(
                    input_dim=20,
                    hidden_dim=64,
                )

                state_path = (
                    run_dir
                    / "mlp_best_state.pt"
                )

            else:

                model = bench.FeatureMatchedCausalGCN(
                    input_dim=20,
                    hidden_dim=64,
                )

                state_path = (
                    run_dir
                    / "causal_gcn_best_state.pt"
                )


            state_sha = sha(
                state_path
            )

            model.load_state_dict(
                load_state(
                    state_path
                ),
                strict=True,
            )

            model.eval()

            parameter_count = int(
                sum(
                    p.numel()
                    for p in model.parameters()
                )
            )

            assert parameter_count == 5569


            rows = []

            clean_regression_pass = True
            max_clean_rmse_difference = 0.0
            max_clean_mae_difference = 0.0


            for condition in condition_cache:

                current = evaluate_model(
                    bench,
                    model_name,
                    model,
                    condition[
                        "x"
                    ],
                    y_t,
                    prev_t,
                    weight_t,
                    frame,
                )


                for domain in DOMAINS:

                    vals = current[
                        domain
                    ]

                    rows.append({
                        "split_seed":
                            split_seed,

                        "train_seed":
                            train_seed,

                        "model":
                            model_name,

                        "condition_index":
                            condition[
                                "condition_index"
                            ],

                        "condition_id":
                            condition[
                                "condition_id"
                            ],

                        "scenario":
                            condition[
                                "scenario"
                            ],

                        "omega_over_pi":
                            condition[
                                "omega_over_pi"
                            ],

                        "perturbation_seed":
                            condition[
                                "perturbation_seed"
                            ],

                        "target_rms_std_units":
                            condition[
                                "target_rms"
                            ],

                        "domain":
                            domain,

                        "n_rows":
                            vals[
                                "n"
                            ],

                        "rmse":
                            vals[
                                "rmse"
                            ],

                        "mae":
                            vals[
                                "mae"
                            ],
                    })


                # -----------------------------------------------------------
                # Clean condition must reproduce the frozen final-v3 metrics.
                # -----------------------------------------------------------

                if condition[
                    "scenario"
                ] == "clean":

                    for domain in DOMAINS:

                        old = old_metrics[
                            (
                                old_metrics[
                                    "_model"
                                ] == model_name
                            )
                            & (
                                old_metrics[
                                    "dataset"
                                ].astype(str)
                                == domain
                            )
                        ]

                        assert len(
                            old
                        ) == 1

                        old = old.iloc[
                            0
                        ]

                        rmse_diff = abs(
                            current[
                                domain
                            ][
                                "rmse"
                            ]
                            - float(
                                old[
                                    "RMSE"
                                ]
                            )
                        )

                        mae_diff = abs(
                            current[
                                domain
                            ][
                                "mae"
                            ]
                            - float(
                                old[
                                    "MAE"
                                ]
                            )
                        )

                        n_match = (
                            int(
                                current[
                                    domain
                                ][
                                    "n"
                                ]
                            )
                            == int(
                                old[
                                    "n"
                                ]
                            )
                        )

                        max_clean_rmse_difference = max(
                            max_clean_rmse_difference,
                            rmse_diff,
                        )

                        max_clean_mae_difference = max(
                            max_clean_mae_difference,
                            mae_diff,
                        )

                        clean_regression_pass = (
                            clean_regression_pass
                            and n_match
                            and rmse_diff
                            <= CLEAN_TOL
                            and mae_diff
                            <= CLEAN_TOL
                        )


            assert clean_regression_pass

            result = pd.DataFrame(
                rows
            )

            assert len(
                result
            ) == 124

            assert (
                result[
                    "condition_id"
                ].nunique()
                == 31
            )

            assert (
                result[
                    "domain"
                ].nunique()
                == 4
            )

            atomic_csv(
                result,
                metric_path,
            )


            meta = {
                "contract":
                    "sinusoidal_frequency_evaluation_run_v1",

                "status":
                    "COMPLETE",

                "split_seed":
                    split_seed,

                "train_seed":
                    train_seed,

                "model":
                    model_name,

                "state_file":
                    str(
                        state_path
                    ),

                "state_file_sha256":
                    state_sha,

                "parameter_count":
                    parameter_count,

                "spectral_protocol_sha256":
                    EXPECTED_PROTOCOL_SHA,

                "clean_graph_fourier_manifest_sha256":
                    EXPECTED_CLEAN_MANIFEST_SHA,

                "runner_script_sha256":
                    script_sha,

                "runner_script_commit":
                    script_commit,

                "training_reexecuted":
                    False,

                "scaler_refit":
                    False,

                "feature_generator_reexecuted":
                    False,

                "GPU_used":
                    False,

                "condition_count":
                    31,

                "primary_frequency_condition_count":
                    30,

                "frequency_count":
                    6,

                "perturbation_seed_count":
                    5,

                "domains_per_condition":
                    4,

                "metric_rows":
                    124,

                "metric_values_surfaced":
                    False,

                "analysis_performed":
                    False,

                "clean_regression_pass":
                    clean_regression_pass,

                "clean_max_abs_rmse_difference":
                    max_clean_rmse_difference,

                "clean_max_abs_mae_difference":
                    max_clean_mae_difference,

                "target_rms_standardized_units":
                    target_rms,

                "max_abs_rms_error":
                    max(
                        c[
                            "max_abs_rms_error"
                        ]
                        for c in condition_cache
                    ),

                "input_tensor_sha256_by_condition":
                    pairing_hashes,

                "metric_file":
                    str(
                        metric_path
                    ),

                "metric_file_sha256":
                    sha(
                        metric_path
                    ),
            }


            atomic_json(
                meta,
                meta_path,
            )

            completed_run_meta.append(
                meta
            )

            split_generated += 1


    print(
        f"split {split_seed}: complete "
        f"generated={split_generated}"
    )


# ===========================================================================
# EXECUTION CLOSURE
# ===========================================================================

metric_files = sorted(
    RUNS.glob(
        "*_frequency_metrics.csv"
    )
)

meta_files = sorted(
    RUNS.glob(
        "*_evaluation.json"
    )
)


assert len(
    metric_files
) == 30

assert len(
    meta_files
) == 30

assert len(
    completed_run_meta
) == 30


keys = {
    (
        int(
            m[
                "split_seed"
            ]
        ),
        int(
            m[
                "train_seed"
            ]
        ),
        str(
            m[
                "model"
            ]
        ),
    )
    for m in completed_run_meta
}

assert len(
    keys
) == 30


for split_seed in range(5):

    group = [
        m
        for m in completed_run_meta
        if int(
            m[
                "split_seed"
            ]
        ) == split_seed
    ]

    assert len(
        group
    ) == 6

    reference = group[
        0
    ][
        "input_tensor_sha256_by_condition"
    ]

    for m in group[
        1:
    ]:

        assert (
            m[
                "input_tensor_sha256_by_condition"
            ]
            == reference
        )


assert all(
    m[
        "training_reexecuted"
    ] is False
    for m in completed_run_meta
)

assert all(
    m[
        "scaler_refit"
    ] is False
    for m in completed_run_meta
)

assert all(
    m[
        "metric_values_surfaced"
    ] is False
    for m in completed_run_meta
)

assert all(
    m[
        "analysis_performed"
    ] is False
    for m in completed_run_meta
)

assert all(
    m[
        "clean_regression_pass"
    ] is True
    for m in completed_run_meta
)


max_clean_rmse_difference = max(
    float(
        m[
            "clean_max_abs_rmse_difference"
        ]
    )
    for m in completed_run_meta
)

max_clean_mae_difference = max(
    float(
        m[
            "clean_max_abs_mae_difference"
        ]
    )
    for m in completed_run_meta
)

global_max_rms_error = max(
    float(
        m[
            "max_abs_rms_error"
        ]
    )
    for m in completed_run_meta
)


manifest = {
    "contract":
        "final_sinusoidal_frequency_execution_v1",

    "status":
        "COMPLETE_BEFORE_VALUE_LEVEL_ANALYSIS",

    "spectral_protocol_sha256":
        EXPECTED_PROTOCOL_SHA,

    "clean_graph_fourier_manifest_sha256":
        EXPECTED_CLEAN_MANIFEST_SHA,

    "final_v3_execution_manifest_sha256":
        EXPECTED_FINAL_V3_EXEC_SHA,

    "runner_script_sha256":
        script_sha,

    "runner_script_commit":
        script_commit,

    "completed_runs":
        30,

    "models":
        MODELS,

    "split_count":
        5,

    "train_initializations_per_split":
        3,

    "frequency_grid_omega_over_pi":
        frequency_grid,

    "frequency_count":
        6,

    "perturbation_seeds":
        perturbation_seeds,

    "perturbation_seed_count":
        5,

    "conditions_per_run":
        31,

    "primary_frequency_conditions_per_run":
        30,

    "domains_per_condition":
        4,

    "metric_rows_per_run":
        124,

    "expected_total_metric_rows":
        3720,

    "target_rms_standardized_units":
        target_rms,

    "max_abs_realized_rms_error":
        global_max_rms_error,

    "paired_realization_across_models":
        True,

    "paired_realization_across_train_initializations":
        True,

    "phase_definition":
        (
            "SHA256(dataset|split_group|feature|perturbation_seed)"
            " mapped deterministically to [0,2pi)"
        ),

    "training_reexecuted":
        False,

    "scaler_refit":
        False,

    "feature_generator_reexecuted":
        False,

    "GPU_used":
        False,

    "clean_final_v3_metric_regression_barrier":
        "PASS",

    "clean_metric_tolerance":
        CLEAN_TOL,

    "clean_max_abs_rmse_difference":
        max_clean_rmse_difference,

    "clean_max_abs_mae_difference":
        max_clean_mae_difference,

    "metric_values_surfaced_during_execution":
        False,

    "analysis_performed":
        False,

    "inferential_unit":
        "split_seed",

    "inferential_n":
        5,

    "train_initializations_role":
        "nested_within_split",

    "perturbation_seeds_role":
        "nested_within_split",

    "metric_files": {
        p.name: {
            "sha256":
                sha(p),

            "rows":
                csv_data_rows(
                    p
                ),
        }
        for p in metric_files
    },

    "evaluation_metadata_files": {
        p.name: {
            "sha256":
                sha(p),
        }
        for p in meta_files
    },
}


atomic_json(
    manifest,
    EXEC_MANIFEST,
)


print()
print("=" * 86)
print("SINUSOIDAL FREQUENCY EXECUTION CLOSURE")
print("=" * 86)

print(
    "completed runs:",
    30
)

print(
    "conditions per run:",
    31
)

print(
    "primary frequency conditions per run:",
    30
)

print(
    "frequencies:",
    6
)

print(
    "perturbation seeds:",
    5
)

print(
    "domains per condition:",
    4
)

print(
    "expected metric rows:",
    3720
)

print(
    "clean final-v3 regression barrier:",
    True
)

print(
    "paired perturbations across models:",
    True
)

print(
    "training reexecuted:",
    False
)

print(
    "scaler refit:",
    False
)

print(
    "metric values surfaced:",
    False
)

print(
    "analysis performed:",
    False
)

print(
    "execution manifest SHA256:",
    sha(
        EXEC_MANIFEST
    )
)

print()
print(
    "SINUSOIDAL_FREQUENCY_EXECUTION_PASS=True"
)
