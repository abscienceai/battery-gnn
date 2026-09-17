from pathlib import Path
import contextlib
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch


# =============================================================================
# Frozen locations / contracts
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

BENCH = (
    REPO
    / "scripts"
    / "revision_2026"
    / "24_feature_matched_soh_benchmark.py"
)

RUNNER = (
    REPO
    / "scripts"
    / "revision_2026"
    / "29_full_clean_soh_v3_runner.py"
)

PROTOCOL = (
    BASE
    / "results"
    / "revision_2026"
    / "feature_matched_soh"
    / "full_clean_benchmark_protocol_v3.json"
)

V3_ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "feature_matched_soh"
    / "final_v3"
)

RUN_LOG_DIR = (
    BASE
    / "logs"
    / "full_clean_soh_v3"
)

STATUS_PATH = (
    V3_ROOT
    / "runner_status.json"
)

FINAL_MANIFEST = (
    V3_ROOT
    / "final_v3_execution_manifest.json"
)


EXPECTED_BENCH_SHA = (
    "5d621f8d5764e417e41bc518ec72b9608566a087c552b8365d438c2344c06c5e"
)

EXPECTED_PROTOCOL_SHA = (
    "2dec14c210ae3f02e5f65e4acc24f9c4adc747bc071509cb144e69dba707f4dd"
)

EXPECTED_BENCHMARK_COMMIT = (
    "3b030ade470754c753ca966e15dd81a362d4b811"
)

EXPECTED_CALIBRATION2_COMMIT = (
    "eb84f1a44844302f96dd93b2a967646f354f1171"
)

MAX_EPOCHS = 5000
PATIENCE = 30
THREADS = 4

PLANNED = [
    (split_seed, train_seed)
    for split_seed in range(5)
    for train_seed in range(3)
]


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


def current_branch():
    return subprocess.check_output(
        [
            "git",
            "branch",
            "--show-current",
        ],
        cwd=REPO,
        text=True,
    ).strip()


def run_dir(
    split_seed,
    train_seed,
):
    return (
        V3_ROOT
        / (
            f"split{split_seed}"
            f"_train{train_seed}"
        )
    )


def required_files(path):
    return [
        path / "config.json",
        path / "metrics.csv",
        path / "mlp_history.csv",
        path / "causal_gcn_history.csv",
        path / "mlp_best_state.pt",
        path / "causal_gcn_best_state.pt",
    ]


# =============================================================================
# Completed-run inspection
#
# IMPORTANT:
# metrics.csv is checked only for existence/non-zero file size.
# Its contents are NOT parsed here.
# =============================================================================

def inspect_completed_run(
    split_seed,
    train_seed,
):

    path = run_dir(
        split_seed,
        train_seed,
    )

    files = required_files(
        path
    )

    if not all(
        p.is_file()
        and p.stat().st_size > 0
        for p in files
    ):
        return None


    try:

        config = json.loads(
            (
                path
                / "config.json"
            ).read_text(
                encoding="utf-8"
            )
        )


        # ---------------------------------------------------------------------
        # Frozen configuration
        # ---------------------------------------------------------------------

        assert config[
            "smoke"
        ] is False

        assert int(
            config[
                "split_seed"
            ]
        ) == split_seed

        assert int(
            config[
                "train_seed"
            ]
        ) == train_seed

        assert int(
            config[
                "hidden_dim"
            ]
        ) == 64

        assert np.isclose(
            float(
                config[
                    "learning_rate"
                ]
            ),
            0.001,
            rtol=0.0,
            atol=0.0,
        )

        assert np.isclose(
            float(
                config[
                    "weight_decay"
                ]
            ),
            0.00001,
            rtol=0.0,
            atol=0.0,
        )

        assert int(
            config[
                "max_epochs"
            ]
        ) == MAX_EPOCHS

        assert int(
            config[
                "patience"
            ]
        ) == PATIENCE

        assert config[
            "device"
        ] == "cpu"

        assert int(
            config[
                "feature_count"
            ]
        ) == 20

        assert int(
            config[
                "eligible_observations"
            ]
        ) == 300553

        assert int(
            config[
                "parameter_count_each_model"
            ]
        ) == 5569

        assert config[
            "parameters_equal"
        ] is True

        assert config[
            "initial_weights_equal_for_same_train_seed"
        ] is True

        assert config[
            "future_edges"
        ] is False

        assert config[
            "reverse_edges"
        ] is False

        assert config[
            "cross_cell_edges"
        ] is False

        assert config[
            "cycle_index_feature"
        ] is False

        assert config[
            "target_feature"
        ] is False


        # ---------------------------------------------------------------------
        # TRAIN / VALIDATION histories only
        # ---------------------------------------------------------------------

        mh = pd.read_csv(
            path
            / "mlp_history.csv",
            low_memory=False,
        )

        gh = pd.read_csv(
            path
            / "causal_gcn_history.csv",
            low_memory=False,
        )


        for h in (
            mh,
            gh,
        ):

            assert set(
                [
                    "epoch",
                    "train_weighted_mse",
                    "validation_balanced_rmse",
                ]
            ).issubset(
                h.columns
            )

            assert 1 <= len(
                h
            ) <= MAX_EPOCHS

            epochs = pd.to_numeric(
                h[
                    "epoch"
                ],
                errors="raise",
            ).to_numpy(
                dtype=int
            )

            assert np.array_equal(
                epochs,
                np.arange(
                    1,
                    len(h) + 1,
                ),
            )

            assert np.isfinite(
                pd.to_numeric(
                    h[
                        "train_weighted_mse"
                    ],
                    errors="raise",
                ).to_numpy(
                    dtype=float
                )
            ).all()

            assert np.isfinite(
                pd.to_numeric(
                    h[
                        "validation_balanced_rmse"
                    ],
                    errors="raise",
                ).to_numpy(
                    dtype=float
                )
            ).all()


        mlp_best = int(
            config[
                "model_training"
            ][
                "MLP"
            ][
                "best_epoch"
            ]
        )

        gcn_best = int(
            config[
                "model_training"
            ][
                "CAUSAL_GCN"
            ][
                "best_epoch"
            ]
        )

        mlp_stop = int(
            mh[
                "epoch"
            ].iloc[-1]
        )

        gcn_stop = int(
            gh[
                "epoch"
            ].iloc[-1]
        )


        assert 1 <= mlp_best <= mlp_stop
        assert 1 <= gcn_best <= gcn_stop


        mlp_hist_best = int(
            mh.loc[
                mh[
                    "validation_balanced_rmse"
                ].idxmin(),
                "epoch",
            ]
        )

        gcn_hist_best = int(
            gh.loc[
                gh[
                    "validation_balanced_rmse"
                ].idxmin(),
                "epoch",
            ]
        )


        assert (
            mlp_best
            == mlp_hist_best
        )

        assert (
            gcn_best
            == gcn_hist_best
        )


        mlp_after_best = (
            mlp_stop
            - mlp_best
        )

        gcn_after_best = (
            gcn_stop
            - gcn_best
        )


        mlp_hit_ceiling = (
            mlp_stop
            >= MAX_EPOCHS
        )

        gcn_hit_ceiling = (
            gcn_stop
            >= MAX_EPOCHS
        )


        # A valid v3 fit must terminate via the frozen
        # 30-epoch patience rule BEFORE epoch 5000.
        mlp_patience_valid = (
            not mlp_hit_ceiling
            and mlp_after_best
            == PATIENCE
        )

        gcn_patience_valid = (
            not gcn_hit_ceiling
            and gcn_after_best
            == PATIENCE
        )


        # ---------------------------------------------------------------------
        # Do NOT parse metrics.csv here.
        # ---------------------------------------------------------------------

        metrics_size = int(
            (
                path
                / "metrics.csv"
            ).stat().st_size
        )

        assert metrics_size > 0


        return {
            "split_seed":
                split_seed,

            "train_seed":
                train_seed,

            "status":
                "complete",

            "mlp_best_epoch":
                mlp_best,

            "mlp_stop_epoch":
                mlp_stop,

            "mlp_epochs_after_best":
                mlp_after_best,

            "mlp_hit_ceiling":
                mlp_hit_ceiling,

            "mlp_patience_valid":
                mlp_patience_valid,

            "gcn_best_epoch":
                gcn_best,

            "gcn_stop_epoch":
                gcn_stop,

            "gcn_epochs_after_best":
                gcn_after_best,

            "gcn_hit_ceiling":
                gcn_hit_ceiling,

            "gcn_patience_valid":
                gcn_patience_valid,

            "mlp_elapsed_seconds":
                float(
                    config[
                        "model_training"
                    ][
                        "MLP"
                    ][
                        "elapsed_seconds"
                    ]
                ),

            "gcn_elapsed_seconds":
                float(
                    config[
                        "model_training"
                    ][
                        "CAUSAL_GCN"
                    ][
                        "elapsed_seconds"
                    ]
                ),

            "config_sha256":
                sha256_file(
                    path
                    / "config.json"
                ),

            "mlp_history_sha256":
                sha256_file(
                    path
                    / "mlp_history.csv"
                ),

            "gcn_history_sha256":
                sha256_file(
                    path
                    / "causal_gcn_history.csv"
                ),

            "mlp_state_sha256":
                sha256_file(
                    path
                    / "mlp_best_state.pt"
                ),

            "gcn_state_sha256":
                sha256_file(
                    path
                    / "causal_gcn_best_state.pt"
                ),

            "metrics_file_exists":
                True,

            "metrics_file_size":
                metrics_size,

            "metrics_values_parsed_by_runner":
                False,
        }


    except Exception:
        return None


# =============================================================================
# Frozen protocol / environment guards
# =============================================================================

assert current_branch() == (
    "major-revision-2026"
)

assert sha256_file(
    BENCH
) == EXPECTED_BENCH_SHA

assert sha256_file(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA


protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)


assert protocol[
    "protocol"
] == (
    "full_clean_feature_matched_soh_v3"
)

assert protocol[
    "status"
] == (
    "FINAL_FROZEN_BEFORE_V3_RUNS"
)

assert protocol[
    "benchmark_model_commit"
] == EXPECTED_BENCHMARK_COMMIT

assert protocol[
    "calibration2_commit"
] == EXPECTED_CALIBRATION2_COMMIT

assert protocol[
    "all_ceiling_decisions_validation_only"
] is True

assert protocol[
    "test_used_for_ceiling_selection"
] is False

assert protocol[
    "v2_exclusion_from_final_aggregation"
] is True

assert int(
    protocol[
        "max_epochs"
    ]
) == MAX_EPOCHS

assert int(
    protocol[
        "early_stopping_patience"
    ]
) == PATIENCE

assert int(
    protocol[
        "planned_paired_run_count"
    ]
) == 15

assert int(
    protocol[
        "planned_model_fit_count"
    ]
) == 30


# CPU-only final execution contract.
assert torch.version.cuda is None
assert torch.cuda.is_available() is False


# Runner itself must be tracked before execution.
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


# =============================================================================
# Import frozen benchmark and redirect output root to FINAL V3
# =============================================================================

spec = importlib.util.spec_from_file_location(
    "benchmark24_frozen_v3",
    BENCH,
)

mod = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    mod
)


assert sha256_file(
    BENCH
) == EXPECTED_BENCH_SHA


mod.RESULT_ROOT = V3_ROOT


V3_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

RUN_LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# Compact start report
# =============================================================================

print("=" * 78)
print("FULL CLEAN SOH V3 RESUME-SAFE RUNNER")
print("=" * 78)

print(
    "benchmark SHA locked:",
    True
)

print(
    "protocol SHA locked:",
    True
)

print(
    "runner tracked before execution:",
    True
)

print(
    "runner commit:",
    RUNNER_COMMIT
)

print(
    "runner SHA256:",
    RUNNER_SHA
)

print(
    "planned paired runs:",
    len(PLANNED)
)

print(
    "planned model fits:",
    len(PLANNED) * 2
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
    "test metric values parsed during series:",
    False
)

print(
    "test metric values printed during series:",
    False
)

print()


records = []

invocation_start = time.time()


# =============================================================================
# Execute / resume all 15 paired runs
# =============================================================================

for pair_index, (
    split_seed,
    train_seed,
) in enumerate(
    PLANNED,
    start=1,
):

    existing = inspect_completed_run(
        split_seed,
        train_seed,
    )


    if existing is not None:

        if not (
            existing[
                "mlp_patience_valid"
            ]
            and existing[
                "gcn_patience_valid"
            ]
        ):

            print(
                f"[{pair_index:02d}/15] "
                f"split={split_seed} "
                f"train={train_seed} "
                f"INVALID_EXISTING_RUN"
            )

            print(
                "CEILING_OR_PATIENCE_GUARD=FAIL"
            )

            raise SystemExit(2)


        records.append(
            existing
        )


        print(
            f"[{pair_index:02d}/15] "
            f"split={split_seed} "
            f"train={train_seed} "
            f"SKIP complete | "
            f"MLP best/stop="
            f"{existing['mlp_best_epoch']}/"
            f"{existing['mlp_stop_epoch']} | "
            f"GCN best/stop="
            f"{existing['gcn_best_epoch']}/"
            f"{existing['gcn_stop_epoch']}"
        )

        continue


    print(
        f"[{pair_index:02d}/15] "
        f"split={split_seed} "
        f"train={train_seed} "
        f"RUNNING..."
    )

    sys.stdout.flush()


    run_log = (
        RUN_LOG_DIR
        / (
            f"split{split_seed}"
            f"_train{train_seed}.log"
        )
    )


    pair_start = time.time()

    old_argv = sys.argv[:]


    sys.argv = [
        str(BENCH),

        "--split-seed",
        str(split_seed),

        "--train-seed",
        str(train_seed),

        "--hidden-dim",
        "64",

        "--lr",
        "0.001",

        "--weight-decay",
        "0.00001",

        "--max-epochs",
        str(MAX_EPOCHS),

        "--patience",
        str(PATIENCE),

        "--threads",
        str(THREADS),
    ]


    failed = False


    try:

        with run_log.open(
            "w",
            encoding="utf-8",
        ) as log:

            with contextlib.redirect_stdout(
                log
            ), contextlib.redirect_stderr(
                log
            ):

                try:

                    mod.main()

                except Exception:

                    traceback.print_exc(
                        file=log
                    )

                    failed = True


    finally:

        sys.argv = old_argv


    if failed:

        print(
            f"[{pair_index:02d}/15] "
            f"split={split_seed} "
            f"train={train_seed} "
            f"FAILED"
        )

        print(
            "Detailed traceback saved to:",
            run_log,
        )

        print(
            "TEST_VALUES_NOT_SURFACED=True"
        )

        raise SystemExit(1)


    result = inspect_completed_run(
        split_seed,
        train_seed,
    )


    if result is None:

        print(
            f"[{pair_index:02d}/15] "
            f"split={split_seed} "
            f"train={train_seed} "
            f"OUTPUT_VALIDATION_FAIL"
        )

        raise SystemExit(1)


    records.append(
        result
    )


    wall = (
        time.time()
        - pair_start
    )


    print(
        f"[{pair_index:02d}/15] "
        f"split={split_seed} "
        f"train={train_seed} "
        f"DONE {wall:.1f}s | "
        f"MLP best/stop="
        f"{result['mlp_best_epoch']}/"
        f"{result['mlp_stop_epoch']} "
        f"patience_ok="
        f"{result['mlp_patience_valid']} | "
        f"GCN best/stop="
        f"{result['gcn_best_epoch']}/"
        f"{result['gcn_stop_epoch']} "
        f"patience_ok="
        f"{result['gcn_patience_valid']}"
    )


    # -------------------------------------------------------------------------
    # Update resume state after every completed pair.
    # No test metric values stored here.
    # -------------------------------------------------------------------------

    atomic_json(
        {
            "execution_contract":
                "full_clean_soh_v3_resume_state_v1",

            "protocol_sha256":
                EXPECTED_PROTOCOL_SHA,

            "benchmark_sha256":
                EXPECTED_BENCH_SHA,

            "runner_sha256":
                RUNNER_SHA,

            "runner_commit":
                RUNNER_COMMIT,

            "completed_pair_count":
                len(records),

            "completed_pairs":
                records,

            "test_metric_values_parsed":
                False,

            "test_metric_values_surfaced":
                False,
        },
        STATUS_PATH,
    )


    # -------------------------------------------------------------------------
    # Frozen final-validity rule.
    # -------------------------------------------------------------------------

    if not (
        result[
            "mlp_patience_valid"
        ]
        and result[
            "gcn_patience_valid"
        ]
    ):

        print()
        print(
            "CEILING_OR_PATIENCE_GUARD=FAIL"
        )

        print(
            "Final aggregation must NOT proceed."
        )

        raise SystemExit(2)


# =============================================================================
# Final execution validation
# =============================================================================

assert len(
    records
) == 15


observed_pairs = {
    (
        int(
            r[
                "split_seed"
            ]
        ),
        int(
            r[
                "train_seed"
            ]
        ),
    )
    for r in records
}


assert observed_pairs == set(
    PLANNED
)


assert all(
    r[
        "mlp_patience_valid"
    ]
    and r[
        "gcn_patience_valid"
    ]
    for r in records
)


assert all(
    not r[
        "mlp_hit_ceiling"
    ]
    and not r[
        "gcn_hit_ceiling"
    ]
    for r in records
)


total_elapsed = (
    time.time()
    - invocation_start
)


# =============================================================================
# Final manifest
#
# Still no parsing of metrics.csv values.
# =============================================================================

manifest = {
    "execution_contract":
        "full_clean_soh_v3_execution_v1",

    "protocol_sha256":
        EXPECTED_PROTOCOL_SHA,

    "benchmark_sha256":
        EXPECTED_BENCH_SHA,

    "runner_sha256":
        RUNNER_SHA,

    "runner_commit":
        RUNNER_COMMIT,

    "planned_pairs":
        [
            {
                "split_seed":
                    s,

                "train_seed":
                    t,
            }
            for s, t in PLANNED
        ],

    "completed_pair_count":
        15,

    "completed_model_fit_count":
        30,

    "all_fits_patience_valid":
        True,

    "all_fits_below_epoch_ceiling":
        True,

    "epoch_ceiling":
        MAX_EPOCHS,

    "patience":
        PATIENCE,

    "device":
        "cpu",

    "test_metric_values_parsed_during_series":
        False,

    "test_metric_values_surfaced_during_series":
        False,

    "test_metric_values_used_to_change_protocol":
        False,

    "records":
        records,

    "runner_wall_seconds_this_invocation":
        total_elapsed,
}


atomic_json(
    manifest,
    FINAL_MANIFEST,
)


# =============================================================================
# Compact completion report
# =============================================================================

print()
print("=" * 78)
print("FINAL V3 EXECUTION SUMMARY")
print("=" * 78)

print(
    "completed paired runs:",
    len(records),
    "/ 15"
)

print(
    "completed model fits:",
    len(records) * 2,
    "/ 30"
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
    "test metric values parsed during series:",
    False
)

print(
    "test metric values surfaced during series:",
    False
)

print(
    "test metrics used to change protocol:",
    False
)

print(
    "execution manifest SHA256:",
    sha256_file(
        FINAL_MANIFEST
    )
)

print(
    "runner status SHA256:",
    sha256_file(
        STATUS_PATH
    )
)

print()
print(
    "FULL_CLEAN_SOH_V3_RUNNER_PASS=True"
)
