from pathlib import Path
import hashlib
import importlib.util
import json
import os
import subprocess

import numpy as np
import pandas as pd
import torch


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo" / "battery-gnn"

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

ORIGINAL_RUNNER = (
    REPO
    / "scripts"
    / "revision_2026"
    / "40_final_temporal_control_runner.py"
)

THIS_SCRIPT = (
    REPO
    / "scripts"
    / "revision_2026"
    / "41_temporal_test_evaluation_resume.py"
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

ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "temporal_controls"
    / "final_temporal_benchmark_v1"
)

PROTOCOL = ROOT / "protocol.json"
CLOSURE = ROOT / "training_closure.json"
RUNS = ROOT / "runs"

MANIFEST = (
    ROOT
    / "test_evaluation_continuation_manifest.json"
)

PROGRESS = (
    ROOT
    / "test_evaluation_continuation_progress.json"
)


EXPECTED = {
    "benchmark":
        "5d621f8d5764e417e41bc518ec72b9608566a087c552b8365d438c2344c06c5e",

    "kernel":
        "23f733540a23954c278722c43a36e994c5431b603bfeb9e7ebd4474451bd3ff7",

    "original_runner":
        "bb1aa8974480a4fccee5dd7219b4c401ae892f4106a42beb8ebbaadf805cbfb3",

    "protocol":
        "b6116666e861c79b7332dbd599594846b86511e420d8305eadf82d30b87f0106",

    "closure":
        "07a75bfc5ab81d467afda76a8b4cd1097ca0609356bf4bdbd6eb2254a2cc1b55",

    "canonical":
        "49a7b7629cf9ee3f8ce0a7e3c52f2807878b752b4c9a30a876285fb891b00c14",

    "assignments":
        "914c8b90131f496ca396f63e0502147ca99848fdfd4b2c6b2a5887225b47d055",

    "base_scalers":
        "b9708b9321b521b87bdd05f969558134d7e0148285ef7c50f648c2028cd86de1",

    "secondary":
        "98607908e0cb8ee2bb9c102c061990d7282bef20f1535c16e9825a2eb3d1a82c",
}


FEATURES = [
    "v_min","v_q10","v_q25","v_median","v_q75","v_q90",
    "v_max","v_mean","v_std","v_phase_slope",
    "i_min","i_q10","i_q25","i_median","i_q75","i_q90",
    "i_max","i_mean","i_std","i_phase_slope",
]

DATASETS = ["HUST", "CALCE", "MATR"]

METHODS = [
    "moving_average3_mlp",
    "lowpass_binomial3_mlp",
    "savgol5_poly2_mlp",
    "kalman_q0p01_mlp",
]

FILTER_NAME = {
    "moving_average3_mlp": "moving_average3",
    "lowpass_binomial3_mlp": "lowpass_binomial3",
    "savgol5_poly2_mlp": "savgol5_poly2",
    "kalman_q0p01_mlp": "kalman_q0p01",
}

EXPECTED_TEST_ROWS = {
    0: 43974,
    1: 46273,
    2: 43818,
    3: 46156,
    4: 43749,
}

SPLIT_SEEDS = [0,1,2,3,4]
TRAIN_SEEDS = [0,1,2]


def sha256_file(path):
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
        json.dumps(obj, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def as_bool(s):
    return (
        s.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true","1","1.0","yes"])
    )


def weighted_r2(pred, y, weight):
    """
    Weighted coefficient of determination.

    Weights are normalized internally. The same frozen
    dataset/cell-balanced weights are used as for RMSE/MAE.
    """
    pred = pred.detach().cpu().double()
    y = y.detach().cpu().double()
    weight = weight.detach().cpu().double()

    assert pred.ndim == 1
    assert y.ndim == 1
    assert weight.ndim == 1
    assert len(pred) == len(y) == len(weight)

    assert torch.isfinite(pred).all()
    assert torch.isfinite(y).all()
    assert torch.isfinite(weight).all()
    assert torch.all(weight >= 0)

    w = weight / weight.sum()

    ybar = torch.sum(w * y)

    sse = torch.sum(
        w * (y - pred).square()
    )

    sst = torch.sum(
        w * (y - ybar).square()
    )

    if float(sst) <= 0.0:
        return float("nan")

    return float(
        1.0 - sse / sst
    )


# =============================================================================
# Immutable provenance guards
# =============================================================================

source_paths = {
    "benchmark": BENCH,
    "kernel": KERNEL,
    "original_runner": ORIGINAL_RUNNER,
    "protocol": PROTOCOL,
    "closure": CLOSURE,
    "canonical": CANON,
    "assignments": ASSIGNMENTS,
    "base_scalers": BASE_SCALERS,
    "secondary": SECONDARY,
}

for name, expected in EXPECTED.items():
    assert sha256_file(
        source_paths[name]
    ) == expected, name


closure = json.loads(
    CLOSURE.read_text(
        encoding="utf-8"
    )
)

assert closure[
    "contract"
] == "temporal_control_training_closure_v1"

assert closure[
    "training_fit_count"
] == 60

assert closure[
    "all_training_fits_patience_valid"
] is True

assert closure[
    "all_training_fits_below_epoch_ceiling"
] is True

assert closure[
    "all_training_fits_test_rows_constructed"
] == 0

assert closure[
    "test_metrics_read_before_closure"
] is False


# This continuation must itself already be tracked.
tracked = subprocess.run(
    [
        "git",
        "ls-files",
        "--error-unmatch",
        str(
            THIS_SCRIPT.relative_to(REPO)
        ),
    ],
    cwd=REPO,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

assert tracked.returncode == 0

CONTINUATION_SHA = sha256_file(
    THIS_SCRIPT
)


assert torch.version.cuda is None
assert torch.cuda.is_available() is False

torch.set_num_threads(4)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# =============================================================================
# Import frozen model + filters
# =============================================================================

spec = importlib.util.spec_from_file_location(
    "benchmark24",
    BENCH,
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


spec = importlib.util.spec_from_file_location(
    "causal_filters",
    KERNEL,
)
filters = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filters)


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
        df["soh_task_eligible"]
    )
].copy()

assert len(df) == 300553


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


def balanced_weights(frame):
    frame = frame.reset_index(drop=True)

    w = np.zeros(
        len(frame),
        dtype=np.float64,
    )

    assert set(
        frame["dataset"].unique()
    ) == set(DATASETS)

    for dataset in DATASETS:
        dm = (
            frame["dataset"]
            == dataset
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

            w[
                gm.to_numpy()
            ] = (
                1.0
                / 3.0
                / len(groups)
                / n_obs
            )

    assert np.isclose(
        w.sum(),
        1.0,
        atol=1e-12,
        rtol=0.0,
    )

    return torch.tensor(
        w.astype(np.float32),
        dtype=torch.float32,
    )


def apply_method(x, method):
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

    raise ValueError(method)


def prepare_test(method, split_seed):
    a = assignments[
        assignments["split_seed"]
        == split_seed
    ][
        [
            "split_group",
            "split",
        ]
    ].copy()

    work = df.merge(
        a,
        on="split_group",
        how="left",
        validate="many_to_one",
    )

    # TEST cells only, and only after verified training closure.
    work = work[
        work["split"]
        == "test"
    ].copy().reset_index(drop=True)

    assert len(work) == EXPECTED_TEST_ROWS[
        split_seed
    ]

    assert work[
        "split_group"
    ].nunique() == 38


    p0 = base_scalers[
        (
            base_scalers["split_seed"]
            == split_seed
        )
        &
        (
            base_scalers["scaler_scope"]
            == "pooled_supervised"
        )
    ].set_index("feature").loc[
        FEATURES
    ]

    raw = (
        work[FEATURES]
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
        - p0["mean"].to_numpy(
            dtype=float
        )
    ) / p0["scale"].to_numpy(
        dtype=float
    )


    filtered = np.empty_like(z)

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
            g["observation_index"],
            errors="raise",
        ).to_numpy(
            dtype=np.int64
        )

        if len(obs) > 1:
            assert np.all(
                np.diff(obs) == 1
            )

        filtered[idx] = apply_method(
            z[idx],
            method,
        )


    p1 = secondary[
        (
            secondary["split_seed"]
            == split_seed
        )
        &
        (
            secondary["filter_name"]
            == FILTER_NAME[method]
        )
    ].set_index("feature").loc[
        FEATURES
    ]

    x = (
        filtered
        - p1["secondary_mean"].to_numpy(
            dtype=float
        )
    ) / p1["secondary_scale"].to_numpy(
        dtype=float
    )

    y = pd.to_numeric(
        work["SOH"],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    assert np.isfinite(x).all()
    assert np.isfinite(y).all()

    meta = work[
        [
            "dataset",
            "split_group",
        ]
    ].reset_index(drop=True)

    return {
        "x":
            torch.tensor(
                x.astype(np.float32),
                dtype=torch.float32,
            ),

        "y":
            torch.tensor(
                y.astype(np.float32),
                dtype=torch.float32,
            ),

        "weight":
            balanced_weights(meta),

        "dataset":
            meta[
                "dataset"
            ].astype(str).to_numpy(),

        "rows":
            len(work),

        "groups":
            int(
                meta[
                    "split_group"
                ].nunique()
            ),
    }


def paths_for(
    method,
    split_seed,
    train_seed,
):
    d = (
        RUNS
        / method
        / (
            f"split{split_seed}"
            f"_train{train_seed}"
        )
    )

    return {
        "fit":
            d / "fit.json",

        "state":
            d / "best_state.pt",

        "metrics":
            d / "test_metrics.csv",

        "evaluation":
            d / "evaluation.json",
    }


def evaluate(
    method,
    split_seed,
    train_seed,
    test,
):
    p = paths_for(
        method,
        split_seed,
        train_seed,
    )

    fit = json.loads(
        p["fit"].read_text(
            encoding="utf-8"
        )
    )

    assert fit[
        "protocol_sha256"
    ] == EXPECTED[
        "protocol"
    ]

    assert fit[
        "test_rows_constructed_during_training"
    ] == 0


    state = torch.load(
        p["state"],
        map_location="cpu",
        weights_only=True,
    )

    model = bench.FeatureMatchedMLP(
        input_dim=20,
        hidden_dim=64,
    )

    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        pred = model(
            test["x"],
            None,
        )

    rows = []


    # Balanced aggregate.
    overall = bench.weighted_metrics(
        pred,
        test["y"],
        test["weight"],
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
            test["rows"],

        "RMSE":
            float(
                overall["rmse"]
            ),

        "MAE":
            float(
                overall["mae"]
            ),

        "R2":
            np.nan,
    })


    # Dataset-specific cell-balanced metrics.
    for dataset in DATASETS:
        mask_np = (
            test["dataset"]
            == dataset
        )

        mask = torch.tensor(
            mask_np,
            dtype=torch.bool,
        )

        w = test["weight"][
            mask
        ].clone()

        w = w / w.sum()

        m = bench.weighted_metrics(
            pred[mask],
            test["y"][mask],
            w,
        )

        r2 = weighted_r2(
            pred[mask],
            test["y"][mask],
            w,
        )

        assert np.isfinite(r2)

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
                    m["rmse"]
                ),

            "MAE":
                float(
                    m["mae"]
                ),

            "R2":
                r2,
        })


    metrics = pd.DataFrame(rows)

    assert len(metrics) == 4

    assert np.isfinite(
        metrics["RMSE"]
    ).all()

    assert np.isfinite(
        metrics["MAE"]
    ).all()

    assert np.isfinite(
        metrics.loc[
            metrics["dataset"]
            != "BALANCED_ALL",
            "R2",
        ]
    ).all()


    metrics.to_csv(
        p["metrics"],
        index=False,
        float_format="%.12g",
    )


    evaluation = {
        "contract":
            "temporal_control_test_evaluation_v2",

        "protocol_sha256":
            EXPECTED[
                "protocol"
            ],

        "training_closure_sha256":
            EXPECTED[
                "closure"
            ],

        "original_runner_sha256":
            EXPECTED[
                "original_runner"
            ],

        "continuation_script_sha256":
            CONTINUATION_SHA,

        "continuation_reason":
            (
                "Original test phase stopped before writing any "
                "test artifact because frozen benchmark "
                "weighted_metrics did not provide an r2 key. "
                "R2 is computed explicitly here using the same "
                "normalized cell-balanced weights."
            ),

        "method":
            method,

        "split_seed":
            split_seed,

        "train_seed":
            train_seed,

        "test_rows":
            test["rows"],

        "test_groups":
            test["groups"],

        "test_evaluated_only_after_training_closure":
            True,

        "training_reexecuted":
            False,

        "r2_definition":
            (
                "1 - weighted_SSE / weighted_SST, "
                "with normalized frozen cell-balanced weights"
            ),

        "test_metrics_used_to_change_protocol":
            False,

        "test_metrics_surfaced_by_continuation":
            False,

        "test_metrics_sha256":
            sha256_file(
                p["metrics"]
            ),
    }


    atomic_json(
        evaluation,
        p["evaluation"],
    )

    return evaluation


def inspect_existing(
    method,
    split_seed,
    train_seed,
):
    p = paths_for(
        method,
        split_seed,
        train_seed,
    )

    if not (
        p["metrics"].is_file()
        and p["evaluation"].is_file()
    ):
        return None

    try:
        e = json.loads(
            p["evaluation"].read_text(
                encoding="utf-8"
            )
        )

        assert e[
            "contract"
        ] == "temporal_control_test_evaluation_v2"

        assert e[
            "training_closure_sha256"
        ] == EXPECTED["closure"]

        assert e[
            "continuation_script_sha256"
        ] == CONTINUATION_SHA

        assert e[
            "training_reexecuted"
        ] is False

        assert e[
            "test_metrics_surfaced_by_continuation"
        ] is False

        assert sha256_file(
            p["metrics"]
        ) == e[
            "test_metrics_sha256"
        ]

        return e

    except Exception:
        return None


# =============================================================================
# Execute 60 TEST-ONLY evaluations
# =============================================================================

print("=" * 80)
print("TEMPORAL TEST-EVALUATION CONTINUATION")
print("=" * 80)

print(
    "training closure exact:",
    True
)

print(
    "training reexecuted:",
    False
)

print(
    "original failure before first test artifact:",
    True
)

print(
    "R2 computation corrected explicitly:",
    True
)

print(
    "test metric values surfaced:",
    False
)

print()


records = []
counter = 0


for method in METHODS:
    for split_seed in SPLIT_SEEDS:

        test = prepare_test(
            method,
            split_seed,
        )

        for train_seed in TRAIN_SEEDS:
            counter += 1

            existing = inspect_existing(
                method,
                split_seed,
                train_seed,
            )

            if existing is not None:
                e = existing
                status = "SKIP"

            else:
                e = evaluate(
                    method,
                    split_seed,
                    train_seed,
                    test,
                )
                status = "DONE"

            records.append(e)

            print(
                f"[TEST {counter:02d}/60] "
                f"{method} "
                f"split={split_seed} "
                f"seed={train_seed} "
                f"{status} "
                f"(values hidden)"
            )

            atomic_json(
                {
                    "contract":
                        "temporal_test_continuation_progress_v1",

                    "completed":
                        len(records),

                    "planned":
                        60,

                    "training_reexecuted":
                        False,

                    "test_metric_values_surfaced":
                        False,
                },
                PROGRESS,
            )


assert len(records) == 60


# Disk re-inspection.
for method in METHODS:
    for split_seed in SPLIT_SEEDS:
        for train_seed in TRAIN_SEEDS:
            assert inspect_existing(
                method,
                split_seed,
                train_seed,
            ) is not None


manifest = {
    "contract":
        "temporal_test_evaluation_continuation_manifest_v1",

    "protocol_sha256":
        EXPECTED["protocol"],

    "training_closure_sha256":
        EXPECTED["closure"],

    "original_runner_sha256":
        EXPECTED["original_runner"],

    "continuation_script_sha256":
        CONTINUATION_SHA,

    "completed_test_evaluations":
        60,

    "training_fit_count_reused":
        60,

    "training_reexecuted":
        False,

    "first_original_test_failure_wrote_test_artifact":
        False,

    "r2_bug_corrected_without_training_changes":
        True,

    "test_metrics_surfaced":
        False,

    "test_metrics_used_to_change_protocol":
        False,

    "records":
        records,
}


atomic_json(
    manifest,
    MANIFEST,
)


print()
print("=" * 80)
print("TEST-EVALUATION CONTINUATION SUMMARY")
print("=" * 80)

print(
    "training fits reused:",
    60,
    "/ 60"
)

print(
    "training reexecuted:",
    False
)

print(
    "test evaluations:",
    60,
    "/ 60"
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
    "continuation manifest SHA256:",
    sha256_file(
        MANIFEST
    )
)

print()
print(
    "TEMPORAL_TEST_EVALUATION_CONTINUATION_PASS=True"
)
