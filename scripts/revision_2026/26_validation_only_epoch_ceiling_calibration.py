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

OUTDIR = (
    BASE
    / "results"
    / "revision_2026"
    / "feature_matched_soh"
    / "calibration_validation_only_split0_train0"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXPECTED_BENCH_SHA = (
    "5d621f8d5764e417e41bc518ec72b9608566a087c552b8365d438c2344c06c5e"
)

SPLIT_SEED = 0
TRAIN_SEED = 0

HIDDEN_DIM = 64
LR = 1e-3
WEIGHT_DECAY = 1e-5

MAX_EPOCHS = 1500
PATIENCE = 30
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


assert sha256_file(
    BENCH
) == EXPECTED_BENCH_SHA


spec = importlib.util.spec_from_file_location(
    "benchmark24",
    BENCH,
)

mod = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    mod
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


# =========================================================================
# Canonical data.
#
# IMPORTANT:
# Only TRAIN and VALIDATION tensors are instantiated.
# No TEST tensor, prediction or metric is constructed in this script.
# =========================================================================

canonical = mod.load_canonical_soh(
    split_seed=SPLIT_SEED
)

data = {
    "train":
        mod.make_split_tensor(
            canonical,
            "train",
        ),

    "validation":
        mod.make_split_tensor(
            canonical,
            "validation",
        ),
}

assert set(data) == {
    "train",
    "validation",
}


print("=" * 80)
print("VALIDATION-ONLY OPTIMIZATION-BUDGET CALIBRATION")
print("=" * 80)

print(
    "benchmark_sha256:",
    sha256_file(
        BENCH
    )
)

print(
    "split_seed / train_seed:",
    SPLIT_SEED,
    "/",
    TRAIN_SEED,
)

print(
    "max_epochs:",
    MAX_EPOCHS
)

print(
    "patience:",
    PATIENCE
)

print(
    "learning_rate:",
    LR
)

print(
    "weight_decay:",
    WEIGHT_DECAY
)

print(
    "hidden_dim:",
    HIDDEN_DIM
)

print(
    "train rows:",
    data[
        "train"
    ][
        "n_rows"
    ]
)

print(
    "validation rows:",
    data[
        "validation"
    ][
        "n_rows"
    ]
)

print(
    "test tensor created:",
    False
)

print(
    "test metrics computed:",
    False
)


def train_validation_only(
    model_name,
):
    mod.set_seed(
        TRAIN_SEED
    )

    if model_name == "MLP":
        model = mod.FeatureMatchedMLP(
            input_dim=len(
                mod.FEATURES
            ),
            hidden_dim=HIDDEN_DIM,
        )

    elif model_name == "CAUSAL_GCN":
        model = mod.FeatureMatchedCausalGCN(
            input_dim=len(
                mod.FEATURES
            ),
            hidden_dim=HIDDEN_DIM,
        )

    else:
        raise ValueError(
            model_name
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
            train["x"],
            train["prev"],
        )

        err = (
            pred
            - train["y"]
        )

        loss = torch.sum(
            train["weight"]
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
                val["x"],
                val["prev"],
            )

            val_metrics = (
                mod.weighted_metrics(
                    val_pred,
                    val["y"],
                    val["weight"],
                )
            )

        val_rmse = float(
            val_metrics[
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

    elapsed = (
        time.time()
        - start
    )

    h = pd.DataFrame(
        history
    )

    h.to_csv(
        OUTDIR
        / (
            model_name.lower()
            + "_validation_history.csv"
        ),
        index=False,
        float_format="%.12g",
    )


    epochs_run = int(
        h[
            "epoch"
        ].iloc[-1]
    )

    best = float(
        h[
            "validation_balanced_rmse"
        ].min()
    )

    assert np.isclose(
        best,
        best_val,
        rtol=0.0,
        atol=1e-12,
    )


    val = h[
        "validation_balanced_rmse"
    ].to_numpy(
        dtype=float
    )


    def value_at(epoch):
        row = h[
            h[
                "epoch"
            ] == epoch
        ]

        if len(row) != 1:
            return np.nan

        return float(
            row[
                "validation_balanced_rmse"
            ].iloc[0]
        )


    v300 = value_at(
        300
    )

    improvement_after_300 = (
        v300 - best
        if np.isfinite(v300)
        else np.nan
    )

    rel_improvement_after_300 = (
        improvement_after_300
        / v300
        if (
            np.isfinite(v300)
            and v300 != 0
        )
        else np.nan
    )


    ntrend = min(
        30,
        len(h),
    )

    xtrend = h[
        "epoch"
    ].to_numpy(
        dtype=float
    )[
        -ntrend:
    ]

    ytrend = val[
        -ntrend:
    ]

    slope_last30 = float(
        np.polyfit(
            xtrend,
            ytrend,
            1,
        )[0]
    )


    result = {
        "model":
            model_name,

        "epochs_run":
            epochs_run,

        "best_epoch":
            int(
                best_epoch
            ),

        "best_validation_rmse":
            float(
                best_val
            ),

        "validation_rmse_epoch300":
            float(
                v300
            ),

        "absolute_improvement_after_epoch300":
            float(
                improvement_after_300
            ),

        "relative_improvement_after_epoch300":
            float(
                rel_improvement_after_300
            ),

        "epochs_after_best":
            int(
                epochs_run
                - best_epoch
            ),

        "stopped_by_patience":
            bool(
                stopped_by_patience
            ),

        "hit_epoch_ceiling":
            bool(
                epochs_run
                == MAX_EPOCHS
                and not stopped_by_patience
            ),

        "validation_slope_final30":
            slope_last30,

        "elapsed_seconds":
            float(
                elapsed
            ),
    }


    return result


results = []

for model_name in (
    "MLP",
    "CAUSAL_GCN",
):
    r = train_validation_only(
        model_name
    )

    results.append(
        r
    )

    print()
    print(
        model_name + ":"
    )

    print(
        "  epochs_run:",
        r[
            "epochs_run"
        ]
    )

    print(
        "  best_epoch:",
        r[
            "best_epoch"
        ]
    )

    print(
        "  best_validation_RMSE:",
        f"{r['best_validation_rmse']:.10f}"
    )

    print(
        "  validation_RMSE_epoch300:",
        f"{r['validation_rmse_epoch300']:.10f}"
    )

    print(
        "  relative_improvement_after_epoch300:",
        f"{100*r['relative_improvement_after_epoch300']:.4f}%"
    )

    print(
        "  epochs_after_best:",
        r[
            "epochs_after_best"
        ]
    )

    print(
        "  stopped_by_patience:",
        r[
            "stopped_by_patience"
        ]
    )

    print(
        "  hit_epoch_ceiling:",
        r[
            "hit_epoch_ceiling"
        ]
    )

    print(
        "  validation_slope_final30:",
        f"{r['validation_slope_final30']:.6e}"
    )

    print(
        "  elapsed_seconds:",
        f"{r['elapsed_seconds']:.2f}"
    )


summary = {
    "purpose":
        (
            "validation-only calibration of sufficient "
            "optimization epoch ceiling"
        ),

    "benchmark_sha256":
        sha256_file(
            BENCH
        ),

    "split_seed":
        SPLIT_SEED,

    "train_seed":
        TRAIN_SEED,

    "hidden_dim":
        HIDDEN_DIM,

    "learning_rate":
        LR,

    "weight_decay":
        WEIGHT_DECAY,

    "max_epochs":
        MAX_EPOCHS,

    "patience":
        PATIENCE,

    "device":
        "cpu",

    "test_tensor_created":
        False,

    "test_metrics_computed":
        False,

    "test_artifacts_read":
        False,

    "results":
        results,
}


summary_path = (
    OUTDIR
    / "validation_only_calibration_summary.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


print()
print(
    "test tensor created:",
    False
)

print(
    "test metrics computed:",
    False
)

print(
    "test artifacts read:",
    False
)

print(
    "summary SHA256:",
    sha256_file(
        summary_path
    )
)

print()
print(
    "VALIDATION_ONLY_EPOCH_CALIBRATION_PASS=True"
)
