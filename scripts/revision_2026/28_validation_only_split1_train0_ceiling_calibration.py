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
        Path.home() / "alper" / "battery_revision_2026",
    )
)

REPO = BASE / "repo" / "battery-gnn"

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
    / "calibration_validation_only_split1_train0_5000"
)

EXPECTED_BENCH_SHA = (
    "5d621f8d5764e417e41bc518ec72b9608566a087c552b8365d438c2344c06c5e"
)

SPLIT_SEED = 1
TRAIN_SEED = 0

HIDDEN_DIM = 64
LR = 1e-3
WEIGHT_DECAY = 1e-5

MAX_EPOCHS = 5000
PATIENCE = 30
THREADS = 4


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


assert sha256_file(BENCH) == EXPECTED_BENCH_SHA

assert torch.version.cuda is None
assert torch.cuda.is_available() is False


spec = importlib.util.spec_from_file_location(
    "benchmark24",
    BENCH,
)

mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


torch.set_num_threads(THREADS)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ============================================================
# TRAIN + VALIDATION ONLY
# ============================================================

canonical = mod.load_canonical_soh(
    split_seed=SPLIT_SEED
)

data = {
    "train": mod.make_split_tensor(
        canonical,
        "train",
    ),
    "validation": mod.make_split_tensor(
        canonical,
        "validation",
    ),
}

assert set(data) == {
    "train",
    "validation",
}


OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


def train_validation_only(model_name):

    mod.set_seed(TRAIN_SEED)

    if model_name == "MLP":
        model = mod.FeatureMatchedMLP(
            input_dim=len(mod.FEATURES),
            hidden_dim=HIDDEN_DIM,
        )

    elif model_name == "CAUSAL_GCN":
        model = mod.FeatureMatchedCausalGCN(
            input_dim=len(mod.FEATURES),
            hidden_dim=HIDDEN_DIM,
        )

    else:
        raise ValueError(model_name)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )


    train = data["train"]
    val = data["validation"]

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

        err = pred - train["y"]

        loss = torch.sum(
            train["weight"]
            * err.square()
        )

        assert torch.isfinite(loss)

        loss.backward()
        optimizer.step()


        model.eval()

        with torch.no_grad():

            vp = model(
                val["x"],
                val["prev"],
            )

            vm = mod.weighted_metrics(
                vp,
                val["y"],
                val["weight"],
            )

        val_rmse = float(
            vm["rmse"]
        )


        history.append({
            "epoch": epoch,
            "train_weighted_mse":
                float(loss.item()),
            "validation_balanced_rmse":
                val_rmse,
        })


        if val_rmse < (
            best_val - 1e-12
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


    h = pd.DataFrame(history)

    history_path = (
        OUTDIR
        / (
            model_name.lower()
            + "_validation_history.csv"
        )
    )

    h.to_csv(
        history_path,
        index=False,
        float_format="%.12g",
    )


    stop_epoch = int(
        h["epoch"].iloc[-1]
    )

    hit_ceiling = bool(
        stop_epoch == MAX_EPOCHS
        and not stopped_by_patience
    )


    return {
        "model":
            model_name,

        "best_epoch":
            int(best_epoch),

        "stop_epoch":
            stop_epoch,

        "epochs_after_best":
            int(
                stop_epoch
                - best_epoch
            ),

        "best_validation_rmse":
            float(best_val),

        "stopped_by_patience":
            bool(stopped_by_patience),

        "hit_epoch_ceiling":
            hit_ceiling,

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


print("=" * 78)
print("VALIDATION-ONLY SPLIT1/TRAIN0 CEILING CALIBRATION")
print("=" * 78)

print(
    "device:",
    "CPU"
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
    "train rows:",
    data["train"]["n_rows"]
)

print(
    "validation rows:",
    data["validation"]["n_rows"]
)

print(
    "test tensor created:",
    False
)

print(
    "test metrics read:",
    False
)


results = []

for model_name in (
    "MLP",
    "CAUSAL_GCN",
):

    r = train_validation_only(
        model_name
    )

    results.append(r)

    print()
    print(model_name + ":")

    print(
        "  best_epoch:",
        r["best_epoch"]
    )

    print(
        "  stop_epoch:",
        r["stop_epoch"]
    )

    print(
        "  epochs_after_best:",
        r["epochs_after_best"]
    )

    print(
        "  best_validation_RMSE:",
        f"{r['best_validation_rmse']:.10f}"
    )

    print(
        "  stopped_by_patience:",
        r["stopped_by_patience"]
    )

    print(
        "  hit_epoch_ceiling:",
        r["hit_epoch_ceiling"]
    )

    print(
        "  elapsed_seconds:",
        f"{r['elapsed_seconds']:.2f}"
    )


summary = {
    "purpose":
        "validation-only optimization ceiling calibration",

    "split_seed":
        SPLIT_SEED,

    "train_seed":
        TRAIN_SEED,

    "benchmark_sha256":
        sha256_file(BENCH),

    "max_epochs":
        MAX_EPOCHS,

    "patience":
        PATIENCE,

    "device":
        "cpu",

    "test_tensor_created":
        False,

    "test_metrics_read":
        False,

    "test_artifacts_read":
        False,

    "results":
        results,
}


summary_path = (
    OUTDIR
    / "summary.json"
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
    "test metrics read:",
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
    "VALIDATION_ONLY_SPLIT1_TRAIN0_CALIBRATION_PASS=True"
)
