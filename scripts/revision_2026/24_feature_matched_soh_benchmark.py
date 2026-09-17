from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =============================================================================
# Immutable data contracts
# =============================================================================

BASE = Path(
    os.environ.get(
        "BATTERY_REVISION_BASE",
        Path.home()
        / "alper"
        / "battery_revision_2026",
    )
)

CANON = (
    BASE
    / "data"
    / "processed"
    / "revision_2026"
    / "canonical_tasks"
)

SPLITDIR = (
    BASE
    / "data"
    / "processed"
    / "revision_2026"
    / "splits_and_scalers"
)

SUPERVISED = (
    CANON
    / "canonical_cycle_supervised_table.csv"
)

ASSIGNMENTS = (
    SPLITDIR
    / "cell_split_assignments.csv"
)

SCALERS = (
    SPLITDIR
    / "train_only_scaler_parameters.csv"
)

RESULT_ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "feature_matched_soh"
)


EXPECTED_HASHES = {
    "supervised":
        "49a7b7629cf9ee3f8ce0a7e3c52f2807878b752b4c9a30a876285fb891b00c14",

    "assignments":
        "914c8b90131f496ca396f63e0502147ca99848fdfd4b2c6b2a5887225b47d055",

    "scalers":
        "b9708b9321b521b87bdd05f969558134d7e0148285ef7c50f648c2028cd86de1",
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

SPLITS = [
    "train",
    "validation",
    "test",
]

BENCHMARK_CONTRACT = (
    "feature_matched_soh_causal_gcn_v1"
)


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(
        True
    )


def parameter_count(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def parameter_signature(model):
    return [
        (
            name,
            tuple(param.shape),
            int(param.numel()),
        )
        for name, param
        in model.named_parameters()
        if param.requires_grad
    ]


# =============================================================================
# Models
# =============================================================================

class FeatureMatchedMLP(nn.Module):

    def __init__(
        self,
        input_dim=20,
        hidden_dim=64,
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.out = nn.Linear(
            hidden_dim,
            1,
        )

        self.act = nn.ReLU()


    def forward(
        self,
        x,
        prev_index=None,
    ):
        h = self.act(
            self.fc1(x)
        )

        h = self.act(
            self.fc2(h)
        )

        return (
            self.out(h)
            .squeeze(-1)
        )


def causal_mean(
    h,
    prev_index,
):
    # First node of every cell points to itself:
    # (self + self) / 2 == self.
    #
    # Every other node t points only to t-1.
    # No reverse/future edge exists.
    return (
        h
        + h[
            prev_index
        ]
    ) * 0.5


class FeatureMatchedCausalGCN(nn.Module):

    def __init__(
        self,
        input_dim=20,
        hidden_dim=64,
    ):
        super().__init__()

        # Exactly the same learnable layers as the MLP.
        self.fc1 = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.out = nn.Linear(
            hidden_dim,
            1,
        )

        self.act = nn.ReLU()


    def forward(
        self,
        x,
        prev_index,
    ):
        # Layer 1:
        # directed causal neighborhood =
        # {self, previous valid observation}.
        h = causal_mean(
            x,
            prev_index,
        )

        h = self.act(
            self.fc1(h)
        )

        # Layer 2:
        # another causal propagation step.
        # Receptive field reaches at most t-2.
        h = causal_mean(
            h,
            prev_index,
        )

        h = self.act(
            self.fc2(h)
        )

        return (
            self.out(h)
            .squeeze(-1)
        )


# =============================================================================
# Architecture-level causal proof
# =============================================================================

def architecture_contract_test():
    set_seed(
        20260911
    )

    mlp = FeatureMatchedMLP()
    mlp_state = copy.deepcopy(
        mlp.state_dict()
    )

    set_seed(
        20260911
    )

    gcn = FeatureMatchedCausalGCN()
    gcn_state = copy.deepcopy(
        gcn.state_dict()
    )

    assert parameter_signature(
        mlp
    ) == parameter_signature(
        gcn
    )

    assert parameter_count(
        mlp
    ) == parameter_count(
        gcn
    )

    # Same RNG reset must generate identical learnable parameters.
    for key in mlp_state:
        assert torch.equal(
            mlp_state[key],
            gcn_state[key],
        ), key


    # -------------------------------------------------------------------------
    # Synthetic two-cell graph.
    #
    # cell A = nodes 0,1,2
    # cell B = nodes 3,4,5
    #
    # prev:
    # 0->0
    # 0->1
    # 1->2
    # 3->3
    # 3->4
    # 4->5
    # -------------------------------------------------------------------------

    model = FeatureMatchedCausalGCN(
        input_dim=2,
        hidden_dim=4,
    )

    with torch.no_grad():
        for p in model.parameters():
            p.fill_(
                0.05
            )

    x = torch.tensor(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
        ],
        dtype=torch.float32,
    )

    prev = torch.tensor(
        [
            0,
            0,
            1,
            3,
            3,
            4,
        ],
        dtype=torch.long,
    )

    base = model(
        x,
        prev,
    ).detach()


    # Perturb future node A2.
    # Earlier outputs A0/A1 must be unchanged.
    future = x.clone()

    future[2] += 100.0

    future_out = model(
        future,
        prev,
    ).detach()

    assert torch.equal(
        base[:2],
        future_out[:2],
    )


    # Perturb cell B.
    # All outputs of cell A must remain unchanged.
    other_cell = x.clone()

    other_cell[3:] += 100.0

    other_out = model(
        other_cell,
        prev,
    ).detach()

    assert torch.equal(
        base[:3],
        other_out[:3],
    )


    # Perturb past node A0.
    # A2 must respond after two causal layers.
    past = x.clone()

    past[0] += 10.0

    past_out = model(
        past,
        prev,
    ).detach()

    assert not torch.equal(
        base[2:3],
        past_out[2:3],
    )


    return {
        "parameter_count":
            parameter_count(
                mlp
            ),

        "parameter_signatures_equal":
            True,

        "initial_weights_equal":
            True,

        "future_to_past_influence":
            False,

        "cross_cell_influence":
            False,

        "two_step_past_influence":
            True,
    }


# =============================================================================
# Dataset construction
# =============================================================================

def load_canonical_soh(
    split_seed,
):
    observed = {
        "supervised":
            sha256_file(
                SUPERVISED
            ),

        "assignments":
            sha256_file(
                ASSIGNMENTS
            ),

        "scalers":
            sha256_file(
                SCALERS
            ),
    }

    assert (
        observed
        == EXPECTED_HASHES
    )


    cols = [
        "dataset",
        "cell_id",
        "dataset_cell_key",
        "split_group",
        "observation_index",
        "SOH",
        "soh_task_eligible",
    ] + FEATURES

    df = pd.read_csv(
        SUPERVISED,
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

    assert (
        df[
            "dataset"
        ].value_counts().to_dict()
        == {
            "HUST": 146122,
            "MATR": 140133,
            "CALCE": 14298,
        }
    )


    a = pd.read_csv(
        ASSIGNMENTS,
        low_memory=False,
    )

    a = a[
        a[
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


    df = df.merge(
        a,
        on="split_group",
        how="left",
        validate="many_to_one",
    )

    assert df[
        "split"
    ].notna().all()


    # Train-only pooled scaler for the requested split seed.
    p = pd.read_csv(
        SCALERS,
        low_memory=False,
    )

    p = p[
        (
            p[
                "split_seed"
            ] == split_seed
        )
        &
        (
            p[
                "scaler_scope"
            ]
            == "pooled_supervised"
        )
    ].copy()

    assert len(
        p
    ) == 20

    p = (
        p.set_index(
            "feature"
        )
        .loc[
            FEATURES
        ]
    )

    assert set(
        p[
            "fit_split"
        ].astype(str)
    ) == {
        "train_only"
    }

    mean = p[
        "mean"
    ].to_numpy(
        dtype=np.float64
    )

    scale = p[
        "scale"
    ].to_numpy(
        dtype=np.float64
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


    x = (
        df[
            FEATURES
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    assert np.isfinite(
        x
    ).all()

    x = (
        x - mean
    ) / scale

    assert np.isfinite(
        x
    ).all()

    for j, col in enumerate(
        FEATURES
    ):
        df[col] = x[
            :,
            j
        ]


    y = pd.to_numeric(
        df[
            "SOH"
        ],
        errors="coerce",
    )

    assert y.notna().all()

    assert np.isfinite(
        y.to_numpy(
            dtype=float
        )
    ).all()


    return df


def make_split_tensor(
    full_df,
    split_name,
):
    d = (
        full_df[
            full_df[
                "split"
            ] == split_name
        ]
        .copy()
    )

    d = (
        d.sort_values(
            [
                "dataset",
                "split_group",
                "observation_index",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    assert len(
        d
    ) > 0

    groups = d[
        "split_group"
    ].astype(str).to_numpy()

    n = len(d)

    prev = np.arange(
        n,
        dtype=np.int64,
    )

    if n > 1:
        same_cell = (
            groups[1:]
            == groups[:-1]
        )

        previous_row = np.arange(
            0,
            n - 1,
            dtype=np.int64,
        )

        current_row = np.arange(
            1,
            n,
            dtype=np.int64,
        )

        prev[1:] = np.where(
            same_cell,
            previous_row,
            current_row,
        )


    # Prove every non-self edge remains within one cell and moves
    # backward in observation index.
    obs = pd.to_numeric(
        d[
            "observation_index"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.int64
    )

    idx = np.arange(
        n,
        dtype=np.int64,
    )

    nonself = (
        prev != idx
    )

    assert np.all(
        groups[
            prev[nonself]
        ]
        ==
        groups[
            nonself
        ]
    )

    assert np.all(
        obs[
            prev[nonself]
        ]
        <
        obs[
            nonself
        ]
    )


    # No internal gaps in the valid sequence.
    for _, g in d.groupby(
        "split_group",
        sort=False,
    ):
        o = pd.to_numeric(
            g[
                "observation_index"
            ],
            errors="raise",
        ).to_numpy(
            dtype=np.int64
        )

        if len(o) > 1:
            assert np.all(
                np.diff(o) == 1
            )


    # -------------------------------------------------------------------------
    # Dataset-balanced + cell-balanced objective.
    #
    # Each dataset gets 1 / number_of_datasets weight.
    # Inside each dataset, each cell gets equal weight.
    # Inside each cell, each observation gets equal weight.
    # -------------------------------------------------------------------------

    w = np.zeros(
        n,
        dtype=np.float64,
    )

    datasets_here = sorted(
        d[
            "dataset"
        ].unique()
        .tolist()
    )

    assert set(datasets_here) == set(DATASETS)

    n_datasets = len(
        datasets_here
    )

    for dataset in datasets_here:

        dm = (
            d[
                "dataset"
            ] == dataset
        )

        dataset_groups = (
            d.loc[
                dm,
                "split_group",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        n_cells = len(
            dataset_groups
        )

        assert n_cells > 0

        for group in dataset_groups:

            gm = (
                d[
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
                /
                n_datasets
                /
                n_cells
                /
                n_obs
            )


    assert np.isclose(
        w.sum(),
        1.0,
        rtol=0.0,
        atol=1e-12,
    )


    x = torch.tensor(
        d[
            FEATURES
        ].to_numpy(
            dtype=np.float32
        ),
        dtype=torch.float32,
    )

    y = torch.tensor(
        pd.to_numeric(
            d[
                "SOH"
            ],
            errors="raise",
        ).to_numpy(
            dtype=np.float32
        ),
        dtype=torch.float32,
    )

    prev_t = torch.tensor(
        prev,
        dtype=torch.long,
    )

    w_t = torch.tensor(
        w.astype(
            np.float32
        ),
        dtype=torch.float32,
    )


    return {
        "frame":
            d,

        "x":
            x,

        "y":
            y,

        "prev":
            prev_t,

        "weight":
            w_t,

        "n_rows":
            n,

        "n_cells":
            d[
                "split_group"
            ].nunique(),
    }


# =============================================================================
# Metrics
# =============================================================================

def weighted_metrics(
    pred,
    target,
    weight,
):
    err = (
        pred - target
    )

    mse = torch.sum(
        weight
        * err.square()
    )

    mae = torch.sum(
        weight
        * err.abs()
    )

    rmse = torch.sqrt(
        mse
    )

    return {
        "rmse":
            float(
                rmse.item()
            ),

        "mae":
            float(
                mae.item()
            ),

        "mse":
            float(
                mse.item()
            ),
    }


def dataset_metrics(
    pred,
    target,
    frame,
):
    pred = (
        pred.detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    target = (
        target.detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    rows = []

    for dataset in DATASETS:

        mask = (
            frame[
                "dataset"
            ].to_numpy()
            == dataset
        )

        p = pred[
            mask
        ]

        y = target[
            mask
        ]

        err = (
            p - y
        )

        mse = float(
            np.mean(
                err ** 2
            )
        )

        rmse = math.sqrt(
            mse
        )

        mae = float(
            np.mean(
                np.abs(
                    err
                )
            )
        )

        denom = float(
            np.sum(
                (
                    y
                    - np.mean(y)
                ) ** 2
            )
        )

        r2 = (
            1.0
            - float(
                np.sum(
                    err ** 2
                )
            )
            / denom
            if denom > 0
            else np.nan
        )

        rows.append({
            "dataset":
                dataset,

            "n":
                int(
                    len(y)
                ),

            "RMSE":
                rmse,

            "MAE":
                mae,

            "R2":
                r2,
        })

    return rows


# =============================================================================
# Training
# =============================================================================

def train_one(
    model_name,
    data,
    train_seed,
    hidden_dim,
    lr,
    weight_decay,
    max_epochs,
    patience,
):
    set_seed(
        train_seed
    )

    if model_name == "MLP":
        model = FeatureMatchedMLP(
            input_dim=len(
                FEATURES
            ),
            hidden_dim=hidden_dim,
        )

    elif model_name == "CAUSAL_GCN":
        model = FeatureMatchedCausalGCN(
            input_dim=len(
                FEATURES
            ),
            hidden_dim=hidden_dim,
        )

    else:
        raise ValueError(
            model_name
        )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    train = data[
        "train"
    ]

    val = data[
        "validation"
    ]


    best_state = None
    best_val = np.inf
    best_epoch = 0
    wait = 0

    history = []


    start = time.time()


    for epoch in range(
        1,
        max_epochs + 1,
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

            vm = weighted_metrics(
                val_pred,
                val["y"],
                val["weight"],
            )


        val_rmse = (
            vm["rmse"]
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
            best_val = (
                val_rmse
            )

            best_epoch = (
                epoch
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            wait = 0

        else:
            wait += 1


        if wait >= patience:
            break


    assert (
        best_state is not None
    )

    model.load_state_dict(
        best_state
    )

    elapsed = (
        time.time()
        - start
    )


    metrics = []

    model.eval()

    with torch.no_grad():

        for split_name in SPLITS:

            d = data[
                split_name
            ]

            pred = model(
                d["x"],
                d["prev"],
            )

            balanced = weighted_metrics(
                pred,
                d["y"],
                d["weight"],
            )

            metrics.append({
                "model":
                    model_name,

                "split":
                    split_name,

                "dataset":
                    "BALANCED_ALL",

                "n":
                    d["n_rows"],

                "RMSE":
                    balanced[
                        "rmse"
                    ],

                "MAE":
                    balanced[
                        "mae"
                    ],

                "R2":
                    np.nan,
            })


            per_dataset = dataset_metrics(
                pred,
                d["y"],
                d["frame"],
            )

            for row in per_dataset:
                row.update({
                    "model":
                        model_name,

                    "split":
                        split_name,
                })

                metrics.append(
                    row
                )


    return {
        "model":
            model,

        "history":
            pd.DataFrame(
                history
            ),

        "metrics":
            pd.DataFrame(
                metrics
            ),

        "best_epoch":
            best_epoch,

        "best_validation_balanced_rmse":
            best_val,

        "elapsed_seconds":
            elapsed,

        "parameter_count":
            parameter_count(
                model
            ),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--train-seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()


    assert args.split_seed in {
        0,
        1,
        2,
        3,
        4,
    }

    assert args.hidden_dim > 0
    assert args.max_epochs > 0
    assert args.patience > 0


    torch.set_num_threads(
        args.threads
    )

    try:
        torch.set_num_interop_threads(
            1
        )
    except RuntimeError:
        pass


    assert (
        torch.version.cuda
        is None
    )

    assert (
        torch.cuda.is_available()
        is False
    )


    architecture = (
        architecture_contract_test()
    )


    df = load_canonical_soh(
        args.split_seed
    )


    data = {
        split:
            make_split_tensor(
                df,
                split,
            )
        for split in SPLITS
    }


    if args.smoke:
        max_epochs = min(
            args.max_epochs,
            2,
        )

        patience = min(
            args.patience,
            2,
        )

        run_name = (
            f"smoke_split{args.split_seed}"
            f"_train{args.train_seed}"
        )

    else:
        max_epochs = (
            args.max_epochs
        )

        patience = (
            args.patience
        )

        run_name = (
            f"split{args.split_seed}"
            f"_train{args.train_seed}"
        )


    outdir = (
        RESULT_ROOT
        / run_name
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )


    results = {}


    for model_name in (
        "MLP",
        "CAUSAL_GCN",
    ):
        results[
            model_name
        ] = train_one(
            model_name=model_name,
            data=data,
            train_seed=args.train_seed,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=max_epochs,
            patience=patience,
        )


    assert (
        results[
            "MLP"
        ][
            "parameter_count"
        ]
        ==
        results[
            "CAUSAL_GCN"
        ][
            "parameter_count"
        ]
    )

    assert (
        results[
            "MLP"
        ][
            "parameter_count"
        ]
        ==
        architecture[
            "parameter_count"
        ]
    )


    metrics = pd.concat(
        [
            results[
                "MLP"
            ][
                "metrics"
            ],

            results[
                "CAUSAL_GCN"
            ][
                "metrics"
            ],
        ],
        ignore_index=True,
    )

    metrics[
        "split_seed"
    ] = args.split_seed

    metrics[
        "train_seed"
    ] = args.train_seed

    metrics[
        "benchmark_contract"
    ] = BENCHMARK_CONTRACT


    metrics.to_csv(
        outdir
        / "metrics.csv",
        index=False,
        float_format="%.12g",
    )


    for model_name in (
        "MLP",
        "CAUSAL_GCN",
    ):
        results[
            model_name
        ][
            "history"
        ].to_csv(
            outdir
            / (
                model_name.lower()
                + "_history.csv"
            ),
            index=False,
            float_format="%.12g",
        )

        torch.save(
            results[
                model_name
            ][
                "model"
            ].state_dict(),
            outdir
            / (
                model_name.lower()
                + "_best_state.pt"
            ),
        )


    config = {
        "benchmark_contract":
            BENCHMARK_CONTRACT,

        "smoke":
            bool(
                args.smoke
            ),

        "task":
            "cycle-level SOH regression",

        "datasets":
            DATASETS,

        "feature_count":
            len(
                FEATURES
            ),

        "features":
            FEATURES,

        "eligible_observations":
            300553,

        "split_seed":
            args.split_seed,

        "train_seed":
            args.train_seed,

        "hidden_dim":
            args.hidden_dim,

        "architecture":
            "20 -> hidden -> hidden -> 1",

        "activation":
            "ReLU",

        "dropout":
            False,

        "batch_normalization":
            False,

        "parameter_count_each_model":
            architecture[
                "parameter_count"
            ],

        "parameters_equal":
            True,

        "initial_weights_equal_for_same_train_seed":
            True,

        "gcn_aggregation":
            (
                "row-normalized causal mean over "
                "{self, previous valid observation}"
            ),

        "gcn_layers_with_aggregation":
            2,

        "maximum_temporal_receptive_field":
            "current plus two previous valid observations",

        "future_edges":
            False,

        "reverse_edges":
            False,

        "cross_cell_edges":
            False,

        "cycle_index_feature":
            False,

        "target_feature":
            False,

        "feature_scaler":
            (
                "seed-specific pooled_supervised "
                "train-only StandardScaler parameters"
            ),

        "loss":
            (
                "dataset-balanced and cell-balanced MSE"
            ),

        "optimizer":
            "AdamW",

        "learning_rate":
            args.lr,

        "weight_decay":
            args.weight_decay,

        "max_epochs":
            max_epochs,

        "patience":
            patience,

        "early_stopping_metric":
            (
                "validation dataset-balanced "
                "cell-balanced RMSE"
            ),

        "device":
            "cpu",

        "torch_version":
            torch.__version__,

        "torch_cuda_build":
            torch.version.cuda,

        "python_version":
            platform.python_version(),

        "input_sha256":
            EXPECTED_HASHES,

        "architecture_contract_test":
            architecture,

        "split_rows": {
            split:
                data[
                    split
                ][
                    "n_rows"
                ]
            for split in SPLITS
        },

        "split_cells": {
            split:
                data[
                    split
                ][
                    "n_cells"
                ]
            for split in SPLITS
        },

        "model_training": {
            name: {
                "best_epoch":
                    results[
                        name
                    ][
                        "best_epoch"
                    ],

                "best_validation_balanced_rmse":
                    results[
                        name
                    ][
                        "best_validation_balanced_rmse"
                    ],

                "elapsed_seconds":
                    results[
                        name
                    ][
                        "elapsed_seconds"
                    ],
            }
            for name in (
                "MLP",
                "CAUSAL_GCN",
            )
        },
    }


    (
        outdir
        / "config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


    print("=" * 80)
    print("FEATURE-MATCHED SOH BENCHMARK")
    print("=" * 80)

    print(
        "mode:",
        "SMOKE"
        if args.smoke
        else "FULL",
    )

    print(
        "device: cpu"
    )

    print(
        "torch:",
        torch.__version__,
    )

    print(
        "split_seed / train_seed:",
        args.split_seed,
        "/",
        args.train_seed,
    )

    print(
        "feature count:",
        len(FEATURES),
    )

    print(
        "eligible observations:",
        len(df),
    )

    print(
        "split rows:",
        {
            split:
                data[
                    split
                ][
                    "n_rows"
                ]
            for split in SPLITS
        },
    )

    print(
        "split cells:",
        {
            split:
                data[
                    split
                ][
                    "n_cells"
                ]
            for split in SPLITS
        },
    )

    print(
        "MLP parameters:",
        results[
            "MLP"
        ][
            "parameter_count"
        ],
    )

    print(
        "CAUSAL_GCN parameters:",
        results[
            "CAUSAL_GCN"
        ][
            "parameter_count"
        ],
    )

    print(
        "parameter signatures equal:",
        architecture[
            "parameter_signatures_equal"
        ],
    )

    print(
        "same-seed initial weights equal:",
        architecture[
            "initial_weights_equal"
        ],
    )

    print(
        "future-to-past influence:",
        architecture[
            "future_to_past_influence"
        ],
    )

    print(
        "cross-cell influence:",
        architecture[
            "cross_cell_influence"
        ],
    )

    print(
        "two-step past influence:",
        architecture[
            "two_step_past_influence"
        ],
    )

    print(
        "dropout:",
        False,
    )

    print(
        "batch normalization:",
        False,
    )

    print(
        "cycle index feature:",
        False,
    )

    print(
        "target-derived feature:",
        False,
    )

    print(
        "loss:",
        "dataset-balanced + cell-balanced MSE",
    )

    print(
        "scaler:",
        "pooled train-only",
    )


    for model_name in (
        "MLP",
        "CAUSAL_GCN",
    ):
        r = results[
            model_name
        ]

        print()
        print(
            model_name,
            "best_epoch:",
            r[
                "best_epoch"
            ],
        )

        print(
            model_name,
            "best_val_balanced_RMSE:",
            f"{r['best_validation_balanced_rmse']:.8f}",
        )

        print(
            model_name,
            "elapsed_seconds:",
            f"{r['elapsed_seconds']:.2f}",
        )


    test_balanced = metrics[
        (
            metrics[
                "split"
            ] == "test"
        )
        &
        (
            metrics[
                "dataset"
            ] == "BALANCED_ALL"
        )
    ][
        [
            "model",
            "RMSE",
            "MAE",
        ]
    ]

    print()
    print(
        "SMOKE test balanced metrics:",
        test_balanced.to_dict(
            orient="records"
        ),
    )


    print()
    print(
        "metrics SHA256:",
        sha256_file(
            outdir
            / "metrics.csv"
        ),
    )

    print(
        "config SHA256:",
        sha256_file(
            outdir
            / "config.json"
        ),
    )

    print()
    print(
        "FEATURE_MATCHED_SOH_BENCHMARK_PASS=True"
    )


if __name__ == "__main__":
    main()
