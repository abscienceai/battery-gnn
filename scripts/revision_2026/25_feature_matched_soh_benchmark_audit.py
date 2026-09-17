from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import os

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

SUP = (
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

SMOKE = (
    BASE
    / "results"
    / "revision_2026"
    / "feature_matched_soh"
    / "smoke_split0_train0"
)

METRICS = (
    SMOKE
    / "metrics.csv"
)

CONFIG = (
    SMOKE
    / "config.json"
)


EXPECTED = {
    "benchmark_script":
        "5d621f8d5764e417e41bc518ec72b9608566a087c552b8365d438c2344c06c5e",

    "supervised":
        "49a7b7629cf9ee3f8ce0a7e3c52f2807878b752b4c9a30a876285fb891b00c14",

    "assignments":
        "914c8b90131f496ca396f63e0502147ca99848fdfd4b2c6b2a5887225b47d055",

    "scalers":
        "b9708b9321b521b87bdd05f969558134d7e0148285ef7c50f648c2028cd86de1",

    "smoke_metrics":
        "fe97b29316c6cfe9ae0db27aa9084e6c0632c67a94d8809d44471fb84901db15",
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


def param_signature(model):
    return [
        (
            name,
            tuple(p.shape),
            p.numel(),
        )
        for name, p
        in model.named_parameters()
        if p.requires_grad
    ]


def param_count(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


print("=" * 80)
print("FEATURE-MATCHED SOH BENCHMARK INDEPENDENT AUDIT")
print("=" * 80)


# =============================================================================
# 1. Immutable source / data / smoke guards
# =============================================================================

assert sha256_file(
    BENCH
) == EXPECTED[
    "benchmark_script"
]

assert sha256_file(
    SUP
) == EXPECTED[
    "supervised"
]

assert sha256_file(
    ASSIGN
) == EXPECTED[
    "assignments"
]

assert sha256_file(
    SCALERS
) == EXPECTED[
    "scalers"
]

assert sha256_file(
    METRICS
) == EXPECTED[
    "smoke_metrics"
]

print(
    "source/data/artifact hashes exact: True"
)


# =============================================================================
# 2. Static source guards
# =============================================================================

source = BENCH.read_text(
    encoding="utf-8"
)

assert "torch_geometric" not in source
assert "GCNConv" not in source
assert "to_undirected" not in source

assert (
    '"cycle index is NOT a model feature"'
    not in source
)

assert (
    '"cycle_index_feature":\n            False'
    in source
)

assert (
    '"target_feature":\n            False'
    in source
)

assert (
    "h\n        + h[\n            prev_index\n        ]"
    in source
)

print(
    "PyG dependency in benchmark: False"
)

print(
    "reverse-edge helper in benchmark: False"
)


# =============================================================================
# 3. Import without executing main
# =============================================================================

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


assert mod.FEATURES == FEATURES
assert mod.DATASETS == DATASETS

assert (
    mod.BENCHMARK_CONTRACT
    == "feature_matched_soh_causal_gcn_v1"
)


# =============================================================================
# 4. Runtime remains CPU only
# =============================================================================

assert (
    torch.__version__.split("+")[0]
    == "2.14.0"
)

assert torch.version.cuda is None

assert (
    torch.cuda.is_available()
    is False
)

print(
    "torch CPU-only contract: True"
)


# =============================================================================
# 5. Architecture equality
# =============================================================================

mod.set_seed(
    20260911
)

mlp = mod.FeatureMatchedMLP(
    input_dim=20,
    hidden_dim=64,
)

mlp_state = copy.deepcopy(
    mlp.state_dict()
)


mod.set_seed(
    20260911
)

gcn = mod.FeatureMatchedCausalGCN(
    input_dim=20,
    hidden_dim=64,
)

gcn_state = copy.deepcopy(
    gcn.state_dict()
)


assert param_count(
    mlp
) == 5569

assert param_count(
    gcn
) == 5569

assert param_signature(
    mlp
) == param_signature(
    gcn
)

for name in mlp_state:
    assert torch.equal(
        mlp_state[name],
        gcn_state[name],
    )


print(
    "MLP parameters:",
    param_count(
        mlp
    )
)

print(
    "CAUSAL_GCN parameters:",
    param_count(
        gcn
    )
)

print(
    "learnable parameter signatures equal: True"
)

print(
    "same-seed initial parameters exact: True"
)


# =============================================================================
# 6. Independent causal-behaviour test
# =============================================================================

m = mod.FeatureMatchedCausalGCN(
    input_dim=2,
    hidden_dim=4,
)

with torch.no_grad():
    for p in m.parameters():
        p.fill_(0.05)


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

# Two independent cells:
# A = 0,1,2
# B = 3,4,5
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


base = m(
    x,
    prev,
).detach()


future = x.clone()
future[2] += 100.0

future_out = m(
    future,
    prev,
).detach()

assert torch.equal(
    base[:2],
    future_out[:2],
)


other_cell = x.clone()
other_cell[3:] += 100.0

other_out = m(
    other_cell,
    prev,
).detach()

assert torch.equal(
    base[:3],
    other_out[:3],
)


past = x.clone()
past[0] += 10.0

past_out = m(
    past,
    prev,
).detach()

assert not torch.equal(
    base[2:3],
    past_out[2:3],
)


# First node self-loop must leave its raw representation
# unchanged under causal_mean.
raw = torch.tensor(
    [
        [2.0, 4.0],
        [8.0, 10.0],
    ]
)

self_prev = torch.tensor(
    [
        0,
        0,
    ]
)

agg = mod.causal_mean(
    raw,
    self_prev,
)

assert torch.equal(
    agg[0],
    raw[0],
)

assert torch.equal(
    agg[1],
    (
        raw[1]
        + raw[0]
    ) / 2.0,
)


print(
    "future-to-past influence: False"
)

print(
    "cross-cell influence: False"
)

print(
    "two-layer t-2 past influence: True"
)

print(
    "first-node self-loop semantics exact: True"
)


# =============================================================================
# 7. Independently reconstruct canonical SOH population
# =============================================================================

cols = [
    "dataset",
    "cell_id",
    "split_group",
    "observation_index",
    "SOH",
    "soh_task_eligible",
] + FEATURES

df = pd.read_csv(
    SUP,
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


assert len(df) == 300553

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


x_all = (
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
    x_all
).all()


print(
    "canonical SOH observations:",
    len(df)
)

print(
    "all 20 raw features finite: True"
)


# =============================================================================
# 8. Sequence continuity
# =============================================================================

first_nonzero = 0
internal_gap_cells = 0


for _, g in df.groupby(
    "split_group",
    sort=True,
):
    obs = (
        pd.to_numeric(
            g[
                "observation_index"
            ],
            errors="raise",
        )
        .sort_values()
        .to_numpy(
            dtype=np.int64
        )
    )

    if obs[0] > 0:
        first_nonzero += 1

    if (
        len(obs) > 1
        and np.any(
            np.diff(obs) != 1
        )
    ):
        internal_gap_cells += 1


assert first_nonzero == 41
assert internal_gap_cells == 0


print(
    "cells beginning after observation 0:",
    first_nonzero
)

print(
    "cells with internal valid-sequence gaps:",
    internal_gap_cells
)


# =============================================================================
# 9. Seed-0 split reconstruction + directed-edge proof
# =============================================================================

a = pd.read_csv(
    ASSIGN,
    low_memory=False,
)

a = a[
    a[
        "split_seed"
    ] == 0
][
    [
        "split_group",
        "split",
    ]
].copy()

assert len(a) == 307


z = df.merge(
    a,
    on="split_group",
    how="left",
    validate="many_to_one",
)

assert z[
    "split"
].notna().all()


expected_rows = {
    "train": 211243,
    "validation": 45336,
    "test": 43974,
}

expected_cells = {
    "train": 182,
    "validation": 39,
    "test": 38,
}


edge_report = {}


for split in (
    "train",
    "validation",
    "test",
):

    d = (
        z[
            z[
                "split"
            ] == split
        ]
        .sort_values(
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
    ) == expected_rows[
        split
    ]

    assert d[
        "split_group"
    ].nunique() == expected_cells[
        split
    ]


    groups = (
        d[
            "split_group"
        ]
        .astype(str)
        .to_numpy()
    )

    obs = pd.to_numeric(
        d[
            "observation_index"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.int64
    )

    n = len(d)

    prev = np.arange(
        n,
        dtype=np.int64,
    )

    if n > 1:
        same = (
            groups[1:]
            == groups[:-1]
        )

        prev[1:] = np.where(
            same,
            np.arange(
                n - 1,
                dtype=np.int64,
            ),
            np.arange(
                1,
                n,
                dtype=np.int64,
            ),
        )


    current = np.arange(
        n,
        dtype=np.int64,
    )

    nonself = (
        prev != current
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

    expected_nonself = (
        n
        - expected_cells[
            split
        ]
    )

    assert int(
        nonself.sum()
    ) == expected_nonself


    edge_report[
        split
    ] = {
        "rows":
            n,

        "cells":
            expected_cells[
                split
            ],

        "causal_previous_edges":
            int(
                nonself.sum()
            ),

        "self_loop_first_nodes":
            expected_cells[
                split
            ],
    }


print(
    "seed0 split rows exact:",
    expected_rows
)

print(
    "seed0 split cells exact:",
    expected_cells
)

print(
    "seed0 causal edge accounting:",
    edge_report
)

print(
    "future edges:",
    0
)

print(
    "cross-cell edges:",
    0
)


# =============================================================================
# 10. Scaler independently verified for benchmark seed 0
# =============================================================================

p = pd.read_csv(
    SCALERS,
    low_memory=False,
)

p = p[
    (
        p[
            "split_seed"
        ] == 0
    )
    &
    (
        p[
            "scaler_scope"
        ]
        == "pooled_supervised"
    )
].copy()

assert len(p) == 20

assert set(
    p[
        "fit_split"
    ].astype(str)
) == {
    "train_only"
}

p = (
    p.set_index(
        "feature"
    )
    .loc[
        FEATURES
    ]
)


train = z[
    z[
        "split"
    ] == "train"
].copy()

train_x = (
    train[
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

mean = np.mean(
    train_x,
    axis=0,
    dtype=np.float64,
)

var = np.var(
    train_x,
    axis=0,
    ddof=0,
    dtype=np.float64,
)

scale = np.sqrt(
    var
)

scale[
    scale == 0
] = 1.0


assert np.allclose(
    mean,
    p[
        "mean"
    ].to_numpy(
        dtype=float
    ),
    rtol=5e-15,
    atol=5e-15,
)

assert np.allclose(
    var,
    p[
        "variance"
    ].to_numpy(
        dtype=float
    ),
    rtol=5e-15,
    atol=5e-15,
)

assert np.allclose(
    scale,
    p[
        "scale"
    ].to_numpy(
        dtype=float
    ),
    rtol=5e-15,
    atol=5e-15,
)


print(
    "seed0 pooled scaler independently reproduced: 20 / 20"
)

print(
    "validation/test statistics in scaler fit:",
    0
)


# =============================================================================
# 11. Independent objective-weight audit
# =============================================================================

max_dataset_delta = 0.0
max_cell_delta = 0.0


for split in (
    "train",
    "validation",
    "test",
):

    d = (
        z[
            z[
                "split"
            ] == split
        ]
        .copy()
    )

    w = np.zeros(
        len(d),
        dtype=np.float64,
    )

    d = d.reset_index(
        drop=True
    )


    for dataset in DATASETS:

        dm = (
            d[
                "dataset"
            ] == dataset
        )

        cells = (
            d.loc[
                dm,
                "split_group",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        n_cells = len(
            cells
        )

        assert n_cells > 0


        for cell in cells:

            cm = (
                d[
                    "split_group"
                ].astype(str)
                == cell
            )

            n_obs = int(
                cm.sum()
            )

            w[
                cm.to_numpy()
            ] = (
                1.0
                / 3.0
                / n_cells
                / n_obs
            )


        dataset_weight = float(
            w[
                dm.to_numpy()
            ].sum()
        )

        max_dataset_delta = max(
            max_dataset_delta,
            abs(
                dataset_weight
                - 1.0 / 3.0
            ),
        )


        target_cell_weight = (
            1.0
            / 3.0
            / n_cells
        )


        for cell in cells:

            cm = (
                d[
                    "split_group"
                ].astype(str)
                == cell
            ).to_numpy()

            cell_weight = float(
                w[
                    cm
                ].sum()
            )

            max_cell_delta = max(
                max_cell_delta,
                abs(
                    cell_weight
                    - target_cell_weight
                ),
            )


    assert np.isclose(
        w.sum(),
        1.0,
        rtol=0.0,
        atol=1e-12,
    )


assert (
    max_dataset_delta
    < 1e-12
)

assert (
    max_cell_delta
    < 1e-12
)


print(
    "dataset-balanced objective exact: True"
)

print(
    "cell-balanced objective exact: True"
)

print(
    "max dataset-weight delta:",
    f"{max_dataset_delta:.3e}"
)

print(
    "max cell-weight delta:",
    f"{max_cell_delta:.3e}"
)


# =============================================================================
# 12. Existing smoke artifact semantic audit
# =============================================================================

metrics = pd.read_csv(
    METRICS,
    low_memory=False,
)

config = json.loads(
    CONFIG.read_text(
        encoding="utf-8"
    )
)


assert len(
    metrics
) == 24

assert set(
    metrics[
        "model"
    ]
) == {
    "MLP",
    "CAUSAL_GCN",
}

assert set(
    metrics[
        "split"
    ]
) == {
    "train",
    "validation",
    "test",
}

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

assert config[
    "smoke"
] is True

assert config[
    "eligible_observations"
] == 300553

assert config[
    "feature_count"
] == 20

assert config[
    "parameter_count_each_model"
] == 5569

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

assert config[
    "device"
] == "cpu"

assert config[
    "split_rows"
] == expected_rows

assert config[
    "split_cells"
] == expected_cells


print(
    "existing smoke metrics rows:",
    len(metrics)
)

print(
    "existing smoke config contract exact: True"
)


# =============================================================================
# 13. In-memory deterministic training reproduction
#
# Does NOT overwrite smoke outputs.
# =============================================================================

canonical = mod.load_canonical_soh(
    split_seed=0
)

data = {
    split:
        mod.make_split_tensor(
            canonical,
            split,
        )
    for split in (
        "train",
        "validation",
        "test",
    )
}


def deterministic_pair(
    model_name,
):
    r1 = mod.train_one(
        model_name=model_name,
        data=data,
        train_seed=0,
        hidden_dim=64,
        lr=1e-3,
        weight_decay=1e-5,
        max_epochs=2,
        patience=2,
    )

    r2 = mod.train_one(
        model_name=model_name,
        data=data,
        train_seed=0,
        hidden_dim=64,
        lr=1e-3,
        weight_decay=1e-5,
        max_epochs=2,
        patience=2,
    )


    h1 = r1[
        "history"
    ].drop(
        columns=[],
        errors="ignore",
    )

    h2 = r2[
        "history"
    ].drop(
        columns=[],
        errors="ignore",
    )

    pd.testing.assert_frame_equal(
        h1,
        h2,
        check_exact=True,
    )


    m1 = (
        r1[
            "metrics"
        ]
        .sort_values(
            [
                "split",
                "dataset",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    m2 = (
        r2[
            "metrics"
        ]
        .sort_values(
            [
                "split",
                "dataset",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    pd.testing.assert_frame_equal(
        m1,
        m2,
        check_exact=True,
    )


    s1 = r1[
        "model"
    ].state_dict()

    s2 = r2[
        "model"
    ].state_dict()

    assert set(
        s1.keys()
    ) == set(
        s2.keys()
    )

    for key in s1:
        assert torch.equal(
            s1[key],
            s2[key],
        ), (
            model_name,
            key,
        )


    return {
        "best_epoch":
            r1[
                "best_epoch"
            ],

        "best_val":
            r1[
                "best_validation_balanced_rmse"
            ],
    }


mlp_det = deterministic_pair(
    "MLP"
)

gcn_det = deterministic_pair(
    "CAUSAL_GCN"
)


print(
    "MLP two-run deterministic history/metrics/state: True"
)

print(
    "CAUSAL_GCN two-run deterministic history/metrics/state: True"
)


# =============================================================================
# Final
# =============================================================================

print()
print(
    "feature count:",
    20
)

print(
    "feature-matched observations:",
    300553
)

print(
    "parameter count each:",
    5569
)

print(
    "causal receptive field:",
    "t, t-1, t-2"
)

print(
    "GPU used by benchmark:",
    False
)

print()
print(
    "FEATURE_MATCHED_SOH_BENCHMARK_AUDIT_PASS=True"
)
