from pathlib import Path
import hashlib
import importlib.util
import os

import numpy as np
import pandas as pd
import torch


BASE = Path.home() / "alper" / "battery_revision_2026"

REPO = (
    BASE
    / "repo"
    / "battery-gnn"
)

TCN = (
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

EXPECTED_CANON_SHA = (
    "49a7b7629cf9ee3f8ce0a7e3c52f2807878b752b4c9a30a876285fb891b00c14"
)

FEATURES = [
    "v_min","v_q10","v_q25","v_median","v_q75","v_q90",
    "v_max","v_mean","v_std","v_phase_slope",
    "i_min","i_q10","i_q25","i_median","i_q75","i_q90",
    "i_max","i_mean","i_std","i_phase_slope",
]


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def as_bool(s):
    return (
        s.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true","1","1.0","yes"])
    )


assert sha256_file(
    CANON
) == EXPECTED_CANON_SHA


spec = importlib.util.spec_from_file_location(
    "causal_tcn",
    TCN,
)

mod = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    mod
)


assert mod.TCN_CONTRACT == (
    "causal_tcn_three_cycle_v1"
)


print("=" * 80)
print("CAUSAL THREE-CYCLE TCN INDEPENDENT AUDIT")
print("=" * 80)


# =============================================================================
# 1. Exact sequence construction
# =============================================================================

x = np.arange(
    6 * 20,
    dtype=np.float64,
).reshape(
    6,
    20,
)


seq = mod.build_causal_three_cycle_sequences(
    x
)


assert seq.shape == (
    6,
    3,
    20,
)


assert np.array_equal(
    seq[0],
    np.stack(
        [
            x[0],
            x[0],
            x[0],
        ]
    ),
)


assert np.array_equal(
    seq[1],
    np.stack(
        [
            x[0],
            x[0],
            x[1],
        ]
    ),
)


assert np.array_equal(
    seq[2],
    np.stack(
        [
            x[0],
            x[1],
            x[2],
        ]
    ),
)


assert np.array_equal(
    seq[5],
    np.stack(
        [
            x[3],
            x[4],
            x[5],
        ]
    ),
)


print(
    "sequence boundary construction exact:",
    True
)

print(
    "sequence receptive field:",
    "{t,t-1,t-2}"
)


# =============================================================================
# 2. Future perturbation test
# =============================================================================

base = mod.build_causal_three_cycle_sequences(
    x
)

changed = x.copy()

changed[5] += 1e9

perturbed = mod.build_causal_three_cycle_sequences(
    changed
)


# Future t=5 cannot affect any t<=4 sequence.
assert np.array_equal(
    base[:5],
    perturbed[:5],
)


print(
    "future-to-past sequence influence:",
    False
)


# =============================================================================
# 3. Cross-cell isolation
# =============================================================================

cell_a = x[:4].copy()

cell_b = (
    x[:4].copy()
    + 100000.0
)


a0 = mod.build_causal_three_cycle_sequences(
    cell_a
)

_ = mod.build_causal_three_cycle_sequences(
    cell_b
)

a1 = mod.build_causal_three_cycle_sequences(
    cell_a
)


assert np.array_equal(
    a0,
    a1,
)


print(
    "cross-cell sequence influence:",
    False
)


# =============================================================================
# 4. Architecture / parameter count
# =============================================================================

torch.manual_seed(0)

model = mod.CausalThreeCycleTCN()


count = mod.parameter_count(
    model
)


assert count == 5489


expected_shapes = {
    "temporal.weight":
        (49, 20, 3),

    "temporal.bias":
        (49,),

    "pointwise.weight":
        (49, 49, 1),

    "pointwise.bias":
        (49,),

    "out.weight":
        (1, 49),

    "out.bias":
        (1,),
}


state = model.state_dict()


assert set(
    state.keys()
) == set(
    expected_shapes.keys()
)


for key, shape in expected_shapes.items():

    assert tuple(
        state[key].shape
    ) == shape

    assert torch.isfinite(
        state[key]
    ).all()


print(
    "TCN learnable parameters:",
    count
)

print(
    "MLP/GCN reference parameters:",
    5569
)

print(
    "parameter difference TCN-reference:",
    count - 5569
)

print(
    "TCN parameter ratio vs reference:",
    f"{count / 5569:.6f}"
)


# =============================================================================
# 5. Forward/autograd smoke
# =============================================================================

tensor = torch.tensor(
    seq.astype(
        np.float32
    ),
    dtype=torch.float32,
    requires_grad=True,
)


pred = model(
    tensor
)


assert pred.shape == (
    6,
)

assert torch.isfinite(
    pred
).all()


loss = pred.square().mean()

loss.backward()


assert tensor.grad is not None

assert torch.isfinite(
    tensor.grad
).all()


for p in model.parameters():

    assert p.grad is not None

    assert torch.isfinite(
        p.grad
    ).all()


print(
    "forward finite:",
    True
)

print(
    "autograd finite:",
    True
)


# =============================================================================
# 6. MODEL-level strict causality
#
# Construct all sequence tensors, evaluate, then perturb future x[5].
# Predictions t<=4 must remain unchanged because their sequence inputs
# are exactly unchanged.
# =============================================================================

model.eval()


with torch.no_grad():

    p0 = model(
        torch.tensor(
            base.astype(
                np.float32
            )
        )
    )


    p1 = model(
        torch.tensor(
            perturbed.astype(
                np.float32
            )
        )
    )


assert torch.equal(
    p0[:5],
    p1[:5],
)


print(
    "future-to-past model influence:",
    False
)


# =============================================================================
# 7. Canonical population audit
# =============================================================================

cols = [
    "split_group",
    "observation_index",
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


assert df[
    "split_group"
].nunique() == 259


rows = 0
cells = 0
first_nonzero = 0
internal_gaps = 0


for _, g in df.groupby(
    "split_group",
    sort=True,
):

    g = g.sort_values(
        "observation_index"
    )


    obs = pd.to_numeric(
        g[
            "observation_index"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.int64
    )


    if obs[0] > 0:
        first_nonzero += 1


    if (
        len(obs) > 1
        and np.any(
            np.diff(obs) != 1
        )
    ):
        internal_gaps += 1


    arr = (
        g[
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


    s = mod.build_causal_three_cycle_sequences(
        arr
    )


    assert s.shape == (
        len(arr),
        3,
        20,
    )


    assert np.isfinite(
        s
    ).all()


    # First cycle strictly self-replicated.
    assert np.array_equal(
        s[0,0],
        arr[0],
    )

    assert np.array_equal(
        s[0,1],
        arr[0],
    )

    assert np.array_equal(
        s[0,2],
        arr[0],
    )


    rows += len(arr)
    cells += 1


assert rows == 300553
assert cells == 259
assert first_nonzero == 41
assert internal_gaps == 0


print(
    "canonical sequence rows:",
    rows
)

print(
    "canonical cells:",
    cells
)

print(
    "cells beginning after observation 0:",
    first_nonzero
)

print(
    "cells with internal gaps:",
    internal_gaps
)

print(
    "all canonical sequences finite:",
    True
)


# =============================================================================
# 8. Contract statements
# =============================================================================

print(
    "future observations used:",
    False
)

print(
    "cross-cell history used:",
    False
)

print(
    "cycle index used as feature:",
    False
)

print(
    "target used as feature:",
    False
)

print(
    "sequence length:",
    3
)

print(
    "causal replicate-padding:",
    True
)

print(
    "TCN test metrics evaluated:",
    False
)

print()
print(
    "CAUSAL_TCN_SEQUENCE_MODEL_AUDIT_PASS=True"
)
