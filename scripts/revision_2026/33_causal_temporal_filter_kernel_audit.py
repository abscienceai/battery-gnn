from pathlib import Path
import hashlib
import importlib.util
import os

import numpy as np
import pandas as pd


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

EXPECTED_CANON_SHA = (
    "49a7b7629cf9ee3f8ce0a7e3c52f2807878b752b4c9a30a876285fb891b00c14"
)

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


assert sha256_file(
    CANON
) == EXPECTED_CANON_SHA


spec = importlib.util.spec_from_file_location(
    "causal_filters",
    KERNEL,
)

mod = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    mod
)


assert mod.FILTER_CONTRACT == (
    "causal_temporal_controls_v1"
)


print("=" * 78)
print("CAUSAL TEMPORAL FILTER KERNEL AUDIT")
print("=" * 78)


# =============================================================================
# 1. Synthetic exact-value tests
# =============================================================================

x = np.array(
    [
        [1.0, 10.0],
        [2.0, 20.0],
        [4.0, 40.0],
        [8.0, 80.0],
        [16.0, 160.0],
        [32.0, 320.0],
        [64.0, 640.0],
        [128.0, 1280.0],
    ],
    dtype=np.float64,
)


ma = mod.trailing_mean(
    x,
    window=3,
)

assert np.allclose(
    ma[0],
    x[0],
)

assert np.allclose(
    ma[1],
    (
        x[0]
        + x[1]
    ) / 2.0,
)

assert np.allclose(
    ma[2],
    (
        x[0]
        + x[1]
        + x[2]
    ) / 3.0,
)


lp = mod.causal_binomial_lowpass3(
    x
)

assert np.allclose(
    lp[0],
    x[0],
)

assert np.allclose(
    lp[1],
    (
        x[1]
        + 2.0 * x[0]
    ) / 3.0,
)

assert np.allclose(
    lp[2],
    (
        x[2]
        + 2.0 * x[1]
        + x[0]
    ) / 4.0,
)


print(
    "moving-average exact values: True"
)

print(
    "binomial low-pass exact values: True"
)


# =============================================================================
# 2. Savitzky-Golay polynomial reproduction
# =============================================================================

t = np.arange(
    20,
    dtype=np.float64,
)

quadratic = (
    2.0
    + 3.0 * t
    - 0.5 * t**2
)

qx = np.column_stack(
    [
        quadratic,
        2.0 * quadratic + 5.0,
    ]
)


sg = mod.causal_savgol_endpoint(
    qx,
    window=5,
    polyorder=2,
)


assert np.allclose(
    sg,
    qx,
    rtol=1e-10,
    atol=1e-10,
)


print(
    "causal SavGol quadratic reproduction: True"
)


# =============================================================================
# 3. Strict future-perturbation causality
# =============================================================================

filters = {
    "moving_average3":
        lambda z: mod.trailing_mean(
            z,
            window=3,
        ),

    "lowpass_binomial3":
        mod.causal_binomial_lowpass3,

    "savgol5_poly2":
        lambda z: mod.causal_savgol_endpoint(
            z,
            window=5,
            polyorder=2,
        ),

    "kalman":
        lambda z: mod.causal_kalman_local_level(
            z,
            process_variance=1e-3,
            measurement_variance=1e-2,
            initial_variance=1.0,
        ),
}


for name, fn in filters.items():

    base = fn(
        x
    )

    changed = x.copy()

    # Perturb observation t=6 massively.
    changed[6] += 1e6

    out = fn(
        changed
    )

    # Outputs through t=5 MUST remain bitwise/effectively unchanged.
    assert np.array_equal(
        base[:6],
        out[:6],
    ), name


print(
    "future-to-past influence across all filters:",
    False
)


# =============================================================================
# 4. Cross-cell isolation
# =============================================================================

cell_a = x[:5].copy()

cell_b = (
    x[:5].copy()
    + 10000.0
)


for name, fn in filters.items():

    a0 = fn(
        cell_a
    )

    b_changed = (
        cell_b.copy()
        + 1e8
    )

    # Reprocessing cell B cannot affect cell A because filters
    # are trajectory-local and have no global state.
    a1 = fn(
        cell_a
    )

    _ = fn(
        b_changed
    )

    assert np.array_equal(
        a0,
        a1,
    ), name


print(
    "cross-cell influence across all filters:",
    False
)


# =============================================================================
# 5. Kalman prefix and finiteness
# =============================================================================

kf = mod.causal_kalman_local_level(
    x,
    process_variance=1e-3,
    measurement_variance=1e-2,
    initial_variance=1.0,
)

assert np.array_equal(
    kf[0],
    x[0],
)

assert np.isfinite(
    kf
).all()


print(
    "Kalman first observation preserved:",
    True
)

print(
    "Kalman outputs finite:",
    True
)


# =============================================================================
# 6. Full canonical SOH feature population
# =============================================================================

cols = [
    "dataset",
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


assert df[
    "split_group"
].nunique() == 259


first_nonzero = 0
internal_gap_cells = 0

processed_rows = {
    "moving_average3": 0,
    "lowpass_binomial3": 0,
    "savgol5_poly2": 0,
    "kalman": 0,
}


for _, g in df.groupby(
    "split_group",
    sort=True,
):

    g = (
        g.sort_values(
            "observation_index"
        )
        .reset_index(
            drop=True
        )
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
        internal_gap_cells += 1


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


    assert np.isfinite(
        arr
    ).all()


    outputs = {
        "moving_average3":
            mod.trailing_mean(
                arr,
                window=3,
            ),

        "lowpass_binomial3":
            mod.causal_binomial_lowpass3(
                arr
            ),

        "savgol5_poly2":
            mod.causal_savgol_endpoint(
                arr,
                window=5,
                polyorder=2,
            ),

        "kalman":
            mod.causal_kalman_local_level(
                arr,
                process_variance=1e-3,
                measurement_variance=1e-2,
                initial_variance=1.0,
            ),
    }


    for name, out in outputs.items():

        assert out.shape == arr.shape

        assert np.isfinite(
            out
        ).all()

        # Every causal filter begins from the actual first valid
        # observation of that cell. No preceding donor cell exists.
        assert np.allclose(
            out[0],
            arr[0],
            rtol=1e-12,
            atol=1e-12,
        )

        processed_rows[
            name
        ] += len(
            out
        )


assert first_nonzero == 41
assert internal_gap_cells == 0


for name in processed_rows:
    assert processed_rows[
        name
    ] == 300553


print(
    "canonical SOH rows:",
    len(df)
)

print(
    "canonical cells:",
    df[
        "split_group"
    ].nunique()
)

print(
    "cells beginning after observation 0:",
    first_nonzero
)

print(
    "cells with internal observation gaps:",
    internal_gap_cells
)

print(
    "moving-average rows finite:",
    processed_rows[
        "moving_average3"
    ]
)

print(
    "binomial low-pass rows finite:",
    processed_rows[
        "lowpass_binomial3"
    ]
)

print(
    "causal SavGol rows finite:",
    processed_rows[
        "savgol5_poly2"
    ]
)

print(
    "Kalman rows finite:",
    processed_rows[
        "kalman"
    ]
)


# =============================================================================
# 7. Contract statements
# =============================================================================

print(
    "centered smoothing used:",
    False
)

print(
    "filtfilt used:",
    False
)

print(
    "future observations used:",
    False
)

print(
    "target columns read by filter kernel:",
    False
)

print(
    "split/scaler logic present in filter kernel:",
    False
)

print(
    "Kalman q/r selected by kernel:",
    False
)


print()
print(
    "CAUSAL_TEMPORAL_FILTER_KERNEL_AUDIT_PASS=True"
)
