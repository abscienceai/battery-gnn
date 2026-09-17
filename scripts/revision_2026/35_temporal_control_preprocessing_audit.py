from pathlib import Path
import hashlib
import importlib.util
import json
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

OUTDIR = (
    BASE
    / "data"
    / "processed"
    / "revision_2026"
    / "temporal_controls"
)

PARAMS = (
    OUTDIR
    / "temporal_filter_secondary_scalers.csv"
)

SUMMARY = (
    OUTDIR
    / "temporal_filter_fit_summary.csv"
)

MANIFEST = (
    OUTDIR
    / "temporal_filter_preprocessing_manifest.json"
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


EXPECTED_TRAIN_ROWS = {
    0: 211243,
    1: 210640,
    2: 211505,
    3: 213879,
    4: 212447,
}


EXPECTED_FILTERS = [
    "identity",
    "moving_average3",
    "lowpass_binomial3",
    "savgol5_poly2",
    "kalman_q0p001",
    "kalman_q0p01",
    "kalman_q0p1",
    "kalman_q1p0",
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


manifest = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)


assert manifest[
    "contract"
] == "temporal_control_preprocessing_v1"

assert manifest[
    "canonical_soh_rows"
] == 300553

assert manifest[
    "canonical_cells"
] == 259

assert manifest[
    "feature_count"
] == 20

assert manifest[
    "kalman_q_grid"
] == [
    0.001,
    0.01,
    0.1,
    1.0,
]

assert manifest[
    "kalman_measurement_variance_r"
] == 1.0

assert manifest[
    "kalman_initial_variance"
] == 1.0

assert manifest[
    "kalman_grid_frozen_before_validation_selection"
] is True

assert manifest[
    "kalman_grid_selected_from_test"
] is False

assert manifest[
    "target_values_loaded"
] is False

assert manifest[
    "test_metrics_read"
] is False


params = pd.read_csv(
    PARAMS,
    low_memory=False,
)

summary = pd.read_csv(
    SUMMARY,
    low_memory=False,
)


assert len(
    params
) == 800

assert len(
    summary
) == 40


assert set(
    params[
        "split_seed"
    ].astype(int)
) == {
    0,
    1,
    2,
    3,
    4,
}

assert set(
    params[
        "filter_name"
    ].astype(str)
) == set(
    EXPECTED_FILTERS
)


for split_seed in range(5):

    for filter_name in EXPECTED_FILTERS:

        p = params[
            (
                params[
                    "split_seed"
                ] == split_seed
            )
            &
            (
                params[
                    "filter_name"
                ] == filter_name
            )
        ]


        assert len(p) == 20

        assert p[
            "feature"
        ].tolist() == FEATURES

        assert set(
            p[
                "fit_split"
            ].astype(str)
        ) == {
            "train_only"
        }

        assert set(
            p[
                "fit_rows"
            ].astype(int)
        ) == {
            EXPECTED_TRAIN_ROWS[
                split_seed
            ]
        }

        assert set(
            p[
                "fit_groups"
            ].astype(int)
        ) == {
            182
        }

        assert np.isfinite(
            p[
                "secondary_mean"
            ]
        ).all()

        assert np.isfinite(
            p[
                "secondary_variance"
            ]
        ).all()

        assert np.isfinite(
            p[
                "secondary_scale"
            ]
        ).all()

        assert np.all(
            p[
                "secondary_scale"
            ].to_numpy(
                dtype=float
            ) > 0
        )


# =============================================================================
# Identity secondary scaler must reproduce standardized training data:
# approximately mean 0, variance 1.
# =============================================================================

identity = params[
    params[
        "filter_name"
    ] == "identity"
].copy()


identity_mean_max = float(
    np.max(
        np.abs(
            identity[
                "secondary_mean"
            ].to_numpy(
                dtype=float
            )
        )
    )
)

identity_var_delta_max = float(
    np.max(
        np.abs(
            identity[
                "secondary_variance"
            ].to_numpy(
                dtype=float
            )
            - 1.0
        )
    )
)


assert identity_mean_max < 1e-10

assert identity_var_delta_max < 1e-10


print("=" * 78)
print("TEMPORAL CONTROL PREPROCESSING INDEPENDENT AUDIT")
print("=" * 78)

print(
    "secondary scaler rows:",
    len(params)
)

print(
    "fit summary rows:",
    len(summary)
)

print(
    "filter configurations:",
    len(
        EXPECTED_FILTERS
    )
)

print(
    "split seeds:",
    5
)

print(
    "all scaler fits train-only:",
    True
)

print(
    "validation/test rows entering secondary scaler fits:",
    0
)

print(
    "identity max |secondary mean|:",
    f"{identity_mean_max:.3e}"
)

print(
    "identity max |secondary variance-1|:",
    f"{identity_var_delta_max:.3e}"
)

print(
    "identity secondary scaler exact within tolerance:",
    True
)


# =============================================================================
# Independently recompute selected splits across ALL filter configurations.
# =============================================================================

spec = importlib.util.spec_from_file_location(
    "causal_filters",
    KERNEL,
)

filters = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    filters
)


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

df = df.reset_index(
    drop=True
)


raw = (
    df[
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


groups = []

for _, g in df.groupby(
    "split_group",
    sort=True,
):

    g = g.sort_values(
        "observation_index"
    )

    groups.append(
        g.index.to_numpy(
            dtype=np.int64
        )
    )


assignments = pd.read_csv(
    ASSIGNMENTS,
    low_memory=False,
)

base_scalers = pd.read_csv(
    BASE_SCALERS,
    low_memory=False,
)


def transform(
    x,
    name,
):

    if name == "identity":
        return filters.identity(
            x
        )

    if name == "moving_average3":
        return filters.trailing_mean(
            x,
            window=3,
        )

    if name == "lowpass_binomial3":
        return filters.causal_binomial_lowpass3(
            x
        )

    if name == "savgol5_poly2":
        return filters.causal_savgol_endpoint(
            x,
            window=5,
            polyorder=2,
        )

    q_map = {
        "kalman_q0p001": 0.001,
        "kalman_q0p01": 0.01,
        "kalman_q0p1": 0.1,
        "kalman_q1p0": 1.0,
    }

    if name in q_map:

        return filters.causal_kalman_local_level(
            x,
            process_variance=q_map[
                name
            ],
            measurement_variance=1.0,
            initial_variance=1.0,
        )

    raise ValueError(
        name
    )


max_mean_delta = 0.0
max_var_delta = 0.0
max_scale_delta = 0.0


for split_seed in (
    0,
    4,
):

    a = assignments[
        assignments[
            "split_seed"
        ] == split_seed
    ][
        [
            "split_group",
            "split",
        ]
    ]


    sf = (
        df[
            [
                "split_group",
            ]
        ]
        .merge(
            a,
            on="split_group",
            how="left",
            validate="many_to_one",
        )
    )


    train_mask = (
        sf[
            "split"
        ].to_numpy()
        == "train"
    )


    p0 = base_scalers[
        (
            base_scalers[
                "split_seed"
            ] == split_seed
        )
        &
        (
            base_scalers[
                "scaler_scope"
            ] == "pooled_supervised"
        )
    ]


    p0 = (
        p0.set_index(
            "feature"
        )
        .loc[
            FEATURES
        ]
    )


    z = (
        raw
        - p0[
            "mean"
        ].to_numpy(
            dtype=float
        )
    ) / p0[
        "scale"
    ].to_numpy(
        dtype=float
    )


    for filter_name in EXPECTED_FILTERS:

        out = np.empty_like(
            z
        )


        for idx in groups:

            out[
                idx
            ] = transform(
                z[
                    idx
                ],
                filter_name,
            )


        train = out[
            train_mask
        ]


        mean = np.mean(
            train,
            axis=0,
            dtype=np.float64,
        )

        var = np.var(
            train,
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


        stored = params[
            (
                params[
                    "split_seed"
                ] == split_seed
            )
            &
            (
                params[
                    "filter_name"
                ] == filter_name
            )
        ].set_index(
            "feature"
        ).loc[
            FEATURES
        ]


        dm = float(
            np.max(
                np.abs(
                    mean
                    - stored[
                        "secondary_mean"
                    ].to_numpy(
                        dtype=float
                    )
                )
            )
        )

        dv = float(
            np.max(
                np.abs(
                    var
                    - stored[
                        "secondary_variance"
                    ].to_numpy(
                        dtype=float
                    )
                )
            )
        )

        ds = float(
            np.max(
                np.abs(
                    scale
                    - stored[
                        "secondary_scale"
                    ].to_numpy(
                        dtype=float
                    )
                )
            )
        )


        max_mean_delta = max(
            max_mean_delta,
            dm,
        )

        max_var_delta = max(
            max_var_delta,
            dv,
        )

        max_scale_delta = max(
            max_scale_delta,
            ds,
        )


assert max_mean_delta < 1e-12
assert max_var_delta < 1e-12
assert max_scale_delta < 1e-12


print(
    "independent recomputation split seeds:",
    "0,4"
)

print(
    "independent filter configurations checked:",
    len(
        EXPECTED_FILTERS
    )
)

print(
    "max secondary mean delta:",
    f"{max_mean_delta:.3e}"
)

print(
    "max secondary variance delta:",
    f"{max_var_delta:.3e}"
)

print(
    "max secondary scale delta:",
    f"{max_scale_delta:.3e}"
)

print(
    "independent scaler reproduction:",
    True
)


# =============================================================================
# Contract / leakage statements
# =============================================================================

print(
    "base scaler source:",
    "pooled_supervised train-only"
)

print(
    "filtering performed within cell:",
    True
)

print(
    "target values used by preprocessing:",
    False
)

print(
    "test metrics read:",
    False
)

print(
    "Kalman grid validation-selected yet:",
    False
)

print(
    "Kalman grid test-selected:",
    False
)

print()
print(
    "TEMPORAL_CONTROL_PREPROCESSING_AUDIT_PASS=True"
)
