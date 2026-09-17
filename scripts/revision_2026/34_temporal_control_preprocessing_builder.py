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

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

PARAMS_OUT = (
    OUTDIR
    / "temporal_filter_secondary_scalers.csv"
)

SUMMARY_OUT = (
    OUTDIR
    / "temporal_filter_fit_summary.csv"
)

MANIFEST_OUT = (
    OUTDIR
    / "temporal_filter_preprocessing_manifest.json"
)


EXPECTED_HASHES = {
    "kernel":
        "23f733540a23954c278722c43a36e994c5431b603bfeb9e7ebd4474451bd3ff7",

    "canonical":
        "49a7b7629cf9ee3f8ce0a7e3c52f2807878b752b4c9a30a876285fb891b00c14",

    "assignments":
        "914c8b90131f496ca396f63e0502147ca99848fdfd4b2c6b2a5887225b47d055",

    "base_scalers":
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


EXPECTED_TRAIN_ROWS = {
    0: 211243,
    1: 210640,
    2: 211505,
    3: 213879,
    4: 212447,
}


KALMAN_Q_GRID = [
    0.001,
    0.01,
    0.1,
    1.0,
]

KALMAN_R = 1.0
KALMAN_INITIAL_VARIANCE = 1.0


FILTER_CONFIGS = [
    {
        "name": "identity",
        "family": "identity",
        "kalman_q": np.nan,
        "kalman_r": np.nan,
    },
    {
        "name": "moving_average3",
        "family": "moving_average",
        "kalman_q": np.nan,
        "kalman_r": np.nan,
    },
    {
        "name": "lowpass_binomial3",
        "family": "lowpass",
        "kalman_q": np.nan,
        "kalman_r": np.nan,
    },
    {
        "name": "savgol5_poly2",
        "family": "savgol_endpoint",
        "kalman_q": np.nan,
        "kalman_r": np.nan,
    },
] + [
    {
        "name":
            (
                "kalman_q"
                + str(q).replace(
                    ".",
                    "p",
                )
            ),
        "family":
            "kalman_local_level",
        "kalman_q":
            q,
        "kalman_r":
            KALMAN_R,
    }
    for q in KALMAN_Q_GRID
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


for key, expected in EXPECTED_HASHES.items():

    path = {
        "kernel": KERNEL,
        "canonical": CANON,
        "assignments": ASSIGNMENTS,
        "base_scalers": BASE_SCALERS,
    }[key]

    assert sha256_file(
        path
    ) == expected


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


# =============================================================================
# Canonical feature-only population.
#
# SOH values, RUL values and test metrics are NOT loaded.
# soh_task_eligible is used only to recover the already-audited
# 300,553-row SOH task population.
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


df = df.reset_index(
    drop=True
)


assert len(
    df
) == 300553

assert df[
    "split_group"
].nunique() == 259


raw_x = (
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


assert np.isfinite(
    raw_x
).all()


# Chronological row indices for each full cell.
group_indices = []


for _, g in df.groupby(
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
        g[
            "observation_index"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.int64
    )

    if len(obs) > 1:
        assert np.all(
            np.diff(obs) == 1
        )

    group_indices.append(
        idx
    )


assert len(
    group_indices
) == 259


assignments = pd.read_csv(
    ASSIGNMENTS,
    low_memory=False,
)

base_scalers = pd.read_csv(
    BASE_SCALERS,
    low_memory=False,
)


def apply_config(
    x,
    config,
):

    name = config[
        "name"
    ]

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

    if config[
        "family"
    ] == "kalman_local_level":

        return filters.causal_kalman_local_level(
            x,
            process_variance=float(
                config[
                    "kalman_q"
                ]
            ),
            measurement_variance=float(
                config[
                    "kalman_r"
                ]
            ),
            initial_variance=KALMAN_INITIAL_VARIANCE,
        )

    raise ValueError(
        name
    )


param_rows = []
summary_rows = []


for split_seed in range(5):

    # -------------------------------------------------------------------------
    # Cell-level split assignment
    # -------------------------------------------------------------------------

    a = assignments[
        assignments[
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


    split_frame = (
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


    assert split_frame[
        "split"
    ].notna().all()


    train_mask = (
        split_frame[
            "split"
        ].to_numpy()
        == "train"
    )


    assert int(
        train_mask.sum()
    ) == EXPECTED_TRAIN_ROWS[
        split_seed
    ]


    train_groups = int(
        split_frame.loc[
            train_mask,
            "split_group",
        ].nunique()
    )


    assert train_groups == 182


    # -------------------------------------------------------------------------
    # Stage 1:
    # existing pooled train-only scaler.
    # -------------------------------------------------------------------------

    p = base_scalers[
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


    base_mean = p[
        "mean"
    ].to_numpy(
        dtype=np.float64
    )

    base_scale = p[
        "scale"
    ].to_numpy(
        dtype=np.float64
    )


    assert np.isfinite(
        base_mean
    ).all()

    assert np.isfinite(
        base_scale
    ).all()

    assert np.all(
        base_scale > 0
    )


    standardized = (
        raw_x
        - base_mean
    ) / base_scale


    assert np.isfinite(
        standardized
    ).all()


    # -------------------------------------------------------------------------
    # Stage 2:
    # causal filtering per full cell.
    #
    # Stage 3:
    # filter-specific secondary scaler fit on TRAIN CELLS ONLY.
    # -------------------------------------------------------------------------

    for config in FILTER_CONFIGS:

        filtered = np.empty_like(
            standardized
        )


        for idx in group_indices:

            segment = standardized[
                idx
            ]

            out = apply_config(
                segment,
                config,
            )

            assert out.shape == segment.shape

            assert np.isfinite(
                out
            ).all()

            filtered[
                idx
            ] = out


        train_x = filtered[
            train_mask
        ]


        assert train_x.shape == (
            EXPECTED_TRAIN_ROWS[
                split_seed
            ],
            20,
        )


        second_mean = np.mean(
            train_x,
            axis=0,
            dtype=np.float64,
        )

        second_variance = np.var(
            train_x,
            axis=0,
            ddof=0,
            dtype=np.float64,
        )

        second_scale = np.sqrt(
            second_variance
        )


        second_scale[
            second_scale == 0
        ] = 1.0


        assert np.isfinite(
            second_mean
        ).all()

        assert np.isfinite(
            second_variance
        ).all()

        assert np.isfinite(
            second_scale
        ).all()

        assert np.all(
            second_scale > 0
        )


        for j, feature in enumerate(
            FEATURES
        ):

            param_rows.append({
                "split_seed":
                    split_seed,

                "filter_name":
                    config[
                        "name"
                    ],

                "filter_family":
                    config[
                        "family"
                    ],

                "kalman_q":
                    config[
                        "kalman_q"
                    ],

                "kalman_r":
                    config[
                        "kalman_r"
                    ],

                "kalman_initial_variance":
                    (
                        KALMAN_INITIAL_VARIANCE
                        if config[
                            "family"
                        ] == "kalman_local_level"
                        else np.nan
                    ),

                "feature":
                    feature,

                "secondary_mean":
                    float(
                        second_mean[j]
                    ),

                "secondary_variance":
                    float(
                        second_variance[j]
                    ),

                "secondary_scale":
                    float(
                        second_scale[j]
                    ),

                "fit_split":
                    "train_only",

                "fit_rows":
                    int(
                        train_mask.sum()
                    ),

                "fit_groups":
                    train_groups,

                "base_scaler_scope":
                    "pooled_supervised",
            })


        summary_rows.append({
            "split_seed":
                split_seed,

            "filter_name":
                config[
                    "name"
                ],

            "filter_family":
                config[
                    "family"
                ],

            "kalman_q":
                config[
                    "kalman_q"
                ],

            "kalman_r":
                config[
                    "kalman_r"
                ],

            "fit_rows":
                int(
                    train_mask.sum()
                ),

            "fit_groups":
                train_groups,

            "secondary_abs_mean_max":
                float(
                    np.max(
                        np.abs(
                            second_mean
                        )
                    )
                ),

            "secondary_variance_min":
                float(
                    np.min(
                        second_variance
                    )
                ),

            "secondary_variance_max":
                float(
                    np.max(
                        second_variance
                    )
                ),

            "secondary_scale_min":
                float(
                    np.min(
                        second_scale
                    )
                ),

            "secondary_scale_max":
                float(
                    np.max(
                        second_scale
                    )
                ),

            "all_filtered_rows_finite":
                bool(
                    np.isfinite(
                        filtered
                    ).all()
                ),
        })


params = pd.DataFrame(
    param_rows
)

summary = pd.DataFrame(
    summary_rows
)


assert len(
    params
) == (
    5
    * len(
        FILTER_CONFIGS
    )
    * 20
)

assert len(
    summary
) == (
    5
    * len(
        FILTER_CONFIGS
    )
)


params.to_csv(
    PARAMS_OUT,
    index=False,
    float_format="%.17g",
)

summary.to_csv(
    SUMMARY_OUT,
    index=False,
    float_format="%.17g",
)


manifest = {
    "contract":
        "temporal_control_preprocessing_v1",

    "pipeline":
        [
            (
                "20 canonical cycle features"
            ),
            (
                "split-specific pooled train-only base standardization"
            ),
            (
                "strictly causal within-cell temporal filter"
            ),
            (
                "filter-specific secondary standardization fit "
                "on training cells only"
            ),
        ],

    "canonical_soh_rows":
        300553,

    "canonical_cells":
        259,

    "feature_count":
        20,

    "feature_order":
        FEATURES,

    "split_seeds":
        [
            0,
            1,
            2,
            3,
            4,
        ],

    "fixed_filters":
        [
            "identity",
            "moving_average3",
            "lowpass_binomial3",
            "savgol5_poly2",
        ],

    "moving_average_window":
        3,

    "lowpass_definition":
        (
            "causal normalized taps [1,2,1] "
            "over {t-2,t-1,t}"
        ),

    "savgol_definition":
        (
            "one-sided trailing endpoint polynomial smoother, "
            "window=5, polyorder=2"
        ),

    "kalman_family":
        "independent scalar local-level filter per feature",

    "kalman_q_grid":
        KALMAN_Q_GRID,

    "kalman_measurement_variance_r":
        KALMAN_R,

    "kalman_initial_variance":
        KALMAN_INITIAL_VARIANCE,

    "kalman_grid_frozen_before_validation_selection":
        True,

    "kalman_grid_selected_from_test":
        False,

    "target_values_loaded":
        False,

    "test_metrics_read":
        False,

    "filtered_feature_matrices_materialized":
        False,

    "input_sha256":
        EXPECTED_HASHES,

    "output_sha256": {
        "secondary_scalers":
            sha256_file(
                PARAMS_OUT
            ),

        "fit_summary":
            sha256_file(
                SUMMARY_OUT
            ),
    },
}


MANIFEST_OUT.write_text(
    json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


print("=" * 78)
print("TEMPORAL CONTROL PREPROCESSING BUILDER")
print("=" * 78)

print(
    "canonical rows:",
    len(df)
)

print(
    "canonical cells:",
    df[
        "split_group"
    ].nunique()
)

print(
    "split seeds:",
    5
)

print(
    "filter configurations per split:",
    len(
        FILTER_CONFIGS
    )
)

print(
    "Kalman q grid:",
    KALMAN_Q_GRID
)

print(
    "Kalman r:",
    KALMAN_R
)

print(
    "Kalman grid frozen before validation selection:",
    True
)

print(
    "target values loaded:",
    False
)

print(
    "test metrics read:",
    False
)

print(
    "secondary scaler rows:",
    len(params)
)

print(
    "fit summary rows:",
    len(summary)
)

print(
    "secondary scaler SHA256:",
    sha256_file(
        PARAMS_OUT
    )
)

print(
    "fit summary SHA256:",
    sha256_file(
        SUMMARY_OUT
    )
)

print(
    "manifest SHA256:",
    sha256_file(
        MANIFEST_OUT
    )
)

print()
print(
    "TEMPORAL_CONTROL_PREPROCESSING_BUILD_PASS=True"
)
