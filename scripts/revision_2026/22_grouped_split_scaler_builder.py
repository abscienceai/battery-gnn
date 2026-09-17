from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd


# =============================================================================
# Immutable paths / contracts
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

SUPERVISED = (
    CANON
    / "canonical_cycle_supervised_table.csv"
)

RWTH_HISTORY = (
    CANON
    / "RWTH_usable_history_cycles.csv"
)

RWTH_TARGETS = (
    CANON
    / "RWTH_sparse_target_table.csv"
)

ALIGN_MANIFEST = (
    CANON
    / "canonical_task_alignment_manifest.json"
)

OUTDIR = (
    BASE
    / "data"
    / "processed"
    / "revision_2026"
    / "splits_and_scalers"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

ASSIGN_OUT = (
    OUTDIR
    / "cell_split_assignments.csv"
)

SCALER_OUT = (
    OUTDIR
    / "train_only_scaler_parameters.csv"
)

SUMMARY_OUT = (
    OUTDIR
    / "train_only_scaler_fit_summary.csv"
)

MANIFEST_OUT = (
    OUTDIR
    / "grouped_split_scaler_manifest.json"
)


CONTRACT_VERSION = (
    "grouped_dataset_censor_stratified_v1"
)

SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

SPLITS = [
    "train",
    "validation",
    "test",
]

RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}


EXPECTED_INPUT_HASHES = {
    "supervised":
        "49a7b7629cf9ee3f8ce0a7e3c52f2807878b752b4c9a30a876285fb891b00c14",

    "rwth_history":
        "4e8dcc9d80e0284e3c6cb9f912505fbff066efff052d3e08996da4028e3a0faa",

    "rwth_targets":
        "6346566441b265e9f5d328527ebeb58a0c9dfb2d18ba70ab72ea7b9f2808eb54",

    "alignment_manifest":
        "f0a9fe3f10c4adc05c3dd76947da824811cf42a5070261f0e14728dbd0b7acd9",
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


def as_bool(series):
    return (
        series
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


def stable_u64(text):
    raw = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        raw[:8],
        byteorder="little",
        signed=False,
    )


def allocation_counts(n):
    exact = np.array(
        [
            RATIOS["train"] * n,
            RATIOS["validation"] * n,
            RATIOS["test"] * n,
        ],
        dtype=float,
    )

    counts = np.floor(
        exact
    ).astype(int)

    remaining = int(
        n - counts.sum()
    )

    fractions = (
        exact - counts
    )

    # Stable tie order:
    # train -> validation -> test.
    order = sorted(
        range(3),
        key=lambda i:
            (
                -fractions[i],
                i,
            ),
    )

    for i in order[:remaining]:
        counts[i] += 1

    out = {
        split:
            int(count)
        for split, count
        in zip(
            SPLITS,
            counts,
        )
    }

    assert sum(
        out.values()
    ) == n

    if n >= 3:
        assert all(
            x >= 1
            for x in out.values()
        )

    return out


def single_value(group, col):
    values = (
        group[col]
        .drop_duplicates()
        .tolist()
    )

    assert len(values) == 1, (
        col,
        values,
    )

    return values[0]


def build_cell_profile(df):
    rows = []

    for split_group, g in df.groupby(
        "split_group",
        sort=True,
    ):
        dataset = str(
            single_value(
                g,
                "dataset",
            )
        )

        cell_id = str(
            single_value(
                g,
                "cell_id",
            )
        )

        cohort = str(
            single_value(
                g,
                "cohort",
            )
        )

        censor = str(
            single_value(
                g,
                "censoring_status",
            )
        )

        rows.append({
            "split_group":
                str(split_group),

            "dataset":
                dataset,

            "cell_id":
                cell_id,

            "cohort":
                cohort,

            "censoring_status":
                censor,

            "stratum":
                (
                    dataset
                    + "::"
                    + censor
                ),
        })

    return pd.DataFrame(
        rows
    )


def make_assignments(profile):
    rows = []

    for seed in SEEDS:

        for stratum, g in profile.groupby(
            "stratum",
            sort=True,
        ):
            groups = np.array(
                sorted(
                    g[
                        "split_group"
                    ].astype(str)
                ),
                dtype=object,
            )

            n = len(groups)

            counts = allocation_counts(
                n
            )

            stable = stable_u64(
                stratum
            )

            ss = np.random.SeedSequence(
                [
                    int(seed),
                    int(
                        stable
                        & 0xFFFFFFFF
                    ),
                    int(
                        (
                            stable
                            >> 32
                        )
                        & 0xFFFFFFFF
                    ),
                ]
            )

            rng = np.random.default_rng(
                ss
            )

            perm = rng.permutation(
                n
            )

            shuffled = groups[
                perm
            ]

            a = 0

            for split in SPLITS:
                b = (
                    a
                    + counts[split]
                )

                chosen = shuffled[
                    a:b
                ]

                for split_group in chosen:

                    row = g[
                        g[
                            "split_group"
                        ].astype(str)
                        == str(
                            split_group
                        )
                    ]

                    assert len(
                        row
                    ) == 1

                    r = row.iloc[0]

                    rows.append({
                        "split_seed":
                            seed,

                        "split_group":
                            str(
                                split_group
                            ),

                        "dataset":
                            str(
                                r[
                                    "dataset"
                                ]
                            ),

                        "cell_id":
                            str(
                                r[
                                    "cell_id"
                                ]
                            ),

                        "cohort":
                            str(
                                r[
                                    "cohort"
                                ]
                            ),

                        "censoring_status":
                            str(
                                r[
                                    "censoring_status"
                                ]
                            ),

                        "stratum":
                            stratum,

                        "split":
                            split,

                        "split_contract_version":
                            CONTRACT_VERSION,
                    })

                a = b

            assert a == n

    out = pd.DataFrame(
        rows
    )

    split_order = {
        "train": 0,
        "validation": 1,
        "test": 2,
    }

    out["_split_order"] = (
        out["split"]
        .map(
            split_order
        )
    )

    out = (
        out
        .sort_values(
            [
                "split_seed",
                "dataset",
                "stratum",
                "_split_order",
                "split_group",
            ]
        )
        .drop(
            columns="_split_order"
        )
        .reset_index(
            drop=True
        )
    )

    return out


def fit_standard_scaler(
    df,
    mask,
):
    x = (
        df.loc[
            mask,
            FEATURES,
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    assert len(x) > 0
    assert np.isfinite(
        x
    ).all()

    mean = np.mean(
        x,
        axis=0,
        dtype=np.float64,
    )

    var = np.var(
        x,
        axis=0,
        ddof=0,
        dtype=np.float64,
    )

    assert np.isfinite(
        mean
    ).all()

    assert np.isfinite(
        var
    ).all()

    assert (
        var >= 0
    ).all()

    scale = np.sqrt(
        var
    )

    zero = (
        scale == 0
    )

    # sklearn StandardScaler-compatible convention:
    # zero-variance feature gets scale 1.
    scale_safe = scale.copy()

    scale_safe[
        zero
    ] = 1.0

    return {
        "n":
            int(len(x)),

        "mean":
            mean,

        "var":
            var,

        "scale":
            scale_safe,

        "zero_variance":
            zero,
    }


def write_csv(df, path):
    df.to_csv(
        path,
        index=False,
        float_format="%.17g",
        na_rep="",
        lineterminator="\n",
    )


# =============================================================================
# Immutable input guards
# =============================================================================

observed_input_hashes = {
    "supervised":
        sha256_file(
            SUPERVISED
        ),

    "rwth_history":
        sha256_file(
            RWTH_HISTORY
        ),

    "rwth_targets":
        sha256_file(
            RWTH_TARGETS
        ),

    "alignment_manifest":
        sha256_file(
            ALIGN_MANIFEST
        ),
}

assert (
    observed_input_hashes
    == EXPECTED_INPUT_HASHES
)


# =============================================================================
# Load only columns required for splitting/scaling
# =============================================================================

supervised_cols = [
    "dataset",
    "cohort",
    "cell_id",
    "dataset_cell_key",
    "split_group",
    "censoring_status",
    "supervised_feature_eligible",
] + FEATURES

sup = pd.read_csv(
    SUPERVISED,
    usecols=supervised_cols,
    low_memory=False,
)

assert len(
    sup
) == 300595

assert sup[
    "split_group"
].nunique() == 259


rwth_target_cols = [
    "dataset",
    "cohort",
    "cell_id",
    "dataset_cell_key",
    "split_group",
    "censoring_status",
]

rt = pd.read_csv(
    RWTH_TARGETS,
    usecols=rwth_target_cols,
    low_memory=False,
)

assert len(
    rt
) == 777

assert rt[
    "split_group"
].nunique() == 48


rh = pd.read_csv(
    RWTH_HISTORY,
    usecols=[
        "canonical_cell_id",
    ] + FEATURES,
    low_memory=False,
)

assert len(
    rh
) == 104940

assert rh[
    "canonical_cell_id"
].nunique() == 48

rh[
    "split_group"
] = (
    "RWTH::"
    + rh[
        "canonical_cell_id"
    ].astype(str)
)


# =============================================================================
# Cell-level split profile
# =============================================================================

sup_profile = build_cell_profile(
    sup[
        [
            "dataset",
            "cohort",
            "cell_id",
            "split_group",
            "censoring_status",
        ]
    ]
)

rwth_profile = build_cell_profile(
    rt[
        [
            "dataset",
            "cohort",
            "cell_id",
            "split_group",
            "censoring_status",
        ]
    ]
)

profile = pd.concat(
    [
        sup_profile,
        rwth_profile,
    ],
    ignore_index=True,
)

assert len(
    profile
) == 307

assert profile[
    "split_group"
].nunique() == 307


# Independent cell-level censoring semantics.
cell_censor_counts = (
    profile.groupby(
        [
            "dataset",
            "censoring_status",
        ]
    )
    .size()
    .to_dict()
)

expected_cell_censor_counts = {
    (
        "HUST",
        "observed_event",
    ): 77,

    (
        "CALCE",
        "observed_event",
    ): 13,

    (
        "MATR",
        "observed_event",
    ): 88,

    (
        "MATR",
        "right_censored_len_plus_1",
    ): 81,

    (
        "RWTH",
        "observed_event",
    ): 48,
}

assert (
    cell_censor_counts
    == expected_cell_censor_counts
), cell_censor_counts


# =============================================================================
# Build five deterministic grouped stratified split realizations
# =============================================================================

assign = make_assignments(
    profile
)

assert len(
    assign
) == (
    307
    * len(SEEDS)
)

assert not assign[
    [
        "split_seed",
        "split_group",
    ]
].duplicated().any()


# Every seed contains every cell exactly once.
for seed in SEEDS:

    a = assign[
        assign[
            "split_seed"
        ] == seed
    ]

    assert len(
        a
    ) == 307

    assert a[
        "split_group"
    ].nunique() == 307

    assert set(
        a["split"]
    ) == set(
        SPLITS
    )


# Expected group allocation after dataset + censoring stratification.
expected_dataset_split_counts = {
    "HUST": {
        "train": 54,
        "validation": 12,
        "test": 11,
    },

    "CALCE": {
        "train": 9,
        "validation": 2,
        "test": 2,
    },

    "MATR": {
        "train": 119,
        "validation": 25,
        "test": 25,
    },

    "RWTH": {
        "train": 34,
        "validation": 7,
        "test": 7,
    },
}


for seed in SEEDS:

    a = assign[
        assign[
            "split_seed"
        ] == seed
    ]

    for dataset, expected in (
        expected_dataset_split_counts.items()
    ):

        observed = (
            a[
                a[
                    "dataset"
                ] == dataset
            ][
                "split"
            ]
            .value_counts()
            .to_dict()
        )

        assert (
            observed
            == expected
        ), (
            seed,
            dataset,
            observed,
            expected,
        )


# MATR censoring strata remain balanced by construction.
expected_matr_strata = {
    "observed_event": {
        "train": 62,
        "validation": 13,
        "test": 13,
    },

    "right_censored_len_plus_1": {
        "train": 57,
        "validation": 12,
        "test": 12,
    },
}


for seed in SEEDS:

    a = assign[
        (
            assign[
                "split_seed"
            ] == seed
        )
        &
        (
            assign[
                "dataset"
            ] == "MATR"
        )
    ]

    for status, expected in (
        expected_matr_strata.items()
    ):

        observed = (
            a[
                a[
                    "censoring_status"
                ] == status
            ][
                "split"
            ]
            .value_counts()
            .to_dict()
        )

        assert (
            observed
            == expected
        ), (
            seed,
            status,
            observed,
            expected,
        )


write_csv(
    assign,
    ASSIGN_OUT,
)


# =============================================================================
# Fit train-only scaler parameters.
#
# Scopes:
# - pooled_supervised: HUST+CALCE+MATR train cells
# - HUST/CALCE/MATR: within-dataset train cells
# - RWTH_history: histories from train RWTH cells only
#
# LODO is intentionally NOT fit here.
# =============================================================================

sup[
    "_eligible"
] = as_bool(
    sup[
        "supervised_feature_eligible"
    ]
)

scaler_rows = []
summary_rows = []


for seed in SEEDS:

    a = assign[
        assign[
            "split_seed"
        ] == seed
    ]

    train_groups = set(
        a.loc[
            a[
                "split"
            ] == "train",
            "split_group",
        ].astype(str)
    )

    validation_groups = set(
        a.loc[
            a[
                "split"
            ] == "validation",
            "split_group",
        ].astype(str)
    )

    test_groups = set(
        a.loc[
            a[
                "split"
            ] == "test",
            "split_group",
        ].astype(str)
    )

    assert train_groups.isdisjoint(
        validation_groups
    )

    assert train_groups.isdisjoint(
        test_groups
    )

    assert validation_groups.isdisjoint(
        test_groups
    )


    # -------------------------------------------------------------------------
    # Pooled supervised
    # -------------------------------------------------------------------------

    pooled_mask = (
        sup[
            "_eligible"
        ]
        &
        sup[
            "split_group"
        ].astype(str).isin(
            train_groups
        )
    )

    result = fit_standard_scaler(
        sup,
        pooled_mask,
    )

    pooled_groups = set(
        sup.loc[
            pooled_mask,
            "split_group",
        ].astype(str)
    )

    assert pooled_groups.issubset(
        train_groups
    )

    scope = (
        "pooled_supervised"
    )

    summary_rows.append({
        "split_seed":
            seed,

        "scaler_scope":
            scope,

        "n_train_rows":
            result["n"],

        "n_train_groups":
            len(
                pooled_groups
            ),

        "zero_variance_features":
            int(
                result[
                    "zero_variance"
                ].sum()
            ),

        "fit_split":
            "train_only",
    })

    for j, feature in enumerate(
        FEATURES
    ):
        scaler_rows.append({
            "split_seed":
                seed,

            "scaler_scope":
                scope,

            "feature":
                feature,

            "mean":
                float(
                    result[
                        "mean"
                    ][j]
                ),

            "variance":
                float(
                    result[
                        "var"
                    ][j]
                ),

            "scale":
                float(
                    result[
                        "scale"
                    ][j]
                ),

            "zero_variance":
                bool(
                    result[
                        "zero_variance"
                    ][j]
                ),

            "n_train_rows":
                result["n"],

            "fit_split":
                "train_only",
        })


    # -------------------------------------------------------------------------
    # Dataset-specific supervised scalers
    # -------------------------------------------------------------------------

    for dataset in (
        "HUST",
        "CALCE",
        "MATR",
    ):

        mask = (
            sup[
                "_eligible"
            ]
            &
            (
                sup[
                    "dataset"
                ] == dataset
            )
            &
            sup[
                "split_group"
            ].astype(str).isin(
                train_groups
            )
        )

        result = fit_standard_scaler(
            sup,
            mask,
        )

        used_groups = set(
            sup.loc[
                mask,
                "split_group",
            ].astype(str)
        )

        expected_train_groups = set(
            a.loc[
                (
                    a[
                        "dataset"
                    ] == dataset
                )
                &
                (
                    a[
                        "split"
                    ] == "train"
                ),
                "split_group",
            ].astype(str)
        )

        assert (
            used_groups
            == expected_train_groups
        )

        summary_rows.append({
            "split_seed":
                seed,

            "scaler_scope":
                dataset,

            "n_train_rows":
                result["n"],

            "n_train_groups":
                len(
                    used_groups
                ),

            "zero_variance_features":
                int(
                    result[
                        "zero_variance"
                    ].sum()
                ),

            "fit_split":
                "train_only",
        })

        for j, feature in enumerate(
            FEATURES
        ):
            scaler_rows.append({
                "split_seed":
                    seed,

                "scaler_scope":
                    dataset,

                "feature":
                    feature,

                "mean":
                    float(
                        result[
                            "mean"
                        ][j]
                    ),

                "variance":
                    float(
                        result[
                            "var"
                        ][j]
                    ),

                "scale":
                    float(
                        result[
                            "scale"
                        ][j]
                    ),

                "zero_variance":
                    bool(
                        result[
                            "zero_variance"
                        ][j]
                    ),

                "n_train_rows":
                    result["n"],

                "fit_split":
                    "train_only",
            })


    # -------------------------------------------------------------------------
    # RWTH history scaler: train RWTH cells only
    # -------------------------------------------------------------------------

    rwth_train_groups = set(
        a.loc[
            (
                a[
                    "dataset"
                ] == "RWTH"
            )
            &
            (
                a[
                    "split"
                ] == "train"
            ),
            "split_group",
        ].astype(str)
    )

    rwth_mask = (
        rh[
            "split_group"
        ].astype(str).isin(
            rwth_train_groups
        )
    )

    result = fit_standard_scaler(
        rh,
        rwth_mask,
    )

    used_groups = set(
        rh.loc[
            rwth_mask,
            "split_group",
        ].astype(str)
    )

    assert (
        used_groups
        == rwth_train_groups
    )

    scope = (
        "RWTH_history"
    )

    summary_rows.append({
        "split_seed":
            seed,

        "scaler_scope":
            scope,

        "n_train_rows":
            result["n"],

        "n_train_groups":
            len(
                used_groups
            ),

        "zero_variance_features":
            int(
                result[
                    "zero_variance"
                ].sum()
            ),

        "fit_split":
            "train_only",
    })

    for j, feature in enumerate(
        FEATURES
    ):
        scaler_rows.append({
            "split_seed":
                seed,

            "scaler_scope":
                scope,

            "feature":
                feature,

            "mean":
                float(
                    result[
                        "mean"
                    ][j]
                ),

            "variance":
                float(
                    result[
                        "var"
                    ][j]
                ),

            "scale":
                float(
                    result[
                        "scale"
                    ][j]
                ),

            "zero_variance":
                bool(
                    result[
                        "zero_variance"
                    ][j]
                ),

            "n_train_rows":
                result["n"],

            "fit_split":
                "train_only",
        })


scaler = pd.DataFrame(
    scaler_rows
)

summary = pd.DataFrame(
    summary_rows
)


assert len(
    scaler
) == (
    len(SEEDS)
    * 5
    * len(FEATURES)
)

assert len(
    summary
) == (
    len(SEEDS)
    * 5
)

assert set(
    scaler[
        "fit_split"
    ]
) == {
    "train_only"
}

assert set(
    summary[
        "fit_split"
    ]
) == {
    "train_only"
}


write_csv(
    scaler,
    SCALER_OUT,
)

write_csv(
    summary,
    SUMMARY_OUT,
)


# =============================================================================
# Manifest
# =============================================================================

seed0 = assign[
    assign[
        "split_seed"
    ] == 0
]


seed0_dataset_counts = {}

for dataset in (
    "HUST",
    "CALCE",
    "MATR",
    "RWTH",
):

    seed0_dataset_counts[
        dataset
    ] = (
        seed0[
            seed0[
                "dataset"
            ] == dataset
        ][
            "split"
        ]
        .value_counts()
        .to_dict()
    )


manifest = {
    "contract_version":
        CONTRACT_VERSION,

    "split_seeds":
        SEEDS,

    "requested_ratios":
        RATIOS,

    "grouping_unit":
        "split_group / cell",

    "stratification":
        [
            "dataset",
            "cell-level censoring_status",
        ],

    "allocation_rule":
        (
            "largest remainder within each "
            "dataset-censoring stratum; stable tie order "
            "train, validation, test"
        ),

    "randomization":
        (
            "NumPy default_rng with SeedSequence "
            "derived from split_seed and SHA256(stratum)"
        ),

    "total_unique_groups":
        307,

    "supervised_unique_groups":
        259,

    "rwth_unique_groups":
        48,

    "cell_censoring_counts":
        {
            "HUST::observed_event":
                77,

            "CALCE::observed_event":
                13,

            "MATR::observed_event":
                88,

            "MATR::right_censored_len_plus_1":
                81,

            "RWTH::observed_event":
                48,
        },

    "seed0_dataset_group_counts":
        seed0_dataset_counts,

    "scaler_method":
        (
            "StandardScaler-equivalent population "
            "mean/variance; ddof=0; zero-variance scale=1"
        ),

    "scaler_fit_policy":
        "train groups only",

    "scaler_scopes":
        [
            "pooled_supervised",
            "HUST",
            "CALCE",
            "MATR",
            "RWTH_history",
        ],

    "scaled_data_materialized":
        False,

    "validation_statistics_used_for_scaler_fit":
        False,

    "test_statistics_used_for_scaler_fit":
        False,

    "lodo_scalers_created":
        False,

    "lodo_policy":
        (
            "fit later per held-out dataset using "
            "source-training datasets only; held-out "
            "dataset statistics must not enter scaler fit"
        ),

    "primary_features":
        FEATURES,

    "input_sha256":
        observed_input_hashes,

    "outputs": {
        "cell_split_assignments": {
            "path":
                str(
                    ASSIGN_OUT.relative_to(
                        BASE
                    )
                ),

            "sha256":
                sha256_file(
                    ASSIGN_OUT
                ),
        },

        "train_only_scaler_parameters": {
            "path":
                str(
                    SCALER_OUT.relative_to(
                        BASE
                    )
                ),

            "sha256":
                sha256_file(
                    SCALER_OUT
                ),
        },

        "train_only_scaler_fit_summary": {
            "path":
                str(
                    SUMMARY_OUT.relative_to(
                        BASE
                    )
                ),

            "sha256":
                sha256_file(
                    SUMMARY_OUT
                ),
        },
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


# =============================================================================
# Compact report
# =============================================================================

print("=" * 78)
print("GROUPED SPLIT + TRAIN-ONLY SCALER BUILDER")
print("=" * 78)

print(
    "split seeds:",
    SEEDS
)

print(
    "unique cell groups:",
    len(profile)
)

print(
    "cell censoring strata:",
    cell_censor_counts
)

print()
print(
    "seed0 group counts:"
)

for dataset in (
    "HUST",
    "CALCE",
    "MATR",
    "RWTH",
):
    print(
        f"  {dataset}:",
        seed0_dataset_counts[
            dataset
        ]
    )

print()
print(
    "MATR seed0 observed-event:",
    seed0[
        (
            seed0[
                "dataset"
            ] == "MATR"
        )
        &
        (
            seed0[
                "censoring_status"
            ] == "observed_event"
        )
    ][
        "split"
    ].value_counts().to_dict()
)

print(
    "MATR seed0 right-censored:",
    seed0[
        (
            seed0[
                "dataset"
            ] == "MATR"
        )
        &
        (
            seed0[
                "censoring_status"
            ]
            == "right_censored_len_plus_1"
        )
    ][
        "split"
    ].value_counts().to_dict()
)


print()
print(
    "seed0 scaler fit summary:"
)

for _, row in summary[
    summary[
        "split_seed"
    ] == 0
].iterrows():

    print(
        " ",
        row[
            "scaler_scope"
        ],
        "rows=",
        int(
            row[
                "n_train_rows"
            ]
        ),
        "groups=",
        int(
            row[
                "n_train_groups"
            ]
        ),
        "zero_var=",
        int(
            row[
                "zero_variance_features"
            ]
        ),
    )


print()
print(
    "all seeds cell-disjoint: True"
)

print(
    "scaler fit split: train_only"
)

print(
    "validation statistics used: False"
)

print(
    "test statistics used: False"
)

print(
    "LODO scaler created: False"
)

print(
    "scaled data materialized: False"
)

print()
print(
    "assignments SHA256:",
    sha256_file(
        ASSIGN_OUT
    )
)

print(
    "scaler parameters SHA256:",
    sha256_file(
        SCALER_OUT
    )
)

print(
    "scaler summary SHA256:",
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
    "GROUPED_SPLIT_SCALER_BUILDER_PASS=True"
)
