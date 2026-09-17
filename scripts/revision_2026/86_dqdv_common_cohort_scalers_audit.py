from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


BASE = Path.home() / "alper/battery_revision_2026"

ROOT = (
    BASE
    / "results/revision_2026/dqdv_leakage_control/final_dqdv_v1"
)

FEATURE_BANK_DIR = ROOT / "features_v1"

FEATURE_BANK_MANIFEST = (
    FEATURE_BANK_DIR
    / "dqdv_feature_bank_manifest.json"
)

CANON = (
    BASE
    / "data/processed/revision_2026/canonical_tasks"
    / "canonical_cycle_supervised_table.csv"
)

RAW_FEATURE_DIR = (
    BASE
    / "data/processed/revision_2026/raw_cycle_features"
)

SPLITS = (
    BASE
    / "data/processed/revision_2026/splits_and_scalers"
    / "cell_split_assignments.csv"
)

OUT = ROOT / "common_cohort_scalers_v1"

COHORT = OUT / "common_cohort_rows.csv"
SCALERS = OUT / "train_only_scaler_parameters.csv"
SUMMARY = OUT / "common_cohort_summary.csv"
MANIFEST = OUT / "common_cohort_scaler_manifest.json"

EXPECTED_FEATURE_BANK_MANIFEST_SHA = (
    "901d692f720a9e5166283f66b462580496192e9bab8cc1706453546c167021dc"
)

DATASETS = [
    "HUST",
    "CALCE",
    "MATR",
]

SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

PRIMARY_FEATURES = [
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

DQDV_FEATURES = [
    "ica_mean_Ah_per_V",
    "ica_std_Ah_per_V",
    "ica_q10_Ah_per_V",
    "ica_q25_Ah_per_V",
    "ica_median_Ah_per_V",
    "ica_q75_Ah_per_V",
    "ica_q90_Ah_per_V",
    "ica_max_Ah_per_V",
    "ica_area_Ah",
    "ica_peak_voltage_V",
]


def sha256(path):

    h = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as f:

        while True:

            b = f.read(
                1024 * 1024
            )

            if not b:
                break

            h.update(
                b
            )

    return h.hexdigest()


def bool_series(s):

    if pd.api.types.is_bool_dtype(
        s
    ):
        return s.astype(
            bool
        )

    x = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
    )

    assert not x.isna().any()

    return x.astype(
        bool
    )


assert sha256(
    FEATURE_BANK_MANIFEST
) == EXPECTED_FEATURE_BANK_MANIFEST_SHA

m = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)

assert m[
    "contract"
] == "dqdv_common_cohort_train_only_scalers_v1"

assert m[
    "status"
] == "COMMON_COHORT_AND_SCALERS_COMPLETE_BEFORE_MODELING"

assert m[
    "feature_bank_manifest_sha256"
] == EXPECTED_FEATURE_BANK_MANIFEST_SHA

assert m[
    "canonical_soh_eligible_rows"
] == 300553

assert m[
    "same_row_ids_all_conditions"
] is True

assert m[
    "scaler_contract"
][
    "validation_rows_used"
] is False

assert m[
    "scaler_contract"
][
    "test_rows_used"
] is False

assert m[
    "target_values_read"
] is False

assert m[
    "capacity_values_read"
] is False

assert m[
    "SOH_values_read"
] is False

assert m[
    "RUL_or_EOL_values_read"
] is False

assert m[
    "model_metrics_read"
] is False

assert m[
    "models_trained"
] is False


for info in m[
    "outputs"
].values():

    p = Path(
        info[
            "path"
        ]
    )

    assert p.is_file()

    assert sha256(
        p
    ) == info[
        "sha256"
    ]


cohort_saved = pd.read_csv(
    COHORT,
    low_memory=False,
)

scalers_saved = pd.read_csv(
    SCALERS,
    low_memory=False,
)

summary_saved = pd.read_csv(
    SUMMARY,
    low_memory=False,
)


assert len(
    scalers_saved
) == 250

assert (
    scalers_saved[
        "fit_split"
    ] == "train_only"
).all()

assert not scalers_saved[
    [
        "split_seed",
        "feature_block",
        "feature",
    ]
].duplicated().any()


# ------------------------------------------------------------------
# Independently rebuild canonical identity.
# ------------------------------------------------------------------

canon = pd.read_csv(
    CANON,
    usecols=[
        "dataset",
        "split_group",
        "soh_task_eligible",
        "feature_source_cell_id",
        "feature_raw_cycle_position_zero_based",
    ],
    low_memory=False,
)

canon = canon[
    canon[
        "dataset"
    ].isin(
        DATASETS
    )
].copy()

canon = canon[
    bool_series(
        canon[
            "soh_task_eligible"
        ]
    )
].copy()

canon = canon.rename(
    columns={
        "feature_source_cell_id":
            "raw_cell_id",

        "feature_raw_cycle_position_zero_based":
            "raw_cycle_position_zero_based",
    }
)

canon[
    "raw_cell_id"
] = canon[
    "raw_cell_id"
].astype(
    str
)

canon[
    "split_group"
] = canon[
    "split_group"
].astype(
    str
)

canon[
    "raw_cycle_position_zero_based"
] = pd.to_numeric(
    canon[
        "raw_cycle_position_zero_based"
    ],
    errors="raise",
).astype(
    int
)

assert len(
    canon
) == 300553


splits = pd.read_csv(
    SPLITS,
    usecols=[
        "split_seed",
        "split_group",
        "dataset",
        "split",
    ],
    low_memory=False,
)

splits = splits[
    splits[
        "dataset"
    ].isin(
        DATASETS
    )
].copy()

splits[
    "split_seed"
] = pd.to_numeric(
    splits[
        "split_seed"
    ],
    errors="raise",
).astype(
    int
)

splits[
    "split_group"
] = splits[
    "split_group"
].astype(
    str
)


identity_cols = [
    "dataset",
    "raw_cell_id",
    "raw_cycle_position_zero_based",
]


parts = []

for dataset in DATASETS:

    raw = pd.read_csv(
        RAW_FEATURE_DIR
        / f"{dataset}_raw_cycle_features.csv",
        usecols=[
            "dataset",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
            "feature_valid",
            *PRIMARY_FEATURES,
        ],
        low_memory=False,
    )

    raw[
        "raw_cell_id"
    ] = raw[
        "raw_cell_id"
    ].astype(
        str
    )

    raw[
        "raw_cycle_position_zero_based"
    ] = pd.to_numeric(
        raw[
            "raw_cycle_position_zero_based"
        ],
        errors="raise",
    ).astype(
        int
    )

    raw = raw[
        bool_series(
            raw[
                "feature_valid"
            ]
        )
    ].drop(
        columns=[
            "feature_valid",
        ]
    )


    bank_cols = [
        "dataset",
        "raw_cell_id",
        "raw_cycle_position_zero_based",
        "full_valid",
    ]

    bank_cols.extend(
        [
            "full_" + f
            for f in DQDV_FEATURES
        ]
    )

    for seed in SEEDS:

        bank_cols.extend(
            [
                f"s{seed}_p25_valid",
                f"s{seed}_p50_valid",
            ]
        )

        bank_cols.extend(
            [
                f"s{seed}_p25_{f}"
                for f in DQDV_FEATURES
            ]
        )

        bank_cols.extend(
            [
                f"s{seed}_p50_{f}"
                for f in DQDV_FEATURES
            ]
        )


    bank = pd.read_csv(
        FEATURE_BANK_DIR
        / f"{dataset}_dqdv_feature_bank.csv",
        usecols=bank_cols,
        low_memory=False,
    )

    bank[
        "raw_cell_id"
    ] = bank[
        "raw_cell_id"
    ].astype(
        str
    )

    bank[
        "raw_cycle_position_zero_based"
    ] = pd.to_numeric(
        bank[
            "raw_cycle_position_zero_based"
        ],
        errors="raise",
    ).astype(
        int
    )


    c = canon[
        canon[
            "dataset"
        ] == dataset
    ].copy()

    x = c.merge(
        raw,
        on=identity_cols,
        how="left",
        validate="one_to_one",
    )

    x = x.merge(
        bank,
        on=identity_cols,
        how="left",
        validate="one_to_one",
    )

    assert len(
        x
    ) == len(
        c
    )

    parts.append(
        x
    )


base = pd.concat(
    parts,
    ignore_index=True,
)

assert len(
    base
) == 300553


# ------------------------------------------------------------------
# Independently reconstruct exact common-cohort row IDs and scalers.
# ------------------------------------------------------------------

cohort_calc_parts = []
scaler_calc_rows = []
summary_calc_rows = []


for seed in SEEDS:

    ss = splits[
        splits[
            "split_seed"
        ] == seed
    ][
        [
            "dataset",
            "split_group",
            "split",
        ]
    ]

    q = base.merge(
        ss,
        on=[
            "dataset",
            "split_group",
        ],
        how="left",
        validate="many_to_one",
    )

    assert not q[
        "split"
    ].isna().any()

    mask = (
        bool_series(
            q[
                "full_valid"
            ]
        )
        & bool_series(
            q[
                f"s{seed}_p25_valid"
            ]
        )
        & bool_series(
            q[
                f"s{seed}_p50_valid"
            ]
        )
    )

    common = q[
        mask
    ].copy()


    cohort = common[
        [
            "dataset",
            "split_group",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
            "split",
        ]
    ].copy()

    cohort.insert(
        0,
        "split_seed",
        seed,
    )

    cohort[
        "common_cohort"
    ] = True

    cohort_calc_parts.append(
        cohort
    )


    grouped = (
        cohort
        .groupby(
            [
                "dataset",
                "split",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="n_rows"
        )
    )

    grouped.insert(
        0,
        "split_seed",
        seed,
    )

    summary_calc_rows.extend(
        grouped.to_dict(
            orient="records"
        )
    )

    summary_calc_rows.append({
        "split_seed":
            seed,

        "dataset":
            "ALL",

        "split":
            "ALL",

        "n_rows":
            int(
                len(
                    cohort
                )
            ),
    })


    train = common[
        common[
            "split"
        ].astype(str) == "train"
    ]

    assert len(
        train
    ) > 0


    blocks = [
        (
            "PRIMARY20",
            PRIMARY_FEATURES,
            PRIMARY_FEATURES,
        ),

        (
            "FULL_DQDV10",
            [
                "full_" + f
                for f in DQDV_FEATURES
            ],
            DQDV_FEATURES,
        ),

        (
            "PREFIX25_DQDV10",
            [
                f"s{seed}_p25_{f}"
                for f in DQDV_FEATURES
            ],
            DQDV_FEATURES,
        ),

        (
            "PREFIX50_DQDV10",
            [
                f"s{seed}_p50_{f}"
                for f in DQDV_FEATURES
            ],
            DQDV_FEATURES,
        ),
    ]


    for block, source_cols, names in blocks:

        for source_col, feature in zip(
            source_cols,
            names,
        ):

            values = pd.to_numeric(
                train[
                    source_col
                ],
                errors="raise",
            ).to_numpy(
                dtype=np.float64
            )

            assert np.isfinite(
                values
            ).all()

            variance = float(
                np.var(
                    values,
                    ddof=0,
                )
            )

            zero = bool(
                variance == 0.0
            )

            scaler_calc_rows.append({
                "split_seed":
                    seed,

                "feature_block":
                    block,

                "feature":
                    feature,

                "source_column":
                    source_col,

                "mean":
                    float(
                        np.mean(
                            values
                        )
                    ),

                "variance":
                    variance,

                "scale":
                    (
                        1.0
                        if zero
                        else float(
                            np.sqrt(
                                variance
                            )
                        )
                    ),

                "zero_variance":
                    zero,

                "n_train_rows":
                    int(
                        len(
                            train
                        )
                    ),

                "fit_split":
                    "train_only",
            })


cohort_calc = pd.concat(
    cohort_calc_parts,
    ignore_index=True,
)

scalers_calc = pd.DataFrame(
    scaler_calc_rows
)

summary_calc = pd.DataFrame(
    summary_calc_rows
)


# ------------------------------------------------------------------
# Exact row-set comparison.
# ------------------------------------------------------------------

cohort_keys = [
    "split_seed",
    "dataset",
    "split_group",
    "raw_cell_id",
    "raw_cycle_position_zero_based",
    "split",
]

for df in [
    cohort_saved,
    cohort_calc,
]:

    df[
        "split_seed"
    ] = pd.to_numeric(
        df[
            "split_seed"
        ],
        errors="raise",
    ).astype(
        int
    )

    df[
        "raw_cell_id"
    ] = df[
        "raw_cell_id"
    ].astype(
        str
    )

    df[
        "split_group"
    ] = df[
        "split_group"
    ].astype(
        str
    )

    df[
        "raw_cycle_position_zero_based"
    ] = pd.to_numeric(
        df[
            "raw_cycle_position_zero_based"
        ],
        errors="raise",
    ).astype(
        int
    )


left = cohort_saved[
    cohort_keys
].merge(
    cohort_calc[
        cohort_keys
    ],
    on=cohort_keys,
    how="outer",
    validate="one_to_one",
    indicator=True,
)

assert (
    left[
        "_merge"
    ] == "both"
).all()

assert len(
    cohort_saved
) == len(
    cohort_calc
)


# ------------------------------------------------------------------
# Independent scaler reproduction.
# ------------------------------------------------------------------

keys = [
    "split_seed",
    "feature_block",
    "feature",
    "source_column",
]

z = scalers_saved.merge(
    scalers_calc,
    on=keys,
    how="outer",
    suffixes=(
        "_saved",
        "_calc",
    ),
    validate="one_to_one",
    indicator=True,
)

assert (
    z[
        "_merge"
    ] == "both"
).all()

assert len(
    z
) == 250

for col in [
    "mean",
    "variance",
    "scale",
]:

    assert np.allclose(
        pd.to_numeric(
            z[
                col
                + "_saved"
            ],
            errors="raise",
        ),
        pd.to_numeric(
            z[
                col
                + "_calc"
            ],
            errors="raise",
        ),
        rtol=1e-12,
        atol=1e-14,
    )

assert (
    pd.to_numeric(
        z[
            "n_train_rows_saved"
        ],
        errors="raise",
    )
    ==
    pd.to_numeric(
        z[
            "n_train_rows_calc"
        ],
        errors="raise",
    )
).all()

assert (
    z[
        "fit_split_saved"
    ] == "train_only"
).all()

assert (
    z[
        "fit_split_calc"
    ] == "train_only"
).all()


# ------------------------------------------------------------------
# Summary reproduction.
# ------------------------------------------------------------------

summary_keys = [
    "split_seed",
    "dataset",
    "split",
    "n_rows",
]

a = summary_saved[
    summary_keys
].copy()

b = summary_calc[
    summary_keys
].copy()

a[
    "split_seed"
] = pd.to_numeric(
    a[
        "split_seed"
    ],
    errors="raise",
).astype(
    int
)

b[
    "split_seed"
] = pd.to_numeric(
    b[
        "split_seed"
    ],
    errors="raise",
).astype(
    int
)

a[
    "n_rows"
] = pd.to_numeric(
    a[
        "n_rows"
    ],
    errors="raise",
).astype(
    int
)

b[
    "n_rows"
] = pd.to_numeric(
    b[
        "n_rows"
    ],
    errors="raise",
).astype(
    int
)

summary_join = a.merge(
    b,
    on=summary_keys,
    how="outer",
    validate="one_to_one",
    indicator=True,
)

assert (
    summary_join[
        "_merge"
    ] == "both"
).all()


# Every retained seed/row appears once only.
assert not cohort_saved[
    [
        "split_seed",
        "dataset",
        "raw_cell_id",
        "raw_cycle_position_zero_based",
    ]
].duplicated().any()


print(
    "DQDV COMMON COHORT / SCALER AUDIT"
)

print(
    "feature bank anchor exact:",
    True
)

print(
    "canonical SOH-eligible identity rows:",
    len(
        canon
    )
)

print(
    "common cohort independently reconstructed:",
    True
)

print(
    "same row IDs all four conditions:",
    True
)

print(
    "split assignments independently applied:",
    True
)

print(
    "scaler parameter rows:",
    len(
        scalers_saved
    )
)

print(
    "scalers independently reproduced:",
    True
)

print(
    "scalers fit from training rows only:",
    True
)

print(
    "validation/test rows used for scaler fit:",
    False
)

print(
    "target/capacity/SOH/RUL/EOL values read:",
    False
)

print(
    "model metrics read:",
    False
)

print(
    "models trained:",
    False
)

print(
    "manifest SHA256:",
    sha256(
        MANIFEST
    )
)

print(
    "DQDV_COMMON_COHORT_SCALERS_AUDIT_PASS=True"
)
