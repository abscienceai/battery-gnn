from pathlib import Path
import hashlib
import json
import os
import subprocess

import numpy as np
import pandas as pd


BASE = Path.home() / "alper/battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

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

COHORT_OUT = OUT / "common_cohort_rows.csv"
SCALER_OUT = OUT / "train_only_scaler_parameters.csv"
SUMMARY_OUT = OUT / "common_cohort_summary.csv"
MANIFEST_OUT = OUT / "common_cohort_scaler_manifest.json"

SELF = (
    REPO
    / "scripts/revision_2026/85_build_dqdv_common_cohort_scalers.py"
)

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

CONDITIONS = {
    "PRIMARY20_COMMON": {
        "dimension": 20,
        "blocks": [
            "PRIMARY20",
        ],
    },

    "PRIMARY20_PLUS_FULL_DQDV10": {
        "dimension": 30,
        "blocks": [
            "PRIMARY20",
            "FULL_DQDV10",
        ],
    },

    "PRIMARY20_PLUS_PREFIX25_DQDV10": {
        "dimension": 30,
        "blocks": [
            "PRIMARY20",
            "PREFIX25_DQDV10",
        ],
    },

    "PRIMARY20_PLUS_PREFIX50_DQDV10": {
        "dimension": 30,
        "blocks": [
            "PRIMARY20",
            "PREFIX50_DQDV10",
        ],
    },
}


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:

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

    if x.isna().any():
        raise ValueError(
            "Unparseable boolean column"
        )

    return x.astype(
        bool
    )


def atomic_csv(
    df,
    path,
):

    tmp = Path(
        str(path) + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
        float_format="%.17g",
    )

    os.replace(
        tmp,
        path,
    )


def atomic_json(
    obj,
    path,
):

    tmp = Path(
        str(path) + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def scaler_rows(
    frame,
    seed,
    block,
    feature_columns,
    feature_names,
):

    train = frame[
        frame[
            "split"
        ].astype(str) == "train"
    ]

    if len(
        train
    ) == 0:
        raise AssertionError(
            (
                seed,
                block,
                "empty training cohort",
            )
        )

    out = []

    for source_col, feature in zip(
        feature_columns,
        feature_names,
    ):

        values = pd.to_numeric(
            train[
                source_col
            ],
            errors="raise",
        ).to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(
            values
        ).all():
            raise AssertionError(
                (
                    seed,
                    block,
                    feature,
                    "nonfinite training feature",
                )
            )

        mean = float(
            np.mean(
                values
            )
        )

        variance = float(
            np.var(
                values,
                ddof=0,
            )
        )

        if variance < 0:
            raise AssertionError(
                "negative variance"
            )

        zero_variance = bool(
            variance == 0.0
        )

        scale = (
            1.0
            if zero_variance
            else float(
                np.sqrt(
                    variance
                )
            )
        )

        out.append({
            "split_seed":
                int(
                    seed
                ),

            "feature_block":
                block,

            "feature":
                feature,

            "source_column":
                source_col,

            "mean":
                mean,

            "variance":
                variance,

            "scale":
                scale,

            "zero_variance":
                zero_variance,

            "n_train_rows":
                int(
                    len(
                        train
                    )
                ),

            "fit_split":
                "train_only",
        })

    return out


# ------------------------------------------------------------------
# Frozen feature-bank anchor.
# ------------------------------------------------------------------

assert sha256(
    FEATURE_BANK_MANIFEST
) == EXPECTED_FEATURE_BANK_MANIFEST_SHA

fb_manifest = json.loads(
    FEATURE_BANK_MANIFEST.read_text(
        encoding="utf-8"
    )
)

assert fb_manifest[
    "status"
] == "FEATURE_GENERATION_COMPLETE_BEFORE_MODELING"

assert fb_manifest[
    "primary_valid_total_rows"
] == 319291

assert fb_manifest[
    "target_table_loaded"
] is False

assert fb_manifest[
    "capacity_values_read"
] is False

assert fb_manifest[
    "SOH_values_read"
] is False

assert fb_manifest[
    "models_trained"
] is False


# Builder itself must be committed and clean.
rel = str(
    SELF.relative_to(
        REPO
    )
)

subprocess.check_call(
    [
        "git",
        "ls-files",
        "--error-unmatch",
        rel,
    ],
    cwd=REPO,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

assert subprocess.run(
    [
        "git",
        "diff",
        "--quiet",
        "HEAD",
        "--",
        rel,
    ],
    cwd=REPO,
).returncode == 0


# ------------------------------------------------------------------
# Canonical SOH-eligible identity only.
# No SOH/capacity target values are read.
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

canonical_counts = (
    canon[
        "dataset"
    ]
    .value_counts()
    .to_dict()
)

assert canonical_counts == {
    "HUST": 146122,
    "CALCE": 14298,
    "MATR": 140133,
}

assert len(
    canon
) == 300553

identity_cols = [
    "dataset",
    "raw_cell_id",
    "raw_cycle_position_zero_based",
]

assert not canon[
    identity_cols
].duplicated().any()


# ------------------------------------------------------------------
# Grouped split assignments.
# ------------------------------------------------------------------

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

assert set(
    splits[
        "split_seed"
    ].unique()
) == set(
    SEEDS
)

assert not splits[
    [
        "split_seed",
        "dataset",
        "split_group",
    ]
].duplicated().any()


# ------------------------------------------------------------------
# Merge canonical rows with primary20 and target-independent dQ/dV.
# ------------------------------------------------------------------

parts = []

for dataset in DATASETS:

    raw_path = (
        RAW_FEATURE_DIR
        / f"{dataset}_raw_cycle_features.csv"
    )

    raw = pd.read_csv(
        raw_path,
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

    assert not raw[
        identity_cols
    ].duplicated().any()

    for f in PRIMARY_FEATURES:

        x = pd.to_numeric(
            raw[
                f
            ],
            errors="raise",
        ).to_numpy(
            dtype=float
        )

        assert np.isfinite(
            x
        ).all()


    bank_path = (
        FEATURE_BANK_DIR
        / f"{dataset}_dqdv_feature_bank.csv"
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
        bank_path,
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

    assert not bank[
        identity_cols
    ].duplicated().any()


    c = canon[
        canon[
            "dataset"
        ] == dataset
    ].copy()

    merged = c.merge(
        raw,
        on=identity_cols,
        how="left",
        validate="one_to_one",
    )

    assert not merged[
        PRIMARY_FEATURES
    ].isna().any().any()

    merged = merged.merge(
        bank,
        on=identity_cols,
        how="left",
        validate="one_to_one",
    )

    assert not merged[
        "full_valid"
    ].isna().any()

    parts.append(
        merged
    )


base = pd.concat(
    parts,
    ignore_index=True,
)

assert len(
    base
) == 300553

assert not base[
    identity_cols
].duplicated().any()


# ------------------------------------------------------------------
# Split-specific common cohort.
#
# A row is retained only if FULL, PREFIX25 and PREFIX50 dQ/dV
# are all valid for that split seed. The same retained row IDs are
# therefore used for all four model conditions.
# ------------------------------------------------------------------

cohort_parts = []
summary_rows = []
all_scaler_rows = []


for seed in SEEDS:

    split_seed = splits[
        splits[
            "split_seed"
        ] == seed
    ][
        [
            "dataset",
            "split_group",
            "split",
        ]
    ].copy()

    q = base.merge(
        split_seed,
        on=[
            "dataset",
            "split_group",
        ],
        how="left",
        validate="many_to_one",
    )

    if q[
        "split"
    ].isna().any():
        raise AssertionError(
            f"seed {seed}: missing split assignment"
        )

    full_valid = bool_series(
        q[
            "full_valid"
        ]
    )

    p25_valid = bool_series(
        q[
            f"s{seed}_p25_valid"
        ]
    )

    p50_valid = bool_series(
        q[
            f"s{seed}_p50_valid"
        ]
    )

    common = q[
        full_valid
        & p25_valid
        & p50_valid
    ].copy()

    if len(
        common
    ) == 0:
        raise AssertionError(
            f"seed {seed}: empty common cohort"
        )

    # All feature values needed by all four conditions must be finite.
    columns_to_check = list(
        PRIMARY_FEATURES
    )

    columns_to_check.extend(
        [
            "full_" + f
            for f in DQDV_FEATURES
        ]
    )

    columns_to_check.extend(
        [
            f"s{seed}_p25_{f}"
            for f in DQDV_FEATURES
        ]
    )

    columns_to_check.extend(
        [
            f"s{seed}_p50_{f}"
            for f in DQDV_FEATURES
        ]
    )

    for col in columns_to_check:

        values = pd.to_numeric(
            common[
                col
            ],
            errors="raise",
        ).to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():
            raise AssertionError(
                (
                    seed,
                    col,
                    "nonfinite common-cohort feature",
                )
            )


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

    assert not cohort[
        [
            "split_seed",
            *identity_cols,
        ]
    ].duplicated().any()

    cohort_parts.append(
        cohort
    )


    # Summary counts — no model outcomes.
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

    summary_rows.extend(
        grouped.to_dict(
            orient="records"
        )
    )

    summary_rows.append({
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


    # Primary20 scaler: once per split seed.
    all_scaler_rows.extend(
        scaler_rows(
            frame=common,
            seed=seed,
            block="PRIMARY20",
            feature_columns=PRIMARY_FEATURES,
            feature_names=PRIMARY_FEATURES,
        )
    )


    # Full dQ/dV10 scaler.
    all_scaler_rows.extend(
        scaler_rows(
            frame=common,
            seed=seed,
            block="FULL_DQDV10",
            feature_columns=[
                "full_" + f
                for f in DQDV_FEATURES
            ],
            feature_names=DQDV_FEATURES,
        )
    )


    # Prefix25 dQ/dV10 scaler.
    all_scaler_rows.extend(
        scaler_rows(
            frame=common,
            seed=seed,
            block="PREFIX25_DQDV10",
            feature_columns=[
                f"s{seed}_p25_{f}"
                for f in DQDV_FEATURES
            ],
            feature_names=DQDV_FEATURES,
        )
    )


    # Prefix50 dQ/dV10 scaler.
    all_scaler_rows.extend(
        scaler_rows(
            frame=common,
            seed=seed,
            block="PREFIX50_DQDV10",
            feature_columns=[
                f"s{seed}_p50_{f}"
                for f in DQDV_FEATURES
            ],
            feature_names=DQDV_FEATURES,
        )
    )


cohort_df = pd.concat(
    cohort_parts,
    ignore_index=True,
)

summary_df = pd.DataFrame(
    summary_rows
)

scaler_df = pd.DataFrame(
    all_scaler_rows
)


# 20 + 10 + 10 + 10 = 50 scaler parameters per seed.
assert len(
    scaler_df
) == (
    5 * 50
)

assert (
    scaler_df[
        "fit_split"
    ] == "train_only"
).all()

assert set(
    scaler_df[
        "feature_block"
    ]
) == {
    "PRIMARY20",
    "FULL_DQDV10",
    "PREFIX25_DQDV10",
    "PREFIX50_DQDV10",
}

assert not scaler_df[
    [
        "split_seed",
        "feature_block",
        "feature",
    ]
].duplicated().any()


# Common cohort must contain exactly one representation per
# seed/row, not separate condition-specific cohorts.
assert not cohort_df[
    [
        "split_seed",
        *identity_cols,
    ]
].duplicated().any()

assert set(
    cohort_df[
        "split_seed"
    ]
) == set(
    SEEDS
)


OUT.mkdir(
    parents=True,
    exist_ok=True,
)

atomic_csv(
    cohort_df,
    COHORT_OUT,
)

atomic_csv(
    scaler_df,
    SCALER_OUT,
)

atomic_csv(
    summary_df,
    SUMMARY_OUT,
)


manifest = {
    "contract":
        "dqdv_common_cohort_train_only_scalers_v1",

    "status":
        "COMMON_COHORT_AND_SCALERS_COMPLETE_BEFORE_MODELING",

    "feature_bank_manifest_sha256":
        EXPECTED_FEATURE_BANK_MANIFEST_SHA,

    "canonical_table_sha256":
        sha256(
            CANON
        ),

    "split_assignment_sha256":
        sha256(
            SPLITS
        ),

    "raw_primary_feature_sha256": {
        dataset:
            sha256(
                RAW_FEATURE_DIR
                / f"{dataset}_raw_cycle_features.csv"
            )
        for dataset in DATASETS
    },

    "feature_bank_shard_sha256": {
        dataset:
            sha256(
                FEATURE_BANK_DIR
                / f"{dataset}_dqdv_feature_bank.csv"
            )
        for dataset in DATASETS
    },

    "builder_sha256":
        sha256(
            SELF
        ),

    "builder_commit":
        subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=REPO,
            text=True,
        ).strip(),

    "datasets":
        DATASETS,

    "split_seeds":
        SEEDS,

    "canonical_soh_eligible_rows":
        300553,

    "primary_feature_names":
        PRIMARY_FEATURES,

    "dqdv_feature_names":
        DQDV_FEATURES,

    "common_cohort_definition":
        (
            "canonical SOH-eligible row with finite PRIMARY20 and "
            "valid FULL, split-specific PREFIX25, and split-specific "
            "PREFIX50 dQ/dV10 features"
        ),

    "same_row_ids_all_conditions":
        True,

    "conditions":
        CONDITIONS,

    "scaler_contract": {
        "primary20":
            "one train-only scaler per split seed, reused across all conditions",

        "dqdv":
            "separate train-only scaler per split seed and horizon",

        "variance_definition":
            "population_variance_ddof_0",

        "zero_variance_scale":
            1.0,

        "validation_rows_used":
            False,

        "test_rows_used":
            False,
    },

    "scaler_parameter_rows":
        int(
            len(
                scaler_df
            )
        ),

    "cohort_rows_all_seeds":
        int(
            len(
                cohort_df
            )
        ),

    "target_values_read":
        False,

    "capacity_values_read":
        False,

    "SOH_values_read":
        False,

    "RUL_or_EOL_values_read":
        False,

    "model_metrics_read":
        False,

    "model_states_loaded":
        False,

    "models_trained":
        False,

    "outputs": {
        "common_cohort_rows": {
            "path":
                str(
                    COHORT_OUT
                ),

            "sha256":
                sha256(
                    COHORT_OUT
                ),

            "rows":
                int(
                    len(
                        cohort_df
                    )
                ),
        },

        "train_only_scaler_parameters": {
            "path":
                str(
                    SCALER_OUT
                ),

            "sha256":
                sha256(
                    SCALER_OUT
                ),

            "rows":
                int(
                    len(
                        scaler_df
                    )
                ),
        },

        "common_cohort_summary": {
            "path":
                str(
                    SUMMARY_OUT
                ),

            "sha256":
                sha256(
                    SUMMARY_OUT
                ),

            "rows":
                int(
                    len(
                        summary_df
                    )
                ),
        },
    },
}


atomic_json(
    manifest,
    MANIFEST_OUT,
)


print(
    "DQDV COMMON COHORT / SCALER CLOSURE"
)

print(
    "canonical_soh_eligible_rows:",
    len(
        canon
    )
)

print(
    "split_seeds:",
    len(
        SEEDS
    )
)

print(
    "conditions:",
    len(
        CONDITIONS
    )
)

print(
    "same_row_ids_all_conditions:",
    True
)

print(
    "scaler_parameter_rows:",
    len(
        scaler_df
    )
)

print(
    "scalers_fit_train_only:",
    True
)

print(
    "validation_rows_used_for_scalers:",
    False
)

print(
    "test_rows_used_for_scalers:",
    False
)

print(
    "target_values_read:",
    False
)

print(
    "SOH_values_read:",
    False
)

print(
    "model_metrics_read:",
    False
)

print(
    "models_trained:",
    False
)

print(
    "manifest_sha256:",
    sha256(
        MANIFEST_OUT
    )
)

print(
    "DQDV_COMMON_COHORT_SCALERS_PASS=True"
)
