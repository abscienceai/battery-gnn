from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


BASE = Path.home() / "alper" / "battery_revision_2026"

ROOT = (
    BASE
    / "results/revision_2026/dqdv_leakage_control/final_dqdv_v1"
)

PROTOCOL = ROOT / "protocol.json"
OUT = ROOT / "barrier_v1"

MANIFEST = OUT / "duration_polarity_barrier_manifest.json"

CANON = (
    BASE
    / "data/processed/revision_2026/canonical_tasks"
    / "canonical_cycle_supervised_table.csv"
)

ASSIGN = (
    BASE
    / "data/processed/revision_2026/splits_and_scalers"
    / "cell_split_assignments.csv"
)

EXPECTED_PROTOCOL_SHA = (
    "c455bf7a46f89649f25769a43abd443f171545b06e4740c799df1842c8bbb8e9"
)

EXPECTED_COUNTS = {
    "HUST": 146122,
    "CALCE": 14298,
    "MATR": 140133,
}

EXPECTED_TOTAL = 300553


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def bool_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)

    m = (
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

    assert not m.isna().any()

    return m.astype(bool)


assert sha256(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

m = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)

assert m[
    "contract"
] == "dqdv_duration_polarity_barrier_v1"

assert m[
    "status"
] == "COMPLETE_BEFORE_DQDV_FEATURE_EXTRACTION"

assert m[
    "protocol_sha256"
] == EXPECTED_PROTOCOL_SHA

assert m[
    "canonical_soh_alignment_rows"
] == EXPECTED_TOTAL

assert m[
    "canonical_soh_alignment_by_dataset"
] == EXPECTED_COUNTS

assert m[
    "cutoff_rows"
] == 15

assert m[
    "cutoff_fit_scope"
] == "train_only_dataset_x_split_seed"

assert m[
    "validation_or_test_duration_used_for_cutoff"
] is False

assert m[
    "individual_test_cycle_endpoint_used_for_cutoff"
] is False

assert m[
    "capacity_values_read"
] is True

assert m[
    "capacity_values_used_only_after_cutoff_fitting"
] is True

assert m[
    "capacity_values_passed_to_sensor_feature_functions"
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
    "model_states_loaded"
] is False

assert m[
    "dqdv_features_computed"
] is False

assert m[
    "models_trained"
] is False

assert m[
    "numeric_validation_values_surfaced_during_execution"
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

    df = pd.read_csv(
        p,
        low_memory=False,
    )

    assert len(
        df
    ) == int(
        info[
            "rows"
        ]
    )


inventory = pd.read_csv(
    m[
        "outputs"
    ][
        "sensor_inventory"
    ][
        "path"
    ],
    low_memory=False,
)

inventory[
    "raw_cycle_position_zero_based"
] = pd.to_numeric(
    inventory[
        "raw_cycle_position_zero_based"
    ],
    errors="raise",
).astype(int)


# ----------------------------------------------------------------------
# Independent cutoff reproduction from sensor inventory.
# No capacity values are used here.
# ----------------------------------------------------------------------

canon_id = pd.read_csv(
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

canon_id = canon_id[
    canon_id[
        "dataset"
    ].isin(
        list(
            EXPECTED_COUNTS
        )
    )
].copy()

canon_id = canon_id[
    bool_series(
        canon_id[
            "soh_task_eligible"
        ]
    )
].copy()

assert (
    canon_id[
        "dataset"
    ]
    .value_counts()
    .to_dict()
) == EXPECTED_COUNTS

canon_id[
    "feature_raw_cycle_position_zero_based"
] = pd.to_numeric(
    canon_id[
        "feature_raw_cycle_position_zero_based"
    ],
    errors="raise",
).astype(int)


aligned = canon_id.merge(
    inventory[
        [
            "dataset",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
            "discharge_duration_s",
        ]
    ],
    left_on=[
        "dataset",
        "feature_source_cell_id",
        "feature_raw_cycle_position_zero_based",
    ],
    right_on=[
        "dataset",
        "raw_cell_id",
        "raw_cycle_position_zero_based",
    ],
    how="left",
    validate="one_to_one",
)

assert len(
    aligned
) == EXPECTED_TOTAL

assert not aligned[
    "discharge_duration_s"
].isna().any()


assign = pd.read_csv(
    ASSIGN,
    usecols=[
        "split_seed",
        "split_group",
        "dataset",
        "split",
    ],
    low_memory=False,
)

assign[
    "split_seed"
] = pd.to_numeric(
    assign[
        "split_seed"
    ],
    errors="raise",
).astype(int)


rows = []


for seed in [
    0,
    1,
    2,
    3,
    4,
]:

    a = assign[
        (
            assign[
                "split_seed"
            ] == seed
        )
        & assign[
            "dataset"
        ].isin(
            list(
                EXPECTED_COUNTS
            )
        )
    ]

    x = aligned.merge(
        a,
        on=[
            "dataset",
            "split_group",
        ],
        how="left",
        validate="many_to_one",
    )

    assert not x[
        "split"
    ].isna().any()

    for dataset in [
        "HUST",
        "CALCE",
        "MATR",
    ]:

        d = x[
            (
                x[
                    "dataset"
                ] == dataset
            )
            & (
                x[
                    "split"
                ] == "train"
            )
        ]

        med = float(
            np.median(
                d[
                    "discharge_duration_s"
                ]
            )
        )

        rows.append({
            "split_seed":
                seed,

            "dataset":
                dataset,

            "n_training_rows":
                len(
                    d
                ),

            "median_training_discharge_duration_s":
                med,

            "prefix25_cutoff_s":
                0.25
                * med,

            "prefix50_cutoff_s":
                0.50
                * med,
        })


calc_cutoff = pd.DataFrame(
    rows
)

saved_cutoff = pd.read_csv(
    m[
        "outputs"
    ][
        "train_only_prefix_cutoffs"
    ][
        "path"
    ]
)

check = calc_cutoff.merge(
    saved_cutoff,
    on=[
        "split_seed",
        "dataset",
    ],
    suffixes=(
        "_calc",
        "_saved",
    ),
    validate="one_to_one",
)

assert len(
    check
) == 15

assert (
    check[
        "n_training_rows_calc"
    ]
    == check[
        "n_training_rows_saved"
    ]
).all()

for col in [
    "median_training_discharge_duration_s",
    "prefix25_cutoff_s",
    "prefix50_cutoff_s",
]:

    assert np.allclose(
        check[
            col
            + "_calc"
        ],
        check[
            col
            + "_saved"
        ],
        rtol=1e-12,
        atol=1e-10,
    )


# ----------------------------------------------------------------------
# Independent polarity-summary reproduction.
# Capacity values are read ONLY here.
# ----------------------------------------------------------------------

canon_cap = pd.read_csv(
    CANON,
    usecols=[
        "dataset",
        "soh_task_eligible",
        "capacity_Ah",
        "feature_source_cell_id",
        "feature_raw_cycle_position_zero_based",
    ],
    low_memory=False,
)

canon_cap = canon_cap[
    canon_cap[
        "dataset"
    ].isin(
        list(
            EXPECTED_COUNTS
        )
    )
].copy()

canon_cap = canon_cap[
    bool_series(
        canon_cap[
            "soh_task_eligible"
        ]
    )
].copy()

canon_cap[
    "feature_raw_cycle_position_zero_based"
] = pd.to_numeric(
    canon_cap[
        "feature_raw_cycle_position_zero_based"
    ],
    errors="raise",
).astype(int)


pol = canon_cap.merge(
    inventory[
        [
            "dataset",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
            "q_discharge_sensor_Ah",
        ]
    ],
    left_on=[
        "dataset",
        "feature_source_cell_id",
        "feature_raw_cycle_position_zero_based",
    ],
    right_on=[
        "dataset",
        "raw_cell_id",
        "raw_cycle_position_zero_based",
    ],
    how="left",
    validate="one_to_one",
)


saved_pol = pd.read_csv(
    m[
        "outputs"
    ][
        "polarity_validation_summary"
    ][
        "path"
    ]
)


thresholds = m[
    "polarity_thresholds"
]

calc_rows = []


for dataset in [
    "HUST",
    "CALCE",
    "MATR",
]:

    d = pol[
        pol[
            "dataset"
        ] == dataset
    ]

    cap = pd.to_numeric(
        d[
            "capacity_Ah"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    qd = pd.to_numeric(
        d[
            "q_discharge_sensor_Ah"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    finite = (
        np.isfinite(cap)
        & (cap > 0)
        & np.isfinite(qd)
        & (qd > 0)
    )

    cf = cap[
        finite
    ]

    qf = qd[
        finite
    ]

    coverage = float(
        np.mean(
            finite
        )
    )

    pearson = float(
        np.corrcoef(
            cf,
            qf,
        )[0, 1]
    )

    ratio = qf / cf

    are = np.abs(
        qf - cf
    ) / cf

    median_ratio = float(
        np.median(
            ratio
        )
    )

    median_are = float(
        np.median(
            are
        )
    )

    passed = bool(
        coverage
        >= thresholds[
            "minimum_coverage"
        ]

        and pearson
        >= thresholds[
            "minimum_pearson_r"
        ]

        and median_are
        <= thresholds[
            "maximum_median_absolute_relative_error"
        ]

        and median_ratio
        >= thresholds[
            "median_ratio_min"
        ]

        and median_ratio
        <= thresholds[
            "median_ratio_max"
        ]
    )

    calc_rows.append({
        "dataset":
            dataset,

        "coverage":
            coverage,

        "pearson_r_capacity_vs_sensor_discharge":
            pearson,

        "median_sensor_to_capacity_ratio":
            median_ratio,

        "median_absolute_relative_error":
            median_are,

        "barrier_pass":
            passed,
    })


calc_pol = pd.DataFrame(
    calc_rows
)

pcheck = calc_pol.merge(
    saved_pol[
        [
            "dataset",
            "coverage",
            "pearson_r_capacity_vs_sensor_discharge",
            "median_sensor_to_capacity_ratio",
            "median_absolute_relative_error",
            "barrier_pass",
        ]
    ],
    on="dataset",
    suffixes=(
        "_calc",
        "_saved",
    ),
    validate="one_to_one",
)

assert len(
    pcheck
) == 3

for col in [
    "coverage",
    "pearson_r_capacity_vs_sensor_discharge",
    "median_sensor_to_capacity_ratio",
    "median_absolute_relative_error",
]:

    assert np.allclose(
        pcheck[
            col
            + "_calc"
        ],
        pcheck[
            col
            + "_saved"
        ],
        rtol=1e-12,
        atol=1e-14,
    )

assert (
    pcheck[
        "barrier_pass_calc"
    ].astype(bool)
    == pcheck[
        "barrier_pass_saved"
    ].astype(bool)
).all()

assert pcheck[
    "barrier_pass_calc"
].astype(bool).all()


print("=" * 84)
print("DQDV DURATION / POLARITY BARRIER AUDIT")
print("=" * 84)

print(
    "protocol exact:",
    True
)

print(
    "sensor inventory rows:",
    len(
        inventory
    )
)

print(
    "canonical SOH alignment rows:",
    EXPECTED_TOTAL
)

print(
    "cutoff rows:",
    15
)

print(
    "cutoffs independently reproduced:",
    True
)

print(
    "cutoff fit uses train only:",
    True
)

print(
    "validation/test duration used:",
    False
)

print(
    "individual test-cycle endpoint used:",
    False
)

print(
    "polarity datasets:",
    3
)

print(
    "polarity summaries independently reproduced:",
    True
)

print(
    "polarity barrier pass:",
    True
)

print(
    "capacity values passed to sensor feature functions:",
    False
)

print(
    "SOH values read:",
    False
)

print(
    "model metrics read:",
    False
)

print(
    "dQdV features computed:",
    False
)

print(
    "manifest SHA256:",
    sha256(
        MANIFEST
    )
)

print()
print(
    "DQDV_DURATION_POLARITY_BARRIER_AUDIT_PASS=True"
)
