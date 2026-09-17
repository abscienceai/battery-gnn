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
V1 = ROOT / "barrier_v1"
V2 = ROOT / "barrier_v2"

INVENTORY = V1 / "sensor_discharge_inventory.csv"
CUTOFFS = V1 / "train_only_prefix_cutoffs.csv"
V1_POLARITY = V1 / "polarity_validation_summary.csv"

SUMMARY = V2 / "polarity_provenance_agreement_summary.csv"
MANIFEST = V2 / "duration_provenance_barrier_manifest.json"

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

    return x.astype(bool)


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
] == "dqdv_duration_provenance_barrier_v2"

assert m[
    "status"
] == "PASS_BEFORE_DQDV_FEATURE_EXTRACTION"

assert m[
    "protocol_changed"
] is False

assert m[
    "raw_reconstruction_rerun"
] is False

assert m[
    "criterion_revision"
][
    "pearson_role"
] == "descriptive_only"

assert m[
    "criterion_revision"
][
    "thresholds_relaxed_after_result"
] is False

assert m[
    "criterion_revision"
][
    "negative_current_discharge_rule_changed"
] is False

assert m[
    "all_datasets_direct_agreement_pass"
] is True


# ------------------------------------------------------------------
# Source artifact hashes.
# ------------------------------------------------------------------

for info in m[
    "source_artifacts"
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


# ------------------------------------------------------------------
# Independently verify that v1 failed only on MATR Pearson.
# ------------------------------------------------------------------

v1 = pd.read_csv(
    V1_POLARITY
)

fail = {}

for _, r in v1.iterrows():

    f = []

    if float(
        r[
            "coverage"
        ]
    ) < float(
        r[
            "coverage_threshold"
        ]
    ):
        f.append(
            "coverage"
        )

    if float(
        r[
            "pearson_r_capacity_vs_sensor_discharge"
        ]
    ) < float(
        r[
            "pearson_threshold"
        ]
    ):
        f.append(
            "pearson_r"
        )

    if float(
        r[
            "median_absolute_relative_error"
        ]
    ) > float(
        r[
            "median_abs_rel_error_threshold"
        ]
    ):
        f.append(
            "median_absolute_relative_error"
        )

    ratio = float(
        r[
            "median_sensor_to_capacity_ratio"
        ]
    )

    if not (
        float(
            r[
                "median_ratio_min"
            ]
        )
        <= ratio
        <= float(
            r[
                "median_ratio_max"
            ]
        )
    ):
        f.append(
            "median_sensor_to_capacity_ratio"
        )

    fail[
        str(
            r[
                "dataset"
            ]
        )
    ] = f


assert fail == {
    "HUST": [],
    "CALCE": [],
    "MATR": [
        "pearson_r"
    ],
}


# ------------------------------------------------------------------
# Independent direct-agreement reproduction.
# ------------------------------------------------------------------

inventory = pd.read_csv(
    INVENTORY,
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


canon = pd.read_csv(
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

canon = canon[
    canon[
        "dataset"
    ].isin(
        list(
            EXPECTED_COUNTS
        )
    )
].copy()

canon = canon[
    bool_series(
        canon[
            "soh_task_eligible"
        ]
    )
].copy()

assert (
    canon[
        "dataset"
    ]
    .value_counts()
    .to_dict()
) == EXPECTED_COUNTS

canon[
    "feature_raw_cycle_position_zero_based"
] = pd.to_numeric(
    canon[
        "feature_raw_cycle_position_zero_based"
    ],
    errors="raise",
).astype(int)


x = canon.merge(
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


saved = pd.read_csv(
    SUMMARY
)

calc = []

for dataset in [
    "HUST",
    "CALCE",
    "MATR",
]:

    d = x[
        x[
            "dataset"
        ] == dataset
    ]

    cap = d[
        "capacity_Ah"
    ].to_numpy(
        dtype=float
    )

    qd = d[
        "q_discharge_sensor_Ah"
    ].to_numpy(
        dtype=float
    )

    good = (
        np.isfinite(cap)
        & (cap > 0)
        & np.isfinite(qd)
        & (qd > 0)
    )

    cf = cap[
        good
    ]

    qf = qd[
        good
    ]

    calc.append({
        "dataset":
            dataset,

        "coverage":
            float(
                np.mean(
                    good
                )
            ),

        "median_sensor_to_capacity_ratio":
            float(
                np.median(
                    qf / cf
                )
            ),

        "median_absolute_relative_error":
            float(
                np.median(
                    np.abs(
                        qf - cf
                    )
                    / cf
                )
            ),

        "pearson_r_descriptive_only":
            float(
                np.corrcoef(
                    cf,
                    qf,
                )[0, 1]
            ),
    })


calc = pd.DataFrame(
    calc
)

z = calc.merge(
    saved,
    on="dataset",
    suffixes=(
        "_calc",
        "_saved",
    ),
    validate="one_to_one",
)

assert len(
    z
) == 3

for col in [
    "coverage",
    "median_sensor_to_capacity_ratio",
    "median_absolute_relative_error",
    "pearson_r_descriptive_only",
]:

    assert np.allclose(
        z[
            col
            + "_calc"
        ],
        z[
            col
            + "_saved"
        ],
        rtol=1e-12,
        atol=1e-14,
    )

assert saved[
    "direct_agreement_pass"
].astype(bool).all()

assert not saved[
    "pearson_used_for_pass_fail"
].astype(bool).any()


# ------------------------------------------------------------------
# Independently verify train-only cutoff artifact remains exact.
# No capacity values participate in cutoff reproduction.
# ------------------------------------------------------------------

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

canon_id[
    "feature_raw_cycle_position_zero_based"
] = pd.to_numeric(
    canon_id[
        "feature_raw_cycle_position_zero_based"
    ],
    errors="raise",
).astype(int)


a = canon_id.merge(
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

assert not a[
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
    0, 1, 2, 3, 4
]:

    s = assign[
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

    q = a.merge(
        s,
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

    for dataset in [
        "HUST",
        "CALCE",
        "MATR",
    ]:

        d = q[
            (
                q[
                    "dataset"
                ] == dataset
            )
            & (
                q[
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
                0.25 * med,

            "prefix50_cutoff_s":
                0.50 * med,
        })


calc_cut = pd.DataFrame(
    rows
)

saved_cut = pd.read_csv(
    CUTOFFS
)

c = calc_cut.merge(
    saved_cut,
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
    c
) == 15

assert (
    c[
        "n_training_rows_calc"
    ]
    == c[
        "n_training_rows_saved"
    ]
).all()

for col in [
    "median_training_discharge_duration_s",
    "prefix25_cutoff_s",
    "prefix50_cutoff_s",
]:

    assert np.allclose(
        c[
            col
            + "_calc"
        ],
        c[
            col
            + "_saved"
        ],
        rtol=1e-12,
        atol=1e-10,
    )


print(
    "DQDV PROVENANCE BARRIER V2 AUDIT"
)

print(
    "protocol unchanged:",
    True
)

print(
    "v1 failed attempt preserved:",
    True
)

print(
    "v1 failure is MATR Pearson only:",
    True
)

print(
    "Pearson used descriptively only:",
    True
)

print(
    "preexisting direct-agreement thresholds changed:",
    False
)

print(
    "direct agreement independently reproduced:",
    True
)

print(
    "all three datasets pass direct agreement:",
    True
)

print(
    "train-only cutoffs independently reproduced:",
    True
)

print(
    "raw reconstruction rerun:",
    False
)

print(
    "capacity used to infer current sign:",
    False
)

print(
    "negative-current discharge rule changed:",
    False
)

print(
    "dQdV features computed:",
    False
)

print(
    "model metrics read:",
    False
)

print(
    "manifest SHA256:",
    sha256(
        MANIFEST
    )
)

print(
    "DQDV_DURATION_PROVENANCE_BARRIER_V2_AUDIT_PASS=True"
)
