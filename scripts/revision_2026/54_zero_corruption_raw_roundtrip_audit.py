from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


BASE = Path.home() / "alper" / "battery_revision_2026"

ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "physical_sensor_noise"
    / "final_physical_v1"
)

ROUNDTRIP = ROOT / "zero_corruption_roundtrip"

PROTOCOL = ROOT / "protocol.json"
SUMMARY = ROUNDTRIP / "roundtrip_summary.json"
ROWS = ROUNDTRIP / "roundtrip_row_audit.csv"

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

EXPECTED_PROTOCOL = (
    "7ae0836fe69c4b14a1f93a094b2d42bb96209e302c8262f7b8bd8625baf78bfc"
)

EXPECTED_CANON = (
    "49a7b7629cf9ee3f8ce0a7e3c52f2807878b752b4c9a30a876285fb891b00c14"
)

EXPECTED_ASSIGNMENTS = (
    "914c8b90131f496ca396f63e0502147ca99848fdfd4b2c6b2a5887225b47d055"
)


def sha(path):
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
        s.fillna(False)
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


assert sha(
    PROTOCOL
) == EXPECTED_PROTOCOL

assert sha(
    CANON
) == EXPECTED_CANON

assert sha(
    ASSIGNMENTS
) == EXPECTED_ASSIGNMENTS


summary = json.loads(
    SUMMARY.read_text(
        encoding="utf-8"
    )
)


assert summary[
    "contract"
] == "zero_corruption_raw_roundtrip_v1"

assert summary[
    "status"
] == "PASS"

assert summary[
    "protocol_sha256"
] == EXPECTED_PROTOCOL

assert summary[
    "canonical_sha256"
] == EXPECTED_CANON

assert summary[
    "assignments_sha256"
] == EXPECTED_ASSIGNMENTS

assert summary[
    "absolute_tolerance"
] == 1e-8

assert summary[
    "relative_tolerance"
] == 1e-8

assert summary[
    "source_member_all_match"
] is True

assert summary[
    "raw_cycle_key_all_match"
] is True

assert summary[
    "feature_contract_all_match"
] is True

assert summary[
    "kernel_hash_all_match"
] is True

assert summary[
    "all_20_features_close"
] is True

assert summary[
    "all_rows_pass"
] is True

assert summary[
    "failed_rows"
] == 0

assert summary[
    "model_evaluated"
] is False

assert summary[
    "physical_noise_generated"
] is False

assert summary[
    "physical_noise_test_metrics_evaluated"
] is False

assert sha(
    ROWS
) == summary[
    "row_audit_sha256"
]


rows = pd.read_csv(
    ROWS,
    low_memory=False,
)


assert len(
    rows
) == summary[
    "union_test_rows_expected"
]

assert len(
    rows
) == summary[
    "union_test_rows_audited"
]


for col in [
    "source_member_match",
    "raw_cycle_key_match",
    "feature_contract_match",
    "kernel_hash_match",
    "all_20_features_close",
    "roundtrip_pass",
]:

    assert as_bool(
        rows[
            col
        ]
    ).all(), col


ratio = pd.to_numeric(
    rows[
        "row_max_tolerance_ratio"
    ],
    errors="raise",
).to_numpy(
    dtype=float
)


assert np.isfinite(
    ratio
).all()

assert float(
    ratio.max()
) <= (
    1.0
    + 1e-12
)


# ============================================================================
# Independently reconstruct exact expected union-of-test row keys
# from canonical table + frozen assignments.
# ============================================================================

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
    as_bool(
        canon[
            "soh_task_eligible"
        ]
    )
].copy()


canon[
    "split_group"
] = canon[
    "split_group"
].astype(str)

canon[
    "feature_source_cell_id"
] = canon[
    "feature_source_cell_id"
].astype(str)

canon[
    "feature_raw_cycle_position_zero_based"
] = pd.to_numeric(
    canon[
        "feature_raw_cycle_position_zero_based"
    ],
    errors="raise",
).astype(int)


a = pd.read_csv(
    ASSIGNMENTS,
    usecols=[
        "split_seed",
        "split_group",
        "split",
    ],
    low_memory=False,
)

a[
    "split_group"
] = a[
    "split_group"
].astype(str)


test_groups = set(
    a.loc[
        a[
            "split"
        ] == "test",
        "split_group",
    ]
)


expected = canon[
    canon[
        "split_group"
    ].isin(
        test_groups
    )
][
    [
        "dataset",
        "feature_source_cell_id",
        "feature_raw_cycle_position_zero_based",
    ]
].copy()


observed = rows[
    [
        "dataset",
        "feature_source_cell_id",
        "feature_raw_cycle_position_zero_based",
    ]
].copy()


observed[
    "feature_source_cell_id"
] = observed[
    "feature_source_cell_id"
].astype(str)

observed[
    "feature_raw_cycle_position_zero_based"
] = pd.to_numeric(
    observed[
        "feature_raw_cycle_position_zero_based"
    ],
    errors="raise",
).astype(int)


expected = expected.sort_values(
    list(
        expected.columns
    ),
    kind="mergesort",
).reset_index(
    drop=True
)

observed = observed.sort_values(
    list(
        observed.columns
    ),
    kind="mergesort",
).reset_index(
    drop=True
)


assert expected.equals(
    observed
)


dataset_counts = (
    observed[
        "dataset"
    ]
    .value_counts()
    .sort_index()
    .to_dict()
)


assert dataset_counts == {
    k: int(v)
    for k, v in summary[
        "dataset_row_counts"
    ].items()
}


print("=" * 80)
print("ZERO-CORRUPTION RAW ROUNDTRIP AUDIT")
print("=" * 80)

print(
    "protocol exact:",
    True
)

print(
    "expected union rows:",
    len(
        expected
    )
)

print(
    "audited union rows:",
    len(
        observed
    )
)

print(
    "exact canonical test-union key set:",
    True
)

print(
    "source metadata exact:",
    True
)

print(
    "feature contract exact:",
    True
)

print(
    "kernel hash exact:",
    True
)

print(
    "20/20 features within tolerance:",
    True
)

print(
    "failed rows:",
    0
)

print(
    "max tolerance ratio:",
    f"{float(ratio.max()):.6g}"
)

print(
    "model evaluated:",
    False
)

print(
    "physical noise generated:",
    False
)

print(
    "physical test metrics evaluated:",
    False
)

print(
    "row audit SHA256:",
    sha(
        ROWS
    )
)

print(
    "roundtrip summary SHA256:",
    sha(
        SUMMARY
    )
)

print()
print(
    "ZERO_CORRUPTION_RAW_ROUNDTRIP_AUDIT_PASS=True"
)
