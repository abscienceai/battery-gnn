#!/usr/bin/env python3

"""
Final MATR negative-time classification audit.

This audit verifies that the production feature table preserves
all raw timing irregularities as provenance while distinguishing:

    574 materially non-causal B4 timestamp anomalies
    223 sub-tolerance B1/B2 floating-point jitters

Only the 574 material B4 anomalies may be timestamp-repaired.

It also cross-checks the production classification against the
independent raw-data audits 15 and 16.
"""

from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd


BASE = Path(
    os.environ["BASE"]
)

FEATURES = (
    BASE
    / "data/processed/revision_2026/"
      "raw_cycle_features/"
      "MATR_raw_cycle_features.csv"
)

AUDIT15 = (
    BASE
    / "data/audit/"
      "MATR_time_discontinuity_diagnostic.csv"
)

AUDIT16 = (
    BASE
    / "data/audit/"
      "MATR_timestamp_repair_audit.csv"
)

OUT = (
    BASE
    / "data/audit/"
      "MATR_negative_time_classification_summary.csv"
)

MANIFEST = (
    BASE
    / "data/audit/"
      "MATR_negative_time_classification_manifest.json"
)


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
    if s.dtype == bool:
        return (
            s.fillna(False)
            .astype(bool)
        )

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
        .fillna(False)
        .astype(bool)
    )


required = [
    "raw_cell_id",
    "raw_cycle_position_zero_based",
    "feature_valid",
    "failure_reason",
    "timestamp_repaired_qc",
    "source_backward_time_step_count_qc",
    "source_material_backward_time_step_count_qc",
    "source_subtolerance_negative_time_step_count_qc",
    "backward_time_step_count_qc",
]

df = pd.read_csv(
    FEATURES,
    usecols=required,
    low_memory=False,
)

assert len(df) == 154411

valid = as_bool(
    df["feature_valid"]
)

repaired = as_bool(
    df[
        "timestamp_repaired_qc"
    ]
)

raw_back = (
    pd.to_numeric(
        df[
            "source_backward_time_step_count_qc"
        ],
        errors="raise",
    )
    .fillna(0)
    .astype(int)
)

material = (
    pd.to_numeric(
        df[
            "source_material_backward_time_step_count_qc"
        ],
        errors="raise",
    )
    .fillna(0)
    .astype(int)
)

subtol = (
    pd.to_numeric(
        df[
            "source_subtolerance_negative_time_step_count_qc"
        ],
        errors="raise",
    )
    .fillna(0)
    .astype(int)
)

kernel_backward = (
    pd.to_numeric(
        df[
            "backward_time_step_count_qc"
        ],
        errors="coerce",
    )
    .fillna(0)
    .astype(int)
)

any_neg = (
    raw_back > 0
)

material_mask = (
    material > 0
)

subtol_mask = (
    subtol > 0
)


# ============================================================
# Global structural contract
# ============================================================

assert int(valid.sum()) == 154365
assert int((~valid).sum()) == 46

assert int(any_neg.sum()) == 797
assert int(material_mask.sum()) == 574
assert int(subtol_mask.sum()) == 223

# Every affected cycle contains exactly one raw negative step.
assert (
    raw_back[
        any_neg
    ] == 1
).all()

assert (
    material[
        material_mask
    ] == 1
).all()

assert (
    subtol[
        subtol_mask
    ] == 1
).all()

# Classes are mutually exclusive and exhaustive.
assert not (
    material_mask
    & subtol_mask
).any()

assert (
    any_neg
    == (
        material_mask
        | subtol_mask
    )
).all()


# ============================================================
# Batch-specific contract
# ============================================================

cell_id = (
    df[
        "raw_cell_id"
    ]
    .astype(str)
)

material_batches = (
    cell_id[
        material_mask
    ]
    .str.extract(
        r"^(b[1-4])",
        expand=False,
    )
    .value_counts()
    .sort_index()
    .to_dict()
)

subtol_batches = (
    cell_id[
        subtol_mask
    ]
    .str.extract(
        r"^(b[1-4])",
        expand=False,
    )
    .value_counts()
    .sort_index()
    .to_dict()
)

assert material_batches == {
    "b4": 574
}

assert subtol_batches == {
    "b1": 123,
    "b2": 100,
}


# ============================================================
# Repair policy
# ============================================================

assert (
    repaired
    == material_mask
).all()

assert not repaired[
    subtol_mask
].any()

# Both material anomalies after repair and sub-tolerance jitter
# must produce valid features.
assert valid[
    material_mask
].all()

assert valid[
    subtol_mask
].all()

# After production handling, the feature kernel sees no
# material backward-time step.
assert (
    kernel_backward[
        material_mask
    ] == 0
).all()

assert (
    kernel_backward[
        subtol_mask
    ] == 0
).all()


# ============================================================
# Known invalid rows must be only matched B1 placeholders
# ============================================================

invalid = df[
    ~valid
].copy()

assert len(invalid) == 46

assert (
    invalid[
        "raw_cell_id"
    ]
    .astype(str)
    .str.startswith(
        "b1c"
    )
    .all()
)

assert (
    pd.to_numeric(
        invalid[
            "raw_cycle_position_zero_based"
        ],
        errors="raise",
    )
    == 0
).all()

failure_counts = (
    invalid[
        "failure_reason"
    ]
    .fillna("")
    .value_counts()
    .to_dict()
)

assert failure_counts == {
    "ValueError: Too few valid samples: 2": 46
}


# ============================================================
# Independent audit cross-check
# ============================================================

a15 = pd.read_csv(
    AUDIT15,
    usecols=[
        "raw_cell_id",
        "cycle_index",
    ],
)

a16 = pd.read_csv(
    AUDIT16,
    usecols=[
        "raw_cell_id",
        "cycle_index",
    ],
)

assert len(a15) == 574
assert len(a16) == 574

prod_material_keys = set(
    zip(
        df.loc[
            material_mask,
            "raw_cell_id",
        ].astype(str),
        pd.to_numeric(
            df.loc[
                material_mask,
                "raw_cycle_position_zero_based",
            ],
            errors="raise",
        ).astype(int),
    )
)

audit15_keys = set(
    zip(
        a15[
            "raw_cell_id"
        ].astype(str),
        pd.to_numeric(
            a15[
                "cycle_index"
            ],
            errors="raise",
        ).astype(int),
    )
)

audit16_keys = set(
    zip(
        a16[
            "raw_cell_id"
        ].astype(str),
        pd.to_numeric(
            a16[
                "cycle_index"
            ],
            errors="raise",
        ).astype(int),
    )
)

assert (
    prod_material_keys
    == audit15_keys
    == audit16_keys
)


# ============================================================
# Compact permanent summary artifact
# ============================================================

summary = pd.DataFrame([
    {
        "class":
            "material_B4_timestamp_anomaly",

        "cycles":
            574,

        "batches":
            "b4",

        "production_action":
            (
                "single isolated timestamp replaced "
                "by midpoint of valid neighbors"
            ),

        "feature_valid_after_policy":
            574,
    },
    {
        "class":
            "subtolerance_B1_B2_negative_jitter",

        "cycles":
            223,

        "batches":
            "b1,b2",

        "production_action":
            "no timestamp modification",

        "feature_valid_after_policy":
            223,
    },
    {
        "class":
            "matched_B1_placeholder",

        "cycles":
            46,

        "batches":
            "b1",

        "production_action":
            "retained structurally; excluded from model X",

        "feature_valid_after_policy":
            0,
    },
])

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

summary.to_csv(
    OUT,
    index=False,
)

manifest = {
    "production_rows":
        154411,

    "feature_valid_rows":
        154365,

    "feature_invalid_rows":
        46,

    "cycles_with_any_raw_negative_dt":
        797,

    "material_B4_timestamp_anomalies":
        574,

    "subtolerance_B1_B2_negative_jitters":
        223,

    "repaired_cycles":
        574,

    "audit15_matches_production_material_set":
        True,

    "audit16_matches_production_material_set":
        True,

    "invalid_rows_are_only_B1_placeholders":
        True,

    "production_csv_sha256":
        sha256_file(
            FEATURES
        ),

    "summary_csv_sha256":
        sha256_file(
            OUT
        ),

    "audit_pass":
        True,
}

MANIFEST.write_text(
    json.dumps(
        manifest,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


print("=" * 78)
print("MATR NEGATIVE-TIME CLASSIFICATION AUDIT")
print("=" * 78)

print(
    "production rows:",
    len(df)
)

print(
    "feature valid / invalid:",
    int(valid.sum()),
    "/",
    int((~valid).sum())
)

print(
    "any raw negative dt:",
    int(any_neg.sum())
)

print(
    "material B4 anomalies:",
    int(material_mask.sum())
)

print(
    "sub-tolerance B1/B2 jitter:",
    int(subtol_mask.sum())
)

print(
    "material batches:",
    material_batches
)

print(
    "sub-tolerance batches:",
    subtol_batches
)

print(
    "repaired == material:",
    bool(
        (
            repaired
            == material_mask
        ).all()
    )
)

print(
    "audit15 material-set match:",
    prod_material_keys
    == audit15_keys
)

print(
    "audit16 material-set match:",
    prod_material_keys
    == audit16_keys
)

print(
    "remaining invalid B1 placeholders:",
    len(invalid)
)

print(
    "production CSV SHA256:",
    sha256_file(
        FEATURES
    )
)

print()
print(
    "MATR_NEGATIVE_TIME_CLASSIFICATION_AUDIT_PASS=True"
)
