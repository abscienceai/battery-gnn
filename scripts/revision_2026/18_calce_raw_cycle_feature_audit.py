#!/usr/bin/env python3

"""
Independent audit of the CALCE source-native raw-cycle feature
table produced by 14_raw_cycle_feature_generator.py.

This audit is deliberately separate from feature construction.
Capacity/Qd and keep_final_batteryml are used here only to
validate alignment and to characterize the small set of
feature-unavailable source fragments.

They are never model features.
"""

from pathlib import Path
import hashlib
import importlib.util
import json
import os

import numpy as np
import pandas as pd


BASE = Path(
    os.environ["BASE"]
)

REPO = Path(
    os.environ["REPO"]
)

FEATURES = (
    BASE
    / "data/processed/revision_2026/"
      "raw_cycle_features/"
      "CALCE_raw_cycle_features.csv"
)

AUDIT_DIR = (
    BASE
    / "data/audit"
)

KERNEL_PATH = (
    REPO
    / "scripts/revision_2026/"
      "cycle_feature_kernel.py"
)

INVALID_OUT = (
    AUDIT_DIR
    / "CALCE_raw_cycle_feature_invalid_audit.csv"
)

MANIFEST = (
    AUDIT_DIR
    / "CALCE_raw_cycle_feature_audit_manifest.json"
)


def sha256_file(path):
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

            h.update(b)

    return h.hexdigest()


def load_kernel():
    spec = (
        importlib.util
        .spec_from_file_location(
            "cycle_feature_kernel",
            KERNEL_PATH,
        )
    )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    module.validate_primary_schema()

    return module


kernel = load_kernel()

feature_names = list(
    kernel.PRIMARY_FEATURE_NAMES
)

assert len(
    feature_names
) == 20


# ============================================================
# Production source-native feature table
# ============================================================

f = pd.read_csv(
    FEATURES,
    low_memory=False,
)

assert len(f) == 18820
assert f[
    "raw_cell_id"
].nunique() == 13

assert (
    f[
        "dataset"
    ] == "CALCE"
).all()

assert (
    f[
        "calce_adapter_contract_version"
    ]
    == "calce_raw_vit_grouped_cycle_v2"
).all()

assert (
    pd.to_numeric(
        f[
            "source_member_count_qc"
        ],
        errors="raise",
    )
    == 1
).all()

assert not f[
    [
        "raw_cell_id",
        "raw_cycle_key",
    ]
].duplicated().any()


valid = (
    f[
        "feature_valid"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("true")
)

assert int(
    valid.sum()
) == 18804

assert int(
    (~valid).sum()
) == 16


X = (
    f[
        feature_names
    ]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
    .to_numpy(
        dtype=float
    )
)

assert np.isfinite(
    X[valid.to_numpy()]
).all()


# ============================================================
# Independently generated CALCE capacity/source audit
# ============================================================

audit_files = sorted(
    AUDIT_DIR.glob(
        "C[XS]2_*_cycles_audit.csv"
    )
)

assert len(
    audit_files
) == 13

parts = []

for path in audit_files:

    cell = path.name.replace(
        "_cycles_audit.csv",
        "",
    )

    a = pd.read_csv(
        path,
        usecols=[
            "batteryml_cycle_number_zero_based",
            "date_batteryml",
            "organized_cycle_index",
            "source_files",
            "keep_final_batteryml",
            "Qd_integrated_Ah",
        ],
        low_memory=False,
    )

    a.insert(
        0,
        "raw_cell_id",
        cell,
    )

    parts.append(
        a
    )


a = pd.concat(
    parts,
    ignore_index=True,
)

assert len(a) == 18820


m = f.merge(
    a,
    left_on=[
        "raw_cell_id",
        "raw_cycle_position_zero_based",
    ],
    right_on=[
        "raw_cell_id",
        "batteryml_cycle_number_zero_based",
    ],
    how="outer",
    validate="one_to_one",
    indicator=True,
    suffixes=(
        "_feature",
        "_audit",
    ),
)

assert len(m) == 18820

assert (
    m["_merge"] == "both"
).all()


date_match = (
    m[
        "date_batteryml_feature"
    ].astype(str)
    ==
    m[
        "date_batteryml_audit"
    ].astype(str)
)

source_match = (
    m[
        "source_member"
    ].astype(str)
    ==
    m[
        "source_files"
    ].astype(str)
)

fx = pd.to_numeric(
    m[
        "organized_cycle_index_feature"
    ],
    errors="coerce",
)

ax = pd.to_numeric(
    m[
        "organized_cycle_index_audit"
    ],
    errors="coerce",
)

organized_match = np.isclose(
    fx,
    ax,
    rtol=0.0,
    atol=0.0,
    equal_nan=True,
)

assert date_match.all()
assert source_match.all()
assert organized_match.all()


# ============================================================
# Validate the 16 source fragments rejected only because
# fewer than the shared kernel's minimum sample count remain.
# ============================================================

m_valid = (
    m[
        "feature_valid"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("true")
)

bad = m[
    ~m_valid
].copy()

assert len(
    bad
) == 16

assert (
    bad[
        "failure_reason"
    ]
    .fillna("")
    .astype(str)
    .str.startswith(
        "ValueError: Too few valid samples:"
    )
    .all()
)

raw_n = pd.to_numeric(
    bad[
        "raw_sample_count_qc"
    ],
    errors="raise",
)

valid_n = pd.to_numeric(
    bad[
        "valid_sample_count_qc"
    ],
    errors="raise",
)

assert int(
    raw_n.max()
) <= 19

assert int(
    valid_n.max()
) <= 19


keep = (
    bad[
        "keep_final_batteryml"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("true")
)

assert int(
    keep.sum()
) == 0


qd = pd.to_numeric(
    bad[
        "Qd_integrated_Ah"
    ],
    errors="raise",
)

assert np.isclose(
    qd.to_numpy(
        dtype=float
    ),
    0.0,
    rtol=0.0,
    atol=1e-12,
).all()


# ============================================================
# Permanent invalid-row evidence
# ============================================================

invalid_cols = [
    "raw_cell_id",
    "raw_cycle_position_zero_based",
    "date_batteryml_feature",
    "organized_cycle_index_feature",
    "source_member",
    "source_format",
    "raw_sample_count_qc",
    "valid_sample_count_qc",
    "dropped_nonfinite_samples_qc",
    "keep_final_batteryml",
    "Qd_integrated_Ah",
    "failure_reason",
]

INVALID_OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

bad[
    invalid_cols
].sort_values(
    [
        "raw_cell_id",
        "raw_cycle_position_zero_based",
    ]
).to_csv(
    INVALID_OUT,
    index=False,
)


source_format_counts = (
    f[
        "source_format"
    ]
    .value_counts()
    .sort_index()
    .to_dict()
)

assert source_format_counts == {
    "txt": 649,
    "xlsx": 18171,
}


manifest = {
    "raw_source_cells":
        13,

    "raw_grouped_cycles":
        18820,

    "feature_valid_rows":
        18804,

    "feature_invalid_rows":
        16,

    "primary_feature_count":
        20,

    "one_to_one_alignment_rows":
        18820,

    "date_matches":
        int(
            date_match.sum()
        ),

    "organized_index_matches":
        int(
            organized_match.sum()
        ),

    "source_member_matches":
        int(
            source_match.sum()
        ),

    "multi_source_grouped_cycles":
        0,

    "invalid_all_too_few_samples":
        True,

    "invalid_inside_keep_final_batteryml":
        0,

    "invalid_qd_all_zero":
        True,

    "source_format_cycle_counts":
        source_format_counts,

    "feature_csv_sha256":
        sha256_file(
            FEATURES
        ),

    "kernel_sha256":
        sha256_file(
            KERNEL_PATH
        ),

    "invalid_audit_csv_sha256":
        sha256_file(
            INVALID_OUT
        ),

    "audit_pass":
        True,
}

MANIFEST.write_text(
    json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


print("=" * 78)
print("CALCE RAW-CYCLE FEATURE AUDIT")
print("=" * 78)

print(
    "raw source cells:",
    f[
        "raw_cell_id"
    ].nunique()
)

print(
    "raw grouped cycles:",
    len(f)
)

print(
    "feature valid / invalid:",
    int(
        valid.sum()
    ),
    "/",
    int(
        (~valid).sum()
    )
)

print(
    "one-to-one audit alignment:",
    int(
        (
            m["_merge"]
            == "both"
        ).sum()
    ),
    "/",
    len(m)
)

print(
    "date / organized / source matches:",
    int(date_match.sum()),
    "/",
    int(
        organized_match.sum()
    ),
    "/",
    int(
        source_match.sum()
    )
)

print(
    "valid rows with all 20 features finite:",
    int(valid.sum()),
    "/",
    int(valid.sum())
)

print(
    "invalid all too-few-sample:",
    True
)

print(
    "invalid inside keep_final_batteryml:",
    int(
        keep.sum()
    ),
    "/",
    len(bad)
)

print(
    "invalid Qd all zero:",
    bool(
        np.isclose(
            qd,
            0.0,
            rtol=0.0,
            atol=1e-12,
        ).all()
    )
)

print(
    "source format cycle counts:",
    source_format_counts
)

print(
    "CALCE_RAW_CYCLE_FEATURE_AUDIT_PASS=True"
)
