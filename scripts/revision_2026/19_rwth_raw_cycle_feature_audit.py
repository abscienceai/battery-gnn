#!/usr/bin/env python3

"""
Independent audit for RWTH source-native cycle features.

The feature generator operates on 105,663 pre-abnormal-filter
structural ZYK cycles and never reads Qd/AhEla/CapDCH1 targets.

Cross-source cycles are treated as structural acquisition
discontinuities rather than having their local Programmdauer
clocks artificially stitched across multi-day source sessions.

The independent reconstruction audit is used here only after
feature generation to establish later eligibility.
"""

from pathlib import Path
import hashlib
import importlib.util
import json
import os
import re

import numpy as np
import pandas as pd


BASE = Path(os.environ["BASE"])
REPO = Path(os.environ["REPO"])

FEATURES = (
    BASE
    / "data/processed/revision_2026/"
      "raw_cycle_features/"
      "RWTH_raw_cycle_features.csv"
)

RECON = (
    BASE
    / "data/audit/"
      "RWTH_ZYK_cycle_reconstruction_all_cells.csv"
)

SOURCES = (
    BASE
    / "data/audit/"
      "RWTH_ZYK_source_manifest.csv"
)

KERNEL_PATH = (
    REPO
    / "scripts/revision_2026/"
      "cycle_feature_kernel.py"
)

OUT = (
    BASE
    / "data/audit/"
      "RWTH_raw_cycle_feature_audit_manifest.json"
)


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


spec = importlib.util.spec_from_file_location(
    "cycle_feature_kernel",
    KERNEL_PATH,
)

kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)

kernel.validate_primary_schema()

features = list(
    kernel.PRIMARY_FEATURE_NAMES
)

assert len(features) == 20


f = pd.read_csv(
    FEATURES,
    low_memory=False,
)

a = pd.read_csv(
    RECON,
    usecols=[
        "cell_id",
        "cycle_number_pre_filter",
        "start_row",
        "end_row_exclusive",
        "n_samples",
        "source_file_count",
        "source_zip_first",
        "source_zip_last",
        "crosses_source_file_boundary",
        "batteryml_abnormal_removed",
        "cycle_number_after_filter",
    ],
    low_memory=False,
)

s = pd.read_csv(
    SOURCES,
    low_memory=False,
)


assert len(f) == 105663
assert f["raw_cell_id"].nunique() == 48
assert len(a) == 105663
assert len(s) == 738

assert (
    f[
        "rwth_adapter_contract_version"
    ]
    == "rwth_zyk_sensor_only_prefilter_v2"
).all()


f["_cycle1"] = (
    pd.to_numeric(
        f[
            "raw_cycle_position_zero_based"
        ],
        errors="raise",
    ).astype(int)
    + 1
)

m = f.merge(
    a,
    left_on=[
        "raw_cell_id",
        "_cycle1",
    ],
    right_on=[
        "cell_id",
        "cycle_number_pre_filter",
    ],
    how="outer",
    validate="one_to_one",
    indicator=True,
    suffixes=(
        "_feature",
        "_audit",
    ),
)

assert len(m) == 105663
assert (
    m["_merge"] == "both"
).all()


def numeric_equal(left, right):
    x = pd.to_numeric(
        left,
        errors="coerce",
    ).to_numpy(dtype=float)

    y = pd.to_numeric(
        right,
        errors="coerce",
    ).to_numpy(dtype=float)

    return np.isclose(
        x,
        y,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


start_match = numeric_equal(
    m["start_row_qc"],
    m["start_row"],
)

end_match = numeric_equal(
    m["end_row_exclusive_qc"],
    m["end_row_exclusive"],
)

sample_match = numeric_equal(
    m["raw_sample_count_qc"],
    m["n_samples"],
)

source_count_match = numeric_equal(
    m["source_file_count_qc"],
    m["source_file_count"],
)

first_match = (
    m[
        "source_zip_first_feature"
    ].astype(str)
    ==
    m[
        "source_zip_first_audit"
    ].astype(str)
).to_numpy()

last_match = (
    m[
        "source_zip_last_feature"
    ].astype(str)
    ==
    m[
        "source_zip_last_audit"
    ].astype(str)
).to_numpy()

boundary_feature = (
    pd.to_numeric(
        m[
            "source_file_count_qc"
        ],
        errors="raise",
    )
    == 2
)

boundary_audit = (
    m[
        "crosses_source_file_boundary"
    ]
    .astype(str)
    .str.lower()
    .eq("true")
)

valid = (
    m[
        "feature_valid"
    ]
    .astype(str)
    .str.lower()
    .eq("true")
)

eligible = (
    m[
        "cycle_number_after_filter"
    ].notna()
)

removed = (
    m[
        "batteryml_abnormal_removed"
    ]
    .astype(str)
    .str.lower()
    .eq("true")
)


assert start_match.all()
assert end_match.all()
assert sample_match.all()
assert source_count_match.all()
assert first_match.all()
assert last_match.all()

assert np.array_equal(
    boundary_feature.to_numpy(),
    boundary_audit.to_numpy(),
)

assert int(boundary_feature.sum()) == 646
assert int(valid.sum()) == 105017
assert int((~valid).sum()) == 646

assert np.array_equal(
    (~valid).to_numpy(),
    boundary_feature.to_numpy(),
)

assert int(eligible.sum()) == 105006
assert int(removed.sum()) == 657

assert int(
    (
        boundary_feature
        & eligible
    ).sum()
) == 66

assert int(
    (
        boundary_feature
        & removed
    ).sum()
) == 580

assert int(
    (
        valid
        & eligible
    ).sum()
) == 104940


X = (
    m.loc[
        valid,
        features,
    ]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
    .to_numpy(dtype=float)
)

assert np.isfinite(X).all()


reason = (
    m.loc[
        ~valid,
        "failure_reason",
    ]
    .fillna("")
    .astype(str)
)

assert (
    reason
    == (
        "StructuralDiscontinuity: "
        "cross-source TBA_Zyk boundary; "
        "Programmdauer resets between archives"
    )
).all()


backward = pd.to_numeric(
    m.loc[
        boundary_feature,
        "backward_time_step_count_qc",
    ],
    errors="raise",
)

assert (
    backward == 1
).all()


# ============================================================
# Archive-session separation audit
# ============================================================

pat = re.compile(
    r"Massenalterung="
    r"(\d{4}-\d{2}-\d{2}) "
    r"(\d{6})="
)


def parse_archive_start(value):
    hit = pat.search(str(value))

    if not hit:
        return pd.NaT

    d = hit.group(1)
    t = hit.group(2)

    return pd.to_datetime(
        (
            f"{d} "
            f"{t[0:2]}:"
            f"{t[2:4]}:"
            f"{t[4:6]}"
        ),
        errors="coerce",
    )


b = m[
    boundary_feature
].copy()

first_dt = (
    b[
        "source_zip_first_audit"
    ]
    .map(parse_archive_start)
)

last_dt = (
    b[
        "source_zip_last_audit"
    ]
    .map(parse_archive_start)
)

assert first_dt.notna().all()
assert last_dt.notna().all()

gap_h = (
    (
        last_dt
        - first_dt
    )
    .dt.total_seconds()
    / 3600.0
)

assert np.isfinite(gap_h).all()
assert (
    gap_h > 48.0
).all()


be = (
    b[
        b[
            "cycle_number_after_filter"
        ].notna()
    ]
    .copy()
)

assert len(be) == 66


# Feature output itself must remain target-free.
forbidden = (
    "capacity",
    "soh",
    "rul",
    "eol",
    "capdch",
    "ahela",
    "ahlad",
    "qd_integrated",
    "qc_integrated",
    "batteryml_abnormal",
    "cycle_number_after_filter",
)

bad_columns = [
    c
    for c in f.columns
    if any(
        token in c.lower()
        for token in forbidden
    )
]

assert bad_columns == []


manifest = {
    "pre_abnormal_filter_cycles":
        105663,

    "contiguous_feature_valid_cycles":
        105017,

    "structural_discontinuity_cycles":
        646,

    "later_qd_eligible_cycles":
        105006,

    "later_qd_removed_cycles":
        657,

    "boundary_later_eligible":
        66,

    "boundary_later_removed":
        580,

    "usable_contiguous_and_later_eligible":
        104940,

    "source_manifest_archives":
        738,

    "structural_alignment_rows":
        105663,

    "archive_start_gap_hours_min":
        float(gap_h.min()),

    "archive_start_gap_hours_median":
        float(gap_h.median()),

    "archive_start_gap_hours_p95":
        float(
            gap_h.quantile(0.95)
        ),

    "archive_start_gap_hours_max":
        float(gap_h.max()),

    "archive_start_gap_le_48h":
        int(
            (gap_h <= 48).sum()
        ),

    "feature_csv_sha256":
        sha256_file(FEATURES),

    "reconstruction_csv_sha256":
        sha256_file(RECON),

    "source_manifest_sha256":
        sha256_file(SOURCES),

    "kernel_sha256":
        sha256_file(KERNEL_PATH),

    "audit_pass":
        True,
}

OUT.write_text(
    json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


print("=" * 78)
print("RWTH RAW-CYCLE FEATURE AUDIT")
print("=" * 78)

print(
    "structural alignment:",
    len(m),
    "/ 105663"
)

print(
    "start/end/sample/source matches:",
    int(start_match.sum()),
    "/",
    int(end_match.sum()),
    "/",
    int(sample_match.sum()),
    "/",
    int(source_count_match.sum()),
)

print(
    "feature valid / structural discontinuity:",
    int(valid.sum()),
    "/",
    int((~valid).sum()),
)

print(
    "invalid == cross-source boundary:",
    bool(
        np.array_equal(
            (~valid).to_numpy(),
            boundary_feature.to_numpy(),
        )
    )
)

print(
    "later eligible / removed:",
    int(eligible.sum()),
    "/",
    int(removed.sum()),
)

print(
    "boundary later eligible / removed:",
    int(
        (
            boundary_feature
            & eligible
        ).sum()
    ),
    "/",
    int(
        (
            boundary_feature
            & removed
        ).sum()
    ),
)

print(
    "usable contiguous + eligible:",
    int(
        (
            valid
            & eligible
        ).sum()
    )
)

print(
    "archive gap h min/median/p95/max:",
    f"{gap_h.min():.3f}",
    "/",
    f"{gap_h.median():.3f}",
    "/",
    f"{gap_h.quantile(0.95):.3f}",
    "/",
    f"{gap_h.max():.3f}",
)

print(
    "archive gap <=48 h:",
    int(
        (gap_h <= 48).sum()
    )
)

print(
    "valid rows all 20 finite:",
    len(X),
    "/",
    int(valid.sum()),
)

print(
    "target-like columns in feature output:",
    len(bad_columns)
)

print()
print(
    "RWTH_RAW_CYCLE_FEATURE_AUDIT_PASS=True"
)
