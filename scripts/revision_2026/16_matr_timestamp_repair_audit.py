#!/usr/bin/env python3

"""
MATR B4 isolated timestamp repair sensitivity audit.

Candidate cycle set comes from independent raw-data audit 15.

For each material anomaly compare:
    A) keep all V/I samples; repair only one timestamp by midpoint
    B) remove the single sample carrying the corrupt timestamp

The repair policy must not materially drive the primary
20-feature representation.
"""

from pathlib import Path
import hashlib
import importlib.util
import json
import os
import re

import h5py
import numpy as np
import pandas as pd


BASE = Path(os.environ["BASE"])

RAW = (
    BASE
    / "data/raw/MATR/"
      "MATR_batch_20190124.mat"
)

AUDIT15 = (
    BASE
    / "data/audit/"
      "MATR_time_discontinuity_diagnostic.csv"
)

KERNEL_PATH = (
    BASE
    / "repo/battery-gnn/"
      "scripts/revision_2026/"
      "cycle_feature_kernel.py"
)

OUT = (
    BASE
    / "data/audit/"
      "MATR_timestamp_repair_audit.csv"
)

MANIFEST = (
    BASE
    / "data/audit/"
      "MATR_timestamp_repair_audit_manifest.json"
)

TIME_TOLERANCE_S = 1e-9


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
        .module_from_spec(spec)
    )

    spec.loader.exec_module(
        module
    )

    module.validate_primary_schema()

    return module


def deref(f, ds, idx):
    return (
        np.asarray(
            f[ds[idx, 0]][:]
        )
        .squeeze()
        .astype(float)
        .reshape(-1)
    )


kernel = load_kernel()

candidates = pd.read_csv(
    AUDIT15,
    usecols=[
        "raw_cell_id",
        "cycle_index",
        "bad_sample_index",
    ],
)

assert len(candidates) == 574
assert candidates[
    "raw_cell_id"
].nunique() == 45

records = []

with h5py.File(RAW, "r") as f:

    batch = f["batch"]

    for _, row in (
        candidates.iterrows()
    ):

        cell_id = str(
            row["raw_cell_id"]
        )

        m = re.fullmatch(
            r"b4c(\d+)",
            cell_id,
        )

        if m is None:
            raise ValueError(
                cell_id
            )

        cell_idx = int(
            m.group(1)
        )

        cycle_idx = int(
            row["cycle_index"]
        )

        expected_bad_idx = int(
            row["bad_sample_index"]
        )

        cycles = f[
            batch["cycles"][
                cell_idx,
                0,
            ]
        ]

        V = deref(
            f,
            cycles["V"],
            cycle_idx,
        )

        I = (
            deref(
                f,
                cycles["I"],
                cycle_idx,
            )
            * 1.1
        )

        t_native = deref(
            f,
            cycles["t"],
            cycle_idx,
        )

        assert (
            len(V)
            == len(I)
            == len(t_native)
        )

        dt_s = (
            np.diff(t_native)
            * 60.0
        )

        neg = np.where(
            np.isfinite(dt_s)
            & (
                dt_s
                < -TIME_TOLERANCE_S
            )
        )[0]

        assert len(neg) == 1

        k = int(neg[0])
        bad_idx = k + 1

        assert (
            bad_idx
            == expected_bad_idx
        )

        assert (
            bad_idx + 1
            < len(t_native)
        )

        left = float(
            t_native[k]
        )

        right = float(
            t_native[
                bad_idx + 1
            ]
        )

        assert right > left

        repaired_native = (
            t_native.copy()
        )

        repaired_native[
            bad_idx
        ] = (
            left + right
        ) / 2.0

        repaired_dt_s = (
            np.diff(
                repaired_native
            )
            * 60.0
        )

        assert np.all(
            repaired_dt_s
            >= -TIME_TOLERANCE_S
        )

        without_t = np.delete(
            t_native,
            bad_idx,
        )

        assert np.all(
            np.diff(
                without_t
            )
            * 60.0
            >= -TIME_TOLERANCE_S
        )

        repair_features, _ = (
            kernel
            .extract_primary_cycle_features(
                V,
                I,
                repaired_native * 60.0,
            )
        )

        drop_features, _ = (
            kernel
            .extract_primary_cycle_features(
                np.delete(
                    V,
                    bad_idx,
                ),
                np.delete(
                    I,
                    bad_idx,
                ),
                without_t * 60.0,
            )
        )

        v_range = max(
            float(
                np.max(V)
                - np.min(V)
            ),
            1e-12,
        )

        i_range = max(
            float(
                np.max(I)
                - np.min(I)
            ),
            1e-12,
        )

        v_slope_delta = (
            abs(
                repair_features[
                    "v_phase_slope"
                ]
                -
                drop_features[
                    "v_phase_slope"
                ]
            )
            / v_range
        )

        i_slope_delta = (
            abs(
                repair_features[
                    "i_phase_slope"
                ]
                -
                drop_features[
                    "i_phase_slope"
                ]
            )
            / i_range
        )

        max_feature_abs_delta = max(
            abs(
                repair_features[name]
                -
                drop_features[name]
            )
            for name in (
                kernel
                .PRIMARY_FEATURE_NAMES
            )
        )

        records.append({
            "batch":
                "b4",

            "raw_cell_id":
                cell_id,

            "cycle_index":
                cycle_idx,

            "bad_sample_index":
                bad_idx,

            "midpoint_repair_monotonic":
                True,

            "single_sample_removal_monotonic":
                True,

            "v_slope_delta_norm_repair_vs_drop":
                float(
                    v_slope_delta
                ),

            "i_slope_delta_norm_repair_vs_drop":
                float(
                    i_slope_delta
                ),

            "max_feature_abs_delta_repair_vs_drop":
                float(
                    max_feature_abs_delta
                ),
        })


df = pd.DataFrame(records)

assert len(df) == 574
assert df["raw_cell_id"].nunique() == 45

assert (
    df[
        "midpoint_repair_monotonic"
    ].all()
)

assert (
    df[
        "single_sample_removal_monotonic"
    ].all()
)

v_delta = (
    df[
        "v_slope_delta_norm_repair_vs_drop"
    ]
)

i_delta = (
    df[
        "i_slope_delta_norm_repair_vs_drop"
    ]
)

# Prior independent result was ~0.0024-0.0025 at p95.
# Keep a conservative 1% normalized guard.
assert float(
    v_delta.quantile(0.95)
) < 0.01

assert float(
    i_delta.quantile(0.95)
) < 0.01

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUT,
    index=False,
)

manifest = {
    "candidate_source":
        "audit15 raw-HDF5 material anomaly set",

    "cycles":
        int(len(df)),

    "affected_cells":
        int(
            df["raw_cell_id"].nunique()
        ),

    "normalized_voltage_slope_delta_median":
        float(v_delta.median()),

    "normalized_voltage_slope_delta_p95":
        float(
            v_delta.quantile(0.95)
        ),

    "normalized_current_slope_delta_median":
        float(i_delta.median()),

    "normalized_current_slope_delta_p95":
        float(
            i_delta.quantile(0.95)
        ),

    "kernel_sha256":
        sha256_file(
            KERNEL_PATH
        ),

    "audit15_csv_sha256":
        sha256_file(
            AUDIT15
        ),

    "audit_csv_sha256":
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
print("MATR TIMESTAMP REPAIR SENSITIVITY AUDIT")
print("=" * 78)

print(
    "cycles / affected cells:",
    len(df),
    "/",
    df[
        "raw_cell_id"
    ].nunique()
)

print(
    "single-sample removal gives monotonic time:",
    int(
        df[
            "single_sample_removal_monotonic"
        ].sum()
    ),
    "/",
    len(df)
)

print(
    "midpoint repair gives monotonic time:",
    int(
        df[
            "midpoint_repair_monotonic"
        ].sum()
    ),
    "/",
    len(df)
)

print(
    "normalized V-slope delta "
    "repair-vs-drop median/p95:",
    f"{v_delta.median():.6e}",
    "/",
    f"{v_delta.quantile(0.95):.6e}"
)

print(
    "normalized I-slope delta "
    "repair-vs-drop median/p95:",
    f"{i_delta.median():.6e}",
    "/",
    f"{i_delta.quantile(0.95):.6e}"
)

print(
    "MATR_TIMESTAMP_REPAIR_AUDIT_PASS=True"
)
