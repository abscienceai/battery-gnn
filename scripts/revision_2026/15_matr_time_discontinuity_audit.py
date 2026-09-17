#!/usr/bin/env python3

"""
Independent raw-data audit of material negative-time anomalies
in MATR batch 4.

Important:
    Candidate cycles are discovered directly from raw HDF5 time
    arrays.  Production feature_valid/failure_reason fields are
    NOT used to select anomalies.

Material backward-time criterion:
    dt_s < -1e-9 s

Previously established MATR time unit:
    stored t = minutes
"""

from pathlib import Path
import hashlib
import json
import os

import h5py
import numpy as np
import pandas as pd


BASE = Path(os.environ["BASE"])

RAW = (
    BASE
    / "data/raw/MATR/"
      "MATR_batch_20190124.mat"
)

OUT = (
    BASE
    / "data/audit/"
      "MATR_time_discontinuity_diagnostic.csv"
)

MANIFEST = (
    BASE
    / "data/audit/"
      "MATR_time_discontinuity_manifest.json"
)

TIME_TOLERANCE_S = 1e-9
EXPECTED_B4_CELLS = 45
EXPECTED_MATERIAL_CYCLES = 574


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def deref(f, ds, idx):
    return (
        np.asarray(
            f[ds[idx, 0]][:]
        )
        .squeeze()
        .astype(float)
        .reshape(-1)
    )


records = []

with h5py.File(RAW, "r") as f:

    batch = f["batch"]

    n_cells = int(
        batch["cycles"].shape[0]
    )

    assert n_cells == EXPECTED_B4_CELLS

    for cell_idx in range(n_cells):

        raw_cell_id = f"b4c{cell_idx}"

        cycles = f[
            batch["cycles"][
                cell_idx,
                0,
            ]
        ]

        n_cycles = int(
            cycles["t"].shape[0]
        )

        for cycle_idx in range(n_cycles):

            t = deref(
                f,
                cycles["t"],
                cycle_idx,
            )

            if len(t) < 2:
                continue

            dt_s = (
                np.diff(t)
                * 60.0
            )

            material_pos = np.where(
                np.isfinite(dt_s)
                & (
                    dt_s
                    < -TIME_TOLERANCE_S
                )
            )[0]

            if len(material_pos) == 0:
                continue

            # Previously observed contract:
            # each affected B4 cycle has exactly one material
            # negative-time edge.
            if len(material_pos) != 1:
                raise AssertionError(
                    f"{raw_cell_id} cycle {cycle_idx}: "
                    f"{len(material_pos)} material backward steps"
                )

            k = int(material_pos[0])
            bad_idx = k + 1

            if bad_idx + 1 >= len(t):
                raise AssertionError(
                    f"{raw_cell_id} cycle {cycle_idx}: "
                    "bad timestamp lacks right neighbour"
                )

            left = float(t[k])
            bad = float(t[bad_idx])
            right = float(t[bad_idx + 1])

            if not (
                np.isfinite(left)
                and np.isfinite(bad)
                and np.isfinite(right)
            ):
                raise AssertionError(
                    "Material anomaly neighbour is nonfinite"
                )

            # Remove only suspect sample.
            without = np.delete(
                t,
                bad_idx,
            )

            without_dt_s = (
                np.diff(without)
                * 60.0
            )

            removal_monotonic = bool(
                np.all(
                    without_dt_s
                    >= -TIME_TOLERANCE_S
                )
            )

            repaired = t.copy()

            repaired[bad_idx] = (
                left + right
            ) / 2.0

            repaired_dt_s = (
                np.diff(repaired)
                * 60.0
            )

            midpoint_monotonic = bool(
                np.all(
                    repaired_dt_s
                    >= -TIME_TOLERANCE_S
                )
            )

            immediate_recovery = bool(
                right > left
            )

            records.append({
                "batch":
                    "b4",

                "raw_cell_id":
                    raw_cell_id,

                "cycle_index":
                    int(cycle_idx),

                "samples":
                    int(len(t)),

                "boundary_index":
                    int(k),

                "bad_sample_index":
                    int(bad_idx),

                "t_before_min":
                    left,

                "t_bad_min":
                    bad,

                "t_after_min":
                    right,

                "backward_step_s":
                    float(
                        dt_s[k]
                    ),

                "single_sample_removal_monotonic":
                    removal_monotonic,

                "midpoint_repair_monotonic":
                    midpoint_monotonic,

                "immediate_recovery":
                    immediate_recovery,
            })


df = pd.DataFrame(records)

assert len(df) == EXPECTED_MATERIAL_CYCLES

assert (
    df["raw_cell_id"].nunique()
    == EXPECTED_B4_CELLS
)

assert (
    df["batch"] == "b4"
).all()

assert (
    df[
        "single_sample_removal_monotonic"
    ].all()
)

assert (
    df[
        "midpoint_repair_monotonic"
    ].all()
)

assert (
    df[
        "immediate_recovery"
    ].all()
)

assert (
    df[
        "backward_step_s"
    ]
    < -TIME_TOLERANCE_S
).all()

assert not df[
    [
        "raw_cell_id",
        "cycle_index",
    ]
].duplicated().any()


OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUT,
    index=False,
)

manifest = {
    "selection_source":
        "raw MATR batch-4 HDF5 time arrays",

    "production_failure_reason_used_for_selection":
        False,

    "time_tolerance_s":
        TIME_TOLERANCE_S,

    "affected_cells":
        int(
            df["raw_cell_id"].nunique()
        ),

    "material_anomaly_cycles":
        int(len(df)),

    "single_sample_removal_monotonic":
        int(
            df[
                "single_sample_removal_monotonic"
            ].sum()
        ),

    "midpoint_repair_monotonic":
        int(
            df[
                "midpoint_repair_monotonic"
            ].sum()
        ),

    "immediate_recovery":
        int(
            df[
                "immediate_recovery"
            ].sum()
        ),

    "raw_file_sha256":
        sha256_file(RAW),

    "audit_csv_sha256":
        sha256_file(OUT),

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
print("MATR RAW B4 TIME-DISCONTINUITY AUDIT")
print("=" * 78)

print(
    "selection uses production failure_reason:",
    False
)

print(
    "B4 raw cells:",
    n_cells
)

print(
    "material anomaly cycles:",
    len(df)
)

print(
    "affected cells:",
    df[
        "raw_cell_id"
    ].nunique()
)

print(
    "single-sample removal monotonic:",
    int(
        df[
            "single_sample_removal_monotonic"
        ].sum()
    ),
    "/",
    len(df)
)

print(
    "midpoint repair monotonic:",
    int(
        df[
            "midpoint_repair_monotonic"
        ].sum()
    ),
    "/",
    len(df)
)

print(
    "immediate recovery:",
    int(
        df[
            "immediate_recovery"
        ].sum()
    ),
    "/",
    len(df)
)

print(
    "MATR_TIME_DISCONTINUITY_AUDIT_PASS=True"
)
