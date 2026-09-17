#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import os

import h5py
import numpy as np
import pandas as pd


BASE = Path(os.environ["BASE"])

FILES = {
    "b1": BASE / "data/raw/MATR/MATR_batch_20170512.mat",
    "b2": BASE / "data/raw/MATR/MATR_batch_20170630.mat",
    "b3": BASE / "data/raw/MATR/MATR_batch_20180412.mat",
    "b4": BASE / "data/raw/MATR/MATR_batch_20190124.mat",
}

OUT_CSV = (
    BASE
    / "data/audit/MATR_cycle_object_to_summary_alignment.csv"
)

OUT_JSON = (
    BASE
    / "data/audit/MATR_cycle_object_to_summary_alignment_manifest.json"
)


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def get_vector(group, name):
    return (
        np.hstack(
            group[name][0, :].tolist()
        )
        .astype(float)
    )


def deref(f, ds, idx):
    return (
        np.asarray(
            f[ds[idx, 0]][:]
        )
        .squeeze()
        .astype(float)
    )


def qd_amount(x):
    """
    Cycle-object Qd magnitude.

    Use range rather than terminal value so the diagnostic
    does not depend on whether the stored trajectory starts
    exactly at zero.
    """
    x = np.asarray(
        x,
        dtype=float
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        return np.nan

    return float(
        np.max(x)
        - np.min(x)
    )


def sample_indices(n):
    candidates = {
        0, 1, 2, 3, 4,
        n // 4,
        n // 2,
        (3 * n) // 4,
        n - 3,
        n - 2,
        n - 1,
    }

    return sorted(
        i
        for i in candidates
        if 0 <= i < n
    )


cell_rows = []
pair_rows = []

for batch, path in FILES.items():

    with h5py.File(
        path,
        "r"
    ) as f:

        b = f["batch"]
        n_cells = int(
            b["summary"].shape[0]
        )

        for i in range(n_cells):

            key = f"{batch}c{i}"

            summary = f[
                b["summary"][i, 0]
            ]

            cycles = f[
                b["cycles"][i, 0]
            ]

            qd_summary = get_vector(
                summary,
                "QDischarge"
            )

            n_summary = len(
                qd_summary
            )

            n_obj = int(
                cycles["Qd"].shape[0]
            )

            # -----------------------------------------------
            # Direct placeholder / endpoint evidence
            # -----------------------------------------------

            obj0 = (
                qd_amount(
                    deref(
                        f,
                        cycles["Qd"],
                        0
                    )
                )
                if n_obj > 0
                else np.nan
            )

            summary0 = (
                float(qd_summary[0])
                if n_summary
                else np.nan
            )

            obj_last = (
                qd_amount(
                    deref(
                        f,
                        cycles["Qd"],
                        n_obj - 1
                    )
                )
                if n_obj
                else np.nan
            )

            summary_last = (
                float(
                    qd_summary[-1]
                )
                if n_summary
                else np.nan
            )

            cell_rows.append({
                "batch":
                    batch,

                "raw_cell_index":
                    i,

                "cell_key":
                    key,

                "summary_rows":
                    n_summary,

                "cycle_object_rows":
                    n_obj,

                "cycle_minus_summary":
                    n_obj - n_summary,

                "summary_qd_first":
                    summary0,

                "object_qd_range_first":
                    obj0,

                "summary_qd_last":
                    summary_last,

                "object_qd_range_last":
                    obj_last,
            })

            # -----------------------------------------------
            # Structural alignment comparison.
            #
            # summary[k] vs cycle_object[k + offset]
            #
            # Only a small deterministic set of positions is
            # needed. No model feature generation occurs here.
            # -----------------------------------------------

            for k in sample_indices(
                n_summary
            ):

                qref = float(
                    qd_summary[k]
                )

                if not np.isfinite(qref):
                    continue

                for offset in [
                    -1, 0, 1, 2
                ]:

                    j = k + offset

                    if not (
                        0 <= j < n_obj
                    ):
                        continue

                    qobj = qd_amount(
                        deref(
                            f,
                            cycles["Qd"],
                            j
                        )
                    )

                    if not np.isfinite(qobj):
                        continue

                    pair_rows.append({
                        "batch":
                            batch,

                        "cell_key":
                            key,

                        "summary_index":
                            k,

                        "object_index":
                            j,

                        "offset":
                            offset,

                        "summary_QDischarge":
                            qref,

                        "object_Qd_range":
                            qobj,

                        "abs_error":
                            abs(
                                qobj
                                - qref
                            ),

                        "ratio":
                            (
                                qobj / qref
                                if abs(qref) > 1e-12
                                else np.nan
                            ),
                    })


cells = pd.DataFrame(
    cell_rows
)

pairs = pd.DataFrame(
    pair_rows
)

assert len(cells) == 185


# ============================================================
# Structural length contract
# ============================================================

length_counts = (
    cells[
        "cycle_minus_summary"
    ]
    .value_counts()
    .sort_index()
)

assert (
    int(
        (
            cells[
                "cycle_minus_summary"
            ] == 0
        ).sum()
    )
    == 184
)

special = cells[
    cells[
        "cell_key"
    ] == "b4c31"
].iloc[0]

assert (
    int(
        special[
            "cycle_minus_summary"
        ]
    )
    == -1
)


# ============================================================
# Direct Qd alignment metrics
# ============================================================

stats = {}

for offset in [
    -1, 0, 1, 2
]:

    x = pairs[
        pairs["offset"] == offset
    ]

    stats[offset] = {
        "n":
            int(len(x)),

        "median_abs_error":
            float(
                x[
                    "abs_error"
                ].median()
            ),

        "p95_abs_error":
            float(
                x[
                    "abs_error"
                ].quantile(0.95)
            ),

        "median_ratio":
            float(
                x[
                    "ratio"
                ]
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .dropna()
                .median()
            ),
    }


same = pairs[
    pairs["offset"] == 0
].copy()

same_nonzero = same[
    same[
        "summary_QDischarge"
    ].abs() > 1e-12
]

exact_1e6 = int(
    (
        same[
            "abs_error"
        ] <= 1e-6
    ).sum()
)

exact_1e4 = int(
    (
        same[
            "abs_error"
        ] <= 1e-4
    ).sum()
)


# ============================================================
# Batch-1 placeholder test
# ============================================================

b1 = cells[
    cells["batch"] == "b1"
].copy()

assert len(b1) == 46

b1_placeholder_pairs = int(
    (
        b1[
            "summary_qd_first"
        ].abs() <= 1e-12
    )
    .mul(
        b1[
            "object_qd_range_first"
        ].abs() <= 1e-12
    )
    .sum()
)


# ============================================================
# b4c31 known final unmatched placeholder
# ============================================================

b4c31_last_summary = float(
    special[
        "summary_qd_last"
    ]
)

b4c31_last_is_zero = bool(
    abs(
        b4c31_last_summary
    ) <= 1e-12
)


# ============================================================
# Hard alignment guards
# ============================================================

# The fixed same-index mapping must be superior to all
# neighbouring offset hypotheses.
for off in [-1, 1, 2]:
    assert (
        stats[0]["median_abs_error"]
        <
        stats[off]["median_abs_error"]
    )

    assert (
        stats[0]["p95_abs_error"]
        <
        stats[off]["p95_abs_error"]
    )

# Every sampled same-index pair must agree within 0.1 mAh.
assert exact_1e4 == len(same)

# Batch-1 index zero is a matched placeholder on BOTH sides,
# not evidence for a +1 shift.
assert b1_placeholder_pairs == 46

# Structural lengths:
# 184 cells are one-to-one; b4c31 has the known final
# unmatched zero QDischarge placeholder.
assert int(
    (
        cells["cycle_minus_summary"] == 0
    ).sum()
) == 184

assert int(
    (
        cells["cycle_minus_summary"] == -1
    ).sum()
) == 1

assert int(
    special["summary_rows"]
) == 834

assert int(
    special["cycle_object_rows"]
) == 833

assert b4c31_last_is_zero

FIXED_ALIGNMENT = (
    "summary[k] -> cycle_object[k]"
)


# ============================================================
# Save cell-level audit
# ============================================================

cells.to_csv(
    OUT_CSV,
    index=False
)

manifest = {
    "raw_cells":
        185,

    "length_delta_counts": {
        str(k): int(v)
        for k, v
        in length_counts.items()
    },

    "alignment_statistics": {
        str(k): v
        for k, v
        in stats.items()
    },

    "same_index_pairs":
        int(len(same)),

    "same_index_error_le_1e6":
        exact_1e6,

    "same_index_error_le_1e4":
        exact_1e4,

    "same_index_median_ratio_nonzero":
        float(
            same_nonzero[
                "ratio"
            ].median()
        ),

    "batch1_cells":
        46,

    "batch1_zero_placeholder_pairs":
        b1_placeholder_pairs,

    "b4c31_summary_rows":
        int(
            special[
                "summary_rows"
            ]
        ),

    "b4c31_cycle_object_rows":
        int(
            special[
                "cycle_object_rows"
            ]
        ),

    "b4c31_last_summary_qd":
        b4c31_last_summary,

    "b4c31_last_summary_is_zero":
        b4c31_last_is_zero,

    "audit_csv_sha256":
        sha256_file(
            OUT_CSV
        ),

    "raw_file_sha256": {
        str(
            p.relative_to(BASE)
        ):
            sha256_file(p)
        for p in FILES.values()
    },
}

OUT_JSON.write_text(
    json.dumps(
        manifest,
        indent=2
    )
    + "\n",
    encoding="utf-8"
)


# ============================================================
# Compact report only
# ============================================================

print("=" * 78)
print("MATR DIRECT Qd STRUCTURAL ALIGNMENT AUDIT")
print("=" * 78)

print(
    "raw cells:",
    len(cells)
)

print(
    "cycle_object_rows - summary_rows:",
    length_counts.to_dict()
)

print()
print("DIRECT Qd ERROR BY OFFSET [Ah]")

for offset in [
    -1, 0, 1, 2
]:
    s = stats[offset]

    print(
        f"offset {offset:+d}: "
        f"n={s['n']} "
        f"median={s['median_abs_error']:.9f} "
        f"p95={s['p95_abs_error']:.9f} "
        f"ratio={s['median_ratio']:.9f}"
    )

print()
print(
    "same-index pairs:",
    len(same)
)

print(
    "same-index error <=1e-6:",
    exact_1e6,
    "/",
    len(same)
)

print(
    "same-index error <=1e-4:",
    exact_1e4,
    "/",
    len(same)
)

print(
    "same-index nonzero median ratio:",
    f"{same_nonzero['ratio'].median():.9f}"
)

print()
print(
    "B1 zero-placeholder correspondence:",
    b1_placeholder_pairs,
    "/ 46"
)

print(
    "b4c31 rows summary/object:",
    int(
        special[
            "summary_rows"
        ]
    ),
    "/",
    int(
        special[
            "cycle_object_rows"
        ]
    )
)

print(
    "b4c31 final summary Qd:",
    f"{b4c31_last_summary:.12g}"
)

print(
    "b4c31 final summary is zero placeholder:",
    b4c31_last_is_zero
)

print()
print(
    "FIXED_INDEX_ALIGNMENT_CANDIDATE="
    "summary[k] -> cycle_object[k]"
)

print(
    "CELL_SPECIFIC_OFFSET_SELECTION_ALLOWED=False"
)

print()
print(
    "MATR_FIXED_ALIGNMENT="
    "summary[k] -> cycle_object[k]"
)

print(
    "MATR_DIRECT_ALIGNMENT_AUDIT_PASS=True"
)
