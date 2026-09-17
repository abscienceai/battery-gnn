#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd


BASE = Path(os.environ["BASE"])

ZYK = (
    BASE
    / "data/audit/"
      "RWTH_ZYK_cycle_reconstruction_all_cells.csv"
)

TARGET = (
    BASE
    / "data/processed/revision_2026/"
      "unified_label_backbone.csv"
)

OUT_CSV = (
    BASE
    / "data/audit/"
      "RWTH_target_to_ZYK_TS_causal_alignment.csv"
)

OUT_JSON = (
    BASE
    / "data/audit/"
      "RWTH_target_to_ZYK_TS_causal_alignment_manifest.json"
)

OUT_TXT = (
    BASE
    / "data/audit/"
      "RWTH_target_to_ZYK_TS_causal_alignment_summary.txt"
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


def canonical_cell(x):
    m = re.search(
        r"(\d{3})$",
        str(x)
    )

    if m is None:
        raise ValueError(
            f"Cannot parse RWTH cell ID: {x}"
        )

    n = int(m.group(1))

    if not 2 <= n <= 49:
        raise ValueError(
            f"RWTH cell outside 002-049: {x}"
        )

    return f"{n:03d}"


def ts_number(x):
    m = re.search(
        r"TS0*(\d+)",
        str(x),
        flags=re.I
    )

    if m is None:
        raise ValueError(
            f"Cannot extract TS identifier: {x}"
        )

    return int(m.group(1))


def embedded_datetime(x):
    m = re.search(
        r"(20\d{2}-\d{2}-\d{2})[ _-]+(\d{6})",
        str(x)
    )

    if m is None:
        return pd.NaT

    return pd.to_datetime(
        f"{m.group(1)} {m.group(2)}",
        format="%Y-%m-%d %H%M%S",
        errors="coerce"
    )


# ============================================================
# Load sources
# ============================================================

z = pd.read_csv(ZYK)

required_zyk = [
    "cell_id",
    "cycle_number_pre_filter",
    "cycle_number_after_filter",
    "Qd_integrated_Ah",
    "source_zip_first",
    "source_zip_last",
]

for c in required_zyk:
    assert c in z.columns, c


t = pd.read_csv(
    TARGET,
    usecols=[
        "dataset",
        "cell_id",
        "observation_index",
        "life_coordinate",
        "source_file",
    ],
)

t = t[
    t["dataset"] == "RWTH"
].copy()

assert len(t) == 777


# ============================================================
# Native -> canonical ID mapping
# ============================================================

z["canonical_cell"] = (
    z["cell_id"]
    .map(canonical_cell)
)

t["canonical_cell"] = (
    t["cell_id"]
    .map(canonical_cell)
)

expected_cells = {
    f"{i:03d}"
    for i in range(2, 50)
}

assert (
    set(z["canonical_cell"])
    == expected_cells
)

assert (
    set(t["canonical_cell"])
    == expected_cells
)


# ============================================================
# Source chronology
# ============================================================

z["zyk_ts_first"] = (
    z["source_zip_first"]
    .map(ts_number)
)

z["zyk_ts_last"] = (
    z["source_zip_last"]
    .map(ts_number)
)

z["embedded_datetime"] = (
    z["source_zip_last"]
    .map(embedded_datetime)
)

t["target_ts"] = (
    t["source_file"]
    .map(ts_number)
)

assert z["zyk_ts_first"].notna().all()
assert z["zyk_ts_last"].notna().all()
assert t["target_ts"].notna().all()
assert z["embedded_datetime"].notna().all()


# TS ordering must agree with embedded chronology.
for cell, g in z.groupby(
    "canonical_cell"
):
    archives = (
        g[
            [
                "zyk_ts_last",
                "embedded_datetime",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "zyk_ts_last"
        )
    )

    d = archives[
        "embedded_datetime"
    ].to_numpy()

    assert not np.any(
        d[1:] < d[:-1]
    ), cell


# Characterization TS must also be chronological.
for cell, g in t.groupby(
    "canonical_cell"
):
    g = g.sort_values(
        "observation_index"
    )

    a = g[
        "target_ts"
    ].to_numpy(dtype=int)

    assert np.all(
        np.diff(a) > 0
    ), cell


# ============================================================
# Canonical cycle eligibility
# ============================================================

post_filter = (
    z[
        "cycle_number_after_filter"
    ].notna()
)

bad_qd = (
    ~np.isfinite(
        z["Qd_integrated_Ah"]
    )
    |
    (
        z["Qd_integrated_Ah"]
        <= 0
    )
)

# Previously diagnosed 538 bad Qd rows must all be rejected.
assert int(bad_qd.sum()) == 538
assert int((bad_qd & post_filter).sum()) == 0

zk = z[
    post_filter
].copy()

assert len(zk) == 105006


# ============================================================
# Strict causal target -> sensor-history mapping
# ============================================================

records = []

for cell in sorted(
    expected_cells
):
    zg = (
        zk[
            zk["canonical_cell"] == cell
        ]
        .sort_values(
            "cycle_number_after_filter"
        )
        .reset_index(drop=True)
    )

    tg = (
        t[
            t["canonical_cell"] == cell
        ]
        .sort_values(
            "observation_index"
        )
        .reset_index(drop=True)
    )

    previous_count = -1

    for _, tr in tg.iterrows():

        target_ts = int(
            tr["target_ts"]
        )

        # STRICT causal rule:
        #
        # A ZYK cycle is available only when the LAST source
        # archive contributing to that cycle precedes the
        # characterization test.
        #
        # target EFC is NOT used for alignment.
        prior = zg[
            zg["zyk_ts_last"]
            < target_ts
        ]

        count = len(prior)

        assert count >= previous_count
        previous_count = count

        if count:
            latest = prior.iloc[-1]

            latest_cycle = float(
                latest[
                    "cycle_number_after_filter"
                ]
            )

            latest_ts = int(
                latest["zyk_ts_last"]
            )

            assert latest_ts < target_ts

        else:
            latest_cycle = np.nan
            latest_ts = np.nan

        records.append({
            "dataset":
                "RWTH",

            "canonical_cell":
                cell,

            "target_native_cell_id":
                tr["cell_id"],

            "target_observation_index":
                int(
                    tr[
                        "observation_index"
                    ]
                ),

            "target_ts":
                target_ts,

            # Retained only for target provenance.
            # Explicitly NOT used for alignment.
            "target_life_coordinate":
                float(
                    tr[
                        "life_coordinate"
                    ]
                ),

            "prior_zyk_cycle_count":
                int(count),

            "latest_prior_zyk_cycle":
                latest_cycle,

            "latest_prior_zyk_ts":
                latest_ts,

            "has_sensor_history":
                bool(count > 0),

            "alignment_rule":
                "zyk_ts_last < target_ts",

            "target_efc_used_for_alignment":
                False,
        })


a = pd.DataFrame(records)

assert len(a) == 777
assert (
    a["canonical_cell"]
    .nunique()
    == 48
)


# ============================================================
# Global invariants
# ============================================================

with_history = int(
    a["has_sensor_history"].sum()
)

without_history = int(
    (~a["has_sensor_history"]).sum()
)

assert with_history == 729
assert without_history == 48


# Every first characterization must be BOL/no-history.
first = (
    a.sort_values(
        [
            "canonical_cell",
            "target_observation_index",
        ]
    )
    .groupby(
        "canonical_cell",
        sort=True
    )
    .first()
)

assert len(first) == 48

assert (
    ~first[
        "has_sensor_history"
    ].astype(bool)
).all()


# Every later characterization must have history.
ordered = (
    a.sort_values(
        [
            "canonical_cell",
            "target_observation_index",
        ]
    )
    .copy()
)

later = ordered[
    ordered.groupby(
        "canonical_cell"
    ).cumcount() > 0
]

assert later[
    "has_sensor_history"
].astype(bool).all()


# No future TS can appear in a mapped history endpoint.
mapped = a[
    a["has_sensor_history"]
]

assert (
    mapped[
        "latest_prior_zyk_ts"
    ]
    <
    mapped[
        "target_ts"
    ]
).all()


# History size must never shrink over time.
for cell, g in a.groupby(
    "canonical_cell"
):
    g = g.sort_values(
        "target_observation_index"
    )

    assert g[
        "prior_zyk_cycle_count"
    ].is_monotonic_increasing


# ============================================================
# Save artifacts
# ============================================================

OUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)

a.to_csv(
    OUT_CSV,
    index=False
)

manifest = {
    "purpose":
        (
            "Strictly causal alignment between RWTH "
            "characterization targets and prior ZYK "
            "sensor-cycle history."
        ),

    "zyk_input":
        str(
            ZYK.relative_to(BASE)
        ),

    "target_input":
        str(
            TARGET.relative_to(BASE)
        ),

    "zyk_input_sha256":
        sha256_file(ZYK),

    "target_input_sha256":
        sha256_file(TARGET),

    "alignment_csv_sha256":
        sha256_file(OUT_CSV),

    "source_cycle_rows":
        int(len(z)),

    "post_filter_cycle_rows":
        int(len(zk)),

    "target_rows":
        int(len(a)),

    "cells":
        48,

    "targets_with_sensor_history":
        with_history,

    "targets_without_sensor_history":
        without_history,

    "first_characterization_without_history":
        48,

    "bad_qd_rows_total":
        int(bad_qd.sum()),

    "bad_qd_rows_surviving_filter":
        int(
            (bad_qd & post_filter).sum()
        ),

    "alignment_rule":
        "zyk_ts_last < target_ts",

    "target_efc_used_for_alignment":
        False,

    "throughput_equality_required":
        False,

    "strict_future_exclusion":
        True,

    "audit_pass":
        True,
}

OUT_JSON.write_text(
    json.dumps(
        manifest,
        indent=2
    )
    + "\n",
    encoding="utf-8"
)

summary = [
    "RWTH strict causal alignment audit",
    "",
    "cells=48",
    "source_ZYK_rows=105663",
    "post_filter_ZYK_rows=105006",
    "target_rows=777",
    "targets_with_history=729",
    "targets_without_history=48",
    "first_characterization_without_history=48/48",
    "bad_Qd_rows=538",
    "bad_Qd_surviving_filter=0",
    "",
    "alignment_rule=zyk_ts_last < target_ts",
    "target_EFC_used_for_alignment=False",
    "throughput_equality_required=False",
    "future_sensor_cycles_allowed=False",
    "",
    "RWTH_CAUSAL_ALIGNMENT_AUDIT_PASS=True",
]

OUT_TXT.write_text(
    "\n".join(summary)
    + "\n",
    encoding="utf-8"
)


# ============================================================
# Compact terminal report
# ============================================================

print("=" * 78)
print("RWTH STRICT CAUSAL ALIGNMENT AUDIT")
print("=" * 78)

print(
    "source / post-filter ZYK cycles:",
    len(z),
    "/",
    len(zk)
)

print(
    "targets / cells:",
    len(a),
    "/",
    a["canonical_cell"].nunique()
)

print(
    "targets with / without history:",
    with_history,
    "/",
    without_history
)

print(
    "first characterization without history:",
    int(
        (
            ~first[
                "has_sensor_history"
            ].astype(bool)
        ).sum()
    ),
    "/ 48"
)

print(
    "bad Qd total / surviving filter:",
    int(bad_qd.sum()),
    "/",
    int(
        (bad_qd & post_filter).sum()
    )
)

print(
    "TS chronology:",
    "PASS"
)

print(
    "strict future exclusion:",
    "PASS"
)

print(
    "alignment rule:",
    "zyk_ts_last < target_ts"
)

print(
    "target EFC used for alignment:",
    False
)

print(
    "CSV SHA256:",
    sha256_file(OUT_CSV)
)

print()
print(
    "RWTH_CAUSAL_ALIGNMENT_AUDIT_PASS=True"
)
