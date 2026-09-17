#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import os
import subprocess

import numpy as np
import pandas as pd


BASE = Path(os.environ["BASE"])
REPO = Path(os.environ["REPO"])

DATA = (
    BASE
    / "data/processed/revision_2026/"
      "unified_label_backbone.csv"
)

MANIFEST = (
    BASE
    / "data/processed/revision_2026/"
      "unified_label_backbone_manifest.json"
)

GENERATOR = (
    REPO
    / "scripts/revision_2026/"
      "08_unified_label_backbone.py"
)

OUT_JSON = (
    BASE
    / "data/audit/"
      "unified_label_backbone_independent_audit.json"
)

OUT_TXT = (
    BASE
    / "data/audit/"
      "unified_label_backbone_independent_audit_summary.txt"
)

OUT_JSON.parent.mkdir(
    parents=True,
    exist_ok=True
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


def resolve_source(path_text):
    p = Path(path_text)

    if p.is_absolute():
        return p

    return BASE / p


def aggregate_source_sha256(paths):
    h = hashlib.sha256()

    paths = sorted(
        [Path(x).resolve() for x in paths],
        key=lambda x: str(x)
    )

    for path in paths:
        assert path.is_file(), path

        try:
            name = str(
                path.relative_to(BASE)
            )
        except ValueError:
            name = str(path)

        digest = sha256_file(path)

        h.update(
            f"{name}\t{digest}\n".encode("utf-8")
        )

    return h.hexdigest()


def fail(msg):
    raise AssertionError(msg)


# ============================================================
# Load artifact without importing generator
# ============================================================

assert DATA.is_file()
assert MANIFEST.is_file()
assert GENERATOR.is_file()

df = pd.read_csv(DATA)
manifest = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# 1. Exact schema / no feature contamination
# ============================================================

EXPECTED_COLUMNS = [
    "dataset",
    "cohort",
    "cell_id",
    "dataset_cell_key",
    "split_group",
    "observation_index",
    "native_cycle",
    "life_coordinate",
    "life_coordinate_name",
    "capacity_Ah",
    "SOH",
    "eol_observation_index",
    "cell_reaches_eol",
    "is_eol_observation",
    "is_at_or_before_eol",
    "RUL_observations",
    "native_RUL",
    "target_source",
    "eol_basis",
    "capacity_is_observed",
    "SOH_is_derived",
    "RUL_is_derived",
    "source_lifetime_label",
    "censoring_status",
    "valid_capacity",
    "source_file",
]

assert list(df.columns) == EXPECTED_COLUMNS, (
    "Unexpected unified backbone schema"
)

# split_group is an identifier only, not an assignment.
FORBIDDEN_ASSIGNMENT_COLUMNS = {
    "split",
    "partition",
    "fold",
    "train",
    "val",
    "validation",
    "test",
    "is_train",
    "is_val",
    "is_test",
    "scaler",
    "scaled",
    "feature_vector",
    "x",
}

assert not (
    set(c.lower() for c in df.columns)
    & FORBIDDEN_ASSIGNMENT_COLUMNS
)

assert manifest[
    "contains_model_features"
] is False

assert manifest[
    "scaler_applied"
] is False

assert manifest[
    "split_assignment_applied"
] is False


# ============================================================
# 2. Global dimensions and canonical key integrity
# ============================================================

assert len(df) == 301372

expected_cells = {
    "HUST": 77,
    "CALCE": 13,
    "MATR": 169,
    "RWTH": 48,
}

expected_rows = {
    "HUST": 146122,
    "CALCE": 14298,
    "MATR": 140175,
    "RWTH": 777,
}

assert set(df["dataset"]) == set(
    expected_cells
)

for dataset in expected_cells:

    g = df[
        df["dataset"] == dataset
    ]

    assert len(g) == expected_rows[
        dataset
    ]

    assert (
        g["cell_id"].nunique()
        == expected_cells[dataset]
    )

assert (
    df[
        [
            "dataset",
            "cell_id",
            "observation_index",
        ]
    ]
    .duplicated()
    .sum()
    == 0
)

assert (
    df["dataset_cell_key"]
    ==
    (
        df["dataset"].astype(str)
        + "::"
        + df["cell_id"].astype(str)
    )
).all()

assert (
    df["split_group"]
    ==
    df["dataset_cell_key"]
).all()

assert (
    df["dataset_cell_key"].nunique()
    == 307
)


# Each cell must have exactly 0...N-1 observation indexing.
bad_index_cells = []

for key, g in df.groupby(
    "dataset_cell_key",
    sort=False
):

    obs = np.sort(
        g[
            "observation_index"
        ].to_numpy(
            dtype=int
        )
    )

    expected = np.arange(
        len(g),
        dtype=int
    )

    if not np.array_equal(
        obs,
        expected
    ):
        bad_index_cells.append(key)

assert not bad_index_cells


# ============================================================
# 3. Generic EOL / RUL invariants
# ============================================================

event_cells = 0
censored_cells = 0

for key, g in df.groupby(
    "dataset_cell_key",
    sort=False
):

    g = g.sort_values(
        "observation_index"
    )

    reaches = set(
        g[
            "cell_reaches_eol"
        ].astype(bool)
    )

    assert len(reaches) == 1

    reaches = reaches.pop()

    eol_flags = int(
        g[
            "is_eol_observation"
        ].astype(bool)
        .sum()
    )

    finite_rul = g[
        "RUL_observations"
    ].notna()

    if reaches:

        event_cells += 1

        assert eol_flags == 1

        eol_row = g[
            g[
                "is_eol_observation"
            ].astype(bool)
        ].iloc[0]

        eol_idx = int(
            eol_row[
                "observation_index"
            ]
        )

        assert np.isclose(
            float(
                eol_row[
                    "RUL_observations"
                ]
            ),
            0.0
        )

        # Before EOL: RUL must be finite and exactly distance-to-EOL.
        before = g[
            g["observation_index"]
            <= eol_idx
        ]

        expected_rul = (
            eol_idx
            - before[
                "observation_index"
            ].to_numpy(
                dtype=float
            )
        )

        actual_rul = before[
            "RUL_observations"
        ].to_numpy(
            dtype=float
        )

        assert np.allclose(
            actual_rul,
            expected_rul,
            atol=0,
            rtol=0,
        )

        # After EOL: canonical RUL must never be provided.
        after = g[
            g["observation_index"]
            > eol_idx
        ]

        assert after[
            "RUL_observations"
        ].isna().all()

        assert (
            before[
                "is_at_or_before_eol"
            ].astype(bool)
            .all()
        )

        assert not (
            after[
                "is_at_or_before_eol"
            ].astype(bool)
            .any()
        )

    else:

        censored_cells += 1

        assert eol_flags == 0
        assert not finite_rul.any()

        assert g[
            "eol_observation_index"
        ].isna().all()

        assert not g[
            "is_at_or_before_eol"
        ].astype(bool).any()


assert event_cells == 226
assert censored_cells == 81


# ============================================================
# 4. HUST independent checks
# ============================================================

h = df[
    df["dataset"] == "HUST"
].copy()

assert h["cell_id"].nunique() == 77
assert h["valid_capacity"].astype(bool).all()
assert h["cell_reaches_eol"].astype(bool).all()

assert np.allclose(
    h["SOH"].to_numpy(dtype=float),
    h["capacity_Ah"].to_numpy(dtype=float)
    / 1.10,
    rtol=1e-10,
    atol=1e-12,
)

assert h["native_RUL"].notna().all()

assert np.allclose(
    h["RUL_observations"].to_numpy(
        dtype=float
    ),
    h["native_RUL"].to_numpy(
        dtype=float
    ) - 1.0,
    rtol=0,
    atol=0,
)

for _, g in h.groupby(
    "cell_id"
):

    g = g.sort_values(
        "observation_index"
    )

    assert int(
        g.iloc[-1][
            "is_eol_observation"
        ]
    ) == 1

    assert np.isclose(
        float(
            g.iloc[-1][
                "native_RUL"
            ]
        ),
        1.0
    )


# ============================================================
# 5. CALCE independent first-crossing checks
# ============================================================

c = df[
    df["dataset"] == "CALCE"
].copy()

assert c["cell_id"].nunique() == 13

for _, g in c.groupby(
    "cell_id"
):

    g = g.sort_values(
        "observation_index"
    )

    eol = g[
        g[
            "is_eol_observation"
        ].astype(bool)
    ].iloc[0]

    idx = int(
        eol[
            "observation_index"
        ]
    )

    assert float(
        eol["SOH"]
    ) <= 0.80 + 1e-12

    previous = g[
        g["observation_index"]
        < idx
    ]

    if len(previous):
        assert (
            previous["SOH"]
            > 0.80
        ).all()


# ============================================================
# 6. RWTH independent Q0 / first-crossing checks
# ============================================================

r = df[
    df["dataset"] == "RWTH"
].copy()

assert len(r) == 777
assert r["cell_id"].nunique() == 48

for _, g in r.groupby(
    "cell_id"
):

    g = g.sort_values(
        "observation_index"
    )

    q0 = float(
        g.iloc[0][
            "capacity_Ah"
        ]
    )

    recomputed = (
        g[
            "capacity_Ah"
        ].to_numpy(dtype=float)
        / q0
    )

    assert np.allclose(
        g["SOH"].to_numpy(
            dtype=float
        ),
        recomputed,
        rtol=1e-12,
        atol=1e-12,
    )

    assert np.isclose(
        float(g.iloc[0]["SOH"]),
        1.0
    )

    eol = g[
        g[
            "is_eol_observation"
        ].astype(bool)
    ].iloc[0]

    idx = int(
        eol[
            "observation_index"
        ]
    )

    assert float(
        eol["SOH"]
    ) <= 0.80 + 1e-12

    previous = g[
        g["observation_index"]
        < idx
    ]

    if len(previous):
        assert (
            previous["SOH"]
            > 0.80
        ).all()


# ============================================================
# 7. MATR censoring/event semantics
# ============================================================

m = df[
    df["dataset"] == "MATR"
].copy()

assert m["cell_id"].nunique() == 169

per_cell = (
    m.groupby(
        "cell_id"
    )
    .agg(
        reaches=(
            "cell_reaches_eol",
            "max"
        ),
        eol_count=(
            "is_eol_observation",
            "sum"
        ),
        finite_rul=(
            "RUL_observations",
            lambda x: int(
                x.notna().sum()
            )
        ),
        n=(
            "observation_index",
            "size"
        ),
        source_label=(
            "source_lifetime_label",
            "first"
        ),
        censoring_status=(
            "censoring_status",
            "first"
        ),
    )
)

observed = per_cell[
    per_cell["reaches"].astype(bool)
]

censored = per_cell[
    ~per_cell["reaches"].astype(bool)
]

assert len(observed) == 88
assert len(censored) == 81

assert observed[
    "eol_count"
].eq(1).all()

assert censored[
    "eol_count"
].eq(0).all()

assert censored[
    "finite_rul"
].eq(0).all()

assert (
    censored[
        "censoring_status"
    ]
    == "right_censored_len_plus_1"
).all()

assert np.allclose(
    censored[
        "source_label"
    ].to_numpy(dtype=float),
    censored[
        "n"
    ].to_numpy(dtype=float)
    + 1.0,
    atol=0,
    rtol=0,
)

# For observed MATR cells, source lifetime label corresponds
# to the one-based EOL observation position.
for cell_id, g in m[
    m["cell_reaches_eol"].astype(bool)
].groupby("cell_id"):

    g = g.sort_values(
        "observation_index"
    )

    source_label = float(
        g[
            "source_lifetime_label"
        ].iloc[0]
    )

    eol_idx = int(
        g.loc[
            g[
                "is_eol_observation"
            ].astype(bool),
            "observation_index",
        ].iloc[0]
    )

    assert np.isclose(
        source_label,
        eol_idx + 1
    )


# ============================================================
# 8. Manifest and cryptographic provenance
# ============================================================

csv_sha = sha256_file(DATA)
generator_sha = sha256_file(
    GENERATOR
)

assert csv_sha == manifest[
    "artifact_sha256"
]

assert generator_sha == manifest[
    "generator_sha256"
]

assert list(df.columns) == manifest[
    "columns"
]

# Verify every directly listed input hash.
input_hashes = manifest[
    "input_file_sha256"
]

assert len(input_hashes) == 21

for name, expected_sha in (
    input_hashes.items()
):

    path = resolve_source(name)

    assert path.is_file(), path

    actual = sha256_file(
        path
    )

    assert actual == expected_sha, (
        f"Input hash mismatch: {name}"
    )


# Independently rebuild RWTH 777-file aggregate hash
rwth_paths = sorted(
    {
        resolve_source(x).resolve()
        for x in r[
            "source_file"
        ].astype(str)
    },
    key=lambda x: str(x)
)

assert len(rwth_paths) == 777

rwth_aggregate = (
    aggregate_source_sha256(
        rwth_paths
    )
)

prov = manifest[
    "direct_source_provenance"
]

assert prov[
    "MATR_raw_files_hashed"
] == 4

assert prov[
    "RWTH_CapDCH1_source_files_hashed"
] == 777

assert (
    rwth_aggregate
    ==
    prov[
        "RWTH_CapDCH1_source_aggregate_sha256"
    ]
)


# Verify official RWTH source repository state independently.
OFF = (
    BASE
    / "external/"
      "RWTH_battery_degradation_trajectory_prediction"
)

repo_head = subprocess.check_output(
    [
        "git",
        "-C",
        str(OFF),
        "rev-parse",
        "HEAD",
    ],
    text=True,
).strip()

repo_status = subprocess.check_output(
    [
        "git",
        "-C",
        str(OFF),
        "status",
        "--porcelain",
    ],
    text=True,
).strip()

assert (
    repo_head
    ==
    prov[
        "RWTH_external_repository_commit"
    ]
)

assert (
    (repo_status == "")
    ==
    prov[
        "RWTH_external_repository_clean"
    ]
)


# ============================================================
# 9. Audit result
# ============================================================

result = {
    "artifact_rows":
        int(len(df)),

    "dataset_cells":
        int(
            df[
                "dataset_cell_key"
            ].nunique()
        ),

    "duplicate_keys":
        int(
            df[
                [
                    "dataset",
                    "cell_id",
                    "observation_index",
                ]
            ]
            .duplicated()
            .sum()
        ),

    "event_cells":
        int(event_cells),

    "right_censored_cells":
        int(censored_cells),

    "MATR_observed_event_cells":
        int(len(observed)),

    "MATR_right_censored_cells":
        int(len(censored)),

    "csv_sha256":
        csv_sha,

    "generator_sha256":
        generator_sha,

    "RWTH_source_aggregate_sha256":
        rwth_aggregate,

    "RWTH_source_repo_commit":
        repo_head,

    "input_hash_entries_verified":
        int(len(input_hashes)),

    "contains_model_features":
        False,

    "scaler_assignment_present":
        False,

    "split_assignment_present":
        False,

    "independent_audit_pass":
        True,
}

OUT_JSON.write_text(
    json.dumps(
        result,
        indent=2
    )
    + "\n",
    encoding="utf-8"
)

summary = [
    "Independent unified label-backbone audit",
    "",
    f"rows={len(df)}",
    (
        "dataset_cells="
        f"{df['dataset_cell_key'].nunique()}"
    ),
    "duplicate_keys=0",
    "",
    f"event_cells={event_cells}",
    f"right_censored_cells={censored_cells}",
    (
        "MATR observed/censored="
        f"{len(observed)}/{len(censored)}"
    ),
    "",
    "HUST native-RUL conversion: PASS",
    "CALCE first-80%-crossing semantics: PASS",
    "RWTH Q0-normalized first-80%-crossing semantics: PASS",
    "MATR event/censoring semantics: PASS",
    "Post-EOL RUL leakage: NONE",
    "Censored-cell RUL leakage: NONE",
    "Canonical key continuity: PASS",
    "No feature/scaler/split assignment contamination: PASS",
    f"input hashes verified={len(input_hashes)}",
    "RWTH 777-source aggregate hash: PASS",
    "RWTH external repository provenance: PASS",
    "",
    f"CSV SHA256={csv_sha}",
    f"GENERATOR SHA256={generator_sha}",
    "",
    "UNIFIED_LABEL_BACKBONE_INDEPENDENT_AUDIT_PASS=True",
]

OUT_TXT.write_text(
    "\n".join(summary)
    + "\n",
    encoding="utf-8"
)


print("=" * 78)
print("UNIFIED LABEL BACKBONE — INDEPENDENT AUDIT")
print("=" * 78)

print(
    "rows / dataset-cells:",
    len(df),
    "/",
    df[
        "dataset_cell_key"
    ].nunique()
)

print(
    "duplicate canonical keys:",
    0
)

print(
    "event / right-censored cells:",
    event_cells,
    "/",
    censored_cells
)

print(
    "MATR observed / censored:",
    len(observed),
    "/",
    len(censored)
)

print(
    "HUST RUL conversion:",
    "PASS"
)

print(
    "CALCE first-crossing:",
    "PASS"
)

print(
    "RWTH first-crossing:",
    "PASS"
)

print(
    "MATR censoring semantics:",
    "PASS"
)

print(
    "post-EOL RUL leakage:",
    "NONE"
)

print(
    "censored RUL leakage:",
    "NONE"
)

print(
    "input hashes verified:",
    len(input_hashes)
)

print(
    "RWTH source files verified:",
    len(rwth_paths)
)

print(
    "CSV SHA256:",
    csv_sha
)

print(
    "generator SHA256:",
    generator_sha
)

print()
print(
    "UNIFIED_LABEL_BACKBONE_INDEPENDENT_AUDIT_PASS=True"
)
