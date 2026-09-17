from pathlib import Path
import csv
import hashlib
import importlib.util
import io
import json
import os
import pickle
import re
import subprocess
import time
import zipfile

import h5py
import numpy as np
import pandas as pd


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo" / "battery-gnn"

THIS_SCRIPT = (
    REPO
    / "scripts"
    / "revision_2026"
    / "53_zero_corruption_raw_roundtrip.py"
)

GENERATOR = (
    REPO
    / "scripts"
    / "revision_2026"
    / "14_raw_cycle_feature_generator.py"
)

KERNEL_PATH = (
    REPO
    / "scripts"
    / "revision_2026"
    / "cycle_feature_kernel.py"
)

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

ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "physical_sensor_noise"
    / "final_physical_v1"
)

PROTOCOL = ROOT / "protocol.json"

ROUNDTRIP = ROOT / "zero_corruption_roundtrip"
CELL_DIR = ROUNDTRIP / "cells"

PROGRESS = ROUNDTRIP / "progress.json"
ROW_AUDIT = ROUNDTRIP / "roundtrip_row_audit.csv"
SUMMARY = ROUNDTRIP / "roundtrip_summary.json"


EXPECTED_PROTOCOL_SHA = (
    "7ae0836fe69c4b14a1f93a094b2d42bb96209e302c8262f7b8bd8625baf78bfc"
)

EXPECTED_CANON_SHA = (
    "49a7b7629cf9ee3f8ce0a7e3c52f2807878b752b4c9a30a876285fb891b00c14"
)

EXPECTED_ASSIGNMENTS_SHA = (
    "914c8b90131f496ca396f63e0502147ca99848fdfd4b2c6b2a5887225b47d055"
)


FEATURES = [
    "v_min",
    "v_q10",
    "v_q25",
    "v_median",
    "v_q75",
    "v_q90",
    "v_max",
    "v_mean",
    "v_std",
    "v_phase_slope",
    "i_min",
    "i_q10",
    "i_q25",
    "i_median",
    "i_q75",
    "i_q90",
    "i_max",
    "i_mean",
    "i_std",
    "i_phase_slope",
]

DATASETS = [
    "HUST",
    "CALCE",
    "MATR",
]

SPLIT_SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

EXPECTED_TEST_ROWS = {
    0: 43974,
    1: 46273,
    2: 43818,
    3: 46156,
    4: 43749,
}

ATOL = 1e-8
RTOL = 1e-8


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


def atomic_json(obj, path):
    path = Path(path)

    tmp = Path(
        str(path) + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def atomic_csv(df, path):
    path = Path(path)

    tmp = Path(
        str(path) + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
        float_format="%.17g",
    )

    os.replace(
        tmp,
        path,
    )


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


def canonical_digest(df):
    cols = [
        "dataset",
        "split_group",
        "feature_source_cell_id",
        "feature_source_member",
        "feature_raw_cycle_key",
        "feature_raw_cycle_position_zero_based",
        "feature_contract_version",
        "feature_kernel_sha256",
    ] + FEATURES

    q = (
        df[
            cols
        ]
        .sort_values(
            [
                "feature_source_cell_id",
                "feature_raw_cycle_position_zero_based",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    payload = q.to_csv(
        index=False,
        float_format="%.17g",
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


def norm_key(x):
    if pd.isna(x):
        return "<NA>"

    s = str(x).strip()

    if re.fullmatch(
        r"[+-]?\d+\.0+",
        s,
    ):
        return str(
            int(
                float(s)
            )
        )

    return s


def safe_name(dataset, cell):
    token = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(cell),
    )

    return (
        dataset
        + "__"
        + token
    )


def raw_stat(path):
    st = Path(path).stat()

    return {
        "path":
            str(
                Path(path)
                .resolve()
            ),

        "size":
            int(
                st.st_size
            ),

        "mtime_ns":
            int(
                st.st_mtime_ns
            ),
    }


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================================
# Immutable guards
# ============================================================================

assert sha256_file(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

assert sha256_file(
    CANON
) == EXPECTED_CANON_SHA

assert sha256_file(
    ASSIGNMENTS
) == EXPECTED_ASSIGNMENTS_SHA


protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)

assert protocol[
    "status"
] == "FINAL_FROZEN_BEFORE_ANY_PHYSICAL_SENSOR_CORRUPTION_EVALUATION"

assert protocol[
    "clean_roundtrip_barrier"
][
    "required_before_noisy_test_metrics"
] is True

assert protocol[
    "clean_roundtrip_barrier"
][
    "feature_tolerance"
][
    "absolute"
] == ATOL

assert protocol[
    "clean_roundtrip_barrier"
][
    "feature_tolerance"
][
    "relative"
] == RTOL


assert sha256_file(
    GENERATOR
) == protocol[
    "source_sha256"
][
    "raw_cycle_generator"
]

assert sha256_file(
    KERNEL_PATH
) == protocol[
    "source_sha256"
][
    "cycle_feature_kernel"
]


tracked = subprocess.run(
    [
        "git",
        "ls-files",
        "--error-unmatch",
        str(
            THIS_SCRIPT.relative_to(
                REPO
            )
        ),
    ],
    cwd=REPO,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

assert tracked.returncode == 0


RUNNER_SHA = sha256_file(
    THIS_SCRIPT
)

RUNNER_COMMIT = subprocess.check_output(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
    text=True,
).strip()


# No physical-noise metric/evaluation artifact may already exist.
for pattern in [
    "physical_test_metrics*.csv",
    "noise_test_metrics*.csv",
    "noise_evaluation*.json",
]:

    assert len(
        list(
            ROOT.rglob(
                pattern
            )
        )
    ) == 0


# ============================================================================
# Import frozen raw adapters/kernel
# ============================================================================

gen = load_module(
    GENERATOR,
    "raw_generator14",
)

kernel = gen.load_kernel()

kernel_hash = sha256_file(
    KERNEL_PATH
)


# ============================================================================
# Canonical union of all test rows across the 5 frozen splits
# ============================================================================

canonical_columns = [
    "dataset",
    "split_group",
    "soh_task_eligible",
    "feature_source_cell_id",
    "feature_source_member",
    "feature_raw_cycle_key",
    "feature_raw_cycle_position_zero_based",
    "feature_contract_version",
    "feature_kernel_sha256",
] + FEATURES


canonical = pd.read_csv(
    CANON,
    usecols=canonical_columns,
    low_memory=False,
)


canonical = canonical[
    as_bool(
        canonical[
            "soh_task_eligible"
        ]
    )
].copy()


assert len(
    canonical
) == 300553


assignments = pd.read_csv(
    ASSIGNMENTS,
    usecols=[
        "split_seed",
        "split_group",
        "split",
    ],
    low_memory=False,
)


test_groups_by_seed = {}

union_test_groups = set()


for seed in SPLIT_SEEDS:

    a = assignments[
        assignments[
            "split_seed"
        ] == seed
    ]

    groups = set(
        a.loc[
            a[
                "split"
            ] == "test",
            "split_group",
        ].astype(str)
    )

    test_groups_by_seed[
        seed
    ] = groups

    union_test_groups.update(
        groups
    )


canonical[
    "split_group"
] = canonical[
    "split_group"
].astype(str)


for seed in SPLIT_SEEDS:

    n = int(
        canonical[
            canonical[
                "split_group"
            ].isin(
                test_groups_by_seed[
                    seed
                ]
            )
        ].shape[
            0
        ]
    )

    assert n == EXPECTED_TEST_ROWS[
        seed
    ], (
        seed,
        n,
        EXPECTED_TEST_ROWS[
            seed
        ],
    )


target = canonical[
    canonical[
        "split_group"
    ].isin(
        union_test_groups
    )
].copy()


assert len(
    target
) > 0

assert target[
    "feature_source_cell_id"
].notna().all()

assert target[
    "feature_source_member"
].notna().all()

assert target[
    "feature_raw_cycle_position_zero_based"
].notna().all()


target[
    "feature_source_cell_id"
] = target[
    "feature_source_cell_id"
].astype(str)

target[
    "feature_source_member"
] = target[
    "feature_source_member"
].astype(str)

target[
    "feature_raw_cycle_position_zero_based"
] = pd.to_numeric(
    target[
        "feature_raw_cycle_position_zero_based"
    ],
    errors="raise",
).astype(int)


key_cols = [
    "dataset",
    "feature_source_cell_id",
    "feature_raw_cycle_position_zero_based",
]


assert not target.duplicated(
    key_cols
).any()


ROUNDTRIP.mkdir(
    parents=True,
    exist_ok=True,
)

CELL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# Compare one reconstructed raw-source cell
# ============================================================================

def compare_cell(
    dataset,
    cell_id,
    expected,
    reconstructed,
    raw_path,
):

    expected = (
        expected
        .sort_values(
            "feature_raw_cycle_position_zero_based"
        )
        .reset_index(
            drop=True
        )
    )


    reconstructed = reconstructed.copy()

    reconstructed[
        "raw_cell_id"
    ] = reconstructed[
        "raw_cell_id"
    ].astype(str)

    reconstructed[
        "raw_cycle_position_zero_based"
    ] = pd.to_numeric(
        reconstructed[
            "raw_cycle_position_zero_based"
        ],
        errors="raise",
    ).astype(int)


    assert not reconstructed.duplicated(
        [
            "raw_cell_id",
            "raw_cycle_position_zero_based",
        ]
    ).any()


    expected_digest = canonical_digest(
        expected
    )


    base = safe_name(
        dataset,
        cell_id,
    )

    shard_path = (
        CELL_DIR
        / f"{base}.csv"
    )

    meta_path = (
        CELL_DIR
        / f"{base}.json"
    )


    source_stat = raw_stat(
        raw_path
    )


    # Resume-safe reuse of a previously completed cell audit.
    if (
        shard_path.is_file()
        and meta_path.is_file()
    ):

        try:
            meta = json.loads(
                meta_path.read_text(
                    encoding="utf-8"
                )
            )

            reusable = (
                meta.get(
                    "contract"
                )
                == "zero_corruption_cell_roundtrip_v1"

                and meta.get(
                    "protocol_sha256"
                )
                == EXPECTED_PROTOCOL_SHA

                and meta.get(
                    "runner_sha256"
                )
                == RUNNER_SHA

                and meta.get(
                    "canonical_digest"
                )
                == expected_digest

                and meta.get(
                    "raw_source_stat"
                )
                == source_stat

                and meta.get(
                    "row_count"
                )
                == len(
                    expected
                )

                and meta.get(
                    "all_rows_pass"
                )
                is True

                and sha256_file(
                    shard_path
                )
                == meta.get(
                    "row_audit_sha256"
                )
            )

            if reusable:
                return (
                    pd.read_csv(
                        shard_path,
                        low_memory=False,
                    ),
                    True,
                )

        except Exception:
            pass


    rec_cols = [
        "raw_cell_id",
        "source_member",
        "raw_cycle_key",
        "raw_cycle_position_zero_based",
        "feature_contract_version",
        "kernel_sha256",
        "feature_valid",
    ] + FEATURES


    missing = [
        x
        for x in rec_cols
        if x not in reconstructed.columns
    ]

    assert not missing, (
        dataset,
        cell_id,
        missing,
    )


    rec = reconstructed[
        rec_cols
    ].copy()


    merged = expected.merge(
        rec,
        left_on=[
            "feature_source_cell_id",
            "feature_raw_cycle_position_zero_based",
        ],
        right_on=[
            "raw_cell_id",
            "raw_cycle_position_zero_based",
        ],
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=(
            "_canonical",
            "_reconstructed",
        ),
    )


    assert len(
        merged
    ) == len(
        expected
    )

    assert (
        merged[
            "_merge"
        ] == "both"
    ).all(), (
        dataset,
        cell_id,
        merged[
            "_merge"
        ].value_counts(
            dropna=False
        ).to_dict(),
    )


    feature_valid = as_bool(
        merged[
            "feature_valid"
        ]
    )


    assert feature_valid.all(), (
        dataset,
        cell_id,
        "reconstructed feature-invalid test cycle",
    )


    source_member_match = (
        merged[
            "feature_source_member"
        ].astype(str)
        ==
        merged[
            "source_member"
        ].astype(str)
    )


    key_match = np.asarray(
        [
            norm_key(a)
            == norm_key(b)
            for a, b in zip(
                merged[
                    "feature_raw_cycle_key"
                ],
                merged[
                    "raw_cycle_key"
                ],
            )
        ],
        dtype=bool,
    )


    contract_match = (
        merged[
            "feature_contract_version_canonical"
        ].astype(str)
        ==
        merged[
            "feature_contract_version_reconstructed"
        ].astype(str)
    )


    kernel_match = (
        merged[
            "feature_kernel_sha256"
        ].astype(str)
        ==
        merged[
            "kernel_sha256"
        ].astype(str)
    )


    canonical_x = merged[
        [
            f"{f}_canonical"
            for f in FEATURES
        ]
    ].to_numpy(
        dtype=np.float64
    )


    reconstructed_x = merged[
        [
            f"{f}_reconstructed"
            for f in FEATURES
        ]
    ].to_numpy(
        dtype=np.float64
    )


    assert np.isfinite(
        canonical_x
    ).all()

    assert np.isfinite(
        reconstructed_x
    ).all()


    abs_diff = np.abs(
        reconstructed_x
        - canonical_x
    )


    tolerance = (
        ATOL
        + RTOL
        * np.abs(
            canonical_x
        )
    )


    tolerance_ratio = (
        abs_diff
        / tolerance
    )


    close_matrix = (
        abs_diff
        <= tolerance
    )


    feature_close = np.all(
        close_matrix,
        axis=1,
    )


    row_max_abs = np.max(
        abs_diff,
        axis=1,
    )

    row_max_ratio = np.max(
        tolerance_ratio,
        axis=1,
    )


    metadata_match = (
        source_member_match.to_numpy()
        & key_match
        & contract_match.to_numpy()
        & kernel_match.to_numpy()
    )


    all_pass = (
        feature_close
        & metadata_match
    )


    audit = pd.DataFrame({
        "dataset":
            dataset,

        "split_group":
            merged[
                "split_group"
            ].astype(str),

        "feature_source_cell_id":
            merged[
                "feature_source_cell_id"
            ].astype(str),

        "feature_raw_cycle_position_zero_based":
            merged[
                "feature_raw_cycle_position_zero_based"
            ].astype(int),

        "source_member_match":
            source_member_match.to_numpy(),

        "raw_cycle_key_match":
            key_match,

        "feature_contract_match":
            contract_match.to_numpy(),

        "kernel_hash_match":
            kernel_match.to_numpy(),

        "all_20_features_close":
            feature_close,

        "row_max_absolute_error":
            row_max_abs,

        "row_max_tolerance_ratio":
            row_max_ratio,

        "roundtrip_pass":
            all_pass,
    })


    atomic_csv(
        audit,
        shard_path,
    )


    meta = {
        "contract":
            "zero_corruption_cell_roundtrip_v1",

        "protocol_sha256":
            EXPECTED_PROTOCOL_SHA,

        "runner_sha256":
            RUNNER_SHA,

        "runner_commit":
            RUNNER_COMMIT,

        "dataset":
            dataset,

        "cell_id":
            str(
                cell_id
            ),

        "canonical_digest":
            expected_digest,

        "raw_source_stat":
            source_stat,

        "row_count":
            int(
                len(
                    audit
                )
            ),

        "source_metadata_all_match":
            bool(
                metadata_match.all()
            ),

        "all_20_features_close":
            bool(
                feature_close.all()
            ),

        "all_rows_pass":
            bool(
                all_pass.all()
            ),

        "max_absolute_error":
            float(
                row_max_abs.max()
                if len(
                    row_max_abs
                )
                else 0.0
            ),

        "max_tolerance_ratio":
            float(
                row_max_ratio.max()
                if len(
                    row_max_ratio
                )
                else 0.0
            ),

        "row_audit_sha256":
            sha256_file(
                shard_path
            ),
    }


    atomic_json(
        meta,
        meta_path,
    )


    return audit, False


# ============================================================================
# Dataset-specific raw reconstruction
# ============================================================================

cell_groups = {
    dataset:
        list(
            target[
                target[
                    "dataset"
                ] == dataset
            ].groupby(
                "feature_source_cell_id",
                sort=True,
            )
        )
    for dataset in DATASETS
}


total_cells = sum(
    len(v)
    for v in cell_groups.values()
)


print("=" * 80)
print("ZERO-CORRUPTION RAW ROUNDTRIP")
print("=" * 80)

print(
    "protocol SHA locked:",
    True
)

print(
    "canonical eligible rows:",
    len(
        canonical
    )
)

print(
    "union test rows:",
    len(
        target
    )
)

print(
    "union raw-source cells:",
    total_cells
)

print(
    "feature tolerance atol/rtol:",
    ATOL,
    RTOL,
)

print(
    "model evaluation:",
    False
)

print(
    "physical corruption generated:",
    False
)

print()


completed = 0
generated = 0
reused = 0
all_shards = []

start_time = time.time()


def update_progress(
    phase,
    current_dataset=None,
    current_cell=None,
):

    atomic_json(
        {
            "contract":
                "zero_corruption_roundtrip_progress_v1",

            "phase":
                phase,

            "protocol_sha256":
                EXPECTED_PROTOCOL_SHA,

            "runner_sha256":
                RUNNER_SHA,

            "total_cells":
                total_cells,

            "completed_cells":
                completed,

            "generated_cells":
                generated,

            "reused_cells":
                reused,

            "current_dataset":
                current_dataset,

            "current_cell":
                current_cell,

            "model_evaluated":
                False,

            "physical_corruption_generated":
                False,

            "elapsed_seconds":
                float(
                    time.time()
                    - start_time
                ),
        },
        PROGRESS,
    )


update_progress(
    "starting"
)


# --------------------------------------------------------------------------
# HUST
# --------------------------------------------------------------------------

if cell_groups[
    "HUST"
]:

    needed_by_member = {}

    for cell_id, expected in cell_groups[
        "HUST"
    ]:

        members = expected[
            "feature_source_member"
        ].astype(str).unique()

        assert len(
            members
        ) == 1, (
            cell_id,
            members,
        )

        needed_by_member[
            members[
                0
            ]
        ] = (
            str(
                cell_id
            ),
            expected,
        )


    with zipfile.ZipFile(
        gen.HUST_ZIP,
        "r",
    ) as zf:

        names = set(
            zf.namelist()
        )


        for member in sorted(
            needed_by_member
        ):

            cell_id, expected = (
                needed_by_member[
                    member
                ]
            )


            assert member in names, (
                "missing HUST member",
                member,
            )


            update_progress(
                "HUST",
                "HUST",
                cell_id,
            )


            with zf.open(
                member
            ) as f:

                obj = pickle.load(
                    f
                )


            assert len(
                obj
            ) == 1


            raw_id = next(
                iter(
                    obj
                )
            )


            assert str(
                raw_id
            ) == str(
                cell_id
            ), (
                raw_id,
                cell_id,
            )


            reconstructed = (
                gen.process_hust_cell(
                    source_member=member,
                    raw_cell_id=raw_id,
                    cell=obj[
                        raw_id
                    ],
                    kernel=kernel,
                    kernel_hash=kernel_hash,
                )
            )


            audit, was_reused = compare_cell(
                "HUST",
                cell_id,
                expected,
                reconstructed,
                gen.HUST_ZIP,
            )


            all_shards.append(
                audit
            )

            completed += 1

            if was_reused:
                reused += 1
            else:
                generated += 1


            if (
                completed % 10 == 0
                or completed == total_cells
            ):
                print(
                    f"progress "
                    f"{completed}/"
                    f"{total_cells} "
                    f"generated={generated} "
                    f"reused={reused}",
                    flush=True,
                )


# --------------------------------------------------------------------------
# CALCE
# --------------------------------------------------------------------------

for cell_id, expected in cell_groups[
    "CALCE"
]:

    cell_id = str(
        cell_id
    )


    update_progress(
        "CALCE",
        "CALCE",
        cell_id,
    )


    zip_path = (
        gen.CALCE_RAW_DIR
        / f"{cell_id}.zip"
    )


    assert zip_path.is_file(), (
        "missing CALCE zip",
        zip_path,
    )


    reconstructed = (
        gen.process_calce_cell(
            zip_path=zip_path,
            raw_cell_id=cell_id,
            kernel=kernel,
            kernel_hash=kernel_hash,
            raw_zip_hash=sha256_file(
                zip_path
            ),
        )
    )


    audit, was_reused = compare_cell(
        "CALCE",
        cell_id,
        expected,
        reconstructed,
        zip_path,
    )


    all_shards.append(
        audit
    )

    completed += 1

    if was_reused:
        reused += 1
    else:
        generated += 1


    if (
        completed % 10 == 0
        or completed == total_cells
    ):
        print(
            f"progress "
            f"{completed}/"
            f"{total_cells} "
            f"generated={generated} "
            f"reused={reused}",
            flush=True,
        )


# --------------------------------------------------------------------------
# MATR
# --------------------------------------------------------------------------

matr_groups_by_source = {}

for cell_id, expected in cell_groups[
    "MATR"
]:

    sources = expected[
        "feature_source_member"
    ].astype(str).unique()

    assert len(
        sources
    ) == 1, (
        cell_id,
        sources,
    )

    matr_groups_by_source.setdefault(
        sources[
            0
        ],
        [],
    ).append(
        (
            str(
                cell_id
            ),
            expected,
        )
    )


def matr_batch_group(f):

    if (
        "batch"
        in f
        and isinstance(
            f[
                "batch"
            ],
            h5py.Group,
        )
        and "cycles"
        in f[
            "batch"
        ]
    ):
        return f[
            "batch"
        ]


    found = []

    for name in f.keys():

        obj = f[
            name
        ]

        if (
            isinstance(
                obj,
                h5py.Group,
            )
            and "cycles"
            in obj
        ):
            found.append(
                obj
            )


    assert len(
        found
    ) == 1, (
        "could not uniquely find MATR batch group",
        list(
            f.keys()
        ),
    )

    return found[
        0
    ]


for source_member in sorted(
    matr_groups_by_source
):

    path = (
        BASE
        / source_member
    )


    assert path.is_file(), (
        "missing MATR source",
        path,
    )


    with h5py.File(
        path,
        "r",
    ) as f:

        batch = matr_batch_group(
            f
        )


        n_cells = int(
            batch[
                "cycles"
            ].shape[
                0
            ]
        )


        for cell_id, expected in sorted(
            matr_groups_by_source[
                source_member
            ],
            key=lambda x: x[
                0
            ],
        ):

            m = re.fullmatch(
                r"b\d+c(\d+)",
                cell_id,
            )

            assert m is not None, (
                "unexpected MATR raw_cell_id",
                cell_id,
            )


            cell_index = int(
                m.group(
                    1
                )
            )


            assert (
                0
                <= cell_index
                < n_cells
            ), (
                cell_id,
                cell_index,
                n_cells,
            )


            update_progress(
                "MATR",
                "MATR",
                cell_id,
            )


            cycles = f[
                batch[
                    "cycles"
                ][
                    cell_index,
                    0,
                ]
            ]


            reconstructed = (
                gen.process_matr_cell(
                    f=f,
                    cycles=cycles,
                    source_member=source_member,
                    raw_cell_id=cell_id,
                    kernel=kernel,
                    kernel_hash=kernel_hash,
                )
            )


            audit, was_reused = compare_cell(
                "MATR",
                cell_id,
                expected,
                reconstructed,
                path,
            )


            all_shards.append(
                audit
            )

            completed += 1

            if was_reused:
                reused += 1
            else:
                generated += 1


            if (
                completed % 10 == 0
                or completed == total_cells
            ):
                print(
                    f"progress "
                    f"{completed}/"
                    f"{total_cells} "
                    f"generated={generated} "
                    f"reused={reused}",
                    flush=True,
                )


assert completed == total_cells


# ============================================================================
# Aggregate barrier artifact
# ============================================================================

combined = pd.concat(
    all_shards,
    ignore_index=True,
)


combined = combined.sort_values(
    [
        "dataset",
        "feature_source_cell_id",
        "feature_raw_cycle_position_zero_based",
    ],
    kind="mergesort",
).reset_index(
    drop=True
)


assert len(
    combined
) == len(
    target
)


assert not combined.duplicated(
    [
        "dataset",
        "feature_source_cell_id",
        "feature_raw_cycle_position_zero_based",
    ]
).any()


atomic_csv(
    combined,
    ROW_AUDIT,
)


source_match = as_bool(
    combined[
        "source_member_match"
    ]
)

key_match = as_bool(
    combined[
        "raw_cycle_key_match"
    ]
)

contract_match = as_bool(
    combined[
        "feature_contract_match"
    ]
)

kernel_match = as_bool(
    combined[
        "kernel_hash_match"
    ]
)

feature_match = as_bool(
    combined[
        "all_20_features_close"
    ]
)

row_pass = as_bool(
    combined[
        "roundtrip_pass"
    ]
)


dataset_counts = {
    dataset:
        int(
            (
                combined[
                    "dataset"
                ] == dataset
            ).sum()
        )
    for dataset in DATASETS
}


summary = {
    "contract":
        "zero_corruption_raw_roundtrip_v1",

    "status":
        (
            "PASS"
            if bool(
                row_pass.all()
            )
            else "FAIL"
        ),

    "protocol_sha256":
        EXPECTED_PROTOCOL_SHA,

    "runner_sha256":
        RUNNER_SHA,

    "runner_commit":
        RUNNER_COMMIT,

    "generator_sha256":
        sha256_file(
            GENERATOR
        ),

    "kernel_sha256":
        kernel_hash,

    "canonical_sha256":
        EXPECTED_CANON_SHA,

    "assignments_sha256":
        EXPECTED_ASSIGNMENTS_SHA,

    "absolute_tolerance":
        ATOL,

    "relative_tolerance":
        RTOL,

    "canonical_soh_eligible_rows":
        int(
            len(
                canonical
            )
        ),

    "union_test_rows_expected":
        int(
            len(
                target
            )
        ),

    "union_test_rows_audited":
        int(
            len(
                combined
            )
        ),

    "union_test_cells":
        int(
            total_cells
        ),

    "dataset_row_counts":
        dataset_counts,

    "per_split_test_row_counts":
        {
            str(
                seed
            ):
                EXPECTED_TEST_ROWS[
                    seed
                ]
            for seed in SPLIT_SEEDS
        },

    "source_member_all_match":
        bool(
            source_match.all()
        ),

    "raw_cycle_key_all_match":
        bool(
            key_match.all()
        ),

    "feature_contract_all_match":
        bool(
            contract_match.all()
        ),

    "kernel_hash_all_match":
        bool(
            kernel_match.all()
        ),

    "all_20_features_close":
        bool(
            feature_match.all()
        ),

    "all_rows_pass":
        bool(
            row_pass.all()
        ),

    "failed_rows":
        int(
            (
                ~row_pass
            ).sum()
        ),

    "max_absolute_error":
        float(
            pd.to_numeric(
                combined[
                    "row_max_absolute_error"
                ],
                errors="raise",
            ).max()
        ),

    "max_tolerance_ratio":
        float(
            pd.to_numeric(
                combined[
                    "row_max_tolerance_ratio"
                ],
                errors="raise",
            ).max()
        ),

    "row_audit_sha256":
        sha256_file(
            ROW_AUDIT
        ),

    "model_evaluated":
        False,

    "physical_noise_generated":
        False,

    "physical_noise_test_metrics_evaluated":
        False,

    "generated_cell_audits_this_run":
        int(
            generated
        ),

    "reused_cell_audits_this_run":
        int(
            reused
        ),

    "elapsed_seconds":
        float(
            time.time()
            - start_time
        ),
}


atomic_json(
    summary,
    SUMMARY,
)


update_progress(
    "complete"
)


print()
print("=" * 80)
print("ZERO-CORRUPTION ROUNDTRIP SUMMARY")
print("=" * 80)

print(
    "union test rows:",
    summary[
        "union_test_rows_audited"
    ]
)

print(
    "union test cells:",
    summary[
        "union_test_cells"
    ]
)

print(
    "dataset rows:",
    summary[
        "dataset_row_counts"
    ]
)

print(
    "source members exact:",
    summary[
        "source_member_all_match"
    ]
)

print(
    "raw cycle keys exact:",
    summary[
        "raw_cycle_key_all_match"
    ]
)

print(
    "feature contracts exact:",
    summary[
        "feature_contract_all_match"
    ]
)

print(
    "kernel hashes exact:",
    summary[
        "kernel_hash_all_match"
    ]
)

print(
    "20/20 features within frozen tolerance:",
    summary[
        "all_20_features_close"
    ]
)

print(
    "failed rows:",
    summary[
        "failed_rows"
    ]
)

print(
    "max tolerance ratio:",
    f"{summary['max_tolerance_ratio']:.6g}"
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
    "row audit SHA256:",
    summary[
        "row_audit_sha256"
    ]
)

print(
    "summary SHA256:",
    sha256_file(
        SUMMARY
    )
)

print()


if not summary[
    "all_rows_pass"
]:
    raise RuntimeError(
        "ZERO_CORRUPTION_ROUNDTRIP_FAILED"
    )


print(
    "ZERO_CORRUPTION_RAW_ROUNDTRIP_PASS=True"
)
