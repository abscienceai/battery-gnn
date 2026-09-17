#!/usr/bin/env python3

"""
Target-independent raw-cycle feature generation for revision_2026.

Current adapter:
    HUST

Critical architectural rule:
    This script reads raw sensor data only.

It MUST NOT read:
    unified_label_backbone.csv
    capacity/SOH/RUL/EOL labels
    train/validation/test splits
    fitted scalers

Model features are produced exclusively through
cycle_feature_kernel.py.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import io
import re
import hashlib
import importlib.util
import json
import os
import pickle
import tempfile
import zipfile

import h5py
import numpy as np
import pandas as pd


BASE = Path(
    os.environ["BASE"]
)

REPO = (
    BASE
    / "repo/battery-gnn"
)

KERNEL_PATH = (
    REPO
    / "scripts/revision_2026/"
      "cycle_feature_kernel.py"
)

HUST_ZIP = (
    BASE
    / "data/raw/HUST/"
      "hust_data.zip"
)

OUT_ROOT = (
    BASE
    / "data/processed/revision_2026/"
      "raw_cycle_features"
)

HUST_SHARDS = (
    OUT_ROOT
    / "HUST_cells"
)

HUST_OUT = (
    OUT_ROOT
    / "HUST_raw_cycle_features.csv"
)

HUST_MANIFEST = (
    OUT_ROOT
    / "HUST_raw_cycle_features_manifest.json"
)

HUST_SUMMARY = (
    OUT_ROOT
    / "HUST_raw_cycle_features_summary.txt"
)



# ---------------------------------------------------------------------------
# CALCE source-native raw-cycle feature generation
# ---------------------------------------------------------------------------

CALCE_RAW_DIR = (
    HUST_ZIP
    .parent
    .parent
    / "CALCE"
)

CALCE_SHARDS = (
    HUST_SHARDS
    .with_name(
        "CALCE_shards"
    )
)

CALCE_OUT = (
    HUST_OUT
    .with_name(
        "CALCE_raw_cycle_features.csv"
    )
)

CALCE_MANIFEST = (
    HUST_MANIFEST
    .with_name(
        "CALCE_raw_cycle_features_manifest.json"
    )
)

CALCE_SUMMARY = (
    HUST_SUMMARY
    .with_name(
        "CALCE_raw_cycle_features_summary.txt"
    )
)

CALCE_RESUME_CONTRACT = (
    CALCE_OUT
    .with_name(
        "CALCE_raw_cycle_features_resume_contract.json"
    )
)

CALCE_EXPECTED_CELLS = (
    "CS2_33",
    "CS2_34",
    "CS2_35",
    "CS2_36",
    "CS2_37",
    "CS2_38",
    "CX2_16",
    "CX2_33",
    "CX2_34",
    "CX2_35",
    "CX2_36",
    "CX2_37",
    "CX2_38",
)

CALCE_EXPECTED_RAW_CELLS = 13
CALCE_EXPECTED_GROUPED_CYCLES = 18820

MATR_FILES = {
    "b1": (
        BASE
        / "data/raw/MATR/"
          "MATR_batch_20170512.mat"
    ),
    "b2": (
        BASE
        / "data/raw/MATR/"
          "MATR_batch_20170630.mat"
    ),
    "b3": (
        BASE
        / "data/raw/MATR/"
          "MATR_batch_20180412.mat"
    ),
    "b4": (
        BASE
        / "data/raw/MATR/"
          "MATR_batch_20190124.mat"
    ),
}

MATR_SHARDS = (
    OUT_ROOT
    / "MATR_cells"
)

MATR_OUT = (
    OUT_ROOT
    / "MATR_raw_cycle_features.csv"
)

MATR_MANIFEST = (
    OUT_ROOT
    / "MATR_raw_cycle_features_manifest.json"
)

MATR_SUMMARY = (
    OUT_ROOT
    / "MATR_raw_cycle_features_summary.txt"
)

MATR_RESUME_CONTRACT = (
    MATR_SHARDS
    / "_resume_contract.json"
)

EXPECTED_MATR_RAW_CELLS = 185
EXPECTED_MATR_B1_CELLS = 46


FEATURE_CONTRACT_VERSION = (
    "primary_vit20_v1"
)

EXPECTED_HUST_CELLS = 77

# Independently established by the prior official raw HUST audit.
# This is a raw-source structural guard, not a target-table lookup.
EXPECTED_HUST_RAW_CYCLES = 146122


def sha256_file(
    path,
    chunk=1024 * 1024,
):
    h = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as f:
        while True:
            b = f.read(chunk)

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


def safe_cycle_sort_key(x):
    """
    HUST cycle dictionary keys are expected to be numeric.
    Preserve the native key while sorting numerically.
    """
    try:
        return (
            0,
            float(x),
        )
    except Exception:
        return (
            1,
            str(x),
        )


def atomic_csv_write(
    df,
    path,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, tmp_name = (
        tempfile.mkstemp(
            prefix=(
                path.name
                + ".tmp."
            ),
            dir=str(
                path.parent
            ),
        )
    )

    os.close(fd)

    tmp = Path(tmp_name)

    try:
        df.to_csv(
            tmp,
            index=False,
        )

        os.replace(
            tmp,
            path,
        )

    finally:
        if tmp.exists():
            tmp.unlink()


def inspect_raw_counts():
    """
    Count HUST cells/cycles from raw PKLs only.

    No labels are read.
    """
    members = []

    cycle_count = 0

    with zipfile.ZipFile(
        HUST_ZIP,
        "r",
    ) as zf:

        members = sorted(
            m
            for m in zf.namelist()
            if (
                m.startswith(
                    "our_data/"
                )
                and m.endswith(
                    ".pkl"
                )
            )
        )

        for member in members:

            with zf.open(
                member
            ) as f:
                obj = pickle.load(f)

            if not isinstance(
                obj,
                dict,
            ):
                raise TypeError(
                    f"{member}: "
                    "top-level object is not dict"
                )

            if len(obj) != 1:
                raise ValueError(
                    f"{member}: "
                    f"expected one cell key, got {len(obj)}"
                )

            raw_cell_id = next(
                iter(obj)
            )

            cell = obj[
                raw_cell_id
            ]

            data = cell.get(
                "data"
            )

            if not isinstance(
                data,
                dict,
            ):
                raise TypeError(
                    f"{member}: "
                    "cell['data'] is not dict"
                )

            cycle_count += len(
                data
            )

    return members, cycle_count


def expected_shard_columns(
    kernel,
):
    metadata = [
        "dataset",
        "raw_cell_id",
        "source_member",
        "raw_cycle_key",
        "raw_cycle_position_zero_based",
        "feature_contract_version",
        "kernel_sha256",
        "feature_valid",
        "failure_reason",
        "raw_sample_count_qc",
        "valid_sample_count_qc",
        "dropped_nonfinite_samples_qc",
        "duration_s_qc_only",
        "duplicate_or_zero_dt_count_qc",
        "backward_time_step_count_qc",
    ]

    return (
        metadata
        + list(
            kernel
            .PRIMARY_FEATURE_NAMES
        )
    )


def shard_is_reusable(
    path,
    *,
    raw_cell_id,
    source_member,
    kernel_hash,
    contract,
    expected_rows,
    columns,
):
    path = Path(path)

    if not path.is_file():
        return False

    try:
        df = pd.read_csv(
            path
        )
    except Exception:
        return False

    if list(
        df.columns
    ) != columns:
        return False

    if len(df) != expected_rows:
        return False

    if len(df) == 0:
        return False

    checks = [
        (
            df["dataset"]
            == "HUST"
        ).all(),

        (
            df["raw_cell_id"]
            .astype(str)
            == str(raw_cell_id)
        ).all(),

        (
            df["source_member"]
            == source_member
        ).all(),

        (
            df[
                "feature_contract_version"
            ]
            == contract
        ).all(),

        (
            df["kernel_sha256"]
            == kernel_hash
        ).all(),
    ]

    return bool(
        all(checks)
    )


def process_hust_cell(
    *,
    source_member,
    raw_cell_id,
    cell,
    kernel,
    kernel_hash,
):
    """
    Generate exactly one structural row per raw HUST cycle.

    cell['dq'] and cell['rul'] are deliberately never accessed.
    """

    data = cell[
        "data"
    ]

    keys = sorted(
        data.keys(),
        key=safe_cycle_sort_key,
    )

    rows = []

    for position, native_key in enumerate(
        keys
    ):

        cycle = data[
            native_key
        ]

        row = {
            "dataset":
                "HUST",

            "raw_cell_id":
                str(raw_cell_id),

            "source_member":
                source_member,

            "raw_cycle_key":
                str(native_key),

            "raw_cycle_position_zero_based":
                int(position),

            "feature_contract_version":
                FEATURE_CONTRACT_VERSION,

            "kernel_sha256":
                kernel_hash,

            "feature_valid":
                False,

            "failure_reason":
                "",

            "raw_sample_count_qc":
                np.nan,

            "valid_sample_count_qc":
                np.nan,

            "dropped_nonfinite_samples_qc":
                np.nan,

            "duration_s_qc_only":
                np.nan,

            "duplicate_or_zero_dt_count_qc":
                np.nan,

            "backward_time_step_count_qc":
                np.nan,
        }

        for feature_name in (
            kernel
            .PRIMARY_FEATURE_NAMES
        ):
            row[
                feature_name
            ] = np.nan

        try:
            if not isinstance(
                cycle,
                pd.DataFrame,
            ):
                raise TypeError(
                    "cycle object is not DataFrame"
                )

            required = [
                "Current (mA)",
                "Voltage (V)",
                "Time (s)",
            ]

            missing = [
                c
                for c in required
                if c not in cycle.columns
            ]

            if missing:
                raise KeyError(
                    "missing columns: "
                    + ",".join(
                        missing
                    )
                )

            V = (
                pd.to_numeric(
                    cycle[
                        "Voltage (V)"
                    ],
                    errors="coerce",
                )
                .to_numpy(
                    dtype=float
                )
            )

            # Official HUST raw current:
            # mA -> A.
            I = (
                pd.to_numeric(
                    cycle[
                        "Current (mA)"
                    ],
                    errors="coerce",
                )
                .to_numpy(
                    dtype=float
                )
                / 1000.0
            )

            t = (
                pd.to_numeric(
                    cycle[
                        "Time (s)"
                    ],
                    errors="coerce",
                )
                .to_numpy(
                    dtype=float
                )
            )

            # Raw count is QC only and never enters model X.
            row[
                "raw_sample_count_qc"
            ] = int(
                len(cycle)
            )

            features, qc = (
                kernel
                .extract_primary_cycle_features(
                    V,
                    I,
                    t,
                )
            )

            row.update(
                features
            )

            row[
                "feature_valid"
            ] = True

            row[
                "valid_sample_count_qc"
            ] = qc[
                "valid_sample_count"
            ]

            row[
                "dropped_nonfinite_samples_qc"
            ] = qc[
                "dropped_nonfinite_samples"
            ]

            row[
                "duration_s_qc_only"
            ] = qc[
                "duration_s_qc_only"
            ]

            row[
                "duplicate_or_zero_dt_count_qc"
            ] = qc[
                "duplicate_or_zero_dt_count"
            ]

            row[
                "backward_time_step_count_qc"
            ] = qc[
                "backward_time_step_count"
            ]

        except Exception as exc:
            row[
                "failure_reason"
            ] = (
                type(exc).__name__
                + ": "
                + str(exc)
            )[:500]

        rows.append(
            row
        )

    return pd.DataFrame(
        rows,
        columns=(
            expected_shard_columns(
                kernel
            )
        ),
    )


def matr_deref(
    f,
    ds,
    idx,
):
    """
    Dereference one MATLAB-v7.3 cycle signal as a 1-D float array.
    """
    return (
        np.asarray(
            f[
                ds[idx, 0]
            ][:]
        )
        .squeeze()
        .astype(float)
        .reshape(-1)
    )



# ===========================================================================
# CALCE adapter
# ===========================================================================


# ===========================================================================
# RWTH source-native adapter
# ===========================================================================

RWTH_RAW_DIR = (
    HUST_ZIP
    .parent
    .parent
    / "RWTH"
)

RWTH_SHARDS = (
    HUST_SHARDS
    .with_name(
        "RWTH_shards"
    )
)

RWTH_OUT = (
    HUST_OUT
    .with_name(
        "RWTH_raw_cycle_features.csv"
    )
)

RWTH_MANIFEST = (
    HUST_MANIFEST
    .with_name(
        "RWTH_raw_cycle_features_manifest.json"
    )
)

RWTH_SUMMARY = (
    HUST_SUMMARY
    .with_name(
        "RWTH_raw_cycle_features_summary.txt"
    )
)

RWTH_RESUME_CONTRACT = (
    RWTH_OUT
    .with_name(
        "RWTH_raw_cycle_features_resume_contract.json"
    )
)

RWTH_ADAPTER_CONTRACT_VERSION = (
    "rwth_zyk_sensor_only_prefilter_v2"
)

RWTH_EXPECTED_CELLS = tuple(
    f"RWTH_{i:03d}"
    for i in range(
        2,
        50,
    )
)

RWTH_EXPECTED_RAW_CELLS = 48
RWTH_EXPECTED_TBA_ZYK_ARCHIVES = 738
RWTH_EXPECTED_PREFILTER_CYCLES = 105663

RWTH_EXPECTED_OUTER_ZIP_SHA256 = (
    "5990780c150c44c1749154f189d28b21"
    "7df4c429b98b8fd5c4145d5af6dc9a0a"
)

RWTH_REQUIRED_SENSOR_COLUMNS = (
    "Zeit",
    "Programmdauer",
    "Strom",
    "Spannung",
)


def rwth_extract_ts_id(
    value,
):
    matches = re.findall(
        r"TS0*(\d+)",
        str(value),
        flags=re.IGNORECASE,
    )

    if not matches:
        raise ValueError(
            "Cannot extract RWTH TS "
            f"identifier: {value}"
        )

    return int(
        matches[0]
    )


def rwth_extract_cell_id(
    member,
):
    """
    TBA_Zyk archive names contain the canonical epSanyoNNN
    raw-cell identifier.
    """
    matches = re.findall(
        r"ep\s+sanyo\s+0*(\d+)\s*=ZYK=",
        str(member),
        flags=re.IGNORECASE,
    )

    if len(matches) != 1:
        raise ValueError(
            "Cannot uniquely extract RWTH "
            f"cell from: {member}"
        )

    number = int(
        matches[0]
    )

    if not (
        2 <= number <= 49
    ):
        raise ValueError(
            "Unexpected RWTH cell number: "
            f"{number}"
        )

    return (
        f"RWTH_{number:03d}"
    )


def rwth_discover_outer_zip():
    """
    Identify the official raw RWTH outer archive by its
    previously audited SHA-256. No processed/audit artifact
    is used for discovery.
    """
    candidates = sorted(
        RWTH_RAW_DIR.rglob(
            "*.zip"
        )
    )

    hits = []

    for path in candidates:
        try:
            if (
                sha256_file(path)
                == RWTH_EXPECTED_OUTER_ZIP_SHA256
            ):
                hits.append(
                    path
                )
        except Exception:
            continue

    if len(hits) != 1:
        raise RuntimeError(
            "Expected exactly one official "
            "RWTH outer ZIP by SHA256; "
            f"found {hits}"
        )

    return hits[0]


def rwth_member_inventory(
    outer_path,
):
    """
    Discover and chronologically order all source-native
    TBA_Zyk nested archives.
    """
    by_cell = {
        cell: []
        for cell in RWTH_EXPECTED_CELLS
    }

    with zipfile.ZipFile(
        outer_path,
        "r",
    ) as outer:

        members = [
            name
            for name in outer.namelist()
            if (
                "=TBA_Zyk=" in name
                and name.lower().endswith(
                    ".zip"
                )
            )
        ]

    if len(
        members
    ) != RWTH_EXPECTED_TBA_ZYK_ARCHIVES:
        raise AssertionError(
            "Unexpected RWTH TBA_Zyk archive "
            f"count: {len(members)}"
        )

    for member in members:
        cell = rwth_extract_cell_id(
            member
        )

        by_cell[
            cell
        ].append(
            member
        )

    if any(
        len(by_cell[cell]) == 0
        for cell in RWTH_EXPECTED_CELLS
    ):
        raise AssertionError(
            "RWTH cell without TBA_Zyk archive"
        )

    # Chronology is carried by the TS identifier.
    for cell in RWTH_EXPECTED_CELLS:
        by_cell[cell] = sorted(
            by_cell[cell],
            key=lambda x: (
                rwth_extract_ts_id(x),
                str(x),
            ),
        )

    return by_cell


def rwth_read_nested_sensor_csv(
    outer,
    member,
    source_index,
):
    """
    Strict sensor/structural-only reader.

    Only Zeit, Programmdauer, Strom and Spannung enter this
    raw feature pipeline.
    """
    nested_bytes = outer.read(
        member
    )

    with zipfile.ZipFile(
        io.BytesIO(
            nested_bytes
        ),
        "r",
    ) as nested:

        csv_members = [
            x
            for x in nested.namelist()
            if x.lower().endswith(
                ".csv"
            )
        ]

        if len(
            csv_members
        ) != 1:
            raise RuntimeError(
                f"{member}: expected one CSV, "
                f"found {len(csv_members)}"
            )

        raw = nested.read(
            csv_members[0]
        )

    df = pd.read_csv(
        io.BytesIO(raw),
        skiprows=[1],
        encoding="utf-8-sig",
        low_memory=False,
        usecols=lambda c:
            c in RWTH_REQUIRED_SENSOR_COLUMNS,
    )

    missing = [
        c
        for c in RWTH_REQUIRED_SENSOR_COLUMNS
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            f"{member}: missing required "
            f"sensor columns {missing}"
        )

    df = df[
        list(
            RWTH_REQUIRED_SENSOR_COLUMNS
        )
    ].copy()

    df[
        "_source_index"
    ] = int(
        source_index
    )

    df[
        "_source_zip"
    ] = (
        Path(member).name
    )

    return df


def rwth_find_time_anomalies(
    time,
    tolerance=1e5,
):
    """
    Exact previously audited BatteryML-compatible RWTH
    Programmdauer filter. Raw Programmdauer unit is ms.
    """
    time = np.asarray(
        time,
        dtype=float,
    )

    if len(time) == 0:
        return np.zeros(
            0,
            dtype=bool,
        )

    prev = time[0]

    result = np.ones(
        len(time),
        dtype=bool,
    )

    for i in range(
        1,
        len(time),
    ):
        if (
            time[i] - prev
            > tolerance
        ):
            result[i] = False
        else:
            prev = time[i]

    return result


def rwth_find_cycle_starts_safe(
    current,
    lag=10,
    tolerance=0.1,
):
    """
    Boundary-safe reproduction of BatteryML's RWTH
    discharge-start detector.
    """
    current = np.asarray(
        current,
        dtype=float,
    )

    is_cycle_start = np.zeros(
        len(current),
        dtype=bool,
    )

    enter_discharge_steps = 0
    nms_size = 500

    for i in range(
        len(current)
    ):
        I = current[i]

        if (
            0 < i
            < len(current) - 1
        ):
            if (
                abs(
                    current[i]
                    - current[i - 1]
                )
                > tolerance
                and
                abs(
                    current[i]
                    - current[i + 1]
                )
                > tolerance
            ):
                I = current[
                    i + 1
                ]

        if I < 0:
            enter_discharge_steps += 1
        else:
            enter_discharge_steps = 0

        if (
            enter_discharge_steps
            == lag
        ):
            t = (
                i - lag + 1
            )

            if (
                t > nms_size
                and np.any(
                    is_cycle_start[
                        t - nms_size:t
                    ]
                )
            ):
                continue

            is_cycle_start[
                t
            ] = True

    return is_cycle_start


def rwth_shard_columns(
    kernel,
):
    base = expected_shard_columns(
        kernel
    )

    feature_names = list(
        kernel.PRIMARY_FEATURE_NAMES
    )

    metadata_count = (
        len(base)
        - len(feature_names)
    )

    return (
        base[:metadata_count]
        + [
            "rwth_adapter_contract_version",
            "raw_outer_zip_sha256",
            "source_index_first",
            "source_index_last",
            "source_zip_first",
            "source_zip_last",
            "source_file_count_qc",
            "crosses_source_file_boundary_qc",
            "start_row_qc",
            "end_row_exclusive_qc",
            "time_unit_source_qc",
        ]
        + base[metadata_count:]
    )


def rwth_atomic_csv_write(
    df,
    path,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
    )

    tmp.replace(
        path
    )


def rwth_atomic_json_write(
    obj,
    path,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
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

    tmp.replace(
        path
    )


def rwth_qc_value(
    qc,
    names,
    default,
):
    if hasattr(
        qc,
        "get",
    ):
        for name in names:
            if name in qc:
                return qc[
                    name
                ]

    return default


def rwth_shard_is_reusable(
    path,
    *,
    raw_cell_id,
    raw_outer_hash,
    kernel_hash,
    expected_rows,
    columns,
):
    path = Path(path)

    if not path.is_file():
        return False

    try:
        df = pd.read_csv(
            path,
            low_memory=False,
        )
    except Exception:
        return False

    if list(
        df.columns
    ) != columns:
        return False

    if len(df) != int(
        expected_rows
    ):
        return False

    if len(df) == 0:
        return False

    checks = [
        (
            df[
                "dataset"
            ] == "RWTH"
        ).all(),

        (
            df[
                "raw_cell_id"
            ].astype(str)
            == str(
                raw_cell_id
            )
        ).all(),

        (
            df[
                "feature_contract_version"
            ]
            == FEATURE_CONTRACT_VERSION
        ).all(),

        (
            df[
                "kernel_sha256"
            ]
            == kernel_hash
        ).all(),

        (
            df[
                "rwth_adapter_contract_version"
            ]
            == RWTH_ADAPTER_CONTRACT_VERSION
        ).all(),

        (
            df[
                "raw_outer_zip_sha256"
            ]
            == raw_outer_hash
        ).all(),
    ]

    return bool(
        all(checks)
    )


def process_rwth_cell(
    *,
    outer_path,
    raw_cell_id,
    members,
    kernel,
    kernel_hash,
    raw_outer_hash,
):
    """
    Produce one feature-table row per PRE-abnormal-filter
    structural RWTH ZYK cycle.

    This function deliberately does not read AhEla, AhLad,
    CapDCH1, Qd, SOH, RUL, EOL, TDS, split or scaler data.
    """
    frames = []

    with zipfile.ZipFile(
        outer_path,
        "r",
    ) as outer:

        for source_index, member in enumerate(
            members,
            start=1,
        ):
            frames.append(
                rwth_read_nested_sensor_csv(
                    outer,
                    member,
                    source_index,
                )
            )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    # BatteryML-compatible chronology contract.
    df = (
        df
        .drop_duplicates(
            "Zeit"
        )
        .sort_values(
            "Zeit"
        )
        .reset_index(
            drop=True
        )
    )

    for column in (
        "Programmdauer",
        "Strom",
        "Spannung",
    ):
        df[
            column
        ] = pd.to_numeric(
            df[
                column
            ],
            errors="coerce",
        )

    bad_time = int(
        (
            ~np.isfinite(
                df[
                    "Programmdauer"
                ].to_numpy(
                    dtype=float
                )
            )
        ).sum()
    )

    bad_current = int(
        (
            ~np.isfinite(
                df[
                    "Strom"
                ].to_numpy(
                    dtype=float
                )
            )
        ).sum()
    )

    if (
        bad_time
        or bad_current
    ):
        raise RuntimeError(
            f"{raw_cell_id}: "
            f"non-finite Programmdauer="
            f"{bad_time}, Strom={bad_current}"
        )

    time_ok = (
        rwth_find_time_anomalies(
            df[
                "Programmdauer"
            ].to_numpy(
                dtype=float
            )
        )
    )

    df = (
        df.loc[
            time_ok
        ]
        .reset_index(
            drop=True
        )
    )

    starts_mask = (
        rwth_find_cycle_starts_safe(
            df[
                "Strom"
            ].to_numpy(
                dtype=float
            )
        )
    )

    detected_starts_all = (
        np.flatnonzero(
            starts_mask
        )
    )

    if len(
        detected_starts_all
    ) < 3:
        raise RuntimeError(
            f"{raw_cell_id}: only "
            f"{len(detected_starts_all)} "
            "discharge starts detected"
        )

    # Exact previously audited BatteryML behavior.
    starts = (
        detected_starts_all[
            1:
        ]
    )

    columns = rwth_shard_columns(
        kernel
    )

    rows = []

    for i in range(
        1,
        len(starts),
    ):
        start = int(
            starts[
                i - 1
            ]
        )

        end = int(
            starts[
                i
            ]
        )

        cycle = df.iloc[
            start:end
        ]

        position = len(
            rows
        )

        source_indices = (
            cycle[
                "_source_index"
            ]
            .astype(int)
            .to_numpy()
        )

        source_zips = (
            cycle[
                "_source_zip"
            ]
            .astype(str)
            .to_numpy()
        )

        unique_sources = list(
            dict.fromkeys(
                source_zips.tolist()
            )
        )

        source_count = len(
            unique_sources
        )

        if source_count not in (
            1,
            2,
        ):
            raise AssertionError(
                f"{raw_cell_id} "
                f"cycle {position+1}: "
                f"source_count={source_count}"
            )

        row = {
            "dataset":
                "RWTH",

            "raw_cell_id":
                str(
                    raw_cell_id
                ),

            "source_member":
                ";".join(
                    unique_sources
                ),

            "raw_cycle_key":
                str(
                    position + 1
                ),

            "raw_cycle_position_zero_based":
                int(
                    position
                ),

            "feature_contract_version":
                FEATURE_CONTRACT_VERSION,

            "kernel_sha256":
                kernel_hash,

            "feature_valid":
                False,

            "failure_reason":
                "",

            "raw_sample_count_qc":
                np.nan,

            "valid_sample_count_qc":
                np.nan,

            "dropped_nonfinite_samples_qc":
                np.nan,

            "duration_s_qc_only":
                np.nan,

            "duplicate_or_zero_dt_count_qc":
                np.nan,

            "backward_time_step_count_qc":
                np.nan,

            "rwth_adapter_contract_version":
                RWTH_ADAPTER_CONTRACT_VERSION,

            "raw_outer_zip_sha256":
                raw_outer_hash,

            "source_index_first":
                int(
                    source_indices[0]
                ),

            "source_index_last":
                int(
                    source_indices[-1]
                ),

            "source_zip_first":
                str(
                    source_zips[0]
                ),

            "source_zip_last":
                str(
                    source_zips[-1]
                ),

            "source_file_count_qc":
                int(
                    source_count
                ),

            "crosses_source_file_boundary_qc":
                bool(
                    source_count > 1
                ),

            "start_row_qc":
                start,

            "end_row_exclusive_qc":
                end,

            "time_unit_source_qc":
                "ms",
        }

        for feature_name in (
            kernel
            .PRIMARY_FEATURE_NAMES
        ):
            row[
                feature_name
            ] = np.nan

        V = (
            cycle[
                "Spannung"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        I = (
            cycle[
                "Strom"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        # Previously audited physical unit conversion:
        # Programmdauer ms -> seconds.
        t = (
            cycle[
                "Programmdauer"
            ]
            .to_numpy(
                dtype=np.float64
            )
            / 1000.0
        )

        raw_n = int(
            len(t)
        )

        finite = (
            np.isfinite(V)
            & np.isfinite(I)
            & np.isfinite(t)
        )

        valid_n = int(
            finite.sum()
        )

        dropped_n = (
            raw_n
            - valid_n
        )

        tf = t[
            finite
        ]

        if len(tf) >= 2:
            dt = np.diff(
                tf
            )

            duration_fallback = float(
                tf[-1]
                - tf[0]
            )

            zero_dt_fallback = int(
                np.sum(
                    np.isclose(
                        dt,
                        0.0,
                        atol=1e-12,
                        rtol=0.0,
                    )
                )
            )

            backward_fallback = int(
                np.sum(
                    dt < -1e-9
                )
            )

        else:
            duration_fallback = (
                np.nan
            )

            zero_dt_fallback = 0
            backward_fallback = 0

        row[
            "raw_sample_count_qc"
        ] = raw_n

        row[
            "valid_sample_count_qc"
        ] = valid_n

        row[
            "dropped_nonfinite_samples_qc"
        ] = dropped_n

        row[
            "duration_s_qc_only"
        ] = duration_fallback

        row[
            "duplicate_or_zero_dt_count_qc"
        ] = zero_dt_fallback

        row[
            "backward_time_step_count_qc"
        ] = backward_fallback

        # Cross-source cycles span distinct TBA_Zyk acquisition
        # sessions. Programmdauer restarts in the new archive and
        # the independently audited archive-session separation is
        # measured in days. We therefore do not invent a continuous
        # within-cycle time axis by stitching these sessions.
        if source_count > 1:
            row[
                "duration_s_qc_only"
            ] = np.nan

            row[
                "failure_reason"
            ] = (
                "StructuralDiscontinuity: "
                "cross-source TBA_Zyk boundary; "
                "Programmdauer resets between archives"
            )

            rows.append(
                row
            )

            continue

        try:
            features, qc = (
                kernel
                .extract_primary_cycle_features(
                    V,
                    I,
                    t,
                )
            )

            row[
                "raw_sample_count_qc"
            ] = rwth_qc_value(
                qc,
                (
                    "raw_sample_count_qc",
                    "raw_sample_count",
                ),
                raw_n,
            )

            row[
                "valid_sample_count_qc"
            ] = rwth_qc_value(
                qc,
                (
                    "valid_sample_count_qc",
                    "valid_sample_count",
                ),
                valid_n,
            )

            row[
                "dropped_nonfinite_samples_qc"
            ] = rwth_qc_value(
                qc,
                (
                    "dropped_nonfinite_samples_qc",
                    "dropped_nonfinite_samples",
                ),
                dropped_n,
            )

            row[
                "duration_s_qc_only"
            ] = rwth_qc_value(
                qc,
                (
                    "duration_s_qc_only",
                    "duration_s",
                ),
                duration_fallback,
            )

            row[
                "duplicate_or_zero_dt_count_qc"
            ] = rwth_qc_value(
                qc,
                (
                    "duplicate_or_zero_dt_count_qc",
                    "duplicate_or_zero_dt_count",
                ),
                zero_dt_fallback,
            )

            row[
                "backward_time_step_count_qc"
            ] = rwth_qc_value(
                qc,
                (
                    "backward_time_step_count_qc",
                    "backward_time_step_count",
                ),
                backward_fallback,
            )

            for feature_name in (
                kernel
                .PRIMARY_FEATURE_NAMES
            ):
                row[
                    feature_name
                ] = features[
                    feature_name
                ]

            row[
                "feature_valid"
            ] = True

        except Exception as exc:
            row[
                "failure_reason"
            ] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        rows.append(
            row
        )

    out = pd.DataFrame(
        rows,
        columns=columns,
    )

    if out[
        "raw_cycle_key"
    ].duplicated().any():
        raise AssertionError(
            f"{raw_cell_id}: "
            "duplicate pre-filter cycle key"
        )

    return out


def rwth_load_resume_contract(
    expected_base,
):
    if not (
        RWTH_RESUME_CONTRACT
        .is_file()
    ):
        return None

    try:
        observed = json.loads(
            RWTH_RESUME_CONTRACT
            .read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None

    for key, value in (
        expected_base.items()
    ):
        if observed.get(
            key
        ) != value:
            return None

    rows_by_cell = (
        observed.get(
            "rows_by_cell"
        )
    )

    if not isinstance(
        rows_by_cell,
        dict,
    ):
        return None

    if set(
        rows_by_cell
    ) != set(
        RWTH_EXPECTED_CELLS
    ):
        return None

    try:
        rows_by_cell = {
            str(k):
                int(v)
            for k, v
            in rows_by_cell.items()
        }
    except Exception:
        return None

    if sum(
        rows_by_cell.values()
    ) != (
        RWTH_EXPECTED_PREFILTER_CYCLES
    ):
        return None

    return {
        **observed,
        "rows_by_cell":
            rows_by_cell,
    }


def generate_rwth(
    *,
    resume=True,
):
    """
    Generate the complete 105,663-cycle PRE-abnormal-filter
    sensor feature pool.

    The later 657-cycle BatteryML Qd-based exclusion is not
    performed here.
    """
    kernel = load_kernel()

    kernel_hash = sha256_file(
        KERNEL_PATH
    )

    outer_path = (
        rwth_discover_outer_zip()
    )

    raw_outer_hash = (
        sha256_file(
            outer_path
        )
    )

    if (
        raw_outer_hash
        != RWTH_EXPECTED_OUTER_ZIP_SHA256
    ):
        raise AssertionError(
            "RWTH outer ZIP SHA mismatch"
        )

    members_by_cell = (
        rwth_member_inventory(
            outer_path
        )
    )

    RWTH_SHARDS.mkdir(
        parents=True,
        exist_ok=True,
    )

    contract_base = {
        "dataset":
            "RWTH",

        "adapter_contract_version":
            RWTH_ADAPTER_CONTRACT_VERSION,

        "feature_contract_version":
            FEATURE_CONTRACT_VERSION,

        "kernel_sha256":
            kernel_hash,

        "raw_outer_zip_sha256":
            raw_outer_hash,

        "expected_cells":
            list(
                RWTH_EXPECTED_CELLS
            ),

        "expected_raw_cells":
            RWTH_EXPECTED_RAW_CELLS,

        "expected_TBA_Zyk_archives":
            RWTH_EXPECTED_TBA_ZYK_ARCHIVES,

        "expected_prefilter_cycles":
            RWTH_EXPECTED_PREFILTER_CYCLES,
    }

    resume_contract = None

    if resume:
        resume_contract = (
            rwth_load_resume_contract(
                contract_base
            )
        )

    columns = rwth_shard_columns(
        kernel
    )

    generated = 0
    reused = 0
    rows_by_cell = {}

    milestones = {
        1,
        8,
        16,
        24,
        32,
        40,
        48,
    }

    for i, cell in enumerate(
        RWTH_EXPECTED_CELLS,
        start=1,
    ):
        shard = (
            RWTH_SHARDS
            / f"{cell}.csv"
        )

        can_reuse = False

        if resume_contract is not None:
            expected_rows = (
                resume_contract[
                    "rows_by_cell"
                ][cell]
            )

            can_reuse = (
                rwth_shard_is_reusable(
                    shard,
                    raw_cell_id=cell,
                    raw_outer_hash=raw_outer_hash,
                    kernel_hash=kernel_hash,
                    expected_rows=expected_rows,
                    columns=columns,
                )
            )

        if can_reuse:
            reused += 1

            rows_by_cell[
                cell
            ] = int(
                resume_contract[
                    "rows_by_cell"
                ][cell]
            )

        else:
            cell_df = (
                process_rwth_cell(
                    outer_path=outer_path,
                    raw_cell_id=cell,
                    members=(
                        members_by_cell[
                            cell
                        ]
                    ),
                    kernel=kernel,
                    kernel_hash=kernel_hash,
                    raw_outer_hash=raw_outer_hash,
                )
            )

            rwth_atomic_csv_write(
                cell_df,
                shard,
            )

            rows_by_cell[
                cell
            ] = len(
                cell_df
            )

            generated += 1

        if i in milestones:
            print(
                f"progress {i}/48 "
                f"generated={generated} "
                f"reused={reused}",
                flush=True,
            )

    parts = []

    for cell in (
        RWTH_EXPECTED_CELLS
    ):
        shard = (
            RWTH_SHARDS
            / f"{cell}.csv"
        )

        df = pd.read_csv(
            shard,
            low_memory=False,
        )

        if list(
            df.columns
        ) != columns:
            raise AssertionError(
                f"{cell}: RWTH shard "
                "schema mismatch"
            )

        if len(
            df
        ) != int(
            rows_by_cell[
                cell
            ]
        ):
            raise AssertionError(
                f"{cell}: RWTH shard "
                "row-count mismatch"
            )

        parts.append(
            df
        )

    out = pd.concat(
        parts,
        ignore_index=True,
    )

    if len(
        rows_by_cell
    ) != 48:
        raise AssertionError(
            "Unexpected RWTH source cell count"
        )

    if len(
        out
    ) != (
        RWTH_EXPECTED_PREFILTER_CYCLES
    ):
        raise AssertionError(
            "Unexpected RWTH pre-filter "
            f"cycle count: {len(out)}"
        )

    if out[
        [
            "raw_cell_id",
            "raw_cycle_key",
        ]
    ].duplicated().any():
        raise AssertionError(
            "Duplicate RWTH cell/cycle key"
        )

    source_counts = (
        pd.to_numeric(
            out[
                "source_file_count_qc"
            ],
            errors="raise",
        )
    )

    source_distribution = (
        source_counts
        .value_counts()
        .sort_index()
        .to_dict()
    )

    if source_distribution != {
        1: 105017,
        2: 646,
    }:
        raise AssertionError(
            "Unexpected RWTH source-file "
            f"multiplicity: {source_distribution}"
        )

    feature_names = list(
        kernel.PRIMARY_FEATURE_NAMES
    )

    if len(
        feature_names
    ) != 20:
        raise AssertionError(
            "Unexpected RWTH primary "
            "feature count"
        )

    forbidden_fragments = (
        "capacity",
        "soh",
        "rul",
        "eol",
        "capdch",
        "qd_integrated",
        "qc_integrated",
        "qdischarge",
        "qcharge",
        "ahela",
        "ahlad",
        "cycle_life",
        "batteryml_abnormal",
        "cycle_number_after_filter",
    )

    bad_columns = [
        c
        for c in out.columns
        if any(
            token in c.lower()
            for token
            in forbidden_fragments
        )
    ]

    if bad_columns:
        raise AssertionError(
            "Forbidden target/Qd-like "
            f"RWTH columns: {bad_columns}"
        )

    valid = (
        out[
            "feature_valid"
        ]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    invalid = ~valid

    rwth_atomic_csv_write(
        out,
        RWTH_OUT,
    )

    output_hash = sha256_file(
        RWTH_OUT
    )

    failure_counts = (
        out.loc[
            invalid,
            "failure_reason",
        ]
        .fillna("")
        .value_counts()
        .to_dict()
    )

    manifest = {
        **contract_base,

        "rows_by_cell":
            {
                k:
                    int(v)
                for k, v
                in rows_by_cell.items()
            },

        "raw_prefilter_cycles":
            int(
                len(out)
            ),

        "feature_valid_rows":
            int(
                valid.sum()
            ),

        "feature_invalid_rows":
            int(
                invalid.sum()
            ),

        "source_file_count_distribution":
            {
                str(k):
                    int(v)
                for k, v
                in source_distribution.items()
            },

        "cross_source_boundary_cycles":
            int(
                (
                    source_counts
                    == 2
                ).sum()
            ),

        "primary_feature_count":
            20,

        "AhEla_AhLad_Qd_CapDCH1_read":
            False,

        "SOH_RUL_EOL_TDS_read":
            False,

        "abnormal_filter_applied":
            False,

        "split_scaler_read":
            False,

        "failure_reason_counts":
            {
                str(k):
                    int(v)
                for k, v
                in failure_counts.items()
            },

        "output_csv_sha256":
            output_hash,

        "audit_pass":
            True,
    }

    rwth_atomic_json_write(
        manifest,
        RWTH_MANIFEST,
    )

    summary_lines = [
        "RWTH RAW-CYCLE FEATURE GENERATION",
        "",
        (
            "raw source cells: "
            f"{len(rows_by_cell)}"
        ),
        (
            "TBA_Zyk archives: "
            f"{sum(len(v) for v in members_by_cell.values())}"
        ),
        (
            "pre-abnormal-filter cycles: "
            f"{len(out)}"
        ),
        (
            "feature valid / invalid: "
            f"{int(valid.sum())} / "
            f"{int(invalid.sum())}"
        ),
        (
            "source-file-count distribution: "
            f"{source_distribution}"
        ),
        (
            "cross-source-boundary cycles: "
            f"{int((source_counts == 2).sum())}"
        ),
        (
            "primary feature count: "
            f"{len(feature_names)}"
        ),
        (
            "AhEla/AhLad/Qd/CapDCH1 read: "
            "False"
        ),
        (
            "SOH/RUL/EOL/TDS read: "
            "False"
        ),
        (
            "abnormal filter applied: "
            "False"
        ),
        (
            "split/scaler read: "
            "False"
        ),
        (
            "output CSV SHA256: "
            f"{output_hash}"
        ),
        "",
        "RWTH_RAW_CYCLE_FEATURE_GENERATION_PASS=True",
    ]

    RWTH_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RWTH_SUMMARY.write_text(
        "\n".join(
            summary_lines
        )
        + "\n",
        encoding="utf-8",
    )

    rwth_atomic_json_write(
        {
            **contract_base,

            "rows_by_cell":
                {
                    k:
                        int(v)
                    for k, v
                    in rows_by_cell.items()
                },

            "output_csv_sha256":
                output_hash,
        },
        RWTH_RESUME_CONTRACT,
    )

    print(
        "raw source cells:",
        len(rows_by_cell),
    )

    print(
        "TBA_Zyk archives:",
        sum(
            len(v)
            for v in members_by_cell.values()
        ),
    )

    print(
        "pre-abnormal-filter cycles:",
        len(out),
    )

    print(
        "feature valid / invalid:",
        int(valid.sum()),
        "/",
        int(invalid.sum()),
    )

    print(
        "source-file-count distribution:",
        source_distribution,
    )

    print(
        "cross-source-boundary cycles:",
        int(
            (
                source_counts == 2
            ).sum()
        ),
    )

    print(
        "primary feature count:",
        len(feature_names),
    )

    if failure_counts:
        print(
            "top invalid reasons:"
        )

        for reason, count in list(
            failure_counts.items()
        )[:6]:
            print(
                f"  {int(count)}: "
                f"{reason}"
            )

    print(
        "AhEla/AhLad/Qd/CapDCH1 read:",
        False,
    )

    print(
        "SOH/RUL/EOL/TDS read:",
        False,
    )

    print(
        "abnormal filter applied:",
        False,
    )

    print(
        "split/scaler read:",
        False,
    )

    print(
        "output CSV SHA256:",
        output_hash,
    )

    print(
        "RWTH_RAW_CYCLE_FEATURE_GENERATION_PASS=True"
    )


CALCE_ADAPTER_CONTRACT_VERSION = (
    "calce_raw_vit_grouped_cycle_v2"
)

CALCE_REQUIRED_SENSOR_COLUMNS = (
    "Cycle_Index",
    "Test_Time(s)",
    "Current(A)",
    "Voltage(V)",
)


def calce_extract_date_batteryml(
    filename,
):
    """
    Reproduce the already-audited BatteryML-compatible CALCE
    filename-date parser.

    Deliberately retains two-digit source years as four-digit
    strings such as 0010 because this field is a provenance /
    grouping key, not a physical feature.
    """
    filename = str(
        filename
    ).upper()

    pattern = (
        r"C[XS]2?_\d+_(\d+)_(\d+)B?_(\d+)"
    )

    matches = re.findall(
        pattern,
        filename,
    )

    if not matches:
        pattern = (
            r"(\d+)_(\d+)_(\d+)_CX2_32"
        )

        matches = re.findall(
            pattern,
            filename,
        )

    if not matches:
        raise ValueError(
            "Cannot parse CALCE date from: "
            f"{filename}"
        )

    month, day, year = (
        matches[0]
    )

    month = int(month)
    day = int(day)
    year = int(year)

    return (
        f"{year:04d}-"
        f"{month:02d}-"
        f"{day:02d}"
    )


def calce_extract_date_human(
    filename,
):
    """
    Human-readable provenance date. Two-digit years map to
    20xx, matching the independent CALCE audit.
    """
    filename = str(
        filename
    ).upper()

    pattern = (
        r"C[XS]2?_\d+_(\d+)_(\d+)B?_(\d+)"
    )

    matches = re.findall(
        pattern,
        filename,
    )

    if not matches:
        pattern = (
            r"(\d+)_(\d+)_(\d+)_CX2_32"
        )

        matches = re.findall(
            pattern,
            filename,
        )

    if not matches:
        raise ValueError(
            "Cannot parse CALCE date from: "
            f"{filename}"
        )

    month, day, year = map(
        int,
        matches[0],
    )

    if year < 100:
        year += 2000

    return (
        f"{year:04d}-"
        f"{month:02d}-"
        f"{day:02d}"
    )


def calce_load_excel_sensor_bytes(
    raw,
    stem,
):
    """
    Sensor-only CALCE Excel loader.

    Capacity columns present in source workbooks are
    deliberately NOT selected or returned.
    """
    xf = pd.ExcelFile(
        io.BytesIO(raw)
    )

    channel_frames = []

    for sheet_name in (
        xf.sheet_names
    ):
        if str(
            sheet_name
        ).startswith(
            "Channel"
        ):
            channel_frames.append(
                xf.parse(
                    sheet_name,
                    usecols=lambda c:
                        c in CALCE_REQUIRED_SENSOR_COLUMNS,
                )
            )

    # BatteryML-compatible fallback for dirty workbook(s).
    if not channel_frames:
        channel_frames = [
            xf.parse(
                sheet_name,
                usecols=lambda c:
                    c in CALCE_REQUIRED_SENSOR_COLUMNS,
            )
            for sheet_name
            in xf.sheet_names
        ]

    df = pd.concat(
        channel_frames,
        ignore_index=True,
    )

    missing = [
        c
        for c
        in CALCE_REQUIRED_SENSOR_COLUMNS
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            f"{stem}: required CALCE "
            f"sensor columns missing: {missing}"
        )

    # Strict sensor-only selection.
    result = df[
        list(
            CALCE_REQUIRED_SENSOR_COLUMNS
        )
    ].copy()

    result[
        "date_batteryml"
    ] = (
        calce_extract_date_batteryml(
            stem
        )
    )

    result[
        "date_human"
    ] = (
        calce_extract_date_human(
            stem
        )
    )

    result[
        "source_file"
    ] = str(stem)

    result[
        "source_format"
    ] = "xlsx"

    return result


def calce_load_txt_sensor_bytes(
    raw,
    stem,
):
    """
    Sensor-only CALCE TXT loader.

    Source units:
        Time -> s
        mA   -> A
        mV   -> V
    """
    required = [
        "Charge count",
        "Time",
        "mA",
        "mV",
    ]

    df = pd.read_csv(
        io.BytesIO(raw),
        sep="\t",
        usecols=required,
    )

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            f"{stem}: required CALCE "
            f"TXT sensor columns missing: {missing}"
        )

    result = pd.DataFrame({
        "Cycle_Index":
            df[
                "Charge count"
            ] // 2 + 1,

        "Test_Time(s)":
            df[
                "Time"
            ],

        "Current(A)":
            df[
                "mA"
            ] / 1000.0,

        "Voltage(V)":
            df[
                "mV"
            ] / 1000.0,
    })

    result[
        "date_batteryml"
    ] = (
        calce_extract_date_batteryml(
            stem
        )
    )

    result[
        "date_human"
    ] = (
        calce_extract_date_human(
            stem
        )
    )

    result[
        "source_file"
    ] = str(stem)

    result[
        "source_format"
    ] = "txt"

    return result


def calce_organize_cycle_index(
    values,
):
    """
    Reproduce the previously audited BatteryML
    organize_cycle_index() behavior exactly.
    """
    arr = np.asarray(
        values
    ).copy()

    if len(arr) == 0:
        return arr

    current_cycle = arr[0]
    prev_value = arr[0]

    for i in range(
        1,
        len(arr),
    ):
        if arr[i] != prev_value:
            current_cycle += 1
            prev_value = arr[i]

        arr[i] = current_cycle

    return arr


def calce_shard_columns(
    kernel,
):
    """
    Extend common 20-feature schema only with non-model CALCE
    provenance and QC columns.
    """
    base = expected_shard_columns(
        kernel
    )

    feature_names = list(
        kernel.PRIMARY_FEATURE_NAMES
    )

    metadata_count = (
        len(base)
        - len(feature_names)
    )

    return (
        base[:metadata_count]
        + [
            "calce_adapter_contract_version",
            "raw_zip_sha256",
            "date_batteryml",
            "date_human",
            "organized_cycle_index",
            "source_format",
            "source_member_count_qc",
        ]
        + base[metadata_count:]
    )


def calce_atomic_csv_write(
    df,
    path,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
    )

    tmp.replace(
        path
    )


def calce_atomic_json_write(
    obj,
    path,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
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

    tmp.replace(
        path
    )


def calce_resume_contract_base(
    *,
    kernel_hash,
    raw_zip_hashes,
):
    return {
        "dataset":
            "CALCE",

        "adapter_contract_version":
            CALCE_ADAPTER_CONTRACT_VERSION,

        "feature_contract_version":
            FEATURE_CONTRACT_VERSION,

        "kernel_sha256":
            kernel_hash,

        "expected_cells":
            list(
                CALCE_EXPECTED_CELLS
            ),

        "expected_raw_cells":
            CALCE_EXPECTED_RAW_CELLS,

        "expected_grouped_cycles":
            CALCE_EXPECTED_GROUPED_CYCLES,

        "raw_zip_sha256":
            raw_zip_hashes,
    }


def calce_load_resume_contract(
    expected_base,
):
    if not (
        CALCE_RESUME_CONTRACT
        .is_file()
    ):
        return None

    try:
        observed = json.loads(
            CALCE_RESUME_CONTRACT
            .read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None

    for key, value in (
        expected_base.items()
    ):
        if observed.get(
            key
        ) != value:
            return None

    rows_by_cell = (
        observed.get(
            "rows_by_cell"
        )
    )

    if not isinstance(
        rows_by_cell,
        dict,
    ):
        return None

    if set(
        rows_by_cell
    ) != set(
        CALCE_EXPECTED_CELLS
    ):
        return None

    try:
        rows_by_cell = {
            str(k):
                int(v)
            for k, v
            in rows_by_cell.items()
        }
    except Exception:
        return None

    if sum(
        rows_by_cell.values()
    ) != (
        CALCE_EXPECTED_GROUPED_CYCLES
    ):
        return None

    return {
        **observed,
        "rows_by_cell":
            rows_by_cell,
    }


def calce_shard_is_reusable(
    path,
    *,
    raw_cell_id,
    raw_zip_hash,
    kernel_hash,
    expected_rows,
    columns,
):
    path = Path(path)

    if not path.is_file():
        return False

    try:
        df = pd.read_csv(
            path,
            low_memory=False,
        )
    except Exception:
        return False

    if list(
        df.columns
    ) != columns:
        return False

    if len(df) != int(
        expected_rows
    ):
        return False

    if len(df) == 0:
        return False

    checks = [
        (
            df[
                "dataset"
            ] == "CALCE"
        ).all(),

        (
            df[
                "raw_cell_id"
            ].astype(str)
            == str(
                raw_cell_id
            )
        ).all(),

        (
            df[
                "feature_contract_version"
            ]
            == FEATURE_CONTRACT_VERSION
        ).all(),

        (
            df[
                "kernel_sha256"
            ]
            == kernel_hash
        ).all(),

        (
            df[
                "calce_adapter_contract_version"
            ]
            == CALCE_ADAPTER_CONTRACT_VERSION
        ).all(),

        (
            df[
                "raw_zip_sha256"
            ]
            == raw_zip_hash
        ).all(),

        (
            pd.to_numeric(
                df[
                    "source_member_count_qc"
                ],
                errors="coerce",
            )
            == 1
        ).all(),
    ]

    return bool(
        all(checks)
    )


def calce_qc_value(
    qc,
    names,
    default,
):
    if hasattr(
        qc,
        "get",
    ):
        for name in names:
            if name in qc:
                return qc[
                    name
                ]

    return default


def process_calce_cell(
    *,
    zip_path,
    raw_cell_id,
    kernel,
    kernel_hash,
    raw_zip_hash,
):
    """
    Generate one structural feature row per source-native
    CALCE grouped cycle.

    STRICT SENSOR-ONLY CONTRACT
    ---------------------------
    Reads:
        Cycle_Index
        Test_Time(s)
        Current(A)
        Voltage(V)

    TXT source conversion:
        Time -> seconds
        mA   -> ampere
        mV   -> volt

    Does NOT select/read for feature construction:
        Charge_Capacity(Ah)
        Discharge_Capacity(Ah)
        Qd / Qc
        capacity target
        SOH / RUL / EOL
        keep_final_batteryml
        split/scaler information

    Raw construction contract:
        all ZIP members
        -> concatenate
        -> stable sort(date_batteryml, Test_Time(s))
        -> organize Cycle_Index
        -> group(date_batteryml, organized cycle index)

    The previously audited CALCE cohort has exactly one source
    member per grouped cycle.
    """
    frames = []

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as zf:

        members = [
            m
            for m
            in zf.namelist()
            if (
                not m.endswith("/")
                and Path(
                    m
                ).suffix.lower()
                in {
                    ".xlsx",
                    ".xls",
                    ".txt",
                }
            )
        ]

        members = sorted(
            members
        )

        if not members:
            raise ValueError(
                f"{raw_cell_id}: "
                "no CALCE raw members"
            )

        for member in members:
            suffix = (
                Path(
                    member
                )
                .suffix
                .lower()
            )

            stem = (
                Path(
                    member
                )
                .stem
            )

            raw = zf.read(
                member
            )

            if suffix in {
                ".xlsx",
                ".xls",
            }:
                frame = (
                    calce_load_excel_sensor_bytes(
                        raw,
                        stem,
                    )
                )
            else:
                frame = (
                    calce_load_txt_sensor_bytes(
                        raw,
                        stem,
                    )
                )

            frames.append(
                frame
            )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    # Exact chronology contract used by independent audit.
    df = (
        df.sort_values(
            [
                "date_batteryml",
                "Test_Time(s)",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    df[
        "Cycle_Index_organized"
    ] = (
        calce_organize_cycle_index(
            df[
                "Cycle_Index"
            ].to_numpy()
        )
    )

    groups = df.groupby(
        [
            "date_batteryml",
            "Cycle_Index_organized",
        ],
        sort=True,
    )

    columns = calce_shard_columns(
        kernel
    )

    rows = []

    for cycle_number, (
        (date, cyc_idx),
        g,
    ) in enumerate(
        groups
    ):
        source_members = sorted(
            set(
                g[
                    "source_file"
                ]
                .astype(str)
            )
        )

        source_formats = sorted(
            set(
                g[
                    "source_format"
                ]
                .astype(str)
            )
        )

        if len(
            source_members
        ) != 1:
            raise AssertionError(
                f"{raw_cell_id} "
                f"cycle {cycle_number}: "
                "expected exactly one "
                "source member, got "
                f"{source_members}"
            )

        if len(
            source_formats
        ) != 1:
            raise AssertionError(
                f"{raw_cell_id} "
                f"cycle {cycle_number}: "
                "multiple source formats"
            )

        date_human_values = (
            g[
                "date_human"
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        if len(
            date_human_values
        ) != 1:
            raise AssertionError(
                f"{raw_cell_id} "
                f"cycle {cycle_number}: "
                "multiple human dates"
            )

        source_member = (
            source_members[0]
        )

        source_format = (
            source_formats[0]
        )

        date_human = (
            date_human_values[0]
        )

        if pd.isna(
            cyc_idx
        ):
            cyc_key = "nan"
        else:
            cyc_key = format(
                float(
                    cyc_idx
                ),
                ".17g",
            )

        row = {
            "dataset":
                "CALCE",

            "raw_cell_id":
                str(
                    raw_cell_id
                ),

            "source_member":
                source_member,

            "raw_cycle_key":
                (
                    f"{date}|"
                    f"{cyc_key}"
                ),

            "raw_cycle_position_zero_based":
                int(
                    cycle_number
                ),

            "feature_contract_version":
                FEATURE_CONTRACT_VERSION,

            "kernel_sha256":
                kernel_hash,

            "feature_valid":
                False,

            "failure_reason":
                "",

            "raw_sample_count_qc":
                np.nan,

            "valid_sample_count_qc":
                np.nan,

            "dropped_nonfinite_samples_qc":
                np.nan,

            "duration_s_qc_only":
                np.nan,

            "duplicate_or_zero_dt_count_qc":
                np.nan,

            "backward_time_step_count_qc":
                np.nan,

            "calce_adapter_contract_version":
                CALCE_ADAPTER_CONTRACT_VERSION,

            "raw_zip_sha256":
                raw_zip_hash,

            "date_batteryml":
                str(
                    date
                ),

            "date_human":
                date_human,

            "organized_cycle_index":
                float(
                    cyc_idx
                ),

            "source_format":
                source_format,

            "source_member_count_qc":
                1,
        }

        for feature_name in (
            kernel
            .PRIMARY_FEATURE_NAMES
        ):
            row[
                feature_name
            ] = np.nan

        V = g[
            "Voltage(V)"
        ].to_numpy(
            dtype=np.float64
        )

        I = g[
            "Current(A)"
        ].to_numpy(
            dtype=np.float64
        )

        t = g[
            "Test_Time(s)"
        ].to_numpy(
            dtype=np.float64
        )

        raw_n = int(
            len(t)
        )

        finite = (
            np.isfinite(V)
            & np.isfinite(I)
            & np.isfinite(t)
        )

        valid_n = int(
            finite.sum()
        )

        dropped_n = (
            raw_n - valid_n
        )

        tf = t[
            finite
        ]

        if len(tf) >= 2:
            dt = np.diff(
                tf
            )

            duration_fallback = float(
                tf[-1]
                - tf[0]
            )

            zero_dt_fallback = int(
                np.sum(
                    np.isclose(
                        dt,
                        0.0,
                        atol=1e-12,
                        rtol=0.0,
                    )
                )
            )

            backward_fallback = int(
                np.sum(
                    dt < -1e-9
                )
            )
        else:
            duration_fallback = (
                np.nan
            )

            zero_dt_fallback = 0
            backward_fallback = 0

        row[
            "raw_sample_count_qc"
        ] = raw_n

        row[
            "valid_sample_count_qc"
        ] = valid_n

        row[
            "dropped_nonfinite_samples_qc"
        ] = dropped_n

        row[
            "duration_s_qc_only"
        ] = duration_fallback

        row[
            "duplicate_or_zero_dt_count_qc"
        ] = zero_dt_fallback

        row[
            "backward_time_step_count_qc"
        ] = backward_fallback

        try:
            features, qc = (
                kernel
                .extract_primary_cycle_features(
                    V,
                    I,
                    t,
                )
            )

            row[
                "raw_sample_count_qc"
            ] = calce_qc_value(
                qc,
                (
                    "raw_sample_count_qc",
                    "raw_sample_count",
                ),
                raw_n,
            )

            row[
                "valid_sample_count_qc"
            ] = calce_qc_value(
                qc,
                (
                    "valid_sample_count_qc",
                    "valid_sample_count",
                ),
                valid_n,
            )

            row[
                "dropped_nonfinite_samples_qc"
            ] = calce_qc_value(
                qc,
                (
                    "dropped_nonfinite_samples_qc",
                    "dropped_nonfinite_samples",
                ),
                dropped_n,
            )

            row[
                "duration_s_qc_only"
            ] = calce_qc_value(
                qc,
                (
                    "duration_s_qc_only",
                    "duration_s",
                ),
                duration_fallback,
            )

            row[
                "duplicate_or_zero_dt_count_qc"
            ] = calce_qc_value(
                qc,
                (
                    "duplicate_or_zero_dt_count_qc",
                    "duplicate_or_zero_dt_count",
                ),
                zero_dt_fallback,
            )

            row[
                "backward_time_step_count_qc"
            ] = calce_qc_value(
                qc,
                (
                    "backward_time_step_count_qc",
                    "backward_time_step_count",
                ),
                backward_fallback,
            )

            for feature_name in (
                kernel
                .PRIMARY_FEATURE_NAMES
            ):
                row[
                    feature_name
                ] = features[
                    feature_name
                ]

            row[
                "feature_valid"
            ] = True

        except Exception as exc:
            row[
                "failure_reason"
            ] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        rows.append(
            row
        )

    out = pd.DataFrame(
        rows,
        columns=columns,
    )

    # Structural source-native key must be unique within cell.
    if out[
        "raw_cycle_key"
    ].duplicated().any():
        raise AssertionError(
            f"{raw_cell_id}: "
            "duplicate CALCE raw cycle key"
        )

    if not (
        pd.to_numeric(
            out[
                "source_member_count_qc"
            ],
            errors="raise",
        )
        == 1
    ).all():
        raise AssertionError(
            f"{raw_cell_id}: "
            "multi-source grouped cycle detected"
        )

    return out


def generate_calce(
    *,
    resume=True,
):
    """
    Generate source-native, target-independent CALCE
    cycle features.

    No capacity-derived QC or label table is read here.
    """
    kernel = load_kernel()

    kernel_hash = sha256_file(
        KERNEL_PATH
    )

    CALCE_SHARDS.mkdir(
        parents=True,
        exist_ok=True,
    )

    zip_paths = sorted(
        CALCE_RAW_DIR.glob(
            "*.zip"
        )
    )

    observed_cells = [
        p.stem
        for p in zip_paths
    ]

    if observed_cells != list(
        CALCE_EXPECTED_CELLS
    ):
        raise AssertionError(
            "Unexpected CALCE ZIP cohort: "
            f"{observed_cells}"
        )

    if len(
        zip_paths
    ) != CALCE_EXPECTED_RAW_CELLS:
        raise AssertionError(
            "Unexpected CALCE cell count"
        )

    raw_zip_hashes = {
        p.name:
            sha256_file(
                p
            )
        for p in zip_paths
    }

    contract_base = (
        calce_resume_contract_base(
            kernel_hash=kernel_hash,
            raw_zip_hashes=raw_zip_hashes,
        )
    )

    resume_contract = None

    if resume:
        resume_contract = (
            calce_load_resume_contract(
                contract_base
            )
        )

    columns = calce_shard_columns(
        kernel
    )

    generated = 0
    reused = 0

    rows_by_cell = {}

    for i, zip_path in enumerate(
        zip_paths,
        start=1,
    ):
        cell = zip_path.stem

        shard = (
            CALCE_SHARDS
            / f"{cell}.csv"
        )

        can_reuse = False

        if resume_contract is not None:
            expected_rows = (
                resume_contract[
                    "rows_by_cell"
                ][cell]
            )

            can_reuse = (
                calce_shard_is_reusable(
                    shard,
                    raw_cell_id=cell,
                    raw_zip_hash=(
                        raw_zip_hashes[
                            zip_path.name
                        ]
                    ),
                    kernel_hash=kernel_hash,
                    expected_rows=expected_rows,
                    columns=columns,
                )
            )

        if can_reuse:
            reused += 1

            rows_by_cell[
                cell
            ] = int(
                resume_contract[
                    "rows_by_cell"
                ][cell]
            )

        else:
            cell_df = (
                process_calce_cell(
                    zip_path=zip_path,
                    raw_cell_id=cell,
                    kernel=kernel,
                    kernel_hash=kernel_hash,
                    raw_zip_hash=(
                        raw_zip_hashes[
                            zip_path.name
                        ]
                    ),
                )
            )

            calce_atomic_csv_write(
                cell_df,
                shard,
            )

            rows_by_cell[
                cell
            ] = len(
                cell_df
            )

            generated += 1

        print(
            f"progress {i}/"
            f"{len(zip_paths)} "
            f"generated={generated} "
            f"reused={reused}",
            flush=True,
        )

    # Always assemble from the persisted shards. This makes
    # first-run and resume-run serialization deterministic.
    parts = []

    for cell in (
        CALCE_EXPECTED_CELLS
    ):
        shard = (
            CALCE_SHARDS
            / f"{cell}.csv"
        )

        df = pd.read_csv(
            shard,
            low_memory=False,
        )

        if list(
            df.columns
        ) != columns:
            raise AssertionError(
                f"{cell}: "
                "CALCE shard schema mismatch"
            )

        if len(
            df
        ) != int(
            rows_by_cell[
                cell
            ]
        ):
            raise AssertionError(
                f"{cell}: "
                "CALCE shard row mismatch"
            )

        parts.append(
            df
        )

    out = pd.concat(
        parts,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Hard structural guards
    # --------------------------------------------------------

    if len(
        rows_by_cell
    ) != CALCE_EXPECTED_RAW_CELLS:
        raise AssertionError(
            "Unexpected CALCE source cell count"
        )

    if len(
        out
    ) != CALCE_EXPECTED_GROUPED_CYCLES:
        raise AssertionError(
            "Unexpected CALCE grouped-cycle count: "
            f"{len(out)}"
        )

    if out[
        [
            "raw_cell_id",
            "raw_cycle_key",
        ]
    ].duplicated().any():
        raise AssertionError(
            "Duplicate CALCE cell/cycle key"
        )

    if not (
        pd.to_numeric(
            out[
                "source_member_count_qc"
            ],
            errors="raise",
        )
        == 1
    ).all():
        raise AssertionError(
            "CALCE grouped cycle with "
            "multiple source members"
        )

    expected_positions = (
        out.groupby(
            "raw_cell_id",
            sort=False,
        )[
            "raw_cycle_position_zero_based"
        ]
        .apply(
            lambda s:
                np.array_equal(
                    pd.to_numeric(
                        s,
                        errors="raise",
                    ).to_numpy(
                        dtype=int
                    ),
                    np.arange(
                        len(s)
                    ),
                )
        )
    )

    if not expected_positions.all():
        raise AssertionError(
            "CALCE raw cycle positions "
            "are not contiguous"
        )

    # Ensure model X remains exactly the shared 20 features.
    feature_names = list(
        kernel.PRIMARY_FEATURE_NAMES
    )

    if len(
        feature_names
    ) != 20:
        raise AssertionError(
            "Unexpected primary feature count"
        )

    forbidden_fragments = (
        "capacity",
        "soh",
        "rul",
        "eol",
        "qdischarge",
        "qcharge",
        "keep_final",
        "cycle_life",
    )

    bad_columns = [
        c
        for c in out.columns
        if any(
            token in c.lower()
            for token
            in forbidden_fragments
        )
    ]

    if bad_columns:
        raise AssertionError(
            "Forbidden target-like "
            "CALCE columns: "
            f"{bad_columns}"
        )

    valid = (
        out[
            "feature_valid"
        ]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    invalid = ~valid

    multi_source = int(
        (
            pd.to_numeric(
                out[
                    "source_member_count_qc"
                ],
                errors="raise",
            )
            != 1
        ).sum()
    )

    calce_atomic_csv_write(
        out,
        CALCE_OUT,
    )

    output_hash = sha256_file(
        CALCE_OUT
    )

    failure_counts = (
        out.loc[
            invalid,
            "failure_reason",
        ]
        .fillna("")
        .value_counts()
        .to_dict()
    )

    source_format_counts = (
        out[
            "source_format"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    manifest = {
        **contract_base,

        "rows_by_cell":
            {
                k:
                    int(v)
                for k, v
                in rows_by_cell.items()
            },

        "raw_grouped_cycles":
            int(
                len(out)
            ),

        "feature_valid_rows":
            int(
                valid.sum()
            ),

        "feature_invalid_rows":
            int(
                invalid.sum()
            ),

        "multi_source_grouped_cycles":
            multi_source,

        "duplicate_cell_cycle_keys":
            int(
                out[
                    [
                        "raw_cell_id",
                        "raw_cycle_key",
                    ]
                ]
                .duplicated()
                .sum()
            ),

        "primary_feature_count":
            len(
                feature_names
            ),

        "source_format_cycle_counts":
            source_format_counts,

        "failure_reason_counts":
            {
                str(k):
                    int(v)
                for k, v
                in failure_counts.items()
            },

        "capacity_target_columns_read":
            False,

        "qd_qc_read":
            False,

        "soh_rul_eol_read":
            False,

        "keep_final_batteryml_read":
            False,

        "split_scaler_read":
            False,

        "output_csv_sha256":
            output_hash,

        "audit_pass":
            True,
    }

    calce_atomic_json_write(
        manifest,
        CALCE_MANIFEST,
    )

    summary_lines = [
        "CALCE RAW-CYCLE FEATURE GENERATION",
        "",
        (
            "raw source cells: "
            f"{len(rows_by_cell)}"
        ),
        (
            "raw grouped cycles: "
            f"{len(out)}"
        ),
        (
            "feature valid / invalid: "
            f"{int(valid.sum())} / "
            f"{int(invalid.sum())}"
        ),
        (
            "multi-source grouped cycles: "
            f"{multi_source}"
        ),
        (
            "duplicate cell/cycle keys: "
            f"{int(out[['raw_cell_id', 'raw_cycle_key']].duplicated().sum())}"
        ),
        (
            "primary feature count: "
            f"{len(feature_names)}"
        ),
        (
            "capacity/Qd/Qc/SOH/RUL/EOL/"
            "keep read: "
            "False / False / False / "
            "False / False / False / False"
        ),
        (
            "scaler/split read: "
            "False / False"
        ),
        (
            "output CSV SHA256: "
            f"{output_hash}"
        ),
        "",
        "CALCE_RAW_CYCLE_FEATURE_GENERATION_PASS=True",
    ]

    CALCE_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CALCE_SUMMARY.write_text(
        "\n".join(
            summary_lines
        )
        + "\n",
        encoding="utf-8",
    )

    # Resume contract is written only after complete successful
    # assembly and all hard guards.
    calce_atomic_json_write(
        {
            **contract_base,
            "rows_by_cell":
                {
                    k:
                        int(v)
                    for k, v
                    in rows_by_cell.items()
                },
            "output_csv_sha256":
                output_hash,
        },
        CALCE_RESUME_CONTRACT,
    )

    print(
        "raw source cells:",
        len(
            rows_by_cell
        ),
    )

    print(
        "raw grouped cycles:",
        len(out),
    )

    print(
        "feature valid / invalid:",
        int(
            valid.sum()
        ),
        "/",
        int(
            invalid.sum()
        ),
    )

    print(
        "multi-source grouped cycles:",
        multi_source,
    )

    print(
        "duplicate cell/cycle keys:",
        int(
            out[
                [
                    "raw_cell_id",
                    "raw_cycle_key",
                ]
            ]
            .duplicated()
            .sum()
        ),
    )

    print(
        "primary feature count:",
        len(
            feature_names
        ),
    )

    print(
        "source format cycle counts:",
        source_format_counts,
    )

    if failure_counts:
        print(
            "top invalid reasons:"
        )

        for reason, count in list(
            failure_counts.items()
        )[:5]:
            print(
                f"  {int(count)}: "
                f"{reason}"
            )

    print(
        "capacity/Qd/Qc/SOH/RUL/EOL/"
        "keep read:",
        "False / False / False / "
        "False / False / False / False",
    )

    print(
        "scaler/split read:",
        "False / False",
    )

    print(
        "output CSV SHA256:",
        output_hash,
    )

    print(
        "CALCE_RAW_CYCLE_FEATURE_GENERATION_PASS=True"
    )


MATR_ADAPTER_CONTRACT_VERSION = (
    "matr_vit20_isolated_timestamp_midpoint_v2"
)

MATR_TIME_TOLERANCE_S = 1e-9


def matr_shard_columns(
    kernel,
):
    """
    MATR extends the common feature-table schema only with
    non-model provenance/QC fields.

    These columns are never part of the 20-dimensional model X.
    """
    base = expected_shard_columns(
        kernel
    )

    feature_names = list(
        kernel.PRIMARY_FEATURE_NAMES
    )

    metadata_count = (
        len(base)
        - len(feature_names)
    )

    return (
        base[:metadata_count]
        + [
            "timestamp_repaired_qc",
            "source_backward_time_step_count_qc",
            "source_material_backward_time_step_count_qc",
            "source_subtolerance_negative_time_step_count_qc",
            "timestamp_repair_policy_qc",
        ]
        + base[metadata_count:]
    )


def matr_shard_is_reusable(
    path,
    *,
    raw_cell_id,
    source_member,
    kernel_hash,
    contract,
    expected_rows,
    columns,
):
    path = Path(path)

    if not path.is_file():
        return False

    try:
        df = pd.read_csv(
            path
        )
    except Exception:
        return False

    if list(
        df.columns
    ) != columns:
        return False

    if len(df) != expected_rows:
        return False

    if len(df) == 0:
        return False

    checks = [
        (
            df["dataset"]
            == "MATR"
        ).all(),

        (
            df["raw_cell_id"]
            .astype(str)
            == str(raw_cell_id)
        ).all(),

        (
            df["source_member"]
            == source_member
        ).all(),

        (
            df[
                "feature_contract_version"
            ]
            == contract
        ).all(),

        (
            df["kernel_sha256"]
            == kernel_hash
        ).all(),
    ]

    return bool(
        all(checks)
    )


def matr_resume_contract_matches(
    expected,
):
    if not MATR_RESUME_CONTRACT.is_file():
        return False

    try:
        found = json.loads(
            MATR_RESUME_CONTRACT.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return False

    return found == expected


def process_matr_cell(
    *,
    f,
    cycles,
    source_member,
    raw_cell_id,
    kernel,
    kernel_hash,
):
    """
    Generate one structural output row per physical MATR cycle object.

    STRICT SENSOR-ONLY CONTRACT
    ---------------------------
    Reads:
        cycles.V
        cycles.I
        cycles.t

    Does NOT read:
        summary
        QDischarge
        QCharge
        capacity
        SOH
        RUL
        EOL
        split/scaler information

    Previously audited MATR raw semantics:
        V_V = stored V
        I_A = stored I * 1.1
        t_s = stored t * 60

    Previously audited alignment:
        cycle-object indices are never shifted.

    Therefore B1 index-0 placeholders remain explicit invalid rows,
    while b4c31 contains only its physically existing 833 objects.
    """

    n_cycles = int(
        cycles["V"].shape[0]
    )

    columns = matr_shard_columns(
        kernel
    )

    rows = []

    for j in range(
        n_cycles
    ):
        row = {
            "dataset":
                "MATR",

            "raw_cell_id":
                str(raw_cell_id),

            "source_member":
                source_member,

            "raw_cycle_key":
                str(j),

            "raw_cycle_position_zero_based":
                int(j),

            "feature_contract_version":
                FEATURE_CONTRACT_VERSION,

            "kernel_sha256":
                kernel_hash,

            "feature_valid":
                False,

            "failure_reason":
                "",

            "raw_sample_count_qc":
                np.nan,

            "valid_sample_count_qc":
                np.nan,

            "dropped_nonfinite_samples_qc":
                np.nan,

            "duration_s_qc_only":
                np.nan,

            "duplicate_or_zero_dt_count_qc":
                np.nan,

            "backward_time_step_count_qc":
                np.nan,

            "timestamp_repaired_qc":
                False,

            "source_backward_time_step_count_qc":
                np.nan,

            "source_material_backward_time_step_count_qc":
                np.nan,

            "source_subtolerance_negative_time_step_count_qc":
                np.nan,

            "timestamp_repair_policy_qc":
                "none",
        }

        for feature_name in (
            kernel
            .PRIMARY_FEATURE_NAMES
        ):
            row[
                feature_name
            ] = np.nan

        try:
            V = matr_deref(
                f,
                cycles["V"],
                j,
            )

            # Audit 11:
            # stored MATR current is C-rate.
            I = (
                matr_deref(
                    f,
                    cycles["I"],
                    j,
                )
                * 1.1
            )

            # Audit 11:
            # stored MATR time is minutes.
            #
            # Keep native minutes during source-anomaly
            # inspection; convert to seconds only after the
            # strictly guarded repair policy below.
            t_native = matr_deref(
                f,
                cycles["t"],
                j,
            )

            if not (
                len(V)
                == len(I)
                == len(t_native)
            ):
                raise ValueError(
                    "V/I/t length mismatch before "
                    "MATR timestamp QC"
                )

            source_dt = np.diff(
                t_native
            )

            raw_negative_positions = np.where(
                np.isfinite(source_dt)
                & (source_dt < 0)
            )[0]

            # Native MATR t is minutes.  The feature kernel
            # receives seconds and defines backward time as
            # dt < -1e-9 s.  Preserve *all* raw negatives for
            # provenance, but distinguish numerical jitter
            # from physically material backward steps.
            source_dt_s = (
                source_dt
                * 60.0
            )

            material_negative_positions = np.where(
                np.isfinite(source_dt_s)
                & (
                    source_dt_s
                    < -MATR_TIME_TOLERANCE_S
                )
            )[0]

            subtolerance_negative_positions = np.where(
                np.isfinite(source_dt_s)
                & (source_dt_s < 0)
                & (
                    source_dt_s
                    >= -MATR_TIME_TOLERANCE_S
                )
            )[0]

            source_backward = int(
                len(
                    raw_negative_positions
                )
            )

            material_backward = int(
                len(
                    material_negative_positions
                )
            )

            subtolerance_backward = int(
                len(
                    subtolerance_negative_positions
                )
            )

            if (
                source_backward
                != (
                    material_backward
                    + subtolerance_backward
                )
            ):
                raise AssertionError(
                    "MATR negative-time classification "
                    "is not exhaustive"
                )

            row[
                "source_backward_time_step_count_qc"
            ] = source_backward

            row[
                "source_material_backward_time_step_count_qc"
            ] = material_backward

            row[
                "source_subtolerance_negative_time_step_count_qc"
            ] = subtolerance_backward

            # ------------------------------------------------
            # Strict B4 isolated-timestamp repair.
            #
            # Audits 15/16 established 574 B4 cycles with:
            #   * exactly one negative step,
            #   * immediate return to the surrounding
            #     chronological time scale,
            #   * monotonic sequence after removing only the
            #     isolated corrupt timestamp,
            #   * monotonic sequence after midpoint repair.
            #
            # V/I values and source sample ordering are never
            # modified.
            # ------------------------------------------------

            if material_backward:

                if not str(
                    raw_cell_id
                ).startswith(
                    "b4c"
                ):
                    raise ValueError(
                        "Material backward timestamp "
                        "outside MATR batch 4"
                    )

                if material_backward != 1:
                    raise ValueError(
                        "MATR B4 timestamp repair requires "
                        "exactly one material backward step"
                    )

                if source_backward != 1:
                    raise ValueError(
                        "MATR B4 material anomaly must be "
                        "the only raw negative step"
                    )

                k = int(
                    material_negative_positions[0]
                )

                bad_idx = (
                    k + 1
                )

                if (
                    bad_idx + 1
                    >= len(
                        t_native
                    )
                ):
                    raise ValueError(
                        "Isolated corrupt timestamp "
                        "has no right neighbour"
                    )

                left = float(
                    t_native[k]
                )

                corrupt = float(
                    t_native[
                        bad_idx
                    ]
                )

                right = float(
                    t_native[
                        bad_idx + 1
                    ]
                )

                if not (
                    np.isfinite(left)
                    and np.isfinite(corrupt)
                    and np.isfinite(right)
                ):
                    raise ValueError(
                        "Timestamp repair neighbours "
                        "must be finite"
                    )

                if not (
                    right > left
                ):
                    raise ValueError(
                        "Timestamp repair requires "
                        "positive left-to-right bridge"
                    )

                without_bad = np.delete(
                    t_native,
                    bad_idx,
                )

                finite_without = (
                    without_bad[
                        np.isfinite(
                            without_bad
                        )
                    ]
                )

                if (
                    len(
                        finite_without
                    ) >= 2
                    and np.any(
                        np.diff(
                            finite_without
                        ) < 0
                    )
                ):
                    raise ValueError(
                        "Removing isolated timestamp "
                        "does not restore monotonicity"
                    )

                repaired_native = (
                    t_native.copy()
                )

                repaired_native[
                    bad_idx
                ] = (
                    left
                    + right
                ) / 2.0

                finite_repaired = (
                    repaired_native[
                        np.isfinite(
                            repaired_native
                        )
                    ]
                )

                if (
                    len(
                        finite_repaired
                    ) >= 2
                    and np.any(
                        np.diff(
                            finite_repaired
                        ) < 0
                    )
                ):
                    raise ValueError(
                        "Midpoint timestamp repair "
                        "does not restore monotonicity"
                    )

                t_native = (
                    repaired_native
                )

                row[
                    "timestamp_repaired_qc"
                ] = True

                row[
                    "timestamp_repair_policy_qc"
                ] = (
                    "isolated_B4_timestamp_"
                    "midpoint_between_valid_neighbors"
                )

            t = (
                t_native
                * 60.0
            )

            row[
                "raw_sample_count_qc"
            ] = int(
                len(V)
            )

            features, qc = (
                kernel
                .extract_primary_cycle_features(
                    V,
                    I,
                    t,
                )
            )

            row.update(
                features
            )

            row[
                "feature_valid"
            ] = True

            row[
                "valid_sample_count_qc"
            ] = qc[
                "valid_sample_count"
            ]

            row[
                "dropped_nonfinite_samples_qc"
            ] = qc[
                "dropped_nonfinite_samples"
            ]

            row[
                "duration_s_qc_only"
            ] = qc[
                "duration_s_qc_only"
            ]

            row[
                "duplicate_or_zero_dt_count_qc"
            ] = qc[
                "duplicate_or_zero_dt_count"
            ]

            row[
                "backward_time_step_count_qc"
            ] = qc[
                "backward_time_step_count"
            ]

        except Exception as exc:
            row[
                "failure_reason"
            ] = (
                type(exc).__name__
                + ": "
                + str(exc)
            )[:500]

        rows.append(
            row
        )

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def inspect_matr_structure():
    """
    Structural inventory from cycles groups only.

    Critically, this function never dereferences batch.summary.
    """

    structure = []

    for batch, path in (
        MATR_FILES.items()
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        with h5py.File(
            path,
            "r"
        ) as f:
            b = f["batch"]

            n_cells = int(
                b["cycles"].shape[0]
            )

            for i in range(
                n_cells
            ):
                cycles = f[
                    b["cycles"][i, 0]
                ]

                n_cycles = int(
                    cycles["V"].shape[0]
                )

                structure.append({
                    "batch":
                        batch,

                    "path":
                        path,

                    "raw_cell_index":
                        int(i),

                    "raw_cell_id":
                        f"{batch}c{i}",

                    "source_member":
                        str(
                            path.relative_to(
                                BASE
                            )
                        ),

                    "raw_cycle_objects":
                        n_cycles,
                })

    return structure


def generate_matr(
    *,
    resume=True,
):
    kernel = load_kernel()

    kernel_hash = sha256_file(
        KERNEL_PATH
    )

    raw_hashes = {
        str(
            path.relative_to(
                BASE
            )
        ):
            sha256_file(
                path
            )
        for path in (
            MATR_FILES.values()
        )
    }

    structure = (
        inspect_matr_structure()
    )

    if len(structure) != (
        EXPECTED_MATR_RAW_CELLS
    ):
        raise AssertionError(
            "Unexpected MATR raw cell count: "
            f"{len(structure)}"
        )

    b1_cells = sum(
        x["batch"] == "b1"
        for x in structure
    )

    if b1_cells != (
        EXPECTED_MATR_B1_CELLS
    ):
        raise AssertionError(
            "Unexpected MATR B1 cell count: "
            f"{b1_cells}"
        )

    raw_cycle_count = int(
        sum(
            x[
                "raw_cycle_objects"
            ]
            for x in structure
        )
    )

    MATR_SHARDS.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = matr_shard_columns(
        kernel
    )

    resume_contract = {
        "dataset":
            "MATR",

        "feature_contract_version":
            FEATURE_CONTRACT_VERSION,

        "matr_adapter_contract_version":
            MATR_ADAPTER_CONTRACT_VERSION,

        "generator_sha256":
            sha256_file(
                Path(__file__)
                .resolve()
            ),

        "kernel_sha256":
            kernel_hash,

        "raw_file_sha256":
            raw_hashes,

        "timestamp_repair_policy":
            (
                "isolated B4 single-timestamp midpoint "
                "repair under strict chronology guards"
            ),
    }

    resume_allowed = (
        resume
        and matr_resume_contract_matches(
            resume_contract
        )
    )

    generated = 0
    reused = 0
    shard_paths = []

    processed = 0

    for batch, path in (
        MATR_FILES.items()
    ):
        with h5py.File(
            path,
            "r"
        ) as f:
            b = f["batch"]

            n_cells = int(
                b["cycles"].shape[0]
            )

            source_member = str(
                path.relative_to(
                    BASE
                )
            )

            for i in range(
                n_cells
            ):
                processed += 1

                raw_cell_id = (
                    f"{batch}c{i}"
                )

                cycles = f[
                    b["cycles"][i, 0]
                ]

                n_cycles = int(
                    cycles["V"].shape[0]
                )

                shard_path = (
                    MATR_SHARDS
                    / f"{raw_cell_id}.csv"
                )

                reusable = False

                if resume_allowed:
                    reusable = (
                        matr_shard_is_reusable(
                            shard_path,
                            raw_cell_id=raw_cell_id,
                            source_member=source_member,
                            kernel_hash=kernel_hash,
                            contract=FEATURE_CONTRACT_VERSION,
                            expected_rows=n_cycles,
                            columns=columns,
                        )
                    )

                if reusable:
                    reused += 1

                else:
                    df = (
                        process_matr_cell(
                            f=f,
                            cycles=cycles,
                            source_member=source_member,
                            raw_cell_id=raw_cell_id,
                            kernel=kernel,
                            kernel_hash=kernel_hash,
                        )
                    )

                    if len(df) != n_cycles:
                        raise AssertionError(
                            f"{raw_cell_id}: "
                            "structural row loss"
                        )

                    atomic_csv_write(
                        df,
                        shard_path,
                    )

                    generated += 1

                shard_paths.append(
                    shard_path
                )

                if (
                    processed == 1
                    or processed % 25 == 0
                    or processed
                    == EXPECTED_MATR_RAW_CELLS
                ):
                    print(
                        "progress "
                        f"{processed}/"
                        f"{EXPECTED_MATR_RAW_CELLS} "
                        f"generated={generated} "
                        f"reused={reused}",
                        flush=True,
                    )

    # --------------------------------------------------------
    # Deterministic assembly
    # --------------------------------------------------------

    frames = [
        pd.read_csv(
            p
        )
        for p in shard_paths
    ]

    out = pd.concat(
        frames,
        ignore_index=True,
    )

    if len(out) != raw_cycle_count:
        raise AssertionError(
            "Combined MATR structural row mismatch: "
            f"{len(out)} != {raw_cycle_count}"
        )

    if out[
        [
            "raw_cell_id",
            "raw_cycle_position_zero_based",
        ]
    ].duplicated().any():
        raise AssertionError(
            "Duplicate MATR raw-cycle key."
        )

    feature_cols = list(
        kernel
        .PRIMARY_FEATURE_NAMES
    )

    valid = (
        out[
            "feature_valid"
        ]
        .astype(bool)
    )

    if out.loc[
        valid,
        feature_cols,
    ].isna().any().any():
        raise AssertionError(
            "NaN in valid MATR feature row."
        )

    if out.loc[
        ~valid,
        feature_cols,
    ].notna().all(
        axis=1
    ).any():
        raise AssertionError(
            "Invalid MATR cycle has complete feature vector."
        )

    # --------------------------------------------------------
    # Known structural contracts
    # --------------------------------------------------------

    b1_placeholder_mask = (
        out[
            "raw_cell_id"
        ].astype(str).str.startswith(
            "b1c"
        )
        &
        (
            out[
                "raw_cycle_position_zero_based"
            ] == 0
        )
    )

    if int(
        b1_placeholder_mask.sum()
    ) != EXPECTED_MATR_B1_CELLS:
        raise AssertionError(
            "B1 placeholder structural count changed."
        )

    if out.loc[
        b1_placeholder_mask,
        "feature_valid",
    ].astype(bool).any():
        raise AssertionError(
            "A B1 matched placeholder became feature-valid."
        )

    b4c31 = out[
        out[
            "raw_cell_id"
        ] == "b4c31"
    ]

    if len(b4c31) != 833:
        raise AssertionError(
            "b4c31 must contain exactly "
            "833 physical cycle objects; "
            f"found {len(b4c31)}"
        )

    if (
        int(
            b4c31[
                "raw_cycle_position_zero_based"
            ].min()
        )
        != 0
        or
        int(
            b4c31[
                "raw_cycle_position_zero_based"
            ].max()
        )
        != 832
    ):
        raise AssertionError(
            "b4c31 raw object indices changed."
        )

    forbidden_columns = {
        "capacity_Ah",
        "SOH",
        "RUL",
        "EOL",
        "cycle_life",
        "QDischarge",
        "QCharge",
        "dq",
        "rul",
    }

    overlap = (
        forbidden_columns
        .intersection(
            out.columns
        )
    )

    if overlap:
        raise AssertionError(
            "Forbidden target-like MATR columns: "
            + ",".join(
                sorted(overlap)
            )
        )

    invalid = out[
        ~valid
    ].copy()

    extra_invalid = out[
        (~valid)
        & (~b1_placeholder_mask)
    ].copy()

    repaired = (
        out[
            "timestamp_repaired_qc"
        ]
        .astype(bool)
    )

    source_backward = (
        pd.to_numeric(
            out[
                "source_backward_time_step_count_qc"
            ],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    material_backward = (
        pd.to_numeric(
            out[
                "source_material_backward_time_step_count_qc"
            ],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    subtolerance_backward = (
        pd.to_numeric(
            out[
                "source_subtolerance_negative_time_step_count_qc"
            ],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    raw_negative_cycles = int(
        (
            source_backward > 0
        ).sum()
    )

    material_negative_cycles = int(
        (
            material_backward > 0
        ).sum()
    )

    subtolerance_negative_cycles = int(
        (
            subtolerance_backward > 0
        ).sum()
    )

    # Audits 15-17 plus the epsilon check established:
    #
    #   797 cycles with any raw negative dt
    # = 574 material B4 anomalies
    # + 223 B1/B2 sub-tolerance numerical jitters.
    #
    # Only the 574 material B4 anomalies are repaired.
    if raw_negative_cycles != 797:
        raise AssertionError(
            "Unexpected total MATR raw-negative "
            f"cycle count: {raw_negative_cycles}"
        )

    if material_negative_cycles != 574:
        raise AssertionError(
            "Unexpected MATR material-negative "
            f"cycle count: {material_negative_cycles}"
        )

    if subtolerance_negative_cycles != 223:
        raise AssertionError(
            "Unexpected MATR sub-tolerance-negative "
            f"cycle count: {subtolerance_negative_cycles}"
        )

    if int(
        repaired.sum()
    ) != 574:
        raise AssertionError(
            "Expected exactly 574 strictly repaired "
            "MATR B4 cycles; found "
            f"{int(repaired.sum())}"
        )

    if not (
        repaired
        == (
            material_backward > 0
        )
    ).all():
        raise AssertionError(
            "Material backward cycles and repaired "
            "cycles are not identical."
        )

    if not (
        out.loc[
            repaired,
            "raw_cell_id",
        ]
        .astype(str)
        .str.startswith(
            "b4c"
        )
        .all()
    ):
        raise AssertionError(
            "Timestamp repair escaped MATR B4."
        )

    jitter_rows = (
        subtolerance_backward > 0
    )

    if not (
        out.loc[
            jitter_rows,
            "raw_cell_id",
        ]
        .astype(str)
        .str.match(
            r"^b[12]c"
        )
        .all()
    ):
        raise AssertionError(
            "Sub-tolerance jitter escaped MATR B1/B2."
        )

    if repaired[
        jitter_rows
    ].any():
        raise AssertionError(
            "Sub-tolerance jitter was incorrectly repaired."
        )

    atomic_csv_write(
        out,
        MATR_OUT,
    )

    failure_counts = (
        invalid[
            "failure_reason"
        ]
        .value_counts()
        .head(20)
        .to_dict()
    )

    manifest = {
        "dataset":
            "MATR",

        "feature_contract_version":
            FEATURE_CONTRACT_VERSION,

        "kernel_sha256":
            kernel_hash,

        "raw_file_sha256":
            raw_hashes,

        "raw_source_cells":
            int(
                len(structure)
            ),

        "raw_cycle_objects":
            int(
                len(out)
            ),

        "feature_valid_cycles":
            int(
                valid.sum()
            ),

        "feature_invalid_cycles":
            int(
                (~valid).sum()
            ),

        "b1_matched_placeholders":
            int(
                b1_placeholder_mask.sum()
            ),

        "b1_matched_placeholders_feature_valid":
            int(
                out.loc[
                    b1_placeholder_mask,
                    "feature_valid",
                ]
                .astype(bool)
                .sum()
            ),

        "invalid_excluding_b1_placeholders":
            int(
                len(
                    extra_invalid
                )
            ),

        "timestamp_repaired_cycles":
            int(
                repaired.sum()
            ),

        "source_cycles_with_any_negative_time":
            raw_negative_cycles,

        "source_cycles_with_material_backward_time":
            material_negative_cycles,

        "source_cycles_with_subtolerance_negative_jitter":
            subtolerance_negative_cycles,

        "time_tolerance_s":
            MATR_TIME_TOLERANCE_S,

        "timestamp_repair_policy":
            (
                "isolated B4 single-timestamp midpoint "
                "between valid chronological neighbors"
            ),

        "matr_adapter_contract_version":
            MATR_ADAPTER_CONTRACT_VERSION,

        "b4c31_physical_cycle_objects":
            int(
                len(
                    b4c31
                )
            ),

        "b4c31_fabricated_summary_placeholder":
            False,

        "current_conversion":
            "I_A = stored_I * 1.1",

        "time_conversion":
            "t_s = stored_t * 60",

        "cycle_object_index_shift":
            0,

        "primary_feature_count":
            len(
                feature_cols
            ),

        "primary_feature_names":
            feature_cols,

        "target_table_read":
            False,

        "summary_read":
            False,

        "QDischarge_read":
            False,

        "QCharge_read":
            False,

        "scaler_applied":
            False,

        "split_applied":
            False,

        "temperature_in_primary_features":
            False,

        "failure_counts_top20":
            failure_counts,

        "output_csv_sha256":
            sha256_file(
                MATR_OUT
            ),
    }

    MATR_MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = [
        "MATR raw-cycle feature generation",
        "",
        f"raw_source_cells={len(structure)}",
        f"raw_cycle_objects={len(out)}",
        f"feature_valid={int(valid.sum())}",
        f"feature_invalid={int((~valid).sum())}",
        (
            "b1_matched_placeholders="
            f"{int(b1_placeholder_mask.sum())}"
        ),
        (
            "invalid_excluding_b1_placeholders="
            f"{len(extra_invalid)}"
        ),
        (
            "timestamp_repaired_cycles="
            f"{int(repaired.sum())}"
        ),
        (
            "source_cycles_with_any_negative_time="
            f"{raw_negative_cycles}"
        ),
        (
            "source_cycles_with_material_backward_time="
            f"{material_negative_cycles}"
        ),
        (
            "source_cycles_with_subtolerance_negative_jitter="
            f"{subtolerance_negative_cycles}"
        ),
        (
            "b4c31_physical_cycle_objects="
            f"{len(b4c31)}"
        ),
        "b4c31_fabricated_placeholder=False",
        "",
        "current_conversion=stored_I*1.1_A",
        "time_conversion=stored_t*60_s",
        "cycle_object_index_shift=0",
        "target_table_read=False",
        "summary_read=False",
        "QDischarge_read=False",
        "QCharge_read=False",
        "scaler_applied=False",
        "split_applied=False",
        "",
        f"kernel_sha256={kernel_hash}",
        f"output_csv_sha256={sha256_file(MATR_OUT)}",
        "",
        "MATR_RAW_CYCLE_FEATURE_GENERATION_PASS=True",
    ]

    MATR_SUMMARY.write_text(
        "\n".join(
            summary
        )
        + "\n",
        encoding="utf-8",
    )

    # Only after successful complete assembly is the resume
    # contract activated for future runs.
    MATR_RESUME_CONTRACT.write_text(
        json.dumps(
            resume_contract,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("MATR RAW-CYCLE FEATURE GENERATION")
    print("=" * 78)

    print(
        "raw source cells:",
        len(structure)
    )

    print(
        "raw cycle objects:",
        len(out)
    )

    print(
        "feature valid / invalid:",
        int(valid.sum()),
        "/",
        int((~valid).sum())
    )

    print(
        "B1 placeholders invalid:",
        int(
            (
                ~out.loc[
                    b1_placeholder_mask,
                    "feature_valid",
                ].astype(bool)
            ).sum()
        ),
        "/",
        EXPECTED_MATR_B1_CELLS
    )

    print(
        "invalid excluding B1 placeholders:",
        len(extra_invalid)
    )

    print(
        "timestamp repaired cycles:",
        int(
            repaired.sum()
        )
    )

    print(
        "source cycles with any negative time:",
        raw_negative_cycles
    )

    print(
        "material backward-time cycles:",
        material_negative_cycles
    )

    print(
        "sub-tolerance negative jitter cycles:",
        subtolerance_negative_cycles
    )

    print(
        "b4c31 physical objects:",
        len(b4c31)
    )

    print(
        "b4c31 fabricated placeholder:",
        False
    )

    print(
        "current conversion:",
        "stored_I*1.1 -> A"
    )

    print(
        "time conversion:",
        "stored_t*60 -> s"
    )

    print(
        "cycle-object index shift:",
        0
    )

    print(
        "target/summary/QDischarge read:",
        False,
        "/",
        False,
        "/",
        False
    )

    print(
        "scaler / split applied:",
        False,
        "/",
        False
    )

    if len(invalid):
        print()
        print(
            "top invalid reasons:"
        )

        for reason, count in (
            invalid[
                "failure_reason"
            ]
            .value_counts()
            .head(5)
            .items()
        ):
            print(
                f"  {count}: {reason}"
            )

    print()
    print(
        "output CSV SHA256:",
        sha256_file(
            MATR_OUT
        )
    )

    print(
        "MATR_RAW_CYCLE_FEATURE_GENERATION_PASS=True"
    )


def generate_hust(
    *,
    resume=True,
):
    kernel = load_kernel()

    kernel_hash = sha256_file(
        KERNEL_PATH
    )

    raw_hash = sha256_file(
        HUST_ZIP
    )

    members, raw_cycle_count = (
        inspect_raw_counts()
    )

    if len(members) != (
        EXPECTED_HUST_CELLS
    ):
        raise AssertionError(
            "Unexpected HUST cell count: "
            f"{len(members)}"
        )

    if raw_cycle_count != (
        EXPECTED_HUST_RAW_CYCLES
    ):
        raise AssertionError(
            "Unexpected HUST raw cycle count: "
            f"{raw_cycle_count}"
        )

    HUST_SHARDS.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = (
        expected_shard_columns(
            kernel
        )
    )

    generated = 0
    reused = 0
    shard_paths = []

    with zipfile.ZipFile(
        HUST_ZIP,
        "r",
    ) as zf:

        for idx, member in enumerate(
            members,
            start=1,
        ):

            with zf.open(
                member
            ) as f:
                obj = pickle.load(f)

            if len(obj) != 1:
                raise ValueError(
                    f"{member}: "
                    "ambiguous cell container"
                )

            raw_cell_id = next(
                iter(obj)
            )

            cell = obj[
                raw_cell_id
            ]

            # Important:
            # Only raw sensor dictionary is structurally needed.
            # Never read dq/rul.
            data = cell[
                "data"
            ]

            shard_name = (
                str(raw_cell_id)
                .replace(
                    "/",
                    "_"
                )
                .replace(
                    "\\",
                    "_"
                )
                + ".csv"
            )

            shard_path = (
                HUST_SHARDS
                / shard_name
            )

            reusable = False

            if resume:
                reusable = (
                    shard_is_reusable(
                        shard_path,
                        raw_cell_id=raw_cell_id,
                        source_member=member,
                        kernel_hash=kernel_hash,
                        contract=FEATURE_CONTRACT_VERSION,
                        expected_rows=len(
                            data
                        ),
                        columns=columns,
                    )
                )

            if reusable:
                reused += 1

            else:
                df = (
                    process_hust_cell(
                        source_member=member,
                        raw_cell_id=raw_cell_id,
                        cell=cell,
                        kernel=kernel,
                        kernel_hash=kernel_hash,
                    )
                )

                if len(df) != len(
                    data
                ):
                    raise AssertionError(
                        f"{raw_cell_id}: "
                        "structural row loss"
                    )

                atomic_csv_write(
                    df,
                    shard_path,
                )

                generated += 1

            shard_paths.append(
                shard_path
            )

            if (
                idx == 1
                or idx % 10 == 0
                or idx == len(
                    members
                )
            ):
                print(
                    "progress "
                    f"{idx}/{len(members)} "
                    f"generated={generated} "
                    f"reused={reused}",
                    flush=True,
                )

    # --------------------------------------------------------
    # Deterministic assembly
    # --------------------------------------------------------

    frames = [
        pd.read_csv(
            p
        )
        for p in shard_paths
    ]

    out = pd.concat(
        frames,
        ignore_index=True,
    )

    if len(out) != (
        EXPECTED_HUST_RAW_CYCLES
    ):
        raise AssertionError(
            "Combined HUST structural row mismatch: "
            f"{len(out)}"
        )

    if out[
        [
            "raw_cell_id",
            "raw_cycle_position_zero_based",
        ]
    ].duplicated().any():
        raise AssertionError(
            "Duplicate HUST raw cycle key."
        )

    # Every feature-valid row must have all 20 model features.
    feature_cols = list(
        kernel
        .PRIMARY_FEATURE_NAMES
    )

    valid = out[
        "feature_valid"
    ].astype(bool)

    if out.loc[
        valid,
        feature_cols,
    ].isna().any().any():
        raise AssertionError(
            "NaN in valid primary feature row."
        )

    # Invalid rows must NOT masquerade as valid model rows.
    if out.loc[
        ~valid,
        feature_cols,
    ].notna().all(
        axis=1
    ).any():
        raise AssertionError(
            "Invalid cycle has complete model feature vector."
        )

    # Explicitly ensure target-like columns never appear.
    forbidden_columns = {
        "capacity_Ah",
        "SOH",
        "RUL",
        "EOL",
        "cycle_life",
        "dq",
        "rul",
    }

    if forbidden_columns.intersection(
        out.columns
    ):
        raise AssertionError(
            "Forbidden target-like output column."
        )

    atomic_csv_write(
        out,
        HUST_OUT,
    )

    invalid = out[
        ~valid
    ]

    failure_counts = (
        invalid[
            "failure_reason"
        ]
        .value_counts()
        .head(20)
        .to_dict()
    )

    manifest = {
        "dataset":
            "HUST",

        "feature_contract_version":
            FEATURE_CONTRACT_VERSION,

        "generator":
            str(
                Path(__file__)
                .resolve()
            ),

        "kernel":
            str(
                KERNEL_PATH
            ),

        "kernel_sha256":
            kernel_hash,

        "raw_zip":
            str(
                HUST_ZIP
            ),

        "raw_zip_sha256":
            raw_hash,

        "raw_cells":
            int(
                len(members)
            ),

        "raw_cycles":
            int(
                len(out)
            ),

        "feature_valid_cycles":
            int(
                valid.sum()
            ),

        "feature_invalid_cycles":
            int(
                (~valid).sum()
            ),

        "primary_feature_count":
            len(
                feature_cols
            ),

        "primary_feature_names":
            feature_cols,

        "target_table_read":
            False,

        "cell_dq_read":
            False,

        "cell_rul_read":
            False,

        "scaler_applied":
            False,

        "split_applied":
            False,

        "absolute_duration_in_model_features":
            False,

        "cycle_number_in_model_features":
            False,

        "temperature_in_primary_features":
            False,

        "failure_counts_top20":
            failure_counts,

        "output_csv_sha256":
            sha256_file(
                HUST_OUT
            ),
    }

    HUST_MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = [
        "HUST raw-cycle feature generation",
        "",
        f"cells={len(members)}",
        f"raw_cycles={len(out)}",
        f"feature_valid={int(valid.sum())}",
        f"feature_invalid={int((~valid).sum())}",
        f"primary_feature_count={len(feature_cols)}",
        f"kernel_sha256={kernel_hash}",
        f"raw_zip_sha256={raw_hash}",
        f"output_csv_sha256={sha256_file(HUST_OUT)}",
        "",
        "target_table_read=False",
        "cell_dq_read=False",
        "cell_rul_read=False",
        "scaler_applied=False",
        "split_applied=False",
        "cycle_number_in_model_features=False",
        "absolute_duration_in_model_features=False",
        "temperature_in_primary_features=False",
        "",
        "HUST_RAW_CYCLE_FEATURE_GENERATION_PASS=True",
    ]

    HUST_SUMMARY.write_text(
        "\n".join(
            summary
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("HUST RAW-CYCLE FEATURE GENERATION")
    print("=" * 78)

    print(
        "cells:",
        len(members)
    )

    print(
        "raw cycles:",
        len(out)
    )

    print(
        "feature valid / invalid:",
        int(valid.sum()),
        "/",
        int((~valid).sum())
    )

    print(
        "primary feature count:",
        len(feature_cols)
    )

    print(
        "target table read:",
        False
    )

    print(
        "dq/rul read:",
        False
    )

    print(
        "scaler / split applied:",
        False,
        "/",
        False
    )

    print(
        "duration/cycle-number model feature:",
        False,
        "/",
        False
    )

    if len(invalid):
        print()
        print(
            "top invalid reasons:"
        )

        for reason, count in (
            invalid[
                "failure_reason"
            ]
            .value_counts()
            .head(5)
            .items()
        ):
            print(
                f"  {count}: {reason}"
            )

    print()
    print(
        "output CSV SHA256:",
        sha256_file(
            HUST_OUT
        )
    )

    print(
        "HUST_RAW_CYCLE_FEATURE_GENERATION_PASS=True"
    )


def main():
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--dataset",
        choices=[
            "HUST",
            "RWTH",
            "CALCE",
            "MATR",
        ],
        required=True,
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
    )

    args = parser.parse_args()

    if args.dataset == "HUST":
        generate_hust(
            resume=(
                not args.no_resume
            )
        )

    elif args.dataset == "RWTH":
        generate_rwth(
            resume=(
                not args.no_resume
            )
        )

    elif args.dataset == "CALCE":
        generate_calce(
            resume=(
                not args.no_resume
            )
        )

    elif args.dataset == "MATR":
        generate_matr(
            resume=(
                not args.no_resume
            )
        )


if __name__ == "__main__":
    main()
