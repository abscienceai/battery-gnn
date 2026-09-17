#!/usr/bin/env python3

from pathlib import Path
import hashlib
import importlib.util
import os
import pickle
import zipfile

import h5py
import numpy as np
import pandas as pd


BASE = Path(
    os.environ["BASE"]
)

KERNEL_PATH = (
    BASE
    / "repo/battery-gnn/"
      "scripts/revision_2026/"
      "cycle_feature_kernel.py"
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


spec = (
    importlib.util
    .spec_from_file_location(
        "cycle_feature_kernel",
        KERNEL_PATH,
    )
)

kernel = (
    importlib.util
    .module_from_spec(spec)
)

spec.loader.exec_module(
    kernel
)


assert (
    kernel.validate_primary_schema()
)

assert len(
    kernel.PRIMARY_FEATURE_NAMES
) == 20


# ============================================================
# HUST representative real cycle
# ============================================================

HUST_ZIP = (
    BASE
    / "data/raw/HUST/"
      "hust_data.zip"
)

with zipfile.ZipFile(
    HUST_ZIP,
    "r"
) as zf:

    with zf.open(
        "our_data/1-1.pkl"
    ) as f:

        cell = pickle.load(f)[
            "1-1"
        ]

df = cell["data"][1]

assert isinstance(
    df,
    pd.DataFrame
)

V_h = (
    pd.to_numeric(
        df["Voltage (V)"],
        errors="coerce",
    )
    .to_numpy(
        dtype=float
    )
)

# Canonical HUST conversion:
# mA -> A
I_h = (
    pd.to_numeric(
        df["Current (mA)"],
        errors="coerce",
    )
    .to_numpy(
        dtype=float
    )
    / 1000.0
)

t_h = (
    pd.to_numeric(
        df["Time (s)"],
        errors="coerce",
    )
    .to_numpy(
        dtype=float
    )
)

fh, qh = (
    kernel
    .extract_primary_cycle_features(
        V_h,
        I_h,
        t_h,
    )
)


# ============================================================
# MATR representative real cycle
# ============================================================

MATR = (
    BASE
    / "data/raw/MATR/"
      "MATR_batch_20170512.mat"
)


def deref(f, ds, idx):
    return (
        np.asarray(
            f[ds[idx, 0]][:]
        )
        .squeeze()
        .astype(float)
    )


with h5py.File(
    MATR,
    "r"
) as f:

    b = f["batch"]

    cycles = f[
        b["cycles"][0, 0]
    ]

    # Audit 12 established:
    # object 0 is the matched B1 placeholder;
    # object 1 is the first physical cycle.
    j = 1

    V_m = deref(
        f,
        cycles["V"],
        j
    )

    # Audit 11 established:
    # stored current is C-rate.
    I_m = (
        deref(
            f,
            cycles["I"],
            j
        )
        * 1.1
    )

    # Audit 11 established:
    # stored time is minutes.
    t_m = (
        deref(
            f,
            cycles["t"],
            j
        )
        * 60.0
    )


fm, qm = (
    kernel
    .extract_primary_cycle_features(
        V_m,
        I_m,
        t_m,
    )
)


# ============================================================
# Hard feature-contract guards
# ============================================================

assert tuple(
    fh.keys()
) == kernel.PRIMARY_FEATURE_NAMES

assert tuple(
    fm.keys()
) == kernel.PRIMARY_FEATURE_NAMES

assert set(
    fh
) == set(
    fm
)


# No target-like field is allowed.
for name in fh:
    low = name.lower()

    for token in (
        kernel
        .FORBIDDEN_PRIMARY_NAME_TOKENS
    ):
        assert token not in low


# Temperature cannot silently enter primary core.
assert all(
    "temp" not in x.lower()
    for x in fh
)


# Duration is QC only, not a model feature.
assert (
    "duration_s_qc_only"
    in qh
)

assert (
    "duration_s_qc_only"
    in qm
)

assert all(
    "duration" not in x.lower()
    for x in fh
)


# ============================================================
# Time-unit invariance test
# ============================================================

# Since primary use of time is normalized phase, multiplying
# every timestamp by a positive constant must not alter the
# model features.

fh_scaled, _ = (
    kernel
    .extract_primary_cycle_features(
        V_h,
        I_h,
        t_h * 17.3,
    )
)

max_time_scale_delta = max(
    abs(
        fh[k]
        - fh_scaled[k]
    )
    for k in fh
)

assert (
    max_time_scale_delta
    < 1e-10
)


# ============================================================
# Determinism test
# ============================================================

fh2, qh2 = (
    kernel
    .extract_primary_cycle_features(
        V_h.copy(),
        I_h.copy(),
        t_h.copy(),
    )
)

for k in fh:
    assert (
        fh[k]
        == fh2[k]
    )

assert qh == qh2


# ============================================================
# Compact physical plausibility checks
# ============================================================

assert (
    1.5
    < fh["v_min"]
    < 5.0
)

assert (
    1.5
    < fh["v_max"]
    < 5.0
)

assert (
    abs(
        fh["i_min"]
    )
    < 10.0
)

assert (
    abs(
        fh["i_max"]
    )
    < 10.0
)

assert (
    1.5
    < fm["v_min"]
    < 5.0
)

assert (
    1.5
    < fm["v_max"]
    < 5.0
)

assert (
    abs(
        fm["i_min"]
    )
    < 10.0
)

assert (
    abs(
        fm["i_max"]
    )
    < 10.0
)


# ============================================================
# Report
# ============================================================

print("=" * 78)
print("PRIMARY CYCLE FEATURE KERNEL AUDIT")
print("=" * 78)

print(
    "primary feature count:",
    len(
        kernel
        .PRIMARY_FEATURE_NAMES
    )
)

print(
    "schema identical HUST/MATR:",
    tuple(fh.keys())
    == tuple(fm.keys())
)

print(
    "target-like feature names:",
    0
)

print(
    "temperature in primary core:",
    False
)

print(
    "absolute duration in primary core:",
    False
)

print()
print(
    "HUST valid samples / QC duration_s:",
    qh["valid_sample_count"],
    "/",
    f"{qh['duration_s_qc_only']:.3f}"
)

print(
    "HUST V range:",
    f"{fh['v_min']:.4f}",
    "to",
    f"{fh['v_max']:.4f}"
)

print(
    "HUST I range [A]:",
    f"{fh['i_min']:.4f}",
    "to",
    f"{fh['i_max']:.4f}"
)

print()
print(
    "MATR valid samples / QC duration_s:",
    qm["valid_sample_count"],
    "/",
    f"{qm['duration_s_qc_only']:.3f}"
)

print(
    "MATR V range:",
    f"{fm['v_min']:.4f}",
    "to",
    f"{fm['v_max']:.4f}"
)

print(
    "MATR I range [A]:",
    f"{fm['i_min']:.4f}",
    "to",
    f"{fm['i_max']:.4f}"
)

print()
print(
    "time-scale invariance max delta:",
    f"{max_time_scale_delta:.3e}"
)

print(
    "deterministic repeat:",
    True
)

print()
print(
    "PRIMARY_FEATURES="
    + ",".join(
        kernel
        .PRIMARY_FEATURE_NAMES
    )
)

print()
print(
    "KERNEL_SHA256=",
    sha256_file(
        KERNEL_PATH
    )
)

print(
    "PRIMARY_CYCLE_FEATURE_KERNEL_AUDIT_PASS=True"
)
