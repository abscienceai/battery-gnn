#!/usr/bin/env python3

from pathlib import Path
import h5py
import numpy as np
import pandas as pd


BASE = Path.home() / "alper/battery_revision_2026"
RAW = BASE / "data/raw/MATR"
OUT = BASE / "data/audit"

FILES = {
    1: RAW / "MATR_batch_20170512.mat",
    2: RAW / "MATR_batch_20170630.mat",
    3: RAW / "MATR_batch_20180412.mat",
    4: RAW / "MATR_batch_20190124.mat",
}

THRESHOLD = 0.88

# Official LoadData.m continuation mapping, converted to zero-based keys.
CONTINUATIONS = {
    "b1c0": ("b2c7", 662),
    "b1c1": ("b2c8", 981),
    "b1c2": ("b2c9", 1060),
    "b1c3": ("b2c15", 208),
    "b1c4": ("b2c16", 482),
}

BATCH2_CONTINUATIONS = {
    "b2c7", "b2c8", "b2c9", "b2c15", "b2c16"
}

# Optional exclusions used in the paper's LoadData.m
BATCH1_NONFINISH = {
    "b1c8", "b1c10", "b1c12", "b1c13", "b1c22"
}


def get_vector(group, name):
    return np.hstack(group[name][0, :].tolist()).astype(float)


def get_scalar(f, ref):
    x = np.asarray(f[ref][:]).squeeze()
    try:
        return float(x)
    except Exception:
        return np.nan


def load_batch(batch_number):
    path = FILES[batch_number]
    result = []

    with h5py.File(path, "r") as f:
        b = f["batch"]
        n = int(b["summary"].shape[0])

        for i in range(n):
            key = f"b{batch_number}c{i}"

            summary = f[b["summary"][i, 0]]

            qd = get_vector(summary, "QDischarge")
            cyc = get_vector(summary, "cycle")

            source_cl = get_scalar(
                f,
                b["cycle_life"][i, 0]
            )

            result.append({
                "cell_id": key,
                "batch": batch_number,
                "original_index_zero_based": i,
                "qd": qd,
                "cycle": cyc,
                "source_cycle_life": source_cl,
            })

    return result


def first_valid_index(qd):
    """
    Ignore nonphysical placeholder values such as Batch-1 QD=0.
    """
    idx = np.where(
        np.isfinite(qd) & (qd > 0.1)
    )[0]

    return int(idx[0]) if len(idx) else None


def first_cross(qd):
    """
    Uniform first physically valid QD < 0.88 crossing.
    Returns 1-based summary position.
    """
    start = first_valid_index(qd)

    if start is None:
        return np.nan

    idx = np.where(
        np.isfinite(qd[start:])
        & (qd[start:] > 0.1)
        & (qd[start:] < THRESHOLD)
    )[0]

    if not len(idx):
        return np.nan

    return float(start + idx[0] + 1)


def sustained_cross(qd, n):
    """
    First position of n consecutive physically valid values <0.88.
    Returns 1-based position of the FIRST point in the run.
    """
    start = first_valid_index(qd)

    if start is None:
        return np.nan

    valid_below = (
        np.isfinite(qd)
        & (qd > 0.1)
        & (qd < THRESHOLD)
    )

    for i in range(start, len(qd) - n + 1):
        if np.all(valid_below[i:i+n]):
            return float(i + 1)

    return np.nan


def paper_loadData_label(qd):
    """
    Exact target rule in Severson et al. LoadData.m:

    if final QDischarge < 0.88:
        first QDischarge < 0.88
    else:
        number of cycles + 1

    This intentionally reproduces the source code rather than
    introducing our own revised definition.
    """
    if qd[-1] < THRESHOLD:
        idx = np.where(qd < THRESHOLD)[0]

        if not len(idx):
            return np.nan

        return float(idx[0] + 1)

    return float(len(qd) + 1)


def cell_analysis_reconstruction(qd):
    """
    Reconstruct source cell_analysis.m cycle_life logic.

    If final capacity is below 0.88, accept a below-threshold point
    when the next capacity point is still lower.
    If testing stops just above failure (<0.01 Ah above threshold),
    assign record length + 1.
    """
    if qd[-1] < THRESHOLD:

        lower = np.where(qd < THRESHOLD)[0]

        for idx in lower:
            if idx + 1 >= len(qd):
                continue

            if qd[idx + 1] < qd[idx]:
                return float(idx + 1)

        return np.nan

    if (qd[-1] - THRESHOLD) < 0.01:
        return float(len(qd) + 1)

    return np.nan


batches = {
    b: load_batch(b)
    for b in range(1, 5)
}

lookup = {
    cell["cell_id"]: cell
    for batch in batches.values()
    for cell in batch
}


# =====================================================================
# 1. Reconstruct original Severson LoadData.m cohort
# =====================================================================

batch1 = [dict(x) for x in batches[1]]
batch2 = [dict(x) for x in batches[2]]
batch3 = [dict(x) for x in batches[3]]


# Append five continuation records to Batch 1.
for b1key, (b2key, _) in CONTINUATIONS.items():

    a = next(x for x in batch1 if x["cell_id"] == b1key)
    b = lookup[b2key]

    a["qd"] = np.hstack([a["qd"], b["qd"]])

    # Cycle numbering does not matter for LoadData bat_label,
    # but create a continuous index for audit interpretation.
    a["cycle"] = np.arange(
        1, len(a["qd"]) + 1, dtype=float
    )


# Remove continuation records from Batch 2.
batch2 = [
    x for x in batch2
    if x["cell_id"] not in BATCH2_CONTINUATIONS
]


# -------------------------
# Official Batch-3 cleaning
# -------------------------

removed_b3_channel = []

# MATLAB: batch3(38) = []
# => zero-based original position 37.
removed_b3_channel.append(batch3[37]["cell_id"])
batch3 = [
    x for i, x in enumerate(batch3)
    if i != 37
]


# MATLAB:
# endcap3(i) = QDischarge(end)
# rind = find(endcap3 > 0.885)
# batch3(rind) = []
removed_b3_endcap = [
    x["cell_id"]
    for x in batch3
    if x["qd"][-1] > 0.885
]

batch3 = [
    x for x in batch3
    if x["qd"][-1] <= 0.885
]


# MATLAB: nind = [3,40:41]
# Applied AFTER prior deletions; convert to zero-based current positions.
noise_positions = {2, 39, 40}

removed_b3_noise = [
    x["cell_id"]
    for i, x in enumerate(batch3)
    if i in noise_positions
]

batch3 = [
    x for i, x in enumerate(batch3)
    if i not in noise_positions
]


# Combine.
severson_before_optional = batch1 + batch2 + batch3

# Optional paper-modeling exclusion:
severson = [
    x for x in severson_before_optional
    if x["cell_id"] not in BATCH1_NONFINISH
]


# =====================================================================
# 2. Audit official-paper labels
# =====================================================================

sev_rows = []

for x in severson:

    qd = x["qd"]

    paper_label = paper_loadData_label(qd)

    stored = x["source_cycle_life"]

    adjusted = stored

    if x["cell_id"] in CONTINUATIONS:
        adjusted = (
            stored
            + CONTINUATIONS[x["cell_id"]][1]
        )

    sev_rows.append({
        "cell_id": x["cell_id"],
        "batch": x["batch"],

        "record_length":
            len(qd),

        "first_QD":
            float(qd[0]),

        "last_QD":
            float(qd[-1]),

        "min_QD":
            float(np.nanmin(qd)),

        "source_cycle_life_raw":
            stored,

        "batteryml_adjusted_source_cycle_life":
            adjusted,

        "official_LoadData_bat_label":
            paper_label,

        "official_minus_adjusted_source":
            (
                paper_label - adjusted
                if np.isfinite(adjusted)
                else np.nan
            ),

        "uniform_first_valid_cross":
            first_cross(qd),

        "uniform_first_3_consecutive":
            sustained_cross(qd, 3),

        "uniform_first_5_consecutive":
            sustained_cross(qd, 5),

        "official_minus_uniform_first_cross":
            (
                paper_label - first_cross(qd)
                if np.isfinite(first_cross(qd))
                else np.nan
            ),

        "record_minus_official_label":
            len(qd) - paper_label,
    })


sev_df = pd.DataFrame(sev_rows)


# =====================================================================
# 3. Batch-4 / Attia collection audit
# =====================================================================

b4_rows = []

for x in batches[4]:

    qd = x["qd"]

    reconstructed = cell_analysis_reconstruction(qd)
    stored = x["source_cycle_life"]

    b4_rows.append({
        "cell_id": x["cell_id"],

        "record_length":
            len(qd),

        "first_QD":
            float(qd[0]),

        "last_QD":
            float(qd[-1]),

        "min_QD":
            float(np.nanmin(qd)),

        "source_cycle_life":
            stored,

        "cell_analysis_reconstructed_cycle_life":
            reconstructed,

        "source_minus_reconstructed":
            (
                stored - reconstructed
                if (
                    np.isfinite(stored)
                    and np.isfinite(reconstructed)
                )
                else np.nan
            ),

        "uniform_first_valid_cross":
            first_cross(qd),

        "uniform_first_3_consecutive":
            sustained_cross(qd, 3),

        "uniform_first_5_consecutive":
            sustained_cross(qd, 5),

        "record_minus_source_cycle_life":
            (
                len(qd) - stored
                if np.isfinite(stored)
                else np.nan
            ),
    })


b4_df = pd.DataFrame(b4_rows)


# =====================================================================
# Save
# =====================================================================

sev_file = OUT / "MATR_Severson_official_label_audit.csv"
b4_file = OUT / "MATR_Batch4_label_audit.csv"
summary_file = OUT / "MATR_label_semantics_summary.txt"

sev_df.to_csv(sev_file, index=False)
b4_df.to_csv(b4_file, index=False)


lines = []

def emit(x=""):
    print(x)
    lines.append(str(x))


emit("=" * 100)
emit("MATR LABEL-SEMANTICS AUDIT")
emit("=" * 100)

emit()
emit("RAW COUNTS")
for b in range(1, 5):
    emit(f"Batch {b}: {len(batches[b])}")

emit()
emit("OFFICIAL SEVERSON LOADData.m CLEANING")

emit(
    "Batch-3 removed channel-problem cell: "
    + ", ".join(removed_b3_channel)
)

emit(
    "Batch-3 removed because terminal QD > 0.885: "
    + ", ".join(removed_b3_endcap)
)

emit(
    "Batch-3 removed noisy cells after prior deletions: "
    + ", ".join(removed_b3_noise)
)

emit(
    "Batch-1 optional non-finish removals: "
    + ", ".join(sorted(BATCH1_NONFINISH))
)

emit()
emit(
    "Severson cells before optional Batch-1 removal: "
    f"{len(severson_before_optional)}"
)

emit(
    "Final Severson paper-modeling cohort: "
    f"{len(severson)}"
)

emit(
    "Batch-4 / Attia cells: "
    f"{len(batches[4])}"
)

emit(
    "Source-QC Severson + Batch4 total: "
    f"{len(severson) + len(batches[4])}"
)

emit()
emit("SEVERSON: OFFICIAL bat_label VS ADJUSTED SOURCE cycle_life")

finite_compare = sev_df[
    np.isfinite(
        sev_df["batteryml_adjusted_source_cycle_life"]
    )
]

exact = np.isclose(
    finite_compare["official_LoadData_bat_label"],
    finite_compare["batteryml_adjusted_source_cycle_life"],
    equal_nan=False
)

emit(
    f"Exact matches: {int(exact.sum())}/{len(finite_compare)}"
)

mismatch = finite_compare.loc[
    ~exact,
    [
        "cell_id",
        "record_length",
        "last_QD",
        "batteryml_adjusted_source_cycle_life",
        "official_LoadData_bat_label",
        "official_minus_adjusted_source",
    ]
]

if len(mismatch):
    emit()
    emit("Mismatches:")
    emit(mismatch.to_string(index=False))
else:
    emit("No mismatches.")

emit()
emit("SEVERSON: OFFICIAL LABEL RELATIVE TO RECORD END")
emit(
    sev_df["record_minus_official_label"]
    .describe()
    .to_string()
)

emit()
emit("SEVERSON: UNIFORM FIRST VALID <0.88 CROSSING")

cross = sev_df[
    np.isfinite(
        sev_df["uniform_first_valid_cross"]
    )
]

emit(
    f"Cells with observed valid <0.88 crossing: "
    f"{len(cross)}/{len(sev_df)}"
)

emit(
    "Official label - uniform first crossing:"
)

emit(
    cross["official_minus_uniform_first_cross"]
    .describe()
    .to_string()
)

emit()
match3 = int(
    np.isclose(
        cross["uniform_first_valid_cross"],
        cross["uniform_first_3_consecutive"],
        equal_nan=False
    ).sum()
)

match5 = int(
    np.isclose(
        cross["uniform_first_valid_cross"],
        cross["uniform_first_5_consecutive"],
        equal_nan=False
    ).sum()
)

emit(
    f"First-cross == 3-consecutive: "
    f"{match3}/{len(cross)}"
)

emit(
    f"First-cross == 5-consecutive: "
    f"{match5}/{len(cross)}"
)

emit()
emit("BATCH-4: STORED cycle_life VS cell_analysis.m RECONSTRUCTION")

finite_b4 = b4_df[
    np.isfinite(b4_df["source_cycle_life"])
    & np.isfinite(
        b4_df["cell_analysis_reconstructed_cycle_life"]
    )
]

exact_b4 = np.isclose(
    finite_b4["source_cycle_life"],
    finite_b4["cell_analysis_reconstructed_cycle_life"],
    equal_nan=False
)

emit(
    f"Exact matches: {int(exact_b4.sum())}/{len(finite_b4)}"
)

b4_mismatch = finite_b4.loc[
    ~exact_b4,
    [
        "cell_id",
        "record_length",
        "last_QD",
        "source_cycle_life",
        "cell_analysis_reconstructed_cycle_life",
        "source_minus_reconstructed",
        "uniform_first_valid_cross",
        "uniform_first_3_consecutive",
        "uniform_first_5_consecutive",
    ]
]

if len(b4_mismatch):
    emit()
    emit("Batch-4 mismatches:")
    emit(b4_mismatch.to_string(index=False))
else:
    emit("No Batch-4 mismatches.")

emit()
emit("BATCH-4: RECORD END MINUS SOURCE cycle_life")
emit(
    b4_df["record_minus_source_cycle_life"]
    .describe()
    .to_string()
)

emit()
emit("FILES")
emit(str(sev_file))
emit(str(b4_file))
emit(str(summary_file))

summary_file.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8"
)
