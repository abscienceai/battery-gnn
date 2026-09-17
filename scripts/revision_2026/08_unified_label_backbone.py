#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import os
import subprocess

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat


BASE = Path(os.environ["BASE"])
REPO = Path(os.environ["REPO"])
OFF = Path(os.environ["OFF"])

AUDIT = BASE / "data/audit"
RAW = BASE / "data/raw"
OUTDIR = BASE / "data/processed/revision_2026"

OUT_CSV = OUTDIR / "unified_label_backbone.csv"
OUT_MANIFEST = OUTDIR / "unified_label_backbone_manifest.json"
OUT_SUMMARY = OUTDIR / "unified_label_backbone_summary.txt"

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# General helpers
# ============================================================

def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            b = f.read(chunk)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def aggregate_source_sha256(paths):
    """
    Content-address a set of source files while also binding
    each digest to its path relative to BASE.
    """
    h = hashlib.sha256()

    paths = sorted(
        [Path(x).resolve() for x in paths],
        key=lambda x: str(x)
    )

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

        try:
            name = str(path.relative_to(BASE))
        except ValueError:
            name = str(path)

        digest = sha256_file(path)

        h.update(
            f"{name}\t{digest}\n".encode("utf-8")
        )

    return h.hexdigest()


def git_head(path):
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(path),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


def git_status_porcelain(path):
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(path),
            "status",
            "--porcelain",
        ],
        text=True,
    ).strip()


def base_record(
    dataset,
    cohort,
    cell_id,
    observation_index,
    native_cycle,
    life_coordinate,
    life_coordinate_name,
    capacity_ah,
    soh,
    eol_index,
    rul,
    native_rul,
    target_source,
    eol_basis,
    valid_capacity,
    source_file,
    source_lifetime_label=None,
    censoring_status=None,
):
    if censoring_status is None:
        censoring_status = (
            "observed_event"
            if eol_index is not None
            else "unknown"
        )

    return {
        "dataset":
            dataset,

        "cohort":
            cohort,

        "cell_id":
            str(cell_id),

        "dataset_cell_key":
            f"{dataset}::{cell_id}",

        "split_group":
            f"{dataset}::{cell_id}",

        "observation_index":
            int(observation_index),

        "native_cycle":
            (
                float(native_cycle)
                if np.isfinite(native_cycle)
                else np.nan
            ),

        "life_coordinate":
            (
                float(life_coordinate)
                if np.isfinite(life_coordinate)
                else np.nan
            ),

        "life_coordinate_name":
            life_coordinate_name,

        "capacity_Ah":
            (
                float(capacity_ah)
                if np.isfinite(capacity_ah)
                else np.nan
            ),

        "SOH":
            (
                float(soh)
                if np.isfinite(soh)
                else np.nan
            ),

        "eol_observation_index":
            (
                int(eol_index)
                if eol_index is not None
                else pd.NA
            ),

        "cell_reaches_eol":
            bool(
                eol_index is not None
            ),

        "is_eol_observation":
            bool(
                eol_index is not None
                and observation_index == eol_index
            ),

        "is_at_or_before_eol":
            bool(
                eol_index is not None
                and observation_index <= eol_index
            ),

        "RUL_observations":
            (
                float(rul)
                if rul is not None
                and np.isfinite(rul)
                else np.nan
            ),

        "native_RUL":
            (
                float(native_rul)
                if native_rul is not None
                and np.isfinite(native_rul)
                else np.nan
            ),

        "target_source":
            target_source,

        "eol_basis":
            eol_basis,

        "capacity_is_observed":
            True,

        "SOH_is_derived":
            True,

        "RUL_is_derived":
            bool(eol_index is not None),

        "source_lifetime_label":
            (
                float(source_lifetime_label)
                if source_lifetime_label is not None
                and np.isfinite(source_lifetime_label)
                else np.nan
            ),

        "censoring_status":
            censoring_status,

        "valid_capacity":
            bool(valid_capacity),

        "source_file":
            source_file,
    }


def rel(path):
    try:
        return str(
            path.relative_to(BASE)
        )
    except Exception:
        return str(path)


# ============================================================
# HUST
#
# Dataset-native RUL is N,...,1.
# Canonical zero-at-EOL RUL = native_RUL - 1.
# ============================================================

def build_hust():

    path = AUDIT / "HUST_cycle_labels.csv"

    df = pd.read_csv(path)

    required = {
        "cell_id",
        "cycle",
        "capacity_Ah_if_dq_is_mAh",
        "SOH_vs_1p10_nominal",
        "rul_source",
    }

    assert required.issubset(
        df.columns
    )

    assert len(df) == 146122
    assert df["cell_id"].nunique() == 77

    rows = []

    for cell_id, g in df.groupby(
        "cell_id",
        sort=True
    ):

        g = (
            g.sort_values(
                "cycle"
            )
            .reset_index(drop=True)
        )

        native_rul = pd.to_numeric(
            g["rul_source"],
            errors="coerce"
        ).to_numpy(
            dtype=float
        )

        assert np.isfinite(
            native_rul
        ).all()

        # Previous source audit established N,...,1.
        assert np.isclose(
            native_rul[-1],
            1.0
        )

        eol_index = len(g) - 1

        for i, r in g.iterrows():

            q = float(
                r[
                    "capacity_Ah_if_dq_is_mAh"
                ]
            )

            soh = float(
                r[
                    "SOH_vs_1p10_nominal"
                ]
            )

            nrul = float(
                r["rul_source"]
            )

            crul = nrul - 1.0

            assert crul >= 0

            rows.append(
                base_record(
                    dataset="HUST",
                    cohort="HUST_1.1Ah_LFP",
                    cell_id=cell_id,
                    observation_index=i,
                    native_cycle=float(
                        r["cycle"]
                    ),
                    life_coordinate=float(
                        r["cycle"]
                    ),
                    life_coordinate_name=
                        "source_cycle",
                    capacity_ah=q,
                    soh=soh,
                    eol_index=eol_index,
                    rul=crul,
                    native_rul=nrul,
                    target_source=
                        "HUST source discharge-capacity label "
                        "converted from mAh to Ah",
                    eol_basis=
                        "dataset-native source RUL terminus",
                    valid_capacity=(
                        np.isfinite(q)
                        and q > 0.1
                    ),
                    source_file=rel(path),
                )
            )

    out = pd.DataFrame(rows)

    assert len(out) == 146122
    assert out["cell_id"].nunique() == 77

    assert (
        out.groupby(
            "cell_id"
        )["is_eol_observation"]
        .sum()
        .eq(1)
        .all()
    )

    return out


# ============================================================
# CALCE
#
# Use only BatteryML-final-clean cycles.
# Capacity = integrated discharge capacity.
# SOH = Qd / cell-specific correct nominal capacity.
# EOL = first clean SOH <= 0.80.
# Post-EOL observations remain in provenance backbone but their
# canonical RUL is NaN.
# ============================================================

def build_calce():

    files = sorted(
        AUDIT.glob(
            "C[XS]2_*_cycles_audit.csv"
        )
    )

    assert len(files) == 13

    rows = []

    for path in files:

        cell_id = path.name.replace(
            "_cycles_audit.csv",
            ""
        )

        df = pd.read_csv(path)

        required = {
            "batteryml_cycle_number_zero_based",
            "Qd_integrated_Ah",
            "keep_final_batteryml",
            "nominal_capacity_correct_Ah",
            "SOH_correct",
        }

        assert required.issubset(
            df.columns
        )

        clean = df[
            df[
                "keep_final_batteryml"
            ].astype(bool)
        ].copy()

        clean = (
            clean.sort_values(
                "batteryml_cycle_number_zero_based"
            )
            .reset_index(drop=True)
        )

        assert len(clean) > 0

        soh = pd.to_numeric(
            clean["SOH_correct"],
            errors="coerce"
        ).to_numpy(
            dtype=float
        )

        cross = np.where(
            np.isfinite(soh)
            & (soh <= 0.80)
        )[0]

        assert len(cross) > 0

        eol_index = int(
            cross[0]
        )

        for i, r in clean.iterrows():

            q = float(
                r[
                    "Qd_integrated_Ah"
                ]
            )

            s = float(
                r["SOH_correct"]
            )

            if i <= eol_index:
                rul = (
                    eol_index - i
                )
            else:
                rul = np.nan

            rows.append(
                base_record(
                    dataset="CALCE",
                    cohort="CALCE_CS_CX",
                    cell_id=cell_id,
                    observation_index=i,
                    native_cycle=float(
                        r[
                            "batteryml_cycle_number_zero_based"
                        ]
                    ),
                    life_coordinate=float(
                        r[
                            "batteryml_cycle_number_zero_based"
                        ]
                    ),
                    life_coordinate_name=
                        "batteryml_clean_cycle_index",
                    capacity_ah=q,
                    soh=s,
                    eol_index=eol_index,
                    rul=rul,
                    native_rul=None,
                    target_source=
                        "CALCE raw current integration after "
                        "validated BatteryML-compatible cycle QC",
                    eol_basis=
                        "first clean SOH <= 0.80 using "
                        "cell-specific nominal capacity",
                    valid_capacity=(
                        np.isfinite(q)
                        and q > 0.1
                    ),
                    source_file=rel(path),
                )
            )

    out = pd.DataFrame(rows)

    assert out["cell_id"].nunique() == 13

    assert (
        out.groupby(
            "cell_id"
        )["is_eol_observation"]
        .sum()
        .eq(1)
        .all()
    )

    return out


# ============================================================
# MATR raw HDF5 loader
# ============================================================

MATR_FILES = {
    1: RAW / "MATR/MATR_batch_20170512.mat",
    2: RAW / "MATR/MATR_batch_20170630.mat",
    3: RAW / "MATR/MATR_batch_20180412.mat",
    4: RAW / "MATR/MATR_batch_20190124.mat",
}

CONTINUATIONS = {
    "b1c0": ("b2c7", 662),
    "b1c1": ("b2c8", 981),
    "b1c2": ("b2c9", 1060),
    "b1c3": ("b2c15", 208),
    "b1c4": ("b2c16", 482),
}


def matr_get_vector(group, name):
    return np.hstack(
        group[name][0, :].tolist()
    ).astype(float)


def matr_get_scalar(f, ref):
    x = np.asarray(
        f[ref][:]
    ).squeeze()

    try:
        return float(x)
    except Exception:
        return np.nan


def matr_load_batch(batch_number):

    path = MATR_FILES[
        batch_number
    ]

    result = {}

    with h5py.File(
        path,
        "r"
    ) as f:

        b = f["batch"]

        n = int(
            b["summary"].shape[0]
        )

        for i in range(n):

            key = (
                f"b{batch_number}c{i}"
            )

            summary = f[
                b["summary"][i, 0]
            ]

            qd = matr_get_vector(
                summary,
                "QDischarge"
            )

            cyc = matr_get_vector(
                summary,
                "cycle"
            )

            source_cl = matr_get_scalar(
                f,
                b["cycle_life"][i, 0]
            )

            # One source-file anomaly is explicitly documented:
            # Batch-4 cell b4c31 contains 834 QDischarge observations
            # but only the contiguous native cycle vector 1..833.
            # The audited summary length is 834 and the final capacity
            # value is a nonphysical zero placeholder. Preserve the raw
            # capacity observation for provenance, but do not invent a
            # native cycle number; assign NaN to that unmatched position.
            if len(qd) != len(cyc):
                if (
                    key == "b4c31"
                    and len(qd) == 834
                    and len(cyc) == 833
                    and np.array_equal(
                        cyc,
                        np.arange(1.0, 834.0)
                    )
                    and np.isclose(
                        qd[-1],
                        0.0,
                        atol=1e-12
                    )
                ):
                    cyc = np.concatenate(
                        [
                            cyc,
                            np.asarray(
                                [np.nan],
                                dtype=float
                            ),
                        ]
                    )
                else:
                    raise RuntimeError(
                        f"{key}: unexpected MATR QDischarge/cycle "
                        f"length mismatch: qd={len(qd)}, "
                        f"cycle={len(cyc)}"
                    )

            assert len(qd) == len(cyc)

            result[key] = {
                "qd":
                    qd,

                "cycle":
                    cyc,

                "source_cycle_life":
                    source_cl,

                "source_file":
                    path,
            }

    return result


def choose_continuation_merge(
    primary,
    continuation,
    expected_length,
):
    """
    Select only a merge whose length matches the previously
    validated MATR audit record_length exactly.

    We do not silently guess.  If no supported merge gives the
    audited length, the generator stops.
    """

    p_q = primary["qd"]
    p_c = primary["cycle"]

    c_q = continuation["qd"]
    c_c = continuation["cycle"]

    candidates = [
        (
            "full_append",
            np.concatenate(
                [p_q, c_q]
            ),
            np.concatenate(
                [p_c, c_c]
            ),
        ),
        (
            "drop_first_continuation",
            np.concatenate(
                [p_q, c_q[1:]]
            ),
            np.concatenate(
                [p_c, c_c[1:]]
            ),
        ),
        (
            "drop_last_primary",
            np.concatenate(
                [p_q[:-1], c_q]
            ),
            np.concatenate(
                [p_c[:-1], c_c]
            ),
        ),
    ]

    matches = [
        x for x in candidates
        if len(x[1])
        == int(expected_length)
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "MATR continuation merge is not uniquely "
            f"resolved: expected_length={expected_length}, "
            f"candidate_lengths="
            f"{[(x[0], len(x[1])) for x in candidates]}"
        )

    return matches[0]


# ============================================================
# MATR
#
# Severson cohort:
#   use official LoadData label audited for 124 cells.
#
# Batch4:
#   use source cycle_life audited for 45 cells.
#
# The model-eligible capacity flag excludes placeholder QD <=0.1,
# but source observation positions are retained for provenance.
# ============================================================

def build_matr():

    sev_path = (
        AUDIT
        / "MATR_Severson_official_label_audit.csv"
    )

    b4_path = (
        AUDIT
        / "MATR_Batch4_label_audit.csv"
    )

    sev = pd.read_csv(
        sev_path
    )

    b4 = pd.read_csv(
        b4_path
    )

    assert len(sev) == 124
    assert len(b4) == 45

    batches = {}

    for b in [1, 2, 3, 4]:
        batches.update(
            matr_load_batch(b)
        )

    rows = []

    continuation_modes = {}

    # --------------------------------------------------------
    # Severson official 124-cell cohort
    # --------------------------------------------------------

    for _, info in sev.iterrows():

        cell_id = str(
            info["cell_id"]
        )

        assert cell_id in batches

        expected_length = int(
            info["record_length"]
        )

        official_label = float(
            info[
                "official_LoadData_bat_label"
            ]
        )

        assert np.isfinite(
            official_label
        )

        primary = batches[
            cell_id
        ]

        qd = primary[
            "qd"
        ].copy()

        cyc = primary[
            "cycle"
        ].copy()

        source_files = [
            rel(
                primary[
                    "source_file"
                ]
            )
        ]

        if cell_id in CONTINUATIONS:

            continuation_id, _ = (
                CONTINUATIONS[
                    cell_id
                ]
            )

            assert (
                continuation_id
                in batches
            )

            mode, qd, cyc = (
                choose_continuation_merge(
                    primary,
                    batches[
                        continuation_id
                    ],
                    expected_length,
                )
            )

            continuation_modes[
                cell_id
            ] = {
                "continuation":
                    continuation_id,
                "merge_mode":
                    mode,
            }

            source_files.append(
                rel(
                    batches[
                        continuation_id
                    ][
                        "source_file"
                    ]
                )
            )

        assert len(qd) == (
            expected_length
        )

        assert len(cyc) == (
            expected_length
        )

        # Severson LoadData.m has two distinct semantics:
        #
        #   label <= record length:
        #       observed threshold event within the record
        #
        #   label == record length + 1:
        #       threshold event was NOT observed; right-censored
        #
        # Never clamp a len+1 label onto the final observation.
        if np.isclose(
            official_label,
            len(qd) + 1
        ):
            eol_index = None
            censoring_status = (
                "right_censored_len_plus_1"
            )

        elif (
            official_label >= 1
            and official_label <= len(qd)
        ):
            eol_index = (
                int(round(
                    official_label
                ))
                - 1
            )

            censoring_status = (
                "observed_event"
            )

        else:
            raise RuntimeError(
                f"{cell_id}: unexpected official lifetime "
                f"label={official_label}, "
                f"record_length={len(qd)}"
            )

        first_valid = np.where(
            np.isfinite(qd)
            & (qd > 0.1)
        )[0]

        assert len(
            first_valid
        ) > 0

        q0 = float(
            qd[
                first_valid[0]
            ]
        )

        for i in range(
            len(qd)
        ):

            q = float(
                qd[i]
            )

            s = (
                q / q0
                if np.isfinite(q)
                else np.nan
            )

            rul = (
                eol_index - i
                if (
                    eol_index is not None
                    and i <= eol_index
                )
                else np.nan
            )

            native_cycle = (
                float(cyc[i])
                if np.isfinite(cyc[i])
                else np.nan
            )

            rows.append(
                base_record(
                    dataset="MATR",
                    cohort=
                        "Severson_Batches1_3_official124",
                    cell_id=cell_id,
                    observation_index=i,
                    native_cycle=native_cycle,
                    life_coordinate=float(
                        i + 1
                    ),
                    life_coordinate_name=
                        "official_cycle_count_reconstruction",
                    capacity_ah=q,
                    soh=s,
                    eol_index=eol_index,
                    rul=rul,
                    native_rul=None,
                    target_source=
                        "MATR summary.QDischarge with "
                        "official LoadData cycle-life label",
                    eol_basis=
                        "official Severson LoadData "
                        "cycle-life label",
                    valid_capacity=(
                        np.isfinite(q)
                        and q > 0.1
                    ),
                    source_file=";".join(
                        source_files
                    ),
                    source_lifetime_label=
                        official_label,
                    censoring_status=
                        censoring_status,
                )
            )

    # --------------------------------------------------------
    # Batch4 / Attia cohort
    # --------------------------------------------------------

    for _, info in b4.iterrows():

        cell_id = str(
            info["cell_id"]
        )

        assert cell_id in batches

        expected_length = int(
            info["record_length"]
        )

        source_label = float(
            info["source_cycle_life"]
        )

        assert np.isfinite(
            source_label
        )

        src = batches[
            cell_id
        ]

        qd = src[
            "qd"
        ].copy()

        cyc = src[
            "cycle"
        ].copy()

        assert len(qd) == (
            expected_length
        )

        assert len(cyc) == (
            expected_length
        )

        eol_index = (
            int(round(
                source_label
            ))
            - 1
        )

        assert (
            0
            <= eol_index
            < len(qd)
        )

        first_valid = np.where(
            np.isfinite(qd)
            & (qd > 0.1)
        )[0]

        assert len(
            first_valid
        ) > 0

        q0 = float(
            qd[
                first_valid[0]
            ]
        )

        for i in range(
            len(qd)
        ):

            q = float(
                qd[i]
            )

            s = (
                q / q0
                if np.isfinite(q)
                else np.nan
            )

            rul = (
                eol_index - i
                if i <= eol_index
                else np.nan
            )

            rows.append(
                base_record(
                    dataset="MATR",
                    cohort=
                        "Attia_Batch4_official45",
                    cell_id=cell_id,
                    observation_index=i,
                    native_cycle=(
                        float(cyc[i])
                        if np.isfinite(
                            cyc[i]
                        )
                        else np.nan
                    ),
                    life_coordinate=float(
                        i + 1
                    ),
                    life_coordinate_name=
                        "source_cycle_count",
                    capacity_ah=q,
                    soh=s,
                    eol_index=eol_index,
                    rul=rul,
                    native_rul=None,
                    target_source=
                        "MATR Batch4 summary.QDischarge "
                        "with source cycle_life label",
                    eol_basis=
                        "Batch4 source cycle_life label",
                    valid_capacity=(
                        np.isfinite(q)
                        and q > 0.1
                    ),
                    source_file=rel(
                        src[
                            "source_file"
                        ]
                    ),
                    source_lifetime_label=
                        source_label,
                    censoring_status=
                        "observed_event",
                )
            )

    out = pd.DataFrame(
        rows
    )

    assert (
        out["cell_id"].nunique()
        == 169
    )

    # MATR contains both observed threshold events and
    # right-censored lifetimes.  The audited cohort has:
    #   88 observed-event cells
    #   81 right-censored cells
    #
    # Therefore only observed-event cells may contain an explicit
    # EOL observation and canonical zero-at-EOL RUL.
    per_cell = (
        out.groupby(
            "cell_id",
            sort=False
        )
        .agg(
            eol_observations=(
                "is_eol_observation",
                "sum"
            ),
            reaches_eol=(
                "cell_reaches_eol",
                "max"
            ),
            finite_rul=(
                "RUL_observations",
                lambda x: int(
                    x.notna().sum()
                )
            ),
        )
    )

    observed = per_cell[
        per_cell[
            "reaches_eol"
        ].astype(bool)
    ]

    censored = per_cell[
        ~per_cell[
            "reaches_eol"
        ].astype(bool)
    ]

    assert len(observed) == 88
    assert len(censored) == 81

    assert (
        observed[
            "eol_observations"
        ].eq(1)
        .all()
    )

    assert (
        censored[
            "eol_observations"
        ].eq(0)
        .all()
    )

    assert (
        censored[
            "finite_rul"
        ].eq(0)
        .all()
    )

    observed_ids = set(
        out.loc[
            out[
                "censoring_status"
            ] == "observed_event",
            "cell_id",
        ].astype(str)
    )

    censored_ids = set(
        out.loc[
            out[
                "censoring_status"
            ] == "right_censored_len_plus_1",
            "cell_id",
        ].astype(str)
    )

    assert len(observed_ids) == 88
    assert len(censored_ids) == 81
    assert observed_ids.isdisjoint(
        censored_ids
    )

    return (
        out,
        continuation_modes
    )


# ============================================================
# RWTH
#
# Official processed characterization capacity:
#   CapDCH1 [Ah]
#
# SOH:
#   CapDCH1 / first valid CapDCH1 of same raw cell
#
# EOL:
#   first characterization SOH <= 0.80
#
# Life coordinate:
#   cumulative AhEla / 1.85 Ah
# ============================================================

def scalar(v):
    try:
        x = np.asarray(
            v
        ).squeeze()

        if np.size(x) != 1:
            return np.nan

        return float(x)

    except Exception:
        return np.nan


def build_rwth():

    p001 = (
        OFF
        / "Processed Experimental Data"
        / "P001"
    )

    rows = []

    for raw_id in range(
        2,
        50
    ):

        cell_id = (
            f"epSanyo{raw_id:03d}"
        )

        folder = (
            p001 / cell_id
        )

        records = []

        for td in folder.iterdir():

            if not td.is_dir():
                continue

            path = (
                td / "lean_data.mat"
            )

            if not path.exists():
                continue

            d = loadmat(
                path,
                simplify_cells=True,
                variable_names=[
                    "StartDate",
                    "AhEla",
                    "CapDCH1",
                ],
            )

            records.append({
                "testset":
                    td.name,

                "start":
                    scalar(
                        d.get(
                            "StartDate",
                            np.nan
                        )
                    ),

                "AhEla":
                    scalar(
                        d.get(
                            "AhEla",
                            np.nan
                        )
                    ),

                "CapDCH1":
                    scalar(
                        d.get(
                            "CapDCH1",
                            np.nan
                        )
                    ),

                "source":
                    path,
            })

        records.sort(
            key=lambda r: (
                r["start"]
                if np.isfinite(
                    r["start"]
                )
                else np.inf,
                r["testset"],
            )
        )

        cumulative_ah = 0.0

        obs = []

        for r in records:

            if np.isfinite(
                r["AhEla"]
            ):
                cumulative_ah += (
                    r["AhEla"]
                )

            if not np.isfinite(
                r["CapDCH1"]
            ):
                continue

            obs.append({
                "capacity":
                    r["CapDCH1"],

                "EFC":
                    cumulative_ah
                    / 1.85,

                "source":
                    r["source"],
            })

        assert len(obs) > 0

        caps = np.asarray(
            [
                x["capacity"]
                for x in obs
            ],
            dtype=float
        )

        q0 = float(
            caps[0]
        )

        soh = (
            caps / q0
        )

        cross = np.where(
            soh <= 0.80
        )[0]

        assert len(cross) > 0

        eol_index = int(
            cross[0]
        )

        for i, x in enumerate(
            obs
        ):

            rul = (
                eol_index - i
                if i <= eol_index
                else np.nan
            )

            rows.append(
                base_record(
                    dataset="RWTH",
                    cohort=
                        "RWTH_epSanyo002_049",
                    cell_id=cell_id,
                    observation_index=i,
                    native_cycle=np.nan,
                    life_coordinate=float(
                        x["EFC"]
                    ),
                    life_coordinate_name=
                        "cumulative_AhEla_div_1p85_Ah",
                    capacity_ah=float(
                        x["capacity"]
                    ),
                    soh=float(
                        soh[i]
                    ),
                    eol_index=eol_index,
                    rul=rul,
                    native_rul=None,
                    target_source=
                        "official processed CapDCH1 "
                        "characterization capacity",
                    eol_basis=
                        "first characterization "
                        "SOH <= 0.80 relative to Q0",
                    valid_capacity=True,
                    source_file=rel(
                        x["source"]
                    ),
                )
            )

    out = pd.DataFrame(
        rows
    )

    assert len(out) == 777

    assert (
        out["cell_id"].nunique()
        == 48
    )

    assert (
        out.groupby(
            "cell_id"
        )["is_eol_observation"]
        .sum()
        .eq(1)
        .all()
    )

    return out


# ============================================================
# Build unified label/provenance backbone
# ============================================================

hust = build_hust()
calce = build_calce()

matr, matr_continuations = (
    build_matr()
)

rwth = build_rwth()

frames = [
    hust,
    calce,
    matr,
    rwth,
]

unified = pd.concat(
    frames,
    ignore_index=True
)


# ============================================================
# Global structural assertions
# ============================================================

expected_datasets = {
    "HUST",
    "CALCE",
    "MATR",
    "RWTH",
}

assert (
    set(
        unified[
            "dataset"
        ].unique()
    )
    == expected_datasets
)

assert (
    unified[
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
    unified[
        "split_group"
    ]
    == unified[
        "dataset_cell_key"
    ]
).all()

assert (
    unified[
        "capacity_is_observed"
    ].all()
)

assert (
    unified[
        "SOH_is_derived"
    ].all()
)

assert (
    unified[
        "RUL_is_derived"
    ].astype(bool)
    ==
    unified[
        "cell_reaches_eol"
    ].astype(bool)
).all()

# This artifact intentionally contains no model feature matrix,
# no split assignment, and no scaling.
for forbidden in [
    "split",
    "train",
    "val",
    "test",
    "scaled",
    "normalized_feature",
]:
    assert forbidden not in {
        c.lower()
        for c in unified.columns
    }


# Nullable integer for EOL position.
unified[
    "eol_observation_index"
] = unified[
    "eol_observation_index"
].astype(
    "Int64"
)

unified.to_csv(
    OUT_CSV,
    index=False
)


# ============================================================
# Manifest
# ============================================================

dataset_summary = {}

for dataset, g in unified.groupby(
    "dataset"
):

    dataset_summary[
        dataset
    ] = {
        "rows":
            int(len(g)),

        "cells":
            int(
                g[
                    "cell_id"
                ].nunique()
            ),

        "valid_capacity_rows":
            int(
                g[
                    "valid_capacity"
                ].sum()
            ),

        "eol_cells":
            int(
                g.groupby(
                    "cell_id"
                )[
                    "cell_reaches_eol"
                ]
                .max()
                .sum()
            ),

        "pre_or_at_eol_rows":
            int(
                g[
                    "is_at_or_before_eol"
                ].sum()
            ),

        "post_eol_rows":
            int(
                (
                    g[
                        "cell_reaches_eol"
                    ]
                    &
                    ~g[
                        "is_at_or_before_eol"
                    ]
                ).sum()
            ),

        "censored_cells":
            int(
                g.groupby(
                    "cell_id"
                )[
                    "cell_reaches_eol"
                ]
                .max()
                .eq(False)
                .sum()
            ),

        "censored_rows":
            int(
                (
                    ~g[
                        "cell_reaches_eol"
                    ]
                ).sum()
            ),
    }


input_files = [
    AUDIT
    / "HUST_cycle_labels.csv",

    AUDIT
    / "MATR_Severson_official_label_audit.csv",

    AUDIT
    / "MATR_Batch4_label_audit.csv",

    AUDIT
    / "RWTH_target_policy.json",
]

input_files += sorted(
    AUDIT.glob(
        "C[XS]2_*_cycles_audit.csv"
    )
)

# MATR is read directly from these four HDF5/MATLAB source files.
input_files += [
    MATR_FILES[b]
    for b in sorted(MATR_FILES)
]

input_hashes = {}

for path in input_files:
    if path.exists():
        input_hashes[
            rel(path)
        ] = sha256_file(
            path
        )


rwth_source_paths = sorted(
    {
        (
            Path(x)
            if Path(x).is_absolute()
            else BASE / Path(x)
        ).resolve()
        for x in rwth["source_file"].astype(str)
    },
    key=lambda x: str(x)
)

assert len(rwth_source_paths) == 777

rwth_source_aggregate_sha256 = (
    aggregate_source_sha256(
        rwth_source_paths
    )
)

rwth_repo_head = git_head(OFF)
rwth_repo_status = git_status_porcelain(OFF)


manifest = {
    "artifact":
        rel(
            OUT_CSV
        ),

    "artifact_sha256":
        sha256_file(
            OUT_CSV
        ),

    "generator":
        rel(
            Path(__file__)
        ),

    "generator_sha256":
        sha256_file(
            Path(__file__)
        ),

    "contains_model_features":
        False,

    "scaler_applied":
        False,

    "split_assignment_applied":
        False,

    "target_feature_separation":
        (
            "This file is label/provenance-only. "
            "Model features must be generated in a "
            "separate artifact and joined only by "
            "(dataset, cell_id, observation_index)."
        ),

    "canonical_RUL_semantics":
        (
            "RUL_observations is zero at the canonical "
            "EOL observation and NaN after EOL."
        ),

    "datasets":
        dataset_summary,

    "matr_continuation_resolution":
        matr_continuations,

    "input_file_sha256":
        input_hashes,

    "direct_source_provenance": {
        "MATR_raw_files_hashed":
            4,

        "RWTH_CapDCH1_source_files_hashed":
            len(rwth_source_paths),

        "RWTH_CapDCH1_source_aggregate_sha256":
            rwth_source_aggregate_sha256,

        "RWTH_external_repository_commit":
            rwth_repo_head,

        "RWTH_external_repository_clean":
            (rwth_repo_status == ""),

        "RWTH_external_repository_status":
            rwth_repo_status,
    },

    "columns":
        list(
            unified.columns
        ),
}

OUT_MANIFEST.write_text(
    json.dumps(
        manifest,
        indent=2
    )
    + "\n",
    encoding="utf-8"
)


# ============================================================
# Human-readable summary
# ============================================================

lines = [
    "Unified label/provenance backbone",
    "",
    f"rows={len(unified)}",
    (
        "cells="
        f"{unified['dataset_cell_key'].nunique()}"
    ),
    "",
]

for dataset in [
    "HUST",
    "CALCE",
    "MATR",
    "RWTH",
]:

    d = dataset_summary[
        dataset
    ]

    lines.append(
        f"{dataset}: "
        f"rows={d['rows']}, "
        f"cells={d['cells']}, "
        f"valid_capacity={d['valid_capacity_rows']}, "
        f"pre_or_at_EOL={d['pre_or_at_eol_rows']}, "
        f"post_EOL={d['post_eol_rows']}, "
        f"censored_cells={d['censored_cells']}"
    )

lines += [
    "",
    "No model features: True",
    "No scaler: True",
    "No train/val/test assignment: True",
    (
        "Canonical join key: "
        "(dataset, cell_id, observation_index)"
    ),
    (
        "Canonical RUL: zero at EOL; "
        "NaN after EOL"
    ),
    "",
    "UNIFIED_LABEL_BACKBONE_PASS=True",
]

OUT_SUMMARY.write_text(
    "\n".join(
        lines
    )
    + "\n",
    encoding="utf-8"
)


# ============================================================
# Compact terminal report
# ============================================================

print("=" * 78)
print("UNIFIED LABEL / PROVENANCE BACKBONE")
print("=" * 78)

print(
    "total rows / cells:",
    len(unified),
    "/",
    unified[
        "dataset_cell_key"
    ].nunique()
)

for dataset in [
    "HUST",
    "CALCE",
    "MATR",
    "RWTH",
]:

    d = dataset_summary[
        dataset
    ]

    print(
        f"{dataset}: "
        f"rows={d['rows']} "
        f"cells={d['cells']} "
        f"valid={d['valid_capacity_rows']} "
        f"pre<=EOL={d['pre_or_at_eol_rows']} "
        f"postEOL={d['post_eol_rows']} "
        f"censored_cells={d['censored_cells']}"
    )

print(
    "MATR continuation cells:",
    len(
        matr_continuations
    )
)

for cell, info in sorted(
    matr_continuations.items()
):
    print(
        f"  {cell} -> "
        f"{info['continuation']} "
        f"[{info['merge_mode']}]"
    )

print(
    "MATR raw files hashed:",
    4
)

print(
    "RWTH CapDCH1 source files hashed:",
    len(rwth_source_paths)
)

print(
    "RWTH source aggregate SHA256:",
    rwth_source_aggregate_sha256
)

print(
    "RWTH source repo HEAD:",
    rwth_repo_head
)

print(
    "RWTH source repo clean:",
    rwth_repo_status == ""
)

print(
    "duplicate canonical keys:",
    int(
        unified[
            [
                "dataset",
                "cell_id",
                "observation_index",
            ]
        ]
        .duplicated()
        .sum()
    )
)

print(
    "contains model features:",
    False
)

print(
    "scaler applied:",
    False
)

print(
    "split assigned:",
    False
)

print()
print(
    "CSV SHA256:",
    manifest[
        "artifact_sha256"
    ]
)

print(
    "CSV:",
    OUT_CSV
)

print(
    "MANIFEST:",
    OUT_MANIFEST
)

print(
    "SUMMARY:",
    OUT_SUMMARY
)

print()
print(
    "UNIFIED_LABEL_BACKBONE_PASS=True"
)
