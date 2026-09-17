from pathlib import Path
from io import BytesIO
import gc
import hashlib
import os
import re
import zipfile

import numpy as np
import pandas as pd


# ============================================================================
# Paths / constants
# ============================================================================

BASE = Path.home() / "alper/battery_revision_2026"

INNER = (
    BASE
    / "data/raw/RWTH/RWTH-2021-04545_818642/Rawdata.zip"
)

AUDIT = Path(
    os.environ.get(
        "RWTH_AUDIT_DIR",
        str(BASE / "data/audit")
    )
).expanduser()

AUDIT.mkdir(
    parents=True,
    exist_ok=True
)

OUT_CYCLES = (
    AUDIT
    / "RWTH_ZYK_cycle_reconstruction_all_cells.csv"
)

OUT_CELLS = (
    AUDIT
    / "RWTH_ZYK_cell_summary.csv"
)

OUT_SOURCES = (
    AUDIT
    / "RWTH_ZYK_source_manifest.csv"
)

OUT_SUMMARY = (
    AUDIT
    / "RWTH_ZYK_dataset_summary.txt"
)

EXPECTED_CELLS = [
    f"{i:03d}"
    for i in range(2, 50)
]

EXPECTED_ARCHIVES = 738

# Verified official non-EIS BOL Part-2 cohort.
EXPECTED_BOL_ARCHIVES = 64

# BatteryML's effective capacity reference for the
# approximately 20--80% SOC cycling window.
WINDOW_REFERENCE_AH = 1.11

# Sensitivity threshold only.
# This is NOT interpreted as 80% of the cell's physical rated capacity.
WINDOW_THRESHOLD_AH = (
    0.80 * WINDOW_REFERENCE_AH
)

RATED_CAPACITY_AH = 2.05

REQUIRED_COLUMNS = {
    "Zeit",
    "Programmdauer",
    "Zustand",
    "Strom",
    "Spannung",
    "AhLad",
    "AhEla",
    "Prozedur",
}

OPTIONAL_COLUMNS = {
    "Schritt",
    "Zyklus",
    "Zyklusebene",
    "Prozedurebene",
    "AhAkku",
    "AhStep",
    "Energie",
    "WhStep",
}


# ============================================================================
# BatteryML-equivalent functions
# ============================================================================

def find_time_anomalies(time, tolerance=1e5):
    """
    Reproduce BatteryML RWTH preprocessing logic.

    RWTH Programmdauer values are milliseconds.
    """

    time = np.asarray(
        time,
        dtype=float
    )

    if len(time) == 0:
        return np.zeros(
            0,
            dtype=bool
        )

    prev = time[0]

    result = np.ones(
        len(time),
        dtype=bool
    )

    for i in range(1, len(time)):

        if time[i] - prev > tolerance:
            result[i] = False

        else:
            prev = time[i]

    return result


def find_cycle_starts_safe(
    current,
    lag=10,
    tolerance=0.1
):
    """
    Boundary-safe reproduction of BatteryML find_cycle_ends().

    Despite its BatteryML name, the original function identifies the
    beginning of a sustained discharge interval after `lag`
    consecutive negative-current samples.

    BatteryML tests `i < len(current)` and subsequently accesses
    current[i+1].  At the final sample that condition is unsafe.
    Here the intended semantics are retained with:
        i < len(current) - 1

    No other segmentation rule is changed.
    """

    current = np.asarray(
        current,
        dtype=float
    )

    is_cycle_start = np.zeros(
        len(current),
        dtype=bool
    )

    enter_discharge_steps = 0
    nms_size = 500

    for i in range(len(current)):

        I = current[i]

        if 0 < i < len(current) - 1:

            if (
                abs(
                    current[i]
                    - current[i - 1]
                ) > tolerance
                and
                abs(
                    current[i]
                    - current[i + 1]
                ) > tolerance
            ):
                I = current[i + 1]

        if I < 0:
            enter_discharge_steps += 1

        else:
            enter_discharge_steps = 0

        if enter_discharge_steps == lag:

            t = i - lag + 1

            if (
                t > nms_size
                and
                np.any(
                    is_cycle_start[
                        t - nms_size:t
                    ]
                )
            ):
                continue

            is_cycle_start[t] = True

    return is_cycle_start


def calc_q(
    current,
    time_ms,
    is_charge
):
    """
    BatteryML-equivalent Coulomb counting.

    A * ms / 3.6e6 = Ah
    """

    current = np.asarray(
        current,
        dtype=float
    )

    time_ms = np.asarray(
        time_ms,
        dtype=float
    )

    q = np.zeros(
        len(current),
        dtype=float
    )

    for i in range(1, len(current)):

        dt = (
            time_ms[i]
            - time_ms[i - 1]
        )

        if (
            is_charge
            and current[i] > 0
        ):
            q[i] = (
                q[i - 1]
                + current[i]
                * dt
                / 3.6e6
            )

        elif (
            (not is_charge)
            and current[i] < 0
        ):
            q[i] = (
                q[i - 1]
                - current[i]
                * dt
                / 3.6e6
            )

        else:
            q[i] = q[i - 1]

    return q


def abnormal_cycle_mask(
    qd,
    eps=0.05,
    window=5
):
    """
    Reproduce BatteryML remove_abnormal_cycle().
    """

    qd = np.asarray(
        qd,
        dtype=float
    )

    remove = np.zeros(
        len(qd),
        dtype=bool
    )

    for i in range(
        window,
        len(qd) - window
    ):

        prev = max(
            0,
            i - window
        )

        if (
            abs(
                qd[i]
                - np.median(
                    qd[prev:i]
                )
            ) > eps
            and
            abs(
                qd[i]
                - np.median(
                    qd[i:i + window]
                )
            ) > eps
        ):
            remove[i] = True

    return remove


# ============================================================================
# Source-file loading
# ============================================================================

def is_cell_temperature_column(name):
    """
    RWTH thermocouple channel is cell dependent:
    Temp1, Temp7, Temp35, Temp116, ...

    Temp0027/28/29/30 auxiliary channels are excluded.
    """

    name = str(name)

    return (
        re.fullmatch(
            r"Temp\d+",
            name
        ) is not None
        and not name.startswith("Temp00")
    )


def temperature_qc_masks(temperature):
    """
    Construct sample-level RWTH temperature QC masks.

    Important:
    - Raw rows are never removed by this function.
    - V/I/time/capacity reconstruction is unchanged.
    - Only the temperature channel is masked for primary
      temperature-feature generation.

    Observed definite-invalid patterns:
      * non-finite values,
      * T <= 0 C reset/missing codes,
      * exact 99.0 C sentinel-like blocks,
      * T > 100 C physically implausible acquisition values.

    Values in 60 < T <= 100 C, except exact 99.0 C, are
    retained. They are flagged separately because the raw
    records contain long, smoothly varying high-temperature
    episodes that must not be discarded by an arbitrary
    60 C clipping rule.
    """

    temperature = pd.to_numeric(
        temperature,
        errors="coerce"
    ).astype(float)

    arr = temperature.to_numpy(
        dtype=float
    )

    finite = np.isfinite(arr)

    nonfinite = ~finite

    nonpositive = (
        finite
        & (arr <= 0.0)
    )

    exact_99 = (
        finite
        & np.isclose(
            arr,
            99.0,
            rtol=0.0,
            atol=1e-12
        )
    )

    gt_100 = (
        finite
        & (arr > 100.0)
    )

    high_60_100_retained = (
        finite
        & (arr > 60.0)
        & (arr <= 100.0)
        & (~exact_99)
    )

    definite_invalid = (
        nonfinite
        | nonpositive
        | exact_99
        | gt_100
    )

    valid = temperature.copy()

    valid.iloc[
        np.flatnonzero(
            definite_invalid
        )
    ] = np.nan

    masks = {
        "nonfinite":
            nonfinite,

        "nonpositive":
            nonpositive,

        "exact_99":
            exact_99,

        "gt_100":
            gt_100,

        "high_60_100_retained":
            high_60_100_retained,

        "definite_invalid":
            definite_invalid,
    }

    return valid, masks


def wanted_column(name):
    name = str(name)

    return (
        name in REQUIRED_COLUMNS
        or name in OPTIONAL_COLUMNS
        or name == "Temp"
        or is_cell_temperature_column(name)
    )


def read_nested_csv(
    outer_zip,
    member,
    cell,
    source_index
):
    """
    Read one official nested TBA_Zyk archive.

    Also returns source provenance for a separate manifest.
    """

    nested_bytes = outer_zip.read(
        member
    )

    nested_sha256 = hashlib.sha256(
        nested_bytes
    ).hexdigest()

    with zipfile.ZipFile(
        BytesIO(nested_bytes),
        "r"
    ) as nested:

        csv_members = [
            x
            for x in nested.namelist()
            if x.lower().endswith(".csv")
        ]

        if len(csv_members) != 1:
            raise RuntimeError(
                f"{member}: expected exactly "
                f"one CSV, got "
                f"{len(csv_members)}"
            )

        csv_name = csv_members[0]

        raw = nested.read(
            csv_name
        )

    df = pd.read_csv(
        BytesIO(raw),
        skiprows=[1],
        encoding="utf-8-sig",
        low_memory=False,
        usecols=wanted_column,
    )

    missing = sorted(
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{member}: missing required "
            f"columns {missing}"
        )

    temp_candidates = [
        c
        for c in df.columns
        if is_cell_temperature_column(c)
    ]

    if len(temp_candidates) == 1:

        temperature_column = (
            temp_candidates[0]
        )

        temperature = pd.to_numeric(
            df[temperature_column],
            errors="coerce"
        )

    elif (
        len(temp_candidates) == 0
        and "Temp" in df.columns
    ):

        temperature_column = "Temp"

        temperature = pd.to_numeric(
            df["Temp"],
            errors="coerce"
        )

    elif len(temp_candidates) == 0:

        temperature_column = None

        temperature = pd.Series(
            np.nan,
            index=df.index,
            dtype=float
        )

    else:
        raise RuntimeError(
            f"{member}: ambiguous cell "
            f"temperature channels: "
            f"{temp_candidates}"
        )

    (
        temperature_valid,
        temperature_masks
    ) = temperature_qc_masks(
        temperature
    )

    # Preserve the selected source channel exactly as parsed.
    df[
        "_temperature_C_raw"
    ] = temperature

    # Primary temperature-feature source after masking only
    # definite-invalid temperature samples. No dataframe rows
    # are removed here.
    df[
        "_temperature_C_valid"
    ] = temperature_valid

    df[
        "_temp_qc_nonfinite"
    ] = temperature_masks[
        "nonfinite"
    ]

    df[
        "_temp_qc_nonpositive"
    ] = temperature_masks[
        "nonpositive"
    ]

    df[
        "_temp_qc_exact99"
    ] = temperature_masks[
        "exact_99"
    ]

    df[
        "_temp_qc_gt100"
    ] = temperature_masks[
        "gt_100"
    ]

    df[
        "_temp_qc_high_60_100_retained"
    ] = temperature_masks[
        "high_60_100_retained"
    ]

    df[
        "_temp_qc_definite_invalid"
    ] = temperature_masks[
        "definite_invalid"
    ]

    # Drop original temperature channels after harmonisation.
    drop_temp = [
        c
        for c in df.columns
        if (
            c == "Temp"
            or is_cell_temperature_column(c)
        )
    ]

    df = df.drop(
        columns=drop_temp,
        errors="ignore"
    )

    df["_source_index"] = (
        source_index
    )

    df["_source_zip"] = (
        Path(member).name
    )

    source_record = {
        "cell_id":
            f"RWTH_{cell}",

        "cell_number":
            cell,

        "source_index":
            source_index,

        "archive_member":
            member,

        "archive_basename":
            Path(member).name,

        "nested_zip_sha256":
            nested_sha256,

        "csv_member":
            csv_name,

        "n_rows":
            len(df),

        "temperature_column":
            (
                temperature_column
                if temperature_column
                is not None
                else ""
            ),

        "Zeit_first":
            (
                str(df["Zeit"].iloc[0])
                if len(df)
                else ""
            ),

        "Zeit_last":
            (
                str(df["Zeit"].iloc[-1])
                if len(df)
                else ""
            ),
    }

    return df, source_record


# ============================================================================
# Helpers
# ============================================================================

def numeric_series(
    df,
    column
):
    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


def safe_stat(
    values,
    func,
    default=np.nan
):
    x = pd.Series(
        values,
        dtype=float
    ).dropna()

    if len(x) == 0:
        return default

    return float(
        func(
            x.to_numpy(
                dtype=float
            )
        )
    )


def first_consecutive_crossing(
    clean,
    threshold,
    consecutive
):
    """
    First start of a run containing `consecutive`
    clean reconstructed cycles at/below threshold.
    """

    qd = clean[
        "Qd_integrated_Ah"
    ].to_numpy(
        dtype=float
    )

    below = (
        np.isfinite(qd)
        & (qd <= threshold)
    )

    run = 0

    for pos, flag in enumerate(below):

        if flag:
            run += 1
        else:
            run = 0

        if run >= consecutive:

            start_pos = (
                pos
                - consecutive
                + 1
            )

            row = clean.iloc[
                start_pos
            ]

            after = (
                len(clean)
                - start_pos
                - 1
            )

            later = clean.iloc[
                start_pos + 1:
            ]

            later_above = int(
                (
                    later[
                        "Qd_integrated_Ah"
                    ] > threshold
                ).sum()
            )

            return {
                "found": True,

                "clean_position":
                    int(
                        row[
                            "cycle_number_after_filter"
                        ]
                    ),

                "pre_filter_cycle":
                    int(
                        row[
                            "cycle_number_pre_filter"
                        ]
                    ),

                "Qd_Ah":
                    float(
                        row[
                            "Qd_integrated_Ah"
                        ]
                    ),

                "cycles_after":
                    int(after),

                "fraction_after":
                    float(
                        after
                        / len(clean)
                    ),

                "later_cycles_above_threshold":
                    later_above,
            }

    return {
        "found": False,
        "clean_position": np.nan,
        "pre_filter_cycle": np.nan,
        "Qd_Ah": np.nan,
        "cycles_after": np.nan,
        "fraction_after": np.nan,
        "later_cycles_above_threshold": np.nan,
    }


# ============================================================================
# One-cell reconstruction
# ============================================================================

def process_cell(
    outer,
    cell,
    members
):
    print()
    print("=" * 110)
    print(
        f"PROCESSING RWTH_{cell} "
        f"({len(members)} TBA_Zyk archives)"
    )
    print("=" * 110)

    frames = []
    source_records = []

    for source_index, member in enumerate(
        members,
        start=1
    ):

        print(
            f"  [{source_index:02d}/"
            f"{len(members):02d}] "
            f"{Path(member).name}",
            flush=True
        )

        part, source_record = (
            read_nested_csv(
                outer,
                member,
                cell,
                source_index
            )
        )

        frames.append(part)
        source_records.append(
            source_record
        )

    df = pd.concat(
        frames,
        ignore_index=True
    )

    del frames
    gc.collect()

    raw_rows = len(df)

    duplicates = int(
        df.duplicated(
            "Zeit"
        ).sum()
    )

    # Match BatteryML:
    # drop_duplicates('Zeit').sort_values('Zeit')
    df = (
        df
        .drop_duplicates("Zeit")
        .sort_values("Zeit")
        .reset_index(drop=True)
    )

    after_duplicates = len(df)

    numeric_cols = [
        "Programmdauer",
        "Strom",
        "Spannung",
        "AhLad",
        "AhEla",
        "_temperature_C_raw",
        "_temperature_C_valid",
    ]

    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    bad_time_before_qc = int(
        (~np.isfinite(
            df[
                "Programmdauer"
            ].to_numpy(dtype=float)
        )).sum()
    )

    bad_current_before_qc = int(
        (~np.isfinite(
            df[
                "Strom"
            ].to_numpy(dtype=float)
        )).sum()
    )

    if (
        bad_time_before_qc > 0
        or bad_current_before_qc > 0
    ):
        raise RuntimeError(
            f"RWTH_{cell}: non-finite "
            f"Programmdauer="
            f"{bad_time_before_qc}, "
            f"Strom="
            f"{bad_current_before_qc}"
        )

    time_ok = find_time_anomalies(
        df[
            "Programmdauer"
        ].to_numpy(
            dtype=float
        )
    )

    time_anomaly_rows = int(
        (~time_ok).sum()
    )

    df = (
        df[
            time_ok
        ]
        .reset_index(
            drop=True
        )
    )

    after_time_qc = len(df)

    temp_raw = df[
        "_temperature_C_raw"
    ]

    temp_valid = df[
        "_temperature_C_valid"
    ]

    temp_raw_nonmissing_n = int(
        temp_raw.notna().sum()
    )

    temp_qc_valid_n = int(
        temp_valid.notna().sum()
    )

    temp_qc_nonfinite_n = int(
        df[
            "_temp_qc_nonfinite"
        ].sum()
    )

    temp_qc_nonpositive_n = int(
        df[
            "_temp_qc_nonpositive"
        ].sum()
    )

    temp_qc_exact99_n = int(
        df[
            "_temp_qc_exact99"
        ].sum()
    )

    temp_qc_gt100_n = int(
        df[
            "_temp_qc_gt100"
        ].sum()
    )

    temp_qc_definite_invalid_n = int(
        df[
            "_temp_qc_definite_invalid"
        ].sum()
    )

    temp_high_60_100_retained_n = int(
        df[
            "_temp_qc_high_60_100_retained"
        ].sum()
    )

    temp_raw_gt_60_n = int(
        (temp_raw > 60).sum()
    )

    temp_raw_gt_100_n = int(
        (temp_raw > 100).sum()
    )

    temp_raw_max = (
        float(
            temp_raw.max()
        )
        if temp_raw.notna().any()
        else np.nan
    )

    temp_valid_max = (
        float(
            temp_valid.max()
        )
        if temp_valid.notna().any()
        else np.nan
    )

    starts_mask = (
        find_cycle_starts_safe(
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
            f"RWTH_{cell}: only "
            f"{len(detected_starts_all)} "
            f"discharge starts detected"
        )

    # BatteryML:
    # cycle_ends =
    # df['Strom'][cycle_ends].index[1:]
    starts = (
        detected_starts_all[
            1:
        ]
    )

    rows = []

    for i in range(
        1,
        len(starts)
    ):

        start = int(
            starts[i - 1]
        )

        end = int(
            starts[i]
        )

        cycle = df.iloc[
            start:end
        ]

        if len(cycle) < 2:
            continue

        I = cycle[
            "Strom"
        ].to_numpy(
            dtype=float
        )

        t = cycle[
            "Programmdauer"
        ].to_numpy(
            dtype=float
        )

        Qd_curve = calc_q(
            I,
            t,
            is_charge=False
        )

        Qc_curve = calc_q(
            I,
            t,
            is_charge=True
        )

        qd = float(
            np.nanmax(
                Qd_curve
            )
        )

        qc = float(
            np.nanmax(
                Qc_curve
            )
        )

        ahela = numeric_series(
            cycle,
            "AhEla"
        )

        native_qd = (
            float(
                ahela.max()
                - ahela.min()
            )
            if ahela.notna().any()
            else np.nan
        )

        voltage = numeric_series(
            cycle,
            "Spannung"
        )

        temperature_raw = cycle[
            "_temperature_C_raw"
        ]

        temperature_valid = cycle[
            "_temperature_C_valid"
        ]

        source_indices = (
            cycle[
                "_source_index"
            ]
            .dropna()
            .astype(int)
            .unique()
        )

        source_files = int(
            len(
                source_indices
            )
        )

        source_first = str(
            cycle[
                "_source_zip"
            ].iloc[0]
        )

        source_last = str(
            cycle[
                "_source_zip"
            ].iloc[-1]
        )

        row = {
            "cell_id":
                f"RWTH_{cell}",

            "cell_number":
                cell,

            "cycle_number_pre_filter":
                i,

            "start_row":
                start,

            "end_row_exclusive":
                end,

            "n_samples":
                len(cycle),

            "Qd_integrated_Ah":
                qd,

            "Qc_integrated_Ah":
                qc,

            "Qd_native_AhEla_range_Ah":
                native_qd,

            "Qd_abs_diff_Ah":
                (
                    abs(
                        qd
                        - native_qd
                    )
                    if np.isfinite(
                        native_qd
                    )
                    else np.nan
                ),

            "window_capacity_fraction_1p11":
                (
                    qd
                    / WINDOW_REFERENCE_AH
                ),

            "rated_capacity_fraction_2p05":
                (
                    qd
                    / RATED_CAPACITY_AH
                ),

            "voltage_min_V":
                (
                    float(
                        voltage.min()
                    )
                    if voltage.notna().any()
                    else np.nan
                ),

            "voltage_max_V":
                (
                    float(
                        voltage.max()
                    )
                    if voltage.notna().any()
                    else np.nan
                ),

            "temp_min_raw_C":
                (
                    float(
                        temperature_raw.min()
                    )
                    if temperature_raw.notna().any()
                    else np.nan
                ),

            "temp_median_raw_C":
                (
                    float(
                        temperature_raw.median()
                    )
                    if temperature_raw.notna().any()
                    else np.nan
                ),

            "temp_max_raw_C":
                (
                    float(
                        temperature_raw.max()
                    )
                    if temperature_raw.notna().any()
                    else np.nan
                ),

            "temp_min_qc_valid_C":
                (
                    float(
                        temperature_valid.min()
                    )
                    if temperature_valid.notna().any()
                    else np.nan
                ),

            "temp_median_qc_valid_C":
                (
                    float(
                        temperature_valid.median()
                    )
                    if temperature_valid.notna().any()
                    else np.nan
                ),

            "temp_max_qc_valid_C":
                (
                    float(
                        temperature_valid.max()
                    )
                    if temperature_valid.notna().any()
                    else np.nan
                ),

            "temp_qc_nonfinite_samples":
                int(
                    cycle[
                        "_temp_qc_nonfinite"
                    ].sum()
                ),

            "temp_qc_nonpositive_samples":
                int(
                    cycle[
                        "_temp_qc_nonpositive"
                    ].sum()
                ),

            "temp_qc_exact99_samples":
                int(
                    cycle[
                        "_temp_qc_exact99"
                    ].sum()
                ),

            "temp_qc_gt100_samples":
                int(
                    cycle[
                        "_temp_qc_gt100"
                    ].sum()
                ),

            "temp_qc_definite_invalid_samples":
                int(
                    cycle[
                        "_temp_qc_definite_invalid"
                    ].sum()
                ),

            "temp_high_60_100_retained_samples":
                int(
                    cycle[
                        "_temp_qc_high_60_100_retained"
                    ].sum()
                ),

            "source_file_count":
                source_files,

            "source_zip_first":
                source_first,

            "source_zip_last":
                source_last,

            "crosses_source_file_boundary":
                (
                    source_files > 1
                ),
        }

        if "Zyklus" in cycle.columns:

            z = numeric_series(
                cycle,
                "Zyklus"
            )

            row[
                "raw_Zyklus_first"
            ] = (
                float(z.iloc[0])
                if len(z)
                and np.isfinite(
                    z.iloc[0]
                )
                else np.nan
            )

            row[
                "raw_Zyklus_last"
            ] = (
                float(z.iloc[-1])
                if len(z)
                and np.isfinite(
                    z.iloc[-1]
                )
                else np.nan
            )

        rows.append(row)

    cycles = pd.DataFrame(
        rows
    )

    if cycles.empty:
        raise RuntimeError(
            f"RWTH_{cell}: "
            f"no reconstructed cycles"
        )

    remove = abnormal_cycle_mask(
        cycles[
            "Qd_integrated_Ah"
        ].to_numpy(
            dtype=float
        )
    )

    cycles[
        "batteryml_abnormal_removed"
    ] = remove

    cycles[
        "cycle_number_after_filter"
    ] = np.nan

    keep = ~remove

    cycles.loc[
        keep,
        "cycle_number_after_filter"
    ] = np.arange(
        1,
        int(
            keep.sum()
        ) + 1
    )

    clean = (
        cycles[
            keep
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    # Validation statistics exclude source-file
    # boundary cycles because AhEla can reset
    # between nested source archives.
    native_valid = clean[
        ~clean[
            "crosses_source_file_boundary"
        ]
    ][
        "Qd_abs_diff_Ah"
    ].dropna()

    qd_clean = clean[
        "Qd_integrated_Ah"
    ].dropna()

    summary = {
        "cell_id":
            f"RWTH_{cell}",

        "cell_number":
            cell,

        "n_TBA_Zyk_archives":
            len(members),

        "raw_rows":
            raw_rows,

        "duplicate_Zeit_rows_removed":
            duplicates,

        "rows_after_duplicate_removal":
            after_duplicates,

        "time_anomaly_rows_removed":
            time_anomaly_rows,

        "rows_after_time_qc":
            after_time_qc,

        "detected_starts_before_initial_skip":
            len(
                detected_starts_all
            ),

        "detected_starts_after_initial_skip":
            len(starts),

        "cycles_before_abnormal_filter":
            len(cycles),

        "batteryml_abnormal_cycles_removed":
            int(
                remove.sum()
            ),

        "clean_cycles":
            len(clean),

        "boundary_cycles_before_filter":
            int(
                cycles[
                    "crosses_source_file_boundary"
                ].sum()
            ),

        "boundary_cycles_clean":
            int(
                clean[
                    "crosses_source_file_boundary"
                ].sum()
            ),

        "Qd_clean_min_Ah":
            safe_stat(
                qd_clean,
                np.min
            ),

        "Qd_clean_median_Ah":
            safe_stat(
                qd_clean,
                np.median
            ),

        "Qd_clean_mean_Ah":
            safe_stat(
                qd_clean,
                np.mean
            ),

        "Qd_clean_max_Ah":
            safe_stat(
                qd_clean,
                np.max
            ),

        "Qd_first_clean_Ah":
            (
                float(
                    qd_clean.iloc[0]
                )
                if len(qd_clean)
                else np.nan
            ),

        "Qd_last_clean_Ah":
            (
                float(
                    qd_clean.iloc[-1]
                )
                if len(qd_clean)
                else np.nan
            ),

        "native_validation_n_nonboundary":
            int(
                len(native_valid)
            ),

        "native_abs_diff_median_Ah":
            safe_stat(
                native_valid,
                np.median
            ),

        "native_abs_diff_p95_Ah":
            safe_stat(
                native_valid,
                lambda x:
                    np.quantile(
                        x,
                        0.95
                    )
            ),

        "native_abs_diff_p99_Ah":
            safe_stat(
                native_valid,
                lambda x:
                    np.quantile(
                        x,
                        0.99
                    )
            ),

        "native_abs_diff_max_Ah":
            safe_stat(
                native_valid,
                np.max
            ),

        "temperature_raw_nonmissing_rows":
            temp_raw_nonmissing_n,

        "temperature_qc_valid_rows":
            temp_qc_valid_n,

        "temperature_qc_nonfinite_rows":
            temp_qc_nonfinite_n,

        "temperature_qc_nonpositive_rows":
            temp_qc_nonpositive_n,

        "temperature_qc_exact99_rows":
            temp_qc_exact99_n,

        "temperature_qc_gt100_rows":
            temp_qc_gt100_n,

        "temperature_qc_definite_invalid_rows":
            temp_qc_definite_invalid_n,

        "temperature_high_60_100_retained_rows":
            temp_high_60_100_retained_n,

        "temperature_raw_rows_gt_60C":
            temp_raw_gt_60_n,

        "temperature_raw_rows_gt_100C":
            temp_raw_gt_100_n,

        "temperature_max_raw_C":
            temp_raw_max,

        "temperature_max_qc_valid_C":
            temp_valid_max,

        "window_threshold_Ah":
            WINDOW_THRESHOLD_AH,

        "clean_cycles_at_or_below_window_threshold":
            int(
                (
                    clean[
                        "Qd_integrated_Ah"
                    ]
                    <= WINDOW_THRESHOLD_AH
                ).sum()
            ),

        "clean_fraction_at_or_below_window_threshold":
            float(
                (
                    clean[
                        "Qd_integrated_Ah"
                    ]
                    <= WINDOW_THRESHOLD_AH
                ).mean()
            ),
    }

    for k in (
        1,
        3,
        5,
        10
    ):

        crossing = (
            first_consecutive_crossing(
                clean,
                WINDOW_THRESHOLD_AH,
                k
            )
        )

        prefix = (
            f"window_crossing_k{k}"
        )

        summary[
            f"{prefix}_found"
        ] = crossing[
            "found"
        ]

        summary[
            f"{prefix}_clean_cycle"
        ] = crossing[
            "clean_position"
        ]

        summary[
            f"{prefix}_pre_filter_cycle"
        ] = crossing[
            "pre_filter_cycle"
        ]

        summary[
            f"{prefix}_Qd_Ah"
        ] = crossing[
            "Qd_Ah"
        ]

        summary[
            f"{prefix}_cycles_after"
        ] = crossing[
            "cycles_after"
        ]

        summary[
            f"{prefix}_fraction_after"
        ] = crossing[
            "fraction_after"
        ]

        summary[
            f"{prefix}_later_cycles_above_threshold"
        ] = crossing[
            "later_cycles_above_threshold"
        ]

    print(
        f"RWTH_{cell}: "
        f"clean={len(clean)}, "
        f"abnormal_removed="
        f"{int(remove.sum())}, "
        f"boundary_clean="
        f"{summary['boundary_cycles_clean']}, "
        f"Qd_first="
        f"{summary['Qd_first_clean_Ah']:.6f}, "
        f"Qd_last="
        f"{summary['Qd_last_clean_Ah']:.6f}, "
        f"k3_cross="
        f"{summary['window_crossing_k3_clean_cycle']}",
        flush=True
    )

    return (
        cycles,
        summary,
        source_records
    )


# ============================================================================
# Dataset reconstruction
# ============================================================================


# ============================================================================
# CANONICAL RWTH BOL PRACTICAL-CAPACITY AUDIT
# ============================================================================

def bol_integrate_charge(
    current,
    time_ms
):
    """
    Integrate positive-current capacity within one BOL state segment.

    RWTH Programmdauer is expressed in milliseconds:
        A * ms / 3.6e6 = Ah

    Negative local time steps are ignored, matching the previously
    validated BOL cohort pilot.
    """

    current = np.asarray(
        current,
        dtype=float
    )

    time_ms = np.asarray(
        time_ms,
        dtype=float
    )

    q = 0.0

    for i in range(
        1,
        len(current)
    ):

        dt = (
            time_ms[i]
            - time_ms[i - 1]
        )

        if (
            not np.isfinite(dt)
            or dt < 0
        ):
            continue

        if (
            np.isfinite(current[i])
            and current[i] > 0
        ):
            q += (
                current[i]
                * dt
                / 3.6e6
            )

    return float(q)


def bol_integrate_discharge(
    current,
    time_ms
):
    """
    Diagnostic discharge integration within the same BOL state segment.
    """

    current = np.asarray(
        current,
        dtype=float
    )

    time_ms = np.asarray(
        time_ms,
        dtype=float
    )

    q = 0.0

    for i in range(
        1,
        len(current)
    ):

        dt = (
            time_ms[i]
            - time_ms[i - 1]
        )

        if (
            not np.isfinite(dt)
            or dt < 0
        ):
            continue

        if (
            np.isfinite(current[i])
            and current[i] < 0
        ):
            q += (
                -current[i]
                * dt
                / 3.6e6
            )

    return float(q)


def run_bol_audit(
    append_to_zyk_summary=False
):
    """
    Reconstruct the verified 48-cell BOL standard-charge
    practical-capacity audit.

    Semantics
    ---------
    * 2.05 Ah:
      rated/manufacturer capacity reference.

    * Qc_practical_median_Ah:
      observed beginning-of-life standard-charge practical-capacity
      estimate from raw TBA_BOL_P2 / TBA_SC segments.

    * 1.11 Ah:
      separate effective cycling-window reference used by BatteryML.

    The proximity of 60% of the observed BOL practical capacity to
    1.11 Ah is reported only as a consistency observation. It is not
    interpreted as proof that BatteryML's value was derived from this
    calculation.
    """

    out_segments = (
        AUDIT
        / "RWTH_BOL_TBA_SC_segments.csv"
    )

    out_cells = (
        AUDIT
        / "RWTH_BOL_practical_capacity_by_cell.csv"
    )

    out_sources = (
        AUDIT
        / "RWTH_BOL_source_manifest.csv"
    )

    out_summary = (
        AUDIT
        / "RWTH_BOL_practical_capacity_summary.txt"
    )

    archive_counts = {
        cell: 0
        for cell in EXPECTED_CELLS
    }

    segment_rows = []
    source_records = []

    cell_re = re.compile(
        r"^Rohdaten/"
        r"e production=ep sanyo "
        r"(\d{3})="
    )

    with zipfile.ZipFile(
        INNER,
        "r"
    ) as outer:

        members = []

        for name in outer.namelist():

            m = cell_re.match(
                name
            )

            if not m:
                continue

            cell = m.group(1)

            if cell not in EXPECTED_CELLS:
                continue

            if (
                "=BOL Part 2=" in name
                and "=TBA_BOL_P2=TS" in name
                and "EISkanal" not in name
                and name.lower().endswith(
                    ".zip"
                )
            ):
                members.append(
                    (
                        cell,
                        name
                    )
                )

        represented = {
            cell
            for cell, _ in members
        }

        if (
            len(members)
            != EXPECTED_BOL_ARCHIVES
        ):
            raise RuntimeError(
                "Expected "
                f"{EXPECTED_BOL_ARCHIVES} "
                "non-EIS BOL Part-2 archives, "
                f"found {len(members)}."
            )

        if (
            represented
            != set(EXPECTED_CELLS)
        ):
            raise RuntimeError(
                "BOL cell coverage mismatch: "
                f"represented="
                f"{sorted(represented)}"
            )

        print()
        print("=" * 110)
        print(
            "RWTH CANONICAL 48-CELL "
            "BOL PRACTICAL-CAPACITY AUDIT"
        )
        print("=" * 110)

        print(
            "Selected non-EIS "
            "BOL Part-2 archives:",
            len(members)
        )

        print(
            "Unique cells represented:",
            len(represented)
        )

        for archive_index, (
            cell,
            member
        ) in enumerate(
            sorted(members),
            start=1
        ):

            archive_counts[
                cell
            ] += 1

            print(
                f"[{archive_index:02d}/"
                f"{len(members):02d}] "
                f"cell {cell}",
                flush=True
            )

            (
                df,
                source_record
            ) = read_nested_csv(
                outer,
                member,
                cell,
                archive_index
            )

            source_record[
                "source_role"
            ] = (
                "BOL_Part2_nonEIS"
            )

            source_records.append(
                source_record
            )

            raw_rows = len(df)

            duplicate_rows = int(
                df.duplicated(
                    "Zeit"
                ).sum()
            )

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

            state = (
                df[
                    "Zustand"
                ]
                .astype(str)
            )

            segment_id = (
                state.ne(
                    state.shift()
                )
                .cumsum()
            )

            for sid, seg in df.groupby(
                segment_id,
                sort=True
            ):

                if len(seg) < 2:
                    continue

                I = pd.to_numeric(
                    seg[
                        "Strom"
                    ],
                    errors="coerce"
                ).to_numpy(
                    dtype=float
                )

                t = pd.to_numeric(
                    seg[
                        "Programmdauer"
                    ],
                    errors="coerce"
                ).to_numpy(
                    dtype=float
                )

                V = pd.to_numeric(
                    seg[
                        "Spannung"
                    ],
                    errors="coerce"
                )

                T_raw = pd.to_numeric(
                    seg[
                        "_temperature_C_raw"
                    ],
                    errors="coerce"
                )

                T_valid = pd.to_numeric(
                    seg[
                        "_temperature_C_valid"
                    ],
                    errors="coerce"
                )

                AhLad = pd.to_numeric(
                    seg[
                        "AhLad"
                    ],
                    errors="coerce"
                )

                AhEla = pd.to_numeric(
                    seg[
                        "AhEla"
                    ],
                    errors="coerce"
                )

                qc = (
                    bol_integrate_charge(
                        I,
                        t
                    )
                )

                qd = (
                    bol_integrate_discharge(
                        I,
                        t
                    )
                )

                procedure = str(
                    seg[
                        "Prozedur"
                    ].iloc[0]
                )

                state_name = str(
                    seg[
                        "Zustand"
                    ].iloc[0]
                )

                # Verified practical-capacity candidate:
                # standard BOL charge, substantial capacity,
                # low-voltage start and high-voltage finish.
                is_candidate = (
                    state_name == "CHA"
                    and procedure == "TBA_SC"
                    and qc >= 1.50
                    and float(
                        V.iloc[0]
                    ) <= 3.45
                    and float(
                        V.iloc[-1]
                    ) >= 4.05
                )

                if not is_candidate:
                    continue

                native_qc = (
                    float(
                        AhLad.max()
                        - AhLad.min()
                    )
                    if AhLad.notna().any()
                    else np.nan
                )

                segment_rows.append({
                    "cell_id":
                        f"RWTH_{cell}",

                    "cell_number":
                        cell,

                    "archive_name":
                        Path(member).name,

                    "archive_member":
                        member,

                    "nested_zip_sha256":
                        source_record[
                            "nested_zip_sha256"
                        ],

                    "raw_rows_in_archive":
                        raw_rows,

                    "duplicate_Zeit_rows":
                        duplicate_rows,

                    "segment_id":
                        int(sid),

                    "state":
                        state_name,

                    "procedure":
                        procedure,

                    "n_samples":
                        len(seg),

                    "duration_s":
                        float(
                            t[-1]
                            - t[0]
                        ) / 1000.0,

                    "current_median_A":
                        float(
                            np.nanmedian(I)
                        ),

                    "voltage_start_V":
                        float(
                            V.iloc[0]
                        ),

                    "voltage_end_V":
                        float(
                            V.iloc[-1]
                        ),

                    "voltage_min_V":
                        float(
                            V.min()
                        ),

                    "voltage_max_V":
                        float(
                            V.max()
                        ),

                    "Qc_integrated_Ah":
                        qc,

                    "Qc_native_AhLad_range_Ah":
                        native_qc,

                    "Qc_abs_diff_Ah":
                        (
                            abs(
                                qc
                                - native_qc
                            )
                            if np.isfinite(
                                native_qc
                            )
                            else np.nan
                        ),

                    "Qd_in_same_segment_Ah":
                        qd,

                    "temp_min_raw":
                        (
                            float(
                                T_raw.min()
                            )
                            if T_raw.notna().any()
                            else np.nan
                        ),

                    "temp_median_raw":
                        (
                            float(
                                T_raw.median()
                            )
                            if T_raw.notna().any()
                            else np.nan
                        ),

                    "temp_max_raw":
                        (
                            float(
                                T_raw.max()
                            )
                            if T_raw.notna().any()
                            else np.nan
                        ),

                    "temp_min_qc_valid_C":
                        (
                            float(
                                T_valid.min()
                            )
                            if T_valid.notna().any()
                            else np.nan
                        ),

                    "temp_median_qc_valid_C":
                        (
                            float(
                                T_valid.median()
                            )
                            if T_valid.notna().any()
                            else np.nan
                        ),

                    "temp_max_qc_valid_C":
                        (
                            float(
                                T_valid.max()
                            )
                            if T_valid.notna().any()
                            else np.nan
                        ),

                    "temp_qc_definite_invalid_samples":
                        int(
                            seg[
                                "_temp_qc_definite_invalid"
                            ].sum()
                        ),
                })

    segments = pd.DataFrame(
        segment_rows
    )

    if segments.empty:
        raise RuntimeError(
            "No BOL TBA_SC practical-capacity "
            "candidates found."
        )

    segments = (
        segments
        .sort_values(
            [
                "cell_number",
                "archive_name",
                "segment_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    cell_rows = []

    for cell in EXPECTED_CELLS:

        g = segments[
            segments[
                "cell_number"
            ] == cell
        ]

        if len(g):

            med = float(
                g[
                    "Qc_integrated_Ah"
                ].median()
            )

            cell_rows.append({
                "cell_id":
                    f"RWTH_{cell}",

                "cell_number":
                    cell,

                "bol_capacity_archives":
                    archive_counts[
                        cell
                    ],

                "n_TBA_SC_candidates":
                    len(g),

                "Qc_practical_median_Ah":
                    med,

                "Qc_practical_min_Ah":
                    float(
                        g[
                            "Qc_integrated_Ah"
                        ].min()
                    ),

                "Qc_practical_max_Ah":
                    float(
                        g[
                            "Qc_integrated_Ah"
                        ].max()
                    ),

                "median_over_2p05":
                    med
                    / RATED_CAPACITY_AH,

                "effective_60pct_of_median_Ah":
                    med
                    * 0.60,

                "difference_from_1p85_Ah":
                    med
                    - 1.85,

                "median_Qc_native_diff_Ah":
                    float(
                        g[
                            "Qc_abs_diff_Ah"
                        ].median()
                    ),

                "max_temp_raw":
                    float(
                        g[
                            "temp_max_raw"
                        ].max()
                    ),
            })

        else:

            cell_rows.append({
                "cell_id":
                    f"RWTH_{cell}",

                "cell_number":
                    cell,

                "bol_capacity_archives":
                    archive_counts[
                        cell
                    ],

                "n_TBA_SC_candidates":
                    0,

                "Qc_practical_median_Ah":
                    np.nan,

                "Qc_practical_min_Ah":
                    np.nan,

                "Qc_practical_max_Ah":
                    np.nan,

                "median_over_2p05":
                    np.nan,

                "effective_60pct_of_median_Ah":
                    np.nan,

                "difference_from_1p85_Ah":
                    np.nan,

                "median_Qc_native_diff_Ah":
                    np.nan,

                "max_temp_raw":
                    np.nan,
            })

    cells = pd.DataFrame(
        cell_rows
    )

    sources = pd.DataFrame(
        source_records
    )

    valid = cells[
        cells[
            "Qc_practical_median_Ah"
        ].notna()
    ].copy()

    if len(valid) != len(
        EXPECTED_CELLS
    ):
        raise RuntimeError(
            "Not all configured cells have "
            "a valid BOL practical-capacity "
            "candidate."
        )

    q = valid[
        "Qc_practical_median_Ah"
    ]

    eff = valid[
        "effective_60pct_of_median_Ah"
    ]

    native_diff = (
        segments[
            "Qc_abs_diff_Ah"
        ]
        .dropna()
    )

    segments.to_csv(
        out_segments,
        index=False
    )

    cells.to_csv(
        out_cells,
        index=False
    )

    sources.to_csv(
        out_sources,
        index=False
    )

    lines = []

    lines.append(
        "=" * 110
    )

    lines.append(
        "RWTH CANONICAL 48-CELL "
        "BOL PRACTICAL-CAPACITY AUDIT"
    )

    lines.append(
        "=" * 110
    )

    lines.append("")

    lines.append(
        "SEMANTICS"
    )

    lines.append(
        "Rated/manufacturer capacity "
        f"reference: "
        f"{RATED_CAPACITY_AH:.2f} Ah."
    )

    lines.append(
        "Qc_practical_median_Ah is an "
        "observed BOL standard-charge "
        "practical-capacity estimate, "
        "not the rated nominal capacity."
    )

    lines.append(
        "BatteryML 1.11 Ah remains a "
        "separate effective cycling-window "
        "reference."
    )

    lines.append(
        "The 60%-of-BOL comparison is "
        "reported only as a consistency "
        "observation, not as a derivation "
        "of the 1.11-Ah value."
    )

    lines.append("")

    lines.append(
        "SOURCE COVERAGE"
    )

    lines.append(
        "Configured cells: "
        f"{len(EXPECTED_CELLS)}"
    )

    lines.append(
        "Selected non-EIS BOL Part-2 "
        f"archives: {len(sources)}"
    )

    lines.append(
        "Cells with valid TBA_SC "
        "practical-capacity candidate: "
        f"{len(valid)}/"
        f"{len(EXPECTED_CELLS)}"
    )

    lines.append(
        "Total TBA_SC candidates: "
        f"{len(segments)}"
    )

    lines.append(
        "Unique candidate archives: "
        f"{segments['archive_name'].nunique()}"
    )

    lines.append("")

    lines.append(
        "PER-CELL MEDIAN TBA_SC "
        "CHARGE CAPACITY [Ah]"
    )

    lines.append(
        q.describe().to_string()
    )

    lines.append("")

    lines.append(
        "Dataset-level median of "
        "cell medians: "
        f"{q.median():.9f} Ah"
    )

    lines.append(
        "Dataset-level mean of "
        "cell medians: "
        f"{q.mean():.9f} Ah"
    )

    lines.append(
        "Dataset-level standard deviation "
        "of cell medians: "
        f"{q.std(ddof=1):.9f} Ah"
    )

    lines.append(
        "Median practical / "
        "2.05-Ah rated capacity: "
        f"{(q / RATED_CAPACITY_AH).median():.9f}"
    )

    lines.append("")

    lines.append(
        "60% PRACTICAL-CAPACITY "
        "CONSISTENCY CHECK"
    )

    lines.append(
        "Median 60% practical capacity: "
        f"{eff.median():.9f} Ah"
    )

    lines.append(
        "Difference from BatteryML "
        "1.11-Ah reference: "
        f"{eff.median() - WINDOW_REFERENCE_AH:+.9f} Ah"
    )

    lines.append("")

    lines.append(
        "INTEGRATED Qc VS RAW AhLad "
        "RANGE [Ah]"
    )

    lines.append(
        f"n: {len(native_diff)}"
    )

    lines.append(
        "median / p95 / p99 / max: "
        f"{native_diff.median():.9f} / "
        f"{native_diff.quantile(0.95):.9f} / "
        f"{native_diff.quantile(0.99):.9f} / "
        f"{native_diff.max():.9f}"
    )

    lines.append("")

    lines.append(
        "FILES"
    )

    lines.append(
        str(out_segments)
    )

    lines.append(
        str(out_cells)
    )

    lines.append(
        str(out_sources)
    )

    summary_text = "\n".join(
        lines
    )

    out_summary.write_text(
        summary_text + "\n",
        encoding="utf-8"
    )

    print()
    print(
        summary_text
    )

    print()
    print(
        "=" * 110
    )

    print(
        "PER-CELL BOL CAPACITY TABLE"
    )

    print(
        "=" * 110
    )

    print(
        cells[
            [
                "cell_id",
                "bol_capacity_archives",
                "n_TBA_SC_candidates",
                "Qc_practical_median_Ah",
                "effective_60pct_of_median_Ah",
                "median_Qc_native_diff_Ah",
                "max_temp_raw",
            ]
        ].to_string(
            index=False
        )
    )

    if append_to_zyk_summary:

        if not OUT_SUMMARY.exists():
            raise RuntimeError(
                "ZYK summary does not exist; "
                "cannot append BOL semantics."
            )

        bol_append = []

        bol_append.append("")

        bol_append.append(
            "BOL PRACTICAL-CAPACITY CROSS-CHECK"
        )

        bol_append.append(
            "Observed BOL standard-charge "
            "practical-capacity estimate "
            "(median of 48 cell medians): "
            f"{q.median():.9f} Ah."
        )

        bol_append.append(
            "Observed per-cell median range: "
            f"{q.min():.9f}--"
            f"{q.max():.9f} Ah."
        )

        bol_append.append(
            "Median of 60% of per-cell "
            "BOL practical capacity: "
            f"{eff.median():.9f} Ah; "
            "difference from the separate "
            "1.11-Ah cycling-window reference: "
            f"{eff.median() - WINDOW_REFERENCE_AH:+.9f} Ah."
        )

        bol_append.append(
            "This proximity is treated only "
            "as a consistency observation; "
            "it is not used to infer the "
            "origin of the 1.11-Ah value."
        )

        with OUT_SUMMARY.open(
            "a",
            encoding="utf-8"
        ) as fh:

            fh.write(
                "\n".join(
                    bol_append
                )
                + "\n"
            )

    return {
        "segments":
            segments,

        "cells":
            cells,

        "sources":
            sources,

        "median_cell_capacity_Ah":
            float(
                q.median()
            ),

        "median_60pct_capacity_Ah":
            float(
                eff.median()
            ),

        "median_60pct_minus_1p11_Ah":
            float(
                eff.median()
                - WINDOW_REFERENCE_AH
            ),
    }


AUDIT_MODE = (
    os.environ.get(
        "RWTH_AUDIT_MODE",
        "all"
    )
    .strip()
    .lower()
)

if AUDIT_MODE not in {
    "all",
    "zyk-only",
    "bol-only",
}:
    raise ValueError(
        "RWTH_AUDIT_MODE must be "
        "'all', 'zyk-only', or 'bol-only'; "
        f"got {AUDIT_MODE!r}"
    )


if AUDIT_MODE == "bol-only":

    run_bol_audit(
        append_to_zyk_summary=False
    )

    raise SystemExit(0)


if not INNER.exists():
    raise FileNotFoundError(
        INNER
    )


all_cycles = []
cell_summaries = []
all_source_records = []

with zipfile.ZipFile(
    INNER,
    "r"
) as outer:

    all_names = outer.namelist()

    members_by_cell = {}

    for cell in EXPECTED_CELLS:

        prefix = (
            f"Rohdaten/"
            f"e production=ep sanyo "
            f"{cell}=ZYK="
        )

        members = [
            x
            for x in all_names
            if (
                x.startswith(prefix)
                and "=TBA_Zyk=" in x
                and x.lower().endswith(
                    ".zip"
                )
            )
        ]

        members_by_cell[
            cell
        ] = members

    missing_cells = [
        cell
        for cell, members
        in members_by_cell.items()
        if len(members) == 0
    ]

    if missing_cells:
        raise RuntimeError(
            "Missing TBA_Zyk archives "
            f"for cells: {missing_cells}"
        )

    total_archives = sum(
        len(x)
        for x in members_by_cell.values()
    )

    print("=" * 110)
    print(
        "RWTH CANONICAL 48-CELL "
        "ZYK DATASET AUDIT"
    )
    print("=" * 110)
    print(
        "Cells:",
        len(
            members_by_cell
        )
    )
    print(
        "TBA_Zyk archives:",
        total_archives
    )
    print(
        "Window reference [Ah]:",
        WINDOW_REFERENCE_AH
    )
    print(
        "80% window-reference "
        "sensitivity threshold [Ah]:",
        WINDOW_THRESHOLD_AH
    )
    print()

    if (
        len(members_by_cell)
        != 48
    ):
        raise RuntimeError(
            "Expected 48 RWTH cells."
        )

    if (
        total_archives
        != EXPECTED_ARCHIVES
    ):
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_ARCHIVES} "
            "TBA_Zyk archives, "
            f"found {total_archives}."
        )

    for cell in EXPECTED_CELLS:

        cycles, summary, sources = (
            process_cell(
                outer,
                cell,
                members_by_cell[
                    cell
                ]
            )
        )

        all_cycles.append(
            cycles
        )

        cell_summaries.append(
            summary
        )

        all_source_records.extend(
            sources
        )

        gc.collect()


cycles_df = pd.concat(
    all_cycles,
    ignore_index=True
)

cells_df = pd.DataFrame(
    cell_summaries
)

sources_df = pd.DataFrame(
    all_source_records
)

cycles_df.to_csv(
    OUT_CYCLES,
    index=False
)

cells_df.to_csv(
    OUT_CELLS,
    index=False
)

sources_df.to_csv(
    OUT_SOURCES,
    index=False
)


# ============================================================================
# Dataset-level summary
# ============================================================================

clean_mask = (
    ~cycles_df[
        "batteryml_abnormal_removed"
    ]
)

clean_all = cycles_df[
    clean_mask
].copy()

clean_nonboundary = clean_all[
    ~clean_all[
        "crosses_source_file_boundary"
    ]
].copy()

native_diff = (
    clean_nonboundary[
        "Qd_abs_diff_Ah"
    ]
    .dropna()
)

summary_lines = []

summary_lines.append(
    "=" * 110
)

summary_lines.append(
    "RWTH CANONICAL 48-CELL "
    "ZYK DATASET AUDIT"
)

summary_lines.append(
    "=" * 110
)

summary_lines.append("")

summary_lines.append(
    "SOURCE / PROVENANCE"
)

summary_lines.append(
    f"Official raw cells reconstructed: "
    f"{cells_df['cell_id'].nunique()}"
)

summary_lines.append(
    f"TBA_Zyk nested archives: "
    f"{len(sources_df)}"
)

summary_lines.append(
    "Raw-cell identifiers: "
    "RWTH_002 through RWTH_049"
)

summary_lines.append("")

summary_lines.append(
    "CAPACITY SEMANTICS"
)

summary_lines.append(
    "Rated cell capacity: "
    "2.05 Ah."
)

summary_lines.append(
    "BatteryML 1.11 Ah value is "
    "treated as an effective "
    "20--80% SOC cycling-window "
    "reference, not as the "
    "physical rated capacity."
)

summary_lines.append(
    "Official RWTH prognosis TDS "
    "uses a separate practical-"
    "capacity-scale degradation "
    "trajectory and is not "
    "equated with this ZYK-window "
    "capacity series."
)

summary_lines.append(
    "The 0.888 Ah threshold reported "
    "below is therefore a sensitivity "
    "threshold equal to 80% of the "
    "1.11 Ah cycling-window reference; "
    "it is not labelled physical-cell "
    "80% SOH."
)

summary_lines.append("")

summary_lines.append(
    "ROW CLEANING"
)

summary_lines.append(
    f"Total raw concatenated rows: "
    f"{int(cells_df['raw_rows'].sum())}"
)

summary_lines.append(
    "Duplicate Zeit rows removed: "
    f"{int(cells_df['duplicate_Zeit_rows_removed'].sum())}"
)

summary_lines.append(
    "Time-anomaly rows removed: "
    f"{int(cells_df['time_anomaly_rows_removed'].sum())}"
)

summary_lines.append("")

summary_lines.append(
    "CYCLE RECONSTRUCTION"
)

summary_lines.append(
    "Cycles before BatteryML "
    "abnormal filter: "
    f"{len(cycles_df)}"
)

summary_lines.append(
    "BatteryML abnormal cycles "
    "removed: "
    f"{int(cycles_df['batteryml_abnormal_removed'].sum())}"
)

summary_lines.append(
    f"Clean reconstructed cycles: "
    f"{len(clean_all)}"
)

summary_lines.append(
    "Clean source-file-boundary "
    "cycles: "
    f"{int(clean_all['crosses_source_file_boundary'].sum())}"
)

summary_lines.append("")

summary_lines.append(
    "CLEAN ZYK Qd [Ah]"
)

summary_lines.append(
    "min / median / mean / max: "
    f"{clean_all['Qd_integrated_Ah'].min():.9f} / "
    f"{clean_all['Qd_integrated_Ah'].median():.9f} / "
    f"{clean_all['Qd_integrated_Ah'].mean():.9f} / "
    f"{clean_all['Qd_integrated_Ah'].max():.9f}"
)

summary_lines.append("")

summary_lines.append(
    "INDEPENDENT AhEla VALIDATION "
    "(clean, non-boundary cycles)"
)

summary_lines.append(
    f"n = {len(native_diff)}"
)

if len(native_diff):

    summary_lines.append(
        "absolute difference "
        "median / p95 / p99 / max [Ah]: "
        f"{native_diff.median():.9f} / "
        f"{native_diff.quantile(0.95):.9f} / "
        f"{native_diff.quantile(0.99):.9f} / "
        f"{native_diff.max():.9f}"
    )

summary_lines.append("")

summary_lines.append(
    "TEMPERATURE QC"
)

summary_lines.append(
    "Temperature QC changes only the temperature channel; "
    "no V/I/time/capacity row is removed."
)

summary_lines.append(
    "Definite-invalid temperature rule: "
    "non-finite, T <= 0 C, exact 99.0 C, or T > 100 C."
)

summary_lines.append(
    "Raw non-missing temperature rows: "
    f"{int(cells_df['temperature_raw_nonmissing_rows'].sum())}"
)

summary_lines.append(
    "QC-valid temperature rows: "
    f"{int(cells_df['temperature_qc_valid_rows'].sum())}"
)

summary_lines.append(
    "Definite-invalid temperature rows: "
    f"{int(cells_df['temperature_qc_definite_invalid_rows'].sum())}"
)

summary_lines.append(
    "  non-finite: "
    f"{int(cells_df['temperature_qc_nonfinite_rows'].sum())}"
)

summary_lines.append(
    "  T <= 0 C: "
    f"{int(cells_df['temperature_qc_nonpositive_rows'].sum())}"
)

summary_lines.append(
    "  exact 99.0 C: "
    f"{int(cells_df['temperature_qc_exact99_rows'].sum())}"
)

summary_lines.append(
    "  T > 100 C: "
    f"{int(cells_df['temperature_qc_gt100_rows'].sum())}"
)

summary_lines.append(
    "High-temperature rows retained "
    "(60 < T <= 100 C, excluding exact 99 C): "
    f"{int(cells_df['temperature_high_60_100_retained_rows'].sum())}"
)

summary_lines.append(
    "Maximum raw temperature value [C]: "
    f"{cells_df['temperature_max_raw_C'].max():.6f}"
)

summary_lines.append(
    "Maximum temperature after definite-invalid masking [C]: "
    f"{cells_df['temperature_max_qc_valid_C'].max():.6f}"
)

summary_lines.append(
    "No generic 60 C clipping is applied because the raw data "
    "contain long, smoothly varying 60--70 C episodes; these "
    "are retained and flagged for sensitivity analysis."
)

summary_lines.append("")

summary_lines.append(
    "0.888 Ah WINDOW-REFERENCE "
    "THRESHOLD SENSITIVITY"
)

for k in (
    1,
    3,
    5,
    10
):

    found_col = (
        f"window_crossing_k{k}_found"
    )

    cycle_col = (
        f"window_crossing_k{k}_clean_cycle"
    )

    found = cells_df[
        found_col
    ].fillna(False)

    cycles_found = (
        cells_df.loc[
            found,
            cycle_col
        ]
        .dropna()
    )

    summary_lines.append(
        f"k={k}: "
        f"{int(found.sum())}/48 cells "
        f"cross; "
        f"median clean-cycle crossing="
        + (
            f"{cycles_found.median():.1f}"
            if len(cycles_found)
            else "NA"
        )
    )

summary_lines.append("")

summary_lines.append(
    "NOTE ON BATTERYML COMPATIBILITY"
)

summary_lines.append(
    "Cycle segmentation reproduces "
    "BatteryML semantics except for "
    "an explicit final-index boundary "
    "guard preventing unsafe "
    "current[i+1] access."
)

summary_lines.append(
    "Programmdauer is treated as "
    "milliseconds for Coulomb counting. "
    "No claim is made here that the "
    "BatteryML CycleData time_in_s "
    "metadata field contains SI-second "
    "values for RWTH."
)

summary_text = "\n".join(
    summary_lines
)

OUT_SUMMARY.write_text(
    summary_text + "\n",
    encoding="utf-8"
)

print()
print(summary_text)

print()
print("=" * 110)
print("OUTPUT FILES")
print("=" * 110)
print(OUT_CYCLES)
print(OUT_CELLS)
print(OUT_SOURCES)
print(OUT_SUMMARY)


# ============================================================================
# Canonical BOL cross-check after successful ZYK reconstruction
# ============================================================================

if AUDIT_MODE == "all":

    run_bol_audit(
        append_to_zyk_summary=True
    )
