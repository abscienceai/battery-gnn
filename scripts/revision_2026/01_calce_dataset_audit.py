#!/usr/bin/env python3

import argparse
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import medfilt


REQUIRED = [
    "Cycle_Index",
    "Test_Time(s)",
    "Current(A)",
    "Voltage(V)",
]


def extract_date_batteryml(filename):
    """
    Reproduce BatteryML date parsing exactly.
    Note: two-digit years such as 10 become 0010.
    """
    filename = filename.upper()

    pat = r"C[XS]2?_\d+_(\d+)_(\d+)B?_(\d+)"
    matches = re.findall(pat, filename)

    if not matches:
        pat = r"(\d+)_(\d+)_(\d+)_CX2_32"
        matches = re.findall(pat, filename)

    if not matches:
        raise ValueError(f"Cannot parse date from: {filename}")

    month, day, year = matches[0]
    month, day, year = int(month), int(day), int(year)

    return f"{year:04}-{month:02}-{day:02}"


def extract_date_human(filename):
    """
    Interpret CALCE's two-digit year as 20xx for provenance display.
    """
    filename = filename.upper()

    pat = r"C[XS]2?_\d+_(\d+)_(\d+)B?_(\d+)"
    matches = re.findall(pat, filename)

    if not matches:
        pat = r"(\d+)_(\d+)_(\d+)_CX2_32"
        matches = re.findall(pat, filename)

    if not matches:
        raise ValueError(f"Cannot parse date from: {filename}")

    month, day, year = map(int, matches[0])

    if year < 100:
        year += 2000

    return f"{year:04}-{month:02}-{day:02}"


def load_excel_bytes(raw, stem):
    xf = pd.ExcelFile(io.BytesIO(raw))

    channel_frames = []

    for sheet_name in xf.sheet_names:
        if str(sheet_name).startswith("Channel"):
            channel_frames.append(xf.parse(sheet_name))

    # Reproduce BatteryML fallback for dirty workbook(s)
    if len(channel_frames) == 0:
        channel_frames = [
            xf.parse(sheet_name)
            for sheet_name in xf.sheet_names
        ]

    df = pd.concat(channel_frames, ignore_index=True)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(
            f"{stem}: required columns missing: {missing}"
        )

    keep = REQUIRED.copy()

    for optional in [
        "Charge_Capacity(Ah)",
        "Discharge_Capacity(Ah)",
    ]:
        if optional in df.columns:
            keep.append(optional)

    result = df[keep].copy()

    result["date_batteryml"] = extract_date_batteryml(stem)
    result["date_human"] = extract_date_human(stem)
    result["source_file"] = stem
    result["source_format"] = "xlsx"

    return result


def load_txt_bytes(raw, stem):
    df = pd.read_csv(io.BytesIO(raw), sep="\t")

    required_txt = [
        "Charge count",
        "Time",
        "mA",
        "mV",
    ]

    missing = [c for c in required_txt if c not in df.columns]

    if missing:
        raise KeyError(
            f"{stem}: TXT columns missing: {missing}"
        )

    result = pd.DataFrame({
        "Cycle_Index": df["Charge count"] // 2 + 1,
        "Test_Time(s)": df["Time"],
        "Current(A)": df["mA"] / 1000.0,
        "Voltage(V)": df["mV"] / 1000.0,
    })

    result["Charge_Capacity(Ah)"] = np.nan
    result["Discharge_Capacity(Ah)"] = np.nan

    result["date_batteryml"] = extract_date_batteryml(stem)
    result["date_human"] = extract_date_human(stem)
    result["source_file"] = stem
    result["source_format"] = "txt"

    return result


def organize_cycle_index(values):
    """
    Reproduce BatteryML organize_cycle_index().
    """
    arr = np.asarray(values).copy()

    if len(arr) == 0:
        return arr

    current_cycle = arr[0]
    prev_value = arr[0]

    for i in range(1, len(arr)):
        if arr[i] != prev_value:
            current_cycle += 1
            prev_value = arr[i]

        arr[i] = current_cycle

    return arr


def calc_q(current, time_s, is_charge):
    """
    Reproduce BatteryML calc_Q() exactly.
    """
    I = np.asarray(current, dtype=np.float64)
    t = np.asarray(time_s, dtype=np.float64)

    Q = np.zeros_like(I, dtype=np.float64)

    for i in range(1, len(I)):
        if is_charge and I[i] > 0:
            Q[i] = (
                Q[i - 1]
                + I[i] * (t[i] - t[i - 1]) / 3600.0
            )

        elif (not is_charge) and I[i] < 0:
            Q[i] = (
                Q[i - 1]
                - I[i] * (t[i] - t[i - 1]) / 3600.0
            )

        else:
            Q[i] = Q[i - 1]

    return Q


def process_cell(zip_path, outdir):
    cell = zip_path.stem

    print()
    print("=" * 78)
    print("CALCE DATASET AUDIT:", cell)
    print("=" * 78)

    frames = []
    format_counts = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [
            m for m in zf.namelist()
            if not m.endswith("/")
            and Path(m).suffix.lower() in {".xlsx", ".xls", ".txt"}
        ]

        for i, member in enumerate(sorted(members), start=1):
            suffix = Path(member).suffix.lower()
            stem = Path(member).stem

            print(
                f"[{i:02d}/{len(members):02d}] {Path(member).name}",
                flush=True
            )

            raw = zf.read(member)

            if suffix in {".xlsx", ".xls"}:
                frame = load_excel_bytes(raw, stem)
                fmt = "xlsx"
            else:
                frame = load_txt_bytes(raw, stem)
                fmt = "txt"

            format_counts[fmt] = format_counts.get(fmt, 0) + 1
            frames.append(frame)

    df = pd.concat(frames, ignore_index=True)

    print("Rows loaded:", len(df))
    print("Formats:", format_counts)

    print(
        "BatteryML date range:",
        df["date_batteryml"].min(),
        "to",
        df["date_batteryml"].max(),
    )

    print(
        "Human date range:",
        df["date_human"].min(),
        "to",
        df["date_human"].max(),
    )

    print(
        "NaN Cycle_Index:",
        int(pd.isna(df["Cycle_Index"]).sum())
    )

    df = df.sort_values(
        ["date_batteryml", "Test_Time(s)"],
        kind="mergesort"
    ).reset_index(drop=True)

    df["Cycle_Index_organized"] = organize_cycle_index(
        df["Cycle_Index"].to_numpy()
    )

    cycle_rows = []

    groups = df.groupby(
        ["date_batteryml", "Cycle_Index_organized"],
        sort=True
    )

    for cycle_number, ((date, cyc_idx), g) in enumerate(groups):

        I = g["Current(A)"].to_numpy(dtype=np.float64)
        t = g["Test_Time(s)"].to_numpy(dtype=np.float64)
        V = g["Voltage(V)"].to_numpy(dtype=np.float64)

        qd = calc_q(I, t, is_charge=False)
        qc = calc_q(I, t, is_charge=True)

        qd_integrated = (
            float(np.max(qd))
            if len(qd) else np.nan
        )

        qc_integrated = (
            float(np.max(qc))
            if len(qc) else np.nan
        )

        native_qd = g["Discharge_Capacity(Ah)"].to_numpy(
            dtype=np.float64
        )

        native_qd = native_qd[np.isfinite(native_qd)]

        native_qd_min = (
            float(np.min(native_qd))
            if len(native_qd) else np.nan
        )

        native_qd_max = (
            float(np.max(native_qd))
            if len(native_qd) else np.nan
        )

        native_qd_range = (
            native_qd_max - native_qd_min
            if len(native_qd) else np.nan
        )

        cycle_rows.append({
            "batteryml_cycle_number_zero_based":
                cycle_number,

            "date_batteryml":
                date,

            "date_human":
                g["date_human"].iloc[0],

            "organized_cycle_index":
                float(cyc_idx),

            "rows":
                len(g),

            "Qd_integrated_Ah":
                qd_integrated,

            "Qc_integrated_Ah":
                qc_integrated,

            "Qd_instrument_min_Ah":
                native_qd_min,

            "Qd_instrument_max_Ah":
                native_qd_max,

            "Qd_instrument_range_Ah":
                native_qd_range,

            "abs_diff_integrated_vs_instrument_range_Ah":
                (
                    abs(qd_integrated - native_qd_range)
                    if np.isfinite(native_qd_range)
                    else np.nan
                ),

            "source_files":
                ";".join(sorted(set(g["source_file"].astype(str)))),


            "V_min":
                float(np.nanmin(V)),

            "V_max":
                float(np.nanmax(V)),

            "I_min_A":
                float(np.nanmin(I)),

            "I_max_A":
                float(np.nanmax(I)),
        })

    cycles = pd.DataFrame(cycle_rows)

    Qd = cycles["Qd_integrated_Ah"].to_numpy()

    # Exact BatteryML cleaning rule
    Qd_med = medfilt(Qd, 21)

    abs_dev = np.abs(Qd - Qd_med)
    ths = float(np.median(abs_dev))

    should_keep = abs_dev < 3.0 * ths

    special_cx234_removed = False

    if cell.upper() == "CX2_34" and len(should_keep):
        should_keep[0] = False
        special_cx234_removed = True

    positive_capacity = Qd > 0.1
    keep_pre_special = should_keep & positive_capacity

    cycles["median_filter_Qd_Ah"] = Qd_med
    cycles["abs_deviation_Ah"] = abs_dev
    cycles["median_abs_deviation_threshold_Ah"] = ths
    cycles["keep_median_rule"] = should_keep
    cycles["keep_Qd_gt_0p1"] = positive_capacity
    cycles["keep_before_CX2_16_special"] = keep_pre_special

    kept_indices = np.where(keep_pre_special)[0].tolist()

    cx216_extra_drop = False

    # BatteryML removes first already-cleaned CX2_16 cycle
    if cell.upper() == "CX2_16" and kept_indices:
        kept_indices = kept_indices[1:]
        cx216_extra_drop = True

    final_keep = np.zeros(len(cycles), dtype=bool)
    final_keep[kept_indices] = True

    cycles["keep_final_batteryml"] = final_keep

    nominal = (
        1.10
        if cell.upper().startswith("CS")
        else 1.35
    )

    cycles["nominal_capacity_correct_Ah"] = nominal

    cycles["SOH_correct"] = (
        cycles["Qd_integrated_Ah"] / nominal
    )

    cycles["SOH_if_1p10_Ah_used"] = (
        cycles["Qd_integrated_Ah"] / 1.10
    )

    cycles["SOH_delta_wrong_minus_correct"] = (
        cycles["SOH_if_1p10_Ah_used"]
        - cycles["SOH_correct"]
    )

    clean = cycles[cycles["keep_final_batteryml"]].copy()

    outfile = outdir / f"{cell}_cycles_audit.csv"
    cycles.to_csv(outfile, index=False)

    finite_diff = (
        cycles[
            "abs_diff_integrated_vs_instrument_range_Ah"
        ]
        .dropna()
    )

    print()
    print("--- SUMMARY ---")
    print("Raw grouped cycles :", len(cycles))
    print("Final clean cycles :", len(clean))
    print("Nominal capacity   :", nominal, "Ah")

    print("Median-filter ths  :", f"{ths:.8f}", "Ah")

    print(
        "Removed median rule:",
        int((~should_keep).sum())
    )

    print(
        "Removed Qd <= 0.1  :",
        int((~positive_capacity).sum())
    )

    print(
        "CX2_34 special     :",
        special_cx234_removed
    )

    print(
        "CX2_16 extra drop  :",
        cx216_extra_drop
    )

    print()
    print("RAW Qd integrated:")
    print(
        cycles["Qd_integrated_Ah"]
        .describe()
        .to_string()
    )

    print()
    print("CLEAN Qd integrated:")
    print(
        clean["Qd_integrated_Ah"]
        .describe()
        .to_string()
    )

    if len(finite_diff):
        print()
        print("Integrated vs instrument Qd |difference|:")
        print(finite_diff.describe().to_string())

    print()
    print("CLEAN SOH using correct nominal capacity:")
    print(
        clean["SOH_correct"]
        .describe()
        .to_string()
    )

    print()
    print("CLEAN SOH if 1.10 Ah were used:")
    print(
        clean["SOH_if_1p10_Ah_used"]
        .describe()
        .to_string()
    )

    if cell.upper().startswith("CX"):
        delta = clean["SOH_delta_wrong_minus_correct"]

        print()
        print("CX FAMILY NORMALIZATION EFFECT:")
        print(
            "Mean absolute SOH shift from using "
            "1.10 instead of 1.35:",
            f"{delta.abs().mean():.6f}"
        )

        ratio = (
            clean["SOH_if_1p10_Ah_used"]
            / clean["SOH_correct"]
        )

        print(
            "Mean multiplicative inflation:",
            f"{ratio.mean():.6f}x"
        )

        print(
            "Expected theoretical inflation:",
            f"{1.35/1.10:.6f}x"
        )

    print()
    print("First 5 clean cycles:")
    print(
        clean[
            [
                "batteryml_cycle_number_zero_based",
                "date_human",
                "Qd_integrated_Ah",
                "Qd_instrument_range_Ah",
                "SOH_correct",
                "SOH_if_1p10_Ah_used",
            ]
        ]
        .head()
        .to_string(index=False)
    )

    print()
    print("Last 5 clean cycles:")
    print(
        clean[
            [
                "batteryml_cycle_number_zero_based",
                "date_human",
                "Qd_integrated_Ah",
                "Qd_instrument_range_Ah",
                "SOH_correct",
                "SOH_if_1p10_Ah_used",
            ]
        ]
        .tail()
        .to_string(index=False)
    )

    print()
    print("Saved:", outfile)

    return {
        "cell_id": cell,
        "source_files": len(members),
        "rows": len(df),
        "raw_grouped_cycles": len(cycles),
        "final_clean_cycles": len(clean),
        "nominal_capacity_Ah": nominal,
        "median_filter_ths_Ah": ths,

        "clean_Qd_min_Ah":
            float(clean["Qd_integrated_Ah"].min()),

        "clean_Qd_median_Ah":
            float(clean["Qd_integrated_Ah"].median()),

        "clean_Qd_max_Ah":
            float(clean["Qd_integrated_Ah"].max()),

        "clean_SOH_correct_min":
            float(clean["SOH_correct"].min()),

        "clean_SOH_correct_median":
            float(clean["SOH_correct"].median()),

        "clean_SOH_correct_max":
            float(clean["SOH_correct"].max()),

        "median_abs_Qd_diff_instrument_Ah":
            (
                float(finite_diff.median())
                if len(finite_diff) else np.nan
            ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rawdir",
        required=True
    )

    parser.add_argument(
        "--outdir",
        required=True
    )

    parser.add_argument(
        "--cells",
        nargs="+",
        default=["CS2_33", "CX2_33"]
    )

    args = parser.parse_args()

    rawdir = Path(args.rawdir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    summaries = []

    for cell in args.cells:
        zip_path = rawdir / f"{cell}.zip"

        if not zip_path.exists():
            raise FileNotFoundError(zip_path)

        summaries.append(
            process_cell(zip_path, outdir)
        )

    summary = pd.DataFrame(summaries)

    summary_file = outdir / "CALCE_dataset_audit_summary.csv"
    summary.to_csv(summary_file, index=False)

    print()
    print("=" * 78)
    print("DATASET COMPARISON")
    print("=" * 78)
    print(summary.to_string(index=False))
    print()
    print("Saved:", summary_file)


if __name__ == "__main__":
    main()
