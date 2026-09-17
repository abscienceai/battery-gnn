#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd


AUDIT_DIR = Path(
    "/home/msoylu/alper/battery_revision_2026/data/audit"
)

OUT = AUDIT_DIR / "CALCE_80pct_EOL_audit.csv"
TXT = AUDIT_DIR / "CALCE_80pct_EOL_summary.txt"


def first_consecutive_below(values, threshold, n):
    below = np.asarray(values) <= threshold

    if len(below) < n:
        return None

    for i in range(len(below) - n + 1):
        if np.all(below[i:i+n]):
            return i + 1  # 1-based clean-sequence position

    return None


rows = []

files = sorted(AUDIT_DIR.glob("C[XS]2_*_cycles_audit.csv"))

print("=" * 90)
print("CALCE 80% EOL AUDIT")
print("=" * 90)
print("Files found:", len(files))

if len(files) != 13:
    raise RuntimeError(
        f"Expected 13 CALCE cycle files, found {len(files)}"
    )

for path in files:
    cell = path.name.replace("_cycles_audit.csv", "")

    df = pd.read_csv(path)

    clean = (
        df[df["keep_final_batteryml"] == True]
        .sort_values("batteryml_cycle_number_zero_based")
        .reset_index(drop=True)
        .copy()
    )

    if clean.empty:
        raise RuntimeError(f"{cell}: no clean cycles")

    nominal_values = clean["nominal_capacity_correct_Ah"].unique()

    if len(nominal_values) != 1:
        raise RuntimeError(
            f"{cell}: inconsistent nominal capacities "
            f"{nominal_values}"
        )

    nominal = float(nominal_values[0])
    threshold_Ah = 0.80 * nominal

    q = clean["Qd_integrated_Ah"].to_numpy(dtype=float)
    soh = q / nominal

    crossing = np.where(q <= threshold_Ah)[0]

    if len(crossing) == 0:
        first_eol_pos = None
        first_eol_source_cycle = None
        q_at_eol = None
        soh_at_eol = None
        cycles_after_eol = None
        old_rul_overestimate_first = None
        post_cross_above = None
    else:
        i = int(crossing[0])

        # 1-based position in CLEAN modeling sequence
        first_eol_pos = i + 1

        first_eol_source_cycle = int(
            clean.loc[i, "batteryml_cycle_number_zero_based"]
        )

        q_at_eol = float(q[i])
        soh_at_eol = float(soh[i])

        cycles_after_eol = len(clean) - first_eol_pos

        # If old label = max recorded clean cycle - current cycle,
        # then at the first modeled cycle the overestimate relative
        # to 80%-EOL RUL is exactly the number of post-EOL cycles.
        old_rul_overestimate_first = cycles_after_eol

        post_cross_above = int(
            np.sum(soh[i+1:] > 0.80)
        )

    first3 = first_consecutive_below(q, threshold_Ah, 3)
    first5 = first_consecutive_below(q, threshold_Ah, 5)

    rows.append({
        "cell_id": cell,
        "nominal_capacity_Ah": nominal,
        "threshold_80pct_Ah": threshold_Ah,

        "clean_recorded_cycles": len(clean),

        "first_clean_Qd_Ah": float(q[0]),
        "first_clean_SOH": float(soh[0]),

        "minimum_clean_Qd_Ah": float(np.min(q)),
        "minimum_clean_SOH": float(np.min(soh)),

        "first_80pct_EOL_clean_position":
            first_eol_pos,

        "first_80pct_EOL_original_grouped_cycle":
            first_eol_source_cycle,

        "Qd_at_first_80pct_EOL_Ah":
            q_at_eol,

        "SOH_at_first_80pct_EOL":
            soh_at_eol,

        "cycle_life_to_80pct_inclusive":
            first_eol_pos,

        "RUL_first_cycle_zero_at_EOL":
            (
                first_eol_pos - 1
                if first_eol_pos is not None
                else None
            ),

        "cycles_recorded_after_first_80pct_EOL":
            cycles_after_eol,

        "fraction_clean_cycles_after_EOL":
            (
                cycles_after_eol / len(clean)
                if cycles_after_eol is not None
                else None
            ),

        "old_end_of_record_RUL_overestimate_cycles":
            old_rul_overestimate_first,

        "cycles_above_80pct_after_first_crossing":
            post_cross_above,

        "first_3_consecutive_below_80_position":
            first3,

        "first_5_consecutive_below_80_position":
            first5,
    })


result = pd.DataFrame(rows).sort_values("cell_id")

result.to_csv(OUT, index=False)

lines = []

def emit(x=""):
    print(x)
    lines.append(str(x))


emit()
emit("=" * 90)
emit("PER-CELL 80% EOL RESULTS")
emit("=" * 90)

cols = [
    "cell_id",
    "nominal_capacity_Ah",
    "clean_recorded_cycles",
    "first_80pct_EOL_clean_position",
    "cycles_recorded_after_first_80pct_EOL",
    "fraction_clean_cycles_after_EOL",
    "cycles_above_80pct_after_first_crossing",
    "first_3_consecutive_below_80_position",
    "first_5_consecutive_below_80_position",
]

emit(result[cols].to_string(index=False))

emit()
emit("=" * 90)
emit("DATASET SUMMARY")
emit("=" * 90)

reached = result["first_80pct_EOL_clean_position"].notna()

emit(
    f"Cells reaching <=80% nominal capacity: "
    f"{int(reached.sum())}/{len(result)}"
)

post = result.loc[
    reached,
    "cycles_recorded_after_first_80pct_EOL"
]

emit()
emit("Recorded cycles AFTER first 80% EOL:")
emit(post.describe().to_string())

frac = result.loc[
    reached,
    "fraction_clean_cycles_after_EOL"
]

emit()
emit("Fraction of clean record AFTER first 80% EOL:")
emit(frac.describe().to_string())

emit()
emit(
    "Total clean cycles: "
    f"{int(result['clean_recorded_cycles'].sum())}"
)

emit(
    "Total cycles through first 80% EOL (inclusive): "
    f"{int(result.loc[reached, 'cycle_life_to_80pct_inclusive'].sum())}"
)

emit(
    "Total clean cycles after first 80% EOL: "
    f"{int(post.sum())}"
)

emit()
emit("Old end-of-record RUL overestimate at first cycle:")
emit(
    result.loc[
        reached,
        "old_end_of_record_RUL_overestimate_cycles"
    ].describe().to_string()
)

emit()
emit("Crossing stability check:")
emit(
    "Cells that later return above 80% after first crossing: "
    f"{int((result['cycles_above_80pct_after_first_crossing'] > 0).sum())}"
)

emit(
    "Cells with identical first-cross / 3-consecutive position: "
    f"{int((result['first_80pct_EOL_clean_position'] == result['first_3_consecutive_below_80_position']).sum())}"
)

emit(
    "Cells with identical first-cross / 5-consecutive position: "
    f"{int((result['first_80pct_EOL_clean_position'] == result['first_5_consecutive_below_80_position']).sum())}"
)

emit()
emit("Saved:")
emit(str(OUT))
emit(str(TXT))

TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
