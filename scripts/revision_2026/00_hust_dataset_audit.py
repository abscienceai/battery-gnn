#!/usr/bin/env python3

import argparse
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


NOMINAL_CAPACITY_AH = 1.10


def natural_cell_key(name):
    stem = Path(name).stem
    a, b = stem.split("-")
    return int(a), int(b)


def integrate_capacity(current_mA, time_s, discharge=True):
    """
    Coulomb counting following the BatteryML HUST convention.

    Current is converted from mA to A.
    For discharge, only I < 0 contributes.
    For charge, only I > 0 contributes.
    """
    I = np.asarray(current_mA, dtype=np.float64) / 1000.0
    t = np.asarray(time_s, dtype=np.float64)

    if len(I) < 2:
        return np.nan

    dt = np.diff(t)
    I_step = I[1:]

    valid = np.isfinite(dt) & np.isfinite(I_step) & (dt >= 0)

    if discharge:
        mask = valid & (I_step < 0)
        return float(np.sum((-I_step[mask]) * dt[mask] / 3600.0))

    mask = valid & (I_step > 0)
    return float(np.sum(I_step[mask] * dt[mask] / 3600.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, dest="zip_path")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    zip_path = Path(args.zip_path).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cell_rows = []
    cycle_rows = []

    print("=" * 78)
    print("HUST 77-CELL DATASET AUDIT")
    print("=" * 78)
    print("Archive:", zip_path)
    print("Nominal capacity basis:", NOMINAL_CAPACITY_AH, "Ah")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [
            x for x in zf.namelist()
            if x.startswith("our_data/") and x.endswith(".pkl")
        ]

        members = sorted(
            members,
            key=lambda x: natural_cell_key(Path(x).stem)
        )

        print("PKL cells found:", len(members))

        if len(members) != 77:
            raise RuntimeError(
                f"Expected 77 HUST cells, found {len(members)}"
            )

        for idx, member in enumerate(members, start=1):
            cell_id = Path(member).stem

            print(f"[{idx:02d}/77] {cell_id}", flush=True)

            with zf.open(member) as f:
                obj = pickle.load(f)

            if cell_id not in obj:
                raise KeyError(
                    f"{cell_id}: cell key missing from pickle object"
                )

            cell = obj[cell_id]

            if not {"dq", "rul", "data"}.issubset(cell.keys()):
                raise KeyError(
                    f"{cell_id}: expected dq/rul/data keys, "
                    f"found {list(cell.keys())}"
                )

            dq = cell["dq"]
            rul = cell["rul"]
            data = cell["data"]

            cycle_keys = sorted(
                set(dq.keys()) &
                set(rul.keys()) &
                set(data.keys())
            )

            if not cycle_keys:
                raise RuntimeError(f"{cell_id}: no common cycle keys")

            dq_raw = np.asarray(
                [dq[k] for k in cycle_keys],
                dtype=np.float64
            )

            # HUST source label is stored at approximately mAh scale.
            dq_ah = dq_raw / 1000.0

            rul_arr = np.asarray(
                [rul[k] for k in cycle_keys],
                dtype=np.float64
            )

            first_cycle = cycle_keys[0]
            last_cycle = cycle_keys[-1]

            first_df = data[first_cycle]
            last_df = data[last_cycle]

            qint_first = integrate_capacity(
                first_df["Current (mA)"],
                first_df["Time (s)"],
                discharge=True
            )

            qint_last = integrate_capacity(
                last_df["Current (mA)"],
                last_df["Time (s)"],
                discharge=True
            )

            first_label_ah = float(dq_ah[0])
            last_label_ah = float(dq_ah[-1])

            expected_rul = np.arange(
                len(cycle_keys), 0, -1,
                dtype=np.float64
            )

            rul_exact_remaining_recorded = bool(
                len(rul_arr) == len(expected_rul)
                and np.array_equal(rul_arr, expected_rul)
            )

            row = {
                "cell_id": cell_id,
                "n_cycles": len(cycle_keys),

                "first_cycle_key": first_cycle,
                "last_cycle_key": last_cycle,

                "dq_raw_min": float(np.min(dq_raw)),
                "dq_raw_median": float(np.median(dq_raw)),
                "dq_raw_max": float(np.max(dq_raw)),

                "capacity_Ah_min": float(np.min(dq_ah)),
                "capacity_Ah_mean": float(np.mean(dq_ah)),
                "capacity_Ah_median": float(np.median(dq_ah)),
                "capacity_Ah_max": float(np.max(dq_ah)),
                "capacity_Ah_std": float(np.std(dq_ah)),

                "capacity_Ah_first": first_label_ah,
                "capacity_Ah_last": last_label_ah,

                "first_capacity_over_nominal":
                    first_label_ah / NOMINAL_CAPACITY_AH,

                "last_capacity_over_nominal":
                    last_label_ah / NOMINAL_CAPACITY_AH,

                "last_capacity_over_first":
                    last_label_ah / first_label_ah,

                "n_capacity_lt_0p05_Ah":
                    int(np.sum(dq_ah < 0.05)),

                "n_capacity_lt_0p50_Ah":
                    int(np.sum(dq_ah < 0.50)),

                "rul_first": float(rul_arr[0]),
                "rul_last": float(rul_arr[-1]),

                "rul_exact_N_to_1":
                    rul_exact_remaining_recorded,

                "integrated_Qdis_first_Ah":
                    qint_first,

                "integrated_Qdis_last_Ah":
                    qint_last,

                "abs_diff_label_vs_integrated_first_Ah":
                    abs(first_label_ah - qint_first),

                "abs_diff_label_vs_integrated_last_Ah":
                    abs(last_label_ah - qint_last),
            }

            cell_rows.append(row)

            for cyc, qraw, qah, r in zip(
                cycle_keys, dq_raw, dq_ah, rul_arr
            ):
                cycle_rows.append({
                    "cell_id": cell_id,
                    "cycle": int(cyc),
                    "dq_raw": float(qraw),
                    "capacity_Ah_if_dq_is_mAh": float(qah),
                    "SOH_vs_1p10_nominal": float(
                        qah / NOMINAL_CAPACITY_AH
                    ),
                    "rul_source": float(r),
                })

            del obj, cell, data, first_df, last_df

    cells = pd.DataFrame(cell_rows)
    cycles = pd.DataFrame(cycle_rows)

    per_cell_file = outdir / "HUST_77cell_audit.csv"
    cycle_file = outdir / "HUST_cycle_labels.csv"
    summary_file = outdir / "HUST_dataset_audit_summary.txt"

    cells.to_csv(per_cell_file, index=False)
    cycles.to_csv(cycle_file, index=False)

    lines = []

    def emit(x=""):
        print(x)
        lines.append(str(x))

    emit()
    emit("=" * 78)
    emit("DATASET-LEVEL SUMMARY")
    emit("=" * 78)

    emit(f"Cells: {len(cells)}")
    emit(f"Cycles: {len(cycles)}")

    emit()
    emit("Cycle-count distribution:")
    emit(cells["n_cycles"].describe().to_string())

    emit()
    emit("All source dq labels interpreted as mAh -> Ah:")
    emit(
        cycles["capacity_Ah_if_dq_is_mAh"]
        .describe(
            percentiles=[
                0.01, 0.05, 0.25,
                0.50, 0.75, 0.95, 0.99
            ]
        )
        .to_string()
    )

    emit()
    emit(
        "Capacity labels < 0.05 Ah: "
        f"{int((cycles['capacity_Ah_if_dq_is_mAh'] < 0.05).sum())}"
    )

    emit(
        "Capacity labels < 0.50 Ah: "
        f"{int((cycles['capacity_Ah_if_dq_is_mAh'] < 0.50).sum())}"
    )

    emit()
    emit("Per-cell capacity minima:")
    emit(cells["capacity_Ah_min"].describe().to_string())

    emit()
    emit("Per-cell capacity maxima:")
    emit(cells["capacity_Ah_max"].describe().to_string())

    emit()
    emit("Per-cell capacity standard deviation:")
    emit(cells["capacity_Ah_std"].describe().to_string())

    emit()
    emit("RUL source-label audit:")
    emit(
        f"Cells with exact N,N-1,...,1 RUL definition: "
        f"{int(cells['rul_exact_N_to_1'].sum())}/{len(cells)}"
    )

    emit()
    emit("Source-label vs independently integrated Qdis:")
    emit(
        "Median |difference| first cycle (Ah): "
        f"{cells['abs_diff_label_vs_integrated_first_Ah'].median():.6f}"
    )
    emit(
        "Median |difference| last cycle  (Ah): "
        f"{cells['abs_diff_label_vs_integrated_last_Ah'].median():.6f}"
    )
    emit(
        "Maximum |difference| first cycle (Ah): "
        f"{cells['abs_diff_label_vs_integrated_first_Ah'].max():.6f}"
    )
    emit(
        "Maximum |difference| last cycle  (Ah): "
        f"{cells['abs_diff_label_vs_integrated_last_Ah'].max():.6f}"
    )

    emit()
    emit("Capacity relative to 1.10 Ah nominal basis:")
    emit(
        "Median first/nominal: "
        f"{cells['first_capacity_over_nominal'].median():.4f}"
    )
    emit(
        "Median last/nominal : "
        f"{cells['last_capacity_over_nominal'].median():.4f}"
    )
    emit(
        "Median last/first   : "
        f"{cells['last_capacity_over_first'].median():.4f}"
    )

    emit()
    emit("Five smallest observed source capacities:")
    smallest = cycles.nsmallest(
        5, "capacity_Ah_if_dq_is_mAh"
    )[
        [
            "cell_id",
            "cycle",
            "dq_raw",
            "capacity_Ah_if_dq_is_mAh",
            "rul_source",
        ]
    ]
    emit(smallest.to_string(index=False))

    emit()
    emit("Five largest observed source capacities:")
    largest = cycles.nlargest(
        5, "capacity_Ah_if_dq_is_mAh"
    )[
        [
            "cell_id",
            "cycle",
            "dq_raw",
            "capacity_Ah_if_dq_is_mAh",
            "rul_source",
        ]
    ]
    emit(largest.to_string(index=False))

    emit()
    emit("Saved:")
    emit(str(per_cell_file))
    emit(str(cycle_file))
    emit(str(summary_file))

    summary_file.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
