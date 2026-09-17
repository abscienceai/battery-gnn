from pathlib import Path
from scipy.io import loadmat
import numpy as np
import json
import os
import re

BASE = Path(os.environ["BASE"])
OFF = Path(os.environ["OFF"])

P001 = OFF / "Processed Experimental Data" / "P001"
PRE = OFF / "Pre-processing Codes"

OUT_JSON = BASE / "data/audit/RWTH_target_policy.json"
OUT_TXT = BASE / "data/audit/RWTH_target_policy_summary.txt"

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)


def scalar(v):
    try:
        a = np.asarray(v).squeeze()
        if np.size(a) != 1:
            return np.nan
        return float(a)
    except Exception:
        return np.nan


def sustained_first(mask, k):
    mask = np.asarray(mask, dtype=bool)

    if len(mask) < k:
        return None

    for i in range(len(mask) - k + 1):
        if mask[i:i+k].all():
            return i

    return None


# ============================================================
# 1. Reconstruct official processed CapDCH1 trajectories
# ============================================================

cells = {}

metadata_records = []

for raw_num in range(2, 50):

    cell_name = f"epSanyo{raw_num:03d}"
    folder = P001 / cell_name

    testsets = []

    for td in sorted(
        [x for x in folder.iterdir() if x.is_dir()],
        key=lambda p: p.name
    ):
        lean_file = td / "lean_data.mat"

        if not lean_file.exists():
            continue

        try:
            lean = loadmat(
                lean_file,
                simplify_cells=True
            )
        except Exception:
            continue

        ah = scalar(
            lean.get("AhEla", np.nan)
        )

        cap = scalar(
            lean.get("CapDCH1", np.nan)
        )

        start = scalar(
            lean.get("StartDate", np.nan)
        )

        prog = lean.get(
            "DigaProgrammName",
            None
        )

        if isinstance(prog, np.ndarray):
            try:
                prog = str(
                    prog.squeeze()
                )
            except Exception:
                prog = None

        testsets.append({
            "name": td.name,
            "start": start,
            "AhEla": ah,
            "CapDCH1": cap,
            "program": (
                str(prog)
                if prog is not None
                else None
            ),
            "path": td,
        })

    # Prefer StartDate chronology, fall back to TS folder name.
    testsets.sort(
        key=lambda r: (
            r["start"]
            if np.isfinite(r["start"])
            else np.inf,
            r["name"],
        )
    )

    cumulative_ah = 0.0
    trajectory = []

    for r in testsets:

        if np.isfinite(r["AhEla"]):
            cumulative_ah += r["AhEla"]

        if not np.isfinite(r["CapDCH1"]):
            continue

        efc_185 = (
            cumulative_ah / 1.85
        )

        trajectory.append({
            "testset": r["name"],
            "CapDCH1_Ah": r["CapDCH1"],
            "cumulative_AhEla_Ah": cumulative_ah,
            "EFC_1p85": efc_185,
            "program": r["program"],
        })

        # Associated official evaluation metadata.
        cp = (
            r["path"]
            / "cell_programm_eval.mat"
        )

        if cp.exists():

            try:
                d = loadmat(
                    cp,
                    simplify_cells=True
                )

                ev = d.get(
                    "cell_programm_eval",
                    {}
                )

                if not isinstance(ev, dict):
                    continue

                capmeta = ev.get(
                    "Cap",
                    {}
                )

                targetmeta = ev.get(
                    "EvalTarget",
                    {}
                )

                metadata_records.append({
                    "cell": raw_num,
                    "testset": r["name"],
                    "NomCap": scalar(
                        ev.get(
                            "NomCap",
                            np.nan
                        )
                    ),
                    "Cap_CRate": (
                        scalar(
                            capmeta.get(
                                "CRate",
                                np.nan
                            )
                        )
                        if isinstance(
                            capmeta,
                            dict
                        )
                        else np.nan
                    ),
                    "Cap_min_span": (
                        scalar(
                            capmeta.get(
                                "min_span",
                                np.nan
                            )
                        )
                        if isinstance(
                            capmeta,
                            dict
                        )
                        else np.nan
                    ),
                    "Cap_max_span": (
                        scalar(
                            capmeta.get(
                                "max_span",
                                np.nan
                            )
                        )
                        if isinstance(
                            capmeta,
                            dict
                        )
                        else np.nan
                    ),
                    "Cap_Type": (
                        str(
                            capmeta.get(
                                "Type",
                                ""
                            )
                        )
                        if isinstance(
                            capmeta,
                            dict
                        )
                        else ""
                    ),
                    "EvalTarget_Cap": (
                        scalar(
                            targetmeta.get(
                                "Cap",
                                np.nan
                            )
                        )
                        if isinstance(
                            targetmeta,
                            dict
                        )
                        else np.nan
                    ),
                })

            except Exception:
                pass

    cells[raw_num] = trajectory


# ============================================================
# 2. Cohort-level target statistics
# ============================================================

counts = {
    cell: len(traj)
    for cell, traj in cells.items()
}

first_caps = {}
last_caps = {}

crossings = {
    1: {},
    2: {},
    3: {},
}

for cell, traj in cells.items():

    caps = np.asarray(
        [
            r["CapDCH1_Ah"]
            for r in traj
        ],
        dtype=float
    )

    if len(caps) == 0:
        continue

    q0 = caps[0]

    first_caps[cell] = q0
    last_caps[cell] = caps[-1]

    soh = caps / q0

    mask = (
        soh <= 0.80
    )

    for k in [1, 2, 3]:

        idx = sustained_first(
            mask,
            k
        )

        crossings[k][cell] = (
            None
            if idx is None
            else {
                "measurement_index": int(idx),
                "testset": traj[idx]["testset"],
                "CapDCH1_Ah": float(caps[idx]),
                "SOH_Q_over_Q0": float(
                    soh[idx]
                ),
                "EFC_1p85": float(
                    traj[idx]["EFC_1p85"]
                ),
            }
        )


# ============================================================
# 3. Evaluation metadata cohort
# ============================================================

nom = np.asarray(
    [
        x["NomCap"]
        for x in metadata_records
        if np.isfinite(
            x["NomCap"]
        )
    ],
    dtype=float
)

crate = np.asarray(
    [
        x["Cap_CRate"]
        for x in metadata_records
        if np.isfinite(
            x["Cap_CRate"]
        )
    ],
    dtype=float
)

vmin = np.asarray(
    [
        x["Cap_min_span"]
        for x in metadata_records
        if np.isfinite(
            x["Cap_min_span"]
        )
    ],
    dtype=float
)

vmax = np.asarray(
    [
        x["Cap_max_span"]
        for x in metadata_records
        if np.isfinite(
            x["Cap_max_span"]
        )
    ],
    dtype=float
)

evalcap = np.asarray(
    [
        x["EvalTarget_Cap"]
        for x in metadata_records
        if np.isfinite(
            x["EvalTarget_Cap"]
        )
    ],
    dtype=float
)

types = sorted(
    set(
        x["Cap_Type"]
        for x in metadata_records
        if x["Cap_Type"]
    )
)


# ============================================================
# 4. Official source-code semantic assertions
# ============================================================

plot_one = (
    OFF
    / "Processed Experimental Data"
    / "PlotOneDataSet.m"
).read_text(
    encoding="utf-8",
    errors="replace"
)

plot_field = (
    OFF
    / "Processed Experimental Data"
    / "PlotDataField.m"
).read_text(
    encoding="utf-8",
    errors="replace"
)

build = (
    OFF
    / "Processed Experimental Data"
    / "build_lifetime_items.m"
).read_text(
    encoding="utf-8",
    errors="replace"
)

plot_capdch_ok = (
    "'AhEla','CapDCH1'"
    in plot_one.replace(" ", "")
)

plot_norm_185_ok = bool(
    re.search(
        r"NominalCapacity\s*=\s*1\.85",
        plot_one
    )
)

plot_cumsum_ok = (
    "cumsum(XDataCell)"
    in plot_field.replace(" ", "")
)

lean_loader_ok = (
    "strcmp(var_name,'lean_data')"
    in build.replace(" ", "")
)


# ============================================================
# 5. Official TDS structure
# ============================================================

tds_file = (
    PRE
    / "Degradation_Prediction_Dataset_ISEA.mat"
)

TDS = np.atleast_1d(
    loadmat(
        tds_file,
        simplify_cells=True
    )["TDS"]
)

tds_ids = sorted(
    set(
        int(x["Cell"])
        for x in TDS
    )
)


# ============================================================
# 6. Assertions
# ============================================================

all_cells_have_cap = (
    len(cells) == 48
    and all(
        counts.get(c, 0) > 0
        for c in range(2, 50)
    )
)

total_cap_records = sum(
    counts.values()
)

first_arr = np.asarray(
    list(
        first_caps.values()
    ),
    dtype=float
)

last_arr = np.asarray(
    list(
        last_caps.values()
    ),
    dtype=float
)

eol_counts = {
    str(k): sum(
        crossings[k][c]
        is not None
        for c in crossings[k]
    )
    for k in [1, 2, 3]
}

assert all_cells_have_cap
assert total_cap_records == 777
assert len(first_arr) == 48

assert len(nom) > 0
assert np.allclose(
    nom,
    2.05,
    atol=1e-12
)

assert len(crate) > 0

assert plot_capdch_ok
assert plot_norm_185_ok
assert plot_cumsum_ok
assert lean_loader_ok

assert len(TDS) == 10686
assert tds_ids == list(
    range(1, 48)
)


# ============================================================
# 7. Canonical methodological policy
# ============================================================

policy = {
    "dataset": "RWTH-2021-04545",

    "raw_cohort": {
        "cell_names": "epSanyo002-epSanyo049",
        "n_cells": 48,
        "CapDCH1_records": total_cap_records,
        "all_48_have_CapDCH1": True,
    },

    "official_capacity_semantics": {
        "capacity_field": "CapDCH1",
        "unit": "Ah",
        "rated_capacity_Ah": 2.05,
        "capacity_evaluation_CRate_median":
            float(np.median(crate)),
        "capacity_evaluation_CRate_min":
            float(np.min(crate)),
        "capacity_evaluation_CRate_max":
            float(np.max(crate)),
        "capacity_evaluation_types":
            types,
        "official_plot_x_field":
            "cumulative AhEla",
        "official_plot_x_normalizer_Ah":
            1.85,
    },

    "revised_benchmark_policy": {
        "RWTH_regression_target":
            "CapDCH1 in Ah",

        "primary_SOH_definition":
            "CapDCH1 / first valid CapDCH1 "
            "for the same raw cell",

        "primary_EOL_definition":
            "first characterization observation "
            "with SOH <= 0.80",

        "EOL_sensitivity":
            "also report 2- and 3-consecutive "
            "characterization-point thresholds",

        "RWTH_life_coordinate":
            "cumulative AhEla / 1.85 Ah "
            "(equivalent-throughput coordinate)",

        "rated_capacity_role":
            "metadata/reference only; not the "
            "primary per-cell SOH denominator",

        "BatteryML_1p11_Ah_role":
            "cycling-window reference only; "
            "not physical rated capacity and "
            "not the primary RWTH SOH denominator",

        "TDS_role":
            "semantic/prospective-trajectory "
            "cross-check only; do not infer "
            "raw cell identities from internal TDS IDs",

        "raw_to_TDS_mapping":
            "unresolved in public release",
    },

    "official_TDS": {
        "samples": int(
            len(TDS)
        ),
        "internal_cell_count":
            len(tds_ids),
        "internal_cell_ids":
            tds_ids,
        "raw_to_TDS_mapping_publicly_exposed":
            False,
    },

    "cohort_statistics": {
        "first_CapDCH1_Ah": {
            "min":
                float(
                    np.min(first_arr)
                ),
            "median":
                float(
                    np.median(first_arr)
                ),
            "mean":
                float(
                    np.mean(first_arr)
                ),
            "max":
                float(
                    np.max(first_arr)
                ),
        },

        "first_CapDCH1_over_rated_2p05": {
            "median":
                float(
                    np.median(
                        first_arr / 2.05
                    )
                )
        },

        "last_CapDCH1_Ah": {
            "min":
                float(
                    np.min(last_arr)
                ),
            "median":
                float(
                    np.median(last_arr)
                ),
            "max":
                float(
                    np.max(last_arr)
                ),
        },

        "EOL_reached_cells": {
            "1_consecutive":
                int(
                    eol_counts["1"]
                ),
            "2_consecutive":
                int(
                    eol_counts["2"]
                ),
            "3_consecutive":
                int(
                    eol_counts["3"]
                ),
        },
    },
}

OUT_JSON.write_text(
    json.dumps(
        policy,
        indent=2
    )
    + "\n",
    encoding="utf-8"
)


summary_lines = [
    "RWTH canonical target-policy audit",
    "",
    f"Raw cells: 48",
    f"CapDCH1 records: {total_cap_records}",
    f"CapDCH1 per cell min/median/max: "
    f"{min(counts.values())} / "
    f"{np.median(list(counts.values())):.1f} / "
    f"{max(counts.values())}",
    "",
    f"First CapDCH1 [Ah] min/median/mean/max: "
    f"{np.min(first_arr):.9f} / "
    f"{np.median(first_arr):.9f} / "
    f"{np.mean(first_arr):.9f} / "
    f"{np.max(first_arr):.9f}",
    "",
    f"Median first CapDCH1 / 2.05 rated: "
    f"{np.median(first_arr / 2.05):.9f}",
    "",
    f"Official capacity C-rate min/median/max: "
    f"{np.min(crate):.9f} / "
    f"{np.median(crate):.9f} / "
    f"{np.max(crate):.9f}",
    "",
    f"Official plot uses AhEla vs CapDCH1: "
    f"{plot_capdch_ok}",
    f"Official plot normalizer is 1.85 Ah: "
    f"{plot_norm_185_ok}",
    f"Official PlotDataField cumulatively sums AhEla: "
    f"{plot_cumsum_ok}",
    "",
    f"TDS samples: {len(TDS)}",
    f"TDS internal cells: {len(tds_ids)}",
    f"TDS internal IDs: {tds_ids[0]}-{tds_ids[-1]}",
    "Raw-to-TDS mapping exposed by public release: False",
    "",
    f"EOL <=80% Q0, 1 consecutive: "
    f"{eol_counts['1']}/48",
    f"EOL <=80% Q0, 2 consecutive: "
    f"{eol_counts['2']}/48",
    f"EOL <=80% Q0, 3 consecutive: "
    f"{eol_counts['3']}/48",
    "",
    "PRIMARY POLICY:",
    "target = CapDCH1 [Ah]",
    "SOH = CapDCH1 / first valid CapDCH1 of same cell",
    "EOL = first SOH <= 0.80 characterization point",
    "report 2/3-consecutive EOL sensitivity",
    "life coordinate = cumulative AhEla / 1.85 Ah",
    "2.05 Ah = rated-capacity metadata/reference",
    "1.11 Ah = separate cycling-window reference only",
    "do not infer raw cell identity from TDS Cell ID",
    "",
    "RWTH_TARGET_POLICY_PASS=True",
]

OUT_TXT.write_text(
    "\n".join(
        summary_lines
    )
    + "\n",
    encoding="utf-8"
)


# ============================================================
# 8. Compact terminal output
# ============================================================

print("=" * 76)
print("RWTH CANONICAL TARGET POLICY AUDIT")
print("=" * 76)

print(
    "raw cells / CapDCH1 records:",
    f"48 / {total_cap_records}"
)

print(
    "CapDCH1 records per cell "
    "min / median / max:",
    min(counts.values()),
    "/",
    f"{np.median(list(counts.values())):.1f}",
    "/",
    max(counts.values())
)

print(
    "first CapDCH1 Ah "
    "min / median / max:",
    f"{np.min(first_arr):.6f}",
    "/",
    f"{np.median(first_arr):.6f}",
    "/",
    f"{np.max(first_arr):.6f}"
)

print(
    "median first CapDCH1 / 2.05:",
    f"{np.median(first_arr / 2.05):.6f}"
)

print(
    "capacity C-rate "
    "min / median / max:",
    f"{np.min(crate):.6f}",
    "/",
    f"{np.median(crate):.6f}",
    "/",
    f"{np.max(crate):.6f}"
)

print(
    "NomCap unique:",
    sorted(
        np.unique(nom).tolist()
    )
)

print(
    "Cap types:",
    types
)

print(
    "official AhEla→cumsum / "
    "CapDCH1 / 1.85 checks:",
    plot_capdch_ok,
    plot_cumsum_ok,
    plot_norm_185_ok
)

print(
    "TDS samples / internal cells:",
    len(TDS),
    "/",
    len(tds_ids)
)

print(
    "TDS internal IDs:",
    f"{tds_ids[0]}-{tds_ids[-1]}"
)

print(
    "80% Q0 EOL reached "
    "1 / 2 / 3 consecutive:",
    eol_counts["1"],
    "/",
    eol_counts["2"],
    "/",
    eol_counts["3"]
)

print()
print(
    "POLICY: target = CapDCH1 [Ah]"
)

print(
    "POLICY: SOH = CapDCH1 / first valid CapDCH1"
)

print(
    "POLICY: 2.05 Ah = rated reference; "
    "1.11 Ah = cycling-window reference only"
)

print(
    "POLICY: do not map internal TDS Cell IDs "
    "onto raw epSanyo IDs"
)

print()
print(
    "JSON:",
    OUT_JSON
)

print(
    "SUMMARY:",
    OUT_TXT
)

print()
print(
    "RWTH_TARGET_POLICY_PASS=True"
)
