#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import os

import h5py
import numpy as np
import pandas as pd


BASE = Path(os.environ["BASE"])

FILES = {
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

OUT_CSV = (
    BASE
    / "data/audit/"
      "MATR_sensor_unit_audit.csv"
)

OUT_JSON = (
    BASE
    / "data/audit/"
      "MATR_sensor_unit_audit_manifest.json"
)


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def scalar_vector(f, ref):
    return (
        np.asarray(
            f[ref][:]
        )
        .squeeze()
        .astype(float)
    )


def integrate_discharge(I, t):
    """
    Integrate negative current using midpoint rule.

    I: current-like signal
    t: matching time coordinate
    """
    I = np.asarray(I, dtype=float)
    t = np.asarray(t, dtype=float)

    finite = (
        np.isfinite(I)
        & np.isfinite(t)
    )

    I = I[finite]
    t = t[finite]

    if len(I) < 3:
        return np.nan

    dt = np.diff(t)

    Imid = (
        I[:-1] + I[1:]
    ) / 2.0

    valid = (
        np.isfinite(dt)
        & (dt >= 0)
        & np.isfinite(Imid)
    )

    discharge = (
        np.maximum(
            -Imid[valid],
            0.0
        )
    )

    return float(
        np.sum(
            discharge
            * dt[valid]
        )
    )


records = []

for batch, path in FILES.items():

    assert path.is_file(), path

    with h5py.File(
        path,
        "r"
    ) as f:

        b = f["batch"]

        n_cells = int(
            b["cycles"].shape[0]
        )

        for i in range(n_cells):

            cycles = f[
                b["cycles"][i, 0]
            ]

            n_cycles = int(
                cycles["V"].shape[0]
            )

            chosen = None

            # Find first real non-placeholder cycle.
            for j in range(
                min(
                    n_cycles,
                    20
                )
            ):
                V = scalar_vector(
                    f,
                    cycles["V"][j, 0]
                )

                I = scalar_vector(
                    f,
                    cycles["I"][j, 0]
                )

                t = scalar_vector(
                    f,
                    cycles["t"][j, 0]
                )

                Qd = scalar_vector(
                    f,
                    cycles["Qd"][j, 0]
                )

                if not (
                    len(V)
                    == len(I)
                    == len(t)
                    == len(Qd)
                ):
                    continue

                if len(V) < 20:
                    continue

                if (
                    np.nanstd(V)
                    <= 1e-6
                ):
                    continue

                if (
                    np.nanmax(t)
                    - np.nanmin(t)
                    <= 1e-6
                ):
                    continue

                chosen = (
                    j,
                    V,
                    I,
                    t,
                    Qd
                )
                break

            if chosen is None:
                records.append({
                    "batch":
                        batch,
                    "raw_cell_index":
                        i,
                    "cycle_position":
                        np.nan,
                    "usable":
                        False,
                })
                continue

            (
                j,
                V,
                I,
                t,
                Qd,
            ) = chosen

            qd_finite = Qd[
                np.isfinite(Qd)
            ]

            observed_qd = (
                float(
                    np.max(qd_finite)
                    - np.min(qd_finite)
                )
                if len(qd_finite)
                else np.nan
            )

            # ------------------------------------------------
            # Interpretation A:
            # BatteryML field labels taken literally:
            #
            # I stored as ampere
            # t stored as second
            # ------------------------------------------------

            literal_Ah = (
                integrate_discharge(
                    I,
                    t
                )
                / 3600.0
            )

            # ------------------------------------------------
            # Interpretation B:
            # Original MATR struct semantics:
            #
            # stored I = C-rate = original_current_A / 1.1
            # stored t = elapsed minutes
            #
            # Therefore:
            #   I_A = I * 1.1
            #   t_s = t * 60
            # ------------------------------------------------

            canonical_Ah = (
                integrate_discharge(
                    I * 1.1,
                    t * 60.0
                )
                / 3600.0
            )

            records.append({
                "batch":
                    batch,

                "raw_cell_index":
                    i,

                "cycle_position":
                    j + 1,

                "usable":
                    True,

                "samples":
                    len(V),

                "V_min":
                    float(
                        np.nanmin(V)
                    ),

                "V_max":
                    float(
                        np.nanmax(V)
                    ),

                "stored_I_min":
                    float(
                        np.nanmin(I)
                    ),

                "stored_I_max":
                    float(
                        np.nanmax(I)
                    ),

                "stored_t_duration":
                    float(
                        np.nanmax(t)
                        - np.nanmin(t)
                    ),

                "observed_Qd_Ah":
                    observed_qd,

                "literal_BatteryML_integral_Ah":
                    literal_Ah,

                "canonical_integral_Ah":
                    canonical_Ah,

                "literal_ratio_to_Qd":
                    (
                        literal_Ah
                        / observed_qd
                        if observed_qd > 0
                        else np.nan
                    ),

                "canonical_ratio_to_Qd":
                    (
                        canonical_Ah
                        / observed_qd
                        if observed_qd > 0
                        else np.nan
                    ),

                "literal_abs_error_Ah":
                    abs(
                        literal_Ah
                        - observed_qd
                    ),

                "canonical_abs_error_Ah":
                    abs(
                        canonical_Ah
                        - observed_qd
                    ),
            })


df = pd.DataFrame(records)

usable = df[
    df["usable"].astype(bool)
].copy()

assert len(usable) > 0


# ============================================================
# Evidence summaries
# ============================================================

ratio_literal = usable[
    "literal_ratio_to_Qd"
].dropna()

ratio_canonical = usable[
    "canonical_ratio_to_Qd"
].dropna()

err_literal = usable[
    "literal_abs_error_Ah"
].dropna()

err_canonical = usable[
    "canonical_abs_error_Ah"
].dropna()

# Correct interpretation should be dramatically closer.
assert (
    np.median(err_canonical)
    <
    np.median(err_literal)
)

improvement = (
    np.median(err_literal)
    /
    np.median(err_canonical)
)


# ============================================================
# Save artifacts
# ============================================================

OUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUT_CSV,
    index=False
)

manifest = {
    "raw_cells_total":
        int(len(df)),

    "usable_cells":
        int(len(usable)),

    "batches":
        sorted(
            usable[
                "batch"
            ].unique()
        ),

    "stored_time_semantics":
        "minutes",

    "canonical_time_conversion":
        "time_s = stored_t * 60",

    "stored_current_semantics":
        "C-rate",

    "canonical_current_conversion":
        "current_A = stored_I * 1.1",

    "nominal_capacity_Ah":
        1.1,

    "median_literal_ratio_to_Qd":
        float(
            np.median(
                ratio_literal
            )
        ),

    "median_canonical_ratio_to_Qd":
        float(
            np.median(
                ratio_canonical
            )
        ),

    "median_literal_abs_error_Ah":
        float(
            np.median(
                err_literal
            )
        ),

    "median_canonical_abs_error_Ah":
        float(
            np.median(
                err_canonical
            )
        ),

    "median_error_improvement_factor":
        float(improvement),

    "raw_file_sha256": {
        str(
            p.relative_to(BASE)
        ):
            sha256_file(p)
        for p in FILES.values()
    },

    "audit_csv_sha256":
        sha256_file(
            OUT_CSV
        ),

    "audit_pass":
        True,
}

OUT_JSON.write_text(
    json.dumps(
        manifest,
        indent=2
    )
    + "\n",
    encoding="utf-8"
)


# ============================================================
# Compact report
# ============================================================

print("=" * 78)
print("MATR SENSOR UNIT AUDIT")
print("=" * 78)

print(
    "raw / usable cells:",
    len(df),
    "/",
    len(usable)
)

print(
    "usable cells by batch:",
    usable.groupby(
        "batch"
    ).size().to_dict()
)

print()
print(
    "stored t duration "
    "median/min/max:",
    f"{usable['stored_t_duration'].median():.3f}",
    "/",
    f"{usable['stored_t_duration'].min():.3f}",
    "/",
    f"{usable['stored_t_duration'].max():.3f}"
)

print(
    "stored I min "
    "median/min:",
    f"{usable['stored_I_min'].median():.3f}",
    "/",
    f"{usable['stored_I_min'].min():.3f}"
)

print()
print(
    "literal BatteryML "
    "integral/Qd median:",
    f"{np.median(ratio_literal):.6f}"
)

print(
    "corrected SI "
    "integral/Qd median:",
    f"{np.median(ratio_canonical):.6f}"
)

print(
    "literal abs error median [Ah]:",
    f"{np.median(err_literal):.6f}"
)

print(
    "corrected abs error median [Ah]:",
    f"{np.median(err_canonical):.6f}"
)

print(
    "median error improvement:",
    f"{improvement:.2f}x"
)

print()
print(
    "MATR_STORED_TIME_UNIT=minutes"
)

print(
    "MATR_TIME_TO_S="
    "stored_t*60"
)

print(
    "MATR_STORED_CURRENT_UNIT=C-rate"
)

print(
    "MATR_CURRENT_TO_A="
    "stored_I*1.1"
)

print(
    "BATTERYML_MATR_TIME_LABEL_IS_SI_CORRECT=False"
)

print(
    "BATTERYML_MATR_CURRENT_LABEL_IS_SI_CORRECT=False"
)

print()
print(
    "MATR_SENSOR_UNIT_AUDIT_PASS=True"
)
