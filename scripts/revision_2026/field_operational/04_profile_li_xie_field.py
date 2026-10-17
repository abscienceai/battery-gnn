from pathlib import Path
from zipfile import ZipFile
import re, warnings
import numpy as np
import pandas as pd

R2 = Path.home() / "alper/EST_R2_field_validation_2026-10"
LI = R2 / "raw/li2026/BatteryData.zip"
XIE = R2 / "raw/xie2026/extracted"
OUT = R2 / "reports"

assert LI.is_file(), LI
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(20261017)
rows = []
pools = {}

def analyse(source, segment, filename, V, I, t=None):
    V = np.asarray(V, dtype=np.float64)
    I = np.asarray(I, dtype=np.float64)

    assert V.ndim == 2
    assert V.shape[1] > 1
    assert len(V) == len(I)

    if source == "Li":
        factor = 1000.0
        eligible = (
            np.isfinite(np.diff(I)) &
            (np.abs(np.diff(I)) <= 1.0)
        )
        median_gap = np.nan
        longest = np.nan
    else:
        factor = 1.0
        dt = np.diff(np.asarray(t, dtype=np.float64)) / 1000.0
        eligible = (
            np.isfinite(dt) &
            (dt >= 1) &
            (dt <= 30) &
            np.isfinite(np.diff(I)) &
            (np.abs(np.diff(I)) <= 0.5)
        )

        positive = dt[np.isfinite(dt) & (dt > 0)]
        median_gap = (
            float(np.median(positive))
            if positive.size else np.nan
        )

        run = 1
        longest = 1
        for good in (np.isfinite(dt) & (dt >= 1) & (dt <= 30)):
            run = run + 1 if good else 1
            longest = max(longest, run)

    dv = np.diff(V, axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        common = np.nanmedian(dv, axis=1, keepdims=True)

    residual = (dv - common)[eligible].reshape(-1)
    residual = residual[np.isfinite(residual)] * factor

    if residual.size:
        indices = rng.choice(
            residual.size,
            size=min(400, residual.size),
            replace=False
        )
        pools.setdefault(segment, []).append(
            residual[indices].astype(np.float32)
        )

    rows.append({
        "source": source,
        "segment": segment,
        "file": Path(filename).name,
        "observations": len(V),
        "cell_channels": V.shape[1],
        "retained_transitions": int(eligible.sum()),
        "residual_points": int(residual.size),
        "abs_increment_p50_mV": (
            float(np.percentile(np.abs(residual), 50))
            if residual.size else np.nan
        ),
        "abs_increment_p95_mV": (
            float(np.percentile(np.abs(residual), 95))
            if residual.size else np.nan
        ),
        "median_gap_seconds": median_gap,
        "longest_1to30s_run": longest
    })

print("===== LI: FULL ARCHIVE =====", flush=True)

with ZipFile(LI) as z:
    members = sorted(
        m for m in z.namelist()
        if m.lower().endswith(".csv")
    )

    print("Li CSV files:", len(members), flush=True)

    for k, member in enumerate(members, 1):
        segment = next(
            (
                s for s in [
                    "real_world_02-03",
                    "real_world_03_04",
                    "real_world_05_06"
                ] if s in member
            ),
            None
        )

        if segment is None:
            continue

        with z.open(member) as f:
            d = pd.read_csv(f)

        vc = sorted(
            [c for c in d if re.fullmatch(r"vol_\d+", c)],
            key=lambda c: int(c.split("_")[-1])
        )

        analyse(
            "Li", segment, member,
            d[vc].to_numpy(float),
            pd.to_numeric(
                d["cur"], errors="coerce"
            ).to_numpy(float)
        )

        if k % 100 == 0:
            print(
                f"Li processed {k}/{len(members)}",
                flush=True
            )

print("\n===== XIE: DENSE TRACES =====", flush=True)

candidates = pd.read_csv(
    OUT / "xie_dense_candidates.csv"
)

for filename in candidates["file"].astype(str):
    p = XIE / filename
    assert p.is_file(), p

    d = pd.read_csv(p, low_memory=False)
    d = d.sort_values(
        "dateTime", kind="mergesort"
    )

    vc = sorted(
        [
            c for c in d
            if re.fullmatch(r"batCoreVoltage\d+", c)
        ],
        key=lambda c: int(
            re.search(r"\d+$", c).group()
        )
    )

    analyse(
        "Xie", "Xie_dense", filename,
        d[vc].to_numpy(float),
        pd.to_numeric(
            d["totalCurrent"], errors="coerce"
        ).to_numpy(float),
        pd.to_numeric(
            d["dateTime"], errors="coerce"
        ).to_numpy(float)
    )

per = pd.DataFrame(rows)

per.to_csv(
    OUT / "r2_li_xie_per_file_profile.csv",
    index=False
)

summary = per.groupby(
    ["source", "segment"], as_index=False
).agg(
    n_files=("file", "count"),
    observations=("observations", "sum"),
    retained_transitions=("retained_transitions", "sum"),
    median_file_p95_mV=("abs_increment_p95_mV", "median"),
    min_file_p95_mV=("abs_increment_p95_mV", "min"),
    max_file_p95_mV=("abs_increment_p95_mV", "max")
)

summary.to_csv(
    OUT / "r2_li_xie_profile_summary.csv",
    index=False
)

np.savez_compressed(
    OUT / "r2_field_increment_proxies.npz",
    **{
        key: np.concatenate(group)
        for key, group in pools.items()
    }
)

print("\n===== FIELD PROFILE SUMMARY =====")
print(summary.to_string(index=False))

print("\n===== XIE INDIVIDUAL TRACES =====")
print(
    per[per.source == "Xie"][
        ["file", "retained_transitions",
         "abs_increment_p95_mV", "longest_1to30s_run"]
    ].to_string(index=False)
)

print("\nPROFILE COMPLETE")
print("Output directory:", OUT)
print("IMPORTANT: These are field variability proxies.")
print("They are NOT measured sensor errors.")
print("Li timestamps are unavailable.")
print("Xie voltage units are provisionally interpreted as mV.")
print("Current-change thresholds are exploratory.")
