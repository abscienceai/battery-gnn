import io
import tarfile
import zipfile
import pickle
import codecs
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path.home() / "alper/EST_R2_field_validation_2026-10"
ARCHIVE = ROOT / "raw/zhang2023/battery_brand3.tar.gz"
OUT = ROOT / "reports"

ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
    ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
    ("_codecs", "encode"): codecs.encode,
}

class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        key = (module, name)
        if key not in ALLOWED:
            raise pickle.UnpicklingError(f"Blocked: {module}.{name}")
        return ALLOWED[key]

def decode(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        names = [n for n in z.namelist() if n.endswith("/data.pkl")]
        if len(names) != 1:
            raise ValueError("Unexpected ZIP structure")
        info = z.getinfo(names[0])
        if info.file_size > 4_000_000:
            raise ValueError("Uncompressed object too large")
        return RestrictedUnpickler(io.BytesIO(z.read(names[0]))).load()

with tarfile.open(ARCHIVE, "r:gz") as tf:
    labels = pd.read_csv(
        tf.extractfile("battery_brand3/label/all_label.csv")
    )

label_map = dict(zip(
    labels["car"].astype(int),
    labels["label"].astype(int)
))

print("Vehicle labels:", labels["label"].value_counts().to_dict())

rows = []
errors = []
example_printed = False
seen_files = 0

with tarfile.open(ARCHIVE, "r|gz") as tf:
    for member in tf:
        if not (
            member.name.startswith("battery_brand3/data/")
            and member.name.endswith(".pkl")
            and member.isfile()
        ):
            continue

        seen_files += 1

        try:
            if member.size > 4_000_000:
                raise ValueError("Archive member too large")

            obj = decode(tf.extractfile(member).read())

            if not isinstance(obj, tuple) or len(obj) != 2:
                raise ValueError("Unexpected record structure")

            arr, meta = obj

            if not isinstance(arr, np.ndarray):
                raise ValueError("Data is not an ndarray")
            if arr.dtype.kind not in "fiub":
                raise ValueError("Unexpected numeric dtype")
            if arr.ndim != 2 or arr.shape[1] != 8:
                raise ValueError(f"Unexpected array shape: {arr.shape}")
            if not isinstance(meta, dict):
                raise ValueError("Missing metadata")

            a = arr.astype(np.float64, copy=False)
            car = int(meta["car"])
            if car not in label_map:
                raise ValueError("Unknown car ID")

            volt = a[:, 0]
            current = a[:, 1]
            soc = a[:, 2]
            vmax = a[:, 3]
            vmin = a[:, 4]
            temp_gap = a[:, 5] - a[:, 6]
            timestamp = a[:, 7]

            valid = (
                np.isfinite(vmax) & np.isfinite(vmin) &
                (vmax >= vmin)
            )
            spread = (vmax - vmin)[valid]

            dt = np.diff(timestamp)
            dt_pos = dt[np.isfinite(dt) & (dt > 0)]

            if not example_printed:
                print("\nEXAMPLE RECORD:", member.name)
                print("Array shape:", a.shape)
                print("Metadata:", meta)
                print("First two measurements:")
                print(a[:2])
                example_printed = True

            def med(x):
                x = np.asarray(x)
                x = x[np.isfinite(x)]
                return float(np.median(x)) if x.size else np.nan

            rows.append({
                "file": Path(member.name).name,
                "car": car,
                "vehicle_fault_label": label_map[car],
                "charge_segment": str(meta.get("charge_segment", "")),
                "n_samples": len(a),
                "median_pack_voltage_raw": med(volt),
                "median_current_raw": med(current),
                "median_soc_raw": med(soc),
                "median_cell_voltage_spread_raw": med(spread),
                "p95_cell_voltage_spread_raw": (
                    float(np.percentile(spread, 95))
                    if len(spread) else np.nan
                ),
                "median_temperature_spread_raw": med(temp_gap),
                "median_positive_timestamp_step_raw": med(dt_pos),
                "nonpositive_timestamp_steps": int(
                    np.sum(np.isfinite(dt) & (dt <= 0))
                ),
                "missing_numeric_values": int(
                    np.size(a) - np.isfinite(a).sum()
                )
            })

        except Exception as exc:
            errors.append((member.name, str(exc)[:130]))

        if seen_files % 5000 == 0:
            print(
                f"Processed {seen_files} records; "
                f"valid={len(rows)}, errors={len(errors)}",
                flush=True
            )

per = pd.DataFrame(rows)
if per.empty:
    raise RuntimeError("No Zhang records decoded")

per.to_csv(OUT / "r2_zhang_segment_profile.csv", index=False)

vehicles = (
    per.groupby(["car", "vehicle_fault_label"], as_index=False)
    .agg(
        segments=("file", "count"),
        total_samples=("n_samples", "sum"),
        median_cell_spread_raw=(
            "median_cell_voltage_spread_raw", "median"
        ),
        median_temperature_spread_raw=(
            "median_temperature_spread_raw", "median"
        ),
        median_time_step_raw=(
            "median_positive_timestamp_step_raw", "median"
        )
    )
)

vehicles.to_csv(OUT / "r2_zhang_vehicle_profile.csv", index=False)

print("\n========== ZHANG SUMMARY ==========")
print("Archive records:", seen_files)
print("Decoded records:", len(per))
print("Errors:", len(errors))
print("Vehicles:", vehicles["car"].nunique())
print("Observed vehicle labels:")
print(vehicles["vehicle_fault_label"].value_counts().to_string())

print("\nSample counts per segment:")
print(per["n_samples"].describe().to_string())

print("\nTimestamp step (raw units):")
print(
    per["median_positive_timestamp_step_raw"]
    .describe().to_string()
)

print("\n========== VEHICLE-LEVEL COMPARISON ==========")
print(
    vehicles.groupby("vehicle_fault_label")
    .agg(
        vehicles=("car", "count"),
        median_segments=("segments", "median"),
        median_cell_spread=(
            "median_cell_spread_raw", "median"
        ),
        median_temperature_spread=(
            "median_temperature_spread_raw", "median"
        ),
        median_timestamp_step=(
            "median_time_step_raw", "median"
        )
    ).to_string()
)

if errors:
    print("\nFirst decoding errors:")
    for name, err in errors[:5]:
        print(name, err)

print("\nOUTPUTS:")
print(OUT / "r2_zhang_segment_profile.csv")
print(OUT / "r2_zhang_vehicle_profile.csv")
print("IMPORTANT: Vehicle faults are NOT verified sensor faults.")
print("Timestamp and voltage units require source verification.")
