from pathlib import Path
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import zipfile

import h5py
import numpy as np
import pandas as pd


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

# The audited raw-cycle generator resolves its project root from
# os.environ["BASE"] at import time.
os.environ["BASE"] = str(BASE)

ROOT = (
    BASE
    / "results/revision_2026/dqdv_leakage_control/final_dqdv_v1"
)

PROTOCOL = ROOT / "protocol.json"

OUT = ROOT / "barrier_v1"

INVENTORY = OUT / "sensor_discharge_inventory.csv"
CUTOFFS = OUT / "train_only_prefix_cutoffs.csv"
POLARITY = OUT / "polarity_validation_summary.csv"
MANIFEST = OUT / "duration_polarity_barrier_manifest.json"

CANON = (
    BASE
    / "data/processed/revision_2026/canonical_tasks"
    / "canonical_cycle_supervised_table.csv"
)

ASSIGN = (
    BASE
    / "data/processed/revision_2026/splits_and_scalers"
    / "cell_split_assignments.csv"
)

GENERATOR = (
    REPO
    / "scripts/revision_2026/14_raw_cycle_feature_generator.py"
)

SELF = (
    REPO
    / "scripts/revision_2026/79_dqdv_duration_polarity_barrier.py"
)

EXPECTED_PROTOCOL_SHA = (
    "c455bf7a46f89649f25769a43abd443f171545b06e4740c799df1842c8bbb8e9"
)

EXPECTED_SOH_COUNTS = {
    "HUST": 146122,
    "CALCE": 14298,
    "MATR": 140133,
}

EXPECTED_SOH_TOTAL = 300553

SPLIT_SEEDS = [0, 1, 2, 3, 4]

POLARITY_MIN_COVERAGE = 0.99
POLARITY_MIN_PEARSON_R = 0.80
POLARITY_MAX_MEDIAN_ABS_REL_ERROR = 0.20
POLARITY_MEDIAN_RATIO_MIN = 0.80
POLARITY_MEDIAN_RATIO_MAX = 1.20


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_csv(df, path):
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False, float_format="%.17g")
    os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bool_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)

    m = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
    )

    if m.isna().any():
        raise ValueError("Unparseable boolean values")

    return m.astype(bool)


def sensor_metrics(voltage_V, current_A, time_s):
    V = np.asarray(voltage_V, dtype=np.float64).reshape(-1)
    I = np.asarray(current_A, dtype=np.float64).reshape(-1)
    t = np.asarray(time_s, dtype=np.float64).reshape(-1)

    if not (len(V) == len(I) == len(t)):
        raise ValueError("V/I/t length mismatch")

    finite = (
        np.isfinite(V)
        & np.isfinite(I)
        & np.isfinite(t)
    )

    V = V[finite]
    I = I[finite]
    t = t[finite]

    if len(t) < 2:
        raise ValueError("Too few finite samples")

    dt = np.diff(t)
    I_step = I[1:]

    valid = (
        np.isfinite(dt)
        & np.isfinite(I_step)
        & (dt > 0)
    )

    discharge = valid & (I_step < 0)
    charge = valid & (I_step > 0)

    d_idx = np.flatnonzero(discharge)

    if len(d_idx) == 0:
        raise ValueError("No discharge-contributing interval")

    q_discharge_Ah = float(
        np.sum(
            (-I_step[discharge])
            * dt[discharge]
            / 3600.0
        )
    )

    q_charge_Ah = float(
        np.sum(
            I_step[charge]
            * dt[charge]
            / 3600.0
        )
    )

    discharge_start_s = float(
        t[d_idx[0]]
    )

    discharge_end_s = float(
        t[d_idx[-1] + 1]
    )

    discharge_duration_s = float(
        discharge_end_s
        - discharge_start_s
    )

    if not (
        np.isfinite(q_discharge_Ah)
        and q_discharge_Ah > 0
        and np.isfinite(discharge_duration_s)
        and discharge_duration_s > 0
    ):
        raise ValueError(
            "Invalid discharge integration metrics"
        )

    return {
        "q_discharge_sensor_Ah":
            q_discharge_Ah,

        "q_charge_sensor_Ah":
            q_charge_Ah,

        "discharge_start_s":
            discharge_start_s,

        "discharge_end_s":
            discharge_end_s,

        "discharge_duration_s":
            discharge_duration_s,

        "n_discharge_intervals":
            int(len(d_idx)),

        "n_charge_intervals":
            int(np.sum(charge)),

        "sensor_sample_count":
            int(len(t)),
    }


class CaptureKernel:
    """
    Proxy around the already-audited cycle feature kernel.

    A capture is recorded ONLY after the original primary feature
    extractor succeeds. Therefore captures align one-to-one, in call
    order, with feature_valid=True rows returned by the source adapter.
    """

    def __init__(self, base_kernel):
        self.base = base_kernel
        self.captures = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def extract_primary_cycle_features(
        self,
        voltage_V,
        current_A,
        time_s,
        *args,
        **kwargs,
    ):
        features, qc = (
            self.base.extract_primary_cycle_features(
                voltage_V,
                current_A,
                time_s,
                *args,
                **kwargs,
            )
        )

        metrics = sensor_metrics(
            voltage_V,
            current_A,
            time_s,
        )

        self.captures.append(metrics)

        return features, qc


def captures_to_rows(dataset, df, captures):
    valid = df[
        bool_series(
            df["feature_valid"]
        )
    ].copy()

    if len(valid) != len(captures):
        raise AssertionError(
            (
                dataset,
                "capture/valid mismatch",
                len(valid),
                len(captures),
            )
        )

    out = valid[
        [
            "raw_cell_id",
            "source_member",
            "raw_cycle_key",
            "raw_cycle_position_zero_based",
        ]
    ].reset_index(drop=True)

    cap = pd.DataFrame(captures)

    assert len(out) == len(cap)

    out = pd.concat(
        [
            out,
            cap,
        ],
        axis=1,
    )

    out.insert(
        0,
        "dataset",
        dataset,
    )

    return out


def build_hust(gen, kernel, kernel_hash):
    parts = []

    with zipfile.ZipFile(
        gen.HUST_ZIP,
        "r",
    ) as zf:

        members = sorted(
            [
                m
                for m in zf.namelist()
                if m.lower().endswith(".pkl")
            ]
        )

        if len(members) != 77:
            raise AssertionError(
                f"HUST member count {len(members)}"
            )

        for member in members:

            with zf.open(
                member
            ) as f:
                obj = pickle.load(f)

            if len(obj) != 1:
                raise AssertionError(
                    f"{member}: unexpected object count"
                )

            raw_cell_id = next(
                iter(obj)
            )

            cell = obj[
                raw_cell_id
            ]

            proxy = CaptureKernel(
                kernel
            )

            df = gen.process_hust_cell(
                source_member=member,
                raw_cell_id=raw_cell_id,
                cell=cell,
                kernel=proxy,
                kernel_hash=kernel_hash,
            )

            parts.append(
                captures_to_rows(
                    "HUST",
                    df,
                    proxy.captures,
                )
            )

    return pd.concat(
        parts,
        ignore_index=True,
    )


def build_calce(gen, kernel, kernel_hash):
    parts = []

    for raw_cell_id in gen.CALCE_EXPECTED_CELLS:

        zip_path = (
            gen.CALCE_RAW_DIR
            / f"{raw_cell_id}.zip"
        )

        if not zip_path.is_file():
            raise FileNotFoundError(
                zip_path
            )

        proxy = CaptureKernel(
            kernel
        )

        df = gen.process_calce_cell(
            zip_path=zip_path,
            raw_cell_id=raw_cell_id,
            kernel=proxy,
            kernel_hash=kernel_hash,
            raw_zip_hash=gen.sha256_file(
                zip_path
            ),
        )

        parts.append(
            captures_to_rows(
                "CALCE",
                df,
                proxy.captures,
            )
        )

    return pd.concat(
        parts,
        ignore_index=True,
    )


def build_matr(gen, kernel, kernel_hash):
    parts = []

    for batch, path in gen.MATR_FILES.items():

        with h5py.File(
            path,
            "r",
        ) as f:

            b = f["batch"]

            n_cells = int(
                b["cycles"].shape[0]
            )

            source_member = str(
                path.relative_to(
                    BASE
                )
            )

            for i in range(
                n_cells
            ):

                raw_cell_id = (
                    f"{batch}c{i}"
                )

                cycles = f[
                    b["cycles"][i, 0]
                ]

                proxy = CaptureKernel(
                    kernel
                )

                df = gen.process_matr_cell(
                    f=f,
                    cycles=cycles,
                    source_member=source_member,
                    raw_cell_id=raw_cell_id,
                    kernel=proxy,
                    kernel_hash=kernel_hash,
                )

                parts.append(
                    captures_to_rows(
                        "MATR",
                        df,
                        proxy.captures,
                    )
                )

    return pd.concat(
        parts,
        ignore_index=True,
    )


# ----------------------------------------------------------------------
# Preconditions
# ----------------------------------------------------------------------

assert sha256(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)

assert protocol[
    "status"
] == "FROZEN_BEFORE_DQDV_FEATURE_EXTRACTION"

assert protocol[
    "raw_trace_contract"
][
    "polarity_validation_barrier_required_before_bulk_generation"
] is True

assert protocol[
    "information_horizons"
][
    "PREFIX25"
][
    "individual_test_cycle_endpoint_used"
] is False

assert protocol[
    "information_horizons"
][
    "PREFIX50"
][
    "individual_test_cycle_endpoint_used"
] is False


rel = str(
    SELF.relative_to(
        REPO
    )
)

subprocess.check_call(
    [
        "git",
        "ls-files",
        "--error-unmatch",
        rel,
    ],
    cwd=REPO,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

assert subprocess.run(
    [
        "git",
        "diff",
        "--quiet",
        "HEAD",
        "--",
        rel,
    ],
    cwd=REPO,
).returncode == 0


OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ----------------------------------------------------------------------
# FIRST RAW NUMERIC V/I/t READ occurs through audited adapters below.
# ----------------------------------------------------------------------

gen = load_module(
    GENERATOR,
    "revision2026_raw_cycle_generator_for_dqdv_barrier",
)

kernel = gen.load_kernel()

kernel_hash = protocol[
    "source_anchors"
][
    "cycle_feature_kernel"
][
    "sha256"
]


hust = build_hust(
    gen,
    kernel,
    kernel_hash,
)

calce = build_calce(
    gen,
    kernel,
    kernel_hash,
)

matr = build_matr(
    gen,
    kernel,
    kernel_hash,
)


inventory = pd.concat(
    [
        hust,
        calce,
        matr,
    ],
    ignore_index=True,
)


# Unique sensor trace identity.
id_cols = [
    "dataset",
    "raw_cell_id",
    "raw_cycle_position_zero_based",
]

if inventory[
    id_cols
].duplicated().any():
    raise AssertionError(
        "Duplicate sensor inventory identity"
    )


for c in [
    "q_discharge_sensor_Ah",
    "q_charge_sensor_Ah",
    "discharge_duration_s",
]:

    inventory[
        c
    ] = pd.to_numeric(
        inventory[
            c
        ],
        errors="raise",
    )

    if not np.isfinite(
        inventory[
            c
        ]
    ).all():
        raise AssertionError(
            f"nonfinite {c}"
        )


if not (
    inventory[
        "q_discharge_sensor_Ah"
    ] > 0
).all():
    raise AssertionError(
        "nonpositive discharge Ah"
    )

if not (
    inventory[
        "discharge_duration_s"
    ] > 0
).all():
    raise AssertionError(
        "nonpositive discharge duration"
    )


inventory = inventory.sort_values(
    id_cols,
    kind="mergesort",
).reset_index(
    drop=True
)

atomic_csv(
    inventory,
    INVENTORY,
)


# ----------------------------------------------------------------------
# Identifier-only canonical alignment for cutoff fitting.
# Capacity and SOH columns are NOT read in this branch.
# ----------------------------------------------------------------------

canon_id = pd.read_csv(
    CANON,
    usecols=[
        "dataset",
        "split_group",
        "soh_task_eligible",
        "feature_source_cell_id",
        "feature_raw_cycle_position_zero_based",
    ],
    low_memory=False,
)

canon_id = canon_id[
    canon_id[
        "dataset"
    ].isin(
        [
            "HUST",
            "CALCE",
            "MATR",
        ]
    )
].copy()

canon_id = canon_id[
    bool_series(
        canon_id[
            "soh_task_eligible"
        ]
    )
].copy()


observed_counts = (
    canon_id[
        "dataset"
    ]
    .value_counts()
    .to_dict()
)

if observed_counts != EXPECTED_SOH_COUNTS:
    raise AssertionError(
        (
            "canonical SOH counts",
            observed_counts,
        )
    )

if len(
    canon_id
) != EXPECTED_SOH_TOTAL:
    raise AssertionError(
        "canonical SOH total mismatch"
    )


canon_id[
    "feature_raw_cycle_position_zero_based"
] = pd.to_numeric(
    canon_id[
        "feature_raw_cycle_position_zero_based"
    ],
    errors="raise",
).astype(int)

inventory[
    "raw_cycle_position_zero_based"
] = pd.to_numeric(
    inventory[
        "raw_cycle_position_zero_based"
    ],
    errors="raise",
).astype(int)


aligned = canon_id.merge(
    inventory[
        [
            "dataset",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
            "discharge_duration_s",
        ]
    ],
    left_on=[
        "dataset",
        "feature_source_cell_id",
        "feature_raw_cycle_position_zero_based",
    ],
    right_on=[
        "dataset",
        "raw_cell_id",
        "raw_cycle_position_zero_based",
    ],
    how="left",
    validate="one_to_one",
)


if len(
    aligned
) != EXPECTED_SOH_TOTAL:
    raise AssertionError(
        "alignment row count mismatch"
    )

if aligned[
    "discharge_duration_s"
].isna().any():
    missing = (
        aligned[
            aligned[
                "discharge_duration_s"
            ].isna()
        ][
            "dataset"
        ]
        .value_counts()
        .to_dict()
    )

    raise AssertionError(
        (
            "SOH-eligible rows missing sensor inventory",
            missing,
        )
    )


assign = pd.read_csv(
    ASSIGN,
    usecols=[
        "split_seed",
        "split_group",
        "dataset",
        "split",
    ],
    low_memory=False,
)

assign[
    "split_seed"
] = pd.to_numeric(
    assign[
        "split_seed"
    ],
    errors="raise",
).astype(int)


cutoff_rows = []


for seed in SPLIT_SEEDS:

    a = assign[
        (
            assign[
                "split_seed"
            ] == seed
        )
        & assign[
            "dataset"
        ].isin(
            [
                "HUST",
                "CALCE",
                "MATR",
            ]
        )
    ].copy()

    if a[
        [
            "dataset",
            "split_group",
        ]
    ].duplicated().any():
        raise AssertionError(
            (
                seed,
                "duplicate assignment",
            )
        )

    m = aligned.merge(
        a,
        on=[
            "dataset",
            "split_group",
        ],
        how="left",
        validate="many_to_one",
    )

    if m[
        "split"
    ].isna().any():
        raise AssertionError(
            (
                seed,
                "missing split assignment",
            )
        )

    for dataset in [
        "HUST",
        "CALCE",
        "MATR",
    ]:

        train = m[
            (
                m[
                    "dataset"
                ] == dataset
            )
            & (
                m[
                    "split"
                ] == "train"
            )
        ].copy()

        if len(
            train
        ) == 0:
            raise AssertionError(
                (
                    seed,
                    dataset,
                    "zero train rows",
                )
            )

        median_duration = float(
            np.median(
                train[
                    "discharge_duration_s"
                ].to_numpy(
                    dtype=np.float64
                )
            )
        )

        if not (
            np.isfinite(
                median_duration
            )
            and median_duration > 0
        ):
            raise AssertionError(
                "invalid train median duration"
            )

        cutoff_rows.append({
            "split_seed":
                seed,

            "dataset":
                dataset,

            "fit_split":
                "train_only",

            "n_training_rows":
                int(
                    len(
                        train
                    )
                ),

            "median_training_discharge_duration_s":
                median_duration,

            "prefix25_cutoff_s":
                0.25
                * median_duration,

            "prefix50_cutoff_s":
                0.50
                * median_duration,

            "validation_or_test_rows_used":
                False,

            "individual_cycle_endpoint_used":
                False,
        })


cutoffs = pd.DataFrame(
    cutoff_rows
)

if len(
    cutoffs
) != 15:
    raise AssertionError(
        "cutoff row count mismatch"
    )

atomic_csv(
    cutoffs,
    CUTOFFS,
)


# ----------------------------------------------------------------------
# Separate polarity validation branch.
#
# FIRST CAPACITY-TARGET VALUE READ occurs here, after sensor inventory
# and train-only cutoff fitting are already complete.
#
# Capacity values are used ONLY for validation and never passed into
# sensor_metrics(), the raw adapters, or cutoff calculation.
# ----------------------------------------------------------------------

canon_cap = pd.read_csv(
    CANON,
    usecols=[
        "dataset",
        "soh_task_eligible",
        "capacity_Ah",
        "feature_source_cell_id",
        "feature_raw_cycle_position_zero_based",
    ],
    low_memory=False,
)

canon_cap = canon_cap[
    canon_cap[
        "dataset"
    ].isin(
        [
            "HUST",
            "CALCE",
            "MATR",
        ]
    )
].copy()

canon_cap = canon_cap[
    bool_series(
        canon_cap[
            "soh_task_eligible"
        ]
    )
].copy()

canon_cap[
    "capacity_Ah"
] = pd.to_numeric(
    canon_cap[
        "capacity_Ah"
    ],
    errors="raise",
)

canon_cap[
    "feature_raw_cycle_position_zero_based"
] = pd.to_numeric(
    canon_cap[
        "feature_raw_cycle_position_zero_based"
    ],
    errors="raise",
).astype(int)


pol = canon_cap.merge(
    inventory[
        [
            "dataset",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
            "q_discharge_sensor_Ah",
            "q_charge_sensor_Ah",
        ]
    ],
    left_on=[
        "dataset",
        "feature_source_cell_id",
        "feature_raw_cycle_position_zero_based",
    ],
    right_on=[
        "dataset",
        "raw_cell_id",
        "raw_cycle_position_zero_based",
    ],
    how="left",
    validate="one_to_one",
)


summary_rows = []


for dataset in [
    "HUST",
    "CALCE",
    "MATR",
]:

    d = pol[
        pol[
            "dataset"
        ] == dataset
    ].copy()

    expected_n = EXPECTED_SOH_COUNTS[
        dataset
    ]

    if len(
        d
    ) != expected_n:
        raise AssertionError(
            (
                dataset,
                "polarity row count",
                len(d),
                expected_n,
            )
        )

    cap = d[
        "capacity_Ah"
    ].to_numpy(
        dtype=np.float64
    )

    qd = d[
        "q_discharge_sensor_Ah"
    ].to_numpy(
        dtype=np.float64
    )

    finite = (
        np.isfinite(cap)
        & (cap > 0)
        & np.isfinite(qd)
        & (qd > 0)
    )

    coverage = float(
        np.mean(
            finite
        )
    )

    cap_f = cap[
        finite
    ]

    qd_f = qd[
        finite
    ]

    ratio = (
        qd_f
        / cap_f
    )

    abs_rel = np.abs(
        qd_f
        - cap_f
    ) / cap_f

    pearson = float(
        np.corrcoef(
            cap_f,
            qd_f,
        )[0, 1]
    )

    median_ratio = float(
        np.median(
            ratio
        )
    )

    median_abs_rel = float(
        np.median(
            abs_rel
        )
    )

    passed = bool(
        coverage
        >= POLARITY_MIN_COVERAGE

        and pearson
        >= POLARITY_MIN_PEARSON_R

        and median_abs_rel
        <= POLARITY_MAX_MEDIAN_ABS_REL_ERROR

        and median_ratio
        >= POLARITY_MEDIAN_RATIO_MIN

        and median_ratio
        <= POLARITY_MEDIAN_RATIO_MAX
    )

    summary_rows.append({
        "dataset":
            dataset,

        "n_expected":
            expected_n,

        "n_finite_positive_pairs":
            int(
                np.sum(
                    finite
                )
            ),

        "coverage":
            coverage,

        "pearson_r_capacity_vs_sensor_discharge":
            pearson,

        "median_sensor_to_capacity_ratio":
            median_ratio,

        "median_absolute_relative_error":
            median_abs_rel,

        "coverage_threshold":
            POLARITY_MIN_COVERAGE,

        "pearson_threshold":
            POLARITY_MIN_PEARSON_R,

        "median_abs_rel_error_threshold":
            POLARITY_MAX_MEDIAN_ABS_REL_ERROR,

        "median_ratio_min":
            POLARITY_MEDIAN_RATIO_MIN,

        "median_ratio_max":
            POLARITY_MEDIAN_RATIO_MAX,

        "barrier_pass":
            passed,
    })


polarity = pd.DataFrame(
    summary_rows
)

if not polarity[
    "barrier_pass"
].all():
    atomic_csv(
        polarity,
        POLARITY,
    )

    raise AssertionError(
        "POLARITY_VALIDATION_BARRIER_FAILED"
    )

atomic_csv(
    polarity,
    POLARITY,
)


manifest = {
    "contract":
        "dqdv_duration_polarity_barrier_v1",

    "status":
        "COMPLETE_BEFORE_DQDV_FEATURE_EXTRACTION",

    "protocol_sha256":
        EXPECTED_PROTOCOL_SHA,

    "runner_commit":
        subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=REPO,
            text=True,
        ).strip(),

    "runner_sha256":
        sha256(
            SELF
        ),

    "raw_adapter_sha256":
        sha256(
            GENERATOR
        ),

    "sensor_inventory_rows":
        int(
            len(
                inventory
            )
        ),

    "canonical_soh_alignment_rows":
        EXPECTED_SOH_TOTAL,

    "canonical_soh_alignment_by_dataset":
        EXPECTED_SOH_COUNTS,

    "cutoff_rows":
        15,

    "cutoff_fit_scope":
        "train_only_dataset_x_split_seed",

    "prefix25_definition":
        "0.25 * median training discharge duration",

    "prefix50_definition":
        "0.50 * median training discharge duration",

    "validation_or_test_duration_used_for_cutoff":
        False,

    "individual_test_cycle_endpoint_used_for_cutoff":
        False,

    "polarity_validation_datasets":
        3,

    "polarity_validation_pass":
        True,

    "polarity_thresholds": {
        "minimum_coverage":
            POLARITY_MIN_COVERAGE,

        "minimum_pearson_r":
            POLARITY_MIN_PEARSON_R,

        "maximum_median_absolute_relative_error":
            POLARITY_MAX_MEDIAN_ABS_REL_ERROR,

        "median_ratio_min":
            POLARITY_MEDIAN_RATIO_MIN,

        "median_ratio_max":
            POLARITY_MEDIAN_RATIO_MAX,
    },

    "capacity_values_read":
        True,

    "capacity_values_used_only_after_cutoff_fitting":
        True,

    "capacity_values_passed_to_sensor_feature_functions":
        False,

    "SOH_values_read":
        False,

    "RUL_or_EOL_values_read":
        False,

    "model_metrics_read":
        False,

    "model_states_loaded":
        False,

    "dqdv_features_computed":
        False,

    "models_trained":
        False,

    "numeric_validation_values_surfaced_during_execution":
        False,

    "outputs": {
        "sensor_inventory": {
            "path":
                str(
                    INVENTORY
                ),

            "sha256":
                sha256(
                    INVENTORY
                ),

            "rows":
                int(
                    len(
                        inventory
                    )
                ),
        },

        "train_only_prefix_cutoffs": {
            "path":
                str(
                    CUTOFFS
                ),

            "sha256":
                sha256(
                    CUTOFFS
                ),

            "rows":
                int(
                    len(
                        cutoffs
                    )
                ),
        },

        "polarity_validation_summary": {
            "path":
                str(
                    POLARITY
                ),

            "sha256":
                sha256(
                    POLARITY
                ),

            "rows":
                int(
                    len(
                        polarity
                    )
                ),
        },
    },
}


atomic_json(
    manifest,
    MANIFEST,
)


print("DQDV DURATION / POLARITY BARRIER CLOSURE")
print(
    "sensor_inventory_rows:",
    len(
        inventory
    )
)
print(
    "canonical_soh_alignment_rows:",
    EXPECTED_SOH_TOTAL
)
print(
    "cutoff_rows:",
    len(
        cutoffs
    )
)
print(
    "polarity_datasets:",
    len(
        polarity
    )
)
print(
    "polarity_barrier_pass:",
    bool(
        polarity[
            "barrier_pass"
        ].all()
    )
)
print(
    "cutoffs_train_only:",
    True
)
print(
    "validation_or_test_duration_used:",
    False
)
print(
    "individual_test_cycle_endpoint_used:",
    False
)
print(
    "capacity_values_used_only_for_validation:",
    True
)
print(
    "capacity_values_passed_to_sensor_feature_functions:",
    False
)
print(
    "SOH_values_read:",
    False
)
print(
    "model_metrics_read:",
    False
)
print(
    "dqdv_features_computed:",
    False
)
print(
    "numeric_validation_values_surfaced:",
    False
)
print(
    "execution_manifest_sha256:",
    sha256(
        MANIFEST
    )
)
print(
    "DQDV_DURATION_POLARITY_BARRIER_PASS=True"
)
