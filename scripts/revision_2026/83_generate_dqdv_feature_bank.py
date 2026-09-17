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

os.environ["BASE"] = str(BASE)

ROOT = (
    BASE
    / "results/revision_2026/dqdv_leakage_control/final_dqdv_v1"
)

PROTOCOL = ROOT / "protocol.json"

PROVENANCE = (
    ROOT
    / "barrier_v2/duration_provenance_barrier_manifest.json"
)

INVENTORY = (
    ROOT
    / "barrier_v1/sensor_discharge_inventory.csv"
)

CUTOFFS = (
    ROOT
    / "barrier_v1/train_only_prefix_cutoffs.csv"
)

RAW_FEATURE_DIR = (
    BASE
    / "data/processed/revision_2026/raw_cycle_features"
)

OUT = ROOT / "features_v1"

MANIFEST = OUT / "dqdv_feature_bank_manifest.json"

GENERATOR = (
    REPO
    / "scripts/revision_2026/14_raw_cycle_feature_generator.py"
)

SELF = (
    REPO
    / "scripts/revision_2026/83_generate_dqdv_feature_bank.py"
)

EXPECTED_PROTOCOL_SHA = (
    "c455bf7a46f89649f25769a43abd443f171545b06e4740c799df1842c8bbb8e9"
)

EXPECTED_PROVENANCE_SHA = (
    "2d4b67d3d1debea5950b399717e22b7db5c1e5df4c583a1f4fbf85be1f1a32a1"
)

DATASETS = [
    "HUST",
    "CALCE",
    "MATR",
]

SPLIT_SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

FEATURES = [
    "ica_mean_Ah_per_V",
    "ica_std_Ah_per_V",
    "ica_q10_Ah_per_V",
    "ica_q25_Ah_per_V",
    "ica_median_Ah_per_V",
    "ica_q75_Ah_per_V",
    "ica_q90_Ah_per_V",
    "ica_max_Ah_per_V",
    "ica_area_Ah",
    "ica_peak_voltage_V",
]

BIN_WIDTH_V = 0.01
BIN_ORIGIN_V = 0.0

MIN_INTERVALS = 20
MIN_BINS = 5


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(
                1024 * 1024
            )

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def bool_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)

    x = (
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

    if x.isna().any():
        raise ValueError(
            "Unparseable boolean values"
        )

    return x.astype(bool)


def atomic_csv(df, path):
    tmp = Path(
        str(path) + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
        float_format="%.17g",
    )

    os.replace(
        tmp,
        path,
    )


def atomic_json(obj, path):
    tmp = Path(
        str(path) + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def invalid_result(
    reason,
    integrated_Ah=np.nan,
    n_intervals=0,
    n_bins=0,
):
    out = {
        "valid":
            False,

        "failure_reason":
            str(
                reason
            ),

        "integrated_discharge_Ah_qc":
            integrated_Ah,

        "discharge_intervals_qc":
            int(
                n_intervals
            ),

        "occupied_voltage_bins_qc":
            int(
                n_bins
            ),
    }

    for f in FEATURES:
        out[
            f
        ] = np.nan

    return out


def dqdv_features(
    voltage_V,
    current_A,
    time_s,
    cutoff_s=None,
):
    V = np.asarray(
        voltage_V,
        dtype=np.float64,
    ).reshape(-1)

    I = np.asarray(
        current_A,
        dtype=np.float64,
    ).reshape(-1)

    t = np.asarray(
        time_s,
        dtype=np.float64,
    ).reshape(-1)

    if not (
        len(V)
        == len(I)
        == len(t)
    ):
        return invalid_result(
            "V_I_t_length_mismatch"
        )

    finite = (
        np.isfinite(V)
        & np.isfinite(I)
        & np.isfinite(t)
    )

    V = V[
        finite
    ]

    I = I[
        finite
    ]

    t = t[
        finite
    ]

    if len(
        t
    ) < 2:
        return invalid_result(
            "too_few_finite_samples"
        )

    dt = np.diff(
        t
    )

    t0 = t[
        :-1
    ]

    t1 = t[
        1:
    ]

    V0 = V[
        :-1
    ]

    V1 = V[
        1:
    ]

    I_step = I[
        1:
    ]

    valid_interval = (
        np.isfinite(dt)
        & np.isfinite(I_step)
        & np.isfinite(V0)
        & np.isfinite(V1)
        & (dt > 0)
    )

    discharge = (
        valid_interval
        & (I_step < 0)
    )

    idx = np.flatnonzero(
        discharge
    )

    if len(
        idx
    ) == 0:
        return invalid_result(
            "no_discharge_interval"
        )

    onset = float(
        t0[
            idx[0]
        ]
    )

    if cutoff_s is None:
        selected = discharge.copy()

        effective_dt = dt[
            selected
        ]

        V_start = V0[
            selected
        ]

        V_end = V1[
            selected
        ]

        I_selected = I_step[
            selected
        ]

    else:
        cutoff_s = float(
            cutoff_s
        )

        if not (
            np.isfinite(
                cutoff_s
            )
            and cutoff_s > 0
        ):
            return invalid_result(
                "invalid_cutoff"
            )

        absolute_cutoff = (
            onset
            + cutoff_s
        )

        selected = (
            discharge
            & (
                t0
                < absolute_cutoff
            )
        )

        if not np.any(
            selected
        ):
            return invalid_result(
                "no_discharge_before_cutoff"
            )

        s_t0 = t0[
            selected
        ]

        s_t1 = t1[
            selected
        ]

        s_dt = dt[
            selected
        ]

        V_start = V0[
            selected
        ]

        original_V_end = V1[
            selected
        ]

        I_selected = I_step[
            selected
        ]

        clipped_end = np.minimum(
            s_t1,
            absolute_cutoff,
        )

        effective_dt = (
            clipped_end
            - s_t0
        )

        keep = (
            effective_dt
            > 0
        )

        effective_dt = effective_dt[
            keep
        ]

        V_start = V_start[
            keep
        ]

        original_V_end = (
            original_V_end[
                keep
            ]
        )

        I_selected = I_selected[
            keep
        ]

        s_dt = s_dt[
            keep
        ]

        fraction = (
            effective_dt
            / s_dt
        )

        V_end = (
            V_start
            + fraction
            * (
                original_V_end
                - V_start
            )
        )

    if len(
        effective_dt
    ) == 0:
        return invalid_result(
            "no_positive_duration_discharge_interval"
        )

    dq = (
        -I_selected
        * effective_dt
        / 3600.0
    )

    good = (
        np.isfinite(dq)
        & (dq > 0)
        & np.isfinite(V_start)
        & np.isfinite(V_end)
    )

    dq = dq[
        good
    ]

    V_start = V_start[
        good
    ]

    V_end = V_end[
        good
    ]

    n_intervals = int(
        len(
            dq
        )
    )

    integrated = float(
        np.sum(
            dq
        )
    ) if n_intervals else np.nan

    if n_intervals == 0:
        return invalid_result(
            "no_positive_charge_interval",
            integrated_Ah=integrated,
        )

    V_mid = (
        0.5
        * (
            V_start
            + V_end
        )
    )

    bin_index = np.floor(
        (
            V_mid
            - BIN_ORIGIN_V
        )
        / BIN_WIDTH_V
    ).astype(
        np.int64
    )

    unique_bins, inverse = np.unique(
        bin_index,
        return_inverse=True,
    )

    bin_charge = np.bincount(
        inverse,
        weights=dq,
    ).astype(
        np.float64
    )

    positive_bin = (
        np.isfinite(
            bin_charge
        )
        & (
            bin_charge
            > 0
        )
    )

    unique_bins = unique_bins[
        positive_bin
    ]

    bin_charge = bin_charge[
        positive_bin
    ]

    n_bins = int(
        len(
            bin_charge
        )
    )

    if n_intervals < MIN_INTERVALS:
        return invalid_result(
            "too_few_discharge_intervals",
            integrated_Ah=integrated,
            n_intervals=n_intervals,
            n_bins=n_bins,
        )

    if n_bins < MIN_BINS:
        return invalid_result(
            "too_few_occupied_voltage_bins",
            integrated_Ah=integrated,
            n_intervals=n_intervals,
            n_bins=n_bins,
        )

    ica = (
        bin_charge
        / BIN_WIDTH_V
    )

    if not np.isfinite(
        ica
    ).all():
        return invalid_result(
            "nonfinite_ica",
            integrated_Ah=integrated,
            n_intervals=n_intervals,
            n_bins=n_bins,
        )

    centers = (
        BIN_ORIGIN_V
        + (
            unique_bins.astype(
                np.float64
            )
            + 0.5
        )
        * BIN_WIDTH_V
    )

    peak_value = float(
        np.max(
            ica
        )
    )

    peak_centers = centers[
        ica
        == peak_value
    ]

    peak_voltage = float(
        np.max(
            peak_centers
        )
    )

    q = np.quantile(
        ica,
        [
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
        ],
    )

    out = {
        "valid":
            True,

        "failure_reason":
            "",

        "integrated_discharge_Ah_qc":
            integrated,

        "discharge_intervals_qc":
            n_intervals,

        "occupied_voltage_bins_qc":
            n_bins,

        "ica_mean_Ah_per_V":
            float(
                np.mean(
                    ica
                )
            ),

        "ica_std_Ah_per_V":
            float(
                np.std(
                    ica,
                    ddof=0,
                )
            ),

        "ica_q10_Ah_per_V":
            float(
                q[
                    0
                ]
            ),

        "ica_q25_Ah_per_V":
            float(
                q[
                    1
                ]
            ),

        "ica_median_Ah_per_V":
            float(
                q[
                    2
                ]
            ),

        "ica_q75_Ah_per_V":
            float(
                q[
                    3
                ]
            ),

        "ica_q90_Ah_per_V":
            float(
                q[
                    4
                ]
            ),

        "ica_max_Ah_per_V":
            peak_value,

        "ica_area_Ah":
            integrated,

        "ica_peak_voltage_V":
            peak_voltage,
    }

    assert set(
        FEATURES
    ).issubset(
        out
    )

    return out


def prefixed(
    prefix,
    result,
):
    out = {}

    for k, v in result.items():
        out[
            f"{prefix}_{k}"
        ] = v

    return out


class DQDVKernel:
    def __init__(
        self,
        base_kernel,
        dataset,
        cutoff_lookup,
    ):
        self.base = base_kernel
        self.dataset = dataset
        self.cutoff_lookup = cutoff_lookup
        self.captures = []

    def __getattr__(
        self,
        name,
    ):
        return getattr(
            self.base,
            name,
        )

    def extract_primary_cycle_features(
        self,
        voltage_V,
        current_A,
        time_s,
        *args,
        **kwargs,
    ):
        features, qc = (
            self.base
            .extract_primary_cycle_features(
                voltage_V,
                current_A,
                time_s,
                *args,
                **kwargs,
            )
        )

        row = {}

        row.update(
            prefixed(
                "full",
                dqdv_features(
                    voltage_V,
                    current_A,
                    time_s,
                    cutoff_s=None,
                ),
            )
        )

        for seed in SPLIT_SEEDS:

            c = self.cutoff_lookup[
                (
                    seed,
                    self.dataset,
                )
            ]

            row.update(
                prefixed(
                    f"s{seed}_p25",
                    dqdv_features(
                        voltage_V,
                        current_A,
                        time_s,
                        cutoff_s=c[
                            "prefix25_cutoff_s"
                        ],
                    ),
                )
            )

            row.update(
                prefixed(
                    f"s{seed}_p50",
                    dqdv_features(
                        voltage_V,
                        current_A,
                        time_s,
                        cutoff_s=c[
                            "prefix50_cutoff_s"
                        ],
                    ),
                )
            )

        self.captures.append(
            row
        )

        return features, qc


def captures_to_df(
    dataset,
    source_df,
    captures,
):
    valid = source_df[
        bool_series(
            source_df[
                "feature_valid"
            ]
        )
    ].copy()

    if len(
        valid
    ) != len(
        captures
    ):
        raise AssertionError(
            (
                dataset,
                "capture mismatch",
                len(
                    valid
                ),
                len(
                    captures
                ),
            )
        )

    ids = valid[
        [
            "raw_cell_id",
            "source_member",
            "raw_cycle_key",
            "raw_cycle_position_zero_based",
        ]
    ].reset_index(
        drop=True
    )

    cap = pd.DataFrame(
        captures
    )

    assert len(
        ids
    ) == len(
        cap
    )

    out = pd.concat(
        [
            ids,
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


def build_hust(
    gen,
    base_kernel,
    kernel_hash,
    cutoff_lookup,
):
    parts = []

    with zipfile.ZipFile(
        gen.HUST_ZIP,
        "r",
    ) as zf:

        members = sorted(
            [
                m
                for m in zf.namelist()
                if m.lower().endswith(
                    ".pkl"
                )
            ]
        )

        assert len(
            members
        ) == 77

        for member in members:

            with zf.open(
                member
            ) as f:
                obj = pickle.load(
                    f
                )

            assert len(
                obj
            ) == 1

            raw_cell_id = next(
                iter(
                    obj
                )
            )

            proxy = DQDVKernel(
                base_kernel,
                "HUST",
                cutoff_lookup,
            )

            df = gen.process_hust_cell(
                source_member=member,
                raw_cell_id=raw_cell_id,
                cell=obj[
                    raw_cell_id
                ],
                kernel=proxy,
                kernel_hash=kernel_hash,
            )

            parts.append(
                captures_to_df(
                    "HUST",
                    df,
                    proxy.captures,
                )
            )

    return pd.concat(
        parts,
        ignore_index=True,
    )


def build_calce(
    gen,
    base_kernel,
    kernel_hash,
    cutoff_lookup,
):
    parts = []

    for raw_cell_id in (
        gen.CALCE_EXPECTED_CELLS
    ):

        zip_path = (
            gen.CALCE_RAW_DIR
            / f"{raw_cell_id}.zip"
        )

        proxy = DQDVKernel(
            base_kernel,
            "CALCE",
            cutoff_lookup,
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
            captures_to_df(
                "CALCE",
                df,
                proxy.captures,
            )
        )

    return pd.concat(
        parts,
        ignore_index=True,
    )


def build_matr(
    gen,
    base_kernel,
    kernel_hash,
    cutoff_lookup,
):
    parts = []

    for batch, path in (
        gen.MATR_FILES.items()
    ):

        with h5py.File(
            path,
            "r",
        ) as f:

            b = f[
                "batch"
            ]

            n_cells = int(
                b[
                    "cycles"
                ].shape[
                    0
                ]
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
                    b[
                        "cycles"
                    ][
                        i,
                        0
                    ]
                ]

                proxy = DQDVKernel(
                    base_kernel,
                    "MATR",
                    cutoff_lookup,
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
                    captures_to_df(
                        "MATR",
                        df,
                        proxy.captures,
                    )
                )

    return pd.concat(
        parts,
        ignore_index=True,
    )


# ------------------------------------------------------------------
# Frozen anchors.
# ------------------------------------------------------------------

assert sha256(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

assert sha256(
    PROVENANCE
) == EXPECTED_PROVENANCE_SHA

protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)

prov = json.loads(
    PROVENANCE.read_text(
        encoding="utf-8"
    )
)

assert prov[
    "status"
] == "PASS_BEFORE_DQDV_FEATURE_EXTRACTION"

assert prov[
    "all_datasets_direct_agreement_pass"
] is True

assert prov[
    "capacity_values_passed_to_feature_generator"
] is False


# Runner must be tracked and clean.
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


# ------------------------------------------------------------------
# Load only non-target cutoff and inventory artifacts.
# ------------------------------------------------------------------

cut = pd.read_csv(
    CUTOFFS,
    low_memory=False,
)

assert len(
    cut
) == 15

cutoff_lookup = {}

for _, r in cut.iterrows():

    assert str(
        r[
            "fit_split"
        ]
    ) == "train_only"

    assert not bool(
        r[
            "validation_or_test_rows_used"
        ]
    )

    assert not bool(
        r[
            "individual_cycle_endpoint_used"
        ]
    )

    key = (
        int(
            r[
                "split_seed"
            ]
        ),
        str(
            r[
                "dataset"
            ]
        ),
    )

    cutoff_lookup[
        key
    ] = {
        "prefix25_cutoff_s":
            float(
                r[
                    "prefix25_cutoff_s"
                ]
            ),

        "prefix50_cutoff_s":
            float(
                r[
                    "prefix50_cutoff_s"
                ]
            ),
    }


assert set(
    cutoff_lookup
) == {
    (
        seed,
        dataset,
    )
    for seed in SPLIT_SEEDS
    for dataset in DATASETS
}


inventory = pd.read_csv(
    INVENTORY,
    usecols=[
        "dataset",
        "raw_cell_id",
        "raw_cycle_position_zero_based",
        "q_discharge_sensor_Ah",
    ],
    low_memory=False,
)

inventory[
    "raw_cycle_position_zero_based"
] = pd.to_numeric(
    inventory[
        "raw_cycle_position_zero_based"
    ],
    errors="raise",
).astype(int)


inventory_counts = (
    inventory[
        "dataset"
    ]
    .value_counts()
    .to_dict()
)

assert inventory_counts == {
    "HUST": 146122,
    "CALCE": 18715,
    "MATR": 154365,
}

assert int(
    sum(
        inventory_counts.values()
    )
) == 319202


def load_primary_valid_ids(
    dataset,
):
    path = (
        RAW_FEATURE_DIR
        / f"{dataset}_raw_cycle_features.csv"
    )

    df = pd.read_csv(
        path,
        usecols=[
            "dataset",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
            "feature_valid",
        ],
        low_memory=False,
    )

    df[
        "raw_cycle_position_zero_based"
    ] = pd.to_numeric(
        df[
            "raw_cycle_position_zero_based"
        ],
        errors="raise",
    ).astype(int)

    df = df[
        bool_series(
            df[
                "feature_valid"
            ]
        )
    ][
        [
            "dataset",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
        ]
    ].copy()

    if df.duplicated().any():
        raise AssertionError(
            f"{dataset}: duplicate primary-valid identity"
        )

    return df


primary_ids_by_dataset = {
    dataset:
        load_primary_valid_ids(
            dataset
        )
    for dataset in DATASETS
}


primary_valid_counts = {
    dataset:
        int(
            len(
                primary_ids_by_dataset[
                    dataset
                ]
            )
        )
    for dataset in DATASETS
}


assert primary_valid_counts == {
    "HUST": 146122,
    "CALCE": 18804,
    "MATR": 154365,
}

assert int(
    sum(
        primary_valid_counts.values()
    )
) == 319291


# Frozen sensor inventory must be a subset of the
# audited primary-feature-valid universe.
for dataset in DATASETS:

    primary = primary_ids_by_dataset[
        dataset
    ]

    inv = inventory[
        inventory[
            "dataset"
        ] == dataset
    ][
        [
            "dataset",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
        ]
    ]

    subset_check = inv.merge(
        primary,
        on=[
            "dataset",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
        ],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not (
        subset_check[
            "_merge"
        ] == "both"
    ).all():
        raise AssertionError(
            f"{dataset}: sensor inventory is not "
            "a subset of primary-valid rows"
        )


# ------------------------------------------------------------------
# Reopen raw sensor sources. No target table is loaded.
# ------------------------------------------------------------------

gen = load_module(
    GENERATOR,
    "revision2026_dqdv_feature_generator_adapter",
)

base_kernel = (
    gen.load_kernel()
)

kernel_hash = protocol[
    "source_anchors"
][
    "cycle_feature_kernel"
][
    "sha256"
]


OUT.mkdir(
    parents=True,
    exist_ok=True,
)


builders = {
    "HUST":
        build_hust,

    "CALCE":
        build_calce,

    "MATR":
        build_matr,
}


output_info = {}


for dataset in DATASETS:

    path = (
        OUT
        / f"{dataset}_dqdv_feature_bank.csv"
    )

    primary_expected = (
        primary_ids_by_dataset[
            dataset
        ]
    )

    inventory_expected = inventory[
        inventory[
            "dataset"
        ] == dataset
    ].copy()

    # Safe per-dataset resume.
    if path.is_file():

        df = pd.read_csv(
            path,
            low_memory=False,
        )

        required_ids = {
            "dataset",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
            "full_valid",
            "full_integrated_discharge_Ah_qc",
        }

        if not required_ids.issubset(
            df.columns
        ):
            raise AssertionError(
                f"{dataset}: unusable existing shard"
            )

        reused = True

    else:

        df = builders[
            dataset
        ](
            gen,
            base_kernel,
            kernel_hash,
            cutoff_lookup,
        )

        reused = False


    df[
        "raw_cycle_position_zero_based"
    ] = pd.to_numeric(
        df[
            "raw_cycle_position_zero_based"
        ],
        errors="raise",
    ).astype(int)


    # --------------------------------------------------------------
    # Feature-bank universe is the audited primary-feature-valid set.
    # --------------------------------------------------------------

    primary_check = df[
        [
            "dataset",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
        ]
    ].merge(
        primary_expected,
        on=[
            "dataset",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
        ],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not (
        primary_check[
            "_merge"
        ] == "both"
    ).all():
        raise AssertionError(
            f"{dataset}: primary-valid identity mismatch"
        )

    if len(
        df
    ) != int(
        primary_valid_counts[
            dataset
        ]
    ):
        raise AssertionError(
            f"{dataset}: primary-valid row mismatch"
        )


    # --------------------------------------------------------------
    # Frozen sensor inventory is a discharge-integrable subset.
    # Full dQ/dV integration must reproduce it on that subset.
    # --------------------------------------------------------------

    regression = inventory_expected.merge(
        df[
            [
                "dataset",
                "raw_cell_id",
                "raw_cycle_position_zero_based",
                "full_integrated_discharge_Ah_qc",
            ]
        ],
        on=[
            "dataset",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
        ],
        how="left",
        validate="one_to_one",
    )

    if regression[
        "full_integrated_discharge_Ah_qc"
    ].isna().any():
        raise AssertionError(
            f"{dataset}: frozen inventory row missing "
            "full discharge integration"
        )

    if not np.allclose(
        pd.to_numeric(
            regression[
                "full_integrated_discharge_Ah_qc"
            ],
            errors="raise",
        ),
        pd.to_numeric(
            regression[
                "q_discharge_sensor_Ah"
            ],
            errors="raise",
        ),
        rtol=1e-11,
        atol=1e-12,
    ):
        raise AssertionError(
            f"{dataset}: full discharge integration mismatch"
        )


    # --------------------------------------------------------------
    # Primary-valid rows outside the discharge-integrable inventory
    # must remain in the feature bank but be dQ/dV-invalid.
    # CALCE is expected to contain exactly 89 such rows.
    # --------------------------------------------------------------

    membership = df.merge(
        inventory_expected[
            [
                "dataset",
                "raw_cell_id",
                "raw_cycle_position_zero_based",
            ]
        ],
        on=[
            "dataset",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
        ],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    outside_inventory = (
        membership[
            "_merge"
        ] == "left_only"
    )

    expected_outside = (
        primary_valid_counts[
            dataset
        ]
        - inventory_counts[
            dataset
        ]
    )

    if int(
        outside_inventory.sum()
    ) != int(
        expected_outside
    ):
        raise AssertionError(
            f"{dataset}: outside-inventory count mismatch"
        )

    if outside_inventory.any():

        if membership.loc[
            outside_inventory,
            "full_valid",
        ].astype(bool).any():
            raise AssertionError(
                f"{dataset}: non-integrable row "
                "unexpectedly has valid full dQ/dV"
            )


    forbidden_tokens = [
        "capacity",
        "soh",
        "rul",
        "eol",
        "target",
    ]

    for col in df.columns:

        low = str(
            col
        ).lower()

        if any(
            x in low
            for x in forbidden_tokens
        ):
            raise AssertionError(
                f"forbidden target-like output column: {col}"
            )


    if not reused:

        atomic_csv(
            df,
            path,
        )


    output_info[
        dataset
    ] = {
        "path":
            str(
                path
            ),

        "sha256":
            sha256(
                path
            ),

        "rows":
            int(
                len(
                    df
                )
            ),

        "primary_valid_rows":
            int(
                primary_valid_counts[
                    dataset
                ]
            ),

        "sensor_inventory_rows":
            int(
                inventory_counts[
                    dataset
                ]
            ),

        "primary_valid_outside_sensor_inventory":
            int(
                primary_valid_counts[
                    dataset
                ]
                - inventory_counts[
                    dataset
                ]
            ),

        "reused_existing_dataset_shard":
            reused,
    }

    print(
        f"DATASET_COMPLETE={dataset}"
    )

    print(
        f"{dataset}_rows={len(df)}"
    )

    print(
        f"{dataset}_reused={reused}"
    )


manifest = {
    "contract":
        "dqdv_feature_bank_v1",

    "status":
        "FEATURE_GENERATION_COMPLETE_BEFORE_MODELING",

    "protocol_sha256":
        EXPECTED_PROTOCOL_SHA,

    "provenance_barrier_sha256":
        EXPECTED_PROVENANCE_SHA,

    "runner_sha256":
        sha256(
            SELF
        ),

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

    "raw_adapter_sha256":
        sha256(
            GENERATOR
        ),

    "cutoff_artifact_sha256":
        sha256(
            CUTOFFS
        ),

    "sensor_inventory_sha256":
        sha256(
            INVENTORY
        ),

    "datasets":
        DATASETS,

    "split_seeds":
        SPLIT_SEEDS,

    "feature_names":
        FEATURES,

    "dqdv_feature_count":
        len(
            FEATURES
        ),

    "voltage_bin_origin_V":
        BIN_ORIGIN_V,

    "voltage_bin_width_V":
        BIN_WIDTH_V,

    "minimum_discharge_intervals":
        MIN_INTERVALS,

    "minimum_occupied_voltage_bins":
        MIN_BINS,

    "horizons": {
        "FULL":
            "all available discharge-contributing intervals",

        "PREFIX25":
            "split-specific train-only fixed elapsed-time cutoff",

        "PREFIX50":
            "split-specific train-only fixed elapsed-time cutoff",
    },

    "raw_sources_reopened":
        True,

    "target_table_loaded":
        False,

    "capacity_values_read":
        False,

    "SOH_values_read":
        False,

    "RUL_or_EOL_values_read":
        False,

    "model_metrics_read":
        False,

    "model_states_loaded":
        False,

    "models_trained":
        False,

    "cutoffs_refit":
        False,

    "test_cycle_total_duration_used":
        False,

    "individual_test_cycle_endpoint_used":
        False,

    "sensor_inventory_total_rows":
        319202,

    "primary_valid_total_rows":
        319291,

    "primary_valid_rows_by_dataset":
        {
            "HUST": 146122,
            "CALCE": 18804,
            "MATR": 154365,
        },

    "sensor_inventory_rows_by_dataset":
        {
            "HUST": 146122,
            "CALCE": 18715,
            "MATR": 154365,
        },

    "primary_valid_outside_sensor_inventory_by_dataset":
        {
            "HUST": 0,
            "CALCE": 89,
            "MATR": 0,
        },

    "feature_bank_identity_universe":
        "audited_primary_feature_valid_rows",

    "sensor_inventory_role":
        "discharge_integrable_subset_for_full_integration_regression",

    "outputs":
        output_info,
}


atomic_json(
    manifest,
    MANIFEST,
)


print(
    "DQDV FEATURE BANK GENERATION CLOSURE"
)

print(
    "datasets:",
    len(
        DATASETS
    )
)

print(
    "primary_valid_total_rows:",
    319291
)

print(
    "sensor_inventory_total_rows:",
    319202
)

print(
    "CALCE_primary_valid_outside_sensor_inventory:",
    89
)

print(
    "split_seeds:",
    len(
        SPLIT_SEEDS
    )
)

print(
    "feature_count_per_horizon:",
    len(
        FEATURES
    )
)

print(
    "raw_sources_reopened:",
    True
)

print(
    "target_table_loaded:",
    False
)

print(
    "capacity_values_read:",
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
    "models_trained:",
    False
)

print(
    "manifest_sha256:",
    sha256(
        MANIFEST
    )
)

print(
    "DQDV_FEATURE_BANK_GENERATION_PASS=True"
)
