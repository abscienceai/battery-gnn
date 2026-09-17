from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import inspect
import json
import os
import pickle
import re
import subprocess
import time
import zipfile

import h5py
import numpy as np
import pandas as pd


# Prevent accidental numerical-library oversubscription when using workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo" / "battery-gnn"

THIS_SCRIPT = (
    REPO
    / "scripts"
    / "revision_2026"
    / "55_physical_sensor_corruption_feature_generator.py"
)

RAW_GENERATOR = (
    REPO
    / "scripts"
    / "revision_2026"
    / "14_raw_cycle_feature_generator.py"
)

KERNEL_PATH = (
    REPO
    / "scripts"
    / "revision_2026"
    / "cycle_feature_kernel.py"
)

CANON = (
    BASE
    / "data"
    / "processed"
    / "revision_2026"
    / "canonical_tasks"
    / "canonical_cycle_supervised_table.csv"
)

ASSIGNMENTS = (
    BASE
    / "data"
    / "processed"
    / "revision_2026"
    / "splits_and_scalers"
    / "cell_split_assignments.csv"
)

ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "physical_sensor_noise"
    / "final_physical_v1"
)

PROTOCOL = ROOT / "protocol.json"

ROUNDTRIP_SUMMARY = (
    ROOT
    / "zero_corruption_roundtrip"
    / "roundtrip_summary.json"
)

OUT = ROOT / "corrupted_features"
CELL_DIR = OUT / "cells"

CONDITIONS_JSON = OUT / "conditions.json"
PROGRESS_JSON = OUT / "progress.json"
CLOSURE_JSON = OUT / "corruption_feature_closure.json"


EXPECTED_PROTOCOL_SHA = (
    "7ae0836fe69c4b14a1f93a094b2d42bb96209e302c8262f7b8bd8625baf78bfc"
)

EXPECTED_ROUNDTRIP_SUMMARY_SHA = (
    "6c783a9d9396e3621dbd043d833ea61224c34f1c25d2fc03c4cf22cfc0dd5f99"
)

EXPECTED_CANON_SHA = (
    "49a7b7629cf9ee3f8ce0a7e3c52f2807878b752b4c9a30a876285fb891b00c14"
)

EXPECTED_ASSIGNMENTS_SHA = (
    "914c8b90131f496ca396f63e0502147ca99848fdfd4b2c6b2a5887225b47d055"
)


FEATURES = [
    "v_min",
    "v_q10",
    "v_q25",
    "v_median",
    "v_q75",
    "v_q90",
    "v_max",
    "v_mean",
    "v_std",
    "v_phase_slope",
    "i_min",
    "i_q10",
    "i_q25",
    "i_median",
    "i_q75",
    "i_q90",
    "i_max",
    "i_mean",
    "i_std",
    "i_phase_slope",
]

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

MAX_WORKERS = 4


# Globals populated once per worker.
_WORKER_GEN = None
_WORKER_KERNEL = None
_WORKER_CONDITIONS = None
_WORKER_LEVELS = None
_WORKER_SCRIPT_SHA = None
_WORKER_CONDITIONS_SHA = None


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(
        data
    ).hexdigest()


def stable_json_bytes(obj):
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def atomic_json(obj, path):
    path = Path(path)

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


def atomic_npz(path, **arrays):
    path = Path(path)

    tmp = Path(
        str(path) + ".tmp"
    )

    with tmp.open("wb") as f:
        np.savez_compressed(
            f,
            **arrays,
        )

    os.replace(
        tmp,
        path,
    )


def as_bool(s):
    return (
        s.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "1.0",
                "yes",
            ]
        )
    )


def safe_name(dataset, cell):
    token = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(cell),
    )

    return (
        dataset
        + "__"
        + token
    )


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def source_stat(path):
    st = Path(path).stat()

    return {
        "path":
            str(
                Path(path).resolve()
            ),

        "size":
            int(st.st_size),

        "mtime_ns":
            int(st.st_mtime_ns),
    }


def stable_seed(*parts):
    text = "|".join(
        str(x)
        for x in parts
    ).encode("utf-8")

    digest = hashlib.sha256(
        text
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


def rng_for(*parts):
    return np.random.default_rng(
        stable_seed(*parts)
    )


def feature_dict_to_array(d):
    return np.asarray(
        [
            float(d[x])
            for x in FEATURES
        ],
        dtype=np.float64,
    )


def build_conditions(protocol):
    levels = protocol[
        "severity_levels"
    ]

    out = [
        {
            "condition_index": 0,
            "condition_id": "clean",
            "scenario": "clean",
            "level": "clean",
            "corruption_seed": -1,
            "severity_axis": 0.0,
        }
    ]

    idx = 1

    for scenario in [
        "gaussian",
        "calibration_offset",
        "sensor_drift",
        "sample_dropout",
    ]:

        for level in [
            "low",
            "medium",
            "high",
        ]:

            for seed in protocol[
                "corruption_seeds"
            ]:

                out.append({
                    "condition_index":
                        idx,

                    "condition_id":
                        (
                            f"{scenario}"
                            f"__{level}"
                            f"__seed{seed}"
                        ),

                    "scenario":
                        scenario,

                    "level":
                        level,

                    "corruption_seed":
                        int(seed),

                    "severity_axis":
                        float(
                            levels[
                                level
                            ][
                                "severity_axis"
                            ]
                        ),
                })

                idx += 1


    for seed in protocol[
        "corruption_seeds"
    ]:

        out.append({
            "condition_index":
                idx,

            "condition_id":
                (
                    "compound_medium"
                    f"__medium"
                    f"__seed{seed}"
                ),

            "scenario":
                "compound_medium",

            "level":
                "medium",

            "corruption_seed":
                int(seed),

            "severity_axis":
                float(
                    levels[
                        "medium"
                    ][
                        "severity_axis"
                    ]
                ),
        })

        idx += 1


    assert len(out) == 66

    assert [
        x[
            "condition_index"
        ]
        for x in out
    ] == list(
        range(66)
    )

    assert len({
        x[
            "condition_id"
        ]
        for x in out
    }) == 66

    return out


def add_constant(
    arr,
    value,
):
    out = np.asarray(
        arr,
        dtype=np.float64,
    ).copy()

    finite = np.isfinite(
        out
    )

    out[
        finite
    ] += float(
        value
    )

    return out


def add_gaussian(
    arr,
    sigma,
    *,
    seed_parts,
):
    out = np.asarray(
        arr,
        dtype=np.float64,
    ).copy()

    finite_idx = np.flatnonzero(
        np.isfinite(
            out
        )
    )

    if len(
        finite_idx
    ):

        rng = rng_for(
            *seed_parts
        )

        out[
            finite_idx
        ] += rng.normal(
            loc=0.0,
            scale=float(
                sigma
            ),
            size=len(
                finite_idx
            ),
        )

    return out


def dropout_forward_fill(
    voltage,
    current,
    time_s,
    probability,
    *,
    seed_parts,
):
    V = np.asarray(
        voltage,
        dtype=np.float64,
    ).copy()

    I = np.asarray(
        current,
        dtype=np.float64,
    ).copy()

    t = np.asarray(
        time_s,
        dtype=np.float64,
    )


    valid = (
        np.isfinite(V)
        & np.isfinite(I)
        & np.isfinite(t)
    )

    idx = np.flatnonzero(
        valid
    )


    if len(idx) <= 1:
        return V, I


    rng = rng_for(
        *seed_parts
    )

    dropped = (
        rng.random(
            len(idx)
        )
        < float(
            probability
        )
    )

    # Protect the first valid sample.
    dropped[
        0
    ] = False


    last_retained = idx[
        0
    ]

    for local_i in range(
        1,
        len(idx),
    ):

        raw_i = idx[
            local_i
        ]

        if dropped[
            local_i
        ]:

            V[
                raw_i
            ] = V[
                last_retained
            ]

            I[
                raw_i
            ] = I[
                last_retained
            ]

        else:

            last_retained = raw_i


    return V, I


def cell_level_bias(
    dataset,
    cell_id,
    corruption_seed,
    level,
    channel,
    sigma,
    component,
):
    rng = rng_for(
        corruption_seed,
        component,
        level,
        dataset,
        cell_id,
        "CELL_CONSTANT",
        channel,
    )

    return float(
        rng.normal(
            loc=0.0,
            scale=float(
                sigma
            ),
        )
    )


def cell_level_sign(
    dataset,
    cell_id,
    corruption_seed,
    level,
    channel,
    component,
):
    rng = rng_for(
        corruption_seed,
        component,
        level,
        dataset,
        cell_id,
        "CELL_CONSTANT",
        channel,
        "sign",
    )

    return (
        -1.0
        if int(
            rng.integers(
                0,
                2,
            )
        ) == 0
        else 1.0
    )


def corrupt_signals(
    voltage,
    current,
    time_s,
    *,
    condition,
    levels,
    dataset,
    cell_id,
    cycle_key,
    cycle_position,
):

    V = np.asarray(
        voltage,
        dtype=np.float64,
    ).copy()

    I = np.asarray(
        current,
        dtype=np.float64,
    ).copy()

    t = np.asarray(
        time_s,
        dtype=np.float64,
    ).copy()


    scenario = condition[
        "scenario"
    ]

    if scenario == "clean":
        return V, I, t


    level = condition[
        "level"
    ]

    corruption_seed = int(
        condition[
            "corruption_seed"
        ]
    )

    cfg = levels[
        level
    ]


    # ------------------------------------------------------------------
    # Calibration offset
    # ------------------------------------------------------------------

    if scenario in {
        "calibration_offset",
        "compound_medium",
    }:

        vbias = cell_level_bias(
            dataset,
            cell_id,
            corruption_seed,
            level,
            "V",
            cfg[
                "calibration_voltage_sigma_V"
            ],
            "calibration",
        )

        ibias = cell_level_bias(
            dataset,
            cell_id,
            corruption_seed,
            level,
            "I",
            cfg[
                "calibration_current_sigma_A"
            ],
            "calibration",
        )

        V = add_constant(
            V,
            vbias,
        )

        I = add_constant(
            I,
            ibias,
        )


    # ------------------------------------------------------------------
    # Progressive cell-level drift
    # ------------------------------------------------------------------

    if scenario in {
        "sensor_drift",
        "compound_medium",
    }:

        v_sign = cell_level_sign(
            dataset,
            cell_id,
            corruption_seed,
            level,
            "V",
            "drift",
        )

        i_sign = cell_level_sign(
            dataset,
            cell_id,
            corruption_seed,
            level,
            "I",
            "drift",
        )


        progress_100 = (
            float(
                cycle_position
            )
            / 100.0
        )


        vmag = min(
            float(
                cfg[
                    "drift_voltage_per_100_cycles_V"
                ]
            )
            * progress_100,

            float(
                cfg[
                    "drift_voltage_cap_V"
                ]
            ),
        )


        imag = min(
            float(
                cfg[
                    "drift_current_per_100_cycles_A"
                ]
            )
            * progress_100,

            float(
                cfg[
                    "drift_current_cap_A"
                ]
            ),
        )


        V = add_constant(
            V,
            v_sign
            * vmag,
        )

        I = add_constant(
            I,
            i_sign
            * imag,
        )


    # ------------------------------------------------------------------
    # Independent Gaussian sensor noise
    # ------------------------------------------------------------------

    if scenario in {
        "gaussian",
        "compound_medium",
    }:

        V = add_gaussian(
            V,
            cfg[
                "gaussian_voltage_sigma_V"
            ],
            seed_parts=(
                corruption_seed,
                scenario,
                level,
                dataset,
                cell_id,
                cycle_key,
                "V",
                "gaussian",
            ),
        )

        I = add_gaussian(
            I,
            cfg[
                "gaussian_current_sigma_A"
            ],
            seed_parts=(
                corruption_seed,
                scenario,
                level,
                dataset,
                cell_id,
                cycle_key,
                "I",
                "gaussian",
            ),
        )


    # ------------------------------------------------------------------
    # Shared-record sample dropout + causal forward-fill
    # ------------------------------------------------------------------

    if scenario in {
        "sample_dropout",
        "compound_medium",
    }:

        V, I = dropout_forward_fill(
            V,
            I,
            t,
            cfg[
                "dropout_probability"
            ],
            seed_parts=(
                corruption_seed,
                scenario,
                level,
                dataset,
                cell_id,
                cycle_key,
                "VI",
                "dropout",
            ),
        )


    assert len(V) == len(I) == len(t)

    return V, I, t


class CorruptingKernelProxy:

    def __init__(
        self,
        *,
        base_kernel,
        conditions,
        levels,
        dataset,
        cell_id,
        target_positions,
    ):

        self.base_kernel = (
            base_kernel
        )

        self.conditions = (
            conditions
        )

        self.levels = (
            levels
        )

        self.dataset = str(
            dataset
        )

        self.cell_id = str(
            cell_id
        )


        self.positions = np.asarray(
            sorted(
                int(x)
                for x in target_positions
            ),
            dtype=np.int64,
        )


        self.position_to_row = {
            int(p):
                i
            for i, p in enumerate(
                self.positions
            )
        }


        self.features = np.full(
            (
                len(
                    conditions
                ),
                len(
                    self.positions
                ),
                len(
                    FEATURES
                ),
            ),
            np.nan,
            dtype=np.float64,
        )


        self.seen = np.zeros(
            len(
                self.positions
            ),
            dtype=bool,
        )


    def __getattr__(
        self,
        name,
    ):
        return getattr(
            self.base_kernel,
            name,
        )


    def extract_primary_cycle_features(
        self,
        voltage_V,
        current_A,
        time_s,
        **kwargs,
    ):

        # First reproduce the exact clean pipeline output.
        clean_features, clean_qc = (
            self.base_kernel
            .extract_primary_cycle_features(
                voltage_V,
                current_A,
                time_s,
                **kwargs,
            )
        )


        caller = (
            inspect
            .currentframe()
            .f_back
        )

        assert caller is not None

        row = caller.f_locals.get(
            "row"
        )

        assert isinstance(
            row,
            dict,
        )


        dataset = str(
            row[
                "dataset"
            ]
        )

        raw_cell_id = str(
            row[
                "raw_cell_id"
            ]
        )

        cycle_key = str(
            row[
                "raw_cycle_key"
            ]
        )

        cycle_position = int(
            row[
                "raw_cycle_position_zero_based"
            ]
        )


        assert dataset == self.dataset
        assert raw_cell_id == self.cell_id


        row_index = (
            self.position_to_row.get(
                cycle_position
            )
        )


        if row_index is None:
            return (
                clean_features,
                clean_qc,
            )


        assert not self.seen[
            row_index
        ], (
            dataset,
            raw_cell_id,
            cycle_position,
        )


        # Clean condition.
        self.features[
            0,
            row_index,
            :,
        ] = feature_dict_to_array(
            clean_features
        )


        # Physical-unit corruptions. Each condition is sent through
        # the exact same frozen feature kernel after corruption.
        for condition in self.conditions[
            1:
        ]:

            Vc, Ic, tc = corrupt_signals(
                voltage_V,
                current_A,
                time_s,
                condition=condition,
                levels=self.levels,
                dataset=dataset,
                cell_id=raw_cell_id,
                cycle_key=cycle_key,
                cycle_position=cycle_position,
            )


            features, _ = (
                self.base_kernel
                .extract_primary_cycle_features(
                    Vc,
                    Ic,
                    tc,
                    **kwargs,
                )
            )


            self.features[
                int(
                    condition[
                        "condition_index"
                    ]
                ),
                row_index,
                :,
            ] = feature_dict_to_array(
                features
            )


        self.seen[
            row_index
        ] = True


        return (
            clean_features,
            clean_qc,
        )


def init_worker(
    conditions,
    levels,
    script_sha,
    conditions_sha,
):

    global _WORKER_GEN
    global _WORKER_KERNEL
    global _WORKER_CONDITIONS
    global _WORKER_LEVELS
    global _WORKER_SCRIPT_SHA
    global _WORKER_CONDITIONS_SHA


    _WORKER_GEN = load_module(
        RAW_GENERATOR,
        (
            "raw_generator14_worker_"
            + str(
                os.getpid()
            )
        ),
    )

    _WORKER_KERNEL = (
        _WORKER_GEN.load_kernel()
    )

    _WORKER_CONDITIONS = (
        conditions
    )

    _WORKER_LEVELS = (
        levels
    )

    _WORKER_SCRIPT_SHA = (
        script_sha
    )

    _WORKER_CONDITIONS_SHA = (
        conditions_sha
    )


def matr_batch_group(f):

    if (
        "batch"
        in f
        and isinstance(
            f["batch"],
            h5py.Group,
        )
        and "cycles"
        in f["batch"]
    ):
        return f["batch"]


    found = []

    for name in f.keys():

        obj = f[name]

        if (
            isinstance(
                obj,
                h5py.Group,
            )
            and "cycles"
            in obj
        ):
            found.append(obj)


    assert len(found) == 1, (
        list(
            f.keys()
        ),
    )

    return found[0]


def expected_digest(task):
    payload = {
        "dataset":
            task[
                "dataset"
            ],

        "cell_id":
            task[
                "cell_id"
            ],

        "source_member":
            task[
                "source_member"
            ],

        "positions":
            task[
                "positions"
            ],

        "cycle_keys":
            task[
                "cycle_keys"
            ],
    }

    return sha256_bytes(
        stable_json_bytes(
            payload
        )
    )


def shard_paths(task):
    base = safe_name(
        task[
            "dataset"
        ],
        task[
            "cell_id"
        ],
    )

    return (
        CELL_DIR
        / f"{base}.npz",

        CELL_DIR
        / f"{base}.json",
    )


def shard_reusable(task):
    npz_path, meta_path = (
        shard_paths(
            task
        )
    )

    if not (
        npz_path.is_file()
        and meta_path.is_file()
    ):
        return False


    try:

        meta = json.loads(
            meta_path.read_text(
                encoding="utf-8"
            )
        )


        if meta.get(
            "contract"
        ) != "physical_corruption_feature_cell_v1":
            return False


        if meta.get(
            "protocol_sha256"
        ) != EXPECTED_PROTOCOL_SHA:
            return False


        if meta.get(
            "generator_sha256"
        ) != _WORKER_SCRIPT_SHA:
            return False


        if meta.get(
            "conditions_sha256"
        ) != _WORKER_CONDITIONS_SHA:
            return False


        if meta.get(
            "expected_digest"
        ) != expected_digest(
            task
        ):
            return False


        if meta.get(
            "row_count"
        ) != len(
            task[
                "positions"
            ]
        ):
            return False


        if meta.get(
            "condition_count"
        ) != 66:
            return False


        if meta.get(
            "feature_count"
        ) != 20:
            return False


        if meta.get(
            "all_features_finite"
        ) is not True:
            return False


        if sha256_file(
            npz_path
        ) != meta.get(
            "npz_sha256"
        ):
            return False


        return True


    except Exception:
        return False


def generate_cell(task):

    npz_path, meta_path = (
        shard_paths(
            task
        )
    )


    if shard_reusable(
        task
    ):
        return {
            "dataset":
                task[
                    "dataset"
                ],

            "cell_id":
                task[
                    "cell_id"
                ],

            "row_count":
                len(
                    task[
                        "positions"
                    ]
                ),

            "reused":
                True,

            "npz_path":
                str(
                    npz_path
                ),

            "meta_path":
                str(
                    meta_path
                ),

            "npz_sha256":
                sha256_file(
                    npz_path
                ),
        }


    dataset = task[
        "dataset"
    ]

    cell_id = task[
        "cell_id"
    ]


    proxy = CorruptingKernelProxy(
        base_kernel=_WORKER_KERNEL,
        conditions=_WORKER_CONDITIONS,
        levels=_WORKER_LEVELS,
        dataset=dataset,
        cell_id=cell_id,
        target_positions=task[
            "positions"
        ],
    )


    raw_source_stat = None


    # ------------------------------------------------------------------
    # HUST
    # ------------------------------------------------------------------

    if dataset == "HUST":

        zip_path = (
            _WORKER_GEN.HUST_ZIP
        )

        member = task[
            "source_member"
        ]


        with zipfile.ZipFile(
            zip_path,
            "r",
        ) as zf:

            assert member in set(
                zf.namelist()
            )

            with zf.open(
                member
            ) as f:

                obj = pickle.load(
                    f
                )


        assert len(obj) == 1

        raw_id = next(
            iter(obj)
        )

        assert str(
            raw_id
        ) == str(
            cell_id
        )


        _WORKER_GEN.process_hust_cell(
            source_member=member,
            raw_cell_id=raw_id,
            cell=obj[
                raw_id
            ],
            kernel=proxy,
            kernel_hash=sha256_file(
                KERNEL_PATH
            ),
        )


        raw_source_stat = (
            source_stat(
                zip_path
            )
        )


    # ------------------------------------------------------------------
    # CALCE
    # ------------------------------------------------------------------

    elif dataset == "CALCE":

        zip_path = (
            _WORKER_GEN.CALCE_RAW_DIR
            / f"{cell_id}.zip"
        )

        assert zip_path.is_file()


        _WORKER_GEN.process_calce_cell(
            zip_path=zip_path,
            raw_cell_id=cell_id,
            kernel=proxy,
            kernel_hash=sha256_file(
                KERNEL_PATH
            ),
            raw_zip_hash=sha256_file(
                zip_path
            ),
        )


        raw_source_stat = (
            source_stat(
                zip_path
            )
        )


    # ------------------------------------------------------------------
    # MATR
    # ------------------------------------------------------------------

    elif dataset == "MATR":

        path = (
            BASE
            / task[
                "source_member"
            ]
        )

        assert path.is_file()


        m = re.fullmatch(
            r"b\d+c(\d+)",
            str(
                cell_id
            ),
        )

        assert m is not None

        cell_index = int(
            m.group(1)
        )


        with h5py.File(
            path,
            "r",
        ) as f:

            batch = matr_batch_group(
                f
            )

            n_cells = int(
                batch[
                    "cycles"
                ].shape[
                    0
                ]
            )

            assert (
                0
                <= cell_index
                < n_cells
            )


            cycles = f[
                batch[
                    "cycles"
                ][
                    cell_index,
                    0,
                ]
            ]


            _WORKER_GEN.process_matr_cell(
                f=f,
                cycles=cycles,
                source_member=task[
                    "source_member"
                ],
                raw_cell_id=cell_id,
                kernel=proxy,
                kernel_hash=sha256_file(
                    KERNEL_PATH
                ),
            )


        raw_source_stat = (
            source_stat(
                path
            )
        )


    else:
        raise AssertionError(
            dataset
        )


    assert proxy.seen.all(), (
        dataset,
        cell_id,
        int(
            (~proxy.seen)
            .sum()
        ),
    )


    assert np.isfinite(
        proxy.features
    ).all(), (
        dataset,
        cell_id,
        "non-finite corrupted feature",
    )


    assert proxy.features.shape == (
        66,
        len(
            task[
                "positions"
            ]
        ),
        20,
    )


    positions = np.asarray(
        task[
            "positions"
        ],
        dtype=np.int64,
    )


    cycle_keys = np.asarray(
        [
            str(x)
            for x in task[
                "cycle_keys"
            ]
        ],
        dtype="U128",
    )


    condition_ids = np.asarray(
        [
            c[
                "condition_id"
            ]
            for c in _WORKER_CONDITIONS
        ],
        dtype="U96",
    )


    atomic_npz(
        npz_path,
        features=proxy.features,
        positions=positions,
        cycle_keys=cycle_keys,
        condition_ids=condition_ids,
    )


    npz_hash = sha256_file(
        npz_path
    )


    meta = {
        "contract":
            "physical_corruption_feature_cell_v1",

        "protocol_sha256":
            EXPECTED_PROTOCOL_SHA,

        "roundtrip_summary_sha256":
            EXPECTED_ROUNDTRIP_SUMMARY_SHA,

        "generator_sha256":
            _WORKER_SCRIPT_SHA,

        "conditions_sha256":
            _WORKER_CONDITIONS_SHA,

        "dataset":
            dataset,

        "cell_id":
            str(
                cell_id
            ),

        "source_member":
            task[
                "source_member"
            ],

        "expected_digest":
            expected_digest(
                task
            ),

        "raw_source_stat":
            raw_source_stat,

        "row_count":
            int(
                len(
                    positions
                )
            ),

        "condition_count":
            66,

        "feature_count":
            20,

        "shape":
            [
                int(x)
                for x in proxy.features.shape
            ],

        "all_features_finite":
            True,

        "contains_model_predictions":
            False,

        "contains_targets":
            False,

        "contains_model_metrics":
            False,

        "npz_sha256":
            npz_hash,
    }


    atomic_json(
        meta,
        meta_path,
    )


    return {
        "dataset":
            dataset,

        "cell_id":
            str(
                cell_id
            ),

        "row_count":
            int(
                len(
                    positions
                )
            ),

        "reused":
            False,

        "npz_path":
            str(
                npz_path
            ),

        "meta_path":
            str(
                meta_path
            ),

        "npz_sha256":
            npz_hash,
    }


# ============================================================================
# Immutable pre-run guards
# ============================================================================

assert sha256_file(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

assert sha256_file(
    ROUNDTRIP_SUMMARY
) == EXPECTED_ROUNDTRIP_SUMMARY_SHA

assert sha256_file(
    CANON
) == EXPECTED_CANON_SHA

assert sha256_file(
    ASSIGNMENTS
) == EXPECTED_ASSIGNMENTS_SHA


protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)

roundtrip = json.loads(
    ROUNDTRIP_SUMMARY.read_text(
        encoding="utf-8"
    )
)


assert roundtrip[
    "status"
] == "PASS"

assert roundtrip[
    "all_rows_pass"
] is True

assert roundtrip[
    "failed_rows"
] == 0

assert roundtrip[
    "model_evaluated"
] is False

assert roundtrip[
    "physical_noise_generated"
] is False


assert protocol[
    "training_policy"
][
    "no_retraining"
] is True

assert protocol[
    "training_policy"
][
    "test_only_corruption"
] is True


assert sha256_file(
    RAW_GENERATOR
) == protocol[
    "source_sha256"
][
    "raw_cycle_generator"
]

assert sha256_file(
    KERNEL_PATH
) == protocol[
    "source_sha256"
][
    "cycle_feature_kernel"
]


tracked = subprocess.run(
    [
        "git",
        "ls-files",
        "--error-unmatch",
        str(
            THIS_SCRIPT.relative_to(
                REPO
            )
        ),
    ],
    cwd=REPO,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

assert tracked.returncode == 0


SCRIPT_SHA = sha256_file(
    THIS_SCRIPT
)

SCRIPT_COMMIT = subprocess.check_output(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
    text=True,
).strip()


# Absolutely no model evaluation artifacts may predate feature closure.
for pattern in [
    "physical_test_metrics*.csv",
    "noise_test_metrics*.csv",
    "noise_evaluation*.json",
    "physical_noise_execution_manifest.json",
]:

    assert not list(
        ROOT.rglob(
            pattern
        )
    ), pattern


# ============================================================================
# Frozen conditions
# ============================================================================

conditions = build_conditions(
    protocol
)

conditions_payload = {
    "contract":
        "physical_sensor_corruption_conditions_v1",

    "protocol_sha256":
        EXPECTED_PROTOCOL_SHA,

    "condition_count":
        len(
            conditions
        ),

    "conditions":
        conditions,
}


CONDITIONS_SHA = sha256_bytes(
    stable_json_bytes(
        conditions_payload
    )
)


conditions_payload[
    "conditions_sha256"
] = CONDITIONS_SHA


OUT.mkdir(
    parents=True,
    exist_ok=True,
)

CELL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


if CONDITIONS_JSON.exists():

    existing = json.loads(
        CONDITIONS_JSON.read_text(
            encoding="utf-8"
        )
    )

    assert existing == (
        conditions_payload
    )

else:

    atomic_json(
        conditions_payload,
        CONDITIONS_JSON,
    )


# ============================================================================
# Canonical union-of-test cells
# ============================================================================

canonical = pd.read_csv(
    CANON,
    usecols=[
        "dataset",
        "split_group",
        "soh_task_eligible",
        "feature_source_cell_id",
        "feature_source_member",
        "feature_raw_cycle_key",
        "feature_raw_cycle_position_zero_based",
    ],
    low_memory=False,
)


canonical = canonical[
    as_bool(
        canonical[
            "soh_task_eligible"
        ]
    )
].copy()


canonical[
    "split_group"
] = canonical[
    "split_group"
].astype(str)

canonical[
    "feature_source_cell_id"
] = canonical[
    "feature_source_cell_id"
].astype(str)

canonical[
    "feature_source_member"
] = canonical[
    "feature_source_member"
].astype(str)

canonical[
    "feature_raw_cycle_position_zero_based"
] = pd.to_numeric(
    canonical[
        "feature_raw_cycle_position_zero_based"
    ],
    errors="raise",
).astype(int)


assign = pd.read_csv(
    ASSIGNMENTS,
    usecols=[
        "split_seed",
        "split_group",
        "split",
    ],
    low_memory=False,
)

assign[
    "split_group"
] = assign[
    "split_group"
].astype(str)


test_groups = set(
    assign.loc[
        assign[
            "split"
        ] == "test",
        "split_group",
    ]
)


target = canonical[
    canonical[
        "split_group"
    ].isin(
        test_groups
    )
].copy()


assert len(
    target
) == roundtrip[
    "union_test_rows_audited"
]

assert len(
    target
) == 166304


tasks = []


for (
    dataset,
    cell_id
), g in target.groupby(
    [
        "dataset",
        "feature_source_cell_id",
    ],
    sort=True,
):

    g = g.sort_values(
        "feature_raw_cycle_position_zero_based",
        kind="mergesort",
    )


    members = (
        g[
            "feature_source_member"
        ]
        .astype(str)
        .unique()
        .tolist()
    )


    # HUST/MATR have one source member per cell.
    # CALCE has multiple per-cycle source members inside one cell ZIP,
    # so the task-level physical source is the cell ZIP itself.
    if dataset in {
        "HUST",
        "MATR",
    }:

        assert len(
            members
        ) == 1

        source_member = (
            members[
                0
            ]
        )

    elif dataset == "CALCE":

        source_member = (
            f"data/raw/CALCE/"
            f"{cell_id}.zip"
        )

    else:
        raise AssertionError(
            dataset
        )


    tasks.append({
        "dataset":
            str(
                dataset
            ),

        "cell_id":
            str(
                cell_id
            ),

        "source_member":
            source_member,

        "positions":
            [
                int(x)
                for x in g[
                    "feature_raw_cycle_position_zero_based"
                ].tolist()
            ],

        "cycle_keys":
            [
                str(x)
                for x in g[
                    "feature_raw_cycle_key"
                ].tolist()
            ],
    })


assert len(
    tasks
) == 143


dataset_cell_counts = {
    dataset:
        sum(
            1
            for task in tasks
            if task[
                "dataset"
            ] == dataset
        )
    for dataset in DATASETS
}


dataset_row_counts = {
    dataset:
        int(
            (
                target[
                    "dataset"
                ] == dataset
            ).sum()
        )
    for dataset in DATASETS
}


print("=" * 80)
print("PHYSICAL SENSOR CORRUPTION FEATURE GENERATION")
print("=" * 80)

print(
    "protocol locked:",
    True
)

print(
    "roundtrip barrier passed:",
    True
)

print(
    "conditions:",
    len(
        conditions
    )
)

print(
    "clean conditions:",
    1
)

print(
    "corrupted conditions:",
    65
)

print(
    "union test rows:",
    len(
        target
    )
)

print(
    "union test cells:",
    len(
        tasks
    )
)

print(
    "dataset cells:",
    dataset_cell_counts
)

print(
    "dataset rows:",
    dataset_row_counts
)

print(
    "workers:",
    MAX_WORKERS
)

print(
    "model loading:",
    False
)

print(
    "model evaluation:",
    False
)

print(
    "metric computation:",
    False
)

print()


start_time = time.time()


completed = 0
generated = 0
reused = 0
rows_completed = 0
records = []


def write_progress(
    phase,
    current=None,
):

    atomic_json(
        {
            "contract":
                "physical_corruption_feature_progress_v1",

            "phase":
                phase,

            "protocol_sha256":
                EXPECTED_PROTOCOL_SHA,

            "generator_sha256":
                SCRIPT_SHA,

            "conditions_sha256":
                CONDITIONS_SHA,

            "total_cells":
                len(
                    tasks
                ),

            "completed_cells":
                completed,

            "generated_cells":
                generated,

            "reused_cells":
                reused,

            "completed_rows":
                rows_completed,

            "total_rows":
                len(
                    target
                ),

            "current":
                current,

            "model_loaded":
                False,

            "model_evaluated":
                False,

            "metric_computed":
                False,

            "elapsed_seconds":
                float(
                    time.time()
                    - start_time
                ),
        },
        PROGRESS_JSON,
    )


write_progress(
    "starting"
)


with ProcessPoolExecutor(
    max_workers=MAX_WORKERS,
    initializer=init_worker,
    initargs=(
        conditions,
        protocol[
            "severity_levels"
        ],
        SCRIPT_SHA,
        CONDITIONS_SHA,
    ),
) as pool:

    future_map = {
        pool.submit(
            generate_cell,
            task,
        ):
            task
        for task in tasks
    }


    for future in as_completed(
        future_map
    ):

        task = future_map[
            future
        ]

        result = future.result()

        records.append(
            result
        )

        completed += 1

        rows_completed += int(
            result[
                "row_count"
            ]
        )

        if result[
            "reused"
        ]:
            reused += 1
        else:
            generated += 1


        write_progress(
            "generating",
            {
                "dataset":
                    result[
                        "dataset"
                    ],

                "cell_id":
                    result[
                        "cell_id"
                    ],
            },
        )


        if (
            completed % 5 == 0
            or completed == len(
                tasks
            )
        ):

            print(
                f"progress "
                f"{completed}/"
                f"{len(tasks)} "
                f"rows={rows_completed}/"
                f"{len(target)} "
                f"generated={generated} "
                f"reused={reused}",
                flush=True,
            )


assert completed == len(
    tasks
)

assert rows_completed == len(
    target
)


records = sorted(
    records,
    key=lambda x: (
        x[
            "dataset"
        ],
        x[
            "cell_id"
        ],
    ),
)


# ============================================================================
# Immutable feature-generation closure
# ============================================================================

for record in records:

    npz_path = Path(
        record[
            "npz_path"
        ]
    )

    meta_path = Path(
        record[
            "meta_path"
        ]
    )

    assert npz_path.is_file()
    assert meta_path.is_file()

    assert sha256_file(
        npz_path
    ) == record[
        "npz_sha256"
    ]


closure = {
    "contract":
        "physical_corruption_feature_closure_v1",

    "status":
        "COMPLETE_BEFORE_ANY_MODEL_EVALUATION",

    "protocol_sha256":
        EXPECTED_PROTOCOL_SHA,

    "roundtrip_summary_sha256":
        EXPECTED_ROUNDTRIP_SUMMARY_SHA,

    "generator_sha256":
        SCRIPT_SHA,

    "generator_commit":
        SCRIPT_COMMIT,

    "conditions_sha256":
        CONDITIONS_SHA,

    "conditions_file_sha256":
        sha256_file(
            CONDITIONS_JSON
        ),

    "condition_count":
        66,

    "clean_condition_count":
        1,

    "corrupted_condition_count":
        65,

    "cell_count":
        len(
            records
        ),

    "row_count":
        int(
            rows_completed
        ),

    "feature_count":
        20,

    "dataset_cell_counts":
        dataset_cell_counts,

    "dataset_row_counts":
        dataset_row_counts,

    "all_shards_complete":
        True,

    "all_features_finite":
        True,

    "model_loaded":
        False,

    "model_evaluated":
        False,

    "metric_computed":
        False,

    "generated_cells_this_invocation":
        int(
            generated
        ),

    "reused_cells_this_invocation":
        int(
            reused
        ),

    "records":
        records,

    "elapsed_seconds":
        float(
            time.time()
            - start_time
        ),
}


atomic_json(
    closure,
    CLOSURE_JSON,
)


write_progress(
    "complete"
)


print()
print("=" * 80)
print("PHYSICAL CORRUPTION FEATURE CLOSURE")
print("=" * 80)

print(
    "cells:",
    closure[
        "cell_count"
    ]
)

print(
    "rows:",
    closure[
        "row_count"
    ]
)

print(
    "conditions:",
    closure[
        "condition_count"
    ]
)

print(
    "corrupted conditions:",
    closure[
        "corrupted_condition_count"
    ]
)

print(
    "all shards complete:",
    closure[
        "all_shards_complete"
    ]
)

print(
    "all features finite:",
    closure[
        "all_features_finite"
    ]
)

print(
    "model loaded:",
    closure[
        "model_loaded"
    ]
)

print(
    "model evaluated:",
    closure[
        "model_evaluated"
    ]
)

print(
    "metric computed:",
    closure[
        "metric_computed"
    ]
)

print(
    "conditions SHA256:",
    closure[
        "conditions_sha256"
    ]
)

print(
    "closure SHA256:",
    sha256_file(
        CLOSURE_JSON
    )
)

print()
print(
    "PHYSICAL_SENSOR_CORRUPTION_FEATURE_GENERATION_PASS=True"
)
