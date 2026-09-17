from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import time

import numpy as np
import pandas as pd
import torch


BASE = Path.home() / "alper/battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

os.environ["BASE"] = str(BASE)

ROOT = (
    BASE
    / "results/revision_2026/dqdv_leakage_control/final_dqdv_v1"
)

TRAINROOT = ROOT / "training_v2"
EXEC = TRAINROOT / "final_execution_v2"

PROTOCOL = TRAINROOT / "dqdv_training_protocol_v2.json"

COMMON_MANIFEST = (
    ROOT
    / "common_cohort_scalers_v1/common_cohort_scaler_manifest.json"
)

CANON = (
    BASE
    / "data/processed/revision_2026/canonical_tasks"
    / "canonical_cycle_supervised_table.csv"
)

COHORT = (
    ROOT
    / "common_cohort_scalers_v1/common_cohort_rows.csv"
)

SCALERS = (
    ROOT
    / "common_cohort_scalers_v1/train_only_scaler_parameters.csv"
)

RAW_FEATURE_DIR = (
    BASE
    / "data/processed/revision_2026/raw_cycle_features"
)

FEATURE_BANK_DIR = ROOT / "features_v1"

BENCHMARK_SCRIPT = (
    REPO
    / "scripts/revision_2026/24_feature_matched_soh_benchmark.py"
)

SELF = (
    REPO
    / "scripts/revision_2026/95_final_dqdv_training_v2_runner.py"
)

RECORD_DIR = EXEC / "run_records"
STATE_DIR = EXEC / "model_states"

STATUS = EXEC / "execution_status.json"
MANIFEST = EXEC / "execution_manifest.json"

EXPECTED_PROTOCOL_SHA = (
    "4612a450eeaf909a04c31be16957bdc0567aac67635241e14d7bbceea507fe9d"
)

EXPECTED_COMMON_MANIFEST_SHA = (
    "3ee48af31ad580c14de0460fa1f163ad572e369368d6da71c0ba6823d28905a1"
)

COMPATIBLE_PRE_INPUT_BINDING_RUNNER_SHA = (
    "46f1ad8ab073e320b5ca5420283cec15c5b2d661f3f2089b41716a5b4d72a76d"
)

PARENT_PROTOCOL_V1_SHA = (
    "43122c4d6ef5a2e70b2d13ae22432a2cfa13027874e0bea6f3b62675e3fb23e9"
)

COMPATIBLE_V1_INPUT_BINDING_RUNNER_SHA = (
    "117f5dea393cb27c487e07e7a0ffc7c46195bb71d69c7809fcc574517caa573f"
)

IMPORT_MANIFEST = (
    EXEC
    / "v1_compatible_import_manifest.json"
)

EXPECTED_IMPORT_MANIFEST_SHA = (
    "068223da9e12c370af136343388285b9930ee3edd42b14082309eb40a0c65117"
)

DATASETS = [
    "HUST",
    "CALCE",
    "MATR",
]

SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

TRAIN_SEEDS = [
    0,
    1,
    2,
]

SPLIT_NAMES = [
    "train",
    "validation",
    "test",
]

PRIMARY_FEATURES = [
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

DQDV_FEATURES = [
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

CONDITIONS = [
    "PRIMARY20_COMMON",
    "PRIMARY20_PLUS_FULL_DQDV10",
    "PRIMARY20_PLUS_PREFIX25_DQDV10",
    "PRIMARY20_PLUS_PREFIX50_DQDV10",
]

PREFLIGHT_ONLY = (
    os.environ.get(
        "DQDV_PREFLIGHT_ONLY",
        "0",
    )
    == "1"
)


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


def atomic_json(obj, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def atomic_torch_save(
    state,
    path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(path) + ".tmp"
    )

    torch.save(
        state,
        tmp,
    )

    os.replace(
        tmp,
        path,
    )


def jsonable(x):

    # train_one() returns its epoch history as a pandas DataFrame.
    # Convert pandas containers before recursively handling their
    # scalar contents. This is serialization-only and does not alter
    # any model, metric, optimizer, or training value.
    if isinstance(
        x,
        pd.DataFrame,
    ):
        return [
            jsonable(
                row
            )
            for row in x.to_dict(
                orient="records"
            )
        ]

    if isinstance(
        x,
        pd.Series,
    ):
        return jsonable(
            x.to_dict()
        )

    if isinstance(
        x,
        pd.Index,
    ):
        return jsonable(
            x.tolist()
        )

    if isinstance(
        x,
        np.bool_,
    ):
        return bool(
            x
        )

    if isinstance(
        x,
        dict,
    ):
        return {
            str(k):
                jsonable(v)
            for k, v in x.items()
        }

    if isinstance(
        x,
        (
            list,
            tuple,
        ),
    ):
        return [
            jsonable(v)
            for v in x
        ]

    if isinstance(
        x,
        (
            np.integer,
        ),
    ):
        return int(x)

    if isinstance(
        x,
        (
            np.floating,
        ),
    ):
        return float(x)

    if isinstance(
        x,
        np.ndarray,
    ):
        return x.tolist()

    if isinstance(
        x,
        Path,
    ):
        return str(x)

    if isinstance(
        x,
        (
            str,
            int,
            float,
            bool,
            type(None),
        ),
    ):
        return x

    raise TypeError(
        f"Unsupported JSON type: {type(x)}"
    )


def load_module(
    path,
    name,
):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            path
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def resolve_identity_columns(df):

    if "feature_source_cell_id" in df.columns:
        cell_col = "feature_source_cell_id"

    elif "raw_cell_id" in df.columns:
        cell_col = "raw_cell_id"

    else:
        raise AssertionError(
            "Benchmark dataframe has no raw-cell identity column"
        )


    if (
        "feature_raw_cycle_position_zero_based"
        in df.columns
    ):
        pos_col = (
            "feature_raw_cycle_position_zero_based"
        )

    elif (
        "raw_cycle_position_zero_based"
        in df.columns
    ):
        pos_col = (
            "raw_cycle_position_zero_based"
        )

    else:
        raise AssertionError(
            "Benchmark dataframe has no raw-cycle-position column"
        )

    return (
        cell_col,
        pos_col,
    )


def condition_blocks(
    condition,
    seed,
):

    if condition == "PRIMARY20_COMMON":
        return [
            (
                "PRIMARY20",
                PRIMARY_FEATURES,
                PRIMARY_FEATURES,
            ),
        ]

    if condition == "PRIMARY20_PLUS_FULL_DQDV10":
        return [
            (
                "PRIMARY20",
                PRIMARY_FEATURES,
                PRIMARY_FEATURES,
            ),
            (
                "FULL_DQDV10",
                [
                    "full_" + f
                    for f in DQDV_FEATURES
                ],
                DQDV_FEATURES,
            ),
        ]

    if condition == "PRIMARY20_PLUS_PREFIX25_DQDV10":
        return [
            (
                "PRIMARY20",
                PRIMARY_FEATURES,
                PRIMARY_FEATURES,
            ),
            (
                "PREFIX25_DQDV10",
                [
                    f"s{seed}_p25_{f}"
                    for f in DQDV_FEATURES
                ],
                DQDV_FEATURES,
            ),
        ]

    if condition == "PRIMARY20_PLUS_PREFIX50_DQDV10":
        return [
            (
                "PRIMARY20",
                PRIMARY_FEATURES,
                PRIMARY_FEATURES,
            ),
            (
                "PREFIX50_DQDV10",
                [
                    f"s{seed}_p50_{f}"
                    for f in DQDV_FEATURES
                ],
                DQDV_FEATURES,
            ),
        ]

    raise ValueError(
        condition
    )


def expected_parameter_count(
    condition,
):

    if condition == "PRIMARY20_COMMON":
        return 5569

    return 6209


def record_path(
    condition,
    split_seed,
    train_seed,
):
    return (
        RECORD_DIR
        / (
            f"{condition}"
            f"__split{split_seed}"
            f"__train{train_seed}.json"
        )
    )


def state_path(
    condition,
    split_seed,
    train_seed,
    model_name,
):
    return (
        STATE_DIR
        / (
            f"{condition}"
            f"__split{split_seed}"
            f"__train{train_seed}"
            f"__{model_name}.pt"
        )
    )


def validate_existing_record_native(
    path,
    condition,
    split_seed,
    train_seed,
):

    if not path.is_file():
        return False

    try:
        r = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        assert r[
            "contract"
        ] == "dqdv_training_pair_record_v1"

        assert r[
            "protocol_sha256"
        ] == EXPECTED_PROTOCOL_SHA

        assert r[
            "condition"
        ] == condition

        record_runner_sha = str(
            r[
                "runner_sha256"
            ]
        )

        current_runner_sha = sha256(
            SELF
        )

        if (
            record_runner_sha
            != current_runner_sha
        ):

            assert (
                record_runner_sha
                == COMPATIBLE_PRE_INPUT_BINDING_RUNNER_SHA
            )

            assert condition == "PRIMARY20_COMMON"
            assert split_seed == 0
            assert train_seed in {
                0,
                1,
                2,
            }

            assert int(
                r[
                    "input_dimension"
                ]
            ) == 20

        assert int(
            r[
                "split_seed"
            ]
        ) == split_seed

        assert int(
            r[
                "train_seed"
            ]
        ) == train_seed

        assert set(
            r[
                "models"
            ]
        ) == {
            "MLP",
            "CAUSAL_GCN",
        }

        for model_name in [
            "MLP",
            "CAUSAL_GCN",
        ]:

            m = r[
                "models"
            ][
                model_name
            ]

            sp = Path(
                m[
                    "state_path"
                ]
            )

            assert sp.is_file()

            assert sha256(
                sp
            ) == m[
                "state_sha256"
            ]

            assert int(
                m[
                    "parameter_count"
                ]
            ) == expected_parameter_count(
                condition
            )

            assert bool(
                m[
                    "patience_valid"
                ]
            )

        return True

    except Exception:
        return False

def imported_v1_record_entry(
    condition,
    split_seed,
    train_seed,
):

    assert IMPORT_MANIFEST.is_file()

    assert sha256(
        IMPORT_MANIFEST
    ) == EXPECTED_IMPORT_MANIFEST_SHA


    m = json.loads(
        IMPORT_MANIFEST.read_text(
            encoding="utf-8"
        )
    )


    assert m[
        "protocol_v2_sha256"
    ] == EXPECTED_PROTOCOL_SHA

    assert m[
        "parent_protocol_v1_sha256"
    ] == PARENT_PROTOCOL_V1_SHA

    assert m[
        "imported_pair_count"
    ] == 4

    assert m[
        "imported_model_fit_count"
    ] == 8

    assert m[
        "invalid_partial_v1_states_imported"
    ] is False

    assert m[
        "invalid_partial_v1_states_reused"
    ] is False


    key = (
        str(
            condition
        ),
        int(
            split_seed
        ),
        int(
            train_seed
        ),
    )


    matches = []

    for item in m[
        "imported_records"
    ]:

        item_key = (
            str(
                item[
                    "condition"
                ]
            ),
            int(
                item[
                    "split_seed"
                ]
            ),
            int(
                item[
                    "train_seed"
                ]
            ),
        )

        if item_key == key:
            matches.append(
                item
            )


    if not matches:
        return None


    assert len(
        matches
    ) == 1

    return matches[
        0
    ]



def validate_existing_record(
    path,
    condition,
    split_seed,
    train_seed,
):

    imported = imported_v1_record_entry(
        condition,
        split_seed,
        train_seed,
    )


    if imported is not None:

        p = Path(
            path
        )

        assert p.is_file()

        assert p.resolve() == Path(
            imported[
                "destination_record_path"
            ]
        ).resolve()


        expected_sha = imported[
            "destination_record_sha256"
        ]

        assert expected_sha == imported[
            "source_record_sha256"
        ]

        assert sha256(
            p
        ) == expected_sha

        assert imported[
            "record_bytes_preserved_exactly"
        ] is True


        for model_name in [
            "MLP",
            "CAUSAL_GCN",
        ]:

            ref = imported[
                "state_references"
            ][
                model_name
            ]

            assert ref[
                "copied_into_v2_namespace"
            ] is False

            assert ref[
                "immutable_v1_reference"
            ] is True


            state_path = Path(
                ref[
                    "source_path"
                ]
            )

            assert state_path.is_file()

            assert sha256(
                state_path
            ) == ref[
                "sha256"
            ]


        # Deliberately do not parse metric contents from imported
        # records during resume qualification.
        return True


    return validate_existing_record_native(
        path,
        condition,
        split_seed,
        train_seed,
    )



def load_feature_master(
    seed,
    cohort_seed,
):

    master = cohort_seed[
        [
            "dataset",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
            "split_group",
            "split",
        ]
    ].copy()

    master[
        "raw_cell_id"
    ] = master[
        "raw_cell_id"
    ].astype(
        str
    )

    master[
        "raw_cycle_position_zero_based"
    ] = pd.to_numeric(
        master[
            "raw_cycle_position_zero_based"
        ],
        errors="raise",
    ).astype(
        int
    )


    identity = [
        "dataset",
        "raw_cell_id",
        "raw_cycle_position_zero_based",
    ]


    raw_parts = []

    for dataset in DATASETS:

        r = pd.read_csv(
            RAW_FEATURE_DIR
            / f"{dataset}_raw_cycle_features.csv",
            usecols=[
                "dataset",
                "raw_cell_id",
                "raw_cycle_position_zero_based",
                *PRIMARY_FEATURES,
            ],
            low_memory=False,
        )

        r[
            "raw_cell_id"
        ] = r[
            "raw_cell_id"
        ].astype(
            str
        )

        r[
            "raw_cycle_position_zero_based"
        ] = pd.to_numeric(
            r[
                "raw_cycle_position_zero_based"
            ],
            errors="raise",
        ).astype(
            int
        )

        raw_parts.append(
            r
        )


    raw = pd.concat(
        raw_parts,
        ignore_index=True,
    )

    assert not raw[
        identity
    ].duplicated().any()

    master = master.merge(
        raw,
        on=identity,
        how="left",
        validate="one_to_one",
    )

    assert not master[
        PRIMARY_FEATURES
    ].isna().any().any()


    bank_parts = []

    for dataset in DATASETS:

        cols = [
            "dataset",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
        ]

        cols.extend(
            [
                "full_" + f
                for f in DQDV_FEATURES
            ]
        )

        cols.extend(
            [
                f"s{seed}_p25_{f}"
                for f in DQDV_FEATURES
            ]
        )

        cols.extend(
            [
                f"s{seed}_p50_{f}"
                for f in DQDV_FEATURES
            ]
        )

        b = pd.read_csv(
            FEATURE_BANK_DIR
            / f"{dataset}_dqdv_feature_bank.csv",
            usecols=cols,
            low_memory=False,
        )

        b[
            "raw_cell_id"
        ] = b[
            "raw_cell_id"
        ].astype(
            str
        )

        b[
            "raw_cycle_position_zero_based"
        ] = pd.to_numeric(
            b[
                "raw_cycle_position_zero_based"
            ],
            errors="raise",
        ).astype(
            int
        )

        bank_parts.append(
            b
        )


    bank = pd.concat(
        bank_parts,
        ignore_index=True,
    )

    assert not bank[
        identity
    ].duplicated().any()

    master = master.merge(
        bank,
        on=identity,
        how="left",
        validate="one_to_one",
    )

    assert len(
        master
    ) == len(
        cohort_seed
    )

    return master


def build_scaled_matrix(
    master,
    scaler_table,
    seed,
    condition,
):

    columns = []

    for block, source_cols, feature_names in condition_blocks(
        condition,
        seed,
    ):

        for source_col, feature in zip(
            source_cols,
            feature_names,
        ):

            q = scaler_table[
                (
                    scaler_table[
                        "split_seed"
                    ] == seed
                )
                & (
                    scaler_table[
                        "feature_block"
                    ] == block
                )
                & (
                    scaler_table[
                        "feature"
                    ] == feature
                )
            ]

            assert len(
                q
            ) == 1

            r = q.iloc[
                0
            ]

            assert str(
                r[
                    "source_column"
                ]
            ) == source_col

            assert str(
                r[
                    "fit_split"
                ]
            ) == "train_only"

            mean = float(
                r[
                    "mean"
                ]
            )

            scale = float(
                r[
                    "scale"
                ]
            )

            assert np.isfinite(
                mean
            )

            assert np.isfinite(
                scale
            )

            assert scale > 0

            values = pd.to_numeric(
                master[
                    source_col
                ],
                errors="raise",
            ).to_numpy(
                dtype=np.float64
            )

            assert np.isfinite(
                values
            ).all()

            columns.append(
                (
                    values - mean
                )
                / scale
            )


    x = np.column_stack(
        columns
    ).astype(
        np.float32,
        copy=False,
    )

    expected_dim = (
        20
        if condition == "PRIMARY20_COMMON"
        else 30
    )

    assert x.shape == (
        len(
            master
        ),
        expected_dim,
    )

    assert np.isfinite(
        x
    ).all()

    return x


def benchmark_frame_for_common_cohort(
    bench,
    seed,
    cohort_seed,
):

    # Authoritative final-v3 benchmark dataframe.
    # It intentionally retains canonical identity, not raw-source
    # feature pointers.
    df = bench.load_canonical_soh(
        seed
    ).copy()

    canonical_key = [
        "dataset",
        "split_group",
        "observation_index",
    ]

    if not set(
        canonical_key
    ).issubset(
        df.columns
    ):
        raise AssertionError(
            "Benchmark dataframe lacks canonical identity key"
        )

    if df.duplicated(
        canonical_key
    ).any():
        raise AssertionError(
            "Benchmark canonical identity is not unique"
        )


    # Common-cohort rows are anchored by raw feature provenance.
    c = cohort_seed[
        [
            "dataset",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
            "split_group",
            "split",
        ]
    ].copy()

    c[
        "raw_cell_id"
    ] = c[
        "raw_cell_id"
    ].astype(
        str
    )

    c[
        "raw_cycle_position_zero_based"
    ] = pd.to_numeric(
        c[
            "raw_cycle_position_zero_based"
        ],
        errors="raise",
    ).astype(
        int
    )

    c = c.rename(
        columns={
            "split_group":
                "_expected_split_group",

            "split":
                "_expected_split",
        }
    )


    # Canonical table provides the deterministic bridge from the
    # audited raw-source identity to the canonical observation identity
    # used by script24. No target/capacity column is read here.
    bridge = pd.read_csv(
        CANON,
        usecols=[
            "dataset",
            "split_group",
            "observation_index",
            "soh_task_eligible",
            "feature_source_cell_id",
            "feature_raw_cycle_position_zero_based",
        ],
        low_memory=False,
    )

    eligible = (
        bridge[
            "soh_task_eligible"
        ]
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

    bridge = bridge[
        eligible
    ].copy()

    bridge = bridge.rename(
        columns={
            "feature_source_cell_id":
                "raw_cell_id",

            "feature_raw_cycle_position_zero_based":
                "raw_cycle_position_zero_based",
        }
    )

    bridge[
        "raw_cell_id"
    ] = bridge[
        "raw_cell_id"
    ].astype(
        str
    )

    bridge[
        "raw_cycle_position_zero_based"
    ] = pd.to_numeric(
        bridge[
            "raw_cycle_position_zero_based"
        ],
        errors="raise",
    ).astype(
        int
    )

    bridge[
        "observation_index"
    ] = pd.to_numeric(
        bridge[
            "observation_index"
        ],
        errors="raise",
    ).astype(
        int
    )


    raw_key = [
        "dataset",
        "raw_cell_id",
        "raw_cycle_position_zero_based",
    ]

    if bridge.duplicated(
        raw_key
    ).any():
        raise AssertionError(
            "Canonical raw-source bridge is not unique"
        )

    if bridge.duplicated(
        canonical_key
    ).any():
        raise AssertionError(
            "Canonical observation bridge is not unique"
        )


    mapped = c.merge(
        bridge[
            [
                *raw_key,
                "split_group",
                "observation_index",
            ]
        ],
        on=raw_key,
        how="left",
        validate="one_to_one",
    )

    if mapped[
        "observation_index"
    ].isna().any():
        raise AssertionError(
            "Common-cohort raw row missing canonical observation"
        )

    if not (
        mapped[
            "split_group"
        ].astype(
            str
        )
        ==
        mapped[
            "_expected_split_group"
        ].astype(
            str
        )
    ).all():
        raise AssertionError(
            "Raw-to-canonical split_group mismatch"
        )


    # Attach final-v3 SOH/split/features using only the unique
    # canonical identity key. Keep raw provenance helper columns so
    # replace_x() can later align the audited 20D/30D feature matrix.
    mapped = mapped.rename(
        columns={
            "raw_cell_id":
                "_raw_cell_id",

            "raw_cycle_position_zero_based":
                "_raw_cycle_position_zero_based",
        }
    )

    q = mapped.merge(
        df,
        on=canonical_key,
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_benchmark",
        ),
    )

    if len(
        q
    ) != len(
        c
    ):
        raise AssertionError(
            "Canonical benchmark bridge changed row count"
        )

    if "SOH" not in q.columns:
        raise AssertionError(
            "Benchmark target missing after canonical bridge"
        )

    if q[
        "SOH"
    ].isna().any():
        raise AssertionError(
            "Common-cohort canonical row missing benchmark target"
        )

    if "split" not in q.columns:
        raise AssertionError(
            "Benchmark split missing after canonical bridge"
        )

    if not (
        q[
            "split"
        ].astype(
            str
        )
        ==
        q[
            "_expected_split"
        ].astype(
            str
        )
    ).all():
        raise AssertionError(
            "Common-cohort split differs from benchmark split"
        )

    if q.duplicated(
        canonical_key
    ).any():
        raise AssertionError(
            "Filtered benchmark canonical identity duplicated"
        )


    # ------------------------------------------------------------
    # Common-cohort filtering can remove observations because FULL,
    # PREFIX25, and PREFIX50 must all be valid. Therefore the
    # original canonical observation_index can contain gaps.
    #
    # Preserve the real canonical index for provenance/causality
    # checks, then create a contiguous LOCAL rank over retained
    # observations only. script24.make_split_tensor() can therefore
    # retain its original no-gap invariant while its prev operator
    # represents exactly the frozen dQ/dV contract:
    #
    #     previous valid / retained observation.
    #
    # observation_index is NOT a model feature.
    # ------------------------------------------------------------

    q[
        "_canonical_observation_index"
    ] = pd.to_numeric(
        q[
            "observation_index"
        ],
        errors="raise",
    ).astype(
        int
    )


    q = (
        q.sort_values(
            [
                "dataset",
                "split_group",
                "_canonical_observation_index",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )


    # Real canonical chronology must be strictly increasing inside
    # every retained cell sequence. Gaps are allowed here.
    for _, g in q.groupby(
        "split_group",
        sort=False,
    ):

        original = g[
            "_canonical_observation_index"
        ].to_numpy(
            dtype=np.int64
        )

        if len(
            original
        ) > 1:

            if not np.all(
                np.diff(
                    original
                ) > 0
            ):
                raise AssertionError(
                    "Retained canonical chronology is not strictly increasing"
                )


    # Local retained-observation rank used only for construction of
    # the causal previous-valid-observation operator.
    q[
        "observation_index"
    ] = (
        q.groupby(
            "split_group",
            sort=False,
        )
        .cumcount()
        .astype(
            int
        )
    )


    # The representation passed to make_split_tensor() must now
    # satisfy its original contiguous-sequence invariant.
    for _, g in q.groupby(
        "split_group",
        sort=False,
    ):

        local = g[
            "observation_index"
        ].to_numpy(
            dtype=np.int64
        )

        if len(
            local
        ) > 1:

            if not np.all(
                np.diff(
                    local
                ) == 1
            ):
                raise AssertionError(
                    "Retained observation rank is not contiguous"
                )


    return q


def authoritative_base_tensors(
    bench,
    filtered_df,
):

    out = {}

    for split_name in SPLIT_NAMES:

        d = bench.make_split_tensor(
            filtered_df,
            split_name,
        )

        assert set(
            [
                "frame",
                "x",
                "y",
                "prev",
                "weight",
                "n_rows",
                "n_cells",
            ]
        ).issubset(
            d
        )

        assert len(
            d[
                "frame"
            ]
        ) == int(
            d[
                "n_rows"
            ]
        )

        assert int(
            d[
                "n_rows"
            ]
        ) > 0

        assert torch.isfinite(
            d[
                "y"
            ]
        ).all()

        assert torch.isfinite(
            d[
                "weight"
            ]
        ).all()

        assert torch.all(
            d[
                "weight"
            ] > 0
        )


        # Independent semantic guard on the causal operator:
        # non-self previous pointers must stay within the same cell
        # and point backward in REAL canonical chronology, not only
        # in the temporary retained-observation rank.
        frame = d[
            "frame"
        ]

        if (
            "_canonical_observation_index"
            not in frame.columns
        ):
            raise AssertionError(
                "Preserved canonical observation index missing"
            )

        prev = (
            d[
                "prev"
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.int64
            )
        )

        idx = np.arange(
            len(
                frame
            ),
            dtype=np.int64,
        )

        nonself = (
            prev != idx
        )

        groups = frame[
            "split_group"
        ].astype(
            str
        ).to_numpy()

        canonical_obs = pd.to_numeric(
            frame[
                "_canonical_observation_index"
            ],
            errors="raise",
        ).to_numpy(
            dtype=np.int64
        )

        if np.any(
            nonself
        ):

            if not np.all(
                groups[
                    prev[
                        nonself
                    ]
                ]
                ==
                groups[
                    nonself
                ]
            ):
                raise AssertionError(
                    "Causal previous pointer crossed cell boundary"
                )

            if not np.all(
                canonical_obs[
                    prev[
                        nonself
                    ]
                ]
                <
                canonical_obs[
                    nonself
                ]
            ):
                raise AssertionError(
                    "Causal previous pointer is not backward in canonical chronology"
                )


        out[
            split_name
        ] = d

    return out


def replace_x(
    base_data,
    master,
    scaled_matrix,
):

    master_index = pd.MultiIndex.from_frame(
        master[
            [
                "dataset",
                "raw_cell_id",
                "raw_cycle_position_zero_based",
            ]
        ]
    )

    assert master_index.is_unique


    out = {}

    for split_name in SPLIT_NAMES:

        d = base_data[
            split_name
        ]

        frame = d[
            "frame"
        ]

        assert "_raw_cell_id" in frame.columns
        assert (
            "_raw_cycle_position_zero_based"
            in frame.columns
        )

        fkey = pd.MultiIndex.from_arrays(
            [
                frame[
                    "dataset"
                ].astype(
                    str
                ),

                frame[
                    "_raw_cell_id"
                ].astype(
                    str
                ),

                pd.to_numeric(
                    frame[
                        "_raw_cycle_position_zero_based"
                    ],
                    errors="raise",
                ).astype(
                    int
                ),
            ]
        )

        mkey = pd.MultiIndex.from_arrays(
            [
                master[
                    "dataset"
                ].astype(
                    str
                ),

                master[
                    "raw_cell_id"
                ].astype(
                    str
                ),

                master[
                    "raw_cycle_position_zero_based"
                ].astype(
                    int
                ),
            ]
        )

        # get_indexer requires exactly the index used here.
        positions = mkey.get_indexer(
            fkey
        )

        assert (
            positions >= 0
        ).all()

        x_np = scaled_matrix[
            positions
        ]

        x = torch.tensor(
            x_np,
            dtype=d[
                "x"
            ].dtype,
            device=d[
                "x"
            ].device,
        )

        nd = dict(
            d
        )

        nd[
            "x"
        ] = x

        assert nd[
            "x"
        ].shape[
            0
        ] == nd[
            "y"
        ].shape[
            0
        ]

        out[
            split_name
        ] = nd

    return out


def evaluate_model(
    bench,
    model,
    model_name,
    data,
):

    model.eval()

    result = {}

    with torch.no_grad():

        for split_name in SPLIT_NAMES:

            d = data[
                split_name
            ]

            if model_name == "MLP":

                pred = model(
                    d[
                        "x"
                    ]
                )

            elif model_name == "CAUSAL_GCN":

                pred = model(
                    d[
                        "x"
                    ],
                    d[
                        "prev"
                    ],
                )

            else:
                raise ValueError(
                    model_name
                )


            balanced = bench.weighted_metrics(
                pred,
                d[
                    "y"
                ],
                d[
                    "weight"
                ],
            )

            by_dataset = bench.dataset_metrics(
                pred,
                d[
                    "y"
                ],
                d[
                    "frame"
                ],
            )


            result[
                split_name
            ] = {
                "balanced":
                    jsonable(
                        balanced
                    ),

                "by_dataset":
                    jsonable(
                        by_dataset
                    ),
            }

    return result


def model_initialization_match(
    bench,
    input_dim,
    train_seed,
):

    bench.set_seed(
        train_seed
    )

    mlp = bench.FeatureMatchedMLP(
        input_dim=input_dim,
        hidden_dim=64,
    )

    bench.set_seed(
        train_seed
    )

    gcn = bench.FeatureMatchedCausalGCN(
        input_dim=input_dim,
        hidden_dim=64,
    )

    a = mlp.state_dict()
    b = gcn.state_dict()

    assert list(
        a
    ) == list(
        b
    )

    for key in a:
        assert torch.equal(
            a[
                key
            ],
            b[
                key
            ],
        )

    return True


def train_one_with_bound_input_dim(
    bench,
    model_name,
    data,
    train_seed,
    hidden_dim,
    lr,
    weight_decay,
    max_epochs,
    patience,
    input_dim,
):

    input_dim = int(
        input_dim
    )

    if input_dim not in {
        20,
        30,
    }:
        raise AssertionError(
            f"Unexpected input dimension: {input_dim}"
        )


    original_mlp = (
        bench.FeatureMatchedMLP
    )

    original_gcn = (
        bench.FeatureMatchedCausalGCN
    )


    # script24.train_one() calls the model constructors without an
    # input_dim argument, which is valid for its original fixed-20D
    # benchmark but not for the 30D dQ/dV conditions.
    #
    # These subclasses change ONLY the constructor default binding.
    # Forward equations, parameter tensors, initialization routine,
    # optimizer, objective, validation metric, and early stopping all
    # remain those of the audited script24 implementation.
    class BoundFeatureMatchedMLP(
        original_mlp
    ):

        def __init__(
            self,
            input_dim_unused=20,
            hidden_dim=64,
            **kwargs,
        ):

            super().__init__(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
            )


    class BoundFeatureMatchedCausalGCN(
        original_gcn
    ):

        def __init__(
            self,
            input_dim_unused=20,
            hidden_dim=64,
            **kwargs,
        ):

            super().__init__(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
            )


    bench.FeatureMatchedMLP = (
        BoundFeatureMatchedMLP
    )

    bench.FeatureMatchedCausalGCN = (
        BoundFeatureMatchedCausalGCN
    )


    try:

        return bench.train_one(
            model_name=model_name,
            data=data,
            train_seed=train_seed,
            hidden_dim=hidden_dim,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
        )

    finally:

        bench.FeatureMatchedMLP = (
            original_mlp
        )

        bench.FeatureMatchedCausalGCN = (
            original_gcn
        )



def fit_pair(
    bench,
    protocol,
    condition,
    split_seed,
    train_seed,
    data,
):

    expected_params = (
        expected_parameter_count(
            condition
        )
    )

    input_dim = int(
        data[
            "train"
        ][
            "x"
        ].shape[
            1
        ]
    )

    assert input_dim == (
        20
        if condition == "PRIMARY20_COMMON"
        else 30
    )

    assert model_initialization_match(
        bench,
        input_dim,
        train_seed,
    )


    model_records = {}


    for model_name in [
        "MLP",
        "CAUSAL_GCN",
    ]:

        result = train_one_with_bound_input_dim(
            bench=bench,
            model_name=model_name,
            data=data,
            train_seed=train_seed,
            hidden_dim=64,
            input_dim=input_dim,
            lr=float(
                protocol[
                    "training_contract"
                ][
                    "learning_rate"
                ]
            ),
            weight_decay=float(
                protocol[
                    "training_contract"
                ][
                    "weight_decay"
                ]
            ),
            max_epochs=int(
                protocol[
                    "training_contract"
                ][
                    "max_epochs"
                ]
            ),
            patience=int(
                protocol[
                    "training_contract"
                ][
                    "patience"
                ]
            ),
        )


        model = result[
            "model"
        ]

        parameter_count = int(
            result[
                "parameter_count"
            ]
        )

        assert parameter_count == expected_params


        history = jsonable(
            result[
                "history"
            ]
        )

        assert len(
            history
        ) > 0

        best_epoch = int(
            result[
                "best_epoch"
            ]
        )

        stop_epoch = int(
            history[
                -1
            ][
                "epoch"
            ]
        )

        patience = int(
            protocol[
                "training_contract"
            ][
                "patience"
            ]
        )

        max_epochs = int(
            protocol[
                "training_contract"
            ][
                "max_epochs"
            ]
        )

        patience_valid = bool(
            stop_epoch
            - best_epoch
            == patience
        )

        below_ceiling = bool(
            stop_epoch
            < max_epochs
        )


        metrics = evaluate_model(
            bench,
            model,
            model_name,
            data,
        )


        sp = state_path(
            condition,
            split_seed,
            train_seed,
            model_name,
        )

        state = {
            k:
                v.detach()
                .cpu()
                .clone()
            for k, v in model.state_dict().items()
        }

        atomic_torch_save(
            state,
            sp,
        )


        model_records[
            model_name
        ] = {
            "parameter_count":
                parameter_count,

            "best_epoch":
                best_epoch,

            "stop_epoch":
                stop_epoch,

            "epochs_after_best":
                (
                    stop_epoch
                    - best_epoch
                ),

            "patience_valid":
                patience_valid,

            "below_epoch_ceiling":
                below_ceiling,

            "best_validation_balanced_rmse":
                float(
                    result[
                        "best_validation_balanced_rmse"
                    ]
                ),

            "elapsed_seconds":
                float(
                    result[
                        "elapsed_seconds"
                    ]
                ),

            "history":
                history,

            "metrics":
                metrics,

            "state_path":
                str(
                    sp
                ),

            "state_sha256":
                sha256(
                    sp
                ),
        }


    assert model_records[
        "MLP"
    ][
        "parameter_count"
    ] == model_records[
        "CAUSAL_GCN"
    ][
        "parameter_count"
    ]

    assert model_records[
        "MLP"
    ][
        "patience_valid"
    ]

    assert model_records[
        "CAUSAL_GCN"
    ][
        "patience_valid"
    ]

    assert model_records[
        "MLP"
    ][
        "below_epoch_ceiling"
    ]

    assert model_records[
        "CAUSAL_GCN"
    ][
        "below_epoch_ceiling"
    ]


    record = {
        "contract":
            "dqdv_training_pair_record_v1",

        "protocol_sha256":
            EXPECTED_PROTOCOL_SHA,

        "runner_sha256":
            sha256(
                SELF
            ),

        "condition":
            condition,

        "split_seed":
            split_seed,

        "train_seed":
            train_seed,

        "input_dimension":
            input_dim,

        "initial_weights_equal_within_pair":
            True,

        "data_counts": {
            split_name:
                int(
                    data[
                        split_name
                    ][
                        "n_rows"
                    ]
                )
            for split_name in SPLIT_NAMES
        },

        "model_selection_uses_test_data":
            False,

        "test_metric_values_used_to_change_protocol":
            False,

        "test_metric_values_surfaced_during_execution":
            False,

        "models":
            model_records,
    }

    return record


def write_status(
    completed_pairs,
    completed_fits,
    current=None,
):

    atomic_json(
        {
            "contract":
                "dqdv_training_execution_status_v1",

            "protocol_sha256":
                EXPECTED_PROTOCOL_SHA,

            "completed_pairs":
                int(
                    completed_pairs
                ),

            "planned_pairs":
                60,

            "completed_model_fits":
                int(
                    completed_fits
                ),

            "planned_model_fits":
                120,

            "current":
                current,

            "test_metric_values_surfaced":
                False,
        },
        STATUS,
    )


# ==================================================================
# PRE-EXECUTION GUARDS
# ==================================================================

assert sha256(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

assert sha256(
    COMMON_MANIFEST
) == EXPECTED_COMMON_MANIFEST_SHA


common_manifest = json.loads(
    COMMON_MANIFEST.read_text(
        encoding="utf-8"
    )
)

assert sha256(
    CANON
) == common_manifest[
    "canonical_table_sha256"
]


protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)

assert protocol[
    "status"
] == "FROZEN_AFTER_VALIDATION_ONLY_CEILING_AMENDMENT_BEFORE_RESUME"

assert protocol[
    "training_contract"
][
    "expected_model_fits"
] == 120

assert protocol[
    "training_contract"
][
    "new_hyperparameter_tuning"
] is False

assert protocol[
    "evaluation_contract"
][
    "test_metric_values_may_be_surfaced_before_execution_audit"
] is False


benchmark_expected_sha = (
    protocol[
        "upstream_anchors"
    ][
        "feature_matched_model_implementation_sha256"
    ]
)

assert sha256(
    BENCHMARK_SCRIPT
) == benchmark_expected_sha


# Runner must be committed and clean before scientific execution.
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


bench = load_module(
    BENCHMARK_SCRIPT,
    "revision2026_feature_matched_benchmark",
)


# CPU-only execution.
torch.set_num_threads(
    1
)

if hasattr(
    torch,
    "set_num_interop_threads",
):
    try:
        torch.set_num_interop_threads(
            1
        )
    except RuntimeError:
        pass


if hasattr(
    bench,
    "DEVICE",
):
    bench.DEVICE = torch.device(
        "cpu"
    )


cohort = pd.read_csv(
    COHORT,
    low_memory=False,
)

scalers = pd.read_csv(
    SCALERS,
    low_memory=False,
)

cohort[
    "split_seed"
] = pd.to_numeric(
    cohort[
        "split_seed"
    ],
    errors="raise",
).astype(
    int
)

cohort[
    "raw_cell_id"
] = cohort[
    "raw_cell_id"
].astype(
    str
)

cohort[
    "raw_cycle_position_zero_based"
] = pd.to_numeric(
    cohort[
        "raw_cycle_position_zero_based"
    ],
    errors="raise",
).astype(
    int
)

scalers[
    "split_seed"
] = pd.to_numeric(
    scalers[
        "split_seed"
    ],
    errors="raise",
).astype(
    int
)


EXEC.mkdir(
    parents=True,
    exist_ok=True,
)

RECORD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

STATE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


planned_set = {
    (
        str(
            r[
                "condition"
            ]
        ),
        int(
            r[
                "split_seed"
            ]
        ),
        int(
            r[
                "train_seed"
            ]
        ),
    )
    for r in protocol[
        "planned_runs"
    ]
}

assert len(
    planned_set
) == 60


completed_pairs = 0

for condition, split_seed, train_seed in sorted(
    planned_set
):

    if validate_existing_record(
        record_path(
            condition,
            split_seed,
            train_seed,
        ),
        condition,
        split_seed,
        train_seed,
    ):
        completed_pairs += 1


write_status(
    completed_pairs,
    completed_pairs * 2,
    current=None,
)


print(
    "DQDV TRAINING EXECUTION START"
)

print(
    "protocol exact:",
    True
)

print(
    "planned pairs:",
    60
)

print(
    "planned model fits:",
    120
)

print(
    "resume completed pairs:",
    completed_pairs
)

print(
    "test metric values surfaced:",
    False
)

print(
    "DATA_INTERFACE_PREFLIGHT_BEGIN=True"
)


# ==================================================================
# TRAINING — seed outer loop keeps one seed's feature data in memory.
# ==================================================================

for split_seed in SEEDS:

    cohort_seed = cohort[
        cohort[
            "split_seed"
        ] == split_seed
    ].copy()

    assert len(
        cohort_seed
    ) > 0


    filtered_benchmark = (
        benchmark_frame_for_common_cohort(
            bench,
            split_seed,
            cohort_seed,
        )
    )


    base_data = (
        authoritative_base_tensors(
            bench,
            filtered_benchmark,
        )
    )


    # Exact split-count agreement with frozen cohort.
    expected_counts = (
        cohort_seed[
            "split"
        ]
        .value_counts()
        .to_dict()
    )

    for split_name in SPLIT_NAMES:

        assert int(
            base_data[
                split_name
            ][
                "n_rows"
            ]
        ) == int(
            expected_counts[
                split_name
            ]
        )


    master = load_feature_master(
        split_seed,
        cohort_seed,
    )


    scaled_by_condition = {}

    data_by_condition = {}


    for condition in CONDITIONS:

        x = build_scaled_matrix(
            master,
            scalers,
            split_seed,
            condition,
        )

        scaled_by_condition[
            condition
        ] = x

        data_by_condition[
            condition
        ] = replace_x(
            base_data,
            master,
            x,
        )


        expected_dim = (
            20
            if condition == "PRIMARY20_COMMON"
            else 30
        )

        assert (
            data_by_condition[
                condition
            ][
                "train"
            ][
                "x"
            ].shape[
                1
            ]
            == expected_dim
        )


    print(
        f"SEED_DATA_READY split_seed={split_seed} "
        f"rows={len(cohort_seed)}"
    )


    if split_seed == 0:
        print(
            "DATA_INTERFACE_PREFLIGHT_PASS=True"
        )


    if PREFLIGHT_ONLY:

        print(
            f"SEED_PREFLIGHT_PASS "
            f"split_seed={split_seed} "
            f"rows={len(cohort_seed)} "
            f"conditions={len(CONDITIONS)} "
            f"previous_valid_observation_contract=True"
        )

        del data_by_condition
        del scaled_by_condition
        del master
        del base_data
        del filtered_benchmark

        continue


    for condition in CONDITIONS:

        data = data_by_condition[
            condition
        ]


        for train_seed in TRAIN_SEEDS:

            key = (
                condition,
                split_seed,
                train_seed,
            )

            assert key in planned_set


            rp = record_path(
                condition,
                split_seed,
                train_seed,
            )


            if validate_existing_record(
                rp,
                condition,
                split_seed,
                train_seed,
            ):

                print(
                    f"PAIR_REUSED "
                    f"condition={condition} "
                    f"split_seed={split_seed} "
                    f"train_seed={train_seed}"
                )

                continue


            write_status(
                completed_pairs,
                completed_pairs * 2,
                current={
                    "condition":
                        condition,

                    "split_seed":
                        split_seed,

                    "train_seed":
                        train_seed,
                },
            )


            t0 = time.time()


            record = fit_pair(
                bench,
                protocol,
                condition,
                split_seed,
                train_seed,
                data,
            )


            record[
                "pair_wall_seconds"
            ] = float(
                time.time()
                - t0
            )


            atomic_json(
                jsonable(
                    record
                ),
                rp,
            )


            assert validate_existing_record(
                rp,
                condition,
                split_seed,
                train_seed,
            )


            completed_pairs += 1


            write_status(
                completed_pairs,
                completed_pairs * 2,
                current=None,
            )


            mlp = record[
                "models"
            ][
                "MLP"
            ]

            gcn = record[
                "models"
            ][
                "CAUSAL_GCN"
            ]


            print(
                f"PAIR_COMPLETE "
                f"condition={condition} "
                f"split_seed={split_seed} "
                f"train_seed={train_seed} "
                f"completed={completed_pairs}/60 "
                f"MLP_best_stop="
                f"{mlp['best_epoch']}/{mlp['stop_epoch']} "
                f"GCN_best_stop="
                f"{gcn['best_epoch']}/{gcn['stop_epoch']} "
                f"test_metrics_surfaced=False"
            )


    # Explicitly drop large seed-local arrays before next seed.
    del data_by_condition
    del scaled_by_condition
    del master
    del base_data
    del filtered_benchmark


# ==================================================================
# PREFLIGHT-ONLY CLOSURE
# ==================================================================

if PREFLIGHT_ONLY:

    print(
        "ALL_SEED_DQDV_DATA_INTERFACE_PREFLIGHT_PASS=True"
    )

    print(
        "split_seeds_checked=5"
    )

    print(
        "conditions_checked_per_seed=4"
    )

    print(
        "previous_valid_observation_contract=True"
    )

    print(
        "real_canonical_chronology_preserved=True"
    )

    print(
        "model_training=False"
    )

    print(
        "test_metric_values_computed=False"
    )

    print(
        "test_metric_values_surfaced=False"
    )

    raise SystemExit(
        0
    )


# ==================================================================
# EXECUTION CLOSURE — no test metric values are read here.
# ==================================================================

records = []

for condition, split_seed, train_seed in sorted(
    planned_set
):

    rp = record_path(
        condition,
        split_seed,
        train_seed,
    )

    assert validate_existing_record(
        rp,
        condition,
        split_seed,
        train_seed,
    )

    r = json.loads(
        rp.read_text(
            encoding="utf-8"
        )
    )

    records.append({
        "condition":
            condition,

        "split_seed":
            split_seed,

        "train_seed":
            train_seed,

        "record_path":
            str(
                rp
            ),

        "record_sha256":
            sha256(
                rp
            ),

        "mlp_best_epoch":
            int(
                r[
                    "models"
                ][
                    "MLP"
                ][
                    "best_epoch"
                ]
            ),

        "mlp_stop_epoch":
            int(
                r[
                    "models"
                ][
                    "MLP"
                ][
                    "stop_epoch"
                ]
            ),

        "mlp_patience_valid":
            bool(
                r[
                    "models"
                ][
                    "MLP"
                ][
                    "patience_valid"
                ]
            ),

        "gcn_best_epoch":
            int(
                r[
                    "models"
                ][
                    "CAUSAL_GCN"
                ][
                    "best_epoch"
                ]
            ),

        "gcn_stop_epoch":
            int(
                r[
                    "models"
                ][
                    "CAUSAL_GCN"
                ][
                    "stop_epoch"
                ]
            ),

        "gcn_patience_valid":
            bool(
                r[
                    "models"
                ][
                    "CAUSAL_GCN"
                ][
                    "patience_valid"
                ]
            ),
    })


assert len(
    records
) == 60

assert all(
    r[
        "mlp_patience_valid"
    ]
    and r[
        "gcn_patience_valid"
    ]
    for r in records
)

assert all(
    r[
        "mlp_stop_epoch"
    ] < 10000
    and r[
        "gcn_stop_epoch"
    ] < 10000
    for r in records
)


execution_manifest = {
    "contract":
        "dqdv_training_execution_v2",

    "status":
        "COMPLETE_PENDING_INDEPENDENT_AUDIT",

    "protocol_sha256":
        EXPECTED_PROTOCOL_SHA,

    "parent_protocol_v1_sha256":
        PARENT_PROTOCOL_V1_SHA,

    "compatible_v1_import_manifest_path":
        str(
            IMPORT_MANIFEST
        ),

    "compatible_v1_import_manifest_sha256":
        EXPECTED_IMPORT_MANIFEST_SHA,

    "compatible_v1_imported_pair_count":
        4,

    "compatible_v1_imported_model_fit_count":
        8,

    "compatible_v1_records_retrained":
        False,

    "invalid_partial_v1_states_reused":
        False,

    "validation_safety_ceiling":
        10000,

    "validation_patience":
        30,

    "common_cohort_manifest_sha256":
        EXPECTED_COMMON_MANIFEST_SHA,

    "runner_path":
        str(
            SELF
        ),

    "runner_sha256":
        sha256(
            SELF
        ),

    "compatible_pre_input_binding_runner_sha256":
        COMPATIBLE_PRE_INPUT_BINDING_RUNNER_SHA,

    "compatible_pre_input_binding_records":
        (
            "PRIMARY20_COMMON split_seed=0 train_seed=0,1,2 only; "
            "20D constructor behavior is identical before/after binding patch"
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

    "benchmark_script_path":
        str(
            BENCHMARK_SCRIPT
        ),

    "benchmark_script_sha256":
        sha256(
            BENCHMARK_SCRIPT
        ),

    "device":
        "cpu",

    "torch_threads":
        1,

    "planned_pair_count":
        60,

    "completed_pair_count":
        60,

    "planned_model_fit_count":
        120,

    "completed_model_fit_count":
        120,

    "all_fits_patience_valid":
        True,

    "all_fits_below_epoch_ceiling":
        True,

    "epoch_ceiling":
        5000,

    "patience":
        30,

    "test_metric_values_parsed_during_closure":
        False,

    "test_metric_values_surfaced_during_execution":
        False,

    "test_metric_values_used_to_change_protocol":
        False,

    "protocol_changed_after_test_evaluation":
        False,

    "records":
        records,
}


atomic_json(
    execution_manifest,
    MANIFEST,
)


write_status(
    60,
    120,
    current=None,
)


print(
    "DQDV TRAINING EXECUTION CLOSURE"
)

print(
    "completed pairs:",
    60
)

print(
    "completed model fits:",
    120
)

print(
    "all fits patience-valid:",
    True
)

print(
    "all fits below epoch ceiling:",
    True
)

print(
    "test metric values surfaced:",
    False
)

print(
    "test metric values used to change protocol:",
    False
)

print(
    "execution_manifest_sha256:",
    sha256(
        MANIFEST
    )
)

print(
    "DQDV_TRAINING_EXECUTION_COMPLETE_READY_FOR_AUDIT=True"
)
