from pathlib import Path
import hashlib
import json

import numpy as np


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo" / "battery-gnn"

ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "physical_sensor_noise"
    / "final_physical_v1"
)

OUT = ROOT / "corrupted_features"
CELL_DIR = OUT / "cells"

CONDITIONS = OUT / "conditions.json"
CLOSURE = OUT / "corruption_feature_closure.json"

PROTOCOL = ROOT / "protocol.json"

ROUNDTRIP_SUMMARY = (
    ROOT
    / "zero_corruption_roundtrip"
    / "roundtrip_summary.json"
)

GENERATOR = (
    REPO
    / "scripts"
    / "revision_2026"
    / "55_physical_sensor_corruption_feature_generator.py"
)


EXPECTED_PROTOCOL_SHA = (
    "7ae0836fe69c4b14a1f93a094b2d42bb96209e302c8262f7b8bd8625baf78bfc"
)

EXPECTED_ROUNDTRIP_SHA = (
    "6c783a9d9396e3621dbd043d833ea61224c34f1c25d2fc03c4cf22cfc0dd5f99"
)


def sha(path):
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


assert sha(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

assert sha(
    ROUNDTRIP_SUMMARY
) == EXPECTED_ROUNDTRIP_SHA


closure = json.loads(
    CLOSURE.read_text(
        encoding="utf-8"
    )
)

conditions = json.loads(
    CONDITIONS.read_text(
        encoding="utf-8"
    )
)


assert closure[
    "contract"
] == "physical_corruption_feature_closure_v1"

assert closure[
    "status"
] == "COMPLETE_BEFORE_ANY_MODEL_EVALUATION"

assert closure[
    "protocol_sha256"
] == EXPECTED_PROTOCOL_SHA

assert closure[
    "roundtrip_summary_sha256"
] == EXPECTED_ROUNDTRIP_SHA

assert closure[
    "generator_sha256"
] == sha(
    GENERATOR
)

assert closure[
    "condition_count"
] == 66

assert closure[
    "clean_condition_count"
] == 1

assert closure[
    "corrupted_condition_count"
] == 65

assert closure[
    "cell_count"
] == 143

assert closure[
    "row_count"
] == 166304

assert closure[
    "feature_count"
] == 20

assert closure[
    "all_shards_complete"
] is True

assert closure[
    "all_features_finite"
] is True

assert closure[
    "model_loaded"
] is False

assert closure[
    "model_evaluated"
] is False

assert closure[
    "metric_computed"
] is False


assert conditions[
    "contract"
] == "physical_sensor_corruption_conditions_v1"

assert conditions[
    "condition_count"
] == 66

assert len(
    conditions[
        "conditions"
    ]
) == 66

assert conditions[
    "conditions"
][
    0
][
    "condition_id"
] == "clean"

assert conditions[
    "conditions"
][
    0
][
    "severity_axis"
] == 0.0


ids = [
    x[
        "condition_id"
    ]
    for x in conditions[
        "conditions"
    ]
]

assert len(
    ids
) == len(
    set(
        ids
    )
) == 66


primary = [
    x
    for x in conditions[
        "conditions"
    ]
    if x[
        "scenario"
    ] in {
        "gaussian",
        "calibration_offset",
        "sensor_drift",
        "sample_dropout",
    }
]

compound = [
    x
    for x in conditions[
        "conditions"
    ]
    if x[
        "scenario"
    ] == "compound_medium"
]


assert len(
    primary
) == 60

assert len(
    compound
) == 5


npz_files = sorted(
    CELL_DIR.glob(
        "*.npz"
    )
)

json_files = sorted(
    CELL_DIR.glob(
        "*.json"
    )
)


assert len(
    npz_files
) == 143

assert len(
    json_files
) == 143


record_map = {
    (
        x[
            "dataset"
        ],
        x[
            "cell_id"
        ]
    ):
        x
    for x in closure[
        "records"
    ]
}


assert len(
    record_map
) == 143


total_rows = 0
dataset_rows = {}
dataset_cells = {}


for meta_path in json_files:

    meta = json.loads(
        meta_path.read_text(
            encoding="utf-8"
        )
    )


    assert meta[
        "contract"
    ] == "physical_corruption_feature_cell_v1"

    assert meta[
        "protocol_sha256"
    ] == EXPECTED_PROTOCOL_SHA

    assert meta[
        "roundtrip_summary_sha256"
    ] == EXPECTED_ROUNDTRIP_SHA

    assert meta[
        "generator_sha256"
    ] == sha(
        GENERATOR
    )

    assert meta[
        "condition_count"
    ] == 66

    assert meta[
        "feature_count"
    ] == 20

    assert meta[
        "all_features_finite"
    ] is True

    assert meta[
        "contains_model_predictions"
    ] is False

    assert meta[
        "contains_targets"
    ] is False

    assert meta[
        "contains_model_metrics"
    ] is False


    dataset = meta[
        "dataset"
    ]

    cell = meta[
        "cell_id"
    ]


    key = (
        dataset,
        cell,
    )

    assert key in record_map


    npz_path = Path(
        record_map[
            key
        ][
            "npz_path"
        ]
    )


    assert npz_path.is_file()

    assert sha(
        npz_path
    ) == meta[
        "npz_sha256"
    ]


    with np.load(
        npz_path,
        allow_pickle=False,
    ) as z:

        features = z[
            "features"
        ]

        positions = z[
            "positions"
        ]

        condition_ids = z[
            "condition_ids"
        ]

        cycle_keys = z[
            "cycle_keys"
        ]


        assert features.shape == (
            66,
            int(
                meta[
                    "row_count"
                ]
            ),
            20,
        )

        assert len(
            positions
        ) == int(
            meta[
                "row_count"
            ]
        )

        assert len(
            cycle_keys
        ) == int(
            meta[
                "row_count"
            ]
        )

        assert condition_ids.tolist() == ids

        assert np.isfinite(
            features
        ).all()

        assert np.all(
            np.diff(
                positions.astype(
                    np.int64
                )
            )
            > 0
        )


    n = int(
        meta[
            "row_count"
        ]
    )

    total_rows += n

    dataset_rows[
        dataset
    ] = (
        dataset_rows.get(
            dataset,
            0,
        )
        + n
    )

    dataset_cells[
        dataset
    ] = (
        dataset_cells.get(
            dataset,
            0,
        )
        + 1
    )


assert total_rows == 166304

assert dataset_rows == {
    k:
        int(v)
    for k, v in closure[
        "dataset_row_counts"
    ].items()
}

assert dataset_cells == {
    k:
        int(v)
    for k, v in closure[
        "dataset_cell_counts"
    ].items()
}


# No evaluation artifact is permitted at this stage.
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


print("=" * 80)
print("PHYSICAL SENSOR CORRUPTION FEATURE AUDIT")
print("=" * 80)

print(
    "protocol exact:",
    True
)

print(
    "roundtrip barrier exact:",
    True
)

print(
    "conditions:",
    66
)

print(
    "clean / corrupted:",
    "1 / 65"
)

print(
    "primary / compound corrupted:",
    "60 / 5"
)

print(
    "cell shards:",
    len(
        npz_files
    )
)

print(
    "audited rows:",
    total_rows
)

print(
    "feature dimensions:",
    20
)

print(
    "all corrupted features finite:",
    True
)

print(
    "all condition order exact:",
    True
)

print(
    "model predictions present:",
    False
)

print(
    "model metrics present:",
    False
)

print(
    "generator SHA256:",
    sha(
        GENERATOR
    )
)

print(
    "closure SHA256:",
    sha(
        CLOSURE
    )
)

print()
print(
    "PHYSICAL_SENSOR_CORRUPTION_FEATURE_AUDIT_PASS=True"
)
