from pathlib import Path
import hashlib
import json


BASE = Path.home() / "alper/battery_revision_2026"

ROOT = (
    BASE
    / "results/revision_2026/dqdv_leakage_control/final_dqdv_v1"
)

V2_PROTOCOL = (
    ROOT
    / "training_v2/dqdv_training_protocol_v2.json"
)

V2_EXEC = (
    ROOT
    / "training_v2/final_execution_v2"
)

IMPORT_MANIFEST = (
    V2_EXEC
    / "v1_compatible_import_manifest.json"
)

EXPECTED_V2_PROTOCOL_SHA = (
    "4612a450eeaf909a04c31be16957bdc0567aac67635241e14d7bbceea507fe9d"
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

            h.update(
                b
            )

    return h.hexdigest()


assert sha256(
    V2_PROTOCOL
) == EXPECTED_V2_PROTOCOL_SHA


p = json.loads(
    V2_PROTOCOL.read_text(
        encoding="utf-8"
    )
)

m = json.loads(
    IMPORT_MANIFEST.read_text(
        encoding="utf-8"
    )
)


assert m[
    "contract"
] == "dqdv_training_v2_compatible_v1_import_v1"

assert m[
    "status"
] == "COMPATIBLE_V1_RECORD_IMPORT_COMPLETE_BEFORE_V2_TRAINING"

assert m[
    "protocol_v2_sha256"
] == EXPECTED_V2_PROTOCOL_SHA

assert m[
    "imported_pair_count"
] == 4

assert m[
    "imported_model_fit_count"
] == 8

assert m[
    "imported_record_count"
] == 4

assert m[
    "state_reference_count"
] == 8

assert m[
    "record_bytes_preserved_exactly"
] is True

assert m[
    "model_states_copied"
] is False

assert m[
    "model_states_referenced_from_v1"
] is True

assert m[
    "invalid_partial_v1_states_imported"
] is False

assert m[
    "invalid_partial_v1_states_reused"
] is False

assert m[
    "test_metric_values_from_v1_records_parsed"
] is False

assert m[
    "test_metric_values_inspected"
] is False

assert m[
    "test_metric_values_surfaced"
] is False

assert m[
    "model_training_performed"
] is False


expected_pairs = {
    (
        "PRIMARY20_COMMON",
        0,
        0,
    ),
    (
        "PRIMARY20_COMMON",
        0,
        1,
    ),
    (
        "PRIMARY20_COMMON",
        0,
        2,
    ),
    (
        "PRIMARY20_PLUS_FULL_DQDV10",
        0,
        0,
    ),
}


seen = set()
state_hashes = set()


for item in m[
    "imported_records"
]:

    key = (
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

    assert key in expected_pairs

    seen.add(
        key
    )


    src = Path(
        item[
            "source_record_path"
        ]
    )

    dst = Path(
        item[
            "destination_record_path"
        ]
    )

    assert src.is_file()
    assert dst.is_file()

    expected_hash = item[
        "source_record_sha256"
    ]

    assert sha256(
        src
    ) == expected_hash

    assert sha256(
        dst
    ) == expected_hash

    assert item[
        "destination_record_sha256"
    ] == expected_hash

    assert item[
        "record_bytes_preserved_exactly"
    ] is True

    # Byte-for-byte check in addition to SHA equality.
    assert src.read_bytes() == dst.read_bytes()


    for model_name in [
        "MLP",
        "CAUSAL_GCN",
    ]:

        q = item[
            "state_references"
        ][
            model_name
        ]

        sp = Path(
            q[
                "source_path"
            ]
        )

        assert sp.is_file()

        assert sha256(
            sp
        ) == q[
            "sha256"
        ]

        assert q[
            "copied_into_v2_namespace"
        ] is False

        assert q[
            "immutable_v1_reference"
        ] is True

        state_hashes.add(
            q[
                "sha256"
            ]
        )


assert seen == expected_pairs

assert len(
    state_hashes
) == 8


# Exactly four imported record files exist locally.
record_files = list(
    (
        V2_EXEC
        / "run_records"
    ).glob(
        "*.json"
    )
)

assert len(
    record_files
) == 4


# No state from either valid or invalid v1 runs has been copied.
assert not (
    V2_EXEC
    / "model_states"
).exists()


# Cross-check amendment still says incomplete states cannot be reused.
amend = p[
    "ceiling_amendment"
]

assert amend[
    "compatible_completed_pair_count"
] == 4

assert amend[
    "invalid_partial_v1_states_reused"
] is False

assert len(
    amend[
        "invalid_partial_v1_states"
    ]
) == 2


print(
    "DQDV TRAINING V2 IMPORT AUDIT"
)

print(
    "protocol v2 exact:",
    True
)

print(
    "imported pair set exact:",
    True
)

print(
    "imported record count:",
    4
)

print(
    "all imported records byte-identical to v1:",
    True
)

print(
    "immutable valid state references verified:",
    8
)

print(
    "model states copied into v2 namespace:",
    False
)

print(
    "invalid partial states present in import:",
    False
)

print(
    "test metric values parsed from imported records:",
    False
)

print(
    "test metric values surfaced:",
    False
)

print(
    "model training performed:",
    False
)

print(
    "import_manifest_sha256:",
    sha256(
        IMPORT_MANIFEST
    )
)

print(
    "DQDV_TRAINING_V2_IMPORT_AUDIT_PASS=True"
)
