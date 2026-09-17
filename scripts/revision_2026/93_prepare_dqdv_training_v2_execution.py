from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess


BASE = Path.home() / "alper/battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

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

RECORD_DIR = V2_EXEC / "run_records"

IMPORT_MANIFEST = (
    V2_EXEC
    / "v1_compatible_import_manifest.json"
)

SELF = (
    REPO
    / "scripts/revision_2026/93_prepare_dqdv_training_v2_execution.py"
)

EXPECTED_V2_PROTOCOL_SHA = (
    "4612a450eeaf909a04c31be16957bdc0567aac67635241e14d7bbceea507fe9d"
)

EXPECTED_V1_PROTOCOL_SHA = (
    "43122c4d6ef5a2e70b2d13ae22432a2cfa13027874e0bea6f3b62675e3fb23e9"
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


def atomic_copy_exact(src, dst):

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(dst) + ".tmp"
    )

    shutil.copyfile(
        src,
        tmp,
    )

    if sha256(
        src
    ) != sha256(
        tmp
    ):
        raise AssertionError(
            "copy hash mismatch"
        )

    os.replace(
        tmp,
        dst,
    )


assert sha256(
    V2_PROTOCOL
) == EXPECTED_V2_PROTOCOL_SHA


# Preparation script itself must be committed and clean.
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


p = json.loads(
    V2_PROTOCOL.read_text(
        encoding="utf-8"
    )
)

assert p[
    "contract"
] == "dqdv_training_protocol_v2"

assert p[
    "parent_protocol_sha256"
] == EXPECTED_V1_PROTOCOL_SHA

amend = p[
    "ceiling_amendment"
]

assert amend[
    "compatible_completed_pair_count"
] == 4

assert amend[
    "compatible_completed_model_fit_count"
] == 8

assert amend[
    "invalid_partial_v1_states_reused"
] is False


compatible = amend[
    "compatible_v1_records"
]

assert len(
    compatible
) == 4


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
imported = []
state_reference_count = 0


for item in compatible:

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
    assert key not in seen

    seen.add(
        key
    )


    source_record = Path(
        item[
            "record_path"
        ]
    )

    assert source_record.is_file()

    assert sha256(
        source_record
    ) == item[
        "record_sha256"
    ]


    # Validate state provenance exclusively from the already frozen
    # v2 protocol metadata. The source record's metric branch does
    # not need to be parsed or inspected here.
    state_refs = {}

    for model_name in [
        "MLP",
        "CAUSAL_GCN",
    ]:

        m = item[
            "models"
        ][
            model_name
        ]

        assert (
            int(
                m[
                    "stop_epoch"
                ]
            )
            -
            int(
                m[
                    "best_epoch"
                ]
            )
        ) == 30

        assert int(
            m[
                "stop_epoch"
            ]
        ) < 5000


        state_path = Path(
            m[
                "state_path"
            ]
        )

        assert state_path.is_file()

        assert sha256(
            state_path
        ) == m[
            "state_sha256"
        ]

        state_refs[
            model_name
        ] = {
            "source_path":
                str(
                    state_path
                ),

            "sha256":
                m[
                    "state_sha256"
                ],

            "copied_into_v2_namespace":
                False,

            "immutable_v1_reference":
                True,
        }

        state_reference_count += 1


    destination = (
        RECORD_DIR
        / source_record.name
    )

    atomic_copy_exact(
        source_record,
        destination,
    )

    assert sha256(
        destination
    ) == item[
        "record_sha256"
    ]


    imported.append({
        "condition":
            key[
                0
            ],

        "split_seed":
            key[
                1
            ],

        "train_seed":
            key[
                2
            ],

        "source_record_path":
            str(
                source_record
            ),

        "destination_record_path":
            str(
                destination
            ),

        "source_record_sha256":
            item[
                "record_sha256"
            ],

        "destination_record_sha256":
            sha256(
                destination
            ),

        "record_bytes_preserved_exactly":
            True,

        "state_references":
            state_refs,
    })


assert seen == expected_pairs

assert state_reference_count == 8


# Incomplete v1 pair states remain outside v2 and are never copied.
invalid_partial = amend[
    "invalid_partial_v1_states"
]

assert len(
    invalid_partial
) == 2

for item in invalid_partial:

    assert item[
        "reusable_in_v2"
    ] is False

    path = Path(
        item[
            "path"
        ]
    )

    assert path.is_file()

    assert sha256(
        path
    ) == item[
        "sha256"
    ]

    assert not (
        V2_EXEC
        in path.parents
    )


assert not (
    V2_EXEC
    / "model_states"
).exists()


manifest = {
    "contract":
        "dqdv_training_v2_compatible_v1_import_v1",

    "status":
        "COMPATIBLE_V1_RECORD_IMPORT_COMPLETE_BEFORE_V2_TRAINING",

    "protocol_v2_sha256":
        EXPECTED_V2_PROTOCOL_SHA,

    "parent_protocol_v1_sha256":
        EXPECTED_V1_PROTOCOL_SHA,

    "imported_pair_count":
        4,

    "imported_model_fit_count":
        8,

    "imported_record_count":
        4,

    "state_reference_count":
        8,

    "record_bytes_preserved_exactly":
        True,

    "model_states_copied":
        False,

    "model_states_referenced_from_v1":
        True,

    "invalid_partial_v1_state_count":
        2,

    "invalid_partial_v1_states_imported":
        False,

    "invalid_partial_v1_states_reused":
        False,

    "test_metric_values_from_v1_records_parsed":
        False,

    "test_metric_values_inspected":
        False,

    "test_metric_values_surfaced":
        False,

    "test_metric_values_used_for_import":
        False,

    "model_training_performed":
        False,

    "imported_records":
        imported,

    "preparer_path":
        str(
            SELF
        ),

    "preparer_sha256":
        sha256(
            SELF
        ),

    "preparer_commit":
        subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=REPO,
            text=True,
        ).strip(),
}


atomic_json(
    manifest,
    IMPORT_MANIFEST,
)


print(
    "DQDV TRAINING V2 EXECUTION IMPORT"
)

print(
    "protocol v2 exact:",
    True
)

print(
    "compatible v1 pairs imported:",
    4
)

print(
    "compatible model fits represented:",
    8
)

print(
    "record bytes preserved exactly:",
    True
)

print(
    "v1 model states referenced:",
    8
)

print(
    "v1 model states copied:",
    False
)

print(
    "invalid partial states imported:",
    False
)

print(
    "invalid partial states reused:",
    False
)

print(
    "v1 test metric values parsed:",
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
    "DQDV_TRAINING_V2_IMPORT_PASS=True"
)
