from pathlib import Path
import hashlib
import json
import os
import shutil


BASE = Path.home() / "alper/battery_revision_2026"

REPO = (
    BASE
    / "repo/battery-gnn"
)

ROOT = (
    BASE
    / "results/revision_2026/dqdv_leakage_control/final_dqdv_v1"
)

EXEC = (
    ROOT
    / "training_v2/final_execution_v2"
)

PROTOCOL = (
    ROOT
    / "training_v2/dqdv_training_protocol_v2.json"
)

IMPORT_MANIFEST = (
    EXEC
    / "v1_compatible_import_manifest.json"
)

MANIFEST = (
    EXEC
    / "execution_manifest.json"
)

BACKUP = (
    EXEC
    / "execution_manifest.pre_metadata_correction.json"
)

RECEIPT = (
    EXEC
    / "execution_manifest_metadata_correction.json"
)

RUNNER = (
    REPO
    / "scripts/revision_2026/95_final_dqdv_training_v2_runner.py"
)


EXPECTED_OLD_MANIFEST_SHA = (
    "a9b74e84eb145cb5b2340ab40c036821d15d82d2ba2ce2caf1a93f490fa4f4a9"
)

EXPECTED_PROTOCOL_SHA = (
    "4612a450eeaf909a04c31be16957bdc0567aac67635241e14d7bbceea507fe9d"
)

EXPECTED_IMPORT_SHA = (
    "068223da9e12c370af136343388285b9930ee3edd42b14082309eb40a0c65117"
)

EXPECTED_RUNNER_SHA = (
    "fc4118473310a9d89536a33565dc06d2247965bba5515e171830ecb9b523c350"
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


def atomic_write_bytes(
    path,
    data,
):

    tmp = Path(
        str(path) + ".tmp"
    )

    tmp.write_bytes(
        data
    )

    os.replace(
        tmp,
        path,
    )


def atomic_json(
    path,
    obj,
):

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


# ------------------------------------------------------------
# Immutable upstream provenance.
# ------------------------------------------------------------

assert sha256(
    MANIFEST
) == EXPECTED_OLD_MANIFEST_SHA

assert sha256(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

assert sha256(
    IMPORT_MANIFEST
) == EXPECTED_IMPORT_SHA

assert sha256(
    RUNNER
) == EXPECTED_RUNNER_SHA


protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)

manifest = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)


assert protocol[
    "training_contract"
][
    "max_epochs"
] == 10000

assert protocol[
    "training_contract"
][
    "patience"
] == 30


assert manifest[
    "contract"
] == "dqdv_training_execution_v2"

assert manifest[
    "status"
] == "COMPLETE_PENDING_INDEPENDENT_AUDIT"

assert manifest[
    "completed_pair_count"
] == 60

assert manifest[
    "completed_model_fit_count"
] == 120

assert manifest[
    "validation_safety_ceiling"
] == 10000

assert manifest[
    "validation_patience"
] == 30

assert manifest[
    "epoch_ceiling"
] == 5000

assert manifest[
    "patience"
] == 30

assert manifest[
    "all_fits_patience_valid"
] is True

assert manifest[
    "all_fits_below_epoch_ceiling"
] is True

assert manifest[
    "test_metric_values_parsed_during_closure"
] is False

assert manifest[
    "test_metric_values_surfaced_during_execution"
] is False

assert manifest[
    "test_metric_values_used_to_change_protocol"
] is False


# ------------------------------------------------------------
# Independently establish actual executed ceiling semantics from
# non-metric execution summaries.
# ------------------------------------------------------------

records = manifest[
    "records"
]

assert len(
    records
) == 60


stop_epochs = []

for r in records:

    stop_epochs.extend([
        int(
            r[
                "mlp_stop_epoch"
            ]
        ),
        int(
            r[
                "gcn_stop_epoch"
            ]
        ),
    ])


assert len(
    stop_epochs
) == 120

assert all(
    x < 10000
    for x in stop_epochs
)

# At least one completed fit exceeded the stale 5000 metadata value.
# Therefore 5000 cannot truthfully describe the v2 execution ceiling.
assert any(
    x >= 5000
    for x in stop_epochs
)


# ------------------------------------------------------------
# Independently verify the executed runner has:
# - actual final closure at <10000
# - a stale manifest-only epoch_ceiling=5000 field.
# ------------------------------------------------------------

runner_text = RUNNER.read_text(
    encoding="utf-8"
)

assert runner_text.count(
    "] < 10000"
) == 2

assert runner_text.count(
    "] < 5000"
) == 0

stale_fragment = '''    "epoch_ceiling":
        5000,'''

assert runner_text.count(
    stale_fragment
) == 1


# ------------------------------------------------------------
# Preserve exact original manifest bytes before correction.
# ------------------------------------------------------------

original_bytes = MANIFEST.read_bytes()

assert hashlib.sha256(
    original_bytes
).hexdigest() == EXPECTED_OLD_MANIFEST_SHA


if BACKUP.exists():

    assert sha256(
        BACKUP
    ) == EXPECTED_OLD_MANIFEST_SHA

else:

    tmp_backup = Path(
        str(BACKUP) + ".tmp"
    )

    shutil.copyfile(
        MANIFEST,
        tmp_backup,
    )

    assert sha256(
        tmp_backup
    ) == EXPECTED_OLD_MANIFEST_SHA

    os.replace(
        tmp_backup,
        BACKUP,
    )


# ------------------------------------------------------------
# Byte-preserving one-field metadata correction.
# ------------------------------------------------------------

old_token = b'"epoch_ceiling": 5000'
new_token = b'"epoch_ceiling": 10000'

assert original_bytes.count(
    old_token
) == 1

assert original_bytes.count(
    new_token
) == 0


corrected_bytes = original_bytes.replace(
    old_token,
    new_token,
    1,
)

assert corrected_bytes != original_bytes

atomic_write_bytes(
    MANIFEST,
    corrected_bytes,
)


corrected = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)

assert corrected[
    "epoch_ceiling"
] == 10000

assert corrected[
    "validation_safety_ceiling"
] == 10000


# Exactly one semantic top-level field changed.
a = dict(
    manifest
)

b = dict(
    corrected
)

old_epoch = a.pop(
    "epoch_ceiling"
)

new_epoch = b.pop(
    "epoch_ceiling"
)

assert old_epoch == 5000
assert new_epoch == 10000
assert a == b


new_sha = sha256(
    MANIFEST
)


receipt = {
    "contract":
        "dqdv_training_v2_execution_manifest_metadata_correction_v1",

    "status":
        "METADATA_CORRECTION_COMPLETE_PENDING_INDEPENDENT_AUDIT",

    "old_execution_manifest_sha256":
        EXPECTED_OLD_MANIFEST_SHA,

    "corrected_execution_manifest_sha256":
        new_sha,

    "original_manifest_backup_path":
        str(
            BACKUP
        ),

    "original_manifest_backup_sha256":
        sha256(
            BACKUP
        ),

    "corrected_field":
        "epoch_ceiling",

    "old_value":
        5000,

    "new_value":
        10000,

    "correction_scope":
        "manifest metadata only",

    "scientific_execution_changed":
        False,

    "pair_records_changed":
        False,

    "model_states_changed":
        False,

    "protocol_changed":
        False,

    "runner_changed":
        False,

    "test_metric_values_read_for_correction":
        False,

    "test_metric_values_surfaced":
        False,

    "basis": {
        "protocol_max_epochs":
            10000,

        "manifest_validation_safety_ceiling":
            10000,

        "runner_final_closure_ceiling":
            10000,

        "all_120_stop_epochs_below_10000":
            True,

        "at_least_one_stop_epoch_not_below_5000":
            True,

        "executed_runner_sha256":
            EXPECTED_RUNNER_SHA,
    },
}


atomic_json(
    RECEIPT,
    receipt,
)


print(
    "DQDV V2 MANIFEST METADATA CORRECTION"
)

print(
    "original manifest preserved exactly:",
    True
)

print(
    "corrected field only:",
    "epoch_ceiling"
)

print(
    "old epoch_ceiling:",
    5000
)

print(
    "new epoch_ceiling:",
    10000
)

print(
    "protocol max_epochs:",
    10000
)

print(
    "runner scientific execution changed:",
    False
)

print(
    "pair records changed:",
    False
)

print(
    "model states changed:",
    False
)

print(
    "test metric values read:",
    False
)

print(
    "corrected_manifest_sha256:",
    new_sha
)

print(
    "correction_receipt_sha256:",
    sha256(
        RECEIPT
    )
)

print(
    "DQDV_V2_MANIFEST_METADATA_CORRECTION_PASS=True"
)
