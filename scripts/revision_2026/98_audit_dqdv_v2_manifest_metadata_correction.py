from pathlib import Path
import hashlib
import json


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

IMPORT_MANIFEST = (
    EXEC
    / "v1_compatible_import_manifest.json"
)


EXPECTED_OLD_MANIFEST_SHA = (
    "a9b74e84eb145cb5b2340ab40c036821d15d82d2ba2ce2caf1a93f490fa4f4a9"
)

EXPECTED_PROTOCOL_SHA = (
    "4612a450eeaf909a04c31be16957bdc0567aac67635241e14d7bbceea507fe9d"
)

EXPECTED_RUNNER_SHA = (
    "fc4118473310a9d89536a33565dc06d2247965bba5515e171830ecb9b523c350"
)

EXPECTED_IMPORT_SHA = (
    "068223da9e12c370af136343388285b9930ee3edd42b14082309eb40a0c65117"
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
    BACKUP
) == EXPECTED_OLD_MANIFEST_SHA

assert sha256(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

assert sha256(
    RUNNER
) == EXPECTED_RUNNER_SHA

assert sha256(
    IMPORT_MANIFEST
) == EXPECTED_IMPORT_SHA


old_bytes = BACKUP.read_bytes()
new_bytes = MANIFEST.read_bytes()


assert old_bytes.count(
    b'"epoch_ceiling": 5000'
) == 1

expected_new_bytes = old_bytes.replace(
    b'"epoch_ceiling": 5000',
    b'"epoch_ceiling": 10000',
    1,
)

assert new_bytes == expected_new_bytes


old = json.loads(
    BACKUP.read_text(
        encoding="utf-8"
    )
)

new = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)

receipt = json.loads(
    RECEIPT.read_text(
        encoding="utf-8"
    )
)

protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)


assert old[
    "epoch_ceiling"
] == 5000

assert new[
    "epoch_ceiling"
] == 10000

assert new[
    "validation_safety_ceiling"
] == 10000

assert protocol[
    "training_contract"
][
    "max_epochs"
] == 10000


a = dict(
    old
)

b = dict(
    new
)

a.pop(
    "epoch_ceiling"
)

b.pop(
    "epoch_ceiling"
)

assert a == b


assert receipt[
    "old_execution_manifest_sha256"
] == EXPECTED_OLD_MANIFEST_SHA

assert receipt[
    "corrected_execution_manifest_sha256"
] == sha256(
    MANIFEST
)

assert receipt[
    "original_manifest_backup_sha256"
] == EXPECTED_OLD_MANIFEST_SHA

assert receipt[
    "corrected_field"
] == "epoch_ceiling"

assert receipt[
    "old_value"
] == 5000

assert receipt[
    "new_value"
] == 10000

assert receipt[
    "scientific_execution_changed"
] is False

assert receipt[
    "pair_records_changed"
] is False

assert receipt[
    "model_states_changed"
] is False

assert receipt[
    "protocol_changed"
] is False

assert receipt[
    "runner_changed"
] is False

assert receipt[
    "test_metric_values_read_for_correction"
] is False

assert receipt[
    "test_metric_values_surfaced"
] is False


assert len(
    new[
        "records"
    ]
) == 60

assert new[
    "completed_pair_count"
] == 60

assert new[
    "completed_model_fit_count"
] == 120

assert new[
    "all_fits_patience_valid"
] is True

assert new[
    "all_fits_below_epoch_ceiling"
] is True


stop_epochs = []

for r in new[
    "records"
]:

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

assert any(
    x >= 5000
    for x in stop_epochs
)


record_files = list(
    (
        EXEC
        / "run_records"
    ).glob(
        "*.json"
    )
)

state_files = list(
    (
        EXEC
        / "model_states"
    ).glob(
        "*.pt"
    )
)

assert len(
    record_files
) == 60

assert len(
    state_files
) == 112


print(
    "DQDV V2 MANIFEST METADATA CORRECTION AUDIT"
)

print(
    "original manifest hash exact:",
    True
)

print(
    "byte-level change restricted to epoch_ceiling:",
    True
)

print(
    "semantic change restricted to epoch_ceiling:",
    True
)

print(
    "corrected epoch_ceiling:",
    10000
)

print(
    "protocol max_epochs agrees:",
    True
)

print(
    "120 stop epochs consistent with corrected ceiling:",
    True
)

print(
    "pair record count unchanged:",
    60
)

print(
    "local v2 state count unchanged:",
    112
)

print(
    "scientific execution changed:",
    False
)

print(
    "test metric values read:",
    False
)

print(
    "corrected_manifest_sha256:",
    sha256(
        MANIFEST
    )
)

print(
    "correction_receipt_sha256:",
    sha256(
        RECEIPT
    )
)

print(
    "DQDV_V2_MANIFEST_METADATA_CORRECTION_AUDIT_PASS=True"
)
