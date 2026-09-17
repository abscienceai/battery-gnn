from pathlib import Path
import hashlib
import json


BASE = Path.home() / "alper" / "battery_revision_2026"

PROTOCOL = (
    BASE
    / "results"
    / "revision_2026"
    / "sequence_controls"
    / "final_tcn_v1"
    / "protocol.json"
)


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


p = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)


assert p[
    "contract"
] == "causal_tcn_soh_benchmark_v1"

assert p[
    "status"
] == "FINAL_FROZEN_BEFORE_ANY_TCN_TRAINING_OR_TEST"

assert p[
    "learnable_parameters"
] == 5489

assert p[
    "reference_mlp_gcn_parameters"
] == 5569

assert p[
    "parameter_difference_vs_reference"
] == -80

assert p[
    "sequence_length"
] == 3

assert p[
    "sequence_receptive_field"
] == [
    "t-2",
    "t-1",
    "t",
]

assert p[
    "future_observations_used"
] is False

assert p[
    "cross_cell_history_used"
] is False

assert p[
    "centered_context_used"
] is False


assert p[
    "split_seeds"
] == [
    0,
    1,
    2,
    3,
    4,
]

assert p[
    "training_initialization_seeds"
] == [
    0,
    1,
    2,
]

assert p[
    "planned_training_fit_count"
] == 15

assert p[
    "higher_level_comparison_units"
] == 5


pre = p[
    "preprocessing"
]

assert pre[
    "secondary_scaler"
] is False

assert pre[
    "validation_statistics_used_in_scaler"
] is False

assert pre[
    "test_statistics_used_in_scaler"
] is False


train = p[
    "training"
]

assert train[
    "execution_device"
] == "CPU"

assert train[
    "full_batch"
] is True

assert train[
    "optimizer"
] == "AdamW"

assert train[
    "learning_rate"
] == 0.001

assert train[
    "weight_decay"
] == 0.00001

assert train[
    "max_epochs"
] == 5000

assert train[
    "early_stopping_patience"
] == 30


barrier = p[
    "training_test_barrier"
]

assert barrier[
    "required"
] is True

assert barrier[
    "test_may_change_training_protocol"
] is False


blind = p[
    "test_blinding_record"
]

assert blind[
    "raw_mlp_test_results_seen_before_tcn_freeze"
] is True

assert blind[
    "causal_gcn_test_results_seen_before_tcn_freeze"
] is True

assert blind[
    "classical_smoothing_test_results_seen_before_tcn_freeze"
] is True

assert blind[
    "reference_results_available_during_tcn_design"
] is True

assert blind[
    "tcn_training_results_seen_before_tcn_freeze"
] is False

assert blind[
    "tcn_validation_results_seen_before_tcn_freeze"
] is False

assert blind[
    "tcn_test_results_seen_before_tcn_freeze"
] is False

assert blind[
    "tcn_test_artifacts_existing_before_freeze"
] == 0


assert p[
    "primary_tcn_comparators"
] == [
    "RAW_MLP",
    "CAUSAL_GCN",
]

assert p[
    "primary_domains"
] == [
    "BALANCED_ALL",
    "HUST",
    "CALCE",
    "MATR",
]

assert p[
    "multiple_testing"
][
    "family_per_error_metric"
] == 8


counts = p[
    "pre_freeze_artifact_counts"
]

assert counts == {
    "fit_json": 0,
    "best_state_pt": 0,
    "test_metrics_csv": 0,
    "evaluation_json": 0,
    "execution_manifest_json": 0,
}


print("=" * 80)
print("CAUSAL TCN BENCHMARK PROTOCOL AUDIT")
print("=" * 80)

print(
    "TCN parameters:",
    p[
        "learnable_parameters"
    ]
)

print(
    "parameter difference vs MLP/GCN:",
    p[
        "parameter_difference_vs_reference"
    ]
)

print(
    "sequence length:",
    p[
        "sequence_length"
    ]
)

print(
    "strictly causal:",
    True
)

print(
    "planned fits:",
    p[
        "planned_training_fit_count"
    ]
)

print(
    "full-batch training:",
    True
)

print(
    "train-only scaler:",
    True
)

print(
    "training/test barrier:",
    True
)

print(
    "reference results already observed:",
    True
)

print(
    "TCN results observed before freeze:",
    False
)

print(
    "TCN pre-freeze fit artifacts:",
    0
)

print(
    "TCN pre-freeze test artifacts:",
    0
)

print(
    "primary comparators:",
    2
)

print(
    "primary domains:",
    4
)

print(
    "Holm family per error metric:",
    8
)

print(
    "seven-method ranking role:",
    "descriptive"
)

print(
    "protocol SHA256:",
    sha256_file(
        PROTOCOL
    )
)

print()
print(
    "CAUSAL_TCN_PROTOCOL_AUDIT_PASS=True"
)
