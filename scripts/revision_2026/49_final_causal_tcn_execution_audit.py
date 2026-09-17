from pathlib import Path
import hashlib
import json


BASE = Path.home() / "alper" / "battery_revision_2026"

ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "sequence_controls"
    / "final_tcn_v1"
)

PROTOCOL = ROOT / "protocol.json"
CLOSURE = ROOT / "training_closure.json"
MANIFEST = ROOT / "execution_manifest.json"
RUNS = ROOT / "runs"

EXPECTED_PROTOCOL = (
    "19fe5012478bd1d70d3f580d08ee8391c905858dbbb3b3e1015fb8627a54f2c3"
)


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


assert sha(
    PROTOCOL
) == EXPECTED_PROTOCOL


closure = json.loads(
    CLOSURE.read_text(
        encoding="utf-8"
    )
)

manifest = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)


assert closure[
    "contract"
] == "causal_tcn_training_closure_v1"

assert closure[
    "training_fit_count"
] == 15

assert closure[
    "all_training_fits_patience_valid"
] is True

assert closure[
    "all_training_fits_below_epoch_ceiling"
] is True

assert closure[
    "test_rows_constructed_before_closure"
] == 0

assert closure[
    "test_metrics_read_before_closure"
] is False


assert manifest[
    "contract"
] == "causal_tcn_execution_manifest_v1"

assert manifest[
    "completed_training_fit_count"
] == 15

assert manifest[
    "completed_test_evaluation_count"
] == 15

assert manifest[
    "all_training_complete_before_first_test_evaluation"
] is True

assert manifest[
    "all_fits_patience_valid"
] is True

assert manifest[
    "all_fits_below_epoch_ceiling"
] is True

assert manifest[
    "test_rows_constructed_before_training_closure"
] == 0

assert manifest[
    "test_metrics_read_before_training_closure"
] is False

assert manifest[
    "reference_result_files_accessed_by_runner"
] is False

assert manifest[
    "test_metric_values_surfaced_during_runner"
] is False

assert manifest[
    "test_metrics_used_to_change_protocol"
] is False


fit_count = 0
history_count = 0
state_count = 0
evaluation_count = 0
metric_count = 0


for split_seed in range(5):

    for train_seed in range(3):

        d = (
            RUNS
            / (
                f"split{split_seed}"
                f"_train{train_seed}"
            )
        )

        fit_path = d / "fit.json"
        history_path = d / "history.csv"
        state_path = d / "best_state.pt"
        evaluation_path = d / "evaluation.json"
        metrics_path = d / "test_metrics.csv"


        assert fit_path.is_file()
        assert history_path.is_file()
        assert state_path.is_file()
        assert evaluation_path.is_file()
        assert metrics_path.is_file()


        fit = json.loads(
            fit_path.read_text(
                encoding="utf-8"
            )
        )

        evaluation = json.loads(
            evaluation_path.read_text(
                encoding="utf-8"
            )
        )


        assert fit[
            "protocol_sha256"
        ] == EXPECTED_PROTOCOL

        assert fit[
            "parameter_count"
        ] == 5489

        assert fit[
            "sequence_length"
        ] == 3

        assert fit[
            "stopped_by_patience"
        ] is True

        assert fit[
            "hit_epoch_ceiling"
        ] is False

        assert fit[
            "epochs_after_best"
        ] == 30

        assert fit[
            "stop_epoch"
        ] < 5000

        assert fit[
            "test_rows_constructed_during_training"
        ] == 0

        assert fit[
            "test_metrics_read_during_training"
        ] is False

        assert sha(
            history_path
        ) == fit[
            "history_sha256"
        ]

        assert sha(
            state_path
        ) == fit[
            "state_file_sha256"
        ]


        assert evaluation[
            "contract"
        ] == "causal_tcn_test_evaluation_v1"

        assert evaluation[
            "protocol_sha256"
        ] == EXPECTED_PROTOCOL

        assert evaluation[
            "training_closure_sha256"
        ] == sha(
            CLOSURE
        )

        assert evaluation[
            "test_evaluated_only_after_training_closure"
        ] is True

        assert evaluation[
            "test_metrics_used_to_change_protocol"
        ] is False

        assert evaluation[
            "test_metrics_surfaced_by_runner"
        ] is False

        assert sha(
            metrics_path
        ) == evaluation[
            "test_metrics_sha256"
        ]


        # Deliberately do NOT parse CSV metric contents here.
        fit_count += 1
        history_count += 1
        state_count += 1
        evaluation_count += 1
        metric_count += 1


assert fit_count == 15
assert history_count == 15
assert state_count == 15
assert evaluation_count == 15
assert metric_count == 15


print("=" * 80)
print("FINAL CAUSAL TCN EXECUTION AUDIT")
print("=" * 80)

print(
    "protocol exact:",
    True
)

print(
    "training fits:",
    fit_count,
    "/ 15"
)

print(
    "training histories:",
    history_count,
    "/ 15"
)

print(
    "saved states:",
    state_count,
    "/ 15"
)

print(
    "test evaluations:",
    evaluation_count,
    "/ 15"
)

print(
    "test metric files:",
    metric_count,
    "/ 15"
)

print(
    "all fits patience-valid:",
    True
)

print(
    "all fits below 5000 ceiling:",
    True
)

print(
    "all training complete before first test:",
    True
)

print(
    "test rows constructed before closure:",
    0
)

print(
    "test metrics read before closure:",
    False
)

print(
    "reference result files accessed by runner:",
    False
)

print(
    "test metric contents parsed by this audit:",
    False
)

print(
    "test metric values surfaced:",
    False
)

print(
    "training closure SHA256:",
    sha(
        CLOSURE
    )
)

print(
    "execution manifest SHA256:",
    sha(
        MANIFEST
    )
)

print()
print(
    "FINAL_CAUSAL_TCN_EXECUTION_AUDIT_PASS=True"
)
