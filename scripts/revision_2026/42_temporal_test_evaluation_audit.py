from pathlib import Path
import hashlib
import json

BASE = (
    Path.home()
    / "alper"
    / "battery_revision_2026"
)

ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "temporal_controls"
    / "final_temporal_benchmark_v1"
)

RUNS = ROOT / "runs"

CLOSURE = ROOT / "training_closure.json"

MANIFEST = (
    ROOT
    / "test_evaluation_continuation_manifest.json"
)

EXPECTED_CLOSURE = (
    "07a75bfc5ab81d467afda76a8b4cd1097ca0609356bf4bdbd6eb2254a2cc1b55"
)

METHODS = [
    "moving_average3_mlp",
    "lowpass_binomial3_mlp",
    "savgol5_poly2_mlp",
    "kalman_q0p01_mlp",
]


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
    CLOSURE
) == EXPECTED_CLOSURE


closure = json.loads(
    CLOSURE.read_text(
        encoding="utf-8"
    )
)

assert closure[
    "training_fit_count"
] == 60

assert closure[
    "test_metrics_read_before_closure"
] is False


manifest = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)

assert manifest[
    "completed_test_evaluations"
] == 60

assert manifest[
    "training_fit_count_reused"
] == 60

assert manifest[
    "training_reexecuted"
] is False

assert manifest[
    "first_original_test_failure_wrote_test_artifact"
] is False

assert manifest[
    "r2_bug_corrected_without_training_changes"
] is True

assert manifest[
    "test_metrics_surfaced"
] is False

assert manifest[
    "test_metrics_used_to_change_protocol"
] is False


fit_count = 0
state_count = 0
eval_count = 0
metric_count = 0


for method in METHODS:
    for split_seed in range(5):
        for train_seed in range(3):

            d = (
                RUNS
                / method
                / (
                    f"split{split_seed}"
                    f"_train{train_seed}"
                )
            )

            fit = d / "fit.json"
            state = d / "best_state.pt"
            evaluation = d / "evaluation.json"
            metrics = d / "test_metrics.csv"

            assert fit.is_file()
            assert state.is_file()
            assert evaluation.is_file()
            assert metrics.is_file()

            f = json.loads(
                fit.read_text(
                    encoding="utf-8"
                )
            )

            e = json.loads(
                evaluation.read_text(
                    encoding="utf-8"
                )
            )

            assert f[
                "test_rows_constructed_during_training"
            ] == 0

            assert f[
                "test_metrics_read_during_training"
            ] is False

            assert e[
                "contract"
            ] == "temporal_control_test_evaluation_v2"

            assert e[
                "training_closure_sha256"
            ] == EXPECTED_CLOSURE

            assert e[
                "training_reexecuted"
            ] is False

            assert e[
                "test_evaluated_only_after_training_closure"
            ] is True

            assert e[
                "test_metrics_used_to_change_protocol"
            ] is False

            assert e[
                "test_metrics_surfaced_by_continuation"
            ] is False

            assert sha(
                metrics
            ) == e[
                "test_metrics_sha256"
            ]

            fit_count += 1
            state_count += 1
            eval_count += 1
            metric_count += 1


assert fit_count == 60
assert state_count == 60
assert eval_count == 60
assert metric_count == 60


print("=" * 80)
print("TEMPORAL TEST-EVALUATION ARTIFACT AUDIT")
print("=" * 80)

print(
    "training closure exact: True"
)

print(
    "training fit artifacts:",
    fit_count,
    "/ 60"
)

print(
    "saved model states:",
    state_count,
    "/ 60"
)

print(
    "test evaluation artifacts:",
    eval_count,
    "/ 60"
)

print(
    "test metric files:",
    metric_count,
    "/ 60"
)

print(
    "training reexecuted:",
    False
)

print(
    "all tests after training closure:",
    True
)

print(
    "original failed test wrote artifact:",
    False
)

print(
    "R2 bug corrected without training changes:",
    True
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
    "continuation manifest SHA256:",
    sha(
        MANIFEST
    )
)

print()
print(
    "TEMPORAL_TEST_EVALUATION_AUDIT_PASS=True"
)
