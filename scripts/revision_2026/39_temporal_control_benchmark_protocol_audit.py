from pathlib import Path
import hashlib
import json
import os


BASE = Path(
    os.environ.get(
        "BATTERY_REVISION_BASE",
        Path.home()
        / "alper"
        / "battery_revision_2026",
    )
)

PROTOCOL = (
    BASE
    / "results"
    / "revision_2026"
    / "temporal_controls"
    / "final_temporal_benchmark_v1"
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
] == "temporal_control_benchmark_v1"

assert p[
    "status"
] == "FINAL_FROZEN_BEFORE_NEW_TEMPORAL_TEST_RUNS"

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
    "new_fit_count"
] == 60

assert p[
    "reference_fit_count_reused"
] == 30

assert p[
    "max_epochs"
] == 5000

assert p[
    "early_stopping_patience"
] == 30

assert p[
    "future_observations_used"
] is False

assert p[
    "centered_smoothing_used"
] is False

assert p[
    "filtfilt_used"
] is False

assert p[
    "cross_cell_filter_state"
] is False

assert p[
    "target_used_by_filter_preprocessing"
] is False

assert p[
    "kalman_selected_using_test"
] is False

assert p[
    "kalman_selected_q"
] == 0.01

assert p[
    "kalman_selected_r"
] == 1.0

assert p[
    "kalman_selection_fit_count"
] == 60


b = p[
    "test_blinding_record"
]


assert b[
    "raw_mlp_and_gcn_reference_test_metrics_previously_observed"
] is True

assert b[
    "new_temporal_control_test_metrics_observed_before_protocol_freeze"
] is False

assert b[
    "reference_test_results_used_to_select_kalman_q"
] is False

assert b[
    "reference_test_results_used_to_fit_secondary_scalers"
] is False

assert b[
    "new_temporal_control_test_results_may_change_training_protocol"
] is False


methods = p[
    "methods"
]


assert set(
    methods
) == {
    "raw_mlp",
    "moving_average3_mlp",
    "lowpass_binomial3_mlp",
    "savgol5_poly2_mlp",
    "kalman_q0p01_mlp",
    "causal_gcn",
}


for name in (
    "moving_average3_mlp",
    "lowpass_binomial3_mlp",
    "savgol5_poly2_mlp",
    "kalman_q0p01_mlp",
):

    assert methods[
        name
    ][
        "new_fit_required"
    ] is True

    assert methods[
        name
    ][
        "learnable_parameters"
    ] == 5569


assert methods[
    "raw_mlp"
][
    "new_fit_required"
] is False

assert methods[
    "causal_gcn"
][
    "new_fit_required"
] is False


assert p[
    "sequence_model_control_in_this_protocol"
] is False


print("=" * 80)
print("TEMPORAL CONTROL BENCHMARK PROTOCOL AUDIT")
print("=" * 80)

print(
    "methods frozen:",
    len(
        methods
    )
)

print(
    "new temporal methods:",
    4
)

print(
    "new fits frozen:",
    60
)

print(
    "reference fits reused:",
    30
)

print(
    "selected Kalman q:",
    p[
        "kalman_selected_q"
    ]
)

print(
    "selected Kalman r:",
    p[
        "kalman_selected_r"
    ]
)

print(
    "new test metrics seen before freeze:",
    False
)

print(
    "test used for Kalman selection:",
    False
)

print(
    "test used for secondary scalers:",
    False
)

print(
    "all new MLP controls parameter matched:",
    True
)

print(
    "causal-only filtering contract:",
    True
)

print(
    "protocol SHA256:",
    sha256_file(
        PROTOCOL
    )
)

print()
print(
    "TEMPORAL_CONTROL_PROTOCOL_AUDIT_PASS=True"
)
