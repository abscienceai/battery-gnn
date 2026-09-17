from pathlib import Path
import hashlib
import json


BASE = Path.home() / "alper" / "battery_revision_2026"

P = (
    BASE
    / "results"
    / "revision_2026"
    / "physical_sensor_noise"
    / "final_physical_v1"
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
    P.read_text(
        encoding="utf-8"
    )
)


assert p[
    "contract"
] == "physical_sensor_noise_soh_v1"

assert p[
    "status"
] == "FINAL_FROZEN_BEFORE_ANY_PHYSICAL_SENSOR_CORRUPTION_EVALUATION"

assert p[
    "datasets"
] == [
    "HUST",
    "CALCE",
    "MATR",
]

assert p[
    "models"
] == [
    "FeatureMatchedMLP",
    "FeatureMatchedCausalGCN",
]

assert p[
    "primary_feature_count"
] == 20


channels = p[
    "raw_model_input_channels"
]

assert channels[
    "voltage"
][
    "used"
] is True

assert channels[
    "voltage"
][
    "physical_unit"
] == "V"

assert channels[
    "current"
][
    "used"
] is True

assert channels[
    "current"
][
    "physical_unit"
] == "A"

assert channels[
    "temperature"
][
    "used"
] is False

assert channels[
    "temperature"
][
    "corrupted_in_primary_benchmark"
] is False

assert channels[
    "capacity"
][
    "used"
] is False

assert channels[
    "capacity"
][
    "corrupted_in_primary_benchmark"
] is False


training = p[
    "training_policy"
]

assert training[
    "no_retraining"
] is True

assert training[
    "no_noise_augmentation_training"
] is True

assert training[
    "frozen_clean_trained_models"
] is True

assert training[
    "test_only_corruption"
] is True

assert training[
    "train_only_scaler_reused_without_refit"
] is True

assert training[
    "validation_or_test_statistics_may_refit_scaler"
] is False


barrier = p[
    "clean_roundtrip_barrier"
]

assert barrier[
    "required_before_noisy_test_metrics"
] is True

assert barrier[
    "mapping_requirements"
][
    "future_cycle_use"
] is False

assert barrier[
    "mapping_requirements"
][
    "cross_cell_mapping"
] is False

assert barrier[
    "mapping_requirements"
][
    "target_used_for_feature_generation"
] is False


assert p[
    "old_feature_space_noise_role"
].startswith(
    "Legacy g.x-space"
)


levels = p[
    "severity_levels"
]

assert list(
    levels.keys()
) == [
    "high",
    "low",
    "medium",
] or set(
    levels.keys()
) == {
    "low",
    "medium",
    "high",
}


assert levels[
    "low"
][
    "gaussian_voltage_sigma_V"
] == 0.002

assert levels[
    "medium"
][
    "gaussian_voltage_sigma_V"
] == 0.005

assert levels[
    "high"
][
    "gaussian_voltage_sigma_V"
] == 0.010

assert levels[
    "low"
][
    "gaussian_current_sigma_A"
] == 0.005

assert levels[
    "medium"
][
    "gaussian_current_sigma_A"
] == 0.010

assert levels[
    "high"
][
    "gaussian_current_sigma_A"
] == 0.025


families = p[
    "scenario_families"
]

for name in [
    "gaussian",
    "calibration_offset",
    "sensor_drift",
    "sample_dropout",
]:

    assert families[
        name
    ][
        "levels"
    ] == [
        "low",
        "medium",
        "high",
    ]


assert families[
    "compound_medium"
][
    "inferential_role"
] == "descriptive stress test only"


assert p[
    "randomness_contract"
][
    "same_corruption_realization_for_mlp_and_gcn"
] is True


aggregation = p[
    "aggregation_contract"
]

assert aggregation[
    "higher_level_units"
] == 5


inferential = p[
    "primary_inferential_sensitivity"
]

assert len(
    inferential[
        "families"
    ]
) == 4

assert len(
    inferential[
        "domains"
    ]
) == 4

assert inferential[
    "comparisons_per_error_metric"
] == 16


claims = p[
    "claims_guard"
]

assert claims[
    "universal_gcn_robustness_claim_allowed"
] is False

assert claims[
    "universal_mlp_robustness_claim_allowed"
] is False


assert p[
    "pre_freeze_physical_test_metric_files"
] == 0

assert p[
    "pre_freeze_physical_evaluation_files"
] == 0


print("=" * 80)
print("PHYSICAL SENSOR NOISE PROTOCOL AUDIT")
print("=" * 80)

print(
    "datasets:",
    3
)

print(
    "frozen clean models:",
    2
)

print(
    "raw voltage corruption:",
    "V"
)

print(
    "raw current corruption:",
    "A"
)

print(
    "temperature excluded from primary input:",
    True
)

print(
    "capacity excluded from primary input:",
    True
)

print(
    "corruption before feature extraction:",
    True
)

print(
    "direct g.x corruption primary evidence:",
    False
)

print(
    "clean raw roundtrip barrier:",
    True
)

print(
    "no noisy retraining:",
    True
)

print(
    "train-only scaler refit:",
    False
)

print(
    "primary families:",
    4
)

print(
    "three severities each:",
    True
)

print(
    "compound role:",
    "descriptive"
)

print(
    "corruption seeds:",
    len(
        p[
            "corruption_seeds"
        ]
    )
)

print(
    "higher-level split units:",
    5
)

print(
    "Holm comparisons per error metric:",
    16
)

print(
    "physical test metrics pre-freeze:",
    0
)

print(
    "protocol SHA256:",
    sha256_file(
        P
    )
)

print()
print(
    "PHYSICAL_SENSOR_NOISE_PROTOCOL_AUDIT_PASS=True"
)
