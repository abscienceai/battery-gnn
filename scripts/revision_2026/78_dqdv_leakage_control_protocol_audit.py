from pathlib import Path
import hashlib
import json
import subprocess


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

OUT = (
    BASE
    / "results/revision_2026/dqdv_leakage_control/final_dqdv_v1"
)

PROTOCOL = OUT / "protocol.json"

EXPECTED_FINAL_V3_PROTOCOL_SHA = (
    "2dec14c210ae3f02e5f65e4acc24f9c4adc747bc071509cb144e69dba707f4dd"
)


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


assert PROTOCOL.is_file()

p = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)


assert p[
    "contract"
] == "dqdv_leakage_control_protocol_v1"

assert p[
    "status"
] == "FROZEN_BEFORE_DQDV_FEATURE_EXTRACTION"


# Source anchors must still be byte-exact.
for name, info in p[
    "source_anchors"
].items():

    path = Path(
        info[
            "path"
        ]
    )

    assert path.is_file(), (
        name,
        path,
    )

    assert sha256(
        path
    ) == info[
        "sha256"
    ], name


assert p[
    "source_anchors"
][
    "final_v3_protocol"
][
    "sha256"
] == EXPECTED_FINAL_V3_PROTOCOL_SHA


task = p[
    "task"
]

assert task[
    "datasets"
] == [
    "HUST",
    "CALCE",
    "MATR",
]

assert task[
    "split_seeds"
] == [
    0,
    1,
    2,
    3,
    4,
]

assert task[
    "train_initialization_seeds"
] == [
    0,
    1,
    2,
]


raw = p[
    "raw_trace_contract"
]

assert raw[
    "only_sensor_inputs"
] == [
    "voltage_V",
    "current_A",
    "time_s",
]

for dataset in [
    "HUST",
    "CALCE",
    "MATR",
]:
    assert raw[
        "dataset_semantics"
    ][
        dataset
    ][
        "discharge_rule"
    ] == "I_A < 0"


ica = p[
    "incremental_capacity_definition"
]

assert ica[
    "voltage_bin_origin_V"
] == 0.0

assert ica[
    "voltage_bin_width_V"
] == 0.01

assert ica[
    "minimum_discharge_intervals"
] == 20

assert ica[
    "minimum_occupied_voltage_bins"
] == 5

assert ica[
    "feature_count"
] == 10

assert len(
    ica[
        "feature_names"
    ]
) == 10

assert len(
    set(
        ica[
            "feature_names"
        ]
    )
) == 10

assert "ica_area_Ah" in ica[
    "feature_names"
]

assert "ica_peak_voltage_V" in ica[
    "feature_names"
]


h = p[
    "information_horizons"
]

assert set(
    h
) == {
    "FULL",
    "PREFIX25",
    "PREFIX50",
}

assert h[
    "FULL"
][
    "deployment_valid"
] is False

for key in [
    "PREFIX25",
    "PREFIX50",
]:

    assert h[
        key
    ][
        "cutoff_fit_scope"
    ] == (
        "training rows only, separately for dataset x split_seed"
    )

    assert h[
        key
    ][
        "validation_or_test_duration_used_to_fit_cutoff"
    ] is False

    assert h[
        key
    ][
        "individual_test_cycle_endpoint_used"
    ] is False

    assert h[
        key
    ][
        "deployment_valid"
    ] is True


cohort = p[
    "cohort_policy"
]

assert cohort[
    "per_split_common_cohort"
] is True

assert cohort[
    "condition_specific_test_cohorts_forbidden"
] is True


conditions = p[
    "feature_sets"
][
    "conditions"
]

assert [
    x[
        "id"
    ]
    for x in conditions
] == [
    "PRIMARY20_COMMON",
    "PRIMARY20_PLUS_FULL_DQDV10",
    "PRIMARY20_PLUS_PREFIX25_DQDV10",
    "PRIMARY20_PLUS_PREFIX50_DQDV10",
]

assert [
    x[
        "input_dimension"
    ]
    for x in conditions
] == [
    20,
    30,
    30,
    30,
]


models = p[
    "models"
]

assert models[
    "parameters_primary20"
] == 5569

assert models[
    "parameters_augmented30"
] == 6209

assert models[
    "mlp_and_gcn_parameter_matched_within_each_condition"
] is True

assert models[
    "future_edges"
] is False

assert models[
    "cross_cell_edges"
] is False

assert models[
    "retrain_required"
] is True


assert p[
    "execution_size"
][
    "expected_model_fits"
] == 120


analysis = p[
    "analysis_contract"
]

assert analysis[
    "new_hypothesis_tests"
] is False

assert analysis[
    "p_values"
] is False

assert analysis[
    "domains"
] == [
    "BALANCED_ALL",
    "HUST",
    "CALCE",
    "MATR",
]


guards = p[
    "claim_guards"
]

assert guards[
    "full_curve_dqdv_is_deployment_valid"
] is False

assert guards[
    "prefix_cutoff_can_use_test_cycle_total_duration"
] is False

assert guards[
    "prefix_cutoff_can_use_test_cycle_endpoint"
] is False

assert guards[
    "target_capacity_can_enter_feature_generator"
] is False

assert guards[
    "full_curve_dqdv_is_positive_control_for_target_leakage"
] is True

assert guards[
    "causal_prefix_dqdv_is_target_independent_at_generation_time"
] is True


barriers = p[
    "freeze_barriers"
]

for key in [
    "raw_numeric_trace_values_read",
    "capacity_values_read",
    "SOH_values_read",
    "model_metrics_read",
    "model_states_loaded",
    "dqdv_features_computed",
    "model_training_run",
    "analysis_performed",
]:
    assert barriers[
        key
    ] is False


freeze_commit = p[
    "freeze_commit"
]

subprocess.check_call(
    [
        "git",
        "cat-file",
        "-e",
        freeze_commit + "^{commit}",
    ],
    cwd=REPO,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)


print(
    "DQDV LEAKAGE-CONTROL PROTOCOL AUDIT"
)

print(
    "protocol_contract_exact:",
    True
)

print(
    "source_anchors_exact:",
    True
)

print(
    "datasets:",
    3
)

print(
    "split_seeds:",
    5
)

print(
    "train_initializations:",
    3
)

print(
    "dqdv_features:",
    10
)

print(
    "information_horizons:",
    3
)

print(
    "fixed_test_endpoint_independent_prefix:",
    True
)

print(
    "common_cohort_required:",
    True
)

print(
    "target_values_forbidden_during_generation:",
    True
)

print(
    "full_curve_positive_control:",
    True
)

print(
    "expected_model_fits:",
    120
)

print(
    "new_hypothesis_tests:",
    False
)

print(
    "raw_numeric_trace_values_read:",
    False
)

print(
    "capacity_values_read:",
    False
)

print(
    "model_metrics_read:",
    False
)

print(
    "protocol_sha256:",
    sha256(
        PROTOCOL
    ),
)

print(
    "DQDV_LEAKAGE_CONTROL_PROTOCOL_AUDIT_PASS=True"
)
