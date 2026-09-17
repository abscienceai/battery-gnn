from pathlib import Path
import hashlib
import json
import subprocess


BASE = Path.home() / "alper/battery_revision_2026"

REPO = (
    BASE
    / "repo/battery-gnn"
)

ROOT = (
    BASE
    / "results/revision_2026/prospective_rul"
)

PROTOCOL = (
    ROOT
    / "prospective_rul_protocol_v1.json"
)

FREEZER = (
    REPO
    / "scripts/revision_2026/101_freeze_prospective_rul_protocol.py"
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


p = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)


assert p[
    "contract"
] == "prospective_rul_protocol_v1"

assert p[
    "status"
] == "FROZEN_BEFORE_TARGET_BUILDING_OR_MODEL_TRAINING"


scope = p[
    "scientific_scope"
]

assert scope[
    "primary_datasets"
] == [
    "CALCE",
    "MATR",
]

assert (
    "no fabricated exact"
    in scope[
        "hust_role"
    ]
)


endpoint = p[
    "endpoint_contract"
]

assert endpoint[
    "threshold"
] == 0.80

assert endpoint[
    "required_consecutive_observations"
] == 2

assert endpoint[
    "consecutive_observation_index_required"
] is True

assert endpoint[
    "rul_at_eol"
] == 0

assert endpoint[
    "post_eol_rows_receive_rul"
] is False

assert endpoint[
    "right_censored_cells_receive_exact_rul"
] is False


lock = p[
    "feasibility_lock"
]

assert lock[
    "HUST"
][
    "primary_k2_event_cells"
] == 0

assert lock[
    "CALCE"
][
    "k2_event_cells"
] == 13

assert lock[
    "CALCE"
][
    "primary_k2_exact_rul_rows"
] == 7355

assert lock[
    "MATR"
][
    "k2_event_cells"
] == 88

assert lock[
    "MATR"
][
    "k3_event_cells"
] == 88

assert lock[
    "MATR"
][
    "k1_only_cells"
] == 41

assert lock[
    "MATR"
][
    "k1_only_crossing_rows_feature_valid"
] == 0

assert lock[
    "MATR"
][
    "primary_k2_exact_rul_rows"
] == 56814


inp = p[
    "input_contract"
]

assert inp[
    "input_eligibility"
] == "feature_valid"

assert inp[
    "existing_rul_task_eligible_used"
] is False

assert inp[
    "feature_count"
] == 20

for key in [
    "cycle_index_as_model_input",
    "observation_index_as_model_input",
    "SOH_as_model_input",
    "capacity_as_model_input",
    "RUL_as_model_input",
    "EOL_as_model_input",
    "lifetime_as_model_input",
    "future_observation_as_model_input",
]:

    assert inp[
        key
    ] is False


pros = p[
    "prospective_context_contract"
]

assert pros[
    "future_nodes_visible"
] is False

assert pros[
    "reverse_future_edges"
] is False

assert pros[
    "cross_cell_edges"
] is False

assert pros[
    "retrospective_global_pooling"
] is False


early = p[
    "early_prediction_contract"
]

assert early[
    "uses_fraction_of_future_lifetime"
] is False

assert early[
    "primary_checkpoints_observed_cycles"
] == [
    25,
    50,
    100,
]

assert early[
    "future_lifetime_needed_to_define_checkpoint"
] is False


split = p[
    "split_contract"
]

assert split[
    "unit"
] == "complete canonical cell"

assert split[
    "split_seeds"
] == [
    0, 1, 2, 3, 4
]

assert split[
    "row_level_random_split"
] is False

assert split[
    "cell_overlap_across_splits"
] is False

assert split[
    "test_data_used_for_model_selection"
] is False


scaling = p[
    "scaling_contract"
]

assert (
    "TRAIN rows only"
    in scaling[
        "feature_scalers"
    ]
)

assert scaling[
    "validation_or_test_rows_used_to_fit_scalers"
] is False


models = p[
    "model_contract"
]

assert models[
    "same_input_features"
] is True

assert models[
    "same_target_definition"
] is True

assert models[
    "same_initial_learnable_weights_within_pair"
] is True

assert models[
    "graph_level_lifetime_regression"
] is False

assert models[
    "node_or_cycle_level_prediction"
] is True


training = p[
    "training_contract"
]

assert training[
    "training_initialization_seeds"
] == [
    0, 1, 2
]

assert training[
    "test_metrics_available_during_training"
] is False

assert training[
    "new_hyperparameter_tuning"
] is False


barrier = p[
    "interpretation_barriers"
]

for key in [
    "claim_universal_gcn_superiority",
    "claim_HUST_80pct_RUL_performance",
    "claim_censored_exact_RUL",
    "claim_future_lifetime_fraction_is_prospective",
    "old_end_of_record_RUL_results_reused",
    "old_early_prediction_results_reused",
]:

    assert barrier[
        key
    ] is False


for item in p[
    "source_artifacts"
].values():

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


assert sha256(
    FREEZER
) == p[
    "protocol_script"
][
    "sha256"
]


state = p[
    "execution_state"
]

assert state == {
    "target_table_built":
        False,

    "scalers_fit":
        False,

    "model_training_performed":
        False,

    "test_metrics_read":
        False,

    "test_metrics_surfaced":
        False,
}


print("PROSPECTIVE RUL PROTOCOL AUDIT")
print(
    "harmonized endpoint locked:",
    True
)
print(
    "CALCE/MATR primary exact-RUL only:",
    True
)
print(
    "HUST fabricated exact-RUL prevented:",
    True
)
print(
    "k2 endpoint primary:",
    True
)
print(
    "k1/k3 sensitivity only:",
    True
)
print(
    "20 feature causal input contract:",
    True
)
print(
    "future nodes prohibited:",
    True
)
print(
    "fixed K25/K50/K100 prospective checkpoints:",
    True
)
print(
    "future lifetime fraction prohibited:",
    True
)
print(
    "old end-of-record RUL prohibited:",
    True
)
print(
    "old early-prediction results prohibited:",
    True
)
print(
    "model training performed:",
    False
)
print(
    "test metrics read:",
    False
)
print(
    "PROSPECTIVE_RUL_PROTOCOL_AUDIT_PASS=True"
)
