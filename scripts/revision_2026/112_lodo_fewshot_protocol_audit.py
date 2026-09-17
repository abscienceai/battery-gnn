from pathlib import Path
import hashlib
import json


BASE = Path.home() / "alper/battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

ROOT = BASE / "results/revision_2026/lodo_fewshot"

PROTO = ROOT / "lodo_fewshot_protocol_v1.json"

FREEZER = (
    REPO
    / "scripts/revision_2026"
    / "111_freeze_lodo_fewshot_protocol.py"
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
    PROTO.read_text(
        encoding="utf-8"
    )
)


assert p["contract"] == "lodo_fewshot_protocol_v1"

assert p["status"] == (
    "FROZEN_BEFORE_SUPPORT_SELECTION_SCALER_FIT_OR_MODEL_TRAINING"
)


scope = p["scientific_scope"]

assert scope["primary_datasets"] == [
    "HUST",
    "CALCE",
    "MATR",
]

assert scope["folds"] == {
    "HUST": [
        "CALCE",
        "MATR",
    ],

    "CALCE": [
        "HUST",
        "MATR",
    ],

    "MATR": [
        "HUST",
        "CALCE",
    ],
}

assert scope[
    "old_legacy_lodo_results_reused"
] is False

assert scope[
    "old_legacy_fewshot_results_reused"
] is False


inp = p["input_contract"]

assert inp["feature_count"] == 20
assert len(inp["features"]) == 20
assert inp["target"] == "SOH"

for key in [
    "dataset_identity_as_model_feature",
    "cycle_index_as_model_feature",
    "observation_index_as_model_feature",
    "capacity_as_model_feature",
    "RUL_as_model_feature",
    "EOL_as_model_feature",
]:

    assert inp[key] is False


split = p["lodo_split_contract"]

assert split["split_seeds"] == [
    0, 1, 2, 3, 4
]

assert split[
    "held_out_dataset_removed_from_base_training"
] is True

assert split[
    "held_out_dataset_removed_from_base_validation"
] is True

assert split[
    "target_train_cells_used_by_zero_shot_model"
] is False

assert split[
    "target_validation_cells_used_by_zero_shot_model"
] is False

assert split[
    "target_test_cells_used_for_selection"
] is False

assert split["reassignment"] is False


sc = p["lodo_scaler_contract"]

for key in [
    "held_out_target_train_statistics_used",
    "held_out_target_validation_statistics_used",
    "held_out_target_test_statistics_used",
    "source_validation_statistics_used",
    "target_support_statistics_refit_scaler",
    "target_scaler",
]:

    assert sc[key] is False

assert sc[
    "same_source_only_scaler_used_for_zero_shot_and_all_few_shot_conditions"
] is True


model = p["zero_shot_model_contract"]

assert model["learnable_parameters_each"] == 5569
assert model["feature_matched"] is True
assert model["parameter_matched"] is True
assert model[
    "same_initial_learnable_weights_within_pair"
] is True
assert model["future_edges"] is False
assert model["reverse_edges"] is False
assert model["cross_cell_edges"] is False
assert model["learning_rate"] == 0.001
assert model["weight_decay"] == 0.00001
assert model[
    "training_initialization_seeds"
] == [
    0, 1, 2
]
assert model["patience"] == 30
assert model[
    "maximum_epochs_safety_ceiling"
] == 10000
assert model[
    "early_stopping_data"
] == "source validation cells only"
assert model["new_hyperparameter_tuning"] is False


support = p["few_shot_support_contract"]

assert support[
    "support_cell_counts"
] == [
    1, 3, 5
]

assert support[
    "percentage_based_support"
] is False

assert support[
    "nested_support"
] == "K1 subset K3 subset K5"

assert support[
    "selection_happens_before_model_training"
] is True

for key in [
    "selection_uses_target_labels",
    "selection_uses_target_features",
    "selection_uses_target_validation",
    "selection_uses_target_test",
]:

    assert support[key] is False

assert support[
    "same_support_cells_for_MLP_and_GCN"
] is True

assert support[
    "same_support_cells_across_training_initialization_seeds"
] is True

assert support[
    "target_validation_cells_reserved_and_never_used"
] is True

assert support[
    "target_test_cells_reserved_for_final_evaluation_only"
] is True


adapt = p["few_shot_adaptation_contract"]

assert adapt[
    "K3_starts_from_K1_state"
] is False

assert adapt[
    "K5_starts_from_K3_state"
] is False

assert adapt[
    "all_layers_trainable"
] is True

assert adapt["learning_rate"] == 0.0001
assert adapt["epochs"] == 100
assert adapt["early_stopping"] is False
assert adapt[
    "target_validation_used_for_epoch_selection"
] is False
assert adapt[
    "target_test_used_during_adaptation"
] is False
assert adapt[
    "feature_scaler_refit_during_adaptation"
] is False
assert adapt[
    "new_adaptation_hyperparameter_tuning"
] is False


ev = p["evaluation_contract"]

assert ev["conditions"] == [
    "ZERO_SHOT",
    "K1",
    "K3",
    "K5",
]

assert ev["primary_metrics"] == [
    "CELL_BALANCED_RMSE",
    "CELL_BALANCED_MAE",
]

assert ev["test_used_for_model_selection"] is False

assert "secondary only" in ev[
    "R2_interpretation"
]


report = p["reporting_contract"]

assert report[
    "five_split_seeds_are_independent_trials"
] is False

assert report[
    "three_initialization_seeds_are_independent_splits"
] is False

assert report["new_inferential_tests"] is False
assert report["model_ranking"] == "descriptive"
assert report[
    "universal_GCN_superiority_claim"
] is False
assert report[
    "failure_is_reportable_result"
] is True


emb = p["metric_embargo"]

assert emb[
    "numeric_test_values_printed_by_runner"
] is False

assert emb[
    "numeric_test_values_surfaced_before_execution_audit"
] is False


scale = p["expected_execution_scale"]

assert scale["zero_shot_model_fits"] == 90
assert scale["few_shot_adaptation_fits"] == 270
assert scale[
    "total_fits_or_adaptations"
] == 360


inv = p["frozen_observed_inventory"]

assert inv["rows"] == {
    "HUST": 146122,
    "CALCE": 14298,
    "MATR": 140133,
}

assert inv["cells"] == {
    "HUST": 77,
    "CALCE": 13,
    "MATR": 169,
}

assert inv[
    "target_train_cell_pool_size_per_split"
] == {
    "HUST": 54,
    "CALCE": 9,
    "MATR": 119,
}


for item in p["source_artifacts"].values():

    path = Path(item["path"])

    assert path.is_file()

    assert sha256(
        path
    ) == item["sha256"]


assert sha256(
    FREEZER
) == p[
    "protocol_script"
][
    "sha256"
]


state = p["execution_state"]

assert state == {
    "support_cells_selected": False,
    "lodo_scalers_fit": False,
    "zero_shot_training_started": False,
    "few_shot_adaptation_started": False,
    "test_metrics_read": False,
    "test_metrics_surfaced": False,
}


print("LODO / FEW-SHOT PROTOCOL AUDIT")
print("three leave-one-dataset-out folds:", True)
print("held-out target excluded from base train/validation:", True)
print("source-train-only scaler:", True)
print("target statistics excluded from scaler fit:", True)
print("feature-matched 5569-parameter models:", True)
print("source-validation-only zero-shot selection:", True)
print("K1/K3/K5 support is cell-based, not percentage-based:", True)
print("support sets deterministic and nested:", True)
print("support selection label/feature blind:", True)
print("target validation reserved and unused:", True)
print("target test final-evaluation-only:", True)
print("adaptation starts independently from zero-shot state:", True)
print("adaptation fixed at 100 epochs without target validation:", True)
print("source-only scaler retained during adaptation:", True)
print("R2 explicitly secondary:", True)
print("new inferential tests:", False)
print("test metric embargo:", True)
print("training started:", False)
print("test metrics read:", False)
print("LODO_FEWSHOT_PROTOCOL_AUDIT_PASS=True")
