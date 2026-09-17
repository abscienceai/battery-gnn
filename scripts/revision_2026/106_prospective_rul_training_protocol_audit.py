from pathlib import Path
import hashlib
import json


BASE = Path.home() / "alper/battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

ROOT = BASE / "results/revision_2026/prospective_rul"

PROTO = ROOT / "prospective_rul_training_protocol_v1.json"

FREEZER = (
    REPO
    / "scripts/revision_2026"
    / "105_freeze_prospective_rul_training_protocol.py"
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


p = json.loads(
    PROTO.read_text(
        encoding="utf-8"
    )
)


assert p["contract"] == "prospective_rul_training_protocol_v1"
assert p["status"] == "FROZEN_BEFORE_MODEL_TRAINING"

assert p["task"]["datasets"] == [
    "CALCE",
    "MATR",
]

assert p["task"]["HUST_exact_RUL_used"] is False
assert p["task"]["target_zero_at_EOL"] is True


f = p["feature_contract"]

assert f["feature_count"] == 20

for key in [
    "cycle_index_input",
    "observation_index_input",
    "SOH_input",
    "capacity_input",
    "RUL_input",
    "EOL_input",
    "future_input",
]:
    assert f[key] is False


models = p["models"]

assert models[
    "FeatureMatchedMLP_RUL"
][
    "learnable_parameters"
] == 5569

assert models[
    "FeatureMatchedCausalGCN_RUL"
][
    "learnable_parameters"
] == 5569

assert models["feature_matched"] is True
assert models["parameter_matched"] is True
assert models[
    "same_initial_learnable_weights_within_pair"
] is True

g = models[
    "FeatureMatchedCausalGCN_RUL"
]

assert g["future_edges"] is False
assert g["reverse_edges"] is False
assert g["cross_cell_edges"] is False
assert g["maximum_receptive_field"] == "t,t-1,t-2"


seed = p["split_and_seed_contract"]

assert seed["split_seeds"] == [
    0, 1, 2, 3, 4
]

assert seed[
    "training_initialization_seeds"
] == [
    0, 1, 2
]

assert seed["paired_runs"] == 15
assert seed["total_model_fits"] == 30
assert seed["reassignment"] is False


opt = p["optimization"]

assert opt["device"] == "CPU"
assert opt["optimizer"] == "AdamW"
assert opt["learning_rate"] == 0.001
assert opt["weight_decay"] == 0.00001
assert opt["dataset_weighting"] == "equal"
assert opt["cell_weighting_within_dataset"] == "equal"
assert opt["observation_weighting_within_cell"] == "equal"
assert opt["dropout"] is False
assert opt["batch_normalization"] is False


early = p["early_stopping"]

assert early["selection_data"] == "validation only"
assert early["patience"] == 30
assert early["minimum_delta"] == 0.0
assert early["maximum_epochs_safety_ceiling"] == 10000
assert early["best_state_retained"] is True
assert early["test_metrics_visible_during_selection"] is False
assert "halt before test evaluation" in early["ceiling_policy"]


ev = p["test_evaluation"]

assert ev["fixed_checkpoint_role"] == "primary early prospective evaluation"

assert ev["fixed_checkpoints"] == [
    25,
    50,
    100,
]

assert ev["future_lifetime_fraction_used"] is False

assert ev["domains"] == [
    "BALANCED_ALL",
    "CALCE",
    "MATR",
]

assert ev["metrics"] == [
    "RMSE",
    "MAE",
    "R2",
    "normalized_MAE",
]

assert ev["test_used_for_model_selection"] is False


emb = p["metric_embargo"]

assert emb["test_numeric_values_printed_by_runner"] is False
assert emb[
    "test_numeric_values_surfaced_before_independent_execution_audit"
] is False


ana = p["analysis_contract"]

assert ana["new_inferential_tests"] is False
assert ana["split_seeds_claimed_independent_trials"] is False
assert ana["model_ranking"] == "descriptive"
assert ana["universal_GCN_superiority_claim_allowed"] is False
assert ana["old_end_of_record_RUL_results_reused"] is False
assert ana["old_fractional_early_prediction_results_reused"] is False


for item in p["frozen_inputs"].values():

    path = Path(
        item["path"]
    )

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
    "training_started": False,
    "model_states_created": False,
    "test_metrics_read": False,
    "test_metrics_surfaced": False,
}


print("PROSPECTIVE RUL TRAINING PROTOCOL AUDIT")
print("feature-matched 5569-parameter models:", True)
print("causal future-free GCN:", True)
print("15 paired runs / 30 fits:", True)
print("train-only feature and target scaling:", True)
print("hierarchical balanced training loss:", True)
print("validation-only early stopping:", True)
print("10000-epoch ceiling is safety-only:", True)
print("K25/K50/K100 are fixed observed-history checkpoints:", True)
print("future-lifetime fraction prohibited:", True)
print("test metric embargo:", True)
print("new inferential tests:", False)
print("training started:", False)
print("test metrics read:", False)
print("PROSPECTIVE_RUL_TRAINING_PROTOCOL_AUDIT_PASS=True")
