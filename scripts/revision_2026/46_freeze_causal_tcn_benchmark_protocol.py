from pathlib import Path
import hashlib
import json
import os


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo" / "battery-gnn"

ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "sequence_controls"
    / "final_tcn_v1"
)

ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

PROTOCOL_OUT = ROOT / "protocol.json"


FILES = {
    "benchmark_script":
        REPO
        / "scripts"
        / "revision_2026"
        / "24_feature_matched_soh_benchmark.py",

    "tcn_model":
        REPO
        / "scripts"
        / "revision_2026"
        / "44_causal_tcn_sequence_model.py",

    "tcn_model_audit":
        REPO
        / "scripts"
        / "revision_2026"
        / "45_causal_tcn_sequence_model_audit.py",

    "canonical":
        BASE
        / "data"
        / "processed"
        / "revision_2026"
        / "canonical_tasks"
        / "canonical_cycle_supervised_table.csv",

    "assignments":
        BASE
        / "data"
        / "processed"
        / "revision_2026"
        / "splits_and_scalers"
        / "cell_split_assignments.csv",

    "base_scalers":
        BASE
        / "data"
        / "processed"
        / "revision_2026"
        / "splits_and_scalers"
        / "train_only_scaler_parameters.csv",

    "clean_protocol":
        BASE
        / "results"
        / "revision_2026"
        / "feature_matched_soh"
        / "full_clean_benchmark_protocol_v3.json",

    "clean_execution_manifest":
        BASE
        / "results"
        / "revision_2026"
        / "feature_matched_soh"
        / "final_v3"
        / "final_v3_execution_manifest.json",

    "temporal_protocol":
        BASE
        / "results"
        / "revision_2026"
        / "temporal_controls"
        / "final_temporal_benchmark_v1"
        / "protocol.json",

    "temporal_analysis_manifest":
        BASE
        / "results"
        / "revision_2026"
        / "temporal_controls"
        / "final_temporal_benchmark_v1"
        / "analysis"
        / "analysis_manifest.json",
}


EXPECTED_HASHES = {
    "benchmark_script":
        "5d621f8d5764e417e41bc518ec72b9608566a087c552b8365d438c2344c06c5e",

    "tcn_model":
        "2fcab55c9e96d4a39e0ba34ec3daad065a75e3d32c98143a5c6fbf6830d6c487",

    "tcn_model_audit":
        "bdcbf69bfa40b74caca7d89dcfa8c8bb062333508ca3b346e8180d9efd698da6",

    "canonical":
        "49a7b7629cf9ee3f8ce0a7e3c52f2807878b752b4c9a30a876285fb891b00c14",

    "assignments":
        "914c8b90131f496ca396f63e0502147ca99848fdfd4b2c6b2a5887225b47d055",

    "base_scalers":
        "b9708b9321b521b87bdd05f969558134d7e0148285ef7c50f648c2028cd86de1",

    "clean_protocol":
        "2dec14c210ae3f02e5f65e4acc24f9c4adc747bc071509cb144e69dba707f4dd",

    "clean_execution_manifest":
        "5a8bfbe0f11a6158f2749a9da4f18f1e62adf8c839976c1bf0af9c9f6c07440d",

    "temporal_protocol":
        "b6116666e861c79b7332dbd599594846b86511e420d8305eadf82d30b87f0106",

    "temporal_analysis_manifest":
        "c67f1cae7264a6495f9454707fca4b0797ce1ca12234a83dc3d6de40481d1f20",
}


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def atomic_json(obj, path):
    tmp = Path(
        str(path)
        + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


# =============================================================================
# Immutable source guards
# =============================================================================

for name, expected in EXPECTED_HASHES.items():

    observed = sha256_file(
        FILES[name]
    )

    assert observed == expected, (
        name,
        observed,
        expected,
    )


# =============================================================================
# Ensure NO TCN fit/test artifact exists before protocol freeze
# =============================================================================

fit_before = list(
    ROOT.rglob(
        "fit.json"
    )
)

state_before = list(
    ROOT.rglob(
        "best_state.pt"
    )
)

test_before = list(
    ROOT.rglob(
        "test_metrics.csv"
    )
)

evaluation_before = list(
    ROOT.rglob(
        "evaluation.json"
    )
)

execution_before = list(
    ROOT.rglob(
        "execution_manifest.json"
    )
)


assert len(
    fit_before
) == 0

assert len(
    state_before
) == 0

assert len(
    test_before
) == 0

assert len(
    evaluation_before
) == 0

assert len(
    execution_before
) == 0


# =============================================================================
# Frozen protocol
# =============================================================================

protocol = {
    "contract":
        "causal_tcn_soh_benchmark_v1",

    "status":
        "FINAL_FROZEN_BEFORE_ANY_TCN_TRAINING_OR_TEST",

    "purpose":
        (
            "Provide an explicit causal sequence-model baseline "
            "to test whether temporal-sequence learning can explain "
            "performance otherwise attributed to causal graph "
            "message passing."
        ),

    "task":
        "cycle-level SOH regression",

    "datasets":
        [
            "HUST",
            "CALCE",
            "MATR",
        ],

    "feature_count":
        20,

    "features":
        (
            "Same 20 voltage/current within-cycle statistical "
            "features as the feature-matched clean benchmark."
        ),

    "cycle_index_as_feature":
        False,

    "target_as_feature":
        False,

    "model":
        "CausalThreeCycleTCN",

    "model_contract":
        "causal_tcn_three_cycle_v1",

    "architecture":
        (
            "Conv1d(20->49,kernel=3) -> ReLU -> "
            "Conv1d(49->49,kernel=1) -> ReLU -> "
            "Linear(49->1)"
        ),

    "learnable_parameters":
        5489,

    "reference_mlp_gcn_parameters":
        5569,

    "parameter_difference_vs_reference":
        -80,

    "parameter_ratio_vs_reference":
        5489 / 5569,

    "dropout":
        False,

    "batch_normalization":
        False,

    "sequence_length":
        3,

    "sequence_receptive_field":
        [
            "t-2",
            "t-1",
            "t",
        ],

    "sequence_boundary_rule":
        (
            "Unavailable historical positions at cell start are "
            "filled by replicating the first valid observation "
            "of that same cell."
        ),

    "future_observations_used":
        False,

    "cross_cell_history_used":
        False,

    "centered_context_used":
        False,

    "split_seeds":
        [
            0,
            1,
            2,
            3,
            4,
        ],

    "training_initialization_seeds":
        [
            0,
            1,
            2,
        ],

    "planned_training_fit_count":
        15,

    "higher_level_comparison_units":
        5,

    "initialization_policy":
        (
            "Three initialization seeds are averaged within each "
            "cell-level split; the five split seeds remain the "
            "higher-level repeated-holdout units."
        ),

    "preprocessing":
        {
            "base_standardization":
                (
                    "split-specific pooled supervised scaler "
                    "fitted using training cells only"
                ),

            "secondary_scaler":
                False,

            "sequence_construction_order":
                (
                    "Apply frozen train-only feature standardization "
                    "to cycle observations, then construct the "
                    "three-cycle causal window independently within "
                    "each cell."
                ),

            "validation_statistics_used_in_scaler":
                False,

            "test_statistics_used_in_scaler":
                False,
        },

    "training":
        {
            "execution_device":
                "CPU",

            "full_batch":
                True,

            "objective":
                "dataset-balanced + cell-balanced MSE",

            "optimizer":
                "AdamW",

            "learning_rate":
                0.001,

            "weight_decay":
                0.00001,

            "max_epochs":
                5000,

            "max_epochs_role":
                "nonbinding safety ceiling",

            "early_stopping_patience":
                30,

            "early_stopping_metric":
                (
                    "validation dataset-balanced "
                    "cell-balanced RMSE"
                ),

            "epoch_ceiling_rule":
                (
                    "If any TCN fit reaches epoch 5000 without "
                    "patience stopping, do not inspect TCN test "
                    "metrics; revisit optimization budget using "
                    "train/validation information only."
                ),
        },

    "training_test_barrier":
        {
            "required":
                True,

            "rule":
                (
                    "All 15 TCN training/validation fits and an "
                    "immutable training-closure artifact must be "
                    "complete before the first TCN test tensor or "
                    "test metric is evaluated."
                ),

            "test_may_change_training_protocol":
                False,
        },

    "test_blinding_record":
        {
            "raw_mlp_test_results_seen_before_tcn_freeze":
                True,

            "causal_gcn_test_results_seen_before_tcn_freeze":
                True,

            "classical_smoothing_test_results_seen_before_tcn_freeze":
                True,

            "reference_results_available_during_tcn_design":
                True,

            "tcn_training_results_seen_before_tcn_freeze":
                False,

            "tcn_validation_results_seen_before_tcn_freeze":
                False,

            "tcn_test_results_seen_before_tcn_freeze":
                False,

            "tcn_test_artifacts_existing_before_freeze":
                0,

            "interpretation":
                (
                    "The TCN experiment is a pre-test-frozen "
                    "sensitivity extension, not a prospectively "
                    "blinded comparison against previously unseen "
                    "reference results."
                ),
        },

    "primary_metrics":
        [
            "RMSE",
            "MAE",
        ],

    "secondary_metric":
        (
            "R2 reported descriptively by physical dataset"
        ),

    "primary_tcn_comparators":
        [
            "RAW_MLP",
            "CAUSAL_GCN",
        ],

    "primary_domains":
        [
            "BALANCED_ALL",
            "HUST",
            "CALCE",
            "MATR",
        ],

    "difference_definition":
        (
            "comparator minus TCN for RMSE/MAE; "
            "positive values favor TCN"
        ),

    "paired_analysis":
        (
            "Average the three TCN initialization seeds within "
            "each split_seed and compare against the corresponding "
            "three-seed split-level means of each frozen comparator."
        ),

    "signflip_test":
        (
            "Exact two-sided paired sign-flip sensitivity test "
            "over the five split-level differences."
        ),

    "signflip_interpretation":
        (
            "Repeated-split sensitivity analysis only; the five "
            "splits are not treated as five independent datasets."
        ),

    "multiple_testing":
        {
            "family_per_error_metric":
                8,

            "definition":
                (
                    "Two primary comparators x four domains. "
                    "Apply Holm adjustment separately to the "
                    "eight RMSE p-values and eight MAE p-values."
                ),
        },

    "all_seven_method_ranking":
        {
            "included":
                True,

            "role":
                "descriptive only",

            "methods":
                [
                    "RAW_MLP",
                    "MOVING_AVERAGE3_MLP",
                    "LOWPASS_BINOMIAL3_MLP",
                    "SAVGOL5_POLY2_MLP",
                    "KALMAN_Q0P01_MLP",
                    "CAUSAL_GCN",
                    "CAUSAL_TCN",
                ],
        },

    "source_commits":
        {
            "tcn_model_and_audit":
                "e64ecc3866a9e0a9dde7ec36ea2fe78e1003dfd8",

            "clean_feature_matched_analysis":
                "a52d37d6760bd40382fd56f8ec57f38989bf1043",

            "temporal_control_analysis":
                "07041dbda011c951c8d089579345f18921ac5cad",
        },

    "source_sha256": {
        name:
            sha256_file(path)
        for name, path in FILES.items()
    },

    "pre_freeze_artifact_counts":
        {
            "fit_json":
                0,

            "best_state_pt":
                0,

            "test_metrics_csv":
                0,

            "evaluation_json":
                0,

            "execution_manifest_json":
                0,
        },
}


if PROTOCOL_OUT.exists():

    existing = json.loads(
        PROTOCOL_OUT.read_text(
            encoding="utf-8"
        )
    )

    assert existing == protocol

else:

    atomic_json(
        protocol,
        PROTOCOL_OUT,
    )


print("=" * 80)
print("CAUSAL TCN BENCHMARK PROTOCOL FREEZE")
print("=" * 80)

print(
    "TCN parameters:",
    protocol[
        "learnable_parameters"
    ]
)

print(
    "reference parameters:",
    protocol[
        "reference_mlp_gcn_parameters"
    ]
)

print(
    "sequence length:",
    protocol[
        "sequence_length"
    ]
)

print(
    "receptive field:",
    "{t,t-1,t-2}"
)

print(
    "planned TCN fits:",
    protocol[
        "planned_training_fit_count"
    ]
)

print(
    "max_epochs:",
    protocol[
        "training"
    ][
        "max_epochs"
    ]
)

print(
    "patience:",
    protocol[
        "training"
    ][
        "early_stopping_patience"
    ]
)

print(
    "reference test results previously observed:",
    True
)

print(
    "TCN training results seen before freeze:",
    False
)

print(
    "TCN validation results seen before freeze:",
    False
)

print(
    "TCN test results seen before freeze:",
    False
)

print(
    "TCN training artifacts before freeze:",
    0
)

print(
    "TCN test artifacts before freeze:",
    0
)

print(
    "primary Holm family per metric:",
    8
)

print(
    "protocol SHA256:",
    sha256_file(
        PROTOCOL_OUT
    )
)

print()
print(
    "CAUSAL_TCN_PROTOCOL_FREEZE_PASS=True"
)
