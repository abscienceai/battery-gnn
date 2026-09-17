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

REPO = (
    BASE
    / "repo"
    / "battery-gnn"
)

ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "temporal_controls"
)

SELECTION_ROOT = (
    ROOT
    / "kalman_validation_selection_v1"
)

FINAL_ROOT = (
    ROOT
    / "final_temporal_benchmark_v1"
)

FINAL_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

PROTOCOL_OUT = (
    FINAL_ROOT
    / "protocol.json"
)


FILES = {
    "benchmark_script":
        REPO
        / "scripts"
        / "revision_2026"
        / "24_feature_matched_soh_benchmark.py",

    "filter_kernel":
        REPO
        / "scripts"
        / "revision_2026"
        / "32_causal_temporal_filter_kernel.py",

    "preprocessing_builder":
        REPO
        / "scripts"
        / "revision_2026"
        / "34_temporal_control_preprocessing_builder.py",

    "preprocessing_audit":
        REPO
        / "scripts"
        / "revision_2026"
        / "35_temporal_control_preprocessing_audit.py",

    "kalman_selector":
        REPO
        / "scripts"
        / "revision_2026"
        / "36_kalman_validation_only_selector.py",

    "kalman_selection_audit":
        REPO
        / "scripts"
        / "revision_2026"
        / "37_kalman_validation_selection_audit.py",

    "secondary_scalers":
        BASE
        / "data"
        / "processed"
        / "revision_2026"
        / "temporal_controls"
        / "temporal_filter_secondary_scalers.csv",

    "preprocessing_manifest":
        BASE
        / "data"
        / "processed"
        / "revision_2026"
        / "temporal_controls"
        / "temporal_filter_preprocessing_manifest.json",

    "kalman_selection_protocol":
        SELECTION_ROOT
        / "selection_protocol.json",

    "kalman_selected":
        SELECTION_ROOT
        / "selected_kalman.json",

    "kalman_selection_manifest":
        SELECTION_ROOT
        / "selection_manifest.json",

    "kalman_q_summary":
        SELECTION_ROOT
        / "q_validation_summary.csv",

    "clean_v3_protocol":
        BASE
        / "results"
        / "revision_2026"
        / "feature_matched_soh"
        / "full_clean_benchmark_protocol_v3.json",

    "clean_v3_execution_manifest":
        BASE
        / "results"
        / "revision_2026"
        / "feature_matched_soh"
        / "final_v3"
        / "final_v3_execution_manifest.json",
}


EXPECTED_FIXED_HASHES = {
    "benchmark_script":
        "5d621f8d5764e417e41bc518ec72b9608566a087c552b8365d438c2344c06c5e",

    "filter_kernel":
        "23f733540a23954c278722c43a36e994c5431b603bfeb9e7ebd4474451bd3ff7",

    "preprocessing_builder":
        "6edd261e0997c39bad2ba76b0536575fefa38e355d5b887e47948f246801ee4a",

    "preprocessing_audit":
        "600ec65db53329166ded63825bf09afbe8387e1cd494a64bd6351135eb6ac6b3",

    "kalman_selector":
        "66444f397f24bf5b26e9caf5009802785c7148e093ee2e6eb096599ac7d01e0c",

    "kalman_selection_audit":
        "0cf82af176f0d9bb25c12ed7c10b76eeb7565015414abe998d0b93d6da7cdffc",

    "secondary_scalers":
        "98607908e0cb8ee2bb9c102c061990d7282bef20f1535c16e9825a2eb3d1a82c",

    "preprocessing_manifest":
        "1397ac3cf505e309fa9d4adf37f95bf58436e4354903a8b7f38345143086e975",

    "clean_v3_protocol":
        "2dec14c210ae3f02e5f65e4acc24f9c4adc747bc071509cb144e69dba707f4dd",

    "clean_v3_execution_manifest":
        "5a8bfbe0f11a6158f2749a9da4f18f1e62adf8c839976c1bf0af9c9f6c07440d",
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


for name, expected in EXPECTED_FIXED_HASHES.items():

    observed = sha256_file(
        FILES[name]
    )

    assert observed == expected, (
        name,
        observed,
        expected,
    )


# =============================================================================
# Kalman selection contract
# =============================================================================

selected = json.loads(
    FILES[
        "kalman_selected"
    ].read_text(
        encoding="utf-8"
    )
)

selection_manifest = json.loads(
    FILES[
        "kalman_selection_manifest"
    ].read_text(
        encoding="utf-8"
    )
)


assert selected[
    "selected_q"
] == 0.01

assert selected[
    "selected_r"
] == 1.0

assert selected[
    "selected_initial_variance"
] == 1.0

assert selected[
    "fit_count"
] == 60

assert selected[
    "test_rows_used"
] == 0

assert selected[
    "test_metrics_read"
] is False

assert selected[
    "final_test_result_root_accessed"
] is False


assert selection_manifest[
    "completed_fit_count"
] == 60

assert selection_manifest[
    "all_fits_patience_valid"
] is True

assert selection_manifest[
    "all_fits_below_epoch_ceiling"
] is True

assert selection_manifest[
    "test_rows_used"
] == 0

assert selection_manifest[
    "test_metrics_read"
] is False


# =============================================================================
# Frozen temporal benchmark protocol
# =============================================================================

protocol = {
    "contract":
        "temporal_control_benchmark_v1",

    "status":
        "FINAL_FROZEN_BEFORE_NEW_TEMPORAL_TEST_RUNS",

    "research_question":
        (
            "Determine whether the causal graph model's behavior "
            "can be explained by generic causal temporal smoothing "
            "rather than graph-specific message passing."
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

    "feature_policy":
        (
            "Same 20 voltage/current within-cycle statistical "
            "features used in the clean feature-matched benchmark."
        ),

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

    "primary_replication_policy":
        (
            "Three initialization seeds are averaged within each "
            "split_seed. The five split_seed units remain the "
            "higher-level repeated-split comparison units."
        ),

    "methods":
        {
            "raw_mlp": {
                "type":
                    "reference_anchor",

                "source":
                    "clean final_v3 MLP",

                "new_fit_required":
                    False,

                "temporal_operation":
                    "none",

                "learnable_parameters":
                    5569,
            },

            "moving_average3_mlp": {
                "type":
                    "new_temporal_control",

                "new_fit_required":
                    True,

                "temporal_operation":
                    (
                        "strictly causal trailing moving average, "
                        "window=3"
                    ),

                "learnable_parameters":
                    5569,
            },

            "lowpass_binomial3_mlp": {
                "type":
                    "new_temporal_control",

                "new_fit_required":
                    True,

                "temporal_operation":
                    (
                        "strictly causal normalized FIR taps "
                        "[1,2,1] over {t-2,t-1,t}"
                    ),

                "learnable_parameters":
                    5569,
            },

            "savgol5_poly2_mlp": {
                "type":
                    "new_temporal_control",

                "new_fit_required":
                    True,

                "temporal_operation":
                    (
                        "one-sided causal endpoint polynomial "
                        "smoother, window=5, polyorder=2"
                    ),

                "learnable_parameters":
                    5569,
            },

            "kalman_q0p01_mlp": {
                "type":
                    "new_temporal_control",

                "new_fit_required":
                    True,

                "temporal_operation":
                    (
                        "independent causal local-level Kalman "
                        "filter per feature"
                    ),

                "kalman_q":
                    0.01,

                "kalman_r":
                    1.0,

                "kalman_initial_variance":
                    1.0,

                "kalman_selection":
                    "validation-only frozen selection",

                "learnable_parameters":
                    5569,
            },

            "causal_gcn": {
                "type":
                    "reference_anchor",

                "source":
                    "clean final_v3 CAUSAL_GCN",

                "new_fit_required":
                    False,

                "temporal_operation":
                    (
                        "two causal parameter-free mean aggregation "
                        "steps with maximum receptive field "
                        "{t,t-1,t-2}"
                    ),

                "learnable_parameters":
                    5569,
            },
        },

    "new_temporal_control_methods":
        [
            "moving_average3_mlp",
            "lowpass_binomial3_mlp",
            "savgol5_poly2_mlp",
            "kalman_q0p01_mlp",
        ],

    "new_fit_count":
        60,

    "new_fit_count_definition":
        "4 new controls x 5 split seeds x 3 initialization seeds",

    "reference_fit_count_reused":
        30,

    "reference_fit_count_definition":
        (
            "15 frozen raw MLP fits + 15 frozen causal GCN fits "
            "from clean final_v3"
        ),

    "model_architecture_for_new_controls":
        "20 -> 64 -> 64 -> 1 MLP with ReLU",

    "dropout":
        False,

    "batch_normalization":
        False,

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
        "validation dataset-balanced cell-balanced RMSE",

    "training_objective":
        "dataset-balanced + cell-balanced MSE",

    "ceiling_rule":
        (
            "Any new temporal-control fit reaching epoch 5000 "
            "invalidates final aggregation until the optimization "
            "budget is revisited using train/validation only."
        ),

    "preprocessing_pipeline_new_controls":
        [
            "split-specific pooled train-only base standardization",
            "strictly causal filtering within each cell",
            (
                "filter-specific secondary standardization fitted "
                "using training cells only"
            ),
        ],

    "raw_mlp_identity_policy":
        (
            "The frozen clean final_v3 raw MLP is reused rather "
            "than retrained. The preprocessing audit demonstrated "
            "that the identity secondary transform has mean 0 and "
            "variance 1 to numerical precision, so a second identity "
            "standardization is redundant."
        ),

    "future_observations_used":
        False,

    "centered_smoothing_used":
        False,

    "filtfilt_used":
        False,

    "cross_cell_filter_state":
        False,

    "target_used_by_filter_preprocessing":
        False,

    "kalman_selected_using_test":
        False,

    "kalman_selected_q":
        0.01,

    "kalman_selected_r":
        1.0,

    "kalman_selection_fit_count":
        60,

    "test_blinding_record": {
        "raw_mlp_and_gcn_reference_test_metrics_previously_observed":
            True,

        "new_temporal_control_test_metrics_observed_before_protocol_freeze":
            False,

        "reference_test_results_used_to_select_kalman_q":
            False,

        "reference_test_results_used_to_fit_secondary_scalers":
            False,

        "new_temporal_control_test_results_may_change_training_protocol":
            False,
    },

    "test_policy":
        (
            "New temporal-control fits may evaluate the held-out "
            "test cells only after model training and validation "
            "early stopping are complete. Test results cannot "
            "change filtering parameters, scalers, architecture, "
            "optimization settings, patience, or epoch ceiling."
        ),

    "primary_metrics":
        [
            "RMSE",
            "MAE",
        ],

    "secondary_metric":
        "R2 descriptive by physical dataset",

    "difference_definition":
        (
            "For RMSE/MAE, comparator minus causal GCN; "
            "positive values favor causal GCN."
        ),

    "planned_analysis":
        (
            "Average three initialization seeds within each "
            "split_seed, then compare methods at the five "
            "split-level repeated-holdout units. Report paired "
            "differences and win/loss counts."
        ),

    "inferential_sensitivity":
        (
            "Exact two-sided paired sign-flip tests over the five "
            "split-level differences are sensitivity analyses, not "
            "claims of five fully independent datasets."
        ),

    "multiple_testing_policy":
        (
            "For each error metric, Holm-adjust the GCN-vs-control "
            "sign-flip p-values across all planned control-by-domain "
            "comparisons."
        ),

    "sequence_model_control_in_this_protocol":
        False,

    "sequence_model_note":
        (
            "A TCN/sequence baseline is reserved for a separate "
            "pre-specified protocol after the classical causal "
            "smoothing benchmark."
        ),

    "source_sha256":
        {
            name:
                sha256_file(path)
            for name, path in FILES.items()
        },

    "source_commits":
        {
            "clean_feature_matched_analysis":
                "a52d37d6760bd40382fd56f8ec57f38989bf1043",

            "causal_filter_kernel":
                "31dd57b7bfb809b40dd45d62fa0dd32e4d6ab349",

            "temporal_preprocessing":
                "f0a4e3a33ddd3cf7290d250942b4fee3c3f17922",

            "kalman_validation_selection":
                "ed61327887609362056c53efc8842dfecefff6d4",
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

    PROTOCOL_OUT.write_text(
        json.dumps(
            protocol,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


print("=" * 80)
print("TEMPORAL CONTROL BENCHMARK PROTOCOL FREEZE")
print("=" * 80)

print(
    "selected Kalman q/r:",
    "0.01 / 1.0"
)

print(
    "new temporal controls:",
    len(
        protocol[
            "new_temporal_control_methods"
        ]
    )
)

print(
    "planned new fits:",
    protocol[
        "new_fit_count"
    ]
)

print(
    "reused reference fits:",
    protocol[
        "reference_fit_count_reused"
    ]
)

print(
    "max_epochs:",
    protocol[
        "max_epochs"
    ]
)

print(
    "patience:",
    protocol[
        "early_stopping_patience"
    ]
)

print(
    "raw/GCN reference test metrics previously observed:",
    True
)

print(
    "new temporal-control test metrics observed before freeze:",
    False
)

print(
    "Kalman selected using test:",
    False
)

print(
    "secondary scalers fitted using test:",
    False
)

print(
    "sequence control included here:",
    False
)

print(
    "protocol SHA256:",
    sha256_file(
        PROTOCOL_OUT
    )
)

print()
print(
    "TEMPORAL_CONTROL_PROTOCOL_FREEZE_PASS=True"
)
