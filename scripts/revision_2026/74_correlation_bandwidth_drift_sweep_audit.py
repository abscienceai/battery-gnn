from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

ROOT = (
    BASE
    / "results/revision_2026/spectral_frequency/final_spectral_v1"
)

PROTOCOL = ROOT / "protocol.json"

CLEAN = (
    ROOT
    / "clean_graph_fourier_v1"
    / "clean_graph_fourier_manifest.json"
)

EVAL = ROOT / "correlation_bandwidth_drift_v1"
RUNS = EVAL / "runs"

EXEC = (
    EVAL
    / "correlation_bandwidth_drift_execution_manifest.json"
)

SCRIPT = (
    REPO
    / "scripts/revision_2026/73_correlation_bandwidth_drift_sweep.py"
)


EXPECTED_PROTOCOL_SHA = (
    "a8d3b65b87980470f326717f0cb50e3d4fb4d2e0429c93171eb0d21d6c713e45"
)

EXPECTED_CLEAN_SHA = (
    "da9653f823f764ab85d69c2726f2de752a0cb7091b7c13d79a7db0e915751d12"
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


def row_count(path):
    with Path(path).open(
        "rb"
    ) as f:

        return sum(
            1
            for _ in f
        ) - 1


assert sha(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

assert sha(
    CLEAN
) == EXPECTED_CLEAN_SHA


m = json.loads(
    EXEC.read_text(
        encoding="utf-8"
    )
)


assert m[
    "contract"
] == "final_correlation_bandwidth_drift_execution_v1"

assert m[
    "status"
] == "COMPLETE_BEFORE_VALUE_LEVEL_ANALYSIS"

assert m[
    "spectral_protocol_sha256"
] == EXPECTED_PROTOCOL_SHA

assert m[
    "clean_graph_fourier_manifest_sha256"
] == EXPECTED_CLEAN_SHA

assert m[
    "runner_script_sha256"
] == sha(
    SCRIPT
)

assert m[
    "completed_runs"
] == 30

assert m[
    "split_count"
] == 5

assert m[
    "train_initializations_per_split"
] == 3

assert m[
    "conditions_per_run"
] == 66

assert m[
    "condition_contract"
] == {
    "clean": 1,
    "ar1": 25,
    "bandwidth": 25,
    "controlled_drift": 15,
    "total": 66,
}

assert m[
    "ar1_rho"
] == [
    -0.9,
    -0.5,
    0.0,
    0.5,
    0.9,
]

assert m[
    "bandwidth_filters"
] == [
    "white",
    "lowpass_ma3",
    "lowpass_ma9",
    "bandpass_ma3_minus_ma9",
    "highpass_first_difference",
]

assert m[
    "drift_terminal_shift_standardized_units"
] == [
    0.10,
    0.25,
    0.50,
]

assert m[
    "domains_per_condition"
] == 4

assert m[
    "metric_rows_per_run"
] == 264

assert m[
    "expected_total_metric_rows"
] == 7920

assert m[
    "paired_realization_across_models"
] is True

assert m[
    "paired_realization_across_train_initializations"
] is True

assert m[
    "training_reexecuted"
] is False

assert m[
    "scaler_refit"
] is False

assert m[
    "feature_generator_reexecuted"
] is False

assert m[
    "GPU_used"
] is False

assert m[
    "clean_final_v3_metric_regression_barrier"
] == "PASS"

assert float(
    m[
        "clean_max_abs_rmse_difference"
    ]
) <= 1e-7

assert float(
    m[
        "clean_max_abs_mae_difference"
    ]
) <= 1e-7

assert float(
    m[
        "bandwidth_max_abs_cell_feature_rms_error"
    ]
) <= 2e-6

assert float(
    m[
        "drift_max_abs_terminal_shift_error"
    ]
) <= 2e-6

assert float(
    m[
        "perturbation_parseval_max_relative_error"
    ]
) <= 1e-10

assert m[
    "metric_values_surfaced_during_execution"
] is False

assert m[
    "perturbation_diagnostic_values_surfaced_during_execution"
] is False

assert m[
    "analysis_performed"
] is False


metric_files = sorted(
    RUNS.glob(
        "*_cbd_metrics.csv"
    )
)

meta_files = sorted(
    RUNS.glob(
        "*_evaluation.json"
    )
)

assert len(
    metric_files
) == 30

assert len(
    meta_files
) == 30


metas = []


for p in meta_files:

    x = json.loads(
        p.read_text(
            encoding="utf-8"
        )
    )

    assert x[
        "contract"
    ] == "correlation_bandwidth_drift_evaluation_run_v1"

    assert x[
        "status"
    ] == "COMPLETE"

    assert x[
        "parameter_count"
    ] == 5569

    assert x[
        "condition_count"
    ] == 66

    assert x[
        "ar1_condition_count"
    ] == 25

    assert x[
        "bandwidth_condition_count"
    ] == 25

    assert x[
        "drift_condition_count"
    ] == 15

    assert x[
        "metric_rows"
    ] == 264

    assert x[
        "training_reexecuted"
    ] is False

    assert x[
        "scaler_refit"
    ] is False

    assert x[
        "metric_values_surfaced"
    ] is False

    assert x[
        "analysis_performed"
    ] is False

    assert x[
        "clean_regression_pass"
    ] is True

    metric_path = Path(
        x[
            "metric_file"
        ]
    )

    assert metric_path.is_file()

    assert sha(
        metric_path
    ) == x[
        "metric_file_sha256"
    ]

    assert row_count(
        metric_path
    ) == 264

    metas.append(
        x
    )


# Pairing proof without reading RMSE/MAE.
for split_seed in range(5):

    group = [
        x
        for x in metas
        if int(
            x[
                "split_seed"
            ]
        ) == split_seed
    ]

    assert len(
        group
    ) == 6

    ref = group[
        0
    ][
        "input_tensor_sha256_by_condition"
    ]

    assert len(
        ref
    ) == 66

    for x in group[
        1:
    ]:

        assert (
            x[
                "input_tensor_sha256_by_condition"
            ]
            == ref
        )


# Identifier-only metric CSV read.
frames = []

for p in metric_files:

    df = pd.read_csv(
        p,
        usecols=[
            "split_seed",
            "train_seed",
            "model",
            "condition_index",
            "condition_id",
            "scenario",
            "rho",
            "filter_name",
            "terminal_shift_std_units",
            "perturbation_seed",
            "domain",
            "n_rows",
        ],
        low_memory=False,
    )

    assert len(
        df
    ) == 264

    frames.append(
        df
    )


ids = pd.concat(
    frames,
    ignore_index=True,
)

assert len(
    ids
) == 7920

assert set(
    ids[
        "scenario"
    ]
) == {
    "clean",
    "ar1_correlation",
    "bandwidth",
    "controlled_drift",
}

assert len(
    ids[
        ids[
            "scenario"
        ] == "clean"
    ]
) == 120

assert len(
    ids[
        ids[
            "scenario"
        ] == "ar1_correlation"
    ]
) == 3000

assert len(
    ids[
        ids[
            "scenario"
        ] == "bandwidth"
    ]
) == 3000

assert len(
    ids[
        ids[
            "scenario"
        ] == "controlled_drift"
    ]
) == 1800


diag_info = m[
    "perturbation_spectral_diagnostics"
]

diag_path = Path(
    diag_info[
        "path"
    ]
)

assert sha(
    diag_path
) == diag_info[
    "sha256"
]

diag = pd.read_csv(
    diag_path
)

assert len(
    diag
) == 325

assert set(
    diag[
        "scenario"
    ]
) == {
    "ar1_correlation",
    "bandwidth",
    "controlled_drift",
}

assert np.isfinite(
    diag[
        "spectral_centroid_omega_over_pi"
    ]
).all()

fractions = (
    diag[
        "low_energy_fraction"
    ]
    + diag[
        "mid_low_energy_fraction"
    ]
    + diag[
        "mid_high_energy_fraction"
    ]
    + diag[
        "high_energy_fraction"
    ]
)

assert np.allclose(
    fractions,
    1.0,
    rtol=1e-10,
    atol=1e-12,
)


print("=" * 90)
print("CORRELATION / BANDWIDTH / DRIFT EXECUTION AUDIT")
print("=" * 90)

print(
    "spectral protocol exact:",
    True
)

print(
    "clean Fourier anchor exact:",
    True
)

print(
    "execution runs:",
    30
)

print(
    "metric files:",
    30
)

print(
    "evaluation metadata files:",
    30
)

print(
    "conditions per run:",
    66
)

print(
    "AR1 conditions:",
    25
)

print(
    "bandwidth conditions:",
    25
)

print(
    "drift conditions:",
    15
)

print(
    "metric rows total:",
    7920
)

print(
    "perturbation diagnostic rows:",
    325
)

print(
    "state parameter count:",
    5569
)

print(
    "clean final-v3 regression barrier:",
    True
)

print(
    "bandwidth RMS barrier:",
    True
)

print(
    "drift terminal-shift barrier:",
    True
)

print(
    "perturbation Parseval barrier:",
    True
)

print(
    "paired realization across models:",
    True
)

print(
    "paired realization across initializations:",
    True
)

print(
    "training reexecuted:",
    False
)

print(
    "scaler refit:",
    False
)

print(
    "metric values parsed by audit:",
    False
)

print(
    "metric values surfaced:",
    False
)

print(
    "analysis already performed:",
    False
)

print(
    "execution manifest SHA256:",
    sha(
        EXEC
    )
)

print()
print(
    "CORRELATION_BANDWIDTH_DRIFT_EXECUTION_AUDIT_PASS=True"
)
