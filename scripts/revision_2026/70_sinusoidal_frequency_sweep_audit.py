from pathlib import Path
import csv
import hashlib
import json

import pandas as pd


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

ROOT = (
    BASE
    / "results/revision_2026/spectral_frequency/final_spectral_v1"
)

PROTOCOL = ROOT / "protocol.json"

CLEAN_MANIFEST = (
    ROOT
    / "clean_graph_fourier_v1"
    / "clean_graph_fourier_manifest.json"
)

EVAL = ROOT / "sinusoidal_frequency_v1"
RUNS = EVAL / "runs"

EXEC_MANIFEST = (
    EVAL
    / "sinusoidal_frequency_execution_manifest.json"
)

SCRIPT = (
    REPO
    / "scripts/revision_2026/69_sinusoidal_frequency_sweep.py"
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

        lines = sum(
            1
            for _ in f
        )

    return lines - 1


assert sha(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

assert sha(
    CLEAN_MANIFEST
) == EXPECTED_CLEAN_SHA


m = json.loads(
    EXEC_MANIFEST.read_text(
        encoding="utf-8"
    )
)


assert m[
    "contract"
] == "final_sinusoidal_frequency_execution_v1"

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
    "frequency_grid_omega_over_pi"
] == [
    0.0625,
    0.125,
    0.25,
    0.5,
    0.75,
    1.0,
]

assert m[
    "perturbation_seeds"
] == [
    0,
    1,
    2,
    3,
    4,
]

assert m[
    "conditions_per_run"
] == 31

assert m[
    "primary_frequency_conditions_per_run"
] == 30

assert m[
    "domains_per_condition"
] == 4

assert m[
    "metric_rows_per_run"
] == 124

assert m[
    "expected_total_metric_rows"
] == 3720

assert float(
    m[
        "target_rms_standardized_units"
    ]
) == 0.10

assert float(
    m[
        "max_abs_realized_rms_error"
    ]
) <= 2e-6

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

assert m[
    "metric_values_surfaced_during_execution"
] is False

assert m[
    "analysis_performed"
] is False

assert m[
    "inferential_unit"
] == "split_seed"

assert m[
    "inferential_n"
] == 5


metric_files = sorted(
    RUNS.glob(
        "*_frequency_metrics.csv"
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
    ] == "sinusoidal_frequency_evaluation_run_v1"

    assert x[
        "status"
    ] == "COMPLETE"

    assert x[
        "parameter_count"
    ] == 5569

    assert x[
        "condition_count"
    ] == 31

    assert x[
        "primary_frequency_condition_count"
    ] == 30

    assert x[
        "frequency_count"
    ] == 6

    assert x[
        "perturbation_seed_count"
    ] == 5

    assert x[
        "domains_per_condition"
    ] == 4

    assert x[
        "metric_rows"
    ] == 124

    assert x[
        "training_reexecuted"
    ] is False

    assert x[
        "scaler_refit"
    ] is False

    assert x[
        "feature_generator_reexecuted"
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

    assert float(
        x[
            "clean_max_abs_rmse_difference"
        ]
    ) <= 1e-7

    assert float(
        x[
            "clean_max_abs_mae_difference"
        ]
    ) <= 1e-7

    assert float(
        x[
            "max_abs_rms_error"
        ]
    ) <= 2e-6

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
    ) == 124

    metas.append(
        x
    )


# ---------------------------------------------------------------------------
# Prove paired perturbations without parsing RMSE/MAE values.
# ---------------------------------------------------------------------------

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
    ) == 31

    for x in group[
        1:
    ]:
        assert (
            x[
                "input_tensor_sha256_by_condition"
            ]
            == ref
        )


# ---------------------------------------------------------------------------
# Read ONLY non-metric identifier columns.
# rmse/mae are deliberately excluded.
# ---------------------------------------------------------------------------

identifier_frames = []

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
            "omega_over_pi",
            "perturbation_seed",
            "target_rms_std_units",
            "domain",
            "n_rows",
        ],
        low_memory=False,
    )

    assert len(
        df
    ) == 124

    identifier_frames.append(
        df
    )


ids = pd.concat(
    identifier_frames,
    ignore_index=True,
)

assert len(
    ids
) == 3720

assert set(
    ids[
        "split_seed"
    ]
) == {
    0,
    1,
    2,
    3,
    4,
}

assert set(
    ids[
        "train_seed"
    ]
) == {
    0,
    1,
    2,
}

assert set(
    ids[
        "model"
    ]
) == {
    "MLP",
    "GCN",
}

assert set(
    ids[
        "domain"
    ]
) == {
    "BALANCED_ALL",
    "HUST",
    "CALCE",
    "MATR",
}


clean = ids[
    ids[
        "scenario"
    ] == "clean"
]

frequency = ids[
    ids[
        "scenario"
    ] == "sinusoidal_frequency"
]


assert len(
    clean
) == 120

assert len(
    frequency
) == 3600


assert sorted(
    frequency[
        "omega_over_pi"
    ].drop_duplicates()
    .tolist()
) == [
    0.0625,
    0.125,
    0.25,
    0.5,
    0.75,
    1.0,
]

assert sorted(
    frequency[
        "perturbation_seed"
    ].drop_duplicates()
    .astype(int)
    .tolist()
) == [
    0,
    1,
    2,
    3,
    4,
]


print("=" * 86)
print("SINUSOIDAL FREQUENCY EXECUTION AUDIT")
print("=" * 86)

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
    31
)

print(
    "primary frequency conditions:",
    30
)

print(
    "frequency points:",
    6
)

print(
    "perturbation seeds:",
    5
)

print(
    "metric rows total:",
    3720
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
    "target RMS barrier:",
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
        EXEC_MANIFEST
    )
)

print()
print(
    "SINUSOIDAL_FREQUENCY_EXECUTION_AUDIT_PASS=True"
)
