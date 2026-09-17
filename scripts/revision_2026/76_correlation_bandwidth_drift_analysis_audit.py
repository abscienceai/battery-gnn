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

EVAL = ROOT / "correlation_bandwidth_drift_v1"
RUNS = EVAL / "runs"
OUT = EVAL / "analysis_v1"

PROTOCOL = ROOT / "protocol.json"

EXEC = (
    EVAL
    / "correlation_bandwidth_drift_execution_manifest.json"
)

SCRIPT = (
    REPO
    / "scripts/revision_2026/75_analyze_correlation_bandwidth_drift.py"
)

MANIFEST = (
    OUT
    / "correlation_bandwidth_drift_analysis_manifest.json"
)

DIAG = (
    EVAL
    / "perturbation_spectral_diagnostics.csv"
)


EXPECTED_PROTOCOL_SHA = (
    "a8d3b65b87980470f326717f0cb50e3d4fb4d2e0429c93171eb0d21d6c713e45"
)

EXPECTED_EXEC_SHA = (
    "f43e69afae5e6d6a9a59edab0ab530990b8dab1dc790c3de3d775ba522835317"
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


assert sha(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

assert sha(
    EXEC
) == EXPECTED_EXEC_SHA


m = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)


assert m[
    "contract"
] == "correlation_bandwidth_drift_analysis_v1"

assert m[
    "status"
] == "COMPLETE"

assert m[
    "spectral_protocol_sha256"
] == EXPECTED_PROTOCOL_SHA

assert m[
    "execution_manifest_sha256"
] == EXPECTED_EXEC_SHA

assert m[
    "analysis_script_sha256"
] == sha(
    SCRIPT
)

assert m[
    "value_level_read_started_only_after_analysis_script_commit"
] is True

assert m[
    "raw_metric_files_read"
] == 30

assert m[
    "raw_metric_rows_read"
] == 7920

assert m[
    "perturbation_spectral_rows_read"
] == 325

assert m[
    "clean_final_v3_reference_barrier"
] == "PASS"

assert m[
    "clean_reference_rows"
] == 120

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
    "nested_count_per_split_setting"
] == 15

assert m[
    "split_units"
] == 5

assert m[
    "analysis_role"
] == "descriptive mechanism analysis"

assert m[
    "new_hypothesis_tests_performed"
] is False

assert m[
    "p_values_computed"
] is False

assert m[
    "models_retrained"
] is False

assert m[
    "scaler_refit"
] is False


expected_rows = {
    "clean_reference_check":
        120,

    "ar1_response_by_split":
        400,

    "ar1_response_summary":
        80,

    "bandwidth_response_by_split":
        400,

    "bandwidth_response_summary":
        80,

    "controlled_drift_response_by_split":
        240,

    "controlled_drift_response_summary":
        48,

    "model_contrast_summary":
        104,

    "perturbation_spectral_summary":
        13,
}


for key, expected in expected_rows.items():

    info = m[
        "outputs"
    ][
        key
    ]

    p = Path(
        info[
            "path"
        ]
    )

    assert p.is_file()

    assert sha(
        p
    ) == info[
        "sha256"
    ]

    df = pd.read_csv(
        p
    )

    assert len(
        df
    ) == expected

    assert int(
        info[
            "rows"
        ]
    ) == expected


clean = pd.read_csv(
    m[
        "outputs"
    ][
        "clean_reference_check"
    ][
        "path"
    ]
)

assert clean[
    "pass"
].astype(bool).all()


# =============================================================================
# Independently recompute all split-level relative degradation values
# =============================================================================

frames = []


for p in sorted(
    RUNS.glob(
        "*_cbd_metrics.csv"
    )
):

    df = pd.read_csv(
        p,
        low_memory=False,
    )

    assert len(
        df
    ) == 264

    frames.append(
        df
    )


raw = pd.concat(
    frames,
    ignore_index=True,
)

assert len(
    raw
) == 7920


clean_raw = raw[
    raw[
        "scenario"
    ] == "clean"
][
    [
        "split_seed",
        "train_seed",
        "model",
        "domain",
        "rmse",
        "mae",
    ]
].copy()


clean_long = clean_raw.melt(
    id_vars=[
        "split_seed",
        "train_seed",
        "model",
        "domain",
    ],
    value_vars=[
        "rmse",
        "mae",
    ],
    var_name="metric",
    value_name="clean_value",
)


pert = raw[
    raw[
        "scenario"
    ] != "clean"
].copy()


long = pert.melt(
    id_vars=[
        "split_seed",
        "train_seed",
        "model",
        "domain",
        "scenario",
        "rho",
        "filter_name",
        "terminal_shift_std_units",
        "perturbation_seed",
    ],
    value_vars=[
        "rmse",
        "mae",
    ],
    var_name="metric",
    value_name="perturbed_value",
)


long = long.merge(
    clean_long,
    on=[
        "split_seed",
        "train_seed",
        "model",
        "domain",
        "metric",
    ],
    how="left",
    validate="many_to_one",
)


long[
    "relative_degradation"
] = (
    long[
        "perturbed_value"
    ]
    - long[
        "clean_value"
    ]
) / long[
    "clean_value"
]


def independent_check(
    scenario,
    parameter,
    output_key,
):

    source = long[
        long[
            "scenario"
        ] == scenario
    ]

    calc = (
        source
        .groupby(
            [
                "split_seed",
                "model",
                "domain",
                "metric",
                parameter,
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            relative_degradation=(
                "relative_degradation",
                "mean",
            ),

            nested_count=(
                "relative_degradation",
                "size",
            ),
        )
    )


    saved = pd.read_csv(
        m[
            "outputs"
        ][
            output_key
        ][
            "path"
        ]
    )


    keys = [
        "split_seed",
        "model",
        "domain",
        "metric",
        parameter,
    ]


    check = calc.merge(
        saved[
            keys
            + [
                "relative_degradation",
                "nested_count",
            ]
        ],
        on=keys,
        suffixes=(
            "_calc",
            "_saved",
        ),
        validate="one_to_one",
    )


    assert len(
        check
    ) == len(
        saved
    )


    assert np.allclose(
        check[
            "relative_degradation_calc"
        ],
        check[
            "relative_degradation_saved"
        ],
        rtol=1e-12,
        atol=1e-14,
    )

    assert (
        check[
            "nested_count_calc"
        ]
        == check[
            "nested_count_saved"
        ]
    ).all()

    assert set(
        check[
            "nested_count_calc"
        ]
    ) == {15}


independent_check(
    "ar1_correlation",
    "rho",
    "ar1_response_by_split",
)

independent_check(
    "bandwidth",
    "filter_name",
    "bandwidth_response_by_split",
)

independent_check(
    "controlled_drift",
    "terminal_shift_std_units",
    "controlled_drift_response_by_split",
)


# =============================================================================
# Independently recompute perturbation spectral setting summaries
# =============================================================================

diag = pd.read_csv(
    DIAG,
    low_memory=False,
)

assert len(
    diag
) == 325


def setting(row):

    if row.scenario == "ar1_correlation":
        return f"rho={float(row.rho):+.2f}"

    if row.scenario == "bandwidth":
        return str(
            row.filter_name
        )

    if row.scenario == "controlled_drift":
        return (
            "terminal="
            f"{float(row.terminal_shift_std_units):.2f}"
        )

    raise AssertionError(
        row.scenario
    )


diag[
    "setting"
] = [
    setting(row)
    for row in diag.itertuples(
        index=False
    )
]


calc = (
    diag
    .groupby(
        [
            "scenario",
            "setting",
        ],
        as_index=False,
        observed=True,
    )
    .agg(
        diagnostic_count=(
            "condition_id",
            "size",
        ),

        global_rms_mean=(
            "global_rms",
            "mean",
        ),

        spectral_centroid_mean=(
            "spectral_centroid_omega_over_pi",
            "mean",
        ),

        low_energy_fraction_mean=(
            "low_energy_fraction",
            "mean",
        ),

        mid_low_energy_fraction_mean=(
            "mid_low_energy_fraction",
            "mean",
        ),

        mid_high_energy_fraction_mean=(
            "mid_high_energy_fraction",
            "mean",
        ),

        high_energy_fraction_mean=(
            "high_energy_fraction",
            "mean",
        ),
    )
)


saved = pd.read_csv(
    m[
        "outputs"
    ][
        "perturbation_spectral_summary"
    ][
        "path"
    ]
)


check = calc.merge(
    saved,
    on=[
        "scenario",
        "setting",
    ],
    suffixes=(
        "_calc",
        "_saved",
    ),
    validate="one_to_one",
)


assert len(
    check
) == 13


for col in [
    "global_rms_mean",
    "spectral_centroid_mean",
    "low_energy_fraction_mean",
    "mid_low_energy_fraction_mean",
    "mid_high_energy_fraction_mean",
    "high_energy_fraction_mean",
]:

    assert np.allclose(
        check[
            col
            + "_calc"
        ],
        check[
            col
            + "_saved"
        ],
        rtol=1e-12,
        atol=1e-14,
    )


assert (
    check[
        "diagnostic_count_calc"
    ] == 25
).all()

assert (
    check[
        "diagnostic_count_saved"
    ] == 25
).all()


fractions = (
    saved[
        "low_energy_fraction_mean"
    ]
    + saved[
        "mid_low_energy_fraction_mean"
    ]
    + saved[
        "mid_high_energy_fraction_mean"
    ]
    + saved[
        "high_energy_fraction_mean"
    ]
)

assert np.allclose(
    fractions,
    1.0,
    rtol=1e-10,
    atol=1e-12,
)


contrast = pd.read_csv(
    m[
        "outputs"
    ][
        "model_contrast_summary"
    ][
        "path"
    ]
)

assert len(
    contrast
) == 104

assert set(
    contrast[
        "gcn_favored_splits"
    ]
    + contrast[
        "mlp_favored_splits"
    ]
    + contrast[
        "ties"
    ]
) == {5}


print("=" * 88)
print("CORRELATION / BANDWIDTH / DRIFT ANALYSIS AUDIT")
print("=" * 88)

print(
    "spectral protocol exact:",
    True
)

print(
    "execution manifest exact:",
    True
)

print(
    "analysis script exact:",
    True
)

print(
    "clean final-v3 reference barrier:",
    True
)

print(
    "clean reference rows:",
    120
)

print(
    "AR1 split-level rows:",
    400
)

print(
    "bandwidth split-level rows:",
    400
)

print(
    "controlled-drift split-level rows:",
    240
)

print(
    "nested observations per split/setting:",
    15
)

print(
    "split-level responses independently reproduced:",
    True
)

print(
    "model contrast rows:",
    104
)

print(
    "perturbation spectral settings:",
    13
)

print(
    "spectral summaries independently reproduced:",
    True
)

print(
    "new hypothesis tests performed:",
    False
)

print(
    "models retrained:",
    False
)

print(
    "scaler refit:",
    False
)

print(
    "manifest SHA256:",
    sha(
        MANIFEST
    )
)

print()
print(
    "CORRELATION_BANDWIDTH_DRIFT_ANALYSIS_AUDIT_PASS=True"
)
