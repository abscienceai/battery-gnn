from pathlib import Path
from itertools import product
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

EVAL = ROOT / "sinusoidal_frequency_v1"
OUT = EVAL / "analysis_v1"

PROTOCOL = ROOT / "protocol.json"

EXEC = (
    EVAL
    / "sinusoidal_frequency_execution_manifest.json"
)

SCRIPT = (
    REPO
    / "scripts/revision_2026/71_analyze_sinusoidal_frequency_response.py"
)

MANIFEST = (
    OUT
    / "frequency_analysis_manifest.json"
)


EXPECTED_PROTOCOL_SHA = (
    "a8d3b65b87980470f326717f0cb50e3d4fb4d2e0429c93171eb0d21d6c713e45"
)

EXPECTED_EXEC_SHA = (
    "ac56254c584990651ee7430786f3a39b3f7ed96d278091e65e30b115651d19fd"
)

FREQUENCIES = np.asarray(
    [
        0.0625,
        0.125,
        0.25,
        0.50,
        0.75,
        1.00,
    ],
    dtype=np.float64,
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


def auc(x, y):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    return float(
        np.sum(
            (
                x[1:]
                - x[:-1]
            )
            * (
                y[1:]
                + y[:-1]
            )
            / 2.0
        )
        / (
            x[-1]
            - x[0]
        )
    )


def signflip(deltas):
    d = np.asarray(
        deltas,
        dtype=np.float64,
    )

    observed = abs(
        float(
            np.mean(d)
        )
    )

    null = []

    for signs in product(
        [-1.0, 1.0],
        repeat=5,
    ):

        s = np.asarray(
            signs,
            dtype=np.float64,
        )

        null.append(
            abs(
                float(
                    np.mean(
                        s * d
                    )
                )
            )
        )

    return float(
        np.mean(
            np.asarray(
                null
            )
            >= observed - 1e-15
        )
    )


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
] == "sinusoidal_frequency_analysis_v1"

assert m[
    "status"
] == "COMPLETE"

assert m[
    "spectral_protocol_sha256"
] == EXPECTED_PROTOCOL_SHA

assert m[
    "frequency_execution_manifest_sha256"
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
] == 3720

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
    "frequency_grid_omega_over_pi"
] == FREQUENCIES.tolist()

assert m[
    "nested_count_per_split_frequency"
] == 15

assert m[
    "inferential_unit"
] == "split_seed"

assert m[
    "inferential_n"
] == 5

assert m[
    "primary_comparisons"
] == 8

assert m[
    "minimum_attainable_two_sided_p"
] == 0.0625

assert m[
    "minimum_possible_first_holm_adjusted_p"
] == 0.25

assert m[
    "models_retrained"
] is False

assert m[
    "scaler_refit"
] is False


expected_rows = {
    "clean_reference_check":
        120,

    "frequency_response_by_split":
        480,

    "frequency_response_summary":
        96,

    "frequency_auc_by_split":
        80,

    "frequency_primary_inference":
        8,

    "analytic_causal_aggregation_transfer":
        30,
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

assert len(
    clean
) == 120

assert clean[
    "pass"
].astype(bool).all()


curve = pd.read_csv(
    m[
        "outputs"
    ][
        "frequency_response_by_split"
    ][
        "path"
    ]
)

assert len(
    curve
) == 480

assert set(
    curve[
        "nested_count"
    ]
) == {15}


auc_table = pd.read_csv(
    m[
        "outputs"
    ][
        "frequency_auc_by_split"
    ][
        "path"
    ]
)

assert len(
    auc_table
) == 80


# Independently recompute every split-level AUC.
for row in auc_table.itertuples(
    index=False
):

    g = curve[
        (
            curve[
                "split_seed"
            ] == row.split_seed
        )
        & (
            curve[
                "model"
            ] == row.model
        )
        & (
            curve[
                "domain"
            ] == row.domain
        )
        & (
            curve[
                "metric"
            ] == row.metric
        )
    ].sort_values(
        "omega_over_pi",
        kind="mergesort",
    )

    assert len(
        g
    ) == 6

    x = g[
        "omega_over_pi"
    ].to_numpy(
        dtype=np.float64
    )

    y = g[
        "relative_degradation"
    ].to_numpy(
        dtype=np.float64
    )

    assert np.allclose(
        x,
        FREQUENCIES,
        rtol=0.0,
        atol=1e-15,
    )

    expected = auc(
        x,
        y,
    )

    assert np.isclose(
        expected,
        float(
            row.normalized_frequency_auc
        ),
        rtol=1e-12,
        atol=1e-14,
    )


inference = pd.read_csv(
    m[
        "outputs"
    ][
        "frequency_primary_inference"
    ][
        "path"
    ]
)

assert len(
    inference
) == 8

assert set(
    inference[
        "metric"
    ]
) == {
    "rmse",
    "mae",
}

for metric in [
    "rmse",
    "mae",
]:

    assert int(
        (
            inference[
                "metric"
            ] == metric
        ).sum()
    ) == 4


# Independently recompute exact sign-flip p-values.
for row in inference.itertuples(
    index=False
):

    g = auc_table[
        (
            auc_table[
                "metric"
            ] == row.metric
        )
        & (
            auc_table[
                "domain"
            ] == row.domain
        )
    ]

    pivot = g.pivot(
        index="split_seed",
        columns="model",
        values="normalized_frequency_auc",
    ).loc[
        [0, 1, 2, 3, 4]
    ]

    delta = (
        pivot[
            "MLP"
        ].to_numpy(
            dtype=np.float64
        )
        - pivot[
            "GCN"
        ].to_numpy(
            dtype=np.float64
        )
    )

    assert np.isclose(
        np.mean(
            delta
        ),
        float(
            row.mean_delta_mlp_minus_gcn
        ),
        rtol=1e-12,
        atol=1e-14,
    )

    assert np.isclose(
        signflip(
            delta
        ),
        float(
            row.p_exact_signflip
        ),
        rtol=0.0,
        atol=1e-15,
    )


assert (
    inference[
        "p_holm"
    ]
    >= inference[
        "p_exact_signflip"
    ]
    - 1e-15
).all()


transfer = pd.read_csv(
    m[
        "outputs"
    ][
        "analytic_causal_aggregation_transfer"
    ][
        "path"
    ]
)

assert len(
    transfer
) == 30


for row in transfer.itertuples(
    index=False
):

    expected = (
        abs(
            np.cos(
                np.pi
                * float(
                    row.omega_over_pi
                )
                / 2.0
            )
        )
        ** int(
            row.depth
        )
    )

    assert np.isclose(
        expected,
        float(
            row.amplitude_gain
        ),
        rtol=1e-13,
        atol=1e-15,
    )

    assert np.isclose(
        expected ** 2,
        float(
            row.power_gain
        ),
        rtol=1e-13,
        atol=1e-15,
    )


print("=" * 86)
print("SINUSOIDAL FREQUENCY ANALYSIS AUDIT")
print("=" * 86)

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
    "split-level frequency curve rows:",
    480
)

print(
    "nested observations per curve point:",
    15
)

print(
    "split-level AUC rows:",
    80
)

print(
    "primary comparisons:",
    8
)

print(
    "AUC independently reproduced:",
    True
)

print(
    "exact sign-flip p-values reproduced:",
    True
)

print(
    "Holm families:",
    "4 RMSE + 4 MAE"
)

print(
    "analytic transfer rows:",
    30
)

print(
    "analytic transfer independently reproduced:",
    True
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
    "SINUSOIDAL_FREQUENCY_ANALYSIS_AUDIT_PASS=True"
)
