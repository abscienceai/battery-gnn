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

OUT = ROOT / "clean_graph_fourier_v1"

PROTOCOL = ROOT / "protocol.json"

SCRIPT = (
    REPO
    / "scripts/revision_2026/67_clean_graph_fourier_diagnostics.py"
)

MANIFEST = (
    OUT
    / "clean_graph_fourier_manifest.json"
)

EXPECTED_PROTOCOL_SHA = (
    "a8d3b65b87980470f326717f0cb50e3d4fb4d2e0429c93171eb0d21d6c713e45"
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


assert sha(PROTOCOL) == EXPECTED_PROTOCOL_SHA


m = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)


assert m[
    "contract"
] == "clean_graph_fourier_diagnostics_v1"

assert m[
    "status"
] == "COMPLETE_BEFORE_FREQUENCY_PERTURBATION_SWEEP"

assert m[
    "spectral_protocol_sha256"
] == EXPECTED_PROTOCOL_SHA

assert m[
    "analysis_script_sha256"
] == sha(SCRIPT)

assert m[
    "models_retrained"
] is False

assert m[
    "scaler_refit"
] is False

assert m[
    "model_states_loaded"
] == 30

assert m[
    "graph"
] == "undirected chronological path graph P_n"

assert m[
    "laplacian"
] == "combinatorial D-A"

assert m[
    "eigenvalue_formula"
] == "lambda_k=2-2*cos(pi*k/n)"

assert m[
    "frequency_coordinate"
] == "omega_k_over_pi=k/n"

assert m[
    "transform"
] == "scipy.fft.dct type=2 norm=ortho"

assert m[
    "clean_metric_regression_barrier"
] == "PASS"

assert m[
    "clean_metric_rows"
] == 120

assert float(
    m[
        "clean_metric_max_abs_rmse_difference"
    ]
) <= 1e-7

assert float(
    m[
        "clean_metric_max_abs_mae_difference"
    ]
) <= 1e-7

assert float(
    m[
        "input_parseval_max_relative_error"
    ]
) <= 1e-10

assert float(
    m[
        "residual_parseval_max_relative_error"
    ]
) <= 1e-10


for info in m[
    "outputs"
].values():

    p = Path(
        info[
            "path"
        ]
    )

    assert p.is_file()
    assert sha(p) == info[
        "sha256"
    ]

    df = pd.read_csv(
        p
    )

    assert len(df) == int(
        info[
            "rows"
        ]
    )


cells = pd.read_csv(
    m[
        "outputs"
    ][
        "path_laplacian_cells"
    ][
        "path"
    ]
)

modes = pd.read_csv(
    m[
        "outputs"
    ][
        "path_laplacian_input_modes"
    ][
        "path"
    ]
)

metrics = pd.read_csv(
    m[
        "outputs"
    ][
        "clean_prediction_metric_regression"
    ][
        "path"
    ]
)

assert len(metrics) == 120
assert metrics["pass"].astype(bool).all()

assert np.allclose(
    cells[
        "lambda_0"
    ],
    0.0,
    atol=1e-14,
    rtol=0.0,
)

assert (
    pd.to_numeric(
        cells[
            "lambda_max"
        ],
        errors="raise",
    )
    < 4.0
).all()


# Verify the actual path-Laplacian formula directly.
k = pd.to_numeric(
    modes[
        "k"
    ],
    errors="raise",
).to_numpy(
    dtype=np.float64
)

n = pd.to_numeric(
    modes[
        "n"
    ],
    errors="raise",
).to_numpy(
    dtype=np.float64
)

observed_lambda = pd.to_numeric(
    modes[
        "eigenvalue"
    ],
    errors="raise",
).to_numpy(
    dtype=np.float64
)

expected_lambda = (
    2.0
    - 2.0
    * np.cos(
        np.pi
        * k
        / n
    )
)

assert np.allclose(
    observed_lambda,
    expected_lambda,
    rtol=1e-13,
    atol=1e-14,
)


observed_omega = pd.to_numeric(
    modes[
        "omega_over_pi"
    ],
    errors="raise",
).to_numpy(
    dtype=np.float64
)

assert np.allclose(
    observed_omega,
    k / n,
    rtol=0.0,
    atol=1e-15,
)


print("=" * 84)
print("CLEAN GRAPH-FOURIER DIAGNOSTICS AUDIT")
print("=" * 84)

print(
    "spectral protocol exact:",
    True
)

print(
    "analysis script exact:",
    True
)

print(
    "frozen state files:",
    30
)

print(
    "clean metric regression barrier:",
    True
)

print(
    "clean metric rows:",
    120
)

print(
    "path-Laplacian eigenvalue formula exact:",
    True
)

print(
    "frequency coordinate exact:",
    True
)

print(
    "input Parseval barrier:",
    True
)

print(
    "residual Parseval barrier:",
    True
)

print(
    "test cell representations:",
    int(
        m[
            "cell_rows"
        ]
    )
)

print(
    "actual Laplacian modes:",
    int(
        m[
            "mode_rows"
        ]
    )
)

print(
    "manifest SHA256:",
    sha(MANIFEST)
)

print()
print(
    "CLEAN_GRAPH_FOURIER_DIAGNOSTICS_AUDIT_PASS=True"
)
