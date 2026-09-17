from pathlib import Path
import hashlib
import json
import math


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo" / "battery-gnn"

ROOT = (
    BASE
    / "results/revision_2026/spectral_frequency/final_spectral_v1"
)

PROTOCOL = ROOT / "protocol.json"

SCRIPT = (
    REPO
    / "scripts/revision_2026/65_freeze_spectral_frequency_protocol.py"
)

BENCHMARK = (
    REPO
    / "scripts/revision_2026/24_feature_matched_soh_benchmark.py"
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


p = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)


assert p[
    "contract"
] == "final_spectral_frequency_v1"

assert p[
    "status"
] == "FROZEN_BEFORE_SPECTRAL_RUNS"

assert p[
    "protocol_script_sha256"
] == sha(SCRIPT)

assert p[
    "benchmark_source_sha256"
] == sha(BENCHMARK)

assert p[
    "final_v3_protocol"
][
    "sha256"
] == (
    "2dec14c210ae3f02e5f65e4acc24f9c4adc747bc071509cb144e69dba707f4dd"
)

assert p[
    "final_v3_execution_manifest"
][
    "sha256"
] == (
    "5a8bfbe0f11a6158f2749a9da4f18f1e62adf8c839976c1bf0af9c9f6c07440d"
)


assert p[
    "model_policy"
][
    "retraining"
] is False

assert p[
    "model_policy"
][
    "scaler_refit"
] is False

assert p[
    "model_policy"
][
    "test_only_perturbation"
] is True

assert p[
    "model_policy"
][
    "paired_realization_across_models"
] is True


assert p[
    "graph_operator"
][
    "aggregation_stages_in_frozen_gcn"
] == 2

assert p[
    "graph_operator"
][
    "directed"
] is True

assert p[
    "graph_operator"
][
    "future_edges"
] is False

assert p[
    "graph_operator"
][
    "cross_cell_edges"
] is False

assert p[
    "graph_operator"
][
    "analytic_depth_grid"
] == [
    0,
    1,
    2,
    3,
    4,
]


assert p[
    "laplacian_graph_fourier"
][
    "laplacian"
] == "combinatorial L=D-A"

assert p[
    "laplacian_graph_fourier"
][
    "exact_eigenvalue_formula"
] == "lambda_k=2-2*cos(pi*k/n)"

assert p[
    "laplacian_graph_fourier"
][
    "frequency_coordinate"
] == "omega_k_over_pi=k/n"


assert p[
    "sinusoidal_frequency_sweep"
][
    "omega_over_pi"
] == [
    0.0625,
    0.125,
    0.25,
    0.50,
    0.75,
    1.00,
]

assert math.isclose(
    p[
        "sinusoidal_frequency_sweep"
    ][
        "rms_per_feature_standardized_units"
    ],
    0.10,
)


assert p[
    "ar1_correlation_sweep"
][
    "rho"
] == [
    -0.90,
    -0.50,
    0.00,
    0.50,
    0.90,
]


filters = p[
    "bandwidth_sweep"
][
    "causal_FIR_filters"
]

assert set(filters) == {
    "white",
    "lowpass_ma3",
    "lowpass_ma9",
    "bandpass_ma3_minus_ma9",
    "highpass_first_difference",
}

assert math.isclose(
    sum(
        filters[
            "bandpass_ma3_minus_ma9"
        ]
    ),
    0.0,
    abs_tol=1e-15,
)


assert p[
    "controlled_drift_sweep"
][
    "terminal_shift_standardized_units"
] == [
    0.10,
    0.25,
    0.50,
]


assert p[
    "primary_frequency_inference"
][
    "comparisons"
] == 8

assert p[
    "primary_frequency_inference"
][
    "paired_unit"
] == "split_seed"

assert p[
    "primary_frequency_inference"
][
    "n"
] == 5

assert p[
    "primary_frequency_inference"
][
    "minimum_attainable_two_sided_p"
] == 0.0625


assert p[
    "claim_guard"
][
    "universal_gcn_robustness_claim"
] is False

assert p[
    "claim_guard"
][
    "spectral_gcn_claim"
] is False

assert p[
    "claim_guard"
][
    "directed_vs_laplacian_distinction_mandatory"
] is True


# No spectral evaluation may have happened before freeze.
for name in [
    "evaluation",
    "analysis",
    "predictions",
]:
    d = ROOT / name

    if d.exists():
        assert not any(
            x.is_file()
            for x in d.rglob("*")
        )


print("=" * 84)
print("SPECTRAL / FREQUENCY PROTOCOL AUDIT")
print("=" * 84)

print(
    "protocol exact:",
    True
)

print(
    "final-v3 anchors exact:",
    True
)

print(
    "frozen models / no retraining:",
    True
)

print(
    "directed causal operator separated from Laplacian diagnostic:",
    True
)

print(
    "actual path-Laplacian eigenvalue formula frozen:",
    True
)

print(
    "frequency sweep frozen:",
    True
)

print(
    "AR1 rho sweep frozen:",
    True
)

print(
    "causal bandwidth sweep frozen:",
    True
)

print(
    "controlled drift sweep frozen:",
    True
)

print(
    "analytic depth sweep frozen:",
    True
)

print(
    "primary comparisons:",
    8
)

print(
    "spectral outcomes existed before freeze:",
    False
)

print(
    "protocol SHA256:",
    sha(PROTOCOL)
)

print()
print(
    "SPECTRAL_FREQUENCY_PROTOCOL_AUDIT_PASS=True"
)
