from pathlib import Path
import hashlib
import json
import math
import os
import subprocess


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo" / "battery-gnn"

THIS = (
    REPO
    / "scripts/revision_2026/65_freeze_spectral_frequency_protocol.py"
)

BENCHMARK = (
    REPO
    / "scripts/revision_2026/24_feature_matched_soh_benchmark.py"
)

FINAL_V3 = (
    BASE
    / "results/revision_2026/feature_matched_soh/final_v3"
)

ROOT = (
    BASE
    / "results/revision_2026/spectral_frequency/final_spectral_v1"
)

PROTOCOL = ROOT / "protocol.json"


EXPECTED_FINAL_V3_PROTOCOL_SHA = (
    "2dec14c210ae3f02e5f65e4acc24f9c4adc747bc071509cb144e69dba707f4dd"
)

EXPECTED_FINAL_V3_EXECUTION_SHA = (
    "5a8bfbe0f11a6158f2749a9da4f18f1e62adf8c839976c1bf0af9c9f6c07440d"
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


def find_json_by_sha(root, expected):
    hits = []

    for p in sorted(
        Path(root).rglob("*.json")
    ):
        if sha(p) == expected:
            hits.append(p)

    assert len(hits) == 1, (
        expected,
        [str(x) for x in hits],
    )

    return hits[0]


assert BENCHMARK.is_file()
assert FINAL_V3.is_dir()

# The frozen v3 protocol is stored at the feature_matched_soh root,
# while the execution manifest is stored inside final_v3.
final_v3_protocol = find_json_by_sha(
    FINAL_V3.parent,
    EXPECTED_FINAL_V3_PROTOCOL_SHA,
)

final_v3_execution = find_json_by_sha(
    FINAL_V3,
    EXPECTED_FINAL_V3_EXECUTION_SHA,
)


if ROOT.exists():
    existing = [
        p
        for p in ROOT.rglob("*")
        if p.is_file()
    ]

    assert not existing


git_commit = subprocess.check_output(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
    text=True,
).strip()


script_sha = sha(THIS)
benchmark_sha = sha(BENCHMARK)


# ---------------------------------------------------------------------------
# Controlled temporal-frequency stress grid.
#
# These are standardized-feature-space mechanism tests, NOT physical sensor
# specifications. Physical-unit V/I corruption is already handled separately.
# ---------------------------------------------------------------------------

frequency_omega_over_pi = [
    0.0625,
    0.125,
    0.25,
    0.50,
    0.75,
    1.00,
]

ar1_rho = [
    -0.90,
    -0.50,
    0.00,
    0.50,
    0.90,
]

drift_terminal_shift_std = [
    0.10,
    0.25,
    0.50,
]

perturbation_seeds = [
    0,
    1,
    2,
    3,
    4,
]

depths = [
    0,
    1,
    2,
    3,
    4,
]


# All FIR filters are causal: tap 0 is current sample,
# subsequent taps are past innovations.
sqrt2 = math.sqrt(2.0)

bandwidth_filters = {
    "white": [
        1.0,
    ],

    "lowpass_ma3": [
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
    ],

    "lowpass_ma9": [
        1.0 / 9.0,
    ] * 9,

    "bandpass_ma3_minus_ma9": [
        2.0 / 9.0,
        2.0 / 9.0,
        2.0 / 9.0,
        -1.0 / 9.0,
        -1.0 / 9.0,
        -1.0 / 9.0,
        -1.0 / 9.0,
        -1.0 / 9.0,
        -1.0 / 9.0,
    ],

    "highpass_first_difference": [
        1.0 / sqrt2,
        -1.0 / sqrt2,
    ],
}


protocol = {
    "contract":
        "final_spectral_frequency_v1",

    "status":
        "FROZEN_BEFORE_SPECTRAL_RUNS",

    "frozen_at_git_commit":
        git_commit,

    "protocol_script_sha256":
        script_sha,

    "benchmark_source_sha256":
        benchmark_sha,

    "final_v3_protocol": {
        "path":
            str(final_v3_protocol),

        "sha256":
            EXPECTED_FINAL_V3_PROTOCOL_SHA,
    },

    "final_v3_execution_manifest": {
        "path":
            str(final_v3_execution),

        "sha256":
            EXPECTED_FINAL_V3_EXECUTION_SHA,
    },

    "scientific_role":
        (
            "Mechanistic temporal-frequency robustness analysis; "
            "not a claim that the causal GCN is a spectral GCN."
        ),

    "primary_task":
        "cycle-level SOH regression",

    "datasets": [
        "HUST",
        "CALCE",
        "MATR",
    ],

    "domains": [
        "BALANCED_ALL",
        "HUST",
        "CALCE",
        "MATR",
    ],

    "models": [
        "FeatureMatchedMLP",
        "FeatureMatchedCausalGCN",
    ],

    "model_policy": {
        "states":
            "frozen final_v3 model states",

        "retraining":
            False,

        "noise_augmentation":
            False,

        "scaler_refit":
            False,

        "test_only_perturbation":
            True,

        "paired_realization_across_models":
            True,
    },

    "feature_space": {
        "dimension":
            20,

        "space":
            "train-only standardized primary feature space",

        "purpose":
            (
                "controlled mechanism experiment; "
                "not physical sensor realism"
            ),

        "cycle_index_feature":
            False,

        "target_feature":
            False,
    },

    "replication": {
        "split_seeds": [
            0,
            1,
            2,
            3,
            4,
        ],

        "train_initializations_per_split":
            3,

        "perturbation_seeds":
            perturbation_seeds,

        "inferential_unit":
            "split_seed",

        "inferential_n":
            5,

        "train_initializations":
            "nested_within_split",

        "perturbation_seeds_role":
            "nested_within_split",
    },

    "graph_operator": {
        "model_operator":
            (
                "causal parameter-free mean aggregation "
                "over {self, previous valid observation}"
            ),

        "boundary_rule":
            "first valid observation uses self only",

        "aggregation_stages_in_frozen_gcn":
            2,

        "maximum_temporal_receptive_field":
            "{t,t-1,t-2}",

        "directed":
            True,

        "future_edges":
            False,

        "cross_cell_edges":
            False,

        "interior_single_stage_transfer":
            "H1(omega)=0.5*(1+exp(-j*omega))",

        "interior_single_stage_magnitude":
            "|H1(omega)|=|cos(omega/2)|",

        "depth_d_magnitude":
            "|Hd(omega)|=|cos(omega/2)|^d",

        "analytic_depth_grid":
            depths,

        "depth_sweep_role":
            (
                "analytic aggregation-operator response only; "
                "not separately trained depth models"
            ),
    },

    "laplacian_graph_fourier": {
        "reason_for_separate_definition":
            (
                "the deployed causal operator is directed and non-symmetric; "
                "an orthonormal graph-Fourier basis is therefore defined "
                "on the corresponding undirected chronological path graph"
            ),

        "graph":
            "undirected path graph P_n for each cell trajectory",

        "laplacian":
            "combinatorial L=D-A",

        "node_order":
            "chronological valid observation order",

        "eigenvalue_index":
            "k=0,...,n-1",

        "exact_eigenvalue_formula":
            "lambda_k=2-2*cos(pi*k/n)",

        "frequency_coordinate":
            "omega_k_over_pi=k/n",

        "orthonormal_basis":
            (
                "q_0(j)=1/sqrt(n); "
                "q_k(j)=sqrt(2/n)*cos(pi*k*(j+0.5)/n), k>0"
            ),

        "input_spectral_quantity":
            (
                "energy of clean standardized feature trajectories "
                "projected onto the path-Laplacian basis"
            ),

        "residual_spectral_quantity":
            (
                "energy of per-cycle prediction residual trajectories "
                "projected onto the same cell-specific basis"
            ),

        "bands_omega_over_pi": {
            "low":
                [0.00, 0.25],

            "mid_low":
                [0.25, 0.50],

            "mid_high":
                [0.50, 0.75],

            "high":
                [0.75, 1.00],
        },

        "band_edge_rule":
            (
                "left-closed/right-open except final band right-closed"
            ),
    },

    "sinusoidal_frequency_sweep": {
        "role":
            "PRIMARY controlled frequency-response experiment",

        "omega_over_pi":
            frequency_omega_over_pi,

        "rms_per_feature_standardized_units":
            0.10,

        "phase":
            (
                "deterministic stable-hash phase per "
                "cell x feature x perturbation seed"
            ),

        "same_phase_realization_for_mlp_and_gcn":
            True,

        "per_cell_feature_waveform":
            (
                "cos(omega*t+phase), scaled to exact requested RMS "
                "when numerically well-defined"
            ),

        "clean_reference":
            "same frozen run with zero perturbation",
    },

    "ar1_correlation_sweep": {
        "role":
            "DESCRIPTIVE mechanism check",

        "rho":
            ar1_rho,

        "stationary_marginal_std_standardized_units":
            0.10,

        "innovation_std":
            "sigma*sqrt(1-rho^2)",

        "initial_state":
            "stationary N(0,sigma^2)",

        "same_realization_for_mlp_and_gcn":
            True,
    },

    "bandwidth_sweep": {
        "role":
            "DESCRIPTIVE mechanism check",

        "source":
            "independent white innovations",

        "causal_FIR_filters":
            bandwidth_filters,

        "post_filter_rms_standardized_units":
            0.10,

        "same_realization_for_mlp_and_gcn":
            True,

        "empirical_check":
            (
                "report perturbation spectral centroid and "
                "Laplacian-band energy fractions"
            ),
    },

    "controlled_drift_sweep": {
        "role":
            "DESCRIPTIVE low-frequency mechanism check",

        "terminal_shift_standardized_units":
            drift_terminal_shift_std,

        "trajectory":
            (
                "linear causal ramp from zero at first observation "
                "to signed terminal shift at final observation"
            ),

        "sign":
            (
                "stable-hash sign per cell x feature x perturbation seed"
            ),

        "same_realization_for_mlp_and_gcn":
            True,

        "physical_reality_note":
            (
                "not a replacement for the separately completed "
                "raw V/I physical-unit drift benchmark"
            ),
    },

    "primary_frequency_inference": {
        "metric_family": [
            "RMSE",
            "MAE",
        ],

        "frequency_response_quantity":
            (
                "relative degradation=(perturbed_error-clean_error)"
                "/clean_error"
            ),

        "summary":
            (
                "trapezoidal AUC of relative degradation over the "
                "frozen omega/pi frequency grid, normalized by "
                "frequency-range width"
            ),

        "delta":
            "MLP_degradation_AUC_minus_GCN_degradation_AUC",

        "positive_delta":
            "GCN lower frequency-integrated relative degradation",

        "paired_unit":
            "split_seed",

        "n":
            5,

        "test":
            "exact two-sided paired sign-flip test",

        "multiple_testing":
            (
                "Holm across four domains separately for RMSE "
                "and four domains separately for MAE"
            ),

        "comparisons":
            8,

        "minimum_attainable_two_sided_p":
            0.0625,
    },

    "descriptive_outputs": [
        "actual path-Laplacian eigenvalue summaries by cell length",
        "clean input graph-Fourier band-energy fractions",
        "clean residual graph-Fourier band-energy fractions by model",
        "frequency-response curves",
        "AR1 rho-response curves",
        "causal-FIR bandwidth-response curves",
        "controlled drift-response curves",
        "analytic aggregation-depth transfer curves",
    ],

    "claim_guard": {
        "universal_gcn_robustness_claim":
            False,

        "spectral_gcn_claim":
            False,

        "mechanism_language":
            (
                "frequency-selective temporal aggregation behavior "
                "and graph-Fourier diagnostics"
            ),

        "directed_vs_laplacian_distinction_mandatory":
            True,

        "physical_sensor_claim_from_feature_space_sweeps":
            False,
    },

    "prohibited_during_protocol_freeze": [
        "read final spectral outcome metrics",
        "change grid after seeing spectral outcomes",
        "retrain frozen final_v3 models",
        "refit train-only scaler",
        "use future observations in model input",
        "call the causal model a spectral GCN",
    ],
}


ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

tmp = Path(
    str(PROTOCOL) + ".tmp"
)

tmp.write_text(
    json.dumps(
        protocol,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

os.replace(
    tmp,
    PROTOCOL,
)


print("=" * 84)
print("SPECTRAL / FREQUENCY PROTOCOL FREEZE")
print("=" * 84)

print(
    "contract:",
    protocol["contract"]
)

print(
    "status:",
    protocol["status"]
)

print(
    "frozen git commit:",
    git_commit
)

print(
    "frequency points:",
    len(
        frequency_omega_over_pi
    )
)

print(
    "frequency omega/pi:",
    ",".join(
        str(x)
        for x in frequency_omega_over_pi
    )
)

print(
    "AR1 rho points:",
    ",".join(
        str(x)
        for x in ar1_rho
    )
)

print(
    "bandwidth filters:",
    len(
        bandwidth_filters
    )
)

print(
    "drift levels:",
    len(
        drift_terminal_shift_std
    )
)

print(
    "analytic depths:",
    ",".join(
        str(x)
        for x in depths
    )
)

print(
    "primary inferential comparisons:",
    8
)

print(
    "inferential unit:",
    "split_seed_n5"
)

print(
    "spectral outcome metrics read:",
    False
)

print(
    "model retraining:",
    False
)

print(
    "scaler refit:",
    False
)

print(
    "protocol SHA256:",
    sha(PROTOCOL)
)

print()
print(
    "SPECTRAL_FREQUENCY_PROTOCOL_FREEZE_PASS=True"
)
