from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os
import subprocess

import numpy as np
import pandas as pd
from scipy.fft import dct
import torch


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

BENCH = (
    REPO
    / "scripts/revision_2026/24_feature_matched_soh_benchmark.py"
)

THIS = (
    REPO
    / "scripts/revision_2026/73_correlation_bandwidth_drift_sweep.py"
)

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

FINAL_V3 = (
    BASE
    / "results/revision_2026/feature_matched_soh/final_v3"
)

EVAL = ROOT / "correlation_bandwidth_drift_v1"
RUNS = EVAL / "runs"

PERTURBATION_DIAGNOSTICS = (
    EVAL
    / "perturbation_spectral_diagnostics.csv"
)

EXEC_MANIFEST = (
    EVAL
    / "correlation_bandwidth_drift_execution_manifest.json"
)


EXPECTED_PROTOCOL_SHA = (
    "a8d3b65b87980470f326717f0cb50e3d4fb4d2e0429c93171eb0d21d6c713e45"
)

EXPECTED_CLEAN_SHA = (
    "da9653f823f764ab85d69c2726f2de752a0cb7091b7c13d79a7db0e915751d12"
)

EXPECTED_FINAL_V3_EXEC_SHA = (
    "5a8bfbe0f11a6158f2749a9da4f18f1e62adf8c839976c1bf0af9c9f6c07440d"
)

CLEAN_TOL = 1e-7
EXACT_RMS_TOL = 2e-6
DRIFT_TERMINAL_TOL = 2e-6
PARSEVAL_RTOL = 1e-10

MODELS = ["MLP", "GCN"]

DOMAINS = [
    "BALANCED_ALL",
    "HUST",
    "CALCE",
    "MATR",
]

AR1_RHO = [
    -0.90,
    -0.50,
    0.00,
    0.50,
    0.90,
]

PERTURBATION_SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

FILTER_ORDER = [
    "white",
    "lowpass_ma3",
    "lowpass_ma9",
    "bandpass_ma3_minus_ma9",
    "highpass_first_difference",
]

DRIFT_LEVELS = [
    0.10,
    0.25,
    0.50,
]

BANDS = [
    ("low", 0.00, 0.25, False),
    ("mid_low", 0.25, 0.50, False),
    ("mid_high", 0.50, 0.75, False),
    ("high", 0.75, 1.00, True),
]


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def sha_bytes(data):
    return hashlib.sha256(
        data
    ).hexdigest()


def stable_seed(*tokens):
    text = "|".join(
        str(x)
        for x in tokens
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        text
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )


def atomic_csv(df, path):
    tmp = Path(
        str(path) + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
        float_format="%.17g",
    )

    os.replace(
        tmp,
        path,
    )


def atomic_json(obj, path):
    tmp = Path(
        str(path) + ".tmp"
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


def csv_data_rows(path):
    with Path(path).open(
        "rb"
    ) as f:

        lines = sum(
            1
            for _ in f
        )

    return lines - 1


def norm_model_name(x):
    s = str(x).strip().upper()

    if "GCN" in s:
        return "GCN"

    if "MLP" in s:
        return "MLP"

    raise AssertionError(
        ("unknown model", x)
    )


def load_state(path):
    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if isinstance(
        state,
        dict,
    ):

        if "model_state_dict" in state:
            state = state[
                "model_state_dict"
            ]

        elif "state_dict" in state:
            state = state[
                "state_dict"
            ]

    assert isinstance(
        state,
        dict,
    )

    return state


def group_records(frame):
    return (
        frame[
            [
                "dataset",
                "split_group",
            ]
        ]
        .drop_duplicates()
        .itertuples(
            index=False
        )
    )


def causal_fir(x, taps):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    taps = np.asarray(
        taps,
        dtype=np.float64,
    )

    y = np.convolve(
        x,
        taps,
        mode="full",
    )[:len(x)]

    assert len(y) == len(x)

    return y


def make_ar1_input(
    base_x,
    frame,
    features,
    rho,
    perturbation_seed,
    sigma,
):
    out = np.asarray(
        base_x,
        dtype=np.float64,
    ).copy()

    dataset_arr = (
        frame[
            "dataset"
        ]
        .astype(str)
        .to_numpy()
    )

    group_arr = (
        frame[
            "split_group"
        ]
        .astype(str)
        .to_numpy()
    )

    innovation_scale = (
        float(sigma)
        * math.sqrt(
            1.0
            - float(rho) ** 2
        )
    )


    for row in group_records(
        frame
    ):

        dataset = str(
            row.dataset
        )

        group = str(
            row.split_group
        )

        idx = np.flatnonzero(
            (dataset_arr == dataset)
            & (group_arr == group)
        )

        assert len(idx) > 0


        for j, feature in enumerate(
            features
        ):

            rng = np.random.default_rng(
                stable_seed(
                    "ar1_shared_innovations",
                    dataset,
                    group,
                    feature,
                    perturbation_seed,
                )
            )

            noise = np.empty(
                len(idx),
                dtype=np.float64,
            )

            noise[0] = (
                float(sigma)
                * rng.normal()
            )

            if len(idx) > 1:

                z = rng.normal(
                    size=len(idx) - 1
                )

                for t in range(
                    1,
                    len(idx),
                ):

                    noise[t] = (
                        float(rho)
                        * noise[t - 1]
                        + innovation_scale
                        * z[t - 1]
                    )


            out[
                idx,
                j,
            ] += noise


    out32 = out.astype(
        np.float32
    )

    delta = (
        out32.astype(
            np.float64
        )
        - np.asarray(
            base_x,
            dtype=np.float32,
        ).astype(
            np.float64
        )
    )

    assert np.isfinite(
        delta
    ).all()

    return (
        out32,
        float(
            np.sqrt(
                np.mean(
                    delta ** 2
                )
            )
        ),
    )


def make_bandwidth_input(
    base_x,
    frame,
    features,
    filter_name,
    taps,
    perturbation_seed,
    target_rms,
):
    out = np.asarray(
        base_x,
        dtype=np.float64,
    ).copy()

    dataset_arr = (
        frame[
            "dataset"
        ]
        .astype(str)
        .to_numpy()
    )

    group_arr = (
        frame[
            "split_group"
        ]
        .astype(str)
        .to_numpy()
    )


    for row in group_records(
        frame
    ):

        dataset = str(
            row.dataset
        )

        group = str(
            row.split_group
        )

        idx = np.flatnonzero(
            (dataset_arr == dataset)
            & (group_arr == group)
        )

        assert len(idx) > 0


        for j, feature in enumerate(
            features
        ):

            # Same underlying white innovations are reused across filters.
            rng = np.random.default_rng(
                stable_seed(
                    "bandwidth_shared_white",
                    dataset,
                    group,
                    feature,
                    perturbation_seed,
                )
            )

            white = rng.normal(
                size=len(idx)
            )

            filtered = causal_fir(
                white,
                taps,
            )

            rms = float(
                np.sqrt(
                    np.mean(
                        filtered ** 2
                    )
                )
            )

            assert rms > 1e-12

            filtered = (
                float(target_rms)
                * filtered
                / rms
            )

            out[
                idx,
                j,
            ] += filtered


    out32 = out.astype(
        np.float32
    )

    delta = (
        out32.astype(
            np.float64
        )
        - np.asarray(
            base_x,
            dtype=np.float32,
        ).astype(
            np.float64
        )
    )


    max_rms_error = 0.0


    for row in group_records(
        frame
    ):

        dataset = str(
            row.dataset
        )

        group = str(
            row.split_group
        )

        idx = np.flatnonzero(
            (dataset_arr == dataset)
            & (group_arr == group)
        )

        rms = np.sqrt(
            np.mean(
                delta[
                    idx,
                    :
                ] ** 2,
                axis=0,
            )
        )

        max_rms_error = max(
            max_rms_error,
            float(
                np.max(
                    np.abs(
                        rms
                        - float(
                            target_rms
                        )
                    )
                )
            ),
        )


    assert max_rms_error <= EXACT_RMS_TOL

    return (
        out32,
        max_rms_error,
    )


def make_drift_input(
    base_x,
    frame,
    features,
    terminal_shift,
    perturbation_seed,
):
    out = np.asarray(
        base_x,
        dtype=np.float64,
    ).copy()

    dataset_arr = (
        frame[
            "dataset"
        ]
        .astype(str)
        .to_numpy()
    )

    group_arr = (
        frame[
            "split_group"
        ]
        .astype(str)
        .to_numpy()
    )

    expected_terminal = {}


    for row in group_records(
        frame
    ):

        dataset = str(
            row.dataset
        )

        group = str(
            row.split_group
        )

        idx = np.flatnonzero(
            (dataset_arr == dataset)
            & (group_arr == group)
        )

        # Frozen drift definition needs distinct first and final observations.
        assert len(idx) >= 2


        ramp = np.linspace(
            0.0,
            1.0,
            len(idx),
            dtype=np.float64,
        )


        for j, feature in enumerate(
            features
        ):

            integer = stable_seed(
                "drift_sign",
                dataset,
                group,
                feature,
                perturbation_seed,
            )

            sign = (
                1.0
                if integer % 2 == 0
                else -1.0
            )

            signed_terminal = (
                sign
                * float(
                    terminal_shift
                )
            )

            expected_terminal[
                (
                    dataset,
                    group,
                    j,
                )
            ] = signed_terminal


            out[
                idx,
                j,
            ] += (
                ramp
                * signed_terminal
            )


    out32 = out.astype(
        np.float32
    )

    base32 = np.asarray(
        base_x,
        dtype=np.float32,
    )

    delta = (
        out32.astype(
            np.float64
        )
        - base32.astype(
            np.float64
        )
    )


    max_terminal_error = 0.0


    for row in group_records(
        frame
    ):

        dataset = str(
            row.dataset
        )

        group = str(
            row.split_group
        )

        idx = np.flatnonzero(
            (dataset_arr == dataset)
            & (group_arr == group)
        )

        # Exact zero at first observation.
        assert np.max(
            np.abs(
                delta[
                    idx[0],
                    :
                ]
            )
        ) <= DRIFT_TERMINAL_TOL


        for j in range(
            len(features)
        ):

            expected = expected_terminal[
                (
                    dataset,
                    group,
                    j,
                )
            ]

            err = abs(
                float(
                    delta[
                        idx[-1],
                        j,
                    ]
                )
                - expected
            )

            max_terminal_error = max(
                max_terminal_error,
                err,
            )


    assert (
        max_terminal_error
        <= DRIFT_TERMINAL_TOL
    )

    return (
        out32,
        max_terminal_error,
    )


def perturbation_spectral_diagnostics(
    base_x,
    perturbed_x,
    frame,
):
    base = np.asarray(
        base_x,
        dtype=np.float32,
    ).astype(
        np.float64
    )

    perturbed = np.asarray(
        perturbed_x,
        dtype=np.float32,
    ).astype(
        np.float64
    )

    delta = (
        perturbed
        - base
    )

    assert np.isfinite(
        delta
    ).all()


    dataset_arr = (
        frame[
            "dataset"
        ]
        .astype(str)
        .to_numpy()
    )

    group_arr = (
        frame[
            "split_group"
        ]
        .astype(str)
        .to_numpy()
    )


    total_energy = 0.0
    centroid_numerator = 0.0

    band_energy = {
        name: 0.0
        for name, _, _, _ in BANDS
    }

    max_parseval_rel = 0.0


    for row in group_records(
        frame
    ):

        dataset = str(
            row.dataset
        )

        group = str(
            row.split_group
        )

        idx = np.flatnonzero(
            (dataset_arr == dataset)
            & (group_arr == group)
        )

        d = delta[
            idx,
            :
        ]

        coeff = dct(
            d,
            type=2,
            axis=0,
            norm="ortho",
        )

        direct = float(
            np.sum(
                d ** 2
            )
        )

        spectral = float(
            np.sum(
                coeff ** 2
            )
        )

        denom = max(
            direct,
            1e-30,
        )

        rel = abs(
            direct
            - spectral
        ) / denom

        max_parseval_rel = max(
            max_parseval_rel,
            rel,
        )

        assert np.isclose(
            direct,
            spectral,
            rtol=PARSEVAL_RTOL,
            atol=1e-12,
        )


        mode_energy = np.sum(
            coeff ** 2,
            axis=1,
        )

        omega = (
            np.arange(
                len(idx),
                dtype=np.float64,
            )
            / float(
                len(idx)
            )
        )


        total_energy += float(
            mode_energy.sum()
        )

        centroid_numerator += float(
            np.sum(
                omega
                * mode_energy
            )
        )


        for name, lo, hi, right_closed in BANDS:

            if right_closed:
                mask = (
                    (omega >= lo)
                    & (omega <= hi)
                )

            else:
                mask = (
                    (omega >= lo)
                    & (omega < hi)
                )

            band_energy[
                name
            ] += float(
                mode_energy[
                    mask
                ].sum()
            )


    assert total_energy > 0.0

    fractions = {
        name:
            band_energy[
                name
            ]
            / total_energy
        for name in band_energy
    }

    assert np.isclose(
        sum(
            fractions.values()
        ),
        1.0,
        rtol=1e-10,
        atol=1e-12,
    )


    return {
        "global_rms":
            float(
                np.sqrt(
                    np.mean(
                        delta ** 2
                    )
                )
            ),

        "spectral_centroid_omega_over_pi":
            float(
                centroid_numerator
                / total_energy
            ),

        "low_energy_fraction":
            fractions[
                "low"
            ],

        "mid_low_energy_fraction":
            fractions[
                "mid_low"
            ],

        "mid_high_energy_fraction":
            fractions[
                "mid_high"
            ],

        "high_energy_fraction":
            fractions[
                "high"
            ],

        "parseval_max_relative_error":
            max_parseval_rel,
    }


def evaluate_model(
    bench,
    model_name,
    model,
    x_t,
    y_t,
    prev_t,
    weight_t,
    frame,
):
    with torch.no_grad():

        if model_name == "MLP":

            pred = model(
                x_t
            )

        else:

            pred = model(
                x_t,
                prev_t,
            )


    assert pred.shape == y_t.shape

    assert torch.isfinite(
        pred
    ).all()


    balanced = bench.weighted_metrics(
        pred,
        y_t,
        weight_t,
    )

    per_dataset = bench.dataset_metrics(
        pred,
        y_t,
        frame,
    )


    result = {
        "BALANCED_ALL": {
            "n":
                int(
                    len(frame)
                ),

            "rmse":
                float(
                    balanced[
                        "rmse"
                    ]
                ),

            "mae":
                float(
                    balanced[
                        "mae"
                    ]
                ),
        }
    }


    for row in per_dataset:

        result[
            str(
                row[
                    "dataset"
                ]
            )
        ] = {
            "n":
                int(
                    row[
                        "n"
                    ]
                ),

            "rmse":
                float(
                    row[
                        "RMSE"
                    ]
                ),

            "mae":
                float(
                    row[
                        "MAE"
                    ]
                ),
        }


    assert set(
        result
    ) == set(
        DOMAINS
    )

    return result


# =============================================================================
# Frozen protocol guards
# =============================================================================

assert sha(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

assert sha(
    CLEAN_MANIFEST
) == EXPECTED_CLEAN_SHA


protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)


assert protocol[
    "status"
] == "FROZEN_BEFORE_SPECTRAL_RUNS"

assert protocol[
    "ar1_correlation_sweep"
][
    "rho"
] == AR1_RHO

assert protocol[
    "replication"
][
    "perturbation_seeds"
] == PERTURBATION_SEEDS

assert protocol[
    "controlled_drift_sweep"
][
    "terminal_shift_standardized_units"
] == DRIFT_LEVELS

assert set(
    protocol[
        "bandwidth_sweep"
    ][
        "causal_FIR_filters"
    ]
) == set(
    FILTER_ORDER
)

assert protocol[
    "final_v3_execution_manifest"
][
    "sha256"
] == EXPECTED_FINAL_V3_EXEC_SHA


ar1_sigma = float(
    protocol[
        "ar1_correlation_sweep"
    ][
        "stationary_marginal_std_standardized_units"
    ]
)

bandwidth_target_rms = float(
    protocol[
        "bandwidth_sweep"
    ][
        "post_filter_rms_standardized_units"
    ]
)

filter_taps = protocol[
    "bandwidth_sweep"
][
    "causal_FIR_filters"
]


assert math.isclose(
    ar1_sigma,
    0.10,
    rel_tol=0.0,
    abs_tol=1e-15,
)

assert math.isclose(
    bandwidth_target_rms,
    0.10,
    rel_tol=0.0,
    abs_tol=1e-15,
)


# Runner must already be committed.
rel = str(
    THIS.relative_to(
        REPO
    )
)

assert subprocess.run(
    [
        "git",
        "ls-files",
        "--error-unmatch",
        rel,
    ],
    cwd=REPO,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
).returncode == 0

assert subprocess.run(
    [
        "git",
        "diff",
        "--quiet",
        "HEAD",
        "--",
        rel,
    ],
    cwd=REPO,
).returncode == 0


script_commit = subprocess.check_output(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
    text=True,
).strip()

script_sha = sha(
    THIS
)


# =============================================================================
# Exact frozen benchmark implementation
# =============================================================================

spec = importlib.util.spec_from_file_location(
    "benchmark24_cbd",
    BENCH,
)

bench = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    bench
)


assert len(
    bench.FEATURES
) == 20

assert list(
    bench.DATASETS
) == [
    "HUST",
    "CALCE",
    "MATR",
]


RUNS.mkdir(
    parents=True,
    exist_ok=True,
)


completed_meta = []
diagnostic_rows = []

global_max_bandwidth_rms_error = 0.0
global_max_drift_terminal_error = 0.0
global_max_parseval_error = 0.0


# =============================================================================
# Five frozen split units
# =============================================================================

for split_seed in range(5):

    full_df = bench.load_canonical_soh(
        split_seed
    )

    test = bench.make_split_tensor(
        full_df,
        "test",
    )

    frame = (
        test[
            "frame"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    base_x = (
        test[
            "x"
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    y_t = test[
        "y"
    ]

    prev_t = test[
        "prev"
    ]

    weight_t = test[
        "weight"
    ]


    conditions = [{
        "condition_index":
            0,

        "condition_id":
            "clean",

        "scenario":
            "clean",

        "rho":
            None,

        "filter_name":
            None,

        "terminal_shift_std_units":
            None,

        "perturbation_seed":
            None,

        "input_tensor_sha256":
            sha_bytes(
                np.ascontiguousarray(
                    base_x
                ).tobytes()
            ),

        "x":
            torch.tensor(
                base_x,
                dtype=torch.float32,
            ),
    }]


    condition_index = 1


    # -------------------------------------------------------------------------
    # AR(1): shared innovations across rho values
    # -------------------------------------------------------------------------

    for rho in AR1_RHO:

        for perturbation_seed in PERTURBATION_SEEDS:

            x32, empirical_rms = make_ar1_input(
                base_x,
                frame,
                bench.FEATURES,
                rho,
                perturbation_seed,
                ar1_sigma,
            )

            diag = perturbation_spectral_diagnostics(
                base_x,
                x32,
                frame,
            )

            global_max_parseval_error = max(
                global_max_parseval_error,
                diag[
                    "parseval_max_relative_error"
                ],
            )

            cid = (
                f"ar1_rho_{rho:+.2f}_"
                f"seed{perturbation_seed}"
            )


            diagnostic_rows.append({
                "split_seed":
                    split_seed,

                "condition_index":
                    condition_index,

                "condition_id":
                    cid,

                "scenario":
                    "ar1_correlation",

                "rho":
                    float(
                        rho
                    ),

                "filter_name":
                    None,

                "terminal_shift_std_units":
                    None,

                "perturbation_seed":
                    perturbation_seed,

                "target_parameter_std_units":
                    ar1_sigma,

                "construction_error":
                    None,

                **diag,
            })


            conditions.append({
                "condition_index":
                    condition_index,

                "condition_id":
                    cid,

                "scenario":
                    "ar1_correlation",

                "rho":
                    float(
                        rho
                    ),

                "filter_name":
                    None,

                "terminal_shift_std_units":
                    None,

                "perturbation_seed":
                    perturbation_seed,

                "input_tensor_sha256":
                    sha_bytes(
                        np.ascontiguousarray(
                            x32
                        ).tobytes()
                    ),

                "x":
                    torch.tensor(
                        x32,
                        dtype=torch.float32,
                    ),
            })

            condition_index += 1


    # -------------------------------------------------------------------------
    # Bandwidth: same underlying white innovations across all filters.
    # -------------------------------------------------------------------------

    for filter_name in FILTER_ORDER:

        taps = filter_taps[
            filter_name
        ]

        for perturbation_seed in PERTURBATION_SEEDS:

            x32, rms_error = make_bandwidth_input(
                base_x,
                frame,
                bench.FEATURES,
                filter_name,
                taps,
                perturbation_seed,
                bandwidth_target_rms,
            )

            global_max_bandwidth_rms_error = max(
                global_max_bandwidth_rms_error,
                rms_error,
            )

            diag = perturbation_spectral_diagnostics(
                base_x,
                x32,
                frame,
            )

            global_max_parseval_error = max(
                global_max_parseval_error,
                diag[
                    "parseval_max_relative_error"
                ],
            )

            cid = (
                f"bandwidth_{filter_name}_"
                f"seed{perturbation_seed}"
            )


            diagnostic_rows.append({
                "split_seed":
                    split_seed,

                "condition_index":
                    condition_index,

                "condition_id":
                    cid,

                "scenario":
                    "bandwidth",

                "rho":
                    None,

                "filter_name":
                    filter_name,

                "terminal_shift_std_units":
                    None,

                "perturbation_seed":
                    perturbation_seed,

                "target_parameter_std_units":
                    bandwidth_target_rms,

                "construction_error":
                    rms_error,

                **diag,
            })


            conditions.append({
                "condition_index":
                    condition_index,

                "condition_id":
                    cid,

                "scenario":
                    "bandwidth",

                "rho":
                    None,

                "filter_name":
                    filter_name,

                "terminal_shift_std_units":
                    None,

                "perturbation_seed":
                    perturbation_seed,

                "input_tensor_sha256":
                    sha_bytes(
                        np.ascontiguousarray(
                            x32
                        ).tobytes()
                    ),

                "x":
                    torch.tensor(
                        x32,
                        dtype=torch.float32,
                    ),
            })

            condition_index += 1


    # -------------------------------------------------------------------------
    # Controlled drift: same signs across magnitude levels.
    # -------------------------------------------------------------------------

    for terminal_shift in DRIFT_LEVELS:

        for perturbation_seed in PERTURBATION_SEEDS:

            x32, terminal_error = make_drift_input(
                base_x,
                frame,
                bench.FEATURES,
                terminal_shift,
                perturbation_seed,
            )

            global_max_drift_terminal_error = max(
                global_max_drift_terminal_error,
                terminal_error,
            )

            diag = perturbation_spectral_diagnostics(
                base_x,
                x32,
                frame,
            )

            global_max_parseval_error = max(
                global_max_parseval_error,
                diag[
                    "parseval_max_relative_error"
                ],
            )

            cid = (
                f"drift_terminal_{terminal_shift:.2f}_"
                f"seed{perturbation_seed}"
            )


            diagnostic_rows.append({
                "split_seed":
                    split_seed,

                "condition_index":
                    condition_index,

                "condition_id":
                    cid,

                "scenario":
                    "controlled_drift",

                "rho":
                    None,

                "filter_name":
                    None,

                "terminal_shift_std_units":
                    float(
                        terminal_shift
                    ),

                "perturbation_seed":
                    perturbation_seed,

                "target_parameter_std_units":
                    float(
                        terminal_shift
                    ),

                "construction_error":
                    terminal_error,

                **diag,
            })


            conditions.append({
                "condition_index":
                    condition_index,

                "condition_id":
                    cid,

                "scenario":
                    "controlled_drift",

                "rho":
                    None,

                "filter_name":
                    None,

                "terminal_shift_std_units":
                    float(
                        terminal_shift
                    ),

                "perturbation_seed":
                    perturbation_seed,

                "input_tensor_sha256":
                    sha_bytes(
                        np.ascontiguousarray(
                            x32
                        ).tobytes()
                    ),

                "x":
                    torch.tensor(
                        x32,
                        dtype=torch.float32,
                    ),
            })

            condition_index += 1


    assert len(
        conditions
    ) == 66

    assert (
        condition_index
        == 66
    )


    pairing_hashes = {
        c[
            "condition_id"
        ]:
            c[
                "input_tensor_sha256"
            ]
        for c in conditions
    }

    assert len(
        pairing_hashes
    ) == 66


    generated = 0


    # -------------------------------------------------------------------------
    # 3 initializations x 2 frozen models
    # -------------------------------------------------------------------------

    for train_seed in range(3):

        run_dir = (
            FINAL_V3
            / f"split{split_seed}_train{train_seed}"
        )

        old_metrics = pd.read_csv(
            run_dir / "metrics.csv",
            low_memory=False,
        )

        old_metrics = old_metrics[
            old_metrics[
                "split"
            ].astype(str).str.lower()
            == "test"
        ].copy()

        assert len(
            old_metrics
        ) == 8

        old_metrics[
            "_model"
        ] = old_metrics[
            "model"
        ].map(
            norm_model_name
        )


        for model_name in MODELS:

            stem = (
                f"split{split_seed}_"
                f"train{train_seed}_"
                f"{model_name.lower()}"
            )

            metric_path = (
                RUNS
                / f"{stem}_cbd_metrics.csv"
            )

            meta_path = (
                RUNS
                / f"{stem}_evaluation.json"
            )


            # Reuse only an atomically closed pair.
            if (
                metric_path.exists()
                and meta_path.exists()
            ):

                meta = json.loads(
                    meta_path.read_text(
                        encoding="utf-8"
                    )
                )

                assert meta[
                    "contract"
                ] == "correlation_bandwidth_drift_evaluation_run_v1"

                assert meta[
                    "status"
                ] == "COMPLETE"

                assert meta[
                    "metric_file_sha256"
                ] == sha(
                    metric_path
                )

                assert meta[
                    "metric_rows"
                ] == 264

                assert csv_data_rows(
                    metric_path
                ) == 264

                assert meta[
                    "input_tensor_sha256_by_condition"
                ] == pairing_hashes

                completed_meta.append(
                    meta
                )

                continue


            if model_name == "MLP":

                model = bench.FeatureMatchedMLP(
                    input_dim=20,
                    hidden_dim=64,
                )

                state_path = (
                    run_dir
                    / "mlp_best_state.pt"
                )

            else:

                model = bench.FeatureMatchedCausalGCN(
                    input_dim=20,
                    hidden_dim=64,
                )

                state_path = (
                    run_dir
                    / "causal_gcn_best_state.pt"
                )


            model.load_state_dict(
                load_state(
                    state_path
                ),
                strict=True,
            )

            model.eval()


            parameter_count = int(
                sum(
                    p.numel()
                    for p in model.parameters()
                )
            )

            assert parameter_count == 5569


            rows = []

            clean_pass = True
            max_clean_rmse_diff = 0.0
            max_clean_mae_diff = 0.0


            for condition in conditions:

                current = evaluate_model(
                    bench,
                    model_name,
                    model,
                    condition[
                        "x"
                    ],
                    y_t,
                    prev_t,
                    weight_t,
                    frame,
                )


                for domain in DOMAINS:

                    vals = current[
                        domain
                    ]

                    rows.append({
                        "split_seed":
                            split_seed,

                        "train_seed":
                            train_seed,

                        "model":
                            model_name,

                        "condition_index":
                            condition[
                                "condition_index"
                            ],

                        "condition_id":
                            condition[
                                "condition_id"
                            ],

                        "scenario":
                            condition[
                                "scenario"
                            ],

                        "rho":
                            condition[
                                "rho"
                            ],

                        "filter_name":
                            condition[
                                "filter_name"
                            ],

                        "terminal_shift_std_units":
                            condition[
                                "terminal_shift_std_units"
                            ],

                        "perturbation_seed":
                            condition[
                                "perturbation_seed"
                            ],

                        "domain":
                            domain,

                        "n_rows":
                            vals[
                                "n"
                            ],

                        "rmse":
                            vals[
                                "rmse"
                            ],

                        "mae":
                            vals[
                                "mae"
                            ],
                    })


                if condition[
                    "scenario"
                ] == "clean":

                    for domain in DOMAINS:

                        old = old_metrics[
                            (
                                old_metrics[
                                    "_model"
                                ] == model_name
                            )
                            & (
                                old_metrics[
                                    "dataset"
                                ].astype(str)
                                == domain
                            )
                        ]

                        assert len(
                            old
                        ) == 1

                        old = old.iloc[
                            0
                        ]

                        rd = abs(
                            current[
                                domain
                            ][
                                "rmse"
                            ]
                            - float(
                                old[
                                    "RMSE"
                                ]
                            )
                        )

                        ad = abs(
                            current[
                                domain
                            ][
                                "mae"
                            ]
                            - float(
                                old[
                                    "MAE"
                                ]
                            )
                        )

                        n_match = (
                            int(
                                current[
                                    domain
                                ][
                                    "n"
                                ]
                            )
                            == int(
                                old[
                                    "n"
                                ]
                            )
                        )

                        max_clean_rmse_diff = max(
                            max_clean_rmse_diff,
                            rd,
                        )

                        max_clean_mae_diff = max(
                            max_clean_mae_diff,
                            ad,
                        )

                        clean_pass = (
                            clean_pass
                            and n_match
                            and rd <= CLEAN_TOL
                            and ad <= CLEAN_TOL
                        )


            assert clean_pass

            result = pd.DataFrame(
                rows
            )

            assert len(
                result
            ) == 264

            assert (
                result[
                    "condition_id"
                ].nunique()
                == 66
            )

            assert (
                result[
                    "domain"
                ].nunique()
                == 4
            )


            # Safe even if a prior run left only one incomplete artifact:
            # atomic replacement is limited to this unclosed run namespace.
            atomic_csv(
                result,
                metric_path,
            )


            meta = {
                "contract":
                    "correlation_bandwidth_drift_evaluation_run_v1",

                "status":
                    "COMPLETE",

                "split_seed":
                    split_seed,

                "train_seed":
                    train_seed,

                "model":
                    model_name,

                "state_file":
                    str(
                        state_path
                    ),

                "state_file_sha256":
                    sha(
                        state_path
                    ),

                "parameter_count":
                    parameter_count,

                "spectral_protocol_sha256":
                    EXPECTED_PROTOCOL_SHA,

                "clean_graph_fourier_manifest_sha256":
                    EXPECTED_CLEAN_SHA,

                "runner_script_sha256":
                    script_sha,

                "runner_script_commit":
                    script_commit,

                "training_reexecuted":
                    False,

                "scaler_refit":
                    False,

                "feature_generator_reexecuted":
                    False,

                "GPU_used":
                    False,

                "condition_count":
                    66,

                "clean_condition_count":
                    1,

                "ar1_condition_count":
                    25,

                "bandwidth_condition_count":
                    25,

                "drift_condition_count":
                    15,

                "domains_per_condition":
                    4,

                "metric_rows":
                    264,

                "metric_values_surfaced":
                    False,

                "analysis_performed":
                    False,

                "clean_regression_pass":
                    clean_pass,

                "clean_max_abs_rmse_difference":
                    max_clean_rmse_diff,

                "clean_max_abs_mae_difference":
                    max_clean_mae_diff,

                "input_tensor_sha256_by_condition":
                    pairing_hashes,

                "metric_file":
                    str(
                        metric_path
                    ),

                "metric_file_sha256":
                    sha(
                        metric_path
                    ),
            }


            atomic_json(
                meta,
                meta_path,
            )

            completed_meta.append(
                meta
            )

            generated += 1


    print(
        f"split {split_seed}: complete "
        f"generated={generated}"
    )


# =============================================================================
# Perturbation spectral diagnostics closure
# =============================================================================

diagnostics = pd.DataFrame(
    diagnostic_rows
)

assert len(
    diagnostics
) == 325

assert (
    diagnostics[
        "condition_id"
    ].nunique()
    == 65
)

assert set(
    diagnostics[
        "scenario"
    ]
) == {
    "ar1_correlation",
    "bandwidth",
    "controlled_drift",
}


diagnostics = diagnostics.sort_values(
    [
        "split_seed",
        "condition_index",
    ],
    kind="mergesort",
).reset_index(
    drop=True
)


atomic_csv(
    diagnostics,
    PERTURBATION_DIAGNOSTICS,
)


# =============================================================================
# Execution closure
# =============================================================================

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

assert len(
    completed_meta
) == 30


keys = {
    (
        int(
            m[
                "split_seed"
            ]
        ),
        int(
            m[
                "train_seed"
            ]
        ),
        str(
            m[
                "model"
            ]
        ),
    )
    for m in completed_meta
}

assert len(
    keys
) == 30


# Paired tensors across model and initialization within each split.
for split_seed in range(5):

    group = [
        m
        for m in completed_meta
        if int(
            m[
                "split_seed"
            ]
        ) == split_seed
    ]

    assert len(
        group
    ) == 6

    reference = group[
        0
    ][
        "input_tensor_sha256_by_condition"
    ]

    assert len(
        reference
    ) == 66

    for m in group[
        1:
    ]:

        assert (
            m[
                "input_tensor_sha256_by_condition"
            ]
            == reference
        )


assert all(
    m[
        "training_reexecuted"
    ] is False
    for m in completed_meta
)

assert all(
    m[
        "scaler_refit"
    ] is False
    for m in completed_meta
)

assert all(
    m[
        "metric_values_surfaced"
    ] is False
    for m in completed_meta
)

assert all(
    m[
        "analysis_performed"
    ] is False
    for m in completed_meta
)

assert all(
    m[
        "clean_regression_pass"
    ] is True
    for m in completed_meta
)


max_clean_rmse = max(
    float(
        m[
            "clean_max_abs_rmse_difference"
        ]
    )
    for m in completed_meta
)

max_clean_mae = max(
    float(
        m[
            "clean_max_abs_mae_difference"
        ]
    )
    for m in completed_meta
)


manifest = {
    "contract":
        "final_correlation_bandwidth_drift_execution_v1",

    "status":
        "COMPLETE_BEFORE_VALUE_LEVEL_ANALYSIS",

    "spectral_protocol_sha256":
        EXPECTED_PROTOCOL_SHA,

    "clean_graph_fourier_manifest_sha256":
        EXPECTED_CLEAN_SHA,

    "final_v3_execution_manifest_sha256":
        EXPECTED_FINAL_V3_EXEC_SHA,

    "runner_script_sha256":
        script_sha,

    "runner_script_commit":
        script_commit,

    "completed_runs":
        30,

    "split_count":
        5,

    "train_initializations_per_split":
        3,

    "perturbation_seeds":
        PERTURBATION_SEEDS,

    "condition_contract": {
        "clean":
            1,

        "ar1":
            25,

        "bandwidth":
            25,

        "controlled_drift":
            15,

        "total":
            66,
    },

    "ar1_rho":
        AR1_RHO,

    "ar1_stationary_marginal_std_standardized_units":
        ar1_sigma,

    "ar1_pairing":
        (
            "same underlying Gaussian innovations reused "
            "across rho values for matched mechanism comparison"
        ),

    "bandwidth_filters":
        FILTER_ORDER,

    "bandwidth_post_filter_rms_standardized_units":
        bandwidth_target_rms,

    "bandwidth_pairing":
        (
            "same underlying white innovations reused "
            "across FIR filters"
        ),

    "drift_terminal_shift_standardized_units":
        DRIFT_LEVELS,

    "drift_pairing":
        (
            "same deterministic signs reused "
            "across drift magnitude levels"
        ),

    "conditions_per_run":
        66,

    "domains_per_condition":
        4,

    "metric_rows_per_run":
        264,

    "expected_total_metric_rows":
        7920,

    "paired_realization_across_models":
        True,

    "paired_realization_across_train_initializations":
        True,

    "training_reexecuted":
        False,

    "scaler_refit":
        False,

    "feature_generator_reexecuted":
        False,

    "GPU_used":
        False,

    "clean_final_v3_metric_regression_barrier":
        "PASS",

    "clean_metric_tolerance":
        CLEAN_TOL,

    "clean_max_abs_rmse_difference":
        max_clean_rmse,

    "clean_max_abs_mae_difference":
        max_clean_mae,

    "bandwidth_max_abs_cell_feature_rms_error":
        global_max_bandwidth_rms_error,

    "drift_max_abs_terminal_shift_error":
        global_max_drift_terminal_error,

    "perturbation_parseval_max_relative_error":
        global_max_parseval_error,

    "perturbation_spectral_diagnostics": {
        "path":
            str(
                PERTURBATION_DIAGNOSTICS
            ),

        "sha256":
            sha(
                PERTURBATION_DIAGNOSTICS
            ),

        "rows":
            len(
                diagnostics
            ),

        "model_performance_values":
            False,
    },

    "metric_values_surfaced_during_execution":
        False,

    "perturbation_diagnostic_values_surfaced_during_execution":
        False,

    "analysis_performed":
        False,

    "scientific_role":
        "descriptive mechanism checks only",

    "metric_files": {
        p.name: {
            "sha256":
                sha(p),

            "rows":
                csv_data_rows(
                    p
                ),
        }
        for p in metric_files
    },

    "evaluation_metadata_files": {
        p.name: {
            "sha256":
                sha(p),
        }
        for p in meta_files
    },
}


atomic_json(
    manifest,
    EXEC_MANIFEST,
)


print()
print("=" * 90)
print("CORRELATION / BANDWIDTH / DRIFT EXECUTION CLOSURE")
print("=" * 90)

print(
    "completed runs:",
    30
)

print(
    "conditions per run:",
    66
)

print(
    "AR1 conditions per run:",
    25
)

print(
    "bandwidth conditions per run:",
    25
)

print(
    "drift conditions per run:",
    15
)

print(
    "domains per condition:",
    4
)

print(
    "expected metric rows:",
    7920
)

print(
    "perturbation diagnostic rows:",
    325
)

print(
    "clean final-v3 regression barrier:",
    True
)

print(
    "paired perturbations across models:",
    True
)

print(
    "paired perturbations across initializations:",
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
    "metric values surfaced:",
    False
)

print(
    "perturbation diagnostic values surfaced:",
    False
)

print(
    "analysis performed:",
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
    "CORRELATION_BANDWIDTH_DRIFT_EXECUTION_PASS=True"
)
