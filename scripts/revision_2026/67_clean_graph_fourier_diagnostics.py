from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os
import re
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
    / "scripts/revision_2026/67_clean_graph_fourier_diagnostics.py"
)

ROOT = (
    BASE
    / "results/revision_2026/spectral_frequency/final_spectral_v1"
)

PROTOCOL = ROOT / "protocol.json"

FINAL_V3 = (
    BASE
    / "results/revision_2026/feature_matched_soh/final_v3"
)

OUT = ROOT / "clean_graph_fourier_v1"

CELL_TABLE = OUT / "path_laplacian_cells.csv"
MODE_TABLE = OUT / "path_laplacian_input_modes.csv"
INPUT_BANDS = OUT / "clean_input_feature_band_energy.csv"
RESIDUAL_BANDS = OUT / "clean_residual_band_energy.csv"
METRIC_CHECK = OUT / "clean_prediction_metric_regression.csv"
SUMMARY = OUT / "clean_graph_fourier_summary.csv"
MANIFEST = OUT / "clean_graph_fourier_manifest.json"


EXPECTED_PROTOCOL_SHA = (
    "a8d3b65b87980470f326717f0cb50e3d4fb4d2e0429c93171eb0d21d6c713e45"
)

EXPECTED_FINAL_V3_PROTOCOL_SHA = (
    "2dec14c210ae3f02e5f65e4acc24f9c4adc747bc071509cb144e69dba707f4dd"
)

EXPECTED_FINAL_V3_EXEC_SHA = (
    "5a8bfbe0f11a6158f2749a9da4f18f1e62adf8c839976c1bf0af9c9f6c07440d"
)

CLEAN_METRIC_TOL = 1e-7
PARSEVAL_RTOL = 1e-10
PARSEVAL_ATOL = 1e-12

SPLITS = [0, 1, 2, 3, 4]
TRAIN_SEEDS = [0, 1, 2]
MODELS = ["MLP", "GCN"]
DOMAINS = ["BALANCED_ALL", "HUST", "CALCE", "MATR"]

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


def atomic_csv(df, path):
    tmp = Path(str(path) + ".tmp")

    df.to_csv(
        tmp,
        index=False,
        float_format="%.17g",
    )

    os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = Path(str(path) + ".tmp")

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(tmp, path)


def band_name(x):
    x = float(x)

    for name, lo, hi, right_closed in BANDS:

        if right_closed:
            if lo <= x <= hi:
                return name

        else:
            if lo <= x < hi:
                return name

    raise AssertionError(
        ("frequency outside frozen bands", x)
    )


def norm_model_name(x):
    s = str(x).strip().upper()

    if "GCN" in s:
        return "GCN"

    if "MLP" in s:
        return "MLP"

    raise AssertionError(
        ("unknown model name", x)
    )


def load_state(path):
    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if isinstance(state, dict):

        if "model_state_dict" in state:
            state = state[
                "model_state_dict"
            ]

        elif "state_dict" in state:
            state = state[
                "state_dict"
            ]

    assert isinstance(state, dict)

    return state


# ===========================================================================
# PRE-OUTCOME GUARDS
# ===========================================================================

assert sha(PROTOCOL) == EXPECTED_PROTOCOL_SHA

protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)

assert protocol[
    "status"
] == "FROZEN_BEFORE_SPECTRAL_RUNS"

assert protocol[
    "final_v3_protocol"
][
    "sha256"
] == EXPECTED_FINAL_V3_PROTOCOL_SHA

assert protocol[
    "final_v3_execution_manifest"
][
    "sha256"
] == EXPECTED_FINAL_V3_EXEC_SHA

assert sha(
    Path(
        protocol[
            "final_v3_protocol"
        ][
            "path"
        ]
    )
) == EXPECTED_FINAL_V3_PROTOCOL_SHA

assert sha(
    Path(
        protocol[
            "final_v3_execution_manifest"
        ][
            "path"
        ]
    )
) == EXPECTED_FINAL_V3_EXEC_SHA


# Script must already be committed and unchanged.
rel = str(
    THIS.relative_to(REPO)
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


if OUT.exists():
    assert not any(
        p.is_file()
        for p in OUT.rglob("*")
    )


script_commit = subprocess.check_output(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
    text=True,
).strip()

script_sha = sha(THIS)


# ===========================================================================
# IMPORT EXACT FROZEN BENCHMARK IMPLEMENTATION
# ===========================================================================

spec = importlib.util.spec_from_file_location(
    "benchmark24_spectral",
    BENCH,
)

bench = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    bench
)


assert list(
    bench.FEATURES
) == [
    "v_min",
    "v_q10",
    "v_q25",
    "v_median",
    "v_q75",
    "v_q90",
    "v_max",
    "v_mean",
    "v_std",
    "v_phase_slope",
    "i_min",
    "i_q10",
    "i_q25",
    "i_median",
    "i_q75",
    "i_q90",
    "i_max",
    "i_mean",
    "i_std",
    "i_phase_slope",
]

assert list(
    bench.DATASETS
) == [
    "HUST",
    "CALCE",
    "MATR",
]


# ===========================================================================
# COLLECT CLEAN INPUT AND RESIDUAL SPECTRA
# ===========================================================================

cell_rows = []
mode_rows = []
input_band_rows = []
residual_band_rows = []
metric_rows = []

max_input_parseval_rel = 0.0
max_residual_parseval_rel = 0.0

state_hashes = {}


for split_seed in SPLITS:

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

    x = (
        test[
            "x"
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
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


    assert len(frame) == len(x)
    assert x.shape[1] == 20

    assert np.isfinite(x).all()

    groups = (
        frame[
            "split_group"
        ]
        .astype(str)
        .to_numpy()
    )

    datasets = (
        frame[
            "dataset"
        ]
        .astype(str)
        .to_numpy()
    )

    observation_index = pd.to_numeric(
        frame[
            "observation_index"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.int64
    )


    unique_groups = (
        frame[
            [
                "dataset",
                "split_group",
            ]
        ]
        .drop_duplicates()
    )


    # -----------------------------------------------------------------------
    # Exact path-Laplacian modes + clean standardized input spectra.
    # -----------------------------------------------------------------------

    for row in unique_groups.itertuples(
        index=False
    ):

        dataset = str(
            row.dataset
        )

        group = str(
            row.split_group
        )

        idx = np.flatnonzero(
            groups == group
        )

        assert len(idx) > 0
        assert np.all(
            datasets[
                idx
            ] == dataset
        )

        obs = observation_index[
            idx
        ]

        if len(obs) > 1:
            assert np.all(
                np.diff(obs) == 1
            )

        cell_x = x[
            idx,
            :
        ]

        n = int(
            len(idx)
        )

        k = np.arange(
            n,
            dtype=np.float64,
        )

        omega_over_pi = (
            k / float(n)
        )

        eigenvalues = (
            2.0
            - 2.0
            * np.cos(
                np.pi
                * k
                / float(n)
            )
        )

        coeff = dct(
            cell_x,
            type=2,
            axis=0,
            norm="ortho",
        )

        original_energy = float(
            np.sum(
                cell_x ** 2
            )
        )

        spectral_energy = float(
            np.sum(
                coeff ** 2
            )
        )

        denom = max(
            abs(
                original_energy
            ),
            1e-30,
        )

        rel = abs(
            original_energy
            - spectral_energy
        ) / denom

        max_input_parseval_rel = max(
            max_input_parseval_rel,
            rel,
        )

        assert np.isclose(
            original_energy,
            spectral_energy,
            rtol=PARSEVAL_RTOL,
            atol=PARSEVAL_ATOL,
        )

        mode_energy = np.sum(
            coeff ** 2,
            axis=1,
        )

        assert np.isclose(
            float(
                mode_energy.sum()
            ),
            spectral_energy,
            rtol=PARSEVAL_RTOL,
            atol=PARSEVAL_ATOL,
        )

        assert spectral_energy > 0.0

        bands = [
            band_name(v)
            for v in omega_over_pi
        ]

        lambda_1 = (
            float(
                eigenvalues[1]
            )
            if n > 1
            else float("nan")
        )

        cell_rows.append({
            "split_seed":
                split_seed,

            "dataset":
                dataset,

            "split_group":
                group,

            "n":
                n,

            "lambda_0":
                float(
                    eigenvalues[0]
                ),

            "lambda_1":
                lambda_1,

            "lambda_max":
                float(
                    eigenvalues[-1]
                ),

            "omega_min_over_pi":
                float(
                    omega_over_pi[0]
                ),

            "omega_max_over_pi":
                float(
                    omega_over_pi[-1]
                ),

            "input_total_energy":
                spectral_energy,

            "input_dc_energy_fraction":
                float(
                    mode_energy[0]
                    / spectral_energy
                ),
        })


        for ii in range(n):

            mode_rows.append({
                "split_seed":
                    split_seed,

                "dataset":
                    dataset,

                "split_group":
                    group,

                "n":
                    n,

                "k":
                    int(ii),

                "omega_over_pi":
                    float(
                        omega_over_pi[ii]
                    ),

                "eigenvalue":
                    float(
                        eigenvalues[ii]
                    ),

                "band":
                    bands[ii],

                "input_energy":
                    float(
                        mode_energy[ii]
                    ),

                "input_energy_fraction":
                    float(
                        mode_energy[ii]
                        / spectral_energy
                    ),
            })


        for feature_index, feature in enumerate(
            bench.FEATURES
        ):

            feature_energy = (
                coeff[
                    :,
                    feature_index,
                ] ** 2
            )

            feature_total = float(
                feature_energy.sum()
            )

            assert feature_total > 0.0

            for band, _, _, _ in BANDS:

                mask = np.asarray(
                    [
                        b == band
                        for b in bands
                    ],
                    dtype=bool,
                )

                e = float(
                    feature_energy[
                        mask
                    ].sum()
                )

                input_band_rows.append({
                    "split_seed":
                        split_seed,

                    "dataset":
                        dataset,

                    "split_group":
                        group,

                    "feature":
                        feature,

                    "band":
                        band,

                    "energy":
                        e,

                    "feature_total_energy":
                        feature_total,

                    "energy_fraction":
                        e
                        / feature_total,
                })


    # -----------------------------------------------------------------------
    # Frozen model predictions + exact final-v3 metric regression barrier.
    # -----------------------------------------------------------------------

    for train_seed in TRAIN_SEEDS:

        run_dir = (
            FINAL_V3
            / f"split{split_seed}_train{train_seed}"
        )

        old_metrics = pd.read_csv(
            run_dir / "metrics.csv"
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


            state_hashes[
                str(
                    state_path.relative_to(
                        FINAL_V3
                    )
                )
            ] = sha(
                state_path
            )


            model.load_state_dict(
                load_state(
                    state_path
                ),
                strict=True,
            )

            model.eval()


            with torch.no_grad():

                if model_name == "MLP":

                    pred_t = model(
                        test["x"]
                    )

                else:

                    pred_t = model(
                        test["x"],
                        prev_t,
                    )


            assert pred_t.shape == y_t.shape

            assert torch.isfinite(
                pred_t
            ).all()


            # Exact benchmark metric functions.
            balanced = bench.weighted_metrics(
                pred_t,
                y_t,
                weight_t,
            )

            per_dataset = bench.dataset_metrics(
                pred_t,
                y_t,
                frame,
            )


            current = {
                "BALANCED_ALL": {
                    "n":
                        int(
                            len(frame)
                        ),

                    "RMSE":
                        float(
                            balanced[
                                "rmse"
                            ]
                        ),

                    "MAE":
                        float(
                            balanced[
                                "mae"
                            ]
                        ),
                }
            }


            for drow in per_dataset:

                current[
                    str(
                        drow[
                            "dataset"
                        ]
                    )
                ] = {
                    "n":
                        int(
                            drow["n"]
                        ),

                    "RMSE":
                        float(
                            drow["RMSE"]
                        ),

                    "MAE":
                        float(
                            drow["MAE"]
                        ),
                }


            assert set(
                current
            ) == set(
                DOMAINS
            )


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

                assert len(old) == 1

                old = old.iloc[0]

                rmse_diff = (
                    current[
                        domain
                    ][
                        "RMSE"
                    ]
                    - float(
                        old["RMSE"]
                    )
                )

                mae_diff = (
                    current[
                        domain
                    ][
                        "MAE"
                    ]
                    - float(
                        old["MAE"]
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
                        old["n"]
                    )
                )

                passed = (
                    n_match
                    and abs(
                        rmse_diff
                    ) <= CLEAN_METRIC_TOL
                    and abs(
                        mae_diff
                    ) <= CLEAN_METRIC_TOL
                )

                metric_rows.append({
                    "split_seed":
                        split_seed,

                    "train_seed":
                        train_seed,

                    "model":
                        model_name,

                    "domain":
                        domain,

                    "n_match":
                        n_match,

                    "rmse_difference":
                        rmse_diff,

                    "mae_difference":
                        mae_diff,

                    "pass":
                        passed,
                })


            # ----------------------------------------------------------------
            # Residual graph-Fourier energy.
            # ----------------------------------------------------------------

            residual = (
                pred_t.detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64
                )
                .reshape(-1)
                -
                y_t.detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64
                )
                .reshape(-1)
            )


            for row in unique_groups.itertuples(
                index=False
            ):

                dataset = str(
                    row.dataset
                )

                group = str(
                    row.split_group
                )

                idx = np.flatnonzero(
                    groups == group
                )

                r = residual[
                    idx
                ]

                n = len(r)

                coeff_r = dct(
                    r,
                    type=2,
                    norm="ortho",
                )

                original_energy = float(
                    np.sum(
                        r ** 2
                    )
                )

                spectral_energy = float(
                    np.sum(
                        coeff_r ** 2
                    )
                )

                denom = max(
                    abs(
                        original_energy
                    ),
                    1e-30,
                )

                rel = abs(
                    original_energy
                    - spectral_energy
                ) / denom

                max_residual_parseval_rel = max(
                    max_residual_parseval_rel,
                    rel,
                )

                assert np.isclose(
                    original_energy,
                    spectral_energy,
                    rtol=PARSEVAL_RTOL,
                    atol=PARSEVAL_ATOL,
                )

                assert spectral_energy > 0.0

                omega = (
                    np.arange(
                        n,
                        dtype=np.float64,
                    )
                    / float(n)
                )

                bands = np.asarray(
                    [
                        band_name(v)
                        for v in omega
                    ],
                    dtype=object,
                )

                energy = (
                    coeff_r ** 2
                )


                for band, _, _, _ in BANDS:

                    mask = (
                        bands == band
                    )

                    e = float(
                        energy[
                            mask
                        ].sum()
                    )

                    residual_band_rows.append({
                        "split_seed":
                            split_seed,

                        "train_seed":
                            train_seed,

                        "model":
                            model_name,

                        "dataset":
                            dataset,

                        "split_group":
                            group,

                        "band":
                            band,

                        "energy":
                            e,

                        "residual_total_energy":
                            spectral_energy,

                        "energy_fraction":
                            e
                            / spectral_energy,
                    })


# ===========================================================================
# GLOBAL BARRIERS
# ===========================================================================

metric_check = pd.DataFrame(
    metric_rows
)

assert len(metric_check) == 120
assert metric_check[
    "pass"
].all()


max_rmse_diff = float(
    np.abs(
        metric_check[
            "rmse_difference"
        ]
    ).max()
)

max_mae_diff = float(
    np.abs(
        metric_check[
            "mae_difference"
        ]
    ).max()
)


cells = pd.DataFrame(
    cell_rows
)

modes = pd.DataFrame(
    mode_rows
)

input_bands = pd.DataFrame(
    input_band_rows
)

residual_bands = pd.DataFrame(
    residual_band_rows
)


assert len(cells) > 0

assert len(modes) == int(
    cells["n"].sum()
)

assert len(input_bands) == (
    len(cells)
    * 20
    * 4
)

assert len(residual_bands) == (
    len(cells)
    * 3
    * 2
    * 4
)


assert np.allclose(
    cells[
        "lambda_0"
    ].to_numpy(
        dtype=np.float64
    ),
    0.0,
    rtol=0.0,
    atol=1e-14,
)


assert (
    cells[
        "lambda_max"
    ] < 4.0
).all()


# Every cell's mode energy fractions sum to one.
mode_fraction_sum = (
    modes
    .groupby(
        [
            "split_seed",
            "dataset",
            "split_group",
        ],
        observed=True,
    )[
        "input_energy_fraction"
    ]
    .sum()
)

assert np.allclose(
    mode_fraction_sum.to_numpy(
        dtype=np.float64
    ),
    1.0,
    rtol=1e-10,
    atol=1e-12,
)


# Every feature/cell band decomposition sums to one.
input_fraction_sum = (
    input_bands
    .groupby(
        [
            "split_seed",
            "dataset",
            "split_group",
            "feature",
        ],
        observed=True,
    )[
        "energy_fraction"
    ]
    .sum()
)

assert np.allclose(
    input_fraction_sum.to_numpy(
        dtype=np.float64
    ),
    1.0,
    rtol=1e-10,
    atol=1e-12,
)


# Every residual/cell/run decomposition sums to one.
residual_fraction_sum = (
    residual_bands
    .groupby(
        [
            "split_seed",
            "train_seed",
            "model",
            "dataset",
            "split_group",
        ],
        observed=True,
    )[
        "energy_fraction"
    ]
    .sum()
)

assert np.allclose(
    residual_fraction_sum.to_numpy(
        dtype=np.float64
    ),
    1.0,
    rtol=1e-10,
    atol=1e-12,
)


# ===========================================================================
# DESCRIPTIVE SUMMARY
# ===========================================================================

input_pooled = (
    modes
    .groupby(
        [
            "dataset",
            "band",
        ],
        as_index=False,
        observed=True,
    )
    .agg(
        energy=(
            "input_energy",
            "sum",
        )
    )
)

input_pooled[
    "total"
] = input_pooled.groupby(
    "dataset",
    observed=True,
)[
    "energy"
].transform(
    "sum"
)

input_pooled[
    "fraction"
] = (
    input_pooled[
        "energy"
    ]
    / input_pooled[
        "total"
    ]
)

input_pooled[
    "quantity"
] = "clean_input_pooled_energy_fraction"

input_pooled[
    "model"
] = "NA"


residual_pooled = (
    residual_bands
    .groupby(
        [
            "model",
            "dataset",
            "band",
        ],
        as_index=False,
        observed=True,
    )
    .agg(
        energy=(
            "energy",
            "sum",
        )
    )
)

residual_pooled[
    "total"
] = residual_pooled.groupby(
    [
        "model",
        "dataset",
    ],
    observed=True,
)[
    "energy"
].transform(
    "sum"
)

residual_pooled[
    "fraction"
] = (
    residual_pooled[
        "energy"
    ]
    / residual_pooled[
        "total"
    ]
)

residual_pooled[
    "quantity"
] = "clean_residual_pooled_energy_fraction"


summary = pd.concat(
    [
        input_pooled[
            [
                "quantity",
                "model",
                "dataset",
                "band",
                "fraction",
            ]
        ],

        residual_pooled[
            [
                "quantity",
                "model",
                "dataset",
                "band",
                "fraction",
            ]
        ],
    ],
    ignore_index=True,
)


# ===========================================================================
# WRITE OUTPUTS
# ===========================================================================

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


cells = cells.sort_values(
    [
        "split_seed",
        "dataset",
        "split_group",
    ],
    kind="mergesort",
).reset_index(drop=True)


modes = modes.sort_values(
    [
        "split_seed",
        "dataset",
        "split_group",
        "k",
    ],
    kind="mergesort",
).reset_index(drop=True)


input_bands = input_bands.sort_values(
    [
        "split_seed",
        "dataset",
        "split_group",
        "feature",
        "band",
    ],
    kind="mergesort",
).reset_index(drop=True)


residual_bands = residual_bands.sort_values(
    [
        "split_seed",
        "train_seed",
        "model",
        "dataset",
        "split_group",
        "band",
    ],
    kind="mergesort",
).reset_index(drop=True)


metric_check = metric_check.sort_values(
    [
        "split_seed",
        "train_seed",
        "model",
        "domain",
    ],
    kind="mergesort",
).reset_index(drop=True)


summary = summary.sort_values(
    [
        "quantity",
        "model",
        "dataset",
        "band",
    ],
    kind="mergesort",
).reset_index(drop=True)


atomic_csv(
    cells,
    CELL_TABLE,
)

atomic_csv(
    modes,
    MODE_TABLE,
)

atomic_csv(
    input_bands,
    INPUT_BANDS,
)

atomic_csv(
    residual_bands,
    RESIDUAL_BANDS,
)

atomic_csv(
    metric_check,
    METRIC_CHECK,
)

atomic_csv(
    summary,
    SUMMARY,
)


manifest = {
    "contract":
        "clean_graph_fourier_diagnostics_v1",

    "status":
        "COMPLETE_BEFORE_FREQUENCY_PERTURBATION_SWEEP",

    "spectral_protocol_sha256":
        EXPECTED_PROTOCOL_SHA,

    "analysis_script_sha256":
        script_sha,

    "analysis_script_commit":
        script_commit,

    "benchmark_source_sha256":
        sha(BENCH),

    "final_v3_protocol_sha256":
        EXPECTED_FINAL_V3_PROTOCOL_SHA,

    "final_v3_execution_manifest_sha256":
        EXPECTED_FINAL_V3_EXEC_SHA,

    "models_retrained":
        False,

    "scaler_refit":
        False,

    "model_states_loaded":
        30,

    "state_file_sha256":
        state_hashes,

    "graph":
        "undirected chronological path graph P_n",

    "laplacian":
        "combinatorial D-A",

    "eigenvalue_formula":
        "lambda_k=2-2*cos(pi*k/n)",

    "frequency_coordinate":
        "omega_k_over_pi=k/n",

    "transform":
        "scipy.fft.dct type=2 norm=ortho",

    "dct_basis_equivalence":
        (
            "orthonormal DCT-II equals the frozen "
            "path-Laplacian eigenvector projection"
        ),

    "bands":
        {
            "low":
                "[0.00,0.25)",

            "mid_low":
                "[0.25,0.50)",

            "mid_high":
                "[0.50,0.75)",

            "high":
                "[0.75,1.00]",
        },

    "clean_metric_regression_barrier":
        "PASS",

    "clean_metric_tolerance":
        CLEAN_METRIC_TOL,

    "clean_metric_rows":
        len(metric_check),

    "clean_metric_max_abs_rmse_difference":
        max_rmse_diff,

    "clean_metric_max_abs_mae_difference":
        max_mae_diff,

    "input_parseval_max_relative_error":
        max_input_parseval_rel,

    "residual_parseval_max_relative_error":
        max_residual_parseval_rel,

    "cell_rows":
        len(cells),

    "mode_rows":
        len(modes),

    "input_feature_band_rows":
        len(input_bands),

    "residual_band_rows":
        len(residual_bands),

    "outputs": {
        "path_laplacian_cells": {
            "path":
                str(CELL_TABLE),

            "sha256":
                sha(CELL_TABLE),

            "rows":
                len(cells),
        },

        "path_laplacian_input_modes": {
            "path":
                str(MODE_TABLE),

            "sha256":
                sha(MODE_TABLE),

            "rows":
                len(modes),
        },

        "clean_input_feature_band_energy": {
            "path":
                str(INPUT_BANDS),

            "sha256":
                sha(INPUT_BANDS),

            "rows":
                len(input_bands),
        },

        "clean_residual_band_energy": {
            "path":
                str(RESIDUAL_BANDS),

            "sha256":
                sha(RESIDUAL_BANDS),

            "rows":
                len(residual_bands),
        },

        "clean_prediction_metric_regression": {
            "path":
                str(METRIC_CHECK),

            "sha256":
                sha(METRIC_CHECK),

            "rows":
                len(metric_check),
        },

        "clean_graph_fourier_summary": {
            "path":
                str(SUMMARY),

            "sha256":
                sha(SUMMARY),

            "rows":
                len(summary),
        },
    },
}


atomic_json(
    manifest,
    MANIFEST,
)


# ===========================================================================
# COMPACT TERMINAL REPORT
# ===========================================================================

print("=" * 92)
print("CLEAN GRAPH-FOURIER DIAGNOSTICS")
print("=" * 92)

print(
    "clean final-v3 metric regression barrier: PASS"
)

print(
    "metric regression rows:",
    len(metric_check)
)

print(
    "max abs RMSE difference:",
    f"{max_rmse_diff:.3e}"
)

print(
    "max abs MAE difference:",
    f"{max_mae_diff:.3e}"
)

print(
    "test cell representations:",
    len(cells)
)

print(
    "actual Laplacian modes:",
    len(modes)
)

print(
    "frozen model states used:",
    len(state_hashes)
)

print(
    "max input Parseval relative error:",
    f"{max_input_parseval_rel:.3e}"
)

print(
    "max residual Parseval relative error:",
    f"{max_residual_parseval_rel:.3e}"
)

print(
    "lambda global min:",
    f"{cells['lambda_0'].min():.9f}"
)

print(
    "lambda global max:",
    f"{cells['lambda_max'].max():.9f}"
)

print()
print("[CLEAN INPUT POOLED GRAPH-FOURIER ENERGY]")

for dataset in [
    "HUST",
    "CALCE",
    "MATR",
]:

    g = input_pooled[
        input_pooled[
            "dataset"
        ] == dataset
    ].set_index(
        "band"
    )

    print(
        dataset,
        " ".join(
            f"{band}={g.loc[band,'fraction']:.4f}"
            for band, _, _, _ in BANDS
        ),
    )


print()
print("[CLEAN RESIDUAL POOLED GRAPH-FOURIER ENERGY]")

for model in MODELS:

    for dataset in [
        "HUST",
        "CALCE",
        "MATR",
    ]:

        g = residual_pooled[
            (
                residual_pooled[
                    "model"
                ] == model
            )
            & (
                residual_pooled[
                    "dataset"
                ] == dataset
            )
        ].set_index(
            "band"
        )

        print(
            model,
            dataset,
            " ".join(
                f"{band}={g.loc[band,'fraction']:.4f}"
                for band, _, _, _ in BANDS
            ),
        )


print()
print(
    "manifest SHA256:",
    sha(MANIFEST)
)

print()
print(
    "CLEAN_GRAPH_FOURIER_DIAGNOSTICS_PASS=True"
)
