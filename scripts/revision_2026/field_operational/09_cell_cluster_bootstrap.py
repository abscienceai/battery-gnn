from pathlib import Path
import os
import numpy as np
import pandas as pd

root = Path(os.environ["R2"])
reports = root / "reports"

def load(prefix):
    paths = [
        reports / f"{prefix}{i}.csv"
        for i in range(11)
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return pd.concat(
        [pd.read_csv(p) for p in paths],
        ignore_index=True
    )

field = load("r2_synthetic_pilot_rank")
gauss = load("r2_gaussian_control_rank")

keys = ["HUST_cell", "source", "model", "seed"]

assert len(field) == len(gauss) == 132
assert not field.duplicated(keys).any()
assert not gauss.duplicated(keys).any()

data = field.merge(
    gauss,
    on=keys,
    suffixes=("_field", "_gaussian"),
    validate="one_to_one"
)

assert np.allclose(
    data.clean_rmse_field,
    data.clean_rmse_gaussian,
    atol=1e-12,
    rtol=0
)

rng = np.random.default_rng(20261017)
results = []

for (source, model), group in data.groupby(
    ["source", "model"], sort=True
):
    cells = group.groupby("HUST_cell").agg(
        clean_mse=(
            "clean_rmse_field",
            lambda x: np.mean(np.square(x))
        ),
        field_mse=(
            "perturbed_rmse_field",
            lambda x: np.mean(np.square(x))
        ),
        gaussian_mse=(
            "perturbed_rmse_gaussian",
            lambda x: np.mean(np.square(x))
        ),
        seeds=("seed", "nunique")
    )

    assert len(cells) == 11
    assert (cells.seeds == 3).all()

    matrix = cells[
        ["clean_mse", "field_mse", "gaussian_mse"]
    ].to_numpy(float)

    def contrasts(a):
        rmse = np.sqrt(a.mean(axis=-2))
        clean, field, gaussian = [
            rmse[..., j] for j in range(3)
        ]
        return (
            field - clean,
            gaussian - clean,
            field - gaussian
        )

    point = contrasts(matrix)
    indices = rng.integers(
        0, 11, size=(10000, 11)
    )
    bootstrap = contrasts(matrix[indices])

    names = [
        "field_minus_clean",
        "gaussian_minus_clean",
        "field_minus_gaussian"
    ]

    for name, value, draws in zip(
        names, point, bootstrap
    ):
        low, high = np.percentile(
            draws, [2.5, 97.5]
        )
        results.append({
            "source": source,
            "model": model,
            "comparison": name,
            "cells": 11,
            "seeds": 3,
            "rmse_difference": float(value),
            "bootstrap_95_low": float(low),
            "bootstrap_95_high": float(high),
            "interval_includes_zero": bool(
                low <= 0 <= high
            )
        })

new = pd.DataFrame(results)
original = pd.read_csv(
    reports / "r2_pilot_cluster_bootstrap.csv"
)

comparison_keys = ["source", "model", "comparison"]
numeric = [
    "rmse_difference",
    "bootstrap_95_low",
    "bootstrap_95_high"
]

joined = new.merge(
    original,
    on=comparison_keys,
    suffixes=("_new", "_original"),
    validate="one_to_one"
)

assert len(joined) == 12

max_difference = max(
    float(np.max(np.abs(
        joined[f"{col}_new"] -
        joined[f"{col}_original"]
    )))
    for col in numeric
)

print(
    "BOOTSTRAP REPRODUCIBILITY:",
    "PASS" if max_difference < 1e-7 else "FAIL"
)
print("Maximum difference:", max_difference)

if max_difference >= 1e-7:
    raise RuntimeError(
        "Bootstrap does not reproduce original output"
    )
