from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo" / "battery-gnn"

ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "physical_sensor_noise"
    / "final_physical_v1"
)

OUT = ROOT / "analysis"

EXECUTION_MANIFEST = (
    ROOT
    / "evaluation"
    / "physical_noise_execution_manifest.json"
)

ANALYSIS_SCRIPT = (
    REPO
    / "scripts"
    / "revision_2026"
    / "59_analyze_final_physical_sensor_noise.py"
)

MANIFEST = (
    OUT
    / "physical_noise_analysis_manifest.json"
)

EXPECTED_EXECUTION_SHA = (
    "7880a61bca31ed2f3329b5bdf9aad0481a4306ffd9f05402e60764b3c0e124ad"
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
    EXECUTION_MANIFEST
) == EXPECTED_EXECUTION_SHA


m = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)


assert m[
    "contract"
] == "final_physical_sensor_noise_analysis_v1"

assert m[
    "status"
] == "COMPLETE"

assert m[
    "execution_manifest_sha256"
] == EXPECTED_EXECUTION_SHA

assert m[
    "analysis_script_sha256"
] == sha(
    ANALYSIS_SCRIPT
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
    "split_count"
] == 5

assert m[
    "train_initializations_nested_per_split"
] == 3

assert m[
    "corruption_seeds_nested_per_split"
] == 5

assert m[
    "inferential_unit"
] == "split_seed"

assert m[
    "inferential_n"
] == 5

assert m[
    "minimum_attainable_two_sided_signflip_p_at_n5"
] == 0.0625

assert m[
    "compound_medium_inferential"
] is False

assert m[
    "clean_reference_check"
] == "PASS"


expected_rows = {
    "level_by_split": 1920,
    "level_summary": 384,
    "auc_by_split": 320,
    "primary_inference": 32,
    "compound_by_split": 120,
    "compound_summary": 24,
    "clean_reference_check": 16,
}


for key, rows in expected_rows.items():

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

    assert int(
        info[
            "rows"
        ]
    ) == rows


inference = pd.read_csv(
    m[
        "outputs"
    ][
        "primary_inference"
    ][
        "path"
    ]
)

assert len(
    inference
) == 32

assert set(
    inference[
        "metric"
    ]
) == {
    "rmse",
    "mae",
}

assert set(
    inference[
        "split_count"
    ]
) == {5}

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
    ) == 16


for c in [
    "mlp_auc_mean",
    "gcn_auc_mean",
    "mean_delta_mlp_minus_gcn",
    "relative_gcn_advantage_pct",
    "p_exact_signflip",
    "p_holm",
]:
    assert np.isfinite(
        pd.to_numeric(
            inference[
                c
            ],
            errors="raise",
        )
    ).all()


assert (
    (
        inference[
            "p_exact_signflip"
        ]
        >= 0
    )
    & (
        inference[
            "p_exact_signflip"
        ]
        <= 1
    )
).all()

assert (
    (
        inference[
            "p_holm"
        ]
        >= 0
    )
    & (
        inference[
            "p_holm"
        ]
        <= 1
    )
).all()


clean = pd.read_csv(
    m[
        "outputs"
    ][
        "clean_reference_check"
    ][
        "path"
    ]
)

assert len(clean) == 16

assert clean[
    "pass"
].astype(bool).all()


auc = pd.read_csv(
    m[
        "outputs"
    ][
        "auc_by_split"
    ][
        "path"
    ]
)

assert len(auc) == 320

assert set(
    auc[
        "split_seed"
    ]
) == {
    0,
    1,
    2,
    3,
    4,
}


print("=" * 80)
print("FINAL PHYSICAL SENSOR NOISE ANALYSIS AUDIT")
print("=" * 80)

print(
    "execution manifest exact:",
    True
)

print(
    "analysis script exact:",
    True
)

print(
    "clean reference regression barrier:",
    True
)

print(
    "split inferential units:",
    5
)

print(
    "train initializations nested:",
    3
)

print(
    "corruption seeds nested:",
    5
)

print(
    "AUC split rows:",
    len(
        auc
    )
)

print(
    "primary inferential comparisons:",
    len(
        inference
    )
)

print(
    "Holm families:",
    "16 RMSE + 16 MAE"
)

print(
    "compound medium inferential test:",
    False
)

print(
    "analysis manifest SHA256:",
    sha(
        MANIFEST
    )
)

print()
print(
    "FINAL_PHYSICAL_SENSOR_NOISE_ANALYSIS_AUDIT_PASS=True"
)
