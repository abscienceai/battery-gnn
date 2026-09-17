from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo" / "battery-gnn"

ROOT = (
    BASE
    / "results/revision_2026/physical_sensor_noise/final_physical_v1"
)

OUT = ROOT / "analysis_v2_domain_contract"

EXEC = (
    ROOT
    / "evaluation_v2_domain_contract"
    / "physical_noise_execution_manifest_v2.json"
)

SCRIPT = (
    REPO
    / "scripts/revision_2026/63_analyze_final_physical_sensor_noise_v2.py"
)

MANIFEST = (
    OUT
    / "physical_noise_analysis_v2_manifest.json"
)

EXPECTED_EXEC_SHA = (
    "cdc02692291041e3fc3e2ef192af857fa7e95c146b7d91dcb7a43bb5e268a127"
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


assert sha(EXEC) == EXPECTED_EXEC_SHA


m = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)


assert m[
    "contract"
] == "final_physical_sensor_noise_analysis_v2"

assert m[
    "status"
] == "COMPLETE"

assert m[
    "execution_manifest_sha256"
] == EXPECTED_EXEC_SHA

assert m[
    "analysis_script_sha256"
] == sha(SCRIPT)

assert m[
    "value_level_read_started_only_after_analysis_script_commit"
] is True

assert m[
    "source_execution_version"
] == "v2_domain_metric_contract"

assert m[
    "v1_metric_analysis_used"
] is False

assert m[
    "raw_metric_files_read"
] == 30

assert m[
    "raw_metric_rows_read"
] == 7920

assert m[
    "primary_metrics"
] == [
    "rmse",
    "mae",
]

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
    "clean_final_v3_runlevel_barrier"
] == "PASS"

assert m[
    "clean_run_count"
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


expected_rows = {
    "clean_check": 120,
    "level_by_split": 1280,
    "level_summary": 256,
    "auc_by_split": 320,
    "primary_inference": 32,
    "compound_by_split": 80,
    "compound_summary": 16,
}


for key, rows in expected_rows.items():

    info = m[
        "outputs"
    ][key]

    p = Path(
        info["path"]
    )

    assert p.is_file()
    assert sha(p) == info["sha256"]
    assert int(
        info["rows"]
    ) == rows


clean = pd.read_csv(
    m[
        "outputs"
    ][
        "clean_check"
    ][
        "path"
    ]
)

assert len(clean) == 120

assert clean[
    "pass"
].astype(bool).all()

assert clean[
    "n_match"
].astype(bool).all()


inference = pd.read_csv(
    m[
        "outputs"
    ][
        "primary_inference"
    ][
        "path"
    ]
)

assert len(inference) == 32

assert set(
    inference["metric"]
) == {
    "rmse",
    "mae",
}

assert set(
    inference["split_count"]
) == {5}


for metric in [
    "rmse",
    "mae",
]:
    assert int(
        (
            inference["metric"]
            == metric
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
    x = pd.to_numeric(
        inference[c],
        errors="raise",
    )

    assert np.isfinite(x).all()


assert (
    inference[
        "p_holm"
    ]
    >= inference[
        "p_exact_signflip"
    ]
    - 1e-15
).all()


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
    auc["split_seed"]
) == {
    0,
    1,
    2,
    3,
    4,
}


print("=" * 82)
print("FINAL PHYSICAL SENSOR NOISE ANALYSIS V2 AUDIT")
print("=" * 82)

print(
    "execution manifest exact:",
    True
)

print(
    "analysis script exact:",
    True
)

print(
    "clean final-v3 run-level barrier:",
    True
)

print(
    "clean run pairs:",
    120
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
    320
)

print(
    "primary inferential comparisons:",
    32
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
    sha(MANIFEST)
)

print()
print(
    "FINAL_PHYSICAL_SENSOR_NOISE_ANALYSIS_V2_AUDIT_PASS=True"
)
