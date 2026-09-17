from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd


BASE = Path.home() / "alper/battery_revision_2026"

ROOT = BASE / "results/revision_2026/prospective_rul"
EXEC = ROOT / "training_v1/final_execution_v1"
OUT = ROOT / "analysis_v1"

EXEC_MANIFEST = EXEC / "prospective_rul_execution_manifest.json"
AUDIT_RECEIPT = EXEC / "prospective_rul_execution_audit_receipt.json"

RUN_OUT = OUT / "prospective_rul_run_metrics.csv"
SPLIT_OUT = OUT / "prospective_rul_split_means.csv"
SUMMARY_OUT = OUT / "prospective_rul_descriptive_summary.csv"
CONTRAST_OUT = OUT / "prospective_rul_model_contrasts.csv"
ANALYSIS_MANIFEST = OUT / "prospective_rul_analysis_manifest.json"


EXPECTED_EXEC_SHA = (
    "bf65abd7e8d6ab5c03a1afae85902df201f67aebeb4c38a57c4ac17ca857dfc6"
)

EXPECTED_AUDIT_SHA = (
    "1446736661d9432a3ae9962d59ffa841aea70b194173f3fb85d2283546286191"
)

SCOPES = [
    "FULL_TRAJECTORY",
    "K25",
    "K50",
    "K100",
]

MODELS = [
    "MLP",
    "GCN",
]

DOMAINS = [
    "BALANCED_ALL",
    "CALCE",
    "MATR",
]

METRICS = [
    "RMSE",
    "MAE",
    "R2",
    "normalized_MAE",
]

LOWER_IS_BETTER = {
    "RMSE",
    "MAE",
    "normalized_MAE",
}


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


assert sha256(EXEC_MANIFEST) == EXPECTED_EXEC_SHA
assert sha256(AUDIT_RECEIPT) == EXPECTED_AUDIT_SHA


receipt = json.loads(
    AUDIT_RECEIPT.read_text(
        encoding="utf-8"
    )
)

assert receipt["audit_pass"] is True
assert receipt[
    "all_test_metrics_independently_reproduced"
] is True


manifest = json.loads(
    EXEC_MANIFEST.read_text(
        encoding="utf-8"
    )
)


# ------------------------------------------------------------
# Independently rebuild run-metric table.
# ------------------------------------------------------------

rows = []
seen = set()


for item in manifest["run_records"]:

    path = Path(item["path"])

    assert sha256(path) == item["sha256"]

    record = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    s = int(record["split_seed"])
    t = int(record["training_initialization_seed"])

    assert (s, t) not in seen
    seen.add((s, t))


    for model in MODELS:

        test = record["models"][model]["test"]


        for scope in SCOPES:

            block = test[scope]


            for domain in DOMAINS:

                source = (
                    block["balanced"]
                    if domain == "BALANCED_ALL"
                    else block["by_dataset"][domain]
                )

                for metric in METRICS:

                    value = float(
                        source[metric]
                    )

                    assert math.isfinite(value)

                    rows.append({
                        "split_seed":
                            s,

                        "training_initialization_seed":
                            t,

                        "scope":
                            scope,

                        "model":
                            model,

                        "domain":
                            domain,

                        "metric":
                            metric,

                        "value":
                            value,

                        "n_rows":
                            int(source["n_rows"]),

                        "n_cells":
                            int(source["n_cells"]),
                    })


assert seen == {
    (s, t)
    for s in range(5)
    for t in range(3)
}


expected_run = pd.DataFrame(rows)

assert len(expected_run) == 1440


# ------------------------------------------------------------
# Rebuild split means.
# ------------------------------------------------------------

split_rows = []


for key, g in expected_run.groupby(
    [
        "split_seed",
        "scope",
        "model",
        "domain",
        "metric",
    ],
    sort=True,
):

    assert len(g) == 3
    assert g["n_rows"].nunique() == 1
    assert g["n_cells"].nunique() == 1

    split_rows.append({
        "split_seed":
            int(key[0]),

        "scope":
            str(key[1]),

        "model":
            str(key[2]),

        "domain":
            str(key[3]),

        "metric":
            str(key[4]),

        "mean_over_3_initializations":
            float(g["value"].mean()),

        "n_rows":
            int(g["n_rows"].iloc[0]),

        "n_cells":
            int(g["n_cells"].iloc[0]),
    })


expected_split = pd.DataFrame(split_rows)

assert len(expected_split) == 480


# ------------------------------------------------------------
# Rebuild summary.
# ------------------------------------------------------------

summary_rows = []


for key, g in expected_run.groupby(
    [
        "scope",
        "model",
        "domain",
        "metric",
    ],
    sort=True,
):

    z = expected_split[
        (expected_split["scope"] == key[0])
        & (expected_split["model"] == key[1])
        & (expected_split["domain"] == key[2])
        & (expected_split["metric"] == key[3])
    ]

    assert len(g) == 15
    assert len(z) == 5

    summary_rows.append({
        "scope":
            str(key[0]),

        "model":
            str(key[1]),

        "domain":
            str(key[2]),

        "metric":
            str(key[3]),

        "mean_over_15_runs":
            float(g["value"].mean()),

        "sd_over_5_split_means":
            float(
                z[
                    "mean_over_3_initializations"
                ].std(
                    ddof=1
                )
            ),

        "minimum_split_mean":
            float(
                z[
                    "mean_over_3_initializations"
                ].min()
            ),

        "maximum_split_mean":
            float(
                z[
                    "mean_over_3_initializations"
                ].max()
            ),

        "n_runs":
            15,

        "n_split_seeds":
            5,

        "initializations_per_split":
            3,
    })


expected_summary = pd.DataFrame(
    summary_rows
)

assert len(expected_summary) == 96


# ------------------------------------------------------------
# Rebuild contrasts.
# ------------------------------------------------------------

pivot = expected_run.pivot(
    index=[
        "split_seed",
        "training_initialization_seed",
        "scope",
        "domain",
        "metric",
    ],
    columns="model",
    values="value",
).reset_index()


pivot[
    "contrast_positive_favors_GCN"
] = np.where(
    pivot["metric"].isin(LOWER_IS_BETTER),
    pivot["MLP"] - pivot["GCN"],
    pivot["GCN"] - pivot["MLP"],
)


contrast_rows = []


for key, g in pivot.groupby(
    [
        "scope",
        "domain",
        "metric",
    ],
    sort=True,
):

    per_split = (
        g.groupby("split_seed")[
            "contrast_positive_favors_GCN"
        ]
        .mean()
    )

    assert len(g) == 15
    assert len(per_split) == 5

    metric = str(key[2])

    mlp_mean = float(
        g["MLP"].mean()
    )

    gcn_mean = float(
        g["GCN"].mean()
    )

    if metric in LOWER_IS_BETTER:
        relative_pct = float(
            100.0
            * (mlp_mean - gcn_mean)
            / mlp_mean
        )
    else:
        relative_pct = np.nan


    contrast_rows.append({
        "scope":
            str(key[0]),

        "domain":
            str(key[1]),

        "metric":
            metric,

        "MLP_mean_over_15_runs":
            mlp_mean,

        "GCN_mean_over_15_runs":
            gcn_mean,

        "mean_paired_contrast_positive_favors_GCN":
            float(
                g[
                    "contrast_positive_favors_GCN"
                ].mean()
            ),

        "sd_over_5_split_mean_contrasts":
            float(
                per_split.std(
                    ddof=1
                )
            ),

        "GCN_better_split_count":
            int(
                (per_split > 0).sum()
            ),

        "tie_split_count":
            int(
                (per_split == 0).sum()
            ),

        "MLP_better_split_count":
            int(
                (per_split < 0).sum()
            ),

        "relative_error_reduction_pct_from_model_means":
            relative_pct,

        "inferential_test":
            False,
    })


expected_contrast = pd.DataFrame(
    contrast_rows
)

assert len(expected_contrast) == 48


# ------------------------------------------------------------
# Canonical sorting before comparison.
# ------------------------------------------------------------

expected_run = expected_run.sort_values(
    [
        "scope",
        "domain",
        "metric",
        "model",
        "split_seed",
        "training_initialization_seed",
    ],
    kind="stable",
).reset_index(drop=True)

expected_split = expected_split.sort_values(
    [
        "scope",
        "domain",
        "metric",
        "model",
        "split_seed",
    ],
    kind="stable",
).reset_index(drop=True)

expected_summary = expected_summary.sort_values(
    [
        "scope",
        "domain",
        "metric",
        "model",
    ],
    kind="stable",
).reset_index(drop=True)

expected_contrast = expected_contrast.sort_values(
    [
        "scope",
        "domain",
        "metric",
    ],
    kind="stable",
).reset_index(drop=True)


actual_run = pd.read_csv(
    RUN_OUT,
    low_memory=False,
)

actual_split = pd.read_csv(
    SPLIT_OUT,
    low_memory=False,
)

actual_summary = pd.read_csv(
    SUMMARY_OUT,
    low_memory=False,
)

actual_contrast = pd.read_csv(
    CONTRAST_OUT,
    low_memory=False,
)


pd.testing.assert_frame_equal(
    actual_run,
    expected_run,
    check_dtype=False,
    check_exact=False,
    rtol=1e-12,
    atol=1e-12,
)

pd.testing.assert_frame_equal(
    actual_split,
    expected_split,
    check_dtype=False,
    check_exact=False,
    rtol=1e-12,
    atol=1e-12,
)

pd.testing.assert_frame_equal(
    actual_summary,
    expected_summary,
    check_dtype=False,
    check_exact=False,
    rtol=1e-12,
    atol=1e-12,
)

pd.testing.assert_frame_equal(
    actual_contrast,
    expected_contrast,
    check_dtype=False,
    check_exact=False,
    rtol=1e-12,
    atol=1e-12,
)


analysis = json.loads(
    ANALYSIS_MANIFEST.read_text(
        encoding="utf-8"
    )
)

assert analysis[
    "contract"
] == "prospective_rul_descriptive_analysis_v1"

assert analysis[
    "status"
] == "COMPLETE_PENDING_INDEPENDENT_ANALYSIS_AUDIT"

assert analysis[
    "execution_audit_passed_before_analysis"
] is True

assert analysis[
    "analysis_semantics"
][
    "new_inferential_tests"
] is False

assert analysis[
    "analysis_semantics"
][
    "model_ranking"
] == "descriptive only"

assert analysis[
    "analysis_semantics"
][
    "five_split_seeds_claimed_independent_trials"
] is False

assert analysis[
    "analysis_semantics"
][
    "initialization_seeds_claimed_independent_splits"
] is False

assert analysis[
    "analysis_semantics"
][
    "primary_early_scopes"
] == [
    "K25",
    "K50",
    "K100",
]

assert analysis[
    "analysis_semantics"
][
    "full_trajectory_role"
] == "supportive"

assert analysis[
    "numeric_test_metrics_printed_by_analysis_script"
] is False

assert analysis[
    "new_inferential_tests"
] is False


for item in analysis["outputs"].values():
    path = Path(item["path"])
    assert path.is_file()
    assert sha256(path) == item["sha256"]


print("PROSPECTIVE RUL DESCRIPTIVE ANALYSIS AUDIT")
print("15 execution records hash-verified:", True)
print("1440 run-metric rows independently reproduced:", True)
print("480 split-mean rows independently reproduced:", True)
print("96 summary rows independently reproduced:", True)
print("48 model contrasts independently reproduced:", True)
print("split-level uncertainty semantics preserved:", True)
print("initialization seeds treated as within-split repetitions:", True)
print("positive contrast consistently favors GCN:", True)
print("K25/K50/K100 retained as primary early evaluation:", True)
print("FULL retained as supportive evaluation:", True)
print("new inferential tests:", False)
print("numeric metric values surfaced by audit:", False)
print("PROSPECTIVE_RUL_DESCRIPTIVE_ANALYSIS_AUDIT_PASS=True")
