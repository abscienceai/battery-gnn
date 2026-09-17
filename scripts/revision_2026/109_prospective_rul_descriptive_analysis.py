from pathlib import Path
import hashlib
import json
import math
import os
import subprocess

import numpy as np
import pandas as pd


BASE = Path.home() / "alper/battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

ROOT = BASE / "results/revision_2026/prospective_rul"
EXEC = ROOT / "training_v1/final_execution_v1"
OUT = ROOT / "analysis_v1"

PROTOCOL = ROOT / "prospective_rul_training_protocol_v1.json"
EXEC_MANIFEST = EXEC / "prospective_rul_execution_manifest.json"
AUDIT_RECEIPT = EXEC / "prospective_rul_execution_audit_receipt.json"

SELF = (
    REPO
    / "scripts/revision_2026"
    / "109_prospective_rul_descriptive_analysis.py"
)

RUN_OUT = OUT / "prospective_rul_run_metrics.csv"
SPLIT_OUT = OUT / "prospective_rul_split_means.csv"
SUMMARY_OUT = OUT / "prospective_rul_descriptive_summary.csv"
CONTRAST_OUT = OUT / "prospective_rul_model_contrasts.csv"
MANIFEST_OUT = OUT / "prospective_rul_analysis_manifest.json"


EXPECTED_PROTOCOL_SHA = (
    "1c43d2f63904d19cf94921be5020f23d01baa4b9d6d3c6badf7560bb15f36bb7"
)

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


def atomic_csv(frame, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(str(path) + ".tmp")

    frame.to_csv(
        tmp,
        index=False,
    )

    os.replace(
        tmp,
        path,
    )


def atomic_json(obj, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(str(path) + ".tmp")

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


assert sha256(PROTOCOL) == EXPECTED_PROTOCOL_SHA
assert sha256(EXEC_MANIFEST) == EXPECTED_EXEC_SHA
assert sha256(AUDIT_RECEIPT) == EXPECTED_AUDIT_SHA


receipt = json.loads(
    AUDIT_RECEIPT.read_text(
        encoding="utf-8"
    )
)

assert receipt["audit_pass"] is True

assert receipt[
    "execution_manifest_sha256"
] == EXPECTED_EXEC_SHA

assert receipt[
    "validated_run_records"
] == 15

assert receipt[
    "validated_model_fits"
] == 30

assert receipt[
    "all_test_metrics_independently_reproduced"
] is True

assert receipt[
    "numeric_test_metrics_surfaced_by_audit"
] is False


manifest = json.loads(
    EXEC_MANIFEST.read_text(
        encoding="utf-8"
    )
)

assert manifest["contract"] == "prospective_rul_execution_v1"
assert manifest["execution"]["paired_runs"] == 15
assert manifest["execution"]["model_fits"] == 30
assert manifest["execution"]["safety_ceiling_hits"] == 0

assert len(
    manifest["run_records"]
) == 15


rel = str(
    SELF.relative_to(REPO)
)

subprocess.check_call(
    [
        "git",
        "ls-files",
        "--error-unmatch",
        rel,
    ],
    cwd=REPO,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

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


rows = []
seen_pairs = set()


for item in manifest["run_records"]:

    path = Path(item["path"])

    assert path.is_file()
    assert sha256(path) == item["sha256"]

    record = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    split_seed = int(
        record["split_seed"]
    )

    train_seed = int(
        record["training_initialization_seed"]
    )

    pair = (
        split_seed,
        train_seed,
    )

    assert pair not in seen_pairs
    seen_pairs.add(pair)

    assert record[
        "training_protocol_sha256"
    ] == EXPECTED_PROTOCOL_SHA

    assert set(
        record["models"]
    ) == set(MODELS)


    for model in MODELS:

        test = record[
            "models"
        ][
            model
        ][
            "test"
        ]

        assert set(test) == set(SCOPES)


        for scope in SCOPES:

            block = test[scope]

            assert set(
                block["by_dataset"]
            ) == {
                "CALCE",
                "MATR",
            }


            for domain in DOMAINS:

                if domain == "BALANCED_ALL":
                    source = block["balanced"]
                else:
                    source = block[
                        "by_dataset"
                    ][
                        domain
                    ]


                n_rows = int(
                    source["n_rows"]
                )

                n_cells = int(
                    source["n_cells"]
                )

                assert n_rows > 0
                assert n_cells > 0


                for metric in METRICS:

                    value = float(
                        source[metric]
                    )

                    assert math.isfinite(value)

                    if metric in LOWER_IS_BETTER:
                        assert value >= 0

                    rows.append({
                        "split_seed":
                            split_seed,

                        "training_initialization_seed":
                            train_seed,

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
                            n_rows,

                        "n_cells":
                            n_cells,
                    })


expected_pairs = {
    (split_seed, train_seed)
    for split_seed in range(5)
    for train_seed in range(3)
}

assert seen_pairs == expected_pairs


run = pd.DataFrame(rows)

assert len(run) == (
    15
    * 4
    * 2
    * 3
    * 4
)

assert len(run) == 1440

assert not run.duplicated(
    [
        "split_seed",
        "training_initialization_seed",
        "scope",
        "model",
        "domain",
        "metric",
    ]
).any()


# ------------------------------------------------------------
# Split-level means:
# initialization seeds are within-split repetitions.
# ------------------------------------------------------------

split_rows = []


for key, g in run.groupby(
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

    assert g[
        "n_rows"
    ].nunique() == 1

    assert g[
        "n_cells"
    ].nunique() == 1

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
            float(
                g["value"].mean()
            ),

        "n_rows":
            int(
                g["n_rows"].iloc[0]
            ),

        "n_cells":
            int(
                g["n_cells"].iloc[0]
            ),
    })


split_means = pd.DataFrame(
    split_rows
)

assert len(split_means) == 480


# ------------------------------------------------------------
# Descriptive summary.
# Mean over 15 paired runs.
# Uncertainty descriptor = SD of the five split-level means.
# ------------------------------------------------------------

summary_rows = []


for key, g in run.groupby(
    [
        "scope",
        "model",
        "domain",
        "metric",
    ],
    sort=True,
):

    assert len(g) == 15

    z = split_means[
        (split_means["scope"] == key[0])
        & (split_means["model"] == key[1])
        & (split_means["domain"] == key[2])
        & (split_means["metric"] == key[3])
    ]

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
            float(
                g["value"].mean()
            ),

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


summary = pd.DataFrame(
    summary_rows
)

assert len(summary) == 96


# ------------------------------------------------------------
# Paired MLP-vs-GCN descriptive contrasts.
#
# Positive contrast ALWAYS means GCN is descriptively better.
# Error metrics: MLP - GCN.
# R2: GCN - MLP.
# ------------------------------------------------------------

pivot = run.pivot(
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

assert len(pivot) == 720

assert set(
    pivot.columns
).issuperset(
    {
        "MLP",
        "GCN",
    }
)


contrast_run = pivot.copy()


contrast_run[
    "contrast_positive_favors_GCN"
] = np.where(
    contrast_run[
        "metric"
    ].isin(
        LOWER_IS_BETTER
    ),
    contrast_run["MLP"]
    - contrast_run["GCN"],
    contrast_run["GCN"]
    - contrast_run["MLP"],
)


contrast_rows = []


for key, g in contrast_run.groupby(
    [
        "scope",
        "domain",
        "metric",
    ],
    sort=True,
):

    assert len(g) == 15

    per_split = (
        g.groupby(
            "split_seed"
        )[
            "contrast_positive_favors_GCN"
        ]
        .mean()
    )

    assert len(per_split) == 5


    metric = str(key[2])

    mlp_mean = float(
        g["MLP"].mean()
    )

    gcn_mean = float(
        g["GCN"].mean()
    )


    if metric in LOWER_IS_BETTER:

        assert mlp_mean > 0

        relative_pct = float(
            100.0
            * (
                mlp_mean
                - gcn_mean
            )
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
                (
                    per_split > 0
                ).sum()
            ),

        "tie_split_count":
            int(
                (
                    per_split == 0
                ).sum()
            ),

        "MLP_better_split_count":
            int(
                (
                    per_split < 0
                ).sum()
            ),

        "relative_error_reduction_pct_from_model_means":
            relative_pct,

        "inferential_test":
            False,
    })


contrasts = pd.DataFrame(
    contrast_rows
)

assert len(contrasts) == 48

assert (
    contrasts[
        "GCN_better_split_count"
    ]
    + contrasts[
        "tie_split_count"
    ]
    + contrasts[
        "MLP_better_split_count"
    ]
    == 5
).all()

assert not contrasts[
    "inferential_test"
].any()


# ------------------------------------------------------------
# Write outputs.
# ------------------------------------------------------------

run = run.sort_values(
    [
        "scope",
        "domain",
        "metric",
        "model",
        "split_seed",
        "training_initialization_seed",
    ],
    kind="stable",
).reset_index(
    drop=True
)

split_means = split_means.sort_values(
    [
        "scope",
        "domain",
        "metric",
        "model",
        "split_seed",
    ],
    kind="stable",
).reset_index(
    drop=True
)

summary = summary.sort_values(
    [
        "scope",
        "domain",
        "metric",
        "model",
    ],
    kind="stable",
).reset_index(
    drop=True
)

contrasts = contrasts.sort_values(
    [
        "scope",
        "domain",
        "metric",
    ],
    kind="stable",
).reset_index(
    drop=True
)


atomic_csv(
    run,
    RUN_OUT,
)

atomic_csv(
    split_means,
    SPLIT_OUT,
)

atomic_csv(
    summary,
    SUMMARY_OUT,
)

atomic_csv(
    contrasts,
    CONTRAST_OUT,
)


analysis_manifest = {
    "contract":
        "prospective_rul_descriptive_analysis_v1",

    "status":
        "COMPLETE_PENDING_INDEPENDENT_ANALYSIS_AUDIT",

    "training_protocol_sha256":
        EXPECTED_PROTOCOL_SHA,

    "execution_manifest_sha256":
        EXPECTED_EXEC_SHA,

    "execution_audit_receipt_sha256":
        EXPECTED_AUDIT_SHA,

    "execution_audit_passed_before_analysis":
        True,

    "analysis_semantics": {
        "scopes":
            SCOPES,

        "models":
            MODELS,

        "domains":
            DOMAINS,

        "metrics":
            METRICS,

        "mean":
            "mean over 15 paired split/init runs",

        "uncertainty_descriptor":
            (
                "sample SD across the five split-level means; "
                "each split-level mean first averages the three "
                "initialization repetitions"
            ),

        "five_split_seeds_claimed_independent_trials":
            False,

        "initialization_seeds_claimed_independent_splits":
            False,

        "contrast_sign":
            (
                "positive always favors GCN; "
                "MLP-GCN for lower-is-better metrics, "
                "GCN-MLP for R2"
            ),

        "new_inferential_tests":
            False,

        "model_ranking":
            "descriptive only",

        "primary_early_scopes": [
            "K25",
            "K50",
            "K100",
        ],

        "full_trajectory_role":
            "supportive",
    },

    "row_counts": {
        "run_metrics":
            len(run),

        "split_means":
            len(split_means),

        "descriptive_summary":
            len(summary),

        "model_contrasts":
            len(contrasts),
    },

    "inputs": {
        "training_protocol": {
            "path":
                str(PROTOCOL),

            "sha256":
                sha256(PROTOCOL),
        },

        "execution_manifest": {
            "path":
                str(EXEC_MANIFEST),

            "sha256":
                sha256(EXEC_MANIFEST),
        },

        "execution_audit_receipt": {
            "path":
                str(AUDIT_RECEIPT),

            "sha256":
                sha256(AUDIT_RECEIPT),
        },
    },

    "outputs": {
        "run_metrics": {
            "path":
                str(RUN_OUT),

            "sha256":
                sha256(RUN_OUT),
        },

        "split_means": {
            "path":
                str(SPLIT_OUT),

            "sha256":
                sha256(SPLIT_OUT),
        },

        "descriptive_summary": {
            "path":
                str(SUMMARY_OUT),

            "sha256":
                sha256(SUMMARY_OUT),
        },

        "model_contrasts": {
            "path":
                str(CONTRAST_OUT),

            "sha256":
                sha256(CONTRAST_OUT),
        },
    },

    "analysis_script": {
        "path":
            str(SELF),

        "sha256":
            sha256(SELF),

        "commit":
            subprocess.check_output(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ],
                cwd=REPO,
                text=True,
            ).strip(),
    },

    "numeric_test_metrics_printed_by_analysis_script":
        False,

    "new_inferential_tests":
        False,
}


atomic_json(
    analysis_manifest,
    MANIFEST_OUT,
)


print("PROSPECTIVE RUL DESCRIPTIVE ANALYSIS")
print("audited input records: 15")
print("run metric rows: 1440")
print("split-mean rows: 480")
print("summary rows: 96")
print("model-contrast rows: 48")
print("primary early scopes: K25,K50,K100")
print("full trajectory role: supportive")
print("new inferential tests: False")
print("numeric test metric values printed: False")
print("analysis_manifest_sha256:", sha256(MANIFEST_OUT))
print("PROSPECTIVE_RUL_DESCRIPTIVE_ANALYSIS_COMPLETE_READY_FOR_AUDIT=True")
