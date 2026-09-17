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

ROOT = BASE / "results/revision_2026/lodo_fewshot"
OUT = ROOT / "analysis_v1"

PROTOCOL = ROOT / "lodo_fewshot_protocol_v1.json"
DATA_MANIFEST = ROOT / "data_v1/lodo_fewshot_data_manifest.json"

ZERO = ROOT / "zero_shot_v1/final_execution_v1"
ZERO_MANIFEST = ZERO / "lodo_zero_shot_execution_manifest.json"
ZERO_AUDIT = ZERO / "lodo_zero_shot_execution_audit_receipt.json"

FEW = ROOT / "few_shot_v1/final_execution_v1"
FEW_MANIFEST = FEW / "lodo_fewshot_execution_manifest.json"
FEW_AUDIT = FEW / "lodo_fewshot_execution_audit_receipt.json"

SELF = (
    REPO
    / "scripts/revision_2026"
    / "119_lodo_fewshot_descriptive_analysis.py"
)

RUN_OUT = OUT / "lodo_fewshot_run_metrics.csv"
SPLIT_OUT = OUT / "lodo_fewshot_split_means.csv"
SUMMARY_OUT = OUT / "lodo_fewshot_descriptive_summary.csv"
ADAPT_OUT = OUT / "lodo_fewshot_vs_zero_contrasts.csv"
MODEL_OUT = OUT / "lodo_fewshot_model_contrasts.csv"
MANIFEST_OUT = OUT / "lodo_fewshot_analysis_manifest.json"


EXPECTED = {
    PROTOCOL:
        "966d8c6b95f29e671ce4be9a785c5088e6d8e7eda9d25b8389f204859641897f",

    DATA_MANIFEST:
        "8d3fa7c8b614305608e8db9f9f8d235388e6b304776cc64ffc8aa2fda6312a67",

    ZERO_MANIFEST:
        "b3e82cf395ce543c6452c62df02e34662b768f2cf24124d7eb530c88e11afbc1",

    ZERO_AUDIT:
        "21f2f4236a8fbb0f8d1293f72e69bccd9ce679be3c1002dbe57c6137cb68bd21",

    FEW_MANIFEST:
        "8c05609ebf554faf15dacd95ae8436205018580e2a7d0b52f06194faa0b625a1",

    FEW_AUDIT:
        "ef83bc676d4cd1f9d8ac0dea837b2895637ad00e52dba228438fe122ddae8d52",
}


HELD = [
    "HUST",
    "CALCE",
    "MATR",
]

CONDITIONS = [
    "ZERO_SHOT",
    "K1",
    "K3",
    "K5",
]

MODELS = [
    "MLP",
    "GCN",
]

SCOPES = [
    "CELL_BALANCED",
    "ROW",
]

METRICS = [
    "RMSE",
    "MAE",
    "R2",
]

LOWER_IS_BETTER = {
    "RMSE",
    "MAE",
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


for path, expected in EXPECTED.items():

    assert path.is_file()
    assert sha256(path) == expected


zero_audit = json.loads(
    ZERO_AUDIT.read_text(
        encoding="utf-8"
    )
)

few_audit = json.loads(
    FEW_AUDIT.read_text(
        encoding="utf-8"
    )
)

assert zero_audit["audit_pass"] is True
assert few_audit["audit_pass"] is True

assert zero_audit[
    "all_target_test_metrics_independently_reproduced"
] is True

assert few_audit[
    "all_target_test_metrics_independently_reproduced"
] is True

assert zero_audit[
    "numeric_test_metrics_surfaced_by_audit"
] is False

assert few_audit[
    "numeric_test_metrics_surfaced_by_audit"
] is False


zero_manifest = json.loads(
    ZERO_MANIFEST.read_text(
        encoding="utf-8"
    )
)

few_manifest = json.loads(
    FEW_MANIFEST.read_text(
        encoding="utf-8"
    )
)

assert len(
    zero_manifest[
        "run_records"
    ]
) == 45

assert len(
    few_manifest[
        "run_records"
    ]
) == 45


rel = str(
    SELF.relative_to(
        REPO
    )
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


# ------------------------------------------------------------
# Exact paired record lookup.
# ------------------------------------------------------------

def load_records(manifest):

    lookup = {}

    for item in manifest[
        "run_records"
    ]:

        path = Path(
            item[
                "path"
            ]
        )

        assert path.is_file()

        assert sha256(
            path
        ) == item[
            "sha256"
        ]

        record = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        key = (
            str(
                record[
                    "held_out"
                ]
            ),
            int(
                record[
                    "split_seed"
                ]
            ),
            int(
                record[
                    "training_initialization_seed"
                ]
            ),
        )

        assert key not in lookup

        lookup[
            key
        ] = record

    return lookup


zero_records = load_records(
    zero_manifest
)

few_records = load_records(
    few_manifest
)


expected_keys = {
    (
        held,
        split_seed,
        train_seed,
    )
    for held in HELD
    for split_seed in range(5)
    for train_seed in range(3)
}

assert set(
    zero_records
) == expected_keys

assert set(
    few_records
) == expected_keys


# ------------------------------------------------------------
# Construct unified run-level table.
# ------------------------------------------------------------

rows = []


for key in sorted(
    expected_keys
):

    held, split_seed, train_seed = key

    zero = zero_records[
        key
    ]

    few = few_records[
        key
    ]


    assert int(
        zero[
            "target_test_rows"
        ]
    ) == int(
        few[
            "target_test_rows"
        ]
    )

    assert int(
        zero[
            "target_test_cells"
        ]
    ) == int(
        few[
            "target_test_cells"
        ]
    )


    for model in MODELS:

        blocks = {
            "ZERO_SHOT":
                zero[
                    "models"
                ][
                    model
                ][
                    "test"
                ],

            "K1":
                few[
                    "models"
                ][
                    model
                ][
                    "K1"
                ][
                    "test"
                ],

            "K3":
                few[
                    "models"
                ][
                    model
                ][
                    "K3"
                ][
                    "test"
                ],

            "K5":
                few[
                    "models"
                ][
                    model
                ][
                    "K5"
                ][
                    "test"
                ],
        }


        for condition in CONDITIONS:

            block = blocks[
                condition
            ]


            for scope in SCOPES:

                source = block[
                    scope
                ]

                assert int(
                    source[
                        "n_rows"
                    ]
                ) == int(
                    zero[
                        "target_test_rows"
                    ]
                )

                assert int(
                    source[
                        "n_cells"
                    ]
                ) == int(
                    zero[
                        "target_test_cells"
                    ]
                )


                for metric in METRICS:

                    value = float(
                        source[
                            metric
                        ]
                    )

                    assert (
                        math.isfinite(
                            value
                        )
                        or (
                            metric == "R2"
                            and math.isnan(
                                value
                            )
                        )
                    )

                    rows.append({
                        "held_out":
                            held,

                        "split_seed":
                            split_seed,

                        "training_initialization_seed":
                            train_seed,

                        "condition":
                            condition,

                        "model":
                            model,

                        "scope":
                            scope,

                        "metric":
                            metric,

                        "value":
                            value,

                        "n_rows":
                            int(
                                source[
                                    "n_rows"
                                ]
                            ),

                        "n_cells":
                            int(
                                source[
                                    "n_cells"
                                ]
                            ),
                    })


run = pd.DataFrame(
    rows
)

assert len(
    run
) == 2160

assert not run.duplicated(
    [
        "held_out",
        "split_seed",
        "training_initialization_seed",
        "condition",
        "model",
        "scope",
        "metric",
    ]
).any()


# ------------------------------------------------------------
# Average 3 initialization repetitions within each split.
# ------------------------------------------------------------

split_rows = []


for key, g in run.groupby(
    [
        "held_out",
        "split_seed",
        "condition",
        "model",
        "scope",
        "metric",
    ],
    sort=True,
):

    assert len(
        g
    ) == 3

    assert g[
        "n_rows"
    ].nunique() == 1

    assert g[
        "n_cells"
    ].nunique() == 1


    split_rows.append({
        "held_out":
            str(
                key[
                    0
                ]
            ),

        "split_seed":
            int(
                key[
                    1
                ]
            ),

        "condition":
            str(
                key[
                    2
                ]
            ),

        "model":
            str(
                key[
                    3
                ]
            ),

        "scope":
            str(
                key[
                    4
                ]
            ),

        "metric":
            str(
                key[
                    5
                ]
            ),

        "mean_over_3_initializations":
            float(
                g[
                    "value"
                ].mean()
            ),

        "n_rows":
            int(
                g[
                    "n_rows"
                ].iloc[
                    0
                ]
            ),

        "n_cells":
            int(
                g[
                    "n_cells"
                ].iloc[
                    0
                ]
            ),
    })


split_means = pd.DataFrame(
    split_rows
)

assert len(
    split_means
) == 720


# ------------------------------------------------------------
# Overall descriptive summary:
# mean across 15 paired split/init runs;
# SD across 5 split-level means.
# ------------------------------------------------------------

summary_rows = []


for key, g in run.groupby(
    [
        "held_out",
        "condition",
        "model",
        "scope",
        "metric",
    ],
    sort=True,
):

    assert len(
        g
    ) == 15


    z = split_means[
        (
            split_means[
                "held_out"
            ]
            == key[
                0
            ]
        )
        & (
            split_means[
                "condition"
            ]
            == key[
                1
            ]
        )
        & (
            split_means[
                "model"
            ]
            == key[
                2
            ]
        )
        & (
            split_means[
                "scope"
            ]
            == key[
                3
            ]
        )
        & (
            split_means[
                "metric"
            ]
            == key[
                4
            ]
        )
    ]


    assert len(
        z
    ) == 5


    summary_rows.append({
        "held_out":
            str(
                key[
                    0
                ]
            ),

        "condition":
            str(
                key[
                    1
                ]
            ),

        "model":
            str(
                key[
                    2
                ]
            ),

        "scope":
            str(
                key[
                    3
                ]
            ),

        "metric":
            str(
                key[
                    4
                ]
            ),

        "mean_over_15_runs":
            float(
                g[
                    "value"
                ].mean()
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

assert len(
    summary
) == 144


# ------------------------------------------------------------
# Few-shot vs ZERO_SHOT paired descriptive contrasts.
#
# Positive ALWAYS means the adaptation condition is better.
# RMSE/MAE: ZERO - K.
# R2: K - ZERO.
# ------------------------------------------------------------

adapt_rows = []


for held in HELD:

    for model in MODELS:

        for scope in SCOPES:

            for metric in METRICS:

                base = run[
                    (
                        run[
                            "held_out"
                        ]
                        == held
                    )
                    & (
                        run[
                            "model"
                        ]
                        == model
                    )
                    & (
                        run[
                            "scope"
                        ]
                        == scope
                    )
                    & (
                        run[
                            "metric"
                        ]
                        == metric
                    )
                    & (
                        run[
                            "condition"
                        ]
                        == "ZERO_SHOT"
                    )
                ][
                    [
                        "split_seed",
                        "training_initialization_seed",
                        "value",
                    ]
                ].rename(
                    columns={
                        "value":
                            "zero_value",
                    }
                )


                assert len(
                    base
                ) == 15


                for condition in [
                    "K1",
                    "K3",
                    "K5",
                ]:

                    adapted = run[
                        (
                            run[
                                "held_out"
                            ]
                            == held
                        )
                        & (
                            run[
                                "model"
                            ]
                            == model
                        )
                        & (
                            run[
                                "scope"
                            ]
                            == scope
                        )
                        & (
                            run[
                                "metric"
                            ]
                            == metric
                        )
                        & (
                            run[
                                "condition"
                            ]
                            == condition
                        )
                    ][
                        [
                            "split_seed",
                            "training_initialization_seed",
                            "value",
                        ]
                    ].rename(
                        columns={
                            "value":
                                "adapted_value",
                        }
                    )


                    paired = base.merge(
                        adapted,
                        on=[
                            "split_seed",
                            "training_initialization_seed",
                        ],
                        how="inner",
                        validate="one_to_one",
                    )


                    assert len(
                        paired
                    ) == 15


                    if metric in LOWER_IS_BETTER:

                        paired[
                            "contrast"
                        ] = (
                            paired[
                                "zero_value"
                            ]
                            - paired[
                                "adapted_value"
                            ]
                        )

                    else:

                        paired[
                            "contrast"
                        ] = (
                            paired[
                                "adapted_value"
                            ]
                            - paired[
                                "zero_value"
                            ]
                        )


                    per_split = (
                        paired.groupby(
                            "split_seed"
                        )[
                            "contrast"
                        ]
                        .mean()
                    )


                    assert len(
                        per_split
                    ) == 5


                    zero_mean = float(
                        paired[
                            "zero_value"
                        ].mean()
                    )

                    adapted_mean = float(
                        paired[
                            "adapted_value"
                        ].mean()
                    )


                    if metric in LOWER_IS_BETTER:

                        assert zero_mean > 0

                        relative_pct = float(
                            100.0
                            * (
                                zero_mean
                                - adapted_mean
                            )
                            / zero_mean
                        )

                    else:

                        relative_pct = np.nan


                    adapt_rows.append({
                        "held_out":
                            held,

                        "condition":
                            condition,

                        "model":
                            model,

                        "scope":
                            scope,

                        "metric":
                            metric,

                        "ZERO_SHOT_mean_over_15_runs":
                            zero_mean,

                        "adapted_mean_over_15_runs":
                            adapted_mean,

                        "mean_paired_contrast_positive_favors_adaptation":
                            float(
                                paired[
                                    "contrast"
                                ].mean()
                            ),

                        "sd_over_5_split_mean_contrasts":
                            float(
                                per_split.std(
                                    ddof=1
                                )
                            ),

                        "adaptation_better_split_count":
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

                        "zero_shot_better_split_count":
                            int(
                                (
                                    per_split < 0
                                ).sum()
                            ),

                        "relative_error_reduction_pct_from_condition_means":
                            relative_pct,

                        "inferential_test":
                            False,
                    })


adapt = pd.DataFrame(
    adapt_rows
)

assert len(
    adapt
) == 108

assert (
    adapt[
        "adaptation_better_split_count"
    ]
    + adapt[
        "tie_split_count"
    ]
    + adapt[
        "zero_shot_better_split_count"
    ]
    == 5
).all()


# ------------------------------------------------------------
# GCN-vs-MLP paired contrasts within every condition.
#
# Positive ALWAYS means GCN is better.
# RMSE/MAE: MLP - GCN.
# R2: GCN - MLP.
# ------------------------------------------------------------

model_rows = []


for held in HELD:

    for condition in CONDITIONS:

        for scope in SCOPES:

            for metric in METRICS:

                z = run[
                    (
                        run[
                            "held_out"
                        ]
                        == held
                    )
                    & (
                        run[
                            "condition"
                        ]
                        == condition
                    )
                    & (
                        run[
                            "scope"
                        ]
                        == scope
                    )
                    & (
                        run[
                            "metric"
                        ]
                        == metric
                    )
                ]


                pivot = z.pivot(
                    index=[
                        "split_seed",
                        "training_initialization_seed",
                    ],
                    columns="model",
                    values="value",
                ).reset_index()


                assert len(
                    pivot
                ) == 15


                if metric in LOWER_IS_BETTER:

                    pivot[
                        "contrast"
                    ] = (
                        pivot[
                            "MLP"
                        ]
                        - pivot[
                            "GCN"
                        ]
                    )

                else:

                    pivot[
                        "contrast"
                    ] = (
                        pivot[
                            "GCN"
                        ]
                        - pivot[
                            "MLP"
                        ]
                    )


                per_split = (
                    pivot.groupby(
                        "split_seed"
                    )[
                        "contrast"
                    ]
                    .mean()
                )


                assert len(
                    per_split
                ) == 5


                mlp_mean = float(
                    pivot[
                        "MLP"
                    ].mean()
                )

                gcn_mean = float(
                    pivot[
                        "GCN"
                    ].mean()
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


                model_rows.append({
                    "held_out":
                        held,

                    "condition":
                        condition,

                    "scope":
                        scope,

                    "metric":
                        metric,

                    "MLP_mean_over_15_runs":
                        mlp_mean,

                    "GCN_mean_over_15_runs":
                        gcn_mean,

                    "mean_paired_contrast_positive_favors_GCN":
                        float(
                            pivot[
                                "contrast"
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


model_contrast = pd.DataFrame(
    model_rows
)

assert len(
    model_contrast
) == 72


# ------------------------------------------------------------
# Canonical sorting and output.
# ------------------------------------------------------------

run = run.sort_values(
    [
        "held_out",
        "condition",
        "model",
        "scope",
        "metric",
        "split_seed",
        "training_initialization_seed",
    ],
    kind="stable",
).reset_index(
    drop=True
)

split_means = split_means.sort_values(
    [
        "held_out",
        "condition",
        "model",
        "scope",
        "metric",
        "split_seed",
    ],
    kind="stable",
).reset_index(
    drop=True
)

summary = summary.sort_values(
    [
        "held_out",
        "condition",
        "model",
        "scope",
        "metric",
    ],
    kind="stable",
).reset_index(
    drop=True
)

adapt = adapt.sort_values(
    [
        "held_out",
        "condition",
        "model",
        "scope",
        "metric",
    ],
    kind="stable",
).reset_index(
    drop=True
)

model_contrast = model_contrast.sort_values(
    [
        "held_out",
        "condition",
        "scope",
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
    adapt,
    ADAPT_OUT,
)

atomic_csv(
    model_contrast,
    MODEL_OUT,
)


analysis_manifest = {
    "contract":
        "lodo_fewshot_descriptive_analysis_v1",

    "status":
        "COMPLETE_PENDING_INDEPENDENT_ANALYSIS_AUDIT",

    "audited_inputs": {
        "zero_shot_execution_manifest_sha256":
            EXPECTED[
                ZERO_MANIFEST
            ],

        "zero_shot_execution_audit_sha256":
            EXPECTED[
                ZERO_AUDIT
            ],

        "few_shot_execution_manifest_sha256":
            EXPECTED[
                FEW_MANIFEST
            ],

        "few_shot_execution_audit_sha256":
            EXPECTED[
                FEW_AUDIT
            ],
    },

    "analysis_semantics": {
        "conditions":
            CONDITIONS,

        "held_out_folds":
            HELD,

        "models":
            MODELS,

        "primary_scope":
            "CELL_BALANCED",

        "secondary_scope":
            "ROW",

        "metrics":
            METRICS,

        "primary_summary":
            "mean over 15 split/init paired runs",

        "uncertainty_descriptor":
            (
                "sample SD over five split-level means after "
                "averaging the three initialization repetitions"
            ),

        "adaptation_contrast_sign":
            (
                "positive always favors K1/K3/K5 over ZERO_SHOT"
            ),

        "model_contrast_sign":
            "positive always favors GCN over MLP",

        "five_split_seeds_claimed_independent_trials":
            False,

        "initialization_seeds_claimed_independent_splits":
            False,

        "new_inferential_tests":
            False,

        "model_ranking":
            "descriptive only",

        "fewshot_monotonicity_assumed":
            False,

        "failure_reportable":
            True,
    },

    "row_counts": {
        "run_metrics":
            len(
                run
            ),

        "split_means":
            len(
                split_means
            ),

        "summary":
            len(
                summary
            ),

        "fewshot_vs_zero_contrasts":
            len(
                adapt
            ),

        "model_contrasts":
            len(
                model_contrast
            ),
    },

    "outputs": {
        "run_metrics": {
            "path":
                str(
                    RUN_OUT
                ),

            "sha256":
                sha256(
                    RUN_OUT
                ),
        },

        "split_means": {
            "path":
                str(
                    SPLIT_OUT
                ),

            "sha256":
                sha256(
                    SPLIT_OUT
                ),
        },

        "descriptive_summary": {
            "path":
                str(
                    SUMMARY_OUT
                ),

            "sha256":
                sha256(
                    SUMMARY_OUT
                ),
        },

        "fewshot_vs_zero_contrasts": {
            "path":
                str(
                    ADAPT_OUT
                ),

            "sha256":
                sha256(
                    ADAPT_OUT
                ),
        },

        "model_contrasts": {
            "path":
                str(
                    MODEL_OUT
                ),

            "sha256":
                sha256(
                    MODEL_OUT
                ),
        },
    },

    "analysis_script": {
        "path":
            str(
                SELF
            ),

        "sha256":
            sha256(
                SELF
            ),

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


print("LODO / FEW-SHOT DESCRIPTIVE ANALYSIS")
print("audited zero-shot records: 45")
print("audited few-shot records: 45")
print("run metric rows: 2160")
print("split-mean rows: 720")
print("summary rows: 144")
print("few-shot-vs-zero contrast rows: 108")
print("GCN-vs-MLP contrast rows: 72")
print("primary scope: CELL_BALANCED")
print("secondary scope: ROW")
print("new inferential tests: False")
print("numeric test metrics printed: False")
print("analysis_manifest_sha256:", sha256(MANIFEST_OUT))
print("LODO_FEWSHOT_DESCRIPTIVE_ANALYSIS_COMPLETE_READY_FOR_AUDIT=True")
