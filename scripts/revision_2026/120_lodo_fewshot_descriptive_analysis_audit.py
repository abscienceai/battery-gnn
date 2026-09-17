from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd


BASE = Path.home() / "alper/battery_revision_2026"

ROOT = BASE / "results/revision_2026/lodo_fewshot"
OUT = ROOT / "analysis_v1"

ZERO = ROOT / "zero_shot_v1/final_execution_v1"
FEW = ROOT / "few_shot_v1/final_execution_v1"

ZERO_MANIFEST = ZERO / "lodo_zero_shot_execution_manifest.json"
ZERO_AUDIT = ZERO / "lodo_zero_shot_execution_audit_receipt.json"

FEW_MANIFEST = FEW / "lodo_fewshot_execution_manifest.json"
FEW_AUDIT = FEW / "lodo_fewshot_execution_audit_receipt.json"

RUN_OUT = OUT / "lodo_fewshot_run_metrics.csv"
SPLIT_OUT = OUT / "lodo_fewshot_split_means.csv"
SUMMARY_OUT = OUT / "lodo_fewshot_descriptive_summary.csv"
ADAPT_OUT = OUT / "lodo_fewshot_vs_zero_contrasts.csv"
MODEL_OUT = OUT / "lodo_fewshot_model_contrasts.csv"
ANALYSIS_MANIFEST = OUT / "lodo_fewshot_analysis_manifest.json"


EXPECTED_ZERO_MANIFEST = (
    "b3e82cf395ce543c6452c62df02e34662b768f2cf24124d7eb530c88e11afbc1"
)

EXPECTED_ZERO_AUDIT = (
    "21f2f4236a8fbb0f8d1293f72e69bccd9ce679be3c1002dbe57c6137cb68bd21"
)

EXPECTED_FEW_MANIFEST = (
    "8c05609ebf554faf15dacd95ae8436205018580e2a7d0b52f06194faa0b625a1"
)

EXPECTED_FEW_AUDIT = (
    "ef83bc676d4cd1f9d8ac0dea837b2895637ad00e52dba228438fe122ddae8d52"
)


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

            b = f.read(
                1024 * 1024
            )

            if not b:
                break

            h.update(b)

    return h.hexdigest()


assert sha256(
    ZERO_MANIFEST
) == EXPECTED_ZERO_MANIFEST

assert sha256(
    ZERO_AUDIT
) == EXPECTED_ZERO_AUDIT

assert sha256(
    FEW_MANIFEST
) == EXPECTED_FEW_MANIFEST

assert sha256(
    FEW_AUDIT
) == EXPECTED_FEW_AUDIT


za = json.loads(
    ZERO_AUDIT.read_text(
        encoding="utf-8"
    )
)

fa = json.loads(
    FEW_AUDIT.read_text(
        encoding="utf-8"
    )
)

assert za["audit_pass"] is True
assert fa["audit_pass"] is True


zm = json.loads(
    ZERO_MANIFEST.read_text(
        encoding="utf-8"
    )
)

fm = json.loads(
    FEW_MANIFEST.read_text(
        encoding="utf-8"
    )
)


def lookup(manifest):

    out = {}

    for item in manifest[
        "run_records"
    ]:

        path = Path(
            item[
                "path"
            ]
        )

        assert sha256(
            path
        ) == item[
            "sha256"
        ]

        r = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        key = (
            str(
                r[
                    "held_out"
                ]
            ),
            int(
                r[
                    "split_seed"
                ]
            ),
            int(
                r[
                    "training_initialization_seed"
                ]
            ),
        )

        assert key not in out

        out[
            key
        ] = r

    return out


zr = lookup(
    zm
)

fr = lookup(
    fm
)


expected_keys = {
    (
        h,
        s,
        i,
    )
    for h in HELD
    for s in range(5)
    for i in range(3)
}

assert set(
    zr
) == expected_keys

assert set(
    fr
) == expected_keys


# ------------------------------------------------------------
# Independent run table reconstruction.
# ------------------------------------------------------------

rows = []


for key in sorted(
    expected_keys
):

    h, s, i = key

    z = zr[
        key
    ]

    f = fr[
        key
    ]

    assert int(
        z[
            "target_test_rows"
        ]
    ) == int(
        f[
            "target_test_rows"
        ]
    )

    assert int(
        z[
            "target_test_cells"
        ]
    ) == int(
        f[
            "target_test_cells"
        ]
    )


    for model in MODELS:

        blocks = {
            "ZERO_SHOT":
                z[
                    "models"
                ][
                    model
                ][
                    "test"
                ],

            "K1":
                f[
                    "models"
                ][
                    model
                ][
                    "K1"
                ][
                    "test"
                ],

            "K3":
                f[
                    "models"
                ][
                    model
                ][
                    "K3"
                ][
                    "test"
                ],

            "K5":
                f[
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

            for scope in SCOPES:

                block = blocks[
                    condition
                ][
                    scope
                ]


                for metric in METRICS:

                    rows.append({
                        "held_out":
                            h,

                        "split_seed":
                            s,

                        "training_initialization_seed":
                            i,

                        "condition":
                            condition,

                        "model":
                            model,

                        "scope":
                            scope,

                        "metric":
                            metric,

                        "value":
                            float(
                                block[
                                    metric
                                ]
                            ),

                        "n_rows":
                            int(
                                block[
                                    "n_rows"
                                ]
                            ),

                        "n_cells":
                            int(
                                block[
                                    "n_cells"
                                ]
                            ),
                    })


expected_run = pd.DataFrame(
    rows
)

assert len(
    expected_run
) == 2160


# ------------------------------------------------------------
# Split means.
# ------------------------------------------------------------

split_rows = []


for key, g in expected_run.groupby(
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


expected_split = pd.DataFrame(
    split_rows
)

assert len(
    expected_split
) == 720


# ------------------------------------------------------------
# Summary.
# ------------------------------------------------------------

summary_rows = []


for key, g in expected_run.groupby(
    [
        "held_out",
        "condition",
        "model",
        "scope",
        "metric",
    ],
    sort=True,
):

    z = expected_split[
        (
            expected_split[
                "held_out"
            ]
            == key[
                0
            ]
        )
        & (
            expected_split[
                "condition"
            ]
            == key[
                1
            ]
        )
        & (
            expected_split[
                "model"
            ]
            == key[
                2
            ]
        )
        & (
            expected_split[
                "scope"
            ]
            == key[
                3
            ]
        )
        & (
            expected_split[
                "metric"
            ]
            == key[
                4
            ]
        )
    ]


    assert len(
        g
    ) == 15

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


expected_summary = pd.DataFrame(
    summary_rows
)

assert len(
    expected_summary
) == 144


# ------------------------------------------------------------
# Adaptation contrasts.
# ------------------------------------------------------------

adapt_rows = []


for held in HELD:

    for model in MODELS:

        for scope in SCOPES:

            for metric in METRICS:

                base = expected_run[
                    (
                        expected_run[
                            "held_out"
                        ]
                        == held
                    )
                    & (
                        expected_run[
                            "model"
                        ]
                        == model
                    )
                    & (
                        expected_run[
                            "scope"
                        ]
                        == scope
                    )
                    & (
                        expected_run[
                            "metric"
                        ]
                        == metric
                    )
                    & (
                        expected_run[
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


                for condition in [
                    "K1",
                    "K3",
                    "K5",
                ]:

                    adapted = expected_run[
                        (
                            expected_run[
                                "held_out"
                            ]
                            == held
                        )
                        & (
                            expected_run[
                                "model"
                            ]
                            == model
                        )
                        & (
                            expected_run[
                                "scope"
                            ]
                            == scope
                        )
                        & (
                            expected_run[
                                "metric"
                            ]
                            == metric
                        )
                        & (
                            expected_run[
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


                    p = base.merge(
                        adapted,
                        on=[
                            "split_seed",
                            "training_initialization_seed",
                        ],
                        validate="one_to_one",
                    )


                    if metric in LOWER_IS_BETTER:

                        p[
                            "contrast"
                        ] = (
                            p[
                                "zero_value"
                            ]
                            - p[
                                "adapted_value"
                            ]
                        )

                    else:

                        p[
                            "contrast"
                        ] = (
                            p[
                                "adapted_value"
                            ]
                            - p[
                                "zero_value"
                            ]
                        )


                    per_split = (
                        p.groupby(
                            "split_seed"
                        )[
                            "contrast"
                        ]
                        .mean()
                    )


                    zero_mean = float(
                        p[
                            "zero_value"
                        ].mean()
                    )

                    adapted_mean = float(
                        p[
                            "adapted_value"
                        ].mean()
                    )


                    if metric in LOWER_IS_BETTER:

                        relative = float(
                            100.0
                            * (
                                zero_mean
                                - adapted_mean
                            )
                            / zero_mean
                        )

                    else:

                        relative = np.nan


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
                                p[
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
                            relative,

                        "inferential_test":
                            False,
                    })


expected_adapt = pd.DataFrame(
    adapt_rows
)

assert len(
    expected_adapt
) == 108


# ------------------------------------------------------------
# Model contrasts.
# ------------------------------------------------------------

model_rows = []


for held in HELD:

    for condition in CONDITIONS:

        for scope in SCOPES:

            for metric in METRICS:

                z = expected_run[
                    (
                        expected_run[
                            "held_out"
                        ]
                        == held
                    )
                    & (
                        expected_run[
                            "condition"
                        ]
                        == condition
                    )
                    & (
                        expected_run[
                            "scope"
                        ]
                        == scope
                    )
                    & (
                        expected_run[
                            "metric"
                        ]
                        == metric
                    )
                ]


                p = z.pivot(
                    index=[
                        "split_seed",
                        "training_initialization_seed",
                    ],
                    columns="model",
                    values="value",
                ).reset_index()


                if metric in LOWER_IS_BETTER:

                    p[
                        "contrast"
                    ] = (
                        p[
                            "MLP"
                        ]
                        - p[
                            "GCN"
                        ]
                    )

                else:

                    p[
                        "contrast"
                    ] = (
                        p[
                            "GCN"
                        ]
                        - p[
                            "MLP"
                        ]
                    )


                per_split = (
                    p.groupby(
                        "split_seed"
                    )[
                        "contrast"
                    ]
                    .mean()
                )


                mlp_mean = float(
                    p[
                        "MLP"
                    ].mean()
                )

                gcn_mean = float(
                    p[
                        "GCN"
                    ].mean()
                )


                relative = (
                    float(
                        100.0
                        * (
                            mlp_mean
                            - gcn_mean
                        )
                        / mlp_mean
                    )
                    if metric in LOWER_IS_BETTER
                    else np.nan
                )


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
                            p[
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
                        relative,

                    "inferential_test":
                        False,
                })


expected_model = pd.DataFrame(
    model_rows
)

assert len(
    expected_model
) == 72


# ------------------------------------------------------------
# Sort and compare.
# ------------------------------------------------------------

sort_run = [
    "held_out",
    "condition",
    "model",
    "scope",
    "metric",
    "split_seed",
    "training_initialization_seed",
]

sort_split = [
    "held_out",
    "condition",
    "model",
    "scope",
    "metric",
    "split_seed",
]

sort_summary = [
    "held_out",
    "condition",
    "model",
    "scope",
    "metric",
]

sort_adapt = [
    "held_out",
    "condition",
    "model",
    "scope",
    "metric",
]

sort_model = [
    "held_out",
    "condition",
    "scope",
    "metric",
]


expected_run = expected_run.sort_values(
    sort_run,
    kind="stable",
).reset_index(
    drop=True
)

expected_split = expected_split.sort_values(
    sort_split,
    kind="stable",
).reset_index(
    drop=True
)

expected_summary = expected_summary.sort_values(
    sort_summary,
    kind="stable",
).reset_index(
    drop=True
)

expected_adapt = expected_adapt.sort_values(
    sort_adapt,
    kind="stable",
).reset_index(
    drop=True
)

expected_model = expected_model.sort_values(
    sort_model,
    kind="stable",
).reset_index(
    drop=True
)


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

actual_adapt = pd.read_csv(
    ADAPT_OUT,
    low_memory=False,
)

actual_model = pd.read_csv(
    MODEL_OUT,
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
    actual_adapt,
    expected_adapt,
    check_dtype=False,
    check_exact=False,
    rtol=1e-12,
    atol=1e-12,
)

pd.testing.assert_frame_equal(
    actual_model,
    expected_model,
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
] == "lodo_fewshot_descriptive_analysis_v1"

assert analysis[
    "status"
] == "COMPLETE_PENDING_INDEPENDENT_ANALYSIS_AUDIT"

assert analysis[
    "analysis_semantics"
][
    "primary_scope"
] == "CELL_BALANCED"

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
    "new_inferential_tests"
] is False

assert analysis[
    "analysis_semantics"
][
    "fewshot_monotonicity_assumed"
] is False

assert analysis[
    "analysis_semantics"
][
    "failure_reportable"
] is True

assert analysis[
    "numeric_test_metrics_printed_by_analysis_script"
] is False


for item in analysis[
    "outputs"
].values():

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


print("LODO / FEW-SHOT DESCRIPTIVE ANALYSIS AUDIT")
print("45 zero-shot records hash-verified:", True)
print("45 few-shot records hash-verified:", True)
print("2160 run metric rows independently reproduced:", True)
print("720 split means independently reproduced:", True)
print("144 descriptive summaries independently reproduced:", True)
print("108 few-shot-vs-zero contrasts independently reproduced:", True)
print("72 GCN-vs-MLP contrasts independently reproduced:", True)
print("identical target test sets preserved across conditions:", True)
print("initializations treated as within-split repetitions:", True)
print("split-level uncertainty semantics preserved:", True)
print("adaptation contrast sign positive=few-shot better:", True)
print("model contrast sign positive=GCN better:", True)
print("few-shot monotonic improvement assumed:", False)
print("new inferential tests:", False)
print("numeric metrics surfaced by audit:", False)
print("LODO_FEWSHOT_DESCRIPTIVE_ANALYSIS_AUDIT_PASS=True")
