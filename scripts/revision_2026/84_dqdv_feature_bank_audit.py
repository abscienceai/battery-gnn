from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


BASE = Path.home() / "alper" / "battery_revision_2026"

ROOT = (
    BASE
    / "results/revision_2026/dqdv_leakage_control/final_dqdv_v1"
)

PROTOCOL = ROOT / "protocol.json"

PROVENANCE = (
    ROOT
    / "barrier_v2/duration_provenance_barrier_manifest.json"
)

INVENTORY = (
    ROOT
    / "barrier_v1/sensor_discharge_inventory.csv"
)

CUTOFFS = (
    ROOT
    / "barrier_v1/train_only_prefix_cutoffs.csv"
)

RAW_FEATURE_DIR = (
    BASE
    / "data/processed/revision_2026/raw_cycle_features"
)

OUT = ROOT / "features_v1"

MANIFEST = (
    OUT
    / "dqdv_feature_bank_manifest.json"
)

EXPECTED_PROTOCOL_SHA = (
    "c455bf7a46f89649f25769a43abd443f171545b06e4740c799df1842c8bbb8e9"
)

EXPECTED_PROVENANCE_SHA = (
    "2d4b67d3d1debea5950b399717e22b7db5c1e5df4c583a1f4fbf85be1f1a32a1"
)

DATASETS = [
    "HUST",
    "CALCE",
    "MATR",
]

FEATURES = [
    "ica_mean_Ah_per_V",
    "ica_std_Ah_per_V",
    "ica_q10_Ah_per_V",
    "ica_q25_Ah_per_V",
    "ica_median_Ah_per_V",
    "ica_q75_Ah_per_V",
    "ica_q90_Ah_per_V",
    "ica_max_Ah_per_V",
    "ica_area_Ah",
    "ica_peak_voltage_V",
]


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


def bool_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)

    x = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
    )

    assert not x.isna().any()

    return x.astype(bool)


assert sha256(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

assert sha256(
    PROVENANCE
) == EXPECTED_PROVENANCE_SHA

m = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)

assert m[
    "contract"
] == "dqdv_feature_bank_v1"

assert m[
    "status"
] == "FEATURE_GENERATION_COMPLETE_BEFORE_MODELING"

assert m[
    "protocol_sha256"
] == EXPECTED_PROTOCOL_SHA

assert m[
    "provenance_barrier_sha256"
] == EXPECTED_PROVENANCE_SHA

assert m[
    "target_table_loaded"
] is False

assert m[
    "capacity_values_read"
] is False

assert m[
    "SOH_values_read"
] is False

assert m[
    "RUL_or_EOL_values_read"
] is False

assert m[
    "model_metrics_read"
] is False

assert m[
    "models_trained"
] is False

assert m[
    "cutoffs_refit"
] is False

assert m[
    "test_cycle_total_duration_used"
] is False

assert m[
    "individual_test_cycle_endpoint_used"
] is False

assert m[
    "primary_valid_total_rows"
] == 319291

assert m[
    "sensor_inventory_total_rows"
] == 319202

assert m[
    "primary_valid_outside_sensor_inventory_by_dataset"
] == {
    "HUST": 0,
    "CALCE": 89,
    "MATR": 0,
}


inventory = pd.read_csv(
    INVENTORY,
    usecols=[
        "dataset",
        "raw_cell_id",
        "raw_cycle_position_zero_based",
        "q_discharge_sensor_Ah",
    ],
    low_memory=False,
)

inventory[
    "raw_cycle_position_zero_based"
] = pd.to_numeric(
    inventory[
        "raw_cycle_position_zero_based"
    ],
    errors="raise",
).astype(int)


total = 0


for dataset in DATASETS:

    info = m[
        "outputs"
    ][
        dataset
    ]

    p = Path(
        info[
            "path"
        ]
    )

    assert p.is_file()

    assert sha256(
        p
    ) == info[
        "sha256"
    ]

    df = pd.read_csv(
        p,
        low_memory=False,
    )

    total += len(
        df
    )

    assert len(
        df
    ) == int(
        info[
            "rows"
        ]
    )

    assert (
        df[
            "dataset"
        ] == dataset
    ).all()

    forbidden = [
        "capacity",
        "soh",
        "rul",
        "eol",
        "target",
    ]

    for col in df.columns:

        low = str(
            col
        ).lower()

        assert not any(
            x in low
            for x in forbidden
        ), col

    df[
        "raw_cycle_position_zero_based"
    ] = pd.to_numeric(
        df[
            "raw_cycle_position_zero_based"
        ],
        errors="raise",
    ).astype(int)


    # --------------------------------------------------------------
    # Exact audited primary-feature-valid identity universe.
    # --------------------------------------------------------------

    raw = pd.read_csv(
        RAW_FEATURE_DIR
        / f"{dataset}_raw_cycle_features.csv",
        usecols=[
            "dataset",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
            "feature_valid",
        ],
        low_memory=False,
    )

    raw[
        "raw_cycle_position_zero_based"
    ] = pd.to_numeric(
        raw[
            "raw_cycle_position_zero_based"
        ],
        errors="raise",
    ).astype(int)

    primary = raw[
        bool_series(
            raw[
                "feature_valid"
            ]
        )
    ][
        [
            "dataset",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
        ]
    ].copy()

    exact = df[
        [
            "dataset",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
        ]
    ].merge(
        primary,
        on=[
            "dataset",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
        ],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    assert (
        exact[
            "_merge"
        ] == "both"
    ).all()


    # --------------------------------------------------------------
    # Frozen sensor inventory is a subset, not the universe.
    # --------------------------------------------------------------

    inv = inventory[
        inventory[
            "dataset"
        ] == dataset
    ].copy()

    z = inv.merge(
        df[
            [
                "dataset",
                "raw_cell_id",
                "raw_cycle_position_zero_based",
                "full_integrated_discharge_Ah_qc",
            ]
        ],
        on=[
            "dataset",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
        ],
        how="left",
        validate="one_to_one",
    )

    assert not z[
        "full_integrated_discharge_Ah_qc"
    ].isna().any()

    assert np.allclose(
        pd.to_numeric(
            z[
                "full_integrated_discharge_Ah_qc"
            ],
            errors="raise",
        ),
        pd.to_numeric(
            z[
                "q_discharge_sensor_Ah"
            ],
            errors="raise",
        ),
        rtol=1e-11,
        atol=1e-12,
    )


    membership = df.merge(
        inv[
            [
                "dataset",
                "raw_cell_id",
                "raw_cycle_position_zero_based",
            ]
        ],
        on=[
            "dataset",
            "raw_cell_id",
            "raw_cycle_position_zero_based",
        ],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    outside = (
        membership[
            "_merge"
        ] == "left_only"
    )

    assert int(
        outside.sum()
    ) == int(
        info[
            "primary_valid_outside_sensor_inventory"
        ]
    )

    if outside.any():

        assert not membership.loc[
            outside,
            "full_valid",
        ].astype(bool).any()


    # FULL model area equals FULL QC integral on valid rows.
    full_valid = (
        df[
            "full_valid"
        ].astype(
            bool
        )
    )

    assert np.allclose(
        pd.to_numeric(
            df.loc[
                full_valid,
                "full_ica_area_Ah",
            ],
            errors="raise",
        ),
        pd.to_numeric(
            df.loc[
                full_valid,
                "full_integrated_discharge_Ah_qc",
            ],
            errors="raise",
        ),
        rtol=1e-12,
        atol=1e-14,
    )


    full_area = pd.to_numeric(
        df[
            "full_integrated_discharge_Ah_qc"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    for seed in [
        0, 1, 2, 3, 4
    ]:

        for horizon in [
            "p25",
            "p50",
        ]:

            prefix = (
                f"s{seed}_{horizon}"
            )

            area = pd.to_numeric(
                df[
                    prefix
                    + "_integrated_discharge_Ah_qc"
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            finite = (
                np.isfinite(
                    area
                )
                & np.isfinite(
                    full_area
                )
            )

            assert np.all(
                area[
                    finite
                ]
                <= (
                    full_area[
                        finite
                    ]
                    + 1e-12
                )
            )

            valid = df[
                prefix
                + "_valid"
            ].astype(
                bool
            )

            assert (
                pd.to_numeric(
                    df.loc[
                        valid,
                        prefix
                        + "_discharge_intervals_qc",
                    ],
                    errors="raise",
                )
                >= 20
            ).all()

            assert (
                pd.to_numeric(
                    df.loc[
                        valid,
                        prefix
                        + "_occupied_voltage_bins_qc",
                    ],
                    errors="raise",
                )
                >= 5
            ).all()

            for f in FEATURES:

                vals = pd.to_numeric(
                    df.loc[
                        valid,
                        prefix
                        + "_"
                        + f,
                    ],
                    errors="raise",
                )

                assert np.isfinite(
                    vals
                ).all()

                invalid_vals = pd.to_numeric(
                    df.loc[
                        ~valid,
                        prefix
                        + "_"
                        + f,
                    ],
                    errors="coerce",
                )

                assert invalid_vals.isna().all()


    for f in FEATURES:

        vals = pd.to_numeric(
            df.loc[
                full_valid,
                "full_" + f,
            ],
            errors="raise",
        )

        assert np.isfinite(
            vals
        ).all()

        invalid_vals = pd.to_numeric(
            df.loc[
                ~full_valid,
                "full_" + f,
            ],
            errors="coerce",
        )

        assert invalid_vals.isna().all()


assert total == 319291


# Frozen cutoff artifact itself remains the one used by runner.
cut = pd.read_csv(
    CUTOFFS,
    low_memory=False,
)

assert len(
    cut
) == 15

assert (
    cut[
        "fit_split"
    ].astype(
        str
    )
    == "train_only"
).all()

assert not cut[
    "validation_or_test_rows_used"
].astype(
    bool
).any()

assert not cut[
    "individual_cycle_endpoint_used"
].astype(
    bool
).any()


print(
    "DQDV FEATURE BANK AUDIT"
)

print(
    "protocol exact:",
    True
)

print(
    "provenance barrier exact:",
    True
)

print(
    "dataset outputs:",
    3
)

print(
    "total feature-bank rows:",
    total
)

print(
    "primary-valid identity exact:",
    True
)

print(
    "sensor inventory is discharge-integrable subset:",
    True
)

print(
    "CALCE primary-valid rows outside inventory:",
    89
)

print(
    "full discharge integration regression:",
    True
)

print(
    "prefix discharge never exceeds full:",
    True
)

print(
    "valid dQdV vectors finite:",
    True
)

print(
    "invalid dQdV vectors masked:",
    True
)

print(
    "cutoffs remain train-only:",
    True
)

print(
    "target table loaded during generation:",
    False
)

print(
    "capacity/SOH/RUL/EOL values read during generation:",
    False
)

print(
    "model metrics read:",
    False
)

print(
    "models trained:",
    False
)

print(
    "manifest SHA256:",
    sha256(
        MANIFEST
    )
)

print(
    "DQDV_FEATURE_BANK_AUDIT_PASS=True"
)
