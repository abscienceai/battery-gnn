from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch


BASE = Path(
    os.environ.get(
        "BATTERY_REVISION_BASE",
        Path.home()
        / "alper"
        / "battery_revision_2026",
    )
)

REPO = (
    BASE
    / "repo"
    / "battery-gnn"
)

ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "feature_matched_soh"
)

V3 = (
    ROOT
    / "final_v3"
)

PROTOCOL = (
    ROOT
    / "full_clean_benchmark_protocol_v3.json"
)

MANIFEST = (
    V3
    / "final_v3_execution_manifest.json"
)

STATUS = (
    V3
    / "runner_status.json"
)

BENCH = (
    REPO
    / "scripts"
    / "revision_2026"
    / "24_feature_matched_soh_benchmark.py"
)

RUNNER = (
    REPO
    / "scripts"
    / "revision_2026"
    / "29_full_clean_soh_v3_runner.py"
)


EXPECTED = {
    "protocol":
        "2dec14c210ae3f02e5f65e4acc24f9c4adc747bc071509cb144e69dba707f4dd",

    "manifest":
        "5a8bfbe0f11a6158f2749a9da4f18f1e62adf8c839976c1bf0af9c9f6c07440d",

    "status":
        "1d8467608831abb49c6111fc1958834e8186a52a7f4348296ec793fbd294febe",

    "benchmark":
        "5d621f8d5764e417e41bc518ec72b9608566a087c552b8365d438c2344c06c5e",

    "runner":
        "b945728c02255f1f2977e47024e33ca75f4e76570c8d381b4881836f17d3dcba",
}


EXPECTED_INPUT_HASHES = {
    "supervised":
        "49a7b7629cf9ee3f8ce0a7e3c52f2807878b752b4c9a30a876285fb891b00c14",

    "assignments":
        "914c8b90131f496ca396f63e0502147ca99848fdfd4b2c6b2a5887225b47d055",

    "scalers":
        "b9708b9321b521b87bdd05f969558134d7e0148285ef7c50f648c2028cd86de1",
}


EXPECTED_ROWS = {
    0: {
        "train": 211243,
        "validation": 45336,
        "test": 43974,
    },

    1: {
        "train": 210640,
        "validation": 43640,
        "test": 46273,
    },

    2: {
        "train": 211505,
        "validation": 45230,
        "test": 43818,
    },

    3: {
        "train": 213879,
        "validation": 40518,
        "test": 46156,
    },

    4: {
        "train": 212447,
        "validation": 44357,
        "test": 43749,
    },
}


EXPECTED_CELLS = {
    "train": 182,
    "validation": 39,
    "test": 38,
}


PLANNED = [
    (split_seed, train_seed)
    for split_seed in range(5)
    for train_seed in range(3)
]


def sha256_file(path):
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


print("=" * 80)
print("FINAL V3 EXECUTION INDEPENDENT AUDIT")
print("=" * 80)


# =============================================================================
# 1. Frozen top-level hashes
# =============================================================================

assert sha256_file(
    PROTOCOL
) == EXPECTED["protocol"]

assert sha256_file(
    MANIFEST
) == EXPECTED["manifest"]

assert sha256_file(
    STATUS
) == EXPECTED["status"]

assert sha256_file(
    BENCH
) == EXPECTED["benchmark"]

assert sha256_file(
    RUNNER
) == EXPECTED["runner"]


print(
    "protocol hash exact: True"
)

print(
    "execution manifest hash exact: True"
)

print(
    "runner status hash exact: True"
)

print(
    "benchmark hash exact: True"
)

print(
    "runner hash exact: True"
)


# =============================================================================
# 2. Protocol semantics
# =============================================================================

protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)


assert protocol[
    "protocol"
] == "full_clean_feature_matched_soh_v3"

assert protocol[
    "status"
] == "FINAL_FROZEN_BEFORE_V3_RUNS"

assert protocol[
    "all_ceiling_decisions_validation_only"
] is True

assert protocol[
    "test_used_for_ceiling_selection"
] is False

assert protocol[
    "v2_exclusion_from_final_aggregation"
] is True

assert protocol[
    "v3_execution_policy"
].startswith(
    "All 15 paired runs are executed from scratch"
)

assert protocol[
    "max_epochs"
] == 5000

assert protocol[
    "early_stopping_patience"
] == 30

assert protocol[
    "planned_paired_run_count"
] == 15

assert protocol[
    "planned_model_fit_count"
] == 30

assert protocol[
    "device"
] == "CPU"


print(
    "protocol semantics exact: True"
)

print(
    "v1/v2 excluded from final aggregation: True"
)

print(
    "ceiling selection used test data: False"
)


# =============================================================================
# 3. Manifest / resume-state semantics
# =============================================================================

manifest = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)

status = json.loads(
    STATUS.read_text(
        encoding="utf-8"
    )
)


assert manifest[
    "execution_contract"
] == "full_clean_soh_v3_execution_v1"

assert manifest[
    "protocol_sha256"
] == EXPECTED["protocol"]

assert manifest[
    "benchmark_sha256"
] == EXPECTED["benchmark"]

assert manifest[
    "runner_sha256"
] == EXPECTED["runner"]

assert manifest[
    "runner_commit"
] == (
    "a10199d966e3cff9e3f037926344f3d7b8d08da2"
)

assert manifest[
    "completed_pair_count"
] == 15

assert manifest[
    "completed_model_fit_count"
] == 30

assert manifest[
    "all_fits_patience_valid"
] is True

assert manifest[
    "all_fits_below_epoch_ceiling"
] is True

assert manifest[
    "epoch_ceiling"
] == 5000

assert manifest[
    "patience"
] == 30

assert manifest[
    "device"
] == "cpu"

assert manifest[
    "test_metric_values_parsed_during_series"
] is False

assert manifest[
    "test_metric_values_surfaced_during_series"
] is False

assert manifest[
    "test_metric_values_used_to_change_protocol"
] is False


assert status[
    "completed_pair_count"
] == 15

assert status[
    "test_metric_values_parsed"
] is False

assert status[
    "test_metric_values_surfaced"
] is False


print(
    "manifest completed pairs:",
    manifest["completed_pair_count"]
)

print(
    "manifest completed model fits:",
    manifest["completed_model_fit_count"]
)

print(
    "runner parsed test metric values:",
    False
)

print(
    "runner surfaced test metric values:",
    False
)


# =============================================================================
# 4. Result-directory inventory
# =============================================================================

observed_dirs = sorted(
    p.name
    for p in V3.iterdir()
    if (
        p.is_dir()
        and p.name.startswith(
            "split"
        )
        and "_train" in p.name
    )
)


expected_dirs = sorted(
    f"split{s}_train{t}"
    for s, t in PLANNED
)


assert observed_dirs == expected_dirs


print(
    "final_v3 pair directories:",
    len(observed_dirs)
)

print(
    "pair directory inventory exact: True"
)


# =============================================================================
# 5. Independently audit every run
#
# metrics.csv CONTENT IS NEVER OPENED.
# Only presence, size and SHA-256 are recorded.
# =============================================================================

records = []

metrics_hashes = []

all_best_epochs = {
    "MLP": [],
    "CAUSAL_GCN": [],
}

all_stop_epochs = {
    "MLP": [],
    "CAUSAL_GCN": [],
}


for split_seed, train_seed in PLANNED:

    run = (
        V3
        / f"split{split_seed}_train{train_seed}"
    )


    required = [
        run / "config.json",
        run / "metrics.csv",
        run / "mlp_history.csv",
        run / "causal_gcn_history.csv",
        run / "mlp_best_state.pt",
        run / "causal_gcn_best_state.pt",
    ]


    assert all(
        p.is_file()
        and p.stat().st_size > 0
        for p in required
    )


    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------

    config = json.loads(
        (
            run
            / "config.json"
        ).read_text(
            encoding="utf-8"
        )
    )


    assert config[
        "benchmark_contract"
    ] == "feature_matched_soh_causal_gcn_v1"

    assert config[
        "smoke"
    ] is False

    assert config[
        "task"
    ] == "cycle-level SOH regression"

    assert config[
        "datasets"
    ] == [
        "HUST",
        "CALCE",
        "MATR",
    ]

    assert config[
        "feature_count"
    ] == 20

    assert config[
        "eligible_observations"
    ] == 300553

    assert config[
        "split_seed"
    ] == split_seed

    assert config[
        "train_seed"
    ] == train_seed

    assert config[
        "hidden_dim"
    ] == 64

    assert config[
        "parameter_count_each_model"
    ] == 5569

    assert config[
        "parameters_equal"
    ] is True

    assert config[
        "initial_weights_equal_for_same_train_seed"
    ] is True

    assert config[
        "future_edges"
    ] is False

    assert config[
        "reverse_edges"
    ] is False

    assert config[
        "cross_cell_edges"
    ] is False

    assert config[
        "cycle_index_feature"
    ] is False

    assert config[
        "target_feature"
    ] is False

    assert config[
        "max_epochs"
    ] == 5000

    assert config[
        "patience"
    ] == 30

    assert config[
        "hidden_dim"
    ] == 64

    assert np.isclose(
        float(
            config[
                "learning_rate"
            ]
        ),
        0.001,
        rtol=0.0,
        atol=0.0,
    )

    assert np.isclose(
        float(
            config[
                "weight_decay"
            ]
        ),
        0.00001,
        rtol=0.0,
        atol=0.0,
    )

    assert config[
        "device"
    ] == "cpu"

    assert config[
        "input_sha256"
    ] == EXPECTED_INPUT_HASHES

    assert config[
        "split_rows"
    ] == EXPECTED_ROWS[
        split_seed
    ]

    assert config[
        "split_cells"
    ] == EXPECTED_CELLS


    # -------------------------------------------------------------------------
    # Histories: train + validation only
    # -------------------------------------------------------------------------

    mh = pd.read_csv(
        run
        / "mlp_history.csv",
        low_memory=False,
    )

    gh = pd.read_csv(
        run
        / "causal_gcn_history.csv",
        low_memory=False,
    )


    for model_name, h in (
        ("MLP", mh),
        ("CAUSAL_GCN", gh),
    ):

        assert list(
            h.columns
        ) == [
            "epoch",
            "train_weighted_mse",
            "validation_balanced_rmse",
        ]

        assert 1 <= len(
            h
        ) < 5000

        epoch = pd.to_numeric(
            h[
                "epoch"
            ],
            errors="raise",
        ).to_numpy(
            dtype=int
        )

        train_loss = pd.to_numeric(
            h[
                "train_weighted_mse"
            ],
            errors="raise",
        ).to_numpy(
            dtype=float
        )

        val = pd.to_numeric(
            h[
                "validation_balanced_rmse"
            ],
            errors="raise",
        ).to_numpy(
            dtype=float
        )


        assert np.array_equal(
            epoch,
            np.arange(
                1,
                len(h) + 1,
            ),
        )

        assert np.isfinite(
            train_loss
        ).all()

        assert np.isfinite(
            val
        ).all()


        history_best_epoch = int(
            epoch[
                np.argmin(
                    val
                )
            ]
        )


        config_best_epoch = int(
            config[
                "model_training"
            ][
                model_name
            ][
                "best_epoch"
            ]
        )


        stop_epoch = int(
            epoch[-1]
        )


        assert (
            history_best_epoch
            == config_best_epoch
        )

        assert (
            stop_epoch
            - history_best_epoch
            == 30
        )

        assert stop_epoch < 5000


        all_best_epochs[
            model_name
        ].append(
            history_best_epoch
        )

        all_stop_epochs[
            model_name
        ].append(
            stop_epoch
        )


    # -------------------------------------------------------------------------
    # Saved states: architecture / finite-value audit
    # -------------------------------------------------------------------------

    for model_name, filename in (
        (
            "MLP",
            "mlp_best_state.pt",
        ),
        (
            "CAUSAL_GCN",
            "causal_gcn_best_state.pt",
        ),
    ):

        state = torch.load(
            run / filename,
            map_location="cpu",
            weights_only=True,
        )


        expected_shapes = {
            "fc1.weight": (64, 20),
            "fc1.bias": (64,),
            "fc2.weight": (64, 64),
            "fc2.bias": (64,),
            "out.weight": (1, 64),
            "out.bias": (1,),
        }


        assert set(
            state.keys()
        ) == set(
            expected_shapes.keys()
        )


        parameter_count = 0


        for key, shape in expected_shapes.items():

            tensor = state[
                key
            ]

            assert tuple(
                tensor.shape
            ) == shape

            assert torch.isfinite(
                tensor
            ).all()

            parameter_count += (
                tensor.numel()
            )


        assert parameter_count == 5569


    # -------------------------------------------------------------------------
    # IMPORTANT:
    # metrics.csv is NOT parsed.
    # -------------------------------------------------------------------------

    metrics_path = (
        run
        / "metrics.csv"
    )

    metrics_hash = sha256_file(
        metrics_path
    )

    metrics_hashes.append(
        metrics_hash
    )


    records.append({
        "split_seed":
            split_seed,

        "train_seed":
            train_seed,

        "MLP_best":
            int(
                config[
                    "model_training"
                ][
                    "MLP"
                ][
                    "best_epoch"
                ]
            ),

        "MLP_stop":
            int(
                mh[
                    "epoch"
                ].iloc[-1]
            ),

        "GCN_best":
            int(
                config[
                    "model_training"
                ][
                    "CAUSAL_GCN"
                ][
                    "best_epoch"
                ]
            ),

        "GCN_stop":
            int(
                gh[
                    "epoch"
                ].iloc[-1]
            ),
    })


# =============================================================================
# 6. Cross-run pairing invariants
# =============================================================================

assert len(
    records
) == 15


assert len({
    (
        r[
            "split_seed"
        ],
        r[
            "train_seed"
        ],
    )
    for r in records
}) == 15


for split_seed in range(5):

    subset = [
        r
        for r in records
        if r[
            "split_seed"
        ] == split_seed
    ]

    assert len(
        subset
    ) == 3

    assert sorted(
        r[
            "train_seed"
        ]
        for r in subset
    ) == [
        0,
        1,
        2,
    ]


print(
    "run configs audited:",
    len(records),
    "/ 15"
)

print(
    "train/validation histories audited:",
    len(records) * 2,
    "/ 30"
)

print(
    "saved model states audited:",
    len(records) * 2,
    "/ 30"
)

print(
    "all fits stop exactly 30 epochs after best:",
    True
)

print(
    "all fits below epoch 5000:",
    True
)

print(
    "all parameter tensors finite:",
    True
)

print(
    "all saved models have 5569 parameters:",
    True
)

print(
    "all 15 split/train pairs unique:",
    True
)


# =============================================================================
# 7. Convergence range only -- no scores
# =============================================================================

print(
    "MLP best-epoch range:",
    (
        min(
            all_best_epochs[
                "MLP"
            ]
        ),
        max(
            all_best_epochs[
                "MLP"
            ]
        ),
    )
)

print(
    "GCN best-epoch range:",
    (
        min(
            all_best_epochs[
                "CAUSAL_GCN"
            ]
        ),
        max(
            all_best_epochs[
                "CAUSAL_GCN"
            ]
        ),
    )
)

print(
    "MLP stop-epoch maximum:",
    max(
        all_stop_epochs[
            "MLP"
        ]
    )
)

print(
    "GCN stop-epoch maximum:",
    max(
        all_stop_epochs[
            "CAUSAL_GCN"
        ]
    )
)


# =============================================================================
# 8. Metrics isolation evidence
# =============================================================================

assert len(
    metrics_hashes
) == 15

assert all(
    len(x) == 64
    for x in metrics_hashes
)


print(
    "metrics files present:",
    len(metrics_hashes),
    "/ 15"
)

print(
    "metrics.csv contents parsed by audit:",
    False
)

print(
    "test metric values read by audit:",
    False
)


# =============================================================================
# Final
# =============================================================================

print()
print(
    "FINAL_V3_EXECUTION_AUDIT_PASS=True"
)
