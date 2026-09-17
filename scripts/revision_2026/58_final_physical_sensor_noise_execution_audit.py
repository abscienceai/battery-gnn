from pathlib import Path
import hashlib
import json


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo" / "battery-gnn"

ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "physical_sensor_noise"
    / "final_physical_v1"
)

EVAL = ROOT / "evaluation"

RUNS = EVAL / "runs"

MANIFEST = (
    EVAL
    / "physical_noise_execution_manifest.json"
)

RUNNER = (
    REPO
    / "scripts"
    / "revision_2026"
    / "57_final_physical_sensor_noise_evaluation.py"
)

FEATURE_CLOSURE = (
    ROOT
    / "corrupted_features"
    / "corruption_feature_closure.json"
)


EXPECTED_PROTOCOL = (
    "7ae0836fe69c4b14a1f93a094b2d42bb96209e302c8262f7b8bd8625baf78bfc"
)

EXPECTED_FEATURE_CLOSURE = (
    "382ea97c8f1d8f219436dd5071bb8fcbd63811f24e76b8c887e8efba4e54675f"
)


def sha(path):
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


assert sha(
    FEATURE_CLOSURE
) == EXPECTED_FEATURE_CLOSURE


m = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)


assert m[
    "contract"
] == "final_physical_sensor_noise_execution_v1"

assert m[
    "status"
] == "COMPLETE_BEFORE_VALUE_LEVEL_ANALYSIS"

assert m[
    "protocol_sha256"
] == EXPECTED_PROTOCOL

assert m[
    "feature_closure_sha256"
] == EXPECTED_FEATURE_CLOSURE

assert m[
    "runner_sha256"
] == sha(
    RUNNER
)

assert m[
    "device"
] == "cpu"

assert m[
    "planned_runs"
] == 30

assert m[
    "completed_runs"
] == 30

assert m[
    "conditions_per_run"
] == 66

assert m[
    "domains_per_condition"
] == 4

assert m[
    "metric_rows_per_run"
] == 264

assert m[
    "expected_total_metric_rows"
] == 7920

assert m[
    "training_reexecuted"
] is False

assert m[
    "state_files_reused"
] is True

assert m[
    "train_only_scalers_reused"
] is True

assert m[
    "scaler_refit"
] is False

assert m[
    "feature_generation_reexecuted"
] is False

assert m[
    "clean_raw_feature_barrier_checked"
] is True

assert m[
    "metric_values_surfaced_during_execution"
] is False

assert m[
    "metric_values_used_to_change_protocol"
] is False

assert m[
    "analysis_performed"
] is False


records = m[
    "records"
]

assert len(
    records
) == 30


assert len({
    (
        x[
            "split_seed"
        ],
        x[
            "train_seed"
        ],
        x[
            "model"
        ],
    )
    for x in records
}) == 30


csvs = sorted(
    RUNS.glob(
        "*_physical_test_metrics.csv"
    )
)

metas = sorted(
    RUNS.glob(
        "*_evaluation.json"
    )
)


assert len(
    csvs
) == 30

assert len(
    metas
) == 30


record_by_csv = {
    Path(
        x[
            "metric_file"
        ]
    ).resolve():
        x
    for x in records
}


expected_header = (
    "split_seed,train_seed,model,"
    "condition_index,condition_id,"
    "scenario,level,corruption_seed,"
    "severity_axis,domain,n_rows,"
    "rmse,mae,r2"
)


for csv_path in csvs:

    rp = csv_path.resolve()

    assert rp in record_by_csv

    rec = record_by_csv[
        rp
    ]

    assert sha(
        csv_path
    ) == rec[
        "metric_file_sha256"
    ]


    # Audit structure only. Never parse metric values.
    with csv_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        header = f.readline().rstrip(
            "\n\r"
        )

        line_count = 1

        for _ in f:
            line_count += 1


    assert header == expected_header

    assert line_count == 265


    meta_path = Path(
        rec[
            "evaluation_meta"
        ]
    )

    assert meta_path.is_file()

    assert sha(
        meta_path
    ) == rec[
        "evaluation_meta_sha256"
    ]


    meta = json.loads(
        meta_path.read_text(
            encoding="utf-8"
        )
    )


    assert meta[
        "contract"
    ] == "physical_noise_test_evaluation_run_v1"

    assert meta[
        "protocol_sha256"
    ] == EXPECTED_PROTOCOL

    assert meta[
        "feature_closure_sha256"
    ] == EXPECTED_FEATURE_CLOSURE

    assert meta[
        "runner_sha256"
    ] == sha(
        RUNNER
    )

    assert meta[
        "parameter_count"
    ] == 5569

    assert meta[
        "condition_count"
    ] == 66

    assert meta[
        "domain_count"
    ] == 4

    assert meta[
        "metric_row_count"
    ] == 264

    assert meta[
        "all_metrics_finite"
    ] is True

    assert meta[
        "training_reexecuted"
    ] is False

    assert meta[
        "test_only_evaluation"
    ] is True

    assert meta[
        "train_only_scaler_reused"
    ] is True

    assert meta[
        "scaler_refit"
    ] is False

    assert meta[
        "clean_raw_feature_barrier_passed"
    ] is True

    assert meta[
        "metric_values_surfaced"
    ] is False

    assert sha(
        Path(
            meta[
                "state_file"
            ]
        )
    ) == meta[
        "state_sha256"
    ]


# Runner source must contain no training pathway.
source = RUNNER.read_text(
    encoding="utf-8"
)

for forbidden in [
    ".backward(",
    "optimizer.step(",
    "AdamW(",
    "loss.backward",
]:
    assert forbidden not in source, forbidden


analysis_dir = (
    ROOT
    / "analysis"
)

if analysis_dir.exists():
    assert not list(
        analysis_dir.glob("*")
    )


print("=" * 80)
print("FINAL PHYSICAL SENSOR NOISE EXECUTION AUDIT")
print("=" * 80)

print(
    "feature closure exact:",
    True
)

print(
    "execution runs:",
    30
)

print(
    "metric files:",
    len(
        csvs
    )
)

print(
    "evaluation metadata files:",
    len(
        metas
    )
)

print(
    "conditions per run:",
    66
)

print(
    "domains per condition:",
    4
)

print(
    "metric rows total:",
    7920
)

print(
    "state parameter count:",
    5569
)

print(
    "training reexecuted:",
    False
)

print(
    "scaler refit:",
    False
)

print(
    "clean raw feature barrier:",
    True
)

print(
    "metric values parsed by audit:",
    False
)

print(
    "metric values surfaced:",
    False
)

print(
    "analysis already performed:",
    False
)

print(
    "execution manifest SHA256:",
    sha(
        MANIFEST
    )
)

print()
print(
    "FINAL_PHYSICAL_SENSOR_NOISE_EXECUTION_AUDIT_PASS=True"
)
