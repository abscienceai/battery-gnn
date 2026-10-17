import ast
import os
import hashlib
import importlib.util
import inspect
import json
import pickle
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from torch import nn

R1 = Path.home() / "alper/battery_revision_2026"
R2 = Path.home() / "alper/EST_R2_field_validation_2026-10"
SRC = R1 / "repo/battery-gnn/scripts/revision_2026"
FINAL = R1 / "results/revision_2026/feature_matched_soh/final_v3/split0_train0"

DATA = R1 / "data/processed/revision_2026"
CANON = DATA / "canonical_tasks/canonical_cycle_supervised_table.csv"
ASSIGN = DATA / "splits_and_scalers/cell_split_assignments.csv"
SCALERS = DATA / "splits_and_scalers/train_only_scaler_parameters.csv"

FEATURES = [
    "v_min","v_q10","v_q25","v_median","v_q75",
    "v_q90","v_max","v_mean","v_std","v_phase_slope",
    "i_min","i_q10","i_q25","i_median","i_q75",
    "i_q90","i_max","i_mean","i_std","i_phase_slope"
]

N = 40
SEEDS = [0, 1, 2]

# Field increments are proxies, NOT measured sensor errors.
pool_file = R2 / "reports/r2_field_increment_proxies.npz"
assert pool_file.is_file(), pool_file

with np.load(pool_file, allow_pickle=False) as z:
    profiles = {
        "Li": z["real_world_02-03"].astype(float) / 1000,
        "Xie": z["Xie_dense"].astype(float) / 1000,
    }

for source, values in profiles.items():
    assert len(values) >= 100 and np.isfinite(values).all()

    values = values - values.mean()
    cap = float(np.percentile(np.abs(values), 95))

    assert cap > 0
    profiles[source] = np.clip(values, -cap, cap)

    print(
        f"PROFILE {source}: n={len(values)}, "
        f"synthetic jitter cap={cap*1000:.4f} mV",
        flush=True
    )

# Load the original, trusted R1 raw-cycle adapter.
source_file = SRC / "14_raw_cycle_feature_generator.py"
spec = importlib.util.spec_from_file_location(
    "r1_raw_adapter", source_file
)
assert spec and spec.loader

adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
kernel = adapter.load_kernel()

# Select held-out HUST cell from original split 0.
assignment = pd.read_csv(
    ASSIGN,
    usecols=["split_seed","split_group","split"]
)

test_groups = set(
    assignment.loc[
        (assignment.split_seed == 0)
        & (assignment.split == "test"),
        "split_group"
    ].astype(str)
)

cols = [
    "dataset","split_group","observation_index",
    "soh_task_eligible","SOH",
    "feature_source_cell_id","feature_source_member",
    "feature_raw_cycle_position_zero_based"
] + FEATURES

canon = pd.read_csv(CANON, usecols=cols, low_memory=False)

eligible = canon.soh_task_eligible.astype(
    str
).str.lower().eq("true")

assert int(eligible.sum()) == 300553

test = canon.loc[
    eligible
    & canon.split_group.astype(str).isin(test_groups)
    & (canon.dataset == "HUST")
].copy()

counts = test.groupby("feature_source_cell_id").size()

candidates = sorted(
    str(k) for k, count in counts.items()
    if count >= N
)

assert candidates, "No suitable HUST test cell found"

rank = int(os.environ.get("R2_CELL_RANK", "0"))
assert 0 <= rank < len(candidates), (
    f"Cell rank {rank} unavailable; "
    f"only {len(candidates)} eligible cells"
)
cell_id = candidates[rank]

target = (
    test.loc[
        test.feature_source_cell_id.astype(str) == cell_id
    ]
    .sort_values("observation_index", kind="mergesort")
    .head(N)
    .copy()
)

positions = target[
    "feature_raw_cycle_position_zero_based"
].astype(int).tolist()

assert len(set(positions)) == N

members = target.feature_source_member.astype(
    str
).unique().tolist()

assert len(members) == 1

# Trusted laboratory archive already used in R1.
with ZipFile(adapter.HUST_ZIP) as z:
    with z.open(members[0]) as f:
        obj = pickle.load(f)

assert len(obj) == 1

raw_id = next(iter(obj))
assert str(raw_id) == cell_id

print(
    f"TEST CELL: {cell_id}; "
    f"observations={N}; source={members[0]}",
    flush=True
)

class KernelProxy:
    def __init__(self, base):
        self.base = base
        self.pos = {p:j for j,p in enumerate(positions)}
        self.clean = np.full((N,20), np.nan)
        self.changed = {
            (source,seed):np.full((N,20),np.nan)
            for source in profiles for seed in SEEDS
        }
        self.seen = set()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def extract_primary_cycle_features(
        self, voltage_V, current_A, time_s, **kwargs
    ):
        clean, qc = self.base.extract_primary_cycle_features(
            voltage_V, current_A, time_s, **kwargs
        )

        frame = inspect.currentframe().f_back
        row = frame.f_locals.get("row") if frame else None

        if not isinstance(row, dict):
            raise RuntimeError("R1 adapter callback missing")

        assert str(row["raw_cell_id"]) == cell_id

        pos = int(row["raw_cycle_position_zero_based"])

        if pos not in self.pos:
            return clean, qc

        assert pos not in self.seen
        self.seen.add(pos)

        j = self.pos[pos]
        self.clean[j] = [clean[f] for f in FEATURES]

        V = np.asarray(voltage_V, dtype=float)

        for (source,seed), arr in self.changed.items():
            token = f"{cell_id}|{pos}|{source}|{seed}".encode()

            seed_value = int.from_bytes(
                hashlib.sha256(token).digest()[:8], "little"
            )

            rng = np.random.default_rng(seed_value)

            # Explicit synthetic stressor:
            # sampled residual increments are used as
            # independent voltage-level jitter.
            jitter = rng.normal(
                0.0, float(np.std(profiles[source])), size=len(V)
            )

            altered, _ = self.base.extract_primary_cycle_features(
                V + jitter, current_A, time_s, **kwargs
            )

            arr[j] = [altered[f] for f in FEATURES]

        return clean, qc

proxy = KernelProxy(kernel)

adapter.process_hust_cell(
    source_member=members[0],
    raw_cell_id=raw_id,
    cell=obj[raw_id],
    kernel=proxy,
    kernel_hash=hashlib.sha256(
        (SRC/"cycle_feature_kernel.py").read_bytes()
    ).hexdigest()
)

assert len(proxy.seen) == N

expected = target[FEATURES].to_numpy(float)

max_difference = float(
    np.max(np.abs(proxy.clean-expected))
)

if not np.allclose(
    proxy.clean, expected, atol=1e-8, rtol=1e-8
):
    raise RuntimeError(
        f"RAW ROUNDTRIP FAILED: {max_difference}"
    )

print(
    f"RAW FEATURE ROUNDTRIP: PASS "
    f"(max difference={max_difference:.3e})",
    flush=True
)

# Original train-only scaler.
scalers = pd.read_csv(SCALERS)

s = scalers.loc[
    (scalers.split_seed == 0)
    & (scalers.scaler_scope == "pooled_supervised")
].set_index("feature").loc[FEATURES]

assert len(s) == 20
assert set(s.fit_split.astype(str)) == {"train_only"}

mean = s["mean"].to_numpy(float)
scale = s["scale"].to_numpy(float)

# Reuse original architecture definitions.
path = SRC / "24_feature_matched_soh_benchmark.py"
tree = ast.parse(path.read_text())

wanted = {
    "causal_mean",
    "FeatureMatchedMLP",
    "FeatureMatchedCausalGCN"
}

nodes = [
    x for x in tree.body
    if isinstance(x, (ast.FunctionDef,ast.ClassDef))
    and x.name in wanted
]

assert {x.name for x in nodes} == wanted

ns = {"np":np, "torch":torch, "nn":nn}

module = ast.fix_missing_locations(
    ast.Module(body=nodes, type_ignores=[])
)
exec(compile(module, str(path), "exec"), ns)

models = {}

for name, class_name, filename in [
    ("MLP","FeatureMatchedMLP","mlp_best_state.pt"),
    ("GCN","FeatureMatchedCausalGCN","causal_gcn_best_state.pt")
]:
    model = ns[class_name](input_dim=20, hidden_dim=64)

    model.load_state_dict(
        torch.load(
            FINAL/filename,
            map_location="cpu",
            weights_only=True
        ),
        strict=True
    )

    model.eval()
    models[name] = model

previous = torch.from_numpy(
    np.r_[0, np.arange(N-1, dtype=np.int64)]
)

y = target["SOH"].to_numpy(float)
torch.set_num_threads(2)

results = []

for model_name, model in models.items():

    def evaluate(x):
        x = torch.from_numpy(
            ((x-mean)/scale).astype(np.float32)
        )

        with torch.inference_mode():
            if model_name == "MLP":
                pred = model(x)
            else:
                pred = model(x, previous)

        error = pred.numpy().reshape(-1) - y

        return (
            float(np.sqrt(np.mean(error**2))),
            float(np.mean(np.abs(error)))
        )

    clean_rmse, clean_mae = evaluate(proxy.clean)

    for (source,seed), changed in proxy.changed.items():
        assert np.isfinite(changed).all()

        changed_rmse, changed_mae = evaluate(changed)

        results.append({
            "source":source,
            "seed":seed,
            "model":model_name,
            "HUST_cell":cell_id,
            "n_cycles":N,
            "clean_rmse":clean_rmse,
            "perturbed_rmse":changed_rmse,
            "delta_rmse":changed_rmse-clean_rmse,
            "clean_mae":clean_mae,
            "perturbed_mae":changed_mae,
            "delta_mae":changed_mae-clean_mae,
            "roundtrip_pass":True,
            "verified_sensor_fault":False
        })

        print(
            f"{source} | seed {seed} | {model_name} | "
            f"RMSE {clean_rmse:.8f} -> "
            f"{changed_rmse:.8f}",
            flush=True
        )

out = R2 / f"reports/r2_gaussian_control_rank{rank}.csv"
pd.DataFrame(results).to_csv(out, index=False)

metadata = {
    "experiment":"exploratory field-scale-informed synthetic jitter",
    "observed_real_sensor_faults":False,
    "independent_voltage_level_jitter":True,
    "temporal_field_dependence_preserved":False,
    "single_test_cell":True,
    "retraining":False,
    "publishable_as_real_fault_validation":False
}

out.with_suffix(".json").write_text(
    json.dumps(metadata, indent=2) + "\n"
)

print("\nRAW PILOT COMPLETE")
print("RESULTS:", out)
print("IMPORTANT: SYNTHETIC SENSITIVITY, NOT REAL FAULT VALIDATION")
