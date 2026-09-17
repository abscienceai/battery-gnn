#!/usr/bin/env python3

"""
Shared leakage-controlled cycle-feature kernel for revision_2026.

PRIMARY MODEL FEATURES
----------------------
Derived only from:
    voltage [V]
    current [A]
    within-cycle time [s]

Absolute cycle duration, cycle index, capacity, Qd/Qc, SOH, RUL,
EOL and lifetime labels are NOT primary model features.

Time is used only to:
    1. validate chronological sample order;
    2. construct normalized within-cycle phase [0, 1].

This deliberately prevents absolute duration from becoming an
implicit constant-current capacity measurement.
"""

from __future__ import annotations

import numpy as np


PRIMARY_FEATURE_NAMES = (
    "v_min",
    "v_q10",
    "v_q25",
    "v_median",
    "v_q75",
    "v_q90",
    "v_max",
    "v_mean",
    "v_std",
    "v_phase_slope",

    "i_min",
    "i_q10",
    "i_q25",
    "i_median",
    "i_q75",
    "i_q90",
    "i_max",
    "i_mean",
    "i_std",
    "i_phase_slope",
)


FORBIDDEN_PRIMARY_NAME_TOKENS = (
    "capacity",
    "soh",
    "rul",
    "eol",
    "cycle_life",
    "qd",
    "qc",
    "dq",
    "duration",
    "sample_count",
    "cycle_number",
    "observation_index",
)


def _as_1d_float(x):
    return (
        np.asarray(
            x,
            dtype=np.float64
        )
        .reshape(-1)
    )


def _signal_features(
    signal,
    phase,
    prefix,
):
    signal = _as_1d_float(
        signal
    )

    phase = _as_1d_float(
        phase
    )

    q10, q25, q50, q75, q90 = (
        np.quantile(
            signal,
            [
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
            ],
        )
    )

    # Least-squares slope against normalized phase.
    #
    # Because phase is dimensionless [0,1], this is invariant
    # to whether the original time coordinate was expressed in
    # seconds, minutes, etc., provided the physical ordering is
    # unchanged.
    p_centered = (
        phase
        - np.mean(phase)
    )

    s_centered = (
        signal
        - np.mean(signal)
    )

    denom = float(
        np.dot(
            p_centered,
            p_centered
        )
    )

    if denom <= 0:
        raise ValueError(
            "Degenerate normalized phase."
        )

    slope = float(
        np.dot(
            p_centered,
            s_centered
        )
        / denom
    )

    return {
        f"{prefix}_min":
            float(
                np.min(signal)
            ),

        f"{prefix}_q10":
            float(q10),

        f"{prefix}_q25":
            float(q25),

        f"{prefix}_median":
            float(q50),

        f"{prefix}_q75":
            float(q75),

        f"{prefix}_q90":
            float(q90),

        f"{prefix}_max":
            float(
                np.max(signal)
            ),

        f"{prefix}_mean":
            float(
                np.mean(signal)
            ),

        f"{prefix}_std":
            float(
                np.std(
                    signal,
                    ddof=0
                )
            ),

        f"{prefix}_phase_slope":
            slope,
    }


def extract_primary_cycle_features(
    voltage_V,
    current_A,
    time_s,
    *,
    min_valid_samples=20,
    time_tolerance_s=1e-9,
):
    """
    Return:
        features: exactly the 20 primary model features.
        qc: non-model quality-control metadata.

    The arrays must already use canonical physical units and the
    source/canonical chronological ordering supplied by each
    dataset adapter.

    This function never sorts samples silently.
    """

    V = _as_1d_float(
        voltage_V
    )

    I = _as_1d_float(
        current_A
    )

    t = _as_1d_float(
        time_s
    )

    if not (
        len(V)
        == len(I)
        == len(t)
    ):
        raise ValueError(
            "V/I/t length mismatch: "
            f"{len(V)}/{len(I)}/{len(t)}"
        )

    n_raw = len(V)

    finite = (
        np.isfinite(V)
        & np.isfinite(I)
        & np.isfinite(t)
    )

    V = V[finite]
    I = I[finite]
    t = t[finite]

    n_valid = len(V)

    if n_valid < min_valid_samples:
        raise ValueError(
            "Too few valid samples: "
            f"{n_valid}"
        )

    dt = np.diff(t)

    backward = int(
        np.sum(
            dt < -time_tolerance_s
        )
    )

    if backward:
        raise ValueError(
            "Non-monotonic within-cycle time: "
            f"{backward} backward steps"
        )

    elapsed = (
        t
        - t[0]
    )

    duration_s = float(
        elapsed[-1]
    )

    if not (
        np.isfinite(duration_s)
        and duration_s > 0
    ):
        raise ValueError(
            "Non-positive cycle duration."
        )

    phase = (
        elapsed
        / duration_s
    )

    features = {}

    features.update(
        _signal_features(
            V,
            phase,
            "v",
        )
    )

    features.update(
        _signal_features(
            I,
            phase,
            "i",
        )
    )

    if tuple(
        features.keys()
    ) != PRIMARY_FEATURE_NAMES:
        raise AssertionError(
            "Primary feature schema changed."
        )

    for name, value in (
        features.items()
    ):
        if not np.isfinite(value):
            raise ValueError(
                f"Non-finite feature: {name}"
            )

    qc = {
        # QC metadata only. These fields must NOT be included in
        # the model feature matrix.
        "raw_sample_count":
            int(n_raw),

        "valid_sample_count":
            int(n_valid),

        "dropped_nonfinite_samples":
            int(
                n_raw - n_valid
            ),

        "duration_s_qc_only":
            duration_s,

        "duplicate_or_zero_dt_count":
            int(
                np.sum(
                    np.abs(dt)
                    <= time_tolerance_s
                )
            ),

        "backward_time_step_count":
            backward,
    }

    return features, qc


def validate_primary_schema():
    if len(
        PRIMARY_FEATURE_NAMES
    ) != 20:
        raise AssertionError(
            "Expected exactly 20 primary features."
        )

    lowered = [
        x.lower()
        for x in PRIMARY_FEATURE_NAMES
    ]

    for name in lowered:
        for token in (
            FORBIDDEN_PRIMARY_NAME_TOKENS
        ):
            if token in name:
                raise AssertionError(
                    "Forbidden primary feature name: "
                    f"{name} contains {token}"
                )

    if len(
        set(PRIMARY_FEATURE_NAMES)
    ) != len(
        PRIMARY_FEATURE_NAMES
    ):
        raise AssertionError(
            "Duplicate primary feature names."
        )

    return True


validate_primary_schema()
