"""
Strictly causal sequence baseline for revision_2026.

Purpose
-------
Provide a trainable temporal-convolution control for comparison with the
parameter-matched MLP and causal graph model.

Input
-----
Canonical 20-feature cycle observations.

For observation t within one cell, construct exactly three chronological
positions:

    [t-2, t-1, t]

At the beginning of a cell, unavailable historical positions are filled by
replicating the FIRST valid observation of that same cell.

Examples
--------
t=0: [x0, x0, x0]
t=1: [x0, x0, x1]
t=2: [x0, x1, x2]

No future observation and no other cell can enter the sequence.

Architecture
------------
Conv1d(20 -> 49, kernel_size=3)
ReLU
Conv1d(49 -> 49, kernel_size=1)
ReLU
Linear(49 -> 1)

The three-cycle input is already an explicitly causal window, therefore the
kernel-size-3 convolution produces one current-time representation without
additional padding.

Learnable parameter count: 5489.
"""

import numpy as np
import torch
from torch import nn


TCN_CONTRACT = "causal_tcn_three_cycle_v1"

FEATURE_COUNT = 20
SEQUENCE_LENGTH = 3
CHANNELS = 49


def build_causal_three_cycle_sequences(x):
    """
    Construct [N, 3, F] causal windows for ONE chronological cell.

    Parameters
    ----------
    x : array-like, shape [N, F]

    Returns
    -------
    seq : np.ndarray, shape [N, 3, F]

    Causal contract
    ---------------
    output[t] depends only on input[0:t+1].

    Boundary handling
    -----------------
    Missing history is filled by replicating x[0].
    """
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    if x.ndim != 2:
        raise ValueError(
            "Expected [time, feature] array"
        )

    if x.shape[0] < 1:
        raise ValueError(
            "Cell trajectory must contain at least one observation"
        )

    if x.shape[1] != FEATURE_COUNT:
        raise ValueError(
            f"Expected {FEATURE_COUNT} features"
        )

    if not np.isfinite(x).all():
        raise ValueError(
            "Input contains non-finite values"
        )

    n = x.shape[0]

    seq = np.empty(
        (
            n,
            SEQUENCE_LENGTH,
            FEATURE_COUNT,
        ),
        dtype=np.float64,
    )

    first = x[0]

    for t in range(n):

        t2 = max(
            0,
            t - 2,
        )

        t1 = max(
            0,
            t - 1,
        )

        seq[t, 0] = (
            first
            if t < 2
            else x[t2]
        )

        seq[t, 1] = (
            first
            if t < 1
            else x[t1]
        )

        seq[t, 2] = x[t]

    return seq


class CausalThreeCycleTCN(nn.Module):
    """
    Three-cycle causal TCN baseline.

    Expected input shape:
        [batch, sequence_length=3, features=20]

    Internally transformed to Conv1d layout:
        [batch, channels/features=20, time=3]
    """

    def __init__(
        self,
        feature_count=FEATURE_COUNT,
        channels=CHANNELS,
    ):
        super().__init__()

        if int(feature_count) != FEATURE_COUNT:
            raise ValueError(
                f"feature_count must be {FEATURE_COUNT}"
            )

        if int(channels) != CHANNELS:
            raise ValueError(
                f"channels must be {CHANNELS}"
            )

        self.temporal = nn.Conv1d(
            in_channels=FEATURE_COUNT,
            out_channels=CHANNELS,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=True,
        )

        self.pointwise = nn.Conv1d(
            in_channels=CHANNELS,
            out_channels=CHANNELS,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.relu = nn.ReLU()

        self.out = nn.Linear(
            CHANNELS,
            1,
            bias=True,
        )


    def forward(self, sequence):
        if sequence.ndim != 3:
            raise ValueError(
                "Expected [batch, time, feature]"
            )

        if sequence.shape[1] != 3:
            raise ValueError(
                "Expected sequence length 3"
            )

        if sequence.shape[2] != 20:
            raise ValueError(
                "Expected 20 features"
            )

        # [B,T,F] -> [B,F,T]
        z = sequence.transpose(
            1,
            2,
        )

        # Kernel size 3 over exactly t-2,t-1,t.
        z = self.relu(
            self.temporal(
                z
            )
        )

        # Time dimension is now 1.
        if z.shape[-1] != 1:
            raise RuntimeError(
                "Unexpected temporal output length"
            )

        z = self.relu(
            self.pointwise(
                z
            )
        )

        # [B,C,1] -> [B,C]
        z = z[
            :,
            :,
            0,
        ]

        y = self.out(
            z
        ).squeeze(
            -1
        )

        return y


def parameter_count(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
