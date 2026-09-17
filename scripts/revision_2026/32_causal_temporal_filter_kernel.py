"""
Leakage-free causal temporal controls for revision_2026.

This module contains NO target logic, train/test logic, scaling logic,
or model-fitting logic.

All transformations operate on ONE chronological cell trajectory at a time.
Future samples are never accessed.
"""

import numpy as np


FILTER_CONTRACT = "causal_temporal_controls_v1"


def _as_2d_float64(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    if x.ndim != 2:
        raise ValueError(
            "Expected 2D [time, feature] array"
        )

    if len(x) < 1:
        raise ValueError(
            "Empty trajectory"
        )

    if not np.isfinite(x).all():
        raise ValueError(
            "Input contains non-finite values"
        )

    return x


def identity(x):
    """
    Raw unsmoothed control.
    """
    x = _as_2d_float64(x)

    return x.copy()


def trailing_mean(x, window=3):
    """
    Strictly causal trailing moving average.

    y_t = mean(x[max(0,t-window+1) : t+1])
    """
    x = _as_2d_float64(x)

    window = int(window)

    if window < 1:
        raise ValueError(
            "window must be >= 1"
        )

    n, d = x.shape

    out = np.empty_like(x)

    cs = np.vstack(
        [
            np.zeros(
                (1, d),
                dtype=np.float64,
            ),
            np.cumsum(
                x,
                axis=0,
                dtype=np.float64,
            ),
        ]
    )

    for t in range(n):
        start = max(
            0,
            t - window + 1,
        )

        count = (
            t - start + 1
        )

        out[t] = (
            cs[t + 1]
            - cs[start]
        ) / count

    return out


def causal_binomial_lowpass3(x):
    """
    Fixed causal 3-tap low-pass FIR.

    Full-history form:
        y_t = (x_t + 2*x_{t-1} + x_{t-2}) / 4

    At the first two samples, only available causal taps are used
    and weights are renormalized.

    This has the same maximum temporal receptive field
    {t, t-1, t-2} as the two-layer causal GCN benchmark.
    """
    x = _as_2d_float64(x)

    n, _ = x.shape

    out = np.empty_like(x)

    for t in range(n):

        if t == 0:
            out[t] = x[t]

        elif t == 1:
            out[t] = (
                x[t]
                + 2.0 * x[t - 1]
            ) / 3.0

        else:
            out[t] = (
                x[t]
                + 2.0 * x[t - 1]
                + x[t - 2]
            ) / 4.0

    return out


def _endpoint_polynomial_weights(
    n_points,
    polyorder,
):
    """
    One-sided polynomial least-squares endpoint weights.

    This is the causal/end-point analogue of a Savitzky-Golay
    smoother: fit a polynomial to the trailing window and evaluate
    it at the CURRENT (last) point.

    No centered/future observations are used.
    """
    n_points = int(
        n_points
    )

    polyorder = int(
        polyorder
    )

    if n_points < 1:
        raise ValueError(
            "n_points must be >= 1"
        )

    if polyorder < 0:
        raise ValueError(
            "polyorder must be >= 0"
        )

    degree = min(
        polyorder,
        n_points - 1,
    )

    # Coordinates end at zero:
    # oldest ... current = negative ... 0.
    z = np.arange(
        -(n_points - 1),
        1,
        dtype=np.float64,
    )

    # Vandermonde:
    # columns = 1, z, z^2, ...
    A = np.vander(
        z,
        N=degree + 1,
        increasing=True,
    )

    # Evaluate fitted polynomial at z=0.
    evaluation = np.zeros(
        degree + 1,
        dtype=np.float64,
    )

    evaluation[0] = 1.0

    # beta = pinv(A) @ y
    # estimate_at_zero = evaluation @ beta
    weights = (
        evaluation
        @ np.linalg.pinv(A)
    )

    return weights.astype(
        np.float64,
        copy=False,
    )


def causal_savgol_endpoint(
    x,
    window=5,
    polyorder=2,
):
    """
    Strictly causal one-sided/end-point Savitzky-Golay analogue.

    For every t:
      - use only the trailing min(window, t+1) samples,
      - fit polynomial of degree min(polyorder, n_points-1),
      - evaluate at CURRENT endpoint.

    The transform never observes t+1 or later.
    """
    x = _as_2d_float64(x)

    window = int(window)
    polyorder = int(polyorder)

    if window < 1:
        raise ValueError(
            "window must be >= 1"
        )

    if polyorder < 0:
        raise ValueError(
            "polyorder must be >= 0"
        )

    n, _ = x.shape

    out = np.empty_like(x)

    # Only window different prefix configurations are required.
    weight_cache = {}

    for points in range(
        1,
        window + 1,
    ):
        weight_cache[points] = (
            _endpoint_polynomial_weights(
                points,
                polyorder,
            )
        )

    for t in range(n):

        points = min(
            window,
            t + 1,
        )

        start = (
            t - points + 1
        )

        segment = x[
            start:t + 1
        ]

        weights = (
            weight_cache[
                points
            ]
        )

        out[t] = (
            weights
            @ segment
        )

    return out


def causal_kalman_local_level(
    x,
    process_variance=1e-3,
    measurement_variance=1e-2,
    initial_variance=1.0,
):
    """
    Independent scalar local-level Kalman filter per feature.

    State:
        z_t = z_{t-1} + w_t
        x_t = z_t + v_t

    Only x_0..x_t are available when producing output t.

    q/r are explicit parameters. This kernel does NOT select them.
    Any benchmark selection must be performed using training /
    validation information only.
    """
    x = _as_2d_float64(x)

    q = float(
        process_variance
    )

    r = float(
        measurement_variance
    )

    p0 = float(
        initial_variance
    )

    if q < 0:
        raise ValueError(
            "process_variance must be >= 0"
        )

    if r <= 0:
        raise ValueError(
            "measurement_variance must be > 0"
        )

    if p0 <= 0:
        raise ValueError(
            "initial_variance must be > 0"
        )

    n, d = x.shape

    out = np.empty_like(x)

    # First state equals first measurement.
    state = x[0].copy()

    variance = np.full(
        d,
        p0,
        dtype=np.float64,
    )

    out[0] = state

    for t in range(
        1,
        n,
    ):

        # Prediction.
        pred_state = state

        pred_variance = (
            variance
            + q
        )

        # Update.
        gain = (
            pred_variance
            / (
                pred_variance
                + r
            )
        )

        state = (
            pred_state
            + gain
            * (
                x[t]
                - pred_state
            )
        )

        variance = (
            1.0
            - gain
        ) * pred_variance

        out[t] = state

    return out


def apply_filter(
    x,
    name,
    **kwargs,
):
    """
    Unified dispatch.
    """
    if name == "identity":
        return identity(
            x
        )

    if name == "moving_average3":
        return trailing_mean(
            x,
            window=3,
        )

    if name == "lowpass_binomial3":
        return causal_binomial_lowpass3(
            x
        )

    if name == "savgol5_poly2":
        return causal_savgol_endpoint(
            x,
            window=5,
            polyorder=2,
        )

    if name == "kalman":
        return causal_kalman_local_level(
            x,
            **kwargs,
        )

    raise ValueError(
        f"Unknown filter: {name}"
    )


PRIMARY_FIXED_FILTERS = (
    "identity",
    "moving_average3",
    "lowpass_binomial3",
    "savgol5_poly2",
)
