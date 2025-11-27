from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

# Default delays/windows (seconds)
V_PRE_WINDOW = 0.8
POST_AVG_HALF_WIDTH = 0.05
FAST_AVG_HALF_WIDTH = 0.25
FAST_FALLBACK_POINTS = 5
SLOPE_WINDOW = 3.0
SLOPE_FRACTION = 0.15
SLOW_TAIL = 5.0


@dataclass
class VoltageWindows:
    t0: float
    I_step: float
    V_pre: float
    V_post: float
    V_fast: float
    V_end: float


def _mean_over_window(values: np.ndarray, mask: np.ndarray) -> float:
    if np.any(mask):
        return float(np.mean(values[mask]))
    return float("nan")


def _derive_post_fast(
    times: np.ndarray,
    volts: np.ndarray,
    start_idx: int,
    end_idx: int,
    t0: float,
) -> tuple[float, float, float]:
    """
    Return (t_post_center, V_post, V_fast) using only derivative-based heuristics.
    """
    slope_mask = (times >= t0) & (times <= times[end_idx])
    if slope_mask.sum() >= 3:
        seg_t = times[slope_mask]
        seg_v = volts[slope_mask]
        dvdt = np.gradient(seg_v, seg_t)
        peak = np.max(np.abs(dvdt))
        peak_idx = 0
        if peak > 0:
            peak_idx = int(np.argmax(np.abs(dvdt)))
            thresh = SLOPE_FRACTION * peak
            idx_candidates = np.where((np.arange(len(dvdt)) >= peak_idx) & (np.abs(dvdt) <= thresh))[0]
            idx_fast = int(idx_candidates[0]) if idx_candidates.size else len(dvdt) - 1
        else:
            idx_fast = 0
        t_post_center = seg_t[peak_idx]
        t_fast_center = max(seg_t[idx_fast], t0 + 0.1)
    else:
        t_post_center = t0
        t_fast_center = t0 + 0.5

    post_mask = (times >= t_post_center - POST_AVG_HALF_WIDTH) & (times <= t_post_center + POST_AVG_HALF_WIDTH)
    V_post = float(np.mean(volts[post_mask])) if np.any(post_mask) else float(volts[start_idx])

    fast_mask = (times >= t_fast_center) & (times <= t_fast_center + FAST_AVG_HALF_WIDTH)
    if np.any(fast_mask):
        V_fast = float(np.mean(volts[fast_mask]))
    else:
        fallback_end = min(end_idx + 1, start_idx + FAST_FALLBACK_POINTS)
        V_fast = float(np.mean(volts[start_idx:fallback_end])) if fallback_end > start_idx else float(volts[start_idx])

    return t_post_center, V_post, V_fast


def compute_voltage_windows(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    time_col: str = "time_s",
    volt_col: str = "volt(V)",
    curr_col: str = "Current(A)",
) -> VoltageWindows:
    """
    Compute V_pre, V_post, V_fast, V_end for a pulse identified by dataframe indices.
    Returns a VoltageWindows dataclass and guarantees V_fast falls back to early samples
    instead of NaN when the nominal window has no data.
    """
    t = df[time_col].to_numpy()
    v = df[volt_col].to_numpy()
    i = df[curr_col].to_numpy()

    t0 = float(t[start_idx])
    I_step = float(np.mean(i[start_idx : start_idx + 5]))

    pre_mask = (t >= t0 - V_PRE_WINDOW) & (t < t0)
    V_pre = _mean_over_window(v, pre_mask)
    if not np.isfinite(V_pre):
        V_pre = float(v[start_idx])

    _, V_post, V_fast = _derive_post_fast(t, v, start_idx, end_idx, t0)

    end_time = float(t[end_idx])
    slow_mask = (t >= end_time - SLOW_TAIL) & (t <= end_time)
    V_end = _mean_over_window(v, slow_mask)

    return VoltageWindows(t0=t0, I_step=I_step, V_pre=V_pre, V_post=V_post, V_fast=V_fast, V_end=V_end)


__all__ = [
    "VoltageWindows",
    "compute_voltage_windows",
    "V_PRE_WINDOW",
    "POST_AVG_HALF_WIDTH",
    "FAST_AVG_HALF_WIDTH",
    "FAST_FALLBACK_POINTS",
    "SLOPE_WINDOW",
    "SLOPE_FRACTION",
    "FAST_AVG_HALF_WIDTH",
    "SLOW_TAIL",
]
