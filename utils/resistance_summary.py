from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

DERIVATIVE_DIR = Path("/home/kcv/Desktop/HPPC_post_processing/post_processing/data/voltage_derivatives")

STITCHED_DIR = Path("/home/kcv/Desktop/HPPC_post_processing/post_processing/data/stitched_detail_data")
FLAT_WINDOW = 5
FLAT_THRESHOLD_FRACTION = 0.008


@dataclass
class PulseMetrics:
    file: str
    cycle_no: int
    step_no: int
    step_name: str
    t_pre: float
    t_post: float
    t_fast: float
    t_end: float
    V_pre: float
    V_post: float
    V_fast: float
    V_end: float
    I_step: float

    @property
    def duration_pre_post(self) -> float:
        return self.t_post - self.t_pre

    @property
    def duration_post_fast(self) -> float:
        return self.t_fast - self.t_post

    @property
    def duration_fast_end(self) -> float:
        return self.t_end - self.t_fast

    @property
    def R0(self) -> float:
        return (self.V_pre - self.V_post) / self.I_step if self.I_step else np.nan

    @property
    def R1(self) -> float:
        return (self.V_post - self.V_fast) / self.I_step if self.I_step else np.nan

    @property
    def R2(self) -> float:
        return (self.V_fast - self.V_end) / self.I_step if self.I_step else np.nan


def _gather_pulses(df: pd.DataFrame) -> List[tuple[pd.DataFrame, pd.DataFrame]]:
    steps = list(df.groupby(["Cycle No", "Step No"], sort=False))
    pairs: List[tuple[pd.DataFrame, pd.DataFrame]] = []

    for idx in range(1, len(steps)):
        step_df = steps[idx][1]
        prev_df = steps[idx - 1][1]
        step_name = str(step_df["Step name"].iloc[0]).strip()
        prev_name = str(prev_df["Step name"].iloc[0]).strip().lower()

        if step_name == "CC_DChg" and "rest" in prev_name:
            pairs.append((prev_df, step_df))
    return pairs


def _compute_metrics_for_pair(file_stem: str, rest: pd.DataFrame, pulse: pd.DataFrame) -> tuple[PulseMetrics | None, pd.DataFrame]:
    combined = pd.concat([rest.tail(5), pulse]).sort_values("time_s").reset_index(drop=True)
    t = combined["time_s"].to_numpy()
    v = combined["volt(V)"].to_numpy()

    dvdt = np.gradient(v, t)
    d2v = np.gradient(dvdt, t)

    t_pre = float(rest["time_s"].iloc[-1])
    V_pre = float(rest["volt(V)"].iloc[-1])

    t_post = float(pulse["time_s"].iloc[0])
    V_post = float(pulse["volt(V)"].iloc[0])

    start_idx = np.searchsorted(t, t_post)
    abs_d2v = np.abs(d2v)
    segment = abs_d2v[start_idx:]
    flat_idx = None
    if segment.size >= FLAT_WINDOW:
        peak = np.nanmax(segment) if np.any(np.isfinite(segment)) else np.nan
        thresh = peak * FLAT_THRESHOLD_FRACTION if np.isfinite(peak) else None
        for idx in range(start_idx, len(t) - FLAT_WINDOW):
            window = abs_d2v[idx : idx + FLAT_WINDOW]
            if thresh is not None and np.all(np.isfinite(window)) and np.nanmax(window) <= thresh:
                flat_idx = idx
                break
    deriv_df = pd.DataFrame({"time_s": t, "dv_dt": dvdt, "d2v_dt2": d2v})
    if flat_idx is None:
        return None, deriv_df
    t_fast = float(t[flat_idx])
    V_fast = float(np.interp(t_fast, t, v))

    t_end = float(pulse["time_s"].iloc[-1])
    V_end = float(pulse["volt(V)"].iloc[-1])

    I_step = float(pulse["Current(A)"].iloc[:5].mean())
    if I_step == 0 or not np.isfinite(I_step):
        return None, deriv_df

    metrics = PulseMetrics(
        file=file_stem,
        cycle_no=int(pulse["Cycle No"].iloc[0]),
        step_no=int(pulse["Step No"].iloc[0]),
        step_name=str(pulse["Step name"].iloc[0]).strip(),
        t_pre=t_pre,
        t_post=t_post,
        t_fast=t_fast,
        t_end=t_end,
        V_pre=V_pre,
        V_post=V_post,
        V_fast=V_fast,
        V_end=V_end,
        I_step=I_step,
    )
    return metrics, deriv_df


def summarize_resistances(output_dir: str | Path, data_dir: str | Path = STITCHED_DIR) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)
    DERIVATIVE_DIR.mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        df["Absolute time"] = pd.to_datetime(df["Absolute time"])
        df["time_s"] = (df["Absolute time"] - df["Absolute time"].min()).dt.total_seconds()

        pairs = _gather_pulses(df)
        summary_rows: List[PulseMetrics] = []

        for rest, pulse in pairs:
            metrics, deriv_df = _compute_metrics_for_pair(csv_path.stem, rest, pulse)
            pulse_id = f"cycle{int(pulse['Cycle No'].iloc[0])}_step{int(pulse['Step No'].iloc[0])}"
            deriv_df.to_csv(DERIVATIVE_DIR / f"{csv_path.stem}_{pulse_id}_derivatives.csv", index=False)
            if metrics:
                summary_rows.append(metrics)

        if not summary_rows:
            continue

        summary_df = pd.DataFrame(
            [
                {
                    "file": m.file,
                    "cycle": m.cycle_no,
                    "step": m.step_no,
                    "step_name": m.step_name,
                    "t_pre": m.t_pre,
                    "t_post": m.t_post,
                    "t_fast": m.t_fast,
                    "t_end": m.t_end,
                    "V_pre": m.V_pre,
                    "V_post": m.V_post,
                    "V_fast": m.V_fast,
                    "V_end": m.V_end,
                    "I_step": m.I_step,
                    "R0": m.R0,
                    "R1": m.R1,
                    "R2": m.R2,
                    "duration_pre_post": m.duration_pre_post,
                    "duration_post_fast": m.duration_post_fast,
                    "duration_fast_end": m.duration_fast_end,
                }
                for m in summary_rows
            ]
        )

        summary_df.to_csv(output_path / f"{csv_path.stem}_resistance_summary.csv", index=False)


__all__ = ["summarize_resistances"]
