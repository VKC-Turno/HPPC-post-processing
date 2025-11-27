from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

STITCHED_DIR = Path("/home/kcv/Desktop/HPPC_post_processing/post_processing/data/stitched_detail_data")
OUTPUT_DIR = Path("/home/kcv/Desktop/HPPC_post_processing/post_processing/data/optimiser_results")
WINDOW_CANDIDATES = (3, 5, 7, 9)
THRESHOLD_CANDIDATES = (1e-4, 5e-4, 1e-3, 5e-3)


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

    @property
    def C1(self) -> float:
        tau1 = self.duration_post_fast
        R1 = self.R1
        return tau1 / R1 if tau1 > 0 and np.isfinite(R1) and not np.isclose(R1, 0.0) else np.nan

    @property
    def C2(self) -> float:
        tau2 = self.duration_fast_end
        R2 = self.R2
        return tau2 / R2 if tau2 > 0 and np.isfinite(R2) and not np.isclose(R2, 0.0) else np.nan

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


def _compute_metrics_for_pair(
    file_stem: str,
    rest: pd.DataFrame,
    pulse: pd.DataFrame,
    flat_window: int,
    flat_threshold: float,
) -> tuple[PulseMetrics | None, pd.DataFrame]:
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
    flat_idx = None
    if abs_d2v[start_idx:].size >= flat_window:
        peak = np.nanmax(abs_d2v[start_idx:]) if np.any(np.isfinite(abs_d2v[start_idx:])) else np.nan
        thresh = peak * flat_threshold if np.isfinite(peak) else None
        if thresh is not None:
            for idx in range(start_idx, len(t) - flat_window):
                window = abs_d2v[idx : idx + flat_window]
                if np.all(np.isfinite(window)) and np.nanmax(window) <= thresh:
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


def _pulse_rmse(metrics: PulseMetrics, raw_df: pd.DataFrame) -> float | None:
    mask = (raw_df["time_s"] >= metrics.t_post) & (raw_df["time_s"] <= metrics.t_end)
    segment = raw_df.loc[mask]
    if segment.empty:
        return None

    t_rel = segment["time_s"].to_numpy() - metrics.t_post
    delta_measured = metrics.V_pre - segment["volt(V)"].to_numpy()

    tau1 = metrics.duration_post_fast
    tau2 = metrics.duration_fast_end
    if tau1 <= 0 or tau2 <= 0:
        return None

    delta_model = metrics.I_step * (
        metrics.R0 + metrics.R1 * (1 - np.exp(-t_rel / tau1)) + metrics.R2 * (1 - np.exp(-t_rel / tau2))
    )
    return float(np.sqrt(np.mean((delta_measured - delta_model) ** 2)))


def _evaluate_parameters(flat_window: int, flat_threshold: float) -> tuple[float, Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.DataFrame]]]:
    per_file_summary: Dict[str, pd.DataFrame] = {}
    per_file_derivatives: Dict[str, Dict[str, pd.DataFrame]] = {}
    rmse_values: List[float] = []

    for csv_path in sorted(STITCHED_DIR.glob("*.csv")):
        df = pd.read_csv(csv_path)
        df["Absolute time"] = pd.to_datetime(df["Absolute time"])
        df["time_s"] = (df["Absolute time"] - df["Absolute time"].min()).dt.total_seconds()

        pairs = _gather_pulses(df)
        summary_rows: List[PulseMetrics] = []
        derivative_records: Dict[str, pd.DataFrame] = {}

        for rest, pulse in pairs:
            metrics, deriv_df = _compute_metrics_for_pair(csv_path.stem, rest, pulse, flat_window, flat_threshold)
            pulse_id = f"cycle{int(pulse['Cycle No'].iloc[0])}_step{int(pulse['Step No'].iloc[0])}"
            derivative_records[pulse_id] = deriv_df

            if metrics:
                rmse = _pulse_rmse(metrics, df)
                if rmse is not None:
                    rmse_values.append(rmse)
                summary_rows.append(metrics)

        if summary_rows:
            per_file_summary[csv_path.stem] = pd.DataFrame(
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
                        "C1": m.C1,
                        "C2": m.C2,
                        "duration_pre_post": m.duration_pre_post,
                        "duration_post_fast": m.duration_post_fast,
                        "duration_fast_end": m.duration_fast_end,
                    }
                    for m in summary_rows
                ]
            )
            per_file_derivatives[csv_path.stem] = derivative_records

    mean_rmse = float(np.mean(rmse_values)) if rmse_values else np.inf
    return mean_rmse, per_file_summary, per_file_derivatives


def optimize_flat_parameters() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    resist_dir = OUTPUT_DIR / "resistances"
    deriv_dir = OUTPUT_DIR / "voltage_derivatives"
    resist_dir.mkdir(exist_ok=True)
    deriv_dir.mkdir(exist_ok=True)

    scores = []
    best_rmse = np.inf
    best_summary: Dict[str, pd.DataFrame] | None = None
    best_deriv: Dict[str, Dict[str, pd.DataFrame]] | None = None
    best_params: tuple[int, float] | None = None

    for window in WINDOW_CANDIDATES:
        for thresh in THRESHOLD_CANDIDATES:
            rmse, summary_map, deriv_map = _evaluate_parameters(window, thresh)
            scores.append({"flat_window": window, "flat_threshold": thresh, "rmse": rmse})
            if rmse < best_rmse:
                best_rmse = rmse
                best_summary = summary_map
                best_deriv = deriv_map
                best_params = (window, thresh)

    pd.DataFrame(scores).to_csv(OUTPUT_DIR / "parameter_scores.csv", index=False)

    if best_summary is None or best_deriv is None or best_params is None:
        print("No valid pulse metrics were produced.")
        return

    for file_stem, summary_df in best_summary.items():
        summary_df.to_csv(resist_dir / f"{file_stem}_resistance_summary.csv", index=False)
        for pulse_id, deriv_df in best_deriv[file_stem].items():
            deriv_df.to_csv(deriv_dir / f"{file_stem}_{pulse_id}_derivatives.csv", index=False)

    msg = f"Best flat_window={best_params[0]}, flat_threshold={best_params[1]}, RMSE={best_rmse}"
    print(msg)
    with open(OUTPUT_DIR / "best_params.txt", "w", encoding="utf-8") as fh:
        fh.write(msg + "\n")


if __name__ == "__main__":
    optimize_flat_parameters()
