from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS: tuple[str, ...] = (
    "Cycle No",
    "Step No",
    "Step name",
    "Absolute time",
    "Record time(m)",
    "Step time(h:m:s.ms)",
    "volt(V)",
    "Current(A)",
    "Capacity(Ah)",
    "Energy(Wh)",
    "Power(mW)",
    "Internal R(Ω)",
    "Charging energy(Wh)",
    "Discharge energy(Wh)",
    "Charging capacity(Ah)",
    "Discharge capacity(Ah)",
)

OUTPUT_COLUMNS: tuple[str, ...] = (
    "Cycle No",
    "Step No",
    "Step name",
    "Absolute time",
    "soc",
)


def _check_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def add_state_of_charge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a dataframe filtered to OUTPUT_COLUMNS with an additional 'soc' column (0-100%).

    The SOC for each (Cycle No, Step No) group is computed by normalizing the absolute
    Capacity(Ah) trace to its maximum value. CC_DChg steps are inverted (1 - SOC) before
    converting to percent.
    """
    _check_columns(df, REQUIRED_COLUMNS)

    working = df.copy()
    working["Capacity(Ah)"] = pd.to_numeric(working["Capacity(Ah)"], errors="coerce").fillna(0.0)
    soc_values = np.zeros(len(working), dtype=float)

    grouped = working.groupby(["Cycle No", "Step No"], sort=False)
    for _, group in grouped:
        idx = group.index.to_numpy()
        if idx.size == 0:
            continue

        cap = group["Capacity(Ah)"].abs().to_numpy()
        max_cap = 150
        # max_cap = cap.max() if cap.size else 0.0
        if max_cap == 0.0:
            soc = np.zeros_like(cap)
        else:
            soc = cap / max_cap

        step_name = str(group["Step name"].iloc[0]).strip().lower()
        if step_name == "cc_dchg":
            soc = 1.0 - soc

        soc_values[idx] = soc * 100.0

    working["soc"] = soc_values
    return working.loc[:, OUTPUT_COLUMNS].copy()


def process_soc_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    file_pattern: str = "*.csv",
) -> list[Path]:
    """
    Read every CSV matching file_pattern under input_dir, append SOC, and write
    the trimmed dataframe to output_dir (mirroring filenames). Returns the list
    of output file paths.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    csv_files = sorted(in_dir.glob(file_pattern))
    if not csv_files:
        raise FileNotFoundError(f"No files matching '{file_pattern}' in {in_dir}")

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        processed = add_state_of_charge(df)
        out_path = out_dir / csv_path.name
        processed.to_csv(out_path, index=False)
        written.append(out_path)

    return written


__all__ = ["add_state_of_charge", "process_soc_directory"]
