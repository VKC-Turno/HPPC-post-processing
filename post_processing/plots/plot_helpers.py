from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, VPacker, TextArea
import pandas as pd


REQUIRED_COLUMNS: tuple[str, ...] = (
    "Cycle No",
    "Step No",
    "Step name",
    "Absolute time",
    "volt(V)",
    "Current(A)",
    "Capacity(Ah)",
)
STEP_NAME_COLORS: dict[str, str] = {
    "CC_Chg": "#006400",  # dark green
    "CCCV_Chg": "#7CFC00",  # light green
    "CC_DChg": "#C00000",  # red
    "Rest": "#1f77b4",  # blue
}
DEFAULT_STEP_COLOR = "#444444"
DEFAULT_MARKER = "o"
SCATTER_MARKER_SIZE = 7
LEGEND_STEP_MARKER_SIZE = 7


def _ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")


def plot_current_voltage_time(
    input_dir: str | Path,
    output_dir: str | Path,
    pdf_name: str = "current_voltage_time.pdf",
) -> Path:
    """
    Build a multi-page PDF where each page contains (1) Current vs Absolute time and
    (2) Voltage vs Absolute time. Colors encode the step names (Rest, CC_Chg, CCCV_Chg,
    CC_DChg), a shared marker style is used everywhere, and vertical dashed lines mark
    the end of each step number using viridis.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(in_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {in_dir}")

    pdf_path = out_dir / pdf_name
    with PdfPages(pdf_path) as pdf:
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            _ensure_columns(df, REQUIRED_COLUMNS)

            df = df.copy()
            df["Absolute time"] = pd.to_datetime(df["Absolute time"], errors="coerce")
            df["Current(A)"] = pd.to_numeric(df["Current(A)"], errors="coerce")
            df["volt(V)"] = pd.to_numeric(df["volt(V)"], errors="coerce")
            df = df.dropna(subset=["Absolute time", "Current(A)", "volt(V)"])
            if df.empty:
                continue

            fig, (ax_current, ax_voltage) = plt.subplots(
                2,
                1,
                figsize=(11, 8),
                sharex=True,
                gridspec_kw={"hspace": 0.05},
            )

            step_boundaries = (
                df.sort_values("Absolute time")
                .groupby("Step No")["Absolute time"]
                .max()
                .dropna()
            )

            def step_color(step_no):
                row = df[df["Step No"] == step_no]
                if row.empty:
                    return DEFAULT_STEP_COLOR
                category = row["step_category"].iloc[-1]
                return STEP_NAME_COLORS.get(category, DEFAULT_STEP_COLOR)

            df["step_category"] = df["Step name"].where(df["Step name"].isin(STEP_NAME_COLORS), "Other")

            def plot_series(axis, series_name: str):
                for category, group in df.groupby("step_category"):
                    color = STEP_NAME_COLORS.get(category, DEFAULT_STEP_COLOR)
                    axis.scatter(
                        group["Absolute time"],
                        group[series_name],
                        color=color,
                        marker=DEFAULT_MARKER,
                        s=SCATTER_MARKER_SIZE,
                        alpha=0.9,
                    )

            plot_series(ax_current, "Current(A)")
            plot_series(ax_voltage, "volt(V)")

            for step_no, end_time in step_boundaries.items():
                color = step_color(step_no)
                for axis in (ax_current, ax_voltage):
                    axis.axvline(end_time, color=color, linestyle="--", linewidth=0.8, alpha=0.8)

            ax_current.set_title(csv_file.stem)
            ax_voltage.set_xlabel("Absolute time")
            ax_current.set_ylabel("Current (A)")
            ax_voltage.set_ylabel("Voltage (V)")
            for axis in (ax_current, ax_voltage):
                axis.grid(True, linestyle="--", alpha=0.3)

            ax_voltage.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
            fig.autofmt_xdate(rotation=30)

            def unique_step_numbers(series: pd.Series) -> list:
                values = []
                for val in series.dropna():
                    if val not in values:
                        values.append(val)
                return values

            def format_step_number(value: object) -> str:
                if isinstance(value, (int, float)):
                    if float(value).is_integer():
                        return str(int(value))
                    return f"{float(value):.2f}".rstrip("0").rstrip(".")
                return str(value)

            order = [name for name in STEP_NAME_COLORS]
            if "Other" in df["step_category"].unique():
                order.append("Other")

            legend_rows = []

            for category in order:
                group = df[df["step_category"] == category]
                if group.empty:
                    continue
                color = STEP_NAME_COLORS.get(category, DEFAULT_STEP_COLOR)
                step_numbers = unique_step_numbers(group["Step No"])
                row_children = [
                    TextArea(
                        "●",
                        textprops=dict(color=color, fontsize=10, fontweight="bold", va="center"),
                    ),
                    TextArea(
                        f" {category}: ",
                        textprops=dict(color="black", fontsize=9),
                    ),
                ]

                if step_numbers:
                    for idx, step_no in enumerate(step_numbers):
                        comma = ", " if idx < len(step_numbers) - 1 else ""
                        row_children.append(
                            TextArea(
                                f"{format_step_number(step_no)}{comma}",
                                textprops=dict(
                                    color=color,
                                    fontsize=9,
                                ),
                            )
                        )
                else:
                    row_children.append(TextArea("—", textprops=dict(color="black", fontsize=9)))

                legend_rows.append(HPacker(children=row_children, align="baseline", pad=0, sep=1))

            if legend_rows:
                legend_box = VPacker(children=legend_rows, align="left", pad=0.2, sep=2)
                anchored = AnchoredOffsetbox(
                    loc="upper right",
                    child=legend_box,
                    pad=0.3,
                    frameon=True,
                    bbox_to_anchor=(1, 1),
                    bbox_transform=ax_current.transAxes,
                )
                anchored.patch.set_alpha(0.9)
                anchored.patch.set_edgecolor("gray")
                ax_current.add_artist(anchored)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return pdf_path


__all__ = ["plot_current_voltage_time"]
