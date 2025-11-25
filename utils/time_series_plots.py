from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from plotly.colors import qualitative
from plotly.subplots import make_subplots


def _load_cell(detail_path: Path, soc_path: Path) -> pd.DataFrame:
    detail = pd.read_csv(detail_path).assign(
        Absolute_time=lambda d: pd.to_datetime(d["Absolute time"]),
        Step_name=lambda d: d["Step name"].fillna("Rest"),
    )
    soc = pd.read_csv(soc_path).assign(
        Absolute_time=lambda d: pd.to_datetime(d["Absolute time"]),
        Step_name=lambda d: d["Step name"].fillna("Rest"),
    )

    df = detail.merge(
        soc[["Cycle No", "Step No", "Step name", "Absolute time", "soc"]],
        on=["Cycle No", "Step No", "Step name", "Absolute time"],
        how="left",
    )
    df["time_s"] = (df["Absolute_time"] - df["Absolute_time"].min()).dt.total_seconds()
    df["time_s_scaled"] = df["time_s"] / 1e4
    return df


def _build_color_map(step_names: Iterable[str]) -> dict[str, str]:
    palette = qualitative.Plotly
    step_names = list(step_names)
    return {name: palette[i % len(palette)] for i, name in enumerate(step_names)}


def _build_figure(df: pd.DataFrame, title: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.05,
        subplot_titles=("Current vs Time", "Voltage vs Time"),
    )

    color_map = _build_color_map(df["Step_name"].unique())

    for step_name, group in df.groupby("Step_name"):
        color = color_map[step_name]
        opacity = 0.1 if step_name == "Rest" else 0.9
        size = 1 if step_name == "Rest" else 4

        fig.add_trace(
            go.Scatter(
                x=group["time_s_scaled"],
                y=group["Current(A)"],
                mode="markers",
                marker=dict(color=color, size=size, symbol="circle", opacity=opacity),
                name=step_name,
                customdata=group[["Step No", "time_s"]],
                hovertemplate="Step: %{customdata[0]}<br>Time: %{customdata[1]:.1f} s<br>Current: %{y:.3f} A",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=group["time_s_scaled"],
                y=group["volt(V)"],
                mode="markers",
                marker=dict(color=color, size=size, symbol="circle", opacity=opacity * 0.8),
                name=f"{step_name} – Voltage",
                customdata=group[["Step No", "time_s"]],
                hovertemplate="Step: %{customdata[0]}<br>Time: %{customdata[1]:.1f} s<br>Voltage: %{y:.3f} V",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title=title,
        height=620,
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color="white"),
        legend=dict(orientation="h"),
    )

    for row in (1, 2):
        fig.update_xaxes(
            showgrid=True,
            gridcolor="#333333",
            zeroline=False,
            color="white",
            tickformat=".1f",
            row=row,
            col=1,
        )

    fig.update_xaxes(title_text="Time (s) ×10⁴", row=2, col=1)
    fig.update_yaxes(title_text="Current (A)", color="white", gridcolor="#333333", zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="Voltage (V)", color="white", gridcolor="#333333", zeroline=False, row=2, col=1)

    return fig


def save_time_series_pdf(
    detail_dir: str | Path,
    soc_dir: str | Path,
    output_pdf: str | Path,
    file_pattern: str = "*.csv",
) -> Path:
    """
    Render the current/voltage vs time plots for every cell and save them into a
    single multi-page PDF (one page per cell). Returns the generated PDF path.
    """
    detail_dir = Path(detail_dir)
    soc_dir = Path(soc_dir)
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(detail_dir.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{file_pattern}' in {detail_dir}")

    pages: list[Image.Image] = []
    for detail_path in files:
        soc_path = soc_dir / detail_path.name
        if not soc_path.exists():
            continue

        df = _load_cell(detail_path, soc_path)
        if df.empty:
            continue

        fig = _build_figure(df, detail_path.stem)
        png_bytes = fig.to_image(format="png", scale=2)
        pages.append(Image.open(BytesIO(png_bytes)).convert("RGB"))

    if not pages:
        raise RuntimeError("No matching cells with SOC data were found.")

    first, rest = pages[0], pages[1:]
    first.save(output_pdf, format="PDF", save_all=True, append_images=rest)
    return output_pdf


__all__ = ["save_time_series_pdf"]
