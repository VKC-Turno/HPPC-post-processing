from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots


STEP_NAMES: Sequence[str] = ("CC_Chg", "CCCV_Chg", "CC_DChg")
STEP_47 = 47


def _sample_viridis(values: Iterable[int]) -> dict[int, str]:
    steps = sorted(set(values))
    if not steps:
        return {}
    if len(steps) == 1:
        return {steps[0]: sample_colorscale("Viridis", [0.5])[0]}
    positions = [i / (len(steps) - 1) for i in range(len(steps))]
    return dict(zip(steps, sample_colorscale("Viridis", positions)))


def _axis_name(prefix: str, idx: int) -> str:
    return prefix if idx == 1 else f"{prefix}{idx}"


def _add_step_subplot(fig: go.Figure, data: pd.DataFrame, row: int, col: int, title: str, subplot_idx: int) -> None:
    xaxis = _axis_name("x", subplot_idx)
    yaxis = _axis_name("y", subplot_idx)

    fig.update_xaxes(title_text="SOC (%)", row=row, col=col)
    fig.update_yaxes(title_text="Voltage (V)", row=row, col=col)

    if data.empty:
        fig.add_annotation(
            text=f"{title}: no data",
            xref=f"{xaxis} domain",
            yref=f"{yaxis} domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#555", size=12),
        )
        return

    step_ids = sorted(data["Step No"].unique())
    color_map = _sample_viridis(step_ids)

    for step_id in step_ids:
        step_df = data[data["Step No"] == step_id].sort_values("soc")
        fig.add_trace(
            go.Scatter(
                x=step_df["soc"],
                y=step_df["volt(V)"],
                mode="lines+markers",
                line=dict(color=color_map.get(step_id), width=1.4),
                marker=dict(color=color_map.get(step_id), size=4),
                hovertemplate="Step %{text}<br>SOC %{x:.1f}%<br>Voltage %{y:.3f} V",
                text=[step_id] * len(step_df),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    legend_text = " &#124; ".join(
        f"<span style='color:{color_map.get(step_id)};'>{int(step_id)}</span>"
        for step_id in step_ids
    )
    fig.add_annotation(
        text=legend_text or "No steps",
        xref=f"{xaxis} domain",
        yref=f"{yaxis} domain",
        x=0.5,
        y=-0.18,
        showarrow=False,
        font=dict(size=11),
    )


def _load_cell(detail_path: Path, soc_path: Path) -> pd.DataFrame:
    detail = pd.read_csv(detail_path)
    soc = pd.read_csv(soc_path)

    detail["Absolute time"] = pd.to_datetime(detail["Absolute time"])
    detail["Step name"] = detail["Step name"].fillna("Rest")
    soc["Absolute time"] = pd.to_datetime(soc["Absolute time"])
    soc["Step name"] = soc["Step name"].fillna("Rest")

    df = detail.merge(
        soc,
        on=["Cycle No", "Step No", "Step name", "Absolute time"],
        how="inner",
    )
    df = df[df["Step name"].str.lower() != "rest"]
    return df


def build_soc_voltage_plots(
    detail_dir: str | Path,
    soc_dir: str | Path,
    output_pdf: str | Path,
    file_pattern: str = "*.csv",
) -> Path:
    """
    For every CSV in detail_dir (matched with soc_dir), create a 2x2 SOC-vs-Voltage
    figure (CC_Chg, CCCV_Chg, CC_DChg, Step 47). Save all figures in a single multi-page
    PDF (one cell per page) and return the PDF path.
    """
    detail_dir = Path(detail_dir)
    soc_dir = Path(soc_dir)
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(detail_dir.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{file_pattern}' in {detail_dir}")

    pdf_pages: list[Image.Image] = []
    for detail_path in files:
        soc_path = soc_dir / detail_path.name
        if not soc_path.exists():
            continue

        df = _load_cell(detail_path, soc_path)
        if df.empty:
            continue

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("CC_Chg", "CCCV_Chg", "CC_DChg", "Step 47"),
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        subplot_idx = 1
        for step_name in STEP_NAMES:
            subset = df[(df["Step name"] == step_name) & (df["Step No"] != STEP_47)]
            row = 1 if subplot_idx <= 2 else 2
            col = 1 if subplot_idx % 2 == 1 else 2
            _add_step_subplot(fig, subset, row, col, step_name, subplot_idx)
            subplot_idx += 1

        step47_data = df[df["Step No"] == STEP_47]
        _add_step_subplot(fig, step47_data, 2, 2, "Step 47", 4)

        fig.update_layout(
            height=850,
            width=1100,
            title=f"{detail_path.stem}: SOC vs Voltage",
            template="plotly_white",
        )

        img_bytes = fig.to_image(format="png", scale=2)
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        pdf_pages.append(image)

    if not pdf_pages:
        raise RuntimeError("No matching cell data found for SOC plots.")

    first, rest = pdf_pages[0], pdf_pages[1:]
    first.save(output_pdf, format="PDF", save_all=True, append_images=rest)
    return output_pdf


__all__ = ["build_soc_voltage_plots"]
