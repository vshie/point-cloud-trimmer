"""Plotly figure builders for the trimmer UI.

The top-down map is rendered server-side with Datashader and embedded into
a Plotly figure as a background image, with the axes set to the data's
local-meter coordinates. Drawn shapes (line, rect) therefore live in real
meters and can be read back directly. Histograms and the cross-section are
built from numpy arrays.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import datashader as ds
import datashader.transfer_functions as tf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

from .geom import Heading

MAP_PIXELS = 800
CROSS_SECTION_PLOT_CAP = 100_000
DEPTH_BINS = 300
POWER_BINS = 200

# Viridis palette as an explicit hex list. datashader's `tf.shade(cmap=...)`
# takes a list of colors (or a bokeh/colorcet palette object), not a name
# string -- passing "viridis" raises `Unknown color`.
VIRIDIS_HEX: list[str] = [
    "#440154", "#482777", "#3f4a8a", "#31678e", "#26838f",
    "#1f9d8a", "#6cce5a", "#b6de2b", "#fee825",
]

# Trace index of the "X" marker overlaid on the top-down map when the user
# hovers a point in the cross-section. Always present so a Patch-based
# update can address `figure.data[XSEC_HOVER_TRACE]` unconditionally.
XSEC_HOVER_TRACE = 1


def _hover_marker_trace() -> go.Scatter:
    return go.Scatter(
        x=[],
        y=[],
        mode="markers",
        marker=dict(
            symbol="x",
            size=16,
            color="#ff3d77",
            line=dict(color="#ffffff", width=1),
        ),
        name="xsec-hover",
        hoverinfo="skip",
        showlegend=False,
    )


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_dark",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _empty_topdown_figure(title: str) -> go.Figure:
    """Empty top-down figure that still carries the hover-marker trace.

    Keeps the trace ordering stable so Patch-based updates can address it.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="markers",
            marker=dict(size=0.0001, color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(_hover_marker_trace())
    fig.update_layout(
        title=title,
        template="plotly_dark",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _png_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def top_down_figure(
    east: np.ndarray,
    north: np.ndarray,
    depth: np.ndarray,
    keep_mask: np.ndarray,
    heading: Optional[Heading] = None,
    half_width: float = 0.5,
) -> go.Figure:
    """Datashader-backed top-down map with optional heading line + corridor box."""
    if not keep_mask.any():
        return _empty_topdown_figure("Top-down map (no points)")

    df = pd.DataFrame(
        {
            "x": east[keep_mask],
            "y": north[keep_mask],
            "z": depth[keep_mask],
        }
    )

    x_min, x_max = float(df["x"].min()), float(df["x"].max())
    y_min, y_max = float(df["y"].min()), float(df["y"].max())

    # Pad bounds slightly so points on the edge are not clipped to half-pixels.
    x_pad = max((x_max - x_min) * 0.02, 1.0)
    y_pad = max((y_max - y_min) * 0.02, 1.0)
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    # Preserve aspect ratio of meters: square pixels are essential since the
    # user will draw a compass-heading line on this raster.
    width_m = x_max - x_min
    height_m = y_max - y_min
    if width_m >= height_m:
        plot_w = MAP_PIXELS
        plot_h = max(1, int(round(MAP_PIXELS * (height_m / width_m))))
    else:
        plot_h = MAP_PIXELS
        plot_w = max(1, int(round(MAP_PIXELS * (width_m / height_m))))

    canvas = ds.Canvas(
        plot_width=plot_w,
        plot_height=plot_h,
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
    )
    agg = canvas.points(df, "x", "y", ds.mean("z"))
    img = tf.shade(agg, cmap=VIRIDIS_HEX, how="linear")
    img = tf.set_background(img, "black")
    pil = img.to_pil()
    uri = _png_data_uri(pil)

    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=uri,
            xref="x",
            yref="y",
            x=x_min,
            y=y_max,
            sizex=x_max - x_min,
            sizey=y_max - y_min,
            sizing="stretch",
            opacity=1.0,
            layer="below",
        )
    )
    # Invisible scatter that anchors the axes to data coordinates so drawn
    # shapes are reported in meters. Stays at trace index 0.
    fig.add_trace(
        go.Scatter(
            x=[x_min, x_max],
            y=[y_min, y_max],
            mode="markers",
            marker=dict(size=0.0001, color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    # Hover-marker trace at index XSEC_HOVER_TRACE; populated by a Patch from
    # the cross-section's hoverData callback.
    fig.add_trace(_hover_marker_trace())

    shapes: list[dict] = []
    if heading is not None:
        rad = np.deg2rad(heading.theta_deg)
        # Direction unit vector for compass azimuth (clockwise from north).
        ux = np.sin(rad)
        uy = np.cos(rad)
        # Perpendicular (left-of-heading) unit vector.
        px = -uy
        py = ux
        L = max(heading.length, 1.0)
        line_x0 = heading.cx - L * ux
        line_y0 = heading.cy - L * uy
        line_x1 = heading.cx + L * ux
        line_y1 = heading.cy + L * uy
        shapes.append(
            dict(
                type="line",
                x0=line_x0,
                y0=line_y0,
                x1=line_x1,
                y1=line_y1,
                line=dict(color="cyan", width=2),
                editable=False,
            )
        )
        # Corridor rectangle as a closed polygon path (handles arbitrary rotation).
        corners = [
            (line_x0 + half_width * px, line_y0 + half_width * py),
            (line_x1 + half_width * px, line_y1 + half_width * py),
            (line_x1 - half_width * px, line_y1 - half_width * py),
            (line_x0 - half_width * px, line_y0 - half_width * py),
        ]
        path = "M " + " L ".join(f"{x},{y}" for x, y in corners) + " Z"
        shapes.append(
            dict(
                type="path",
                path=path,
                line=dict(color="cyan", width=1, dash="dot"),
                fillcolor="rgba(0, 255, 255, 0.08)",
                editable=False,
            )
        )

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(
            range=[x_min, x_max],
            title="Easting (local m)",
            constrain="domain",
        ),
        yaxis=dict(
            range=[y_min, y_max],
            title="Northing (local m)",
            scaleanchor="x",
            scaleratio=1,
        ),
        dragmode="drawline",
        newshape=dict(line=dict(color="yellow", width=2)),
        shapes=shapes,
        title=f"Top-down map ({int(keep_mask.sum()):,} kept points)",
    )
    return fig


def depth_histogram_figure(
    depth: np.ndarray,
    mask: np.ndarray,
    selected_range: Optional[tuple[float, float]] = None,
    title: str = "Depth distribution",
) -> go.Figure:
    if not mask.any():
        return _empty_figure(title + " (no points)")
    vals = depth[mask]
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return _empty_figure(title + " (no finite values)")
    counts, edges = np.histogram(finite, bins=DEPTH_BINS)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig = go.Figure(
        go.Bar(
            x=centers,
            y=counts,
            width=edges[1] - edges[0],
            marker=dict(color="#4fc3f7"),
            hovertemplate="depth=%{x:.2f} m<br>count=%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=20, t=40, b=40),
        title=f"{title}  (n={finite.size:,})",
        xaxis=dict(title="depth (m)"),
        yaxis=dict(title="count"),
        bargap=0.0,
    )
    if selected_range is not None:
        z_lo, z_hi = selected_range
        fig.add_vrect(
            x0=z_lo,
            x1=z_hi,
            fillcolor="rgba(255, 235, 59, 0.15)",
            line_width=0,
        )
    return fig


def power_histogram_figure(
    power: np.ndarray,
    mask: np.ndarray,
    selected_range: Optional[tuple[float, float]] = None,
) -> go.Figure:
    if not mask.any():
        return _empty_figure("Power distribution (no points)")
    vals = power[mask]
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return _empty_figure("Power distribution (no finite values)")
    counts, edges = np.histogram(finite, bins=POWER_BINS)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig = go.Figure(
        go.Bar(
            x=centers,
            y=counts,
            width=edges[1] - edges[0],
            marker=dict(color="#ffb74d"),
            hovertemplate="power=%{x:.1f} dB<br>count=%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=20, t=40, b=40),
        title=f"Power distribution  (n={finite.size:,})",
        xaxis=dict(title="power (dB)"),
        yaxis=dict(title="count"),
        bargap=0.0,
    )
    if selected_range is not None:
        p_lo, p_hi = selected_range
        fig.add_vrect(
            x0=p_lo,
            x1=p_hi,
            fillcolor="rgba(255, 235, 59, 0.15)",
            line_width=0,
        )
    return fig


def cross_section_figure(
    corridor_idx: np.ndarray,
    along: np.ndarray,
    across: np.ndarray,
    depth: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
    heading: Heading,
    half_width: float,
    selected_depth_range: Optional[tuple[float, float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[go.Figure, np.ndarray]:
    """Cross-section plot of depth vs along-track for the current corridor.

    Returns the figure plus the corridor-local indices of the plotted points
    (`sample_local`), so the caller can cache it and translate a lasso
    selection's `pointIndex` back to global row indices.

    `customdata` is (N, 3) per plotted point: ``[local_idx, east, north]``.
    The east/north entries let the top-down hover-marker callback place an
    "X" at the original geographic location without any rotation math.
    """
    n = corridor_idx.size
    if n == 0:
        return _empty_figure("Cross-section (empty corridor)"), np.empty(0, dtype=np.int64)

    z = depth[corridor_idx]
    e_full = east[corridor_idx]
    n_full = north[corridor_idx]

    if n > CROSS_SECTION_PLOT_CAP:
        rng = rng or np.random.default_rng(0)
        sample_local = rng.choice(n, size=CROSS_SECTION_PLOT_CAP, replace=False)
        sample_local.sort()
    else:
        sample_local = np.arange(n)
    sample_local = sample_local.astype(np.int64, copy=False)

    plot_along = along[sample_local]
    plot_across = across[sample_local]
    plot_z = z[sample_local]
    plot_e = e_full[sample_local]
    plot_n = n_full[sample_local]

    # (N, 3) customdata so each point exposes [local_idx, east, north].
    custom = np.column_stack(
        [
            sample_local.astype(np.float64),
            plot_e.astype(np.float64),
            plot_n.astype(np.float64),
        ]
    )

    fig = go.Figure(
        go.Scattergl(
            x=plot_along,
            y=plot_z,
            mode="markers",
            marker=dict(
                size=3,
                color=plot_across,
                colorscale="RdBu",
                cmin=-half_width,
                cmax=half_width,
                showscale=True,
                colorbar=dict(title="across (m)"),
            ),
            customdata=custom,
            hovertemplate=(
                "along=%{x:.2f} m"
                "<br>depth=%{y:.3f} m"
                "<br>across=%{marker.color:.2f} m"
                "<br>E=%{customdata[1]:.2f}  N=%{customdata[2]:.2f}"
                "<extra></extra>"
            ),
        )
    )

    title = (
        f"Cross-section  heading={heading.theta_deg:.1f}\u00b0  "
        f"half-width={half_width:.2f} m  "
        f"({n:,} corridor pts, plotting {sample_local.size:,})"
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=50, r=20, t=40, b=40),
        title=title,
        xaxis=dict(title="along-track (m)"),
        yaxis=dict(title="depth (m)"),
        dragmode="select",
        hovermode="closest",
        hoverdistance=20,
    )
    if selected_depth_range is not None:
        z_lo, z_hi = selected_depth_range
        fig.add_hrect(
            y0=z_lo,
            y1=z_hi,
            fillcolor="rgba(255, 235, 59, 0.12)",
            line_width=0,
        )
    return fig, sample_local


def transect_3d_figure(
    corridor_idx: np.ndarray,
    along: np.ndarray,
    across: np.ndarray,
    depth: np.ndarray,
    sample_local: np.ndarray,
    half_width: float,
    selected_depth_range: Optional[tuple[float, float]] = None,
) -> go.Figure:
    """Rotatable 3D scatter of the corridor: along / across / depth.

    Uses ``aspectmode='data'`` so all three axes are scaled by their real
    data extents -- a long, thin, shallow transect looks long, thin, and
    shallow. The Plotly modebar lets the user orbit/pan/zoom the view.
    """
    if corridor_idx.size == 0 or sample_local.size == 0:
        fig = _empty_figure("Transect 3D (empty corridor)")
        fig.update_layout(scene=dict(bgcolor="#111"))
        return fig

    plot_along = along[sample_local]
    plot_across = across[sample_local]
    plot_z = depth[corridor_idx][sample_local]

    if selected_depth_range is not None:
        cmin, cmax = float(selected_depth_range[0]), float(selected_depth_range[1])
    else:
        finite_z = plot_z[np.isfinite(plot_z)]
        if finite_z.size:
            cmin = float(finite_z.min())
            cmax = float(finite_z.max())
        else:
            cmin, cmax = -1.0, 0.0

    fig = go.Figure(
        go.Scatter3d(
            x=plot_along,
            y=plot_across,
            z=plot_z,
            mode="markers",
            marker=dict(
                size=2,
                color=plot_z,
                colorscale="Viridis",
                cmin=cmin,
                cmax=cmax,
                showscale=True,
                colorbar=dict(title="depth (m)", thickness=12),
            ),
            hovertemplate=(
                "along=%{x:.2f} m"
                "<br>across=%{y:.2f} m"
                "<br>depth=%{z:.3f} m"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        scene=dict(
            xaxis=dict(title="along (m)"),
            yaxis=dict(
                title="across (m)",
                range=[-half_width, half_width],
            ),
            zaxis=dict(title="depth (m)"),
            aspectmode="data",
            bgcolor="#111",
            camera=dict(eye=dict(x=1.4, y=-1.4, z=0.9)),
        ),
    )
    return fig
