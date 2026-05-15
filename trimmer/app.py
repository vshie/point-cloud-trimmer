"""Dash application layout and callbacks for the Omniscan3D trimmer.

State model:
    - The single source of truth is `trimmer.data.STATE.keep_mask`.
    - Callbacks store small derived values (heading, corridor cache, slider
      bounds) in `dcc.Store` components; large arrays are never serialized.
    - Each user action that mutates the mask first calls `STATE.push_history()`
      via `STATE.replace_mask()` so the undo deque captures the previous state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

from . import data as data_module
from . import filters as F
from . import views
from .geom import Heading, corridor, heading_from_line


# --- helpers --------------------------------------------------------------

def _heading_from_store(payload: Optional[dict]) -> Optional[Heading]:
    if not payload:
        return None
    return Heading(
        cx=float(payload["cx"]),
        cy=float(payload["cy"]),
        theta_deg=float(payload["theta_deg"]),
        length=float(payload.get("length", 0.0)),
    )


def _heading_to_store(h: Heading) -> dict:
    return dict(cx=h.cx, cy=h.cy, theta_deg=h.theta_deg, length=h.length)


def _apply_and_replace(new_mask: np.ndarray) -> tuple[int, int]:
    """AND `new_mask` with the current keep-mask and store with undo.

    Returns `(removed, kept_after)` so the caller can build a status string.
    """
    ds = data_module.get()
    before = ds.kept_count
    combined = ds.keep_mask & new_mask
    ds.replace_mask(combined)
    after = ds.kept_count
    return before - after, after


def _status(label: str, removed: int, kept: int) -> str:
    return f"{label}: removed {removed:,}, now keeping {kept:,}"


def _fmt_duration(seconds: float) -> str:
    """Human-friendly duration: '45s', '3m 24s', '1h 12m 5s'."""
    s = max(0, int(round(seconds)))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


# --- layout ---------------------------------------------------------------

def build_layout(ds: data_module.Dataset) -> html.Div:
    z_min, z_max = float(np.nanmin(ds.depth)), 0.0
    pwr_finite = ds.power[np.isfinite(ds.power)]
    if pwr_finite.size:
        p_min = float(pwr_finite.min())
        p_max = float(pwr_finite.max())
    else:
        p_min, p_max = 0.0, 120.0

    x_min, x_max = float(ds.east.min()), float(ds.east.max())
    y_min, y_max = float(ds.north.min()), float(ds.north.max())
    diag = float(np.hypot(x_max - x_min, y_max - y_min))

    return html.Div(
        style={
            "display": "grid",
            "gridTemplateRows": "auto auto 1fr",
            "height": "100vh",
            "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
            "backgroundColor": "#111",
            "color": "#eee",
        },
        children=[
            # Top bar -----------------------------------------------------
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "16px",
                    "padding": "10px 14px",
                    "borderBottom": "1px solid #333",
                    "flexWrap": "wrap",
                },
                children=[
                    html.Div(
                        style={"fontWeight": "bold"},
                        children=f"Omniscan3D Trimmer  \u2014  {ds.csv_path.name}",
                    ),
                    html.Div(id="kept-counter", style={"color": "#aaa"}),
                    html.Div(id="last-action", style={"color": "#9cf", "fontSize": "12px"}),
                    html.Button("Undo", id="btn-undo", n_clicks=0),
                    html.Button("Reset", id="btn-reset", n_clicks=0),
                    html.Button(
                        "Export trimmed CSV",
                        id="btn-export",
                        n_clicks=0,
                        style={
                            "backgroundColor": "#2e7d32",
                            "color": "white",
                            "border": "none",
                            "padding": "6px 12px",
                            "cursor": "pointer",
                        },
                    ),
                    html.Span(id="export-status", style={"color": "#9e9", "marginLeft": "6px"}),
                    dcc.Interval(
                        id="export-interval",
                        interval=500,
                        n_intervals=0,
                        disabled=True,
                    ),
                ],
            ),
            # Auto-filter row --------------------------------------------
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "20px",
                    "padding": "8px 14px",
                    "borderBottom": "1px solid #333",
                    "backgroundColor": "#181818",
                    "flexWrap": "wrap",
                },
                children=[
                    dcc.Checklist(
                        id="auto-filters",
                        options=[
                            {"label": " Percentile clip (0.5/99.5)", "value": "pct"},
                            {"label": " Global MAD (k=5)", "value": "mad"},
                            {"label": " Grid MAD (k=4, cell=1 m)", "value": "gridmad"},
                        ],
                        value=[],
                        inline=True,
                        labelStyle={"marginRight": "12px"},
                    ),
                    html.Div(
                        style={"display": "flex", "gap": "6px", "alignItems": "center"},
                        children=[
                            html.Label("grid cell (m):"),
                            dcc.Input(
                                id="grid-cell",
                                type="number",
                                value=1.0,
                                step=0.25,
                                min=0.1,
                                style={"width": "70px"},
                            ),
                            html.Label("k:"),
                            dcc.Input(
                                id="grid-k",
                                type="number",
                                value=4.0,
                                step=0.5,
                                min=1.0,
                                style={"width": "60px"},
                            ),
                            html.Button(
                                "Apply auto-filters",
                                id="btn-apply-auto",
                                n_clicks=0,
                            ),
                        ],
                    ),
                    html.Div(
                        style={"display": "flex", "gap": "6px", "alignItems": "center"},
                        children=[
                            html.Label("Reject pings:"),
                            dcc.Input(
                                id="ping-reject-spec",
                                type="text",
                                placeholder="27346, 27400-27410",
                                debounce=True,
                                style={"width": "220px"},
                            ),
                            html.Button(
                                "Apply ping reject",
                                id="btn-apply-pingreject",
                                n_clicks=0,
                            ),
                        ],
                    ),
                ],
            ),
            # Main split --------------------------------------------------
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "minmax(0, 1fr) minmax(0, 1fr)",
                    "gap": "8px",
                    "padding": "8px",
                    "overflow": "hidden",
                },
                children=[
                    # Left: top-down map + heading controls
                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateRows": "1fr auto",
                            "gap": "8px",
                            "minWidth": 0,
                        },
                        children=[
                            dcc.Graph(
                                id="topdown",
                                config={
                                    "modeBarButtonsToAdd": [
                                        "drawline",
                                        "eraseshape",
                                    ],
                                    "displaylogo": False,
                                },
                                style={"height": "100%", "minHeight": "300px"},
                            ),
                            html.Div(
                                style={
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "gap": "6px",
                                    "padding": "6px 4px",
                                    "borderTop": "1px solid #333",
                                },
                                children=[
                                    # Row 1: heading and origin inputs.
                                    html.Div(
                                        style={
                                            "display": "flex",
                                            "gap": "14px",
                                            "alignItems": "center",
                                            "flexWrap": "wrap",
                                        },
                                        children=[
                                            html.Div(id="heading-readout", style={"color": "#9cf"}),
                                            html.Label("heading (\u00b0):"),
                                            dcc.Input(
                                                id="heading-input",
                                                type="number",
                                                min=0,
                                                max=359.9,
                                                step=0.1,
                                                debounce=True,
                                                style={"width": "80px"},
                                            ),
                                            html.Label("origin E:"),
                                            dcc.Input(
                                                id="origin-x",
                                                type="number",
                                                step=0.5,
                                                debounce=True,
                                                style={"width": "100px"},
                                            ),
                                            html.Label("origin N:"),
                                            dcc.Input(
                                                id="origin-y",
                                                type="number",
                                                step=0.5,
                                                debounce=True,
                                                style={"width": "100px"},
                                            ),
                                        ],
                                    ),
                                    # Row 2: corridor half-width slider on its own row so
                                    # it has the full bar width to render into.
                                    html.Div(
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": "auto auto 1fr",
                                            "gap": "10px",
                                            "alignItems": "center",
                                        },
                                        children=[
                                            html.Label(
                                                "Corridor half-width (m):",
                                                style={"whiteSpace": "nowrap"},
                                                title=(
                                                    "Half the across-track width of the corridor "
                                                    "strip around the heading line."
                                                ),
                                            ),
                                            html.Span(
                                                id="half-width-value",
                                                style={
                                                    "color": "#9cf",
                                                    "minWidth": "56px",
                                                    "textAlign": "right",
                                                },
                                                children="0.50 m",
                                            ),
                                            html.Div(
                                                dcc.Slider(
                                                    id="half-width",
                                                    min=0.1,
                                                    max=max(5.0, diag / 50),
                                                    step=0.1,
                                                    value=0.5,
                                                    tooltip={"placement": "bottom"},
                                                    marks=None,
                                                ),
                                                style={"width": "100%", "padding": "0 8px"},
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # Right: tabs
                    dcc.Tabs(
                        id="right-tabs",
                        value="tab-depth",
                        children=[
                            dcc.Tab(
                                label="Depth distribution",
                                value="tab-depth",
                                style={"backgroundColor": "#181818", "color": "#eee"},
                                selected_style={"backgroundColor": "#222", "color": "#fff"},
                                children=html.Div(
                                    style={
                                        "display": "grid",
                                        "gridTemplateRows": "1fr auto auto",
                                        "height": "100%",
                                        "minHeight": "300px",
                                    },
                                    children=[
                                        dcc.Graph(id="depth-hist", style={"height": "100%"}),
                                        dcc.Checklist(
                                            id="hist-corridor-only",
                                            options=[
                                                {"label": " histogram of corridor only", "value": "on"}
                                            ],
                                            value=[],
                                            style={"padding": "4px 8px"},
                                        ),
                                        html.Div(
                                            style={
                                                "padding": "0 14px 8px 14px",
                                                "display": "flex",
                                                "gap": "10px",
                                                "alignItems": "center",
                                            },
                                            children=[
                                                dcc.RangeSlider(
                                                    id="depth-range",
                                                    min=z_min,
                                                    max=z_max,
                                                    step=0.01,
                                                    value=[z_min, z_max],
                                                    tooltip={"placement": "bottom"},
                                                    marks=None,
                                                ),
                                                html.Button(
                                                    "Apply depth range",
                                                    id="btn-apply-depth",
                                                    n_clicks=0,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ),
                            dcc.Tab(
                                label="Power distribution",
                                value="tab-power",
                                style={"backgroundColor": "#181818", "color": "#eee"},
                                selected_style={"backgroundColor": "#222", "color": "#fff"},
                                children=html.Div(
                                    style={
                                        "display": "grid",
                                        "gridTemplateRows": "1fr auto",
                                        "height": "100%",
                                        "minHeight": "300px",
                                    },
                                    children=[
                                        dcc.Graph(id="power-hist", style={"height": "100%"}),
                                        html.Div(
                                            style={
                                                "padding": "0 14px 8px 14px",
                                                "display": "flex",
                                                "gap": "10px",
                                                "alignItems": "center",
                                            },
                                            children=[
                                                dcc.RangeSlider(
                                                    id="power-range",
                                                    min=p_min,
                                                    max=p_max,
                                                    step=0.1,
                                                    value=[p_min, p_max],
                                                    tooltip={"placement": "bottom"},
                                                    marks=None,
                                                ),
                                                html.Button(
                                                    "Apply power range",
                                                    id="btn-apply-power",
                                                    n_clicks=0,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ),
                            dcc.Tab(
                                label="Cross-section",
                                value="tab-xsec",
                                style={"backgroundColor": "#181818", "color": "#eee"},
                                selected_style={"backgroundColor": "#222", "color": "#fff"},
                                children=html.Div(
                                    style={
                                        "display": "grid",
                                        "gridTemplateRows": "1fr auto",
                                        "height": "100%",
                                        "minHeight": "300px",
                                    },
                                    children=[
                                        dcc.Graph(
                                            id="cross-section",
                                            config={"displaylogo": False},
                                            style={"height": "100%"},
                                        ),
                                        html.Div(
                                            style={
                                                "padding": "6px 14px",
                                                "display": "flex",
                                                "gap": "10px",
                                                "alignItems": "center",
                                            },
                                            children=[
                                                html.Button(
                                                    "Exclude lasso selection",
                                                    id="btn-exclude-lasso",
                                                    n_clicks=0,
                                                ),
                                                html.Button(
                                                    "Keep only lasso selection (in corridor)",
                                                    id="btn-keep-lasso",
                                                    n_clicks=0,
                                                ),
                                                html.Span(id="xsec-status", style={"color": "#999"}),
                                            ],
                                        ),
                                    ],
                                ),
                            ),
                        ],
                    ),
                ],
            ),
            # Hidden state stores ---------------------------------------
            dcc.Store(id="mask-version", data=0),
            dcc.Store(id="heading-store"),
            dcc.Store(id="corridor-version", data=0),
            dcc.Store(id="xsec-hover-sink"),
            dcc.Store(
                id="bounds-store",
                data=dict(
                    x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                    z_min=z_min, z_max=z_max, p_min=p_min, p_max=p_max,
                ),
            ),
        ],
    )


# --- callbacks ------------------------------------------------------------

def register_callbacks(app: Dash) -> None:
    @app.callback(
        Output("topdown", "figure"),
        Output("kept-counter", "children"),
        Input("mask-version", "data"),
        Input("heading-store", "data"),
        Input("half-width", "value"),
    )
    def _update_topdown(_mask_version, heading_payload, half_width):
        ds = data_module.get()
        heading = _heading_from_store(heading_payload)
        fig = views.top_down_figure(
            east=ds.east,
            north=ds.north,
            depth=ds.depth,
            keep_mask=ds.keep_mask,
            heading=heading,
            half_width=float(half_width or 0.5),
        )
        counter = f"kept {ds.kept_count:,} / {ds.n:,}"
        return fig, counter

    @app.callback(
        Output("heading-store", "data"),
        Output("heading-readout", "children"),
        Output("heading-input", "value"),
        Output("origin-x", "value"),
        Output("origin-y", "value"),
        Input("topdown", "relayoutData"),
        Input("heading-input", "value"),
        Input("origin-x", "value"),
        Input("origin-y", "value"),
        State("heading-store", "data"),
        State("bounds-store", "data"),
        prevent_initial_call=True,
    )
    def _update_heading(relayout, h_in, ox_in, oy_in, current, bounds):
        trigger = ctx.triggered_id
        heading = _heading_from_store(current)

        if trigger == "topdown" and relayout:
            shapes = relayout.get("shapes") or []
            new_shape = None
            for key, val in relayout.items():
                if key.startswith("shapes[") and isinstance(val, dict):
                    if val.get("type") == "line":
                        new_shape = val
            if not new_shape:
                for s in shapes:
                    if isinstance(s, dict) and s.get("type") == "line":
                        new_shape = s
            if new_shape is None:
                return no_update, no_update, no_update, no_update, no_update
            try:
                x0 = float(new_shape["x0"])
                y0 = float(new_shape["y0"])
                x1 = float(new_shape["x1"])
                y1 = float(new_shape["y1"])
            except (KeyError, TypeError, ValueError):
                return no_update, no_update, no_update, no_update, no_update
            heading = heading_from_line(x0, y0, x1, y1)

        elif trigger == "heading-input" and h_in is not None and heading is not None:
            heading = Heading(
                cx=heading.cx,
                cy=heading.cy,
                theta_deg=float(h_in) % 360.0,
                length=heading.length,
            )
        elif trigger in ("origin-x", "origin-y") and heading is not None:
            cx = float(ox_in) if ox_in is not None else heading.cx
            cy = float(oy_in) if oy_in is not None else heading.cy
            heading = Heading(
                cx=cx, cy=cy, theta_deg=heading.theta_deg, length=heading.length
            )
        else:
            return no_update, no_update, no_update, no_update, no_update

        if heading is None:
            return no_update, no_update, no_update, no_update, no_update

        # Fallback line length if the user only typed a heading without drawing.
        if heading.length <= 0:
            diag = float(np.hypot(bounds["x_max"] - bounds["x_min"], bounds["y_max"] - bounds["y_min"]))
            heading = Heading(cx=heading.cx, cy=heading.cy, theta_deg=heading.theta_deg, length=diag * 0.25)

        readout = f"heading {heading.theta_deg:6.1f}\u00b0  origin ({heading.cx:.1f}, {heading.cy:.1f}) m"
        return (
            _heading_to_store(heading),
            readout,
            round(heading.theta_deg, 1),
            round(heading.cx, 2),
            round(heading.cy, 2),
        )

    @app.callback(
        Output("cross-section", "figure"),
        Output("corridor-version", "data"),
        Input("heading-store", "data"),
        Input("half-width", "value"),
        Input("mask-version", "data"),
        Input("depth-range", "value"),
        State("corridor-version", "data"),
    )
    def _update_cross_section(heading_payload, half_width, _mv, depth_rng, corr_version):
        ds = data_module.get()
        heading = _heading_from_store(heading_payload)
        if heading is None:
            ds.cross_section_sample_local = None
            return views._empty_figure("Cross-section (draw a line on the map)"), corr_version
        hw = float(half_width or 0.5)
        idx, along_idx, across_idx = corridor(
            east=ds.east,
            north=ds.north,
            heading=heading,
            half_width=hw,
            keep_mask=ds.keep_mask,
        )
        # Caches kept on the Dataset rather than in a dcc.Store because the
        # arrays can be many MB.
        ds.corridor_idx = idx
        fig, sample_local = views.cross_section_figure(
            corridor_idx=idx,
            along=along_idx,
            across=across_idx,
            depth=ds.depth,
            east=ds.east,
            north=ds.north,
            heading=heading,
            half_width=hw,
            selected_depth_range=tuple(depth_rng) if depth_rng else None,
        )
        ds.cross_section_sample_local = sample_local
        return fig, (corr_version or 0) + 1

    @app.callback(
        Output("depth-hist", "figure"),
        Input("mask-version", "data"),
        Input("depth-range", "value"),
        Input("hist-corridor-only", "value"),
        Input("corridor-version", "data"),
    )
    def _update_depth_hist(_mv, depth_rng, corridor_only, _cv):
        ds = data_module.get()
        if corridor_only and ds.corridor_idx is not None and ds.corridor_idx.size:
            mask = np.zeros(ds.n, dtype=bool)
            mask[ds.corridor_idx] = True
            mask &= ds.keep_mask
            title = "Depth distribution (corridor)"
        else:
            mask = ds.keep_mask
            title = "Depth distribution"
        return views.depth_histogram_figure(
            depth=ds.depth,
            mask=mask,
            selected_range=tuple(depth_rng) if depth_rng else None,
            title=title,
        )

    @app.callback(
        Output("power-hist", "figure"),
        Input("mask-version", "data"),
        Input("power-range", "value"),
    )
    def _update_power_hist(_mv, power_rng):
        ds = data_module.get()
        return views.power_histogram_figure(
            power=ds.power,
            mask=ds.keep_mask,
            selected_range=tuple(power_rng) if power_rng else None,
        )

    @app.callback(
        Output("mask-version", "data", allow_duplicate=True),
        Output("last-action", "children", allow_duplicate=True),
        Input("btn-apply-auto", "n_clicks"),
        State("auto-filters", "value"),
        State("grid-cell", "value"),
        State("grid-k", "value"),
        State("mask-version", "data"),
        prevent_initial_call=True,
    )
    def _apply_auto(_n, choices, cell, k, version):
        if not choices:
            return no_update, "auto-filters: no checkboxes selected"
        ds = data_module.get()
        m = np.ones(ds.n, dtype=bool)
        applied: list[str] = []
        if "pct" in choices:
            m &= F.percentile_clip(ds.depth, 0.5, 99.5)
            applied.append("percentile 0.5/99.5")
        if "mad" in choices:
            m &= F.mad_clip(ds.depth, k=5.0)
            applied.append("global MAD k=5")
        if "gridmad" in choices:
            m &= F.grid_mad_clip(
                ds.east, ds.north, ds.depth,
                cell=float(cell or 1.0),
                k=float(k or 4.0),
            )
            applied.append(f"grid MAD cell={float(cell or 1.0):g} k={float(k or 4.0):g}")
        removed, kept = _apply_and_replace(m)
        return (version or 0) + 1, _status(f"auto-filters [{', '.join(applied)}]", removed, kept)

    @app.callback(
        Output("mask-version", "data", allow_duplicate=True),
        Output("last-action", "children", allow_duplicate=True),
        Input("btn-apply-depth", "n_clicks"),
        State("depth-range", "value"),
        State("mask-version", "data"),
        prevent_initial_call=True,
    )
    def _apply_depth(_n, depth_rng, version):
        if not depth_rng:
            return no_update, no_update
        ds = data_module.get()
        zmin, zmax = float(depth_rng[0]), float(depth_rng[1])
        removed, kept = _apply_and_replace(F.depth_range(ds.depth, zmin, zmax))
        return (version or 0) + 1, _status(
            f"depth range [{zmin:.2f}, {zmax:.2f}] m", removed, kept
        )

    @app.callback(
        Output("mask-version", "data", allow_duplicate=True),
        Output("last-action", "children", allow_duplicate=True),
        Input("btn-apply-power", "n_clicks"),
        State("power-range", "value"),
        State("mask-version", "data"),
        prevent_initial_call=True,
    )
    def _apply_power(_n, power_rng, version):
        if not power_rng:
            return no_update, no_update
        ds = data_module.get()
        pmin, pmax = float(power_rng[0]), float(power_rng[1])
        removed, kept = _apply_and_replace(F.power_threshold(ds.power, pmin, pmax))
        return (version or 0) + 1, _status(
            f"power range [{pmin:.1f}, {pmax:.1f}] dB", removed, kept
        )

    @app.callback(
        Output("mask-version", "data", allow_duplicate=True),
        Output("last-action", "children", allow_duplicate=True),
        Input("btn-apply-pingreject", "n_clicks"),
        State("ping-reject-spec", "value"),
        State("mask-version", "data"),
        prevent_initial_call=True,
    )
    def _apply_pingreject(_n, spec, version):
        if not spec:
            return no_update, "ping reject: spec is empty"
        ds = data_module.get()
        bad = F.parse_ping_spec(spec)
        if bad.size == 0:
            return no_update, "ping reject: spec parsed to zero pings"
        removed, kept = _apply_and_replace(F.ping_reject(ds.ping, bad))
        return (version or 0) + 1, _status(
            f"ping reject ({bad.size:,} pings)", removed, kept
        )

    @app.callback(
        Output("mask-version", "data", allow_duplicate=True),
        Output("xsec-status", "children"),
        Output("last-action", "children", allow_duplicate=True),
        Input("btn-exclude-lasso", "n_clicks"),
        Input("btn-keep-lasso", "n_clicks"),
        State("cross-section", "selectedData"),
        State("mask-version", "data"),
        prevent_initial_call=True,
    )
    def _apply_lasso(_excl, _keep, selected, version):
        ds = data_module.get()
        if ds.corridor_idx is None or ds.corridor_idx.size == 0:
            return no_update, "no corridor; draw a heading line first", no_update
        if not selected or not selected.get("points"):
            return no_update, "no points selected", no_update
        points = selected["points"]
        sample = ds.cross_section_sample_local

        # Prefer pointIndex against the cached sample; customdata is a fallback
        # for older Plotly versions where pointIndex is occasionally absent.
        plotted_idx: list[int] = []
        for p in points:
            pi = p.get("pointIndex")
            if pi is None:
                pi = p.get("pointNumber")
            if pi is not None:
                plotted_idx.append(int(pi))

        if plotted_idx and sample is not None and sample.size:
            arr = np.asarray(plotted_idx, dtype=np.int64)
            valid = (arr >= 0) & (arr < sample.size)
            local = sample[arr[valid]]
        else:
            cd_list = []
            for p in points:
                cd = p.get("customdata")
                if cd is None:
                    continue
                cd_list.append(cd[0] if isinstance(cd, (list, tuple)) else cd)
            if not cd_list:
                return no_update, "selection had no plotted points", no_update
            local = np.asarray(cd_list, dtype=np.int64)

        if local.size == 0:
            return no_update, "selection had no plotted points", no_update

        global_idx = ds.corridor_idx[local]

        trigger = ctx.triggered_id
        before = ds.kept_count
        if trigger == "btn-exclude-lasso":
            new_mask = ds.keep_mask.copy()
            new_mask[global_idx] = False
            ds.replace_mask(new_mask)
            label = f"lasso exclude ({local.size:,} sel.)"
        else:
            new_mask = ds.keep_mask.copy()
            corridor_mask = np.zeros(ds.n, dtype=bool)
            corridor_mask[ds.corridor_idx] = True
            new_mask &= ~corridor_mask
            new_mask[global_idx] = True
            ds.replace_mask(new_mask)
            label = f"lasso keep-only ({local.size:,} sel.)"
        removed = before - ds.kept_count
        kept_after = ds.kept_count
        return (
            (version or 0) + 1,
            f"{label}: removed {removed:,}",
            _status(label, removed, kept_after),
        )

    # Live readout for the half-width slider so the current value is always
    # visible without hovering for the tooltip.
    app.clientside_callback(
        """
        function(v) {
            if (v === null || v === undefined) return '';
            var n = Number(v);
            if (!isFinite(n)) return '';
            return n.toFixed(2) + ' m';
        }
        """,
        Output("half-width-value", "children"),
        Input("half-width", "value"),
    )

    # Hover-marker: clientside callback that calls Plotly.restyle directly on
    # the top-down chart. Bypasses the server entirely so hover stays smooth,
    # and uses the underlying Plotly.js API so the marker reliably re-renders
    # from an initially empty trace.
    app.clientside_callback(
        """
        function(hoverData) {
            try {
                var idx = %d;
                var root = document.getElementById('topdown');
                if (!root) return '';
                // Plotly attaches the chart as the first descendant with the
                // js-plotly-plot class.
                var gd = root.classList && root.classList.contains('js-plotly-plot')
                    ? root
                    : root.querySelector('.js-plotly-plot');
                if (!gd || !window.Plotly) return '';
                var x = [], y = [];
                if (hoverData && hoverData.points && hoverData.points.length > 0) {
                    var cd = hoverData.points[0].customdata;
                    if (cd && cd.length >= 3) {
                        x = [cd[1]];
                        y = [cd[2]];
                    }
                }
                window.Plotly.restyle(gd, {x: [x], y: [y]}, [idx]);
            } catch (e) {
                if (window.console && console.warn) {
                    console.warn('xsec-hover restyle failed:', e);
                }
            }
            return '';
        }
        """ % views.XSEC_HOVER_TRACE,
        Output("xsec-hover-sink", "data"),
        Input("cross-section", "hoverData"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("mask-version", "data", allow_duplicate=True),
        Output("last-action", "children", allow_duplicate=True),
        Input("btn-undo", "n_clicks"),
        State("mask-version", "data"),
        prevent_initial_call=True,
    )
    def _undo(_n, version):
        ds = data_module.get()
        if not ds.undo():
            return no_update, "undo: nothing to undo"
        return (version or 0) + 1, f"undo: now keeping {ds.kept_count:,}"

    @app.callback(
        Output("mask-version", "data", allow_duplicate=True),
        Output("last-action", "children", allow_duplicate=True),
        Input("btn-reset", "n_clicks"),
        State("mask-version", "data"),
        prevent_initial_call=True,
    )
    def _reset(_n, version):
        ds = data_module.get()
        ds.reset()
        return (version or 0) + 1, f"reset: now keeping {ds.kept_count:,}"

    @app.callback(
        Output("export-status", "children"),
        Output("export-interval", "disabled"),
        Output("btn-export", "disabled"),
        Input("btn-export", "n_clicks"),
        prevent_initial_call=True,
    )
    def _start_export(_n):
        ds = data_module.get()
        if ds.is_exporting():
            return "export already in progress", False, True
        out = ds.csv_path.with_name(ds.csv_path.stem + "_trimmed.csv")
        ds.start_export(out)
        return f"starting export -> {out.name} ...", False, True

    @app.callback(
        Output("export-status", "children", allow_duplicate=True),
        Output("export-interval", "disabled", allow_duplicate=True),
        Output("btn-export", "disabled", allow_duplicate=True),
        Input("export-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def _poll_export(_n):
        ds = data_module.get()
        s = ds.export_state
        phase = s.get("phase", "idle")
        if phase == "running":
            rows_p = int(s.get("rows_processed", 0))
            rows_w = int(s.get("rows_written", 0))
            rows_t = int(s.get("rows_total", 1))
            progress = float(s.get("progress", 0.0))
            elapsed = float(s.get("elapsed", 0.0))
            pct = progress * 100.0
            if elapsed > 0.5 and progress > 0.01:
                eta = elapsed / progress - elapsed
                eta_s = f", ETA {_fmt_duration(eta)}"
            else:
                eta_s = ""
            msg = (
                f"exporting {pct:5.1f}%  scanned {rows_p:,}/{rows_t:,},  "
                f"wrote {rows_w:,},  elapsed {_fmt_duration(elapsed)}{eta_s}"
            )
            return msg, False, True
        if phase == "done":
            return (
                f"wrote {int(s.get('rows_written', 0)):,} / {int(s.get('rows_total', 0)):,} "
                f"rows to {Path(s.get('out_path') or '').name} "
                f"in {_fmt_duration(float(s.get('elapsed', 0.0)))}",
                True,
                False,
            )
        if phase == "error":
            return (
                f"export failed: {s.get('error', 'unknown error')}",
                True,
                False,
            )
        return no_update, True, False


# --- entry point ----------------------------------------------------------

def build_app(csv_path: str | Path) -> Dash:
    ds = data_module.load(csv_path)
    assets_path = (Path(__file__).parent / "assets").resolve()
    app = Dash(
        __name__,
        title=f"Omniscan3D Trimmer - {ds.csv_path.name}",
        assets_folder=str(assets_path),
    )
    app.layout = build_layout(ds)
    register_callbacks(app)
    return app
