"""Compass-heading geometry for cross-section corridors.

The top-down map uses easting (X) on the horizontal axis and northing (Y)
on the vertical axis. Compass heading `theta` is degrees clockwise from
north, so heading 0 -> +Y, 90 -> +X.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Heading:
    """A user-defined cross-section heading.

    `cx`, `cy` are the origin of the cross-section in local meters (the
    midpoint of the drawn line). `theta_deg` is the compass azimuth in
    degrees [0, 360). `length` is the half-extent of the drawn line in
    meters, useful for the cross-section X-axis range.
    """

    cx: float
    cy: float
    theta_deg: float
    length: float = 0.0


def heading_from_line(x0: float, y0: float, x1: float, y1: float) -> Heading:
    """Compass heading from a two-point line drawn on the top-down map.

    The midpoint becomes the cross-section origin and the line's direction
    (from start to end) becomes the heading. `length` is the half-length of
    the drawn line in meters.
    """
    dx = x1 - x0
    dy = y1 - y0
    theta = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    length = 0.5 * float(np.hypot(dx, dy))
    return Heading(cx=float(cx), cy=float(cy), theta_deg=float(theta), length=length)


def project(
    east: np.ndarray,
    north: np.ndarray,
    heading: Heading,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate (east, north) into (along, across) for the given heading.

    `along` is signed distance in meters along the heading direction;
    `across` is signed perpendicular distance (left of heading is positive
    in the conventional right-handed sense).
    """
    rad = np.deg2rad(heading.theta_deg)
    dx = east - heading.cx
    dy = north - heading.cy
    along = dy * np.cos(rad) + dx * np.sin(rad)
    across = -dy * np.sin(rad) + dx * np.cos(rad)
    return along, across


def corridor(
    east: np.ndarray,
    north: np.ndarray,
    heading: Heading,
    half_width: float,
    keep_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indices of kept points inside the heading corridor, plus their along/across.

    Returns `(idx, along_idx, across_idx)` where `idx` is a global index
    array into the full point cloud (sorted by `along` for plotting) and
    `along_idx`/`across_idx` are the corresponding coordinates.
    """
    along, across = project(east, north, heading)
    in_corr = np.abs(across) <= half_width
    if keep_mask is not None:
        in_corr &= keep_mask
    idx = np.flatnonzero(in_corr)
    a = along[idx]
    order = np.argsort(a)
    idx = idx[order]
    return idx, along[idx], across[idx]
