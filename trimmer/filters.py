"""Filter functions that produce/refine the keep-mask.

Each filter returns a fresh boolean mask of length N; the caller is
responsible for AND-combining it with the current `keep_mask` and calling
`Dataset.replace_mask` so the previous state is pushed onto the undo stack.

The functions are pure: they read the supplied arrays and never mutate the
shared dataset, which keeps them trivially testable.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def drop_positive(depth: np.ndarray) -> np.ndarray:
    """Keep only strictly negative depths."""
    return depth < 0


def percentile_clip(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Keep depths between the given percentiles of the currently-finite values.

    `lo` and `hi` are in [0, 100]. NaNs and inf are always dropped.
    """
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros_like(depth, dtype=bool)
    vals = depth[finite]
    z_lo, z_hi = np.percentile(vals, [lo, hi])
    return finite & (depth >= z_lo) & (depth <= z_hi)


def mad_clip(depth: np.ndarray, k: float = 5.0) -> np.ndarray:
    """Keep depths within k * MAD of the global median.

    MAD is the median absolute deviation; multiply by 1.4826 to approximate
    a standard deviation for normally-distributed data.
    """
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros_like(depth, dtype=bool)
    vals = depth[finite]
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    if mad == 0:
        return finite
    spread = 1.4826 * mad
    lo = med - k * spread
    hi = med + k * spread
    return finite & (depth >= lo) & (depth <= hi)


def grid_mad_clip(
    east: np.ndarray,
    north: np.ndarray,
    depth: np.ndarray,
    cell: float = 1.0,
    k: float = 4.0,
) -> np.ndarray:
    """Per-cell median-absolute-deviation outlier filter.

    Bins points into a 2D grid of `cell`-meter squares and drops any point
    whose depth is more than `k * 1.4826 * MAD` from that cell's median.
    Cells with fewer than 4 points are passed through unchanged (too few to
    establish a robust median).
    """
    finite = np.isfinite(east) & np.isfinite(north) & np.isfinite(depth)
    if not finite.any():
        return np.zeros_like(depth, dtype=bool)

    ix = np.floor(east / cell).astype(np.int64)
    iy = np.floor(north / cell).astype(np.int64)
    ix -= ix.min()
    iy -= iy.min()
    nx = int(ix.max()) + 1
    cell_id = iy * nx + ix
    cell_id[~finite] = -1

    # Sort points by cell so each cell occupies a contiguous run.
    order = np.argsort(cell_id, kind="stable")
    cell_sorted = cell_id[order]
    depth_sorted = depth[order]

    keep_sorted = np.ones(depth.size, dtype=bool)

    starts = np.flatnonzero(np.diff(np.r_[-2, cell_sorted]))
    ends = np.r_[starts[1:], depth.size]

    for s, e in zip(starts, ends):
        cid = cell_sorted[s]
        if cid < 0:
            keep_sorted[s:e] = False
            continue
        if (e - s) < 4:
            continue
        seg = depth_sorted[s:e]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        if mad == 0:
            continue
        spread = 1.4826 * mad
        lo = med - k * spread
        hi = med + k * spread
        keep_sorted[s:e] = (seg >= lo) & (seg <= hi)

    out = np.empty(depth.size, dtype=bool)
    out[order] = keep_sorted
    return out & finite


def depth_range(depth: np.ndarray, zmin: float, zmax: float) -> np.ndarray:
    """Keep depths in [zmin, zmax]."""
    return np.isfinite(depth) & (depth >= zmin) & (depth <= zmax)


def corridor_depth_range(
    depth: np.ndarray,
    corridor_idx: np.ndarray,
    zmin: float,
    zmax: float,
) -> np.ndarray:
    """Restrict a depth-range filter to the supplied corridor indices.

    Outside the corridor the returned mask is True (i.e. unaffected); inside
    the corridor only points within [zmin, zmax] survive.
    """
    out = np.ones_like(depth, dtype=bool)
    seg = depth[corridor_idx]
    keep_seg = np.isfinite(seg) & (seg >= zmin) & (seg <= zmax)
    out[corridor_idx] = keep_seg
    return out


def lasso_exclude(
    n: int,
    corridor_idx: np.ndarray,
    selected_local_indices: Sequence[int],
) -> np.ndarray:
    """Build a mask that drops the lasso-selected subset of the corridor.

    `selected_local_indices` are positions into `corridor_idx` (i.e. the
    indices of the points that appear in the cross-section figure that the
    user lassoed).
    """
    out = np.ones(n, dtype=bool)
    if len(selected_local_indices) == 0:
        return out
    sel_local = np.asarray(selected_local_indices, dtype=np.int64)
    global_idx = corridor_idx[sel_local]
    out[global_idx] = False
    return out


def power_threshold(power: np.ndarray, p_min: float, p_max: float) -> np.ndarray:
    """Keep returns whose power (dB) is within [p_min, p_max]."""
    return np.isfinite(power) & (power >= p_min) & (power <= p_max)


def parse_ping_spec(spec: str) -> np.ndarray:
    """Parse a ping-reject spec into a sorted unique int array.

    Accepts comma- and whitespace-separated tokens; each token is either an
    integer (e.g. `27346`) or an inclusive range (e.g. `27400-27410`).
    Returns an empty array if `spec` is empty/None.
    """
    if not spec:
        return np.empty(0, dtype=np.int64)
    bad: list[int] = []
    for raw in spec.replace(",", " ").split():
        token = raw.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                lo, hi = hi, lo
            bad.extend(range(lo, hi + 1))
        else:
            bad.append(int(token))
    return np.unique(np.asarray(bad, dtype=np.int64))


def ping_reject(ping: np.ndarray, bad_pings: np.ndarray | Iterable[int]) -> np.ndarray:
    """Drop any point whose ping number is in `bad_pings`."""
    bad = np.asarray(list(bad_pings), dtype=np.int64) if not isinstance(bad_pings, np.ndarray) else bad_pings
    if bad.size == 0:
        return np.ones(ping.size, dtype=bool)
    return ~np.isin(ping, bad)
