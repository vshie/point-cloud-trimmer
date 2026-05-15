"""Filter functions that produce/refine the keep-mask.

Each filter returns a fresh boolean mask of length N; the caller is
responsible for AND-combining it with the current `keep_mask` and calling
`Dataset.replace_mask` so the previous state is pushed onto the undo stack.

The functions are pure: they read the supplied arrays and never mutate the
shared dataset, which keeps them trivially testable.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import numpy as np

# Type alias for a progress callback used by the slow finishing filters.
# Receives (phase_label, fraction_in_0_1).
ProgressFn = Callable[[str, float], None]


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


# ---------------------------------------------------------------------------
# Finishing filters (slow; intended to run after the other tools as a final
# pass on already-trimmed data).
# ---------------------------------------------------------------------------


def power_weighted_grid_filter(
    east: np.ndarray,
    north: np.ndarray,
    depth: np.ndarray,
    power: np.ndarray,
    cell: float = 1.0,
    k: float = 4.0,
    top_pct: float = 50.0,
    subset_mask: Optional[np.ndarray] = None,
    progress: Optional[ProgressFn] = None,
) -> np.ndarray:
    """Per-cell MAD outlier filter, with the cell median computed only from
    the highest-power returns in the cell.

    For each 2D cell of size `cell` meters:
      1. Take the `top_pct` strongest returns by `power (dB)` as the
         "trusted" subset.
      2. Compute the median depth of that trusted subset.
      3. Compute MAD over **all** the cell's points around that trusted
         median.
      4. Keep points whose `|depth - trusted_median| <= k * 1.4826 * MAD`.

    Designed to suppress side-lobes and multipath (which are usually a few
    dB weaker than the main return) without throwing away weak returns
    globally.

    `subset_mask`, if given, restricts processing to those rows; points
    outside the subset always pass (True). Points inside the subset that
    fail the per-cell test fail in the returned mask.

    Cells with fewer than 4 trusted points (or where MAD == 0) pass
    through unchanged.
    """
    n = depth.size
    if subset_mask is None:
        subset_mask = np.ones(n, dtype=bool)
    finite = (
        np.isfinite(east) & np.isfinite(north)
        & np.isfinite(depth) & np.isfinite(power)
    )
    active = subset_mask & finite

    out = np.ones(n, dtype=bool)
    if not active.any():
        if progress:
            progress("power-weighted grid", 1.0)
        return out

    idx_active = np.flatnonzero(active)
    east_a = east[idx_active]
    north_a = north[idx_active]
    depth_a = depth[idx_active]
    power_a = power[idx_active]

    ix = np.floor(east_a / cell).astype(np.int64)
    iy = np.floor(north_a / cell).astype(np.int64)
    ix -= ix.min()
    iy -= iy.min()
    nx = int(ix.max()) + 1
    cell_id = iy * nx + ix

    # Sort by cell so each cell occupies a contiguous run.
    order = np.argsort(cell_id, kind="stable")
    cell_sorted = cell_id[order]
    depth_sorted = depth_a[order]
    power_sorted = power_a[order]
    idx_sorted = idx_active[order]

    starts = np.flatnonzero(np.diff(np.r_[-2, cell_sorted]))
    ends = np.r_[starts[1:], depth_sorted.size]

    n_cells = starts.size
    if progress:
        progress("power-weighted grid", 0.0)
    progress_step = max(1, n_cells // 100)
    spread_factor = 1.4826
    keep_local = np.ones(depth_sorted.size, dtype=bool)

    for ci, (s, e) in enumerate(zip(starts, ends)):
        n_pts = e - s
        if n_pts < 4:
            continue
        seg_depth = depth_sorted[s:e]
        seg_power = power_sorted[s:e]
        # Take the top `top_pct` % of points by power as the trusted subset.
        n_trust = max(1, int(np.ceil(n_pts * top_pct / 100.0)))
        if n_trust < n_pts:
            # argpartition is O(n) and faster than a full sort.
            cut = n_pts - n_trust
            part = np.argpartition(seg_power, cut)
            trusted = seg_depth[part[cut:]]
        else:
            trusted = seg_depth
        if trusted.size < 4:
            continue
        med = np.median(trusted)
        mad = np.median(np.abs(seg_depth - med))
        if mad == 0:
            continue
        spread = spread_factor * mad
        lo = med - k * spread
        hi = med + k * spread
        keep_local[s:e] = (seg_depth >= lo) & (seg_depth <= hi)

        if progress and (ci % progress_step == 0):
            progress("power-weighted grid", (ci + 1) / n_cells)

    if progress:
        progress("power-weighted grid", 1.0)

    rejected_local = ~keep_local
    if rejected_local.any():
        out[idx_sorted[rejected_local]] = False
    return out


def knn_sor(
    east: np.ndarray,
    north: np.ndarray,
    depth: np.ndarray,
    k: int = 8,
    m: float = 3.0,
    chunk_size: int = 500_000,
    subset_mask: Optional[np.ndarray] = None,
    progress: Optional[ProgressFn] = None,
) -> np.ndarray:
    """Statistical outlier removal via mean k-nearest-neighbor distance.

    For each kept point, compute the mean Euclidean distance to its `k`
    nearest 3D neighbors (excluding self) using a single `scipy.cKDTree`.
    Keep points whose mean distance is at most `mean + m * sigma` of the
    distribution across all kept points.

    Doesn't assume the surface is smooth - only that valid points are
    spatially near other valid points. Designed as a finishing pass that
    catches isolated multipath / side-lobe returns from already-trimmed
    data.

    Queries are issued in chunks of `chunk_size` rows with
    `workers=-1` (all cores) so memory peaks at one chunk of distances
    instead of N * k floats.

    `subset_mask`, if given, restricts the kNN computation to those rows;
    points outside the subset always pass (True).
    """
    # Local import so the rest of the module doesn't pay scipy's import cost.
    from scipy.spatial import cKDTree

    n = depth.size
    if subset_mask is None:
        subset_mask = np.ones(n, dtype=bool)
    finite = np.isfinite(east) & np.isfinite(north) & np.isfinite(depth)
    active = subset_mask & finite

    out = np.ones(n, dtype=bool)
    if not active.any():
        if progress:
            progress("kNN SOR", 1.0)
        return out

    idx_active = np.flatnonzero(active)
    n_sub = idx_active.size
    if n_sub <= k + 1:
        # Not enough neighbors to compute the statistic robustly.
        if progress:
            progress("kNN SOR", 1.0)
        return out

    if progress:
        progress("kNN: building kd-tree", 0.0)
    # cKDTree wants float64 contiguous; materialize once.
    pts = np.empty((n_sub, 3), dtype=np.float64)
    pts[:, 0] = east[idx_active]
    pts[:, 1] = north[idx_active]
    pts[:, 2] = depth[idx_active]
    tree = cKDTree(pts)

    mean_d = np.empty(n_sub, dtype=np.float64)
    cs = max(1, int(chunk_size))
    n_chunks = (n_sub + cs - 1) // cs
    for ci in range(n_chunks):
        a = ci * cs
        b = min(a + cs, n_sub)
        # query k+1 because the first neighbor is the point itself.
        dists, _ = tree.query(pts[a:b], k=k + 1, workers=-1)
        # dists shape: (m, k+1); drop the first column (self).
        if dists.ndim == 1:
            # Edge case: k+1 == 1; shouldn't happen given the guard above.
            mean_d[a:b] = dists
        else:
            mean_d[a:b] = dists[:, 1:].mean(axis=1)
        if progress:
            progress("kNN query", b / n_sub)

    global_mean = float(mean_d.mean())
    global_std = float(mean_d.std())
    threshold = global_mean + m * global_std

    keep_subset = mean_d <= threshold
    rejected_subset = ~keep_subset
    if rejected_subset.any():
        out[idx_active[rejected_subset]] = False

    if progress:
        progress("kNN SOR", 1.0)
    return out
