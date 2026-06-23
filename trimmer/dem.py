"""Gridded DEM export for QGIS contour extraction.

The trimmer's main export streams the original CSV row-for-row. This module
adds a *surface* export instead: it bins the kept points' true UTM
coordinates onto a regular grid and writes an Esri ASCII Grid (``.asc``)
plus a ``.prj`` sidecar. QGIS reads that pair natively, so the user can
drop it in and run ``Raster > Extraction > Contour`` (``gdal_contour``)
to pull isolines directly -- no interpolation step and no GDAL/rasterio
dependency on our side.

The vertical value gridded is ``altitude (m)`` (the same column the app
trims on), aggregated per cell. UTM easting/northing are used for the
horizontal grid because, unlike the ``local m`` columns, they share a
single origin across runs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np

NODATA = -9999.0

# Aggregations that can be computed in a single streaming pass (no need to
# hold every value that falls in a cell).
AGGREGATIONS = ("mean", "min", "max")


def parse_utm_epsg(projection: Optional[str]) -> Optional[int]:
    """Map an Omniscan3D ``coordinate projection`` string to an EPSG code.

    Recognizes strings like ``"UTM zone 5N"`` / ``"UTM zone 17S"`` and
    returns the WGS84 / UTM EPSG code (326xx north, 327xx south). Returns
    ``None`` if the string is missing or not a UTM specifier.
    """
    if not projection:
        return None
    m = re.search(r"UTM\s*zone\s*(\d{1,2})\s*([NS])", str(projection), re.IGNORECASE)
    if not m:
        return None
    zone = int(m.group(1))
    if not (1 <= zone <= 60):
        return None
    hemi = m.group(2).upper()
    return (32600 if hemi == "N" else 32700) + zone


def utm_wkt(epsg: int) -> str:
    """Build an OGC WKT1 string for a WGS84 / UTM zone EPSG code.

    Used for the ``.prj`` sidecar so QGIS georeferences the grid without a
    dependency on pyproj/GDAL.
    """
    north = 32600 < epsg <= 32660
    zone = epsg - (32600 if north else 32700)
    central_meridian = zone * 6 - 183
    false_northing = 0 if north else 10_000_000
    hemi = "N" if north else "S"
    return (
        f'PROJCS["WGS 84 / UTM zone {zone}{hemi}",'
        'GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
        'AUTHORITY["EPSG","6326"]],'
        'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
        'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
        'AUTHORITY["EPSG","4326"]],'
        'PROJECTION["Transverse_Mercator"],'
        'PARAMETER["latitude_of_origin",0],'
        f'PARAMETER["central_meridian",{central_meridian}],'
        'PARAMETER["scale_factor",0.9996],'
        'PARAMETER["false_easting",500000],'
        f'PARAMETER["false_northing",{false_northing}],'
        'UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
        'AXIS["Easting",EAST],AXIS["Northing",NORTH],'
        f'AUTHORITY["EPSG","{epsg}"]]'
    )


class GridAccumulator:
    """Streaming per-cell accumulator for a regular DEM grid.

    Horizontal extent and cell size are fixed up front (computed in a first
    pass over the data); points are then added batch-by-batch. The grid is
    stored row-major with row 0 = the NORTH-most row, matching the Esri
    ASCII Grid convention so it can be written out directly.
    """

    def __init__(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        cell: float,
        agg: str = "mean",
    ) -> None:
        if cell <= 0:
            raise ValueError("cell size must be positive")
        if agg not in AGGREGATIONS:
            raise ValueError(f"agg must be one of {AGGREGATIONS}, got {agg!r}")
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.x_max = float(x_max)
        self.y_max = float(y_max)
        self.cell = float(cell)
        self.agg = agg
        self.ncols = int(np.floor((self.x_max - self.x_min) / self.cell)) + 1
        self.nrows = int(np.floor((self.y_max - self.y_min) / self.cell)) + 1
        self.ncells = self.ncols * self.nrows
        # Lower-left corner of the lower-left cell (Esri ASCII convention).
        # Columns are anchored at the west edge (x_min); rows are anchored at
        # the north edge (y_max, since grid row 0 is the north-most row), so
        # the south edge is y_max - nrows*cell.
        self.xllcorner = self.x_min
        self.yllcorner = self.y_max - self.nrows * self.cell

        if agg == "mean":
            self._sum = np.zeros(self.ncells, dtype=np.float64)
            self._cnt = np.zeros(self.ncells, dtype=np.int64)
        elif agg == "min":
            self._val = np.full(self.ncells, np.inf, dtype=np.float64)
        else:  # max
            self._val = np.full(self.ncells, -np.inf, dtype=np.float64)

    def _flat_index(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        col = np.floor((x - self.x_min) / self.cell).astype(np.int64)
        # Row 0 is the north-most row, so measure down from y_max.
        row = np.floor((self.y_max - y) / self.cell).astype(np.int64)
        np.clip(col, 0, self.ncols - 1, out=col)
        np.clip(row, 0, self.nrows - 1, out=row)
        return row * self.ncols + col

    def add(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> int:
        """Add a batch of points. Returns the count of finite points used."""
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if not finite.all():
            x, y, z = x[finite], y[finite], z[finite]
        if x.size == 0:
            return 0
        flat = self._flat_index(x, y)
        if self.agg == "mean":
            np.add.at(self._sum, flat, z)
            np.add.at(self._cnt, flat, 1)
        elif self.agg == "min":
            np.minimum.at(self._val, flat, z)
        else:
            np.maximum.at(self._val, flat, z)
        return int(x.size)

    def result(self) -> np.ndarray:
        """Return the finished grid (row 0 = north), NODATA for empty cells."""
        if self.agg == "mean":
            with np.errstate(invalid="ignore", divide="ignore"):
                grid = np.where(self._cnt > 0, self._sum / np.maximum(self._cnt, 1), NODATA)
            populated = int((self._cnt > 0).sum())
        else:
            filled = np.isfinite(self._val)
            grid = np.where(filled, self._val, NODATA)
            populated = int(filled.sum())
        self.populated_cells = populated
        return grid.reshape(self.nrows, self.ncols)


def write_asc(
    out_path: str | Path,
    grid: np.ndarray,
    x_min: float,
    y_min: float,
    cell: float,
    nodata: float = NODATA,
) -> None:
    """Write an Esri ASCII Grid (``.asc``). ``grid[0]`` must be the north row."""
    nrows, ncols = grid.shape
    with open(out_path, "w", newline="\n") as f:
        f.write(f"ncols {ncols}\n")
        f.write(f"nrows {nrows}\n")
        f.write(f"xllcorner {x_min:.6f}\n")
        f.write(f"yllcorner {y_min:.6f}\n")
        f.write(f"cellsize {cell:.6f}\n")
        f.write(f"NODATA_value {nodata:g}\n")
        np.savetxt(f, grid, fmt="%.3f", delimiter=" ")


def write_prj(out_path: str | Path, epsg: Optional[int]) -> None:
    """Write a ``.prj`` sidecar with the UTM WKT, if the EPSG is known."""
    if not epsg:
        return
    with open(out_path, "w", newline="\n") as f:
        f.write(utm_wkt(int(epsg)))


def write_geotiff(
    out_path: str | Path,
    grid: np.ndarray,
    x_min: float,
    y_min: float,
    cell: float,
    epsg: Optional[int],
    nodata: float = NODATA,
) -> None:
    """Write a compressed, tiled GeoTIFF with embedded CRS and overviews.

    ``grid[0]`` must be the NORTH-most row (as produced by
    :class:`GridAccumulator`). Uses rasterio/GDAL: float32, LZW compression,
    internal tiling, a NoData tag, and average-resampled overviews so QGIS
    renders large grids quickly. Imported lazily so the ``.asc`` path has no
    GDAL dependency.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin

    nrows, ncols = grid.shape
    # Row 0 is the north-most row, so the raster origin (top-left corner) is at
    # the west edge and the north edge (y_min is the lower-left corner).
    north_edge = y_min + nrows * cell
    transform = from_origin(x_min, north_edge, cell, cell)
    crs = CRS.from_epsg(int(epsg)) if epsg else None
    data = np.ascontiguousarray(grid, dtype=np.float32)

    profile = {
        "driver": "GTiff",
        "height": nrows,
        "width": ncols,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
        "predictor": 3,  # floating-point predictor: better ratio for DEMs
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(data, 1)
        # Internal overviews for fast pan/zoom in QGIS on large grids.
        factors = [f for f in (2, 4, 8, 16, 32) if min(nrows, ncols) // f >= 1]
        if factors:
            dst.build_overviews(factors, Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")


# --- Google Earth (KMZ ground overlay) -----------------------------------

# A handful of viridis anchor colors; we linearly interpolate between them.
# viridis is perceptually uniform and reads well as a depth ramp (dark = deep).
_VIRIDIS = np.array(
    [
        [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
        [38, 130, 142], [31, 158, 137], [53, 183, 121], [110, 206, 88],
        [181, 222, 43], [253, 231, 37],
    ],
    dtype=np.float64,
)


def colorize(
    grid: np.ndarray,
    nodata: float = NODATA,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> tuple[np.ndarray, float, float]:
    """Map a float grid to an RGBA uint8 image via the viridis ramp.

    Empty/NoData cells become fully transparent. ``vmin``/``vmax`` default to
    the 2nd/98th percentiles of the finite data so a few outliers don't wash
    out the gradient. Returns ``(rgba, vmin, vmax)``.
    """
    valid = np.isfinite(grid) & (grid != nodata)
    fin = grid[valid]
    if fin.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        if vmin is None:
            vmin = float(np.percentile(fin, 2))
        if vmax is None:
            vmax = float(np.percentile(fin, 98))
        if vmax <= vmin:
            vmax = vmin + 1.0

    norm = np.clip((grid - vmin) / (vmax - vmin), 0.0, 1.0)
    xs = np.linspace(0.0, 1.0, len(_VIRIDIS))
    h, w = grid.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = np.interp(norm, xs, _VIRIDIS[:, 0]).astype(np.uint8)
    rgba[..., 1] = np.interp(norm, xs, _VIRIDIS[:, 1]).astype(np.uint8)
    rgba[..., 2] = np.interp(norm, xs, _VIRIDIS[:, 2]).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    return rgba, float(vmin), float(vmax)


def _reproject_to_4326(grid, x_min, y_min, cell, epsg, nodata):
    """Warp a north-up UTM grid to EPSG:4326. Returns (dst, (w, s, e, n))."""
    import rasterio  # noqa: F401  (ensures GDAL env is initialized)
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    nrows, ncols = grid.shape
    north_edge = y_min + nrows * cell
    src_transform = from_origin(x_min, north_edge, cell, cell)
    src_crs = CRS.from_epsg(int(epsg))
    dst_crs = CRS.from_epsg(4326)
    left, bottom, right, top = x_min, y_min, x_min + ncols * cell, north_edge

    dst_transform, dw, dh = calculate_default_transform(
        src_crs, dst_crs, ncols, nrows, left, bottom, right, top
    )
    dst = np.full((dh, dw), nodata, dtype=np.float32)
    reproject(
        source=np.ascontiguousarray(grid, dtype=np.float32),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=nodata,
        dst_nodata=nodata,
        resampling=Resampling.nearest,
    )
    west = dst_transform.c
    north = dst_transform.f
    east = west + dw * dst_transform.a
    south = north + dh * dst_transform.e  # e is negative (north-up)
    return dst, (west, south, east, north)


def write_kmz(
    out_path: str | Path,
    grid: np.ndarray,
    x_min: float,
    y_min: float,
    cell: float,
    epsg: Optional[int],
    nodata: float = NODATA,
) -> None:
    """Write a Google-Earth KMZ: a depth-colorized WGS84 ground overlay.

    Unlike the GeoTIFF/asc (raw float data), this bakes the depth gradient
    into a transparent PNG and wraps it as a ``<GroundOverlay>`` in lat/lon,
    which is what Google Earth actually renders. ``grid[0]`` must be the
    north-most row.
    """
    if not epsg:
        raise ValueError("KMZ export needs a known CRS (EPSG) to reproject to WGS84.")

    import io
    import zipfile

    from PIL import Image

    dst, (west, south, east, north) = _reproject_to_4326(
        grid, x_min, y_min, cell, epsg, nodata
    )
    rgba, vmin, vmax = colorize(dst, nodata)

    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    png_bytes = buf.getvalue()

    name = Path(out_path).stem
    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        "  <GroundOverlay>\n"
        f"    <name>{name}</name>\n"
        f"    <description>Depth (m): {vmin:.2f} (deep/dark) to {vmax:.2f} "
        "(shallow/bright), viridis colormap.</description>\n"
        "    <Icon><href>overlay.png</href></Icon>\n"
        "    <LatLonBox>\n"
        f"      <north>{north:.10f}</north>\n"
        f"      <south>{south:.10f}</south>\n"
        f"      <east>{east:.10f}</east>\n"
        f"      <west>{west:.10f}</west>\n"
        "    </LatLonBox>\n"
        "  </GroundOverlay>\n"
        "</kml>\n"
    )

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml)
        z.writestr("overlay.png", png_bytes)
