"""Module-level dataset state for the trimmer.

Only five compact arrays are held in memory plus a boolean keep-mask. The
full 15-column row table is never resident; export streams the original CSV
in record-batch chunks. See plan: bounded RAM regardless of file size.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
from pyarrow import csv as pacsv

from . import dem as dem_module

EASTING_COL = "easting (local m)"
NORTHING_COL = "northing (local m)"
DEPTH_COL = "altitude (m)"
POWER_COL = "power (dB)"
PING_COL = "ping number"

# Used only by the DEM/contour export, which needs globally-consistent
# coordinates (the `local m` columns use a per-run origin).
UTM_EASTING_COL = "easting (UTM m)"
UTM_NORTHING_COL = "northing (UTM m)"
PROJECTION_COL = "coordinate projection"

_F32 = pa.float32()
_I32 = pa.int32()
COLUMN_TYPES: dict[str, pa.DataType] = {
    EASTING_COL: _F32,
    NORTHING_COL: _F32,
    DEPTH_COL: _F32,
    POWER_COL: _F32,
    PING_COL: _I32,
}

HISTORY_MAXLEN = 10


class Dataset:
    """In-memory state for a single loaded CSV.

    Held as a module-level singleton (`STATE`). Dash callbacks read/write
    arrays in place rather than passing payloads around.
    """

    csv_path: Path
    east: np.ndarray
    north: np.ndarray
    depth: np.ndarray
    power: np.ndarray
    ping: np.ndarray
    keep_mask: np.ndarray
    history: deque
    corridor_idx: Optional[np.ndarray]
    # Corridor-local indices that the cross-section figure last plotted. Lets
    # the lasso callback map `pointIndex` -> corridor index -> global row.
    cross_section_sample_local: Optional[np.ndarray]
    # Live state for a streaming export running on a worker thread; the Dash
    # progress-poll callback reads this. Plain dict + lock; CPython dict
    # writes are atomic per key which is good enough for UI display.
    export_state: dict
    export_lock: threading.Lock
    # Same idea, for the slow finishing filters (kNN SOR and
    # power-weighted grid median).
    filter_state: dict
    filter_lock: threading.Lock
    n: int

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = Path(csv_path)
        t0 = time.perf_counter()
        table = pacsv.read_csv(
            str(self.csv_path),
            convert_options=pacsv.ConvertOptions(
                include_columns=list(COLUMN_TYPES),
                column_types=COLUMN_TYPES,
            ),
        )
        self.east = table.column(EASTING_COL).to_numpy(zero_copy_only=False)
        self.north = table.column(NORTHING_COL).to_numpy(zero_copy_only=False)
        self.depth = table.column(DEPTH_COL).to_numpy(zero_copy_only=False)
        self.power = table.column(POWER_COL).to_numpy(zero_copy_only=False)
        self.ping = table.column(PING_COL).to_numpy(zero_copy_only=False)
        del table
        self.n = int(self.depth.size)
        # Auto-drop positive depths at load time; the user can recover via reset.
        self.keep_mask = self.depth < 0
        self.history = deque(maxlen=HISTORY_MAXLEN)
        self.corridor_idx = None
        self.cross_section_sample_local = None
        self.export_lock = threading.Lock()
        self.export_state = self._idle_export_state()
        self.filter_lock = threading.Lock()
        self.filter_state = self._idle_filter_state()
        # Coordinate projection (for the DEM/contour export). Read lazily from
        # the first data row so load stays cheap; mapped to an EPSG code.
        self.coordinate_projection = self._read_first_projection()
        self.epsg = dem_module.parse_utm_epsg(self.coordinate_projection)
        self.load_seconds = time.perf_counter() - t0

    def _read_first_projection(self) -> Optional[str]:
        """Read the `coordinate projection` value from the first data row."""
        try:
            reader = pacsv.open_csv(
                str(self.csv_path),
                convert_options=pacsv.ConvertOptions(
                    include_columns=[PROJECTION_COL],
                ),
            )
            for batch in reader:
                if batch.num_rows:
                    val = batch.column(0)[0].as_py()
                    return str(val) if val is not None else None
            return None
        except Exception:
            return None

    @staticmethod
    def _idle_export_state() -> dict:
        return {
            "phase": "idle",
            "progress": 0.0,
            "rows_processed": 0,
            "rows_written": 0,
            "rows_total": 0,
            "elapsed": 0.0,
            "out_path": None,
            "error": None,
        }

    @staticmethod
    def _idle_filter_state() -> dict:
        return {
            "phase": "idle",
            "stage": "",
            "progress": 0.0,
            "elapsed": 0.0,
            "applied": [],
            "removed": 0,
            "kept_before": 0,
            "kept_after": 0,
            "error": None,
        }

    def is_exporting(self) -> bool:
        return self.export_state.get("phase") == "running"

    def is_filtering(self) -> bool:
        return self.filter_state.get("phase") == "running"

    def push_history(self) -> None:
        """Snapshot the current mask onto the undo stack (packed, ~1 bit/point)."""
        self.history.append(np.packbits(self.keep_mask))

    def undo(self) -> bool:
        if not self.history:
            return False
        packed = self.history.pop()
        self.keep_mask[:] = np.unpackbits(packed, count=self.n).astype(bool)
        return True

    def reset(self) -> None:
        """Revert to the initial post-load mask (positives dropped)."""
        self.push_history()
        self.keep_mask[:] = self.depth < 0

    def replace_mask(self, new_mask: np.ndarray) -> None:
        """Replace the keep-mask, snapshotting the previous state for undo."""
        if new_mask.shape != self.keep_mask.shape:
            raise ValueError(
                f"mask shape mismatch: got {new_mask.shape}, expected {self.keep_mask.shape}"
            )
        self.push_history()
        # In-place assignment so external references to `keep_mask` stay valid.
        self.keep_mask[:] = new_mask.astype(bool, copy=False)

    @property
    def kept_count(self) -> int:
        return int(self.keep_mask.sum())

    def kept_indices(self) -> np.ndarray:
        return np.flatnonzero(self.keep_mask)

    def start_export(self, out_path: Path, chunk_rows: int = 1_000_000) -> bool:
        """Kick off a streaming export on a daemon thread.

        Returns False if another export is already running; otherwise resets
        the export state to a fresh `running` snapshot and returns True. The
        UI polls `self.export_state` for progress.
        """
        with self.export_lock:
            if self.is_exporting():
                return False
            self.export_state = {
                "phase": "running",
                "progress": 0.0,
                "rows_processed": 0,
                "rows_written": 0,
                "rows_total": self.n,
                "elapsed": 0.0,
                "out_path": str(out_path),
                "error": None,
            }
        t = threading.Thread(
            target=self._run_export,
            args=(Path(out_path), int(chunk_rows)),
            daemon=True,
            name="omniscan-export",
        )
        t.start()
        return True

    def export(self, out_path: Path, chunk_rows: int = 1_000_000) -> dict:
        """Synchronous export. Used by tests/CLI; the Dash UI uses start_export."""
        out_path = Path(out_path)
        self.export_state = {
            "phase": "running",
            "progress": 0.0,
            "rows_processed": 0,
            "rows_written": 0,
            "rows_total": self.n,
            "elapsed": 0.0,
            "out_path": str(out_path),
            "error": None,
        }
        self._run_export(out_path, chunk_rows)
        s = self.export_state
        if s.get("phase") == "error":
            raise RuntimeError(s.get("error") or "export failed")
        return {
            "out_path": s.get("out_path"),
            "rows_written": s.get("rows_written", 0),
            "rows_total": s.get("rows_total", self.n),
            "elapsed_seconds": s.get("elapsed", 0.0),
        }

    def _run_export(self, out_path: Path, chunk_rows: int) -> None:
        """Inner streaming loop. Updates `self.export_state` as it goes.

        Peak RAM is one record batch (a few hundred MB at most), independent
        of the total file size.
        """
        t0 = time.perf_counter()
        try:
            # block_size is in bytes; ~256 B per CSV row is a conservative upper
            # bound for this schema so chunk_rows ends up close to the target.
            reader = pacsv.open_csv(
                str(self.csv_path),
                read_options=pacsv.ReadOptions(block_size=chunk_rows * 256),
            )

            cursor = 0
            kept = 0
            first_batch = True
            with open(out_path, "w", newline="") as f:
                for batch in reader:
                    n = batch.num_rows
                    sub = self.keep_mask[cursor : cursor + n]
                    cursor += n
                    if sub.any():
                        filtered = batch.filter(pa.array(sub))
                        df_chunk = filtered.to_pandas(types_mapper=None)
                        df_chunk.to_csv(f, index=False, header=first_batch)
                        kept += filtered.num_rows
                        first_batch = False

                    self.export_state.update(
                        {
                            "progress": cursor / max(self.n, 1),
                            "rows_processed": cursor,
                            "rows_written": kept,
                            "elapsed": time.perf_counter() - t0,
                        }
                    )

            if cursor != self.n:
                raise RuntimeError(
                    f"Row count mismatch during export: streamed {cursor} rows "
                    f"but dataset has {self.n}. The source CSV may have changed "
                    "since load."
                )

            self.export_state.update(
                {
                    "phase": "done",
                    "progress": 1.0,
                    "rows_processed": self.n,
                    "rows_written": kept,
                    "elapsed": time.perf_counter() - t0,
                }
            )
        except Exception as e:
            self.export_state.update(
                {
                    "phase": "error",
                    "error": str(e),
                    "elapsed": time.perf_counter() - t0,
                }
            )

    # --- DEM / contour export ---------------------------------------------

    def start_dem_export(
        self,
        out_path: Path,
        cell: float,
        agg: str = "mean",
        epsg: Optional[int] = None,
        fmt: str = "asc",
        chunk_rows: int = 1_000_000,
    ) -> bool:
        """Kick off a streaming DEM export (`fmt` in {"asc", "gtiff"}).

        Shares `export_state` with the CSV export (they are mutually
        exclusive). Returns False if an export is already running.
        """
        with self.export_lock:
            if self.is_exporting():
                return False
            self.export_state = {
                "phase": "running",
                "progress": 0.0,
                "rows_processed": 0,
                "rows_written": 0,
                "rows_total": self.n,
                "elapsed": 0.0,
                "out_path": str(out_path),
                "message": None,
                "error": None,
            }
        t = threading.Thread(
            target=self._run_dem_export,
            args=(Path(out_path), float(cell), str(agg), epsg, str(fmt), int(chunk_rows)),
            daemon=True,
            name="omniscan-dem-export",
        )
        t.start()
        return True

    # Refuse grids that would blow past a sane memory budget; the accumulator
    # holds up to 16 bytes/cell, so ~50M cells is ~0.8 GB.
    _MAX_DEM_CELLS = 50_000_000

    def _dem_columns(self):
        return {
            UTM_EASTING_COL: pa.float64(),
            UTM_NORTHING_COL: pa.float64(),
            DEPTH_COL: pa.float32(),
        }

    def _run_dem_export(
        self,
        out_path: Path,
        cell: float,
        agg: str,
        epsg: Optional[int],
        fmt: str,
        chunk_rows: int,
    ) -> None:
        """Two streaming passes: pass 1 finds the UTM extent, pass 2 bins.

        Peak RAM is one record batch plus the grid accumulator, independent
        of the source file size.
        """
        t0 = time.perf_counter()
        try:
            col_types = self._dem_columns()
            include = list(col_types)

            def _open():
                return pacsv.open_csv(
                    str(self.csv_path),
                    read_options=pacsv.ReadOptions(block_size=chunk_rows * 256),
                    convert_options=pacsv.ConvertOptions(
                        include_columns=include,
                        column_types=col_types,
                    ),
                )

            # --- Pass 1: extent over kept, finite points ---------------------
            x_min = y_min = np.inf
            x_max = y_max = -np.inf
            cursor = 0
            kept_seen = 0
            reader = _open()
            for batch in reader:
                n = batch.num_rows
                sub = self.keep_mask[cursor : cursor + n]
                cursor += n
                if sub.any():
                    x = batch.column(UTM_EASTING_COL).to_numpy(zero_copy_only=False)[sub]
                    y = batch.column(UTM_NORTHING_COL).to_numpy(zero_copy_only=False)[sub]
                    z = batch.column(DEPTH_COL).to_numpy(zero_copy_only=False)[sub]
                    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                    if finite.any():
                        xf, yf = x[finite], y[finite]
                        x_min = min(x_min, float(xf.min()))
                        x_max = max(x_max, float(xf.max()))
                        y_min = min(y_min, float(yf.min()))
                        y_max = max(y_max, float(yf.max()))
                        kept_seen += int(finite.sum())
                self.export_state.update(
                    {
                        "progress": 0.45 * (cursor / max(self.n, 1)),
                        "rows_processed": cursor,
                        "elapsed": time.perf_counter() - t0,
                    }
                )

            if cursor != self.n:
                raise RuntimeError(
                    f"Row count mismatch during DEM export: streamed {cursor} "
                    f"rows but dataset has {self.n}. The source CSV may have "
                    "changed since load."
                )
            if kept_seen == 0 or not np.isfinite(x_min):
                raise RuntimeError(
                    "No kept points with valid UTM coordinates to grid."
                )

            acc = dem_module.GridAccumulator(
                x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
                cell=cell, agg=agg,
            )
            if acc.ncells > self._MAX_DEM_CELLS:
                raise RuntimeError(
                    f"Grid would be {acc.ncols} x {acc.nrows} = {acc.ncells:,} "
                    f"cells, over the {self._MAX_DEM_CELLS:,}-cell limit. "
                    f"Increase the cell size (current: {cell:g} m)."
                )

            # --- Pass 2: accumulate ------------------------------------------
            cursor = 0
            reader = _open()
            for batch in reader:
                n = batch.num_rows
                sub = self.keep_mask[cursor : cursor + n]
                cursor += n
                if sub.any():
                    x = batch.column(UTM_EASTING_COL).to_numpy(zero_copy_only=False)[sub]
                    y = batch.column(UTM_NORTHING_COL).to_numpy(zero_copy_only=False)[sub]
                    z = batch.column(DEPTH_COL).to_numpy(zero_copy_only=False)[sub]
                    acc.add(x, y, z)
                self.export_state.update(
                    {
                        "progress": 0.45 + 0.5 * (cursor / max(self.n, 1)),
                        "rows_processed": cursor,
                        "elapsed": time.perf_counter() - t0,
                    }
                )

            grid = acc.result()
            self.export_state.update({"progress": 0.97, "elapsed": time.perf_counter() - t0})

            out_path = Path(out_path)
            if fmt == "gtiff":
                dem_module.write_geotiff(
                    out_path, grid, acc.xllcorner, acc.yllcorner, cell, epsg,
                )
            elif fmt == "kmz":
                dem_module.write_kmz(
                    out_path, grid, acc.xllcorner, acc.yllcorner, cell, epsg,
                )
            else:
                dem_module.write_asc(out_path, grid, acc.xllcorner, acc.yllcorner, cell)
                dem_module.write_prj(out_path.with_suffix(".prj"), epsg)

            populated = int(getattr(acc, "populated_cells", 0))
            epsg_txt = f"EPSG:{epsg}" if epsg else "no CRS"
            kind = {"gtiff": "GeoTIFF", "kmz": "KMZ"}.get(fmt, "ASCII grid")
            message = (
                f"wrote {kind} {acc.ncols}x{acc.nrows} ({agg}, {cell:g} m, {epsg_txt}), "
                f"{populated:,} filled cells -> {out_path.name} "
                f"in {self._fmt_secs(time.perf_counter() - t0)}"
            )
            self.export_state.update(
                {
                    "phase": "done",
                    "progress": 1.0,
                    "rows_processed": self.n,
                    "rows_written": populated,
                    "elapsed": time.perf_counter() - t0,
                    "message": message,
                }
            )
        except Exception as e:
            self.export_state.update(
                {
                    "phase": "error",
                    "error": str(e),
                    "elapsed": time.perf_counter() - t0,
                }
            )

    @staticmethod
    def _fmt_secs(seconds: float) -> str:
        s = max(0, int(round(seconds)))
        if s < 60:
            return f"{s}s"
        m, s = divmod(s, 60)
        return f"{m}m {s:02d}s"

    # --- Finishing filters (kNN SOR + power-weighted grid median) ---------

    def start_finishing(self, params: dict) -> bool:
        """Kick off a finishing-filter pass on a daemon thread.

        `params` is a dict with at least:
            run_pwg: bool
            run_knn: bool
            pwg_cell, pwg_k, pwg_top_pct: floats
            knn_k: int, knn_m: float, knn_chunk: int

        Returns False if a finishing run is already in flight.
        """
        with self.filter_lock:
            if self.is_filtering():
                return False
            applied: list[str] = []
            if params.get("run_pwg"):
                applied.append("pwg")
            if params.get("run_knn"):
                applied.append("knn")
            self.filter_state = {
                "phase": "running",
                "stage": "starting",
                "progress": 0.0,
                "elapsed": 0.0,
                "applied": applied,
                "removed": 0,
                "kept_before": self.kept_count,
                "kept_after": 0,
                "error": None,
            }
        t = threading.Thread(
            target=self._run_finishing,
            args=(dict(params),),
            daemon=True,
            name="omniscan-finishing",
        )
        t.start()
        return True

    def _run_finishing(self, params: dict) -> None:
        """Worker that runs the selected finishing filters in sequence.

        Operates on a *copy* of the current keep mask so the UI's `keep_mask`
        view (and any concurrent reads) stay consistent until the run
        finishes; the result is then committed via `replace_mask` (which
        pushes the previous state to the undo stack).
        """
        # Local import to keep startup snappy when no finishing is run.
        from . import filters as F

        t0 = time.perf_counter()
        try:
            subset = self.keep_mask.copy()
            applied_labels: list[str] = []

            def make_progress_cb(stage_prefix: str, overall_lo: float, overall_hi: float):
                def cb(stage: str, frac: float) -> None:
                    frac = float(max(0.0, min(1.0, frac)))
                    overall = overall_lo + (overall_hi - overall_lo) * frac
                    self.filter_state.update(
                        {
                            "stage": f"{stage_prefix}: {stage}",
                            "progress": overall,
                            "elapsed": time.perf_counter() - t0,
                        }
                    )
                return cb

            run_pwg = bool(params.get("run_pwg"))
            run_knn = bool(params.get("run_knn"))

            # Allocate progress budget across whichever filters were chosen.
            stages: list[tuple[str, tuple[float, float]]] = []
            if run_pwg and run_knn:
                # kNN dominates by a large margin; give it 85% of the bar.
                stages.append(("pwg", (0.0, 0.15)))
                stages.append(("knn", (0.15, 1.0)))
            elif run_pwg:
                stages.append(("pwg", (0.0, 1.0)))
            elif run_knn:
                stages.append(("knn", (0.0, 1.0)))
            stage_ranges = dict(stages)

            if run_pwg:
                lo, hi = stage_ranges["pwg"]
                self.filter_state.update({"stage": "power-weighted grid", "progress": lo})
                mask_pwg = F.power_weighted_grid_filter(
                    self.east, self.north, self.depth, self.power,
                    cell=float(params.get("pwg_cell", 1.0)),
                    k=float(params.get("pwg_k", 4.0)),
                    top_pct=float(params.get("pwg_top_pct", 50.0)),
                    subset_mask=subset,
                    progress=make_progress_cb("power-weighted grid", lo, hi),
                )
                subset &= mask_pwg
                applied_labels.append(
                    f"PW grid (cell={params.get('pwg_cell', 1.0):g}, "
                    f"k={params.get('pwg_k', 4.0):g}, "
                    f"top={params.get('pwg_top_pct', 50.0):g}%)"
                )

            if run_knn:
                lo, hi = stage_ranges["knn"]
                self.filter_state.update({"stage": "kNN SOR", "progress": lo})
                mask_knn = F.knn_sor(
                    self.east, self.north, self.depth,
                    k=int(params.get("knn_k", 8)),
                    m=float(params.get("knn_m", 3.0)),
                    chunk_size=int(params.get("knn_chunk", 500_000)),
                    subset_mask=subset,
                    progress=make_progress_cb("kNN SOR", lo, hi),
                )
                subset &= mask_knn
                applied_labels.append(
                    f"kNN SOR (k={int(params.get('knn_k', 8))}, "
                    f"m={float(params.get('knn_m', 3.0)):g})"
                )

            kept_before = int(self.filter_state.get("kept_before", 0))
            kept_after = int(subset.sum())
            removed = max(0, kept_before - kept_after)

            self.replace_mask(subset)

            self.filter_state.update(
                {
                    "phase": "done",
                    "stage": "done",
                    "progress": 1.0,
                    "elapsed": time.perf_counter() - t0,
                    "applied": applied_labels,
                    "removed": removed,
                    "kept_after": kept_after,
                }
            )
        except Exception as e:
            self.filter_state.update(
                {
                    "phase": "error",
                    "stage": "error",
                    "error": str(e),
                    "elapsed": time.perf_counter() - t0,
                }
            )


STATE: Optional[Dataset] = None


def load(csv_path: str | Path) -> Dataset:
    """Load a CSV into the module-level singleton and return it."""
    global STATE
    STATE = Dataset(Path(csv_path))
    return STATE


def get() -> Dataset:
    """Accessor for the loaded dataset; raises if `load()` was not called."""
    if STATE is None:
        raise RuntimeError("Dataset not loaded. Call trimmer.data.load(path) first.")
    return STATE
