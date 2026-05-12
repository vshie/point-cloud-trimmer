"""CLI entry point for the Omniscan3D trimmer Dash app.

Usage:
    python -m scripts.run /path/to/file.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trimmer.app import build_app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="omniscan3d-trimmer",
        description="Interactive Dash trimmer for Omniscan3D pointcloud CSVs.",
    )
    p.add_argument(
        "csv_path",
        type=Path,
        help="Path to the Omniscan3D CSV file to load.",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to serve the Dash app on (default: 8050).",
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Run Dash in debug mode (hot reload, verbose errors).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.csv_path.exists():
        print(f"error: CSV not found: {args.csv_path}", file=sys.stderr)
        return 2
    app = build_app(args.csv_path)
    print(f"Serving Omniscan3D trimmer on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
