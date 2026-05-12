from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pedflow.calibration import (
    DEFAULT_CHARUCO_DICTIONARY,
    DEFAULT_CHARUCO_DPI,
    DEFAULT_CHARUCO_MARGIN_MM,
    DEFAULT_CHARUCO_MARKER_LENGTH_MM,
    DEFAULT_CHARUCO_SQUARE_LENGTH_MM,
    DEFAULT_CHARUCO_SQUARES_X,
    DEFAULT_CHARUCO_SQUARES_Y,
    generate_charuco_board,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a printable ChArUco calibration board with JSON metadata.",
    )
    parser.add_argument("--output-dir", default="outputs/charuco_board", help="Output folder")
    parser.add_argument("--basename", default="charuco_board", help="Output file basename")
    parser.add_argument(
        "--squares-x",
        type=int,
        default=DEFAULT_CHARUCO_SQUARES_X,
        help="Number of board squares horizontally",
    )
    parser.add_argument(
        "--squares-y",
        type=int,
        default=DEFAULT_CHARUCO_SQUARES_Y,
        help="Number of board squares vertically",
    )
    parser.add_argument(
        "--square-length-mm",
        type=float,
        default=DEFAULT_CHARUCO_SQUARE_LENGTH_MM,
        help="Square side length",
    )
    parser.add_argument(
        "--marker-length-mm",
        type=float,
        default=DEFAULT_CHARUCO_MARKER_LENGTH_MM,
        help="ArUco marker side length",
    )
    parser.add_argument(
        "--dictionary", default=DEFAULT_CHARUCO_DICTIONARY, help="OpenCV ArUco dictionary"
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_CHARUCO_DPI, help="Printable output DPI")
    parser.add_argument(
        "--margin-mm",
        type=float,
        default=DEFAULT_CHARUCO_MARGIN_MM,
        help="White margin around board",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = generate_charuco_board(
        output_dir=args.output_dir,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_mm=args.square_length_mm,
        marker_length_mm=args.marker_length_mm,
        dictionary_name=args.dictionary,
        dpi=args.dpi,
        margin_mm=args.margin_mm,
        basename=args.basename,
    )

    print("Generated ChArUco board:")
    print(f"  PNG:      {metadata['png_path']}")
    print(f"  PDF:      {metadata['pdf_path']}")
    print(f"  Metadata: {metadata['metadata_path']}")
    print(f"  Size:     {metadata['board_width_m']:.3f} m x {metadata['board_height_m']:.3f} m")
    print("Print the PDF at 100% scale on A3 paper. Do not fit-to-page.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
