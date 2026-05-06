import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import FileDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

WINDOW_WIDTH = 200
WINDOW_LEVEL = 40


def apply_window(
    hu_array: np.ndarray,
    window_width: float,
    window_level: float,
) -> np.ndarray:
    """Apply windowing to a HU array and return an 8-bit grayscale image."""

    lower = window_level - window_width / 2
    upper = window_level + window_width / 2
    windowed = np.clip(hu_array, lower, upper)
    windowed = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
    return windowed


def hu_to_array(ds: FileDataset) -> np.ndarray:
    """Convert DICOM pixel data to a HU array using Rescale Slope and Intercept."""

    raw = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    return raw * slope + intercept


def extract_slices_from_file(dicom_path: str, output_dir: str) -> int:
    """Extract slices from a single DICOM file and save as PNG images. Returns the number of slices extracted."""

    ds = pydicom.dcmread(dicom_path)

    if not hasattr(ds, "pixel_array"):
        logger.warning("No pixel data found in %s. Skipping.", dicom_path)
        return 0

    hu_array = hu_to_array(ds)
    is_volume = hu_array.ndim == 3
    slices = hu_array if is_volume else hu_array[np.newaxis, ...]
    num_slices = slices.shape[0]

    base_name = os.path.splitext(os.path.basename(dicom_path))[0]

    for idx in range(num_slices):
        windowed = apply_window(slices[idx], WINDOW_WIDTH, WINDOW_LEVEL)
        img = Image.fromarray(windowed, mode="L")
        slice_tag = f"slice{idx:04d}" if is_volume else "slice0000"
        filename = f"{base_name}_{slice_tag}.png"
        out_path = os.path.join(output_dir, filename)
        img.save(out_path)
        logger.debug("Saved %s", out_path)

    return num_slices


def process_dicom_file(task: tuple[str, str]) -> tuple[str, int, str | None]:
    """Worker function to process a single DICOM file. Returns (dicom_path, slice_count, error_message)."""

    dicom_path, output_dir = task
    try:
        count = extract_slices_from_file(dicom_path, output_dir)
        return dicom_path, count, None
    except Exception as exc:
        return dicom_path, 0, str(exc)


def collect_dicom_files(input_dir: str) -> list[str]:
    """Recursively collect all valid DICOM files from the input directory."""

    dicom_files = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                pydicom.dcmread(fpath, stop_before_pixels=True)
                dicom_files.append(fpath)
            except Exception:
                pass
    return sorted(dicom_files)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Extract DICOM slices with brain/ICH windowing."
    )
    parser.add_argument(
        "input_dir",
        help="Path to a directory containing DICOM files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="dicom_slices",
        help="Output directory for extracted slice images. Default: dicom_slices/",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers. Default: 1",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        parser.error(f"input_dir is not a valid directory: {args.input_dir}")

    if args.workers < 1:
        parser.error(f"workers must be >= 1: {args.workers}")

    return args


def main() -> None:
    args = parse_args()

    dicom_files = collect_dicom_files(args.input_dir)
    if not dicom_files:
        logger.error("No valid DICOM files found in: %s", args.input_dir)
        sys.exit(1)

    logger.info(
        "Found %d DICOM file(s). Window — width: %d HU, level: %d HU. Workers: %d.",
        len(dicom_files),
        WINDOW_WIDTH,
        WINDOW_LEVEL,
        args.workers,
    )
    os.makedirs(args.output, exist_ok=True)

    total_slices = 0
    if args.workers == 1:
        for fpath in dicom_files:
            logger.info("Processing: %s", fpath)
            try:
                count = extract_slices_from_file(fpath, args.output)
                total_slices += count
                logger.info("  -> Extracted %d slice(s).", count)
            except Exception as exc:
                logger.error("Failed to process %s: %s", fpath, exc)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_dicom_file, (fpath, args.output)): fpath
                for fpath in dicom_files
            }
            for future in as_completed(futures):
                fpath, count, err = future.result()
                if err is None:
                    total_slices += count
                    logger.info("Processed: %s -> Extracted %d slice(s).", fpath, count)
                else:
                    logger.error("Failed to process %s: %s", fpath, err)

    logger.info("Done. Total slices extracted: %d.", total_slices)


if __name__ == "__main__":
    main()
