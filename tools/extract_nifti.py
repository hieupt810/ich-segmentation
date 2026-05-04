import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np

BRAIN_WINDOW: tuple[int, int] = (40, 120)


def load_nifti(path: Path, canonical: bool = True) -> np.ndarray:
    image = nib.load(path)
    if canonical:
        image = nib.as_closest_canonical(image)
    return np.asarray(image.dataobj, dtype=np.float32)


def apply_window(image: np.ndarray, window: tuple[int, int]) -> np.ndarray:
    center, width = window
    lowest: float = center - width / 2.0
    highest: float = center + width / 2.0

    clipped: np.ndarray = np.clip(image, lowest, highest)
    normalized: np.ndarray = (clipped - lowest) / (highest - lowest) * 255.0
    return normalized.astype(np.uint8)


def apply_pipeline_and_save_image(
    image: np.ndarray, output_path: Path, image_size: int = 256
) -> None:
    image = cv2.resize(image, (image_size, image_size))
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    cv2.imwrite(str(output_path), image)


def process_nifti(input_path: Path, output_path: Path) -> None:
    image = load_nifti(input_path)
    image = apply_window(image, BRAIN_WINDOW)

    name = input_path.name.replace(".nii.gz", "").replace(".nii", "")
    slice_dir = output_path / name
    slice_dir.mkdir(parents=True, exist_ok=True)

    for i in range(image.shape[2]):
        apply_pipeline_and_save_image(image[:, :, i], slice_dir / f"{i:04d}.png")


def _worker(args: tuple[Path, Path]) -> str:
    input_path, output_path = args
    process_nifti(input_path, output_path)
    return input_path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PNG slices from NIfTI files")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Directory containing the NIfTI files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Directory to save the extracted PNG files",
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=1, help="Number of worker processes"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nifti_files = sorted(args.input.rglob("*.nii*"))

    if not nifti_files:
        print("No NIfTI files found.")
        return

    tasks = [(path, args.output) for path in nifti_files]

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for name in executor.map(_worker, tasks):
                print(f"Processed {name}")
    else:
        for path, output_path in tasks:
            print(f"Processing {path.name}...")
            process_nifti(path, output_path)

    print("Extraction complete!")


if __name__ == "__main__":
    main()
