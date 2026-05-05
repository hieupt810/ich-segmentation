import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from pydicom.errors import InvalidDicomError

# Configure logging for monitoring
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DicomSliceExtractor:
    def __init__(self, input_dir: str | Path, output_dir: str | Path) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        # Ensure the output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_pixel_array(self, pixel_array: np.ndarray) -> np.ndarray:
        array_min = pixel_array.min()
        array_max = pixel_array.max()

        # Prevent division by zero
        if array_max - array_min == 0:
            return np.zeros_like(pixel_array, dtype=np.uint8)

        # Min-max normalization to [0, 255]
        normalized = (pixel_array - array_min) / (array_max - array_min)
        return (normalized * 255).astype(np.uint8)

    def process_single_dicom(self, file_path: Path) -> Path | None:
        try:
            # Read the DICOM file
            dataset = pydicom.dcmread(file_path)

            # Ensure the dataset contains pixel data
            if not hasattr(dataset, "pixel_array"):
                logger.warning(f"No pixel data found in {file_path.name}.")
                return None

            # Extract and normalize the pixel array
            raw_array = dataset.pixel_array
            normalized_array = self._normalize_pixel_array(raw_array)

            # Convert to a Pillow Image and save
            image = Image.fromarray(normalized_array)
            output_filename = self.output_dir / f"{file_path.stem}.png"
            image.save(output_filename)

            logger.info(f"Successfully extracted: {output_filename.name}")
            return output_filename
        except InvalidDicomError:
            logger.error(f"Invalid DICOM file encountered: {file_path.name}")
            return None
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}. Reason: {e}")
            return None

    def extract_all_slices(self) -> list[Path]:
        saved_files: list[Path] = []
        for item in self.input_dir.iterdir():
            if item.is_file() and item.suffix.lower() in [".dcm", ".dicom"]:
                result = self.process_single_dicom(item)
                if result:
                    saved_files.append(result)

        logger.info(
            f"Extraction complete. Successfully processed {len(saved_files)} files."
        )
        return saved_files


def build_parser() -> Namespace:
    parser = ArgumentParser(
        description="Extract slices from DICOM files and save as PNG images."
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        type=Path,
        help="Directory containing DICOM files.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        type=Path,
        help="Directory to save extracted PNG images.",
    )
    return parser.parse_args()


def main() -> None:
    args = build_parser()
    extractor = DicomSliceExtractor(args.input_dir, args.output_dir)
    extractor.extract_all_slices()


if __name__ == "__main__":
    main()
