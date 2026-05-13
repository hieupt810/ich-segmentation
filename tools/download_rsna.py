import logging

import kagglehub

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    path = kagglehub.competition_download(
        "rsna-intracranial-hemorrhage-detection", output_dir="data"
    )
    logger.info(f"Dataset downloaded to: {path}")


if __name__ == "__main__":
    main()
