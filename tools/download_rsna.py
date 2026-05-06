import logging

import kagglehub

# Configure logging for monitoring
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    path = kagglehub.competition_download(
        "rsna-intracranial-hemorrhage-detection", output_dir="data"
    )
    logger.info(f"Dataset downloaded to: {path}")


if __name__ == "__main__":
    main()
