import argparse


def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["train-mae", "train-segmentation"],
        help="Choose the training mode.",
    )
    return parser.parse_args()


def main():
    args = build_parser()
    if args.mode == "train-mae":
        from src import run_mae_pipeline

        run_mae_pipeline()
    elif args.mode == "train-segmentation":
        pass


if __name__ == "__main__":
    main()
