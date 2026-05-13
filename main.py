import argparse

from src import mae


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
        cfg = mae.MAEConfig()
        mae.train_mae(cfg)
    elif args.mode == "train-segmentation":
        pass


if __name__ == "__main__":
    main()
