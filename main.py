import argparse

from ct_mae import MAEConfig, train


def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CT-MAE model")
    parser.add_argument(
        "--train", action="store_true", help="Whether to train the model"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = build_parser()
    if args.train:
        cfg = MAEConfig()
        train(cfg)
