import argparse


def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["train-mae", "plot-reconstruction", "train-segmentation"],
        help="Choose the training mode.",
    )
    parser.add_argument(
        "--image",
        nargs="+",
        help="One or more image paths (required for plot-reconstruction).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip plt.show(); only save the figure.",
    )
    return parser.parse_args()


def main():
    args = build_parser()
    if args.mode == "train-mae":
        from src import mae

        cfg = mae.MAEConfig()
        mae.train_mae(cfg)
    elif args.mode == "plot-reconstruction":
        if not args.image:
            raise SystemExit("plot-reconstruction requires --image PATH [PATH ...]")

        from src.eval import plot_reconstruction

        plot_reconstruction(args.image, show=not args.no_show)
    elif args.mode == "train-segmentation":
        pass


if __name__ == "__main__":
    main()
