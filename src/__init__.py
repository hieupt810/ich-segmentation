def run_mae_pipeline():
    from . import mae

    cfg = mae.MAEConfig()
    mae.train_mae(cfg)


__all__: list[str] = ["run_mae_pipeline"]
