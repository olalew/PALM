from strenum import StrEnum

component_type = StrEnum(
    "component_type",
    [
        "embedder", "dimred", "predictor"
    ])

run_mode = StrEnum(
    "run_mode",
    [
        "train",
        "test",
        "embed"
    ])


def validate_predictor(cfg):
    if cfg.predictor.model_type == "classifier":
        if cfg.dataset.data_type == "binary":
            pass
        elif cfg.dataset.data_type == "real-valued" and cfg.dataset.cutoff_value is None:
            raise ValueError(
                "cutoff_value must be defined if real-valued data is provided when training a classifier"
            )
    if cfg.predictor.model_type == "regression":
        pass
