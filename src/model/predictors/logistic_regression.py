import logging
from typing import Any

import numpy as np
from omegaconf import DictConfig

from src.model.abstract.abstract_skl_predictor_model import SKLPredictorModel
from src.model.common import validate_predictor, run_mode
from sklearn.linear_model import (
    LogisticRegression as SKLogisticRegression,
)
from pathlib import Path
from skops.io import load as sk_load
from skops.io import dump as sk_dump

logger = logging.getLogger(__name__)

class LogisticRegression(SKLPredictorModel):
    """
    A logistic regression predictor model.
    Args:
        cfg (DictConfig): The configuration for the model.
    Attributes:
        model: The logistic regression model.
        trained (bool): Indicates if the model has been trained.
    Methods:
        forward(embeddings:np.array) -> np.array: Forward pass of the logistic regression model.
        save_model(dir_path:Path) -> None: Save the logistic regression model.
        update_predictor_name() -> None: Update the predictor name based on the class name and hyperparameters.
        get_hparams_string() -> str: Get a string representation of the hyperparameters.
        get_param_grid() -> dict: Get the parameter grid for hyperparameter tuning.
        update_config_hparams() -> None: Update the configuration hyperparameters based on the model's current hyperparameters.
    Raises:
        ValueError: If the run mode is not defined.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        validate_predictor(cfg)

        if self.cfg.general.run_mode == run_mode.train:
            if self.cfg.predictor.model_name is None:
                self.model = SKLogisticRegression(random_state=self.cfg.general.random_state)
            else:
                raise ValueError(
                    f"Model name must be null (not {self.cfg.predictor.model_name}) while in training mode"
                )
        elif self.cfg.general.run_mode == run_mode.test:
            if self.cfg.general.composite_model_path:
                model_path = Path(
                    self.cfg.general.composite_model_path,
                    self.cfg.predictor.model_name + ".skops",
                )
            else:
                model_path = Path(self.cfg.predictor.model_name + ".skops")
            self.model = sk_load(model_path, trusted=True)
            self.trained = True
        else:
            raise ValueError(f"run_mode: {self.cfg.general.run_mode} not defined")

    def forward(self, embeddings: np.ndarray) -> tuple[Any, Any]:
        """
        Forward pass of the logistic regression model.
        Args:
            embeddings (np.array): The input embeddings.
        Returns:
            np.array: The predicted labels.
        """
        return (
            self.model.predict(embeddings),
            self.model.predict_proba(embeddings)[:, -1],
        )

    def save_model(self, dir_path: Path) -> Path:
        """
        Save the logistic regression model.
        Args:
            dir_path (Path): The directory path to save the model.
        Returns:
            model_path (str): The path to where the model was saved
        """
        model_path = Path(dir_path, self.cfg.predictor.model_name + ".skops")
        sk_dump(self.model, model_path)
        return model_path

    def update_predictor_name(self) -> None:
        """Update the predictor name based on the class name and hyperparameters."""
        self.cfg.predictor.model_name = f"{self.cfg.dataset.data_name}{self.cfg.embedder.model_name}{self.cfg.predictor.class_name}{self.get_hparams_string()}"

    def get_hparams_string(self) -> str:
        """
        Get a string representation of the hyperparameters.
        Returns:
            str: The hyperparameters string.
        """
        c_val = self.cfg.predictor.hparams.C if self.cfg.predictor.hparams.C is not None else np.nan
        penalty = (
            self.cfg.predictor.hparams.penalty
            if self.cfg.predictor.hparams.penalty is not None
            else np.nan
        )
        solver = (
            self.cfg.predictor.hparams.solver
            if self.cfg.predictor.hparams.solver is not None
            else np.nan
        )
        class_weight = (
            self.cfg.predictor.hparams.class_weight
            if self.cfg.predictor.hparams.class_weight is not None
            else ""
        )
        hparams_string = f"-c:{c_val}-penalty:{penalty}-solver:{solver}-classweight:{class_weight}"
        return hparams_string

    def get_param_grid(self) -> dict:
        """
        Get the parameter grid for hyperparameter tuning.
        Returns:
            dict: The parameter grid.
        """
        param_grid = {
            "C": list(self.cfg.predictor.hparam_tuning.C_values),
            "penalty": list(self.cfg.predictor.hparam_tuning.penalties),
            "solver": [self.cfg.predictor.hparam_tuning.solver],
            "random_state": [self.cfg.general.random_state],
            "max_iter": [self.cfg.predictor.hparam_tuning.max_iter],
            "class_weight": [self.cfg.predictor.hparam_tuning.class_weight],
        }
        return param_grid

    def update_config_hparams(self) -> None:
        """Update the configuration hyperparameters based on the model's current hyperparameters."""
        updated_params = self.model.get_params()
        for param_name in self.cfg.predictor.hparams:
            try:
                self.cfg.predictor.hparams[param_name] = updated_params[param_name]
            except KeyError:
                logger.info(
                    {
                        f"Invalid param_name {param_name} not present in the current list of hyperparameters"
                    }
                )
