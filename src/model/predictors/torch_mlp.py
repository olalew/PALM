import numpy as np
import torch
import logging
from omegaconf import DictConfig

from src.helpers.pytorch.mlp import TorchMLPModule
from src.helpers.pytorch.utilities import EmbeddingsDataset
from src.model.abstract.abstract_torch_predictor_model import TorchPredictorModel
from src.model.common import validate_predictor, run_mode
from pathlib import Path

logger = logging.getLogger(__name__)


class TorchMLP(TorchPredictorModel):
    def __init__(self, cfg: DictConfig):
        logger.info(f"Load class (TorchMLP): {self.__class__.__name__}")
        super().__init__(cfg)

        self.dataset = EmbeddingsDataset
        # use torch.utils.data.DataLoader default
        self.collate_fn = None

        validate_predictor(cfg)
        if self.cfg.general.run_mode == run_mode.train:
            if self.cfg.predictor.model_name is None:
                if cfg.predictor.model_type == "regression":
                    # Lazy load (until we know the embedding dimension)
                    self.model = None
                elif cfg.predictor.model_type == "classification_binary":
                    raise NotImplementedError("Classification not implemented for TorchMLP")
                else:
                    raise ValueError(
                        f'Only model_type "regression" is supported,'  # noqa: F541
                        'not "{self.cfg.predictor.model_type}"'
                    )
            else:
                raise ValueError(
                    f"Model name must be null (not {self.cfg.predictor.model_name}) while in training mode"
                )
        elif self.cfg.general.run_mode == run_mode.test:
            if self.cfg.general.composite_model_path:
                model_path = Path(
                    self.cfg.general.composite_model_path,
                    self.cfg.predictor.model_name + ".pt",
                )
            else:
                model_path = Path(self.cfg.predictor.model_name + ".pt")
            logger.info(f"Loading model from {model_path}")
            self.model = torch.load(model_path)
            self.trained = True
            self.model.eval()
        else:
            raise ValueError(f"run_mode: {self.cfg.general.run_mode} not defined")

    def init_torch_module(self, embedding_dim: int):
        self.model = TorchMLPModule(
            embedding_dim,
            self.cfg.predictor.hparams.hidden_size,
            self.cfg.predictor.hparams.dropout_rate,
            self.cfg.predictor.hparams.learning_rate,
        )

    def forward(self, embeddings):
        """
        Performs forward pass on the TorchMLP model.
        Args:
            embeddings (list[torch.tensor | np.array]): The input embeddings.
        Returns:
            np.array: The predicted values.
            np.array: Predicted probabilites (None for regression)
        """
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).to(torch.float)

        logger.info(f"Moving embeddings to device: {device}")
        embeddings = embeddings.to(device)
        with torch.inference_mode():
            predictions = self.model.forward(embeddings).cpu().numpy()
        return predictions, None

    def save_model(self, dir_path):
        """
        Saves the model.
        Args:
            dir_path (Path): The directory path to save the model.
        Returns:
            model_path (str): The path to where the model was saved
        """
        model_path = Path(dir_path, self.cfg.predictor.model_name + ".pt")
        torch.save(self.model, model_path)
        return model_path

    def update_predictor_name(self) -> None:
        """Updates the predictor model name according to the hparams selected during training"""
        self.cfg.predictor.model_name = "torchmlp_" + self.get_hparams_string()

    def get_hparams_string(self) -> str:
        """
        Get a string representation of the hyperparameters.
        Returns:
            str: The string representation of the hyperparameters.
        """
        return "_".join([f"{k}_{v}" for k, v in dict(self.cfg.predictor.hparams).items()])

    def get_param_grid(self) -> dict:
        """
        Get the parameter grid for hyperparameter tuning.
        Returns:
            dict: The parameter grid.
        """
        return dict()

    def update_config_hparams(self) -> None:
        """
        Updates the configuration hyperparameters based on the model's current parameters.
        """
        pass
