import numpy as np
import pandas as pd
import torch
import logging
from omegaconf import DictConfig, open_dict
from sklearn.metrics import matthews_corrcoef

from src.helpers.pytorch.lstm import LSTM_custom_collate, LSTMModule
from src.helpers.pytorch.utilities import EmbeddingsDataset
from src.model.abstract.abstract_torch_predictor_model import TorchPredictorModel
from src.model.common import validate_predictor, run_mode
from pathlib import Path

logger = logging.getLogger(__name__)


class LSTM(TorchPredictorModel):
    def __init__(self, cfg: DictConfig):
        """
        Initialize the LSTM predictor.
        Args:
            cfg (DictConfig): The configuration for the model.
        Attributes:
            model: The LSTM model.
            trained (bool): Indicates if the model has been trained.
        Methods:
            forward(embeddings:np.array) -> np.array: Forward pass of the LSTM model.
            save_model(dir_path:Path) -> None: Save the LSTM model.
            update_predictor_name() -> None: Update the LSTM name based on the class name and hyperparameters.
            get_hparams_string() -> str: Get a string representation of the hyperparameters.
            get_param_grid() -> dict: Get the parameter grid for hyperparameter tuning.
            update_config_hparams() -> None: Update the configuration hyperparameters based on the model's current hyperparameters.
        """
        logger.info(f"Load class (LSTM): {self.__class__.__name__}")
        super().__init__(cfg)

        self.dataset = EmbeddingsDataset
        self.collate_fn = LSTM_custom_collate

        validate_predictor(cfg)
        if self.cfg.general.run_mode == run_mode.train:
            self.optimal_cutoff = None
            if self.cfg.predictor.model_name is None:
                if cfg.predictor.model_type == "regression":
                    raise NotImplementedError("Regression not implemented for LightAttention")
                elif cfg.predictor.model_type == "classification_binary":
                    # Lazy load (once we know the embedding dimension)
                    self.model = None
                else:
                    raise ValueError(
                        'Only model_type "classification_binary" is supported,'
                        'not "{self.cfg.predictor.model_type}"'
                    )
            else:
                raise ValueError(
                    f"Model name must be null (not {self.cfg.predictor.model_name}) while in training mode"
                )
        elif self.cfg.general.run_mode == run_mode.test:
            if not self.cfg.predictor.hparams.optimal_cutoff:
                raise ValueError("Optimal cutoff must be defined in test mode")
            self.optimal_cutoff = self.cfg.predictor.hparams.optimal_cutoff
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
        self.model = LSTMModule(
            embeddings_dim=embedding_dim,
            output_dim=1,
            dropout=self.cfg.predictor.hparams.dropout,
            optimizer_type=self.cfg.predictor.hparams.optimizer_type,
            learning_rate=self.cfg.predictor.hparams.learning_rate,
        )

    def forward(self, embeddings):
        """
        Performs forward pass on the LSTM model.
        Args:
            embeddings (list[torch.tensor]): The input embeddings.
        Returns:
            np.array: The predicted values.
            np.array: Predicted probabilities (None for regression)
        """
        embeddings, _ = self.collate_fn([[x] for x in embeddings])
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logger.info(f"Moving embeddings to device: {device}")
        embeddings = embeddings.to(device)
        if self.model_type == "classification_binary":
            with torch.inference_mode():
                predicted_probabilities = self.model.forward(embeddings).cpu().numpy().squeeze()
            if self.optimal_cutoff:
                predictions = predicted_probabilities >= self.optimal_cutoff
            else:
                predictions = None
        else:
            raise NotImplementedError

        return predictions, predicted_probabilities

    def post_train_model(self, x_val, y_val):
        """
        After training, find the optimal cutoff for classification.
        """
        _, predicted_probabilities = self.forward(x_val)

        # scan over cutoffs with a step size of 0.01
        cutoffs = np.linspace(0, 1, 101)
        thresholded_probabilities = predicted_probabilities > np.expand_dims(cutoffs, axis=-1)

        mcc_df = pd.DataFrame(
            {"cutoffs": cutoffs, "predictions": [x for x in thresholded_probabilities]}
        )
        mcc_df["mcc"] = mcc_df.apply(lambda x: matthews_corrcoef(y_val, x.predictions), axis=1)
        self.optimal_cutoff = float(mcc_df.iloc[mcc_df.mcc.idxmax()].cutoffs)

        with open_dict(self.cfg):
            self.cfg.predictor.hparams.optimal_cutoff = self.optimal_cutoff

        logger.info(f"Optimal cutoff is: {self.optimal_cutoff} with MCC: {mcc_df.mcc.max()}")

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
        self.cfg.predictor.model_name = "lstm_" + self.get_hparams_string()

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
