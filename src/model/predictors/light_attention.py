import torch
import logging

from omegaconf import DictConfig, open_dict
from pathlib import Path

from src.helpers.pytorch.light_attention import LA_custom_collate, LA_custom_collate_residue_level, LightAttentionModule
from src.helpers.pytorch.utilities import EmbeddingsDataset, EmbeddingsDatasetResidueLevel
from src.helpers.utilities import find_optimal_cutoff
from src.model.abstract.abstract_torch_predictor_model import TorchPredictorModel
from src.model.common import run_mode, validate_predictor

logger = logging.getLogger(__name__)


class LightAttention(TorchPredictorModel):
    """
        Attributes:
            model: The KNearestNeighbors model.
            trained (bool): Indicates if the model has been trained.
        Methods:
            forward(embeddings:np.array) -> np.array: Forward pass of the LightAttention model.
            save_model(dir_path:Path) -> None: Save the LightAttention model.
            update_predictor_name() -> None: Update the LightAttention name based on the class name and hyperparameters.
            get_hparams_string() -> str: Get a string representation of the hyperparameters.
            get_param_grid() -> dict: Get the parameter grid for hyperparameter tuning.
            update_config_hparams() -> None: Update the configuration hyperparameters based on the model's current hyperparameters.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the LightAttention predictor.
        Args:
            cfg (DictConfig): The configuration for the model.
        """
        logger.info(f"Load class (LightAttention): {self.__class__.__name__}")
        super().__init__(cfg)

        self.residue_prediction_mode = cfg.predictor.residue_prediction_mode
        self.dataset = (
            EmbeddingsDataset
            if not self.cfg.predictor.residue_prediction_mode
            else EmbeddingsDatasetResidueLevel
        )
        self.collate_fn = (
            LA_custom_collate
            if not self.cfg.predictor.residue_prediction_mode
            else LA_custom_collate_residue_level
        )

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
                    "model_state_dict.pt",
                )
            else:
                model_path = Path("model_state_dict.pt")
            logger.info(f"Loading model from {model_path}")
            emb_dims = {
                "esm2_t6_8M_UR50D": 320,
                "onehot": 20
            }
            self.init_torch_module(emb_dims[self.cfg.embedder.model_name])
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.trained = True
            self.model.eval()
        else:
            raise ValueError(f"run_mode: {self.cfg.general.run_mode} not defined")

    def init_torch_module(self, embedding_dim: int):
        self.model = LightAttentionModule(
            embeddings_dim=embedding_dim,
            output_dim=self.cfg.predictor.hparams.output_dim,
            kernel_size=self.cfg.predictor.hparams.kernel_size,
            dropout=self.cfg.predictor.hparams.dropout,
            conv_dropout=self.cfg.predictor.hparams.conv_dropout,
            optimizer_type=self.cfg.predictor.hparams.optimizer_type,
            learning_rate=self.cfg.predictor.hparams.learning_rate,
            post_attention=self.cfg.predictor.hparams.post_attention,
            conv1d_output_dim=self.cfg.predictor.hparams.conv1d_output_dim,
            residue_prediction_mode=self.cfg.predictor.residue_prediction_mode,
            reduction_mode=self.cfg.predictor.hparams.reduction_mode,
            penalty_weight=self.cfg.predictor.hparams.penalty_weight,
        )

    def forward(self, embeddings):
        """
        Performs forward pass on the LightAttention model.
        Args:
            embeddings (list[torch.tensor]): The input embeddings.
        Returns:
            np.array: The predicted values.
            np.array: Predicted probabilities (None for regression)
        """
        padded_embeddings, mask, _ = self.collate_fn([[x] for x in embeddings])
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logger.info(f"Moving embeddings to device: {device}")
        padded_embeddings, mask = padded_embeddings.to(device), mask.to(device)
        if self.model_type == "classification_binary":
            with torch.inference_mode():
                predicted_probabilities = self.model.forward(padded_embeddings, mask)
                predicted_probabilities = self.model.convert_to_numpy(predicted_probabilities, mask)
            predictions = (
                predicted_probabilities >= self.optimal_cutoff if self.optimal_cutoff else None
            )
        else:
            raise NotImplementedError

        return predictions, predicted_probabilities

    def post_train_model(self, x_val, y_val):
        """
        After training, find the optimal cutoff for classification.
        """
        _, predicted_probabilities = self.forward(x_val)
        if self.residue_prediction_mode:
            # flatten the array and select the non-padding positions
            predicted_probabilities = predicted_probabilities.compressed()
        self.optimal_cutoff, mcc_val = find_optimal_cutoff(y_val, predicted_probabilities)
        with open_dict(self.cfg):
            self.cfg.predictor.hparams.optimal_cutoff = self.optimal_cutoff
        logger.info(f"Optimal cutoff is: {self.optimal_cutoff} with MCC: {mcc_val}")

    def save_model(self, dir_path):
        """
        Saves the model.
        Args:
            dir_path (Path): The directory path to save the model.
        Returns:
            model_path (str): The path to where the model was saved
        """
        model_path = Path(dir_path, "model.pt")
        torch.save(self.model, model_path)
        return model_path

    def update_predictor_name(self) -> None:
        """Updates the predictor model name according to the hparams selected during training"""
        self.cfg.predictor.model_name = "lightattention_" + self.get_hparams_string()

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
