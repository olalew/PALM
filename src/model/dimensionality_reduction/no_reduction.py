import logging
import numpy as np

from pathlib import Path
from omegaconf import DictConfig

from src.model.abstract.abstract_dimention_reduction_model import DimRedModel

logger = logging.getLogger(__name__)


class NoReduction(DimRedModel):
    """
    Class for performing no dimensionality reduction.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initializes the NoReduction class.
        Args:
            cfg (DictConfig): Configuration dictionary.
        Attributes:
            fit (bool): Whether the model is fit.
        Methods:
            forward(embeddings: np.array) -> np.array: Perform forward pass of the dimensionality reduction.
            fitData(embeddings_train: np.array) -> np.array: Fit the training data to find the transform and number of PCs required.
            saveModel(dir_path: Path) -> None: Save the model.
        """
        logger.info(f"Load class (NoReduction): {self.__class__.__name__}")
        super().__init__(cfg)

        # No transformation
        self.fit = True

    def forward(self, embeddings) -> np.ndarray:
        return embeddings

    def fit_data(self, embeddings_train):
        return

    def save_model(self, path: Path):
        # No transformation
        return
