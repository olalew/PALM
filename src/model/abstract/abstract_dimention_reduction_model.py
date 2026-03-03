import logging
from pathlib import Path

import numpy as np

from abc import ABCMeta, abstractmethod
from omegaconf import DictConfig

from src.model.common import component_type

logger = logging.getLogger(__name__)


class DimRedModel(metaclass=ABCMeta):
    """Abstract class that templates all dimensionality reduction classes"""

    # https://peps.python.org/pep-0487/#subclass-registration
    subclass_registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclass_registry[cls.__name__] = cls

    def __init__(self, cfg: DictConfig):
        """
        Initializes a new instance of the DimRedModel class.

        Args:
            cfg (DictConfig): The configuration for the dimensionality reduction model.

        Attributes:
            ctype (component_type): The type of component (dimensionality reduction).
            cfg (DictConfig): The configuration for the model.
            fit (bool): Whether the model is fit.

        Methods:
            forward: Forward pass of the model.
            fit_data: Fit the data to the model.
            save_model: Save the trained model.
        """
        logger.info(f"Load class (DimRed Model): {self.__class__.__name__}")
        super().__init__()
        self.ctype = component_type.dimred  # for convenient access
        self.cfg = cfg
        self.fit = False

    @abstractmethod
    def forward(self, embeddings: np.ndarray):
        """
        Abstract property representing the forward pass of the dimensionality reduction model.
        """
        pass

    @abstractmethod
    def fit_data(self, embeddings_train: np.ndarray):
        """
        Abstract property representing the fitting of data to the dimensionality reduction model.
        """
        pass

    @abstractmethod
    def save_model(self, dir_path: Path):
        """
        Abstract property representing the saving of the dimensionality reduction model.
        """
        pass
