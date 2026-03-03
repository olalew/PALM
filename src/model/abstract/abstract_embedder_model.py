import logging
from abc import ABCMeta, abstractmethod

import numpy as np
from omegaconf import DictConfig
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from skops.io import dump as sk_dump
from skops.io import load as sk_load

from src.model.common import component_type

logger = logging.getLogger(__name__)


class EmbedderModel(metaclass=ABCMeta):
    """Abstract class that templates all embedder models"""

    # https://peps.python.org/pep-0487/#subclass-registration
    subclass_registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclass_registry[cls.__name__] = cls

    def __init__(self, cfg: DictConfig):
        """
        Initializes a new instance of the EmbedderModel class.
        Args:
            cfg (DictConfig): The configuration for the embedder model.
        Attributes:
            ctype (component_type): The type of component (embedder).
            cfg (DictConfig): The configuration for the model.
        Methods:
            forward: Forward pass of the model.
        """
        logger.info(f"Load class (Embedder Model): {self.__class__.__name__}")
        super().__init__()
        self.ctype = component_type.embedder  # for convenient access
        self.cfg = cfg

        # Initialize scaler as None
        self.scaler = None

    def standardize_embeddings(self, embeddings, fit=False):
        """
        Standardize embeddings using sklearn's MinMaxScaler.
        Args:
            embeddings: Can be either:
                - numpy array of shape (n_sequences, n_features) for mean-pooled embeddings or embeddings from biophys,amortport,etc
                - list of torch tensors of shape (seq_len, n_features) for per-residue embeddings
            fit (bool): Whether to fit the scaler on this data. Should be True for training data.
        Returns:
            Standardized embeddings in the same format as input
        """
        if self.scaler is None and fit:
            if self.cfg.embedder.scalar_type == "MinMaxScaler":
                self.scaler = MinMaxScaler()
            elif self.cfg.embedder.scalar_type == "StandardScaler":
                self.scaler = StandardScaler()
            elif self.cfg.embedder.scalar_type == "RobustScaler":
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaler type: {self.cfg.embedder.scaler_type}")

        if isinstance(embeddings, np.ndarray):
            # logger.info(f"Before standardization - min: {embeddings.min()}, max: {embeddings.max()}")
            if fit:
                result = self.scaler.fit_transform(embeddings)
            else:
                result = self.scaler.transform(embeddings)
            # logger.info(f"After standardization - min: {result.min()}, max: {result.max()}")
            return result

        # In future add logic for embeddings along residue when mean_pool is set to false
        else:
            raise ValueError("Embeddings must be either a numpy array or a list of torch tensors")

    def save_scaler(self, path):
        """
        Save the fitted scaler to disk.

        Args:
            path (str or Path): Path to save the scaler
        """
        if self.scaler is not None:
            sk_dump(self.scaler, path)
            logger.info(f"Saved scaler to {path}")

    def load_scaler(self, path):
        """
        Load a previously fitted scaler from disk.

        Args:
            path (str or Path): Path to the saved scaler
        """
        self.scaler = sk_load(path)
        logger.info(f"Loaded scaler from {path}")

    @abstractmethod
    def forward(self, sequences: list):
        """
        Abstract method for the forward pass of the embedder model.
        """
        pass
