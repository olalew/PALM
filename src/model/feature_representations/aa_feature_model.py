import logging
from abc import abstractmethod
from typing import Any

import numpy as np
import sklearn
import torch

from omegaconf import DictConfig
from torch import Tensor

from src.model.abstract.abstract_embedder_model import EmbedderModel

sklearn.set_config(enable_metadata_routing=True)

"""
All composite models consist of embedder, dimensionality reducer, and predictor modules chained together. 
The methods of these modules are templated by the abstract base classes defined here, while specific implementations 
defined in distinct source files. From a practical perspective, this helps ensure that new modules implement
the required methods for use within composite models.
"""

logger = logging.getLogger(__name__)


class AAFeaturizerModel(EmbedderModel):
    """
    Abstract class that templates all amino-acid level featurizers
    """

    def __init__(self, cfg: DictConfig):
        """
        Initializes a new instance of the AAFeaturizer class.
        Args:
            cfg (DictConfig): The configuration for the featurizer model.
        Attributes:
            ctype (component_type): The type of component (embedder).
            cfg (DictConfig): The configuration for the model.
        Methods:
            forward: Forward pass of the model.
        """
        logger.info(f"Load class (AAFeaturizer): {self.__class__.__name__}")
        super().__init__(cfg)

        # If True, will mean pool the embeddings over the sequence length dimension
        self.mean_pool = (
            self.cfg.embedder.mean_pool
        )

    @property
    @abstractmethod
    def aa_feature_mapping(self):
        """
        A residue-level mapping to feature vectors
        """
        pass

    @abstractmethod
    def validate_sequences(self, sequences: list):
        """
        To ensure that the sequences do not contain AA outside the alphabet
        """
        pass

    def forward(self, sequences: list) -> list[Tensor] | Any:
        self.validate_sequences(sequences)

        sequence_aa_features = [
            torch.tensor([self.aa_feature_mapping[char] for char in seq], dtype=torch.float)
            for seq in sequences
        ]
        if self.mean_pool:
            sequence_features = self.mean_pool_embeddings(sequence_aa_features)
            logger.info(f"Final mean-pooled shape: {sequence_features.shape}")
            return sequence_features
        else:
            logger.info(f"Final results len: {len(sequence_aa_features)}")
            return sequence_aa_features

    @staticmethod
    def mean_pool_embeddings(residue_embeddings: list, to_numpy=True):
        """
        Take the mean over the sequence length dimension
        Args:
            residue_embeddings (list[torch.tensor]): List of residue embeddings. b x l x h
            to_numpy:
        Returns:
            sequence_embeddings (list[torch.tensor]): List of sequence embeddings b x h
        """
        sequence_embeddings = torch.stack([x.mean(dim=0) for x in residue_embeddings])
        return sequence_embeddings.numpy() if to_numpy else sequence_embeddings
