import logging
from abc import ABC

import numpy as np
import torch
from omegaconf import DictConfig

from src.helpers.utilities import CudaOutOfMemoryError
from src.model.abstract.abstract_embedder_model import EmbedderModel

logger = logging.getLogger(__name__)


class LLMEmbedderModel(EmbedderModel, ABC):
    """
    Abstract class that templates all LLM embedder models
    """

    def __init__(self, cfg: DictConfig):
        """
        Initializes a new instance of the LLMEmbedderModel class.
        Args:
            cfg (DictConfig): The configuration for the embedder model.
        Attributes:
            ctype (component_type): The type of component (embedder).
            cfg (DictConfig): The configuration for the model.
        Methods:
            forward: Forward pass of the model.
        """
        logger.info(f"Load class (LLMEmbedderModel): {self.__class__.__name__}")
        super().__init__(cfg)

        self.batch_scale_factors = {
            "esm2_t6_8M_UR50D": (164.216311, 6796.137809, 1.258964e05),
            "esm2_t12_35M_UR50D": (165.052996, 8850.407079, 5.208933e05),
            "esm2_t30_150M_UR50D": (165.341865, 11189.307840, 1.045312e06),
            "esm2_t33_650M_UR50D": (142.660817, 36701.930699, 3.602179e06),
            "esm2_t36_3B_UR50D": (
                142.660817,
                66701.930699,
                6.602179e06,
            ),  # these are estimates
            "esm2_t48_15B_UR50D": (
                142.660817,
                66701.930699,
                6.602179e06,
            )
        }
        self.buffer_scale_factor = self.cfg.embedder.buffer_scale_factor  # Extra memory left unused
        self.max_batch_size = (
            self.cfg.embedder.max_batch_size
        )  # The maximum number of sequences allowed in a batch
        self.strict = (
            self.cfg.embedder.strict
        )  # If True, will throw an error if there is insufficient memory to embed a sequence
        self.verbose = self.cfg.embedder.verbose  # If True, will print info about the batches
        self.simple_batching = (
            self.cfg.embedder.simple_batching
        )  # If True, will batch each sequence separately (very slow)
        self.mean_pool = (
            self.cfg.embedder.mean_pool
        )  # If True, will mean pool the embeddings over the sequence length dimension

    @staticmethod
    def quadratic_mem_util(X: tuple[float], a: float, b: float,
                           c: float):  # noqa: N803 argument name X should be lower case.
        """
        Returns the memory required to embed a batch in bytes
        Args:
            X (tuple[float]): The sequence length and batch size
            a (float): The first embedder-specific constant
            b (float): The second embedder-specific constant
            c (float): The third embedder-specific constant
        """
        x1, x2 = X
        return a * (x1 ** 2) * x2 + b * x1 * x2 + c * x2

    def mem_req_for_batch(self, batch: list[str]):
        """
        Predict the amount of additional memory (in bytes) that will be required to embed the batch
        Args:
            batch (list[str]): List of sequences
        Returns:
            mem_req (float): The bytes required to embed the batch
        """
        batch_size = float(len(batch))
        max_len = float(np.array([len(x) for x in batch]).max())
        return (
                self.quadratic_mem_util(
                    (max_len, batch_size), self.factor_1, self.factor_2, self.factor_3
                )
                * self.buffer_scale_factor
        )

    def get_batches(self, sequences: list[str]):
        """
        Generator that batches sequences
        Args:
            sequences (list[str]): List of sequences
        Returns:
            batch (list[str]): A valid batch
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == 'cpu':
            logger.info("No GPU found, switching to simple batching.")

            for seq in sequences:
                yield [seq]
        else:
            total_mem_avail = torch.cuda.get_device_properties(0).total_memory
            base_mem_allocated = torch.cuda.memory_allocated(0)
            logger.info(
                f"Total memory: {total_mem_avail / 1e9}GB\nMemory already allocated: {base_mem_allocated / 1e9}GB"
            )

            batch = []
            for seq in sequences:
                seq_mem_req = self.mem_req_for_batch([seq])
                if base_mem_allocated + seq_mem_req > total_mem_avail:
                    logger.info(
                        f"Cannot embed sequence of length: {len(seq)}, estimated memory requirement: {seq_mem_req / 1e9:0.2f}GB"
                    )
                    if self.strict:
                        raise CudaOutOfMemoryError("Unable to embed sequence", [len(seq)])

                if len(batch) >= self.max_batch_size or (
                        base_mem_allocated + self.mem_req_for_batch(batch + [seq]) > total_mem_avail
                ):
                    # Sequence doesn't fit in current batch, process current batch, and then try again with an empty batch
                    if self.verbose:
                        self.summarize_batch(batch)
                    yield batch
                    batch = []

                # Add sequence to batch and iterate
                batch.append(seq)

            if self.verbose:
                self.summarize_batch(batch)
            yield batch

    @staticmethod
    def summarize_batch(sequences: list[str]):
        """
        Print the length of each sequence in the batch
        Args:
            sequences (list[str]): List of sequences.
        Returns:
            None
        """
        logger.info(
            f"Batch with {len(sequences)} sequences of length: {[len(x) for x in sequences]}"
        )

    @staticmethod
    def mean_pool_embeddings(residue_embeddings: list[torch.Tensor], to_numpy=True):
        """
        Take the mean over the sequence length dimension
        Args:
            residue_embeddings (list[torch.tensor]): List of residue embeddings. b x l x h
        Returns:
            sequence_embeddings (list[torch.tensor]): List of sequence embeddings b x h
        """
        sequence_embeddings = torch.stack([x.mean(dim=0) for x in residue_embeddings])
        return sequence_embeddings.numpy() if to_numpy else sequence_embeddings

    def validate_layer_idx(self) -> None:
        """
        Validate the layer index against model configuration.
        This method should be called before processing any sequences to ensure
        the specified layer index is valid for the model.
        Raises:
            ValueError: If layer_idx is specified but greater than the model's number of layers
            NotImplementedError: If called from base class (must be implemented by child class)
        """
        if not hasattr(self, "model"):
            raise NotImplementedError(
                "_validate_layer_idx called on base class. Must be implemented by child class."
            )

        if hasattr(self.cfg.embedder, "layer_idx"):
            layer_idx = self.cfg.embedder.layer_idx
            if layer_idx is not None:  # Check if layer_idx is specified
                try:
                    num_layers = self.model.config.num_hidden_layers
                    if layer_idx > num_layers:
                        raise ValueError(
                            f"Layer index {layer_idx} is greater than the number "
                            f"of hidden layers in the model ({num_layers}). "
                            f"Please specify a layer index between 1 and {num_layers}."
                        )
                except AttributeError as e:
                    logger.error(f"Model configuration appears invalid: {e}")
                    raise

    def should_use_specific_layer(self) -> bool:
        """Check if embeddings should be extracted from a specific layer.
        Returns:
            bool: True if a specific layer should be used, False otherwise
        """
        return (
                hasattr(self.cfg.embedder, "output_hidden_states")
                and hasattr(self.cfg.embedder, "layer_idx")
                and self.cfg.embedder.output_hidden_states
                and self.cfg.embedder.layer_idx
                and self.cfg.embedder.layer_idx <= self.model.config.num_hidden_layers
        )
