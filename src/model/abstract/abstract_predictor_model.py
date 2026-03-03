from abc import ABCMeta, abstractmethod
from omegaconf import DictConfig
import logging

from src.model.common import component_type

logger = logging.getLogger(__name__)


class PredictorModel(metaclass=ABCMeta):
    """Abstract class that templates all predictor models"""

    # https://peps.python.org/pep-0487/#subclass-registration
    subclass_registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclass_registry[cls.__name__] = cls

    def __init__(self, cfg: DictConfig):
        """
        Initialize the PredictorModel.

        Args:
            cfg (DictConfig): The configuration for the model.

        Attributes:
            ctype (component_type): The type of component (predictor).
            cfg (DictConfig): The configuration for the model.
            trained (bool): Whether the model is trained.

        Methods:
            forward: Forward pass of the model.
            save_model: Save the trained model.
            get_hparams_string: Get a string representation of the hyperparameters.
            get_param_grid: Get the parameter grid for hyperparameter tuning.
        """
        logger.info(f"Load class (Predictor Model): {self.__class__.__name__}")
        super().__init__()
        self.ctype = component_type.predictor  # for convenient access
        self.cfg = cfg
        self.trained = False
        self.hparamsresults_df = None

    @abstractmethod
    def forward(self, embeddings):
        """
        Abstract property to get the forward pass of the model.
        """
        pass

    @abstractmethod
    def train_model(self, x_train, y_train, x_val, y_val, sample_weights):
        """
        Abstract property to train the model.
        """
        pass

    @abstractmethod
    def save_model(self, dir_path):
        """
        Abstract property to save the trained model.
        """
        pass

    @abstractmethod
    def get_hparams_string(self) -> str:
        """
        Abstract property to get a string representation of the hyperparameters.

        Returns:
            str: The string representation of the hyperparameters.
        """
        pass

    @abstractmethod
    def get_param_grid(self) -> dict:
        """
        Abstract property to get the parameter grid for hyperparameter tuning.

        Returns:
            dict: The parameter grid.
        """
        pass
