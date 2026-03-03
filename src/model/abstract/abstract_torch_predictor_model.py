import logging
import os
from abc import ABC, abstractmethod

import mlflow
import lightning as L  # noqa: N812

from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from datetime import timedelta
from omegaconf import DictConfig

from src.model.abstract.abstract_predictor_model import PredictorModel

logger = logging.getLogger(__name__)


class TorchPredictorModel(PredictorModel, ABC):
    """
    Abstract class that templates all Pytorch Lightning predictor models
    """

    @abstractmethod
    def init_torch_module(self, embedding_dim):
        pass

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    @property
    def collate_fn(self):
        return self._collate_fn

    @collate_fn.setter
    def collate_fn(self, value):
        self._collate_fn = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def __init__(self, cfg: DictConfig):
        """
        Initializes a new instance of the TorchPredictorModel class.
        Args:
            cfg (DictConfig): The configuration for the predictor model.
        Attributes:
            cfg (DictConfig): The configuration for the model.
        Methods:
            forward: Forward pass of the model.
        """
        logger.info(f"Load class (TorchPredictorModel): {self.__class__.__name__}")
        super().__init__(cfg)
        self.model_type = cfg.predictor.model_type
        self.max_time = timedelta(days=0, hours=6)
        self.batch_size = cfg.predictor.hparams.batch_size
        self.max_epochs = cfg.predictor.hparams.max_epochs
        self.patience = cfg.predictor.hparams.patience

    def train_model(self, x_train, y_train, x_val, y_val, sample_weights):
        assert (
                x_train is not None
                and y_train is not None
                and x_val is not None
                and y_val is not None
        ), "Missing train/test data"

        train_set = self.dataset(x_train, y_train)
        val_set = self.dataset(x_val, y_val)

        if self.model is None:
            self.init_torch_module(train_set.embedding_dim())

        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        logger.info(f"This is the train loader {train_loader}")
        logger.info(f"This is the val loader {val_loader}")

        early_stopping = EarlyStopping(monitor="val.loss", patience=self.patience)
        checkpointer = ModelCheckpoint(save_top_k=1, mode="min", monitor="val.loss")
        mlflow_logger = MLFlowLogger(
            experiment_name=mlflow.get_experiment(mlflow.active_run().info.experiment_id).name,
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
            run_id=mlflow.active_run().info.run_id,
            log_model=True,
        )  # prefix argument automatically uses "-" to join, otherwise I would use it

        trainer = L.Trainer(
            callbacks=[early_stopping, checkpointer],
            max_time=self.max_time,
            deterministic=False,
            max_epochs=self.max_epochs,
            enable_progress_bar=False,
            log_every_n_steps=1,
            logger=mlflow_logger,
        )

        mlflow.pytorch.autolog(checkpoint_monitor="val.loss")
        trainer.fit(model=self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        stopped_epoch = early_stopping.stopped_epoch
        best_dir = checkpointer.best_model_path
        best_score = checkpointer.best_model_score
        logger.info(f"MODEL STOPPED BY EARLY STOPPER AT EPOCH {stopped_epoch}")
        logger.info(f"BEST MODEL VALIDATION LOSS IS {best_score}")
        logger.info(f"BEST MODEL SAVED AT {best_dir}")
        logger.info(f"FINAL MODEL SAVED AT {trainer.log_dir}")
        final_dir = "/".join(best_dir.split("/")[:-1]) + "/final.ckpt"
        trainer.save_checkpoint(filepath=final_dir)

        # Get the torch lightning module class
        pytorch_model_type = type(self.model)
        self.model = pytorch_model_type.load_from_checkpoint(best_dir)
        self.model.eval()

        # In case the derived class must do things post training
        post_train_model = getattr(self, "post_train_model", None)
        if callable(post_train_model):
            y_val = val_loader.dataset.get_labels()
            post_train_model(x_val, y_val)
