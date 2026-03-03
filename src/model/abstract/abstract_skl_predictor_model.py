import logging
from abc import ABC

import pandas as pd
from joblib import parallel_backend
from omegaconf import DictConfig
from sklearn.metrics import make_scorer, mean_squared_error, matthews_corrcoef
from sklearn.model_selection import GridSearchCV

from src.helpers.utilities import nJobs
from src.model.abstract.abstract_predictor_model import PredictorModel

logger = logging.getLogger(__name__)


class SKLPredictorModel(PredictorModel, ABC):
    """
    Abstract class that templates all Scikit-learn predictor models
    """

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def __init__(self, cfg: DictConfig):
        """
        Initializes a new instance of the SKLPredictorModel class.
        Args:
            cfg (DictConfig): The configuration for the predictor model.
        Attributes:
            cfg (DictConfig): The configuration for the model.
        Methods:
            forward: Forward pass of the model.
        """
        logger.info(f"Load class (SKLPredictorModel): {self.__class__.__name__}")
        super().__init__(cfg)

        self.model_type = cfg.predictor.model_type
        self.class_name = cfg.predictor.class_name
        self.mem_per_job = cfg.predictor.mem_per_job
        self.n_splits = cfg.predictor.hparam_tuning.n_splits

    def train_model(self, x_train, y_train, x_val, y_val, sample_weights=None):
        """
        Train a sklearn predictor model
        Args:
            x_train (np.array): The training data (n x h)
            y_train (np.array): The training data labels (n x 1)
            x_val (np.array): The validation data, not used here (n x h)
            y_val (np.array): The validation data labels, not used here (n x 1)
            sample_weights: ...
        """
        # Get the model selector
        param_grid = self.get_param_grid()
        logger.info(f"param_grid: {param_grid}")
        logger.info(f"Model type: {self.model_type}")
        if self.model_type == "regression":
            logger.info("Model type = regression, scorer = mean_squared_error")
            scorer = make_scorer(mean_squared_error)
        elif self.model_type == "classification_binary":
            logger.info("Model type = classifier, scorer = matthews_corrcoef")
            scorer = make_scorer(matthews_corrcoef)
        else:
            raise NotImplementedError

        if self.cfg.dataset.use_sample_weights is not None:
            scorer.set_score_request(sample_weight=True)
            self.model.set_fit_request(sample_weight=True)

        # Calculate the max number of jobs that we can run without an OOM error
        logger.info(f"The predictor is {self}")
        if self.model_type == "classifier" and self.class_name != "MLP":
            mem_per_job = (
                # If we don't know how many GB are required, then we assume a large value
                self.mem_per_job if self.mem_per_job else 16.0
            )
            n_jobs = nJobs(mem_per_job)
            logger.info(f"Training model with {n_jobs} workers in parallel")
            param_grid["n_jobs"] = [n_jobs]
            model_selector = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=self.n_splits,
                scoring=scorer,
                error_score="raise",
                verbose=3,
            )
        elif self.model_type == "classifier" and self.class_name == "MLP":
            # use joblib parallel backend for sklearn mlp as it can't take number of jobs as a parameter
            with parallel_backend("multiprocessing", n_jobs=-1):
                model_selector = GridSearchCV(
                    estimator=self.model,
                    param_grid=param_grid,
                    cv=self.n_splits,
                    scoring=scorer,
                    error_score="raise",
                    verbose=3,
                )
        else:
            model_selector = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=self.n_splits,
                scoring=scorer,
                error_score="raise",
                verbose=3,
            )

        # Search the hyperparameter space for the best model w.r.t. the scoring function
        logger.info(f"X_train shape: {x_train.shape}")
        logger.info(f"y_train shape: {y_train.shape}")
        logger.info(f"Parameter grid of different hyperparamenters: {self.get_param_grid()}")
        if sample_weights is not None:
            logger.info("Fitting the model with sample weights")

        model_selector.fit(x_train, y_train, sample_weight=sample_weights)
        self.model = model_selector.best_estimator_
        self.hparamsresults_df = pd.DataFrame(model_selector.cv_results_)
