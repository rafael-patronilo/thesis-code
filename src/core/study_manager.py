import logging
from logging_setup import NOTIFY
from typing import Iterable, Callable, Literal, Optional
from .storage_management.study_file_manager import StudyFileManager
from .datasets import SplitDataset
from . import ModelDetails
from .trainer import Trainer
from .metrics_logger import MetricsLogger
from . import modules

logger = logging.getLogger(__name__)

class StudyManager:
    def __init__(
            self, 
            file_manager : StudyFileManager, 
            dataset : SplitDataset,
            val_metrics : list[modules.metrics.NamedMetricFunction],
            compare_strategy : Callable[[dict, dict], bool] | tuple[str, Literal["max", "min"]],
            num_epochs : int = 10,
            ):
        self.name = file_manager.study_name
        self.file_manager = file_manager
        self.dataset = dataset
        self.compare_strategy : Callable[[dict, dict], bool]
        self.val_metrics = val_metrics
        self.num_epochs = num_epochs
        self.best_results : Optional[tuple[str, MetricsLogger]] = None
        self.results = {}
        if isinstance(compare_strategy, tuple):
            metric, favored = compare_strategy
            if favored == "max":
                self.compare_strategy = lambda x, y: x[metric] > y[metric]
            elif favored == "min":
                self.compare_strategy = lambda x, y: x[metric] < y[metric]
            else:
                raise ValueError(f"Invalid compare strategy {(metric,favored)}")
        else:
            self.compare_strategy = compare_strategy

    def _create_trainer(self, model_file_manager, model_details : ModelDetails, metrics_logger : MetricsLogger):
        model = model_details.architecture
        loss_fn = modules.get_loss_function(model_details.loss_fn)
        optimizer = modules.get_optimizer(model_details.optimizer, model)
        
        trainer = Trainer(
                model = model,
                loss_fn = loss_fn,
                optimizer = optimizer,
                training_loader = self.dataset.for_training(),
                model_file_manager = model_file_manager,
                metric_loggers = [metrics_logger],
                epoch = 0,
                checkpoint_each = None
            )
        return trainer
    
    def run_experiment(self, experiment_name, details):
        with self.file_manager.new_experiment(experiment_name) as model_file_manager:
            metrics_logger = MetricsLogger(
                identifier="val",
                metric_functions=self.val_metrics,
                dataset=self.dataset.for_validation
            )
            
            logger.info(f"Running experiment {experiment_name}")
            trainer = self._create_trainer(model_file_manager, details, metrics_logger)
            trainer.train_epochs(self.num_epochs)
            logger.info(f"Experiment {experiment_name} complete")
            logger.info(f"Metrics: {metrics_logger.last_record}")
            assert metrics_logger.last_record is not None
            self.results[experiment_name] = metrics_logger.last_record
            if self.best_results is None:
                self.best_results = (experiment_name, metrics_logger)
                logger.debug(f"Saved first experiment {experiment_name} as best")
            else:
                assert self.best_results[1].last_record is not None
                if self.compare_strategy(metrics_logger.last_record, self.best_results[1].last_record):
                    self.best_results = (experiment_name, metrics_logger)
                    logger.info(f"New best experiment {experiment_name}")


    def run(self, model_generator : Iterable[tuple[str, ModelDetails]]):
        for experiment_name, details in model_generator:
            self.run_experiment(experiment_name, details)
        logger.log(NOTIFY, f"Training {experiment_name} complete")
        if self.best_results is not None:
            logger.info(f"Best experiment: {self.best_results[0]}")
            logger.info(f"Metrics: {self.best_results[1].last_record}")
            self.file_manager.save_results(self.results, self.best_results[0])
