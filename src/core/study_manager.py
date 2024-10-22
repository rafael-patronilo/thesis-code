import logging
from logging_setup import NOTIFY
from typing import Iterable, Callable, Literal, Optional, Any, assert_never
from .storage_management.study_file_manager import StudyFileManager
from .datasets import SplitDataset
from .trainer import Trainer, TrainerConfig
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
            stop_criteria : Optional[
                Callable[[dict], bool] | 
                tuple[str, Any] | 
                tuple[str, Any, Literal['ge', 'le', 'eq']]
                ] = None,
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
        if isinstance(stop_criteria, tuple):
            if len(stop_criteria) == 2:
                metric, value = stop_criteria
                strategy = 'eq'
            else:
                metric, value, strategy = stop_criteria
            match strategy:
                case 'eq':
                    self.stop_criteria = lambda x: x[metric] == value
                case 'ge':
                    self.stop_criteria = lambda x: x[metric] >= value
                case 'le':
                    self.stop_criteria = lambda x: x[metric] <= value
                case _:
                    assert_never(strategy)
        else:
            self.stop_criteria = stop_criteria

    def _create_trainer(self, model_file_manager, config : TrainerConfig, metrics_logger : MetricsLogger):
        trainer = Trainer.from_config(config)
        trainer.metric_loggers.append(metrics_logger)
        trainer.init_file_manager(model_file_manager)
        return trainer
    
    def run_experiment(self, experiment_name, config: TrainerConfig):
        with self.file_manager.new_experiment(experiment_name) as model_file_manager:
            metrics_logger = MetricsLogger(
                identifier="val",
                metric_functions=self.val_metrics,
                dataset=self.dataset.for_validation
            )
            
            logger.info(f"Running experiment {experiment_name}")
            trainer = self._create_trainer(model_file_manager, config, metrics_logger)
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
                if self.stop_criteria is not None and self.stop_criteria(metrics_logger.last_record):
                    logger.info(f"Stopping criteria met for {experiment_name}")
                    return

    def run_with_script(self, script : str, config_generator : Iterable[tuple[str, list, dict]]):
        generator : Iterable[tuple[str, TrainerConfig]] = (
            (name, dict(build_script=script, args=args, kwargs=kwargs)) 
            for name, args, kwargs in config_generator
        ) # type: ignore
        self.run(generator)

    def run(self, config_generator : Iterable[tuple[str, TrainerConfig]]):
        for experiment_name, details in config_generator:
            self.run_experiment(experiment_name, details)
        logger.log(NOTIFY, f"Training {experiment_name} complete")
        if self.best_results is not None:
            logger.info(f"Best experiment: {self.best_results[0]}")
            logger.info(f"Metrics: {self.best_results[1].last_record}")
            self.file_manager.save_results(self.results, self.best_results[0])
