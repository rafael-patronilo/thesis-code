import logging
from core.logging import NOTIFY
from typing import Iterable, Callable, Literal, Optional, Any, assert_never
from core.storage_management.study_file_manager import StudyFileManager
from core.training.trainer import Trainer, TrainerConfig, ResultsDict

module_logger = logging.getLogger(__name__)

class StudyManager:
    def __init__(
            self, 
            file_manager : StudyFileManager,
            compare_strategy : Callable[[ResultsDict, ResultsDict], bool] | Literal["max", "min"],
            metric_key : Optional[tuple[str, str]] = ('train', 'loss'),
            stop_criteria : Optional[
                Callable[[ResultsDict], bool] |
                tuple[Any, Literal['ge', 'le', 'eq']]
                ] = None,
            num_epochs : int = 10,
            ):
        self.logger = module_logger.getChild(file_manager.study_name)
        self.name = file_manager.study_name
        self.logger = self.logger.getChild(self.name)
        self.logger.debug(f"Logger context switch")
        self.file_manager = file_manager
        self.compare_strategy : Callable[[dict, dict], bool]
        self.num_epochs = num_epochs
        self.best_results : Optional[tuple[str, ResultsDict]] = None
        self.results = {}
        self.skip_comparison = False
        if metric_key is not None:
            logger_identifier, metric_name = metric_key
        metric_getter = lambda x: x[logger_identifier][metric_name]
        if isinstance(compare_strategy, str):
            if metric_key is None:
                raise ValueError("Must provide metric_key if compare_strategy is a string")
            favored = compare_strategy
            match favored:
                case 'max':
                    self.compare_strategy = lambda x, y: metric_getter(x) > metric_getter(y)
                case 'min':
                    self.compare_strategy = lambda x, y: metric_getter(x) < metric_getter(y)
                case _:
                    assert_never(favored)
        else:
            self.compare_strategy = compare_strategy
        if isinstance(stop_criteria, tuple):
            if metric_key is None:
                raise ValueError("Must provide metric_key if stop_criteria is a tuple")
            value, strategy = stop_criteria
            match strategy:
                case 'eq':
                    self.stop_criteria = lambda x: metric_getter(x) == value
                case 'ge':
                    self.stop_criteria = lambda x: metric_getter(x) >= value
                case 'le':
                    self.stop_criteria = lambda x: metric_getter(x) <= value
                case _:
                    assert_never(strategy)
        else:
            self.stop_criteria : Optional[Callable[[dict], bool]] = stop_criteria

    def _create_trainer(self, model_file_manager, config : TrainerConfig):
        trainer = Trainer.from_config(config)
        trainer.init_file_manager(model_file_manager)
        checkpoint = model_file_manager.load_last_checkpoint()
        if checkpoint is not None:
            self.logger.info(f"Checkpoint found, loading...")
            trainer.load_state_dict(checkpoint)
        else:
            self.logger.info("No checkpoint")
        return trainer
    
    def run_experiment(self, experiment_name : str, config: TrainerConfig):
        with self.file_manager.new_experiment(experiment_name) as model_file_manager:
            self.logger.info(f"Running experiment {experiment_name} for {self.num_epochs} epochs")
            trainer = self._create_trainer(model_file_manager, config)
            model_file_manager.save_config(config)
            if trainer.epoch >= self.num_epochs:
                self.logger.info(f"Experiment {experiment_name} was already complete.")
            else:
                trainer.train_until_epoch(self.num_epochs)
                self.logger.log(NOTIFY, f"Experiment {experiment_name} complete")
            if self.skip_comparison:
                self.logger.warning("Skipping automatic results comparison")
            else:
                try:
                    self._evaluate_experiment(experiment_name, trainer)
                except BaseException as e:
                    self.logger.error(f"Error comparing experiment {experiment_name}: {e}\n" +
                                 "Training will continue skipping comparison")
                    self.skip_comparison = True


    def _evaluate_experiment(self, experiment_name : str, trainer : Trainer):
        snapshot = trainer.get_results_dict()
        self.results[experiment_name] = snapshot
        if self.best_results is None:
                self.best_results = (experiment_name, snapshot)
                self.logger.debug(f"Saved first experiment {experiment_name} as best")
        else:
            
            if self.compare_strategy(snapshot, self.best_results[1]):
                self.best_results = (experiment_name, snapshot)
                self.logger.info(f"New best experiment {experiment_name}")
            else:
                self.logger.info(f"Best experiment remains {self.best_results[0]}")

    def check_stop_criteria(self) -> bool:
        if self.best_results is None:
            return False
        experiment_name, snapshot = self.best_results
        if self.stop_criteria is not None and self.stop_criteria(snapshot):
            self.logger.info(f"Stopping criteria met for {experiment_name}")
            return True
        else:
            return False
    
    def run_with_script(self, script : str, config_generator : Iterable[tuple[str, list, dict]]):
        generator : Iterable[tuple[str, TrainerConfig]] = (
            (name, dict(build_script=script, build_args=args, build_kwargs=kwargs)) 
            for name, args, kwargs in config_generator
        ) # type: ignore
        self.run(generator)

    def run(self, config_generator : Iterable[tuple[str, TrainerConfig]]):
        for experiment_name, details in config_generator:
            self.run_experiment(experiment_name, details)
            if self.check_stop_criteria():
                break
        self.logger.log(NOTIFY, f"Study complete")
        self.store_results()

    def store_results(self):
        if self.best_results is not None:
            self.logger.info(f"Best experiment: {self.best_results[0]}")
            self.logger.info(f"Metrics: {self.best_results[1]}")
            self.file_manager.save_results(self.results, self.best_results[0])
        else:
            self.logger.warning("No results to store")