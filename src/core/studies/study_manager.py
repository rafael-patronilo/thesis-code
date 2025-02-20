import logging
from core.logging import NOTIFY
from typing import Iterable, Callable, Literal, Optional, Any, Sized, assert_never
from core.storage_management.study_file_manager import StudyFileManager
from core.training.trainer import Trainer, TrainerConfig, ResultsDict
from core.eval.objectives import Objective
from core.util.progress_trackers import LogProgressContextManager
from core.training.stop_criteria.goal_reached import GoalReached

module_logger = logging.getLogger(__name__)

experiments_progress_cm = LogProgressContextManager(module_logger)

class StudyManager:
    def __init__(
            self, 
            file_manager : StudyFileManager,
            objective : Optional[Objective] = None,
            goal : Optional[GoalReached] | Literal['from_trainer'] = None,
            max_epochs : Optional[int] = 100,
            ):
        self.logger = module_logger.getChild(file_manager.study_name)
        self.name = file_manager.study_name
        self.logger = self.logger.getChild(self.name)
        self.logger.debug(f"Logger context switch")
        self.file_manager = file_manager
        self.compare_strategy : Callable[[dict, dict], bool]
        self.num_epochs = max_epochs
        self.best_results : Optional[tuple[str, ResultsDict]] = None
        self.stop = False
        self.goal = goal
        self.results = {}
        self.skip_comparison = False
        self.objective = objective

    def _create_trainer(self, model_file_manager, config : TrainerConfig):
        trainer = Trainer.from_config(config)
        trainer.init_file_manager(model_file_manager)
        checkpoint = model_file_manager.load_checkpoint(prefer='last')
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
            if self.num_epochs is None:
                trainer.train_indefinitely()
            else:
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
                                 "Training will continue skipping comparison", exc_info=True)
                    self.skip_comparison = True

    def _evaluate_experiment(self, experiment_name : str, trainer : Trainer):
        if trainer.best_checkpoint_results is not None:
            results = trainer.best_checkpoint_results
        else:
            results = trainer.get_results_dict()
        self.results[experiment_name] = results
        if results is None:
            raise ValueError(f"Results for experiment {experiment_name} are None")
        if self.objective is None:
            trainer_objective = trainer.objective
            if trainer_objective is None:
                raise ValueError("No objective found")
            self.objective = trainer_objective
        if self.goal == 'auto':
            trainer_goal = [crit for crit in trainer.stop_criteria if isinstance(crit, GoalReached)]
            if len(trainer_goal) >= 1:
                if len(trainer_goal) > 1:
                    self.logger.warning(f"Multiple GoalReached criteria found, selecting first")
                self.goal = trainer_goal[0]
            else:
                self.logger.warning("No GoalReached criteria found, "
                                    "no goal has been set and all experiments will be run")
                self.goal = None
        if isinstance(self.goal, GoalReached):
            if self.goal(trainer):
                self.logger.info(f"Goal reached by experiment {experiment_name}")
                self.best_results = (experiment_name, results)
                self.stop = True
        if self.best_results is None:
                self.best_results = (experiment_name, results)
                self.logger.debug(f"Saved first experiment {experiment_name} as best")
        else:
            if self.objective.compare_strict(results, self.best_results[1]):
                self.best_results = (experiment_name, results)
                self.logger.info(f"New best experiment {experiment_name}")
            else:
                self.logger.info(f"Best experiment remains {self.best_results[0]}")
    
    def run_with_script(self, script : str, config_generator : Iterable[tuple[str, list, dict]]):
        generator : Iterable[tuple[str, TrainerConfig]] = (
            (name, dict(build_script=script, build_args=args, build_kwargs=kwargs)) 
            for name, args, kwargs in config_generator
        ) # type: ignore
        self.run(generator)

    def run(self, config_generator : Iterable[tuple[str, TrainerConfig]]):
        with experiments_progress_cm.track(self.name, 'experiments', config_generator) as progress:
            for experiment_name, details in config_generator:
                if self.stop:
                    self.logger.info("Goal reached, stopping further experiments")
                    break
                self.run_experiment(experiment_name, details)
                progress.tick()
        self.store_results()
        self.logger.log(NOTIFY, f"Study complete")

    def store_results(self):
        if self.best_results is not None:
            self.logger.info(f"Best experiment: {self.best_results[0]}")
            self.logger.info(f"Metrics: {self.best_results[1]}")
            self.file_manager.save_results(self.results, self.best_results[0])
        else:
            self.logger.warning("No results to store")