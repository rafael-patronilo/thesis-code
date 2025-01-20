from collections import OrderedDict
import logging
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from typing import Optional, Any, Callable, Literal, TypedDict, Iterable, TYPE_CHECKING
from core.training.metrics_recorder import MetricsRecorder, TrainingRecorder
from core.datasets import SplitDataset
from core.storage_management.model_file_manager import ModelFileManager
from core.logging import NOTIFY, log_file

import os
import importlib
import torch
from torch import nn
from datetime import timedelta
from core import util as utils
from core.util.progress_trackers import LogProgressContextManager
from core.eval.objectives import Objective
if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer

module_logger = logging.getLogger(__name__)


def _dataloader_worker_init_fn(worker_id):
    module_logger.debug(f"Training Dataloader worker {worker_id} initialized")
    torch.set_default_device('cpu')

def debug_model(error : BaseException, model : nn.Module, x, y):
    sb = [
        "\nModel debug info",
        f"\tError: {error}",
        f"\tModel: {model}"]
    if x is not None and y is not None:
        sb.append(f"\tX shape: {x.shape}")
        sb.append(f"\tY shape: {y.shape}")
        sb.append(f"\tX dtype: {x.dtype}")
        sb.append(f"\tY dtype: {y.dtype}")
        sb.append(f"\tX device: {x.device}")
        sb.append(f"\tY device: {y.device}")
    def debug_tensor(x) -> str:
        if isinstance(x, torch.Tensor):
            return f"Tensor {x.shape} {x.dtype} {x.device}"
        elif isinstance(x, tuple):
            return f"({', '.join(debug_tensor(sub) for sub in x)})"
        elif isinstance(x, list):
            return f"[{', '.join(debug_tensor(sub) for sub in x)}]"
        else:
            return str(x)
    def forward_hook(module, input, output):
        sb.append(f"\t\t{module.__class__.__name__} : {debug_tensor(input)} -> {debug_tensor(output)}")
    sb.append(f"\tSub modules:")
    for module in model.modules():
        sb.append(f"\t\t{module.__class__.__name__}")
    
    sb.append(f"\tDebugging forward pass:")
    torch.nn.modules.module.register_module_forward_hook(forward_hook, always_call=True)
    try:
        result = model.forward(x)
        sb.append(f"\tResult shape: {result.shape}")
        sb.append(f"\tExpected shape: {y.shape}")
    except BaseException as e:
        sb.append(f"\tError during forward pass: {e}")
        pass
    return "\n".join(sb)

type ResultsDict = dict[str, dict[str, Any]]

type CheckpointReason = Literal[
    'interrupt', 
    'force_interrupt', 
    "error", 
    "eval_error", 
    'triggered', 
    'periodic', 
    'end'
]

# Checkpoint reasons that may indicate a checkpoint in the middle of an epoch
# 
# Note: errors during evaluation don't count as abrupt because
#   at that point the training epoch should be complete and
#   the model weights are not in an incomplete update state
ABRUPT_CHECKPOINT_REASONS = [
    'force_interrupt', 
    "error"
]

class TrainerConfig(TypedDict, total=True):
    build_script : str
    build_args : list[Any]
    build_kwargs : dict[str, Any]

class _EvalExceptionWrapper(BaseException):
    def __init__(self, inner : BaseException):
        self.inner = inner


class Trainer:
    def __init__(
            self,
            model : 'nn.Module',
            loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            optimizer : Callable[[Any], 'Optimizer'],
            training_set : 'SplitDataset | Dataset',
            metric_loggers : list[MetricsRecorder],
            epoch : int = 0,
            batch_size : int = 32,
            num_loaders : int = int(os.getenv('NUM_THREADS', 4)),
            shuffle : bool = True,
            checkpoint_each : Optional[int] = 10,
            checkpoint_triggers : Optional[list[Callable[[Any], bool]]] = None,
            stop_criteria : Optional[list[Callable[['Trainer'], bool]]] = None,
            objective : Optional[Objective] = None
        ):
        self._set_logger(module_logger)
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer(model.parameters())
        self.training_set = training_set
        self.model_file_manager : Optional[ModelFileManager] = None
        self.metric_loggers : list[MetricsRecorder] = metric_loggers
        self.train_logger = None
        _train_logger = [logger for logger in metric_loggers if isinstance(logger, TrainingRecorder)]
        if len(_train_logger) > 1:
            raise ValueError("Multiple TrainLossLoggers found")
        elif len(_train_logger) == 1:
            self.train_logger = _train_logger[0]
        #self.callbacks = callbacks or {}
        self.start_epoch : int = epoch
        self.epoch : int = epoch
        self.checkpoint_triggers = checkpoint_triggers or []
        self.stop_criteria = stop_criteria or []
        self.checkpoint_each = checkpoint_each
        self.batch_size = batch_size
        self.num_loaders = num_loaders
        self.shuffle = shuffle
        self.first_epoch = True
        self.epoch_checkpoint = False
        self.best_checkpoint_results : Optional[ResultsDict] = None
        self.objective = objective

    def _set_logger(self, logger : logging.Logger):
        self.logger = logger
        self.progress_cm = LogProgressContextManager(self.logger, cooldown=timedelta(minutes=5))
        if self.logger is not module_logger:
            self.logger.debug(f"Logger context switch")

    def checkpoint_metadata(self, reason : CheckpointReason):
        assert self.model_file_manager is not None
        return {
            "epoch" : self.epoch,
            "start_epoch" : self.start_epoch,
            "path" : str(self.model_file_manager.path),
            "reason" : reason,
            "device" : torch.get_default_device().type,
            "logfile" : str(log_file)
        }
    
    def init_file_manager(self, model_file_manager : ModelFileManager) -> 'Trainer':
        self.model_file_manager = model_file_manager
        model_file_manager.init_directory()
        for metric_logger in self.metric_loggers:
            self.model_file_manager.init_metrics_file(metric_logger.identifier, metric_logger.metrics_header)
        self._set_logger(module_logger.getChild(model_file_manager.model_name))
        return self

    
    def state_dict(self, reason : CheckpointReason):
        metadata = self.checkpoint_metadata(reason)
        self.logger.debug(f"Checkpoint metadata: {metadata}")
        stop_criteria = {
            criterion.__class__.__name__ : criterion.state_dict() # type: ignore # hasattr ensures state_dict
            for criterion in self.stop_criteria
            if hasattr(criterion, 'state_dict')
        }
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "metrics": {metric_logger.identifier : metric_logger.state_dict() for metric_logger in self.metric_loggers},
            "stop_criteria" : stop_criteria,
            "metadata": metadata
        }
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.epoch = state_dict["epoch"]
        self.start_epoch = state_dict["epoch"]
        for metric_logger in self.metric_loggers:
            metric_logger.load_state_dict(state_dict["metrics"][metric_logger.identifier])
        for criterion in self.stop_criteria:
            if hasattr(criterion, 'load_state_dict'):
                if criterion.__class__.__name__ not in state_dict['stop_criteria']:
                    self.logger.warning(f"Stop criterion {criterion.__class__.__name__} not found in checkpoint")
                    continue
                try:
                    criterion.load_state_dict(state_dict['stop_criteria'][criterion.__class__.__name__]) #type: ignore # hasattr ensures load_state_dict
                except BaseException as e:
                    self.logger.error(f"Error loading stop criterion {criterion.__class__.__name__}: {e}",
                                      exc_info=True)
        metadata = state_dict['metadata']
        self.logger.info(f"Loaded checkpoint: {utils.multiline_str(metadata)}")
        if metadata['reason'] in ABRUPT_CHECKPOINT_REASONS:
            self.logger.warning(f"Abrupt checkpoint: Checkpoint was saved due to {metadata['reason']}")
        elif metadata['reason'] == 'eval_error':
            self.logger.warning(f"Checkpoint was saved due to evaluation error. Attempting to evaluate again...")
            self.eval_metrics()
            if self.model_file_manager is not None:
                self.model_file_manager.flush()
            self.logger.info(self._metrics_str())
        self.epoch += 1

    def train_indefinitely(self):
        self.train_until([])

    def train_until_epoch(self, end_epoch : int):
        self.train_until([lambda x : x.epoch >= end_epoch])

    def train_epochs(self, num_epochs):
        self.train_until_epoch(self.epoch + num_epochs)

    def get_results_dict(self) -> ResultsDict | None:
        if self.first_epoch:
            return None
        snapshot = {}
        for metric_logger in self.metric_loggers:
            snapshot[metric_logger.identifier] = metric_logger.last_record
        return snapshot

    def train_until(self, criteria : list[Callable[['Trainer'], bool]]):
        if self.model_file_manager is None:
            raise ValueError("Model file manager not initialized")
        criteria = criteria + self.stop_criteria
        self.logger.info("Initiating training loop...\n"
                    f"Model: {self.model_file_manager.model_name}\n"
                    f"Epoch: {self.epoch}\n"
                    f"Checkpoint each: {self.checkpoint_each}\n"
                    f"Keyboard interrupt to save checkpoint and exit.")
        
        try:
            no_interrupt = utils.NoInterrupt("mid epoch", self.logger)
            while not any(criterion(self) for criterion in criteria):
                with no_interrupt:
                    if not self.first_epoch:
                        self.epoch += 1
                    self._train_epoch(self.epoch)
            self._checkpoint('end')
            self.logger.log(NOTIFY, "Training complete.")
        except (KeyboardInterrupt, utils.NoInterrupt.InterruptException):
            self.logger.log(NOTIFY, "Training safely interrupted. Saving checkpoint...")
            self._checkpoint('interrupt')
            raise
        except (SystemExit, utils.NoInterrupt.ForcedInterruptException):
            self.logger.error("Forced interrupt. Saving checkpoint...")
            self._checkpoint('force_interrupt')
            raise
        except _EvalExceptionWrapper as e:
            self.logger.error(f"Exception caught during evaluation. Saving checkpoint...")
            self._checkpoint('eval_error')
            raise e.inner
        except:
            if self.epoch == 0:
                self.logger.error("Exception caught at epoch 0. Skipping checkpoint...")
            else:
                self.logger.error("Exception caught while training. Saving checkpoint...")
                self._checkpoint('error')
            raise
    
    def _metrics_str(self):
        averages = ""
        averages_last_n = ""
        last_record = ""
        n = None
        #to_display = [name for logger in self.metric_loggers for name,_ in logger.ordered_metrics]
        def format_dict(values : dict | None):
            if values is None:
                return 'N/A'
            def fmt_value(value):
                if value is None:
                    return 'N/A'
                elif value < 0.01:
                    return f"{value:.4e}"
                else:
                    return f"{value:.4f}"
            return '\n'.join(f"\t\t{metric} : {fmt_value(value)}" for metric, value in values.items() )
        for metric_logger in self.metric_loggers:
            n = n or len(metric_logger.last_n)
            averages += f"\t{metric_logger.identifier} : {format_dict(metric_logger.averages())}\n"
            averages_last_n += f"\t{metric_logger.identifier} : {format_dict(metric_logger.averages_last_n())}\n"
            last_record += f"\t{metric_logger.identifier} : {format_dict(metric_logger.last_record)}\n"
        return (
            f"Metrics:\n{last_record}"
            f"Avg:\n{averages}"
            f"Avg last {n}:\n{averages_last_n}"
            )

    def _checkpoint(self, reason : CheckpointReason):
        if self.first_epoch:
            self.logger.warning("No training done yet, skipping checkpoint")
            return
        if self.epoch_checkpoint:
            self.logger.warning("Checkpoint already saved for this epoch, skipping")
            return
        if self.model_file_manager is None:
            raise ValueError("Model file manager not initialized")
        is_abrupt = reason in ABRUPT_CHECKPOINT_REASONS
        is_best = False
        results = self.get_results_dict()
        if results is not None and not is_abrupt and self.objective is not None:
            if self.best_checkpoint_results is None:
                self.logger.debug("Initializing best checkpoint results with first checkpoint")
                is_best = True
            elif self.objective.compare_strict(results, self.best_checkpoint_results):
                self.logger.info("New best checkpoint")
                is_best = True
            if is_best:
                self.best_checkpoint_results = self.get_results_dict()
        state_dict = self.state_dict(reason)
        self.model_file_manager.save_checkpoint(self.epoch, state_dict, is_abrupt, is_best)
        self.model_file_manager.flush()
        self.epoch_checkpoint = True
        self.logger.info(f"Checkpoint at Epoch {self.epoch}\n{self._metrics_str()}")
        find_illegal_children(state_dict, self.logger)

    def make_loader(self, dataset : Dataset) -> DataLoader:
        generator = torch.Generator(device=torch.get_default_device())
        return DataLoader(
            dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_loaders,
            shuffle=self.shuffle,
            generator=generator,
            worker_init_fn=_dataloader_worker_init_fn,
            pin_memory=True
        )

    def training_loader(self):
        if isinstance(self.training_set, SplitDataset):
            training_set = self.training_set.for_training()
        else:
            training_set = self.training_set
        return self.make_loader(training_set)

    def eval_metrics(self):
        if self.model_file_manager is None:
            raise ValueError("Model file manager not initialized")
        for metric_logger in self.metric_loggers:
            record = metric_logger.log_record(self.epoch, self.model)
            self.model_file_manager.write_metrics(metric_logger.identifier, record)

    def _train_epoch(self, epoch):
        if self.model_file_manager is None:
            raise ValueError("Model file manager not initialized")
        self.epoch = epoch
        self.epoch_checkpoint = False
        #logger.info(f"Epoch {epoch} - Starting")
        self.model.train()
        batches = 0
        update_metrics = lambda _, __ : None
        if self.train_logger is not None:
            self.train_logger.prepare_torch_metrics()
            update_metrics = self.train_logger.update_torch_metrics
        loss_sum = 0
        x = None
        y = None
        try:
            loader = self.training_loader()
            with self.progress_cm.track(f'Epoch {epoch}', 'batches', loader) as progress_tracker:
                for x, y in loader:
                    x = x.to(torch.get_default_device())
                    y = y.to(torch.get_default_device())
                    loss, pred = self.__train_step(x, y)
                    loss_sum += loss.item()
                    update_metrics(pred, y)
                    batches += 1
                    progress_tracker.tick()
            self.first_epoch = False
        except BaseException as e:
            self.logger.error(f"Exception during training loop {debug_model(e, self.model, x, y)}")
            raise
        
        try:
            self.eval_metrics()
        except BaseException as e:
            raise _EvalExceptionWrapper(e)
        if epoch > 0 and self.checkpoint_each is not None and epoch % self.checkpoint_each == 0:
            self.logger.info(f"Periodic checkpoint")
            self._checkpoint('periodic')
        elif any(trigger(self) for trigger in self.checkpoint_triggers):
            self.logger.info(f"Triggered checkpoint")
            self._checkpoint('triggered')
        elif self.first_epoch:
            self.logger.info(self._metrics_str())
        self.logger.info(f"Epoch {epoch} avg loss = {loss_sum / batches}\n\n\n")
            
        #for callback in self.callbacks:
        #    callback(self.model, self.optimizer, self.metrics, epoch)

    def __train_step(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        pred = self.model.forward(x)
        loss = self.loss_fn(pred, y)
        loss.backward()
        self.optimizer.step()
        if self.train_logger is not None:
            self.train_logger.update_loss(loss.detach())
        return loss, pred

    def try_get_validation_set(self) -> Dataset:
        if isinstance(self.training_set, SplitDataset):
            return self.training_set.for_validation()
        elif hasattr(self.training_set, 'dataset') and isinstance(self.training_set.dataset, SplitDataset): # type: ignore
            return self.training_set.dataset.for_validation() # type: ignore
        elif hasattr(self.training_set, 'for_validation'):
            return self.training_set.for_validation() # type: ignore
        else:
            return self.training_set

    def __repr__(self) -> str:
        fields = []
        for k, v in self.__dict__.items():
            if k == 'model_file_manager':
                continue
            if k.startswith('_'):
                continue
            if k == 'training_set':
                if isinstance(v, SplitDataset):
                    fields.append(f"{k} = {v}")
                elif hasattr(v, 'dataset'):
                    fields.append(f"{k} = {v.dataset}")
                else:
                    fields.append(f"{k} = {v.__class__.__name__}")
                continue
            fields.append(f"{k} = {v}")
        return f"Trainer(\n{',\n'.join(fields)}\n)"

    @classmethod
    def _get_build_module(cls, build_script : str):
        # Sanitation checks before loading the script
        assert utils.is_import_safe(build_script), f"Invalid build script name: {build_script}"
        return importlib.import_module('.' + build_script, 'models')

    @classmethod
    def _from_config_with_script(cls, config : TrainerConfig) -> tuple['Trainer', Any]:
        build_script = config['build_script']
        build_args = config.get('build_args', [])
        build_kwargs = config.get('build_kwargs', {})
        for key in config.keys():
            if key not in ['build_script', 'build_args', 'build_kwargs']:
                module_logger.warning(f"Unused key in config: {key}")
        module_logger.info(f"Creating trainer from script {build_script}")
        script = cls._get_build_module(build_script)
        trainer = script.create_trainer(*build_args, **build_kwargs)
        module_logger.info(trainer)
        return trainer, script

    @classmethod
    def from_config(cls, config : TrainerConfig) -> 'Trainer':
        trainer, _ = cls._from_config_with_script(config)
        return trainer

    @classmethod
    def load_checkpoint(
            cls, 
            file_manager : ModelFileManager,
            file : Path | None = None,
            prefer : Literal['last', 'best'] = 'last'
            ) -> 'Trainer':
        module_logger.info(f"Loading Model {file_manager.model_name}")
        config = file_manager.load_config()
        trainer, script = cls._from_config_with_script(config)
        trainer.init_file_manager(file_manager)
        module_logger.info(f"{trainer}")

        module_logger.info(f"Looking for checkpoints")
        checkpoint = file_manager.load_checkpoint(file, prefer)
        if checkpoint is not None:
            module_logger.info(f"Loading checkpoint")
            metadata = checkpoint['metadata']
            module_logger.info(f"Checkpoint info: {metadata}")
            trainer.load_state_dict(checkpoint)
        else:
            module_logger.info("No checkpoint found.")
            if hasattr(script, "init_weights"):
                module_logger.info("Calling 'init_weights' from script")
                script.init_weigths(trainer)
        return trainer
    

def find_illegal_children(state_dict, logger : logging.Logger | None = None) -> None | list[tuple[str, Any]]:
    def walk(obj, tree_trace : list[str], invalid_objects : list[tuple[str, Any]]):
        obj_type = type(obj)
        if obj_type is dict or obj_type is OrderedDict:
            for i, (k, v) in enumerate(obj.items()):
                tree_trace.append(f'.{k}')
                walk(v, tree_trace, invalid_objects)
                tree_trace.pop()
                tree_trace.append(f'.keys()[{i}]')
                walk(k, tree_trace, invalid_objects)
                tree_trace.pop()
        elif obj_type is list or obj_type is tuple:
            for i, v in enumerate(obj):
                tree_trace.append(f'[{i}]')
                walk(v, tree_trace, invalid_objects)
                tree_trace.pop()
        elif not (
            isinstance(obj, torch.Tensor) or
            obj_type is int or
            obj_type is float or
            obj_type is str or
            obj_type is bool or
            obj is None
        ):
            invalid_objects.append((''.join(tree_trace), obj))
    tree_trace_ = ['state_dict']
    invalid_objects_ : list[tuple[str, Any]] = []
    walk(state_dict, tree_trace_, invalid_objects_)
    if len(invalid_objects_) == 0:
        return None
    else:
        if logger is not None:
            logger.error(
                "Found illegal objects in state_dict:\n" + 
                '\n'.join(f"\t{path} : {repr(obj)[:30]} of type {type(obj)}" for path, obj in invalid_objects_) +
                "\ntorch.load() will required weights_only=False"
            )
        return invalid_objects_

