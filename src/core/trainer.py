from collections import OrderedDict
import logging
from torch.utils.data import DataLoader, IterableDataset, Dataset, RandomSampler
from typing import Optional, Any, Callable, Literal, NamedTuple, TypedDict, TYPE_CHECKING
from .metrics_logger import MetricsLogger, TrainingLogger
from .datasets import SplitDataset
from .storage_management.model_file_manager import ModelFileManager
from .datasets import get_dataset
from logging_setup import NOTIFY, logfile

import os
import importlib
import torch
from . import modules
from . import util as utils
import time
if TYPE_CHECKING:
    from torch import nn
    from torch.optim.optimizer import Optimizer

# Deprecated TODO: Remove\Replace
ModelDetails = NamedTuple(
    "ModelDetails",
    [
        ("architecture", Any),
        ("optimizer", str),
        ("loss_fn", str),
        ("dataset", SplitDataset),
        ("metrics", list['MetricsLogger']),
        ("batch_size", int),
    ]
)

def _dataloader_worker_init_fn(worker_id):
    logger.debug(f"Training Dataloader worker {worker_id} initialized")
    torch.set_default_device('cpu')

def debug_model(error : BaseException, model : 'nn.Module', X, Y):
    sb = []
    sb.append("\nModel debug info")
    sb.append(f"\tError: {error}")
    sb.append(f"\tModel: {model}")
    if X is not None and Y is not None:
        sb.append(f"\tX shape: {X.shape}")
        sb.append(f"\tY shape: {Y.shape}")
        sb.append(f"\tX dtype: {X.dtype}")
        sb.append(f"\tY dtype: {Y.dtype}")
        sb.append(f"\tX device: {X.device}")
        sb.append(f"\tY device: {Y.device}")
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
        result = model.forward(X)
        sb.append(f"\tResult shape: {result.shape}")
        sb.append(f"\tExpected shape: {Y.shape}")
    except BaseException as e:
        sb.append(f"\tError during forward pass: {e}")
        pass
    return "\n".join(sb)

type MetricsSnapshot = dict[str, dict[str, Any]]

type CheckpointReason = Literal['interrupt', 'force_interrupt', "error", "eval_error", 'triggered', 'periodic', 'end']

ABRUPT_CHECKPOINT_REASONS = ['force_interrupt', "error"]

class TrainerConfig(TypedDict, total=True):
    build_script : str
    build_args : list[Any]
    build_kwargs : dict[str, Any]

logger = logging.getLogger(__name__)

LOG_STEP_EVERY = 5*60 #seconds

class _EvalExceptionWrapper(BaseException):
    def __init__(self, inner : BaseException):
        self.inner = inner


class Trainer:

    def __init__(
            self,
            model : 'nn.Module',
            loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            optimizer : Callable[[Any], 'Optimizer'],
            training_set : 'Dataset',
            metric_loggers : list[MetricsLogger],
            #callbacks = None,
            epoch : int = 0,
            batch_size : int = 32,
            num_loaders : int = int(os.getenv('NUM_THREADS', 4)),
            shuffle : bool = True,
            checkpoint_each : Optional[int] = 10,
            checkpoint_triggers : Optional[list[Callable[[Any], bool]]] = None,
            stop_criteria : Optional[list[Callable[['Trainer'], bool]]] = None
        ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer(model.parameters())
        self.training_set = training_set
        self.model_file_manager : Optional[ModelFileManager] = None
        self.metric_loggers : list[MetricsLogger] = metric_loggers
        self.train_logger = None
        _train_logger = [logger for logger in metric_loggers if isinstance(logger, TrainingLogger)]
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
        self._training_loader = None

    def checkpoint_metadata(self, reason : CheckpointReason):
        assert self.model_file_manager is not None
        return {
            "epoch" : self.epoch,
            "start_epoch" : self.start_epoch,
            "path" : self.model_file_manager.path,
            "reason" : reason,
            "device" : torch.get_default_device().type,
            "logfile" : logfile
        }
    
    def init_file_manager(self, model_file_manager : ModelFileManager) -> 'Trainer':
        self.model_file_manager = model_file_manager
        model_file_manager.init_directory()
        for metric_logger in self.metric_loggers:
            self.model_file_manager.init_metrics_file(metric_logger.identifier, metric_logger.metrics_header)
        return self

    
    def state_dict(self, reason : CheckpointReason):
        metadata = self.checkpoint_metadata(reason)
        logger.debug(f"Checkpoint metadata: {metadata}")
        stop_criteria = {
            criterion.__class__.__name__ : criterion.state_dict() 
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
                criterion.load_state_dict(state_dict['stop_criteria'][criterion.__class__.__name__])
        metadata = state_dict['metadata']
        logger.info(f"Loaded checkpoint: {utils.multiline_str(metadata)}")
        if metadata['reason'] in ABRUPT_CHECKPOINT_REASONS:
            logger.warning(f"Abrupt checkpoint: Checkpoint was saved due to {metadata['reason']}")
        elif metadata['reason'] == 'eval_error':
            logger.warning(f"Checkpoint was saved due to evaluation error. Attempting to evaluate again...")
            self.eval_metrics()
            if self.model_file_manager is not None:
                self.model_file_manager.flush()
            logger.info(self._metrics_str())
        self.epoch += 1

    def train_indefinitely(self):
        self.train_until([])

    def train_until_epoch(self, end_epoch):
        self.train_until([lambda x : x.epoch >= end_epoch])

    def train_epochs(self, num_epochs):
        self.train_until_epoch(self.epoch + num_epochs)

    def metrics_snapshot(self) -> MetricsSnapshot:
        snapshot = {}
        for metric_logger in self.metric_loggers:
            snapshot[metric_logger.identifier] = metric_logger.last_record
        return snapshot

    def train_until(self, criteria : list[Callable[['Trainer'], bool]]):
        if self.model_file_manager is None:
            raise ValueError("Model file manager not initialized")
        criteria = criteria + self.stop_criteria
        logger.info("Initiating training loop...\n"
                    f"Model: {self.model_file_manager.model_name}\n"
                    f"Epoch: {self.epoch}\n"
                    f"Checkpoint each: {self.checkpoint_each}\n"
                    f"Keyboard interrupt to save checkpoint and exit.")
        
        try:
            first = True
            no_interrupt = utils.NoInterrupt("mid epoch", logger)
            while not any(criterion(self) for criterion in criteria):
                with no_interrupt:
                    if not first:
                        self.epoch += 1
                    self._train_epoch(self.epoch, first=first)
                    first = False
            self.epoch += 1
            self._checkpoint('end')
            logger.log(NOTIFY, "Training complete.")
        except (KeyboardInterrupt, utils.NoInterrupt.InterruptException):
            logger.log(NOTIFY, "Training safely interrupted. Saving checkpoint...")
            self._checkpoint('interrupt')
            raise
        except (SystemExit, utils.NoInterrupt.ForcedInterruptException):
            logger.error("Forced interrupt. Saving checkpoint...")
            self._checkpoint('force_interrupt')
            raise
        except _EvalExceptionWrapper as e:
            logger.error(f"Exception caught during evaluation. Saving checkpoint...")
            self._checkpoint('eval_error')
            raise e.inner
        except:
            if self.epoch == 0:
                logger.error("Exception caught at epoch 0. Skipping checkpoint...")
            else:
                logger.error("Exception caught while training. Saving checkpoint...")
                self._checkpoint('error')
            raise
    
    def _metrics_str(self):
        averages = ""
        averages_last_n = ""
        last_record = ""
        for metric_logger in self.metric_loggers:
            averages += f"\t{metric_logger.identifier} : {metric_logger.averages()}\n"
            averages_last_n += f"\t{metric_logger.identifier} : {metric_logger.averages_last_n()}\n"
            last_record += f"\t{metric_logger.identifier} : {metric_logger.last_record}\n"
        return (
            f"Metrics:\n{last_record}"
            f"Avg:\n{averages}"
            f"Avg last {len(self.metric_loggers[0].last_n)}:\n{averages_last_n}"
            )

    def _checkpoint(self, reason : CheckpointReason):
        if self.model_file_manager is None:
            raise ValueError("Model file manager not initialized")
        is_abrupt = reason in ABRUPT_CHECKPOINT_REASONS
        self.model_file_manager.save_checkpoint(self.epoch, self.state_dict(reason), is_abrupt)
        self.model_file_manager.flush()
        logger.info(f"Checkpoint at Epoch {self.epoch}\n{self._metrics_str()}")
        
    def training_loader(self):
        if self._training_loader is None:
            generator = torch.Generator(device=torch.get_default_device())
            self._training_loader = DataLoader(
                self.training_set, 
                batch_size=self.batch_size,
                num_workers=self.num_loaders,
                shuffle=self.shuffle,
                generator=generator,
                worker_init_fn=_dataloader_worker_init_fn,
                pin_memory=True
            )
        return self._training_loader
    
    def eval_metrics(self):
        if self.model_file_manager is None:
            raise ValueError("Model file manager not initialized")
        for metric_logger in self.metric_loggers:
            record = metric_logger.log_record(self.epoch, self.model)
            self.model_file_manager.write_metrics(metric_logger.identifier, record)

    def _train_epoch(self, epoch, first = False):
        if self.model_file_manager is None:
            raise ValueError("Model file manager not initialized")
        self.epoch = epoch
        logger.info(f"Epoch {epoch} - Starting")
        self.model.train()
        batches = 0
        epoch_start_time = time.time()
        last_log = epoch_start_time
        update_metrics = lambda _, __ : None
        if self.train_logger is not None:
            self.train_logger._prepare_torch_metrics()
            update_metrics = self.train_logger._update_torch_metrics
        try:
            X = None
            Y = None
            for X, Y in self.training_loader():
                X = X.to(torch.get_default_device())
                Y = Y.to(torch.get_default_device())
                _, pred = self.__train_step(X, Y)
                update_metrics(pred, Y)
                batches += 1
                now = time.time()
                if now - last_log > LOG_STEP_EVERY:
                    logger.info(f"Epoch {epoch} - \t\tRunning longer than {LOG_STEP_EVERY} seconds ({now - epoch_start_time:.0f}), current batch = {batches}")
                    last_log = now
        except BaseException as e:
            logger.error(f"Exception during training loop {debug_model(e, self.model, X, Y)}")
            raise
        logger.info(f"Epoch {epoch} - Training complete ({time.time() - epoch_start_time:.0f} seconds)")
        try:
            self.eval_metrics()
        except BaseException as e:
            raise _EvalExceptionWrapper(e)
        if epoch > 0 and self.checkpoint_each is not None and epoch % self.checkpoint_each == 0:
            logger.info(f"Periodic checkpoint")
            self._checkpoint('periodic')
        elif any(trigger(self) for trigger in self.checkpoint_triggers):
            logger.info(f"Triggered checkpoint")
            self._checkpoint('triggered')
        elif first:
            logger.info(self._metrics_str())

            
        #for callback in self.callbacks:
        #    callback(self.model, self.optimizer, self.metrics, epoch)

    def __repr__(self) -> str:
        fields = []
        for k, v in self.__dict__.items():
            if k == 'model_file_manager':
                continue
            if k.startswith('_'):
                continue
            if k == 'training_set':
                if hasattr(v, 'dataset'):
                    fields.append(f"{k} = {v.dataset}")
                else:
                    fields.append(f"{k} = {v.__class__.__name__}")
                continue
            fields.append(f"{k} = {v}")
        return f"Trainer(\n{',\n'.join(fields)}\n)"

    def __train_step(self, X, Y):
        self.model.train()
        self.optimizer.zero_grad()
        pred = self.model.forward(X)
        loss = self.loss_fn(pred, Y)
        loss.backward()
        self.optimizer.step()
        return loss, pred


    @classmethod
    def from_config(cls, config : TrainerConfig) -> 'Trainer':
        build_script = config['build_script']
        build_args = config.get('build_args', [])
        build_kwargs = config.get('build_kwargs', {})
        for key in config.keys():
            if key not in ['build_script', 'build_args', 'build_kwargs']:
                logger.warning(f"Unused key in config: {key}")
        logger.info(f"Creating trainer from script {build_script}")
        # Sanitazion checks before loading the script
        assert utils.is_import_safe(build_script), f"Invalid build script name: {build_script}"
        trainer = importlib.import_module('.' + build_script, 'models').create_trainer(*build_args, **build_kwargs)
        logger.info(trainer)
        return trainer

    @classmethod
    def load_checkpoint(
            cls, 
            file_manager : ModelFileManager
            ) -> 'Trainer':
        logger.info(f"Loading Model {file_manager.model_name}")
        config = file_manager.load_config()
        trainer = cls.from_config(config)
        trainer.init_file_manager(file_manager)
        logger.info(f"{trainer}")

        logger.info(f"Looking for checkpoints")
        checkpoint = file_manager.load_last_checkpoint()
        if checkpoint is not None:
            logger.info(f"Loading checkpoint")
            metadata = checkpoint['metadata']
            logger.info(f"Checkpoint info: {metadata}")
            trainer.load_state_dict(checkpoint)
        else:
            logger.info("No checkpoint found.")

        return trainer
