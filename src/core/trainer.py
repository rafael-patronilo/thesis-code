import logging
from torch.utils.data import DataLoader
from typing import Optional, Any, Callable, Literal, NamedTuple, TypedDict, TYPE_CHECKING
from .metrics_logger import MetricsLogger
from .datasets import SplitDataset
from .storage_management.model_file_manager import ModelFileManager
from .datasets import get_dataset
from logging_setup import NOTIFY, logfile
import re
import importlib
import torch
from . import modules
from . import util as utils
if TYPE_CHECKING:
    from torch import nn
    from torch.optim.optimizer import Optimizer
    from torch.utils.data import Dataset

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

type MetricsSnapshot = dict[str, dict[str, Any]]

type CheckpointReason = Literal['interrupt', 'force_interrupt', "error", 'triggered', 'periodic', 'end']
class TrainerConfig(TypedDict, total=True):
    build_script : str
    build_args : list[Any]
    build_kwargs : dict[str, Any]

logger = logging.getLogger(__name__)

class Trainer:

    def __init__(
            self,
            model : 'nn.Module',
            loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            optimizer : Callable[[Any], 'Optimizer'],
            training_set : 'Dataset',
            metric_loggers : list[MetricsLogger],
            #callbacks = None,
            epoch = 0,
            checkpoint_each : Optional[int] = 10,
            checkpoint_triggers : Optional[list[Callable[[Any], bool]]] = None,
        ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer(model.parameters())
        self.training_set = training_set
        self.model_file_manager : Optional[ModelFileManager] = None
        self.metric_loggers : list[MetricsLogger] = metric_loggers
        #self.callbacks = callbacks or {}
        self.start_epoch = epoch
        self.epoch = epoch
        self.checkpoint_triggers = checkpoint_triggers or []
        self.checkpoint_each = checkpoint_each

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
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "metrics": {metric_logger.identifier : metric_logger.state_dict() for metric_logger in self.metric_loggers},
            "metadata": metadata
        }
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.epoch = state_dict["epoch"]
        self.start_epoch = state_dict["epoch"]
        for metric_logger in self.metric_loggers:
            metric_logger.load_state_dict(state_dict["metrics"][metric_logger.identifier])
        logger.info(f"Loaded checkpoint: {utils.multiline_str(state_dict['metadata'])}")

    def train_indefinitely(self):
        self.train_until([])

    def train_until_epoch(self, end_epoch):
        self.train_until([lambda x : x.epoch >= end_epoch])

    def train_epochs(self, num_epochs):
        self.train_until_epoch(self.epoch + num_epochs)

    def metrics_snapshot(self) -> MetricsSnapshot:
        snapshot = {}
        for metric_logger in self.metric_loggers:
            if metric_logger.last_record is not None:
                snapshot[metric_logger.identifier] = metric_logger.last_record
        return snapshot

    def train_until(self, criteria : list[Callable[['Trainer'], bool]]):
        if self.model_file_manager is None:
            raise ValueError("Model file manager not initialized")
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
            self._checkpoint('end')
            logger.log(NOTIFY, "Training complete.")
        except (KeyboardInterrupt, utils.NoInterrupt.InterruptException):
            logger.log(NOTIFY, "Training safely interrupted. Saving checkpoint...")
            self._checkpoint('interrupt')
        except (SystemExit, utils.NoInterrupt.ForcedInterruptException):
            logger.error("Forced interrupt. Saving checkpoint...")
            self._checkpoint('force_interrupt')
        except:
            logger.error("Exception caught while training. Saving checkpoint...")
            self._checkpoint('error')
            raise
    
    

    def _checkpoint(self, reason : CheckpointReason):
        if self.model_file_manager is None:
            raise ValueError("Model file manager not initialized")
        is_abrupt = reason in ['force_interrupt', "error"]
        self.model_file_manager.save_checkpoint(self.epoch, self.state_dict(reason), is_abrupt)
        self.model_file_manager.flush()
        averages = ""
        averages_last_n = ""
        last_record = ""
        for metric_logger in self.metric_loggers:
            averages += f"\t{metric_logger.identifier} : {metric_logger.averages()}\n"
            averages_last_n += f"\t{metric_logger.identifier} : {metric_logger.averages_last_n()}\n"
            last_record += f"\t{metric_logger.identifier} : {metric_logger.last_record}\n"
        logger.info(
            f"Checkpoint at Epoch {self.epoch}\n" 
            f"Metrics:\n{last_record}"
            f"Avg:\n{averages}"
            f"Avg last {len(self.metric_loggers[0].last_n)}:\n{averages_last_n}"
            )
    
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
        logger.info(f"Epoch {epoch}")
        self.model.train()
        batches = 0
        for X, Y in self.training_set:
            self.__train_step(X, Y)
            batches += 1
        logger.debug("Epoch complete")
        self.eval_metrics()
        if epoch > 0 and self.checkpoint_each is not None and epoch % self.checkpoint_each == 0:
            logger.info(f"Periodic checkpoint")
            self._checkpoint('periodic')
        elif any(trigger(self) for trigger in self.checkpoint_triggers):
            logger.info(f"Triggered checkpoint")
            self._checkpoint('triggered')
        elif first:
            metrics = {metric_logger.identifier : metric_logger.last_record for metric_logger in self.metric_loggers}
            logger.info(f"Metrics: {metrics}")

            
        #for callback in self.callbacks:
        #    callback(self.model, self.optimizer, self.metrics, epoch)

    def __repr__(self) -> str:
        fields = []
        for k, v in self.__dict__.items():
            if k == 'model_file_manager':
                continue
            if k == 'training_set' and hasattr(v, 'dataset'):
                fields.append(f"{k} = {v.dataset}")
                continue
            fields.append(f"{k} = {v}")
        return f"Trainer(\n{',\n'.join(fields)}\n)"

    def __train_step(self, X, Y):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model.forward(X), Y)
        loss.backward()
        self.optimizer.step()
        return loss


    @classmethod
    def from_config(cls, config : TrainerConfig) -> 'Trainer':
        build_script = config['build_script']
        build_args = config.get('build_args', [])
        build_kwargs = config.get('build_kwargs', {})
        logger.info(f"Creating trainer from script {build_script}")
        # Sanitazion checks before loading the script
        assert utils.is_import_safe(build_script), f"Invalid build script name: {build_script}"
        trainer = importlib.import_module('.' + build_script, 'models').create_trainer(*build_args, **build_kwargs)
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
            trainer.load_state_dict(checkpoint)
        else:
            logger.info("No checkpoint found.")

        return trainer
