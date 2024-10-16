import logging
from torch.utils.data import DataLoader
from typing import Optional, Any, Callable, Literal, NamedTuple
from .metrics_logger import MetricsLogger
from .datasets import SplitDataset
from .storage_management.model_file_manager import ModelFileManager
from .datasets import get_dataset
from logging_setup import NOTIFY, logfile
import torch
from . import modules
from . import util as utils

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

CheckpointReason = Literal['interrupt', 'force_interrupt', "error", 'triggered', 'periodic', 'end']

logger = logging.getLogger(__name__)

class Trainer:

    def __init__(
            self,
            model,
            loss_fn,
            optimizer,
            training_loader,
            model_file_manager : ModelFileManager,
            metric_loggers : list[MetricsLogger],
            #callbacks = None,
            epoch = 0,
            checkpoint_each : Optional[int] = 10,
            checkpoint_triggers : Optional[list[Callable[[Any], bool]]] = None
        ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.training_loader = training_loader
        self.model_file_manager = model_file_manager
        self.metric_loggers : list[MetricsLogger] = metric_loggers
        #self.callbacks = callbacks or {}
        self.start_epoch = epoch
        self.epoch = epoch
        self.checkpoint_triggers = checkpoint_triggers or []
        self.checkpoint_each = checkpoint_each

    def checkpoint_metadata(self, reason : CheckpointReason):
        return {
            "epoch" : self.epoch,
            "start_epoch" : self.start_epoch,
            "path" : self.model_file_manager.path,
            "reason" : reason,
            "device" : torch.get_default_device().type,
            "logfile" : logfile
        }
    
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

    def train_epochs(self, num_epochs):
        self.train_until([lambda x : x.epoch >= num_epochs])

        
    def train_until(self, criteria : list[Callable[['Trainer'], bool]]):
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
        self.model_file_manager.save_checkpoint(self.epoch, self.state_dict(reason))
        averages = ""
        averages_last_n = ""
        last_record = ""
        for metric_logger in self.metric_loggers:
            metric_logger.flush()
            averages += f"\t{metric_logger.identifier} : {metric_logger.averages()}\n"
            averages_last_n += f"\t{metric_logger.identifier} : {metric_logger.averages_last_n()}\n"
            last_record += f"\t{metric_logger.identifier} : {metric_logger.last_record}\n"
        logger.info(
            f"Checkpoint at Epoch {self.epoch}\n" 
            f"Metrics:\n{last_record}"
            f"Avg:\n{averages}"
            f"Avg last {len(self.metric_loggers[0].last_n)}:\n{averages_last_n}"
            )
    
    def _train_epoch(self, epoch, first = False):
        self.epoch = epoch
        logger.info(f"Epoch {epoch}")
        self.model.train()
        batches = 0
        for X, Y in self.training_loader:
            self.__train_step(X, Y)
            batches += 1
        logger.debug("Epoch complete")
        for metric_logger in self.metric_loggers:
            metric_logger.log(epoch, self.model)
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


    def __train_step(self, X, Y):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model.forward(X), Y)
        loss.backward()
        self.optimizer.step()
        return loss


    @classmethod
    def load_checkpoint(
            cls, 
            file_manager : ModelFileManager
            ):
        logger.info(f"Loading Model {file_manager.model_name} from checkpoint...")
        model_details = file_manager.load_model_details()
        logger.info(f"Model details: {model_details}")
        logger.debug(f"Loading architecture")
        model = model_details.architecture

        loss_fn = modules.get_loss_function(model_details.loss_fn)
        optimizer = modules.get_optimizer(model_details.optimizer, model)
        metric_loggers = model_details.metrics

        logger.info(f"Loading dataset {model_details.dataset}")
        batch_size = model_details.batch_size
        dataset = model_details.dataset

        training_loader = DataLoader(
            dataset.for_training(),
            batch_size=batch_size,
            shuffle=True
        )

        trainer = cls(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            training_loader=training_loader,
            model_file_manager=file_manager,
            metric_loggers=metric_loggers
        )

        checkpoint = file_manager.load_last_checkpoint()
        if checkpoint is not None:
            trainer.load_state_dict(checkpoint)

        return trainer
