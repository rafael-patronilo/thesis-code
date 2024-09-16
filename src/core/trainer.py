import logging
import typing
import torch
from torch.utils.data import DataLoader
from typing import Optional, Any
import itertools
from metrics_logger import MetricsLogger
from model_file_manager import ModelFileManager
from core.datasets import dataset_registry
from log_setup import NOTIFY
import modules
from . import ModelDetails

logger = logging.getLogger(__name__)

class Trainer:

    def __init__(
            self,
            model,
            loss_fn,
            optimizer,
            training_loader,
            device,
            model_file_manager : ModelFileManager,
            metric_loggers : list[MetricsLogger],
            #callbacks = None,
            epoch = 0,
            checkpoint_each = 10
        ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.training_loader = training_loader
        self.model_file_manager = model_file_manager
        self.metric_loggers = metric_loggers
        #self.callbacks = callbacks or {}
        self.epoch = epoch
        self.checkpoint_each = checkpoint_each

    
    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "metrics": {metric_logger.identifier : metric_logger.state_dict() for metric_logger in self.metric_loggers}
        }
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.epoch = state_dict["epoch"]
        for metric_logger in self.metric_loggers:
            metric_logger.load_state_dict(state_dict["metrics"][metric_logger.identifier])
    
    def train(self, epochs):
        logger.info(f"Training for {epochs} epochs")
        self.train_until(lambda self: self.epoch >= epochs)
        
    def train_until(self, criterion):
        logger.info("Initiating training loop...\n"
                    f"    Model: {self.model_file_manager.model_name}\n"
                    f"    Epoch: {self.epoch}\n"
                    f"    Checkpoint each: {self.checkpoint_each}\n"
                    f"    Keyboard interrupt to save checkpoint and exit.")
        try:
            while not criterion(self):
                self.__train_epoch(self.epoch)
                self.epoch += 1
            self.model_file_manager.save_checkpoint(self.epoch, self.state_dict())
            logger.log(NOTIFY, "Training complete.")
        except KeyboardInterrupt | SystemExit:
            logger.log(NOTIFY, "Training interrupted. Saving checkpoint...")
            self.model_file_manager.save_checkpoint(self.epoch, self.state_dict())
        except Exception as e:
            logger.error("Exception caught while training. Saving checkpoint...")
            self.model_file_manager.save_checkpoint(self.epoch, self.state_dict())
            raise e
    
    def __train_epoch(self, epoch):
        self.epoch = epoch
        self.model.train()
        batches = 0
        for X, Y in self.training_loader:
            self.__train_step(X, Y)
            batches += 1
        for metric_logger in self.metric_loggers:
            metric_logger.log(epoch, self.model)
        if epoch % self.checkpoint_each == 0:
            self.model_file_manager.save_checkpoint(epoch, self.state_dict())
            averages = {}
            averages_last_n = {}
            for metric_logger in self.metric_loggers:
                metric_logger.flush()
                averages[metric_logger.identifier] = metric_logger.averages(epoch)
                averages_last_n[metric_logger.identifier] = metric_logger.averages_last_n(epoch)
            logger.info(
                f"Epoch {epoch}\n" 
                f"    Avg: {averages}\n"
                f"    Avg last {averages_last_n}"
            )
        else:
            logger.info(f"Epoch {epoch}")

            
        #for callback in self.callbacks:
        #    callback(self.model, self.optimizer, self.metrics, epoch)


    def __train_step(self, X, Y):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model(X), Y)
        loss.backward()
        self.optimizer.step()
        return loss


    @classmethod
    def load_checkpoint(
            cls, 
            file_manager : ModelFileManager,
            device
            ):
        logger.info(f"Loading Model {file_manager.model_name} from checkpoint...")
        model_details = file_manager.load_model_details()
        logger.info(f"Model details: {model_details}")
        logger.debug(f"Loading architecture")
        model = model_details.architecture.to(device)

        loss_fn = modules.get_loss_function(model_details.loss_fn)
        optimizer = modules.get_optimizer(model_details.optimizer, model)

        logger.info(f"Loading dataset {model_details.dataset}")
        batch_size = model_details.batch_size
        dataset = model_details.dataset
        dataset = dataset_registry[dataset]

        training_loader = DataLoader(
            dataset.for_training(),
            batch_size=batch_size,
            pin_memory=True,
            shuffle=True,
            pin_memory_device=device)
        
        validation_loader = DataLoader(
            dataset.for_validation(),
            batch_size=batch_size,
            pin_memory=True,
            shuffle=True,
            pin_memory_device=device)
        
        val_metrics = model_details.metrics
        train_metrics = model_details.metrics
        if model_details.train_metrics is not None:
            train_metrics = model_details.train_metrics
        
        train_metrics = modules.metrics.select_metrics(train_metrics)
        val_metrics = modules.metrics.select_metrics(val_metrics)
        train_metrics['loss'] = loss_fn
        val_metrics['loss'] = loss_fn

        trainer = cls(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            training_loader=training_loader,
            device=device,
            model_file_manager=file_manager,
            metric_loggers=[
                MetricsLogger('train', file_manager, train_metrics, dataloader=training_loader),
                MetricsLogger('val', file_manager, model_details.metrics, dataloader=validation_loader),
            ]
        )

        checkpoint = file_manager.load_last_checkpoint()
        if checkpoint is not None:
            trainer.load_state_dict(checkpoint)

        return trainer
