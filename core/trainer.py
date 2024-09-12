import logging
import typing
import torch
from typing import Optional
import itertools
from metrics_logger import MetricsLogger
from model_file_manager import ModelFileManager
from log_setup import NOTIFY

logger = logging.getLogger(__name__)

class Trainer:

    def __init__(
            self,
            model,
            loss_fn,
            optimizer,
            training_loader,
            validation_loader,
            device,
            model_file_manager : ModelFileManager,
            metrics : MetricsLogger,
            #callbacks = None,
            epoch = 0,
            checkpoint_each = 10
        ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.model_file_manager = model_file_manager
        self.metrics = metrics
        #self.callbacks = callbacks or {}
        self.epoch = epoch
        self.checkpoint_each = checkpoint_each

    
    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "metrics": self.metrics.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.epoch = state_dict["epoch"]
        self.metrics.load_state_dict(state_dict["metrics"])
    
    def train(self, epochs):
        self.train_until(lambda self: self.epoch >= epochs)
        
    def train_until(self, criterion):
        logger.info("Training model until criterion is met")
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
        loss_sum = 0
        batches = 0
        for batch in self.training_loader:
            loss_sum += self.__train_step(batch)
            batches += 1
        loss = loss_sum / batches
        y_pred, y_true = self.__eval_epoch()
        self.metrics.log(epoch, loss, y_pred=y_pred, y_true=y_true)
        if epoch % self.checkpoint_each == 0:
            self.model_file_manager.save_checkpoint(epoch, self.state_dict())
            logger.info(
                f"Epoch {epoch}\n" 
                f"    Avg: {self.metrics.averages(epoch)}\n"
                f"    Avg last {len(self.metrics.last_n)}: {self.metrics.averages_last_n(epoch)}"
            )

            self.metrics.flush()
        #for callback in self.callbacks:
        #    callback(self.model, self.optimizer, self.metrics, epoch)

    def __eval_epoch(self):
        self.model.eval()
        y_pred = []
        y_true = []
        for X, Y in self.validation_loader:
            with torch.no_grad():
                y_pred.append(self.model(X))
                y_true.append(Y)
        return torch.cat(y_pred), torch.cat(y_true)


    def __train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model(batch), batch)
        loss.backward()
        self.optimizer.step()
        return loss


    @classmethod
    def load_checkpoint(
            cls, 
            file_manager : ModelFileManager,
            loss_fn,
            metrics : MetricsLogger, 
            training_loader, 
            validation_loader, 
            device
            ):
        model_details = file_manager.load_model_details()
        model = model_details.architecture
        optimizer = model_details.optimizer
        trainer = cls(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            training_loader=training_loader,
            validation_loader=validation_loader,
            device=device,
            model_file_manager=file_manager,
            metrics=metrics
        )
        
        checkpoint = file_manager.load_last_checkpoint()
        trainer.load_state_dict(checkpoint)

        return trainer