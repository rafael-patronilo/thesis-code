from typing import Optional, Any
from .storage_management.model_file_manager import ModelFileManager
import logging
from collections import deque
import torch
from .modules.metrics import select_metrics
import math
from logging_setup import tensors as tensor_logging

logger = logging.getLogger(__name__)

class MetricsLogger:
    def __init__(self, 
                identifier : str,
                model_file_manager : ModelFileManager,
                metric_functions : dict[str, Any] | list[str],
                dataloader,
                last_n_size = 10,
                tensorboard_writer = None,
                ):
        self.identifier = identifier
        self.__metric_functions : dict[str, Any]
        if type(metric_functions) is list:
            self.__metric_functions = select_metrics(metric_functions)
        else:
            self.__metric_functions = metric_functions # type: ignore
        self.last_record = None
        def sort_key(metric):
            metric_name = metric[0]
            return (0 if metric_name.lower() == 'loss' else 1, metric_name)
        self.ordered_metrics = sorted([(k, v) for k, v in self.__metric_functions.items()] , key=sort_key)
        self.metrics_header = ['epoch'] + [k for (k, _) in self.ordered_metrics]
        self.sums : dict = {k: 0.0 for k, _ in self.ordered_metrics}
        self.last_n = deque([{k: 0.0 for k, _ in self.ordered_metrics} for _ in range(last_n_size)])
        self.sums_last_n : dict = {k: 0.0 for k, _ in self.ordered_metrics}
        self.total_measures = {k: 0.0 for k, _ in self.ordered_metrics}
        self.buffered_records = []
        self.dataloader = dataloader
        self.tensorboard_writer = tensorboard_writer
        self.file_handle = model_file_manager.init_metrics_file(self.metrics_header, identifier)

    def state_dict(self):
        return {
            "sums": self.sums,
            "last_n" : list(self.last_n),
            "total_measures": self.total_measures
        }
    
    def load_state_dict(self, state_dict):
        sums : dict = state_dict["sums"]
        last_n = state_dict["last_n"]
        total_measures = state_dict["total_measures"]
        if sums.keys() != self.__metric_functions.keys():
            logger.error(f"Invalid sums keys ({sums.keys()}), ignoring value")
        else:
            self.sums = sums
        if total_measures.keys() != self.__metric_functions.keys():
            logger.error(f"Invalid total measures keys ({total_measures.keys()}), ignoring value")
        else:
            self.total_measures = total_measures
        if any(x.keys() != self.__metric_functions.keys() for x in last_n):
            logger.error(f"Invalid last n keys, ignoring value")
        else:
            self.last_n = deque(last_n)
            self.sums = {k: 0.0 for k, _ in self.ordered_metrics}
            for entry in last_n:
                for k, v in entry.items():
                    sums[k] += v

    def _safe_div(self, a, b):
        if b == 0:
            return float('nan')
        return a / b

    def averages(self):
        return {
            k : self._safe_div(v, self.total_measures[k]) 
            for k,v in self.sums.items()
        }
    
    def averages_last_n(self):
        return {
            k : self._safe_div(v, min(len(self.last_n), self.total_measures[k])) 
            for k,v in self.sums_last_n.items()
        }
    
    def flush(self):
        if len(self.buffered_records) > 0:
            self.file_handle.write(
                "\n".join(",".join(map(str, record)) for record in self.buffered_records) + "\n")
            self.buffered_records = []
            self.file_handle.flush()

    def _eval(self, model):
        model.eval()
        y_preds = []
        y_trues = []
        with torch.no_grad():
            for x, y_true in self.dataloader:
                y_preds.append(model(x))
                y_trues.append(y_true)
        y_preds = torch.cat(y_preds)
        y_trues = torch.cat(y_trues)
        
        # store predictions for debugging
        tensor_logging.log_table(f"{self.identifier}_preds.csv", y_preds= y_preds, y_trues=y_trues)
        return y_preds, y_trues
    

    def log(self, epoch, model):
        record = []
        record.append(epoch)
        self.last_record = {}
        discarded = self.last_n.pop()
        for k, v in discarded.items():
            if not math.isnan(v):
                self.sums_last_n[k] -= v
        self.last_n.appendleft({})
        y_pred, y_true = self._eval(model)
        for metric_name, metric_fn in self.ordered_metrics:
            value = metric_fn(epoch=epoch, y_pred=y_pred, y_true=y_true)
            record.append(value)
            self.last_record[metric_name] = value
            if not math.isnan(value):
                self.sums[metric_name] += value
                self.sums_last_n[metric_name] += value
                self.last_n[0][metric_name] = value
                self.total_measures[metric_name] += 1
                if self.tensorboard_writer is not None:
                    self.tensorboard_writer.add_scalar(f"{metric_name}/{self.identifier}", value, epoch)
        
        self.buffered_records.append(record)
        if len(self.buffered_records) >= 10:
            self.flush()
