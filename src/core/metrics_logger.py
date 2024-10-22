from typing import Optional, Any, Sequence, Callable
from .storage_management.model_file_manager import ModelFileManager
import logging
from collections import OrderedDict, deque
import torch
from torch.utils.data import DataLoader
from .modules.metrics import select_metrics, NamedMetricFunction, MetricFunction
import math
from core.util import debug, safe_div
from core.util.typing import TorchDataset

logger = logging.getLogger(__name__)

class MetricsLogger:
    def __init__(self, 
                identifier : str,
                metric_functions : dict[str, MetricFunction] | Sequence[NamedMetricFunction],
                dataset : Callable[[],TorchDataset],
                target_module : Optional[str] = None,
                last_n_size = 10
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
        self.dataset_ref = dataset
        self.dataloader = DataLoader(self.dataset_ref())
        self.target_module = target_module

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        del state["dataloader"]
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.dataloader = DataLoader(self.dataset_ref())

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
        if any(x.keys() != self.__metric_functions.keys() for x in last_n) or len(last_n) != len(self.last_n):
            logger.error(f"Invalid last n, ignoring value\nlast_n: {last_n}\n keys: {self.__metric_functions.keys()}")
        else:
            self.last_n = deque(last_n)
            self.sums = {k: 0.0 for k, _ in self.ordered_metrics}
            for entry in last_n:
                for k, v in entry.items():
                    sums[k] += v

    def averages(self):
        return {
            k : safe_div(v, self.total_measures[k]) 
            for k,v in self.sums.items()
        }
    
    def averages_last_n(self):
        return {
            k : safe_div(v, min(len(self.last_n), self.total_measures[k])) 
            for k,v in self.sums_last_n.items()
        }


    def get_target_module(self, model):
        if self.target_module is None:
            return model
        else:
            path = self.target_module.split(".")
            submodule = model
            for p in path:
                if len(p) == 0:
                    continue
                submodule = getattr(submodule, p)
            return submodule

    def _eval(self, model):
        model.eval()
        module = self.get_target_module(model)
        y_preds = []
        y_trues = []
        with torch.no_grad():
            for x, y_true in self.dataloader:
                y_preds.append(module(x))
                y_trues.append(y_true)
        y_preds = torch.cat(y_preds)
        y_trues = torch.cat(y_trues)
        
        # store predictions for debugging
        debug.debug_table(f"{self.identifier}_preds.csv", y_preds=y_preds, y_trues=y_trues)
        debug.debug_table(f"{self.identifier}_round_preds.csv", 
                          y_preds=lambda:torch.round(y_preds), 
                          y_trues=lambda:torch.round(y_trues))
        return y_preds, y_trues
    

    def log_record(self, epoch, model) -> OrderedDict[str, Any]:
        record = OrderedDict()
        record['epoch'] = epoch
        self.last_record = {}
        discarded = self.last_n.pop()
        for k, v in discarded.items():
            if not math.isnan(v):
                self.sums_last_n[k] -= v
        self.last_n.appendleft({})
        y_pred, y_true = self._eval(model)
        for metric_name, metric_fn in self.ordered_metrics:
            value = metric_fn(epoch=epoch, y_pred=y_pred, y_true=y_true)
            record[metric_name] = value
            self.last_record[metric_name] = value
            if not math.isnan(value):
                self.sums[metric_name] += value
                self.sums_last_n[metric_name] += value
                self.last_n[0][metric_name] = value
                self.total_measures[metric_name] += 1
        return record
    
    def __repr__(self):
        return f"MetricsLogger({self.identifier}, metrics=[{self.ordered_metrics}])"