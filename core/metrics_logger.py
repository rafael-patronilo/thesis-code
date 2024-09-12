from typing import NamedTuple, Any
from model_file_manager import ModelFileManager
import logging
from collections import deque

logger = logging.getLogger(__name__)

class MetricsLogger:
    def __init__(self, 
                model_file_manager : ModelFileManager,
                metric_functions : dict[str, Any],
                dataloader,
                last_n_size = 10):
        self.model_file_manager = model_file_manager
        self.__metric_functions = metric_functions
        
        self.ordered_metrics = sorted([(k, v) for k, v in metric_functions.items()] , key=lambda x: x[0])
        self.metrics_header = ['epoch', 'loss'] + [k for (k, _) in self.ordered_metrics]
        self.sums : dict = {k: 0.0 for k, _ in self.ordered_metrics}
        self.last_n = deque([{k: 0.0 for k, _ in self.ordered_metrics} for _ in range(last_n_size)])
        self.sums_last_n : dict = {k: 0.0 for k, _ in self.ordered_metrics}
        self.buffered_records = []
        self.dataloader = dataloader
        self.model_file_manager.init_metrics_file(self.metrics_header)

    def state_dict(self):
        return {
            "sums": self.sums,
            "last_n" : list(self.last_n),
        }
    
    def load_state_dict(self, state_dict):
        sums : dict = state_dict["sums"]
        last_n = state_dict["last_n"]
        if sums.keys() != self.__metric_functions.keys():
            logger.error(f"Invalid sums keys ({sums.keys()}), ignoring value")
        else:
            self.sums = sums
        if any(x.keys() != self.__metric_functions.keys() for x in last_n):
            logger.error(f"Invalid last n keys, ignoring value")
        else:
            self.last_n = deque(last_n)
            self.sums = {k: 0.0 for k, _ in self.ordered_metrics}
            for entry in last_n:
                for k, v in entry.items():
                    sums[k] += v

    def averages(self, epoch):
        return {k : v / epoch for k,v in self.sums.items()}
    
    def averages_last_n(self, epoch):
        return {k : v / min(epoch, len(self.last_n)) for k,v in self.sums_last_n.items()}
    
    def flush(self):
        if len(self.buffered_records) > 0:
            self.model_file_manager.append_metrics(
                "\n".join(",".join(map(str, record)) for record in self.buffered_records) + "\n")
            self.buffered_records = []
    
    def log(self, epoch, loss, y_pred, y_true):
        record = []
        record.append(epoch)
        record.append(loss)
        discarded = self.last_n.pop()
        for k, v in discarded:
            self.sums_last_n[k] -= v
        self.last_n.appendleft({})
        for metric_name, metric_fn in self.ordered_metrics:
            value = metric_fn(y_pred=y_pred, y_true=y_true)
            self.sums[metric_name] += value
            self.sums_last_n[metric_name] += value
            self.last_n[0][metric_name] = value
            record.append(value)
        
        self.buffered_records.append(record)
