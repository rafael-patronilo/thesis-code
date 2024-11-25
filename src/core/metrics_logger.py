from typing import Literal, Optional, Any, Sequence, Callable
from .storage_management.model_file_manager import ModelFileManager
import logging
from collections import OrderedDict, deque
import torch
from torch.utils.data import DataLoader
from .metrics import select_metrics, NamedMetricFunction, MetricFunction
import time
import math
from core.util import debug, safe_div
from core.util.typing import TorchDataset
from torcheval.metrics import Metric as TorchMetric
import os

logger = logging.getLogger(__name__)
def dataloader_worker_init_fn(worker_id):
    logger.debug(f"Evaluation dataloader worker {worker_id} initialized")
    torch.set_default_device('cpu')

LOG_STEP_EVERY = 5*60 # 5 minutes


class MetricsLogger:
    def __init__(self, 
                identifier : str,
                metric_functions : dict[str, TorchMetric],
                dataset : Callable[[],TorchDataset],
                target_module : Optional[str] = None,
                last_n_size = 10,
                batch_size = 64,
                num_loaders : int = int(os.getenv('NUM_THREADS', 4)),
                ):
        #TODO deprecated checks in this constructor
        self.identifier = identifier
        self.__metric_functions : dict[str, Any]
        if type(metric_functions) is list:
            self.__metric_functions = select_metrics(metric_functions)
        else:
            self.__metric_functions = metric_functions # type: ignore
        self.torch_metrics = {k: v for k, v in self.__metric_functions.items() if isinstance(v, TorchMetric)}
        self.last_record = None
        def sort_key(metric):
            metric_name = metric[0]
            return (0 if metric_name.lower() == 'loss' else 1, metric_name)
        self.ordered_metrics : list[tuple[str, Any]] = sorted(
            [(k, v) for k, v in self.__metric_functions.items()] , key=sort_key
        )
        self.metrics_header = ['epoch'] + [k for (k, _) in self.ordered_metrics]
        self.sums : dict = {k: 0.0 for k, _ in self.ordered_metrics}
        self.last_n : deque[OrderedDict] = deque(OrderedDict((k, 0.0) for k in self.metrics_header) for _ in range(last_n_size))
        self.sums_last_n : dict = {k: 0.0 for k, _ in self.ordered_metrics}
        self.total_measures = {k: 0.0 for k, _ in self.ordered_metrics}
        self.dataset_ref = dataset
        if dataset is not None:
            self.dataloader = DataLoader(
                self.dataset_ref(), 
                batch_size=batch_size, 
                num_workers=num_loaders,
                worker_init_fn=dataloader_worker_init_fn,
                pin_memory=True
            )
        self.target_module = target_module


    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        del state["dataloader"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def state_dict(self):
        return {
            "sums": self.sums,
            "last_n" : list(self.last_n),
            "total_measures": self.total_measures
        }
    
    def _load_last_n(self, last_n):
        sb = []
        fatal_error = False
        if len(last_n) > len(self.last_n):
            sb.append(f"Invalid last_n length ({len(last_n)}), ignoring last {len(last_n) - len(self.last_n)} records")
            last_n = last_n[:len(self.last_n)]
            return
        new_last_n = deque(maxlen=len(self.last_n))
        new_sums_last_n = {k: 0.0 for k, _ in self.ordered_metrics}
        for i, record in enumerate(last_n):
            new_record = OrderedDict()
            for key in self.metrics_header:
                if key not in record:
                    sb.append(f"Missing key {key} in record {i}, adding NaN")
                    new_record[key] = float('nan')
                else:
                    new_record[key] = record[key]
                    if key != 'epoch':
                        new_sums_last_n[key] += record[key]
            for key, value in record.items():
                if key not in new_record:
                    sb.append(f"Ivalid {key} in record {i} with value {value}, ignoring")
            new_last_n.append(new_record)
        if len(new_last_n) < len(self.last_n):
            sb.append(f"Invalid last_n length ({len(new_last_n)}), appending 0s")
            for _ in range(len(new_last_n), len(self.last_n)):
                new_last_n.append(OrderedDict((k, 0.0) for k, in self.metrics_header))
        if len(new_last_n) != len(self.last_n):
            sb.append(f"FATAL: Invalid last_n length (got {len(new_last_n)}, expected {len(self.last_n)}) after all corrections")
            fatal_error = True
        if fatal_error:
            logger.error("Fatal error while trying to load last n records from checkpoint:\n"
                         + "\n".join(f"\t{m}" for m in sb)
                         + "\nLast n will be reset")
        else:
            self.last_n = new_last_n
            self.sums_last_n = new_sums_last_n
            self.last_record = self.last_n[0]
            if len(sb) > 0:
                logger.warning("Errors while trying to load last n records from checkpoint.\n"
                               + "\n".join(f"\t{m}" for m in sb)
                               + "\nLast n was corrected")

    def load_state_dict(self, state_dict):
        sums : dict = state_dict["sums"]
        last_n : list[dict] = state_dict["last_n"]
        total_measures = state_dict["total_measures"]
        if sums.keys() != self.__metric_functions.keys():
            logger.error(f"Invalid sums keys ({sums.keys()}), ignoring value")
        else:
            self.sums = sums
        if total_measures.keys() != self.__metric_functions.keys():
            logger.error(f"Invalid total measures keys ({total_measures.keys()}), ignoring value")
        else:
            self.total_measures = total_measures
        self._load_last_n(last_n)

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

    def _prepare_torch_metrics(self):
        for metric_fn in self.torch_metrics.values():
            metric_fn.to(device=torch.get_default_device())
            metric_fn.reset()

    def _update_torch_metrics(self, y_pred, y_true):
        y_pred = torch.flatten(y_pred, start_dim=1)
        y_true = torch.flatten(y_true, start_dim=1)
        for metric_fn in self.torch_metrics.values():
            metric_fn.update(y_pred, y_true)

    def _eval(self, model):
        if hasattr(self, '_evaluated_externally') and self._evaluated_externally: # type: ignore
            return
        model.eval()
        module = self.get_target_module(model)
        self._prepare_torch_metrics()
        y_pred = None
        y_true = None
        #logger.info(f"Producing predictions for metrics logger {self.identifier}")
        start_time = time.time()
        last_log = start_time
        batches = 0
        with torch.no_grad():
            for x, y_true in self.dataloader:
                x = x.to(torch.get_default_device())
                y_true = y_true.to(torch.get_default_device())
                y_pred = module(x)
                self._update_torch_metrics(y_pred, y_true)
                now = time.time()
                if now - last_log > LOG_STEP_EVERY:
                    logger.info(f"\t\tEvaluating longer than {LOG_STEP_EVERY} seconds ({now - start_time:.0f}), current batch = {batches}")
                    last_log = now
                batches += 1
        if y_pred is not None and y_true is not None and len(y_pred.shape) <= 2:
            # store predictions for debugging
            debug.debug_table(f"{self.identifier}_preds.csv", y_preds=y_pred, y_trues=y_true)
            debug.debug_table(f"{self.identifier}_round_preds.csv", 
                          y_preds=lambda:torch.round(y_pred), 
                          y_trues=lambda:torch.round(y_true))
        
    

    def log_record(self, epoch, model, record : OrderedDict[str, Any] | None = None) -> OrderedDict[str, Any]:
        record = record or self.produce_record(epoch, model)
        if 'epoch' not in record:
            record['epoch'] = epoch
            record.move_to_end('epoch', last=False)
        discarded = self.last_n.pop()
        for k, v in discarded.items():
            if not math.isnan(v) and not k == 'epoch':
                self.sums_last_n[k] -= v
        for metric_name, _ in self.ordered_metrics:
            value = record[metric_name]
            if not math.isnan(value):
                self.sums[metric_name] += value
                self.sums_last_n[metric_name] += value
                self.total_measures[metric_name] += 1
        self.last_n.appendleft(record)
        self.last_record = record
        return record
    
    def produce_record(self, epoch, model) -> OrderedDict[str, Any]:
        record = OrderedDict()
        record['epoch'] = epoch
        
        self._eval(model) #TODO rework this class and the concept of metric
        for metric_name, metric_fn in self.ordered_metrics:
            if isinstance(metric_fn, TorchMetric):
                value = metric_fn.compute()
                if isinstance(value, torch.Tensor):
                    value = value.item()
            else:
                value = metric_fn(epoch=epoch)
            record[metric_name] = value
        return record

    def __repr__(self):
        return f"MetricsLogger({self.identifier}, metrics=[{self.ordered_metrics}])"

class TrainingLogger(MetricsLogger):
    def __init__(
            self, 
            metric_functions : dict[str, MetricFunction] | Sequence[NamedMetricFunction],
            identifier='train'):
        super(TrainingLogger, self).__init__(identifier, metric_functions=metric_functions, dataset=None) # type:ignore
        self._evaluated_externally = True