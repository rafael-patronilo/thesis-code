import pandas as pd
from typing import Callable
import torch
from torcheval.metrics import Metric

class MetricCrosser:
    def __init__(
            self, 
            pred_labels : list[str] |tuple[str, int], 
            true_labels : list[str] | tuple[str, int],
            metrics : dict[str, Callable[[], Metric]]):
        if isinstance(pred_labels, tuple):
            pred_labels = [f'{pred_labels[0]}{i}' for i in range(pred_labels[1])]
        if isinstance(true_labels, tuple):
            true_labels = [f'{true_labels[0]}{i}' for i in range(true_labels[1])]
        self.pred_labels = pred_labels
        self.pred_size = len(pred_labels)
        self.true_labels = true_labels
        self.true_size = len(true_labels)

        self.metrics = {
            metric_name : self._make_dataframe(metric_factory)
            for metric_name, metric_factory in metrics.items()
        }

    def _make_dataframe(self, metric_factory : Callable[[],Metric]):
        return pd.DataFrame([
                [metric_factory() for _ in self.true_labels]
                for _ in self.pred_labels
            ],
            index = self.pred_labels,
            columns = self.true_labels)
    
    def _assert_right_size(self, preds : torch.Tensor, trues : torch.Tensor):
        if not preds.size(1) == self.pred_size: 
            raise ValueError(f'Invalid pred size {preds.size(1)} (Expected {self.pred_size})')
        if not trues.size(1) == self.true_size: 
            raise ValueError(f'Invalid true size {trues.size(1)} (Expected {self.true_size})')

    def update_metric(self, table : pd.DataFrame, preds : torch.Tensor, trues : torch.Tensor):
        self._assert_right_size(preds, trues)
        for i in range(self.pred_size):
            for j in range(self.true_size):
                metric : Metric = table.iat[i,j]
                metric.update(preds[:,i], trues[:,j])

    def update(self, preds : torch.Tensor, trues : torch.Tensor):
        for table in self.metrics.values():
            self.update_metric(table, preds, trues)

    def compute(self, output_float : bool = True):
        if output_float:
            mapper = lambda x : x.compute().item()
        else:
            mapper = lambda x : x.compute()
        return {
            name : table.map(mapper) 
            for name, table in self.metrics.items()
        }

    def reset(self):
        for table in self.metrics.values():
            table.map(lambda x : x.reset())
