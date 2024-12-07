import torcheval
from torcheval.metrics import BinaryConfusionMatrix, Metric
import torch


class __ConfusionMatrix:
    
    def __init__(self, threshold: float = 0.5):
        raise NotImplementedError("Work in progress")
        self.cm = torch.zeros(2, 2)
        self.threshold = threshold

    def true_positives(self):
        pass

    def update(self, y_pred : torch.Tensor, y_true : torch.Tensor):
        raise NotImplementedError("Work in progress")
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shapes of y_pred and y_true must be the same. Got {y_pred.shape} and {y_true.shape}")
        if y_true.dtype.is_floating_point:
            raise ValueError(f"y_true must be of integer type (1 and 0). Got {y_true.dtype}")
        if y_pred.dim() != 1:
            raise ValueError(f"Tensors should have dimension 1, got {y_pred.dim()}")
        y_pred = torch.where(y_pred >= self.threshold, 1, 0)
        
        

class BinaryBalancedAccuracy(BinaryConfusionMatrix):
    def __init__(self):
        assert torcheval.version.__version__ == '0.0.7', "confusion matrix order may have been changed: https://github.com/pytorch/torcheval/issues/183"
        super().__init__()

    def compute(self):
        cm = super().compute()
        # docs are wrong: https://github.com/pytorch/torcheval/issues/183
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
        specificity = tn / (tn + fp)
        recall = tp / (tp + fn)
        return (specificity + recall) / 2

class BinarySpecificity(BinaryConfusionMatrix):
    def __init__(self):
        assert torcheval.version.__version__ == '0.0.7', "confusion matrix order may have been changed: https://github.com/pytorch/torcheval/issues/183"
        super().__init__()

    def compute(self):
        cm = super().compute()
        # docs are wrong: https://github.com/pytorch/torcheval/issues/183
        tn = cm[0, 0]
        fp = cm[0, 1]
        return tn / (tn + fp)