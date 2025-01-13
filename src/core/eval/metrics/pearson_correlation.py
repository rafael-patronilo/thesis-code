from typing import Iterable, Mapping, TypedDict, Self

from torcheval.metrics import Metric
import torch


class _State(TypedDict):
    """
    TypedDict with terms of the formula
     to make sure there are no name mistakes in update or compute.
    """
    x : torch.Tensor
    y : torch.Tensor
    x_sq : torch.Tensor # the square of x
    y_sq : torch.Tensor
    xy : torch.Tensor


def _blank_state() -> _State:
    return _State(
        x = torch.tensor(0.0, requires_grad=False),
        y = torch.tensor(0.0, requires_grad=False),
        x_sq = torch.tensor(0.0, requires_grad=False),
        y_sq = torch.tensor(0.0, requires_grad=False),
        xy = torch.tensor(0.0, requires_grad=False)
    )


class PearsonCorrelationCoefficient(Metric):

    def __init__(self):
        self.sums : _State = _blank_state()
        self.n = torch.tensor(0)
        super().__init__()

    def reset(self):
        self.n.zero_()
        tensors : Iterable[torch.Tensor] = self.sums.values() # type: ignore
        for value in tensors:
            value.zero_()
        return self

    def state_dict(self): # type: ignore # I seem to be unable to get TState correct
        return dict(
            sums = self.sums,
            n = self.n
        )

    def load_state_dict(self, state_dict, strict: bool = True):
        self.sums : _State = state_dict['sums']
        self.n = state_dict['n']


    def merge_state(self, metrics: Iterable[Self]):
        for metric in metrics:
            for key, value in metric.sums.items():
                self.sums[key] += value # type: ignore
            self.n += metric.n
        return self

    def update(self, x: torch.Tensor, y: torch.Tensor):
        x = x.detach()
        y = y.detach()
        self.sums['x'] += x.sum()
        self.sums['y'] += y.sum()
        self.sums['x_sq'] += (x ** 2).sum()
        self.sums['y_sq'] += (y ** 2).sum()
        self.sums['xy'] += (x * y).sum()
        self.n += x.size(0)
        return self

    def compute(self):
        # TODO I got the formula in https://en.wikipedia.org/wiki/Correlation, find another source (sklearn?)
        means : _State = {key : value / self.n for key, value in self.sums.items()} #type: ignore
        cov = means['xy'] - means['x'] * means['y']
        std_x = (means['x_sq'] - means['x'] ** 2).sqrt()
        std_y = (means['y_sq'] - means['y'] ** 2).sqrt()
        return cov / (std_x * std_y)

