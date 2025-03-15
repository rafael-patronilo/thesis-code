import torch
from torch import nn
from typing import overload
from core.util.progress_trackers import ProgressContextManager, NULL_PROGRESS_CM

def add_layers(model : nn.Module, *layers : nn.Module) -> nn.Module:
    if isinstance(model, nn.Sequential):
        model.append(*layers)
        return model
    else:
        return nn.Sequential(model, *layers)

class MinMaxNormalizer(nn.Module):
    def __init__(
            self, 
            min : torch.Tensor, 
            max : torch.Tensor, 
            nan_value : float | None = 0,
            clamp : bool = True,
            *args, **kwargs) -> None:
        self.min = min
        self.max = max
        self.nan_value = nan_value
        self.clamp = clamp
        super().__init__(*args, **kwargs)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        result = (x - self.min) / (self.max - self.min)
        if self.nan_value is not None:
            result[torch.isnan(result)] = self.nan_value
        if self.clamp:
            result = torch.clamp(result, 0, 1)
        return result

    @classmethod
    def fit(cls, 
            model : nn.Module, 
            loader : torch.utils.data.DataLoader, 
            device = None,
            progress_cm : ProgressContextManager = NULL_PROGRESS_CM,
              **kwargs):
        if device is None:
            device = torch.get_default_device()
        min : torch.Tensor = None #type: ignore
        max : torch.Tensor = None #type: ignore
        model.eval()
        with progress_cm.track('MinMaxNormalizer fit', 'batches', loader) as progress_tracker:
            with torch.no_grad():
                for x, _ in loader:
                    x : torch.Tensor = x.to(device)
                    z : torch.Tensor = model(x)
                    if min is None:
                        min = z.min(0).values
                        max = z.max(0).values
                    else:
                        min = torch.min(min, z.min(0).values)
                        max = torch.max(max, z.max(0).values)
                    progress_tracker.tick()
        if min is None or max is None:
            raise ValueError("No data found")
        return cls(min, max, **kwargs)


class Reorder(torch.nn.Module):
    def __init__(self, attribution: list[int]):
        super().__init__()
        self.attribution = attribution

    def forward(self, x):
        return x[:, self.attribution]
