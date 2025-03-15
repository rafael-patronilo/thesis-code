from pathlib import Path

from typing import Literal, TYPE_CHECKING, assert_never
from dataclasses import dataclass, field

import warnings


if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

import torch
import numpy as np

class CrossBinaryHistogram:
    def __init__(
            self, 
            preds : list[str], 
            trues : list[str],
            min_values: torch.Tensor,
            max_values: torch.Tensor,
            bins : int = 100
        ) -> None:
        if len(min_values) != len(preds):
            raise ValueError(f"Expected {len(preds)} min values, got {len(min_values)}")
        if len(max_values) != len(preds):
            raise ValueError(f"Expected {len(preds)} max values, got {len(max_values)}")
        self.preds = preds
        self.trues = trues
        self.max_values = max_values
        self.min_values = min_values
        self._clipping_warning_done = False
        self.sizes = max_values - min_values
        self.bins = bins
        self.histograms = torch.zeros(
            len(self.preds),
            len(self.trues),
            2,
            int(self.bins)
        )

    def update(self, preds : torch.Tensor,  trues : torch.Tensor):
        for i in range(len(self.preds)):
            for j in range(len(self.trues)):
                for c in range(2):
                    indices = trues[:,j] == c
                    min_value = self.min_values[i].item()
                    max_value = self.max_values[i].item()
                    target_preds = preds[indices,i]
                    if target_preds.numel() == 0:
                        continue
                    if target_preds.min() < min_value or target_preds.max() > max_value:
                        target_preds = target_preds.clip(min_value, max_value)
                        if not self._clipping_warning_done:
                            warnings.warn(
                                f"Values are outside the range "
                                f"[{min_value}, {max_value}]. Clipping will occur."
                            )
                            self._clipping_warning_done = True
                    batch_histc = target_preds.histc(self.bins, min_value, max_value)

                    self.histograms[i,j,c].add_(batch_histc)

    @dataclass
    class CreateFigureArgs:
        axes_width : float = 3
        axes_height : float = 2
        subplots_kw : dict = field(default_factory=lambda : { 
            'layout' : 'constrained', 
            'sharey' : 'row',
            'sharex' : 'all',
            'dpi' : 500
        })
        axes_kw : dict = field(default_factory=lambda : {'frame_on' : False})
        base_hist_kw : dict = field(default_factory = lambda: {'linestyle':'--', 'linewidth':0.2})
        neg_kw : dict = field( default_factory=lambda : {'edgecolor' : (1,0,0,1), 'facecolor' : (1,0,0,0.5)})
        pos_kw : dict = field( default_factory=lambda : {'edgecolor' : (0,1,0,1),'facecolor' : (0,1,0,0.5)})

    

    def create_figure(
            self,
            mode : Literal['stacked', 'overlayed']='stacked',
            args : 'CreateFigureArgs' = CreateFigureArgs()
    ) -> 'Figure':
        import matplotlib.pyplot as plt
        fig : 'Figure'
        axes : list[list['Axes']]
        args.subplots_kw.setdefault('figsize', (len(self.trues) * args.axes_width, len(self.preds) * args.axes_height))
        fig, axes = plt.subplots(len(self.preds), len(self.trues), **args.subplots_kw) # type: ignore
        y2 = torch.zeros_like(self.histograms)
        match mode:
            case 'overlayed':
                density = self.histograms / self.histograms.sum(dim=3, keepdim=True)
            case 'stacked':
                density = self.histograms / self.histograms.sum(dim=(2, 3)).unsqueeze(-1).unsqueeze(-1)
                y2[:, :, 1] = density[:, :, 0]
                density[:, :, 1] += density[:, :, 0]
            case _:
                assert_never(mode)
        density = density.numpy(force=True)
        y2 = y2.numpy(force=True)
        class_kw = [args.neg_kw, args.pos_kw]

        for i in range(len(self.preds)):
            max_value = self.max_values[i].item()
            min_value = self.min_values[i].item()
            x = np.linspace(min_value, max_value, self.bins)
            for j in range(len(self.trues)):
                axes[i][j].set_title(f'{self.preds[i]} vs {self.trues[j]}')
                axes[i][j].set(**args.axes_kw)
                axes[i][j].set_xlim(min_value, max_value)
                for c in range(2):
                    kwargs = args.base_hist_kw | class_kw[c]
                    y = density[i,j,c]
                    axes[i][j].fill_between(x, y, y2[i,j,c], **kwargs)
        return fig

    @dataclass
    class CreateFigurePredsArgs:
        num_rows: int = 1
        axes_width: float = 3
        axes_height: float = 3
        subplots_kw: dict = field(default_factory=lambda: {
            'layout': 'constrained',
            'dpi': 500
        })
        axes_kw: dict = field(default_factory=lambda: {'frame_on': False})
        base_hist_kw: dict = field(default_factory=lambda: {
            'linestyle': '--',
            'linewidth': 0.2,
            'edgecolor': (1, 1, 1, 1),
            'facecolor': (0.0, 0.0, 1.0, 0.5)
        })

    def create_figure_preds(self, args : 'CreateFigurePredsArgs' = CreateFigurePredsArgs()) -> 'Figure':
        import matplotlib.pyplot as plt
        fig: 'Figure'
        axes: list['Axes']
        n_cols = (len(self.preds) + args.num_rows - 1) // args.num_rows
        args.subplots_kw.setdefault('figsize', (n_cols * args.axes_width, args.num_rows * args.axes_height))

        fig, _ = plt.subplots(args.num_rows, n_cols, squeeze=True, **args.subplots_kw)  # type: ignore
        axes = fig.get_axes()
        histograms = self.histograms[:,0].sum(dim=1)
        assert histograms.ndim == 2
        density = histograms / histograms.sum(dim=1, keepdim=True)
        density = density.numpy(force=True)

        for i in range(len(self.preds)):
            max_value = self.max_values[i].item()
            min_value = self.min_values[i].item()
            x = np.linspace(min_value, max_value, self.bins)
            axes[i].set_title(f'{self.preds[i]}')
            axes[i].set(**args.axes_kw)
            axes[i].set_xlim(min_value, max_value)
            kwargs = args.base_hist_kw
            y = density[i]
            axes[i].fill_between(x, y, **kwargs)
        return fig

