from pathlib import Path

from typing import Self, Literal, TYPE_CHECKING, assert_never
from dataclasses import dataclass, field

import warnings

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection
    from matplotlib.backend_bases import RendererBase

import torch
import numpy as np

class CrossBinaryHistogram:
    def __init__(
            self, 
            preds : list[str], 
            trues : list[str],
            bins : int = 100,
            min_value : float = 0.0,
            max_value : float = 1.0
        ) -> None:
        self.preds = preds
        self.trues = trues
        self.max = max_value
        self.min = min_value
        self.size = max_value - min_value
        self.bins = bins
        self.histogram = torch.zeros(
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
                    batch_histc = preds[indices,i].histc(self.bins, self.min, self.max)
                    self.histogram[i,j,c].add_(batch_histc)

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
        args.subplots_kw.setdefault('figsize', (len(self.preds) * args.axes_width, len(self.trues) * args.axes_height))
        fig, axes = plt.subplots(len(self.preds), len(self.trues), **args.subplots_kw) # type: ignore
        y2 = torch.zeros_like(self.histogram)
        match mode:
            case 'overlayed':
                density = self.histogram / self.histogram.sum(dim=3, keepdim=True)
            case 'stacked':
                density = self.histogram / self.histogram.sum(dim=(2, 3)).unsqueeze(-1).unsqueeze(-1)
                y2[:, :, 1] = density[:, :, 0]
                density[:, :, 1] += density[:, :, 0]
            case _:
                assert_never(mode)
        density = density.numpy(force=True)
        y2 = y2.numpy(force=True)
        class_kw = [args.neg_kw, args.pos_kw]
        x = np.linspace(self.min, self.max, self.bins)
        for i in range(len(self.preds)):
            for j in range(len(self.trues)):
                axes[i][j].set_title(f'{self.preds[i]} vs {self.trues[j]}')
                axes[i][j].set(**args.axes_kw)
                axes[i][j].set_xlim(self.min, self.max)
                for c in range(2):
                    kwargs = args.base_hist_kw | class_kw[c]
                    y = density[i,j,c]
                    axes[i][j].fill_between(x, y, y2[i,j,c], **kwargs)
        return fig



