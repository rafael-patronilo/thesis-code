from pathlib import Path

from typing import Self, Literal, TYPE_CHECKING
from dataclasses import dataclass, field
import warnings

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection
    from matplotlib.backend_bases import RendererBase

import torch
import numpy as np


@warnings.deprecated('Unused class and inefficient implementation. To be removed')
class CrossPlotter:
    """Utility class to efficiently plot large amounts of data points.

    How it works:
        Rather than retaining the input data until a call to `save`, the data is plotted immediately.
        This way overlapping points are drawn on top of each other, reducing the memory footprint.

    Adapted from https://stackoverflow.com/questions/20250689/plotting-a-large-number-of-points-using-matplotlib-and-running-out-of-memory
    """
    def __init__(
            self, 
            preds : list[str], 
            trues : list[str], 
            axis : tuple[float, float, float, float],
            plot_styles : dict[str, dict],
            figure_args : dict = {},
            subplot_args : dict = {},
            gridspec_args : dict = {'wspace' : 1, 'hspace' : 1},
            dpi : float = 1000.0,
            base_plot_style : dict = {'marker' : 's', 's': 0.1, 'linewidths' : 0}
            ):
        import matplotlib.pyplot as plt
        self.preds = preds
        self.preds_size = len(preds)
        self.trues = trues
        self.trues_size = len(trues)
        self.plotted = set()
        self.dpi = dpi
        self.figure : 'Figure' = plt.figure(dpi=self.dpi, **figure_args)

        self.axes : list[list['Axes']] = self.figure.subplots(
            self.preds_size, self.trues_size, gridspec_kw=gridspec_args, **subplot_args) #type: ignore
        self.artists : list[list[dict[str, 'PathCollection']]] = []
        self._init_axes(axis)
        self.figure.canvas.draw() # init canvas
        self._init_artists(plot_styles, base_plot_style)
        self._capture_backend()

    def _init_axes(self,axis : tuple[float, float, float, float]):
        for i in range(self.preds_size):
            for j in range(self.trues_size):
                self.axes[i][j].axis(axis)
                self.axes[i][j].set_title(f'{self.preds[i]} vs {self.trues[j]}', fontsize=4)

    def _capture_backend(self):
        if hasattr(self.figure.canvas, 'get_renderer'):
            self.renderer : RendererBase = self.figure.canvas.get_renderer() #type: ignore
            if not hasattr(self.renderer, 'buffer_rgba'):
                raise ValueError("Backend not supported (buffer_rgba() not found)")
        else:
            raise ValueError("Cannot get renderer from Backend")

    def _init_artists(self, artists_specs : dict[str, dict], base_plot_style : dict):
        for i in range(self.preds_size):
            self.artists.append([])
            for j in range(self.trues_size):
                self.artists[i].append({})
                for artist_name, plot_style in artists_specs.items():
                    plot_style = plot_style | base_plot_style
                    artist = self.axes[i][j].scatter([0.0], [0.0],**plot_style)
                    self.artists[i][j][artist_name] = artist

    def update(self, preds : torch.Tensor, trues : torch.Tensor, plot_style : str):
        for i in range(self.preds_size):
            for j in range(self.trues_size):
                artist = self.artists[i][j][plot_style]
                artist.set_offsets(torch.hstack((preds[:,[i]], trues[:,[j]])).cpu())
                self.axes[i][j].draw_artist(artist)
                artist.set_offsets([0.0, 0.0]) # Unset tensors to clear them for garbage collection

    def save(self, file : Path | str):
        from matplotlib.image import imsave
        format = Path(file).suffix[1:]
        if hasattr(self.renderer, 'buffer_rgba'):
            buffer = self.renderer.buffer_rgba() #type: ignore
            imsave(file, buffer, format=format, origin='upper', dpi=self.figure.dpi)
        else:
            raise ValueError("Cannot save image")

    @classmethod
    def basic_config(cls, preds : list[str], trues : list[str], axis : tuple[float, float, float, float]) -> Self:
        plotter = cls(
            preds,
            trues,
            axis,
            subplot_args=dict(
                sharex='all',
                sharey='all'
            ),
            plot_styles={
                'train' : {'c' : 'blue'},
                'val' : {'c' : 'red'}
            }
        )
        #xlim_tensor = torch.tensor([axis[0], axis[1]]).unsqueeze(1).expand(-1, len(preds))
        #ylim_tensor = torch.tensor([axis[2], axis[3]]).unsqueeze(1).expand(-1, len(trues))
        #plotter.update(xlim_tensor, ylim_tensor, 'ref')
        return plotter

class CrossBinaryHistogram:
    def __init__(
            self, 
            preds : list[str], 
            trues : list[str],
            bins : int = 100, 
            min : float = 0.0, 
            max : float = 1.0
        ) -> None:
        self.preds = preds
        self.trues = trues
        self.max = max
        self.min = min
        self.size = max - min
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

    

    def create_figure(self, mode : Literal['stacked', 'overlayed']='stacked', args : 'CreateFigureArgs' = CreateFigureArgs()) -> 'Figure':
        import matplotlib.pyplot as plt
        fig : 'Figure'
        axes : list[list['Axes']]
        args.subplots_kw.setdefault('figsize', (len(self.preds) * args.axes_width, len(self.trues) * args.axes_height))
        fig, axes = plt.subplots(len(self.preds), len(self.trues), **args.subplots_kw) # type: ignore
        y2 = torch.zeros_like(self.histogram)
        if mode == 'overlayed':
            density = self.histogram / self.histogram.sum(3, keepdim=True)
        elif mode == 'stacked':
            density = self.histogram / self.histogram.sum((2,3)).unsqueeze(-1).unsqueeze(-1)
            y2[:,:,1] = density[:,:,0]
            density[:,:,1] += density[:,:,0]
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



