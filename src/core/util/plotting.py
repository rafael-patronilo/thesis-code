from pathlib import Path
import matplotlib.image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.backend_bases import RendererBase
import matplotlib
from typing import Self
import numpy as np

import torch


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
            base_plot_style : dict = {'marker' : 's', 's': 0.01, 'linewidths' : 0}
            ):
        self.preds = preds
        self.preds_size = len(preds)
        self.trues = trues
        self.trues_size = len(trues)
        self.plotted = set()
        self.dpi = dpi
        self.figure : Figure = plt.figure(dpi=self.dpi, **figure_args)

        self.axes : list[list[Axes]] = self.figure.subplots(
            self.preds_size, self.trues_size, gridspec_kw=gridspec_args, **subplot_args) #type: ignore
        self.artists : list[list[dict[str, PathCollection]]] = []
        self._init_axes(axis)
        self.figure.canvas.draw() # init canvas
        self._init_artists(plot_styles, base_plot_style)
        self._capture_backend()

    def _init_axes(self,axis : tuple[float, float, float, float]):
        for i in range(self.preds_size):
            for j in range(self.trues_size):
                self.axes[i][j].axis(axis)
                self.axes[i][j].set_title(f'{self.preds[i]} vs {self.trues[j]}', fontsize=8)

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
                artist.set_offsets(torch.hstack((preds[:,[i]], trues[:,[j]])))
                self.axes[i][j].draw_artist(artist)
                artist.set_offsets(torch.tensor([0.0, 0.0])) # Unset tensors to clear them for garbage collection

    def save(self, file : Path | str):
        format = Path(file).suffix[1:]
        if hasattr(self.renderer, 'buffer_rgba'):
            buffer = self.renderer.buffer_rgba() #type: ignore
            matplotlib.image.imsave(file, buffer, format=format, origin='upper', dpi=self.figure.dpi)
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


