from datetime import timedelta

import torch
from torch.utils import data as torch_data

from core.util.progress_trackers import LogProgressContextManager
import logging
logger = logging.getLogger(__name__)
from pathlib import Path

progress_cm = LogProgressContextManager(logger, cooldown=timedelta(minutes=2))


def analyze_dataset(loader : 'torch_data.DataLoader',
                    destination : 'Path',
                    dataset_description : str,
                    class_names : list[str]):
    hist = torch.zeros(len(class_names), 2)
    with progress_cm.track(f'Analyzing dataset {dataset_description}', 'batches', loader) as progress_tracker:
        for _, y in loader:
            for j in range(len(class_names)):
                col : torch.Tensor= y[:, j]
                pos = y[:, j].sum(0)
                hist[j][1] += pos
                hist[j][0] += col.size(0) - pos
            progress_tracker.tick()
    logger.info(f"Dataset {dataset_description} histogram:\n{hist}")
    densities = hist / hist.sum(-1, keepdim=True)
    logger.info(f"Dataset {dataset_description} densities:\n{densities}")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(layout='constrained')
    hist = hist.numpy(force=True)
    densities = densities.numpy(force=True)
    p = ax.bar(class_names, hist[:, 1], label='Positive', color='g')
    ax.bar_label(p, labels=[f'{d:.2%}' for d in densities[:,1]], label_type='center')
    p = ax.bar(class_names, hist[:, 0], bottom=hist[:, 1], label='Negative', color='r')
    ax.bar_label(p, labels=[f'{d:.2%}' for d in densities[:,0]], label_type='center')
    ax.set_title(f"{dataset_description} concept histogram")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend()
    fig.savefig(destination.joinpath(f"{dataset_description}_histogram.png"))
    import pandas as pd
    pd.DataFrame(hist, index=class_names, columns=['Negative', 'Positive'] #type: ignore
                 ).to_csv(destination.joinpath(f"{dataset_description}_histogram.csv"))
