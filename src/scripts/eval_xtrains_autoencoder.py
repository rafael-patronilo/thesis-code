from collections import OrderedDict
import sys
import time
from typing import Iterable, Sequence
import math

import torcheval.metrics

from core.util.debug import debug_tensor
from core import Trainer, ModelFileManager
from core.datasets import SplitDataset, dataset_wrappers

from core.metrics import metric_wrappers as metric_wrappers
import torch
import torchvision
import torch.utils.data as torch_data
import os
from core.metrics_logger import dataloader_worker_init_fn
from core.datasets import get_dataset
import torcheval
import logging
from logging_setup import NOTIFY
import json

logger = logging.getLogger(__name__)
SEED = 3892
NUM_IMAGES = 16
NUM_WORKERS = int(os.getenv('NUM_THREADS', 4))

CLASSES = [
    'PassengerCar',
    'FreightWagon',
    'EmptyWagon',
    'LongWagon',
    'ReinforcedCar',
    'LongPassengerCar',
    'AtLeast2PassengerCars',
    'AtLeast2FreightWagons',
    'AtLeast3Wagons'
]


class Tests:
    PERMUTATIONS = torch.tensor([
            [0, 1, 2],
            [0, 2, 1],
            [2, 1, 0]
        ], dtype=torch.int32)
    
    PRED_Y = torch.tensor([
            [0.9, 0.2, 0.3],
            [0.4, 0.6, 0.7]
        ])
    
    TRUE_Y = torch.tensor([
            [0, 1, 0],
            [1, 1, 0]            
        ])
    
    REARRANGED_PRED = torch.tensor([
            [
                [0.9, 0.2, 0.3],
                [0.9, 0.3, 0.2],
                [0.3, 0.2, 0.9]
            ],
            [
                [0.4, 0.6, 0.7],
                [0.4, 0.7, 0.6],
                [0.7, 0.6, 0.4]
            ]
        ])
    REARRANGED_TRUE = torch.tensor([
            [
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0]
            ],
            [
                [1, 1, 0],
                [1, 1, 0],
                [1, 1, 0]
            ]
        ])
    
    EXPECTED_COUNTS = torch.tensor([
            [0, 1, 1],
            [0, 1, 1],
            [2, 1, 1]
        ])
    @classmethod
    def test_rearrange_batch(cls):
        permutations = cls.PERMUTATIONS
        pred_y = cls.PRED_Y
        rearranged_pred = cls.REARRANGED_PRED
        true_y = cls.TRUE_Y
        rearranged_true = cls.REARRANGED_TRUE
        result_py, result_ty = rearrange_batch(pred_y, true_y, permutations)
        assert torch.all(result_py == rearranged_pred), f"Expected {rearranged_pred}, got {result_py}"
        assert torch.all(result_ty == rearranged_true), f"Expected {rearranged_true}, got {result_ty}"
    
    @classmethod
    def test_produce_counts(cls):
        pred_y = cls.REARRANGED_PRED
        true_y = cls.REARRANGED_TRUE
        expected_counts = cls.EXPECTED_COUNTS
        result = produce_counts(pred_y, true_y)
        assert torch.all(result == expected_counts), f"Expected {expected_counts}, got {result}"

    @classmethod
    def run_all(cls):
        cls.test_rearrange_batch()
        cls.test_produce_counts()


def generate_permutations(elements : list):
    """Generate permutations using Heap's algorithm 
    (adapted from https://en.wikipedia.org/wiki/Heap's_algorithm)

    Args:
        elements (list): the elements to permute
    """
    c = [0] * len(elements)
    yield elements
    i = 1
    while i < len(elements):
        if c[i] < i:
            if i % 2 == 0:
                elements[0], elements[i] = elements[i], elements[0]
            else:
                elements[c[i]], elements[i] = elements[i], elements[c[i]]
            yield elements
            c[i] += 1
            i = 1
        else:
            c[i] = 0
            i += 1

def generate_permutations_tensor(elements : list[int]):
    num_elements = len(elements)
    num_permutations = math.factorial(num_elements)
    tensor = torch.empty((num_permutations, num_elements), dtype=torch.int32)
    for i, permutation in enumerate(generate_permutations(elements)):
        tensor[i] = torch.tensor(permutation)
    return tensor


def evaluate_batch(
        pred_y : torch.Tensor,
        true_y : torch.Tensor,
        permutations : torch.Tensor,
        correct_counts : torch.Tensor,
        ):
    logger.debug(f"Original\n\tpred_y: {pred_y}, true_y: {true_y}")
    r_pred_y, r_true_y = rearrange_batch(pred_y, true_y, permutations)
    logger.debug(f"Rearranged with permutations\n\tpred_y: {r_pred_y}, true_y: {r_true_y}")
    counts = produce_counts(r_pred_y, r_true_y)
    debug_tensor('counts.txt', counts)
    correct_counts += counts


def produce_counts(pred_y, true_y):
    result = torch.where((pred_y > 0.5) == (true_y > 0.5), 1, 0)
    return result.sum(dim=0)

def rearrange_batch(
        pred_y : torch.Tensor,
        true_y : torch.Tensor,
        permutations : torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
    new_py = pred_y[:, permutations]
    new_ty = true_y.unsqueeze(1).expand_as(new_py)
    return new_py, new_ty

def get_min_max(model : torch.nn.Module, dataloader : torch_data.DataLoader):
    """Create a MinMax normalization function

    Args:
        trainer (Trainer): _description_
        dataloader (torch_data.DataLoader): _description_
    """
    min_value = float('inf')
    max_value = float('-inf')
    for x, _ in dataloader:
        pred : torch.Tensor = model(x)
        min_value = min(min_value, pred.min().item())
        max_value = max(max_value, pred.max().item())
    return min_value, max_value

def find_best_permutation(
        permutations : torch.Tensor,
        num_classes : int,
        trainer: Trainer,
        loader : torch_data.DataLoader
    ):
    correct_counts = torch.zeros(permutations.size(0), num_classes)
    model : torch.nn.Module = trainer.model.encoder
    total_samples : int = 0
    min, max = get_min_max(model, loader)
    value_range = max - min
    def normalize(tensor : torch.Tensor):
        return (tensor - min) / value_range
    last_log = time.time()
    for x, y in loader:
        x = x.to(torch.get_default_device())
        y = y.to(torch.get_default_device())
        total_samples += x.size(0)
        pred = normalize(trainer.model.encoder(x))
        del x
        evaluate_batch(pred, y, permutations, correct_counts)
        now = time.time()
        if now - last_log >= 60:
            last_log = now
            logger.info(f"Processed samples: {total_samples}")
    logger.info("Evaluation done")
    negated_counts = total_samples - correct_counts
    negate = negated_counts > correct_counts
    best_counts = torch.where(negate, negated_counts, correct_counts)
    best_sums = best_counts.sum(dim=1)
    best_permutation_i = best_sums.argmax()
    best_permutation = permutations[best_permutation_i]
    best_negate = negate[best_permutation_i]
    accuracies = best_counts[best_permutation_i] / total_samples
    accuracies = accuracies.tolist()
    accuracy = best_sums[best_permutation_i] / (total_samples * num_classes)
    accuracy = accuracy.item()
    logger.log(NOTIFY, f"Best permutation is {best_permutation} with the feature inversion {best_negate}\n"
                f"\tGlobal accuracy: {accuracy}\n"
                f"\tConcept accuracies: {accuracies}")
    return normalize, best_permutation, best_negate, accuracy, accuracies

def evaluate_permutations(
        trainer : Trainer,
        file_manager : ModelFileManager, 
        dataset : SplitDataset,
        classes : list[str]
        ):
    num_classes = len(classes)
    indices = list(range(num_classes))
    logger.info(f"Evaluating permutations for {classes}")
    
    logger.info("Generating permutations tensor...")
    permutations = generate_permutations_tensor(indices)
    logger.info(f"{permutations.size(0)} permutations generated")

    training_loader = torch_data.DataLoader(
        dataset.for_training(), 
        batch_size=64, num_workers=NUM_WORKERS, worker_init_fn=dataloader_worker_init_fn)
    
    logger.info("Starting evaluation on training dataset...")
    normalize, best_permutation, best_negate, train_accuracy, train_accuracies = find_best_permutation(
        permutations, num_classes, trainer, training_loader)
    del permutations
    val_loader = torch_data.DataLoader(
        dataset.for_training(), 
        batch_size=64, num_workers=NUM_WORKERS, worker_init_fn=dataloader_worker_init_fn)
    counts = torch.zeros(num_classes)
    total_samples = 0
    logger.info("Starting evaluation on validation dataset...")
    for x, y in val_loader:
        pred : torch.Tensor = trainer.model.encoder(x)
        pred = normalize(pred)
        pred = pred[best_permutation]
        pred = torch.where(best_negate, -pred, pred)
        counts += produce_counts(pred, y)
        total_samples += x.size(0)
    accuracies = counts / total_samples
    accuracies = accuracies.tolist()
    accuracy = counts.sum() / (total_samples*num_classes)
    accuracy = accuracy.item()
    logger.info("Validation:\n"                
                f"\tGlobal accuracy: {accuracy}\n"
                f"\tConcept accuracies: {accuracies}")
    best_permutation = best_permutation.tolist()
    best_negate = (-1 if val else 1 for val in best_negate)
    result_json = json.dumps({
        'best_permutation':best_permutation,
        'best_negate':best_negate,
        'train':{
            'accuracy' : train_accuracy,
            'concept_accuracies' : {cls : acc for cls, acc in zip(classes, train_accuracies)}
        },
        'val':{
            'accuracy' : accuracy,
            'concept_accuracies' : {cls : acc for cls, acc in zip(classes, accuracies)}
        }
    }, indent=4)
    file_manager.results_dest.joinpath('permutations.json').write_text(result_json)


def sample_images(
        trainer : Trainer, 
        file_manager : ModelFileManager, 
        dataset : torch_data.Dataset
    ):
    generator = torch.Generator(device=torch.get_default_device())
    generator.manual_seed(SEED)
    loader = torch_data.DataLoader(
        dataset, batch_size=NUM_IMAGES, shuffle=True, generator=generator)
    batch, _ = next(iter(loader))
    batch = batch.to(torch.get_default_device())
    results = trainer.model(batch)
    torchvision.utils.save_image(batch, file_manager.results_dest.joinpath("original.png"))
    torchvision.utils.save_image(results, file_manager.results_dest.joinpath("reconstructed.png"))


def main():
    model_name = sys.argv[1]
    model_path = None
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    Tests.run_all()
    with torch.no_grad():
        with ModelFileManager(model_name, model_path) as file_manager:
            trainer = Trainer.load_checkpoint(file_manager)
            trainer.model.eval()
            dataset = get_dataset('xtrains_with_concepts')
            val_set = dataset.for_validation()
            sample_images(trainer, file_manager, val_set)
            logger.info("Sampling images done")

            label_indices = dataset.get_collumn_references().get_label_indices(CLASSES)
            selected_set = dataset_wrappers.SelectCols(dataset, select_y=label_indices)
            evaluate_permutations(trainer, file_manager, selected_set, CLASSES)
            logger.log(NOTIFY, 'Done')




