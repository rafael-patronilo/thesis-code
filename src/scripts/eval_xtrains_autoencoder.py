from collections import OrderedDict
import sys
from typing import Iterable, Sequence
import math

import torcheval.metrics
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

logger = logging.getLogger(__name__)
SEED = 3892
NUM_IMAGES = 10

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
            [2, 0, 1],
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
                [0.3, 0.9, 0.2],
                [0.3, 0.2, 0.9]
            ],
            [
                [0.4, 0.6, 0.7],
                [0.7, 0.4, 0.6],
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
            [2, 1, 1],
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
    r_pred_y, r_true_y = rearrange_batch(pred_y, true_y, permutations)
    correct_counts += produce_counts(r_pred_y, r_true_y)

def produce_counts(pred_y, true_y):
    result = torch.where(torch.abs(pred_y - true_y) < 0.5, 1, 0)
    return result.sum(dim=1)

def rearrange_batch(
        pred_y : torch.Tensor,
        true_y : torch.Tensor,
        permutations : torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
    new_py = pred_y[:, permutations]
    new_ty = true_y.unsqueeze(1).expand_as(new_py)
    return new_py, new_ty


def evaluate_permutations(
        trainer : Trainer,
        file_manager : ModelFileManager, 
        dataset : torch_data.Dataset,
        classes : list[str]
        ):
    num_classes = len(classes)
    indices = list(range(num_classes))
    logger.info(f"Evaluating permutations for {classes}")
    
    logger.info("Generating permutations tensor...")
    permutations = generate_permutations_tensor(indices)
    logger.info(f"{permutations.size(0)} permutations generated")

    correct_counts = torch.zeros(permutations.size(0), num_classes, dtype=torch.int32)
    loader = torch_data.DataLoader(
        dataset, batch_size=128, num_workers=4, worker_init_fn=dataloader_worker_init_fn)
    total_samples = 0
    
    logger.info("Starting evaluation...")
    for x, y in loader:
        total_samples += x.size(0)
        pred = trainer.model(x)
        evaluate_batch(pred, y, permutations, correct_counts)
        logger.info(f"Batch finished, total samples: {total_samples}")
    
    logger.info("Evaluation done, storing results...")
    torch.save(correct_counts, file_manager.debug_dir.joinpath("permut_correct_counts.pt"))
    global_accuracy = correct_counts.sum(dim=1).float() / (total_samples*num_classes)

    file_manager.init_metrics_file("permutations", ["permutation", "accuracy"])
    best_permutation = None,
    best_accuracy = -1
    for i in range(permutations.size(0)):
        record = OrderedDict()
        permutation =  '_'.join(map(str, permutations[i].tolist()))
        record['permutation'] = permutation
        accuracy = global_accuracy[i].item()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_permutation = permutation
        record['accuracy'] = accuracy
        for j, cls in enumerate(classes):
            record[f"{cls}_acc"] = correct_counts[i, j].item()
        file_manager.write_metrics("permutations", record)
    file_manager.flush()
    logger.info("Results stored")
    logger.info(f"Best permutation: {best_permutation} with accuracy {best_accuracy}")



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
    Tests.run_all()
    with torch.no_grad():
        with ModelFileManager(model_name) as file_manager:
            trainer = Trainer.load_checkpoint(file_manager)
            trainer.model.eval()
            dataset = get_dataset('xtrains_with_concepts')
            val_set = dataset.for_validation()
            sample_images(trainer, file_manager, val_set)
            logger.info("Sampling images done")

            label_indices = dataset.get_collumn_references().get_label_indices(CLASSES)
            selected_val_set = dataset_wrappers.SelectCols(dataset, select_y=label_indices).for_validation()
            evaluate_permutations(trainer, file_manager, selected_val_set, CLASSES)




