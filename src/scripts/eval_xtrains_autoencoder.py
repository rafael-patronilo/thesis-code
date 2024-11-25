from collections import OrderedDict
import sys
import time
from typing import Callable, Iterable, Sequence
import math

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
        logger.debug("Rearrange batch test passed")
    
    @classmethod
    def test_produce_counts(cls):
        pred_y = cls.REARRANGED_PRED
        true_y = cls.REARRANGED_TRUE
        expected_counts = cls.EXPECTED_COUNTS
        result = produce_counts(pred_y, true_y)
        assert torch.all(result == expected_counts), f"Expected {expected_counts}, got {result}"
        logger.debug("Produce counts test passed")

    @classmethod
    def run_all(cls):
        cls.test_rearrange_batch()
        cls.test_produce_counts()
        logger.debug("All tests passed")


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
    counts = produce_counts(r_pred_y, r_true_y)
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

def create_min_max_normalizer(model : torch.nn.Module, num_classes : int, dataloader : torch_data.DataLoader):
    """Create a MinMax normalization function

    Args:
        trainer (Trainer): _description_
        dataloader (torch_data.DataLoader): _description_
    """
    min_value = torch.full((num_classes,), float('inf'))
    max_value = torch.full((num_classes,), float('-inf'))
    for x, _ in dataloader:
        x = x.to(torch.get_default_device())
        pred : torch.Tensor = model(x)
        pred_min, pred_max = pred.aminmax(dim=0)
        torch.maximum(pred_max, max_value, out=max_value)
        torch.minimum(pred_min, min_value, out=min_value)
    logger.info(f"Min: {min_value}, Max: {max_value}")
    value_range = max_value - min_value
    return lambda tensor : (tensor - min_value) / value_range

def evaluate_permutations(
        permutations : torch.Tensor,
        normalize : Callable[[torch.Tensor], torch.Tensor],
        classes : list[str],
        model : torch.nn.Module,
        loader : torch_data.DataLoader,
        topk : int = 10,
        negate : torch.Tensor | None = None
    ):
    num_classes = len(classes)
    correct_counts = torch.zeros(permutations.size(0), num_classes)
    total_samples : int = 0
    
    last_log = time.time()
    for x, y in loader:
        x = x.to(torch.get_default_device())
        y = y.to(torch.get_default_device())
        total_samples += x.size(0)
        pred = normalize(model(x))
        del x
        evaluate_batch(pred, y, permutations, correct_counts)
        now = time.time()
        if now - last_log >= 60:
            last_log = now
            logger.info(f"Processed samples: {total_samples}")
    logger.info("Computing results")
    negated_counts = total_samples - correct_counts
    if negate is None:
        negate = negated_counts > correct_counts
    best_counts = torch.where(negate, negated_counts, correct_counts)
    best_sums = best_counts.sum(dim=1)
    best_permutations = best_sums.topk(topk)
    results : list[OrderedDict[str,str|float]] = []
    for index, value in zip(best_permutations.indices, best_permutations.values):
        record = OrderedDict()
        permutation = '_'.join(map(str, permutations[index].tolist()))
        global_accuracy = value / (total_samples*num_classes)
        concept_accuracies = best_counts[index] / total_samples
        record['permutation'] = permutation
        record['negate'] = '_'.join(['-1' if neg else '1' for neg in negate[index]])
        record['global_accuracy'] = global_accuracy.item()
        for cls, acc in zip(classes, concept_accuracies):
            record[f'{cls}_accuracy'] = acc.item()
        logger.info(f"Permutation {permutation} (at {index}) with negation {negate[index]}\n"
                    f"\tGlobal Accuracy    : {global_accuracy}\n"
                    f"\tConcept accuracies : {concept_accuracies}")
        logger.debug("\n"
                     f"\tCorrect counts    : {best_counts[index]}\n"
                     f"\tTotal samples     : {total_samples}")
        results.append(record)
    return permutations[best_permutations.indices], negate[best_permutations.indices], results

def evaluate_encoding(
        trainer : Trainer,
        file_manager : ModelFileManager, 
        dataset : SplitDataset,
        classes : list[str]
        ):
    num_classes = len(classes)
    indices = list(range(num_classes))
    encoder = trainer.model.encoder
    logger.info(f"Evaluating if Encoder can find classes {classes}")
    
    logger.info("Generating permutations tensor...")
    permutations = file_manager.cache('permutations', lambda:generate_permutations_tensor(indices))
    logger.info(f"{permutations.size(0)} permutations generated")

    training_loader = torch_data.DataLoader(
        dataset.for_training(), 
        batch_size=64, num_workers=NUM_WORKERS, worker_init_fn=dataloader_worker_init_fn)
    
    logger.info("Computing min and max for normalization...")
    normalize = create_min_max_normalizer(encoder, num_classes, training_loader)

    logger.info("Starting evaluation on training dataset...")
    training_permutations, training_negate, training_results = evaluate_permutations(
        permutations, normalize, classes, encoder, training_loader)
    del permutations
    del training_loader

    val_loader = torch_data.DataLoader(
        dataset.for_validation(), 
        batch_size=64, num_workers=NUM_WORKERS, worker_init_fn=dataloader_worker_init_fn)
    logger.info("Starting evaluation on validation dataset...")
    _, _, val_results = evaluate_permutations(
        training_permutations, normalize, classes, encoder, val_loader, negate=training_negate)
    del val_loader
    logger.info("Saving results...")
    file_manager.results_dest.joinpath('encoder_concepts_results.json').write_text(
        json.dumps({
            'training': training_results,
            'validation': val_results
        }, indent=4))


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
            evaluate_encoding(trainer, file_manager, selected_set, CLASSES)
            logger.log(NOTIFY, 'Done')

