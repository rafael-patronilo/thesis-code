from dataclasses import field
from pathlib import Path
from typing import Literal

import torch

from core.datasets import get_dataset
from core.eval.metrics import BinaryBalancedAccuracy, metric_wrappers
from core.eval.metrics.pearson_correlation import PearsonCorrelationCoefficient
from core.init.options_parsing import option, positional
from core.storage_management import ModelFileManager
from core.training import Trainer
import logging

from core.eval.metrics_crosser import MetricCrosser

logger = logging.getLogger(__name__)

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

SHORT_CLASSES = [
    'Passenger',
    'Freight',
    'Empty',
    'Long',
    'Reinforced',
    'LongPassenger',
    '2Passenger',
    '2Freight',
    '3Wagons'
]

ENCODING_SIZE = 16


class Options:
    model: Path = field(
        metadata=positional(Path, help_="The model to load")
    )

    preferred_checkpoint: str = field(default='best',
                                      metadata=option(str, help_=
                                      "Either 'best' or 'last'. Defaults to 'best'. "
                                      "Specifies which checkpoint to prefer "
                                      "during checkpoint discovery. "
                                      "If the checkpoint option is specified, "
                                      "this option is ignored.")
                                      )

    checkpoint: Path | None = field(default=None,
                                    metadata=option(Path, help_=
                                    "The checkpoint file to load. If not specified, "
                                    "the script will automatically decide which checkpoint to load, "
                                    "in accordance to the preferred_checkpoint option."))
    eval_each : int = field(
        default=20,
        metadata=option(int, help_="Evaluate the model each `eval_each` epochs")
    )
    num_epochs: int | None = field(default=None,
                                   metadata=option(int, help_="Number of epochs to train for"))
    clear_stop_criteria: bool = field(default=True,
                                      metadata=option(bool, help_="Clear stop criteria defined by the build script"))


class Evaluator:
    def __init__(self, eval_each : int):
        self.eval_each = eval_each
        enc_labels = [f"E{i}" for i in range(ENCODING_SIZE)]
        self.concept_crosser = MetricCrosser(enc_labels, CLASSES, {
            'correlation' : PearsonCorrelationCoefficient,
            'balanced_accuracy' : lambda : metric_wrappers.ToDtype(
                BinaryBalancedAccuracy(), torch.int32, apply_to_pred=False)
        })
        self.encoding_crosser = MetricCrosser(enc_labels, enc_labels, {
            'correlation' : PearsonCorrelationCoefficient,
        })
        self.dataset = get_dataset('xtrains_with_concepts')


    def eval(self, trainer : Trainer):
        trainer.model.eval()
        model = trainer.model.encoder
        for x, y in trainer.make_loader(self.dataset.for_validation()):
            with torch.no_grad():
                pred = model(x)
                self.concept_crosser.update(pred, y)
                self.encoding_crosser.update(pred, pred)


    def __call__(self, trainer : Trainer) -> bool:
        if trainer.epoch % self.eval_each == 0 and trainer.epoch > 0:
            self.eval(trainer)
        return False

def main(options : Options):
    """
    Resume training from the last checkpoint
    """
    with ModelFileManager(path=options.model) as file_manager:
        # Load trainer
        assert options.preferred_checkpoint in ['best', 'last']
        preferred_checkpoint : Literal['last', 'best']
        preferred_checkpoint = options.preferred_checkpoint  # type: ignore
        trainer = Trainer.load_checkpoint(file_manager, options.checkpoint, preferred_checkpoint)
        if options.clear_stop_criteria:
            trainer.stop_criteria = []
        trainer.stop_criteria.append(Evaluator(options.eval_each))
        if options.num_epochs is not None:
            trainer.train_epochs(options.num_epochs)
        else:
            trainer.train_indefinitely()