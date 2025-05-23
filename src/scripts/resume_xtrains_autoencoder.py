import typing
from dataclasses import field, dataclass
from datetime import timedelta

from core.init.options_parsing import option, positional
from pathlib import Path

from core.init import DO_SCRIPT_IMPORTS
from core.logging import NOTIFY

if typing.TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from typing import Literal
    from core.datasets import get_dataset, dataset_wrappers
    from core.eval.metrics import BinaryBalancedAccuracy, metric_wrappers
    from core.eval.metrics.pearson_correlation import PearsonCorrelationCoefficient
    from core.util.progress_trackers import LogProgressContextManager
    from core.storage_management import ModelFileManager
    from core.training import Trainer
    from core.nn import layers

    import torch

    from torch import nn
    import logging
    from core.eval.metrics_crosser import MetricCrosser
    import torch.utils.data as torch_data
    import torchvision
    logger = logging.getLogger(__name__)
    progress_cm = LogProgressContextManager(logger, cooldown=timedelta(minutes=2))

SEED = 3892
NUM_IMAGES = 16

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

@dataclass
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
        dataset = get_dataset('xtrains_with_concepts')
        dataset.for_validation() # Ensure that the dataset is loaded
        label_indices = dataset.get_column_references().get_label_indices(CLASSES)
        selected_dataset = dataset_wrappers.SelectCols(dataset, select_y=label_indices)
        self.dataset = dataset
        self.selected_dataset = selected_dataset

    def produce_images(self, trainer : 'Trainer', dest : Path):
        generator = torch.Generator(device=torch.get_default_device())
        generator.manual_seed(SEED)
        loader = torch_data.DataLoader(
            self.dataset.for_validation(), batch_size=NUM_IMAGES, shuffle=True, generator=generator)
        batch, _ = next(iter(loader))
        batch = batch.to(torch.get_default_device())
        results = trainer.model(batch)
        file_manager = trainer.model_file_manager
        assert file_manager is not None
        torchvision.utils.save_image(batch, dest.joinpath("original.png"))
        torchvision.utils.save_image(results, dest.joinpath("reconstructed.png"))

    def eval_metrics(self, model : 'torch.nn.Module', trainer : 'Trainer', dest : Path):
        normalizer = layers.MinMaxNormalizer.fit(model, trainer.make_loader(self.selected_dataset.for_training()),
                                                 progress_cm=progress_cm)
        logger.info(f"Normalizer min: {normalizer.min}; max: {normalizer.max}")
        model = nn.Sequential(model, normalizer)
        loader = trainer.make_loader(self.selected_dataset.for_validation())
        with progress_cm.track('Evaluating encoding on val set', 'batches', len(loader)) as tracker:
            for x, y in loader:
                x = x.to(torch.get_default_device())
                y = y.to(torch.get_default_device())
                with torch.no_grad():
                    pred = model(x)
                    self.concept_crosser.update(pred, y)
                    self.encoding_crosser.update(pred, pred)
                tracker.tick()
        concept_results = self.concept_crosser.compute()
        for metric, result in concept_results.items():
            result.to_csv(dest.joinpath(f"concepts_{metric}.csv"))
        encoding_results = self.encoding_crosser.compute()
        for metric, result in encoding_results.items():
            result.to_csv(dest.joinpath(f"encoding_{metric}.csv"))
        for metric, result in concept_results.items():
            logger.info(f"Concepts {metric}\n"
                        f"Max : {result.max()}\n"
                        f"Min : {result.min()}")

    def eval(self, trainer : 'Trainer'):
        file_manager = trainer.model_file_manager
        assert file_manager is not None
        dest = file_manager.results_dest.joinpath("per_epoch").joinpath(f'epoch_{trainer.epoch}')
        if dest.exists():
            logger.warning(f"Destination {dest} already exists. Overwriting")
        dest.mkdir(parents=True, exist_ok=True)

        trainer.model.eval()
        model = trainer.model.encoder
        logger.info(f"Outputting results to {dest}")
        self.produce_images(trainer, dest)
        logger.info("Images saved")
        self.eval_metrics(model, trainer, dest)
        logger.info("Metrics saved")


    def __call__(self, trainer : 'Trainer') -> bool:
        if trainer.epoch % self.eval_each == 0 and trainer.epoch > 0:
            logger.info(f"Evaluating correlations : Epoch {trainer.epoch}")
            try:
                self.eval(trainer)
                logger.log(NOTIFY, f"Evaluation at Epoch {trainer.epoch} complete")
            except Exception as e:
                logger.critical(f"Failed to evaluate model: {e}", exc_info=True)
        return False

def main(options : Options):
    """
    Resume training from the last checkpoint,
    stopping periodically to evaluate the model's capacity to identify concepts
    """
    with ModelFileManager(path=options.model) as file_manager:
        # Load trainer
        assert options.preferred_checkpoint in ['best', 'last']
        preferred_checkpoint : Literal['last', 'best']
        preferred_checkpoint = options.preferred_checkpoint  # type: ignore
        trainer = Trainer.load_checkpoint(file_manager, options.checkpoint, preferred_checkpoint)
        if options.clear_stop_criteria:
            trainer.stop_criteria = []
        evaluator = Evaluator(options.eval_each)
        trainer.stop_criteria.append(evaluator)
        evaluator.eval(trainer)
        if options.num_epochs is not None:
            trainer.train_epochs(options.num_epochs)
        else:
            trainer.train_indefinitely()