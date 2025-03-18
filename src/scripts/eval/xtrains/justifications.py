
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal, NamedTuple, Optional


from core.init import DO_SCRIPT_IMPORTS
from dataclasses import dataclass, field
from core.init.options_parsing import option, positional, comma_split

if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    import logging

    logger = logging.getLogger(__name__)

    from core.util.progress_trackers import LogProgressContextManager
    progress_cm = LogProgressContextManager(logger)

    import pandas as pd
    from torch.utils.data import Dataset as TorchDataset
    import torch
    from torch import nn
    from core.nn.layers import Reorder

    from core.training.trainer import Trainer
    from core.storage_management.model_file_manager import ModelFileManager

    from core.eval.justifier_wrapper.justifier import (
        ParallelJustifierWrapper, JustifierArgs, JustifierConfig, JustifierResult)
    from core.eval.justifier_wrapper.justifier_result import Justification

    from datasets.xtrains import CONCEPTS, CLASSES
    from analysis_tools.xtrains_utils import class_to_manchester_assertion, prepare_pn_with_attribution

    from core.datasets import get_dataset

COLS = ['All Correct', 'Some Correct', 'None Correct', 'No Justifications']
SEED=51612

def preprocessor(batch : tuple['torch.Tensor', 'torch.Tensor', int]) -> 'JustifierArgs':
    beliefs, correct_preds, train_type = batch
    assert correct_preds.dtype == torch.bool
    observations = []
    correct_dict : dict[str, bool] = {}
    for i, concept in enumerate(CONCEPTS):
        belief = beliefs[i].item()
        negate = belief < 0.5
        if negate:
            belief = 1.0 - belief
        manchester = class_to_manchester_assertion(concept, negate)
        observations.append((manchester, belief))
        correct_dict[manchester] = correct_preds[i].item() # type: ignore # (asserted above)
    entailment = class_to_manchester_assertion(CLASSES[train_type])
    return JustifierArgs(entailment, observations, metadata=(correct_dict, train_type))

class CorrectnessCount(NamedTuple):
    train_type : int
    all_justifications : 'pd.Series'
    best_justification : 'pd.Series'

def postprocessor(result : 'JustifierResult') -> CorrectnessCount:
    correct_dict : dict[str, bool]
    train_type : int
    correct_dict, train_type = result.args.metadata
    correct = 0
    incorrect = 0
    justifications = result.justifications
    series = pd.Series([0, 0, 0, 0], index=COLS)
    best_series = pd.Series([0, 0, 0, 0], index=COLS)
    if justifications in ['inconsistent', 'not_entailed']:
        series['No Justifications'] = 1
        best_series['No Justifications'] = 1
    else:
        is_first = True
        for justification in justifications:
            assert isinstance(justification, Justification)
            is_correct = all(correct_dict[observation.concept_name]
                             for observation in justification.used_observations)
            if is_correct:
                correct += 1
                if is_first:
                    best_series['All Correct'] = 1
            else:
                incorrect += 1
                if is_first:
                    best_series['None Correct'] = 1
            is_first = False
        if correct > 0:
            if incorrect == 0:
                series['All Correct'] = 1
            else:
                series['Some Correct'] = 1
        elif incorrect > 0:
            series['None Correct'] = 1
        else:
            series['No Justifications'] = 1
            best_series['No Justifications'] = 1
    return CorrectnessCount(train_type, series, best_series)


def run_justifier(
        datasets : list['TorchDataset'],
        config : 'JustifierConfig',
        pn : 'torch.nn.Module',
        rn : 'torch.nn.Module',
        trainer : 'Trainer',
        max_samples : Optional[int]
) -> tuple['pd.DataFrame', 'pd.DataFrame']:
    justifier = ParallelJustifierWrapper(config)
    pn.eval()
    rn.eval()
    if max_samples is None:
        max_samples = sum(len(d) for d in datasets) # type: ignore
    assert max_samples is not None
    with justifier:
        def producer():
            queued_samples = 0
            with torch.no_grad():
                for dataset in datasets:
                    for x, y in trainer.make_loader(dataset, force_shuffle=True, seed=SEED):
                        x = x.to(torch.get_default_device())
                        y = y.to(torch.get_default_device())
                        y_concepts = y[:, 4:]
                        y_classes = y[:, :4]
                        concepts = pn(x)
                        correct_preds = (concepts > 0.5) == (y_concepts > 0.5)
                        for i in range(min(y.size(0), max_samples - queued_samples)):
                            for j in [0, 1, 2]:
                                if y_classes[i][j] > 0.5:
                                    yield concepts[i].cpu(), correct_preds[i].cpu(), j
                            queued_samples += 1
        result = pd.DataFrame(0, index=pd.Index(CLASSES), columns=pd.Index(COLS))
        result_best = pd.DataFrame(0, index=pd.Index(CLASSES), columns=pd.Index(COLS))
        def reducer(sample_results : Iterable[CorrectnessCount]):
            for i, sample_result in enumerate(sample_results):
                result.loc[CLASSES[sample_result.train_type]] += sample_result.all_justifications
                result_best.loc[CLASSES[sample_result.train_type]] += sample_result.best_justification
                if (i+1) % 500 == 0:
                    logger.info(f'Processed {i+1} samples. Results so far:\n{result}\n'
                                f'Best results so far:\n{result_best}')
        with progress_cm.track('Running Justifier', 'samples', max_samples) as tracker:
            justifier.from_producer(
                producer(),
                preprocessor,
                postprocessor,
                reducer,
                tracker
            )
    return result, result_best

def output_results(path : Path, dataframes : tuple['pd.DataFrame', 'pd.DataFrame']):
    logger.info(dataframes)
    for name, results in zip(['all', 'best'], dataframes):
        logger.info(f'Results ({name} justifications):\n{results}')
        ratio = results / results.sum(axis='rows')
        logger.info(f'Ratio:\n{ratio}')
        path.mkdir(parents=True, exist_ok=True)
        results.to_csv(path.joinpath(f'results_{name}.csv'))
        ratio.to_csv(path.joinpath(f'ratio_{name}.csv'))
        ratio.to_latex(path.joinpath(f'ratio_{name}.tex'), float_format='{:.2%}'.format)

@dataclass
class Options:
    model_name : str = field(metadata=positional(str, help_='The name of the model to justify'))
    checkpoint : Optional[Path] = field(default=None, metadata=option(Path, help_='The checkpoint to load'))
    prefer : Literal['last', 'best'] = field(default='best', metadata=option(str, help_='The checkpoint to prefer'))
    with_training : bool = field(default=False, metadata=option(bool, help_='Whether to include the training set'))
    ontology_file : Path = field(
        default=Path('ontologies/XTRAINS.owl'),
        metadata=option(Path, help_='The ontology file to use')
    )
    max_samples : Optional[int] = field(default=None,
                                        metadata=option(int, help_='The maximum number of samples to justify'))
    attribution : Optional[list[str]] = field(default=None,
              metadata=option(comma_split, help_="Concept attribution to use "
                                                     "between the perception network and the reasoning network"))
    binary_threshold : Optional[float] = field(default=None,
        metadata=option(float, help_="Threshold to distinguish positive and negative"
                                     " classification for binary metrics. "
                                     "Specifying this option will turn outputs precisely to 1 and 0, "
                                     "which means best justification results will hold no meaning."))

def main(options : Options):
    justifier_config = JustifierConfig(ontology_file=options.ontology_file)
    split_dataset = get_dataset('xtrains_with_concepts')
    with ModelFileManager(options.model_name) as mfm:
        trainer = Trainer.load_checkpoint(mfm, options.checkpoint, options.prefer)
        results_path = mfm.results_dest.joinpath('justifications')

        pn = trainer.model.perception_network
        rn = trainer.model.reasoning_network
        if options.attribution is not None:
            pn = prepare_pn_with_attribution(pn, options.attribution, options.binary_threshold)

        dfs = run_justifier([split_dataset.for_validation()], justifier_config,
                           pn, rn, trainer, options.max_samples)
        output_results(results_path.joinpath('val'), dfs)
        if options.with_training:
            dfs = run_justifier([split_dataset.for_training()], justifier_config,
                               pn, rn, trainer, options.max_samples)
            output_results(results_path.joinpath('train'), dfs)
