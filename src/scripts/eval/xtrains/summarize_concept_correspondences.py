from pathlib import Path
from typing import TYPE_CHECKING, Optional


from core.init import DO_SCRIPT_IMPORTS
from core.init.options_parsing import option, positional
from dataclasses import dataclass, field

if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from analysis_tools.xtrains_utils import class_to_latex_cmd, SHORT_CONCEPTS
    from analysis_tools.sort_attributions import greedy_attribution, rank_concepts, ABSDifference
    import pandas as pd
    import logging
    logger = logging.getLogger(__name__)
@dataclass
class Options:
    results_path : Path = field(
        metadata=positional(Path, help_="Path to the directory containing the results for different sets."))
    target_csv : Path = field(
        metadata=positional(Path, help_="Name of the csv files to summarize")
    )
    mode : str = field(default='abs',
                       metadata=option(str, help_="The mode to use for determining the best attribution. "
                                   "Options are 'abs', 'max', 'min'."))
    include_expectations : bool = field(default=False,
        metadata=option(bool, help_="Whether to include the expected concepts in the ranking.")
    )
    mid_point : Optional[float] = field(default=None,
        metadata=option(float, help_="The mid point of the range. "
                                     "If not provided attempt to determine automatically.")
    )
    exclude_classes : bool = field(default=True,
        metadata=option(bool, help_="Whether to exclude the final classes from the ranking."))

def determine_mid_point(csv_file : Path) -> float:
    if 'accuracy' in csv_file.name:
        return 0.5
    elif 'correlation' in csv_file.name:
        return 0.0
    else: raise ValueError(f"Cannot determine mid point for {csv_file}")

def handle_csv_file(csv_file : Path, sort_by, maximize) -> 'pd.DataFrame':
    dest_path = csv_file.with_suffix('')
    dest_path = dest_path.with_name(dest_path.name + '_ranking')
    dest_path = dest_path.with_suffix('.csv')


    df = pd.read_csv(csv_file, index_col=0)
    ranking = rank_concepts(df, sort_by, maximize)
    logger.info(f"Saving to {dest_path}")
    assert not (dest_path.exists() and dest_path.samefile(csv_file))
    ranking.to_csv(dest_path)
    return ranking

def main(options : Options):
    if options.mid_point is None:
        mid_point = determine_mid_point(options.target_csv)
    else:
        mid_point = options.mid_point
    sort_by = None
    maximize = True
    if options.mode == 'abs':
        sort_by = ABSDifference(mid_point)
    elif options.mode == 'min':
        maximize = False

    training_file = options.results_path.joinpath('train', options.target_csv)
    training_results = pd.read_csv(training_file, index_col=0)
    ranking = handle_csv_file(training_file, sort_by, maximize)
    validation_file = options.results_path.joinpath('val', options.target_csv)
    validation_results = pd.read_csv(validation_file, index_col=0)
    handle_csv_file(validation_file, sort_by, maximize)
    attributions = ['Best Without Repeat']
    if options.include_expectations:
        attributions.insert(0, 'Expected')
    summary_cols = pd.MultiIndex.from_product(
        [attributions, ['Concept', 'Rank', 'Training', 'Validation']],
    )
    summary = pd.DataFrame(index=ranking.index, columns=summary_cols)

    def handle_negation(concept, row):
        training_result = training_results[concept][row]
        validation_result = validation_results[concept][row]
        if training_result < mid_point:
            training_result = 2 * mid_point - training_result
            validation_result = 2 * mid_point - validation_result
            concept = f"!{concept}"
        return concept, training_result, validation_result

    if options.include_expectations:
        for concept, row in zip(SHORT_CONCEPTS, ranking.index):
            summary.loc[row, ('Expected', 'Concept')] = concept
            summary.loc[row, ('Expected', 'Rank')] = ranking[concept][row]
            summary.loc[row, ('Expected', 'Training')] = training_results[concept][row]
            summary.loc[row, ('Expected', 'Validation')] = validation_results[concept][row]
    attribution : list[tuple[str, str]] = greedy_attribution(training_results, sort_by=sort_by, maximize=maximize)
    for concept, row in attribution:
        if not pd.isna(summary[('Best Without Repeat', 'Concept')][row]): # type: ignore
            raise ValueError("Multiple concepts found for the same neuron")
        summary.loc[row, ('Best Without Repeat', 'Rank')] = ranking[concept][row]
        concept, train_result, val_result = handle_negation(concept, row)
        summary.loc[row, ('Best Without Repeat', 'Concept')] = concept
        summary.loc[row, ('Best Without Repeat', 'Training')] = train_result
        summary.loc[row, ('Best Without Repeat', 'Validation')] = val_result

    score_cols =  [col for col in summary.columns if col[1] in ['Training', 'Validation']]
    mean = summary[score_cols].mean()
    logger.info(f"Mean:\n{mean}")
    summary.loc['Mean', score_cols] = mean

    csv_summary_path = (options.results_path.joinpath(
        options.target_csv.with_suffix('').name + '_summary').with_suffix('.csv'))
    latex_path = options.results_path.joinpath(options.target_csv.name).with_suffix('.tex')
    summary.to_csv(csv_summary_path)

    formatters = {
        k : class_to_latex_cmd
        for k in summary.columns if k[1] == 'Concept'
    } | {
        k : lambda x: f"{x:.4f}"
        for k in score_cols
    } | {
        k : lambda x: str(x+1)
        for k in summary.columns if k[1] == 'Rank'
    }

    summary.to_latex(
        latex_path,
        formatters = formatters,
        na_rep = ''
    )
    del summary

    summary_best_cols = ['Best Neuron', 'Training', 'Validation']

    if options.include_expectations:
        summary_best_cols.insert(0, 'Expected Neuron')

    summary_best = pd.DataFrame(index=ranking.columns, columns=pd.Index(summary_best_cols))

    if options.include_expectations:
        for i, concept in enumerate(ranking.columns):
            summary_best.loc[concept, 'Expected Neuron'] = i
    best_neurons : list[tuple[str, str]] = ranking[ranking == 0].stack().index.tolist()  # type: ignore
    for row, concept in best_neurons:
        if not pd.isna(summary_best['Best Neuron'][concept]):  # type: ignore
            raise ValueError("Multiple rank 0 neurons found")
        neg_concept, train_result, val_result = handle_negation(concept, row)
        summary_best.loc[concept, 'Best Neuron'] = f"{row} (negated)" if neg_concept != concept else row
        summary_best.loc[concept, 'Training'] = train_result
        summary_best.loc[concept, 'Validation'] = val_result

    summary_best.loc['Mean', ['Training', 'Validation']] = summary_best[['Training', 'Validation']].mean()

    csv_summary_path = (options.results_path.joinpath(
        options.target_csv.with_suffix('').name + '_best_summary').with_suffix('.csv'))
    summary_best.to_csv(csv_summary_path)
    summary_best.to_latex(
        csv_summary_path.with_suffix('.tex'),
        formatters = {
            'Expected Neuron' : lambda x: f"{x}",
            'Best Neuron' : lambda x: f"{x}",
            'Training' : lambda x: f"{x:.4f}",
            'Validation' : lambda x: f"{x:.4f}",
        },
        na_rep = ''
    )




