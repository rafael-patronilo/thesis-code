from dataclasses import dataclass
from typing import Callable, Hashable, Literal, NamedTuple, Optional
import pandas as pd
from collections import OrderedDict
import logging

from analysis_tools.xtrains_utils import SHORT_CONCEPTS
logger = logging.getLogger(__name__)

def _identity_fn(x : float): return x

class ABSDifference(NamedTuple):
    midpoint : float
    def __call__(self, score : float) -> float:
        return abs(score - self.midpoint)


def greedy_attribution(
        scores : pd.DataFrame,
        *,
        sort_by : Optional[Callable[[float], float]] = None,
        attribution_mode : Literal['rows_to_cols', 'cols_to_rows'] = 'rows_to_cols',
        maximize : bool = True,
        concepts_only : bool = True
) -> list[tuple[str, str]]:
    """
    Given a matrix of compatibility scores between variables,
    find the best attribution of each variable, so that no variable is assigned more than once.

    This function follows a greedy strategy:
        it chooses attributions in order of the best score,
        discarding the respective column and row for the future attributions
    :param concepts_only:
    :param scores: The compatibility scores between variables
    :param sort_by: An optional function to map each score before sorting
    :param attribution_mode: Whether to attribute rows to columns or columns to rows
    :param maximize: Whether to maximize or minimize the scores
    :return:
    """
    if sort_by is None:
        sort_by = _identity_fn
    if maximize:
        def prefer(a, b) -> bool:
            return a > b
        def idx_best(series_: pd.Series) -> Hashable:
            return series_.idxmax()
    else:
        def prefer(a, b) -> bool:
            return a < b
        def idx_best(series_: pd.Series) -> Hashable:
            return series_.idxmin()
    if attribution_mode == 'cols_to_rows':
        scores = scores.T
    if concepts_only:
        scores = scores[SHORT_CONCEPTS] # type: ignore
    # from here one we can assume we are always attributing rows to columns

    @dataclass
    class _Attribution:
        variable : Hashable
        assigned_to : Hashable
        value : float
        sort_key : float

    scores_mapped = scores.map(sort_by)
    variables = scores.columns.tolist()
    attributions : OrderedDict[Hashable, Optional[_Attribution]] = OrderedDict(
        (var, None)
        for var in variables
    )
    for _ in variables:
        best_attr : Optional[_Attribution] = None
        for row, series in scores_mapped.iterrows():
            col = idx_best(series)
            sort_key = series.loc[col]
            if best_attr is None or prefer(sort_key, best_attr.sort_key):
                best_attr = _Attribution(col, row, scores.at[row, col], sort_key)
        if best_attr is None:
            logger.warning("Run out of variables to assign")
            break
        attributions[best_attr.variable] = best_attr
        scores_mapped.drop(best_attr.assigned_to, inplace=True)
        scores_mapped.drop(best_attr.variable, axis=1, inplace=True)
    results = []
    for v, a in attributions.items():
        if a is None:
            raise ValueError(f"Variable {v} was not assigned")
        results.append((v, a.assigned_to))
    return results # type: ignore

def rank_concepts(
        scores : pd.DataFrame,
        sort_by : Optional[Callable[[float], float]] = None,
        maximize : bool = True,
        concepts_only : bool = True,
        sort_axis : Literal['rows', 'columns'] = 'columns'
) -> pd.DataFrame:
    if sort_by is None:
        sort_by = _identity_fn
    if concepts_only:
        scores = scores[SHORT_CONCEPTS] # type: ignore
    result = pd.DataFrame(index = scores.index, columns = scores.columns)
    if sort_axis == 'columns':
        scores = scores.transpose()
        result = result.transpose()
    for row in scores.index:
        series : pd.Series = scores.loc[row]
        series = series.map(sort_by)
        series.sort_values(ascending=not maximize, inplace=True)
        for i, (col, _) in enumerate(series.items()):
            result.at[row, col] = i
    if sort_axis == 'columns':
        result = result.transpose()
    return result