"""Work in progress, potential replacement for EarlyStop and CheckpointTrigger repeated logic"""


from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Iterable, overload, override, final
from functools import singledispatchmethod
if TYPE_CHECKING:
    from core.trainer import MetricsSnapshot

class CompareKeyGetter(ABC):
    @abstractmethod
    def get_compare_key(self, snapshot : 'MetricsSnapshot') -> float:
        pass

@final
class BasicGetter(CompareKeyGetter):
    def __init__(self, metric_logger : str, metric_name : str):
        self.metric_logger = metric_logger
        self.metric_name = metric_name
    
    def get_compare_key(self, snapshot : 'MetricsSnapshot') -> float:
        return snapshot[self.metric_logger][self.metric_name]


class MetricCompareStrategy(ABC):
    @overload
    def __init__(self, metric_getter : CompareKeyGetter): pass
    
    @overload
    def __init__(self, metric_logger : str, metric_name : str): pass
    
    @singledispatchmethod
    def __init__(self, *args, **kwargs):
        pass

    @__init__.register
    def _(self, metric_getter : CompareKeyGetter):
        self.metric_getter = metric_getter

    @__init__.register
    def _(self, metric_logger : str, metric_name : str):
        self.metric_getter = BasicGetter(metric_logger, metric_name)

    def compare(self, snapshot1 : 'MetricsSnapshot', snapshot2 : 'MetricsSnapshot', threshold : float = 0.0) -> bool:
        """Determines if the metrics of a snapshot are preferable

        Args:
            snapshot1 (MetricsSnapshot): the first snapshot
            snapshot2 (MetricsSnapshot): the second snapshot

        Returns:
            bool: true if the first snapshot is preferable, false otherwise
        """
        return self.compare_float(
            self.metric_getter.get_compare_key(snapshot1), 
            self.metric_getter.get_compare_key(snapshot2), 
            threshold
        )


    @abstractmethod
    def compare_float(self, value1 : float, value2 : float, threshold : float = 0.0) -> bool:
        """Determines if the value of a metric is preferable

        Args:
            value1 (float): the value of the metric in the first snapshot
            value2 (float): the value of the metric in the second snapshot

        Returns:
            bool: true if the first value is preferable, false otherwise
        """
        pass

@final
class PreferMax(MetricCompareStrategy):
    @override
    def compare_float(self, value1 : float, value2 : float, threshold : float = 0.0) -> bool:
        return value1 > value2 and abs(value1 - value2) > threshold

@final
class PreferMin(MetricCompareStrategy):
    @override
    def compare_float(self, value1: float, value2: float, threshold: float = 0) -> bool:
        return value1 < value2 and abs(value1 - value2) > threshold


class AggregatingGetter(CompareKeyGetter, ABC):
    def __init__(self, *metric_getters : CompareKeyGetter | tuple[str, str]):
        self.metric_getters = [
            getter if isinstance(getter, CompareKeyGetter) else BasicGetter(*getter)
            for getter in metric_getters
        ]
    

    @abstractmethod
    def aggregate(self, values : Iterable[float]) -> float:
        pass

    def get_compare_key(self, snapshot : 'MetricsSnapshot') -> float:
        return self.aggregate(getter.get_compare_key(snapshot) for getter in self.metric_getters)

@final
class MinOf(AggregatingGetter):
    def aggregate(self, values: Iterable[float]) -> float:
        return min(values)

@final
class MaxOf(AggregatingGetter):
    def aggregate(self, values: Iterable[float]) -> float:
        return max(values)

@final
class CustomAggregator(AggregatingGetter):
    def __init__(self, aggregator : Callable[[Iterable[float]], float], *metric_getters : CompareKeyGetter | tuple[str, str]):
        super().__init__(*metric_getters)
        self.aggregator = aggregator

    def aggregate(self, values: Iterable[float]) -> float:
        return self.aggregator(values)