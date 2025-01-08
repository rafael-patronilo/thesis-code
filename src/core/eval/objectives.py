from pandas.io.common import abstractmethod
from .metrics import MetricResult
from abc import ABC

type ResultsDict = dict[str, dict[str, MetricResult]]

class Objective(ABC):
    @abstractmethod
    def compare(self, a : ResultsDict, b : ResultsDict) -> bool:
        """
        Determine whether `a` is significantly better than `b`
        May return False even if `a` is better than `b` if the difference is not significant
        :param a: a metrics dictionary
        :param b: a metrics dictionary
        :return: True if `a` is significantly better than `b`, False otherwise
        """
        raise NotImplementedError()

    @abstractmethod
    def compare_strict(self, a : ResultsDict, b : ResultsDict) -> bool:
        """
        Determine whether `a` is better than `b`
        Opposite to `compare` this method strictly compares `a` and `b` and will
        return True if `a` is better than `b` even if the difference is not significant

        :param a: a metrics dictionary
        :param b: a metrics dictionary
        :return: True if `a` is better than `b`, False otherwise
        """
        raise NotImplementedError()

class FloatObjective(ABC):
    def __init__(self, metrics_group : str, metric : str, threshold : float = 0) -> None:
        self.metrics_group = metrics_group
        self.metric = metric
        self.threshold = threshold

    def _get_values(self, a : ResultsDict, b : ResultsDict) -> tuple[float, float]:
        value_a = a[self.metrics_group][self.metric]
        value_b = b[self.metrics_group][self.metric]
        assert isinstance(value_a, float) and isinstance(value_b, float)
        return value_a, value_b

    @abstractmethod
    def diff(self, a : float, b : float) -> float:
        """
        This method should calculate which value is better and how much better it is:
        - If `a` is better than `b`, it should return a positive value representing
            how much better `a` is
        - If `b` is better than `a`, it should return a negative value representing
            how much better `b` is
        :param a: the value to compare
        :param b: the value to compare
        :return: a positive value if `a` is better than `b`, a negative value if `b` is better than `a`
        """
        raise NotImplementedError()

    def compare_strict(self, a : ResultsDict, b : ResultsDict) -> bool:
        value_a, value_b = self._get_values(a, b)
        return self.diff(value_a, value_b) > 0

    def compare(self, a : ResultsDict, b : ResultsDict) -> bool:
        value_a, value_b = self._get_values(a, b)
        return self.diff(value_a, value_b) >= self.threshold


class Maximize(FloatObjective):
    def diff(self, a : float, b : float) -> float:
        return a - b

class Minimize(FloatObjective):
    def diff(self, a : float, b : float) -> float:
        return b - a
