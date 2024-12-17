import unittest
from torcheval.metrics import BinaryAccuracy

from core.eval.metrics import *
from core.eval.metrics.metric_wrappers import SelectCol

class SelectColTests(unittest.TestCase):
    def test_select_col(self):
        manual = BinaryAccuracy()
        rng = torch.Generator().manual_seed(91)
        y_pred = torch.rand(10, 5, generator=rng)
        y_true = torch.randint(0, 2, (10, 5), generator=rng)
        manual.update(y_pred[:, 2], y_true[:, 2])

        inner = BinaryAccuracy()
        metric = SelectCol(inner, 2)
        metric.update(y_pred, y_true)
        self.assertEqual(metric.compute(), manual.compute())
