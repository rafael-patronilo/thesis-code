from types import SimpleNamespace
import unittest
from core.eval.metrics import PearsonCorrelationCoefficient

import torch


class PearsonCorrelationTests(unittest.TestCase):

    def test_pearson_correlation(self):
        metric = PearsonCorrelationCoefficient()
        rng = torch.Generator().manual_seed(67)
        x  = torch.randn(100, generator=rng)
        y = torch.randn(100, generator=rng)
        noise = torch.randn(100, generator=rng)

        metric.reset()
        metric.update(x, x)
        self.assertAlmostEqual(1.0, metric.compute().item(), places=4)

        metric.reset()
        metric.update(x, y)
        self.assertAlmostEqual(0.0829, metric.compute().item(), places=4)

        metric.reset()
        metric.update(x, -x)
        self.assertAlmostEqual(-1.0, metric.compute().item(), places=4)

        metric.reset()
        metric.update(x, 2*x)
        self.assertAlmostEqual(1.0, metric.compute().item(), places=4)

        metric.reset()
        metric.update(x, 2*x + 0.1 * noise)
        self.assertAlmostEqual(0.9992, metric.compute().item(), places=4)
