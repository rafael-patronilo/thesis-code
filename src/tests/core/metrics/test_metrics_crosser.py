import unittest
from torcheval.metrics import BinaryAccuracy, BinaryRecall, BinaryPrecision, BinaryF1Score
from collections import Counter

from core.eval.metrics.metrics_crosser import *

class MetricCrosserTests(unittest.TestCase):

    def setUp(self) -> None:
        
        self.addTypeEqualityFunc(pd.DataFrame,
            lambda first, second, msg: pd.testing.assert_frame_equal(first, second, obj=msg))
        return super().setUp()

    def test_accuracy(self):
        preds = torch.Tensor([
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [1, 1, 1],
            [1, 0, 1]
        ])
        trues = torch.Tensor([
            [1, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0, 0],
            [0, 1],
            [1, 1]
        ])

        trues2 = trues.clone()
        trues2[6,1] = 0
        expected = pd.DataFrame({
            'p1' : {'t1' : 4, 't2' : 4},
            'p2' : {'t1' : 2, 't2' : 4},
            'p3' : {'t1' : 4, 't2' : 4}
        }).transpose()
        expected2 = expected.copy()
        expected2.at['p1','t2'] -= 1
        expected2.at['p2','t2'] += 1
        expected2.at['p3','t2'] -= 1
        expected = expected.map(lambda x : torch.div(x, 7).item())
        expected2 = expected2.map(lambda x : torch.div(x, 7).item())
        cols = ['t1', 't2']
        rows = ['p1', 'p2', 'p3']

        crosser = MetricCrosser(rows, cols, {'accuracy' : BinaryAccuracy})
        crosser.update(preds[0:2,:], trues[0:2,:])
        crosser.update(preds[2: ,:], trues[2: ,:])
        result = crosser.compute()['accuracy']

        self.assertDictEqual(Counter(result.index), Counter(rows))
        self.assertDictEqual(Counter(result.columns), Counter(cols))
        self.assertEqual(expected, result)
        self.assertEqual(torch.div(4, 7).item(), result.at['p2', 't2'])
        self.assertNotEqual(torch.div(3, 7).item(), result.at['p3', 't2'])

        crosser.reset()
        crosser.update(preds, trues2)
        result2 = crosser.compute()['accuracy']
        self.assertEqual(expected2, result2)

    def multiple_test(self, rng, crosser, metrics):
        preds = torch.rand(30, 5, generator=rng)
        trues = torch.randint(high=2, size=(30, 9), generator=rng)
        
        def compute_cell(factory,i,j):
            metric : Metric = factory()
            metric.update(input=preds[:,i], target=trues[:,j])
            return metric.compute().item()
        manual = {
            name : pd.DataFrame([
                [compute_cell(factory,i,j) for j in range(9)]
                for i in range(5)
            ],
            index = [f'p{i}' for i in range(5)],
            columns = [f't{j}' for j in range(9)])
            for name, factory in metrics.items()
        }
        crosser.update(preds, trues)
        result = crosser.compute()
        for name, table in result.items():
            with self.subTest(metric=name):
                self.assertEqual(manual[name], table)

    def test_multiple_reset(self):
        metrics = {
            'accuracy' : BinaryAccuracy,
            'precision' : BinaryPrecision,
            'recall' : BinaryRecall,
            'f1' : BinaryF1Score
        }
        crosser = MetricCrosser(
            pred_labels = [f'p{i}' for i in range(5)],
            true_labels = [f't{j}' for j in range(9)],
            metrics = metrics
        )
        rng = torch.Generator().manual_seed(299)
        self.multiple_test(rng, crosser, metrics)
        crosser.reset()
        self.multiple_test(rng, crosser, metrics)