import unittest
from unittest.mock import Mock
from torch import nn
import torch
from ..layers import *

class TestMinMaxNormalizer(unittest.TestCase):

    def make_mock_model(self):
        class MockModule(nn.Module):
            def forward(self, x):
                return x
        return MockModule()

    def test_min_max_normalizer(self):
        tensor = torch.tensor([
            [-20, 0, 10],
            [  5, 1, 20],
        ])
        normalizer = MinMaxNormalizer(tensor.min(), tensor.max())
        result = normalizer(tensor)
        self.assertGreaterEqual(result.min().item(), 0)
        self.assertLessEqual(result.max().item(), 1)
    
    def test_min_max_normalizer_append(self):
        tensor = torch.tensor([
            [-20, 0, 10],
            [  5, 1, 20],
        ])
        model = self.make_mock_model()
        normalizer = MinMaxNormalizer(tensor.min(dim=0).values, tensor.max(dim=0).values)
        model = add_layers(model, normalizer)
        result = model(tensor)
        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 1)

    def test_min_max_normalizer_fit(self):
        rng = torch.Generator().manual_seed(56)
        model = self.make_mock_model()
        dataset = torch.utils.data.TensorDataset(
            torch.rand(32, 5, generator=rng) * 100,
            torch.empty((32,))
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=8)
        normalizer = MinMaxNormalizer.fit(model, loader)
        self.assertEqual(normalizer.min.size(0), 5)
        self.assertEqual(normalizer.max.size(0), 5)

        model = add_layers(model, normalizer)
        result : torch.Tensor = model(dataset.tensors[0])
        self.assertGreaterEqual(result.min().item(), 0)
        self.assertLessEqual(result.max().item(), 1)