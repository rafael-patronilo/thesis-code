from typing import Iterator
import unittest
from unittest import mock
from torch import nn
import torch

from ..trainer import Trainer, find_illegal_children
from ..storage_management import ModelFileManager

class TestTrainer(unittest.TestCase):

    def test_basic_cycle(self):
        model = nn.Linear(2, 2)
        dataset = torch.utils.data.TensorDataset(
            torch.empty((128,2)),
            torch.empty((128,2))
        )
        mock_file_manager = mock.Mock(spec=ModelFileManager('test_test_test'))
        loss = nn.MSELoss()
        optimizer = torch.optim.Adam
        trainer = Trainer(model, loss, optimizer, dataset, [])
        trainer.init_file_manager(mock_file_manager)
        trainer.train_epochs(7)
        mock_file_manager.save_checkpoint.assert_called()
        epoch, state_dict, abrupt = mock_file_manager.save_checkpoint.call_args
        self.assertEqual(epoch, 3)
        self.assertFalse(abrupt)
        self.assertIsNone(find_illegal_children(state_dict))