import unittest
from unittest import mock
from torch import nn
import torch

from core.training.trainer import Trainer, find_illegal_children
from core.storage_management import ModelFileManager


def create_mock_file_manager(path: str = 'test_test_test') -> mock.Mock:
    file_manager = mock.Mock(ModelFileManager)
    for name, value in ModelFileManager(path).__dict__.items():
        setattr(type(file_manager), name, mock.PropertyMock(return_value=value))
    return file_manager


class TestTrainer(unittest.TestCase):

    def test_basic_cycle(self):
        model = nn.Linear(2, 2)
        dataset = torch.utils.data.TensorDataset(
            torch.empty((128,2)),
            torch.empty((128,2))
        )

        mock_file_manager = create_mock_file_manager()
        loss = nn.MSELoss()
        optimizer = torch.optim.Adam
        trainer = Trainer(model, loss, optimizer, dataset, [])
        trainer.init_file_manager(mock_file_manager)
        trainer.train_epochs(7)
        mock_file_manager.save_checkpoint.assert_called()
        epoch, state_dict, abrupt = mock_file_manager.save_checkpoint.call_args.args
        self.assertEqual(epoch, 7)
        self.assertFalse(abrupt)
        self.assertIsNone(find_illegal_children(state_dict))