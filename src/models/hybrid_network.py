from typing import Literal
from core import ModelFileManager, Trainer


def load_reasoning_network(model_name : str, model_path : str | None = None):
    with ModelFileManager(model_name, model_path) as file_manager:
        trainer = Trainer.load_checkpoint(file_manager)
        return trainer.model


def create_perception_network(
        input_shape : tuple[int, int, int],
        num_concepts: int,
        conv_layers : list[int | tuple[Literal['pool'], int]],
        kernel_size : int = 3,
        use_batch_norm : bool = True,
        use_dropout : bool = True,
        activation : Literal['relu'] = 'relu'
    ):
    raise NotImplementedError("This function is not implemented yet.")